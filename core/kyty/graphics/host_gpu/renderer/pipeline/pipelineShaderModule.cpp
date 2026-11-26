#include "graphics/host_gpu/renderer/pipeline/pipelineManifest.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineShaderDiagnostics.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineShaderOptions.h"

#include "common/debugCounters.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/rayTracing.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/shaderCompiler.h"

#include <algorithm>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

uint32_t SubgroupStages(vk::ShaderStageFlags supported) {
	uint32_t stages = 0;
	for (const auto stage: {ShaderType::Vertex, ShaderType::Pixel, ShaderType::Compute, ShaderType::Mesh,
	                        ShaderType::Local, ShaderType::TessellationControl, ShaderType::TessellationEvaluation}) {
		if (supported & NativeShaderStage(stage)) {
			stages |= 1u << static_cast<uint32_t>(stage);
		}
	}
	return stages;
}
vk::Format NarrowInputFormat(vk::Format format, uint32_t used_components) {
	switch (format) {
		case vk::Format::eR32G32B32A32Sfloat:
			switch (used_components) {
				case 1: return vk::Format::eR32Sfloat;
				case 2: return vk::Format::eR32G32Sfloat;
				case 3: return vk::Format::eR32G32B32Sfloat;
				default: return format;
			}
		case vk::Format::eR32G32B32Sfloat:
			switch (used_components) {
				case 1: return vk::Format::eR32Sfloat;
				case 2: return vk::Format::eR32G32Sfloat;
				default: return format;
			}
		case vk::Format::eR16G16B16A16Sfloat:
			switch (used_components) {
				case 1: return vk::Format::eR16Sfloat;
				case 2: return vk::Format::eR16G16Sfloat;
				default: return format;
			}
		case vk::Format::eR8G8B8A8Unorm:
			switch (used_components) {
				case 1: return vk::Format::eR8Unorm;
				case 2: return vk::Format::eR8G8Unorm;
				default: return format;
			}
		case vk::Format::eR8G8B8A8Snorm:
			return used_components == 2 ? vk::Format::eR8G8Snorm : format;
		case vk::Format::eR8G8B8A8Uint:
			switch (used_components) {
				case 1: return vk::Format::eR8Uint;
				case 2: return vk::Format::eR8G8Uint;
				default: return format;
			}
		default: return format;
	}
}

vk::Format InputFormat(const ShaderBufferResource& resource, uint32_t used_components) {
	switch (resource.RawFormat()) {
		case 113: return NarrowInputFormat(vk::Format::eR32G32B32A32Sfloat, used_components);
		case 121: return vk::Format::eR16G16Sfloat;
		default: break;
	}
	const auto format = VulkanFormat(resource.Format());
	if (format == vk::Format::eUndefined ||
	    ShaderRecompiler::Format::GetFormatInfo(resource.Format()).component_count == 0) {
		EXIT("unknown vertex format: fmt = %u\n", resource.RawFormat());
	}
	return NarrowInputFormat(format, used_components);
}

std::vector<vk::Format> VertexInputFormats(const ShaderVertexInputInfo&                    input_info,
                                           const ShaderRecompiler::IR::CompiledShaderInfo& program) {
	std::vector<vk::Format> formats(input_info.resources_num);
	for (int index = 0; index < input_info.resources_num; index++) {
		const auto registers_num = input_info.resources_dst[index].registers_num;
		EXIT_IF(registers_num < 1 || registers_num > 4);
		const auto fetched = program.info.vertex_fetch_components[index];
		formats[index] = InputFormat(input_info.resources[index],
		                             fetched > 0 ? fetched : static_cast<uint32_t>(registers_num));
	}
	return formats;
}

}

void SetShaderCompileOptions(ShaderRecompiler::CompileOptions& options, uint64_t hash,
                             std::span<const uint32_t> user_data, std::span<const uint32_t> back_code,
                             bool native, bool lane_pairs) {
	options.shader_hash        = hash;
	options.user_data          = user_data;
	options.back_code          = back_code;
	options.dump_ir            = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
	options.emit_debug_info    = Config::GraphicsDebugDumpEnabled();
	options.native_ray_tracing = native;
	options.ray_stats          = native && Config::RayTracingStatsEnabled();
	options.rt_oracle          = native ? Config::RayTracingOracle() : 0u;
	options.wave_lane_pairs    = lane_pairs;
}

void AppendNativeRayTracingKey(ShaderType stage, std::vector<uint32_t>& key, bool native) {
	if (native && stage == ShaderType::Pixel && RayTracing::NativeTraversals() && Config::RayTracingOracle() != 0) {
		key.push_back(0x52544f00u | Config::RayTracingOracle());
		if (Config::RayTracingStatsEnabled()) {
			key.push_back(0x52545354u);
		}
	}
	if (native && stage == ShaderType::Compute && RayTracing::NativeTraversals()) {
		key.push_back(0x52545041u);
		if (Config::RayTracingStatsEnabled()) {
			key.push_back(0x52545354u);
		}
		if (const auto oracle = Config::RayTracingOracle(); oracle != 0) {
			key.push_back(0x52544f00u | oracle);
		}
	}
}

ShaderRecompiler::Spirv::Profile SpirvProfile(const GraphicContext& graphics) {
	const auto operations = graphics.subgroup_supported_operations;
	return {.maintenance8           = graphics.maintenance8_enabled,
	        .native_float_min_max   = graphics.physical_device_properties.vendorID == 0x1002u,
	        .device_clock_period_ps = graphics.device_clock_period_ps,
	        .robust_buffer_access   = graphics.robust_buffer_access_enabled,
	        .r8_srgb_views          = graphics.r8_srgb_views,
	        .r8g8_srgb_views        = graphics.r8g8_srgb_views,
	        .relaxed_float          = Config::RelaxedShaderFloatEnabled(),
	        .ray_query              = graphics.ray_query_enabled,
	        .lds_lane_sync          = LdsLaneSyncEnabled(),
	        .native_float16         = graphics.shader_float16_enabled,
	        .f32_product_no_contraction = F32ProductNoContractionEnabled(),
	        .subgroup_arithmetic = bool(operations & vk::SubgroupFeatureFlagBits::eArithmetic),
	        .subgroup_ballot     = bool(operations & vk::SubgroupFeatureFlagBits::eBallot),
	        .subgroup_shuffle    = bool(operations & vk::SubgroupFeatureFlagBits::eShuffle),
	        .subgroup_quad       = bool(operations & vk::SubgroupFeatureFlagBits::eQuad),
	        .subgroup_stages     = SubgroupStages(graphics.subgroup_supported_stages)};
}

uint64_t ManifestModule::ProgramHash() const {
	if (back_hash == 0) {
		return code_hash;
	}
	const uint64_t hashes[] = {code_hash, back_hash};
	return XXH3_64bits(hashes, sizeof(hashes));
}

void PipelineCache::SetHostInput(ManifestModule& module) const {
	const auto& limits = m_graphics.GetPhysicalDeviceProperties().limits;
	if (auto* compute = std::get_if<ShaderComputeInputInfo>(&module.input)) {
		compute->host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
		compute->lds_storage = compute->lds_size_dwords * 4u > limits.maxComputeSharedMemorySize;
	} else if (auto* vertex = std::get_if<ShaderVertexInputInfo>(&module.input)) {
		if (module.stage == ShaderType::Mesh) {
			vertex->mesh.host_subgroup_size = m_graphics.subgroup_size;
		}
		if (vertex->clip_space.enabled) {
			vertex->clip_space.half_extent[0] =
			    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
			vertex->clip_space.half_extent[1] =
			    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		}
	}
}

ShaderModule PipelineCache::BuildModule(const ProgramKey&                                  key,
                                       const ShaderRecompiler::CompileOptions&            options,
                                       ShaderRecompiler::TranslateResult                  translated,
                                       const ShaderRecompiler::IR::ResourceSpecialization& specialization,
                                       std::span<const uint32_t> function_code,
                                       uint32_t push_data_cursor, std::vector<uint32_t>& spirv,
                                       std::vector<uint32_t>& plain_spirv,
                                       std::string* table_refusal) const {
	Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::ShadersCompiled,
	                                         Common::DebugCounters::Counter::ShaderCompileNs,
	                                         Common::DebugCounters::Peak::ShaderCompileNs);
	const auto profile         = SpirvProfile(m_graphics);
	auto       compile_options = options;
	compile_options.profile    = &profile;
	compile_options.plain_mip_stats_variant = key.stage == ShaderType::Pixel;
	auto result = ShaderRecompiler::CompileProgram(std::move(translated), compile_options,
	                                               specialization, push_data_cursor);
	if (compile_options.table_mode && result.spirv.empty()) {
		if (table_refusal != nullptr) *table_refusal = std::move(result.table_refusal);
		return {};
	}
	auto info = std::move(result.program).TakeCompiledInfo();

	ShaderRecompiler::IR::Writer inputs;
	ShaderRecompiler::IR::IoAll(inputs, key.stage, key.hash, key.user_data_count, key.static_state,
	                            info.bindings.push_data_start_dword, specialization);
	inputs.Raw(function_code.data(), function_code.size_bytes());
	const auto id = XXH3_64bits(inputs.bytes.data(), inputs.bytes.size());

	const auto* stage = ShaderRecompiler::StageName(key.stage);
	DumpShader(result.spirv, fmt::format("{}_{:016x}_{:016x}.spv", stage, key.hash, id));
	ValidateShaderSpirv(stage, key.hash, result.spirv, profile.maintenance8);
	ShaderModule module {
	    .id             = id,
	    .module         = CompileSPV(result.spirv, m_graphics.device),
	    .info           = std::move(info),
	    .specialization = specialization,
	    .function_code  = {function_code.begin(), function_code.end()},
	};
	module.spirv_words = static_cast<uint32_t>(result.spirv.size());
	if (key.stage == ShaderType::Vertex || key.stage == ShaderType::Local) {
		module.vertex_formats = VertexInputFormats(*options.input_info.vertex, module.info);
	} else if (key.stage == ShaderType::Pixel &&
	           options.input_info.pixel->parameter_mode == ShaderPixelParameterMode::Rectangle) {
		module.rect_parameters = RectListParameters(*options.input_info.pixel, module.info);
	} else if (key.stage == ShaderType::Mesh) {
		module.mesh_task = options.input_info.vertex->indirect_draw;
	} else if (key.stage == ShaderType::Compute) {
		module.guest_traversal = RayTracing::NativeTraversals() && !options.native_ray_tracing;
	}
	if (!result.spirv_plain.empty()) {
		ValidateShaderSpirv(stage, key.hash, result.spirv_plain, profile.maintenance8);
		module.plain = MakePlainModule(module, result.spirv_plain, m_graphics.device);
		EXIT_IF(module.plain == nullptr);
	}
#if defined(ANYPS5_KYTY)
	LogShaderVariantBuild(stage, key.hash, key.user_data_count, key.static_state, push_data_cursor, module);
#endif
	spirv       = std::move(result.spirv);
	plain_spirv = std::move(result.spirv_plain);
	return module;
}

std::shared_ptr<const ShaderModule> MakePlainModule(const ShaderModule&       module,
                                                    std::span<const uint32_t> spirv,
                                                    vk::Device                device) {
	auto plain = std::make_shared<ShaderModule>();
	plain->id     = module.id ^ PlainModuleIdXor;
	plain->module = CompileSPV(spirv, device);
	if (plain->module == nullptr) {
		return nullptr;
	}
	plain->info            = module.info;
	plain->specialization  = module.specialization;
	plain->function_code   = module.function_code;
	plain->vertex_formats  = module.vertex_formats;
	plain->rect_parameters = module.rect_parameters;
	plain->spirv_words     = static_cast<uint32_t>(spirv.size());
	return plain;
}

}
