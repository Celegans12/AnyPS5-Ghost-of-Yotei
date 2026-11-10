#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/envRead.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/startupTrace.h"
#include "common/shaderEnv.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/perfCounters.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderPrecompile.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/rectListShader.h"
#include "graphics/shader/shader.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <limits>
#include <map>
#include <mutex>
#include <span>
#include <vector>

namespace Libs::Graphics {

namespace {

// PIPELINE_REPEAT (diagnostic): a driver compile (50 ms or more)
// of a VS+PS pair (shader hashes) already compiled in this session, with the inputs that differ from its closest earlier
// variant - which pipeline state makes the same shaders compile again (dynamic state or a pipeline library, stage 2).
struct PipelineVariant {
	uint64_t                 vs_id = 0, ps_id = 0; // the translated programs (another permutation of the same hash)
	uint64_t                 vs_module = 0, ps_module = 0; // their VkShaderModule handles
	uint64_t                 layout_hash = 0;              // descriptor set 0 bindings (binding, type, count, stages)
	bool                     rect_list = false;
	std::array<uint32_t, ShaderVertexInputInfo::RES_MAX> attribute_formats {};
	PipelineStaticParameters static_params;
	PipelineRenderingState   rendering;
	PipelineVertexInputState vertex_input;
};

std::string PipelineVariantDiff(const PipelineVariant& a, const PipelineVariant& b) {
	std::string out;
	const auto  add = [&](bool differs, const char* name) {
        if (!differs) return;
        if (!out.empty()) out += ',';
        out += name;
	};
	const auto& x       = a.static_params;
	const auto& y       = b.static_params;
	const auto  differs = [](const auto& p, const auto& q) { return std::memcmp(&p, &q, sizeof(p)) != 0; };
	add(a.vs_id != b.vs_id, "vs_program");
	add(a.vs_module != b.vs_module, "vs_module");
	add(a.ps_module != b.ps_module, "ps_module");
	add(a.layout_hash != b.layout_hash, "layout");
	add(a.rect_list != b.rect_list, "rect_list");
	add(a.attribute_formats != b.attribute_formats, "vertex_formats");
	add(a.ps_id != b.ps_id, "ps_program");
	add(x.topology != y.topology, "topology");
	add(x.primitive_restart_enable != y.primitive_restart_enable, "primitive_restart");
	add(x.cull_front != y.cull_front || x.cull_back != y.cull_back, "cull");
	add(x.face != y.face, "front_face");
	add(x.polygon_mode != y.polygon_mode, "polygon_mode");
	add(x.depth_clip_enable != y.depth_clip_enable, "depth_clip");
	add(x.negative_one_to_one != y.negative_one_to_one, "clip_space");
	add(x.provoking_vtx_last != y.provoking_vtx_last, "provoking");
	add(x.samples != y.samples, "samples");
	add(x.sample_shading_enable != y.sample_shading_enable, "sample_shading");
	add(x.alpha_to_coverage_enable != y.alpha_to_coverage_enable, "alpha_to_coverage");
	add(differs(x.color_mask, y.color_mask), "color_mask");
	add(differs(x.blend_enable, y.blend_enable), "blend_enable");
	add(differs(x.color_srcblend, y.color_srcblend) || differs(x.color_destblend, y.color_destblend) ||
	        differs(x.color_comb_fcn, y.color_comb_fcn) || differs(x.alpha_srcblend, y.alpha_srcblend) ||
	        differs(x.alpha_destblend, y.alpha_destblend) || differs(x.alpha_comb_fcn, y.alpha_comb_fcn) ||
	        differs(x.separate_alpha_blend, y.separate_alpha_blend),
	    "blend_equation");
	add(a.rendering.color_count != b.rendering.color_count, "color_count");
	add(a.rendering.color_formats != b.rendering.color_formats, "color_formats");
	add(a.rendering.depth_format != b.rendering.depth_format || a.rendering.stencil_format != b.rendering.stencil_format,
	    "depth_format");
	const auto& v = a.vertex_input;
	const auto& w = b.vertex_input;
	add(v.binding_count != w.binding_count || v.attribute_count != w.attribute_count, "vertex_counts");
	bool strides = false, rates = false, offsets = false;
	for (size_t i = 0; i < v.bindings.size(); i++) {
		strides |= v.bindings[i].stride != w.bindings[i].stride;
		rates |= v.bindings[i].instance != w.bindings[i].instance;
	}
	for (size_t i = 0; i < v.attributes.size(); i++) offsets |= !(v.attributes[i] == w.attributes[i]);
	add(strides, "vertex_strides");
	add(rates, "vertex_rates");
	add(offsets, "vertex_offsets");
	return out.empty() ? std::string("none") : out;
}

void NotePipelineVariant(uint64_t vs_hash, uint64_t ps_hash, const PipelineVariant& variant, double ms, bool background) {
	static std::mutex                                                           mutex;
	static std::map<std::pair<uint64_t, uint64_t>, std::vector<PipelineVariant>> seen;
	std::string                                                                 diff;
	size_t                                                                      earlier = 0;
	{
		std::lock_guard lock(mutex);
		auto&           list = seen[{vs_hash, ps_hash}];
		earlier              = list.size();
		size_t best          = SIZE_MAX;
		for (const auto& old: list) {
			auto   candidate = PipelineVariantDiff(variant, old);
			size_t fields    = static_cast<size_t>(std::count(candidate.begin(), candidate.end(), ',')) + 1;
			if (fields < best) {
				best = fields;
				diff = std::move(candidate);
			}
		}
		if (list.size() < 64) list.push_back(variant);
	}
	if (earlier == 0 || ms < 50.0) return;
	std::fprintf(stderr, "PIPELINE_REPEAT %s vs=0x%016llx ps=0x%016llx %.0fms earlier=%zu diff=%s\n",
	             background ? "graphics-background" : "graphics", static_cast<unsigned long long>(vs_hash),
	             static_cast<unsigned long long>(ps_hash), ms, earlier, diff.c_str());
}

} // namespace

// IDK: maybe we can remove it?
constexpr uint8_t kTemporaryVertexAttribFormat113 =
    static_cast<uint8_t>(Prospero::VertexAttribFormat::k16_16SInt);
constexpr uint32_t kTemporaryPs5BufferFormat121 = 121u;

static bool NarrowInputFormat(vk::Format& format, uint32_t& size, uint32_t used_components) {
	if (used_components == 0 || used_components >= size) {
		return false;
	}

	switch (format) {
		case vk::Format::eR32G32B32A32Sfloat:
			switch (used_components) {
				case 1: format = vk::Format::eR32Sfloat; break;
				case 2: format = vk::Format::eR32G32Sfloat; break;
				case 3: format = vk::Format::eR32G32B32Sfloat; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR32G32B32Sfloat:
			switch (used_components) {
				case 1: format = vk::Format::eR32Sfloat; break;
				case 2: format = vk::Format::eR32G32Sfloat; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR16G16B16A16Sfloat:
			switch (used_components) {
				case 1: format = vk::Format::eR16Sfloat; break;
				case 2: format = vk::Format::eR16G16Sfloat; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR8G8B8A8Unorm:
			switch (used_components) {
				case 1: format = vk::Format::eR8Unorm; break;
				case 2: format = vk::Format::eR8G8Unorm; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR8G8B8A8Snorm:
			if (used_components != 2) {
				return false;
			}
			format = vk::Format::eR8G8Snorm;
			size   = 2;
			return true;
		case vk::Format::eR8G8B8A8Uint:
			switch (used_components) {
				case 1: format = vk::Format::eR8Uint; break;
				case 2: format = vk::Format::eR8G8Uint; break;
				default: return false;
			}
			size = used_components;
			return true;
		default: break;
	}

	return false;
}

static void GetInputFormat(const ShaderBufferResource& res, vk::Format& format, uint32_t& size,
                           uint32_t used_components) {
	const auto fmt        = res.Format();
	const auto raw_format = res.RawFormat();
	if (raw_format == kTemporaryVertexAttribFormat113) {
		// Atomic: background compiles (shaderPrecompile.h) create pipelines too.
		static std::atomic_bool logged_113 {false};
		if (!logged_113.exchange(true, std::memory_order_relaxed)) {
			LOGF("InputFormat: temporary: accepting invalid PS5 buffer format 113 as "
			     "vk::Format::eR32G32B32A32Sfloat\n");
		}
		format = vk::Format::eR32G32B32A32Sfloat;
		size   = 4;
		if (NarrowInputFormat(format, size, used_components)) {
			LOGF("InputFormat: narrowing fmt=%u to %s for used_components=%u\n", raw_format,
			     vk::to_string(format).c_str(), used_components);
		}
		return;
	}
	if (raw_format == kTemporaryPs5BufferFormat121) {
		static std::atomic_bool logged_121 {false};
		if (!logged_121.exchange(true, std::memory_order_relaxed)) {
			LOGF("InputFormat: accepting PS5 buffer format 121 as vk::Format::eR16G16Sfloat\n");
		}
		format = vk::Format::eR16G16Sfloat;
		size   = 2;
		return;
	}

	format = VulkanFormat(fmt);
	size   = ShaderRecompiler::Format::GetFormatInfo(fmt).component_count;
	if (format == vk::Format::eUndefined || size == 0) {
		EXIT("unknown vertex format: fmt = %u\n", raw_format);
	}

	if (NarrowInputFormat(format, size, used_components)) {
		static std::atomic<uint64_t> log_count = 0;
		auto                         log_id    = log_count.fetch_add(1);
		if (log_id < 32) {
			LOGF("VertexInput: narrowed vertex format to %" PRIu32
			     " component(s) for shader fetch\n",
			     used_components);
		}
	}
}

static vk::BlendFactor GetBlendFactor(uint32_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kZero: return vk::BlendFactor::eZero;
		case Prospero::BlendFactor::kOne: return vk::BlendFactor::eOne;
		case Prospero::BlendFactor::kSrcColor: return vk::BlendFactor::eSrcColor;
		case Prospero::BlendFactor::kOneMinusSrcColor: return vk::BlendFactor::eOneMinusSrcColor;
		case Prospero::BlendFactor::kSrcAlpha: return vk::BlendFactor::eSrcAlpha;
		case Prospero::BlendFactor::kOneMinusSrcAlpha: return vk::BlendFactor::eOneMinusSrcAlpha;
		case Prospero::BlendFactor::kDstAlpha: return vk::BlendFactor::eDstAlpha;
		case Prospero::BlendFactor::kOneMinusDstAlpha: return vk::BlendFactor::eOneMinusDstAlpha;
		case Prospero::BlendFactor::kDstColor: return vk::BlendFactor::eDstColor;
		case Prospero::BlendFactor::kOneMinusDstColor: return vk::BlendFactor::eOneMinusDstColor;
		case Prospero::BlendFactor::kSrcAlphaSaturate: return vk::BlendFactor::eSrcAlphaSaturate;
		case Prospero::BlendFactor::kConstantColor: return vk::BlendFactor::eConstantColor;
		case Prospero::BlendFactor::kOneMinusConstantColor:
			return vk::BlendFactor::eOneMinusConstantColor;
		case Prospero::BlendFactor::kSrc1Color: return vk::BlendFactor::eSrc1Color;
		case Prospero::BlendFactor::kOneMinusSrc1Color: return vk::BlendFactor::eOneMinusSrc1Color;
		case Prospero::BlendFactor::kSrc1Alpha: return vk::BlendFactor::eSrc1Alpha;
		case Prospero::BlendFactor::kOneMinusSrc1Alpha: return vk::BlendFactor::eOneMinusSrc1Alpha;
		case Prospero::BlendFactor::kConstantAlpha: return vk::BlendFactor::eConstantAlpha;
		case Prospero::BlendFactor::kOneMinusConstantAlpha:
			return vk::BlendFactor::eOneMinusConstantAlpha;
		default: EXIT("unknown factor: %u\n", factor);
	}
	return vk::BlendFactor::eZero;
}

static vk::BlendOp GetBlendOp(uint32_t op) {
	switch (static_cast<Prospero::BlendOp>(op)) {
		case Prospero::BlendOp::kAdd: return vk::BlendOp::eAdd;
		case Prospero::BlendOp::kSubtract: return vk::BlendOp::eSubtract;
		case Prospero::BlendOp::kMin: return vk::BlendOp::eMin;
		case Prospero::BlendOp::kMax: return vk::BlendOp::eMax;
		case Prospero::BlendOp::kReverseSubtract: return vk::BlendOp::eReverseSubtract;
		default: EXIT("unknown op: %u\n", op);
	}
	return vk::BlendOp::eAdd;
}

// KYTY_BINDLESS: a stage samples images through the set-1 bindless arrays.
// KYTY_TRACE_PIPELINE_STATS=<all|comma separated hash prefixes> (diagnostic): pipelines whose
// shader hash starts with a listed prefix are created with statistics capture, and the driver's
// per-executable statistics (registers, local memory/spills, ...) are printed as PIPELINE_STATS.
static bool PipelineStatsWanted(const GraphicContext& graphics, uint64_t a, uint64_t b) {
	if (!graphics.pipeline_stats_enabled) {
		return false;
	}
	static const std::vector<std::string> prefixes = [] {
		std::vector<std::string> list;
		const char*              value = Common::ShaderEnv::Read("KYTY_TRACE_PIPELINE_STATS");
		std::string              all   = value != nullptr ? value : "";
		size_t                   start = 0;
		while (start <= all.size()) {
			const size_t end = std::min(all.find(',', start), all.size());
			if (end > start) list.push_back(all.substr(start, end - start));
			start = end + 1;
		}
		return list;
	}();
	char text[2][17];
	std::snprintf(text[0], sizeof(text[0]), "%016llx", static_cast<unsigned long long>(a));
	std::snprintf(text[1], sizeof(text[1]), "%016llx", static_cast<unsigned long long>(b));
	for (const auto& prefix: prefixes) {
		if (prefix == "all" || prefix == "1") return true;
		for (const auto* hash: text) {
			if (std::strncmp(hash, prefix.c_str(), prefix.size()) == 0) return true;
		}
	}
	return false;
}

static void PrintPipelineStats(const GraphicContext& graphics, vk::Pipeline pipeline, const char* kind,
                               uint64_t a, uint64_t b) {
	vk::PipelineInfoKHR pipeline_info {};
	pipeline_info.pipeline = pipeline;
	uint32_t count         = 0;
	if (graphics.device.getPipelineExecutablePropertiesKHR(&pipeline_info, &count, nullptr) !=
	    vk::Result::eSuccess) {
		return;
	}
	std::vector<vk::PipelineExecutablePropertiesKHR> executables(count);
	if (graphics.device.getPipelineExecutablePropertiesKHR(&pipeline_info, &count, executables.data()) !=
	    vk::Result::eSuccess) {
		return;
	}
	for (uint32_t index = 0; index < count; index++) {
		vk::PipelineExecutableInfoKHR executable {};
		executable.pipeline        = pipeline;
		executable.executableIndex = index;
		uint32_t stat_count        = 0;
		if (graphics.device.getPipelineExecutableStatisticsKHR(&executable, &stat_count, nullptr) !=
		    vk::Result::eSuccess) {
			continue;
		}
		std::vector<vk::PipelineExecutableStatisticKHR> stats(stat_count);
		if (graphics.device.getPipelineExecutableStatisticsKHR(&executable, &stat_count, stats.data()) !=
		    vk::Result::eSuccess) {
			continue;
		}
		std::string line;
		for (const auto& stat: stats) {
			char value[64] = {};
			switch (stat.format) {
				case vk::PipelineExecutableStatisticFormatKHR::eBool32:
					std::snprintf(value, sizeof(value), "%u", static_cast<unsigned>(stat.value.b32));
					break;
				case vk::PipelineExecutableStatisticFormatKHR::eInt64:
					std::snprintf(value, sizeof(value), "%lld", static_cast<long long>(stat.value.i64));
					break;
				case vk::PipelineExecutableStatisticFormatKHR::eUint64:
					std::snprintf(value, sizeof(value), "%llu", static_cast<unsigned long long>(stat.value.u64));
					break;
				case vk::PipelineExecutableStatisticFormatKHR::eFloat64:
					std::snprintf(value, sizeof(value), "%g", stat.value.f64);
					break;
			}
			line += " ";
			line += stat.name.data();
			line += "=";
			line += value;
		}
		std::fprintf(stderr, "PIPELINE_STATS %s %016llx %016llx exe=%u(%s) subgroup=%u:%s\n", kind,
		             static_cast<unsigned long long>(a), static_cast<unsigned long long>(b), index,
		             executables[index].name.data(), executables[index].subgroupSize, line.c_str());
	}
	std::fflush(stderr);
}

static bool UsesBindless(const ShaderRecompiler::IR::CompiledShaderInfo& program) {
	return std::ranges::any_of(program.info.images, &ShaderRecompiler::IR::ImageResource::bindless);
}

// Set 0 is the pipeline's own descriptors; set 1, when a stage samples bindless images, is the
// shared bindless table.
static void FillSetLayouts(GraphicContext& graphics, const PipelineCache::Pipeline& pipeline,
                           std::array<vk::DescriptorSetLayout, 2>& layouts, uint32_t& count) {
	layouts[0] = pipeline.descriptor_set_layout;
	count      = 1;
	// Set 1 is our bindless table (KYTY_BINDLESS). Upstream's resident image heap (27dc7c281) would also take set 1
	// (graphics.image_heap_layout); it is never in use here (RenderContext::ImageHeapEnabled), so no program asks for it.
	EXIT_IF(pipeline.uses_image_heap);
	if (pipeline.uses_bindless) {
		EXIT_IF(graphics.bindless_layout == nullptr);
		layouts[1] = graphics.bindless_layout;
		count      = 2;
	}
}

static void AddLayoutBindings(std::vector<vk::DescriptorSetLayoutBinding>& descriptor_bindings,
                              const ShaderRecompiler::IR::CompiledShaderInfo& program,
                              vk::ShaderStageFlagBits              stage) {
	for (auto mask = program.bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
		const auto index = std::countr_zero(mask);
		const auto kind = static_cast<ShaderRecompiler::IR::DescriptorBindingKind>(index);
		descriptor_bindings.push_back(
		    {ShaderRecompiler::IR::NativeBinding(program.stage, kind), NativeDescriptorType(kind),
		     program.bindings.descriptor_counts[index], stage, nullptr});
	}
}

// False only on a background compile (shaderPrecompile.h) the driver refused; the GPU thread exits then.
static bool CreateDescriptorLayout(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                                   std::span<const vk::DescriptorSetLayoutBinding> bindings) {
	const auto& limits = graphics.GetPhysicalDeviceProperties().limits;
	uint64_t descriptor_count = 0;
	uint64_t storage_buffers = 0;
	uint64_t samplers = 0;
	std::array<uint64_t, 32> stage_buffers {}, stage_resources {}, stage_samplers {};
	for (const auto& binding: bindings) {
		descriptor_count += binding.descriptorCount;
		const bool storage = binding.descriptorType == vk::DescriptorType::eStorageBuffer;
		const bool sampler = binding.descriptorType == vk::DescriptorType::eSampler;
		if (storage) storage_buffers += binding.descriptorCount;
		if (sampler) samplers += binding.descriptorCount;
		for (auto stages = static_cast<uint32_t>(binding.stageFlags); stages != 0; stages &= stages - 1u) {
			const auto stage = std::countr_zero(stages);
			if (storage) stage_buffers[stage] += binding.descriptorCount;
			if (sampler) stage_samplers[stage] += binding.descriptorCount;
			else
				stage_resources[stage] += binding.descriptorCount;
			if (stage_buffers[stage] > limits.maxPerStageDescriptorStorageBuffers ||
			    stage_resources[stage] > limits.maxPerStageResources ||
			    stage_samplers[stage] > limits.maxPerStageDescriptorSamplers) {
				EXIT("native descriptor stage %u exceeds device limits: buffers=%" PRIu64
				     "/%u resources=%" PRIu64 "/%u samplers=%" PRIu64 "/%u\n", stage, stage_buffers[stage],
				     limits.maxPerStageDescriptorStorageBuffers, stage_resources[stage], limits.maxPerStageResources,
				     stage_samplers[stage], limits.maxPerStageDescriptorSamplers);
			}
		}
	}
	if (storage_buffers > limits.maxDescriptorSetStorageBuffers) {
		EXIT("native storage-buffer descriptors exceed device limit: %" PRIu64 "/%u\n",
		     storage_buffers, limits.maxDescriptorSetStorageBuffers);
	}
	if (samplers > limits.maxDescriptorSetSamplers) {
		EXIT("native sampler descriptors exceed device limit: %" PRIu64 "/%u\n",
		     samplers, limits.maxDescriptorSetSamplers);
	}
	pipeline.uses_push_descriptors = descriptor_count <= graphics.max_push_descriptors;

	vk::DescriptorSetLayoutCreateInfo create {};
	create.flags        = pipeline.uses_push_descriptors
	                          ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
	                          : vk::DescriptorSetLayoutCreateFlags {};
	create.bindingCount = static_cast<uint32_t>(bindings.size());
	create.pBindings    = bindings.data();
	const auto result =
	    graphics.device.createDescriptorSetLayout(&create, nullptr, &pipeline.descriptor_set_layout);
	if (result != vk::Result::eSuccess && IsBackgroundCompileThread()) {
		pipeline.descriptor_set_layout = nullptr;
		return false;
	}
	EXIT_IF(result != vk::Result::eSuccess);
	return true;
}

// The blend factors and operations GetBlendFactor and GetBlendOp map (they exit on any other).
static bool BlendFactorSupported(uint32_t factor) {
	return factor <= static_cast<uint32_t>(Prospero::BlendFactor::kSrcAlphaSaturate) ||
	       (factor >= static_cast<uint32_t>(Prospero::BlendFactor::kConstantColor) &&
	        factor <= static_cast<uint32_t>(Prospero::BlendFactor::kOneMinusConstantAlpha));
}

static bool BlendOpSupported(uint32_t op) {
	return op <= static_cast<uint32_t>(Prospero::BlendOp::kReverseSubtract);
}

// The vertex formats GetInputFormat maps (it exits on any other).
static bool InputFormatSupported(const ShaderBufferResource& res) {
	const auto raw_format = res.RawFormat();
	if (raw_format == kTemporaryVertexAttribFormat113 || raw_format == kTemporaryPs5BufferFormat121) {
		return true;
	}
	return VulkanFormat(res.Format()) != vk::Format::eUndefined &&
	       ShaderRecompiler::Format::GetFormatInfo(res.Format()).component_count != 0;
}

const char* GraphicsPipelineUnsupported(const GraphicContext& graphics, const PipelineRenderingState& rendering,
                                        const PipelineVertexInputState&        vertex_input,
                                        std::span<const ShaderVertexInputInfo> vertex_info,
                                        const ShaderPixelInputInfo*            ps_input_info,
                                        const PipelineStaticParameters&        static_params) {
	if (vertex_info.size() != 1 && vertex_info.size() != 3) {
		return "vertex stage count";
	}
	for (const auto& stage: vertex_info) {
		if (!stage.stage) {
			return "vertex stage program";
		}
	}
	if (ps_input_info != nullptr && !ps_input_info->stage) {
		return "pixel stage program";
	}
	const auto& vs_input_info = vertex_info.front();
	const bool  mesh          = vs_input_info.stage.program->stage == ShaderType::Mesh;
	if (mesh) {
		if (!graphics.mesh_shader_enabled) {
			return "mesh shaders unsupported";
		}
		// PipelineCache::GetGraphicsPrograms exits on a mesh draw beyond these.
		const auto& mesh_info = vs_input_info.mesh;
		const auto& limits    = graphics.mesh_shader_properties;
		if (mesh_info.wave_size == 0) {
			return "mesh wave size";
		}
		const auto logical_threads = mesh_info.threads_num[0] * mesh_info.threads_num[1] * mesh_info.threads_num[2];
		const auto host_threads    = ((logical_threads + mesh_info.wave_size - 1u) / mesh_info.wave_size) *
		                          std::min(mesh_info.host_subgroup_size, mesh_info.wave_size);
		if (host_threads > limits.maxMeshWorkGroupInvocations || host_threads > limits.maxMeshWorkGroupSize[0] ||
		    mesh_info.max_vertices > limits.maxMeshOutputVertices ||
		    mesh_info.max_primitives > limits.maxMeshOutputPrimitives ||
		    mesh_info.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			return "mesh shader exceeds host limits";
		}
	}
	if (vertex_info.size() == 3 &&
	    (vs_input_info.tess.input_control_points == 0 ||
	     vs_input_info.tess.input_control_points > graphics.GetPhysicalDeviceProperties().limits.maxTessellationPatchSize)) {
		return "tessellation patch size";
	}
	if (static_params.provoking_vtx_last && !graphics.provoking_vertex_last_enabled) {
		return "provoking vertex last unsupported";
	}
	if (static_params.sample_shading_enable && !graphics.sample_rate_shading_enabled) {
		return "sample-rate shading unsupported";
	}
	if (vulkan_sample_count(static_params.samples) == vk::SampleCountFlagBits {}) {
		return "sample count";
	}
	if (static_cast<uint32_t>(static_params.topology) > static_cast<uint32_t>(vk::PrimitiveTopology::ePatchList) ||
	    static_cast<uint32_t>(static_params.polygon_mode) > static_cast<uint32_t>(vk::PolygonMode::ePoint)) {
		return "topology or polygon mode";
	}
	if (rendering.color_count > RENDER_COLOR_ATTACHMENTS_MAX) {
		return "color attachment count";
	}
	for (uint32_t i = 0; i < rendering.color_count; i++) {
		if ((static_params.color_mask[i] & ~0x0fu) != 0) {
			return "color write mask";
		}
		if (!BlendFactorSupported(static_params.color_srcblend[i]) ||
		    !BlendFactorSupported(static_params.color_destblend[i]) ||
		    !BlendOpSupported(static_params.color_comb_fcn[i]) ||
		    (static_params.separate_alpha_blend[i] &&
		     (!BlendFactorSupported(static_params.alpha_srcblend[i]) ||
		      !BlendFactorSupported(static_params.alpha_destblend[i]) ||
		      !BlendOpSupported(static_params.alpha_comb_fcn[i])))) {
			return "blend state";
		}
	}
	if (!mesh) {
		if (vertex_input.binding_count > ShaderVertexInputInfo::RES_MAX ||
		    vertex_input.attribute_count > ShaderVertexInputInfo::RES_MAX) {
			return "vertex input count";
		}
		for (uint32_t index = 0; index < vertex_input.attribute_count; index++) {
			const auto registers_num = vs_input_info.resources_dst[index].registers_num;
			if (vertex_input.attributes[index].binding >= vertex_input.binding_count) {
				return "vertex input binding";
			}
			if (registers_num < 1 || registers_num > 4) {
				return "vertex input registers";
			}
			if (!InputFormatSupported(vs_input_info.resources[index])) {
				return "vertex input format";
			}
		}
	}
	bool bindless = ps_input_info != nullptr && UsesBindless(*ps_input_info->stage.program);
	for (const auto& stage: vertex_info) {
		bindless = bindless || UsesBindless(*stage.stage.program);
	}
	if (bindless && graphics.bindless_layout == nullptr) {
		return "bindless table missing";
	}
	return nullptr;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const PipelineRenderingState&          rendering,
                            const PipelineVertexInputState&        vertex_input,
                            std::span<const ShaderVertexInputInfo> vertex_info,
                            const ShaderPixelInputInfo*            ps_input_info,
                            const PipelineCache::GraphicsPrograms& programs,
                            const PipelineStaticParameters&        static_params,
                            vk::PipelineCache                      driver_cache) {
	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	const bool  tessellation   = vertex_info.size() == 3;
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(!vertex_program || (ps_active && !pixel_program));
	const bool with_depth = rendering.depth_format != vk::Format::eUndefined ||
	                        rendering.stencil_format != vk::Format::eUndefined;
	EXIT_IF(!vs_input_info.stage);
	// A background compile (shaderPrecompile.h) gives up where the GPU thread exits: what it created is destroyed
	// (the caller destroys the layouts and pipeline left in `pipeline`). It checks the state first
	// (GraphicsPipelineUnsupported); the driver calls can still fail.
	const bool background = IsBackgroundCompileThread();
	const bool mesh = vs_input_info.stage.program->stage == ShaderType::Mesh;
	if (background && mesh && !graphics.mesh_shader_enabled) {
		return;
	}
	EXIT_NOT_IMPLEMENTED(mesh && !graphics.mesh_shader_enabled);
	const bool rect_list =
	    !mesh && !tessellation && static_params.topology == vk::PrimitiveTopology::ePatchList;

	vk::ShaderModule tess_control_shader_module = nullptr;
	vk::ShaderModule tess_eval_shader_module    = nullptr;

	if (rect_list) {
		const auto shaders =
		    BuildRectListShaders(vs_input_info, ps_active ? ps_input_info : nullptr);
		tess_control_shader_module = background ? TryCompileSPV(shaders.control, graphics.device)
		                                        : CompileSPV(shaders.control, graphics.device);
		if (graphics_debug_dump_enabled()) {
			LOGF("PipelineTrace: vkCreateShaderModule RectList TCS done module=%p\n",
			     static_cast<void*>(tess_control_shader_module));
		}

		tess_eval_shader_module = background ? TryCompileSPV(shaders.evaluation, graphics.device)
		                                     : CompileSPV(shaders.evaluation, graphics.device);
		if (graphics_debug_dump_enabled()) {
			LOGF("PipelineTrace: vkCreateShaderModule RectList TES done module=%p\n",
			     static_cast<void*>(tess_eval_shader_module));
		}
	}
	const auto destroy_rect_list_modules = [&] {
		if (tess_control_shader_module != nullptr) {
			graphics.device.destroyShaderModule(tess_control_shader_module, nullptr);
		}
		if (tess_eval_shader_module != nullptr) {
			graphics.device.destroyShaderModule(tess_eval_shader_module, nullptr);
		}
	};

	if (background && rect_list &&
	    (tess_control_shader_module == nullptr || tess_eval_shader_module == nullptr)) {
		destroy_rect_list_modules();
		return;
	}
	EXIT_NOT_IMPLEMENTED(
	    rect_list && (tess_control_shader_module == nullptr || tess_eval_shader_module == nullptr));

	vk::PipelineShaderStageCreateInfo shader_stages[4] {};
	uint32_t                          shader_stage_count = 0;
	for (uint32_t i = 0; i < vertex_info.size(); i++) {
		shader_stages[shader_stage_count++] = {.stage =
		                                           NativeShaderStage(vertex_info[i].logical_stage),
		                                       .module = programs.vertex[i].module,
		                                       .pName  = "main"};
	}
	if (rect_list) {
		shader_stages[shader_stage_count++] = {.stage =
		                                           vk::ShaderStageFlagBits::eTessellationControl,
		                                       .module = tess_control_shader_module,
		                                       .pName  = "main"};
		shader_stages[shader_stage_count++] = {.stage =
		                                           vk::ShaderStageFlagBits::eTessellationEvaluation,
		                                       .module = tess_eval_shader_module,
		                                       .pName  = "main"};
	}
	if (ps_active) {
		shader_stages[shader_stage_count++] = {.stage  = vk::ShaderStageFlagBits::eFragment,
		                                       .module = pixel_program.module,
		                                       .pName  = "main"};
	}

	vk::VertexInputAttributeDescription input_attr[ShaderVertexInputInfo::RES_MAX] {};
	vk::VertexInputBindingDescription   input_desc[ShaderVertexInputInfo::RES_MAX] {};

	for (uint32_t binding = 0; binding < vertex_input.binding_count; binding++) {
		input_desc[binding].binding   = binding;
		input_desc[binding].stride    = vertex_input.bindings[binding].stride;
		input_desc[binding].inputRate = vertex_input.bindings[binding].instance
		                                    ? vk::VertexInputRate::eInstance
		                                    : vk::VertexInputRate::eVertex;
	}
	for (uint32_t index = 0; index < vertex_input.attribute_count; index++) {
		input_attr[index].binding  = vertex_input.attributes[index].binding;
		input_attr[index].location = index;
		input_attr[index].offset   = vertex_input.attributes[index].offset;

		uint32_t   attr_size     = 4;
		const auto registers_num = vs_input_info.resources_dst[index].registers_num;
		const auto compiled_components =
		    vs_input_info.stage.program->info.vertex_fetch_components[index];
		const auto used_components =
		    compiled_components > 0 ? static_cast<int>(compiled_components) : registers_num;
		GetInputFormat(vs_input_info.resources[index], input_attr[index].format, attr_size,
		               static_cast<uint32_t>(used_components));

		if (graphics_debug_dump_enabled()) {
			static std::atomic_uint log_count = 0;
			const auto              log_id    = log_count.fetch_add(1, std::memory_order_relaxed);
			if (log_id < 128) {
				LOGF("VertexInputState[%u]: attr=%u binding=%u offset=%u stride=%u fmt=%d "
				     "src_fmt=%u dst=v%u regs=%u"
				     " fetched_components=%u attr_size=%u swizzle=%u,%u,%u,%u\n",
				     log_id, index, input_attr[index].binding, input_attr[index].offset,
				     input_desc[input_attr[index].binding].stride,
				     static_cast<int>(input_attr[index].format),
				     static_cast<uint32_t>(vs_input_info.resources[index].Format()),
				     static_cast<uint32_t>(vs_input_info.resources_dst[index].register_start),
				     static_cast<uint32_t>(registers_num), static_cast<uint32_t>(used_components),
				     attr_size, static_cast<uint32_t>(vs_input_info.resources[index].DstSelX()),
				     static_cast<uint32_t>(vs_input_info.resources[index].DstSelY()),
				     static_cast<uint32_t>(vs_input_info.resources[index].DstSelZ()),
				     static_cast<uint32_t>(vs_input_info.resources[index].DstSelW()));
			}
		}

		if (vs_input_info.resources[index].OutOfBounds() != 0) {
			static std::atomic_bool logged {false};
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				LOGF("VertexInput: temporary: accepting PS5 out-of-bounds behavior %" PRIu8 "\n",
				     vs_input_info.resources[index].OutOfBounds());
			}
		}

		EXIT_IF(registers_num < 1 || registers_num > 4);
	}

	vk::PipelineVertexInputStateCreateInfo vertex_input_info {};
	vertex_input_info.vertexBindingDescriptionCount   = vertex_input.binding_count;
	vertex_input_info.pVertexBindingDescriptions      = input_desc;
	vertex_input_info.vertexAttributeDescriptionCount = vertex_input.attribute_count;
	vertex_input_info.pVertexAttributeDescriptions    = input_attr;

	vk::PipelineInputAssemblyStateCreateInfo input_assembly {};
	input_assembly.topology = static_params.topology;
	input_assembly.primitiveRestartEnable =
	    static_params.primitive_restart_enable ? VK_TRUE : VK_FALSE;

	vk::PipelineViewportDepthClipControlCreateInfoEXT depth_clip_control {};
	depth_clip_control.negativeOneToOne = (static_params.negative_one_to_one ? VK_TRUE : VK_FALSE);

	vk::PipelineViewportStateCreateInfo viewport_state {};
	viewport_state.pNext = &depth_clip_control;

	vk::CullModeFlags cull_mode = vk::CullModeFlagBits::eNone;
	if (static_params.cull_back) {
		cull_mode |= vk::CullModeFlagBits::eBack;
	}
	if (static_params.cull_front) {
		cull_mode |= vk::CullModeFlagBits::eFront;
	}

	vk::FrontFace front_face =
	    (static_params.face ? vk::FrontFace::eClockwise : vk::FrontFace::eCounterClockwise);

	vk::PipelineRasterizationDepthClipStateCreateInfoEXT clip_ext {};
	clip_ext.depthClipEnable = static_params.depth_clip_enable ? VK_TRUE : VK_FALSE;

	vk::PipelineRasterizationStateCreateInfo rasterizer {};
	// MoltenVK lacks VK_EXT_depth_clip_enable; omit the depth-clip struct on macOS and accept
	// Vulkan's default depth clipping (enabled) instead of the PS5's clamp behavior.
#if !defined(__APPLE__)
	// The DB clamps depth to the viewport range after polygon offset is applied.
	rasterizer.depthClampEnable = VK_TRUE;
	rasterizer.pNext = &clip_ext;
#endif
	vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT provoking_vertex {};
	if (background && static_params.provoking_vtx_last && !graphics.provoking_vertex_last_enabled) {
		destroy_rect_list_modules();
		return;
	}
	EXIT_NOT_IMPLEMENTED(static_params.provoking_vtx_last &&
	                     !graphics.provoking_vertex_last_enabled);
	if (graphics.provoking_vertex_last_enabled) {
		provoking_vertex.provokingVertexMode = static_params.provoking_vtx_last
		    ? vk::ProvokingVertexModeEXT::eLastVertex : vk::ProvokingVertexModeEXT::eFirstVertex;
		provoking_vertex.pNext = rasterizer.pNext;
		rasterizer.pNext = &provoking_vertex;
	}
	rasterizer.cullMode  = cull_mode;
	rasterizer.frontFace = front_face;
	rasterizer.polygonMode = static_params.polygon_mode;
	rasterizer.lineWidth = 1.0f;

	vk::PipelineMultisampleStateCreateInfo multisampling {};
	multisampling.sampleShadingEnable  = static_params.sample_shading_enable ? VK_TRUE : VK_FALSE;
	multisampling.rasterizationSamples = vulkan_sample_count(static_params.samples);
	multisampling.minSampleShading     = 1.0f;
	multisampling.alphaToCoverageEnable =
	    static_params.alpha_to_coverage_enable ? VK_TRUE : VK_FALSE;

	vk::PipelineColorBlendAttachmentState color_blend_attachment[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	for (uint32_t i = 0; i < rendering.color_count; i++) {
		EXIT_NOT_IMPLEMENTED((static_params.color_mask[i] & ~0x0fu) != 0);
		color_blend_attachment[i].colorWriteMask =
		    vk::ColorComponentFlags {static_params.color_mask[i]};
		color_blend_attachment[i].blendEnable = static_params.blend_enable[i] ? VK_TRUE : VK_FALSE;
		color_blend_attachment[i].srcColorBlendFactor =
		    GetBlendFactor(static_params.color_srcblend[i]);
		color_blend_attachment[i].dstColorBlendFactor =
		    GetBlendFactor(static_params.color_destblend[i]);
		color_blend_attachment[i].colorBlendOp = GetBlendOp(static_params.color_comb_fcn[i]);
		color_blend_attachment[i].srcAlphaBlendFactor =
		    (static_params.separate_alpha_blend[i]
		         ? GetBlendFactor(static_params.alpha_srcblend[i])
		         : color_blend_attachment[i].srcColorBlendFactor);
		color_blend_attachment[i].dstAlphaBlendFactor =
		    (static_params.separate_alpha_blend[i]
		         ? GetBlendFactor(static_params.alpha_destblend[i])
		         : color_blend_attachment[i].dstColorBlendFactor);
		color_blend_attachment[i].alphaBlendOp =
		    (static_params.separate_alpha_blend[i] ? GetBlendOp(static_params.alpha_comb_fcn[i])
		                                           : color_blend_attachment[i].colorBlendOp);
	}

	vk::Bool32 color_write_enable[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	for (uint32_t i = 0; i < rendering.color_count; i++) {
		color_write_enable[i] = VK_TRUE;
	}

	vk::PipelineColorWriteCreateInfoEXT color_write {};
	color_write.attachmentCount    = rendering.color_count;
	color_write.pColorWriteEnables = color_write_enable;

	vk::PipelineColorBlendStateCreateInfo color_blending {};
	// MoltenVK lacks VK_EXT_color_write_enable; drop the dynamic color-write struct on macOS
	// and rely on each attachment's static colorWriteMask (all channels enabled by default).
#if !defined(__APPLE__)
	color_blending.pNext = &color_write;
#endif
	color_blending.logicOp         = vk::LogicOp::eCopy;
	color_blending.attachmentCount = rendering.color_count;
	color_blending.pAttachments    = color_blend_attachment;

	std::vector<vk::DescriptorSetLayoutBinding> descriptor_bindings;
	vk::ShaderStageFlags graphics_stages = vk::ShaderStageFlagBits::eFragment;
	pipeline.uses_bindless = false;
	for (const auto& stage: vertex_info) {
		const auto native_stage = NativeShaderStage(stage.logical_stage);
		AddLayoutBindings(descriptor_bindings, *stage.stage.program, native_stage);
		pipeline.uses_image_heap |= stage.stage.program->info.uses_image_heap;
		graphics_stages |= native_stage;
		pipeline.uses_bindless |= UsesBindless(*stage.stage.program);
	}
	if (ps_active) {
		EXIT_IF(!ps_input_info->stage);
		AddLayoutBindings(descriptor_bindings, *ps_input_info->stage.program,
		                  vk::ShaderStageFlagBits::eFragment);
		pipeline.uses_bindless |= UsesBindless(*ps_input_info->stage.program);
		pipeline.uses_image_heap |= ps_input_info->stage.program->info.uses_image_heap;
	}
	if (!CreateDescriptorLayout(graphics, pipeline, descriptor_bindings) ||
	    (background && pipeline.uses_bindless && graphics.bindless_layout == nullptr)) {
		destroy_rect_list_modules();
		return;
	}
	const vk::PushConstantRange push_constants {graphics_stages, 0,
	                                            ShaderRecompiler::IR::NativePushConstantSize};

	std::array<vk::DescriptorSetLayout, 2> set_layouts {};
	uint32_t                               set_layout_count = 0;
	FillSetLayouts(graphics, pipeline, set_layouts, set_layout_count);
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = set_layout_count;
	pipeline_layout_info.pSetLayouts            = set_layouts.data();
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_constants;

	EXIT_IF(pipeline.pipeline_layout != nullptr);

	if (graphics_debug_dump_enabled()) {
		LOGF("PipelineTrace: vkCreatePipelineLayout begin VS=%" PRIu64 " PS=%" PRIu64
		     " set_layouts=1 push_constants=%" PRIu32 "\n",
		     vertex_program.id, ps_active ? pixel_program.id : 0, 1u);
	}
	auto result = graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                   &pipeline.pipeline_layout);
	if (graphics_debug_dump_enabled()) {
		LOGF("PipelineTrace: vkCreatePipelineLayout done result=%s layout=%p\n",
		     vk::to_string(result).c_str(), static_cast<void*>(pipeline.pipeline_layout));
	}
	if (background && (result != vk::Result::eSuccess || pipeline.pipeline_layout == nullptr)) {
		pipeline.pipeline_layout = nullptr;
		destroy_rect_list_modules();
		return;
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	EXIT_NOT_IMPLEMENTED(pipeline.pipeline_layout == nullptr);

	vk::PipelineDepthStencilStateCreateInfo depth_stencil_info {};

	std::vector<vk::DynamicState> dynamic_states {
	    vk::DynamicState::eViewportWithCount,
	    vk::DynamicState::eScissorWithCount,
	    vk::DynamicState::eLineWidth,
	    vk::DynamicState::eDepthTestEnable,
	    vk::DynamicState::eDepthWriteEnable,
	    vk::DynamicState::eDepthCompareOp,
	    vk::DynamicState::eDepthBiasEnable,
	    vk::DynamicState::eDepthBias,
	    vk::DynamicState::eStencilTestEnable,
	    vk::DynamicState::eStencilOp,
	    vk::DynamicState::eStencilCompareMask,
	    vk::DynamicState::eStencilReference,
	    vk::DynamicState::eStencilWriteMask,
	    vk::DynamicState::eBlendConstants,
	};
#if !defined(__APPLE__)
	dynamic_states.push_back(vk::DynamicState::eDepthBoundsTestEnable);
	dynamic_states.push_back(vk::DynamicState::eDepthBounds);
	if (rendering.color_count != 0) {
		dynamic_states.push_back(vk::DynamicState::eColorWriteEnableEXT);
	}
#endif
	if (graphics.attachment_feedback_loop_enabled) {
		dynamic_states.push_back(vk::DynamicState::eAttachmentFeedbackLoopEnableEXT);
	}

	vk::PipelineDynamicStateCreateInfo dynamic_state {};
	dynamic_state.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
	dynamic_state.pDynamicStates    = dynamic_states.data();

	vk::GraphicsPipelineCreateInfo  pipeline_info {};
	vk::PipelineRenderingCreateInfo rendering_info {};
	rendering_info.colorAttachmentCount    = rendering.color_count;
	rendering_info.pColorAttachmentFormats = rendering.color_formats.data();
	rendering_info.depthAttachmentFormat   = rendering.depth_format;
	rendering_info.stencilAttachmentFormat = rendering.stencil_format;
	pipeline_info.pNext                    = &rendering_info;
	pipeline_info.stageCount               = shader_stage_count;
	pipeline_info.pStages                  = shader_stages;
	pipeline_info.pVertexInputState        = mesh ? nullptr : &vertex_input_info;
	pipeline_info.pInputAssemblyState      = mesh ? nullptr : &input_assembly;
	vk::PipelineTessellationStateCreateInfo tessellation_state {};
	tessellation_state.patchControlPoints =
	    tessellation ? vs_input_info.tess.input_control_points : 3u;
	pipeline_info.pTessellationState = (rect_list || tessellation) ? &tessellation_state : nullptr;
	pipeline_info.pViewportState          = &viewport_state;
	pipeline_info.pRasterizationState     = &rasterizer;
	pipeline_info.pMultisampleState       = &multisampling;
	pipeline_info.pDepthStencilState      = (with_depth ? &depth_stencil_info : nullptr);
	pipeline_info.pColorBlendState        = &color_blending;
	pipeline_info.pDynamicState           = &dynamic_state;
	pipeline_info.layout                  = pipeline.pipeline_layout;
	pipeline_info.basePipelineIndex       = -1;

	EXIT_IF(pipeline.pipeline != nullptr);

	if (graphics_debug_dump_enabled()) {
		LOGF("PipelineTrace: vkCreateGraphicsPipelines begin VS=%" PRIu64 " PS=%" PRIu64
		     " topology=%" PRIu32 " color_mask=0x%08" PRIx32
		     " depth=%s blend=%s dyn_states=%" PRIu32 "\n",
		     vertex_program.id, ps_active ? pixel_program.id : 0,
		     static_cast<uint32_t>(static_params.topology), static_params.color_mask[0],
		     (with_depth ? "true" : "false"), (static_params.blend_enable[0] ? "true" : "false"),
		     dynamic_state.dynamicStateCount);
	}
	const uint64_t stats_vs   = vs_input_info.stage.program->shader_hash;
	const uint64_t stats_ps   = ps_active && ps_input_info->stage ? ps_input_info->stage.program->shader_hash : 0;
	const bool     want_stats = PipelineStatsWanted(graphics, stats_vs, stats_ps);
	if (want_stats) {
		pipeline_info.flags |= vk::PipelineCreateFlagBits::eCaptureStatisticsKHR;
	}
	const auto create_begin = std::chrono::steady_clock::now();
	result = graphics.device.createGraphicsPipelines(driver_cache, 1, &pipeline_info, nullptr,
	                                                 &pipeline.pipeline);
	// KYTY_TRACE_STARTUP: every graphics pipeline creation (GPU thread or background), 50 ms+ ones also as slow.
	if (Common::StartupTrace::Enabled()) {
		const auto ms =
		    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - create_begin).count();
		using Common::StartupTrace::Counter;
		Common::StartupTrace::AddMs(background ? Counter::PipeBackground : Counter::PipeGraphics, ms);
		if (ms >= 50.0) Common::StartupTrace::Add(Counter::PipeSlow, 0);
	}
	// KYTY_TRACE_PERF: a driver pipeline compile over 50 ms (a pipeline-cache hit takes well under 1 ms); always, one on the
	// GPU thread over 200 ms (the game waits for it: 25.10, stalls of 4-70 s in play were all driver compiles there).
	if (const auto create_ms =
	        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - create_begin).count();
	    (PerfEnabled() && create_ms >= 50.0) || (!background && create_ms >= 200.0)) {
		std::fprintf(stderr, "PIPELINE_TIME %s vs=0x%016llx ps=0x%016llx %.0fms created_ago=%.0f/%.0fms\n",
		             background ? "graphics-background" : "graphics", static_cast<unsigned long long>(stats_vs),
		             static_cast<unsigned long long>(stats_ps), create_ms, ShaderCreatedMsAgo(stats_vs),
		             ShaderCreatedMsAgo(stats_ps));
		std::fflush(stderr);
	}
	if (result == vk::Result::eSuccess) { // PIPELINE_REPEAT
		const auto create_ms =
		    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - create_begin).count();
		NotePipelineVariant(stats_vs, stats_ps,
		                    {.vs_id         = vertex_program.id,
		                     .ps_id         = ps_active ? pixel_program.id : 0,
		                     .vs_module     = reinterpret_cast<uint64_t>(static_cast<VkShaderModule>(vertex_program.module)),
		                     .ps_module     = ps_active ? reinterpret_cast<uint64_t>(static_cast<VkShaderModule>(pixel_program.module)) : 0,
		                     .layout_hash   = [&] {
			                     uint64_t h = 1469598103934665603ull;
			                     for (const auto& b: descriptor_bindings) {
				                     for (const uint64_t v: {uint64_t {b.binding}, uint64_t(b.descriptorType), uint64_t {b.descriptorCount},
				                                             uint64_t(static_cast<uint32_t>(b.stageFlags))}) {
					                     h = (h ^ v) * 1099511628211ull;
				                     }
			                     }
			                     return h;
		                     }(),
		                     .rect_list     = rect_list,
		                     .attribute_formats = [&] {
			                     std::array<uint32_t, ShaderVertexInputInfo::RES_MAX> f {};
			                     for (uint32_t i = 0; i < vertex_input.attribute_count; i++) f[i] = static_cast<uint32_t>(input_attr[i].format);
			                     return f;
		                     }(),
		                     .static_params = static_params,
		                     .rendering     = rendering,
		                     .vertex_input  = vertex_input},
		                    create_ms, background);
	}
	if (want_stats && result == vk::Result::eSuccess) {
		PrintPipelineStats(graphics, pipeline.pipeline, background ? "draw-background" : "draw", stats_vs, stats_ps);
	}
	if (graphics_debug_dump_enabled()) {
		LOGF("PipelineTrace: vkCreateGraphicsPipelines done result=%s pipeline=%p\n",
		     vk::to_string(result).c_str(), static_cast<void*>(pipeline.pipeline));
	}
	if (result != vk::Result::eSuccess) {
		std::fprintf(stderr, "vkCreateGraphicsPipelines failed%s: %s VS id=%llu PS id=%llu\n",
		             background ? " (background)" : "", vk::to_string(result).c_str(),
		             static_cast<unsigned long long>(vertex_program.id),
		             static_cast<unsigned long long>(ps_active ? pixel_program.id : 0));
		std::fflush(stderr);
	}
	if (background && (result != vk::Result::eSuccess || pipeline.pipeline == nullptr)) {
		pipeline.pipeline = nullptr;
		destroy_rect_list_modules();
		return;
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	EXIT_NOT_IMPLEMENTED(pipeline.pipeline == nullptr);

	if (tess_control_shader_module != nullptr) {
		graphics.device.destroyShaderModule(tess_control_shader_module, nullptr);
	}
	if (tess_eval_shader_module != nullptr) {
		graphics.device.destroyShaderModule(tess_eval_shader_module, nullptr);
	}
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderComputeInputInfo& input_info,
                            vk::ShaderModule compute_module, vk::PipelineCache driver_cache) {
	EXIT_IF(compute_module == nullptr);

	vk::PipelineShaderStageCreateInfo                     comp_shader_stage_info {};
	vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo comp_subgroup_size {};
	comp_shader_stage_info.stage  = vk::ShaderStageFlagBits::eCompute;
	comp_shader_stage_info.module = compute_module;
	comp_shader_stage_info.pName  = "main";
	EXIT_IF(!input_info.stage);
	const auto wave_size = input_info.stage.program->wave_size;
	if (graphics.compute_subgroup_size_control_enabled &&
	    wave_size >= graphics.min_subgroup_size && wave_size <= graphics.max_subgroup_size) {
		comp_subgroup_size.requiredSubgroupSize = wave_size;
		comp_shader_stage_info.pNext            = &comp_subgroup_size;
	}

	// A background compile (shaderPrecompile.h) gives up where the GPU thread exits (the caller destroys what
	// `pipeline` holds then).
	const bool background = IsBackgroundCompileThread();
	std::vector<vk::DescriptorSetLayoutBinding> descriptor_bindings;
	AddLayoutBindings(descriptor_bindings, *input_info.stage.program,
	                  vk::ShaderStageFlagBits::eCompute);
	pipeline.uses_image_heap = input_info.stage.program->info.uses_image_heap; // upstream 27dc7c281 (never set here)
	pipeline.uses_bindless = UsesBindless(*input_info.stage.program);
	if (!CreateDescriptorLayout(graphics, pipeline, descriptor_bindings) ||
	    (background && pipeline.uses_bindless && graphics.bindless_layout == nullptr)) {
		return;
	}
	const vk::PushConstantRange push_constants {vk::ShaderStageFlagBits::eCompute, 0,
	                                            ShaderRecompiler::IR::NativePushConstantSize};

	std::array<vk::DescriptorSetLayout, 2> set_layouts {};
	uint32_t                               set_layout_count = 0;
	FillSetLayouts(graphics, pipeline, set_layouts, set_layout_count);
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = set_layout_count;
	pipeline_layout_info.pSetLayouts            = set_layouts.data();
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_constants;

	EXIT_IF(pipeline.pipeline_layout != nullptr);

	LOGF("PipelineTrace: vkCreatePipelineLayout CS begin set_layouts=1 push_constants=%u\n",
	     1u);
	auto result = graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                  &pipeline.pipeline_layout);
	LOGF("PipelineTrace: vkCreatePipelineLayout CS done result=%s layout=%p\n",
	     vk::to_string(result).c_str(), static_cast<void*>(pipeline.pipeline_layout));
	if (background && (result != vk::Result::eSuccess || pipeline.pipeline_layout == nullptr)) {
		pipeline.pipeline_layout = nullptr;
		return;
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	EXIT_NOT_IMPLEMENTED(pipeline.pipeline_layout == nullptr);

	vk::ComputePipelineCreateInfo info {};
	if (input_info.stage.program->info.uses_gds) {
		info.flags = vk::PipelineCreateFlagBits::eDispatchBase;
	}
	info.stage             = comp_shader_stage_info;
	info.layout            = pipeline.pipeline_layout;
	info.basePipelineIndex = -1;

	EXIT_IF(pipeline.pipeline != nullptr);

	if (std::FILE* f = Common::EnvRead::ShaderStagesFile()) { std::fprintf(f, "STAGE cs-pipeline-begin\n"); std::fclose(f); }
	LOGF("PipelineTrace: vkCreateComputePipelines begin layout=%p\n",
	     static_cast<void*>(pipeline.pipeline_layout));
	const uint64_t stats_cs   = input_info.stage.program->shader_hash;
	const bool     want_stats = PipelineStatsWanted(graphics, stats_cs, stats_cs);
	if (want_stats) {
		info.flags |= vk::PipelineCreateFlagBits::eCaptureStatisticsKHR;
	}
	const auto create_begin = std::chrono::steady_clock::now();
	result = graphics.device.createComputePipelines(driver_cache, 1, &info, nullptr,
	                                                &pipeline.pipeline);
	// KYTY_TRACE_STARTUP: every compute pipeline creation (GPU thread or background), 50 ms+ ones also as slow.
	if (Common::StartupTrace::Enabled()) {
		const auto ms =
		    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - create_begin).count();
		using Common::StartupTrace::Counter;
		Common::StartupTrace::AddMs(background ? Counter::PipeBackground : Counter::PipeCompute, ms);
		if (ms >= 50.0) Common::StartupTrace::Add(Counter::PipeSlow, 0);
	}
	if (const auto create_ms =
	        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - create_begin).count();
	    (PerfEnabled() && create_ms >= 50.0) || (!background && create_ms >= 200.0)) {
		std::fprintf(stderr, "PIPELINE_TIME %s cs=0x%016llx %.0fms created_ago=%.0fms\n",
		             background ? "compute-background" : "compute",
		             static_cast<unsigned long long>(stats_cs), create_ms, ShaderCreatedMsAgo(stats_cs));
		std::fflush(stderr);
	}
	if (want_stats && result == vk::Result::eSuccess) {
		PrintPipelineStats(graphics, pipeline.pipeline, "dispatch", stats_cs, 0);
	}
	if (std::FILE* f = Common::EnvRead::ShaderStagesFile()) { std::fprintf(f, "STAGE cs-pipeline-done\n"); std::fclose(f); }
	LOGF("PipelineTrace: vkCreateComputePipelines done result=%s pipeline=%p\n",
	     vk::to_string(result).c_str(), static_cast<void*>(pipeline.pipeline));
	if (background && (result != vk::Result::eSuccess || pipeline.pipeline == nullptr)) {
		pipeline.pipeline = nullptr;
		return;
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	EXIT_NOT_IMPLEMENTED(pipeline.pipeline == nullptr);
}

} // namespace Libs::Graphics
