#include "graphics/host_gpu/graphicContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"

#include <spirv-tools/libspirv.hpp>
#include <spirv/unified1/spirv.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace Libs::Graphics;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;

constexpr const char* Name = "kyty_shim_mesh_per_primitive_block";

constexpr std::array<uint32_t, 31> MeshWithLayerCode {
    0xbefc03ffu, 0x00002003u, 0xbf900009u, 0x7e0202f3u, 0x7e0402f3u, 0x7e060280u, 0x7e0802f2u,
    0x7e0a02ffu, 0x40400000u, 0x7d840084u, 0x02020b01u, 0x7d840088u, 0x02040b02u, 0x7e0c0280u,
    0x7e0e0281u, 0x7d840084u, 0x020c0f06u, 0x7e1002ffu, 0x00200400u, 0x38101008u, 0x7e1202ffu,
    0x00000801u, 0x38121209u, 0x02101308u, 0xf80000cfu, 0x04030201u, 0xf80000d4u, 0x00060000u,
    0xf8000941u, 0x00000008u, 0xbf810000u,
};

constexpr std::array<uint32_t, 26> MeshWithoutLayerCode {
    0xbefc03ffu, 0x00002003u, 0xbf900009u, 0x7e0202f3u, 0x7e0402f3u, 0x7e060280u, 0x7e0802f2u,
    0x7e0a02ffu, 0x40400000u, 0x7d840084u, 0x02020b01u, 0x7d840088u, 0x02040b02u, 0x7d840084u,
    0x7e1002ffu, 0x00200400u, 0x38101008u, 0x7e1202ffu, 0x00000801u, 0x38121209u, 0x02101308u,
    0xf80000cfu, 0x04030201u, 0xf8000941u, 0x00000008u, 0xbf810000u,
};

constexpr std::array<uint32_t, 72> SolidPixelSpirv {
    0x07230203u, 0x00010300u, 0x0008000bu, 0x0000000cu, 0x00000000u, 0x00020011u, 0x00000001u,
    0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0x00000000u, 0x0003000eu,
    0x00000000u, 0x00000001u, 0x0006000fu, 0x00000004u, 0x00000004u, 0x6e69616du, 0x00000000u,
    0x00000009u, 0x00030010u, 0x00000004u, 0x00000007u, 0x00040047u, 0x00000009u, 0x0000001eu,
    0x00000000u, 0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u,
    0x00000006u, 0x00000020u, 0x00040017u, 0x00000007u, 0x00000006u, 0x00000004u, 0x00040020u,
    0x00000008u, 0x00000003u, 0x00000007u, 0x0004003bu, 0x00000008u, 0x00000009u, 0x00000003u,
    0x0004002bu, 0x00000006u, 0x0000000au, 0x3f800000u, 0x0007002cu, 0x00000007u, 0x0000000bu,
    0x0000000au, 0x0000000au, 0x0000000au, 0x0000000au, 0x00050036u, 0x00000002u, 0x00000004u,
    0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u, 0x0003003eu, 0x00000009u, 0x0000000bu,
    0x000100fdu, 0x00010038u,
};

void Require(bool condition, const std::string& message) {
	if (condition) return;
	std::fprintf(stderr, "%s: FAILED: %s\n", Name, message.c_str());
	std::exit(1);
}

template <class T>
T Checked(vk::ResultValue<T> result, const char* what) {
	Require(result.result == vk::Result::eSuccess,
	        std::string(what) + " failed: " + vk::to_string(result.result));
	return std::move(result.value);
}

void Checked(vk::Result result, const char* what) {
	Require(result == vk::Result::eSuccess, std::string(what) + " failed: " + vk::to_string(result));
}

int NoDevice(const char* why) {
	const char* required = std::getenv("ANYPS5_REQUIRE_VULKAN");
	Require(required == nullptr || std::strcmp(required, "1") != 0,
	        std::string("ANYPS5_REQUIRE_VULKAN=1 and ") + why);
	std::printf("%s_draw: skipped, %s\n", Name, why);
	return 77;
}

std::vector<uint32_t> CompileMesh(bool with_layer, uint32_t host_subgroup_size) {
	std::vector<uint32_t> code;
	if (with_layer) {
		code.assign(MeshWithLayerCode.begin(), MeshWithLayerCode.end());
	} else {
		code.assign(MeshWithoutLayerCode.begin(), MeshWithoutLayerCode.end());
	}

	ShaderVertexInputInfo vertex {};
	vertex.logical_stage              = ShaderType::Mesh;
	vertex.wave_size                  = 64u;
	vertex.pa_cl_vs_out_cntl          = with_layer ? ((1u << 21u) | (1u << 18u)) : 0u;
	vertex.mesh.threads_num[0]        = 64u;
	vertex.mesh.threads_num[1]        = 1u;
	vertex.mesh.threads_num[2]        = 1u;
	vertex.mesh.host_subgroup_size    = host_subgroup_size;
	vertex.mesh.vertices_per_group    = 3u;
	vertex.mesh.primitives_per_group  = 2u;
	vertex.mesh.max_vertices          = 64u;
	vertex.mesh.max_primitives        = 2u;
	vertex.mesh.input_primitive       = static_cast<uint32_t>(Prospero::PrimitiveType::kPointList);
	vertex.mesh.output_primitive_size = 3u;
	vertex.mesh.provoking_vertex      = 0u;
	vertex.mesh.fast_launch           = false;

	std::array<uint32_t, 2>    user_data {};
	Recompiler::Spirv::Profile profile {};
	Recompiler::CompileOptions options {};
	options.stage             = ShaderType::Mesh;
	options.wave_size         = 64u;
	options.user_data_base    = 8u;
	options.user_data         = user_data;
	options.shader_hash       = with_layer ? 0x6d65736800000001ull : 0x6d65736800000002ull;
	options.profile           = &profile;
	options.input_info.vertex = &vertex;

	const auto                       source     = Recompiler::PrepareShaderSource(code, options);
	auto                             translated = Recompiler::TranslateProgram(source.decoded, options);
	const auto                       plan       = Recompiler::IR::ExtractResourcePlan(translated.program);
	const Recompiler::IR::SrtRuntime runtime {};
	Recompiler::IR::ResourceSnapshot       snapshot;
	Recompiler::IR::ResourceSpecialization specialization;
	Require(Recompiler::IR::MaterializeResources(plan, runtime, snapshot, specialization),
	        "failed to materialize mesh resources");
	auto compiled = Recompiler::CompileProgram(std::move(translated), options, specialization,
	                                           Recompiler::IR::PushData::MeshDrawDwordCount);

	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer(
	    [&messages](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
		    messages += message;
		    messages += '\n';
	    });
	spvtools::ValidatorOptions validator;
	validator.SetUniformBufferStandardLayout(true);
	validator.SetScalarBlockLayout(true);
	Require(tools.Validate(compiled.spirv.data(), compiled.spirv.size(), validator),
	        std::string("SPIR-V validation failed:\n") + messages);
	return compiled.spirv;
}

struct PerPrimitiveLayout {
	std::set<uint32_t>                                      block_structs;
	std::map<uint32_t, std::vector<uint32_t>>               struct_members;
	std::map<std::pair<uint32_t, uint32_t>, spv::BuiltIn>   member_builtins;
	std::set<std::pair<uint32_t, uint32_t>>                 member_per_primitive;
	uint32_t                                                standalone_layer = 0;
	uint32_t                                                standalone_cull  = 0;
};

PerPrimitiveLayout Inspect(const std::vector<uint32_t>& words) {
	PerPrimitiveLayout layout;
	for (size_t at = 5; at < words.size();) {
		const auto count  = words[at] >> 16u;
		const auto opcode = static_cast<spv::Op>(words[at] & 0xffffu);
		if (count == 0 || at + count > words.size()) break;
		const auto word = [&](uint32_t index) { return words[at + index]; };
		if (opcode == spv::OpTypeStruct && count >= 3) {
			std::vector<uint32_t> members;
			for (uint32_t i = 2; i < count; i++) members.push_back(word(i));
			layout.struct_members[word(1)] = std::move(members);
		} else if (opcode == spv::OpDecorate && count >= 3) {
			const auto target = word(1);
			const auto deco   = static_cast<spv::Decoration>(word(2));
			if (deco == spv::DecorationBlock) {
				layout.block_structs.insert(target);
			} else if (deco == spv::DecorationBuiltIn && count >= 4) {
				const auto builtin = static_cast<spv::BuiltIn>(word(3));
				if (builtin == spv::BuiltInLayer) layout.standalone_layer++;
				if (builtin == spv::BuiltInCullPrimitiveEXT) layout.standalone_cull++;
			}
		} else if (opcode == spv::OpMemberDecorate && count >= 4) {
			const auto target = word(1);
			const auto member = word(2);
			const auto deco   = static_cast<spv::Decoration>(word(3));
			if (deco == spv::DecorationBuiltIn && count >= 5) {
				layout.member_builtins[{target, member}] = static_cast<spv::BuiltIn>(word(4));
			} else if (deco == spv::DecorationPerPrimitiveEXT) {
				layout.member_per_primitive.insert({target, member});
			}
		}
		at += count;
	}
	return layout;
}

void VerifyCodegen() {
	for (const uint32_t subgroup: {32u, 64u}) {
		const auto with_layer = Inspect(CompileMesh(true, subgroup));
		Require(with_layer.standalone_layer == 0u, "BuiltIn Layer must not be a standalone output variable");
		Require(with_layer.standalone_cull == 0u,
		        "BuiltIn CullPrimitiveEXT must not be a standalone output variable");
		uint32_t block_id = 0;
		for (const auto id: with_layer.block_structs) {
			const auto cull_it  = with_layer.member_builtins.find({id, 0u});
			const auto layer_it = with_layer.member_builtins.find({id, 1u});
			if (cull_it != with_layer.member_builtins.end() &&
			    cull_it->second == spv::BuiltInCullPrimitiveEXT &&
			    layer_it != with_layer.member_builtins.end() &&
			    layer_it->second == spv::BuiltInLayer) {
				block_id = id;
				break;
			}
		}
		Require(block_id != 0u, "missing Block struct carrying CullPrimitiveEXT and Layer");
		Require(with_layer.member_per_primitive.contains({block_id, 0u}),
		        "CullPrimitiveEXT member missing PerPrimitiveEXT decoration");
		Require(with_layer.member_per_primitive.contains({block_id, 1u}),
		        "Layer member missing PerPrimitiveEXT decoration");

		const auto without_layer = Inspect(CompileMesh(false, subgroup));
		Require(without_layer.standalone_layer == 0u, "unexpected standalone Layer without layer export");
		Require(without_layer.standalone_cull == 0u,
		        "BuiltIn CullPrimitiveEXT must stay in a Block struct without layer export");
		uint32_t cull_block = 0;
		for (const auto id: without_layer.block_structs) {
			const auto cull_it = without_layer.member_builtins.find({id, 0u});
			if (cull_it != without_layer.member_builtins.end() &&
			    cull_it->second == spv::BuiltInCullPrimitiveEXT) {
				cull_block = id;
				break;
			}
		}
		Require(cull_block != 0u, "missing Block struct carrying CullPrimitiveEXT without layer export");
		Require(without_layer.struct_members.at(cull_block).size() == 1u,
		        "per-primitive Block struct without layer export should have one member");
		Require(without_layer.member_per_primitive.contains({cull_block, 0u}),
		        "CullPrimitiveEXT member missing PerPrimitiveEXT decoration without layer export");
	}
}

int DrawMeshPerPrimitiveBlock() {
	const vk::detail::DynamicLoader loader;
	if (!loader.success()) return NoDevice("the Vulkan loader is missing");
	auto& dispatch = VULKAN_HPP_DEFAULT_DISPATCHER;
	dispatch.init(loader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr"));
	vk::ApplicationInfo application;
	application.apiVersion = VK_API_VERSION_1_3;
	vk::InstanceCreateInfo instance_info;
	instance_info.pApplicationInfo = &application;
	const auto created             = vk::createInstance(instance_info);
	if (created.result != vk::Result::eSuccess) return NoDevice("no Vulkan 1.3 instance");
	const auto instance = created.value;
	dispatch.init(instance);

	vk::PhysicalDevice physical;
	uint32_t           family           = 0;
	uint32_t           subgroup         = 0;
	bool               has_gpl          = false;
	bool               skipped_llvmpipe = false;
	std::string        llvmpipe_info;
	for (const auto candidate: Checked(instance.enumeratePhysicalDevices(), "device enumeration")) {
		const auto extensions = Checked(candidate.enumerateDeviceExtensionProperties(nullptr),
		                                "device extension enumeration");
		const auto has_ext    = [&](const char* ext) {
			return std::ranges::any_of(extensions, [ext](const auto& props) {
				return std::string_view(props.extensionName.data()) == ext;
			});
		};
		if (!has_ext(VK_EXT_MESH_SHADER_EXTENSION_NAME)) continue;
		const bool candidate_gpl = has_ext(VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME) &&
		                           has_ext(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);

		vk::PhysicalDeviceMeshShaderFeaturesEXT             mesh_features;
		vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT gpl_features;
		vk::PhysicalDeviceVulkan12Features                  v12_features;
		vk::PhysicalDeviceVulkan13Features                  v13_features;
		vk::PhysicalDeviceFeatures2                         features;
		v13_features.pNext  = &v12_features;
		v12_features.pNext  = &mesh_features;
		mesh_features.pNext = candidate_gpl ? &gpl_features : nullptr;
		features.pNext      = &v13_features;
		candidate.getFeatures2(&features);

		const auto families = candidate.getQueueFamilyProperties();
		const auto graphics = std::ranges::find_if(families, [](const auto& queue) {
			return static_cast<bool>(queue.queueFlags & vk::QueueFlagBits::eGraphics);
		});
		if (!mesh_features.meshShader || !v12_features.shaderOutputLayer ||
		    !v12_features.bufferDeviceAddress || !v12_features.uniformBufferStandardLayout ||
		    !v13_features.dynamicRendering || !features.features.shaderInt64 ||
		    graphics == families.end()) {
			continue;
		}
		vk::PhysicalDeviceDriverProperties      driver_props;
		vk::PhysicalDeviceVulkan11Properties    props11;
		vk::PhysicalDeviceMeshShaderPropertiesEXT mesh_props;
		vk::PhysicalDeviceProperties2           props;
		driver_props.pNext = &mesh_props;
		props11.pNext      = &driver_props;
		props.pNext        = &props11;
		candidate.getProperties2(&props);
		if (mesh_props.maxMeshWorkGroupInvocations < 64u ||
		    mesh_props.maxMeshWorkGroupSize[0] < 64u || mesh_props.maxMeshOutputVertices < 64u ||
		    mesh_props.maxMeshOutputPrimitives < 2u || props11.subgroupSize == 0u) {
			continue;
		}
		if (driver_props.driverID == vk::DriverId::eMesaLlvmpipe &&
		    props.properties.driverVersion <= VK_MAKE_API_VERSION(0, 26, 2, 4)) {
			skipped_llvmpipe = true;
			llvmpipe_info    = driver_props.driverInfo.data();
			continue;
		}
		const bool software = props.properties.deviceType == vk::PhysicalDeviceType::eCpu;
		if (physical && software) continue;
		physical = candidate;
		family   = static_cast<uint32_t>(graphics - families.begin());
		subgroup = props11.subgroupSize;
		has_gpl  = candidate_gpl && gpl_features.graphicsPipelineLibrary;
		if (!software) break;
	}
	if (!physical) {
		instance.destroy();
		if (skipped_llvmpipe) {
			std::printf("%s_draw: skipped, the device has no mesh shader compiler that handles "
			            "these programs (llvmpipe %s)\n",
			            Name, llvmpipe_info.c_str());
			return 77;
		}
		return NoDevice("no device with VK_EXT_mesh_shader, shaderOutputLayer, and dynamic rendering");
	}

	const auto mesh_with_layer    = CompileMesh(true, subgroup);
	const auto mesh_without_layer = CompileMesh(false, subgroup);

	std::vector<const char*> enabled_extensions {VK_EXT_MESH_SHADER_EXTENSION_NAME};
	if (has_gpl) {
		enabled_extensions.push_back(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
		enabled_extensions.push_back(VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME);
	}
	const float               priority = 1.0f;
	vk::DeviceQueueCreateInfo queue_info;
	queue_info.queueFamilyIndex = family;
	queue_info.queueCount       = 1;
	queue_info.pQueuePriorities = &priority;

	vk::PhysicalDeviceMeshShaderFeaturesEXT              mesh_features;
	mesh_features.meshShader = VK_TRUE;
	vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT gpl_features;
	gpl_features.graphicsPipelineLibrary = VK_TRUE;
	vk::PhysicalDeviceVulkan12Features v12_features;
	v12_features.shaderOutputLayer           = VK_TRUE;
	v12_features.bufferDeviceAddress         = VK_TRUE;
	v12_features.uniformBufferStandardLayout = VK_TRUE;
	vk::PhysicalDeviceVulkan13Features v13_features;
	v13_features.dynamicRendering = VK_TRUE;
	v13_features.pNext            = &v12_features;
	v12_features.pNext            = &mesh_features;
	mesh_features.pNext           = has_gpl ? &gpl_features : nullptr;
	vk::PhysicalDeviceFeatures2 features;
	features.features.shaderInt64 = VK_TRUE;
	features.pNext                = &v13_features;

	vk::DeviceCreateInfo device_info;
	device_info.pNext                   = &features;
	device_info.queueCreateInfoCount    = 1;
	device_info.pQueueCreateInfos       = &queue_info;
	device_info.enabledExtensionCount   = static_cast<uint32_t>(enabled_extensions.size());
	device_info.ppEnabledExtensionNames = enabled_extensions.data();
	const auto device                   = Checked(physical.createDevice(device_info), "device creation");
	dispatch.init(device);

	vk::PhysicalDeviceDriverProperties driver;
	vk::PhysicalDeviceProperties2      properties;
	properties.pNext = &driver;
	physical.getProperties2(&properties);
	const auto gpu_name = std::string(properties.properties.deviceName.data()) + " (" +
	                      driver.driverName.data() + ", " + driver.driverInfo.data() + ")";
	const auto queue = device.getQueue(family, 0);

	const auto make_module = [&](const std::vector<uint32_t>& words, const char* what) {
		vk::ShaderModuleCreateInfo info;
		info.codeSize = words.size() * sizeof(uint32_t);
		info.pCode    = words.data();
		return Checked(device.createShaderModule(info), what);
	};
	const auto layer_mesh_module = make_module(mesh_with_layer, "layered mesh shader module");
	const auto cull_mesh_module  = make_module(mesh_without_layer, "cull-only mesh shader module");
	vk::ShaderModuleCreateInfo pixel_mod_info;
	pixel_mod_info.codeSize = SolidPixelSpirv.size() * sizeof(uint32_t);
	pixel_mod_info.pCode    = SolidPixelSpirv.data();
	const auto pixel_module = Checked(device.createShaderModule(pixel_mod_info), "pixel shader module");

	vk::PushConstantRange push_range;
	push_range.stageFlags = vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eFragment;
	push_range.offset     = 0;
	push_range.size       = Recompiler::IR::NativePushConstantSize;
	vk::PipelineLayoutCreateInfo layout_info;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push_range;
	const auto layout = Checked(device.createPipelineLayout(layout_info), "pipeline layout");

	constexpr uint32_t Side       = 4u;
	constexpr uint32_t Layers     = 2u;
	constexpr uint32_t Pixels     = Side * Side;
	constexpr uint32_t LayerBytes = Pixels * 4u;
	constexpr auto     Format     = vk::Format::eR8G8B8A8Unorm;

	vk::Viewport viewport {0.0f, 0.0f, static_cast<float>(Side), static_cast<float>(Side), 0.0f, 1.0f};
	vk::Rect2D   scissor {{0, 0}, {Side, Side}};
	vk::PipelineViewportStateCreateInfo viewports;
	viewports.viewportCount = 1;
	viewports.pViewports    = &viewport;
	viewports.scissorCount  = 1;
	viewports.pScissors     = &scissor;

	vk::PipelineRasterizationStateCreateInfo raster;
	raster.polygonMode = vk::PolygonMode::eFill;
	raster.cullMode    = vk::CullModeFlagBits::eNone;
	raster.lineWidth   = 1.0f;

	vk::PipelineMultisampleStateCreateInfo multisample;
	multisample.rasterizationSamples = vk::SampleCountFlagBits::e1;

	vk::PipelineColorBlendAttachmentState blend_target;
	blend_target.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
	                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
	vk::PipelineColorBlendStateCreateInfo blend;
	blend.attachmentCount = 1;
	blend.pAttachments    = &blend_target;

	const auto build_pipeline = [&](vk::ShaderModule mesh_mod, const char* label) {
		vk::PipelineShaderStageCreateInfo mesh_stage;
		mesh_stage.stage  = vk::ShaderStageFlagBits::eMeshEXT;
		mesh_stage.module = mesh_mod;
		mesh_stage.pName  = "main";

		if (has_gpl) {
			vk::PipelineRenderingCreateInfo          view_only;
			vk::GraphicsPipelineLibraryCreateInfoEXT lib_info;
			lib_info.flags = vk::GraphicsPipelineLibraryFlagBitsEXT::ePreRasterizationShaders;
			lib_info.pNext = &view_only;
			vk::GraphicsPipelineCreateInfo lib_pipe_info;
			lib_pipe_info.pNext               = &lib_info;
			lib_pipe_info.flags               = vk::PipelineCreateFlagBits::eLibraryKHR;
			lib_pipe_info.stageCount          = 1;
			lib_pipe_info.pStages             = &mesh_stage;
			lib_pipe_info.pViewportState      = &viewports;
			lib_pipe_info.pRasterizationState = &raster;
			lib_pipe_info.layout              = layout;
			const auto library = Checked(device.createGraphicsPipeline(nullptr, lib_pipe_info),
			                             (std::string("pre-rasterization library: ") + label).c_str());
			device.destroyPipeline(library);
		}

		std::array<vk::PipelineShaderStageCreateInfo, 2> stages {};
		stages[0]        = mesh_stage;
		stages[1].stage  = vk::ShaderStageFlagBits::eFragment;
		stages[1].module = pixel_module;
		stages[1].pName  = "main";

		vk::PipelineRenderingCreateInfo rendering;
		rendering.colorAttachmentCount    = 1;
		rendering.pColorAttachmentFormats = &Format;

		vk::GraphicsPipelineCreateInfo pipeline_info;
		pipeline_info.pNext               = &rendering;
		pipeline_info.stageCount          = static_cast<uint32_t>(stages.size());
		pipeline_info.pStages             = stages.data();
		pipeline_info.pViewportState      = &viewports;
		pipeline_info.pRasterizationState = &raster;
		pipeline_info.pMultisampleState   = &multisample;
		pipeline_info.pColorBlendState    = &blend;
		pipeline_info.layout              = layout;
		return Checked(device.createGraphicsPipeline(nullptr, pipeline_info),
		               (std::string("whole mesh pipeline: ") + label).c_str());
	};

	const auto layer_pipeline = build_pipeline(layer_mesh_module, "with layer");
	const auto cull_pipeline  = build_pipeline(cull_mesh_module, "without layer");

	const auto memory_types = physical.getMemoryProperties();
	const auto allocate     = [&](const vk::MemoryRequirements& needs, vk::MemoryPropertyFlags flags) {
		for (uint32_t type = 0; type < memory_types.memoryTypeCount; type++) {
			if (((needs.memoryTypeBits >> type) & 1u) == 0u ||
			    (memory_types.memoryTypes[type].propertyFlags & flags) != flags) {
				continue;
			}
			vk::MemoryAllocateInfo info;
			info.allocationSize  = needs.size;
			info.memoryTypeIndex = type;
			return Checked(device.allocateMemory(info), "memory allocation");
		}
		Require(false, "no suitable memory type");
		return vk::DeviceMemory {};
	};

	vk::ImageCreateInfo image_info;
	image_info.imageType   = vk::ImageType::e2D;
	image_info.format      = Format;
	image_info.extent      = vk::Extent3D {Side, Side, 1u};
	image_info.mipLevels   = 1u;
	image_info.arrayLayers = Layers;
	image_info.samples     = vk::SampleCountFlagBits::e1;
	image_info.tiling      = vk::ImageTiling::eOptimal;
	image_info.usage       = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc;
	const auto image        = Checked(device.createImage(image_info), "image creation");
	const auto image_memory = allocate(device.getImageMemoryRequirements(image),
	                                   vk::MemoryPropertyFlagBits::eDeviceLocal);
	Checked(device.bindImageMemory(image, image_memory, 0), "image bind");

	const vk::ImageSubresourceRange whole {vk::ImageAspectFlagBits::eColor, 0u, 1u, 0u, Layers};
	vk::ImageViewCreateInfo         view_info;
	view_info.image            = image;
	view_info.viewType         = vk::ImageViewType::e2DArray;
	view_info.format           = Format;
	view_info.subresourceRange = whole;
	const auto view            = Checked(device.createImageView(view_info), "image view creation");

	vk::BufferCreateInfo buffer_info;
	buffer_info.size  = LayerBytes * Layers;
	buffer_info.usage = vk::BufferUsageFlagBits::eTransferDst;
	const auto buffer = Checked(device.createBuffer(buffer_info), "readback buffer creation");
	const auto buffer_memory =
	    allocate(device.getBufferMemoryRequirements(buffer),
	             vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	Checked(device.bindBufferMemory(buffer, buffer_memory, 0), "buffer bind");

	vk::CommandPoolCreateInfo pool_info;
	pool_info.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	pool_info.queueFamilyIndex = family;
	const auto pool            = Checked(device.createCommandPool(pool_info), "command pool");
	vk::CommandBufferAllocateInfo command_info;
	command_info.commandPool        = pool;
	command_info.commandBufferCount = 1;
	const auto command =
	    Checked(device.allocateCommandBuffers(command_info), "command buffer").front();

	const auto run_draw = [&](vk::Pipeline pipeline, bool cull0, bool cull1) -> std::pair<uint32_t, uint32_t> {
		const auto move = [&](vk::ImageLayout from, vk::ImageLayout to, vk::AccessFlags source,
		                      vk::AccessFlags target, vk::PipelineStageFlags before,
		                      vk::PipelineStageFlags after) {
			vk::ImageMemoryBarrier barrier;
			barrier.srcAccessMask       = source;
			barrier.dstAccessMask       = target;
			barrier.oldLayout           = from;
			barrier.newLayout           = to;
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.image               = image;
			barrier.subresourceRange    = whole;
			command.pipelineBarrier(before, after, {}, 0, nullptr, 0, nullptr, 1, &barrier);
		};

		Checked(command.reset(), "command reset");
		Checked(command.begin(vk::CommandBufferBeginInfo {}), "command begin");
		move(vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, {},
		     vk::AccessFlagBits::eColorAttachmentWrite, vk::PipelineStageFlagBits::eTopOfPipe,
		     vk::PipelineStageFlagBits::eColorAttachmentOutput);
		vk::RenderingAttachmentInfo target;
		target.imageView   = view;
		target.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
		target.loadOp      = vk::AttachmentLoadOp::eClear;
		target.storeOp     = vk::AttachmentStoreOp::eStore;
		vk::RenderingInfo pass;
		pass.renderArea           = scissor;
		pass.layerCount           = Layers;
		pass.colorAttachmentCount = 1;
		pass.pColorAttachments    = &target;
		command.beginRendering(&pass);
		command.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
		std::array<uint32_t, Recompiler::IR::NativePushConstantSize / 4u> push {};
		push[0]                                      = 3u;
		push[Recompiler::IR::PushData::MeshDrawDwordCount + 0u] = cull0 ? 0x80000000u : 0u;
		push[Recompiler::IR::PushData::MeshDrawDwordCount + 1u] = cull1 ? 0x80000000u : 0u;
		command.pushConstants(layout, push_range.stageFlags, 0, Recompiler::IR::NativePushConstantSize,
		                      push.data());
		command.drawMeshTasksEXT(1u, 1u, 1u);
		command.endRendering();
		move(vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferSrcOptimal,
		     vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eTransferRead,
		     vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eTransfer);
		vk::BufferImageCopy copy;
		copy.imageSubresource = vk::ImageSubresourceLayers {vk::ImageAspectFlagBits::eColor, 0u, 0u, Layers};
		copy.imageExtent      = vk::Extent3D {Side, Side, 1u};
		command.copyImageToBuffer(image, vk::ImageLayout::eTransferSrcOptimal, buffer, 1, &copy);
		Checked(command.end(), "command end");

		vk::SubmitInfo submit;
		submit.commandBufferCount = 1;
		submit.pCommandBuffers    = &command;
		Checked(queue.submit(1, &submit, nullptr), "queue submit");
		Checked(device.waitIdle(), "device waitIdle");

		const auto* bytes =
		    static_cast<const uint8_t*>(Checked(device.mapMemory(buffer_memory, 0, LayerBytes * Layers),
		                                        "readback map"));
		uint32_t layer0_lit = 0;
		uint32_t layer1_lit = 0;
		for (uint32_t i = 0; i < Pixels; i++) {
			if (bytes[i * 4u] != 0u) layer0_lit++;
			if (bytes[LayerBytes + i * 4u] != 0u) layer1_lit++;
		}
		device.unmapMemory(buffer_memory);
		return {layer0_lit, layer1_lit};
	};

	const auto expect = [&](const char* what, std::pair<uint32_t, uint32_t> got, uint32_t want0,
	                        uint32_t want1) {
		Require(got.first == want0 && got.second == want1,
		        std::string(what) + ": got layer0=" + std::to_string(got.first) + " layer1=" +
		            std::to_string(got.second) + ", expected " + std::to_string(want0) + "/" +
		            std::to_string(want1));
	};

	expect("layered dynamic cull (0 culled, 1 visible)", run_draw(layer_pipeline, true, false), 0u, Pixels);
	expect("layered dynamic cull (0 visible, 1 culled)", run_draw(layer_pipeline, false, true), Pixels, 0u);
	expect("layered dynamic cull (both culled)", run_draw(layer_pipeline, true, true), 0u, 0u);
	expect("layered dynamic cull (both visible)", run_draw(layer_pipeline, false, false), Pixels, Pixels);

	expect("cull-only dynamic cull (both culled)", run_draw(cull_pipeline, true, true), 0u, 0u);
	expect("cull-only dynamic cull (0 culled, 1 visible)", run_draw(cull_pipeline, true, false), Pixels, 0u);
	expect("cull-only dynamic cull (0 visible, 1 culled)", run_draw(cull_pipeline, false, true), Pixels, 0u);

	device.destroyCommandPool(pool);
	device.destroyBuffer(buffer);
	device.freeMemory(buffer_memory);
	device.destroyImageView(view);
	device.destroyImage(image);
	device.freeMemory(image_memory);
	device.destroyPipeline(cull_pipeline);
	device.destroyPipeline(layer_pipeline);
	device.destroyPipelineLayout(layout);
	device.destroyShaderModule(pixel_module);
	device.destroyShaderModule(cull_mesh_module);
	device.destroyShaderModule(layer_mesh_module);
	device.destroy();
	instance.destroy();
	std::printf("%s_draw: ok, subgroup=%u gpl=%s dynamic-cull+layer passed on %s\n", Name, subgroup,
	            has_gpl ? "on" : "off", gpu_name.c_str());
	return 0;
}

}

int main(int argc, char** argv) {
	if (argc > 1 && std::strcmp(argv[1], "draw") == 0) return DrawMeshPerPrimitiveBlock();
	VerifyCodegen();
	std::printf("%s: ok\n", Name);
	return 0;
}
