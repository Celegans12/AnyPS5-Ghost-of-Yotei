#pragma once

#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/rectListShader.h"
#include "graphics/shader/shader.h"

#include <memory>

namespace Libs::Graphics {

struct ShaderProgramKey {
	ShaderType            stage           = ShaderType::Unknown;
	uint64_t              hash            = 0;
	uint32_t              user_data_count = 0;
	std::vector<uint32_t> static_state;

	bool operator==(const ShaderProgramKey&) const = default;
};

struct ShaderModule {
	uint64_t                                     id     = 0;
	vk::ShaderModule                             module = nullptr;
	ShaderRecompiler::IR::CompiledShaderInfo     info;
	ShaderRecompiler::IR::ResourceSpecialization specialization;
	std::vector<uint32_t>                        function_code;
	std::vector<vk::Format>                      vertex_formats;
	std::vector<RectListParameter>               rect_parameters;
	bool                                         mesh_task = false;
	uint32_t                                     spirv_words = 0;
	std::shared_ptr<const ShaderModule>          plain;
	bool                                         guest_traversal = false;
	const ShaderRecompiler::IR::ResourcePlan*    plan = nullptr;
#if defined(ANYPS5_KYTY)
	mutable bool                                 pipeline_requested = false;
	bool                                         shared_variant     = false;
#endif
};

inline constexpr uint64_t PlainModuleIdXor = 0x9e3779b97f4a7c15ull;

}
