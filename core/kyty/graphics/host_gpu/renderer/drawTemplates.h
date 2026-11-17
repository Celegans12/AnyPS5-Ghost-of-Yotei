#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_DRAWTEMPLATES_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_DRAWTEMPLATES_H_

#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/renderTarget.h"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace Libs::Graphics {

struct ShaderStageRuntime;
struct ShaderVertexInputInfo;

namespace HW {
class Context;
class UserConfig;
class Shader;
}

struct DrawTemplateStage {
	const void* program = nullptr;
	std::vector<ShaderRecompiler::IR::DescriptorValue> image_values;
	std::vector<ShaderRecompiler::IR::DescriptorValue> sampler_values;
	std::vector<TextureBinding> images;
	std::vector<uint32_t>    image_versions;
	std::vector<vk::Sampler> samplers;
	uint64_t image_set_epoch = 0;
	uint64_t touched_tick = 0;
};

struct DrawTemplate {
	uint64_t                         key         = 0;
	uint32_t                         stage_count = 0;
	uint64_t                         refused_tick = 0;
	std::array<DrawTemplateStage, 4> stages;

	void* pipeline = nullptr;
	bool                                                       targets_valid = false;
	RenderState                                                rendering;
	vk::ImageAspectFlags                                       feedback_aspects;
	uint32_t                                                   target_count = 0;
	std::array<uint32_t, RENDER_COLOR_ATTACHMENTS_MAX + 1>     target_images {};
	std::array<uint32_t, RENDER_COLOR_ATTACHMENTS_MAX + 1>     target_versions {};
	std::array<vk::ImageLayout, RENDER_COLOR_ATTACHMENTS_MAX + 1>  target_layouts {};
	std::array<vk::AccessFlags2, RENDER_COLOR_ATTACHMENTS_MAX + 1> target_access {};

	struct CommitSlot {
		uint8_t  stage  = 0;
		uint8_t  kind   = 0;
		uint16_t count  = 0;
		uint32_t offset = 0;
	};
	bool                         commit_valid = false;
	vk::PipelineLayout           commit_layout;
	vk::DescriptorUpdateTemplate commit_template;
	std::vector<uint8_t>         commit_data;
	std::vector<CommitSlot>      commit_slots;
};

class DrawTemplateTable {
public:
	static constexpr uint32_t Size = 1u << 14u;

	DrawTemplateTable(): m_slots(Size) {}

	[[nodiscard]] DrawTemplate* Find(uint64_t key) {
		for (uint32_t i = Index(key), probes = 0; probes < MaxProbes; i = (i + 1) & (Size - 1), probes++) {
			auto* slot = m_slots[i].get();
			if (slot == nullptr || slot->key == 0) {
				return nullptr;
			}
			if (slot->key == key) {
				return slot;
			}
		}
		return nullptr;
	}

	[[nodiscard]] DrawTemplate& Insert(uint64_t key) {
		if (m_count >= Size / 2) {
			Clear();
		}
		for (uint32_t i = Index(key), probes = 0; probes < MaxProbes; i = (i + 1) & (Size - 1), probes++) {
			auto& slot = m_slots[i];
			if (slot == nullptr || slot->key == 0) {
				if (slot == nullptr) {
					slot = std::make_unique<DrawTemplate>();
				}
				slot->key           = key;
				slot->stage_count   = 0;
				slot->pipeline      = nullptr;
				slot->targets_valid = false;
				slot->commit_valid  = false;
				m_count++;
				return *slot;
			}
			if (slot->key == key) {
				return *slot;
			}
		}
		Clear();
		return Insert(key);
	}

	void Clear() {
		for (auto& slot: m_slots) {
			if (slot != nullptr) {
				slot->key = 0;
			}
		}
		m_count = 0;
	}

private:
	static constexpr uint32_t MaxProbes = 64;

	[[nodiscard]] static uint32_t Index(uint64_t key) {
		return static_cast<uint32_t>((key * 0x9e3779b97f4a7c15ull) >> 50u) & (Size - 1);
	}

	std::vector<std::unique_ptr<DrawTemplate>> m_slots;
	uint32_t                                   m_count = 0;
};

[[nodiscard]] uint64_t DrawResourceKey(std::span<const ShaderStageRuntime* const> stages);

[[nodiscard]] uint64_t DrawContextKey(const HW::Context& context);

[[nodiscard]] uint64_t DrawUserConfigKey(const HW::UserConfig& user_config);

[[nodiscard]] uint64_t DrawStateKey(uint64_t resource_key, uint64_t context_key, uint64_t user_config_key,
                                    const HW::Shader& shaders, const ShaderVertexInputInfo& vertex_input);

[[nodiscard]] bool DrawTemplatesEnabled();
[[nodiscard]] bool DrawStateTemplatesEnabled();
[[nodiscard]] bool DrawBuffersEnabled();

}

#endif
