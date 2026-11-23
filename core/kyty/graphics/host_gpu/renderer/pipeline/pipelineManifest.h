#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINEMANIFEST_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINEMANIFEST_H_

#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include <array>
#include <charconv>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace Libs::Graphics {

namespace ManifestText {

inline bool ParseNumber(std::string_view text, uint64_t& value, int base = 10) {
	if (text.empty()) return false;
	const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value, base);
	return error == std::errc() && end == text.data() + text.size();
}

inline std::vector<std::string_view> Split(std::string_view text, char separator) {
	std::vector<std::string_view> parts;
	for (size_t start = 0;;) {
		const auto end = text.find(separator, start);
		parts.push_back(text.substr(start, end == std::string_view::npos ? end : end - start));
		if (end == std::string_view::npos) return parts;
		start = end + 1;
	}
}

}

inline constexpr uint32_t ManifestFormat = 1;

struct ManifestModule {
	ShaderType stage = ShaderType::Unknown;
	uint64_t code_hash        = 0;
	uint64_t back_hash        = 0;
	uint32_t user_data_count  = 0;
	uint32_t push_data_cursor = 0;
	std::variant<ShaderVertexInputInfo, ShaderPixelInputInfo, ShaderComputeInputInfo> input;
	ShaderRecompiler::IR::ResourceSpecialization specialization;
	bool from_assets = false;

	[[nodiscard]] uint64_t ProgramHash() const;
};

struct ManifestPipeline {
	static constexpr uint32_t NoStage = UINT32_MAX;

	std::array<uint32_t, 4> stages {NoStage, NoStage, NoStage, NoStage};
	bool plain_pixel = false;
	GraphicsPipelineKey key {};
	ManifestFloatModes  float_modes {};
	bool from_assets = false;
};

struct PipelineManifest {
	std::vector<ManifestModule>                                   modules;
	std::vector<ManifestPipeline>                                 pipelines;
	std::vector<std::pair<uint64_t, ShaderTessellationInputInfo>> tessellation_layouts;
};

void NormalizeManifestInput(ManifestModule& module);

[[nodiscard]] std::vector<uint8_t> SaveManifest(const PipelineManifest& manifest);
bool LoadManifest(std::span<const uint8_t> bytes, PipelineManifest& manifest,
                  uint32_t* skipped = nullptr, std::vector<std::string>* unlinked = nullptr);

[[nodiscard]] uint64_t ManifestModuleIdentity(const ManifestModule&     module,
                                              std::span<const uint32_t> static_state);
[[nodiscard]] uint64_t ManifestPipelineIdentity(const ManifestPipeline& pipeline);
[[nodiscard]] ManifestFloatModes ReadManifestFloatModes(const HW::Shader&             shaders,
                                                        const std::array<uint64_t, 4>& stage_ids);

}

#endif
