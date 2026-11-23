#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineManifest.h"

#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace Libs::Graphics;

int g_failures = 0;

void Check(bool condition, const char* what) {
	if (!condition) {
		std::printf("FAILED: %s\n", what);
		g_failures++;
	}
}

constexpr ManifestFloatMode Mode(uint8_t float_mode, bool ieee, bool dx10_clamp) {
	return {.float_mode = float_mode, .ieee = ieee, .dx10_clamp = dx10_clamp, .recorded = true};
}

PipelineManifest DrawManifest() {
	PipelineManifest manifest;
	ManifestModule   vertex {.stage = ShaderType::Vertex, .code_hash = 0x1111222233334444ull};
	vertex.input.emplace<ShaderVertexInputInfo>().logical_stage = ShaderType::Vertex;
	ManifestModule pixel {.stage = ShaderType::Pixel, .code_hash = 0x5555666677778888ull};
	pixel.input.emplace<ShaderPixelInputInfo>();
	manifest.modules = {vertex, pixel};
	ManifestPipeline pipeline;
	pipeline.stages         = {0, ManifestPipeline::NoStage, ManifestPipeline::NoStage, 1};
	pipeline.float_modes[0] = Mode(0xc0, false, true);
	pipeline.float_modes[3] = Mode(0xf0, true, false);
	manifest.pipelines      = {pipeline};
	return manifest;
}

std::string Text(std::span<const uint8_t> bytes) {
	return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

std::vector<uint8_t> Bytes(std::string_view text) {
	return {text.begin(), text.end()};
}

std::string Replace(std::string text, std::string_view from, std::string_view to) {
	const auto at = text.find(from);
	if (at != std::string::npos) {
		text.replace(at, from.size(), to);
	}
	return text;
}

void RoundTrip() {
	const auto text = Text(SaveManifest(DrawManifest()));
	Check(text.find(" fv=c0.0.1") != std::string::npos, "the vertex stage's float mode is written");
	Check(text.find(" fp=f0.1.0") != std::string::npos, "the pixel stage's float mode is written");
	Check(text.find(" fh=") == std::string::npos && text.find(" fd=") == std::string::npos,
	      "stages without a module write no float mode");
	PipelineManifest loaded;
	uint32_t         skipped = 0;
	Check(LoadManifest(Bytes(text), loaded, &skipped), "the manifest loads");
	Check(skipped == 0 && loaded.pipelines.size() == 1, "the pipeline loads");
	if (loaded.pipelines.size() == 1) {
		const auto& modes = loaded.pipelines[0].float_modes;
		Check(modes[0] == Mode(0xc0, false, true), "the vertex stage's float mode reads back");
		Check(modes[3] == Mode(0xf0, true, false), "the pixel stage's float mode reads back");
		Check(!modes[1].recorded && !modes[2].recorded, "unrecorded stages stay unrecorded");
	}
	Check(Text(SaveManifest(loaded)) == text, "the manifest writes back the same");
}

void OlderManifest() {
	const auto text = Replace(Text(SaveManifest(DrawManifest())), " fv=c0.0.1 fp=f0.1.0", "");
	Check(text.find(" fv=") == std::string::npos, "the older line has no float modes");
	PipelineManifest loaded;
	uint32_t         skipped = 0;
	Check(LoadManifest(Bytes(text), loaded, &skipped), "a manifest without float modes loads");
	Check(skipped == 0 && loaded.pipelines.size() == 1, "its pipeline loads");
	if (loaded.pipelines.size() == 1) {
		for (const auto& mode: loaded.pipelines[0].float_modes) {
			Check(!mode.recorded, "its float modes are not recorded");
		}
	}
}

void BadFloatModes() {
	const auto text = Text(SaveManifest(DrawManifest()));
	for (const auto& [from, to]: {std::pair {" fp=f0.1.0", " fp=f0.1.0 fh=c0.0.0"},
	                             std::pair {" fp=f0.1.0", " fp=100.1.0"},
	                             std::pair {" fp=f0.1.0", " fp=f0.2.0"},
	                             std::pair {" fp=f0.1.0", " fp=f0.1"}}) {
		PipelineManifest loaded;
		uint32_t         skipped = 0;
		Check(LoadManifest(Bytes(Replace(text, from, to)), loaded, &skipped), "the manifest loads");
		Check(skipped == 1 && loaded.pipelines.empty(), "a pipeline with a bad float mode is left out");
	}
}

void StageRegisters() {
	HW::Shader shaders;
	shaders.SetGsShaderResource1({.float_mode = 0xc0, .dx10_clamp = true});
	shaders.SetHsShaderResource1({.float_mode = 0x30, .ieee_mode = true});
	shaders.SetPsShaderResource1({.float_mode = 0xf0, .dx10_clamp = true, .ieee_mode = true});
	const auto draw = ReadManifestFloatModes(shaders, {1, 0, 0, 2});
	Check(draw[0] == Mode(0xc0, false, true), "a vertex stage reads SPI_SHADER_PGM_RSRC1_GS");
	Check(!draw[1].recorded && !draw[2].recorded, "absent stages read nothing");
	Check(draw[3] == Mode(0xf0, true, true), "a pixel stage reads SPI_SHADER_PGM_RSRC1_PS");
	const auto tessellation = ReadManifestFloatModes(shaders, {1, 2, 3, 0});
	Check(tessellation[0] == Mode(0x30, true, false), "a local stage reads SPI_SHADER_PGM_RSRC1_HS");
	Check(tessellation[1] == Mode(0x30, true, false), "a hull stage reads SPI_SHADER_PGM_RSRC1_HS");
	Check(tessellation[2] == Mode(0xc0, false, true),
	      "an evaluation stage reads SPI_SHADER_PGM_RSRC1_GS");
	Check(!tessellation[3].recorded, "a draw without a pixel stage reads no pixel float mode");
}

}

int main() {
	RoundTrip();
	OlderManifest();
	BadFloatModes();
	StageRegisters();
	if (g_failures != 0) {
		std::printf("kyty_shim_manifest_float_modes: %d check(s) FAILED\n", g_failures);
		return 1;
	}
	std::printf("kyty_shim_manifest_float_modes: ok\n");
	return 0;
}
