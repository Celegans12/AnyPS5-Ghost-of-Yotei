#include "graphics/host_gpu/renderer/pipeline/pipelineCacheTrace.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/envRead.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/shaderEnv.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"

#include <fmt/format.h>
#include <spirv-tools/libspirv.hpp>

#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <xxhash.h>

// The shader dumps and traces of the pipeline cache's compile path: every one is off unless its name is set (a compile is
// no hot path; the gates are read where the tool runs).
namespace Libs::Graphics::PipelineTrace {

void DumpShaderSpirv(const char* stage_name, uint64_t shader_hash,
                     const std::vector<uint32_t>& spirv) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / fmt::format("{:04d}_new_shader_{}_{:016x}.spv",
	                                                             id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(spirv.data(), spirv.size() * sizeof(uint32_t));
}

void DumpShaderOriginal(const char* stage_name, uint64_t shader_hash,
                        std::span<const uint32_t> code) {
	// KYTY_TRACE_SHADER_CODE_DIR=<dir> (diagnostic): raw guest code of each compiled shader.
	if (const char* dir = Common::EnvRead::ShaderCodeDir(); dir != nullptr && !code.empty()) {
		const auto name = fmt::format("{}/{}_{:016x}.bin", dir, stage_name, shader_hash);
		if (FILE* f = std::fopen(name.c_str(), "wb")) {
			std::fwrite(code.data(), 1, code.size_bytes(), f);
			std::fclose(f);
		}
		// KYTY_TRACE_DISASM=<hex,...>: also the decoded RDNA2 text of the listed shaders, written here
		// so it does not depend on a pipeline cache miss: <dir>/<stage>_<hash>.asm.
		if (const char* list = std::getenv("KYTY_TRACE_DISASM");
		    list != nullptr && std::string_view(list).find(fmt::format("{:016x}", shader_hash)) != std::string_view::npos) {
			ShaderRecompiler::Decoder::Program decoded;
			ShaderRecompiler::Decoder::DecodeProgram(code, decoded);
			const auto text = ShaderRecompiler::Decoder::ProgramToString(decoded);
			const auto asm_name = fmt::format("{}/{}_{:016x}.asm", dir, stage_name, shader_hash);
			if (FILE* f = std::fopen(asm_name.c_str(), "w")) {
				std::fwrite(text.data(), 1, text.size(), f);
				std::fclose(f);
			}
		}
	}
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	EXIT_IF(code.empty());
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / "original" /
	                  fmt::format("{:04d}_new_shader_{}_{:016x}.bin", id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(code.data(), code.size_bytes());
}

bool ValidateShaderSpirv(const char* label, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv, bool maintenance8_enabled) {
	if (!Config::ShaderValidationEnabled()) {
		return true;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	spvtools::ValidatorOptions validator_options;
	validator_options.SetAllowOffsetTextureOperand(maintenance8_enabled);
	if (tools.Validate(spirv.data(), spirv.size(), validator_options)) {
		return true;
	}
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF_COLOR(Log::Color::BrightRed, "%s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s",
	           label, shader_hash, messages.c_str());
	LOGF("%s\n", text.c_str());
	std::fprintf(stderr, "SPIRV_VALIDATE_FAIL %s hash=0x%016llx: %s", label,
	             static_cast<unsigned long long>(shader_hash), messages.c_str());
	return false;
}

void DumpModule(const char* stage_name, uint64_t shader_hash, const std::vector<uint32_t>& spirv) {
	// KYTY_TRACE_SPIRV_ALL_DIR=<dir> (diagnostic): every emitted module, <stage>_<hash>_<SPIR-V hash>.spv (one file
	// per distinct module), for comparing two builds or switch settings offline.
	if (const char* dir = Common::EnvRead::SpirvAllDir(); dir != nullptr && !spirv.empty()) {
		const auto spirv_hash = XXH3_64bits(spirv.data(), spirv.size() * sizeof(uint32_t));
		const auto name = fmt::format("{}/{}_{:016x}_{:016x}.spv", dir, stage_name, shader_hash, spirv_hash);
		if (FILE* f = std::fopen(name.c_str(), "wb")) {
			std::fwrite(spirv.data(), sizeof(uint32_t), spirv.size(), f);
			std::fclose(f);
		}
	}
	// KYTY_TRACE_SHADER_CODE_DIR: SPIR-V of oversized modules (> 400k words) for inspection.
	if (const char* dir = Common::EnvRead::ShaderCodeDir(); dir != nullptr && spirv.size() > 400000u) {
		const auto name = fmt::format("{}/big_{:016x}_{}.spv", dir, shader_hash, spirv.size());
		if (FILE* f = std::fopen(name.c_str(), "wb")) {
			std::fwrite(spirv.data(), sizeof(uint32_t), spirv.size(), f);
			std::fclose(f);
		}
	}
}

void ReplaceSpirv(uint64_t shader_hash, std::vector<uint32_t>& spirv) {
	// KYTY_DIAG_SPIRV_REPLACE=<hex hash>=<file.spvasm>[;<hex hash>=<file.spvasm>...]: the listed shaders' SPIR-V is
	// replaced by an edited disassembly (measurement probes). Logged; the edited module is still validated.
	const char* replace_list = Common::ShaderEnv::Read("KYTY_DIAG_SPIRV_REPLACE");
	for (std::string all = replace_list != nullptr ? replace_list : ""; !all.empty();) {
		const auto        end  = all.find(';');
		const std::string text = all.substr(0, end);
		all                    = end == std::string::npos ? std::string() : all.substr(end + 1);
		const auto        separator = text.find('=');
		if (separator != std::string::npos &&
		    std::strtoull(text.substr(0, separator).c_str(), nullptr, 16) == shader_hash) {
			std::string source;
			if (FILE* f = std::fopen(text.substr(separator + 1).c_str(), "rb")) {
				char   chunk[65536];
				size_t read = 0;
				while ((read = std::fread(chunk, 1, sizeof(chunk), f)) > 0) source.append(chunk, read);
				std::fclose(f);
			}
			const auto env = source.find("; Version: 1.3") != std::string::npos ? SPV_ENV_VULKAN_1_1
			               : source.find("; Version: 1.5") != std::string::npos ? SPV_ENV_VULKAN_1_2
			                                                                    : SPV_ENV_VULKAN_1_3;
			spvtools::SpirvTools  assembler(env);
			std::vector<uint32_t> words;
			std::string           messages;
			assembler.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t& position,
			                                 const char* message) {
				messages += fmt::format("line {}: {}; ", position.line, message);
			});
			if (!source.empty() && assembler.Assemble(source, &words)) {
				std::fprintf(stderr, "DIAG_SPIRV_REPLACE APPROXIMATION hash=0x%016llx words=%zu (was %zu)\n",
				             static_cast<unsigned long long>(shader_hash), words.size(),
				             spirv.size());
				spirv = std::move(words);
			} else {
				std::fprintf(stderr, "DIAG_SPIRV_REPLACE failed hash=0x%016llx bytes=%zu %s\n",
				             static_cast<unsigned long long>(shader_hash), source.size(),
				             messages.c_str());
			}
		}
	}
}

void DumpSpirvText(const char* stage_name, uint64_t shader_hash, const std::vector<uint32_t>& spirv) {
	// KYTY_TRACE_SPIRV_TEXT=<hex,...> (with KYTY_TRACE_SHADER_CODE_DIR): disassembled SPIR-V of the
	// listed shaders, for reading what a pass computes.
	if (const char* list = std::getenv("KYTY_TRACE_SPIRV_TEXT"); list != nullptr) {
		const auto hash_text = fmt::format("{:016x}", shader_hash);
		const char* dir      = Common::EnvRead::ShaderCodeDir();
		if (dir != nullptr && std::string(list).find(hash_text) != std::string::npos) {
			spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
			std::string          text;
			disassembler.Disassemble(spirv, &text,
			                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
			                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT));
			const auto name = fmt::format("{}/{}_{}_{}.spvasm", dir, stage_name, hash_text, spirv.size());
			if (FILE* f = std::fopen(name.c_str(), "wb")) {
				std::fwrite(text.data(), 1, text.size(), f);
				std::fclose(f);
			}
		}
	}
}

void TracePermutation(uint64_t shader_hash, const ShaderRecompiler::IR::ResourceSpecialization* previous, size_t permutations,
                      const ShaderRecompiler::IR::ResourceSpecialization& next) {
	// KYTY_TRACE_PERMUTATIONS=1: why a shader that already has a permutation compiles another.
	static const bool trace_permutations = std::getenv("KYTY_TRACE_PERMUTATIONS") != nullptr;
	if (trace_permutations && previous != nullptr) {
		std::string reason = fmt::format("buffers={}/{} images={}/{} samplers={}/{}", previous->buffers.size(),
		                                 next.buffers.size(), previous->images.size(), next.images.size(),
		                                 previous->samplers.size(), next.samplers.size());
		for (size_t i = 0; i < std::min(previous->images.size(), next.images.size()); ++i) {
			const auto& a = previous->images[i];
			const auto& b = next.images[i];
			if (a == b) continue;
			reason += fmt::format(" img{}: class {}/{} dim {}/{} mips {}/{} conv {}/{} swz {:x}/{:x} root {}/{} map {}/{} iter {}/{} cube {}/{}",
			                      i, static_cast<int>(a.numeric_class), static_cast<int>(b.numeric_class),
			                      static_cast<int>(a.dimension), static_cast<int>(b.dimension), a.mip_count, b.mip_count,
			                      static_cast<int>(a.conversion_format), static_cast<int>(b.conversion_format),
			                      a.shader_swizzle, b.shader_swizzle, a.indirect_root, b.indirect_root,
			                      a.indirect_mapping_offset, b.indirect_mapping_offset,
			                      a.indirect_search_iterations, b.indirect_search_iterations, a.cube, b.cube);
			break;
		}
		for (size_t i = 0; i < std::min(previous->buffers.size(), next.buffers.size()); ++i) {
			const auto& a = previous->buffers[i];
			const auto& b = next.buffers[i];
			if (a == b) continue;
			reason += fmt::format(" buf{}: stride {:x}/{:x} fmt {}/{} swz {:x}/{:x} oob {}/{} root {}/{} map {}/{} iter {}/{}",
			                      i, a.packed_stride, b.packed_stride, static_cast<int>(a.descriptor_format),
			                      static_cast<int>(b.descriptor_format), a.descriptor_swizzle, b.descriptor_swizzle,
			                      a.zero_stride_oob, b.zero_stride_oob, a.indirect_root, b.indirect_root,
			                      a.indirect_mapping_offset, b.indirect_mapping_offset,
			                      a.indirect_search_iterations, b.indirect_search_iterations);
			break;
		}
		// Every image table of the new permutation: root index, child count, b = bindless heap,
		// @ = mapping offset in the flattened SRT.
		reason += " roots:";
		for (size_t i = 0; i < next.images.size(); ++i) {
			if (next.images[i].indirect_root != i) continue;
			size_t children = 0;
			for (size_t j = 0; j < next.images.size(); ++j)
				children += j != i && next.images[j].indirect_root == i ? 1u : 0u;
			reason += fmt::format(" {}:{}{}@{}", i, children, next.images[i].bindless ? "b" : "",
			                      next.images[i].indirect_mapping_offset);
		}
		std::fprintf(stderr, "PERMUTATION hash=0x%016llx count=%zu %s\n",
		             static_cast<unsigned long long>(shader_hash), permutations,
		             reason.c_str());
	}
}

} // namespace Libs::Graphics::PipelineTrace
