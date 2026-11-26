#include "graphics/host_gpu/renderer/pipeline/pipelineShaderDiagnostics.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineStoreTypes.h"

#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/Serialize.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <mutex>
#include <spirv-tools/libspirv.hpp>
#include <unordered_set>
#include <xxhash.h>

namespace Libs::Graphics {

#if defined(ANYPS5_KYTY)
void LogShaderVariantBuild(const char* stage, uint64_t shader_hash, uint32_t user_data_count,
                           const std::vector<uint32_t>& static_state, uint32_t push_data_cursor,
                           const ShaderModule& module) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SHADER_VARIANT_LOG");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	if (!enabled || module.module == nullptr) return;

	ShaderRecompiler::IR::Writer writer;
	const auto fingerprint = [&writer](const auto& fields) {
		writer.bytes.clear();
		ShaderRecompiler::IR::IoAll(writer, fields);
		return XXH3_64bits(writer.bytes.data(), writer.bytes.size());
	};
	const auto& specialization = module.specialization;
	const auto& functions      = module.function_code;
	const auto plain_fields = module.plain != nullptr
	                              ? fmt::format(" plain_module={:016x} plain_spirv_words={}",
	                                            module.plain->id, module.plain->spirv_words)
	                              : std::string {};
	Log::WriteToConsoleAndLog(fmt::format(
	    "[kyty-variant] build stage={} shader={:016x} module={:016x} user_data_count={} "
	    "static={:016x} push_start={} push_cursor={} buffers={:016x} images={:016x} "
	    "samplers={:016x} linked_code={:016x} table={} spirv_words={}{} @{}\n",
	    stage, shader_hash, module.id, user_data_count, fingerprint(static_state),
	    module.info.bindings.push_data_start_dword, push_data_cursor,
	    fingerprint(specialization.buffers), fingerprint(specialization.images),
	    fingerprint(specialization.samplers), XXH3_64bits(functions.data(), functions.size() * sizeof(uint32_t)),
	    module.info.table_mode, module.spirv_words, plain_fields,
	    std::chrono::steady_clock::now().time_since_epoch().count()));
}
#endif

void DumpShader(std::span<const uint32_t> words, const std::string& name) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	const auto path = Config::GetShaderLogFolder() / name;
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(words.data(), words.size_bytes());
}

void ValidateShaderSpirv(const char* stage, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv, bool maintenance8_enabled) {
	if (!Config::ShaderValidationEnabled()) {
		return;
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
		return;
	}
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF("%s\n", text.c_str());
	EXIT("ShaderRecompiler %s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s", stage,
	     shader_hash, messages.c_str());
}

void NoteStoreRefused(ShaderType stage, uint64_t hash,
                      const ShaderRecompiler::IR::EncodeFailure& failure) {
	using Reason = ShaderRecompiler::IR::EncodeFailure::Reason;
	static std::mutex                   mutex;
	static std::unordered_set<uint64_t> noted;
	{
		std::scoped_lock lock(mutex);
		if (!noted.insert(hash).second) {
			return;
		}
	}
	std::string why;
	switch (failure.reason) {
		case Reason::ValueType:
			why = fmt::format("a value of type 0x{:08x} has no cache encoding", failure.type);
			break;
		case Reason::UnknownInstruction:
			why = "a value refers to an instruction outside the program's plan";
			break;
		case Reason::PhiBlock: why = "a phi operand names a predecessor block"; break;
		case Reason::None: return;
	}
	const std::string_view opcode =
	    failure.in_instruction ? ShaderRecompiler::IR::ValueOpcodeName(failure.opcode)
	                           : std::string_view("none, outside the instructions");
	Log::WriteToConsoleAndLog(fmt::format(
	    "Shader cache: program 0x{:016x} ({}) is not written to the store: {} (opcode {}); the game "
	    "goes on with the program compiled in memory\n",
	    hash, ShaderRecompiler::StageName(stage), why, opcode));
}

}
