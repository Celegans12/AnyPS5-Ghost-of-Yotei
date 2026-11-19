#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineStoreFormat.h"

#include "common/logging/log.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"

#include <fmt/format.h>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace Libs::Graphics {
namespace {

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

bool EncodeModuleRecord(ShaderType stage, uint64_t hash, uint32_t user_data_count,
                        const std::vector<uint32_t>&              static_state,
                        const ShaderRecompiler::IR::ResourcePlan& plan, const ShaderModule& module,
                        const std::vector<uint32_t>& spirv, const std::vector<uint32_t>& plain_spirv,
                        std::vector<uint8_t>& record) {
	using ShaderRecompiler::IR::IoAll;
	using ShaderRecompiler::IR::Writer;
	using namespace ShaderStoreFormat;

	std::vector<uint8_t>                plan_bytes;
	ShaderRecompiler::IR::EncodeFailure failure;
	if (!ShaderRecompiler::IR::SaveResourcePlan(plan, plan_bytes, failure)) {
		NoteStoreRefused(stage, hash, failure);
		return false;
	}
	Writer writer;
	IoAll(writer, Record::Module, stage, hash, user_data_count, static_state, plan_bytes);
	IoStoreModuleInfo(writer, module);
	IoAll(writer, spirv, plain_spirv);
	if (writer.failure.Failed()) {
		NoteStoreRefused(stage, hash, writer.failure);
		return false;
	}
	record = std::move(writer.bytes);
	return true;
}

}
