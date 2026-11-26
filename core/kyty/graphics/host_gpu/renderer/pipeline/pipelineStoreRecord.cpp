#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineShaderDiagnostics.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineStoreFormat.h"

#include "graphics/shader/recompiler/ShaderRecompiler.h"

#include <istream>
#include <string>
#include <utility>

namespace Libs::Graphics {

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

#if defined(ANYPS5_KYTY)
bool LoadStoredComputeModule(std::istream& input, std::span<const uint8_t> expected_header,
                             uint64_t id, ShaderModule& module, std::vector<uint32_t>& spirv,
                             std::string& error) {
	using ShaderRecompiler::IR::IoAll;
	using ShaderRecompiler::IR::Reader;
	using namespace ShaderStoreFormat;

	constexpr size_t MaxRecordBytes = 64u * 1024u * 1024u;
	const auto fail = [&](const char* message) {
		error = message;
		return false;
	};
	const auto read = [&](void* data, size_t size) {
		return bool(input.read(static_cast<char*>(data), static_cast<std::streamsize>(size)));
	};
	std::vector<uint8_t> header(expected_header.size());
	if (header.empty() || !read(header.data(), header.size()) || !std::ranges::equal(header, expected_header)) {
		return fail("shader store is unreadable or incompatible with this Kyty build and device");
	}
	std::vector<uint8_t> compressed, raw;
	ShaderModule selected;
	std::vector<uint32_t> selected_spirv;
	while (input.peek() != std::char_traits<char>::eof()) {
		Frame frame;
		if (!read(&frame, sizeof(frame)) || frame.raw_size > MaxRecordBytes ||
		    frame.compressed_size > ZSTD_compressBound(MaxRecordBytes)) {
			return fail("invalid shader store frame size");
		}
		compressed.resize(frame.compressed_size);
		if (!read(compressed.data(), compressed.size()) || !DecodeShaderStoreFrame(frame, compressed, raw)) {
			return fail("invalid or incomplete shader store frame");
		}
		Reader reader {.bytes = raw, .allocation_budget = MaxRecordBytes};
		Record type {};
		IoAll(reader, type);
		if (!reader.ok || type > Record::TessellationLayout) {
			return fail("invalid shader store record kind");
		}
		if (type != Record::Module) continue;
		ShaderStoreFormat::Key key;
		ShaderModule candidate;
		std::vector<uint8_t> plan;
		std::vector<uint32_t> words, plain;
		IoStoreModule(reader, key, candidate, plan, words, plain);
		if (!reader.ok || !reader.bytes.empty()) return fail("invalid shader store module record");
		if (candidate.id != id) continue;
		if (!selected_spirv.empty()) return fail("duplicate module ID in shader store");
		if (key.stage != ShaderType::Compute || candidate.info.stage != ShaderType::Compute) {
			return fail("requested module is not a compute shader");
		}
		if (words.empty()) return fail("compute module has no SPIR-V");
		selected = std::move(candidate);
		selected_spirv = std::move(words);
	}
	if (input.bad() || !input.eof()) return fail("could not finish reading the shader store");
	if (selected_spirv.empty()) return fail("compute module ID was not found in the shader store");
	selected.spirv_words = static_cast<uint32_t>(selected_spirv.size());
	module = std::move(selected);
	spirv = std::move(selected_spirv);
	return true;
}
#endif

}
