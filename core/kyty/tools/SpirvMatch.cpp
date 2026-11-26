#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineStoreFormat.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <tuple>

namespace {

using namespace Libs::Graphics;
using namespace ShaderStoreFormat;
using ShaderRecompiler::IR::IoAll;
using ShaderRecompiler::IR::Reader;
using ShaderRecompiler::IR::Writer;

constexpr size_t MaxRecordBytes = 64 * 1024 * 1024;
constexpr std::array Stages{"unknown", "vertex", "pixel", "compute", "mesh", "local",
                            "tessellation-control", "tessellation-evaluation"};

struct Dump {
	std::vector<uint32_t> words;
	std::set<std::tuple<ShaderType, uint64_t, uint64_t, bool>> matches;
};

bool IsSpirv(std::span<const uint32_t> words) {
	return words.size() >= 5 && words.front() == 0x07230203 && words[3] != 0 && words[4] == 0;
}

template <class T> bool Read(std::istream& input, std::span<T> bytes) {
	return bool(input.read(reinterpret_cast<char*>(bytes.data()), bytes.size_bytes()));
}

bool ReadDump(const std::filesystem::path& path, Dump& dump) {
	std::ifstream input(path, std::ios::binary | std::ios::ate);
	const auto size = static_cast<std::streamoff>(input.tellg());
	if (size < 0 || static_cast<uint64_t>(size) > MaxRecordBytes || size % sizeof(uint32_t) != 0) return false;
	dump.words.resize(size / sizeof(uint32_t));
	input.seekg(0);
	return Read(input, std::span(dump.words)) && IsSpirv(dump.words);
}

template <size_t Flags>
using EarlierProfile = std::tuple<bool, bool, uint32_t, std::array<bool, Flags>>;

bool ReadStoreHeader(std::istream& input) {
	input.seekg(0, std::ios::end);
	const auto end = input.tellg();
	const auto try_profile = [&](auto profile) {
		uint64_t stamp = 0;
		uint32_t max_shared_memory = 0;
		Writer layout;
		IoStoreHeader(layout, stamp, profile, max_shared_memory);
		const auto size = static_cast<std::streamoff>(layout.bytes.size());
		if (end < size) return false;
		input.clear();
		input.seekg(0);
		if (!Read(input, std::span(layout.bytes))) return false;
		Reader reader{layout.bytes};
		IoStoreHeader(reader, stamp, profile, max_shared_memory);
		if (!reader.ok || stamp == 0 || max_shared_memory == 0) return false;
		if (end == size) return true;
		uint32_t magic = 0;
		input.seekg(size + static_cast<std::streamoff>(sizeof(Frame)));
		if (!Read(input, std::span(&magic, 1)) || magic != ZSTD_MAGICNUMBER) return false;
		input.seekg(size);
		return bool(input);
	};
	return try_profile(ShaderRecompiler::Spirv::Profile{}) || try_profile(EarlierProfile<9>{}) ||
	       try_profile(EarlierProfile<8>{}) || try_profile(EarlierProfile<7>{}) || try_profile(EarlierProfile<6>{});
}

bool MatchShaderDump(std::span<const uint8_t> raw, std::span<Dump> dumps,
                     std::optional<bool>& shader_info_layout) {
	Reader reader{raw};
	Record type{};
	IoAll(reader, type);
	if (!reader.ok) return false;
	switch (type) {
		case Record::GraphicsPipeline:
			return reader.bytes.size() == sizeof(GraphicsPipelineKey);
		case Record::TessellationLayout: {
			uint64_t key = 0;
			ShaderTessellationInputInfo layout;
			if (reader.bytes.size() == sizeof(key) + 9 * sizeof(uint32_t)) return true;
			IoAll(reader, key, layout);
			return reader.ok && reader.bytes.empty();
		}
		case Record::Module: break;
		default: return false;
	}

	const auto decode = [&](bool has_relative_vgprs) {
		Reader module_reader{.bytes = reader.bytes, .allocation_budget = MaxRecordBytes,
		                     .shader_info_has_relative_vgprs = has_relative_vgprs};
		ShaderStoreFormat::Key key;
		ShaderModule module;
		std::vector<uint8_t> resource_plan;
		std::vector<uint32_t> spirv;
		std::vector<uint32_t> plain_spirv;
		IoStoreModule(module_reader, key, module, resource_plan, spirv, plain_spirv);
		if (!module_reader.ok || !module_reader.bytes.empty() || key.stage == ShaderType::Unknown ||
		    static_cast<size_t>(key.stage) >= Stages.size() || !IsSpirv(spirv) ||
		    (!plain_spirv.empty() && !IsSpirv(plain_spirv))) {
			return false;
		}
		for (auto& dump: dumps) {
			if (dump.words == spirv) dump.matches.emplace(key.stage, key.hash, module.id, false);
			if (dump.words == plain_spirv) dump.matches.emplace(key.stage, key.hash, module.id ^ PlainModuleIdXor, true);
		}
		return true;
	};
	if (shader_info_layout.has_value()) return decode(*shader_info_layout);
	for (const bool has_relative_vgprs: {true, false}) {
		if (decode(has_relative_vgprs)) {
			shader_info_layout = has_relative_vgprs;
			return true;
		}
	}
	return false;
}

bool ReadStore(const std::filesystem::path& path, std::span<Dump> dumps) {
	std::ifstream input(path, std::ios::binary);
	if (!ReadStoreHeader(input)) {
		std::cerr << "Invalid or unreadable Kyty shader store header\n";
		return false;
	}
	std::vector<uint8_t> compressed;
	std::vector<uint8_t> raw;
	std::optional<bool> shader_info_layout;
	for (size_t index = 1; input.peek() != std::char_traits<char>::eof(); ++index) {
		Frame frame;
		if (Read(input, std::span(&frame, 1)) && frame.raw_size <= MaxRecordBytes &&
		    frame.compressed_size <= ZSTD_compressBound(MaxRecordBytes)) {
			compressed.resize(frame.compressed_size);
			if (Read(input, std::span(compressed)) && DecodeShaderStoreFrame(frame, compressed, raw) &&
			    MatchShaderDump(raw, dumps, shader_info_layout)) {
				continue;
			}
		}
		std::cerr << "Invalid or incomplete Kyty shader store record " << index << '\n';
		return false;
	}
	if (input.eof() && !input.bad()) return true;
	std::cerr << "Could not finish reading the Kyty shader store\n";
	return false;
}

template <class Char> int Run(int argc, Char** argv) {
	if (argc < 3) {
		const bool help = argc == 2 && std::filesystem::path(argv[1]) == "--help";
		(help ? std::cout : std::cerr) << "usage: kyty_spirv_match <shader-store.shaders> <module.spv> [module.spv ...]\n";
		return help ? 0 : 64;
	}
	std::vector<Dump> dumps(argc - 2);
	for (size_t i = 0; i < dumps.size(); ++i) {
		if (!ReadDump(argv[i + 2], dumps[i])) {
			std::cerr << "Invalid or unreadable SPIR-V input " << (i + 1) << '\n';
			return 2;
		}
	}
	if (!ReadStore(argv[1], dumps)) return 2;
	bool all_matched = true;
	for (size_t i = 0; i < dumps.size(); ++i) {
		if (dumps[i].matches.empty()) {
			std::cout << "input=" << (i + 1) << " no match\n";
			all_matched = false;
		}
		for (const auto& [stage, shader, id, plain]: dumps[i].matches) {
			std::cout << "input=" << (i + 1) << " stage=" << Stages[static_cast<size_t>(stage)]
			          << std::hex << std::setfill('0') << " shader=" << std::setw(16) << shader
			          << " module=" << std::setw(16) << id << std::dec
			          << " variant=" << (plain ? "plain" : "standard") << '\n';
		}
	}
	return all_matched ? 0 : 1;
}

}

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
	try {
		return Run(argc, argv);
	} catch (const std::exception&) {
		std::cerr << "Could not process the shader store or SPIR-V inputs\n";
		return 2;
	}
}
