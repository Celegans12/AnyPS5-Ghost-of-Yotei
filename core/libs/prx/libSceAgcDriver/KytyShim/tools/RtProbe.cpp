#include "RtProbeCache.hpp"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/RayQuery.h"

#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace Recompiler = Libs::Graphics::ShaderRecompiler;
using Libs::Graphics::ShaderComputeInputInfo;
using Libs::Graphics::ShaderType;

struct Settings {
	uint32_t wave      = 64;
	uint32_t user      = 16;
	uint32_t group[3]  = {8, 8, 1};
	uint32_t scratch   = 0;
	uint32_t lds       = 0;
	bool     dump      = false;
	bool     emit      = false;
	bool     disabled  = false;
	std::string spirv;
};

bool Read(const char* path, std::vector<uint32_t>& code) {
	std::ifstream file(path, std::ios::binary);
	if (!file) return false;
	std::vector<char> bytes {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
	code.resize(bytes.size() / 4);
	std::memcpy(code.data(), bytes.data(), code.size() * 4);
	return !code.empty();
}

uint64_t HashOf(const char* path) {
	std::string name = path;
	if (const auto slash = name.find_last_of('/'); slash != std::string::npos) name = name.substr(slash + 1);
	return std::strtoull(name.c_str(), nullptr, 16);
}

int Probe(const char* label, const std::vector<uint32_t>& code, uint64_t hash, const Settings& settings) {
	ShaderComputeInputInfo compute {};
	compute.wave_size          = settings.wave;
	compute.host_subgroup_size = settings.wave;
	for (int k = 0; k < 3; ++k)
		compute.threads_num[k] = settings.group[k];
	compute.thread_ids_num = settings.group[2] > 1 ? 3 : settings.group[1] > 1 ? 2 : 1;
	compute.group_id[0] = compute.group_id[1] = compute.group_id[2] = true;
	compute.workgroup_register = static_cast<int>(settings.user);
	compute.lds_size_dwords     = settings.lds;
	compute.scratch_size_dwords = settings.scratch;
	static Libs::Graphics::ShaderRecompiler::Spirv::Profile profile {.maintenance8 = true, .ray_query = true};
	std::vector<uint32_t> user_data(settings.user, 0u);
	Recompiler::CompileOptions options;
	options.stage                 = ShaderType::Compute;
	options.wave_size             = settings.wave;
	options.shader_hash           = hash;
	options.dump_ir               = settings.dump;
	options.profile               = &profile;
	options.native_ray_tracing    = !settings.disabled;
	options.user_data             = user_data;
	options.input_info.compute    = &compute;
	auto source = Recompiler::PrepareShaderSource(code, options);
	if (!source.calls.empty() || source.unsupported_call_pc != UINT32_MAX) {
		std::printf("%s: calls\n", label);
		return 1;
	}
	auto translated = Recompiler::TranslateProgram(source.decoded, options);
	uint32_t bvh = 0, native = 0;
	for (const auto* block: translated.program.blocks) {
		for (const auto& inst: *block) {
			if (inst.GetOpcode() != Recompiler::IR::ValueOpcode::BvhIntersect) continue;
			++bvh;
			if ((inst.Flags<uint32_t>() & Recompiler::IR::BvhIntersectNative) != 0) ++native;
		}
	}
	const auto& info = translated.program.info;
	std::printf("%s: words=%zu bvh=%u native=%u user_sgpr=%u header=0x%x root=0x%x layout=%08x,%08x,%08x,%08x\n", label,
	            code.size(), bvh, native, info.ray_query_user_sgpr, info.ray_query_header_offset, info.ray_query_root,
	            info.ray_query_layout[0], info.ray_query_layout[1], info.ray_query_layout[2], info.ray_query_layout[3]);
	if (settings.emit) {
		Recompiler::IR::ResourceSpecialization specialization;
		specialization.buffers.resize(translated.program.info.buffers.size());
		specialization.images.resize(translated.program.info.images.size());
		specialization.samplers.resize(translated.program.info.samplers.size());
		for (size_t k = 0; k < specialization.images.size(); ++k) {
			specialization.images[k].numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float;
			specialization.images[k].dimension     = translated.program.info.images[k].dimension;
		}
		auto compiled = Recompiler::CompileProgram(std::move(translated), options, specialization);
		std::printf("%s: spirv=%zu words ray_query=%d\n", label, compiled.spirv.size(),
		            compiled.program.info.ray_query ? 1 : 0);
		if (!settings.spirv.empty()) {
			std::ofstream out(settings.spirv, std::ios::binary);
			out.write(reinterpret_cast<const char*>(compiled.spirv.data()),
			          static_cast<std::streamsize>(compiled.spirv.size() * 4));
		}
	}
	return native == bvh && bvh != 0 ? 0 : 2;
}

int Usage() {
	std::fprintf(stderr, "usage: kyty_rt_probe [--wave 32|64] [--user N] [--group X Y Z] [--scratch dwords] [--lds dwords] [--dump] [--emit] [--spirv out.spv] [--off] (<shader.bin>... | --cache <file.code> --hash <hex>)\n");
	return 64;
}

bool ParseShaderHash(std::string_view input, uint64_t& hash) {
	if (input.starts_with("0x") || input.starts_with("0X")) input.remove_prefix(2);
	const auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), hash, 16);
	return error == std::errc{} && end == input.data() + input.size() && hash != 0;
}

}

int main(int argc, char** argv) {
	Settings settings;
	std::vector<std::pair<const char*, Settings>> files;
	const char* cachePath = nullptr;
	const char* cacheHash = nullptr;
	for (int i = 1; i < argc; ++i) {
		const std::string_view arg = argv[i];
		if (arg == "--wave" && i + 1 < argc) {
			settings.wave = static_cast<uint32_t>(std::atoi(argv[++i]));
		} else if (arg == "--user" && i + 1 < argc) {
			settings.user = static_cast<uint32_t>(std::atoi(argv[++i]));
		} else if (arg == "--group" && i + 3 < argc) {
			for (auto& size: settings.group)
				size = static_cast<uint32_t>(std::atoi(argv[++i]));
		} else if (arg == "--scratch" && i + 1 < argc) {
			settings.scratch = static_cast<uint32_t>(std::atoi(argv[++i]));
		} else if (arg == "--lds" && i + 1 < argc) {
			settings.lds = static_cast<uint32_t>(std::atoi(argv[++i]));
		} else if (arg == "--dump") {
			settings.dump = true;
		} else if (arg == "--emit") {
			settings.emit = true;
		} else if (arg == "--off") {
			settings.disabled = true;
		} else if (arg == "--spirv" && i + 1 < argc) {
			settings.spirv = argv[++i];
			settings.emit  = true;
		} else if (arg == "--cache") {
			if (cachePath || i + 1 >= argc) return Usage();
			cachePath = argv[++i];
		} else if (arg == "--hash") {
			if (cacheHash || i + 1 >= argc) return Usage();
			cacheHash = argv[++i];
		} else {
			files.emplace_back(argv[i], settings);
		}
	}
	if (cachePath || cacheHash) {
		uint64_t hash = 0;
		if (!cachePath || !*cachePath || !cacheHash || !files.empty() || !ParseShaderHash(cacheHash, hash)) return Usage();
		std::ifstream cache(cachePath, std::ios::binary);
		if (!cache) {
			std::fprintf(stderr, "%s: cannot open manifest cache\n", cachePath);
			return 2;
		}
		if (!settings.spirv.empty()) {
			std::error_code error;
			if (std::filesystem::equivalent(cachePath, settings.spirv, error)) {
				std::fprintf(stderr, "%s: SPIR-V output aliases the manifest cache\n", settings.spirv.c_str());
				return 2;
			}
			if (error && error != std::errc::no_such_file_or_directory) {
				std::fprintf(stderr, "%s: cannot verify SPIR-V output path: %s\n", settings.spirv.c_str(), error.message().c_str());
				return 2;
			}
		}
		const auto result = AgcDriver::ReadCachedShaderCode(cache, hash);
		if (result.error != nullptr) {
			std::fprintf(stderr, "%s: shader %s: %s\n", cachePath, cacheHash, result.error);
			return 2;
		}
		return Probe(cacheHash, result.code, hash, settings) == 0 ? 0 : 2;
	}
	if (files.empty()) return Usage();
	bool failed = false;
	for (const auto& [path, options] : files) {
		std::vector<uint32_t> code;
		if (!Read(path, code)) {
			std::printf("%s: unreadable\n", path);
			failed = true;
			continue;
		}
		if (Probe(path, code, HashOf(path), options) != 0) failed = true;
	}
	return failed ? 2 : 0;
}
