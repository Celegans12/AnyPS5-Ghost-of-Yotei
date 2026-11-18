#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/RayQuery.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
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

int Probe(const char* path, const Settings& settings) {
	std::vector<uint32_t> code;
	if (!Read(path, code)) {
		std::printf("%s: unreadable\n", path);
		return 1;
	}
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
	options.shader_hash           = HashOf(path);
	options.dump_ir               = settings.dump;
	options.profile               = &profile;
	options.native_ray_tracing    = !settings.disabled;
	options.user_data             = user_data;
	options.input_info.compute    = &compute;
	auto source = Recompiler::PrepareShaderSource(code, options);
	if (!source.calls.empty() || source.unsupported_call_pc != UINT32_MAX) {
		std::printf("%s: calls\n", path);
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
	std::printf("%s: words=%zu bvh=%u native=%u user_sgpr=%u header=0x%x root=0x%x layout=%08x,%08x,%08x,%08x\n", path,
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
		std::printf("%s: spirv=%zu words ray_query=%d\n", path, compiled.spirv.size(),
		            compiled.program.info.ray_query ? 1 : 0);
		if (!settings.spirv.empty()) {
			std::ofstream out(settings.spirv, std::ios::binary);
			out.write(reinterpret_cast<const char*>(compiled.spirv.data()),
			          static_cast<std::streamsize>(compiled.spirv.size() * 4));
		}
	}
	return native == bvh && bvh != 0 ? 0 : 2;
}

}

int main(int argc, char** argv) {
	Settings settings;
	int      failures = 0, files = 0;
	for (int i = 1; i < argc; ++i) {
		const std::string arg = argv[i];
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
		} else {
			++files;
			if (Probe(argv[i], settings) != 0) ++failures;
		}
	}
	if (files == 0) {
		std::fprintf(stderr, "usage: kyty_rt_probe [--wave 32|64] [--user N] [--group X Y Z] [--scratch dwords] [--lds dwords] [--dump] [--emit] [--spirv out.spv] [--off] <shader.bin>...\n");
		return 64;
	}
	return failures == 0 ? 0 : 2;
}
