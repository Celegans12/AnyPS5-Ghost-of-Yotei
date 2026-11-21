#include "RtProbeCache.hpp"
#include "RtSurvey.hpp"
#include "prx/libSceAgcDriver/TraceRevision.hpp"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/host_gpu/renderer/rayTracingReplay.h"
#include "graphics/shader/recompiler/ir/RayQuery.h"

#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace Recompiler = Libs::Graphics::ShaderRecompiler;
using Libs::Graphics::ShaderComputeInputInfo;
using Libs::Graphics::ShaderPixelInputInfo;
using Libs::Graphics::ShaderType;

struct Settings {
	uint32_t wave      = 64;
	uint32_t subgroup  = 0;
	uint32_t user      = 16;
	uint32_t group[3]  = {8, 8, 1};
	uint32_t scratch   = 0;
	uint32_t lds       = 0;
	bool     dump      = false;
	bool     emit      = false;
	bool     disabled  = false;
	bool     stats     = false;
	bool     pixel     = false;
	uint32_t inputs    = 8;
	uint32_t oracle    = 0;
	bool     replay    = false;
	bool     oracleSet = false;
	bool     robust    = false;
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

std::filesystem::path NormalizeSpirvPath(const std::filesystem::path& path, std::error_code& error) {
	const auto absolute = std::filesystem::absolute(path, error);
	if (error) return {};
	return std::filesystem::weakly_canonical(absolute, error);
}

bool CheckSpirvAlias(const std::string& output, const std::filesystem::path& outputPath,
                     const char* input, const char* description) {
	std::error_code error;
	const auto inputPath = NormalizeSpirvPath(input, error);
	if (!error) {
		if (outputPath == inputPath || std::filesystem::equivalent(outputPath, inputPath, error)) {
			std::fprintf(stderr, "%s: SPIR-V output aliases %s (%s)\n", output.c_str(), description, input);
			return false;
		}
		if (!error || error == std::errc::no_such_file_or_directory) return true;
	}
	std::fprintf(stderr, "%s: cannot verify SPIR-V output against %s: %s\n",
	             output.c_str(), input, error.message().c_str());
	return false;
}

bool CheckSpirvOutput(const std::string& output, std::span<const char* const> inputs,
                      const char* description, const char* replayPeer = nullptr) {
	std::error_code error;
	auto status = std::filesystem::symlink_status(output, error);
	if (error && error != std::errc::no_such_file_or_directory) {
		std::fprintf(stderr, "%s: cannot verify SPIR-V output path: %s\n", output.c_str(), error.message().c_str());
		return false;
	}
	if (std::filesystem::is_symlink(status)) {
		status = std::filesystem::status(output, error);
		if (error || !std::filesystem::exists(status)) {
			std::fprintf(stderr, "%s: cannot verify SPIR-V output path: unresolved symbolic link\n", output.c_str());
			return false;
		}
	}
	if (std::filesystem::is_directory(status)) {
		std::fprintf(stderr, "%s: cannot open SPIR-V output: path is a directory\n", output.c_str());
		return false;
	}
	const auto outputPath = NormalizeSpirvPath(output, error);
	if (error) {
		std::fprintf(stderr, "%s: cannot verify SPIR-V output path: %s\n", output.c_str(), error.message().c_str());
		return false;
	}
	for (const auto input: inputs) {
		if (!CheckSpirvAlias(output, outputPath, input, description)) return false;
	}
	return !replayPeer || CheckSpirvAlias(output, outputPath, replayPeer, "the other SPIR-V output");
}

bool CheckSpirvOutputs(const Settings& settings, std::span<const char* const> inputs, const char* description) {
	if (settings.spirv.empty()) return true;
	if (!CheckSpirvOutput(settings.spirv, inputs, description)) return false;
	return !settings.replay ||
	       CheckSpirvOutput(settings.spirv + ".guest.spv", inputs, description, settings.spirv.c_str());
}

bool WriteSpirv(const std::string& path, const std::vector<uint32_t>& spirv) {
	if (path.empty()) return true;
	std::ofstream out(path, std::ios::binary);
	if (!out) {
		std::fprintf(stderr, "%s: cannot open SPIR-V output\n", path.c_str());
		return false;
	}
	if (!out.write(reinterpret_cast<const char*>(spirv.data()),
	               static_cast<std::streamsize>(spirv.size() * sizeof(uint32_t)))) {
		std::fprintf(stderr, "%s: cannot write SPIR-V output\n", path.c_str());
		return false;
	}
	out.close();
	if (!out) {
		std::fprintf(stderr, "%s: cannot close SPIR-V output\n", path.c_str());
		return false;
	}
	return true;
}

int Probe(const char* label, const std::vector<uint32_t>& code, uint64_t hash, const Settings& settings) {
	ShaderComputeInputInfo compute {};
	compute.wave_size          = settings.wave;
	compute.host_subgroup_size = settings.subgroup != 0 ? settings.subgroup : settings.wave;
	for (int k = 0; k < 3; ++k)
		compute.threads_num[k] = settings.group[k];
	compute.thread_ids_num = settings.group[2] > 1 ? 3 : settings.group[1] > 1 ? 2 : 1;
	compute.group_id[0] = compute.group_id[1] = compute.group_id[2] = true;
	compute.workgroup_register = static_cast<int>(settings.user);
	compute.lds_size_dwords     = settings.lds;
	compute.scratch_size_dwords = settings.scratch;
	static Libs::Graphics::ShaderRecompiler::Spirv::Profile profile {.maintenance8 = true, .ray_query = true};
	profile.robust_buffer_access   = settings.robust;
	profile.device_clock_period_ps = settings.robust ? 1000u : 0u;
	std::vector<uint32_t> user_data(settings.user, 0u);
	Recompiler::CompileOptions options;
	options.stage                 = ShaderType::Compute;
	options.wave_size             = settings.wave;
	options.shader_hash           = hash;
	options.dump_ir               = settings.dump;
	options.profile               = &profile;
	options.native_ray_tracing    = !settings.disabled;
	options.ray_stats             = settings.stats && !settings.disabled;
	options.rt_oracle             = settings.disabled ? 0u : settings.oracle;
	options.user_data             = user_data;
	options.input_info.compute    = &compute;
	ShaderPixelInputInfo pixel {};
	if (settings.pixel) {
		pixel.wave_size           = settings.wave;
		pixel.input_num           = settings.inputs;
		pixel.ps_pos_x            = pixel.ps_pos_y = pixel.ps_pos_z = pixel.ps_pos_w = true;
		pixel.scratch_size_dwords = settings.scratch;
		options.stage             = ShaderType::Pixel;
		options.input_info.pixel  = &pixel;
	}
	auto source = Recompiler::PrepareShaderSource(code, options);
	if (!source.calls.empty() || source.unsupported_call_pc != UINT32_MAX) {
		std::printf("%s: calls\n", label);
		return 1;
	}
	auto translated = Recompiler::TranslateProgram(source.decoded, options);
	uint32_t bvh = 0, native = 0, oracle = 0, lane = 0, empty = 0;
	for (const auto* block: translated.program.blocks) {
		for (const auto& inst: *block) {
			if (inst.GetOpcode() != Recompiler::IR::ValueOpcode::BvhIntersect) continue;
			const auto flags = inst.Flags<uint32_t>();
			++bvh;
			if ((flags & Recompiler::IR::BvhIntersectNative) != 0) ++native;
			if ((flags & Recompiler::IR::BvhIntersectOracle) != 0) ++oracle;
			if ((flags & Recompiler::IR::BvhIntersectOracleLane) != 0) ++lane;
			if ((flags & Recompiler::IR::BvhIntersectEmpty) != 0) ++empty;
		}
	}
	const auto& info = translated.program.info;
	std::printf("%s: words=%zu bvh=%u native=%u user_sgpr=%u header=0x%x root=0x%x layout=%08x,%08x,%08x,%08x oracle=%u lane=%u "
	            "empty=%u\n",
	            label, code.size(), bvh, native, info.ray_query_user_sgpr, info.ray_query_header_offset, info.ray_query_root,
	            info.ray_query_layout[0], info.ray_query_layout[1], info.ray_query_layout[2], info.ray_query_layout[3], oracle,
	            lane, empty);
	for (size_t at = 0; at < translated.oracle_sources.size(); ++at) {
		const auto& hypothesis = translated.oracle_sources[at];
		std::printf("%s: hypothesis %zu/%zu user_sgpr=%u header=0x%x vsharp=0x%x root=0x%x layout=%08x,%08x,%08x,%08x\n", label, at,
		            translated.oracle_sources.size(), hypothesis.user_sgpr, hypothesis.header_offset, hypothesis.header_vsharp,
		            hypothesis.root, hypothesis.layout[0], hypothesis.layout[1], hypothesis.layout[2], hypothesis.layout[3]);
	}
	for (size_t at = 0; at < translated.oracle_improvements.size(); ++at) {
		const auto& improvement = translated.oracle_improvements[at];
		std::printf("%s: improvement %zu/%zu option=0x%x mask=%u:%u:%u\n", label, at, translated.oracle_improvements.size(),
		            improvement.option, improvement.dword, improvement.shift, improvement.width);
	}
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
		if (!WriteSpirv(settings.spirv, compiled.spirv)) return 2;
		std::printf("%s: spirv=%zu words ray_query=%d\n", label, compiled.spirv.size(),
		            compiled.program.info.ray_query ? 1 : 0);
	}
	return native == bvh && bvh != 0 ? 0 : 2;
}

bool CompileVariant(const std::vector<uint32_t>& code, uint64_t hash, const Settings& settings, bool native,
                    Recompiler::CompileResult& result) {
	ShaderComputeInputInfo compute {};
	compute.wave_size          = settings.wave;
	compute.host_subgroup_size = settings.subgroup != 0 ? settings.subgroup : settings.wave;
	for (int k = 0; k < 3; ++k)
		compute.threads_num[k] = settings.group[k];
	compute.thread_ids_num = settings.group[2] > 1 ? 3 : settings.group[1] > 1 ? 2 : 1;
	compute.group_id[0] = compute.group_id[1] = compute.group_id[2] = true;
	compute.workgroup_register  = static_cast<int>(settings.user);
	compute.lds_size_dwords     = settings.lds;
	compute.scratch_size_dwords = settings.scratch;
	static Libs::Graphics::ShaderRecompiler::Spirv::Profile profile {.maintenance8 = true, .ray_query = true};
	profile.robust_buffer_access   = settings.robust;
	profile.device_clock_period_ps = settings.robust ? 1000u : 0u;
	std::vector<uint32_t> user_data(settings.user, 0u);
	Recompiler::CompileOptions options;
	options.stage              = ShaderType::Compute;
	options.wave_size          = settings.wave;
	options.shader_hash        = hash;
	options.profile            = &profile;
	options.native_ray_tracing = native;
	options.rt_oracle          = native ? settings.oracle : 0u;
	options.user_data          = user_data;
	options.input_info.compute = &compute;
	ShaderPixelInputInfo pixel {};
	if (settings.pixel) {
		pixel.wave_size           = settings.wave;
		pixel.input_num           = settings.inputs;
		pixel.ps_pos_x            = pixel.ps_pos_y = pixel.ps_pos_z = pixel.ps_pos_w = true;
		pixel.scratch_size_dwords = settings.scratch;
		options.stage             = ShaderType::Pixel;
		options.input_info.pixel  = &pixel;
	}
	auto source = Recompiler::PrepareShaderSource(code, options);
	if (!source.calls.empty() || source.unsupported_call_pc != UINT32_MAX) return false;
	auto translated = Recompiler::TranslateProgram(source.decoded, options);
	Recompiler::IR::ResourceSpecialization specialization;
	specialization.buffers.resize(translated.program.info.buffers.size());
	specialization.images.resize(translated.program.info.images.size());
	specialization.samplers.resize(translated.program.info.samplers.size());
	for (size_t k = 0; k < specialization.images.size(); ++k) {
		specialization.images[k].numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float;
		specialization.images[k].dimension     = translated.program.info.images[k].dimension;
	}
	result = Recompiler::CompileProgram(std::move(translated), options, specialization);
	return true;
}

int Replay(const char* label, const std::vector<uint32_t>& code, uint64_t hash, const Settings& settings) {
	Recompiler::CompileResult served, guest;
	if (!CompileVariant(code, hash, settings, true, served) || !CompileVariant(code, hash, settings, false, guest)) {
		std::printf("%s: replay=no reason=calls\n", label);
		return 1;
	}
	uint32_t bvh = 0, native = 0, oracle = 0;
	for (const auto* block: served.program.blocks) {
		for (const auto& inst: *block) {
			if (inst.GetOpcode() != Recompiler::IR::ValueOpcode::BvhIntersect) continue;
			++bvh;
			if ((inst.Flags<uint32_t>() & Recompiler::IR::BvhIntersectNative) != 0) ++native;
			if ((inst.Flags<uint32_t>() & Recompiler::IR::BvhIntersectOracle) != 0) ++oracle;
		}
	}
	const auto written = [](const Recompiler::IR::Program& program) {
		uint32_t buffers = 0, images = 0;
		for (const auto& buffer: program.info.buffers) buffers += buffer.written ? 1u : 0u;
		for (const auto& image: program.info.images) images += image.written ? 1u : 0u;
		return std::pair {buffers, images};
	};
	const auto [served_buffers, served_images] = written(served.program);
	const auto [guest_buffers, guest_images]   = written(guest.program);
	const bool served_natively = served.program.info.bvh && served.program.info.ray_query;
	const char* reason = !served_natively ? "not served natively"
	                     : settings.pixel ? "draws are not played twice"
	                     : served_buffers != guest_buffers || served_images != guest_images
	                         ? "the two variants write different outputs"
	                         : Libs::Graphics::RayTracingCheckUnreplayable(served.program);
	if (!settings.spirv.empty() &&
	    (!WriteSpirv(settings.spirv, served.spirv) || !WriteSpirv(settings.spirv + ".guest.spv", guest.spirv))) {
		return 2;
	}
	std::printf("%s: replay=%s reason=%s bvh=%u native=%u oracle=%u written_buffers=%u written_images=%u "
	            "address_writes=%d served_words=%zu guest_words=%zu\n",
	            label, reason == nullptr ? "yes" : "no", reason == nullptr ? "-" : reason, bvh, native, oracle,
	            served_buffers, served_images, served.program.has_address_writes ? 1 : 0, served.spirv.size(),
	            guest.spirv.size());
	return reason == nullptr ? 0 : 3;
}

int Usage() {
	std::fprintf(stderr, "usage: kyty_rt_probe [--wave 32|64] [--subgroup N] [--user N] [--group X Y Z] [--scratch dwords] [--lds dwords] [--dump] [--emit] [--spirv out.spv] [--off] [--oracle 0-63] [--stats] [--robust] [--replay-check] [--stage compute|pixel] [--inputs N] (<shader.bin>... | --cache <file.code> --hash <hex> | --cache <file.code> --all [--kpm <file.kpm>] [--subgroup 32|64] [--lds-limit bytes] [--no-validate])\n       kyty_rt_probe --version\n");
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
	const char* kpmPath   = nullptr;
	bool        all       = false;
	bool        noValidate = false;
	uint32_t    ldsLimit  = 49152;
	for (int i = 1; i < argc; ++i) {
		const std::string_view arg = argv[i];
		if (arg == "--wave" && i + 1 < argc) {
			settings.wave = static_cast<uint32_t>(std::atoi(argv[++i]));
		} else if (arg == "--subgroup" && i + 1 < argc) {
			settings.subgroup = static_cast<uint32_t>(std::atoi(argv[++i]));
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
		} else if (arg == "--oracle" && i + 1 < argc) {
			settings.oracle    = static_cast<uint32_t>(std::atoi(argv[++i])) & 63u;
			settings.oracleSet = true;
		} else if (arg == "--stage" && i + 1 < argc) {
			const std::string_view stage = argv[++i];
			if (stage != "pixel" && stage != "compute") return Usage();
			settings.pixel = stage == "pixel";
		} else if (arg == "--inputs" && i + 1 < argc) {
			settings.inputs = static_cast<uint32_t>(std::atoi(argv[++i]));
		} else if (arg == "--robust") {
			settings.robust = true;
		} else if (arg == "--stats") {
			settings.stats = true;
		} else if (arg == "--off") {
			settings.disabled = true;
		} else if (arg == "--replay-check") {
			settings.replay = true;
		} else if (arg == "--spirv" && i + 1 < argc) {
			settings.spirv = argv[++i];
			settings.emit  = true;
		} else if (arg == "--version") {
			std::printf("kyty_rt_probe %s%s\n", APS5_TRACE_COMMIT, APS5_TRACE_DIRTY > 0 ? "+modified" : "");
			return 0;
		} else if (arg == "--all") {
			all = true;
		} else if (arg == "--kpm" && i + 1 < argc) {
			kpmPath = argv[++i];
		} else if (arg == "--lds-limit" && i + 1 < argc) {
			ldsLimit = static_cast<uint32_t>(std::atoi(argv[++i]));
		} else if (arg == "--no-validate") {
			noValidate = true;
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
	if (all) {
		if (!cachePath || !*cachePath || cacheHash || !files.empty()) return Usage();
		AgcDriver::SurveySettings survey;
		survey.cache         = cachePath;
		survey.manifest      = kpmPath != nullptr ? kpmPath : "";
		survey.oracle        = settings.oracleSet ? settings.oracle : 1u;
		survey.subgroup      = settings.subgroup != 0 ? settings.subgroup : 32u;
		survey.ldsLimitBytes = ldsLimit;
		survey.validate      = !noValidate;
		survey.build         = std::string(APS5_TRACE_COMMIT) + (APS5_TRACE_DIRTY > 0 ? "+modified" : "");
		return AgcDriver::RunSurvey(survey);
	}
	if (kpmPath != nullptr || noValidate) return Usage();
	if (cachePath || cacheHash) {
		uint64_t hash = 0;
		if (!cachePath || !*cachePath || !cacheHash || !files.empty() || !ParseShaderHash(cacheHash, hash)) return Usage();
		std::ifstream cache(cachePath, std::ios::binary);
		if (!cache) {
			std::fprintf(stderr, "%s: cannot open manifest cache\n", cachePath);
			return 2;
		}
		if (!CheckSpirvOutputs(settings, {&cachePath, 1}, "the manifest cache")) return 2;
		const auto result = AgcDriver::ReadCachedShaderCode(cache, hash);
		if (result.error != nullptr) {
			std::fprintf(stderr, "%s: shader %s: %s\n", cachePath, cacheHash, result.error);
			return 2;
		}
		if (settings.replay) return Replay(cacheHash, result.code, hash, settings) == 0 ? 0 : 2;
		return Probe(cacheHash, result.code, hash, settings) == 0 ? 0 : 2;
	}
	if (files.empty()) return Usage();
	std::vector<const char*> inputs;
	inputs.reserve(files.size());
	for (const auto& file: files) inputs.push_back(file.first);
	for (const auto& file: files) {
		if (!CheckSpirvOutputs(file.second, inputs, "a shader input")) return 2;
	}
	bool failed = false;
	for (const auto& [path, options] : files) {
		std::vector<uint32_t> code;
		if (!Read(path, code)) {
			std::printf("%s: unreadable\n", path);
			failed = true;
			continue;
		}
		if (options.replay) {
			if (Replay(path, code, HashOf(path), options) != 0) failed = true;
			continue;
		}
		if (Probe(path, code, HashOf(path), options) != 0) failed = true;
	}
	return failed ? 2 : 0;
}
