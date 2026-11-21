#include "common/runtimeSwitches.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "graphics/shader/shaderCompiler.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace {

using namespace Libs::Graphics;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;
namespace IR         = Recompiler::IR;
namespace Switches   = Common::RuntimeSwitches;

constexpr std::array<uint32_t, 31> LoopWords {
    0xf4080600u, 0xfa000100u, 0x7e080d00u, 0x060808f0u, 0x100808ffu, 0x3c800000u, 0x7e0a02f0u, 0x340c0084u,
    0x7e100280u, 0x7e120280u, 0x7e140280u, 0x7e160280u, 0xbe830380u, 0xbf0a0203u, 0xbf84000du, 0x8f248503u,
    0xf40c0200u, 0x48000000u, 0xbf8cc07fu, 0xf09c0f08u, 0x00c20004u, 0xbf8c3f70u, 0x3a100108u, 0x3a120309u,
    0x3a14050au, 0x3a16070bu, 0x80038103u, 0xbf82fff1u, 0xe0781000u, 0x80010806u, 0xbf810000u,
};
constexpr std::array<uint32_t, 7> RawWords {
    0x34020082u, 0xe0301000u, 0x80000201u, 0xbf8c0000u, 0xe0701000u, 0x80010201u, 0xbf810000u,
};

constexpr uint64_t GuestRegion   = 0x200800000ull;
constexpr size_t   GuestBytes    = 0x40000u;
constexpr uint64_t TableAddress  = 0x200810000ull;
constexpr uint64_t TextureA      = 0x200820000ull;
constexpr uint64_t TextureB      = 0x200820100ull;
constexpr uint64_t TextureC      = 0x200820200ull;
constexpr uint64_t OutputAddress = 0x200830000ull;
constexpr uint32_t Format8888    = 56;
constexpr uint32_t Format32UInt  = 20;

int       g_failures = 0;
uint32_t* g_guest    = nullptr;

void Check(bool condition, const std::string& what) {
	if (condition) return;
	std::fprintf(stderr, "FAILED: %s\n", what.c_str());
	g_failures++;
}

using Texture = std::array<uint32_t, 8>;

Texture MakeTexture(uint64_t address, uint32_t format) {
	const auto base = address >> 8u;
	return {static_cast<uint32_t>(base), static_cast<uint32_t>((base >> 32u) & 0xffu) | (format << 20u) | (3u << 30u),
	        15u, 0xfacu | (9u << 28u), 0u, 0u, 0u, 0u};
}

bool MapGuest() {
#if defined(_WIN32)
	g_guest = static_cast<uint32_t*>(VirtualAlloc(reinterpret_cast<void*>(GuestRegion), GuestBytes,
	                                              MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
	void* mapped = mmap(reinterpret_cast<void*>(GuestRegion), GuestBytes, PROT_READ | PROT_WRITE,
	                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
	g_guest      = mapped == MAP_FAILED ? nullptr : static_cast<uint32_t*>(mapped);
#endif
	return g_guest == reinterpret_cast<uint32_t*>(GuestRegion);
}

void Poke(uint64_t address, uint32_t value) {
	g_guest[(address - GuestRegion) / 4u] = value;
}

bool ReadMemory(void*, uint64_t address, std::span<uint32_t> values) {
	for (size_t index = 0; index < values.size(); ++index) {
		const auto at = address + index * 4u;
		values[index] = at >= GuestRegion && at + 4u <= GuestRegion + GuestBytes ? g_guest[(at - GuestRegion) / 4u] : 0u;
	}
	return true;
}

void RecordLoad(void*, uint64_t, uint32_t, uint32_t) {}

void RecordLoads(void* userdata, std::span<const uint64_t> loads) {
	auto& list = *static_cast<std::vector<uint64_t>*>(userdata);
	list.insert(list.end(), loads.begin(), loads.end());
}

struct Plan {
	ShaderComputeInputInfo info {};
	IR::ResourcePlan       plan;
};

Plan Translate(std::span<const uint32_t> words, uint64_t hash) {
	Plan result;
	auto& info              = result.info;
	info.wave_size          = 64;
	info.host_subgroup_size = 64;
	info.threads_num[0]     = 64;
	info.threads_num[1]     = 1;
	info.threads_num[2]     = 1;
	info.thread_ids_num     = 1;
	const std::vector<uint32_t> user_data(16, 0u);
	const std::vector<uint32_t> code(words.begin(), words.end());
	Recompiler::CompileOptions  options;
	options.stage              = ShaderType::Compute;
	options.wave_size          = 64;
	options.shader_hash        = hash;
	options.dump_ir            = false;
	options.user_data          = user_data;
	options.input_info.compute = &info;
	const auto source          = Recompiler::PrepareShaderSource(code, options);
	auto       translated      = Recompiler::TranslateProgram(source.decoded, options);
	result.plan                = IR::ExtractResourcePlan(translated.program);
	return result;
}

struct Walk {
	bool                       ok = false;
	IR::ResourceSnapshot       snapshot;
	IR::ResourceSpecialization specialization;
	std::vector<uint64_t>      loads;
};

bool Same(const Walk& a, const Walk& b) {
	return a.ok == b.ok && a.snapshot.flattened_srt == b.snapshot.flattened_srt && a.snapshot.buffers == b.snapshot.buffers &&
	       a.snapshot.images == b.snapshot.images && a.snapshot.samplers == b.snapshot.samplers &&
	       a.snapshot.specialization_reads == b.snapshot.specialization_reads && a.specialization == b.specialization &&
	       a.loads == b.loads;
}

Walk Materialize(const IR::ResourcePlan& plan, std::span<const uint32_t, 16> user_data, bool ahead) {
	Walk           result;
	IR::SrtRuntime runtime {.user_data = user_data, .read_specialization_memory = ReadMemory};
	if (ahead) {
		runtime.userdata     = &result.loads;
		runtime.record_load  = RecordLoad;
		runtime.record_loads = RecordLoads;
	}
	result.ok = IR::MaterializeResources(plan, runtime, result.snapshot, result.specialization);
	return result;
}

std::array<uint32_t, 16> TableUserData(const std::vector<Texture>& table, uint32_t count) {
	std::fill_n(g_guest, GuestBytes / 4u, 0u);
	for (uint32_t entry = 0; entry < table.size(); ++entry) {
		for (uint32_t word = 0; word < 8u; ++word) Poke(TableAddress + entry * 32u + word * 4u, table[entry][word]);
	}
	const std::array<uint32_t, 4> sampler {0x12u, 0u, 0u, 0u};
	for (uint32_t word = 0; word < 4u; ++word) Poke(TableAddress + 0x100u + word * 4u, sampler[word]);
	std::array<uint32_t, 16> user_data {};
	user_data[0] = static_cast<uint32_t>(TableAddress);
	user_data[1] = static_cast<uint32_t>(TableAddress >> 32u);
	user_data[2] = count;
	user_data[4] = static_cast<uint32_t>(OutputAddress);
	user_data[5] = static_cast<uint32_t>(OutputAddress >> 32u);
	user_data[6] = 64u * 16u;
	user_data[7] = 0xfacu | (Format32UInt << 12u) | (1u << 24u) | (3u << 28u);
	return user_data;
}

std::array<uint32_t, 16> RawUserData(uint32_t records) {
	std::fill_n(g_guest, GuestBytes / 4u, 0u);
	std::array<uint32_t, 16> user_data {};
	const auto sharp = [&](uint32_t at, uint64_t address) {
		user_data[at]      = static_cast<uint32_t>(address);
		user_data[at + 1u] = static_cast<uint32_t>((address >> 32u) & 0xffffu);
		user_data[at + 2u] = records;
		user_data[at + 3u] = 0xfacu | (Format32UInt << 12u) | (1u << 24u) | (3u << 28u);
	};
	sharp(0, TableAddress);
	sharp(4, OutputAddress);
	return user_data;
}

struct Case {
	std::string              name;
	const IR::ResourcePlan*  plan = nullptr;
	std::array<uint32_t, 16> user_data {};
};

uint32_t Run(bool both) {
	const auto loop = Translate(LoopWords, 0x9e2u);
	const auto raw  = Translate(RawWords, 0x9e3u);
	Check(loop.plan.srt_code != nullptr && raw.plan.srt_code != nullptr, "both shaders get a walker");
	Check((loop.plan.srt_long_walker != nullptr) == (both || IR::SrtShortWalkersMode() == IR::SrtShortWalkers::Verify),
	      "a plan carries the long walker only for an A/B or the check");
	const auto a = MakeTexture(TextureA, Format8888);
	const auto b = MakeTexture(TextureB, Format8888);
	const auto c = MakeTexture(TextureC, Format32UInt);
	const std::vector<std::pair<std::string, std::vector<Texture>>> tables {
	    {"two Float descriptors", {a, b}}, {"one descriptor twice", {a, a}}, {"a Uint descriptor", {a, c}},
	    {"three descriptors", {a, b, c}},  {"no descriptor", {}},
	};
	uint32_t walks = 0;
	const auto compare = [&](const std::string& name, const IR::ResourcePlan& plan, const auto& fill) {
		for (const bool ahead: {false, true}) {
			const auto what = name + (ahead ? ", ahead" : ", execution");
			if (!both) {
				static_cast<void>(Materialize(plan, fill(), ahead));
				walks++;
				continue;
			}
			Switches::Assign(Switches::Switch::ShortSrtWalkers, false);
			const auto long_form = Materialize(plan, fill(), ahead);
			Switches::Assign(Switches::Switch::ShortSrtWalkers, true);
			const auto short_form = Materialize(plan, fill(), ahead);
			walks += 2u;
			Check(Same(long_form, short_form), what + ": the short walker gives the long walker's result");
			Check(!ahead || !long_form.ok || !long_form.loads.empty() || &plan == &raw.plan,
			      what + ": an ahead walk of the table shader hands over its loads");
		}
	};
	for (const auto& [name, table]: tables) {
		for (const uint32_t count: {static_cast<uint32_t>(table.size()), 0u, 1u}) {
			compare(name + " count " + std::to_string(count), loop.plan, [&] { return TableUserData(table, count); });
		}
	}
	for (const uint32_t records: {0u, 128u}) {
		compare("raw buffer records " + std::to_string(records), raw.plan, [&] { return RawUserData(records); });
	}
	return walks;
}

}

int main() {
	const char* ab      = std::getenv("KYTY_AB");
	const bool  both    = ab != nullptr && *ab != '\0';
	const bool  verify  = IR::SrtShortWalkersMode() == IR::SrtShortWalkers::Verify;
	IR::SetComputeIndirectFloors(256, 8);
	if (!MapGuest()) {
		std::fprintf(stderr, "FAILED: the guest region could not be mapped\n");
		return 1;
	}
	Check(!both || Common::Ab::Listed(Switches::Switch::ShortSrtWalkers), "KYTY_AB lists KYTY_SHORT_SRT_WALKERS");
	const auto walks  = Run(both);
	const auto checks = IR::SrtWalkerChecksSoFar();
	if (verify) {
		Check(checks.walks >= walks, "every walk ran with both walkers");
		Check(checks.mismatches == 0 && checks.unstable == 0, "the check found no difference");
	} else {
		Check(checks.walks == 0, "no walk is checked without verify");
	}
	if (g_failures != 0) {
		std::fprintf(stderr, "%d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("kyty_shim_short_srt_walkers%s: ok (%u walks, %llu checked)\n", verify ? "_verify" : both ? "_ab" : "", walks,
	            static_cast<unsigned long long>(checks.walks));
	return 0;
}
