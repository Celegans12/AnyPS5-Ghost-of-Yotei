#include "graphics/host_gpu/renderer/e2Stats.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "graphics/shader/shaderCompiler.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <string>
#include <vector>

namespace {

using namespace Libs::Graphics;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;
namespace IR         = Recompiler::IR;
using IR::SrtLoadClass;

constexpr uint32_t ShiftLane     = 0x34020082u;
constexpr uint32_t LoadCount[]   = {0xf4000200u, 0xfa000000u};
constexpr uint32_t LoadTarget[]  = {0xf4080301u, 0xfa000000u};
constexpr uint32_t WaitAll       = 0xbf8c0000u;
constexpr uint32_t CountIsZero   = 0xbf068008u;
constexpr uint32_t SkipStore     = 0xbf850003u;
constexpr uint32_t MoveCount     = 0x7e040208u;
constexpr uint32_t StoreCount[]  = {0xe0701000u, 0x80030201u};
constexpr uint32_t EndProgram    = 0xbf810000u;
constexpr uint32_t Format32UInt  = 20;
constexpr uint32_t SwizzleXyzw   = 0xfacu;
constexpr uint64_t TargetAddress = 0x200820000ull;
constexpr uint32_t Records       = 256u;

int g_failures = 0;

void Check(bool condition, const std::string& what) {
	if (condition) return;
	std::fprintf(stderr, "FAILED: %s\n", what.c_str());
	g_failures++;
}

using Sharp = std::array<uint32_t, 4>;

constexpr Sharp StorageSharp(uint64_t address, uint32_t bytes) {
	return {static_cast<uint32_t>(address), static_cast<uint32_t>(address >> 32u), bytes,
	        SwizzleXyzw | (Format32UInt << 12u) | (1u << 24u) | (3u << 28u)};
}

struct alignas(4096) GuestWords {
	uint32_t word = 0;
	uint32_t pad[15] {};
	Sharp    sharps[2] {};
};

GuestWords            g_guest;
std::vector<uint64_t> g_gpu_written;

constexpr uint16_t Sinks(std::initializer_list<SrtLoadClass> classes, uint16_t extra = 0) {
	uint16_t sinks = extra;
	for (const auto load_class: classes) sinks |= IR::SrtLoadBit(load_class);
	return sinks;
}

IR::SrtWalkerPage GuestPage(uint64_t address, bool, const uint8_t*& backing) {
	backing           = nullptr;
	const auto first  = reinterpret_cast<uint64_t>(&g_guest);
	const bool inside = address >= first && address < first + sizeof(g_guest);
	return inside ? IR::SrtWalkerPage::PerLoad : IR::SrtWalkerPage::Direct;
}

bool GuestRead(uint64_t address, uint32_t& value, bool ahead, const uint8_t*) {
	if (ahead && std::ranges::find(g_gpu_written, address) != g_gpu_written.end()) return false;
	std::memcpy(&value, reinterpret_cast<const void*>(address), sizeof(value));
	return true;
}

void RecordLoad(void*, uint64_t, uint32_t, uint32_t) {}

IR::ResourcePlan Translate(const std::vector<uint32_t>& code, uint64_t hash) {
	ShaderComputeInputInfo info {};
	info.wave_size          = 64;
	info.host_subgroup_size = 64;
	info.threads_num[0]     = 64;
	info.threads_num[1]     = 1;
	info.threads_num[2]     = 1;
	info.thread_ids_num     = 1;
	const std::vector<uint32_t> user_data(16, 0u);
	Recompiler::CompileOptions  options;
	options.stage              = ShaderType::Compute;
	options.wave_size          = 64;
	options.shader_hash        = hash;
	options.dump_ir            = false;
	options.user_data          = user_data;
	options.input_info.compute = &info;
	const auto source          = Recompiler::PrepareShaderSource(code, options);
	auto       translated      = Recompiler::TranslateProgram(source.decoded, options);
	return IR::ExtractResourcePlan(translated.program);
}

struct Walk {
	bool                    ok = false;
	IR::SrtRefusalClasses   classes;
	std::vector<uint32_t>   faulted;
};

Walk Materialize(const IR::ResourcePlan& plan, const std::array<uint32_t, 16>& user_data,
                 std::vector<uint64_t> gpu_written) {
	g_gpu_written = std::move(gpu_written);
	IR::SrtRuntime runtime {.user_data = user_data};
	runtime.record_load = RecordLoad;
	IR::ResourceSnapshot       snapshot;
	IR::ResourceSpecialization specialization;
	IR::ClearSrtFaultedLoadsOfThread();
	Walk walk;
	walk.ok      = IR::MaterializeResources(plan, runtime, snapshot, specialization);
	const auto faulted = IR::SrtFaultedLoadsOfThread();
	walk.faulted.assign(faulted.begin(), faulted.end());
	walk.classes = IR::ClassifySrtRefusal(plan.srt_load_sinks, faulted);
	return walk;
}

std::array<uint32_t, 16> Pointers(const void* first, const void* second = nullptr) {
	const auto a = reinterpret_cast<uint64_t>(first);
	const auto b = reinterpret_cast<uint64_t>(second);
	return {static_cast<uint32_t>(a), static_cast<uint32_t>(a >> 32u), static_cast<uint32_t>(b),
	        static_cast<uint32_t>(b >> 32u)};
}

std::vector<uint64_t> SharpWords(const Sharp& sharp) {
	std::vector<uint64_t> words;
	for (const auto& word: sharp) words.push_back(reinterpret_cast<uint64_t>(&word));
	return words;
}

void Reduction() {
	using enum SrtLoadClass;
	const std::vector<uint16_t> sinks {Sinks({Address, Data}), Sinks({Address, VSharp}), Sinks({Data}),
	                                   Sinks({VSharp}, IR::SrtWrittenVSharpSink), Sinks({KeyTable, Address}),
	                                   Sinks({ImageTable, VSharp}), Sinks({TSharp, ImageTable}), 0};
	Check(IR::SrtLoadClassOf(sinks[0]) == Address, "a word that only forms addresses of data loads is address_only");
	Check(IR::SrtLoadClassOf(sinks[1]) == VSharp, "an address of a V# word takes the V# class");
	Check(IR::SrtLoadClassOf(sinks[2]) == Data, "a data word is data");
	Check(IR::SrtLoadClassOf(sinks[4]) == KeyTable, "an address of a key table takes the key table class");
	Check(IR::SrtLoadClassOf(sinks[5]) == ImageTable, "image tables come after V# words");
	Check(IR::SrtLoadClassOf(sinks[6]) == TSharp, "T# words come after image tables");
	Check(IR::SrtLoadClassOf(sinks[7]) == Other, "a word with no known sink is other");

	const std::vector<uint32_t> faulted {0, 1, 3, IR::SrtUnknownLoad};
	const auto refusal = IR::ClassifySrtRefusal(sinks, faulted);
	Check(refusal.first == Address, "the first faulted load gives the first class");
	Check(refusal.mask == (IR::SrtLoadBit(Address) | IR::SrtLoadBit(VSharp) | IR::SrtLoadBit(Other)),
	      "every faulted load adds its class");
	Check(refusal.loads[static_cast<uint32_t>(Address)] == 1 && refusal.loads[static_cast<uint32_t>(VSharp)] == 2 &&
	          refusal.loads[static_cast<uint32_t>(Other)] == 1,
	      "loads are counted per class");
	Check(refusal.written_vsharp_loads == 1, "a V# of a written buffer is counted as written through");
	Check(IR::ClassifySrtRefusal(sinks, {}).mask == 0, "no faulted load, no class");

	const auto sets = E2Stats::CoverageSets();
	const auto covered = [&](std::vector<uint32_t> loads) {
		E2Stats::Refusal value {.classes = IR::ClassifySrtRefusal(sinks, loads), .cause = E2Stats::Cause::AheadWalk};
		uint32_t         mask = 0;
		for (uint32_t set = 0; set < sets.size(); set++) {
			if (E2Stats::Covers(sets[set].classes, value)) mask |= 1u << set;
		}
		return mask;
	};
	Check(covered({0}) == 0x1fu, "an address_only refusal is covered by every set");
	Check(covered({0, 4}) == 0x1cu, "a key table refusal needs the key table sets");
	Check(covered({1}) == 0x18u, "a V# refusal needs a set with V# words");
	Check(covered({3}) == 0u, "a written GPU-formed V# keeps its refusal on today's path");
	Check(covered({5}) == 0u, "an image table refusal is covered by no set");
	Check(covered({2}) == 0x10u, "a data refusal is covered by the widest set only");
	E2Stats::Refusal unsplit {.cause = E2Stats::Cause::RevalidationWalk};
	Check(!E2Stats::Covers(sets[4].classes, unsplit), "a walker word of a revalidation is never covered");
}

void PlanClasses() {
	const auto plan = Translate({ShiftLane, LoadCount[0], LoadCount[1], LoadTarget[0], LoadTarget[1], WaitAll,
	                             CountIsZero, SkipStore, MoveCount, StoreCount[0], StoreCount[1], EndProgram},
	                            0x5eu);
	Check(!plan.srt_load_sinks.empty(), "the walker classes its loads");
	g_guest.word      = 7u;
	g_guest.sharps[0] = StorageSharp(TargetAddress, Records * 4u);
	const auto user_data = Pointers(&g_guest.word, &g_guest.sharps[0]);

	const auto clean = Materialize(plan, user_data, {});
	Check(clean.ok && clean.faulted.empty(), "a walk on clean lines records no faulted load");

	const auto sharp = Materialize(plan, user_data, SharpWords(g_guest.sharps[0]));
	Check(!sharp.ok, "a GPU-written V# refuses the walk");
	Check(sharp.faulted.size() == 4u, "each word of the GPU-written V# is a faulted load");
	Check(sharp.classes.first == SrtLoadClass::VSharp && sharp.classes.mask == IR::SrtLoadBit(SrtLoadClass::VSharp),
	      "the words of a V# are vsharp_key");
	Check(sharp.classes.written_vsharp_loads == 4u, "the V# of the stored buffer is written through");

	const auto count = Materialize(plan, user_data, {reinterpret_cast<uint64_t>(&g_guest.word)});
	Check(!count.ok && count.faulted.size() == 1u, "a GPU-written count refuses the walk");
	Check(count.classes.first == SrtLoadClass::Data, "a count used as a branch and as data is data");

	const auto both = Materialize(plan, user_data,
	                              {reinterpret_cast<uint64_t>(&g_guest.word),
	                               reinterpret_cast<uint64_t>(&g_guest.sharps[0][0])});
	Check(both.classes.mask == (IR::SrtLoadBit(SrtLoadClass::Data) | IR::SrtLoadBit(SrtLoadClass::VSharp)),
	      "a refusal on two lines involves both classes");
	Check(both.classes.first == (both.faulted.empty() || both.faulted[0] >= plan.srt_load_sinks.size()
	                                 ? SrtLoadClass::Other
	                                 : IR::SrtLoadClassOf(plan.srt_load_sinks[both.faulted[0]])),
	      "the first class is the class of the first faulted load");
	IR::SetSrtLoadFaultRecording(false);
	const auto off = Materialize(plan, user_data, SharpWords(g_guest.sharps[0]));
	Check(!off.ok && off.faulted.empty(), "with recording off the walk is refused and records nothing");
	IR::SetSrtLoadFaultRecording(true);
}

void ScalarBufferSharp() {
	const auto plan = Translate({0xf4080100u, 0xfa000000u, 0xf4080200u, 0xfa000010u, 0xbf8cc07fu, 0xf4200302u,
	                             0xfa000000u, 0xbf8cc07fu, 0xbf0d800cu, 0x858e80c1u, 0x7d8a0080u, 0x87ea6a0eu,
	                             0xbe90246au, 0xbf880002u, 0xe0702000u, 0x80020000u, EndProgram},
	                            0x5fu);
	g_guest.word      = 0xffff0001u;
	g_guest.sharps[0] = StorageSharp(reinterpret_cast<uint64_t>(&g_guest.word), 16u);
	g_guest.sharps[1] = StorageSharp(TargetAddress, Records * 4u);
	const auto user_data = Pointers(&g_guest.sharps[0]);
	const auto sharp     = Materialize(plan, user_data, SharpWords(g_guest.sharps[0]));
	Check(!sharp.ok && sharp.faulted.size() == 4u, "a GPU-written V# of a scalar buffer refuses the walk");
	Check(sharp.classes.first == SrtLoadClass::VSharp && sharp.classes.mask == IR::SrtLoadBit(SrtLoadClass::VSharp),
	      "a scalar buffer V# bound as a resource is vsharp_key, even when it only loads a control word");
	const auto mask = Materialize(plan, user_data, {reinterpret_cast<uint64_t>(&g_guest.word)});
	Check(!mask.ok && mask.faulted.size() == 1u && mask.classes.first == SrtLoadClass::Data,
	      "a GPU-written lane mask is data");
}

void PointerChain() {
	const auto plan = Translate({ShiftLane, 0xf4040200u, 0xfa000000u, LoadTarget[0], LoadTarget[1], WaitAll, 0xf4000284u,
	                             0xfa000000u, WaitAll, 0x7e04020au, StoreCount[0], StoreCount[1], EndProgram},
	                            0x60u);
	const auto count = [&](SrtLoadClass load_class) {
		return std::ranges::count_if(plan.srt_load_sinks,
		                             [&](uint16_t sinks) { return sinks != 0 && IR::SrtLoadClassOf(sinks) == load_class; });
	};
	Check(count(SrtLoadClass::Address) == 2, "both words of a pointer to a data word are address_only");
	Check(count(SrtLoadClass::VSharp) == 4, "the four words of the stored buffer's V# are vsharp_key");
	const auto pointer   = reinterpret_cast<uint64_t>(&g_guest.word);
	g_guest.pad[0]       = static_cast<uint32_t>(pointer);
	g_guest.pad[1]       = static_cast<uint32_t>(pointer >> 32u);
	g_guest.word         = 7u;
	g_guest.sharps[0]    = StorageSharp(TargetAddress, Records * 4u);
	const auto user_data = Pointers(&g_guest.pad[0], &g_guest.sharps[0]);
	const auto clean     = Materialize(plan, user_data, {});
	Check(clean.ok && clean.faulted.empty(), "the pointer chain walks on clean lines");
	const auto leaf = Materialize(plan, user_data, {pointer});
	Check(leaf.faulted.empty() || leaf.classes.first == SrtLoadClass::Data,
	      "the data word at the end of a pointer chain is data when it faults");
}

}

int main() {
	IR::SetSrtWalkerGuestMemory({&GuestPage, &GuestRead});
	IR::SetSrtLoadFaultRecording(true);
	Reduction();
	PlanClasses();
	ScalarBufferSharp();
	PointerChain();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d check(s) failed\n", g_failures);
		return 1;
	}
	std::printf("kyty_shim_e2_load_classes: ok\n");
	return 0;
}
