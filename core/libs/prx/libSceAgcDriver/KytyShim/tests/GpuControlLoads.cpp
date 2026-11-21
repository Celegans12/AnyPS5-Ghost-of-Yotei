#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "graphics/shader/shaderCompiler.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace Libs::Graphics;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;
namespace IR         = Recompiler::IR;

constexpr uint32_t ShiftLane      = 0x34020082u;
constexpr uint32_t LoadCount[]    = {0xf4000200u, 0xfa000000u};
constexpr uint32_t LoadTarget[]   = {0xf4080301u, 0xfa000000u};
constexpr uint32_t WaitAll        = 0xbf8c0000u;
constexpr uint32_t CountIsZero    = 0xbf068008u;
constexpr uint32_t SkipStore      = 0xbf850003u;
constexpr uint32_t MoveCount      = 0x7e040208u;
constexpr uint32_t StoreCount[]   = {0xe0701000u, 0x80030201u};
constexpr uint32_t EndProgram     = 0xbf810000u;
constexpr uint32_t Format32UInt   = 20;
constexpr uint32_t SwizzleXyzw    = 0xfacu;
constexpr uint64_t TargetAddress  = 0x200820000ull;
constexpr uint32_t Records        = 256u;

int g_failures = 0;

void Check(bool condition, const std::string& what) {
	if (condition) return;
	std::fprintf(stderr, "FAILED: %s\n", what.c_str());
	g_failures++;
}

using Sharp = std::array<uint32_t, 4>;

struct alignas(64) GuestWords {
	uint32_t count = 0;
	uint32_t pad[15] {};
	Sharp    target {};
};

GuestWords g_guest;
uint64_t   g_produced = 0;

struct Load {
	uint64_t address   = 0;
	uint32_t value     = 0;
	uint32_t flat_slot = 0;
};

std::vector<Load> g_loads;

void RecordLoad(void*, uint64_t address, uint32_t value, uint32_t flat_slot) {
	g_loads.push_back({address, value, flat_slot});
}

bool Produced(uint64_t address) {
	return address == g_produced;
}

IR::ResourcePlan Translate() {
	const std::vector<uint32_t> code {ShiftLane,  LoadCount[0], LoadCount[1], LoadTarget[0], LoadTarget[1],
	                                  WaitAll,    CountIsZero,  SkipStore,    MoveCount,     StoreCount[0],
	                                  StoreCount[1], EndProgram};
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
	options.shader_hash        = 0x52u;
	options.dump_ir            = false;
	options.user_data          = user_data;
	options.input_info.compute = &info;
	const auto source          = Recompiler::PrepareShaderSource(code, options);
	auto       translated      = Recompiler::TranslateProgram(source.decoded, options);
	return IR::ExtractResourcePlan(translated.program);
}

struct Walk {
	bool                       ok = false;
	IR::ResourceSnapshot       snapshot;
	IR::ResourceSpecialization specialization;
	std::vector<Load>          loads;
};

Walk Materialize(const IR::ResourcePlan& plan, uint32_t count, bool ahead, bool produced) {
	g_guest.count = count;
	g_produced    = produced ? reinterpret_cast<uint64_t>(&g_guest.count) : 0;
	g_loads.clear();
	const auto count_address  = reinterpret_cast<uint64_t>(&g_guest.count);
	const auto target_address = reinterpret_cast<uint64_t>(&g_guest.target);
	const std::array<uint32_t, 16> user_data {static_cast<uint32_t>(count_address),
	                                          static_cast<uint32_t>(count_address >> 32u),
	                                          static_cast<uint32_t>(target_address),
	                                          static_cast<uint32_t>(target_address >> 32u)};
	IR::SrtRuntime runtime {.user_data = user_data};
	if (ahead) {
		runtime.record_load  = RecordLoad;
		runtime.gpu_produced = Produced;
	}
	Walk result;
	result.ok    = IR::MaterializeResources(plan, runtime, result.snapshot, result.specialization);
	result.loads = g_loads;
	return result;
}

uint32_t TargetSlot(const IR::ResourcePlan& plan) {
	for (uint32_t i = 0; i < plan.info.buffers.size(); i++) {
		if (plan.info.buffers[i].written) return i;
	}
	return UINT32_MAX;
}

bool Holds(const Walk& walk, uint32_t slot, const Sharp& sharp) {
	if (slot >= walk.snapshot.buffers.size()) return false;
	const auto& value = walk.snapshot.buffers[slot];
	return value.dword_count == 4u && std::equal(sharp.begin(), sharp.end(), value.dwords.begin());
}

const Load* CountLoad(const Walk& walk) {
	const auto address = reinterpret_cast<uint64_t>(&g_guest.count);
	for (const auto& load: walk.loads) {
		if (load.address == address) return &load;
	}
	return nullptr;
}

void ControlAndData() {
	const auto plan = Translate();
	const auto slot = TargetSlot(plan);
	Check(slot != UINT32_MAX, "the stored buffer is planned");
	Check(!plan.control_flow.empty(), "the branch on the count is part of the plan");
	if (slot == UINT32_MAX) return;
	const Sharp target {static_cast<uint32_t>(TargetAddress), static_cast<uint32_t>(TargetAddress >> 32u), Records * 4u,
	                    SwizzleXyzw | (Format32UInt << 12u) | (1u << 24u) | (3u << 28u)};
	g_guest.target = target;

	const auto taken = Materialize(plan, 7u, false, false);
	Check(taken.ok, "a nonzero count materializes");
	Check(Holds(taken, slot, target), "a nonzero count binds the stored buffer");
	Check(std::ranges::find(taken.snapshot.flattened_srt, 7u) != taken.snapshot.flattened_srt.end(),
	      "a nonzero count reaches the shader data");

	const auto skipped = Materialize(plan, 0u, false, false);
	Check(skipped.ok, "a zero count materializes");
	Check(!Holds(skipped, slot, target), "a zero count leaves the stored buffer out");

	const auto read_ahead = Materialize(plan, 0u, true, false);
	const auto* read      = CountLoad(read_ahead);
	Check(read_ahead.ok && read != nullptr, "ahead, a count the GPU does not produce is read");
	Check(read != nullptr && read->flat_slot == IR::NoFlatSlot,
	      "ahead, a count the GPU does not produce is checked as an exact value");
	Check(!Holds(read_ahead, slot, target), "ahead, a zero count the GPU does not produce leaves the buffer out");

	const auto produced = Materialize(plan, 0u, true, true);
	const auto* deferred = CountLoad(produced);
	Check(produced.ok, "ahead, a count the GPU produces materializes");
	Check(deferred != nullptr, "ahead, a count the GPU produces is recorded");
	if (deferred == nullptr) return;
	Check(deferred->flat_slot != IR::NoFlatSlot && deferred->flat_slot != IR::ZeroTestSlot &&
	          !IR::IsDescriptorSlot(deferred->flat_slot),
	      "ahead, a count the GPU produces is a word of the shader data, not a value to check");
	Check(deferred->flat_slot < produced.snapshot.flattened_srt.size() &&
	          deferred->flat_slot < taken.snapshot.flattened_srt.size() &&
	          taken.snapshot.flattened_srt[deferred->flat_slot] == 7u,
	      "the recorded word is the slot of the count in the shader data");
	Check(Holds(produced, slot, target), "ahead, a count the GPU produces binds the buffer of both branches");
	Check(produced.specialization == taken.specialization,
	      "the variant of both branches is the variant of the taken branch");
}

}

int main() {
	ControlAndData();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d check(s) failed\n", g_failures);
		return 1;
	}
	std::printf("kyty_shim_gpu_control_loads: ok\n");
	return 0;
}
