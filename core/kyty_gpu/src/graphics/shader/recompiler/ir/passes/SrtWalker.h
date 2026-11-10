#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace Libs::Graphics::ShaderRecompiler::IR {

class Value;

using SrtMemoryReader = bool (*)(void* userdata, uint64_t address, std::span<uint32_t> values);

// KYTY_TRACE_SRT_CLASSES=1 (diagnostic, default off; M1 of the "GPU reads shader-only SRT words" project): what the SRT
// read in progress on this thread is for, so the GPU thread can count reads, waits and Adopt work by class and address
// region. Class: 0 not a tape read, 1 the CPU needs the value (descriptor, pointer, branch, fill, indirect key or
// selector), 2 only the shader reads it (a flattened SRT word), 3 a fused run with words of both kinds.
bool                        SrtClassTrace();
inline thread_local uint8_t t_srt_read_class = 0;
// 0 a real call's walk, 1 Adopt re-verifying a resolved log.
inline thread_local uint8_t t_srt_read_phase = 0;
// ReadsUnchangedFast: the log entry that read other bytes (SIZE_MAX: none known).
inline thread_local size_t t_srt_changed_entry = SIZE_MAX;

namespace SrtClasses {
inline constexpr uint32_t Classes = 4;
inline constexpr uint32_t Regions = 4; // SRT ring, data ring, 0x50xxxxxxxx, other
[[nodiscard]] inline uint32_t Region(uint64_t address) {
	if (address >= 0x2010000000ull && address < 0x2015E00000ull) return 0;
	if (address >= 0x2002600000ull && address < 0x2010000000ull) return 1;
	if ((address >> 32u) == 0x50u) return 2;
	return 3;
}
// GPU-thread counters (plain integers: only the GPU thread writes and prints them).
struct Counters {
	uint64_t reads[2][Classes][Regions]   = {}; // [phase] reads through the GPU thread's memory hooks
	uint64_t bytes[2][Classes][Regions]   = {};
	uint64_t wait_ns[2][Classes][Regions] = {}; // waits for GPU writes / downloads inside those reads (and page proofs)
	uint64_t waits[2][Classes][Regions]   = {}; // of them >= 20 us
	uint64_t adopted[Classes][Regions]    = {}; // entries of adopted logs
	uint64_t adopted_bytes[Classes][Regions] = {};
	uint64_t rejected[Classes + 1][Regions]  = {}; // rejected adopts by the first changed entry ([Classes]: not known)
};
[[nodiscard]] inline Counters& Get() {
	static Counters counters;
	return counters;
}
inline void AddWait(uint32_t cls, uint64_t address, uint64_t ns) {
	auto&          c      = Get();
	const uint32_t region = Region(address);
	c.wait_ns[t_srt_read_phase & 1u][cls & 3u][region] += ns;
	if (ns >= 20000u) c.waits[t_srt_read_phase & 1u][cls & 3u][region]++;
}
} // namespace SrtClasses

// Upstream's image heap (27dc7c281): image descriptors at base + first + k * step, k < count, within size bytes
// that a sampled root selects by GPU-computed byte offset. Our MaterializeIndirectDescriptor does not request heaps
// (SESJA-SYNC1: the resource system stays ours); the types keep imageHeap.cpp compiling.
struct ImageHeapRequest {
	uint64_t             base  = 0;
	uint64_t             size  = 0;
	uint64_t             step  = 0;
	uint64_t             first = 0;
	uint64_t             count = 0;
	const ImageResource* root  = nullptr;
};

struct ImageHeapView {
	uint64_t                         slots      = 0; // Device address of one slot per heap dword.
	uint32_t                         slot_count = 0; // Keys past it select a trailing null slot.
	std::span<const DescriptorValue> kinds; // One representative descriptor per kind.
};

class ImageHeapSource {
public:
	// False when the host does not keep the table resident.
	virtual bool Acquire(const ImageHeapRequest& request, ImageHeapView& view) = 0;

protected:
	~ImageHeapSource() = default;
};

struct SrtRuntime {
	std::span<const uint32_t> user_data;
	uint64_t                  shader_base                = 0;
	SrtMemoryReader           read_memory                = nullptr;
	void*                     userdata                   = nullptr;
	SrtMemoryReader           read_specialization_memory = nullptr;
	std::span<const uint32_t> workgroup_counts;
	uint64_t                  indirect_args_address = 0;
	ImageHeapSource*          image_heap            = nullptr; // upstream's image heap; never set (see above)
};

class SrtReadCapture {
public:
	SrtReadCapture(SrtRuntime source, std::vector<std::pair<uint64_t, uint64_t>>& ranges):
	    m_source(source), m_ranges(ranges) {}
	SrtRuntime ObservedRuntime();

private:
	static bool ReadStrict(void* userdata, uint64_t address, std::span<uint32_t> values);
	static bool ReadOrdinary(void* userdata, uint64_t address, std::span<uint32_t> values);

	SrtRuntime                                  m_source;
	std::vector<std::pair<uint64_t, uint64_t>>& m_ranges;
};

// Retained reads no longer need the guest instruction PC. A clean read evaluates
// its address and bounds through the strict reader as well as the final DWORD.
struct SrtReadFlags {
	uint32_t index = 0;
	uint32_t clean = 0;
};

enum class RuntimeValueType { Any, Integer, ImmutableInteger };

bool ValidateRuntimeValue(const ResourcePlan& program, Value value,
                          RuntimeValueType type = RuntimeValueType::Any);
// Uses the strict reader for values that affect shader specialization.
SrtRuntime CleanRuntime(SrtRuntime runtime);

// One memoized evaluation session shared by the entire shader resource refresh.
class SrtWalker {
public:
	SrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
	          SrtWalker* clean_evaluator = nullptr, Value active_mask = {});
	~SrtWalker();
	SrtWalker(const SrtWalker&)            = delete;
	SrtWalker& operator=(const SrtWalker&) = delete;

	bool Evaluate(Value value, uint32_t& result);
	bool EvaluateDescriptor(uint32_t source, DescriptorValue& result);
	// Refreshes reachable scalar reads and active descriptor sources in one walk.
	bool RefreshFlatBuffer(std::vector<uint32_t>& flat);

private:
	static ResourcePlan::EvaluationContext& AcquireContext(const ResourcePlan& program);
	static float Float32(uint64_t bits);
	bool EvaluateWide(Value value, uint64_t& result);
	bool Arg(const Inst& inst, size_t index, uint64_t& result);
	bool EvaluatePhi(const Inst& inst, uint64_t& result);
	bool EvaluateExtract(const Inst& inst, uint64_t& result);
	bool EvaluateRawRead(const Inst& inst, uint64_t& result);
	bool EvaluateInst(const Inst& inst, uint64_t& result);

	const ResourcePlan&              m_program;
	SrtRuntime                      m_runtime;
	SrtWalker*                      m_clean_evaluator = nullptr;
	Value                           m_active_mask;
	ResourcePlan::EvaluationContext& m_context;
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_ */
