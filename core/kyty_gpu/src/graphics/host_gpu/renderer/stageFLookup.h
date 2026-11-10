#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_STAGEFLOOKUP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_STAGEFLOOKUP_H_

#include "common/slotVector.h"

#include <cstdint>
#include <memory>

// Step F2 of the stage F design. The resolve thread materializes the resource snapshot of
// upcoming draws ahead of the GPU thread; they also compute what FindBuffers will compute for that snapshot and store it
// next to the snapshot (a ResolveRecord). When ProgramCache::Get adopts the snapshot, the record moves with it (a pointer
// swap) and the draw's ShaderStageRuntime points at it.
//
// Buffers (see BufferOutcome below): the buffer cache had no lock (GPU thread only). Its page table and slot
// insertions/erasures are made under a spin lock (BufferCache::m_lookup_lock), which the GPU thread takes only for those
// changes (a few hundred per frame) and the resolve thread takes with try_lock for one slot's lookups; the GPU thread's own finds
// stay unlocked (it is the only writer). The clamp is computed under the virtual ranges' existing mutex (the one
// ClampRangeSize takes), never while holding the buffer lock.
//
// KYTY_TRACE_PERF=1: a "PERF stagef_lookup" line every ~5 s (used, misses by reason, the resolve thread's batches).
//
namespace Libs::Graphics {

class RenderContext;

namespace ShaderRecompiler::IR {
struct ResourceSnapshot;
} // namespace ShaderRecompiler::IR

namespace StageF {


// Buffers (RenderExecutor::FindBuffers: the storage-buffer V#s and the record-keyed ranges of a stage's snapshot). The
// resolve thread computes what FindBuffers computes for each: the V# decode and size rules, ClampRangeSize (with the virtual
// range generation it saw, Memory::ClampRangeSizeAhead) and BufferCache::FindBuffer (LookupBuffer, with the buffer
// registry epoch DrawRepeat::g_buffer_registry_epoch). The GPU thread uses the result for a binding when the V# words
// (or the range) equal the snapshot's, the binding is not a written sentinel V# (its size depends on the other stages)
// and both the buffer registry epoch and the range generation are unchanged; FindBuffer has no side effect when it
// finds a buffer, ClampRangeSize none when it does not clamp, so nothing is replayed.
enum class BufferOutcome : uint8_t {
	None,     // not looked up
	Found,    // the resolve thread's source holds
	Busy,     // the buffer cache's lookup lock was held by the GPU thread
	Short,    // fewer than 4 descriptor dwords
	Null,     // address or size 0 (FindBuffers binds nothing; cheap on the GPU thread)
	Clamped,  // ClampRangeSize would clamp (and log) or exit
	NoBuffer, // FindBuffer would create a buffer (or exit on an invalid range)
	Count
};

enum class BufferUse : uint8_t {
	Used,        // the resolve thread's source replaced the decode, clamp and lookup
	NoRecord,    // the stage's snapshot has no resolved record
	NotAhead,    // the record has no entry for this binding
	Words,       // the record's V# (or range) differs from the snapshot's
	Written,     // a written sentinel V# (its size depends on the other stages' reads)
	Stale,       // the buffer registry epoch moved since the resolve thread's lookup
	StaleRanges, // the virtual range generation moved since the resolve thread's clamp
	Count
};

// GPU thread.
void NoteBufferMiss(BufferOutcome outcome);
void NoteBufferUse(BufferUse use, uint64_t count = 1);
// Resolve thread: one batch of lookups (busy: the lock was held by another thread; nothing looked up).
void NoteResolveBufferBatch(bool busy, uint64_t lookups, uint64_t found);

// GPU thread (GuestGpu's PERF report, KYTY_TRACE_PERF): the "PERF stagef_lookup" line.
void ReportLookupPerf();

// Defined in descriptors.cpp.
struct ResolveRecord;
struct ResolveRecordDeleter {
	void operator()(ResolveRecord* record) const noexcept;
};
using ResolveRecordPtr = std::unique_ptr<ResolveRecord, ResolveRecordDeleter>;

// Resolve thread, after MaterializeResourcesAhead succeeded: fills `record` (allocated on first use)
// with FindBuffers' sources for `snapshot`. Read-only towards everything shared.
void BuildResolveRecord(RenderContext& renderer, const ShaderRecompiler::IR::ResourceSnapshot& snapshot,
                        ResolveRecordPtr& record);
// Resolve thread: the slot has no result; its record (if any) describes nothing.
void ClearResolveRecord(ResolveRecord* record) noexcept;

} // namespace StageF
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_STAGEFLOOKUP_H_
