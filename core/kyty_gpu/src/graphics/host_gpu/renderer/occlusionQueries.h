#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_OCCLUSIONQUERIES_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_OCCLUSIONQUERIES_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <cstdint>
#include <vector>

namespace Libs::Graphics {

class CommandBuffer;
class RenderContext;

// KYTY_FIX_OCCLUSION_QUERIES=1 (default off; read once, logged once as FIX_OCCLUSION_QUERIES): the guest's
// occlusion counters (EVENT_WRITE PIXEL_PIPE_STAT_DUMP) are backed by host Vulkan occlusion queries and written into
// guest memory by the GPU, at the point of the command stream where the guest dumped them. Without it every dump
// publishes a synthetic count that always reads "visible" (Ghost of Yotei: the sun/moon flares of
// Frame>Postprocessing>Bloom shine through walls and mountains, because the CS that turns 4 visible/total query
// pairs into flare visibility reads 16/16 = 1.0 every frame).
[[nodiscard]] bool OcclusionQueriesRequested();

// The guest's occlusion queries. Everything runs on the GPU thread, in command order.
//
// Guest layout (PS5, GFX10 DBs): a dump at A writes the 64-bit sample counter of each of the 16 DBs to
// A + db * 16 and sets bit 63 once the value is there; a query is a begin dump at A and an end dump at A + 8, and
// its result is the sum over the DBs of (end - begin). The PS5 counts the samples that pass the depth/stencil
// tests of every draw between the two dumps, also of draws that write no color, depth or stencil (a "total" pass
// with the depth test off draws nothing at all but still counts).
//
// Host side: while a guest query is open, every render pass instance in which a guest draw is recorded is counted
// by one host query (begun lazily right before the first guest draw of the instance, ended before the instance
// ends or at the next dump), so the emulator's own blits and copies are never counted. Host queries are numbered
// in one sequence, so the samples of any guest query (also nested or interleaved ones) are exactly the host
// queries numbered between its begin and its end dump. A begin dump writes {0, ready} to the 16 begin slots; an end
// dump writes {0, ready} to the 16 end slots and then copies the host results (32-bit, the low dword of each slot)
// into the end slots of DB 1..15 with vkCmdCopyQueryPoolResults; DB 0 holds the samples of host queries that had to
// be summed on the CPU when a guest query spanned more than 15 of them (rare: a submit and wait). The guest's sum
// over the DBs is then the exact sample count, written on the GPU timeline before any later command reads it.
class OcclusionQueries {
public:
	static constexpr uint32_t DbCount  = 16;
	static constexpr uint64_t DbStride = 16;
	// Bytes a dump writes, from its address: the 16 slots of one half (begin or end) of the 256-byte record.
	static constexpr uint64_t DumpBytes = (DbCount - 1u) * DbStride + sizeof(uint64_t);
	// Bit 63 of a slot (ready), in its high dword.
	static constexpr uint32_t ReadyHigh = 0x80000000u;
	// Host query slots (a ring) and the slots kept reset ahead of the next allocation.
	static constexpr uint32_t Capacity   = 8192;
	static constexpr uint32_t ResetBatch = 256;
	static constexpr uint32_t ResetLow   = 16;
	// Host results an end dump copies into DB 1..15 (DB 0: the CPU-summed part).
	static constexpr uint32_t MaxCopied = DbCount - 1u;
	// A begin dump without its end is forgotten after this many presented frames.
	static constexpr uint32_t MaxOpenFrames = 8;

	explicit OcclusionQueries(RenderContext& context);
	~OcclusionQueries();
	KYTY_CLASS_NO_COPY(OcclusionQueries);

	// Host queries back the counters (KYTY_FIX_OCCLUSION_QUERIES=1 and the query pool exists).
	[[nodiscard]] bool Enabled() const noexcept { return m_pool != nullptr; }
	// A guest query is open: the samples of draws count on the PS5, also those of draws without any output.
	[[nodiscard]] bool Counting() const noexcept { return !m_open.empty(); }

	// KYTY_FIX_OCCLUSION_QUERIES: PIXEL_PIPE_STAT_DUMP to `address` (8-byte aligned) at this point of the command
	// stream. False when the host queries are not available (the caller then publishes the synthetic count).
	[[nodiscard]] bool Dump(CommandBuffer& buffer, uint64_t address);
	// Without host queries: the dump the caller published synthetically, only to know which guest queries are open
	// (draw-drop diagnostics).
	void NoteSyntheticDump(uint64_t address);
	// Right before a guest draw command is recorded inside a render pass instance.
	void BeforeDraw(vk::CommandBuffer recorder);
	// CommandBuffer::EndRendering: before vkCmdEndRendering (ends the active host query in the instance) and after
	// it (resets slots ahead; outside any render pass instance).
	void BeforeEndRendering(vk::CommandBuffer recorder);
	void AfterEndRendering(vk::CommandBuffer recorder);

private:
	struct Open {
		uint64_t address    = 0; // begin dump
		uint64_t first_slot = 0; // first host query of the guest query
		uint32_t frame      = 0; // DebugPresentCount() at the begin dump
		bool     uncounted  = false;
	};

	// Pairs a dump with the open guest queries: the open entry it ends (end dump at begin + 8), or m_open.end() for
	// a begin dump. Forgets begin dumps older than MaxOpenFrames.
	std::vector<Open>::iterator Pair(uint64_t address);
	[[nodiscard]] bool          CreatePool(vk::CommandBuffer recorder);
	[[nodiscard]] uint64_t      OldestNeeded() const;
	void                        ResetAhead(vk::CommandBuffer recorder);
	// Submits, waits and sums the host results of [first, end) on the CPU.
	[[nodiscard]] uint64_t      SumOnCpu(uint64_t first, uint64_t end);
	// Records the guest writes of a dump: the 16 slots at `address` get {low, ready} from `low_dwords`, then
	// (end dumps) the results of host queries [first, end) go to the low dwords of DB 1..
	void                        WriteSlots(CommandBuffer& buffer, uint64_t address,
	                                       const std::array<uint32_t, DbCount>& low_dwords, uint64_t first,
	                                       uint64_t end);

	RenderContext&        m_context;
	vk::Device            m_device = nullptr;
	vk::QueryPool         m_pool   = nullptr;
	vk::QueryControlFlags m_control;
	bool                  m_failed      = false;
	bool                  m_active      = false;
	uint32_t              m_active_slot = 0;
	uint64_t              m_next_slot   = 0; // next host query (sequence number; ring index = number % Capacity)
	uint64_t              m_reset_end   = 0; // host queries before this number are reset for their current use
	std::vector<Open>     m_open;
	uint64_t              m_dumps = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_OCCLUSIONQUERIES_H_
