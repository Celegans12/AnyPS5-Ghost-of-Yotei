#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_SCOUT_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_SCOUT_H_

#include "graphics/guest_gpu/pm4.h"

#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <span>
#include <vector>

namespace Libs::Graphics {
class RenderContext;
using GpuResourceManager = RenderContext;
}

namespace Libs::Graphics::Scout {

[[nodiscard]] bool Running() noexcept;

void Start(RenderContext& resources);
void Stop();

[[nodiscard]] constexpr bool IsDraw(uint32_t opcode) {
	return opcode == Pm4::IT_DRAW_INDEX_2 || opcode == Pm4::IT_DRAW_INDEX_OFFSET_2 || opcode == Pm4::IT_DRAW_INDEX_AUTO ||
	       opcode == Pm4::IT_DRAW_INDIRECT || opcode == Pm4::IT_DRAW_INDEX_INDIRECT ||
	       opcode == Pm4::IT_DRAW_INDIRECT_MULTI || opcode == Pm4::IT_DRAW_INDEX_INDIRECT_MULTI;
}

void BeginSubmission(uint64_t sequence);
void NoteDraw(const uint32_t* packet);
void SkipDraws(uint32_t count);
void NoteDrawAt(const uint32_t* packet, uint64_t sequence, uint32_t ordinal);
void NoteWrite(const void* address, uint32_t size);
void NoteHostWrite(uint64_t address, uint64_t size, bool write_back = false);
inline void NoteHostWriteIfOn(uint64_t address, uint64_t size, bool write_back = false) {
	NoteHostWrite(address, size, write_back);
}

struct Run {
	std::array<uint16_t, 3>   address {};
	uint16_t                  records = NotBounded;
	int32_t                   immediate = 0;
	uint16_t                  first = 0, count = 0;
	uint16_t                  at = 0;
	static constexpr uint16_t NotBounded = 0xffff;
};
struct Segment {
	uint16_t first = 0, count = 0;
};

struct View {
	std::array<uint64_t, 3> key {};
	uint64_t                version = 0;
	uint32_t                stages  = 0;
	struct Stage {
		std::vector<Run>      runs;
		std::vector<Segment>  segments;
		std::vector<uint16_t> level_ends, image_words, buffer_words;
		std::vector<uint8_t>  image_counts;
		std::vector<uint32_t> constants;
		uint32_t              user_words = 0, block_dwords = 0, lod_stats_count = 0, buffers = 0, slot_words = 0;
	};
	std::array<Stage, 2> stage;
	uint64_t             retired = 0;
};
[[nodiscard]] uint64_t NextVersion();
void Publish(View* view, uint64_t key_hash);
void Retire(uint64_t key_hash, uint64_t version);

inline constexpr uint32_t ResolvedEntries = 65536;
static_assert(std::has_single_bit(ResolvedEntries));

struct alignas(64) Ticket {
	uint64_t                                sequence = 0;
	uint32_t                                ordinal  = 0;
	uint32_t                                used     = 0;
	uint64_t                                packet   = 0;
	uint64_t                                version  = 0;
	std::array<uint64_t, 3>                 key {};
	uint64_t                                mapping_epoch = 0;
	uint64_t                                written       = 0;
	uint64_t                                first_page = 0, last_page = 0;
	std::array<std::array<uint32_t, 32>, 2> user_data {};
	std::array<uint32_t, 2>                 user_words {};
	std::array<uint16_t, 2>                 slot_words {}, buffers {};
	uint16_t                                image_words = 0, lod_stats = 0, pages = 0, stages = 0;
	uint16_t                                backing = 0;
	std::array<uint16_t, 2>                 segments {};
	static constexpr uint32_t Capacity = 928;
	static constexpr uint32_t PageBacking = 0x80000000u;
	uint32_t                  data[Capacity];
};
static_assert(sizeof(Ticket) <= 4096);
[[nodiscard]] inline const uint32_t* Pages(const Ticket& ticket) {
	return ticket.data + ticket.slot_words[0] + ticket.slot_words[1] + ticket.segments[0] + ticket.segments[1] +
	       ticket.image_words + ticket.lod_stats + 5u * (ticket.buffers[0] + ticket.buffers[1]);
}
[[nodiscard]] inline uint64_t PageAt(const uint32_t* pages, uint32_t i) {
	return pages[2 * i] | uint64_t {pages[2 * i + 1] & ~Ticket::PageBacking} << 32u;
}
[[nodiscard]] bool WrittenSince(const Ticket& ticket, bool& write_back);

[[nodiscard]] bool EvaluateAhead(const std::array<uint64_t, 3>& key, uint64_t key_hash, std::span<const uint32_t> gs_user,
                                 std::span<const uint32_t> ps_user, Ticket& ticket);

}

#endif
