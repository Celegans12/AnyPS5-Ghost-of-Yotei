#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"

#include "SceTypes.hpp"

#include <sys/mman.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace Pm4 {

constexpr uint32_t IT_NOP                       = 0x10;
constexpr uint32_t IT_SET_BASE                  = 0x11;
constexpr uint32_t IT_CLEAR_STATE               = 0x12;
constexpr uint32_t IT_INDEX_BUFFER_SIZE         = 0x13;
constexpr uint32_t IT_DISPATCH_DIRECT           = 0x15;
constexpr uint32_t IT_DISPATCH_INDIRECT         = 0x16;
constexpr uint32_t IT_SET_PREDICATION           = 0x20;
constexpr uint32_t IT_COND_EXEC                 = 0x22;
constexpr uint32_t IT_DRAW_INDIRECT             = 0x24;
constexpr uint32_t IT_DRAW_INDEX_INDIRECT       = 0x25;
constexpr uint32_t IT_INDEX_BASE                = 0x26;
constexpr uint32_t IT_DRAW_INDEX_2              = 0x27;
constexpr uint32_t IT_CONTEXT_CONTROL           = 0x28;
constexpr uint32_t IT_INDEX_TYPE                = 0x2A;
constexpr uint32_t IT_DRAW_INDIRECT_MULTI       = 0x2C;
constexpr uint32_t IT_DRAW_INDEX_AUTO           = 0x2D;
constexpr uint32_t IT_NUM_INSTANCES             = 0x2F;
constexpr uint32_t IT_DRAW_INDEX_OFFSET_2       = 0x35;
constexpr uint32_t IT_WRITE_DATA                = 0x37;
constexpr uint32_t IT_DRAW_INDEX_INDIRECT_MULTI = 0x38;
constexpr uint32_t IT_DRAW_INDEX_MULTI_INST     = 0x3A;
constexpr uint32_t IT_INDIRECT_BUFFER           = 0x3F;
constexpr uint32_t IT_COPY_DATA                 = 0x40;
constexpr uint32_t IT_PFP_SYNC_ME               = 0x42;
constexpr uint32_t IT_EVENT_WRITE               = 0x46;
constexpr uint32_t IT_RELEASE_MEM               = 0x49;
constexpr uint32_t IT_DMA_DATA                  = 0x50;
constexpr uint32_t IT_ACQUIRE_MEM               = 0x58;
constexpr uint32_t IT_SET_SH_REG_INDIRECT       = 0x63;
constexpr uint32_t IT_SET_CONFIG_REG            = 0x68;
constexpr uint32_t IT_SET_CONTEXT_REG           = 0x69;
constexpr uint32_t IT_SET_SH_REG                = 0x76;
constexpr uint32_t IT_SET_UCONFIG_REG           = 0x79;
constexpr uint32_t IT_SET_UCONFIG_REG_INDEX     = 0x7A;
constexpr uint32_t IT_GET_LOD_STATS             = 0x8E;
constexpr uint32_t IT_SET_CONTEXT_REG_INDIRECT  = 0x9F;

}

namespace {

constexpr size_t   ArenaBytes   = 16u << 20;
constexpr size_t   StreamDwords = 1u << 14;
constexpr uint64_t DataOffset   = 1u << 20;
constexpr uint64_t DataSpan     = 4u << 20;
constexpr uint64_t IndirectOffset = 8u << 20;

struct Op {
	uint32_t code;
	uint32_t minimum;
	bool     exact;
};

constexpr std::array<Op, 20> Ops = {{
    {Pm4::IT_NOP, 1, false},
    {Pm4::IT_SET_BASE, 3, true},
    {Pm4::IT_CLEAR_STATE, 1, true},
    {Pm4::IT_INDEX_BUFFER_SIZE, 1, true},
    {Pm4::IT_DISPATCH_DIRECT, 4, true},
    {Pm4::IT_DISPATCH_INDIRECT, 3, true},
    {Pm4::IT_DRAW_INDIRECT, 4, true},
    {Pm4::IT_DRAW_INDEX_INDIRECT, 4, true},
    {Pm4::IT_INDEX_BASE, 2, true},
    {Pm4::IT_DRAW_INDEX_2, 5, true},
    {Pm4::IT_INDEX_TYPE, 1, true},
    {Pm4::IT_DRAW_INDEX_AUTO, 2, true},
    {Pm4::IT_NUM_INSTANCES, 1, true},
    {Pm4::IT_WRITE_DATA, 4, false},
    {Pm4::IT_PFP_SYNC_ME, 1, true},
    {Pm4::IT_EVENT_WRITE, 1, true},
    {Pm4::IT_RELEASE_MEM, 7, true},
    {Pm4::IT_DMA_DATA, 6, true},
    {Pm4::IT_ACQUIRE_MEM, 6, true},
    {Pm4::IT_GET_LOD_STATS, 4, true},
}};

uint8_t* g_arena = nullptr;

uint64_t Address(uint32_t value, uint64_t base, uint64_t span) {
	return reinterpret_cast<uint64_t>(g_arena) + base + (static_cast<uint64_t>(value) * 4u) % span;
}

void SetAddress(uint32_t* body, size_t at, uint32_t value, uint64_t base, uint64_t span) {
	const uint64_t address = Address(value, base, span);
	body[at]     = static_cast<uint32_t>(address);
	body[at + 1] = static_cast<uint32_t>(address >> 32);
}

void FixAddresses(uint32_t code, uint32_t* body, uint32_t length, uint32_t seed) {
	switch (code) {
		case Pm4::IT_WRITE_DATA: SetAddress(body, 1, seed, DataOffset, DataSpan); break;
		case Pm4::IT_COPY_DATA:
			SetAddress(body, 1, seed, DataOffset, DataSpan);
			SetAddress(body, 3, seed * 7u + 1u, DataOffset, DataSpan);
			break;
		case Pm4::IT_DMA_DATA:
			SetAddress(body, 1, seed, DataOffset, DataSpan);
			SetAddress(body, 3, seed * 7u + 1u, DataOffset, DataSpan);
			body[5] &= 0xfffu;
			break;
		case Pm4::IT_RELEASE_MEM: SetAddress(body, 2, seed, DataOffset, DataSpan); break;
		case Pm4::IT_COND_EXEC: SetAddress(body, 0, seed, DataOffset, DataSpan); break;
		case Pm4::IT_SET_BASE:
			body[0] = 1;
			SetAddress(body, 1, seed, DataOffset, DataSpan);
			break;
		case Pm4::IT_INDEX_BASE: SetAddress(body, 0, seed, DataOffset, DataSpan); break;
		case Pm4::IT_INDIRECT_BUFFER:
			SetAddress(body, 0, seed, IndirectOffset, 1u << 20);
			body[2] = (body[2] & ~0xfffffu) | (4u + (seed & 0x3fu));
			break;
		case Pm4::IT_DISPATCH_INDIRECT:
			SetAddress(body, 0, seed, DataOffset, DataSpan);
			body[2] = 1;
			break;
		case Pm4::IT_DISPATCH_DIRECT:
			body[0] &= 3u;
			body[1] &= 3u;
			body[2] &= 3u;
			body[3] = 1;
			break;
		case Pm4::IT_DRAW_INDIRECT:
		case Pm4::IT_DRAW_INDEX_INDIRECT:
			body[0] &= 0x3fffu;
			body[3] = code == Pm4::IT_DRAW_INDEX_INDIRECT ? 0u : 2u;
			break;
		case Pm4::IT_CLEAR_STATE: body[0] &= 0xfu; break;
		case Pm4::IT_DRAW_INDEX_AUTO: body[1] &= 0x22u; break;
		case Pm4::IT_DRAW_INDEX_2:
			SetAddress(body, 1, seed, DataOffset, DataSpan);
			body[3] = body[3] % (body[0] + 1u);
			body[4] &= 0x20u;
			break;
		case Pm4::IT_SET_SH_REG_INDIRECT:
		case Pm4::IT_SET_CONTEXT_REG_INDIRECT: SetAddress(body, 1, seed, DataOffset, DataSpan); break;
		default: break;
	}
	static_cast<void>(length);
}

size_t Build(const uint8_t* data, size_t size, uint32_t* out, size_t capacity) {
	size_t used = 0;
	size_t at   = 0;
	while (at + 3 <= size && used + 24 < capacity) {
		const Op&      op       = Ops[data[at] % Ops.size()];
		const uint32_t length   = op.exact ? op.minimum : std::max<uint32_t>(op.minimum, 1u + (data[at + 1] % 12u));
		const uint32_t seed     = data[at + 2] | (static_cast<uint32_t>(data[at + 1]) << 8);
		at += 3;
		uint32_t* body = out + used + 1;
		for (uint32_t k = 0; k < length; ++k) {
			uint32_t word = 0;
			for (uint32_t b = 0; b < 4 && at < size; ++b) word |= static_cast<uint32_t>(data[at++]) << (8 * b);
			body[k] = word;
		}
		FixAddresses(op.code, body, length, seed);
		out[used] = 0xc0000000u | ((length - 1u) << 16) | (op.code << 8);
		used += 1u + length;
	}
	return used;
}

void FillIndirect(const uint8_t* data, size_t size) {
	auto* indirect = reinterpret_cast<uint32_t*>(g_arena + IndirectOffset);
	const size_t words = Build(data, size, indirect, 1u << 16);
	if (words + 1 < (1u << 16)) indirect[words] = 0xc0001000u;
}

}

extern "C" int LLVMFuzzerInitialize(int*, char***) {
	setenv("ANYPS5_NO_SHADER_CACHE", "1", 1);
	void* mapped = mmap(nullptr, ArenaBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (mapped == MAP_FAILED) {
		std::fputs("kyty_pm4_fuzzer: cannot map the arena\n", stderr);
		std::exit(2);
	}
	g_arena = static_cast<uint8_t*>(mapped);
	AgcDriverTrackSerials_nid_postfix();
	return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
	if (size < 3) return 0;
	std::memset(g_arena, 0, ArenaBytes);
	const size_t half = size / 2;
	FillIndirect(data + half, size - half);
	auto*        stream = reinterpret_cast<uint32_t*>(g_arena);
	const size_t words  = Build(data, half, stream, StreamDwords);
	if (words == 0) return 0;
	Packet packet {stream, static_cast<uint32_t>(words), 0, {}};
	sceAgcDriverSubmitDcb(&packet);
	AgcDriverWaitIdle_nid_postfix();
	return 0;
}
