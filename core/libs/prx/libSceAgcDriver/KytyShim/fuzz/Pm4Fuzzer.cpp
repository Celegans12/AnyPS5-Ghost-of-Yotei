#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"

#include "SceTypes.hpp"
#include "graphics/guest_gpu/pm4.h"

#include <sys/mman.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

namespace Pm4 = Libs::Graphics::Pm4;

constexpr size_t   ArenaBytes   = 16u << 20;
constexpr size_t   StreamDwords = 1u << 14;
constexpr uint64_t DataOffset   = 1u << 20;
constexpr uint64_t DataSpan     = 4u << 20;
constexpr uint64_t IndirectOffset = 8u << 20;

struct Op {
	uint32_t code;
	uint32_t minimum;
};

constexpr std::array<Op, 34> Ops = {{
    {Pm4::IT_NOP, 1},
    {Pm4::IT_SET_BASE, 3},
    {Pm4::IT_CLEAR_STATE, 1},
    {Pm4::IT_INDEX_BUFFER_SIZE, 1},
    {Pm4::IT_DISPATCH_DIRECT, 3},
    {Pm4::IT_DISPATCH_INDIRECT, 2},
    {Pm4::IT_SET_PREDICATION, 4},
    {Pm4::IT_COND_EXEC, 4},
    {Pm4::IT_DRAW_INDIRECT, 3},
    {Pm4::IT_INDEX_BASE, 2},
    {Pm4::IT_DRAW_INDEX_2, 4},
    {Pm4::IT_CONTEXT_CONTROL, 2},
    {Pm4::IT_INDEX_TYPE, 1},
    {Pm4::IT_DRAW_INDIRECT_MULTI, 6},
    {Pm4::IT_DRAW_INDEX_AUTO, 2},
    {Pm4::IT_NUM_INSTANCES, 1},
    {Pm4::IT_DRAW_INDEX_OFFSET_2, 4},
    {Pm4::IT_WRITE_DATA, 4},
    {Pm4::IT_DRAW_INDEX_MULTI_INST, 3},
    {Pm4::IT_INDIRECT_BUFFER, 3},
    {Pm4::IT_COPY_DATA, 5},
    {Pm4::IT_PFP_SYNC_ME, 1},
    {Pm4::IT_EVENT_WRITE, 1},
    {Pm4::IT_RELEASE_MEM, 7},
    {Pm4::IT_DMA_DATA, 6},
    {Pm4::IT_ACQUIRE_MEM, 6},
    {Pm4::IT_SET_CONFIG_REG, 2},
    {Pm4::IT_SET_CONTEXT_REG, 2},
    {Pm4::IT_SET_SH_REG, 2},
    {Pm4::IT_SET_UCONFIG_REG, 2},
    {Pm4::IT_SET_UCONFIG_REG_INDEX, 2},
    {Pm4::IT_SET_SH_REG_INDIRECT, 3},
    {Pm4::IT_SET_CONTEXT_REG_INDIRECT, 3},
    {Pm4::IT_GET_LOD_STATS, 4},
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
		case Pm4::IT_INDEX_BASE: SetAddress(body, code == Pm4::IT_SET_BASE ? 1 : 0, seed, DataOffset, DataSpan); break;
		case Pm4::IT_INDIRECT_BUFFER:
			SetAddress(body, 0, seed, IndirectOffset, 1u << 20);
			body[2] = (body[2] & ~0xfffffu) | (4u + (seed & 0x3fu));
			break;
		case Pm4::IT_DISPATCH_INDIRECT:
		case Pm4::IT_DRAW_INDIRECT:
		case Pm4::IT_DRAW_INDIRECT_MULTI: body[0] &= 0x3fffu; break;
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
		const uint32_t length   = std::max<uint32_t>(op.minimum, 1u + (data[at + 1] % 12u));
		const uint32_t seed     = data[at + 2] | (static_cast<uint32_t>(data[at + 1]) << 8);
		at += 3;
		uint32_t* body = out + used + 1;
		for (uint32_t k = 0; k < length; ++k) {
			uint32_t word = 0;
			for (uint32_t b = 0; b < 4 && at < size; ++b) word |= static_cast<uint32_t>(data[at++]) << (8 * b);
			body[k] = word;
		}
		FixAddresses(op.code, body, length, seed);
		out[used] = 0xc0000000u | ((length - 1u) << 16) | (op.code << 8) | ((seed & 1u) << 1);
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
