#include "ShaderRecompilerRun.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

extern "C" size_t LLVMFuzzerMutate(uint8_t* data, size_t size, size_t max_size);

namespace {

using KytyFuzz::Run;

constexpr uint32_t SEndPgm = 0xbf810000u;
constexpr size_t   Header  = 4;
constexpr size_t   MaxWords = 8192;

constexpr std::array<uint32_t, 24> KnownWords = {
    0xbf810000u, 0xbf800000u, 0xbe800381u, 0x7e000301u, 0x7e020300u, 0x06000101u, 0x06020300u, 0x50000100u,
    0xd5190000u, 0xd7010000u, 0xbf8c0000u, 0xbf8c007fu, 0x7e000280u, 0x7e000281u, 0xbe820300u, 0x80000100u,
    0x81000001u, 0xbe80037eu, 0x7c000101u, 0xbf860001u, 0xbf850001u, 0xbefe0480u, 0xc0020000u, 0xe0301000u};

}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
	if (size < Header + 4) return 0;
	const size_t words = std::min((size - Header) / 4, MaxWords);
	std::vector<uint32_t> code(words);
	std::memcpy(code.data(), data + Header, words * 4);
	Run(code, data[0], data[1]);
	return 0;
}

extern "C" size_t LLVMFuzzerCustomMutator(uint8_t* data, size_t size, size_t max_size, unsigned int seed) {
	std::mt19937 rng(seed);
	if (max_size < Header + 8) return size;
	if (size < Header + 8) {
		size = Header + 8;
		std::memset(data, 0, size);
		data[0] = static_cast<uint8_t>(rng());
		data[1] = static_cast<uint8_t>(rng());
		const uint32_t first = KnownWords[rng() % KnownWords.size()];
		std::memcpy(data + Header, &first, 4);
		std::memcpy(data + Header + 4, &SEndPgm, 4);
	}
	size                = Header + (size - Header) / 4 * 4;
	const size_t words  = (size - Header) / 4;
	const size_t rounds = 1 + rng() % 3;
	for (size_t round = 0; round < rounds; ++round) {
		const size_t at = Header + 4 * (rng() % words);
		uint32_t     word = 0;
		std::memcpy(&word, data + at, 4);
		switch (rng() % 7) {
			case 0: word ^= 1u << (rng() % 32); break;
			case 1: word = KnownWords[rng() % KnownWords.size()]; break;
			case 2: word = static_cast<uint32_t>(rng()); break;
			case 3: word = (word & 0xffff0000u) | (rng() & 0xffffu); break;
			case 4: word = (word & 0x0000ffffu) | (rng() << 16); break;
			case 5:
				if (size + 4 <= max_size) {
					std::memmove(data + at + 4, data + at, size - at);
					word = KnownWords[rng() % KnownWords.size()];
					size += 4;
				}
				break;
			default: {
				const size_t other = Header + 4 * (rng() % words);
				uint32_t     swap  = 0;
				std::memcpy(&swap, data + other, 4);
				std::memcpy(data + other, &word, 4);
				word = swap;
				break;
			}
		}
		std::memcpy(data + at, &word, 4);
	}
	if (rng() % 16 == 0) data[rng() % 2] = static_cast<uint8_t>(rng());
	if (rng() % 8 == 0) size = LLVMFuzzerMutate(data, size, max_size);
	uint32_t last = 0;
	if (size >= Header + 4) std::memcpy(&last, data + size - 4, 4);
	if (last != SEndPgm && size + 4 <= max_size) {
		std::memcpy(data + size, &SEndPgm, 4);
		size += 4;
	}
	return size;
}
