#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_COMPAREWRITEBACK_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_COMPAREWRITEBACK_H_

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace Libs::Graphics {

struct WriteBackCounts {
	uint64_t written = 0;
	uint64_t kept    = 0;
};

template <class Write>
WriteBackCounts CompareWriteBack(uint64_t address, const uint8_t* result, const uint8_t* snapshot,
                                 const uint8_t* current, uint64_t size, Write&& write) {
	constexpr uint64_t Chunk = 64;
	WriteBackCounts    counts;
	uint64_t           run_begin = 0;
	uint64_t           run_size  = 0;
	const auto         flush     = [&] {
		if (run_size != 0) {
			write(address + run_begin, result + run_begin, run_size);
			counts.written += run_size;
			run_size = 0;
		}
	};
	const auto append = [&](uint64_t offset, uint64_t bytes) {
		if (run_size != 0 && run_begin + run_size == offset) {
			run_size += bytes;
			return;
		}
		flush();
		run_begin = offset;
		run_size  = bytes;
	};
	for (uint64_t offset = 0; offset < size; offset += Chunk) {
		const uint64_t bytes    = std::min(Chunk, size - offset);
		const bool     produced = std::memcmp(result + offset, snapshot + offset, bytes) != 0;
		if (std::memcmp(current + offset, snapshot + offset, bytes) == 0) {
			if (produced) {
				append(offset, bytes);
			}
			continue;
		}
		for (uint64_t index = offset; index < offset + bytes; index++) {
			if (current[index] != snapshot[index]) {
				counts.kept += result[index] != current[index] ? 1u : 0u;
			} else if (result[index] != snapshot[index]) {
				append(index, 1);
			}
		}
	}
	flush();
	return counts;
}

}

#endif
