#include "graphics/host_gpu/commandBlocks.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>

namespace {

using Libs::Graphics::CommandBlocks;

constexpr uint64_t Block     = CommandBlocks::BlockSize;
constexpr uint64_t LeafSpan  = uint64_t {4096} * 64 * Block;
constexpr uint64_t PoolStart = 0x240000000;

void Require(bool value, const char* what) {
	if (value) return;
	std::fprintf(stderr, "kyty_shim_command_blocks: FAILED: %s\n", what);
	std::exit(1);
}

uint64_t BlockOf(uint64_t address) { return address / Block * Block; }

void FreshTable() {
	auto blocks = std::make_unique<CommandBlocks>();
	Require(!blocks->Contains(PoolStart), "a fresh table holds no command block");
	blocks->Forget(PoolStart, 64 * Block);
	Require(!blocks->Contains(PoolStart), "forgetting an unmarked range marks nothing");
	Require(!blocks->Contains(CommandBlocks::AddressSize), "an address past the tracked space is never a command block");
	blocks->Mark(CommandBlocks::AddressSize, Block);
	Require(!blocks->Contains(CommandBlocks::AddressSize), "a buffer past the tracked space marks nothing");
	blocks->Mark(PoolStart, 0);
	Require(!blocks->Contains(PoolStart), "an empty buffer marks nothing");
}

void MarkReadForget() {
	auto           blocks = std::make_unique<CommandBlocks>();
	const uint64_t buffer = PoolStart + 3 * Block + 0x123;
	const uint64_t first  = BlockOf(buffer);
	blocks->Mark(buffer, 2 * Block);
	Require(!blocks->Contains(first - 1), "the block before the buffer stays unmarked");
	Require(blocks->Contains(buffer), "the block holding the first command is marked");
	Require(blocks->Contains(first), "the whole block of the first command is marked");
	Require(blocks->Contains(first + Block), "a block inside the buffer is marked");
	Require(blocks->Contains(buffer + 2 * Block - 1), "the block holding the last command is marked");
	Require(!blocks->Contains(first + 3 * Block), "the block after the last command stays unmarked");

	blocks->Forget(first + Block + 0x10, 0x20);
	Require(blocks->Contains(first), "an unmapping leaves the blocks it does not touch");
	Require(!blocks->Contains(first + Block), "a block an unmapping touches is forgotten");
	Require(blocks->Contains(first + 2 * Block), "an unmapping leaves the next block");

	blocks->Forget(first, 3 * Block);
	for (uint64_t at = first; at < first + 3 * Block; at += Block / 4) {
		Require(!blocks->Contains(at), "an unmapped buffer leaves no mark");
	}

	blocks->Mark(buffer, 4);
	Require(blocks->Contains(buffer), "a buffer the processor reads again after an unmapping is marked again");
	Require(!blocks->Contains(first + Block), "marking one command marks only its block");
}

void WordAndLeafBoundaries() {
	auto           blocks = std::make_unique<CommandBlocks>();
	const uint64_t start  = 5 * LeafSpan - 70 * Block;
	blocks->Mark(start, 140 * Block);
	Require(!blocks->Contains(start - 1), "the block before a buffer across two leaves stays unmarked");
	Require(!blocks->Contains(start + 140 * Block), "the block after a buffer across two leaves stays unmarked");
	for (uint64_t index = 0; index < 140; index++) {
		Require(blocks->Contains(start + index * Block), "every block of a buffer across words and leaves is marked");
	}
	blocks->Forget(start + 60 * Block, 20 * Block);
	for (uint64_t index = 0; index < 140; index++) {
		const bool kept = index < 60 || index >= 80;
		Require(blocks->Contains(start + index * Block) == kept, "an unmapping across a leaf clears exactly its blocks");
	}
	blocks->Forget(0, CommandBlocks::AddressSize);
	for (uint64_t index = 0; index < 140; index++) {
		Require(!blocks->Contains(start + index * Block), "unmapping everything clears every mark");
	}
}

void TopOfSpace() {
	auto           blocks = std::make_unique<CommandBlocks>();
	const uint64_t last   = CommandBlocks::AddressSize - Block;
	blocks->Mark(last + 0x40, UINT64_MAX - last);
	Require(blocks->Contains(last), "a buffer reaching past the tracked space marks its last block");
	Require(!blocks->Contains(last - 1), "and nothing before it");
	blocks->Forget(last, UINT64_MAX - last);
	Require(!blocks->Contains(last), "an unmapping reaching past the tracked space clears the last block");
}

void ReadWhileMarking() {
	auto           blocks = std::make_unique<CommandBlocks>();
	const uint64_t steady = PoolStart + 7 * Block;
	blocks->Mark(steady, Block);
	std::atomic<bool>     stop {false};
	std::atomic<uint64_t> misses {0};
	std::thread           reader([&] {
		while (!stop.load(std::memory_order_acquire)) {
			if (!blocks->Contains(steady)) misses.fetch_add(1, std::memory_order_relaxed);
		}
	});
	for (uint64_t round = 0; round < 20000; round++) {
		const uint64_t other = 3 * LeafSpan + (round % 512) * Block;
		blocks->Mark(other, Block);
		blocks->Mark(steady - Block, Block);
		blocks->Forget(steady - Block, Block);
		blocks->Forget(other, Block);
	}
	stop.store(true, std::memory_order_release);
	reader.join();
	Require(misses.load() == 0, "a mark stays visible while its neighbours are marked and forgotten");
	Require(blocks->Contains(steady), "the steady mark survives");
	Require(!blocks->Contains(steady - Block), "the neighbour mark is gone");
}

}

int main() {
	FreshTable();
	MarkReadForget();
	WordAndLeafBoundaries();
	TopOfSpace();
	ReadWhileMarking();
	std::printf("kyty_shim_command_blocks: ok\n");
	return 0;
}
