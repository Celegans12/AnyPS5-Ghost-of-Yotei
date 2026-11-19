#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_COMMANDBLOCKS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_COMMANDBLOCKS_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>

namespace Libs::Graphics {

class CommandBlocks final {
public:
	static constexpr uint64_t BlockSize   = 64 * 1024;
	static constexpr uint64_t AddressSize = uint64_t {1} << 44u;

	CommandBlocks() = default;
	~CommandBlocks() {
		for (auto& leaf: m_leaves) {
			delete leaf.load(std::memory_order_relaxed);
		}
	}
	CommandBlocks(const CommandBlocks&)            = delete;
	CommandBlocks& operator=(const CommandBlocks&) = delete;

	void Mark(uint64_t vaddr, uint64_t size) {
		ForEachWord<true>(vaddr, size, [](std::atomic<uint64_t>& word, uint64_t mask) {
			if ((word.load(std::memory_order_relaxed) & mask) != mask) {
				word.fetch_or(mask, std::memory_order_release);
			}
		});
	}

	void Forget(uint64_t vaddr, uint64_t size) {
		ForEachWord<false>(vaddr, size, [](std::atomic<uint64_t>& word, uint64_t mask) {
			if ((word.load(std::memory_order_relaxed) & mask) != 0) {
				word.fetch_and(~mask, std::memory_order_release);
			}
		});
	}

	[[nodiscard]] bool Contains(uint64_t vaddr) const noexcept {
		if (vaddr >= AddressSize) {
			return false;
		}
		const auto  block = vaddr / BlockSize;
		const auto* leaf  = m_leaves[block / LeafBlocks].load(std::memory_order_acquire);
		if (leaf == nullptr) {
			return false;
		}
		const auto index = block % LeafBlocks;
		return (((*leaf)[index / 64].load(std::memory_order_acquire) >> (index % 64)) & 1u) != 0;
	}

private:
	static constexpr uint64_t LeafWords  = 4096;
	static constexpr uint64_t LeafBlocks = LeafWords * 64;
	static constexpr uint64_t LeafCount  = AddressSize / BlockSize / LeafBlocks;
	using Leaf                           = std::array<std::atomic<uint64_t>, LeafWords>;

	template <bool create>
	Leaf* GetLeaf(uint64_t index) {
		auto* leaf = m_leaves[index].load(std::memory_order_acquire);
		if constexpr (create) {
			if (leaf == nullptr) {
				auto  fresh    = std::make_unique<Leaf>();
				Leaf* expected = nullptr;
				if (m_leaves[index].compare_exchange_strong(expected, fresh.get(), std::memory_order_acq_rel,
				                                            std::memory_order_acquire)) {
					return fresh.release();
				}
				return expected;
			}
		}
		return leaf;
	}

	template <bool create, typename Func>
	void ForEachWord(uint64_t vaddr, uint64_t size, Func&& func) {
		if (size == 0 || vaddr >= AddressSize) {
			return;
		}
		const auto end   = size > AddressSize - vaddr ? AddressSize : vaddr + size;
		auto       block = vaddr / BlockSize;
		const auto stop  = (end - 1) / BlockSize + 1;
		while (block < stop) {
			const auto leaf_index = block / LeafBlocks;
			const auto leaf_stop  = std::min(stop, (leaf_index + 1) * LeafBlocks);
			if (auto* leaf = GetLeaf<create>(leaf_index); leaf != nullptr) {
				for (auto at = block; at < leaf_stop;) {
					const auto index = at % LeafBlocks;
					const auto bit   = index % 64;
					const auto count = std::min<uint64_t>(64 - bit, leaf_stop - at);
					const auto mask  = (count == 64 ? ~uint64_t {0} : (uint64_t {1} << count) - 1u) << bit;
					func((*leaf)[index / 64], mask);
					at += count;
				}
			}
			block = leaf_stop;
		}
	}

	std::array<std::atomic<Leaf*>, LeafCount> m_leaves {};
};

}

#endif
