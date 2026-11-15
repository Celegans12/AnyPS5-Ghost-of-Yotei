#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_WRITEWATCHCOVERAGE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_WRITEWATCHCOVERAGE_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <limits>
#include <unordered_map>

namespace AgcDriver::GuestMemory {

class WriteWatchCoverage {
public:
    void Initialize(std::uint64_t address, std::size_t bytes, bool summarize = true) {
        base = address;
        size = bytes;
        excluded = std::vector<std::atomic<std::uint16_t>>(bytes / BlockBytes + (bytes % BlockBytes != 0));
        excludedGenerations.clear();
        summary = std::vector<std::atomic<std::uint64_t>>(summarize ? (excluded.size() + BlocksPerWord - 1) / BlocksPerWord : 0);
    }

    bool Summarized() const { return !summary.empty(); }

    bool Covers(std::uint64_t address, std::size_t bytes) const {
        if (bytes == 0 || bytes > size || address < base || address - base > size - bytes) return false;
        const auto first = (address - base) / BlockBytes;
        const auto last = (address - base + bytes - 1) / BlockBytes;
        const auto firstPage = (address - base) % BlockBytes / PageBytes;
        const auto lastPage = (address - base + bytes - 1) % BlockBytes / PageBytes;
        if (first == last) return (excludedPages(first) & pageMask(firstPage, lastPage + 1)) == 0;
        if ((excludedPages(first) & pageMask(firstPage, PagesPerBlock)) != 0 || (excludedPages(last) & pageMask(0, lastPage + 1)) != 0) return false;
        if (!summary.empty()) return noneExcluded(first + 1, last);
        return std::none_of(excluded.begin() + first + 1, excluded.begin() + last, [](const auto& mask) { return mask.load(std::memory_order_relaxed) != 0; });
    }

    bool ExcludedSpan(std::uint64_t address, std::size_t bytes, std::uint64_t* begin, std::uint64_t* end) const {
        std::size_t first = 0;
        std::size_t last = 0;
        if (!excludedBlocks(address, bytes, first, last)) return false;
        *begin = base + first * BlockBytes;
        *end = std::min<std::uint64_t>(base + (last + 1) * BlockBytes, base + size);
        return true;
    }
    bool Exclude(std::uint64_t address, std::size_t bytes, std::uint64_t generation = 1) {
        std::size_t first = 0;
        std::size_t last = 0;
        if (!excludedBlocks(address, bytes, first, last)) return false;
        const auto offset = std::max(address, base) - base;
        const auto firstPage = offset / PageBytes;
        const auto endPage = (std::min<std::uint64_t>(address + bytes - base, size) + PageBytes - 1) / PageBytes;
        bool changed = false;
        for (auto block = first; block <= last; ++block) {
            const auto from = block == first ? firstPage % PagesPerBlock : 0;
            const auto to = block == last ? (endPage - 1) % PagesPerBlock + 1 : PagesPerBlock;
            const auto mask = pageMask(from, to);
            const auto held = excludedPages(block);
            changed = changed || (held & mask) != mask;
            auto& stamps = excludedGenerations[block];
            for (auto page = from; page < to; ++page) stamps[page] = std::max(stamps[page], std::max<std::uint64_t>(generation, 1));
            excluded[block].store(static_cast<std::uint16_t>(held | mask), std::memory_order_relaxed);
            updateSummary(block);
        }
        return changed;
    }

    bool Restore(std::uint64_t address, std::size_t bytes, std::uint64_t generation = std::numeric_limits<std::uint64_t>::max()) {
        if (bytes == 0 || size == 0) return false;
        if (address < base) {
            if (base - address >= bytes) return false;
            bytes -= base - address;
            address = base;
        }
        const auto offset = address - base;
        if (offset >= size) return false;
        bytes = static_cast<std::size_t>(std::min<std::uint64_t>(bytes, size - offset));
        const auto firstPage = offset / PageBytes + (offset % PageBytes != 0);
        const auto stop = offset + bytes;
        const auto endPage = stop / PageBytes + (stop == size && stop % PageBytes != 0);
        if (firstPage >= endPage) return false;
        const auto first = firstPage / PagesPerBlock;
        const auto last = (endPage - 1) / PagesPerBlock;
        bool changed = false;
        for (auto block = first; block <= last; ++block) {
            const auto from = block == first ? firstPage % PagesPerBlock : 0;
            const auto to = block == last ? (endPage - 1) % PagesPerBlock + 1 : PagesPerBlock;
            auto held = excludedPages(block);
            if ((held & pageMask(from, to)) == 0) continue;
            const auto& stamps = excludedGenerations.at(block);
            for (auto page = from; page < to; ++page) {
                const auto bit = static_cast<std::uint16_t>(1u << page);
                if ((held & bit) == 0 || stamps[page] >= generation) continue;
                held &= static_cast<std::uint16_t>(~bit);
                changed = true;
            }
            excluded[block].store(held, std::memory_order_relaxed);
            if (held == 0) excludedGenerations.erase(block);
            updateSummary(block);
        }
        return changed;
    }

private:
    static std::uint16_t pageMask(std::size_t first, std::size_t end) {
        return static_cast<std::uint16_t>(((1u << end) - 1u) & ~((1u << first) - 1u));
    }
    std::uint16_t excludedPages(std::size_t block) const {
        return excluded[block].load(std::memory_order_relaxed);
    }
    void updateSummary(std::size_t block) {
        if (summary.empty()) return;
        const auto bit = std::uint64_t{1} << (block % BlocksPerWord);
        auto& word = summary[block / BlocksPerWord];
        const auto held = word.load(std::memory_order_relaxed);
        word.store(excludedPages(block) != 0 ? held | bit : held & ~bit, std::memory_order_relaxed);
    }
    bool noneExcluded(std::size_t from, std::size_t to) const {
        if (from >= to) return true;
        const auto firstWord = from / BlocksPerWord;
        const auto lastWord = (to - 1) / BlocksPerWord;
        const auto lowMask = ~std::uint64_t{0} << (from % BlocksPerWord);
        const auto highMask = ~std::uint64_t{0} >> (BlocksPerWord - 1 - (to - 1) % BlocksPerWord);
        const auto word = [&](std::size_t index) { return summary[index].load(std::memory_order_relaxed); };
        if (firstWord == lastWord) return (word(firstWord) & lowMask & highMask) == 0;
        if ((word(firstWord) & lowMask) != 0 || (word(lastWord) & highMask) != 0) return false;
        return std::none_of(summary.begin() + firstWord + 1, summary.begin() + lastWord, [](const auto& held) { return held.load(std::memory_order_relaxed) != 0; });
    }
    bool excludedBlocks(std::uint64_t address, std::size_t bytes, std::size_t& first, std::size_t& last) const {
        if (bytes == 0 || size == 0) return false;
        if (address < base) {
            if (base - address >= bytes) return false;
            bytes -= base - address;
            address = base;
        }
        const auto offset = address - base;
        if (offset >= size) return false;
        bytes = static_cast<std::size_t>(std::min<std::uint64_t>(bytes, size - offset));
        first = offset / BlockBytes;
        last = (offset + bytes - 1) / BlockBytes;
        return true;
    }
    static constexpr std::size_t BlockBytes = 65536;
    static constexpr std::size_t PageBytes = 4096;
    static constexpr std::size_t PagesPerBlock = BlockBytes / PageBytes;
    static constexpr std::size_t BlocksPerWord = 64;
    std::uint64_t base = 0;
    std::size_t size = 0;
    std::vector<std::atomic<std::uint16_t>> excluded;
    std::vector<std::atomic<std::uint64_t>> summary;
    std::unordered_map<std::uint64_t, std::array<std::uint64_t, PagesPerBlock>> excludedGenerations;
};

}

#endif
