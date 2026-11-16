#include "prx/libSceAgcDriver/Core/include/LabelBatch.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace AgcDriver::Core {

namespace {

constexpr std::uint64_t BlockBytes = std::uint64_t{1} << 16u;
constexpr std::uint64_t BlockBase = ~(BlockBytes - 1u);

std::uint64_t lastByte(std::uint64_t address, std::uint64_t bytes) {
    const std::uint64_t extent = bytes == 0 ? 0 : bytes - 1u;
    const std::uint64_t room = std::numeric_limits<std::uint64_t>::max() - address;
    return extent > room ? std::numeric_limits<std::uint64_t>::max() : address + extent;
}

}

bool LabelBatch::First(std::uint64_t address, std::uint64_t bytes) {
    ++counts.labels;
    const std::uint64_t last = lastByte(address, bytes) & BlockBase;
    bool fresh = false;
    for (std::uint64_t block = address & BlockBase;; block += BlockBytes) {
        const std::uint64_t* begin = blocks.data();
        const std::uint64_t* end = begin + used;
        if (std::find(begin, end, block) == end) {
            if (used == blocks.size()) {
                ++counts.overflowed;
                return true;
            }
            blocks[used] = block;
            ++used;
            fresh = true;
        }
        if (block == last) break;
    }
    if (fresh) {
        ++counts.first;
    } else {
        ++counts.repeated;
    }
    return fresh;
}

std::span<const std::uint64_t> LabelBatch::Blocks() const {
    return std::span<const std::uint64_t>(blocks.data(), used);
}

void LabelBatch::Close() {
    used = 0;
}

LabelBatch::Counts LabelBatch::Counted() const {
    return counts;
}

}
