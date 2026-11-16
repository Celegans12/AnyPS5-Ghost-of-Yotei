#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_WRITEBLOCKS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_WRITEBLOCKS_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>

namespace AgcDriver::Core {

class WriteBlocks {
public:
    using Range = std::pair<std::uint64_t, std::uint64_t>;
    static constexpr std::uint32_t BlockShift = 16;
    static constexpr std::uint64_t BlockBytes = std::uint64_t{1} << BlockShift;
    static constexpr std::uint32_t LeafShift = 32;
    static constexpr std::uint32_t CoveredShift = 47;
    static constexpr std::uint64_t CoveredBytes = std::uint64_t{1} << CoveredShift;
    static constexpr std::uint32_t InlineHolders = 254;

    WriteBlocks();
    ~WriteBlocks();
    WriteBlocks(const WriteBlocks&) = delete;
    WriteBlocks& operator=(const WriteBlocks&) = delete;

    void Enter(std::span<const Range> ranges);
    void Leave(std::span<const Range> ranges);
    [[nodiscard]] bool Clear(std::uint64_t address, std::uint64_t bytes) const noexcept;
    [[nodiscard]] bool Clear(std::span<const Range> ranges) const noexcept;

    [[nodiscard]] std::uint64_t Holders(std::uint64_t address) const;
    [[nodiscard]] std::size_t Ranges() const;
    [[nodiscard]] std::size_t Leaves() const;
    [[nodiscard]] std::size_t Spilled() const;

private:
    struct State;
    std::unique_ptr<State> state;
};

}

#endif
