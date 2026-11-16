#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_LABELBATCH_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_LABELBATCH_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace AgcDriver::Core {

class LabelBatch {
public:
    static constexpr std::size_t Blocks64k = 64;
    [[nodiscard]] bool First(std::uint64_t address, std::uint64_t bytes);
    [[nodiscard]] std::span<const std::uint64_t> Blocks() const;
    void Close();
    struct Counts { std::uint64_t labels = 0, first = 0, repeated = 0, overflowed = 0; };
    [[nodiscard]] Counts Counted() const;

private:
    std::array<std::uint64_t, Blocks64k> blocks{};
    std::size_t used = 0;
    Counts counts;
};

}

#endif
