#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_OPHOST_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_OPHOST_HPP

#include "prx/libSceAgcDriver/Core/include/Op.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <utility>

namespace AgcDriver::Core {

class OpHost {
public:
    using Range = std::pair<std::uint64_t, std::uint64_t>;
    static constexpr std::uint64_t Every = std::numeric_limits<std::uint64_t>::max();
    struct Settings { std::size_t depth = 256; bool verifies = false; bool counts = false; bool turns = false; };
    struct Gate { std::uint64_t number = 0; std::uint64_t ops = 0; };
    struct Counts { std::uint64_t posted = 0, retired = 0, gates = 0, passed = 0, roomWaits = 0, wakes = 0, wide = 0, queries = 0, scanned = 0; };

    explicit OpHost(Settings settings);
    ~OpHost();
    OpHost(const OpHost&) = delete;
    OpHost& operator=(const OpHost&) = delete;

    [[nodiscard]] bool Post(const Op& op, std::span<const Range> writes, std::span<const Range> reads, std::uint64_t collectEpoch);
    [[nodiscard]] bool Producing() const;
    [[nodiscard]] Gate Queue();

    [[nodiscard]] std::uint64_t Posted() const;
    [[nodiscard]] std::uint64_t Retired() const;
    [[nodiscard]] std::size_t Pending(std::uint64_t before = Every) const;
    [[nodiscard]] bool Overlaps(std::span<const Range> ranges, std::uint64_t before = Every) const;
    [[nodiscard]] bool OverlapsSince(std::uint64_t first, std::span<const Range> ranges, std::uint64_t before = Every) const;
    [[nodiscard]] bool Reads(std::uint64_t address, std::uint64_t end) const;
    void AwaitRetired(std::uint64_t position);

    [[nodiscard]] bool Runnable() const;
    [[nodiscard]] const Op* Front(std::uint64_t& collectEpoch);
    void ReleaseReads();
    void Retire();
    void Check(Gate gate) const;
    void Pass(Gate gate);
    void Sleep(bool sleeping);
    [[nodiscard]] bool Sleeps() const;

    void Open();
    void Close();
    [[nodiscard]] bool Closed() const;
    [[nodiscard]] Counts Counted() const;

private:
    struct State;
    std::unique_ptr<State> state;
};

}

#endif
