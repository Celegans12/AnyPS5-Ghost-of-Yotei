#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_OPQUEUE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_OPQUEUE_HPP

#include "prx/libSceAgcDriver/Core/include/Op.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace AgcDriver::Core {

class OpQueue {
public:
    struct Settings { std::size_t depth = 256; std::size_t wakeBatch = 32; std::uint32_t spins = 256; bool verifies = false; };
    explicit OpQueue(Settings settings);
    ~OpQueue();
    OpQueue(const OpQueue&) = delete;
    OpQueue& operator=(const OpQueue&) = delete;
    [[nodiscard]] Op* Stage();
    std::uint64_t Publish();
    void Flush();
    void AwaitRoom();
    void AwaitRetired(std::uint64_t position);
    [[nodiscard]] std::span<const Op> Take(std::size_t most);
    [[nodiscard]] std::span<const Op> TryTake(std::size_t most);
    void Retire(std::size_t count);
    void Close();
    [[nodiscard]] bool Closed() const;
    [[nodiscard]] std::uint64_t Published() const;
    [[nodiscard]] std::uint64_t Retired() const;
    struct Counts { std::uint64_t published = 0, taken = 0, parks = 0, batchWakes = 0, flushWakes = 0, roomWaits = 0, spun = 0; };
    [[nodiscard]] Counts Counted() const;

private:
    struct State;
    std::unique_ptr<State> state;
};

}

#endif
