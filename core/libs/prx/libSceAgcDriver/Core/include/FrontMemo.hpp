#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_FRONTMEMO_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_FRONTMEMO_HPP

#include "prx/libSceAgcDriver/Core/include/BlockRing.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace AgcDriver::Core {

struct MemoKey { std::uint64_t address = 0; std::uint32_t bytes = 0; std::uint32_t first = 0; bool operator==(const MemoKey&) const = default; };

class FrontMemo {
public:
    explicit FrontMemo(std::size_t slots = 1024);
    ~FrontMemo();
    FrontMemo(const FrontMemo&) = delete;
    FrontMemo& operator=(const FrontMemo&) = delete;
    [[nodiscard]] bool Find(const MemoKey& key, std::uint64_t epoch, const BlockRing& ring, RingSlice& copy) const;
    void Keep(const MemoKey& key, std::uint64_t epoch, const RingSlice& copy);
    void Clear();
    struct Counts { std::uint64_t finds = 0, hits = 0, stale = 0, dead = 0; };
    [[nodiscard]] Counts Counted() const;

private:
    struct State;
    std::unique_ptr<State> state;
};

}

#endif
