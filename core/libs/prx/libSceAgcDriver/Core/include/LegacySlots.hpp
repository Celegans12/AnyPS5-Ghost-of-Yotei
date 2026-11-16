#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_LEGACYSLOTS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_LEGACYSLOTS_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

namespace AgcDriver::Core {

class LegacySlots {
public:
    using Work = std::function<void()>;
    static constexpr std::uint32_t Full = 0xffffffffu;
    explicit LegacySlots(std::size_t slots = 1024);
    ~LegacySlots();
    LegacySlots(const LegacySlots&) = delete;
    LegacySlots& operator=(const LegacySlots&) = delete;
    [[nodiscard]] std::uint32_t Hold(Work work);
    void Run(std::uint32_t slot);
    void Skip(std::uint32_t slot);
    std::size_t Reclaim();
    struct Counts { std::uint64_t held = 0, run = 0, skipped = 0, reclaimed = 0, full = 0; };
    [[nodiscard]] Counts Counted() const;

private:
    struct State;
    std::unique_ptr<State> state;
};

}

#endif
