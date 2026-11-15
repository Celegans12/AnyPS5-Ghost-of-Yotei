#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PASSSTATE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PASSSTATE_HPP

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace AgcDriver::Graphics {

class PassState {
public:
    enum class Mode : std::uint8_t { Environment, Off, On, Verify };
    enum class Served : std::uint8_t { Access, Targets, Depth, Kept, Count };
    enum class Refusal : std::uint8_t { NoSummary, Unaccepted, PassBroken, Batch, Gate, Stamp, Targets, Evicted, Depth, DepthCalls, Count };
    static constexpr std::size_t ServedKinds = static_cast<std::size_t>(Served::Count);
    static constexpr std::size_t Refusals = static_cast<std::size_t>(Refusal::Count);
    static constexpr std::array<const char*, ServedKinds> ServedNames{"pass accesses", "targets", "depth view", "kept objects"};
    static constexpr std::array<const char*, Refusals> RefusalNames{"no read-only summary", "summary new to the pass", "pass broken", "another batch", "run ended", "proof stamp moved", "other targets", "target evicted", "other depth target", "depth view taken since"};

    struct Counts {
        std::uint64_t draws = 0;
        std::uint64_t runs = 0;
        std::uint64_t verified = 0;
        std::uint64_t mismatches = 0;
        std::array<std::uint64_t, ServedKinds> served{};
        std::array<std::uint64_t, Refusals> refused{};
    };

    static Mode Current() {
        const auto forced = tested.load(std::memory_order_relaxed);
        if (forced != Mode::Environment) return forced;
        static const Mode configured = std::getenv("APS5_VERIFY_PASS_STATE") != nullptr ? Mode::Verify : std::getenv("APS5_PASS_STATE") != nullptr ? Mode::On : Mode::Off;
        return configured;
    }
    static bool Enabled() { return Current() != Mode::Off; }
    static bool Verifies() { return Current() == Mode::Verify; }
    static void SetForTest(Mode mode) { tested.store(mode, std::memory_order_relaxed); }

    static void Break() { gate.fetch_add(1, std::memory_order_acq_rel); }
    static std::uint64_t Gate() { return gate.load(std::memory_order_acquire); }
    static void NoteDepthView() { depthViews.fetch_add(1, std::memory_order_acq_rel); }
    static std::uint64_t DepthViews() { return depthViews.load(std::memory_order_acquire); }
    static std::uint64_t NextId() { return ids.fetch_add(1, std::memory_order_relaxed) + 1; }

    static void CountDraw() { draws.fetch_add(1, std::memory_order_relaxed); }
    static void CountRun() { runs.fetch_add(1, std::memory_order_relaxed); }
    static void CountVerified() { verified.fetch_add(1, std::memory_order_relaxed); }
    static void CountMismatch() { mismatches.fetch_add(1, std::memory_order_relaxed); }
    static void Count(Served kind) { served[static_cast<std::size_t>(kind)].fetch_add(1, std::memory_order_relaxed); }
    static void Count(Refusal reason) { refused[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed); }
    static Counts Counted() {
        Counts counts;
        counts.draws = draws.load(std::memory_order_relaxed);
        counts.runs = runs.load(std::memory_order_relaxed);
        counts.verified = verified.load(std::memory_order_relaxed);
        counts.mismatches = mismatches.load(std::memory_order_relaxed);
        for (std::size_t index = 0; index < ServedKinds; ++index) counts.served[index] = served[index].load(std::memory_order_relaxed);
        for (std::size_t index = 0; index < Refusals; ++index) counts.refused[index] = refused[index].load(std::memory_order_relaxed);
        return counts;
    }

private:
    static inline std::atomic<Mode> tested{Mode::Environment};
    static inline std::atomic<std::uint64_t> gate{0};
    static inline std::atomic<std::uint64_t> depthViews{0};
    static inline std::atomic<std::uint64_t> ids{0};
    static inline std::atomic<std::uint64_t> draws{0};
    static inline std::atomic<std::uint64_t> runs{0};
    static inline std::atomic<std::uint64_t> verified{0};
    static inline std::atomic<std::uint64_t> mismatches{0};
    static inline std::array<std::atomic<std::uint64_t>, ServedKinds> served{};
    static inline std::array<std::atomic<std::uint64_t>, Refusals> refused{};
};

}

#endif
