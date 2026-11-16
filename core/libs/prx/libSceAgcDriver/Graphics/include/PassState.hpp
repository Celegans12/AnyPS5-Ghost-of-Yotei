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

    static Mode Current();
    static bool Enabled();
    static bool Verifies();
    static void SetForTest(Mode mode);

    static void Break();
    static std::uint64_t Gate();
    static void NoteDepthView();
    static std::uint64_t DepthViews();
    static std::uint64_t NextId();

    static void CountDraw();
    static void CountRun();
    static void CountVerified();
    static void CountMismatch();
    static void Count(Served kind);
    static void Count(Refusal reason);
    static Counts Counted();

private:
    static std::atomic<Mode> tested;
    static std::atomic<std::uint64_t> gate;
    static std::atomic<std::uint64_t> depthViews;
    static std::atomic<std::uint64_t> ids;
    static std::atomic<std::uint64_t> draws;
    static std::atomic<std::uint64_t> runs;
    static std::atomic<std::uint64_t> verified;
    static std::atomic<std::uint64_t> mismatches;
    static std::array<std::atomic<std::uint64_t>, ServedKinds> served;
    static std::array<std::atomic<std::uint64_t>, Refusals> refused;
};

}

#endif
