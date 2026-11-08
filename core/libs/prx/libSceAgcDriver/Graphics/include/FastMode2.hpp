#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_FASTMODE2_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_FASTMODE2_HPP

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <chrono>
#include <cstdlib>

#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"

// Fast mode v2 (local, opt-in, inexact; research/speed/fm2/design.md). APS5_FM2 is the master RuntimeSwitch (an APS5_AB
// arm flips it); an item runs only while the master and its own RuntimeSwitch are set:
// - APS5_FM2_TRUST: a cached draw set proved in full once in the current present skips the per-texture proof for the rest
//   of that present (identity, eviction, prefix and tail checks stay). APS5_FM2_TRUST_PLAIN=1 (start-time) limits it to
//   sets without storage images or GPU-written sampled surfaces.
// - APS5_FM2_SNAP: no per-draw snapshots of in-place constant buffers (moved-buffer overrides stay).
// - APS5_FM2_TARGET: the last resident colour target per slot is reused within a present while the storage pending
//   serial is unmoved and the image is still cached.
// With APS5_FM2 unset nothing here changes behaviour.
namespace AgcDriver::Graphics::FastMode2 {

inline bool On(RuntimeSwitch item) {
    return RuntimeSwitches::Set(RuntimeSwitch::Fm2) && RuntimeSwitches::Set(item);
}

inline bool TrustPlainOnly() {
    static const bool plain = std::getenv("APS5_FM2_TRUST_PLAIN") != nullptr;
    return plain;
}

inline std::uint64_t TrustPresents() {
    static const std::uint64_t presents = [] {
        const char* text = std::getenv("APS5_FM2_TRUST_PRESENTS");
        const auto value = text != nullptr ? std::strtoull(text, nullptr, 10) : 1ull;
        return value == 0 ? 1ull : value;
    }();
    return presents;
}

inline bool SnapMovedInPlace() {
    static const bool on = std::getenv("APS5_FM2_SNAP_NO_MOVED") == nullptr;
    return on;
}

struct Counters {
    std::atomic<std::uint64_t> trustAsked{0};
    std::atomic<std::uint64_t> trusted{0};
    std::atomic<std::uint64_t> trustNotProved{0};
    std::atomic<std::uint64_t> trustMoved{0};
    std::atomic<std::uint64_t> trustEvicted{0};
    std::atomic<std::uint64_t> trustPlain{0};
    std::atomic<std::uint64_t> trustArmed{0};
    std::atomic<std::uint64_t> snapSkipped{0};
    std::atomic<std::uint64_t> targetAsked{0};
    std::atomic<std::uint64_t> targetReused{0};
    std::atomic<std::uint64_t> provesTrusted{0}, provesTrustedNs{0}, provesOther{0}, provesOtherNs{0}, prefixNs{0};
    std::atomic<std::uint64_t> revalidates{0}, revalidateNs{0}, movedCalls{0}, movedNs{0}, movedNonEmpty{0};
    std::atomic<std::uint64_t> bindCalls{0}, bindNs{0}, bindWithMoved{0}, bindWithMovedNs{0}, bindLoopNs{0}, snapInPlace{0}, snapNotInPlace{0};
};

inline bool Timing() {
    static const bool timing = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return timing;
}

inline Counters& Count() {
    static Counters counters;
    return counters;
}

inline void Report() {
    auto& c = Count();
    const auto load = [](const std::atomic<std::uint64_t>& value) { return static_cast<unsigned long long>(value.load(std::memory_order_relaxed)); };
    std::fprintf(stderr, "[fm2] on %d trust %d snap %d target %d; trust: asked %llu trusted %llu (not proved this present %llu, moved %llu, evicted %llu, plain-only %llu), armed %llu; snap: skipped %llu; target: asked %llu reused %llu\n", RuntimeSwitches::Set(RuntimeSwitch::Fm2) ? 1 : 0, On(RuntimeSwitch::Fm2Trust) ? 1 : 0, On(RuntimeSwitch::Fm2Snap) ? 1 : 0, On(RuntimeSwitch::Fm2Target) ? 1 : 0, load(c.trustAsked), load(c.trusted), load(c.trustNotProved), load(c.trustMoved), load(c.trustEvicted), load(c.trustPlain), load(c.trustArmed), load(c.snapSkipped), load(c.targetAsked), load(c.targetReused));
    std::fprintf(stderr, "[fm2-time] proofs trusted %llu in %llu us, other %llu in %llu us, prefix %llu us; draw-hit revalidates %llu in %llu us; moved-buffer checks %llu in %llu us (%llu non-empty); draw bindings %llu in %llu us, of them with moved buffers %llu in %llu us (snapshot loop %llu us); trust presents %llu; moved in place %llu, not in place %llu\n", load(c.provesTrusted), load(c.provesTrustedNs) / 1000, load(c.provesOther), load(c.provesOtherNs) / 1000, load(c.prefixNs) / 1000, load(c.revalidates), load(c.revalidateNs) / 1000, load(c.movedCalls), load(c.movedNs) / 1000, load(c.movedNonEmpty), load(c.bindCalls), load(c.bindNs) / 1000, load(c.bindWithMoved), load(c.bindWithMovedNs) / 1000, load(c.bindLoopNs) / 1000, static_cast<unsigned long long>(TrustPresents()), load(c.snapInPlace), load(c.snapNotInPlace));
}

}

#endif
