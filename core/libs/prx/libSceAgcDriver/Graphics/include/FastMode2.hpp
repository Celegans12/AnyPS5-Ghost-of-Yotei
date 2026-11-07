#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_FASTMODE2_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_FASTMODE2_HPP

#include <atomic>
#include <cstdint>
#include <cstdio>
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
};

inline Counters& Count() {
    static Counters counters;
    return counters;
}

inline void Report() {
    auto& c = Count();
    const auto load = [](const std::atomic<std::uint64_t>& value) { return static_cast<unsigned long long>(value.load(std::memory_order_relaxed)); };
    std::fprintf(stderr, "[fm2] on %d trust %d snap %d target %d; trust: asked %llu trusted %llu (not proved this present %llu, moved %llu, evicted %llu, plain-only %llu), armed %llu; snap: skipped %llu; target: asked %llu reused %llu\n", RuntimeSwitches::Set(RuntimeSwitch::Fm2) ? 1 : 0, On(RuntimeSwitch::Fm2Trust) ? 1 : 0, On(RuntimeSwitch::Fm2Snap) ? 1 : 0, On(RuntimeSwitch::Fm2Target) ? 1 : 0, load(c.trustAsked), load(c.trusted), load(c.trustNotProved), load(c.trustMoved), load(c.trustEvicted), load(c.trustPlain), load(c.trustArmed), load(c.snapSkipped), load(c.targetAsked), load(c.targetReused));
}

}

#endif
