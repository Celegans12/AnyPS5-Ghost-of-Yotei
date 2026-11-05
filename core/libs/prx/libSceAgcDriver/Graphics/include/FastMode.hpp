#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_FASTMODE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_FASTMODE_HPP

// APS5_FAST_MODE (PLAN v3 lane B): an opt-in, inexact fast mode. Never the default and never compared to T1; it
// bundles the P0 ablations that follow KytyPS5-Yotei's inexact choices (ideas only, no code). Unset, no item takes
// any new branch.
//
// Master switch: APS5_FAST_MODE. Static items read it once at startup (the environment); runtime items follow the
// RuntimeSwitch of the same name, so APS5_AB=60:APS5_FAST_MODE flips them in-run. Each item has a sub-switch
// APS5_FM_<ITEM>; unset = the item's default level, 0 = off, n = level n.
//   PROOFS  (runtime, default 4)  proof walks and cached-set proofs only every n-th present; in between,
//                                  CollectWrites returns the tracker's generation and Revalidate's full proof
//                                  accepts (P0-cpu APS5_P0C_NO_PROOFS, staleness bounded to n presents)
//   QUIET   (runtime, default 1)  bindless publishes record nothing and a bindless draw continues the open pass
//                                  (APS5_QUIET_PUBLISH + APS5_P0D_HEAP_CONTINUE)
//   MERGE   (runtime, default 1)  an open pass of the same key continues through hazards and forced breaks
//                                  (APS5_P0D_MERGE_PASSES)
//   STAGING (runtime, default 1)  1: resident staged ranges without the 32-entry / 8-use / 64 KiB limits (exact
//                                  design, VRAM-capped); 2 (static: needs APS5_FAST_MODE at start): the P0-mem E1
//                                  ceiling (one persistent shadow per range, copied in once, never copied back)
//   PUBLISH (runtime, default 1)  shadow publishes for address-based heaps are skipped and left pending (BDA
//                                  readers read the import as it is; a later hook or eviction publishes them);
//                                  2 (static): the P0-mem publish ceiling
//   CLONE   (static, default 0)   MiB budget of device-local import clones for BDA reads (P0-gpu
//                                  APS5_P0G_CLONE_IMPORTS: made at first use, never refreshed, stores land in them)
//   FLOAT   (static, default 1)   native f32 in the recompiler (P0-shader APS5_P0S_NATIVE_FLOAT; cache-keyed, read
//                                  in core/shader/recompiler/P0sCeiling.hpp)
// APS5_FM_FRAME_COUNTS=1 (log only, any mode): one [fm-frame] line per present with the guest draw and dispatch
// packets since the previous present (validation rule (a)).
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace AgcDriver::Graphics::FastMode {

enum class Item : unsigned { Proofs, Quiet, Merge, Staging, Publish, Clone, Float, Count };

inline constexpr std::array<const char*, static_cast<std::size_t>(Item::Count)> SubNames{"APS5_FM_PROOFS", "APS5_FM_QUIET", "APS5_FM_MERGE", "APS5_FM_STAGING", "APS5_FM_PUBLISH", "APS5_FM_CLONE", "APS5_FM_FLOAT"};
inline constexpr std::array<std::uint32_t, static_cast<std::size_t>(Item::Count)> Defaults{4u, 1u, 1u, 1u, 1u, 0u, 1u};
inline constexpr std::array<bool, static_cast<std::size_t>(Item::Count)> RuntimeItem{true, true, true, true, true, false, false};

inline bool StartedOn() {
    static const bool on = std::getenv("APS5_FAST_MODE") != nullptr;
    return on;
}

inline std::uint32_t Level(Item item) {
    static const auto levels = [] {
        std::array<std::uint32_t, static_cast<std::size_t>(Item::Count)> values{};
        for (std::size_t index = 0; index < values.size(); ++index) {
            const char* text = std::getenv(SubNames[index]);
            values[index] = text != nullptr && *text != '\0' ? static_cast<std::uint32_t>(std::strtoul(text, nullptr, 0)) : Defaults[index];
        }
        return values;
    }();
    return levels[static_cast<std::size_t>(item)];
}

inline void Announce() {
    static const bool done = [] {
        if (!StartedOn() && std::getenv("APS5_AB") == nullptr) return true;
        std::fprintf(stderr, "[fast-mode] APS5_FAST_MODE %s at start (inexact, opt-in):", StartedOn() ? "set" : "unset");
        for (std::size_t index = 0; index < SubNames.size(); ++index) std::fprintf(stderr, " %s=%u%s", SubNames[index] + 8, Level(static_cast<Item>(index)), RuntimeItem[index] ? "(rt)" : "");
        std::fprintf(stderr, "\n");
        return true;
    }();
    static_cast<void>(done);
}

// The item's level now: a static item's when APS5_FAST_MODE was set at start, a runtime item's while the
// RuntimeSwitch is set; else 0.
inline std::uint32_t On(Item item) {
    const bool master = RuntimeItem[static_cast<std::size_t>(item)] ? RuntimeSwitches::Set(RuntimeSwitch::FastMode) : StartedOn();
    if (!master) return 0;
    Announce();
    return Level(item);
}

// A level that only takes effect when APS5_FAST_MODE was set at start (the static ceilings of runtime items).
inline bool StaticLevel(Item item, std::uint32_t level) {
    return StartedOn() && Level(item) == level;
}

inline std::atomic<std::uint64_t>& PresentCounter() {
    static std::atomic<std::uint64_t> presents{0};
    return presents;
}

// PROOFS: whether a proof may be skipped now (fast mode on and this is not a proof present). APS5_FM_PROOF_PARTS
// (bit mask, default 1): 1 = CollectWrites walks; 2 = Revalidate's full proof of cached resource sets (hangs the GPU and crashes at exit in replay: not in the default).
inline std::uint32_t ProofParts() {
    static const std::uint32_t parts = [] {
        const char* text = std::getenv("APS5_FM_PROOF_PARTS");
        return text != nullptr && *text != '\0' ? static_cast<std::uint32_t>(std::strtoul(text, nullptr, 0)) : 1u;
    }();
    return parts;
}

inline bool SkipProof(std::uint32_t part) {
    if ((ProofParts() & part) == 0) return false;
    const auto every = On(Item::Proofs);
    if (every == 0) return false;
    return every == 1 ? false : PresentCounter().load(std::memory_order_relaxed) % every != 0;
}

inline bool FrameCounts() {
    static const bool on = std::getenv("APS5_FM_FRAME_COUNTS") != nullptr;
    return on;
}

struct Counts {
    std::atomic<std::uint64_t> draws{0}, dispatches{0};
};

inline Counts& FrameCounters() {
    static Counts counts;
    return counts;
}

inline void CountDraw() {
    if (FrameCounts()) FrameCounters().draws.fetch_add(1, std::memory_order_relaxed);
}

inline void CountDispatch() {
    if (FrameCounts()) FrameCounters().dispatches.fetch_add(1, std::memory_order_relaxed);
}

inline void NotePresent(std::uint64_t presents) {
    PresentCounter().store(presents, std::memory_order_relaxed);
    if (!FrameCounts()) return;
    const auto draws = FrameCounters().draws.exchange(0, std::memory_order_relaxed);
    const auto dispatches = FrameCounters().dispatches.exchange(0, std::memory_order_relaxed);
    std::fprintf(stderr, "[fm-frame] present %llu draws %llu dispatches %llu fast %d\n", static_cast<unsigned long long>(presents), static_cast<unsigned long long>(draws), static_cast<unsigned long long>(dispatches), RuntimeSwitches::Set(RuntimeSwitch::FastMode) ? 1 : 0);
}

}

#endif
