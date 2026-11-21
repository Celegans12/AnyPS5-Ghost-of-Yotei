#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CX_COMPAT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CX_COMPAT_H_

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <string_view>

extern "C++" {
inline volatile std::atomic_uint32_t kyty_local_table_xpr_mode {2};
inline volatile std::atomic_uint32_t kyty_local_table_memo_verify {0};
inline volatile std::atomic_uint32_t kyty_local_speculate_mode {0};
inline volatile std::atomic_uint32_t kyty_local_speculations_live {0};
}

namespace Libs::Graphics {

namespace TableCount {
enum Kind : uint32_t { Event, PairRefused, DrawRefused, Kinds };
struct Slot {
	std::atomic<const char*> what {nullptr};
	std::atomic<uint64_t>    count {0};
};
inline constexpr size_t                            Slots = 256;
inline std::array<std::array<Slot, Slots>, Kinds> g_slots {};
inline void Add(const char* what, Kind kind = Event, uint64_t n = 1) noexcept {
	if (what == nullptr || n == 0) return;
	auto&  slots = g_slots[kind];
	size_t at    = (reinterpret_cast<uintptr_t>(what) * 0x9e3779b97f4a7c15ull) >> 56u;
	for (size_t probe = 0; probe < Slots; ++probe, at = (at + 1) & (Slots - 1)) {
		auto&       slot = slots[at];
		const char* held = slot.what.load(std::memory_order_acquire);
		if (held == nullptr && slot.what.compare_exchange_strong(held, what, std::memory_order_acq_rel)) held = what;
		if (held == what) {
			slot.count.fetch_add(n, std::memory_order_relaxed);
			return;
		}
	}
}
inline uint64_t TableNow() noexcept {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
inline uint64_t TableSampleStart() noexcept {
	static thread_local uint32_t calls = 0;
	return (++calls & 15u) == 0 ? TableNow() : 0;
}
}

namespace Spec {
struct State {
	const char* refused = nullptr;
	void        NoteSet(uint64_t) {}
	void        NoteBufferRange(uint64_t, uint64_t) {}
	void        NoteCleanRead(uint64_t, uint64_t) {}
	bool        JournalIntersects(uint64_t, uint64_t) const { return false; }
};
[[nodiscard]] inline State* Current() noexcept { return nullptr; }
[[nodiscard]] inline bool   On() noexcept { return false; }
inline void Refuse(const char* why) noexcept {
	TableCount::Add(why);
}
}

namespace LiveCounters {
enum Id : uint32_t {
	TableDrawNative, TableStoreDeferred, TableStoreTargets, TableRingSharedBytes, TableRingShared, TableRegistered,
	TableStoreVariant, TableRingCopies, TableRingBytes, TableRefusedSets,
	TableNoSet, TableMemoVerified, TableMemoMismatch, TableMemoHits, TableGatherMemoHits,
	TableEvaluations, TableDraws, TableContinued,
	ScoutUsedBacking, ScoutUsed, ScoutLostWritten, ScoutLostWriteBack, ScoutLostWatched,
	ScoutLostRegisters, ScoutLostPair, ScoutLostNoView, ScoutLostMissing, ScoutLostEpoch,
	ScoutLostDirty, ScoutLostChanged, ScoutChecks,
	Count
};
inline void Add(Id, uint64_t = 1) noexcept {}
}

namespace LiveCensus {
enum Kind : uint32_t { DrawPhase };
struct Scope {
	Scope(Kind, uint32_t, uint32_t) noexcept {}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;
};
}

namespace LiveSwitch {
inline std::atomic<int64_t>& Find(std::string_view, int64_t fallback) {
	static std::atomic<int64_t> value {0};
	value.store(fallback, std::memory_order_relaxed);
	return value;
}
}

namespace SlowLog {
inline double HitchThreshold() noexcept { return 0.0; }
}

namespace FrameGen {
inline bool Enabled() noexcept { return false; }
inline bool WantsCamera() noexcept { return false; }
inline bool IsGbufferVertexShader(uint64_t) noexcept { return false; }
}

}

#endif
