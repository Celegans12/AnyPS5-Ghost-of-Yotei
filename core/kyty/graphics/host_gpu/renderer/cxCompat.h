#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CX_COMPAT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CX_COMPAT_H_

#include <atomic>
#include <cstdint>
#include <string_view>

extern "C++" {
inline volatile std::atomic_uint32_t kyty_local_table_xpr_mode {2};
inline volatile std::atomic_uint32_t kyty_local_table_memo_verify {0};
inline volatile std::atomic_uint32_t kyty_local_speculate_mode {0};
inline volatile std::atomic_uint32_t kyty_local_speculations_live {0};
}

namespace Libs::Graphics {

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
inline void Refuse(const char*) noexcept {}
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
inline bool IsTaaShader(uint64_t) noexcept { return false; }
inline bool IsGbufferVertexShader(uint64_t) noexcept { return false; }
inline void NoteScene() noexcept {}
}

}

#endif
