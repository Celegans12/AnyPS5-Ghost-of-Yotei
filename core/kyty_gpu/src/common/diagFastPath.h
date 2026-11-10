#ifndef KYTY_COMMON_DIAGFASTPATH_H_
#define KYTY_COMMON_DIAGFASTPATH_H_

#include <atomic>
#include <cstdint>

// KYTY_PROFILER_BLOCK (profiler.h): tracy::ProfilerAvailable() is an out-of-line call per zone, so the zone asks the
// profiler bit below (set by Profiler::Initialize, cleared by Shutdown) instead: one relaxed load of a global and a test.
namespace Common::DiagFastPath {

inline constexpr uint8_t kProfiler = 4u; // the Tracy profiler is running (Profiler::Initialize .. Profiler::Shutdown)

inline constinit std::atomic<uint8_t> g_state {0};

// Profiler::Initialize / Shutdown: whether the Tracy profiler is running.
void NoteProfiler(bool running);

} // namespace Common::DiagFastPath

#endif // KYTY_COMMON_DIAGFASTPATH_H_
