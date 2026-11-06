#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_C3LOG_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_C3LOG_HPP

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

// v3 C3 log-only experiment (APS5_V3E_C3_LOG=1, start-time, default off; no behaviour change). When a cacheable
// draw builds its resources, the guest bytes its capture read (the snapshots the build consumed) are copied into a
// log keyed by the resource-cache key. When a later draw finds that cache entry, the committer compares the log with
// the draw's snapshots (the bytes the walk reads now, in queue order) by memcmp, next to today's proof (Revalidate +
// the moved-buffer check), and counts the pair of verdicts with the proof's failure class, the log's bytes and both
// times. One [c3log] line per window (frame windows in a replay, 10 s in the game).
namespace AgcDriver::C3Log {

bool Enabled();

enum class Proof : std::uint8_t { Served, ValidMoved, Imports, Evicted, Pending, Changed, Keys, Other, Count };
enum class Skip : std::uint8_t { Uncacheable, NoEntry, InUse, NoLog, Count };

using Region = std::pair<std::uint64_t, std::span<const std::byte>>;

void Record(std::span<const std::uint32_t> key, std::span<const Region> regions);
bool Same(std::span<const std::uint32_t> key, std::span<const Region> regions, std::uint64_t& bytes, bool& logged, std::uint64_t& compareNs);
void NoteCompared(bool memcmpSame, Proof proof, std::uint64_t proofNs, std::uint64_t memcmpNs, std::uint64_t bytes, std::size_t regions);
void NoteSkipped(Skip reason);

enum class Step : std::uint8_t { None, NotReusable, InUse, Snapshots, Space, Served, Prove, Imports, Threw, Count };
struct Split {
    std::uint64_t keyNs = 0, findNs = 0, snapshotsNs = 0, spaceNs = 0, proveNs = 0, importsNs = 0, movedNs = 0;
    Step refused = Step::None;
    bool epochReused = false;
    bool keyBuilt = false;
};
Split& ThreadSplit();
void NoteSplit(const Split& split, bool served);
void Close(const char* label);
void MaybeReport();

}

#endif
