#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PREPARESHADOW_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PREPARESHADOW_HPP

#include <cstdint>
#include <functional>

// v3 lock split, prepare-shadow census (start-time, log-only; nothing uses its results).
// APS5_V3E_PREPARE_SHADOW=1 (v1): each pipelined draw takes a PrepareStamp, the tracker generation and the pending-write
// snapshot when it is prepared (by the preparer when it runs ahead, else at enqueue); its commit's draw-set proof, under
// the record lock, builds the read set from the committed set's ranges and runs GuestMemory::Validate (mem's
// [v3m-validate]: site "draw" enqueue-stamped, "other" preparer-stamped).
// APS5_V3E_PREPARE_SHADOW=2 (v2): the read set is built on the preparer, from the set the resource cache holds for the
// draw's key at prepare time (the read-only lookup phase (ii) would make), under the resource-set cache lock (shared;
// needs APS5_V3E_LOCK_SPLIT for the commits' exclusive stage holds): per range a tracker generation (write-watched) or
// a dirty token armed then, and the facts PendingWrites, StorageResults and Shadows. The commit validates it when it
// commits that same set; a commit with no set at prepare time, or another set, counts as a re-prepare
// ([prepare-shadow2], and [v3m-validate] site "other").
namespace AgcDriver::Graphics::PrepareShadow {

bool Enabled();
int Mode();
// Wraps a draw item's commit and preparation so both see one capture.
void Wrap(std::function<void()>& commit, std::function<void()>& prepare);
using Ranges = std::function<void(const std::function<void(std::uint64_t, std::uint64_t)>&)>;
// v1/v2, at the commit's draw-set proof (record lock held): validates once per commit.
bool Active();
void NoteDrawSet(const void* set, const Ranges& ranges);
// v2, on the preparer inside PrepareDraw: the set the resource cache holds for the draw's key now (null when none).
bool Preparing();
void NotePreparedSet(const void* set, const Ranges& ranges);

}

#endif
