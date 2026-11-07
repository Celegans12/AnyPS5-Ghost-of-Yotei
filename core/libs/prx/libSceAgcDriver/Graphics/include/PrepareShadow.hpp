#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PREPARESHADOW_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PREPARESHADOW_HPP

#include <cstdint>
#include <functional>

// v3 lock split, prepare-shadow census (APS5_V3E_PREPARE_SHADOW=1, start-time, log-only): before phase (ii) moves the
// draw-set proof out of the record lock, measure how often its prepared result would fail validation. Each pipelined
// draw takes a PrepareStamp, the write tracker's generation and the pending-write snapshot when it is prepared (by
// the preparer thread when it runs ahead, else when the worker enqueues it); its commit's draw-set proof, under the
// record lock, then builds the read set it would have needed (the set's direct regions, described ranges, storage
// images and validated textures, with the generation and, for pending writes, whether each range overlapped the
// prepare-time snapshot) and runs GuestMemory::Validate on it. The result goes to mem's [v3m-validate] census: site
// "draw" for enqueue-stamped commits, "other" for preparer-stamped ones. Nothing uses the result.
namespace AgcDriver::Graphics::PrepareShadow {

bool Enabled();
// Wraps a draw item's commit and preparation so both see one capture.
void Wrap(std::function<void()>& commit, std::function<void()>& prepare);
// Called by the draw-set proof with the set's read ranges; validates once per commit, under the record lock.
bool Active();
void NoteDrawSet(const std::function<void(const std::function<void(std::uint64_t, std::uint64_t)>&)>& ranges);

}

#endif
