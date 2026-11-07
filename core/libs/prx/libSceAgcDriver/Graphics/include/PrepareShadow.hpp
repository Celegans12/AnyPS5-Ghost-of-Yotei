#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PREPARESHADOW_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PREPARESHADOW_HPP

#include <cstdint>
#include <functional>
#include <span>

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
// ([prepare-shadow2], and [v3m-validate] site "other"). Ranges a copied writer overlaps, and every set holding a BDA
// lease, are entered unvouched (AddUnwiredEntry); unarmed ranges are counted by GuestDirty's refusal reason.
// Phase (ii), APS5_V3E_PREPARE_PROOF (RuntimeSwitch, with APS5_V3E_LOCK_SPLIT): the preparer builds the read set from
// the set's published ProofView and runs ShaderResources::ProveAhead on it, taking no cache lock; the commit's revalidateFull serves the set from that verdict when the read
// set validates, the storage-results and shadow registries did not move and its own re-checks pass, else it runs
// today's proof ([prepare-proof]). APS5_V3E_PREPARE_PROOF_VERIFY=throw runs today's proof too and throws on a difference.
namespace AgcDriver::Graphics {
class ShaderResources;
struct CompiledShader;
}

namespace AgcDriver::Graphics::PrepareShadow {

bool Enabled();
int Mode();
// Whether draw items are wrapped: a census mode, or APS5_V3E_PREPARE_PROOF named in the environment or APS5_AB.
bool Wrapping();
// Wraps a draw item's commit and preparation so both see one capture.
void Wrap(std::function<void()>& commit, std::function<void()>& prepare);
using Ranges = std::function<void(const std::function<void(std::uint64_t, std::uint64_t)>&)>;
// v1/v2, at the commit's draw-set proof (record lock held): validates once per commit.
bool Active();
void NoteDrawSet(const void* set, const Ranges& ranges);
// v2 / phase (ii), on the preparer inside PrepareDraw: the set the resource cache holds for the draw's key now (null when
// none).
bool Preparing();
void NotePreparedSet(const ShaderResources* set, std::span<const CompiledShader> shaders);

// Phase (ii), at the commit (record lock held).
bool ProofOn();
bool VerifyProof();
// Diagnostics (RuntimeSwitches): APS5_V3E_PP_PUBLISH_ONLY publishes ProofViews with phase (ii) off (the publication
// cost alone); APS5_V3E_PP_NO_PUBLISH stops republishing with phase (ii) on (a set's first view, at the cache insert,
// stays: the prepare work alone).
bool PublishOnly();
bool NoPublish();
// Whether this commit may serve `set` from its prepared verdict: same set, verdict Pass, read set Valid (the storage-results
// and shadow checks re-run at the commit when their serials moved: RecheckPending). Counts the outcome; once per commit.
bool TakePrepared(const void* set);
enum class Fallback : std::uint8_t { None, Moved, Evicted, Depth, Imports, Untracked, Pending, Count };
// Whether this commit's storage-results or shadow serial moved since prepare (the commit re-runs those checks).
bool RecheckPending();
void NoteFallback(Fallback reason);
void NoteServed(bool served);
void NoteVerify(bool same);

}

#endif
