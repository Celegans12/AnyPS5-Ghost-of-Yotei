# Direct path: design, draft 3

Status: approved by the user 2026-11-11, including the three decisions at the end as recommended
(red-zone rule reversed for titles that pass the phase 1 gate; LEGO for the GPU track and AoM for
the coherence track, 007 once it has a repeatable gameplay scene; 30 fps first for AoM and 007).
Divinity is not a test title. Branch base: upstream 1b485ff745 merged 2026-11-11. This draft
supersedes the earlier "Gates from here" and "007 follow-up" plans; the rest of the design (direct
shaders, heap slots, table mirror, fallback) stays as built. Branch `rewrite/direct`.

## Why a new draft

Draft 2 made each draw cheaper to prepare. Measurements since then say draw preparation is no
longer the only limit, and on some titles not the main one.

| Title | Frame now | GPU busy per frame | What limits it (measured) |
|---|---|---|---|
| LEGO | about 48 ms | 39 ms | GPU: one translated compute program takes 23 ms |
| FF Resonance | about 40 ms | not re-profiled | unknown on this build; was CPU-bound |
| AoM title | about 298 ms | 274 ms | GPU timeline: about 300 pass breaks, barriers, 300 MiB of storage copies; draws 267 us each |
| 007 intro | about 550 ms | 51 to 58 ms | old path only so far: uploads (3.0 s per 10 s), bindings (1.35 s), five workers at 99 % while the GPU idles 90 % |

Target: 16.7 ms per frame. That needs the CPU worker at or under about 10 ms per frame and the GPU
at or under about 14 ms. No title is within 2x on either side.

Three limits are left, and the direct path as built has a mechanism for none of them:

1. **Coherence.** The CPU re-establishes what guest memory holds at every ordering point: write-watch
   walks, collect epochs, memo revalidation, uploads of data that did not change, syncs before
   captures. The cost scales with how often memory is *used*, not how often it *changes*.
2. **Waiting.** Workers poll for labels and for each other while the GPU is idle.
3. **GPU time.** Translated shaders, pass breaks, barriers and copies cost 3x to 16x the budget.

## Principles

1. Work happens when memory changes, and the change is an event. Nothing asks "did this change?"
   per draw, per epoch or per submission.
2. Descriptors stay CPU-known. Shaders never chase guest pointers on the GPU.
3. GPU results stay on the GPU until the CPU actually reads them.
4. A worker never polls. It blocks on the event that satisfies its wait.
5. The GPU timeline carries shader work: one pass for consecutive draws into the same targets,
   barriers only for real hazards.
6. Every title keeps running: each mechanism has the existing path as its fallback, chosen per
   title or per draw.
7. Footprint target 16 GiB combined. No change is kept for a timing gain that grows a cache limit.

## Architecture

### A. Event-driven memory ownership

Each 64 KiB block of GPU-registered guest memory is in one of three states:

| State | Protection | Meaning | Leaves the state when |
|---|---|---|---|
| CPU-clean | read-only | The GPU-side copy and all memos of it are current | a guest write faults: block becomes CPU-dirty and is appended to the dirty list |
| CPU-dirty | read-write | The CPU may be writing | a submission consumes the dirty list: block is uploaded once, re-armed, becomes CPU-clean |
| GPU-owned | no access | The newest content is on the GPU | a guest read or write faults: the block is written back, then becomes CPU-dirty |

- The dirty list replaces `CollectWrites` walks, collect epochs and the visibility policies tried in
  earlier work. A submission reads a list; it does not walk ranges.
- GPU-owned replaces eager write-back and the syncs before CPU reads of GPU results (DMA_DATA
  sources, storage image copies). Data nobody reads on the CPU is never copied back.
- Blocks that fault every frame stay CPU-dirty permanently (no fault cost) and are uploaded per
  submission, as today.

**The precondition is red-zone safety.** A fault on a guest thread makes Windows write the exception
frame below RSP, into the 128 bytes that System V leaf functions use for locals. That is why
upstream removed protection-based tracking (#9181, #5596) and why it was ruled out here. These
titles are converted, so the fix belongs in the relinker: `--frame-red-zone` already frames leaf
functions and reports the ones it skips. Section A is allowed for a title only when that report
shows no skipped function in any converted module (eboot and every PRX). A title that cannot reach
that keeps the write-watch model.

### B. Residency

The device-local mirror, now fed by section A instead of walks:

- CPU-clean and GPU-owned blocks have a device-local copy; shaders and draws read that, never a
  host import.
- One ledger tracks owner, dirty state and in-flight uses for imports, mirrors and staging pools,
  under one budget from `VK_EXT_memory_budget`. Least recently used blocks are evicted by writing
  back and pointing the range at RAM.
- An upload happens once per change. Today 007 spends 3.0 s per 10 s re-uploading.

### C. Waits

- A wait whose value comes from a recorded or queued label store is resolved from the record, in
  order, with no GPU completion.
- A wait on another worker's not-yet-recorded store blocks on a condition that the recording
  signals. `PollSleep` goes away.
- Only a wait on a value the GPU computes blocks on GPU completion, through a fence or timeline
  semaphore.

### D. Direct path (as built, with new validity)

Unchanged in form: light capture, data-independent shaders, heap slots, table mirror, fallback per
draw. One change: memos (slot memo, table mirror, column stamps, binding plans) are valid until a
dirty event on their source blocks, instead of per collect epoch. With A in place the per-draw
proof becomes a flag read.

### E. GPU stage

Two separate pieces of work, both per title:

- **Timeline:** merge consecutive draws into one pass, drop barriers without a hazard, keep storage
  images on the GPU (section A makes this legal). AoM is the test: 274 ms busy for about 40 ms of
  shaders.
- **Translation:** the dominant program per title. LEGO's is latency-bound with no single emulation
  cost to remove (first GPU profile). The open question is whether the wave64 mapping can
  be restructured (one lane per invocation changes occupancy and the LDS pattern) or whether the
  guest workload at 3840x2160 is the cost. This is the least certain part of the plan.

## Phases

| Phase | Work | Gate to continue | If the gate fails |
|---|---|---|---|
| 0 | Hygiene. Reject stage A (slower in 3 pairs), save as patch. Split and commit or shelve the sparse-settle change. One fixed gameplay scene per title; 007 run with `APS5_DIRECT_DRAW=1`. Worker and GPU profile of each title on the direct path. | A ranked list per title: ms per frame in coherence, waiting, GPU timeline, shaders | n/a |
| 1 | Red-zone audit. Convert all four titles with `--frame-red-zone`; get skipped functions to zero. Build a fault-storm test: arm everything, fault constantly with AVX state live. | Zero skipped in every module; storm runs 30 min clean per title; the same storm *without* framing reproduces corruption (proves the test can fail) | Keep write-watch. Do one hot-set walk per submission instead of A |
| 2 | Section A, CPU-clean and CPU-dirty only. Start from the `APS5_PROTECT_PRIVATE_WATCH` prototype (branch proto/fault-tracking), with its correctness fixes. | Walks and epoch bumps near zero; the phase 0 coherence share falls by at least half on AoM and FF; pictures match | Revert, keep as patch |
| 3 | Section C. | Worker wait time under 2 ms per frame on LEGO and 007; GPU idle gaps under 20 % of the frame where the CPU is not the limit | Revert per wait class |
| 4 | Section D validity change, then section B and the GPU-owned state. | Uploads and write-backs per frame proportional to changed bytes; 007 and LEGO upload time down by 5x; footprint not above today's | Keep eager write-back for the failing resource class |
| 5 | Section E, timeline first (AoM), then translation (LEGO). | GPU busy at or under 14 ms on at least one title | Report the floor per title; that is the result for the pitch |
| 6 | Per-draw CPU toward 20 us (light capture, bindings). | Worker at or under 10 ms per frame | n/a |

Phases 3 and 5 do not depend on phase 1 and can run while the audit is open. Each kept change
gets one regression pair on a second title, as now.

## Phase 0 result (2026-11-11, merged build, direct path)

Per frame, with draw and GPU profiling on (frames are 1 to 3 ms slower than unprofiled):

| | LEGO | FF Resonance | AoM title |
|---|---|---|---|
| Frame p50 | 48.6 ms | 37.0 ms | 254 ms |
| GPU busy / idle gaps | 39.0 / 8.3 ms | 20.2 / 16.8 ms | 256 / 0.5 ms |
| Bound by | GPU | CPU (worker 0 at 97 %) | GPU |
| Shader programs (GPU) | 31 ms, one program 22.8 ms | 4.4 ms | 42 ms |
| Draw passes (GPU) | 2.3 ms | 7.9 ms | 200 ms |
| Storage upload, write-back, staging, publish (GPU) | 4.5 ms (3.0 GiB uploaded) | 5.7 ms | 7.9 ms |
| Draws per frame x CPU each | 1,208 x 12 us = 14.6 ms | 618 x 22 us = 14.7 ms | 757 x 25 us = 19 ms |
| Render passes begun per frame | 112 | 267 (191 forced) | 745 (702 forced) |
| Other CPU | dispatches 12.2 ms; presenter waits 10.8 ms on frames in flight | memory_wait 10.3 ms; stage materialize 10.8 ms | worker waits 207 ms on the GPU fence |

Ranking and consequences:

1. **AoM is a pass-break problem, not a coherence problem.** 200 of 254 ms is GPU time in draw
   passes: almost every draw starts its own render pass. A pass is forced (Draw.cpp, the
   `StartDrawPass` call) when the draw captures, reads its target, uses GPU-side indirect arguments,
   is a mesh indirect draw, or has staged copies. AoM's draws are GPU-side indirect. Its CPU cost
   per draw is already 25 us. So AoM moves to the GPU timeline track and is the first phase 5 item:
   keep GPU-indirect draws in the open pass (stage their arguments before the pass).
2. **FF Resonance is the CPU-bound title** and replaces AoM on the coherence and waits track. It is
   also the closest to target: its shaders take 4.4 ms. Removing the forced passes helps it too.
3. **LEGO is bound by one compute program** (22.8 of 39 ms). Everything else on its GPU is about
   16 ms, so LEGO cannot pass 60 fps without that program at about a third of its cost.
4. Per-draw CPU is 12 to 25 us on all three; phase 6 is nearly met for these scenes.

Order from here: phase 5 timeline item (forced passes) first, since it is the largest measured cost
and does not depend on phase 1; phase 1 audit in parallel; then phases 2 to 4 measured on FF.

### Phase 5, item 1 done: indirect draws continue the pass (9673e74b06, 2026-11-11)

A GPU-side indirect draw whose arguments the CPU can snapshot records no command, so it no longer
forces a pass. `APS5_NO_INDIRECT_IN_PASS=1` restores the old behaviour. Alternating pairs, same
build:

| | Off | On |
|---|---|---|
| AoM title | 256.9 / 256.1 ms | 220.6 / 221.6 ms (passes per frame 745 -> 72, none forced) |
| FF Resonance | 32.9 / 32.9 ms | 32.4 / 32.5 ms (about 41 passes per frame still forced by indirect draws that cannot be snapshotted) |

CORRECTION to point 1 above: the pass boundaries were about 35 ms of AoM's frame, not 200. The
"draw" class of `[gputime]` covers everything inside draw passes, including shading; with the
passes merged it is still 158 ms per frame. Per-draw timing (`APS5_PROFILE_GPU_DRAWS=1`, per frame at 43 presents per 10 s):

| Pixel program (variant, code address) | Draws per frame | GPU per frame |
|---|---|---|
| 0x7f8, 0x3f8610600 | 129 | 47 ms |
| 0x7f6, 0x3f8648000 | 70 | 44 ms |
| 0x7a2, 0x3f7c69200 | 1 | 24 ms |
| 0x78f, 0x3f7c80400 | 1 | 12 ms |
| 0x789, 0x3f802a900 | 340 | 10 ms |
| 0x7d2, 0x3f85cd100 | 6 | 8 ms |

Four pixel programs take about 127 ms of the frame. So AoM is bound by translated pixel shaders,
like LEGO is by one compute program: phase 5 "translation" is the main item for both. Next: dump
these four (AoM code addresses are stable across runs) and find what in the translation costs the
time (table lookups through the heap slot map, host-import reads, float paths).

### AoM pixel shaders: first look (2026-11-11)

Dumps: `draw_<code address>.spv` and `.txt` in the `APS5_DIRECT_DUMP_SPIRV` directory (the switch
also writes direct draw stages). The 47 ms program (0x3f8610600) is small: 12 texture samples,
one loop, 8.6k words. Two things in it are not in the guest shader:

- A fault buffer (storage buffer with atomics and stores) and a final conditional `OpKill`, with no
  `EarlyFragmentTests`: side effects make the GPU shade pixels that depth would reject.
- Its loop is driven by subgroup ballots and a shuffle across neighbouring pixels (the emulation of
  the guest wave's control flow).

Ceiling experiment `APS5_X_FORCE_EARLY_Z=1` (early tests forced on every pixel shader without a
depth, stencil or coverage export; wrong picture for alpha-tested draws; experiment only, not in
the tree): GPU busy 213 -> 176 ms per frame, frame 213 -> 219
ms. So shading rejected pixels is at most about 37 ms of AoM's GPU time, and with the GPU at 176 ms
the frame did not get shorter: something else holds the frame near 215 ms. Open questions, in order:

ANSWER to question 1 (profile of the forced run): the CPU. With the GPU at 180 ms the worker is
99 % busy and spends about 200 ms per frame in packets: draws 118 ms (of which stage preparation,
`Shader.DrawStage`, 61 ms) and dispatches 80 ms (140 per frame, `Vulkan.Dispatch` 54 ms). That is
about 156 us per draw, not the 25 us in the phase 0 table: the `[draws]` average counts only the
record step (`Graphics::Draw`), and in the phase 0 run the worker's time was hidden behind its 207 ms
wait on the GPU. CORRECTION to the phase 0 table and ranking: AoM is bound by CPU and GPU at about
the same level (about 200 ms each), so it needs both tracks; the LEGO and FF figures were checked
against their packet times and stand (15.8 and 14.7 ms per frame in draws).

1. What bounds the frame when GPU busy falls (profile the forced run's worker and presenter waits).
2. What the ballot loop costs (a ceiling switch that runs the loop per pixel without ballots).
3. Whether fault recording can leave pixel shaders (it blocks early rejection wherever the guest
   allows it; worth up to the 37 ms).

### AoM GPU time: host-import reads confirmed (2026-11-11)

Not resolution: the trace (`APS5_TRACE_DRAWS=1`) shows the scene is rendered
at 1920x1080 (about 133 draws per frame into 6 targets), shadows at 2048x2048 depth-only (about
527), and only about 42 draws go to the 3840x2160 target. Not the ballot loop either: it is the
guest's own readfirstlane loop for per-pixel texture indices and runs once for one material.

`APS5_HOST_IMPORT_MIB=64` (imports refused, buffers served from device-local mirrors; per frame,
3 frames in the window, same draw counts):

| Program | Host imports (normal) | Device-local mirrors |
|---|---|---|
| draw 0x7f8 (G-buffer, about 125 draws) | 47 ms | 4.3 ms |
| draw 0x7f6 (about 67 draws) | 44 ms | 5 ms |
| draw 0x789 (shadows, about 330 draws) | 10 ms | 3 ms |
| compute 0x3f7a97000 (about 40 dispatches) | 10.5 ms | 3.3 ms |
| draw 0x7a2 (full screen, samples images) | 24 ms | 27 ms |
| draw 0x78f (full screen) | 12 ms | 12.7 ms |

Draws that read guest buffers get about 10x faster when the buffers are in VRAM; draws that only
sample images do not change. About 90 ms of AoM's roughly 213 ms of GPU time is host-import
reads. The crude mode itself is unusable: today's mirrors compare and copy blocks on every use
(7.7 GiB of staging copies and 2.9 s per frame). So section B is confirmed as AoM's largest GPU
item, and what it needs is exactly its design: a device-local copy that is refreshed only for
blocks that changed.

First build step that does not need phase 1 (walk-based): for buffer ranges that draws and
dispatches read and the write watch has shown clean for several frames, bind a device-local copy;
drop back to the import when a write is collected. Measure on AoM (target: the four buffer-reading
programs near their mirror times with frame time not worse), then FF and LEGO.

### Phase 1 audit, first pass (2026-11-11)

`relinker --windows --frame-red-zone --dry-run` on each title's executable (LEGO and FF need
`--sce-module-path <game>\app0`).

| Title | Functions with bounds | Leaves using the red zone | Framed | Skipped |
|---|---|---|---|---|
| LEGO | 35,883 | 733 | 575 | 158 |
| FF Resonance | 17,288 | 292 | 253 | 39 |
| AoM | 123,536 | 672 | 525 | 147 |

The phase 1 gate (zero skipped) fails on all three today. Three gaps:

1. **Skipped leaves, by reason (LEGO / FF / AoM):** exit shorter than a jump 87 / 3 / 50; push or
   pop in a leaf without pushes 0 / 5 / 45; indirect jump 15 / 8 / 6; other use of rsp 14 / 9 / 10;
   prologue shorter than a jump 14 / 1 / 11; branch into the instructions before a short exit
   14 / 3 / 8; branch to the start of a widened prologue 2 / 5 / 9; branch leaves the function
   7 / 4 / 4; exit does not pop the prologue's registers 3 / 0 / 1; rsp operand that cannot move
   2 / 1 / 3.
2. **Bundled modules are not framed.** Each run converts six PRX modules and prints one framing
   report, for the executable only.
3. **Only functions with `.eh_frame` bounds are examined.** FF's 179 MB executable reports 17,288
   functions; whether code without bounds exists there is not checked yet.

So protection-based tracking (sections A and B) needs relinker work first: frame the skipped
shapes (the two "shorter than a jump" classes and the push/pop class are about 70 % of them), frame
modules, and account for code without bounds. Until then phases 2 and 4 use the walk-based fallback.

## What this can and cannot deliver

- Phases 1 to 4 remove CPU work and waiting. They cannot make a title faster than its GPU time:
  LEGO stays near 39 ms until phase 5.
- 60 fps on LEGO needs its top program at roughly a third of today's cost. Nothing measured so far
  says that is reachable; phase 5 answers it.
- AoM and 007 are 18x and 33x away. Treat 30 fps there as the first milestone.
- If phase 1 fails for a title, that title gets the walk-based variant and a lower ceiling.

## Decisions (approved 2026-11-11 as recommended)

1. **Red zone.** Allow protection-based tracking on `rewrite/direct` for titles that pass the phase 1
   gate. This reverses the current rule; without it sections A and B fall back to walks.
2. **Benchmark.** One primary title and scene until phase 2 is done. Recommendation: LEGO for the
   GPU track, AoM for the coherence track, 007 only once it reaches a repeatable gameplay scene.
3. **Milestones.** Accept 30 fps as the first gate for AoM and 007.

## Not in this plan

- GPU-side descriptor reads, hash tables and miss rings (measured slower).
- Further tuning of the locked draw preparation (stage A, binding-plan caches) before phase 0 ranks
  it as a top cost.
- Visibility policies for clean ranges (measured flat).
- Larger caches.
