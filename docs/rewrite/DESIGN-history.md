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

### Section B, first build step: result and rule change (2026-11-12)

The step as written above made no copy on AoM. Its draws read two 64 MiB buffers, used in
alternate frames, and the GPU itself rewrites each one before the frame that reads it: the driver
marks all 1,025 blocks written once per frame, none by the CPU. A range like that is never clean
for several frames. Two thirds of the bindings also had that write still pending when they were
prepared.

Rule as built (approved by the user 2026-11-12, replaces "clean for several frames, drop on a
write"):

- A range of at least 64 KiB that draws bind in 3 frames gets a device-local copy, filled by a GPU
  copy from the import. No host snapshot and no compare.
- The copy serves every binding while no write is collected over the range.
- After a collected write the copy is filled again, at most once per frame. A second write in the
  same frame reads the import until the next frame.
- A binding prepared while a recorded write is pending may use the copy: the fill is recorded
  after that write on the same queue.
- Budget 2 GiB of copies, least recently used first out. `APS5_DIRECT_NO_RESIDENT=1` turns the
  copies off (A/B lever).

Same build, alternating runs, per frame:

| | Copies off | Copies on |
|---|---|---|
| AoM title | 215 to 217 ms | 174 to 181 ms |
| AoM title, GPU busy | 215 ms | 178 ms |
| AoM scene after the title (CPU-bound) | 147 to 151 ms | 147 to 155 ms |
| FF Resonance | 32.8 ms | 32.7 ms |
| LEGO | 48.0 ms | 48.5 ms (no range qualifies) |

AoM per program (ms per frame, off / on): draw 0x7f8 47.7 / 26.2, draw 0x7f6 44.6 / 29.8, draw
0x789 10.1 / 8.1, compute 0x3f7a97000 10.6 / 11.2 (dispatches do not use the copies yet). The
fill costs about 2.6 ms of GPU time per frame (one 64 MiB copy). AoM holds 128 MiB of copies, FF
17 MiB.

The target (the four programs near their mirror times of 3 to 5 ms) is not met. What the draws
still read from host memory, per frame on the AoM title: about 2,700 small uploads (1.4 KiB
each), about 400 host snapshots (15 KiB each), about 2,600 bindings that are not guest memory
(direct-path data such as table maps, 123 KiB each), and vertex and index data. Next within
section B, in this order: measure which of these the two programs spend their time on, then move
that class to device-local memory with one upload per change.

How this feeds section C: each copy's record is kept in the block states' terms. While no write
is collected since its generation the blocks are clean and the copy is current; after one the
import holds the newest content. The walk-based form never withholds a GPU write from guest
memory, so no block is GPU-owned yet and waits and CPU reads need nothing from the copy. The
record also keeps the serial of the batch whose completion makes the copy current. When phase 4
withholds the host write (GPU-owned), a wait on such a block blocks on that batch (section C,
third rule) and never polls.

### Section B, second step: mesh data read through the address table (2026-11-12)

Measured first (experiments only, saved as `probe-patches\xlocal-placement-and-xprobe.patch` and
`probe-patches\vertex-pointer-reads-ceiling.patch`):

- None of the classes listed above is the cost. With small uploads, snapshots, table maps, data
  buffers, vertex and index snapshots and the address range table all placed in VRAM, the two
  programs do not change.
- It is the vertex stage: with a one-pixel scissor the two programs keep about 90 % of their GPU
  time (369 and 155 us per draw against about 410 and 180).
- The shared vertex shader (code 0x3f849c900) reads guest memory one byte at a time through the
  address range table. The draws declare no ranges of their own, so every such read lands in a
  whole-heap host import (table ranges of 7.7 GiB and 4 GiB).
- Ceiling: five guessed windows around the index data (304 MiB) held in VRAM and entered in the
  table. Picture correct; per draw about 400 -> 116 to 236 us and about 180 -> 63 to 166 us, GPU
  time per 10 s 8.9 s -> 4.4 to 8.3 s depending on the title's camera scene. The windows covered
  the mesh data only partly.

Step (approved by the user 2026-11-12):

- The class is guest memory that read-only draws reach through the address table. The shader's
  addresses are not known to the CPU, so the copy set is derived from what is: the index range of
  each such draw. The fixed-size window of guest memory around an index range that such draws use
  in 3 frames gets a device-local copy, filled by a GPU copy from the import.
- The copies are entered in the address table of read-only draws when the table is built. A table
  is never rewritten while in use (that loses the device); a change of the copy set changes a
  generation, and a draw whose table is older builds a new one, as it does for the multisample
  guard.
- A table that names a copy keeps it alive and current: before a draw that uses the table, the
  64 KiB blocks with a collected write since the last fill are copied again on the same queue.
  One upload per change, proportional to the changed bytes.
- A window whose blocks keep changing (refilled in most frames) leaves the set; so does one that
  no draw used for some frames. The copies share the 2 GiB budget of the first step.
- Dispatches and draws that store through addresses keep the imports.

Gate: the two programs near their mirror times (4 to 5 ms per frame), picture unchanged, frame
time not worse; then one regression pair on FF and LEGO. If the windows around index data do not
cover the vertex data, the next measurement is which addresses the vertex stage still reads from
the import (unreadable-outside-copies table, fault addresses), not a larger guess.

Result (2026-11-12, kept):

- The vertex stage's buffer descriptors (SRT probe, `probe-patches\srt-pointer-probe.patch`) name
  only the two 64 MiB raw buffers that already have copies; the mesh addresses come from pointers
  loaded inside the shader, so the index range stays the CPU's only handle on them.
- As first built, 4 of 7 windows (0x3e4, 0x3ec, 0x3f8, 0x43e) were dropped as changing: a few
  blocks in them are written every frame, and the test counted frames refilled, not bytes. Changed:
  a window leaves the set when its block refills copy more than twice its size within 32 frames.
  Now 6 of 7 are held (192 MiB); the hot index windows refill 1 to 9 MiB per 32 frames. 0x3ec
  (inside the 64 MiB raw buffer area) still churns.
- The per-draw check ran again whenever the write tracker moved (196k checks in 130 s) and cost
  about 10 ms of frame time in the CPU-bound scene. Changed to the approved rule: one check per
  copy per frame (a mapping change still forces one).
- AoM (no profile flags, head against change): title frame p50 167-178 -> 146-158 ms, GPU per
  present 178-186 -> 146-151 ms; scene after the pad press frame 153-161 -> 148-158 ms, GPU
  105-107 -> 81-97 ms. Picture unchanged. Per draw the two programs 408 -> 225 us and 190 -> 133 us;
  per frame about 26 and 30 -> 13 and 12 ms. FF 32.9 / 33.0 ms, LEGO 49.1 / 48.6 ms (level).
- Gate not met yet: the two programs are at 12 to 13 ms, not 4 to 5. Next, per the gate: measure
  which addresses the vertex stage still reads from the import.

Measured (2026-11-13, probe `probe-patches\outside-copy-reads-probe.patch`, `APS5_X_PROBE_OUTSIDE=1`,
run `aom\xoutside2`): read-only draws' tables mark every range that is not a copy; a read lookup that
lands in one notes its 4 KiB page in the fault buffer's written-page record (unused by read-only
draws), and the CPU sums the pages per 1 MiB. Full coverage per draw, picture unchanged. Over 320k
read-only draw uses (title and the scene after the pad press), 85 % still read the import:

| Guest range | Uses touching it | Pages per use | In a window? |
|---|---|---|---|
| 0x3e1000000 (one page) | 136k (43 %) | 1 | no (no index data near) |
| 0x3e1200000 (one page) | 137k (43 %) | 1 | no |
| 0x3ed500000..0x3edd00000 (8 MiB) | 47k (15 %) | 1.3 | window 0x3ec, barred as churning |
| 0x3f7b00000 (1 MiB) | 5k | 1 | no |
| 0x43a..0x442 (held windows) | under 50 each | | edges just outside the copies |

- The mesh data the two programs read outside the copies is the 8 MiB at 0x3ed5..0x3edd, inside
  the one window the byte rule bars (0x3ec, used in 553 frames, refills over 2 MiB per frame).
- Two single pages at 0x3e1000000 and 0x3e1200000 are read by almost half of all read-only draws,
  far from any index data; the index-range rule cannot reach them. Whether they change per frame is
  not measured.
- Not attributed per program (the probe sums all read-only draws).
- Options put to the user: (a) hold the 0x3ec window despite its churn (refill only the changed
  blocks), (b) admit the two hot pages. User: "whichever you deem best"; (a) tried first.
- (a) result, not kept (`failed-patches\resident-window-no-churn-bar.patch`, runs `aom\nc-head1`
  against `aom\nc-on1`, both with the GPU profile): 0x3ec held through the title (about 3 MiB
  refilled per frame), but title GPU per present 150-158 -> 149-155 ms, frame p50 160-163 ->
  162-169 ms, the two programs 9.3 / 7.8 -> 8.1 / 7.6 ms per frame in the later scene (noise
  level). Without the bar, window 0x424 refills 17 MiB per frame. The 8 MiB outside reads are not
  the remaining cost.
- (b) not tried: two pages read once per draw are cache-resident on the GPU after the first read;
  they cannot explain 8 ms per frame.
- Suspects tested next (2026-11-13, all against `aom\nc-head1`, same profile flags; none kept):
  - Table search: the read-only tables hold 13 ranges, not about 1,200 (an old code comment).
    Cut to the 2 large imports plus the copies (`probe-patches\small-address-table-probe.patch`,
    run `aom\xsmall1`): no faults, no change (per draw 146 / 242 us against 154 / 238).
  - The two hot pages, (b) after all: a window forced over 0x3e0000000 is barred as churning
    (the pages change every frame); held with the bar off
    (`probe-patches\force-window-no-churn-probe.patch`, run `aom\xforce2`, about 2 MiB refilled per
    frame, 0x3ec held too): title GPU 144-155 ms, per draw 154 / 260 us. No change.
  - Index fetch: every read-only draw already binds a copied index buffer (140k copied, 0 read in
    place; `probe-patches\index-from-window-copy-probe.patch`, runs `aom\xindex1`, `aom\xindex2`).
- The mirror ceiling still reproduces on head (`APS5_HOST_IMPORT_MIB=64`, run `aom\mirror-now1`):
  0x7f8 35 us and 0x7f6 74 us per draw, against about 150 and 250 us now. But in that mode the GPU
  runs about 4 frames per 10 s, with the draws separated by seconds of staging copies, and GPU busy
  per present is 270 ms against 150. Per-draw timestamps of isolated draws and of draws that
  overlap in a full pass may not be comparable, so the 4-5 ms target may not be a like-for-like
  ceiling.
- Every host-memory class of the read-only draws is now ruled out by measurement (descriptor-bound
  ranges, table ranges, outside pages, table size, indices). Next: a GPU-stage question for an
  Nsight GPU Trace (`ngfx-mcp`) of one title frame on head: what stalls the vertex stage of code
  0x3f849c900 (memory throughput vs latency vs instruction issue), and whether the program's time
  in a full pass matches its time in isolation. If it is instruction-bound (byte reads and
  per-byte lookups), the work is section E translation, not B.

**GPU Trace result (2026-11-13, head 9817f28fbe, title frame, 147 ms;
`traces\game_2026_11_13_01_07_27.ngfx-gputrace`).** `ngfx-mcp.exe` has to run elevated (not
elevated: `NVPA_STATUS_INTERNAL_ERROR`). The trace call reports "No LOP collected data chunks
intersected the region of interest" but still writes the file.

- Shader samples: fragment e78003fe 17 %, compute 69cf7464 13 % (Wait / Selected: issue-bound),
  then the vertex shader 11b452b5 under two shader indices, 11.5 % + 3.8 %. That is the shared
  vertex code 0x3f849c900, one index per program.
- The vertex stage is memory-latency-bound, not instruction-bound: almost all samples are Long
  Scoreboard, with some LG Throttle.
- Hot program (index 111), using SPIR-V lines from `runs\aom\spv\draw_3f849c900.spv`:
  - About 30 % of its samples are in `probe_bda_pointer`'s binary search. Line 842 is the compare
    that waits on the range-begin loads from `bda_pagetable`; line 791 is the header load.
    `probe_bda_pointer` is reached from `read_bda_span` / `read_bda_span_stop`, which are called
    13 times per invocation (the vertex fetches), so every fetch pays a chain of dependent
    loads: header, then about 4 search steps, then the data.
  - About 25 % is in the joins after the vertex fetches (lines 1787, 1425, 1820), i.e. waiting
    on the vertex data itself.
  - The other program (index 101) is mostly `get_bda_pointer` and the main entry.
- This conflicts with the table-size probe, where cutting the table to 2 ranges changed nothing
  measurable. That probe still leaves the header load and one search step, and other warps may
  hide that latency. So the stall attribution alone does not prove that removing the search
  saves time.
- Timeline: draws sum to 49 ms (566 draws; one draw is 12.5 ms and one is 4.8 ms) and dispatches
  to 35 ms (133), against a 147 ms frame. Action durations carry no shader id, so isolated vs
  in-pass time per program is not answerable from this reader.
- Next: two candidates; take the cheaper one first.
  - Translation (E), principle 2: the per-fetch search is the shader chasing guest pointers.
    Resolve each fetch base's table range when the binding plan is built, and keep it in the
    section D memo, valid until a dirty event or a table change. Do not re-resolve or check it per
    draw (principle 1). The shader uses the stored range, keeps the bounds test, and falls back to
    `probe_bda_pointer` when an address is outside it (principle 6). Probe: bench the AoM title.
  - Timeline (E): find the 12.5 ms draw and the 8 ms dispatch, and the roughly 60 ms of the
    frame outside draws and dispatches.

### Section D fix: heap-slot images in the pass hazard check (2026-11-13)

Reported on PPSA34501 (Black Ops 1): with the direct path the main menu is a pale field with dark
specks that flickers; `APS5_NO_DIRECT=1` is clean. Bisected to direct draws, then pixel shaders,
then one program (code 0x400088b700, one fullscreen draw per frame) sampling the scene's
R16G16B16A16 HDR target.

- Cause: a direct stage binds its images in heap slots, outside its ShaderResources, so the draw's
  pass access (`StorageImages`, `ReadsImage`) never named them. The composite read the HDR target
  inside the open pass with no barrier after the draws that wrote it.
- Fix: `ResolveDirectStages` notes, per draw, the GPU-written storage images it binds (sampled
  storage sources as reads, storage images with their write flag, table-mirror elements included);
  `recordDraw` adds them to the pass images and the reads-its-target test. Hazard information
  only (principle 5); no per-draw change check.
- Result: BO1 menu clean in 4 runs (1.5 % near-white pixels against 97 % before). Regression pairs,
  head against fix: FF 33.1 / 30.6 against 32.8 / 32.3 ms, LEGO 46.7 / 46.9 against 47.6 / 47.3 ms
  (inside the noise floor; any LEGO cost would be the passes this hazard now breaks).
- Not kept (saved as a patch outside the tree): sampler compare,
  unnormalized and integer-border flags and the base-level view for unnormalized images in direct
  requests. They match the classic path but changed nothing on BO1. A per-use `RefreshDepthCopy`
  was also tried and dropped: no effect, and it is a per-draw change check (principle 1).
- Seen, not examined: `APS5_DIRECT_NO_FOLD_SRT=1` alone breaks BO1 (28 draws per 10 s,
  `BDA access failed`).

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

## Per-phase status detail to 2026-11-13 (moved from DESIGN.md when it was condensed)

## Current status (2026-11-13, head 9817f28fbe)

| Title | Frame | GPU busy | Bound by | Track |
|---|---|---|---|---|
| LEGO | 47 to 49 ms | 39 ms | GPU: one compute program, 22.8 ms | 5, translation |
| FF Resonance | 30.6 to 33 ms | about 20 ms | CPU: worker 0 at 97 % (phase 0 profile only) | 1 to 4 |
| AoM title | 146 to 158 ms | 146 to 151 ms | GPU | 5, translation and timeline |
| AoM scene after the pad press | 148 to 158 ms | 81 to 97 ms | CPU | 6 |
| 007 title screen | 127 to 133 ms | about 70 ms | CPU: the draw committer (below) | 4 |

Per phase:

- **0:** done for LEGO, FF and AoM.
  - **007** was added on 2026-11-13 with the title screen as its scene (run `007\p0-head`; worker
    samples in `runs\007\stacks-p0-head-samples.txt`).
  - Per frame, summed over the five workers: packets 2,469 ms, of which the `memory_wait` zone is
    1,732 ms, dispatch 426 ms and draw 284 ms.
  - The worker samples show 57 % self time in one unresolved ntdll wait (`ntdll+0x1628a4`). What
    calls it is not resolved: `DrawPipeline::Enqueue` (36.5 % inclusive) is a likely caller, but
    only 1.5 % of samples sit under `waitMemory`, where `PollSleep` is 58 %.
  - Real work in the samples:
    - storage textures created and uploaded per use (`cachedStorageTexture` 19 %;
      `StorageTexture::Create` 10 %, `::upload` 11 %)
    - `ResolveImageTables` 8 % self
    - `materializeDrawStage` 15 %
    - `VulkanDevice::Dispatch` 22 % (in the NVIDIA driver)
  - Over 300 s, the log shows:
    - 326k buffer uploads, with 100 s of guest reads
    - 37.5 GiB copied out of imports and 37.5 GiB written back (eager write-back)
    - 26.7k dispatch-cache evictions against 43k hits
  - **Cause of the blocked time:**
    - 69 % of the ntdll wait is `DrawPipeline::enqueue` waiting for room and 12 % is `Drain`.
      The five workers are back-pressured by the single committer thread.
    - The committer runs 14,940 commits per 10 s at 453 us each, 6.8 s busy. Workers wait 292 us
      per draw. Draw resources are 4.4 s per 10 s, of which uploads are 4.0 s.
  - **Storage images**, per 10 s:
    - About 810 created (39 per frame, about 5.6 MiB each) against about 11,000 reused.
    - "first" uploads 4.5 GiB, and write-backs at storage image creation 3.6 GiB.
    - 837 LRU evictions, none of which flush anything.
    - The cache sits at its budget: 7.4 GiB of a 14.9 GiB device budget, 12.4 GiB in use.
  - **Reading:** per frame, new storage images over guest memory evict dead ones, and creating
    them writes back the images they overlap. That is eager write-back (principle 3, section B /
    GPU-owned), not a budget problem: there is no VRAM headroom to raise it.
  - **Storage creation probe** (`probe-patches\storage-create-probe.patch`, run `007\xstorage1`,
    150 s):
    - 10,937 creations (53.8 GiB) over 3,501 distinct addresses:
      - 48 % reuse a known address with another shape;
      - 20 % reuse a known address with the same shape (an entry recreated after being dropped);
      - 32 % are first uses of an address.
    - The game aliases a transient pool: one 2 MiB address is created as 44 shapes and formats.
    - Its dynamic resolution walks 1280x720 to 1920x1080 in 64-pixel steps, so every frame brings
      new sizes.
    - 76 % of creations are of written images and 58 % flush an overlapping one.
    - Each creation costs about 1.4 ms on the committer (image, memory and first upload).
    - So the cost is one storage image per (address, shape), created and filled through guest
      memory each time the transient pool is re-used.
  - Storage images were not the limit: they are built on the workers, which are mostly blocked.
    Sharing device memory per range (approved option 1) is parked until it is.
  - **Committer** (`APS5_SAMPLE_COMMITTER`, run `007\committer1`): 39 % idle, 51 % draw commits.
    `UploadFinish` took 29 % of its samples, as CPU copies of guest buffers (`copyRegion`).
  - **Copy probe** (`probe-patches\copy-region-probe.patch`): every copy was a binding with no
    import. 92 % of the bytes were one 12.5 MiB binding (0x3f00094200), about 920 copies per
    second. It starts inside a 2 MiB registered range and runs into the adjacent 12 MiB one;
    `joinedRun` joined such runs only for a binding from the run's base.
  - **Fixed in 375d360e24:** a binding from inside the first range also joins when that range is
    imported on its own. 007 title went from 387-426 to 127-133 ms per frame; picture correct;
    FF and LEGO level. `agc_driver_graphics` fails the "read-only V# across two imports" check on
    head too (pre-existing).
  - **Next for 007:** re-profile the committer and workers on 375d360e24. The menus are now about
    8 fps; move to gameplay once they are reasonable.
  - **First cutscene, cold cache (manual run, 2026-11-13):** rendering stops for minutes at a time
    while queue worker 0 is inside the NVIDIA compiler (`vkCreate*Pipelines`). The other workers
    wait in `PollSleep`; game audio continues.
    - The cutscene's shaders are very large: 26 cache entries over 2 MiB, eight of them 7 to
      9 MiB. One is 2.29 M SPIR-V words (476k instructions, 16 % `OpSelect`) from about 98 KiB of
      guest code.
    - 0x30032da200 falls back to two lanes per invocation (a barrier under per-lane control flow).
    - Compiled pipelines are cached (our cache plus NVIDIA's), so each run gets further.
    - Open: translation size (section E). The levers measured on LEGO are cheaper NaN checks (-5 %
      words), `APS5_FAST_F32` (-22 %) and the spanned wave (-39 %).
  - **Translation size, measured (2026-11-13, bc7ef96c09 and cb4485926e):** 158 compute requests
    dumped to the title (`APS5_DUMP_SHADERS=1`, run `007\dump1`), replayed offline. All 29 large
    programs fell back from spanned to paired: the spanned uniformity check counts every guest-memory
    load as per-invocation once the module stores guest memory, and the deciding loads are guest
    scalar loads (s_buffer_load, s_load, SRT reads), which are one value per wave in the guest's
    scalar cache unless the program invalidates it.
    - Change (`probe-patches\spanned-scalar-loads.patch`): the emitter records the id ranges of
      scalar-load emission; the check exempts loads in them unless the program invalidates caches.
      0 fallbacks; 21.3 M -> 13.9 M words. NVIDIA compile of the largest (0x3500c6aa00), timed
      offline: 92.9 s -> 42.7 s; 0x3000b8fc00 13.5 -> 6.3 s.
    - Change (`probe-patches\cheaper-nan-check.patch`): `FPIsNan32` instead of And + UGreaterThan in
      the f32 NaN propagation, exact, -3.4 % words.
    - Cold start (empty NVIDIA and our caches, 300 s; runs `007\cold-head`, `007\cold-span`): head
      still black at 250 s (236 presents); change past the stall at about 150 s, boot notice shown.
    - Warm, steady (runs `007\warm-span1/2`, `007\warm-head2`): 121-129 ms with the change vs
      126-141 ms on head; no runtime cost seen. Tests: only `agc_driver_graphics` and
      `agc_driver_frame_replay_rewind_copy` fail, both on head too. LEGO 45.9-47.1 vs 47.1-47.6 ms,
      FF 31.5-31.7 vs 32.3 ms (runs `<title>\span-on1/2`, `span-head1/2`), pictures correct.
    - Next: the largest program still takes 43 s; `APS5_FAST_F32` (-18 % here) is not exact.
- **1:** first audit pass fails on all three titles. Skipped functions: LEGO 158, FF 39, AoM 147.
  The bundled PRX modules are not framed, and code without `.eh_frame` bounds is not checked. The
  two "shorter than a jump" shapes and the push/pop shape are about 70 % of the skipped functions.
  - **RIP-relative widening (2026-11-13, 58acf93710, `probe-patches\redzone-rip-relative-widening.patch`):**
    prologue, exit and operand stubs may now cover RIP-relative instructions, with a relocation
    per copied displacement. Skipped: LEGO 158 -> 77 (short exits 87 -> 17, short prologues
    14 -> 2), AoM 147 -> 136, FF 39 unchanged. LEGO converted with `--frame-red-zone` runs the
    bench scene normally (46.6-48.1 vs 46.8 ms unframed, same picture; runs `lego\rz-plain1`,
    `lego\rz-framed1/2`); the game folder was restored to its previous conversion afterwards
    (backup `lego-convert-backup`). Tests: red-zone unit and execution tests pass (new execution
    case: a short exit widened over a RIP-relative lea, checked under the clobber).
  - **Skipped leaves without stores (measured, not used):** a leaf that stores only through rsp
    cannot take a write-protection fault. FF 13 of 39, LEGO 28 of 158 (before the change above).
  - **Return thunks (96f2e8d266):** a leaf without pushes, allocation or frame pointer whose exits
    cannot be patched gets an entry stub that lowers rsp and calls the rest of itself, so each ret
    lands on `lea rsp, [rsp+shift]; ret`; its pushes and pops must balance on every path. Stubs may
    also carry a jcc. Skipped: AoM 136 -> 66 (66 thunks), LEGO 77 -> 52 (23), FF 39 -> 31 (7).
    LEGO framed runs normally (46.8-48.1 ms, runs `lego\rz-thunk1/2`), folder restored after.
  - **Tail-jump entries and AH (1d54fb40d5):** only a branch from inside the leaf to its start
    blocks a widened prologue; `ah` (register 4 without REX) is no longer read as rsp. Skipped:
    AoM 66 -> 44, LEGO 52 -> 42, FF 31 -> 18.
  - **Thunk gap fix (110ddcf025):** a leaf with pushes after an early-out ret went through a thunk
    whose gap did not cover those pushes, so its spills overwrote its saved registers (framed AoM
    crashed at start-up in crc32 at 0x39f1900). The gap now covers the pushes before each red-zone
    operand; past 128 bytes the leaf is skipped ("too deep for a thunk": AoM 5, 007 5).
    Framed runs: AoM title screen correct, p50 158 ms vs 163-167 plain (`aom\rz-aom-framed2`,
    `rz-aom-plain3/4`); 007 p50 125 ms, no crash (`007\rz-framed1`). Folders restored after.
  - **rsp arithmetic in the body (b4eec7ba80):** frame-pointer leaves whose `[rbp-x]` slots lie in
    a `sub rsp` allocation freed only in epilogues are not red-zone users (007: 31). Leaves without
    a frame pointer that move rsp outside their prologue allocation and frees return through a
    thunk whose walk tracks pushed slots and allocations; the gap holds the whole frame (up to
    2 KiB, disp32 lea). Crash checks (40 s, framed, folders restored): AoM, 007, LEGO, FF clean.
  - **Tail-call exits (bb0218c541):** a jmp to another function's start (or outside every
    function), or an indirect jmp right after the prologue's pops, is an exit; its stub relocates
    the jump. Indirect jumps without pushes or pops first (jump tables) stay refused. Crash checks
    (40 s) clean on all four titles.
  - **Jump tables (a3fe6be7cc, 7e21d150bd):** `lea; movsxd [base+i*4]; add; jmp reg` tables are
    read (from any load segment) until an entry misses an instruction of the function; a lea
    hoisted out of the window takes every table the function's RIP-relative leas address. Cases
    become branch targets and walk successors. Function-pointer jumps (`jmp [reg+disp]`,
    `mov reg,[..]; jmp reg`) are tail calls. Entries or branches past the first int3 refuse.
  - **Thunk retry (59d1573627) and thunk tail calls (68a1d63afa):** a leaf without a frame pointer
    whose exits cannot be patched is retried as a thunk with its pushes in the body; a thunk's
    tail call leaves through a stub raising rsp over the gap and slot (callee sees the caller's
    return address and stack arguments).
  - Crash checks (40 s, all four titles) clean after each step.
  - **Whole-leaf relocation (98cf5da8d9): skipped 0 on all four titles** (from 132 / 147 / 158 /
    39 at the start of phase 1). A leaf that neither in-place framing nor a thunk can take is
    copied into a stub that runs it with rsp 0x80 lower: rsp operands +0x80, pushes and pops
    rewritten as mov + lea so every stack slot keeps its address, every exit (ret, tail call,
    branch out) raises rsp first. The start jumps to the stub; jump-table cases get a join site
    each (none needed in the four titles). Absolute tables are read from RELATIVE addends; a table
    of other functions' starts (007's threaded interpreter handlers) is a tail call. A function
    entered from outside its start is relocated instead of patched (none in the four titles).
    Relocated: 007 8, AoM 7, LEGO 2, FF 1; 007's last leaf framed in place once its dispatch
    window skips VEX vector instructions. Crash checks (40 s) clean; folders restored.
- **2 and 3:** not started.
- **4:** section B is built in its walk-based form (bound ranges and index windows). It took the
  AoM title from 216 to about 150 ms. The section D validity change is not started.
- **5:**
  - The timeline item for indirect draws in the pass is done (AoM 745 to 72 passes per frame).
  - **AoM translation**, from a GPU Trace of one title frame
    (`traces\game_2026_11_13_01_07_27.ngfx-gputrace`):
    - The vertex code 0x3f849c900 shared by the two hot programs is memory-latency bound.
    - About 30 % of the hot program's samples are the per-fetch `probe_bda_pointer` binary search
      (13 fetches per invocation), and about 25 % wait on the vertex data.
    - Every host-memory class has been ruled out by probes.
  - **AoM timeline:** draws sum to 49 ms and dispatches to 35 ms of the 147 ms frame. One draw takes
    12.5 ms and one dispatch takes 8 ms.
  - **LEGO translation:** the program is latency-bound with no single emulation cost to remove.
    Open: restructure the wave64 mapping, or accept that the guest workload at 3840x2160 is the
    cost.
- **6:** per-draw record cost is 12 to 25 us. On the AoM worker, stage preparation brings this to
  about 156 us per draw.


## Phase 1 and 2 status detail to 2026-11-15 (moved from DESIGN.md when its rows were condensed)

| Phase | State |
|---|---|
| 1 | Eboot framing done: every red-zone leaf of the four eboots is framed, 0 skipped (from 132 / 147 / 158 / 39; 18 relocated whole, 98cf5da8d9). Framed crash checks (40 s) clean on all four. Bundled PRX modules framed too (5 / 11 / 16 / 1 modules for FF / AoM / LEGO / 007, 0 skipped, every module has `.eh_frame` bounds; the one skip was libc's longjmp `mov rsp, [mem]`, now loaded and lowered by the shift); framed crash checks clean. Code without `.eh_frame` bounds is framed too: FF's FDEs cover only 6.5 of 107.6 MB of its eboot's code, so code outside every FDE is split into functions at int3 padding and direct call targets (nops neither start nor end one) and framed like the rest (FF eboot 289,831 such functions, 411 more red-zone leaves; LEGO 550, AoM 1,337, 007 563 plus modules). New relocated shapes: `pop rsp` and `mov [mem], rsp`, and `mov rsp, [mem]` now goes through a pushed scratch so rsp never rises above the lowered rsp. 0 skipped in every image; crash checks clean (FF only reaches its title menu in 40 s). Fault storm (6317f55d61, `APS5_FAULT_STORM`): a storm thread single-steps bursts of 256 instructions on threads in guest code and overwrites the 128 bytes below rsp at every step (on this 9800X3D a real frame reaches the red zone only while AVX-512 state is in use, measured). Gate, 2000 landings/s: unframed, all four titles crash within 25 s with the pattern in registers (FF in zlib inflate_fast); framed, FF, LEGO and 007 run 30 min clean (24 M / 63 M / 674 M steps); framed AoM aborts in libc about 30 s in (3 of 3; not without the clobber, and also with the clobber limited to guest code), so a red-zone use in AoM's guest code is still unframed. Found by bisecting the clobber with `APS5_FAULT_STORM_CLOBBER_RANGE`: libc's bounds-checked copies (ELF 0x6d360, 0x6d270 and two more) are frame-pointer functions that call, yet keep `[rbp-8]` below rsp before their first call to compare against their own frame, and the framer framed leaves only. Now a frame-pointer function with an access below its frame built so far (pushes and `sub rsp` before the first call; `lea`, indexed operands and `and rsp` realignment excluded) is framed too; one that pushes only rbp leaves through `leave` (its `pop rbp` rewritten), so no exit stub. Libc's four copies are framed in every title. Splitting uncovered code now also starts a function after a ret, jmp or ud2 at code no earlier branch reaches (FF's shrink-wrapped functions had merged with the one before), and a leaf that calls a helper inside its own bounds (inline-asm AES key expansion in 007) stays a leaf. 0 skipped in every image; framed AoM with the storm: 90 s clean (it aborted at 30 s before). A 30 min gate run then aborted at 256 s inside our native AJM code (an invalid job size in a batch from libfmod): the storm's clobber also ran at steps a burst took into native code, and our `sysv_abi` exports may keep values in the red zone (MinGW GCC uses it for System V functions), the same hazard a real fault there would hit. With the clobber limited to guest code, framed AoM ran 10 min clean. Gate rule changed with the user's approval (2026-11-13): the storm overwrites the red zone only at steps in guest code, and native modules are built with `-mno-red-zone` (top-level CMakeLists, MinGW). Gate result under that rule (`logs\storm-gate3-20261113.out`, libs `libs\storm3`): unframed AoM crashed at 20 s with an access violation and `a5a5…` in rax (the fill pattern); framed, all four titles ran the 30 min storm clean (AoM 27.5 M steps, FF 24.2 M, LEGO 59.1 M, 007 680 M; 179 intervals each, no FATAL). Phase 1 gate passed for all four titles. |
| 2 | **Kept and committed (aba95ed26b, 2026-11-14) for FF, AoM and 007.** **007 fix (6bccb2bfa7):** 007's written shader-storage regions shift base every frame, which made hundreds of overlapping 60 to 100 MiB store imports; the device was lost at a submission and `vkDeviceWaitIdle` then hung for about 50 s. Store imports of 1 MiB or more now snap to a 16 MiB grid and a same-base import that already covers the range is reused. 7 consecutive 200 s on-runs clean (about 3 of 12 failed before); on about 191.6 ms against 187 off (the "11 % slower" was probe cost). **Rebase onto upstream 3c310597b6 (2026-11-14, 4,538 upstream commits):** upstream replaced the thread collect memo with `GuestMemory::observe()` (observation ticks shared across threads) and added ordered waits (`bumpWaitEpoch`). Section A is kept inside it: `observe()` skips ranges the arena reports clean and `BumpCollectEpoch` keeps the skip, but the unclean-walk counter is global again because observations are shared, so almost no bump is skipped now. Checked on the rebased tree (libs `libs\merge1`, framed, same binary on/off): phase 1 on all four (0 leaves skipped, marked framed, 2 min storm, no fatal). FF walks 3.00 M to 1.39 M, collect 6.2 to 4.2 s, epochs 40.4 k to 37.1 k (2.6 k skipped), frame 35.4 to 36.1 ms. AoM walks 1.44 M to 0.20 M, collect 2.6 to 0.9 s, frame 240 to 245 ms. 007 walks 5.21 M to 1.59 M, collect 11.4 to 5.4 s, frame 226 to 221 ms, no device loss in two 300 s on-runs. LEGO 49.2 ms. Pictures correct on all four. **Slowdown after the rebase, found and fixed (2026-11-15, libs `libs\merge4`):** AoM was at 240 to 247 ms (157 to 159 before) and 007 at 221 to 226 ms (187 to 191 before), on and off alike. AoM: upstream befe8dc17b no longer credits the imports a larger import replaces while a recorder is active, so the 4 GiB heap never fitted the import budget, stayed in 64 MiB pieces, and every draw binding across two pieces took an address-based build ("not space-only" 21.8 k per 10 s, no resident copies). Now the replaced imports are credited when the host heap's own report has room for the run beside them (upstream's rule unchanged without a report, so its tests pass): AoM 162 to 163 ms, reuse back to 30 to 34 k per 10 s. 007: the binding join had been narrowed in the merge for an upstream test, which sent 77 GiB per 300 s through CPU copies (guest read 13.8 to 41.5 s); the join now applies when the run's first range is small (007's shape) and upstream's rule covers a large first range: guest read 7 to 11 s, copies of the 0x426 region 9,462 to 3. 007 after its scene change (about 103 s) is now bimodal on one build, about 205 or about 221 ms with GPU 88 to 94 or 100 to 105 ms (185 to 190 ms and 72 to 80 ms before the rebase); before the scene change it is faster than before the rebase (160 to 175 against 178 to 200 ms). FF 35.5, LEGO 49.1 ms. **Still open on 007:** that GPU time after the scene change; CPU counters match the old build. Not the cause: upstream's staging shadow cap (`APS5_STAGING_SHADOW_MAX_MIB=4096` no change) and its CPU-side indirect reads (#18670; off is slower on 007 and AoM). Next step there: per-program GPU times (`APS5_PROFILE_GPU=1` with `APS5_PROFILE_GPU_DRAWS=1`) against the pre-rebase build with its own cache folder. The epoch skip lost with the shared observations goes with the FF target in phase 4. Test subset on the rebased tree with both fixes: 1,790 of 1,790 on and off (18 skipped by their own conditions); upstream's new `agc_driver_depth_expand_out_of_place_inline` failed about 2 runs in 9 earlier and passes alone. The stall watchdog (`APS5_STALL_DUMP_SECS=15`) is only valid for a warm 007: FF goes 15 s without a submission while loading and a cold NVIDIA pipeline compile on the draw committer can take longer. As built (third build): collected memory is armed per 64 KiB block; GPU stores go through small dedicated store imports that are made uniform at their storing submission, so heap imports stay partly armed; submissions name their stores; collects skip clean ranges; an epoch bump is skipped while the bumping thread walked nothing unclean and no armed memory was opened. Gate pairs (framed, same binary on/off): FF walks 1.50 M to 0.34 M, collect 2.9 to 1.1 s, real epoch bumps 25.4 k to 18.6 k, frame 36.3 to 36.6 ms, pictures match, 9.9 GiB combined. AoM walks 3.3 M to 0.07 M, collect 5.6 to 1.1 s, epoch bumps 2.0 k to 1.7 k, frame 171-176 to 157-159 ms; both sides render correctly but the title camera gives no matched shot; 25 GiB combined on and off alike (not from phase 2). **Gate judgement (user, 2026-11-14):** the coherence share is more than halved on both; FF walks and epoch bumps are not near zero because the walks left consume real writes (16.7 k of 18.6 k real bumps follow the thread's own walk of written memory) and memos are still valid per epoch. That target moves to phase 4 with the section D validity change. Rejected on the way: hot blocks answered "written" without a walk (FF 36.7 to 52.5 ms and missing particles, `failed-patches\phase2-hot-no-walk.patch`): a hot answer may only cover pages the write watch reports. **007 (framed, 300 s, `runs\007\gate2-007-*`):** off runs clean at 187 ms; on, one of two runs stopped submitting at about 120 s (waits of 100 s on three queues), then lost the device (`vkQueueSubmit` -4); the other ran 300 s at 208 ms with probes on (walks 6.8 M to 1.4 M, collect 13.8 to 6.4 s). 602 stores still went through ordinary imports and 98 import groups were opened. Open: find the lost submission or deadlock before 007 runs with section A. **First build (2026-11-13), gate not met, reverted, kept as `failed-patches\phase2-section-a.patch` (libs `libs\protect6`).** Built: (1) libc `WindowsMappings`: a resetting collect arms consumed private and resident memory read-only (CPU-clean); a store faults once and opens its 64 KiB block (CPU-dirty); blocks that fault repeatedly stay dirty (strikes); host writes, pins and guest mprotect open armed memory; armed pages report read-write to the guest and the driver. (2) Blocks the GPU writes stay CPU-dirty: the recorder's pending-write notes and storage targets open every host import they land in, for good (also after the guest range is released and reused, as the import can outlive it). (3) The dirty-event answer replaces walks: `collectWrites` skips any range libc reports clean, with no walk and no epoch. Counters on `[guestmem]`; unit test in `windows_write_watch`; `guest_memory`'s resident test states the new rule. ctest subset: no new failures. **Hardware finding (RTX 5080, driver 617.42; probe `probe-patches\import_protect_probe.cpp`):** a GPU write into a host import with some pages read-only and others writable silently drops the whole submission in about 1 of 12 runs (5/60 shader stores, 5/60 transfer fills, 5/60 even when the stores avoid the read-only pages); reads of such an import (0/60), writes into a uniformly read-only import (0/120) and imports reopened before the write (0/60) are unaffected. **Gate result (framed, profiled, same binary on/off):** FF: 346 imports GPU-written, 69 MiB armed, walks 3.02 M to 2.50 M, collect 5.4 to 5.8 s, epochs about 39-40 k both, frame 35.9 to 36.6 ms. AoM: its 13 GPU-written whole-heap imports cover everything collected, 0 MiB armed, walks 3.81 M to 3.47 M, epochs 2.2 k both, frame 166.7 to 168.7 ms. Walks and epoch bumps are not near zero and the coherence share does not fall: the guest heaps the collects walk are host imports the GPU also writes. Arming them anyway (probe) gave FF 540 MiB armed and collect 3.9 to 1.6 s but drops GPU work. **Follow-up probe (same program, 64 fills in flight):** only the protection at submission matters. Mixed at submission: 6/60 lost. Uniform read-only, or mixed and re-armed to uniform before submission: 0/60 (fills and shader stores). Pieces opened or armed while the submission runs: 1 lost in 420. **Route to the gate within section A:** the as-built version armed after a collect and kept GPU-written imports open; the table says a submission consumes the dirty list and re-arms. Built that way, every import a submission can write is uniform at submission: its dirty blocks are re-armed (CPU-clean) as the submission consumes them, so whole heaps can be armed, GPU-written or not. A block that faults every frame then costs one fault per frame instead of staying open; where that costs more than the walk, the whole import stays CPU-dirty (uniform read-write). With every registered block clean or listed, the dirty list also answers the collect memo, so epoch bumps are skipped while the list is empty. Expected from the unsafe probe: FF collect 3.9 to 1.6 s. **Second build (2026-11-14, in progress, uncommitted):** re-arm at submission is built (every `vkQueueSubmit` runs inside an arena bracket that first makes each import group uniform). Only guest allocations are armed; an import group holding other memory stays open. **Second hardware finding (same probe, `PROBE_OVERLAP`, `PROBE_PRIOR`, `PROBE_LATE`):** when two imports cover the same pages, a write through an import whose first GPU use happens with its pages read-only is dropped (40 of 40; a later-made import over an already used one 29 of 30; in the engine the device was lost). An import used once while writable, even by a 4-byte read, is unaffected (0 of 30). So every import the arena may arm gets a 4-byte read at creation, inside its host-access bracket. The collect skip for clean ranges now records its observation in the collect memo like a walk that found nothing. Open: the test subset on and off, the FF and AoM pairs against the gate, and the epoch-skip item. |
