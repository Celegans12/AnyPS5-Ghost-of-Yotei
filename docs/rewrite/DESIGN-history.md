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
