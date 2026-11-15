# Direct path: design, draft 3

Branch `rewrite/direct`, approved 2026-11-11 with the decisions below. This file holds the design
and the latest progress only; dated measurements, probes and earlier status are in
`DESIGN-history.md`.

Target: 16.7 ms per frame, which needs the CPU worker at or under about 10 ms and the GPU at or
under about 14 ms per frame.

Three limits remain:

1. **Coherence.** The CPU re-establishes what guest memory holds at every ordering point
   (write-watch walks, collect epochs, memo revalidation, uploads of unchanged data). The cost
   scales with how often memory is used, not with how often it changes.
2. **Waiting.** Workers poll for labels and for each other while the GPU is idle.
3. **GPU time.** Translated shaders, pass breaks, barriers and copies.

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
7. Footprint target is 16 GiB combined. Raising a limit is fine; total usage is the constraint.

## Architecture

### A. Event-driven memory ownership

Each 64 KiB block of GPU-registered guest memory is in one of three states:

| State | Protection | Meaning | Leaves the state when |
|---|---|---|---|
| CPU-clean | read-only | The GPU-side copy and all memos of it are current | a guest write faults: the block becomes CPU-dirty and joins the dirty list |
| CPU-dirty | read-write | The CPU may be writing | a submission consumes the dirty list: the block is uploaded once, re-armed and becomes CPU-clean |
| GPU-owned | no access | The newest content is on the GPU | a guest read or write faults: the block is written back, then becomes CPU-dirty |

- The dirty list replaces `CollectWrites` walks and collect epochs: a submission reads a list
  instead of walking ranges.
- GPU-owned replaces eager write-back and the syncs before CPU reads of GPU results. Data nobody
  reads on the CPU is never copied back.
- Blocks that fault every frame stay CPU-dirty and are uploaded per submission.
- **Precondition: red-zone safety.** A fault on a guest thread makes Windows write the exception
  frame below RSP, into the 128 bytes that System V leaf functions use. The relinker's
  `--frame-red-zone` frames leaf functions and reports the ones it skips. Section A is allowed for
  a title only when no function is skipped in any converted module (eboot and every PRX).
  Otherwise the title keeps the write-watch model. As built: the relinker marks the executable
  (8 bytes in the DOS header's reserved words) only when `--frame-red-zone` left no leaf skipped
  in the executable and every module, and libc arms memory only when the running executable
  carries the mark. Programs without guest code (tests, probes) opt in with
  `APS5_PROTECT_TRACKING=1`.
- **Stacks are never armed.** An exception cannot be delivered on a read-only stack (the process
  ends without a report), which framing does not cover. Guest threads run on host stacks, which
  are not guest allocations and so are never armed; fiber stacks are pinned; the guest main stack
  is excluded explicitly. A title that switches to a stack of its own inside GPU-imported heap
  memory is not covered and not detected (open risk).
- **Red-zone framing, as built.** Each leaf that keeps values below rsp is framed by the first of
  these that fits:
  1. In place: the prologue and exits become stubs that lower and raise rsp, and rsp operands move
     by the shift (in place, or in operand stubs where they cannot).
  2. A return thunk: the entry calls the rest of the leaf, so every ret lands on a stub that
     raises rsp.
  3. Relocated whole: the leaf is copied into a stub that runs it with rsp 0x80 lower. Pushes and
     pops keep their slots, every exit raises rsp first, and jump-table cases jump into the stub.
     A function entered from outside its start is always relocated, since that entry sees rsp as
     the leaf keeps it.

### B. Residency

- CPU-clean and GPU-owned blocks have a device-local copy. Shaders and draws read that copy, never
  a host import.
- One ledger tracks owner, dirty state and in-flight uses for imports, mirrors and staging pools,
  under one budget from `VK_EXT_memory_budget`. Least recently used blocks are evicted by writing
  them back and pointing the range at RAM.
- An upload happens once per change, in proportion to the changed bytes.
- Built so far in walk-based form:
  - Bound ranges of at least 64 KiB that draws bind in 3 frames get a device-local copy, refilled
    at most once per frame after a collected write (budget 2 GiB, LRU;
    `APS5_DIRECT_NO_RESIDENT=1` turns it off).
  - Fixed 32 MiB windows around the index data of read-only draws get a copy, entered in those
    draws' address tables; a changed copy set bumps a generation and older tables are rebuilt.
  - Each copy's record is in block-state terms (`ResidentCopy::currentFrom`, plus the batch serial
    that makes it current), so section C can block on that batch once blocks are GPU-owned.

### C. Waits

- A wait whose value comes from a recorded or queued label store is resolved from the record, in
  order, with no GPU completion.
- A wait on another worker's not-yet-recorded store blocks on a condition that the recording
  signals. `PollSleep` goes away.
- Only a wait on a value the GPU computes blocks on GPU completion, through a fence or a timeline
  semaphore.

### D. Direct path

- Kept as built: light capture, data-independent shaders, heap slots, table mirror, and the
  fallback per draw. `APS5_NO_DIRECT=1` restores the old path as a whole.
- One change: memos (slot memo, table mirror, column stamps, binding plans) are valid until a dirty
  event on their source blocks, instead of per collect epoch. With section A in place, the per-draw
  proof becomes a flag read.
- Pass hazards include the GPU-written storage images that a direct stage binds in heap slots.
  This is hazard information only, not a per-draw change check.

### E. GPU stage

- **Timeline:** merge consecutive draws into one pass, drop barriers without a hazard, and keep
  storage images on the GPU (section A makes this legal). GPU-side indirect draws whose arguments
  the CPU can snapshot continue the open pass (`APS5_NO_INDIRECT_IN_PASS=1` restores the old
  behaviour).
- **Translation:** the dominant program of each title. Address lookups that the CPU can answer are
  resolved when the binding plan is built and held by the section D memo (principles 1 and 2). The
  shader keeps the bounds test and falls back to the table search outside the stored range.

## Phases

| Phase | Work | Gate to continue | If the gate fails |
|---|---|---|---|
| 0 | Hygiene; a fixed scene and a worker and GPU profile per title | A ranked list per title | n/a |
| 1 | Red-zone audit. Convert every title with `--frame-red-zone`, skipped functions to zero. Fault-storm test: arm everything, fault constantly with AVX state live. | Zero skipped in every module; the storm (red zone overwritten at steps in guest code) runs 30 min clean per title; the same storm without framing reproduces corruption (guest-only clobber, 2000 landings/s: unframed FF, LEGO, 007 and AoM all crash with the 0xa5 fill in registers). Native modules are built with `-mno-red-zone`, so the guest's calls into them are covered too | Keep write-watch; one hot-set walk per submission instead of A |
| 2 | Section A, CPU-clean and CPU-dirty only, built fresh on `rewrite/direct` | Walks and epoch bumps near zero; the coherence share falls by at least half on AoM and FF; pictures match | Revert, keep as a patch |
| 3 | Section C | Worker wait time under 2 ms per frame on LEGO and 007; GPU idle gaps under 20 % of the frame where the CPU is not the limit | Revert per wait class |
| 4 | Section D validity change, then section B and GPU-owned | Uploads and write-backs proportional to changed bytes; 007 and LEGO upload time down 5x; footprint not above today's; walks and epoch bumps near zero on FF (moved here from phase 2 on 2026-11-14: it needs memos valid per block) | Keep eager write-back for the failing class |
| 5 | Section E: timeline, then translation | GPU busy at or under 14 ms on at least one title | Report the floor per title |
| 6 | Per-draw CPU toward 20 us | Worker at or under 10 ms per frame | n/a |

Phases 3 and 5 do not depend on phase 1. Each kept change gets one regression pair on a second
title.

## Progress (2026-11-15)

### Per game

Completion is the share of the 60 fps target reached: 16.7 ms divided by today's frame time.

| Title | Where it runs | Frame | Completion | Bound by | Phase work open |
|---|---|---|---|---|---|
| FF Resonance | in game, bench scene | 35.3 to 36.6 ms | **47 %** | CPU: worker 0 at 97 % | 4 (carries the walk and epoch target) |
| LEGO | in game, bench scene | 49 to 50 ms | **34 %** | GPU: about 40 ms per frame, one compute program on top | 5 (translation) |
| 007 | title screen and menus; first cutscene stalls on cold shader compiles | 160 to 175 ms before the scene change, about 200 to 221 ms after it | **8 to 10 %** | CPU: the draw committer; queue 0 waits 14 to 23 ms per frame on GPU-computed values | GPU time after the scene change higher since the rebase (open), then 4 and 5 |
| AoM | title screen and the scene after the pad press | 155 to 163 ms | **10 to 11 %** | title GPU, scene CPU | 4, then 5 and 6 |

Frame times are on the rebased tree (upstream 3c310597b6), which is slower than the pre-rebase
numbers this table held before (FF 30.6 to 33, LEGO 44 to 49, 007 121 to 129, AoM 146 to 158 ms).

Divinity is not a test title. BO1 was removed from testing on 2026-11-13 (game deleted).

### Per phase

| Phase | State |
|---|---|
| 0 | Done for all four titles. |
| 1 | **Gate passed for all four titles (2026-11-13).** Every red-zone leaf of the four eboots and their bundled modules is framed, 0 skipped, including code outside `.eh_frame` bounds. Fault storm at 2000 landings/s: unframed, all four crash with the fill pattern in registers; framed, all four ran 30 min clean. Rule approved 2026-11-13: the storm overwrites the red zone only at steps in guest code, and native modules are built with `-mno-red-zone`. Detail: `DESIGN-history.md`, "Phase 1 and 2 status detail to 2026-11-15". |
| 2 | **Kept for all four titles** (aba95ed26b; 007 store-import grid fix 6bccb2bfa7; rebase onto upstream 3c310597b6 and its two fixes 0bc66be545, libs `libs\merge4`). As built: collected memory is armed per 64 KiB block; GPU stores go through small dedicated store imports made uniform at their storing submission; submissions name their stores; collects skip clean ranges. On the rebased tree, on against off: FF walks 3.00 M to 1.39 M, AoM 1.44 M to 0.20 M, 007 5.21 M to 1.59 M; pictures correct. Gate judgement (user, 2026-11-14): coherence share more than halved; the FF walk and epoch target moved to phase 4. **Open:** 007 GPU time after its scene change (about 205 or 221 ms against 185 to 190 before the rebase; next step per-program GPU times against `libs\protect24`). Detail: `DESIGN-history.md`, "Phase 1 and 2 status detail to 2026-11-15". |
| 3 | **Built and kept (user, 2026-11-15); gate met on LEGO, not on 007.** All three section C items are built (`WaitEvents.hpp`): a blocked wait or worker sleeps on events instead of a timed poll. Events: a label store or recorded GPU write over the awaited range, a finished GPU batch (only for waits whose value is on the GPU), driver queue state, release of the GPU mutex after a failed try, a closed label group. The timer left is the label-flush deadline, a 2 ms fallback with a known producer and a 200 us wait for values no driver event announces (a guest CPU store). `PollSleep` remains only in the REWIND wait (a guest CPU store) and the capture-hold test loop. Queue 0 time in waits per frame: LEGO about 1.0 ms to 0.4-0.5 ms (gate met, and met before); 007 about 22 ms to 14-23 ms (gate not met). 007's queue 0 waits on compute queues that themselves wait 150 to 200 ms on GPU-computed values, so its wait is bounded by GPU time (phases 4 and 5), not by wait mechanics. Almost every wait ends on an event (waits ended after a timer wake: 0, once 1, per 10 s). Frame times unchanged within run variance: FF 35.3 to 36.6, LEGO 49.2 to 49.9, AoM 155 to 158, 007 about 200 ms; no errors. GPU idle gaps not measured. Tried and dropped: trusting late label stores (0 on queue 0), and limiting mutex-release wakes to the wait's own consults (wake counts unchanged). A blocked LEGO compute queue still wakes about 15 k times per 10 s, almost all on GPU mutex releases. Tests: see the note below the table. |
| 4 | **Started 2026-11-15; nothing kept yet.** Earlier: section B built in walk-based form (AoM title 216 to about 150 ms); 007 binding join fix 375d360e24 (387-426 to 127-133 ms). Section D validity change: a memo of whole direct stages (request plus replayed guest reads) gave no gain and is in `failed-patches\phase4-direct-stage-memo.patch` (FF 0 % hits, LEGO under 1 % and 51.8 ms against 49.0, AoM 20 % with no frame change): every draw's stage carries buffer addresses that are new each frame. The slot memo is cleared by the proof serial, not by collect epochs (FF 148 k clears per run); its misses cost about 1.5 ms per FF frame and 1.4 ms per LEGO frame of worker time, so the per-block rule can save at most that. Gate baselines and the build plan for section B and GPU-owned: "Phase 4 baselines" below. |
| 5 | Indirect draws continue the pass (AoM 745 to 72 passes per frame). 007 translation size cut 35 % (spanned scalar loads, cheaper NaN check). AoM's hot vertex program is latency-bound (about 30 % in the per-fetch `probe_bda_pointer` search, 25 % waiting on vertex data). LEGO's top program has no single emulation cost to remove. |
| 6 | Per-draw record cost 12 to 25 us; about 156 us per draw with stage preparation on the AoM worker. |

Tests (2026-11-15, subset of 1,790): pass with section A on and off on 1732c4cf56. Three tests failed
intermittently under parallel load before that commit; both causes are fixed in it:

- `agc_driver_draw_unsupported_buffer_type` (heap corruption, 0xc0000374): `DrawCapture.cpp` kept
  refusal text in `thread_local` strings. MinGW destroys such an object at thread exit after its
  storage is freed. Rule for new code: no `thread_local` object with a destructor on a thread that
  can exit (about 80 older declarations in the driver are unchanged and not known to fail).
- `agc_driver_depth_expand_out_of_place` and `_inline` ("the depth draw did not write its gradient
  at the read base", about 6 % of runs under load): an import's deferred settle finished after the
  depth draw, the range went back into the write watch, and the rewatch stamp made the depth surface
  reload stale guest memory. The first sync after a plane is watched again now only takes a new
  baseline. The same race exists upstream; the direct path draws sooner and so hits it.
- Still open, also on plain upstream (about 1 % under load): the same test's "expanded planes
  stored ahead of the copies: 0 (expected 1)", when an idle flush stores the plane before the copy
  does. A timing assumption of the test, not a wrong picture.

### Phase 4 baselines (2026-11-15, `libs\c8`, `APS5_PROFILE_GPU=1`, last `[gputime]` line, per present)

| Class | FF | LEGO | AoM | 007 |
|---|---|---|---|---|
| storage-upload (guest memory into a storage image) | 0.5 ms, 59 MiB | 3.2 ms | 0.01 ms | 7.1 ms, 258 MiB |
| storage-writeback (image results into guest memory) | 1.1 ms, 78 MiB | 0.1 ms | 7.5 ms, 210 MiB | 9 to 12 ms, 310 MiB |
| staging-in (import into a device-local buffer) | 1.5 ms, 39 MiB | 0.7 ms, 9 MiB | 0.3 ms | 2.2 ms, 82 MiB |
| staging-out (written sub-ranges back) | 1.5 ms, 39 MiB | 0.7 ms, 9 MiB | 0.3 ms | 2.2 ms, 82 MiB |
| shadow-publish | 1.3 ms, 59 MiB | none | 0.06 ms | 1.7 ms, 71 MiB |
| draw | 3.3 ms | 2.2 ms | 82.6 ms | 1.4 to 1.6 ms |

- 007's GPU time is about 60 ms per present. One compute program (0x3001052300, three runs per
  present) is 34 ms of it (phase 5). The copy classes above are about 25 ms.
- 007 storage uploads are all "guest memory changed" (`APS5_TRACE_UPLOAD=1`). The title reuses the
  same guest memory within a frame for different images (0x425aa00000 as 128x1, 16x16, 54x31 and
  640x360; 0x425b13d000 as 640x360 and 1280x720). Each image is written back to host memory so
  that the next one can be uploaded from it: 570 MiB per present through the import.
- AoM writes 210 MiB of image results back per present and uploads almost none of it again.
- LEGO's storage-upload byte count in that line is not credible (3 GiB per present in 3.2 ms);
  the time is.

Correction (2026-11-15, later): step 1 below already exists upstream as unit shadows
(`UnitShadow.cpp`: storage write-backs go to device-local slabs, uploads read them, host memory
is written only for host-facing readers). It covers only surfaces at a 64 KiB base. 007's pool
surfaces are at 4 KiB bases and its pool is mapped in 2 to 48 MiB pieces, so its images and
buffer regions often span two mappings and no single import served them. Found and changed
(a05fb7e28b, kept by the user, `libs\e2`):

- About 30 buffer regions per frame (2 to 8 MiB) were copied on the CPU, each behind the GPU
  results pending in it (50 to 90 ms per dispatch; 32 GiB read per run). `joinedRun` now joins a
  run whose mappings are all 2 MiB or larger (`APS5_NO_LARGE_RANGE_JOIN=1` turns it off).
- About 12 storage-image uploads per frame took the CPU path (guest read plus a waited batch).
  A surface without block units that several imports serve is now gathered on the GPU into a
  device-local buffer and detiled from it (`APS5_NO_GATHERED_UPLOAD=1`). With the join in place
  this only removes the last 72 such uploads per run; frame time is the same with and without it.
- 007: 142 to 167 ms per frame before its scene change (was 160 to 175) and 180 to 188 after
  (was 197 to 221), two runs; CPU-copied regions 6,419 to 128 in 40 s. FF 36.2, LEGO 49.1 and
  50.6, AoM 161.4: unchanged. No errors; tests 1,790 of 1,790 with section A on and off.
- Image tables (c4d8f47c8f, `libs\f3`): 007's direct stages cost 67 us each, 50 us of it
  resolving image tables. Its table columns are pointer columns and key-narrowed columns, which
  bypass the column memo, so each capture held equal values in blocks of its own; the resolution
  memo matched shared blocks only, and the heap mirror was rebuilt for every stage (0 reuses).
  The memo now also matches equal content (`APS5_TABLE_MEMO_SHARED_ONLY=1` restores the old
  match). Direct stage 15 us, mirrors reused, `Shader.DrawStage` 65 to 70 ms per frame gone from
  the top costs. 007: 117 to 135 ms per frame over the whole scene, two runs. FF 36.6, LEGO 49.1,
  AoM 157.1: unchanged. Tests: 1,808 of 1,808 with section A on; with it off one label test
  (`agc_driver_unregistered_label_on_gpu`) failed once and passed 70 repeats after.
- 007's storage cache sits at its 7.2 GiB budget and makes about 41 images per present: the
  title allocates the same kind of target at hundreds of addresses in a cycle (1280x720 format
  36 at 338 addresses). Each new image uploads its whole surface ("first", 231 MiB per present).
  Not changed.

Build plan that follows from this, in order:

1. Section B for storage images: a device-local copy of the guest ranges that storage images
   exchange. A write-back whose reader is another image goes to that copy, and the upload reads
   it. Host memory is still written (eagerly) at this step. Expected: 007 storage-upload from the
   import gone (7 ms). Exists for 64 KiB bases (see the correction above); what is left is block
   units for surfaces at 4 KiB bases.
2. GPU-owned: blocks whose newest content is in that copy become no-access; a guest or host read
   faults and stores them first. Then the eager write-back to host memory goes (007 9 to 12 ms,
   AoM 7.5 ms, FF 1.1 ms). Needs a read-fault path that can wait for the GPU from a guest thread;
   today's `HandleRead` only rechecks the protection.
3. Staging: skip the copy-in when the staging buffer is still what the import holds (no stamp on
   its blocks since its own copy-back); then the copy-back under GPU-owned.
4. Slot memo and table mirror validity per source block (small; see the phase 4 row).

### Next, in phase order

1. **Phases 1 and 2:** passed and kept (see their rows). Open from phase 2: 007's GPU time after its scene change.
2. **Phase 3** (section C, waits): built and kept; 007's wait gate is left to phases 4 and 5 (see the phase 3 row).
3. **Phase 4** in progress: step 1 of the build plan in "Phase 4 baselines" is next.

Parked until their phase comes up: the AoM fetch-range probe and timeline outliers, the FF worker
profile, the LEGO translation question (phase 5), the 007 committer re-profile and storage-image
sharing (phase 4).

## What this can and cannot deliver

- Phases 1 to 4 remove CPU work and waiting. They cannot make a title faster than its GPU time.
- 60 fps on LEGO needs its top program at about a third of today's cost. Nothing measured so far
  says that is reachable.
- AoM and 007 have 30 fps as the first milestone (AoM at 22 %, 007 at 27 % of it today).
- A title that fails phase 1 gets the walk-based variant and a lower ceiling.

## Decisions (approved 2026-11-11)

1. **Red zone.** Protection-based tracking is allowed on `rewrite/direct` for titles that pass the
   phase 1 gate.
2. **Titles.** LEGO and AoM are on the GPU track. FF Resonance is on the coherence and waits track;
   it replaced AoM there after phase 0. 007 joins once it has a repeatable gameplay scene.
   Divinity is not a test title.
3. **Milestones.** 30 fps first for AoM and 007.

## Related work to take later (noted 2026-11-14)

`simnJS:integ/stack-12` (compare `boykopovar/AnyPS5` main against it): 13 commits of NVIDIA-on-Windows
performance work, measured on TCG Card Shop Simulator (PPSA36301, RTX 4080 SUPER). Read from the
commit messages and file list only, not the diffs. It sits on an upstream main of 2026-11-14 (this
branch is on 2026-11-11), so it needs an upstream merge first, and it rewrites staging and
copy-back code that phase 2 changed (`GuestBufferMemory.cpp`, `Recorder.cpp`,
`ShaderResources.cpp`, `Texture.cpp`).

| Their change | Their measured effect | Where it fits here |
|---|---|---|
| Staged regions copied back with a buffer copy instead of a compute pass | 63 to 7 ms per frame | 007 (about 41 GiB staged in and back in two minutes); phase 4 |
| Staging shadow handed to the next dispatch; one shared shadow per region | 271-334 to 10-19 MiB staged per frame | 007; phase 4. Would also shrink the large store regions of the 007 device loss |
| Staging shadows of cached builds bounded | avoids a collapse to 1.4 fps | footprint (principle 7) |
| Device-local input twins for read-only dispatch inputs | 114-121 to 2-8 us per dispatch | section B for dispatches (built here for draw buffers only) |
| Draw inputs behind a pending GPU write copied on the GPU | removes a 35 ms committer wait | 007 committer; section C |
| Draw proofs reused while nothing they read moved; twin verdict per collect epoch | not measured in the title | section D validity change (phase 4); keyed on the collect epoch, so the epoch skips extend it |
| Draw shape reuse across moved user words; relocation learned where the draw finishes | not measured in the title | phase 6 |
| Single-key storage image upload on the GPU | not measured | one committer sync less |
| One write per log line, prints off the GPU mutex; clock reads only when profiling | n/a | general |

Rule when taking it: every copy-back is a GPU store into a host import, so it must go through a
store import and be named at its submission (section A), or the dropped-write hazard returns.
Order: staging commits and input twins first (007's two largest costs), proofs with phase 4, shape
reuse with phase 6.

## Not in this plan

- GPU-side descriptor reads, hash tables and miss rings (measured slower).
- Visibility policies for clean ranges (measured flat).
- Tuning of the locked draw preparation until a profile ranks it as a top cost.
