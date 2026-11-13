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
  Otherwise the title keeps the write-watch model.
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
| 1 | Red-zone audit. Convert every title with `--frame-red-zone`, skipped functions to zero. Fault-storm test: arm everything, fault constantly with AVX state live. | Zero skipped in every module; the storm runs 30 min clean per title; the same storm without framing reproduces corruption | Keep write-watch; one hot-set walk per submission instead of A |
| 2 | Section A, CPU-clean and CPU-dirty only (from the `APS5_PROTECT_PRIVATE_WATCH` prototype, branch proto/fault-tracking) | Walks and epoch bumps near zero; the coherence share falls by at least half on AoM and FF; pictures match | Revert, keep as a patch |
| 3 | Section C | Worker wait time under 2 ms per frame on LEGO and 007; GPU idle gaps under 20 % of the frame where the CPU is not the limit | Revert per wait class |
| 4 | Section D validity change, then section B and GPU-owned | Uploads and write-backs proportional to changed bytes; 007 and LEGO upload time down 5x; footprint not above today's | Keep eager write-back for the failing class |
| 5 | Section E: timeline, then translation | GPU busy at or under 14 ms on at least one title | Report the floor per title |
| 6 | Per-draw CPU toward 20 us | Worker at or under 10 ms per frame | n/a |

Phases 3 and 5 do not depend on phase 1. Each kept change gets one regression pair on a second
title.

## Progress (2026-11-13)

### Per game

Completion is the share of the 60 fps target reached: 16.7 ms divided by today's frame time.

| Title | Where it runs | Frame | Completion | Bound by | Phase work open |
|---|---|---|---|---|---|
| FF Resonance | in game, bench scene | 30.6 to 33 ms | **52 %** | CPU: worker 0 at 97 % | 1, then 2 to 4 |
| LEGO | in game, bench scene | 44 to 49 ms | **36 %** | GPU: one compute program, 22.8 ms | 1, then 5 (translation) |
| 007 | title screen and menus; first cutscene stalls on cold shader compiles | 121 to 129 ms | **13 %** | CPU: the draw committer | 1, then 4 |
| AoM | title screen and the scene after the pad press | 146 to 158 ms | **11 %** | title GPU, scene CPU | 1, then 5 and 6 |

Divinity is not a test title. BO1 was removed from testing on 2026-11-13 (game deleted).

### Per phase

| Phase | State |
|---|---|
| 0 | Done for all four titles. |
| 1 | Eboot framing done: every red-zone leaf of the four eboots is framed, 0 skipped (from 132 / 147 / 158 / 39; 18 relocated whole, 98cf5da8d9). Framed crash checks (40 s) clean on all four. Bundled PRX modules framed too (5 / 11 / 16 / 1 modules for FF / AoM / LEGO / 007, 0 skipped, every module has `.eh_frame` bounds; the one skip was libc's longjmp `mov rsp, [mem]`, now loaded and lowered by the shift); framed crash checks clean. Open: code without `.eh_frame` bounds, the fault-storm test. |
| 2 | Not started (waits on the phase 1 gate). |
| 3 | Not started. |
| 4 | Section B built in walk-based form (AoM title 216 to about 150 ms); 007 binding join fix 375d360e24 (387-426 to 127-133 ms). Section D validity change not started. |
| 5 | Indirect draws continue the pass (AoM 745 to 72 passes per frame). 007 translation size cut 35 % (spanned scalar loads, cheaper NaN check). AoM's hot vertex program is latency-bound (about 30 % in the per-fetch `probe_bda_pointer` search, 25 % waiting on vertex data). LEGO's top program has no single emulation cost to remove. |
| 6 | Per-draw record cost 12 to 25 us; about 156 us per draw with stage preparation on the AoM worker. |

### Next, in phase order

1. **Phase 1:** account for code without `.eh_frame` bounds (code an FDE does not cover, in the
   eboots and modules), and build the fault-storm test. Check each change with one framed run per title for crashes (40 s,
   no benchmarks needed).
2. **Phase 2** once a title passes the phase 1 gate.
3. **Phase 3** (waits) after that; it does not depend on phase 1.

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

## Not in this plan

- GPU-side descriptor reads, hash tables and miss rings (measured slower).
- Visibility policies for clean ranges (measured flat).
- Tuning of the locked draw preparation until a profile ranks it as a top cost.
