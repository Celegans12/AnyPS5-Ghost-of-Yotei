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
| 1 | Eboot framing done: every red-zone leaf of the four eboots is framed, 0 skipped (from 132 / 147 / 158 / 39; 18 relocated whole, 98cf5da8d9). Framed crash checks (40 s) clean on all four. Bundled PRX modules framed too (5 / 11 / 16 / 1 modules for FF / AoM / LEGO / 007, 0 skipped, every module has `.eh_frame` bounds; the one skip was libc's longjmp `mov rsp, [mem]`, now loaded and lowered by the shift); framed crash checks clean. Code without `.eh_frame` bounds is framed too: FF's FDEs cover only 6.5 of 107.6 MB of its eboot's code, so code outside every FDE is split into functions at int3 padding and direct call targets (nops neither start nor end one) and framed like the rest (FF eboot 289,831 such functions, 411 more red-zone leaves; LEGO 550, AoM 1,337, 007 563 plus modules). New relocated shapes: `pop rsp` and `mov [mem], rsp`, and `mov rsp, [mem]` now goes through a pushed scratch so rsp never rises above the lowered rsp. 0 skipped in every image; crash checks clean (FF only reaches its title menu in 40 s). Fault storm (6317f55d61, `APS5_FAULT_STORM`): a storm thread single-steps bursts of 256 instructions on threads in guest code and overwrites the 128 bytes below rsp at every step (on this 9800X3D a real frame reaches the red zone only while AVX-512 state is in use, measured). Gate, 2000 landings/s: unframed, all four titles crash within 25 s with the pattern in registers (FF in zlib inflate_fast); framed, FF, LEGO and 007 run 30 min clean (24 M / 63 M / 674 M steps); framed AoM aborts in libc about 30 s in (3 of 3; not without the clobber, and also with the clobber limited to guest code), so a red-zone use in AoM's guest code is still unframed. Found by bisecting the clobber with `APS5_FAULT_STORM_CLOBBER_RANGE`: libc's bounds-checked copies (ELF 0x6d360, 0x6d270 and two more) are frame-pointer functions that call, yet keep `[rbp-8]` below rsp before their first call to compare against their own frame, and the framer framed leaves only. Now a frame-pointer function with an access below its frame built so far (pushes and `sub rsp` before the first call; `lea`, indexed operands and `and rsp` realignment excluded) is framed too; one that pushes only rbp leaves through `leave` (its `pop rbp` rewritten), so no exit stub. Libc's four copies are framed in every title. Splitting uncovered code now also starts a function after a ret, jmp or ud2 at code no earlier branch reaches (FF's shrink-wrapped functions had merged with the one before), and a leaf that calls a helper inside its own bounds (inline-asm AES key expansion in 007) stays a leaf. 0 skipped in every image; framed AoM with the storm: 90 s clean (it aborted at 30 s before). A 30 min gate run then aborted at 256 s inside our native AJM code (an invalid job size in a batch from libfmod): the storm's clobber also ran at steps a burst took into native code, and our `sysv_abi` exports may keep values in the red zone (MinGW GCC uses it for System V functions), the same hazard a real fault there would hit. With the clobber limited to guest code, framed AoM ran 10 min clean. Gate rule changed with the user's approval (2026-11-13): the storm overwrites the red zone only at steps in guest code, and native modules are built with `-mno-red-zone` (top-level CMakeLists, MinGW). Gate result under that rule (`logs\storm-gate3-20261113.out`, libs `libs\storm3`): unframed AoM crashed at 20 s with an access violation and `a5a5…` in rax (the fill pattern); framed, all four titles ran the 30 min storm clean (AoM 27.5 M steps, FF 24.2 M, LEGO 59.1 M, 007 680 M; 179 intervals each, no FATAL). Phase 1 gate passed for all four titles. |
| 2 | **Kept and committed (aba95ed26b, 2026-11-14) for FF and AoM; 007 fails and stays on write-watch until fixed.** As built (third build): collected memory is armed per 64 KiB block; GPU stores go through small dedicated store imports that are made uniform at their storing submission, so heap imports stay partly armed; submissions name their stores; collects skip clean ranges; an epoch bump is skipped while the bumping thread walked nothing unclean and no armed memory was opened. Gate pairs (framed, same binary on/off): FF walks 1.50 M to 0.34 M, collect 2.9 to 1.1 s, real epoch bumps 25.4 k to 18.6 k, frame 36.3 to 36.6 ms, pictures match, 9.9 GiB combined. AoM walks 3.3 M to 0.07 M, collect 5.6 to 1.1 s, epoch bumps 2.0 k to 1.7 k, frame 171-176 to 157-159 ms; both sides render correctly but the title camera gives no matched shot; 25 GiB combined on and off alike (not from phase 2). **Gate judgement (user, 2026-11-14):** the coherence share is more than halved on both; FF walks and epoch bumps are not near zero because the walks left consume real writes (16.7 k of 18.6 k real bumps follow the thread's own walk of written memory) and memos are still valid per epoch. That target moves to phase 4 with the section D validity change. Rejected on the way: hot blocks answered "written" without a walk (FF 36.7 to 52.5 ms and missing particles, `failed-patches\phase2-hot-no-walk.patch`): a hot answer may only cover pages the write watch reports. **007 (framed, 300 s, `runs\007\gate2-007-*`):** off runs clean at 187 ms; on, one of two runs stopped submitting at about 120 s (waits of 100 s on three queues), then lost the device (`vkQueueSubmit` -4); the other ran 300 s at 208 ms with probes on (walks 6.8 M to 1.4 M, collect 13.8 to 6.4 s). 602 stores still went through ordinary imports and 98 import groups were opened. Open: find the lost submission or deadlock before 007 runs with section A. **First build (2026-11-13), gate not met, reverted, kept as `failed-patches\phase2-section-a.patch` (libs `libs\protect6`).** Built: (1) libc `WindowsMappings`: a resetting collect arms consumed private and resident memory read-only (CPU-clean); a store faults once and opens its 64 KiB block (CPU-dirty); blocks that fault repeatedly stay dirty (strikes); host writes, pins and guest mprotect open armed memory; armed pages report read-write to the guest and the driver. (2) Blocks the GPU writes stay CPU-dirty: the recorder's pending-write notes and storage targets open every host import they land in, for good (also after the guest range is released and reused, as the import can outlive it). (3) The dirty-event answer replaces walks: `collectWrites` skips any range libc reports clean, with no walk and no epoch. Counters on `[guestmem]`; unit test in `windows_write_watch`; `guest_memory`'s resident test states the new rule. ctest subset: no new failures. **Hardware finding (RTX 5080, driver 617.42; probe `probe-patches\import_protect_probe.cpp`):** a GPU write into a host import with some pages read-only and others writable silently drops the whole submission in about 1 of 12 runs (5/60 shader stores, 5/60 transfer fills, 5/60 even when the stores avoid the read-only pages); reads of such an import (0/60), writes into a uniformly read-only import (0/120) and imports reopened before the write (0/60) are unaffected. **Gate result (framed, profiled, same binary on/off):** FF: 346 imports GPU-written, 69 MiB armed, walks 3.02 M to 2.50 M, collect 5.4 to 5.8 s, epochs about 39-40 k both, frame 35.9 to 36.6 ms. AoM: its 13 GPU-written whole-heap imports cover everything collected, 0 MiB armed, walks 3.81 M to 3.47 M, epochs 2.2 k both, frame 166.7 to 168.7 ms. Walks and epoch bumps are not near zero and the coherence share does not fall: the guest heaps the collects walk are host imports the GPU also writes. Arming them anyway (probe) gave FF 540 MiB armed and collect 3.9 to 1.6 s but drops GPU work. **Follow-up probe (same program, 64 fills in flight):** only the protection at submission matters. Mixed at submission: 6/60 lost. Uniform read-only, or mixed and re-armed to uniform before submission: 0/60 (fills and shader stores). Pieces opened or armed while the submission runs: 1 lost in 420. **Route to the gate within section A:** the as-built version armed after a collect and kept GPU-written imports open; the table says a submission consumes the dirty list and re-arms. Built that way, every import a submission can write is uniform at submission: its dirty blocks are re-armed (CPU-clean) as the submission consumes them, so whole heaps can be armed, GPU-written or not. A block that faults every frame then costs one fault per frame instead of staying open; where that costs more than the walk, the whole import stays CPU-dirty (uniform read-write). With every registered block clean or listed, the dirty list also answers the collect memo, so epoch bumps are skipped while the list is empty. Expected from the unsafe probe: FF collect 3.9 to 1.6 s. **Second build (2026-11-14, in progress, uncommitted):** re-arm at submission is built (every `vkQueueSubmit` runs inside an arena bracket that first makes each import group uniform). Only guest allocations are armed; an import group holding other memory stays open. **Second hardware finding (same probe, `PROBE_OVERLAP`, `PROBE_PRIOR`, `PROBE_LATE`):** when two imports cover the same pages, a write through an import whose first GPU use happens with its pages read-only is dropped (40 of 40; a later-made import over an already used one 29 of 30; in the engine the device was lost). An import used once while writable, even by a 4-byte read, is unaffected (0 of 30). So every import the arena may arm gets a 4-byte read at creation, inside its host-access bracket. The collect skip for clean ranges now records its observation in the collect memo like a walk that found nothing. Open: the test subset on and off, the FF and AoM pairs against the gate, and the epoch-skip item. |
| 3 | Not started. |
| 4 | Section B built in walk-based form (AoM title 216 to about 150 ms); 007 binding join fix 375d360e24 (387-426 to 127-133 ms). Section D validity change not started. |
| 5 | Indirect draws continue the pass (AoM 745 to 72 passes per frame). 007 translation size cut 35 % (spanned scalar loads, cheaper NaN check). AoM's hot vertex program is latency-bound (about 30 % in the per-fetch `probe_bda_pointer` search, 25 % waiting on vertex data). LEGO's top program has no single emulation cost to remove. |
| 6 | Per-draw record cost 12 to 25 us; about 156 us per draw with stage preparation on the AoM worker. |

### Next, in phase order

1. **Phase 1:** gate passed for all four titles (see the phase 1 row).
2. **Phase 2** (section A): kept for FF and AoM; the FF walk and epoch target moved to phase 4 (see the phase 2 row). Open before phase 3: 007 with section A on stops submitting and loses the device in one of two runs.
3. **Phase 3** (section C, waits) after the 007 failure is understood.

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
