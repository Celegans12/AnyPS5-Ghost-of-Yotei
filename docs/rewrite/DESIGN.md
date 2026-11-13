# Direct path: design, draft 3

Branch `rewrite/direct`. Approved by the user on 2026-11-11, including the decisions below. Dated
measurements, probes and results are in `DESIGN-history.md`; this file holds the design and where
it stands now.

Target: 16.7 ms per frame. That needs the CPU worker at or under about 10 ms and the GPU at or
under about 14 ms per frame.

Three limits remain:

1. **Coherence.** The CPU re-establishes what guest memory holds at every ordering point
   (write-watch walks, collect epochs, memo revalidation, uploads of unchanged data). The cost scales
   with how often memory is used, not with how often it changes.
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

### B. Residency

- CPU-clean and GPU-owned blocks have a device-local copy. Shaders and draws read that copy, never
  a host import.
- One ledger tracks owner, dirty state and in-flight uses for imports, mirrors and staging pools,
  under one budget from `VK_EXT_memory_budget`. Least recently used blocks are evicted by writing
  them back and pointing the range at RAM.
- An upload happens once per change, in proportion to the changed bytes.

As built before phase 1 (walk-based):

- **Bound ranges.**
  - A range of at least 64 KiB that draws bind in 3 frames gets a device-local copy, filled by a
    GPU copy from the import.
  - The copy serves every binding while no write over the range is collected. After a collected
    write it is refilled at most once per frame; a second write in the same frame reads the import
    until the next frame.
  - A binding prepared while a recorded write is pending may use the copy, because the fill is
    recorded after that write on the same queue.
  - Budget 2 GiB, least recently used first out. `APS5_DIRECT_NO_RESIDENT=1` turns the copies off.
- **Index windows (memory read through the address table).**
  - A fixed 32 MiB window around the index range of read-only draws, used in 3 frames, gets a copy.
    It is entered in those draws' address tables when a table is built.
  - A table is never rewritten while in use. A change to the copy set bumps a generation, and an
    older table is rebuilt.
  - Changed 64 KiB blocks are refilled before use, with one check per copy per frame.
  - A window leaves the set when its refills copy more than twice its size within 32 frames, or
    when no draw uses it for some frames.
  - Dispatches and draws that store through addresses keep the imports.
- Each copy's record is kept in block-state terms (`ResidentCopy::currentFrom`, plus the serial of
  the batch that makes it current), so section C can block on that batch once blocks are GPU-owned.

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
- Pass hazards include the GPU-written storage images that a direct stage binds in heap slots
  (`ResolveDirectStages` notes them, `recordDraw` adds them). This is hazard information only, not
  a per-draw change check.

### E. GPU stage

- **Timeline:** merge consecutive draws into one pass, drop barriers without a hazard, and keep
  storage images on the GPU (section A makes this legal).
  - GPU-side indirect draws whose arguments the CPU can snapshot continue the open pass.
    `APS5_NO_INDIRECT_IN_PASS=1` restores the old behaviour.
- **Translation:** the dominant program of each title.
  - Address lookups that the CPU can answer are resolved when the binding plan is built and held by
    the section D memo (principles 1 and 2).
  - The shader keeps the bounds test and falls back to the table search for any address outside
    the stored range.

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
  - **Remaining skipped: 007 9, AoM 7, LEGO 2, FF 1** (from 132 / 147 / 158 / 39). 7 have `[rsp]`
    operands without a displacement next to branch targets (no room for a stub), 4 read a pushed
    slot through rsp in a thunk, 4 unresolved indirect jumps, 4 single cases. These need the leaf
    relocated whole into stub space (all rsp operands rewritten, branches and tables relocated).
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

Next, in phase order:

1. **Phase 1.** Relocate the last skipped leaves whole (19 across the four titles), or decide
   they are acceptable. Check each change with one framed run per title for crashes (40 s, no
   benchmarks needed). Then frame the bundled PRX modules, account for
   code without `.eh_frame` bounds, and build the fault-storm test.
2. **Phase 2** once a title passes the phase 1 gate.
3. **Phase 3** (waits) after that; it does not depend on phase 1.

Parked until their phase comes up: the AoM fetch-range probe and timeline outliers (phase 5), the
FF worker profile, and the LEGO translation question (phase 5).

Open:

- The AoM mirror ceiling (35 and 74 us per draw) was measured on isolated draws, so it may not be a
  like-for-like target.

## What this can and cannot deliver

- Phases 1 to 4 remove CPU work and waiting. They cannot make a title faster than its GPU time.
- 60 fps on LEGO needs its top program at about a third of today's cost. Nothing measured so far
  says that is reachable.
- AoM and 007 have 30 fps as the first milestone.
- A title that fails phase 1 gets the walk-based variant and a lower ceiling.

## Decisions (approved 2026-11-11)

1. **Red zone.** Protection-based tracking is allowed on `rewrite/direct` for titles that pass the
   phase 1 gate.
2. **Titles.**
   - LEGO and AoM are on the GPU track.
   - FF Resonance is on the coherence and waits track; it replaced AoM there after phase 0.
   - 007 joins once it has a repeatable gameplay scene.
   - Divinity is not a test title.
3. **Milestones.** 30 fps first for AoM and 007.

## Not in this plan

- GPU-side descriptor reads, hash tables and miss rings (measured slower).
- Visibility policies for clean ranges (measured flat).
- Tuning of the locked draw preparation until a profile ranks it as a top cost.
