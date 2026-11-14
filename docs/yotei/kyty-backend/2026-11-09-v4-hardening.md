# v4h (PLAN v4 phases 3/4): adapter hardening (2026-11-09)

Lane v4h (v4h3 since 09:35). Status: **v4h3 done 10:58** (§5 traced-pair analysis, §6 throw located and adapter
switches A/B'd; nothing of this lane is running). PLAN, STATE, HANDOFF and upstream/ are untouched. Files: `research/v4/h/`. Tags: [M] measured (run tag named),
[E] estimate (reasoning named), [C] read in code (file:line).

## 0. Branches and switches
- Kyty `v4/h` (worktree `C:/ps5/kyty-src-wt/h`, from `v4/m1b` 72f2250b):
  - bb0d387b Graphics: `KYTY_TRACE_H` (diagnostic, default off), the guest link of each flip, the same hooks in Kyty
    standalone and in kyty_gpu;
  - 78d323dd kyty_gpu: ready flips AnyPS5 never presents are reported; `KYTY_APS5_FLIP_RESYNC=1` (default off) drops them;
  - 68c90276 kyty_gpu: `KYTY_APS5_HEADER` checks at build time that `src/aps5/GpuBackend.h` is AnyPS5's.
- AnyPS5 `local/v4-h` (worktree `C:/ps5/AnyPS5-scratch/v4-h`, from `local/v4` 7c142723ff6): uncommitted until it builds.
  - kyty_gpu.prx is no longer imported by libSceAgcDriver.prx: with `APS5_GPU_BACKEND=kyty` the adapter loads it
    (`LoadLibraryExW`, beside libSceAgcDriver.prx) at its first use, outside the loader lock; memory and shader
    notifications before that are queued and replayed in order. The old backend never loads it.
  - the `[present]` and frame-dump lines of the kyty path no longer go through `ProfileOutput` (whose static destructor
    joined a thread `ExitProcess` had already killed: the v4b-k-w1 exit hang).
  - adapter switches, default off: `protect-srw`, `lean-counters`; set for a run with `APS5_ADAPTER_SWITCHES=a,b` or
    A/B'd in-run with `APS5_ADAPTER_AB=a+b` (`APS5_ADAPTER_AB_PERIOD_S` 60, `_START` 1500, `_WARMUP` 20).

## 1. What the G2 profile already says (offline, from v4m's v4m-k-prof / v4m-s-prof; scenes not matched)
- Kyty's PERF counters per 5-s window (the same frame range, ~frames 2,100-2,700):
  - execution thread idle (waiting for operations): ours 8,268 idles / 1,053 ms, standalone 2,974 / 109 ms; CP idle
    (no submission to process): 1,486 vs 924 ms. Ours has the stage threads starved for guest work; standalone's
    execution thread is saturated (and its CP drains more: direct 1,327 vs 2,292 ms) [M v4m-k-prof, v4m-s-prof].
  - protect-after-copy re-checks that found the bytes changed: **15.7 vs 1.0 per frame**, of 272 vs 250 re-checked ranges
    (5.8 % vs 0.4 %); commits 13.3 vs 8.5 per frame [M `tools/pacdelta.py`].
  - tracker protections (PERF locks, per call): ours 3.7 us on the GPU threads and 2.5 us on others, standalone 2.8 and
    1.8 us; calls per frame ~2,600 vs ~3,000 [M].
  - tracker protections split at AnyPS5's view boundaries: 372 in the whole 1,200-s run (0.06 per frame) [M adapter line]:
    the views do not explain the x13 (m1b section 4.7's suspicion).
- tsamp (v4m): guest job threads 218.8 ms/frame busy in ours against 154.6 standalone; by IP class, the extra is all in
  the game's poll loops (ours 78.6 % of job samples, standalone 64 %); kyty_gpu code 0.8 % of job samples (~1.8 ms/frame,
  the fault handler included) [M tana w1/w2].

## 2. Lane v4h2 (continuation from 06:40; v4h handed back early, `research/v4/h/handback-1.md`)
Status lines are kept current; results land in sections 3-6 as they come.
- **Gates so far.**
  - Switch-on T1 `v4h-k-t1a` (local/v4-h 519f59c6a2a + v4/h 68c90276, DLL md5 f8cb45e7): coordinator's
    `t1align.py v4h-k-t1a` gives **VERDICT: MATCH (phase-tolerant, exact)**, 60/60 frames byte-identical to
    `t1-k-1080-dense` at offset +7 [M v4h-k-t1a]. kyty_gpu.prx is loaded in eboot.exe (tasklist /m, 40 s in) [M].
  - Switch-off T1 `v4h-t1-off1`: queued in chain1 (attempts 1-4 lost the slot, "lock busy").
- **Binaries of the gate runs.** All gate runs (both T1s, the world run) use the same binaries: local/v4-h 519f59c6a2a
  `build-local` with the DLL of v4/h 68c90276. v4/h's tip 5fcfc57e adds only the env-gated C++ throw tracer
  (`KYTY_APS5_TRACE_CXX_THROW=1`, default off); it is built separately (`logs/build-gpu2.log`, out/ keeps the 68c90276
  DLL and exe as `kyty_gpu-68c90276.prx` / `kyty_emulator-68c90276.exe`) and used only in a diagnostic run.
- **Queued (06:52):**
  - `chain2.sh` (after chain1 ends): `v4h2-k-w1` world gate + exit test (run1080 long, 900 s, default switches, no
    tracing, `APS5_ADAPTER_TEST_EXIT_AT_PRESENT=4500`, a watcher times the exit), then `v4h2-k-tr1` ours traced
    (chain1's recipe; side jobs tsamp w1 90 s from frame 4200 and w2 45 s with IP samples from 4900).
  - `chain2s.sh`: `v4h2-s-tr1` standalone traced (chain1's recipe, exe 956de14e of 68c90276; kyty_run.sh yields to
    every waiting run, so it runs whenever the queue is empty).
  - `build-gpu2` (guarded, class build): the 5fcfc57e DLL.

## 3. Ceiling question, first pass on the G2 profile pair (offline; scenes not matched; the traced pair refines it)
Tool: `research/v4/h/tools/execfr.py <tsamp dir>`: per presented frame (FRAMETIME marks) the execution thread's busy ms
(cycles, so its spin counts as busy), its idle gaps (< 10 % busy for >= 1 ms) and where they start, then a least-squares
fit of frame time on exec busy and the frames binned by exec busy.
| per frame (tsamp w1, 90 s) | ours v4m-k-prof (549 frames) | standalone v4m-s-prof (601 frames) |
|---|---|---|
| frame | 163.8 ms (61.0/10 s) | 149.5 ms (66.9/10 s) |
| exec busy | 132.4 ms | 130.1 ms |
| exec idle gaps >= 1 ms | 19.4 ms | 14.3 ms |
| rest (short idle < 1 ms, partly-busy ticks) | 12.0 ms | 5.1 ms |
| idle by gap start after the present | 0-20 ms: 4.2; 20-60: 6.4; 60-120: 2.0; 120-180: 6.1 | 0-20 ms: 9.8; 20-120: 0.3; 120-180: 4.2 |
| fit frame = a + b * exec busy | 24.3 + 1.054 b (r 0.950) | 16.1 + 1.025 b (r 0.925) |
| fit idle = c + d * exec busy | 15.6 + 0.029 b (r 0.09) | 12.5 + 0.013 b (r 0.04) |
| lowest decile of exec busy | busy 110.1 -> frame 137.3, idle 16.1 | busy 115.9 -> frame 134.1, idle 13.0 |
[M v4m-k-prof, v4m-s-prof tsamp w1]
- Reading [E]: on both, the frame is exec busy plus a fixed overhead that does **not** grow when exec work shrinks (the
  idle gaps are flat across the deciles, 110-171 ms of exec busy). In this range the guest keeps up; there is no sign of
  a guest floor yet. Ours' overhead is ~8 ms/frame larger, and it sits in different places: standalone has one link gap
  right after the present (it presents at its 60-Hz tick, ~10 ms), ours a shorter one (~4 ms: it presents as soon as the
  flip is ready) plus mid-frame starvation at 20-60 ms (6.4 ms/frame) and 120-180 ms that standalone does not have.
  At 20-60 ms the async compute queues wait for CPU-written labels from the guest's jobs (v4g §3), so the extra gaps
  point at the guest's jobs finishing later in ours, not at the flip path.
- Projection [E, extrapolated below the measured range, so a ceiling estimate, not a measurement]: exec busy 95 ms
  (BAR readback gone) -> ours ~124 ms (~80/10 s), standalone ~114 ms (~88/10 s); exec busy 60 ms -> ours ~88 ms
  (~114/10 s), standalone ~78 ms. The fixed gap (~8 ms) weighs more as exec work shrinks: ×0.93 now, ×0.91 at 95 ms,
  ×0.89 at 60 ms. Whether the guest's own critical path caps this earlier needs the traced pair (section 4).

## 4. Hand-back state (v4h2, 06:58; forced hand-back before any of the queued runs got the slot)
- Nothing measured in-game yet by v4h2: the slot was v4o-pl2's (1,500 s) with v4r-k-w01, v4h-t1-off1, v4e-t1off and
  v4x-t1-eager waiting before ours. Left running, detached: chain1 (pid 87437, v4h-t1-off1, then ends on STOP),
  chain2.sh (pid 176135, waits for chain1, then v4h2-k-w1 and v4h2-k-tr1), chain2s.sh (pid 176134, kyty_run.sh pid
  176338 waiting, v4h2-s-tr1), the guarded build-gpu2 of 5fcfc57e (machine_guard pid 170926). `research/v4/h/STOP2`
  stops chain2/chain2s before their next run.
- What became of them (v4h3, 09:40): chain1 ended; `v4h-t1-off1` ran. chain2 ran `v4h2-k-w1` (world gate + exit test)
  and `v4h2-k-tr1`; chain2s ran `v4h2-s-tr1`; build-gpu2 built 5fcfc57e (DLL `out/kyty_gpu.prx` md5
  a327d946ffaa0b6d7fa76a7d94c98fa9, exe a381716e). No v4h2 process is left.
- `v4h2-k-w1` [M]: 767 s, fatal none, **"process exited with code 0" at present 4500** (the
  `APS5_ADAPTER_TEST_EXIT_AT_PRESENT` line is in the err.log), run1080 restore OK, 61.84 presents/10 s ([present]
  metric, 9 blocks, SE 1.17). The exit test passed. Its watcher wrote "ended without the TEST_EXIT line" at 08:33, six
  minutes **before** this run started (08:39): it saw another lane's eboot.exe end. So the exit-to-gone time was not
  measured, but the run ended by itself (no kill, no hang stop).

## 5. The traced pair (v4h3): gap split, PAC, ceiling (offline)

Runs: ours `v4h2-k-tr1` (local/v4-h 519f59c6a2a build-local + DLL f8cb45e7 of v4/h 68c90276; no pad presses; 900 s;
61.48 presents/10 s [present] metric, 61.17 ft.py) and standalone `v4h2-s-tr1` (exe 956de14e of the same Kyty tree,
`research/v4/h/app-h`, pad-w; 900 s; 55.12 ft.py). Same tracing on both (`KYTY_TRACE_H`, PERF, FRAMETIME, sync-timeline
detail at frame 4000). tsamp w1 = 90 s of cycles from frame 4200, w2 = 45 s of IP samples from 4900. New tool:
`research/v4/h/tools/gaplink.py` (places the exec idle gaps against each flip's ready/done and the guest's first
submissions, from the HT marks in tsamp). Outputs: `research/v4/h/profile/<tag>/tana-w1.txt`, `tana-w2.txt`.

### 5.1 Scene match
- Captures `research/v4/h/shots/<tag>-f{1600,2600,4000}.png`: the same river-village bank, **both overcast** (f1600 has
  leaf shadows on both). Positions differ: pad-w made standalone's character jump onto the canoe, and its camera sits
  further back and higher (more of the village in view). Ours stands beside the canoe, camera lower and closer.
- Work per frame [M PERF, frames >= 3000]: draws 3,227 vs 3,354 (+4 % standalone), dispatches 2,457 vs 2,839 (+16 %),
  operations 7,259 vs 7,728; exec busy +11 % in standalone. So: **same weather and place, standalone ~6-11 % more
  work**. Compare at matched execution-thread busy, not by fps.

### 5.2 Gap split, per presented frame (ours vs standalone)
| item | ours v4h2-k-tr1 | standalone v4h2-s-tr1 | source |
|---|---|---|---|
| frame (tsamp w1) | 162.7 ms (61.5/10 s) | 181.6 ms (55.1/10 s) | [M execfr w1] |
| exec busy (cycles; its spin counts as busy) | 134.0 ms | 148.8 ms | [M execfr w1] |
| exec idle gaps >= 1 ms / short idle (< 1 ms, partly busy ticks) | 20.4 / 8.3 ms | 22.9 / 9.9 ms | [M execfr w1] |
| **overhead = frame − exec busy** | **28.7 ms** | **32.8 ms** | [M] |
| fit frame = a + b·busy, value at b = 134 | 14.0 + 1.109 b → 162.6 | 0.6 + 1.217 b → 163.7 | [M execfr w1] |
| matched deciles (exec busy → frame) | 136.0 → 164.1; 140.4 → 170.0; 147.4 → 174.7 | 136.0 → 164.8; 140.4 → 169.6; 148.1 → 184.0 | [M execfr w1] |
| Kyty `execution_idle` / CP idle / resolve idle | 12.4 / 38.1 / 42.0 ms | 33.4 / 56.5 / 79.8 ms | [M PERF perfcmp, 2,344 / 2,240 frames] |
| **exec idle by link interval (relative to the flip's ready):** | | | [M gaplink w1] |
| – ready → done | 0.02 ms | 1.6 ms | |
| – done → the guest's first submission (any queue) | **6.6 ms** | 0.1 ms | |
| – first submission → first queue-0 submission | 0.03 ms | **7.5 ms** | |
| – after the first queue-0 submission | **13.6 ms** | **13.6 ms** | |
| – by gap start after ready: 0-20 / 20-60 / 60-120 / 120-180 / 180+ ms | 6.5 / 1.5 / 1.9 / 9.6 / 0.8 | 2.7 / 5.8 / 2.3 / 9.3 / 2.5 | |
| – by gap start after FRAMETIME (≈ 9 ms before ready on both): 0-20 / 20-60 / 120-180 ms | 13.1 / 2.2 / 3.1 | 7.5 / 4.9 / 5.0 | [M execfr w1] |
| flip: ready → present start / → done (mean) | 0.05 / 1.46 ms | 8.42 / 9.62 ms (waits for its 60-Hz tick) | [M HT, frames >= 3000] |
| guest waits on the flip | 1 rendering wait per flip; flip queue full 0 of 5,353; 0.85 interrupts per flip; exec flip-done wait 1 per frame, 3.3 µs | exec flip-done wait 1 per frame, 0.5 µs | [M adapter line, HT fw] |
| the guest's submissions after ready: first any / first q0, mean (median) | 25.4 (25.0) / 25.5 (25.0) ms | 4.2 (3.3) / 53.4 (49.5) ms | [M gaplink, HT anyr/q0r] |
| submissions per frame, q0 / other queues | 1.00 / 12.3 | 1.00 / 13.1 | [M HT] |
| GPU-queue waits on guest labels (WAIT_REG_MEM, detail frames 4001-4003) | 17 per frame, 207-232 ms summed over queues; same labels and pattern as standalone: cq38/cq48 on the frame counter 0x2000000020 ~23 ms, gfx 0x2000000160 ~21 ms, cq30 per-frame flag 67-84 ms, then four cq30/cq38 flags of 16-21 ms | 13-15 per frame, 184-294 ms (its detail frames are perturbed: 496 / 219 / 402 ms, the f4000 PrintWindow); plus a cq20 wait (~25 ms) ours does not show | [M synctl] |
| guest-thread faults, handler time | 2,446 per frame, 113.5 µs each, 277.7 ms per frame summed over threads | 2,353, 149.9 µs, 352.8 ms | [M HT fg] |
| exec-thread faults | 8.6 per frame, 9.3 µs each (0.08 ms) | 10.3, 4.2 µs (0.04 ms) | [M HT fe] |
| outer layer (AnyPS5's / Kyty's VEH, all threads) | 2,454, 113.1 µs each: nothing over the handler | 2,364, 149.4 µs | [M HT fx] |
| guest threads waiting for the exec thread to take a write-fault request | 181.9 ms per frame (120 requests with a download) | 255.9 ms (129) | [M PERF sync_time / readers invalidate] |
| protect-after-copy: changed / re-checked ranges / commits | **17.2** / 253 / 11.5 | **4.9** / 250 / 10.0 | [M HT pac, PERF protect_batch] |
| tracker protections per frame, exec / other threads (µs each) | 1,175 (3.5) / 1,626 (2.9) | 1,212 (3.2) / 1,429 (2.0) | [M PERF locks] |
| job threads busy / guest main thread busy | 152-160 ms (70 % of IP samples in the poll loops 0x152a550 / 0x152ad10) / 133 ms | 246 ms / 128 ms | [M execfr, tana w2] |

**Which items make up the ~8 ms fixed overhead and the 20-60 ms starvation [M + E]:**
- **In this pair there is no fixed overhead on our side.** At the same execution-thread busy, ours' frame equals
  standalone's within ±4 ms (matched deciles), and ours' overhead (frame − exec busy) is 28.7 ms against standalone's
  32.8 ms. Ours' overhead is stable across runs (G2 v4m-k-prof 31.4 ms, now 28.7). Standalone's moved from 19.4 ms
  (v4m-s-prof: sunny, camera A/B, the clang2 exe) to 32.8 ms (now: overcast, on the canoe). **The ~8 ms of §3 was the
  standalone run, not our adapter** [E: the side whose overhead changed is the side whose scene (sunny → overcast,
  camera) and exe (clang2 865cdd5e → 956de14e) changed; ours did not move across its DLL/adapter change 8b7c92c0 →
  f8cb45e7]. Caveat: ours has not been traced in the sunny scene, so a scene-specific few ms cannot be excluded; G2
  §2.6's same-scene ft.py comparison (×0.98, 2-3 %) remains the upper bound for any adapter share.
- **The 20-60 ms starvation moved with the scene as well:** ours 6.4 ms (G2) → 2.2 ms now; standalone 0.3 → 4.9 ms
  (FRAMETIME-relative, as in §3).
- What the remaining idle is, on both (the same size, so not a gap):
  - (a) **~7 ms per frame at the frame start: the exec thread has drained the previous frame and waits for the guest's
    next submissions.** Ours: nothing arrives until 25 ms after ready, and the exec thread is busy 14.9 ms of that
    with the previous frame's tail. Standalone: compute arrives at 4 ms, queue 0 only at 53 ms, and it waits between.
  - (b) **~13.6 ms after the first queue-0 submission, mostly 120-180 ms after ready.** The async queues block on
    per-frame flags the guest's job threads write (cq30 67-84 ms, then four 16-21 ms waits). Meanwhile those job threads
    sit in write faults waiting for the exec thread to take their request: 10-13 invalidate requests per detail frame,
    each queued ~10 ms behind the exec thread's current operation (frame 4002 at 141.9 ms: ten job threads on page
    0x50316b8000, queue_us 10.0-10.1 ms) [M synctl]. This is Kyty's write-fault service on the execution thread (v4e's
    code). v4e's census §3.5 already shows it shrinking with the GPU copy: guest waiting for request pickup 218 → 44-67
    ms per frame.
  - (c) ~8-10 ms of sub-ms idle.
  - (a) is the guest's own pacing (v4g: the guest spins on end-of-frame values), (b) is v4e's area.
- The flip path is not a cost: ours delivers done 1.5 ms after ready (standalone 9.6 ms), the flip queue is never
  full, and the exec thread's flip-done wait is 3 µs per frame. AnyPS5's outer fault layer adds nothing measurable
  (fx ≈ fg per fault). Fault counts differ by +4 % (scene).

### 5.3 The protect-after-copy re-checks that find changed bytes (×13 in G2, ×3.5 here)
- Mechanism [C `src/graphics/host_gpu/renderer/cache/bufferCache.cpp:2399-2484` on v4/h]: a commit stages each
  CPU-modified range while it is still writable (`ReadGuestPlain` + XXH3), records the upload, runs
  `CommitDeferredProtections`, then hashes the range again. A different hash means **someone wrote the bytes between the
  staging read and the protect**: the range is marked CPU-modified again and re-uploaded before its next use. Correct
  either way; it costs one re-upload.
- Which pages [M HT_PAC, HT_PAC_SAMPLE, frames >= 3000; 661 vs 175 samples]: ours 0x2004xxxxxx (55 %) and 0x200bxxxxxx
  (44 %); hottest 64-KiB pages 0x2004520000-0x2004560000, 0x200bc20000-0x200bc70000, 0x200bdb0000. Standalone: the same
  two families at other addresses (0x20048b0000-0x20049b0000, 0x200c150000-0x200c1c0000; 52 % / 40 %). They are **the
  same two guest data rings** (v4e's census names them: 0x20028c0000 +74 MiB and 0x2009fc0000 +76 MiB in ours). Kyty's
  direct ranges (0x2010000000-0x2015e00000 by default [C bufferCache.cpp:1534]) do not cover them, so these are
  ordinary uploaded buffers.
- Who writes [M samples]: the first changed 8 bytes sit in the **last 4-KiB page of the range in 65 % (ours) / 68 %
  (standalone)** of the samples, at offset 0 in 31 % / 27 %, mid-range in 2 % / 3 %. The values are float pairs
  (constants, transforms) or turn to 0. That is the ring's write head: **guest CPU code** (job threads filling the next
  draws' data) writes into the page it shares with the range being uploaded. Not the GPU: GPU writes land in Kyty's
  buffers, not in these guest pages (no direct range here), and Kyty's own downloads run on the commit's thread.
- Does ours re-protect something standalone does not? **No** [M]: re-checked ranges 253 vs 250 per frame, commits 11.5
  vs 10.0, protections on the exec thread 1,175 vs 1,212; tracker protections that cross AnyPS5's views 0.06 per frame
  (§1). The difference is only how often the guest is writing the ring's current page during the commit window: ours
  has changes in 69 % of frames, standalone in 42 %. Per frame, the count correlates with the frame being faster
  (r −0.23) and not with the submission timing (q0r, r 0.02) [M]. The commit takes as long in both (0.32 vs 0.30 ms per
  commit), so the window is not longer in ours; the guest's ring writes overlap the commits more often [E].
- Cost [E]: ~0.4 MiB more re-uploaded per frame than standalone (detail frames: changed_kb 590-746 vs 193-340), about
  0.1 ms of copy and hash on the exec thread, under 0.1 % of the frame. **Nothing to fix.**

### 5.4 Ceiling [E]
- Model: frame ≈ exec busy + overhead. Ours' overhead is 28.7 ms (31.4 in G2) and flat across the exec-busy deciles
  (idle 18-26 ms for exec busy 115-169 ms) [M]. v4e-ab1 shows the 1:1 part in-run: "both" switches took ~30 ms of CPU
  BAR reads off the exec thread, and the median frame fell 162.1 → 133.5 ms (−28.6 ms) [M v4e-ab1].
- **If exec busy goes 134 → ~95 ms** (most of BAR readback and part of the readback service gone): frame ≈ 95 + 28.7 ≈
  124 ms → **≈ 81 presents/10 s**; the fitted slope (1.109) gives 119 ms → 84 [E]. With v4e's switches as measured
  today (+21.9 % clean), this pair's scene goes 61.5 → **≈ 75/10 s** [E, scaled].
- What limits it then, in order [E]:
  1. the exec thread, still the longest stage at ~95 ms: the remaining readback service (v4e "both": ~29 ms per frame,
     of which SRT-walker waits for GPU results 12-14 ms) and the per-operation cost (~7,260 operations per frame);
  2. the ~29 ms overhead: the frame-start wait for the guest's submissions (~7 ms); the label waits whose writers sit in
     write-fault requests behind the exec thread (~13.6 ms; this should shrink when the write-fault service gets faster,
     so 81 is the conservative end); sub-ms idle (~8 ms);
  3. not limiting at ~120 ms: record and resolve show ~110-117 ms of cycles per frame, but mostly spin (78 % and 37 % of
     their IP samples in pause buckets [M tana w2]: real work ~25 / ~70 ms); the GPU (~61 ms busy per frame, G2
     timestamps); the guest's own work (v4g: main ~13-21 ms, job code 8-27 ms of non-spin work per frame).

### 5.5 Fixes on our side
- **No cause found in the adapter, the flip path, AnyPS5's memory notifications or the aps5 glue** that costs time on
  the critical path (§5.2): flip done 1.5 ms after ready, outer fault layer ≈ 0, re-check count equal, frame equal at
  matched exec busy. So there is no new switch to implement.
- The two adapter switches v4h built (`protect-srw`, `lean-counters`, default off; local/v4-h 519f59c6a2a) target the
  only measured adapter-side extra: protections cost 3.5 vs 3.2 µs on the exec thread (+0.4 ms per frame, ~0.25 % [E])
  and 2.9 vs 2.0 µs on guest threads (off the critical path). Expected gain < +1 % [E]. They get one in-run A/B in the
  diagnostic run (§6) so they can be closed with a number.
- Leads for v4e (Kyty's execution-thread code; described, not changed):
  - the write-fault requests of the guest's job threads queue ~10 ms behind the exec thread's current operation while
    the async queues wait on labels those threads write (§5.2 (b));
  - deferred end-of-pipe labels are published by the GPU thread at packet boundaries [C
    `src/graphics/guest_gpu/graphicsRun.cpp:505-511`], so one long exec-thread operation (a BAR read of up to 11 ms,
    v4e §3.5) also delays every label the guest waits on: `label_at_completion` lag 5.7 ms average in ours, 5.2 ms
    standalone [M PERF]. BAR removal should shorten both; worth reading in v4e's next census.

## 6. Diagnostic run v4h3-k-ab1 (queued 09:50)
- `research/v4/h/chain3.sh` (log `logs/chain3.log`), started detached 09:50:16 (chain pid 380548, side job 380679,
  run1080 380682 → run_when_free.sh 380716, waiting in the game queue behind v4x-w-diag, v4o-b2, v4r-k-w03 and
  v4e-diag1). `research/v4/h/STOP3` stops it before the run.
- Build: `research/v4/h/bview` = local/v4-h 519f59c6a2a `build-local` as hard links (core/libs/libs, core/libs/prx,
  relinker.exe, CMakeCache.txt), with only `core/libs/libs/kyty_gpu.prx` a real copy of the 5fcfc57e DLL (a327d946);
  build-local's own DLL is unchanged (f8cb45e7).
- Env: `APS5_GPU_BACKEND=kyty APS5_KYTY_APS5_TRACE_CXX_THROW=1 APS5_KYTY_TRACE_FRAMETIME=1
  APS5_ADAPTER_AB=protect-srw+lean-counters APS5_ADAPTER_AB_PERIOD_S=30`, run1080 long, 900 s. The throw tracer is a
  last vectored handler that only acts on 0xE06D7363 [C 5fcfc57e kytyBackend.cpp], so it does not touch the tracker
  faults and costs both A/B arms the same.
- What is known before the run [M v4h2-k-tr1, v4h2-k-w1 err.logs]: one `[crash] exception 0xe06d7363 (flags 0x81) at
  KERNELBASE.dll+0xc408a` per run, on a short-lived thread (5804 in tr1, 16364 in w1; not alive later, absent from
  tsamp's thread list), during frame 1 (between the first `FIX_DCC_KNOWN_KEYS` lines). 0xE06D7363 is MSVC's C++ throw
  code; our MinGW/llvm-mingw builds (AnyPS5, Kyty) throw with the GCC code, so the thrower is an MSVC-built module in
  the process (driver, loader, OS component) [E].

### 6.1 Result (run 10:38-10:57; chain3 done 10:57, nothing of this lane left running)
- Run [M v4h3-k-ab1]: 900 s, fatal none, fatal-pattern lines 0, RESTORE OK, RUN1080 OK, runtime manifest identical;
  5,239 presents, **61.15 presents/10 s** ([present] metric, SE 0.58), 60.94 ft.py; no pipeline compile on the GPU
  thread after present 1000; tracker faults 2,396 per present. Captures `shots/v4h3-k-ab1-f{1600,4000}.png`.
- **The first-chance throw** [M]: the tracer logged two throws, both right after the first present (FRAMETIME
  frame=1), and AnyPS5's `[crash]` reporter logged the first (same thread 5180):
  - `MONZA::DdiThreadingContext<AdapterTraits_Gen12LP>::msg_end` on thread 5180 and
    `MONZA::IgcThreadingContext<AdapterTraits_Gen12LP>::msg_end` on thread 5532;
  - both raised at `igd10um64xe.DLL+0x1061a3` (→ KERNELBASE RaiseException) on threads that start in
    `igd10um64xe.DLL+0x117a2f` via `KERNEL32` BaseThreadInitThunk; no Kyty, AnyPS5 or game frame on the stack (the
    first frame, kyty_gpu.prx+0x5da1c4, is the tracer itself).
  - So the thrower is **Intel's iGPU D3D user-mode driver** (igd10um64xe = Intel Graphics D3D10/11 UMD; Gen12LP = the
    i9-11900H's UHD graphics): two of its worker threads (the D3D DDI threading context and the IGC shader-compiler
    context) end by throwing the driver's own `msg_end` message, which the driver catches itself.
  - **Harmless** [M]: first chance only, no second chance, the run goes on for 900 s with no fatal, and the same single
    `[crash]` line appears in every Kyty-path run (tr1, w1, this one) without any effect.
  - Why only on the Kyty path [E, not verified]: the laptop panel is driven by the Intel iGPU, so a present from the
    NVIDIA Vulkan device goes through Windows' cross-adapter path, which keeps a D3D device on the Intel adapter. The
    Kyty path creates its own Vulkan device and swapchain on AnyPS5's window; a cross-adapter context torn down or
    replaced around the first present ends those two threads. Nothing to fix on our side; AnyPS5's `[crash]` line
    for it could be filtered by type if it confuses readers.
- **Adapter switches A/B** [M v4h3-k-ab1, `tools/absplit.py`]: `protect-srw+lean-counters` vs off: **+0.15 % per cycle
  (SE 1.22 %, 10 complete 30-s cycles, faster in 5 of 10)**; pooled 60.92 vs 60.90 presents/10 s (1,631 / 1,633
  presents, 268 s each). Under the +3 % stop rule: **leave both off** (they stay in local/v4-h as default-off
  switches; no merge needed for speed).
