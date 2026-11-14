# v4o (PLAN v4 phase 4): a clang PGO build of the Kyty DLL, and thread placement on this CPU (2026-11-09)

Lane v4o. Status: **DONE, handed back 2026-11-09 ~12:50.** Steps 1 and 2 are measured; step 3 (`-march=native`) is built
but not measured (§0). Nothing of this lane runs. PLAN, STATE, HANDOFF and upstream/ are untouched; nothing pushed or
posted (this lane changed no AnyPS5 code, so there is no `local/v4-o` branch to back up). Files: `research/v4/o/`.
Tags: [M] measured (run tag named), [E] estimate (reasoning named).

## 0. Answer
- **PGO: above the +3 % bar; worth turning on (CMake `KYTY_PGO=USE`, default OFF).** A-B-A-B-A, 900-s world runs, the
  G2 DLL against the PGO DLL of the same source [M v4o-a1, b1, a2, b2, a3]:
  - +5.3 % in the one scene-matched view (B2 vs A3, frames 4000-5000: 70.2 vs 66.7 presents/10 s; +3.8 % in 3000-4000);
  - +8.9 % in the frame window all five runs share (1500-3000: B 61.6, A 56.6);
  - +10.4 % over the whole runs (B 65.1, A 59.0; scenes not matched).
  The camera alone moves a run of the same DLL by ~11 %, so the true gain lies around +4…10 %, above the bar in every
  reading. Gates: T1 byte-identical to `t1-k-1080` and MATCH against `t1-k-1080-dense`; switch-off T1 MATCH
  `t1-ref-1080`; five world runs with no fatal.
- **Placement: below the +3 % bar; leave it off.** Three in-run A/Bs (1,500 s each, 30-s phases) [M v4o-pl1, pl2, pl3]:
  - the only layout that gains opens the siblings of the recording, resolve and CP cores (CPUs 5, 7, 9) to every
    other thread: `default:0,1,5,7,9,10-15` **+2.4 % and +1.0 %** (20 cycles, mean +1.7 %), and the same without core 7
    `default:0,1,5,7,9,10-13` **+3.5 % and +1.7 %** (18 cycles, mean +2.5 %). Consistent (faster in 7-8 of 10 cycles)
    but under +3 %;
  - flat or worse: the BAR copy helpers on free siblings (+0.6 %/−0.8 %), all four siblings open (−1.5 %), a core left
    idle for clock (−1.9…−3.4 %), record and resolve sharing a core to free one for the guest (−0.9 %), the guest's job
    workers on the stage siblings (−3.6 %), the CP on the exec core's sibling (−6.3 %).
  - Kyty's G2 placement (stages pinned to whole cores, their siblings reserved) is right for this CPU within ~2 %.
  - The switch stays in the DLL, default 0 (G2), as the tool for re-checking layouts when the frame changes (e.g. after
    v4e's BAR work). Placement does not change pixels (T1 MATCH with a layout on).
- `-march=native` (report only): built (`out/native/kyty_gpu.prx`, md5 fb173d3627f7c44d20abbed385672dc4), **not
  measured**. One run cannot resolve a few-% effect when the scene alone moves runs by ~11 %, and an A-B-A-B-A costs
  ~1.5 h of the shared game slot for a non-portable number.
- **Side finding for all phase-4 lanes:** in the world the package runs at **~3.0 GHz** (`% Processor Performance`
  111-128 % of the 2.5-GHz nominal on every logical CPU) [M during v4o-pl1]; Tiger Lake's cores share one clock.

## 1. What was built
- Kyty branch **`v4/o`, tip 6e0837f2**, worktree `C:/ps5/kyty-src-wt/o`, on `v4/m1b` 72f2250b (the G2 DLL's source),
  not pushed; commits as Celegans12, `AI-assisted: yes`, no trailer:
  - **564440c2 `Build: KYTY_PGO, clang IR profile-guided optimisation (default OFF)`.** CMake cache options
    `KYTY_PGO` = `OFF` (default) | `GEN` | `USE`, `KYTY_PGO_DIR` (GEN: where .profraw files go, default
    `<build>/pgo`), `KYTY_PGO_PROFILE` (USE: the merged .profdata). Flags go to everything built after
    `config_compiler_and_linker()` (3rdparty included): GEN `-fprofile-generate=<dir>` (compile and link), USE
    `-fprofile-use=<file> -Wno-profile-instr-unprofiled`. Clang with the GNU driver only. In GEN builds the DLL writes its
    counters every `KYTY_PGO_WRITE_SECONDS` (env, default 60; `APS5_KYTY_PGO_WRITE_SECONDS` in a run; 0 = only at exit)
    into `KYTY_PGO_DIR` (env override of the configure-time dir) as `kyty_gpu-<pid>-<seconds>s.profraw`
    (`src/aps5/pgoWriter.cpp`, called from `Start` after the settings), because AnyPS5's runs end by killing the process
    and the profile runtime writes only at a clean exit. Without GEN the writer is an empty function.
  - **6aa3c400 + 6e0837f2 `Common: KYTY_PERF_PLACEMENT …`** (`src/common/gpuPlacement.{h,cpp}`, hooks in
    `emulatorConfig.cpp` `ConfigureGpuStageThread`, the GPU thread's frame boundary in `graphicsRun.cpp`, the BAR copy
    helpers in `bufferCache.cpp`, `liveAb.{h,cpp}`):
    - runtime switch **`KYTY_PERF_PLACEMENT=<n>`, default 0 = the G2 start placement, where nothing new runs** (one
      once-read check per flip);
    - layouts `KYTY_PLACEMENT_LAYOUT_1..4` = fields joined by `+` (or `/`): `stages:<e>,<r>,<v>,<c>`,
      `default:reserve|all|<cpus>` (process default CPU sets), `barcopy:<cpus>` (one CPU each), `jobs:<cpus>` (CPU sets
      of guest threads named `KYTY_PLACEMENT_JOB_PREFIX`*, default `JobWorker`), `prio:0|1`; a field left out keeps G2's
      choice;
    - live: `KYTY_LIVE_AB` may list `KYTY_PERF_PLACEMENT=<n>` once per layout (a new per-value switch kind,
      `SwitchInfo::per_value`); the GPU thread applies a phase's layout at its frame boundary (70-330 µs; ~30 ms when a
      jobs field snapshots the threads), and layout 0 restores G2 exactly (stage pins and priorities, the reserved
      default sets, cleared helper/guest CPU sets).
- Build script `research/v4/o/build-gpu.sh <name> <log> [cmake -D…]` (v4b's recipe: llvm-mingw 20261104 clang 23.1.5,
  ThinLTO, libc++, native TLS, `-O3 -march=x86-64-v3 -ffp-contract=off`; refuses an uncommitted tree; `NOCCACHE=1`).
  DLLs (stripped) in `research/v4/o/out/<name>/kyty_gpu.prx`:

  | name | what | md5 |
  |---|---|---|
  | base | v4/o 6e0837f2, no PGO | 7d2b6d7f84a72b162577e0681150365d |
  | gen | KYTY_PGO=GEN (instrumented; training only) | 655180f4f02e724789c74d2f17d8016b |
  | **pgo** | **KYTY_PGO=USE with `pgo/kyty_gpu-6e0837f.profdata`** | **54f7c11800138d82a66aa3226e62b350** |
  | native | no PGO, `-march=native` (Tiger Lake) after x86-64-v3 | fb173d3627f7c44d20abbed385672dc4 |

- Run views `research/v4/o/bd-<name>` = hard-linked copies of the G2 build's `core/libs/libs`, relinker and
  ShaderCacheVersion with only `kyty_gpu.prx` replaced (`bd-g2` keeps the G2 DLL 8b7c92c0…). Nothing ran from the G2
  worktree or its build dir. Chains `research/v4/o/chain{A,C,D,E,F,H}.sh` (+ `chain-lib.sh`), logs in `logs/`.

## 2. PGO

### 2.1 Training (v4o-gen1, 11-09 04:07-04:22)
- The instrumented DLL in view `bd-gen`, `run1080.sh long`, `AB_SECONDS=900`: boot, the title intro, the save load and
  ~800 s in the river village (web-teshio-b, pad-nojump), `APS5_KYTY_PGO_WRITE_SECONDS=60`. Never a measured run.
- 15 files (60 s … 900 s, 6 MB each, 2-9 ms per write) in `research/v4/o/pgo/raw/`; killed at 900 s as usual (RESTORE OK,
  no fatal). The 900-s file → `llvm-profdata merge` → **`research/v4/o/pgo/kyty_gpu-6e0837f.profdata`** (4.2 MB; 12,477
  functions, 148,740 blocks).
- Instrumented speed: 39.4 presents/10 s in the world ([present] metric) against ~61 for G2 [M v4o-gen1].
- Hottest functions: the per-draw paths (`SrtTapeWalker::Eval`, `ReadLog::Clear`, `RenderExecutor::CommitBindings` /
  `PlanFlags`, `FastVerifyLoggedReads`, `ClampRangeSizeAt`), BAR readback (`ScanBarDownload`, `WaitPendingReadbacks`)
  and the stage threads' `SpinWait::Spin`; the load phase does not dominate.

### 2.2 The PGO build
- `build-gpu.sh pgo … -DKYTY_PGO=USE -DKYTY_PGO_PROFILE=C:/ps5/work/research/v4/o/pgo/kyty_gpu-6e0837f.profdata`, the
  same commit, no ccache: rc 0, 13.4 MB (G2 13.2 MB), same imports as G2.
- clang matched the profile to every function but two (`function control flow change detected (hash mismatch) … count
  discarded [-Wbackend-plugin]`): `Aps5::StartPgoWriter` (its body exists only in GEN builds) and `TraceFrameTime`
  (swapchain.cpp, once per frame; why it differs between GEN and USE of one commit is not known).

### 2.3 A-B-A-B-A (G2 DLL / PGO DLL; run1080.sh long, 900 s each; ft.py rule)
| run | DLL | presents/10 s, frames 1500-end | scene (captures at frames 1600 / 4000 / 5000) | notes |
|---|---|---|---|---|
| v4o-a1 (07:23) | G2 8b7c92c0 | 57.72 (62 windows, SE 0.87) | sunny; canoe level, the village huts in view | no compile in the world |
| v4o-b1 (07:50) | PGO 54f7c118 | **63.69** (62, SE 0.77) | overcast/misty; canoe level | 1 compile (1.2-s frame, present ~2,200) |
| v4o-a2 (08:17) | G2 | 55.57 (60, SE 0.37) | sunny; facing the village, then down river (a view no other run had) | 5 compiles (1.2- and 3.5-s frames, presents 2,080-2,440) |
| v4o-b2 (09:57) | PGO | **66.48** (61, SE 1.48) | sunny; canoe level, then angled with the steps at the right | 13 compiles at present ~2,404 (one 8.6-s frame), none after |
| v4o-a3 (11:16) | G2 | 63.63 (61, SE 1.23) | sunny; facing the village, then B2's view (steps at the right) | no compile in the world |

Same frame ranges (`tools/ftwin.py`), presents/10 s:

| run | 1500-3000 (jump presses, dumps) | 3000-4000 | 4000-5000 | 3000-5000 |
|---|---|---|---|---|
| A1 G2, sunny | 55.08 | 59.37 | 59.94 | 59.67 |
| B1 PGO, overcast | 61.72 | 64.55 | 65.22 | 64.89 |
| A2 G2, sunny, other camera | 55.06 | 55.84 | 56.23 (to 4926) | 55.99 |
| B2 PGO, sunny | 61.45 | 68.25 | 70.18 | 69.16 |
| A3 G2, sunny | 59.52 | 65.74 | 66.65 | 66.19 |

[M; all five: fatal none, RESTORE OK, in-run CPU samples clean (foreign 31-77 % of one CPU, largest single ≤ 13 %).]
- **The camera moves the number as much as the build.** A1 and A3 are the same DLL and weather; from frame 4000 they
  differ by 11 % (59.9 vs 66.7): A1 looks at the village huts, A3 and B2 at the steps with one hut at the edge
  (`shots/v4o-a1-f5000.jpg`, `v4o-a3-f5000.jpg`, `v4o-b2-f4000.jpg`). The first three runs had no matched pair (A sunny,
  B1 overcast), hence B2 and A3.
- **Scene-matched pair B2 vs A3: +5.3 % (frames 4000-5000), +3.8 % (3000-4000).** Shared window 1500-3000: +8.9 %. Whole
  runs: +10.4 %. The lowest B (B1 63.7) equals the highest A (A3 63.6): the scene spread is the size of the effect, so a
  tighter number needs a traced pair (execution-thread busy per draw, `KYTY_TRACE_PERF`), not more untraced runs.

### 2.4 Where the profile lives, and how it is refreshed
- Profile: **`C:\ps5\work\research\v4\o\pgo\kyty_gpu-6e0837f.profdata`** (4.2 MB, from v4o-gen1's 900-s .profraw in
  `pgo/raw/`). It is not in the Kyty tree; a build passes it with `-DKYTY_PGO=USE -DKYTY_PGO_PROFILE=<path>`.
- **Staleness.** clang's IR PGO keys each function's counts by name and a hash of its control-flow graph. A function
  whose code changed since the profile gets no data (one `hash mismatch … count discarded` warning each in the build
  log), new functions get none, and every unchanged function keeps its counts whatever changed elsewhere. A stale
  profile never makes a build wrong, and at worst leaves the changed functions as in a no-PGO build. The v4 lanes each
  change a few hot files (v4e `bufferCache.cpp` BAR paths, v4x draw and shader paths, v4h flip and trace paths), so after
  each merge into `v4/int` the mismatch count in the USE build's log says how much of the profile is lost.
- **Refresh: `research/v4/o/pgo-refresh.sh <kyty worktree> <name>`** (launch detached; ~35-40 min of machine: two guarded
  ~8-min builds and one 15-min run): KYTY_PGO=GEN build of the committed tip → one `run1080.sh long` 900-s training run in
  a view of it (`APS5_KYTY_PGO_WRITE_SECONDS=60`) → `llvm-profdata merge` of the last complete .profraw →
  `pgo/kyty_gpu-<git hash>.profdata` → KYTY_PGO=USE build of the same tip. Refresh at each integration merge that touches
  the execution thread's hot paths, or when the mismatch count grows; the training run is never measured. Warm the
  pipeline cache with the new DLL's first run (a new revision precompiles other builds' journal entries, 923 pipelines
  in 89 s on its first start).

## 3. Placement

### 3.1 Knobs
- Kyty's own, all start-time env: `KYTY_GPU_THREAD_CPUS=<e>,<r>,<v>,<c>|off` (stage pins; default from
  `ChooseGpuStageCpus`, here 2, 6, 8, 4), `KYTY_GPU_THREAD_MIN_CORES` (the reservation needs this many other physical
  cores; default 4: reserved here), `KYTY_GPU_THREAD_PRIORITY=0` (exec and CP not raised), `KYTY_GPU_SPIN_US` (stage spin
  budget before a kernel wait, default 2,000). The BAR copy helpers (2, `BarCopyThreads`) and the pipeline-compile
  threads are not pinned. AnyPS5 applies no guest affinity (scePthreadSetaffinity only stores it) unless
  `APS5_JOB_AFFINITY_MASK`.
- New: `KYTY_PERF_PLACEMENT` + `KYTY_PLACEMENT_LAYOUT_1..4` (§1), live under `KYTY_LIVE_AB`.

### 3.2 This CPU under the game
- Logical CPUs 2k and 2k+1 are one physical core. G2: stages on 2/6/8/4 (cores 1, 3, 4, 2), process default CPU sets
  {0, 1, 10-15} (cores 0, 5, 6, 7), the siblings 3/5/7/9 idle but for other processes.
- The package runs at ~3.0 GHz in the world (`% Processor Performance` 111-128 % of nominal, every logical CPU; exec
  CPU 2: 128 %, utility 111 %) [M 10 × 1 s during v4o-pl1]. Tiger Lake's cores share one clock.

### 3.3 Three in-run A/Bs (bd-base, 1,500 s, 30-s phases from frame 1500, 10 warm-up frames, alternating order)
| run | layout | vs off, all cycles (SE; faster in) | vs off without cycle 0 (SE; faster in) |
|---|---|---|---|
| pl1 | `barcopy:5,9` (BAR copy helpers on free siblings) | +0.55 % (1.46; 3/10) | −0.79 % (0.65; 2/9) |
| pl1 | **`default:0,1,5,7,9,10-15`** (record/resolve/CP siblings open; the exec core's sibling stays empty) | **+2.43 %** (1.40; 7/10) | +1.23 % (0.81; 6/9) |
| pl1 | `default:0,1,3,5,7,9,10-15` (all four siblings open) | −1.46 % (0.83; 4/10) | −1.79 % (0.86; 3/9) |
| pl2 | `default:0,1,10-13` (core 7 left idle, for clock) | −1.93 % (1.83; 1/8) | −3.38 % (1.30; 0/7) |
| pl2 | **`default:0,1,5,7,9,10-13`** (as above without core 7) | **+3.48 %** (1.43; 6/8) | +2.57 % (1.27; 5/7) |
| pl2 | `jobs:3,5,7,9` (the 20 JobWorker/JobWorkerLow threads on the siblings) | −3.57 % (1.45; 1/8) | −4.28 % (1.46; 0/7) |
| pl2 | `stages:2,6,8,3` (CP on the exec core's sibling; core 2 to the rest) | −6.27 % (1.13; 0/8) | −6.67 % (1.22; 0/7) |
| pl3 | **`default:0,1,5,7,9,10-15`** (repeat) | **+1.00 %** (0.54; 7/10) | +1.31 % (0.49; 7/9) |
| pl3 | **`default:0,1,5,7,9,10-13`** (repeat) | **+1.65 %** (0.60; 8/10) | +2.02 % (0.53; 8/9) |
| pl3 | `stages:2,6,7,4+default:0,1,5,8-15` (record and resolve share core 3; core 4 to the rest) | −0.90 % (1.29; 4/10) | +0.18 % (0.81; 4/9) |

[M v4o-pl1 (10 cycles), v4o-pl2 (8), v4o-pl3 (10); `tools/labtab.py`. Off phases 5.74 / 5.67 / 5.87 fps; whole runs
57.5 / 55.7 / 59.0 presents/10 s from frame 1500; no fatal, no compile in the world; RESTORE OK.]

Reading: the guest wants more CPUs (opening the record/resolve/CP siblings helps a little; idling a core or crowding the
job workers onto siblings hurts), and the exec core wants its sibling empty (opening it, or putting the CP there, hurts).
The best layout repeats at +1.0…+3.5 % per run, **mean +1.7 % (`…10-15`, 20 cycles) and +2.5 % (`…10-13`, 18
cycles): under the +3 % bar, so it stays off.** Placing the BAR copy helpers is flat.

## 4. Gates
- **PGO DLL, switch on, placement off: T1 identical [M v4o-pgo-t1].** `run1080.sh bd-pgo t1 … APS5_GPU_BACKEND=kyty`:
  all 60 frames **byte-identical to `t1-k-1080`** (v4r's registered Kyty reference = v4r-k-t1a, the G2 DLL), and t1align
  against `t1-k-1080-dense`: **MATCH (phase-tolerant, exact)**, offsets +0..+8. Index by index against `v4b-k-t1e` it
  reads DIFFERS, which is only the T1 start offset v4r found (`reports/2026-11-09-v4-refs.md` §1: v4b-k-t1e had offset 5,
  today's runs 8). 595 presents, fatal none, RESTORE OK, RUN1080 OK.
- **Placement switch on: T1 identical [M v4o-plt1].** The PGO DLL with `APS5_KYTY_PERF_PLACEMENT=1
  APS5_KYTY_PLACEMENT_LAYOUT_1=default:0,1,5,7,9,10-15` from the first flip: **VERDICT: MATCH against t1-k-1080,
  tolerance 0, frames 0-59**; t1align MATCH. Placement does not change pixels.
- **Switch off (old backend, the PGO DLL loaded): T1 MATCH [M v4o-pgo-off]** against `t1-ref-1080`, tolerance 0,
  frames 0-59; 542 presents, RESTORE OK.
- **World runs with no fatal:** v4o-b1, v4o-b2 (PGO DLL, 900 s each), v4o-pl1/pl2/pl3 (placement code with layouts
  switching every 30 s, 1,500 s each): no fatal, no device loss, RESTORE OK, RUN1080 OK.
- **The layout really applies [M tsamp, v4o-pl3]:** 8 s into a layout-1 phase, the process default CPU sets read 256, 257,
  261, 263, 265, 266-271 = CPUs 0, 1, 5, 7, 9, 10-15; stage threads unchanged (exec affinity 0x4 priority 2, record 0x40,
  resolve 0x100, CP 0x10 priority 2) (`research/v4/o/tsamp/v4o-pl3/`). (In v4o-plt1 the tsamp came before the first flip
  and still saw the start sets.)

## 5. Open items and notes for other lanes
- **Merging** (`git merge-tree` against the lanes' tips at 09:10): `v4/o` with `v4/x` merges clean; with `v4/e` it
  conflicts in `liveAb.cpp`/`liveAb.h` (both append switches to the enum and the table: keep both; `SwitchInfo`'s new
  fourth field `per_value` has a default, so v4e's entries stay valid); with `v4/h` it conflicts in
  `src/aps5/kytyBackend.cpp` (both add lines near the includes and `Start`: keep both).
- **Turning PGO on in `v4/int`:** take 564440c2 (and the placement commits or not), then run `pgo-refresh.sh` on the
  integrated tip so the profile matches it; ship the USE build. The profile is per source; a USE build with an older
  profile is still correct.
- **For v4e (BAR readback):** per ~5 s of world (G2 profile run v4m-k-prof, `PERF bar_copy_sizes`): 1,842 reads < 4 KiB
  (22 ms), 2,948 of 4-64 KiB (190 ms), **154 of 64 KiB-1 MiB carrying 52 MB in 1,180 ms (~44 MB/s)**, none ≥ 1 MiB. Only
  reads ≥ 256 KiB use the two `BarCopyThreads` helpers (3 streams with the caller); where the helpers run is flat
  (pl1), so their number and the 256-KiB threshold are the lever, if CPU reads of ReBAR stay at all.
- **For every phase-4 lane:** the package clock is shared and runs at ~3.0 GHz in the world; work anywhere in the process
  can slow the execution thread. Measure levers on the whole frame.
- Not done: `-march=native` measurement (§0); a traced PGO pair for a scene-independent number (§2.3); the
  `TraceFrameTime` GEN/USE hash mismatch.

## 6. Machine state at hand-back (12:50)
- Nothing of this lane runs: chains A, F, H finished; C, D, G were stopped by me by PID while they only waited (C after
  its v4o-pl1; D in a pre-run CPU sample, before any run was queued; G before it started anything); E stopped itself on
  a stale name (D's pre-run sample file, renamed `logs/v4o-pl2.cpu.aborted-chainD.txt`). No eboot, tsamp, guard or wait
  loop of mine is left; I deleted no lock or marker. `research/v4/o/STOP` was removed after chainA ended.
- All runs RESTORE OK, RUN1080 OK (runtime manifest identical). Runs of this lane: v4o-gen1, v4o-pgo-t1, v4o-pl1, v4o-pl2,
  v4o-a1, v4o-b1, v4o-a2, v4o-plt1, v4o-pgo-off, v4o-b2, v4o-a3, v4o-pl3.
- Shared state changed: Kyty's cache `shader_caches/22c401ea7ca3796f/kyty` grew with these runs (the new revision
  6e0837f's warm-list section; warm list 11,765 → 20,730 with all lanes).
- Disk: C: 89 GB free. This lane: ~3 GB in `research/v4/o` (build dirs b-base, b-pgo; out/ incl. unstripped gen and
  native DLLs; deps; views). Deleted (manifest `logs/cleanup-manifest.txt`): my build dirs b-gen (1.3 GB) and b-native
  (1.0 GB) after keeping their unstripped DLLs.
