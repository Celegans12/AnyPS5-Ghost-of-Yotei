# v4g (PLAN v4 phase 0, step 0d): the guest-side floor on AnyPS5 and on KytyPS5-Yotei (2026-11-08)

Lane v4g. Same scene on both emulators: river village, web-teshio-b save, 1080p output, this laptop. Two runs per
emulator (one light, one traced). Nothing built in our tree, nothing posted. PLAN, STATE and HANDOFF are untouched.
Tags: [M] measured here (run tag and window named), [E] estimate (reasoning named).

## 0. Answer

1. **The guest's own work is not on AnyPS5's critical path.** It is not on Kyty's either. On both emulators the
   game's frame threads (main + ten job workers) spend most of their CPU in the game's own rdtsc back-off loops. These
   are the same four loops in the same game code. They wait for values written by the GPU or by the emulator.
   - CPU-weighted shares [M v4g-a2, v4g-k2, 5 % IP threshold]:
     - AnyPS5: main 98 %, jobs 94 %;
     - Kyty: main 90 %, jobs 48 %, plus 16 % in Kyty's own spin locks.
   - What is left per frame, everything that is not a spin loop (guest code + HLE + OS):
     - AnyPS5: main ~21 ms (guest code 7, libSceAgcDriver 5, OS 8); jobs ~74 ms (guest code 8, libSceAgcDriver 4,
       OS waits/wakes 60);
     - Kyty: main ~13 ms (guest code 4, OS 6, Kyty 2); jobs ~143 ms (guest code 27, OS waits 69, Kyty spin locks
       43, Kyty 3).
   - AnyPS5's HLE is not heavier than Kyty's on these threads. libkernel and libSceJobManager.prx are 0.0 %.
2. **AnyPS5's "guest link" is the backend, not the guest.** After the flip packet, queue 0 waits **102 ms** (median,
   p10 93, p90 132; v4g-a2 driver trace) for the guest's next submission. The untraced a1 shows the same ~100 ms; the
   older yb-next trace ggp-tr1 showed 144 ms. In that time:
   - the job threads are busy 1.5 ms per frame;
   - the main thread spins: 91.6 % of its samples are in the polling loops, the rest is the submit calls;
   - our queue-0x20 worker records the frame-start compute work: 256 dispatches, 88 ms busy, GPU-mutex waits
     included. Our ordering rule had held those ACBs behind queue 0 for ~300 ms;
   - the GPU runs them (48 ms busy in the link), the guest sees the label and submits 0x30/0x38/0x48 at ~95-101 ms,
     then queue 0 of frame N+1.
   - The GPU had finished frame N's own work 3 ms after the flip.
3. **Kyty's link is 11 ms.** The execution thread runs dry ~8 ms after each present and resumes ~19.5 ms after it:
   - median gap 11.3 ms [M v4g-k1 w1/w2, v4g-k2];
   - meanwhile the main thread spins (99.3 %) on the same loop as ours, waiting for the end-of-frame values. Kyty
     publishes labels 8-13 ms after the work (PERF label_at_completion).
   - Everything else overlaps: of the guest's 400 ms of busy CPU per frame, 374 ms runs while the executor is busy.
4. **Projection.** A Kyty backend in AnyPS5 would give a frame of **≈ 190-215 ms (≈ 47-53 presents/10 s)**, the same
   as Kyty here, unless one of these holds:
   - **Z1.** The adapter keeps AnyPS5's submission ordering. That rule (an async ACB waits until queue 0 has
     processed everything submitted before it) is what makes today's link ~100 ms.
   - **Z2.** The adapter delivers labels, EOPs or flips to the guest later than Kyty does. Each extra ms adds 1:1
     with one frame in flight.
   - **Z3.** The HLE JobManager's one-ACB-per-job compute path turns out costly in Kyty's command processor. Each of
     those ACBs carries an ACQUIRE_MEM. This is a phase-1 census item, not a floor.

   Assumptions and the arithmetic are in §8. **The guest side does not lower G2.** Keep G2 at ≥ 35 presents/10 s,
   with 45-50 expected. A result between 25 and 35 would point at the adapter's ordering or latency (Z1-Z3), not at
   the guest.
5. **Something to fix first: two stray busy loops have run on this machine since 11-05 16:30.**
   - What they are: `python -I -c "while True: pass"`, PIDs 19972 and 19012 (parents 20052/12668, bash 10236/8376),
     from the v3-mem lane's `research/speed/v3/mem/direct-inner2.sh` "spin" arm. Its `kill` reached only the
     WindowsApps launcher; the interpreter kept running, and the script's bash is still waiting for it.
   - They burn 2 cores (217,592 CPU-s each). On Kyty they sit on the SMT siblings that Kyty reserves for its stage
     threads (CPUs 3/5/7/9 busy 34-80 %).
   - Every run since then is affected: kyty-prof w2/w3 (48.1), all v3/fm2 runs, and these. Kyty's 48.1 was therefore
     measured under load, while our yotei/best references (11-02 to 11-04) were not.
   - I did not stop them: they are not my processes. Please stop them by PID and rerun one short Kyty reference.

## 1. What was run

| tag | emulator, build | env / tracing | sampler windows | result |
|---|---|---|---|---|
| v4g-k1 | Kyty v0.1.4 release (`research/speed/v3/kyty/app`) | kyty-prof w2 recipe: `KYTY_PERF_DIRECT_RANGES=0`, `KYTY_TRACE_FRAMETIME=1`, `KYTY_VIDEOOUT_1080P=1`, pad-w | w1 frames 2500+ for 120 s (cycles only); w2 frames 5008+ for 120 s (+ IP samples, 50 % tick threshold) | 1200 s, ok. w1 49.8, w2 55.1 presents/10 s |
| v4g-a1 | AnyPS5 yotei/best build (`yb-combo/build-local`, 72af1e5ea23 + an off-by-default budget gate, SV 22c401ea7ca3796f, warm) | `run1080.sh long` (README env, web-teshio-b), untraced | w1 t0+420 s (cycles only); w2 t0+900 s (+ IP) | 1200 s, RUN1080 OK, runtime manifest identical. ~9.0 / ~10.0 presents/10 s in w1 / w2 ([present] lines) |
| v4g-k2 | Kyty release | k1 env + `KYTY_TRACE_PERF=1`, `KYTY_TRACE_SYNC_TIMELINE=1` (detail frames 3001-3030) | frames 2985+ for 45 s (+ IP, 5 % threshold) | 1000 s, ok. 50.1 presents/10 s in the window |
| v4g-a2 | AnyPS5 yotei/best build | a1 env + driver trace `APS5_TRACE_FRAMES=1240:12` + `APS5_TRACE_WAITS=1` | from frame dump 120 for 75 s (+ IP, 5 %); covers the 12 traced presents | 900 s, RUN1080 OK, RESTORE OK. ~10 presents/10 s untraced; the 12 traced frames 1,117 ms median |

- The ggp-tr1 driver trace from 10-30 is reused offline as a cross-check. It is lane G-gaps': yb-next, 1080p, the
  same README env, `APS5_TRACE_FRAMES=880:12`.
- Every run took the game slot through kyty_run.sh or run1080.sh/run_when_free.sh, one at a time, after v4k's and
  v4p's guarded jobs.

## 2. Method

- **Sampler** (`research/v4/guest-floor/tools/tsamp.cpp`, ~450 lines, no admin). Every ~1 ms (high-resolution
  waitable timer) it reads `QueryThreadCycleTime` for every thread of the emulator process.
  - That read is live and exact on this machine: probe3 shows 2,49x k cycles per 1 ms on a spinning thread. The
    per-tick deltas give each thread's CPU timeline at ~1 ms.
  - Every 20 ms, `NtQuerySystemInformation` gives thread states (Running, Ready, Waiting), per-CPU times and the
    other processes' cycles.
  - IP samples: up to 4 running threads per tick are suspended for one `GetThreadContext`, ~25-50 µs each. The rip is
    classified by module: eboot.exe and app0 = guest on AnyPS5; non-module executable memory = guest on Kyty;
    `*.prx` = AnyPS5 HLE; kyty_emulator.exe = Kyty; ntdll/kernel32/... = OS.
  - Spin loops are recognised by rdtsc or pause in the hot code bytes. The four guest loops are disassembled in §6.
  - Log lines (FRAMETIME, [present], frame dumps, SYNC_TL) are stamped with the same QPC clock.
- **Frame anchors.**
  - Kyty: its FRAMETIME line, one per present.
  - AnyPS5 untraced: the presenter thread's burst (≥ 3 ms of CPU within 40 ms). It catches 91 % of the presents
    (9.1 vs 10.0 per 10 s), so a few frames read double; medians are used. The burst leads the flip packet by
    23-41 ms (a2).
  - AnyPS5 traced: the driver trace's flip packet on queue 0 (`Driver.Packet.flip_flush`), which coincides with
    its present instant.
- **Analysis**:
  - `tools/tana.py`: per thread and group, frames, timelines, idle gaps, onsets, bursts, contention, IP classes;
  - `klink.py` (Kyty link), `alink.py` (AnyPS5 link), `partial.py` (CPU in ticks the IP sampler does not see),
    `atrace.py` (driver trace + sampler joined by queue-0 cross-correlation), `strip.py` (text strip charts).
- **What the IP samples miss.** With the 50 % threshold, short bursts are not sampled. Those CPU-ms are reported as
  "partial-tick" and treated as non-spin work, which is an upper bound for real work: a spinner fills whole ticks.
  The traced runs used a 5 % threshold.
- **Overhead and tracing.**
  - The cycle sampler costs ~2-3 % of one core.
  - IP sampling at 4 per tick costs 8-15 % of one core, plus ~30 µs pauses of the sampled threads. On both
    emulators the IP windows (w2) were as fast as or faster than the cycle-only windows (w1). The scene drift between
    windows (±10 %) is larger than the sampler's effect.
  - The Kyty traced run held 50.1 presents/10 s in its window, against 49.8-55.1 untraced.
  - The AnyPS5 driver trace inflates the frame ~9 % (G-gaps: 1,359 vs 1,249 ms). Its link numbers are used for
    structure and ordering; the untraced a1 gives the sizes.

## 3. Kyty: per-frame timeline (v4g-k1 w2, 661 frames, mean 181.5 ms, median 176.8 ms)

Busy cores per thread group, 20 ms bins after the present (FRAMETIME):

| ms after present | exec | resolve | record | CP | job (10) | main | audio | BarCopy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.43 | 0.43 | 0.43 | 0.20 | 0.51 | 0.75 | 0.49 | 0.06 |
| 20 | 0.94 | 0.67 | 0.59 | 0.52 | 0.22 | 0.58 | 0.58 | 0.53 |
| 40 | 0.94 | 0.96 | 0.94 | 0.88 | 0.97 | 0.80 | 0.61 | 0.03 |
| 60 | 0.98 | 0.97 | 0.96 | 0.94 | 1.95 | 0.89 | 0.64 | 0.00 |
| 80 | 0.98 | 0.95 | 0.93 | 0.92 | 1.97 | 0.94 | 0.63 | 0.00 |
| 100 | 0.96 | 0.84 | 0.82 | 0.79 | 1.92 | 0.66 | 0.62 | 0.01 |
| 120 | 0.87 | 0.69 | 0.63 | 0.44 | 0.65 | 0.62 | 0.56 | 0.39 |
| 140 | 0.85 | 0.44 | 0.42 | 0.11 | 1.57 | 0.37 | 0.53 | 0.75 |
| 160 | 0.75 | 0.38 | 0.38 | 0.05 | 2.07 | 0.46 | 0.49 | 0.49 |

Per frame [M k1 w2]:
- **exec busy 158 ms (87 %)**, resolve 131, record 126, CP 96.
- Guest threads: job 258 ms, main 126 ms, audio 105 ms.
- **Executor idle gap after the present:**
  - starts 8.0 ms after it, ends 19.3 ms after it; median 11.3 ms (p10 8.4, p90 13.9), 651 of 661 frames;
  - total exec idle 15.7 ms/frame. Kyty spins 2 ms before it sleeps, so add ≤ 2 ms per gap.
  - Same shape in w1 (11.2 ms, 595 frames) and in the traced k2 (11.3 ms, 214 frames).
- In the gap the main thread is 99.3-99.4 % in its polling loops. The CP picks up the next submission ~8-10 ms after
  the present.
- **Frame length follows the executor:**
  - corr(frame, exec busy) = 0.87-0.99; the longest 10 % of frames carry more exec work (338 ms of exec in 365 ms
    frames, w1);
  - the job threads' extra work in long frames (up to 4 cores) overlaps it.
- Traced detail (k2 SYNC_TL_GUEST, frames 3001-3030):
  - the async queues (cq30/38/48) block on WAIT_REG_MEM at ~22-32 ms and pass at ~53 ms;
  - more blocks on CPU-written labels follow: cq30 waits 42-54 ms for a job's label, passed at 85 and 127 ms;
  - the graphics stream meets its frame-counter waits at ~50 and ~83 ms and passes them in 0.1-0.2 ms; the compute
    queues' frame-counter waits pass in ~2.6 ms.
  - The guest's CPU jobs feed the async GPU work inside the frame, overlapped with the graphics queue.
  - Labels reach the guest 8-13 ms after the work (avg; PERF label_at_completion). SuspendPoint never blocks (18
    per 5 s, 0.0 ms).

## 4. AnyPS5: per-frame timeline (v4g-a1 w2, ~110 frames, ~1.0-1.1 s each)

Busy cores, 50 ms bins after the presenter burst. In a2 that burst leads queue 0's flip packet by 23-41 ms (median
~32 ms), so subtract ~30 ms to read the times as "after the flip".

| ms after present | q0 worker | committer (host1) | other queue workers | job (10) | main |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.32 | 0.14 | 0.31 | 0.00 | 0.98 |
| 50 | **0.03** | 0.00 | **0.85** (0x20) | **0.01** | 0.98 |
| 100 | 0.15 | 0.00 | 0.58 | 0.10 | 0.98 |
| 150 | 0.89 | 0.01 | 0.41 | 1.63 | 0.97 |
| 200-600 | 0.5-0.7 | 0.1 → 0.86 | 0.05-0.8 | **1.94-1.96** | 0.98 |
| 650-850 | 0.5-0.6 | 0.5-0.8 | 0.2-0.6 (0x51/0x52) | 1.8 → 0.3 | 0.6-0.9 |
| 900-1050 | 0.4-0.55 | 0.35-0.5 | 0.15-0.25 | 0.0 | 0.98 |

Per frame:
- q0 568 ms, committer 483 ms, other queue workers 354 ms, second unnamed host thread 229 ms.
- Guest: job 1,240 ms, main 1,014 ms (0.93 cores, the whole frame), audio 617 ms.
- **The link** (`alink.py`): q0 drops below 0.2 cores ~15 ms after the presenter burst and is back above 0.5 at
  ~135 ms.
  - The presenter burst comes 23-41 ms before the flip packet. That is measured in a2, where the driver trace and
    the sampler line up to ~7 ms.
  - So queue 0 resumes **~100-105 ms after the flip**, the same as a2's trace (102 ms).
  - The low stretch lasts 118-123 ms by this measure (p10 108-110, p90 132) in the two windows.
  - Inside it: the 0x20 worker is busy 66 ms, all other queue workers 12 ms, the **job threads 0.0 ms**, main 119 ms
    of spinning.
- Two job threads sit in the game's poll loops from ~150 to ~650 ms (1.95 cores), waiting for GPU work inside the
  frame. That is the same pattern as Kyty's two ~50 ms bursts at 50-110 and 140-180 ms, stretched to our backend's
  pace.

## 5. The link, side by side

### 5.1 Numbers

| | AnyPS5 (yotei/best) | Kyty |
|---|---:|---:|
| frame, untraced | ~1,000-1,100 ms | 181-201 ms |
| link: the emulator's ordered stage has nothing to do after the flip/present | **102 ms** (q0, flip → resume; a2 trace; a1 ~100-105), 144 ms in yb-next (ggp-tr1) | **11.3 ms** (executor; k1, k2) |
| guest threads in the link | jobs 1.5 ms; main spins (91.6 %, the rest is submit calls) | jobs ~0; main spins (99.3 %) |
| emulator in the link | 0x20 worker records 256 dispatches (88 ms busy incl. GPU-mutex waits) after a ~300 ms order hold; GPU 48 ms on them | CP takes the next submission at ~8 ms; executor resumes at ~19.5 ms |
| GPU tail of frame N after the flip | 3.3 ms (a2), 5.4 ms (ggp-tr1) | n/a (label lag 8-13 ms) |
| guest busy that overlaps the ordered stage | the link is serial; the rest of the frame overlaps the committer and q0 | 374 of 400 ms of guest CPU per frame overlaps a busy executor (k1 w2) |

### 5.2 Why AnyPS5's link is long

The driver trace (ggp-tr1, 11 frames):
- the flip packet on queue 0 is followed by queue 0 waiting for a submission for 121-158 ms;
- in that time, queue 0x20 runs the 5 ACBs the guest submitted earlier in the frame. They held ~406 ms in
  `dequeue_order`: the rule that an async submission waits until queue 0 has processed everything received before
  it;
- 0x20 decodes 307 dispatches in the first 150 ms after the flip. Its GPU-mutex waits come to 45 ms per frame;
- the GPU executes them (81.5 ms busy in the link), and 0x30/0x38/0x48 follow at ~135-143 ms;
- then the guest submits queue 0 for frame N+1. The committer is idle in the link (0.4 ms).

Kyty has no such hold: its command processor takes the queues round-robin and stops only at WAIT_REG_MEM. The same
async work runs inside the frame, and Kyty's frame ends with an 11 ms gap.

### 5.3 v4g-a2 (driver trace + sampler in the same run)

The trace covers presents 1240-1252. `atrace.py` aligns it to the sampler by queue-0 cross-correlation (offset
34,959 ms, r = 0.35). The "[trace] recording" log line, stamped at 34,952.5 ms, confirms it to ~7 ms.

| per frame, 11 frames | median (p10, p90) |
|---|---|
| frame (traced) | 1,117 ms (1,061, 1,256) |
| **link: flip packet → queue 0 resumes** | **102.0 ms (93.2, 132.4)** |
| GPU still busy with frame N after the flip | 3.3 ms |
| GPU busy inside the link | 48.0 ms |
| committer / queue 0 busy inside the link | 0.4 / 0.7 ms |
| 0x20 worker busy inside the link | 87.5 ms, from 1 to 89 ms after the flip; 256 of its 365 dispatches per frame; 4.1 submissions per frame; GPU-mutex waits 42.5 ms per frame; held 295 ms (median) in `dequeue_order` first |
| 0x30 / 0x38 / 0x48 | ~7 ms each, at 95-101 ms after the flip (they follow 0x20's completion) |
| 0x50 | 38 ms, at 0-38 ms after the flip |
| guest main thread inside the link | 99.7 ms busy; IP: 91.6 % poll loops, 3.8 % libSceAgcDriver (its submits), 1.7 % ntdll, 1.4 % winpthread, 0.8 % other guest code |
| guest job threads inside the link | 1.5 ms busy in total (only wake/wait syscalls) |

The whole link is our backend working through the frame-start compute ACBs it held behind queue 0. The game runs no
work in it.

**Tracing correction.**
- The traced frames run at 1,117 ms. The untraced a1 and a2 windows run at ~1,000 ms (10 presents/10 s).
- The link is 102 ms traced; ~100-105 ms untraced (a1, via the presenter offset).
- So the driver trace inflates the long queue-0 stretch of the frame, not the link.

## 6. What the guest threads execute

The four hot guest loops are the same code on both emulators. Kyty maps the game image 0x10000 lower than our PE's
module offsets:

| loop | AnyPS5 (eboot+) | Kyty (0x900000000+) | polls | threads |
|---|---|---|---|---|
| A | 0xda2940 | 0xd92940 | `[rbx+4] > 0` (counter) | main |
| B | 0x1139580 | 0x1129580 | `r12d` against a sequence table | main |
| C | 0x152a550 | 0x151a550 | `[r14] == 1` (flag) | job |
| D | 0x152ad10 | 0x151ad10 | `[r14] == rbx` (counter) | job |

- Each is `deadline = rdtsc + 20 µs × calibrated ticks; while (rdtsc < deadline);` followed by one memory check: a
  polling back-off that burns a full core.
- G-1 (10-27) found the main thread's third loop at eboot+0xaefbd0 polling a GPU-written counter (Kyty: +0xadfbd0).

IP classes, share of samples [M a1 w2, k1 w2, k2]:

| thread group | AnyPS5 | Kyty |
|---|---|---|
| main | guest spin 97.6 %, guest other 1.2 %, libSceAgcDriver 0.5 %, ntdll 0.4 % | guest spin 84.2 %, guest other 6.7 %, Kyty 0.8 % + its spin 1.5 %, ntdll 6.3 % |
| job (10) | guest spin 98.3 %, guest other 0.6 %, HLE 0.1 % (libSceJobManager.prx **0.0 %**), ntdll 0.9 % | guest spin 54.6 %, guest other 13.2 %, Kyty spin lock (xchg+pause at kyty+0x1137920) 13.4 %, Kyty 1.1 %, ntdll 17.2 % (WaitForAlertByThreadId 12.8, ProtectVirtualMemory 2.8) |
| audio | guest 51.7 %, ntdll 38.8 %, libSceAjm.native 4.9 % | guest 49.9 %, ntdll 37.4 %, Kyty 8.7 % |

**CPU per frame by class.** These are the traced runs at a 5 % IP threshold, so short bursts are sampled too. Each
sample is weighted by its thread's cycles in that tick (`ipw.py`). AnyPS5 has ~0.9 frames/s in this window, Kyty
~5.0.

| ms per frame | AnyPS5 main | AnyPS5 jobs (10) | Kyty main | Kyty jobs (10) |
|---|---:|---:|---:|---:|
| total CPU | 970 | 1,290 | 127 | 274 |
| game poll loops (spin) | 948 | 1,217 | 114 | 131 |
| other game code | 7 | 8 | 4 | 27 |
| emulator HLE / Kyty | libSceAgcDriver 5 | libSceAgcDriver 4, **libSceJobManager 0** | Kyty 0.5 + Kyty spin 1.7 | Kyty 3 + **Kyty spin locks 43** |
| OS (waits, wakes, locks) | ntdll 6, winpthread 2 | ntdll 59, winpthread 1 | ntdll 6, win32u 0.4 | ntdll 68, win32u 1.4 |
| **non-spin total** | **~21** | **~74** | **~13** | **~143** |

The light runs agree:
- a1 w2 and k1 w2 at the 50 % threshold, with short bursts added as non-spin: main 32 vs 25 ms, jobs 125 vs 138 ms.
- AnyPS5's main thread does ~8 ms per frame more in libSceAgcDriver, ntdll and winpthread than Kyty's: its submit
  calls through our driver. Under a Kyty backend those calls go to the adapter.

**The JobManager point (lane v4s).**
- Kyty runs the game's native libSceJobManager; AnyPS5 runs its HLE (`libSceJobManager/Export.cpp`).
- Measured, the HLE costs the job threads nothing visible:
  - libSceJobManager.prx is < 0.2 ms/s, not among the top classes at either threshold;
  - its scheduling shows up only as the OS waits and wakes of its condition variables: ntdll ~59 ms per ~1.1 s frame
    over ten threads;
  - on Kyty, the native module's equivalent costs more per frame: ntdll 68 ms + Kyty spin locks 43 ms.
  - On Kyty the job threads run 66 % in the eboot image (mostly loops C/D) and 1.0-1.3 % in the second guest code
    region at 0x910000000; the rest is Kyty and the OS.
- In the link the job threads are idle on both emulators. **The JobManager HLE is not a measurable part of the
  ~100 ms link** (1.5 ms of job-thread CPU per frame inside it).
- The HLE's other difference is on the GPU side. `Sequence::SubmitCompute` builds and submits **one ACB per compute
  job** (ACQUIRE_MEM + registers + dispatch), and `FlushCompute` adds a RELEASE_MEM ACB per sync point. The native
  module batches them in guest code.
- Under the old backend this path is not where the link comes from: 0x20 had 5 ACBs and 411 dispatches per frame in
  ggp-tr1, so those are the game's own large ACBs.
- Under Kyty's backend every HLE ACB would be a separate CP submission with its own global barrier.
- **Candidate phase-1 fixes:** batch the HLE's compute jobs per sync point, or load the native module as Kyty does.
  Do whichever the census in phase 1 shows to be needed. The fix belongs in phase 1, not in the floor.

## 7. CPU cores and contention

- **Kyty** reserves 4 whole cores for its stage threads:
  - execution on CPU 2, CP on 4, record on 6, resolve on 8, at priority 2 for exec and CP;
  - it sets the process's default CPU sets so everything else gets 0, 1 and 10-15, i.e. cores 0, 5, 6 and 7.
  - So the game's ~2.7 cores of busy threads (job 1.4, main 0.7, audio 0.6), plus Kyty's BarCopy threads (0.2),
    share 8 logical CPUs.
  - Job threads are Ready (runnable, no CPU) 19-25 % of their runnable time, main 6-10 %; stage threads ≤ 2.5 %.
  - The guest is crowded, but in its own cores. The siblings 3/5/7/9 should be idle; they run 34-80 % busy. That is
    the two stray spinners (2.0 cores), sitting next to Kyty's executor and CP.
- **AnyPS5** pins nothing on this CPU:
  - affinity 0xffff on every thread. Its job-worker pinning (`APS5_JOB_AFFINITY`) is off by default and matches
    another title's thread names;
  - Ready shares: jobs 4.4 %, main 1.1 %, q0 2.5 %, committer 1.8 %;
  - the machine runs 7.8 of 16 logical CPUs busy: eboot 4.4, the stray spinners 2.0, the rest ~1.4.
  - The guest's own spinners (main 0.93 core all frame, two job threads ~2 cores for 500 ms of each frame) share
    physical cores with the queue workers and the committer whenever Windows puts them on SMT siblings. This is not
    measurable without ETW [E].
- **With the Kyty backend in AnyPS5** the stage threads would come with Kyty's pinning and reservation. That is
  process-wide (`SetProcessDefaultCpuSets`), so AnyPS5's guest threads would be fenced off the same way.
  - The guest's spin time per frame falls with the frame. On Kyty: main 0.6 core and jobs ~0.8 core of spin.
  - Keep the reservation: it is part of Kyty's 26 µs/draw. Their docs: pinning +18 %.

## 8. Projection

Model: one frame in flight, so frame = (executor-bound part) + (exposed link).
- On Kyty, measured here: executor-bound ≈ 165-190 ms, link ≈ 11-13 ms. That gives 181-208 ms (k1, kyty-prof w2),
  under the stray-spinner load.
- With Kyty's backend in AnyPS5:
  - the executor-bound part is the same backend on the same GPU work: ≈ 165-190 ms [E: same code, same draws and
    dispatches per frame, M kyty-prof §1];
  - the link is Kyty's chain (GPU done → label → the guest's poll → submit). The guest side of that chain is the
    same game code: main's poll loop A/B and the frame-start submits. AnyPS5's HLE share on those threads is ≤ 1 %
    → ≈ 11-13 ms;
  - the guest's own non-spin work overlaps the executor on Kyty: main ~13-25 ms and jobs ~140 ms per frame across
    ten threads, so 374 of 400 ms of guest busy time per frame. It would overlap the same way, because the
    dependency structure is the game's;
  - the residue: our main thread spends ~8 ms per frame more than Kyty's in submit calls (libSceAgcDriver, ntdll,
    winpthread). Under the Kyty backend those calls go to the adapter. Even if they stayed and were fully serial,
    that is < 10 ms.
- **Frame ≈ 190-215 ms → ≈ 47-53 presents/10 s.** That is Kyty-level here, ×4.6-5 today's 9.3-10.6.
- **Unless:**
  - **Z1.** The adapter routes guest submissions through AnyPS5's `Driver` queue workers with `waitFree`/
    `queue0Before` ordering. That ordering is today's ~100 ms link (§5.3): it would add ~80-100 ms. Submissions
    must go straight to Kyty's per-queue CP.
  - **Z2.** Label, EOP or flip delivery to the guest is slower than Kyty's 8-13 ms: +1 ms of frame per ms. This
    covers AnyPS5's equeue, VideoOut and EOP interrupt paths and the EQ context-id difference v4s found.
  - **Z3.** The HLE JobManager's compute path costs real time in Kyty's CP or on the GPU. It submits one ACB per
    compute job, each with an ACQUIRE_MEM, so one global barrier each in Kyty. Its CPU side measured nil (§6); the
    GPU side is unknown. Phase-1 census; fix: batch per sync point, or load the native module.
  - **Z4.** The process-wide core reservation is not kept: the guest's spinners would share cores with the
    executor.
- **G2:**
  - the guest side leaves no reason to lower 35 presents/10 s. Expect 45-50 on a clean machine;
  - a 25-35 result is to be read as Z1-Z4 (adapter ordering or latency, JobManager path, pinning), not as a guest
    floor;
  - if the stray spinners are stopped, rerun the Kyty yardstick first: Kyty's 48.1 is likely understated by its
    loaded SMT siblings [E].

## 9. Machine, processes, disk, artifacts

- Game slot: four runs, each through its slot script, one at a time. They waited for v4p's prelink and v4k's
  guarded build; no lock was taken over.
  - Both AnyPS5 runs: RUN1080 OK, RESTORE OK, runtime manifest identical.
  - Both Kyty runs: 0 changed entries in the game dir.
- Processes:
  - the sampler and its side scripts ended with their windows;
  - one side script was mine and I stopped it by PID. side_aps5.sh matched `eboot.exe`, while bash lists
    `/c/.../eboot`; side_aps5b.sh replaced it, using eboot's start time.
  - No sampler, watcher or monitor of this lane is left.
- Known sampler quirk: proc.tsv's "other processes" sum breaks when a pid is reused. Here it was recomputed per
  sample with the outliers dropped: 2.6-2.7 cores of other processes, mostly the two stray spinners.
- The stray python spinners (§0.5) were left running: they are not mine.
- Disk: C: 129 GB free at the end; this lane uses ~0.3 GB (`research/v4/guest-floor/runs`).
- Artifacts, all in `C:\ps5\work\research\v4\guest-floor\`:
  - `tools/`: tsamp.cpp and tsamp.exe, the probes, tana.py, klink.py, alink.py, partial.py, presanchor.py,
    strip.py, atrace.py, ipw.py, kframes.py, and the side scripts;
  - `runs/<tag>/w*/`: the sampler data;
  - `runs/trace-a2/`: the driver trace;
  - `v4g-chain1.sh` / `v4g-chain2.sh` and their logs.
- Kyty run logs are in `research/speed/v3/kyty/runs/v4g-k1`, `v4g-k2`. AnyPS5's are in `game_runs/v4g-a1`,
  `v4g-a2` and `C:/ps5/YoteiPC-latest/logs/v4g-a*.err.log`.
