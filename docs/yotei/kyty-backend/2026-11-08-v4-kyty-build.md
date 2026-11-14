# v4k (2026-11-08): our own KytyPS5-Yotei build with MinGW GCC, and the 1.008 DIRECT_RANGES fix

Lane v4k of PLAN v4, phase 0 (steps 0a and 0e). User go for phase 0: 11-08. Nothing was posted to GitHub; no git in
`C:\ps5\AnyPS5`; PLAN/STATE/HANDOFF untouched.

Tags: **[M]** measured here (run tag given), **[E]** estimate or inference (reasoning given).

**Load.** Two orphaned CPU spinners (`python -I -c "while True: pass"`, PIDs 19972 and 19012, since 11-05 16:30) ran on
the machine until some time between 09:16 (when I last saw them) and 11:05:46 (when the coordinator found them gone).
Runs are marked **L** (loaded: before 09:16), **L?** (loaded or unknown: 09:16-11:05) or **C** (clean: started after
11:05:46). Nothing in this lane stopped them. kyty-prof w2 (11-05 18:04) and v4g-k1/k2 were loaded too.

## 0. Answer

1. **Our own Kyty builds and runs with our MinGW GCC 15.2 toolchain**, after fixing 10 classes of build problems (§2).
   All are fixed on `v4/base`, and none needed clang. It reaches the river village and renders it like the release.
2. **Gate 0a (≥ 45 presents/10 s): not met, but close on a clean machine.**
   - Clean, best setting (direct ranges on): **44.0** presents/10 s (median 44.6) [M C v4k-c-gccon2].
   - Loaded, `KYTY_PERF_DIRECT_RANGES=0`: 35.8-38.2 [M L/L?].
   - **The gate's own yardstick has moved.** The clean v0.1.4 release reaches 49.4 [M C v4k-c-rel014]. The **v0.1.5
     release reaches 63.4** [M C v4k-c-rel015], and 56.9-61.6 under load [M L?].
3. **The compiler costs ~a third.** Same v0.1.5 source, same scene (rain, camera on the huts):
   - clang-cl release 56.9 [L? v4k-rel015b] against our GCC build 42.5 with direct ranges on [L v4k-on1], and 37.2 /
     35.8 with them off [L, scene not captured];
   - **GCC ≈ ×0.63-0.75 of the clang-cl release.** Clean pair: 44.0 (GCC, on, rain) vs 63.4 (release, off, dry) → ×0.69; ×0.75 after the ~8 % the rain costs the release.

   This matters for option A: Kyty's code built inside AnyPS5's GCC/MinGW build starts about a third below Kyty-level
   (§3). Likely causes [E]: no LTO, emulated TLS for `thread_local`, winpthreads locking. Not split yet.
4. **0e is fixed in source, exactly (commit a2f25e1f, +71/−8).**
   - Root cause: a direct range is activated lazily by the first buffer lookup that reaches it. Activation finishes all
     recorded GPU work and downloads, so it waits.
   - On 1.008 the first lookup of the SRT ring comes from the BDA fault pass, which runs as a deferred operation. Waits
     are forbidden there: `EXIT_IF(g_deferred_callback_scheduler == this)`.
   - Fix: the fault pass marks the range wanted, and the frame boundary activates it.
   - Result: boots and plays with direct ranges on [M v4k-on1 1,503 s, v4k-on2, v4k-c-gccon2]. The identical build
     without the commit dies at present 2 [M v4k-base-onboot].
5. **ON vs OFF: not resolved by separate runs.** The scene (weather, camera) changes from run to run, by up to ~10 %.
   - Separate runs: on1/off1 +14 % (rain vs unknown scene); on2/off2 +1 % (both dry).
   - An in-run A/B with Kyty's own `KYTY_LIVE_AB` is prepared (`chain6.sh`, one 1,500 s run). It was not run: I handed
     the slot to the presets lane.
6. **One silent exit** of the GCC build during boot: present 5, exit code 127, no FATAL [M C v4k-c-gccon]. The retry ran
   clean. 1 of 10 boots of the final GCC binaries, cause unknown; a stability item for phase 1.

## 1. Source and branch
- github.com/Erlou93/KytyPS5-Yotei, read-only `gh api` on 11-08: one release after v0.1.4, **v0.1.5** (11-06 15:57,
  tag 529c5acd3a3 = main; no commits after it).
  - It targets 2.002 ("1.512 still starts but is not tested"; 1.008 not named).
  - Changes since v0.1.4: page-manager lock back-off (their +13-18 % median fps while walking), texture relocation
    always on, a stencil-shadow crash fix, 2.002 hints, diagnostics. Nothing touches direct ranges.
  - **It keeps 1.008 behaviour:** it plays with DIRECT_RANGES=0, and asserts at present 2 with it on.
- Clone: `C:\ps5\kyty-src` (submodules recursive). Local branch **`v4/base`** from tag v0.1.5 (529c5acd3a3):

  | commit | what |
  |---|---|
  | d573ce07 | Build with MinGW GCC 15.2: source fixes (§2 items 1-5) |
  | da750748 | CMake: 64 MiB stack reserve for MinGW, as clang-cl's /STACK |
  | 2ad0dffb | CMake: per-file workaround for a GCC 15.2 internal compiler error |
  | 7ed0e3b7 | CMake: MinGW link (VirtualAlloc2 & co. import library, bcrypt, -municode, cpuinfo's max) |
  | 4e2b357d | RenderExecutor: two inline thread_local vectors defined in descriptors.cpp (MinGW TLS link error) |
  | 59695076 | CMake: -Wa,-muse-unaligned-vector-move (GCC PR 54412) and a fixed image base |
  | **a2f25e1f** | **Direct ranges: the fault pass no longer activates a range inside its deferred callback (0e)** |

  - Binaries in `research/v4/kyty-build/exe/` (debug info stripped; unstripped fix binary in `exe/full/` for
    symbolising):
    - **base** = 59695076 (v0.1.5 + build-only commits);
    - **fix** = a2f25e1f.
  - Commits are as Celegans12, with `AI-assisted: yes`.

## 2. Build with MinGW GCC 15.2 (our toolchain): what it took
- Toolchain: `. /c/ps5/toolchain/env.sh` (WinLibs GCC 15.2.0 ucrt posix-seh r7, CMake 4.3.1, Ninja).
- Build: Release with their GCC flags (`-O3 -march=x86-64-v3 -ffp-contract=off -fno-omit-frame-pointer -g`, static
  libgcc/libstdc++, no LTO), launcher off (no Qt), its own ccache dir.
- Every build ran through `machine_guard.sh`.
- Their CMake accepts GCC, but this tree evidently has not been built with MinGW GCC for a while.
- **These are the integration items for building Kyty's code inside AnyPS5's GCC build:**

| # | problem (GCC 15.2 / MinGW-w64) | where | fix on v4/base |
|---|---|---|---|
| 1 | MinGW `<pthread.h>` defines `pthread_attr_getguardsize`/`_setguardsize` as macros → "expected unqualified-id" in 64 TUs | kernel/pthread.h, pthread.cpp | `#undef` both |
| 2 | vulkan.hpp struct assigned from a braced list (`x.extent = {w,h,1}`): ambiguous `operator=` (C++ struct vs Vk C struct) | 44 sites, 13 files | name the type (`vk::Extent3D {w,h,1}`) |
| 3 | SEH `__try/__except` (MSVC/clang only) | perfSampler.cpp, refCapture.cpp, kernel/memory.cpp | kept for clang/MSVC. Under GCC the two guarded copies use ReadProcessMemory on the own process (fails instead of faulting), and the sampler's unwind runs unguarded (diagnostics only) |
| 4 | `std::thread::native_handle()` is a winpthreads `pthread_t`, not a HANDLE | shaderPrecompile.cpp | `pthread_gethandle()` under libstdc++ |
| 5 | union whose members have default member initializers: GCC deletes its default constructor | libs/ngs2.cpp | `{}` on the first member |
| 6 | **internal compiler error** "in choose_baseaddr, i386.cc:7447" (pass pro_and_epilogue), one function each in 5 host files, at -O1 and up. Tried without effect: -mno-avx, -march=x86-64, -fomit-frame-pointer, -mstackrealign, -fno-shrink-wrap(-separate), -mno-stack-arg-probe, -g0 | eventQueue.cpp, audio.cpp, avPlayer.cpp, libKernel.cpp, runtimeLinker.cpp | audio.cpp gets `-fno-tree-vectorize`; the other four get `-fno-asynchronous-unwind-tables` (their functions then have no SEH unwind data) |
| 7 | link problems: no MinGW import library has `VirtualAlloc2`, `MapViewOfFile3`, `UnmapViewOfFile2` (clang-cl takes them from onecore); the FFmpeg prebuilt needs bcrypt; the entry point is `wmain`; cpuinfo uses `max()`, which the global NOMINMAX removes | CMakeLists.txt | import library from a .def (KERNELBASE.dll) via dlltool; bcrypt after FFmpeg; -municode; cpuinfo built with -UNOMINMAX |
| 8 | link: 48 × "multiple definition of TLS init function" for `static inline thread_local std::vector` | render.h | define the two vectors in descriptors.cpp |
| 9 | **runtime**: access violation at start. `vmovdqa ymm0 → [rbp-0x60]` with rbp-0x60 only 16-byte aligned, in Swapchain::Submit (GCC PR 54412: MinGW GCC does not realign the stack for AVX spills) | everything built with -march=x86-64-v3 | `-Wa,-muse-unaligned-vector-move` globally. The aligned stack moves left are FFmpeg's own asm |
| 10 | parity with clang-cl: MinGW threads got a 2 MiB stack (clang-cl: /STACK 64 MiB), and the image was relocated by ASLR (clang-cl: /DYNAMICBASE:NO) | CMakeLists.txt | `-Wl,--stack,67108864`, `-Wl,--disable-dynamicbase` |

Not fixed:
- Warnings worth a look later: ~30 `-Wreturn-type` (switches over enums), `-Warray-bounds` in the FlipQueue heap
  code, and 100+ `-Wclass-memaccess`.
- **Emulated TLS.** GCC 15.2 has no native TLS on MinGW (`-fno-emutls` is rejected, tested). Every `thread_local`
  access is a call: the exe has 402 emutls variables and 5,475 calls to `__emutls_get_address` [M objdump].

## 3. Runs
How the runs were made:
- All through `research/speed/v3/kyty/scripts/kyty_run.sh`: the game slot, yielding to waiting runs. The slot was
  shared with v4g and v4p all morning.
- App dirs:
  - self-built binaries in their own app dir, `research/v4/kyty-build/app-gcc`;
  - the official v0.1.5 release in `research/v4/kyty-build/app-rel015/` (zip sha256 bf0662bb…a89, which is the release
    digest). Its `_PipelineCache` was copied from app-gcc, so it started warm;
  - the v0.1.4 release copy `research/speed/v3/kyty/app` was used unmodified for the clean yardstick, as the
    coordinator asked.
- Env as kyty-prof w2:
  - `KYTY_TRACE_FRAMETIME=1 KYTY_VIDEOOUT_1080P=1`, `pad-w.txt`, 1920×1080;
  - save web-teshio-b, restored before each run;
  - "off" adds `KYTY_PERF_DIRECT_RANGES=0`.
- presents/10 s = `scripts/ft.py <stderr> 1500` (10-s windows from present 1500).
- Window captures use PrintWindow (`research/v4/kyty-build/shotpw.ps1`). A maximised Photos window and the Windows
  Firewall prompt cover the screen, so a plain screen capture shows those instead.

**The scene is not the same in every run.** With the same save and pad script, the weather and the camera angle change
from run to run:
- rain with the camera on the huts: w2, on1, rel015b, c-gccon2;
- dry with the camera on the huts: off2, on2;
- dry with the camera turned: rel015, off3, c-rel014.

It is not tied to the binary: the release had both. Cost in the release: rain 56.9 vs dry 61.6 (also different camera),
~8 %. Compare runs with the same scene where possible (column "scene").

| tag | load | binary | direct ranges | s | presents/10 s (n, sd) | frame ms mean/median | scene | notes |
|---|---|---|---|---:|---|---|---|---|
| v4k-w2 | L | GCC base | off | 1,204 | 39.2 (94, 1.6) | 255/250 | rain, huts | warm-up run (first run of this app dir) |
| **v4k-on1** | L | GCC fix | **on** | 1,503 | **42.5** (127, 1.5) | 235/233 | rain, huts | boots, plays 1,503 s, no FATAL; range active from the first frame boundary |
| **v4k-off1** | L | GCC fix | off | 1,502 | **37.2** (127, 2.1) | 269/266 | not captured | |
| **v4k-base1** | L | GCC base | off | 1,505 | **35.8** (126, 1.6) | 279/275 | not captured | gate 0a run as planned |
| v4k-base-onboot | L? | GCC base | on | 5 | — | — | — | **control: the 1.008 assert at present 2** (commandScheduler.cpp:421) |
| v4k-rel015 | L? | release v0.1.5 | off | 1,503 | 61.6 (133, 1.6) | 162/162 | dry, camera turned | |
| v4k-off2 | L? | GCC fix | off | 902 | 38.2 (68, 1.1) | 262/257 | dry, huts | |
| **v4k-rel015b** | L? | release v0.1.5 | off | 905 | **56.9** (73, 2.6) | 176/175 | **rain, huts** | same scene as on1 / c-gccon2 |
| v4k-on2 | L? | GCC fix | on | 902 | 38.6 (68, 1.0) | 259/256 | dry (overcast), huts | |
| v4k-off3 | L? → C | GCC fix | off | 900 | 40.5 (68, 2.7): 37.8 before 11:05:46, **41.8** after | 247/238 | dry, camera turned | spans the moment the load was found gone |
| **v4k-c-rel014** | **C** | release v0.1.4 | off | 906 | **49.4** (72, 3.1) | 202/172 | dry, camera turned | clean re-measure of the w2 recipe (48.1 loaded) |
| v4k-c-gccon | C | GCC fix | on | 4 | — | — | — | **silent exit at present 5, rc 127**, during shader translation; no FATAL, no WER event |
| **v4k-c-gccon2** | **C** | GCC fix | **on** | 903 | **44.0** (68, 3.8; median 44.6) | 228/221 | **rain, huts** | the same run again; two hitch windows (26, 35) |
| **v4k-c-rel015** | **C** | release v0.1.5 | off | 900 | **63.4** (75, 1.2) | 158/157 | dry (overcast), huts | the yardstick for "Kyty-level" from now on |
| (w2, 11-05) | L | release v0.1.4 | off | 1,503 | 48.1 (128, 4.2) | 208/191 | not captured | kyty-prof yardstick |
| (v4g-k1) | L | release v0.1.4 | off | 1,201 | 48.5 (101, 3.8) | 206/185 | not captured | v4g lane, with side sampler |

Clean next to loaded:

| | loaded | clean |
|---|---|---|
| release v0.1.4, w2 recipe | 48.1 (w2), 48.5 (v4g-k1) | **49.4** (c-rel014) |
| release v0.1.5 | 56.9 (rel015b, rain), 61.6 (rel015, dry) | **63.4** (c-rel015, dry) |
| GCC fix, on | 42.5 (on1, rain), 38.6 (on2, dry) | **44.0** (c-gccon2, rain) |
| GCC fix, off | 37.2 (off1), 38.2 (off2), 37.8 (off3 before) | 41.8 (off3 after 11:05:46, part of one run) |

The load cost the clang-cl release ~2-3 % and our GCC build ~4-10 % [M, same-scene pairs where available].

What the numbers say:
- **GCC vs clang-cl, same v0.1.5 source.**
  - Rain, huts: 42.5 (GCC, on) vs 56.9 (release, off), both under load → **×0.75**. With both off: ≤ 37.2 / 56.9 →
    ~×0.65 (off1's scene not captured).
  - Clean: 44.0 (GCC, on, rain) vs 63.4 (release, off, dry, huts) → ×0.69. The rain costs the release ~8 %, so the
    same-scene ratio is about ×0.75.
  - This, not the version, is why the gate is missed. The release itself is far above it.
- **v0.1.5 vs v0.1.4 (both clang-cl releases):** clean 63.4 vs 49.4, **+28 %** (dry in both, different camera); loaded
  56.9-61.6 vs 48.1-48.5. That is their page-manager lock change and texture
  relocation. "Kyty-level" on this machine is now the v0.1.5 figure.
- **Why GCC is slower is not measured.** Candidates [E]:
  - no LTO: their CI builds with `CMAKE_INTERPROCEDURAL_OPTIMIZATION=ON`;
  - emulated TLS: 5,475 call sites, and Kyty's hot paths use thread_local;
  - winpthreads mutex and condition variables under libstdc++'s std::mutex/std::thread, against the MSVC STL's SRW locks;
  - codegen.

  The split costs one build and one run: clang against the same MinGW runtime (llvm-mingw has native TLS, and LTO
  works there), measured in the same scene.

## 4. The 1.008 crash (0e): root cause and fix

**Mechanism** (v0.1.5 source; the same in every release):
1. `KYTY_PERF_DIRECT_RANGES` (default on) makes the "Yotei SRT ring" 0x2010000000-0x2015E00000 one buffer of imported
   host memory.
   - The range is **activated lazily**, by the first `DirectBufferFor` call that reaches it (from `FindBuffer`,
     `ObtainBuffer` or `TryCpWriteThrough`).
   - Activation (`ActivateDirect`) calls `CommandScheduler::Finish()`: submit, wait for all recorded work, then
     `PopPendingOperations`.
   - It also downloads the GPU-written bytes of the ordinary buffers it replaces, with `DownloadBufferMemory<false>`.
     That waits and then calls `WaitPriorityOperations`.
   - Growing a range does the same through `DeactivateDirect`.
2. The BDA fault pass (`FaultManager::ProcessFaultBuffer`) records a compute pass and **defers** a callback
   (`DeferOperation`).
   - The callback reads the missed pages and calls `FindBuffer` on each (2 MiB windows; `PERF_BDA_PROMOTE_KB` is 2048
     by default).
   - Deferred callbacks run inside `CommandScheduler::RunOperation`, which sets `g_deferred_callback_scheduler`.
   - `WaitPriorityOperations` and `DrainPriorityOperations` refuse to run there
     (`EXIT_IF(g_deferred_callback_scheduler == this)`, commandScheduler.cpp:414/421). A wait inside a deferred
     operation can wait on itself or re-enter the pop.
3. **On 1.008 a shader reads the SRT ring through a BDA pointer before any binding covers it.**
   - Frame 2: `BDA_FAULT frame=2 count=5 first=0x2011884000`. The fault callback is the first lookup of the range, so it
     activates the range inside the deferred operation, and the emulator stops.
   - On 2.002 a binding gets to the range first: `ObtainBuffer`, on the GPU thread, outside any deferred operation. The
     fault pass then only ever finds the range already active.
   - Kyty hit the same path on 2.002 in its own commit 1cf3b782 ("from the start it stopped at frame 2 in a nested
     Finish of the fault-buffer callback"), when KYTY_DIAG_SKIP_DIRECT_DRAWS removed those bindings.
4. Stack of the control run v4k-base-onboot (our base build, symbolised with `exe/full`):
   - ExecutionThread → Execute → ExecuteDispatchDirect → … → RunOperation (the fault callback) → FindBuffer →
     DirectBufferFor → ActivateDirect → Finish → PopPendingOperations → WaitPriorityOperations → EXIT.
   - The release's stack (kyty-prof) reached the other of the activation's two waits:
     … → DownloadBufferMemory<false> → WaitPriorityOperations.

**The fix** (commit **a2f25e1f** on `v4/base`, 4 files, +71/−8; its message is written as an upstream-ready
description):
- `FindBuffer(vaddr, size, bool defer_direct = false)`. The fault callback passes `true`.
- `DirectBufferFor(vaddr, size, bool* deferred)`:
  - with `deferred` set, a range that would have to be activated, grown, or (live A/B switch) ended is left alone;
  - it is marked wanted (the union of the requested bytes), and a null id is returned;
  - `FindBuffer` then returns no buffer for that request. It does not create an ordinary buffer that activation would
    replace, and it never joins one with an active direct buffer.
- `BufferCache::ActivateWantedDirectRanges()`:
  - called from `RenderContext::RunGarbageCollector`: the frame boundary, on the GPU thread, outside any deferred
    operation. The fault pass and texture relocation already run and wait there;
  - it activates or grows each wanted range exactly as a lookup of the requested bytes would.
- Unchanged:
  - every other caller;
  - with `KYTY_PERF_DIRECT_RANGES=0`, nothing at all.
- Until the frame boundary the missed pages read through their BDA fallback entries, as they did before the pass
  (v4k-on1: `fallback=5 zeros=0`).
- **Why it is exact.** It changes only when the activation happens (from inside the deferred callback to the next frame
  boundary, at most a frame later), not what it does. The same activation code runs, on the same thread and at the same
  kind of point as on 2.002.
- The alternatives were worse:
  - waiting inside the callback would need the scheduler's re-entrancy rules changed;
  - activating at start would need the range mapped and free of images before the game runs.

**Verification** [M]:
- With the fix, direct ranges on, every run logs `PERF_DIRECT_RANGES 0x2010000000-0x2015e00000 wanted by a deferred
  lookup (0x2011800000+0x400000)`, then `PERF_DIRECT_RANGES active … (0 old buffers replaced)`:
  - v4k-on1: 6,978 presents in 1,503 s, no FATAL/EXIT, and the river village renders normally
    (`runs/v4k-on1/shot-pw-a.jpg`);
  - v4k-on2 and v4k-c-gccon2: 900 s each, no FATAL/EXIT.
- v4k-base-onboot, the identical build minus the commit, with direct ranges on: dies at present 2 with the assert.
- The v4k-c-gccon silent exit came after the activation, at present 5, during shader translation. Its retry ran clean.
  Nothing links it to the fix, but it is not proven unrelated: the other boots of the fix binary were 6 clean (on and off),
  and the base binary booted clean twice (plus the expected control assert).

**ON vs OFF, same fix binary** [M]:
- on1 42.5 vs off1 37.2 (+14 %; off1's scene was not captured).
- on2 38.6 vs off2 38.2 (+1 %, both dry, camera on the huts).
- The scene noise is as large as the effect. Kyty's code note measured +1 % on 2.002 (6.34 vs 6.28 fps).
- The fair measurement is Kyty's in-run A/B, prepared as `research/v4/kyty-build/chain6.sh`:
  - `KYTY_LIVE_AB=KYTY_PERF_DIRECT_RANGES`, 60-s phases from present 1500, 20 warm-up frames per phase, alternating
    order;
  - one 1,500 s run. Not run: the slot went to the presets lane.

## 5. Machine, disk, processes
- **Builds:** all through `machine_guard.sh`, GUARD_CLASS=build:
  - v4k-build-gcc1/2, v4k-ice-test/2/3, v4k-build-both (passes 2-6), v4k-tls-test;
  - they waited for v4g/v4p game runs (up to 37 min) and never ran beside one.
- **Game slot:** every start went through kyty_run.sh; no lock was taken over.
  - 3 failed starts before build fix 9: v4k-warm1, v4k-fix-on, v4k-fix-off. Their dirs are deleted.
  - 1 control crash, 1 silent exit, 12 measured runs.
  - My first chain was stopped by PID after those failed starts (its runner released game.lock through its trap). No
    other process was touched.
- **Windows Firewall prompt** for the new `app-gcc\kyty_emulator.exe` is open on the desktop. I did not touch it (a
  security setting). It does not stop the runs. The user can cancel it.
- **Disk:** C: 134 GB free at the start, 130 GB at the end. Kept:
  - `C:\ps5\kyty-src`: 3.7 GB with `_Build/gcc`, for incremental builds in phase 1;
  - `research/v4/kyty-build/`, 2.3 GB:
    - ccache, 1.0 GB;
    - `app-gcc`, 0.2 GB, with a warm pipeline cache;
    - `app-rel015`, 0.24 GB;
    - `exe/`: the two stripped binaries and the unstripped fix binary.
- **Deleted** (my own outputs):
  - the first broken binaries (1.6 GB);
  - the v0.1.5 zip (its sha256 is above);
  - the TLS test, the screen captures that showed other windows;
  - the scratch worktree and branch `v4/fix-direct`.
- **Files** in `research/v4/kyty-build/`:
  - scripts: `build-gcc.sh`, `build-both.sh`, `chain.sh`, `chain3.sh`, `chain4.sh`, `chain6.sh` (the A/B, not run),
    `chain7.sh`, `chain8.sh`, `shotpw.ps1`, `errlist.py`, `ice-test*.sh`;
  - logs: `v4k-build-*.log` (`old/` holds the failing passes), `gcc1-errors.txt` (the first error list),
    `chain-v4k-*.log`.
- Run outputs: `research/speed/v3/kyty/runs/v4k-*`, with `shot-pw-*.jpg`.
