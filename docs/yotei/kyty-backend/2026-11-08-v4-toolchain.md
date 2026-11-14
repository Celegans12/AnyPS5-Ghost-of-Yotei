# v4t (2026-11-08): the toolchain split for the vendored Kyty backend (PLAN v4 step 0f)

Lane v4t of PLAN v4 (G0 package, user go 11-08). Nothing was posted anywhere; no git in `C:\ps5\AnyPS5`; PLAN, STATE
and HANDOFF are untouched; `v4/base` and the checkout in `C:\ps5\kyty-src` were not changed.

Tags: **[M]** measured here (run tag named), **[E]** estimate (reasoning named). "presents/10 s" is always
`scripts/ft.py <stderr> 1500` (10-s windows from present 1500), mean of the windows. Every run here is clean: no stray
load (a 5-s per-process CPU sample during v4t-gcc2 showed only kyty_emulator), and no build beside a run (all through
the game slot and machine_guard).

## 0. Answer

### 0.1 Presents/10 s per build, by scene

Same v0.1.5 source (`v4/base` a2f25e1f plus build-only commits), same recipe as v4k (1080p, pad-w, web-teshio-b).
`KYTY_PERF_DIRECT_RANGES=0` in every build-vs-build run, because the release cannot run with it on (0e). Scene =
weather and camera, from the PrintWindow captures at presents 1600, 2600 and 4000 (each run kept its scene from 1600
to the end, except v4t-ab1). Camera A: canoe level across the frame; B: canoe angled to the right; C: turned left,
huts at the right edge.

| scene | clang-cl release v0.1.5 | **llvm-mingw clang, libc++, native TLS, ThinLTO** | llvm-mingw clang, GCC's libstdc++ + winpthreads, native TLS, ThinLTO | GCC 15.2 + LTO | GCC 15.2 (v4k's build) |
|---|---|---|---|---|---|
| overcast, camera A | 61.7 [M v4t-rel1]; 63.4 [M v4k-c-rel015] | | | | |
| overcast, camera B | | **63.9** [M v4t-clang3] | | | 35.7 (median 39.0; 40.6 without its 19 stalls) [M v4t-gcc2] |
| overcast, camera C | **66.3** [M v4t-rel2] | 64.4-69.7 (the A/B's off phases after the rain) [M v4t-ab1] | **51.8** [M v4t-clanglstd2] | | |
| sunny | | **62.4** (camera B) [M v4t-clang2] | | 42.6 (camera A) [M v4t-gcclto3] | |
| rain, camera A | | | | | 44.0 (direct ranges **on**) [M v4k-c-gccon2] |

Same-scene and near pairs:
- **clang for MinGW (libc++) = the release.** 62.4-63.9 against the release's 61.7-66.3; every clang number sits
  inside the release's spread (±4 %), though no pair has the exact same scene [M].
- **GCC is at two thirds, with or without LTO.**
  - Same scene (overcast, camera B), the same 600 s from present 1500: GCC 35.5 vs clang 63.6 → **×0.56**; ×0.64
    after removing GCC's stalls (40.5) [M v4t-gcc2, v4t-clang3].
  - Same weather (sunny): GCC + LTO 42.6 vs clang 62.4 → **×0.68** [M v4t-gcclto3, v4t-clang2].
  - LTO itself is worth roughly +5-10 % (40.6 stall-free → 42.6, different weather): at most a fifth of the gap
    [M, scene not matched].
- **The C++ runtime is a third of the gap.** Same scene (overcast, camera C), first 600 s: clang with GCC's libstdc++
  and winpthreads 52.3 vs the release 66.3 → **×0.79** [M v4t-clanglstd2, v4t-rel2]. The same compiler with libc++
  runs at release speed (row above), so libstdc++ + winpthreads (locks, condition variables, `unordered_map`
  hashing, …) cost ~20 % on this workload. GCC's code generation and emulated TLS cost about the same again
  (42.6 / 52 ≈ ×0.81, scenes not matched) [M/E].
- **GCC-only stalls.** Builds with GCC's SEH unwind tables stall 1-6 s at irregular intervals: 19 times in v4t-gcc2,
  3 in v4k-c-gccon2, never in the 7 loaded v4k GCC runs. GCC + LTO (built without asynchronous unwind tables) and
  every clang and clang-cl run had none [M, cause not investigated].

### 0.2 Recommendation: **(b) a clang-built Kyty DLL with libc++ behind a C seam**; (a), (c) and the hybrid rejected

- **(a) GCC with a fix: no.** LTO is the only lever on the compiler side and it leaves ×0.68 [M]. Emulated TLS cannot
  be switched off in MinGW GCC 15.2 (v4k), and libstdc++ + winpthreads alone cost ×0.79 [M]. Replacing the 83
  `std::mutex`, 6 `std::condition_variable` and 146 `std::unordered_map` mentions in Kyty's graphics and common code
  would be a large private patch on the
  vendored subtree, and would still leave GCC's own ×0.81 [E]. A GCC-built backend starts G2 (≥ 40) on the edge: v4g
  projected 47-53 for a Kyty-level backend, ×0.68 of that is 32-36 [E].
- **(c) all of AnyPS5 with clang: no.** It does not build today and the port is deep (§6): AnyPS5's Windows exception
  model is GCC-specific, the libraries' CI and docs are GCC-only, and the old backend's T1 would have to be re-proven.
  Even then, AnyPS5 would have to move to libc++ to get the speed (the libstdc++ result above). 1-2 weeks with an
  open-ended exception-handling risk [E].
- **Hybrid (clang objects in the GCC-built prx, sharing AnyPS5's libstdc++, no seam): no.** Measured at ×0.79 because
  it keeps libstdc++ and winpthreads [M v4t-clanglstd2]. It also needs a patch on Kyty's `std::call_once` (clang's
  native TLS does not link against GCC's emulated-TLS libstdc++; with `-femulated-tls` clang builds do not start), and
  GNU ld cannot do LTO on clang objects.
- **(b) yes.** It is the one configuration measured at release speed: llvm-mingw + libc++ + native TLS + ThinLTO.
  Cost on top of the ~2,950-line adapter: **~1.3-1.8k lines and 4-7 agent-days** for the C seam, plus a second
  toolchain in build and deploy (§5) [E]. Phase 1 (19-25 agent-days to the title) grows by about a quarter.

### 0.3 Direct ranges: **default ON** for the vendored backend

In-run `KYTY_LIVE_AB` on the clang build, 1,500 s, 10 complete cycles of 60-s phases (alternating order, 20 warm-up
frames per phase), rain then overcast, camera C [M v4t-ab1]:
- off 6.443 fps pooled (64.4 presents/10 s), on 6.732 (67.3);
- per-cycle difference **+4.70 % (standard error 1.65 %)**; on was faster in 9 of 10 cycles (about +3.2 % without
  cycle 0, the heaviest rain);
- the GPU thread is busy 97 % of the time in both arms (148.8 vs 144.1 ms per frame): the gain is CPU work saved on
  the bottleneck thread.

### 0.4 For the coordinator and the user

- **Decision needed:** (b) for phase 1's build lane. The vendored subtree builds with llvm-mingw as its own DLL; the
  v4s adapter spec stays valid, but its C++ seam becomes the C table of §5, and v4s's "patch `sync.cpp` to AnyPS5's
  `KernelEqueueEvent`" becomes a callback.
- **Tell the user:** our Kyty build was at ×0.69 because of the toolchain, not the source. clang for MinGW with libc++
  reaches the release; GCC + LTO does not; about a third of GCC's loss is libstdc++ + winpthreads.
- `v4/clang` (865cdd5e) is the Kyty build to use from now on: release speed, MinGW target, native TLS, same PE layout
  as the release.
- Pipeline caches: a clang + libc++ build emits different SPIR-V from GCC/libstdc++ builds for some shaders (25 run-time
  compiles on its first run; the libstdc++ builds had none), probably container iteration order. Whatever toolchain
  the backend uses, its pipeline cache must be warmed by that toolchain's build [M v4t-clang2, v4t-clanglstd2; cause E].
- A Windows Firewall prompt for `research/v4/toolchain/app-v4t/kyty_emulator.exe` may be open; I did not touch it.

## 1. Builds

All through `machine_guard.sh` (GUARD_CLASS=build). Build dirs `research/v4/toolchain/b-<name>`, ccache
`research/v4/toolchain/ccache`, script `research/v4/toolchain/build-tc.sh` (`build-tc2.sh`: the same with the worktree
as a variable). Release, `-O3 -march=x86-64-v3 -ffp-contract=off -g`, Kyty's flags otherwise. Stripped copies
(`--strip-debug`) are in `research/v4/toolchain/exe/` (`*.rev` names the source); the unstripped exes stay in the build
dirs.

Toolchain installed: **llvm-mingw 20261104** (clang 23.1.5, ucrt x86_64), the official release from
github.com/mstorsjo/llvm-mingw, sha256 `552191b5…1b40f` checked against the release digest, in
`C:\ps5\toolchain\llvm-mingw-20261104`.

| name | source | compiler, C++ runtime | LTO | TLS | result | exe in `exe/` |
|---|---|---|---|---|---|---|
| **clang** (`b-clang`) | `v4/clang` **865cdd5e** | llvm-mingw clang, libc++ (Win32 threads: SRW locks), static | ThinLTO | native (0 `__emutls_get_address` calls) | built first time: 8 min cold, 1 min relink | `kyty_emulator-clang2.exe` (md5 b0c59e66) |
| **gcclto3** (`b-gcclto3`) | `v4/gcc-lto` **071976bf** + `-fno-asynchronous-unwind-tables` on the command line | WinLibs GCC 15.2, libstdc++, winpthreads | `-flto=auto`, 16 LTRANS jobs | emulated (5,713 calls) | third attempt, 12 min | `kyty_emulator-gcclto3.exe` (md5 74812849) |
| gcc (v4k) | `v4/base` a2f25e1f | WinLibs GCC 15.2 | none | emulated (5,475) | v4k's fix binary | `kyty_emulator-gcc.exe` |
| **clanglstd2** (`b-clanglstd2`) | `v4/clang-lstd` **ae9b0c89** | clang, `--sysroot` WinLibs `-stdlib=libstdc++ -rtlib=libgcc` (GCC's libstdc++, static winpthreads) | ThinLTO | native | built; runs | `kyty_emulator-clanglstd2.exe` (md5 a855e7eb) |
| clangemu (`b-clangemu`) | 865cdd5e + `-femulated-tls` | clang, libc++ | ThinLTO | emulated (9,340 calls) | built; **does not start** (rc 139) | `kyty_emulator-clangemu.exe` |
| clanglstdn (`b-clanglstdn`) | 865cdd5e, as clanglstd2 + `-femulated-tls` | clang, GCC's libstdc++ | none | emulated (8,085 calls) | built; **does not start** (rc 139) | `kyty_emulator-clanglstdn.exe` |
| clanglstd, clanglstde | 865cdd5e, GCC's libstdc++ | clang | ThinLTO | native / emulated | **link fails**: `std::__once_callable`, `std::__once_call` undefined | — (dirs deleted) |

What each needed:
- **`v4/clang` (865cdd5e, +12/−9 in 4 files):**
  - the MinGW link block (KernelBase import library for VirtualAlloc2 & co., bcrypt after FFmpeg, `-municode`,
    `--disable-dynamicbase`, cpuinfo's `max()`) applies to every MinGW compiler, not only GCC;
  - SEH `__try/__except` only under `_MSC_VER`. clang for MinGW parses `__try` with `-fms-extensions` but emits no
    handler: a test copy from address 0x10 crashed instead of returning false [M]. MinGW clang takes v4k's
    ReadProcessMemory path, as GCC does (diagnostic code only: guarded copies in the trace and capture paths, and the
    sampler's unwind);
  - nothing else: v4k's ten GCC fixes are compiler-neutral or GCC-only. Image base 0x140000000, no ASLR, 64 MiB stack,
    fully static (no `libwinpthread-1.dll`): the same PE header values as the clang-cl release [M objdump].
  - **Trap:** Kyty disables its Vulkan pipeline cache for a "-dirty" git hash. My first clang binary was built before
    the commit, so v4t-clang1 compiled every pipeline at run time; I stopped it by PID after 457 s; it is invalid.
- **`v4/gcc-lto` (071976bf, +13 lines in CMakeLists.txt):**
  - attempt 1: v4k's GCC 15.2 internal compiler error ("in choose_baseaddr", pass pro_and_epilogue) comes back in
    the link-time compile, in `kernel/memory.cpp` (`RemoveMappedUnlocked`): v4k's per-file
    `-fno-asynchronous-unwind-tables` does not reach the LTRANS compile;
  - attempt 2: the six affected files out of LTO (`-fno-lto`, the commit); the error moves to
    `kernel/fileSystem.cpp`;
  - attempt 3: the whole program without asynchronous unwind tables (as AnyPS5's Windows libraries are built), plus
    `MAKE=mingw32-make` and `-flto=16` (without `make` on PATH, lto-wrapper runs its 128 LTRANS jobs one by one).
    Builds and runs.
  - So GCC + LTO works here only without SEH unwind data. Inside AnyPS5 that would be the normal case anyway.
- **`v4/clang-lstd` (ae9b0c89, lab branch from `v4/clang`):** `Common::CallOnce`/`Common::OnceFlag`
  (`src/common/callOnce.h`), which is `std::call_once` except for clang against libstdc++, at the 6 `call_once`
  sites. libstdc++ keeps call_once's callable in TLS variables that GCC's MinGW libstdc++ defines with emulated TLS;
  clang's native TLS cannot reference them.
- **clang with emulated TLS does not start** (clangemu with compiler-rt's emutls, clanglstdn with libgcc's): both exit
  139 right after "FIX_VIRTUAL_GPU_CLOCK enabled", before the guest's first VideoOut call, where every working build
  continues. Not investigated (time box). It rules out "clang with GCC's TLS model" as a way to share TLS with GCC code.
- Worktrees: `C:\ps5\kyty-src-wt\v4t` (`v4/clang`, checked out again at the end) and `C:\ps5\kyty-src-wt\v4t-gcclto`
  (`v4/gcc-lto`). Submodules are shallow clones from `C:\ps5\kyty-src\.git\modules` (url insteadOf on the command line
  only). Commits as Celegans12, ending with `AI-assisted: yes`, no Co-Authored-By.

## 2. Runs

All through `research/speed/v3/kyty/scripts/kyty_run.sh`, from my chain script `research/v4/toolchain/v4t-chain.sh`
(v4k's chain8 with its own app dir and captures at presents 1600, 2600, 4000). Env `KYTY_TRACE_FRAMETIME=1
KYTY_VIDEOOUT_1080P=1`, `pad-w.txt`, save web-teshio-b, 1920×1080. App dirs: `research/v4/toolchain/app-v4t` (a copy
of v4k's `app-gcc`, warm pipeline cache) for self-built binaries; v4k's `app-rel015` for the release. Run dirs
`research/speed/v3/kyty/runs/v4t-*` (with `shot-pw-*.jpg`), chain logs `research/v4/toolchain/logs/v4t-chain-{A..G}.log`.

| tag | binary | direct ranges | s | presents/10 s (n, sd) | median | frame ms mean | stalls > 1 s | scene | notes |
|---|---|---|---:|---|---:|---:|---:|---|---|
| v4t-clang1 | clang, "-dirty" | off | 457 | — | | | | | **invalid**: pipeline cache disabled by the dirty hash; stopped by PID |
| v4t-rel1 | release v0.1.5 | off | 907 | **61.65** (73, 1.52) | 61.99 | 162 | 0 | overcast, A | |
| v4t-clang2 | clang | off | 904 | **62.43** (58, 1.64) | 62.42 | 160 | 0 | sunny, B | first run of this revision: 25 pipeline compiles while loading (present 1500 at 317 s), none after present 1500 |
| v4t-gcc2 | GCC (v4k's fix exe) | off | 907 | **35.68** (63, 7.99) | 38.97 | 281 | 19 (85 s) | overcast, B | 40.6 without the stalls |
| v4t-gcclto3 | GCC + LTO | off | 905 | **42.60** (70, 1.27) | 42.70 | 235 | 0 | sunny, A | |
| v4t-clangemu1 | clang + emulated TLS | off | 0 | — | | | | | exit 139 before the second frame |
| v4t-ab1 | clang | live A/B | 1,502 | 66.37 (133, 3.58), both arms | 66.66 | 151 | 0 | rain to ~present 3000, then overcast; C | §0.3 |
| v4t-rel2 | release v0.1.5 | off | 902 | **66.28** (75, 1.30) | 66.41 | 151 | 0 | overcast, C | |
| v4t-clang3 | clang | off | 905 | **63.93** (74, 2.25) | 64.29 | 156 | 0 | overcast, B | same scene as v4t-gcc2 |
| v4t-clanglstdn1 | clang + libstdc++, emulated TLS, no LTO | off | 0 | — | | | | | exit 139 at the same point as v4t-clangemu1 |
| v4t-clanglstd2 | clang + libstdc++, native TLS, ThinLTO | off | 905 | **51.84** (71, 2.33) | 51.91 | 193 | 0 | overcast, C | same scene as v4t-rel2; no pipeline compiles at all |

Earlier reference runs (v4k, clean): v4k-c-rel015 63.37 (overcast, A); v4k-c-gccon2 43.97 (rain, A, direct ranges on,
3 stalls).

The scene also changes inside a run (v4t-ab1: rain until about present 3000, then overcast, and its last phases reach
70-73). Weather and camera moved the release between 61.7 and 66.3 here; rain costs it ~8 % (v4k). The ratios between
builds (×0.56-0.79) are well outside that.

## 3. Direct ranges A/B (the `KYTY_LIVE_AB` run v4k prepared)

`chain6.sh`'s recipe, pointed at the clang binary (chain log `v4t-chain-D.log`): `KYTY_LIVE_AB=KYTY_PERF_DIRECT_RANGES`,
`KYTY_LIVE_AB_PERIOD_MS=60000`, `KYTY_LIVE_AB_START_FRAME=1500`, `KYTY_LIVE_AB_WARMUP_FRAMES=20`, alternating order.
Direct ranges are on at start: the 0e fix activates the SRT ring at the first frame boundary ("wanted by a deferred
lookup", then "active 0x2010000000-0x2015e00000 (0 old buffers replaced)").

| cycle | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| off fps | 5.698 | 6.327 | 6.250 | 6.438 | 6.487 | 6.570 | 6.525 | 6.540 | 6.655 | 6.931 |
| on fps | 6.748 | 6.571 | 6.433 | 6.739 | 6.515 | 6.871 | 6.865 | 6.889 | 6.776 | 6.909 |
| on vs off | +18.4 % | +3.9 % | +2.9 % | +4.7 % | +0.4 % | +4.6 % | +5.2 % | +5.3 % | +1.8 % | −0.3 % |

Kyty's summary line: off 6.442 ± 0.320, on 6.732 ± 0.169, **+4.70 % ± 1.65 % (SE)**. About +3.2 % without cycle 0
(heavy rain; the first phase also ends the range that was active from the start). Kyty's code note had +1 % on 2.002.
Default ON.

## 4. Which factor (half-day budget)

| factor | evidence | effect |
|---|---|---|
| libstdc++ + winpthreads vs libc++ (same clang, same flags, native TLS, ThinLTO) | v4t-clanglstd2 52.3 vs v4t-rel2 66.3 (same scene, first 600 s); clang + libc++ at release speed elsewhere | **×0.79** [M]. Not split further (locks vs containers vs hashing): the binary has 773 direct `pthread_mutex_lock` calls where the libc++ build has 766 SRW-lock calls |
| GCC code generation + emulated TLS (both with LTO, libstdc++) | v4t-gcclto3 42.6 (sunny A) vs v4t-clanglstd2 51.8 (overcast C) | ≈ ×0.81 [M, scenes not matched]. TLS not split: clang with emulated TLS does not start |
| LTO in GCC | GCC 40.6 (stall-free, overcast B) → GCC + LTO 42.6 (sunny A) | +5-10 % [M, scenes not matched] |
| GCC's stalls (builds with SEH unwind tables) | 19 stalls (85 s) in v4t-gcc2, none in GCC + LTO without unwind tables, none in clang | a defect of that build, 0-12 % of a run [M]; cause not looked at |

×0.79 × ×0.81 ≈ ×0.64, consistent with GCC + LTO at ×0.66-0.68 of clang + libc++.

Not done (outside the budget): a sampling profile. Kyty's own sampler (`KYTY_TRACE_PERF_SAMPLE`) unwinds stacks without
a guard on MinGW (§1), and GCC + LTO has no unwind data, so it would likely crash these builds.

## 5. Option (b): what the C seam must carry

Starting point: `research/v4/seam/adapter-spec.md` (v4s). Under (a) or (c) the adapter is ordinary C++ on both sides.
Under (b) every call that crosses between AnyPS5 code (GCC, libstdc++; every prx imports the shared
`libstdc++-6.dll` and `libgcc_s_seh-1.dll` [M objdump of `libSceAgcDriver.prx`]) and Kyty code (clang, static libc++)
becomes a C function with plain-data arguments. No `std::` type, virtual interface, `std::function` or exception may
cross.

### 5.1 AnyPS5 → Kyty DLL (exports, ~35)

| group | calls | AnyPS5 types to flatten |
|---|---|---|
| lifecycle | `kyty_init(const KytyConfig*, const KytyHostApi*)`, `kyty_shutdown()` | the ~45 `Config::*` values from env, the cache root, the callback table of 5.2 |
| submission (edge A) | `kyty_submit_dcb(addr, dw)`, `kyty_submit_acb(queue, addr, dw)`, `kyty_suspend_point()`, `kyty_wait_idle()`, `kyty_static_table_init(block)` | none |
| shaders | `kyty_register_shader(code, const KytyShaderInfo*)` | AnyPS5 `Shader` → plain struct (type, user-data pointer, input semantics pointer and count, sizes) |
| event queues | `kyty_add_eq_event(eq, id, udata)`, `kyty_delete_eq_event(eq, id)`, `kyty_eq_context_id(ev)` | `KernelEqueue*` → opaque handle; v4s's "patch `sync.cpp` to AnyPS5's `KernelEqueueEvent`" is not possible across runtimes and becomes the `eq_trigger` callback |
| presentation (edge B) | `kyty_attach_window(const KytyWindowDesc*)`, `kyty_register_video_output(handle, const KytyVideoOutApi*, ctx)`, `kyty_present_buffer(window, const KytyDisplayBuffer*, gpu_ready_fn, ctx)`, `kyty_present_clear(...)`, `kyty_submit_cpu_flip(const KytyFlipInfo*)` | `PresentationWindow` (extension list, surface and drawable-size callbacks) → C arrays and function pointers; `IVideoOutput`/`IFlipRequest` (virtual, `std::optional<DisplayBuffer>`) → a C function table and opaque request handles; `DisplayBuffer`, `FlipInfo` → plain structs |
| guest memory (edge C, AnyPS5 → Kyty) | `kyty_map_memory(va, n)`, `kyty_unmap_memory(va, n)`, `kyty_protect_changed(va, n, prot)`, `kyty_host_write_begin/end(va, n)`, `kyty_host_read_prepare(va, n)` | none; called from the `MappingChange` observer and the host write/read hooks |
| faults | `kyty_handle_fault(access, addr) → handled`, or `kyty_install_fault_handler()` | the vectored handler is a Windows-ABI callback and crosses as is; AnyPS5 keeps control of the order (Kyty before `GuestArenaHandleWrite`) |
| T1 and tools | frame-dump settings, screenshot request, stats as plain structs | the `[present]` and `[gpu] frame_NNN.bmp` lines are printed inside the DLL in AnyPS5's format |

### 5.2 Kyty DLL → AnyPS5 (one callback table, ~25 function pointers)

| group | callbacks | notes |
|---|---|---|
| memory (v4s §4.3 shim) | `query_direct_mapping(addr, *start, *end, *offset, *type)`, `backing_base/size`, `reserved_range(addr, *len)`, `prt_aperture(...)`, `direct_memory_views(phys, len, out[], n)` | the 22-function `kernel/memory.h` shim moves into the DLL; only these lookups call out. The range map and generation counter can live in the DLL |
| event queues | `eq_trigger(eq, ident, filter, data)`, `eq_pin(eq)` | AnyPS5 triggers with its own types |
| VideoOut | `try_reserve(ctx, const KytyFlipInfo*) → request`, `wait_for_flip_room(ctx)`, `capture_rendering_wait(ctx, index) → wait`, `wait(wait)`, `buffer(ctx, index, KytyDisplayBuffer*) → bool`, `gpu_ready(request, timing)`, `release(request)` | each C++ object becomes a handle owned by AnyPS5 |
| services | `read_tsc`, `tsc_frequency`, `process_time`, `process_time_counter`, `is_guest_thread`, `title_id/content_version/title(buf, n)`, `vulkan_loader_lock/unlock`, `log(level, msg, len)` | strings into caller buffers |

### 5.3 Rules the seam enforces

- **Allocation ownership.** Nothing allocated on one side is freed on the other. Strings go into caller buffers;
  objects are opaque handles released by their owner. Both runtimes end in the UCRT heap (which AnyPS5 redirects to
  mimalloc), so a slip would not corrupt the heap at once, but libc++ and libstdc++ objects have different layouts:
  no STL object crosses.
- **Exceptions.** Kyty builds with `-fno-exceptions`; AnyPS5 throws (exact-or-throw). Every AnyPS5 callback is a
  `noexcept` C trampoline that catches everything and returns an error code; nothing may unwind through DLL frames.
  The unwinders differ: AnyPS5's prx carry DWARF CFI that their own libc unwinder reads (`.ehfram`, §6); the DLL has
  SEH `.pdata/.xdata`.
- **SEH and the vectored handler.** One vectored handler for Kyty's page tracking, installed (or called) in AnyPS5's
  order. AnyPS5's crash report and `WorkerSampler` walk `.ehfram`; for DLL frames they need `RtlVirtualUnwind` on the
  DLL's `.pdata`. No `__try` in the DLL: clang for MinGW emits no handler for it (§1).
- **Thread creation.** Kyty creates and pins its own stage threads (libc++ `std::thread` over `CreateThread`) and sets
  the process default CPU sets, which also moves AnyPS5's guest threads (wanted, Z4). AnyPS5 threads (guest, VideoOut
  presenter) call into the DLL; their exit runs the DLL's `thread_local` destructors.
- **TLS.** Native TLS in the DLL (`_tls_index`, `gs:[0x58]`), emulated TLS in the GCC prx. They do not interact as
  long as no `thread_local` is shared across the seam; sharing one is exactly what breaks the hybrid (§1). Windows
  gives a DLL's static TLS to every thread, also when it is loaded with `LoadLibrary`. Guest code must never repoint
  `gs` (AnyPS5 relinks guest `fs` use; `gs` stays the TEB) [E, check once in phase 1].
- **Vulkan.** The DLL creates its own instance and device; AnyPS5's loader lock is taken through the callback.
- **Pipeline cache.** The DLL's cache must be warmed by the DLL's own build (libc++ changes some SPIR-V, §0.4).
- **Build and deploy.** llvm-mingw as a CMake ExternalProject (or a pinned prebuilt step); `kyty_gpu.dll` beside the
  prx in `game_run.sh`, `run1080.sh` and packaging. Linux builds the same sources with GCC there, or with clang +
  libc++ as a `.so`.

### 5.4 Size [E]

- C header: ~250-350 lines (~35 exports, ~25 callbacks, ~12 plain structs).
- AnyPS5 side: ~500-700 lines (trampolines; C++ objects to handles and structs).
- Kyty side: ~300-500 lines (the shim headers implemented over the callbacks).
- Build, deploy, CI: ~150-250 lines.
- **Total ~1.3-1.8k lines, 4-7 agent-days, on top of the ~2,950-line adapter.** A second toolchain stays, which
  upstream AnyPS5 would not take as is.

## 6. Option (c): does AnyPS5 build with clang today? No.

Read in `C:\ps5\AnyPS5-scratch\v4` (`local/v4` = yotei/best 72af1e5ea23) and its `.github/workflows`, plus small
compile tests with llvm-mingw in my scratch dir. No configure or build of AnyPS5 with clang was run.

- **CI and docs: GCC only for the libraries.** `build.yml`, `windows-msys2.yml` and `release.yml` build the libraries
  with `gcc`/`g++` (WinLibs 15.2 posix-seh or MSYS2 UCRT64 GCC). clang appears only in the relinker-only build on
  macOS (`ANYPS5_RELINKER_ONLY`). `docs/dev/BUILD.md`: "Windows: x86-64 MinGW-w64 GCC with POSIX threads".
- **The exception model is GCC-specific (the blocker).** `core/libs/CMakeLists.txt` checks for "DWARF exceptions with
  `-fno-asynchronous-unwind-tables`". If present, every library is compiled that way and `.eh_frame` is renamed to
  `.ehfram` with objcopy; AnyPS5's libc then unwinds those frames itself (`prx/libc/src/exception/Unwind.cpp`, 686
  lines, calling `__gxx_personality_v0`). The fallback path is "GCC with System V ABI SEH" plus `SehPersonalities.cpp`,
  which takes `__gxx_personality_seh0` and `__gcc_personality_seh0`.
  - clang for MinGW passes the DWARF check (it only rejects SJLJ) but ignores `-fno-asynchronous-unwind-tables`: it
    still emits SEH `.pdata/.xdata` and no `.eh_frame` [M object dump]. The libraries would build with no DWARF data
    for AnyPS5's unwinder.
  - `-fdwarf-exceptions` is accepted and emits `.eh_frame`, but llvm-mingw's libc++abi/libunwind are SEH-only: the
    link fails on `__gxx_personality_v0` [M].
  - The SEH path works in principle (clang passes AnyPS5's `sysv_abi` throw/catch check at run time [M]), but
    llvm-mingw has no `__gcc_personality_seh0` [M nm], and AnyPS5 would have to switch paths and re-prove the guest/host
    exception interplay.
- **Surface:** 368 prx directories; `APS5_VABI` (`sysv_abi`) 26,748 times in 1,089 files; 101 files with
  `thread_local` (emulated → native TLS changes their ABI); 1,484 registered tests. CMake hard-codes GCC's runtime DLLs
  for the tests (`libgcc_s_seh-1.dll`, `libstdc++-6.dll`, `libwinpthread-1.dll`) and uses GNU-ld options
  (`--exclude-libs`, `--exclude-symbols` with libstdc++'s mangled `operator new`, `-mcrtdll=ucrt`).
- **And it would not be enough:** with libstdc++ the backend stays at ×0.79 (§4), so AnyPS5 would have to move to
  libc++ as well, a change to every prx's ABI.
- **Cost [E]:** 1-2 weeks to a clang build of all libraries with the test suite green and the old backend's T1 MATCH,
  open-ended in the exception path; more for libc++; permanent divergence from upstream's toolchain.

## 7. Machine state at the end (17:25)

- No process of mine is running: all chains (A-G) and guarded builds finished; `ps` shows no `v4t`, `kyty_run`,
  `machine_guard`, `kyty_emulator`, clang or ninja process. No lock held when I left; I deleted no lock or marker.
- One process was stopped: my own `kyty_emulator.exe` of the invalid v4t-clang1 run (Windows PID 11312, game.lock tag
  `kyty-v4t-clang1`), and my own queued `machine_guard.sh` waiter (PID 2183852, before it started; requeued).
- Disk: C: 116 GB free (129 GB at the start; ≥ 85 kept throughout).
  - `research/v4/toolchain/`: 9.8 GB (b-clang, b-gcclto3, b-clanglstd2, b-clangemu, b-clanglstdn, ccache 2.5 GB,
    app-v4t 0.3 GB, exe/ 0.2 GB). Deleted: my failed build dirs b-gcclto, b-gcclto2, b-clanglstd, b-clanglstde, and
    the llvm-mingw zip after checking it.
  - `C:\ps5\toolchain\llvm-mingw-20261104`: 0.7 GB.
  - Worktrees `C:\ps5\kyty-src-wt\v4t` (`v4/clang`) and `C:\ps5\kyty-src-wt\v4t-gcclto` (`v4/gcc-lto`), 235 MB each.
- Branches in `C:\ps5\kyty-src` (refs only, nothing pushed): `v4/clang` 865cdd5e, `v4/gcc-lto` 071976bf,
  `v4/clang-lstd` ae9b0c89 (lab).
- Runs shared the slot with lanes v4a (T1 and Kyty runs) and v4p (its baseline run finished at 12:48); runs always went
  through `kyty_run.sh`, builds through the guard.
