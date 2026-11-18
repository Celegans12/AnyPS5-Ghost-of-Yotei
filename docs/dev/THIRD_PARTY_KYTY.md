# Kyty code in AnyPS5

`core/kyty/` and `core/libs/prx/libSceAgcDriver/KytyShim/include/` hold code taken from
a fork of [KytyPS5](https://github.com/KytyPS5/KytyPS5), at tag
`v24`, commit `40ddebbe21cd8a75f9da41fb66dad8104c341904`. It is the second GPU driver of plan B, built only with
`-DANYPS5_GPU_BACKEND=kyty`; the default `-DANYPS5_GPU_BACKEND=anyps5` builds AnyPS5 exactly as before and reads none of
these files.

## Licence

Kyty is distributed under the GNU General Public License version 2, copied in
[core/kyty/LICENSE](../../core/kyty/LICENSE). The original Kyty code by InoriRus was published under the MIT licence, copied
in [core/kyty/LICENSES/Kyty-MIT.txt](../../core/kyty/LICENSES/Kyty-MIT.txt). Two files carry a notice of their own, kept
verbatim at their top:

| File | Notice |
|---|---|
| `core/kyty/common/alignment.h` | 2014 Jannik Vogel, CC0-1.0 |
| `core/kyty/graphics/host_gpu/shaders/fault_buffer_process.comp` | 2024 shadPS4 Emulator Project, GPL-2.0-or-later |

## Folders

| AnyPS5 path | Kyty path at `40ddebbe` | Files |
|---|---|---|
| `core/kyty/graphics/` | `src/graphics/` | every file |
| `core/kyty/common/` | `src/common/` | every file except `CMakeLists.txt` |
| `core/kyty/cmake/` | `src/` | `embed_spirv.cmake`, `embed_localization.cmake`, `cmake_config.h.in`, `kytyGitVersion.h.in` |
| `core/libs/prx/libSceAgcDriver/KytyShim/include/kernel/` | `src/kernel/` | `eventQueue.h`, `fileSystem.h`, `guestPause.h`, `memory.h`, `pthread.h` |
| `core/libs/prx/libSceAgcDriver/KytyShim/include/libs/` | `src/libs/` | `agc.h`, `controller.h`, `errno.h`, `hmd2.h`, `ime.h`, `imeCommon.h`, `imeDialog.h`, `libs.h` |
| `core/libs/prx/libSceAgcDriver/KytyShim/include/loader/` | `src/loader/` | `demonsSoulsPageFree.h`, `demonsSoulsTitle.h`, `guestOodle.h`, `systemContent.h`, `timer.h` |

Outside those folders, code adapted from Kyty is built only with `-DANYPS5_GPU_BACKEND=kyty`:

| AnyPS5 path | Kyty path at `40ddebbe` | What |
|---|---|---|
| `core/libs/prx/libSceAgc/Kyty/` | `src/libs/agc.cpp` | `AgcGetFusedShaderSize`, `AgcFuseShaderHalves_0200`, `AgcFuseShaderHalves`, `AgcCreatePrimState`, `AgcUpdatePrimState`, `AgcGetGsOversubscription`, `AgcCreateInterpolantMapping`, `AgcCreateInterpolantMapping2`, `AgcGetRegisterDefaults2`, `AgcGetRegisterDefaults2Internal`, as AnyPS5's `sceAgc…` exports of the same NIDs; a missing register throws instead of dereferencing null |

The headers under `KytyShim/include/` are the contract of the shim: they are the declarations of the KytyPS5 kernel,
library and loader functions that `graphics/` and `common/` call, and the shim implements them on top of AnyPS5.
`libs/dialog.h` is left out: only `graphics/presentation/systemOverlay.cpp`, which `kyty_gpu` does not build, includes
it, and its `ErrorDialogOpen` and `ErrorDialogClose` would be second declarations of AnyPS5 exports for
`core/libs/tests/test_public_headers.py`, which reads every header under `core/` without namespaces.

## Changes from upstream

- Every comment is removed, except the two notices above. `core/kyty/tools/kyty_import.py` does it with a C++ lexer
  that keeps string, character and raw string literals and digit separators intact, and checks each file: the token
  stream that `g++ -fpreprocessed` produces is the same before and after, preprocessor line ends included.
- Fixes made in [KytyPS5](https://github.com/KytyPS5/KytyPS5) after `v24`, merged on its `main` or open as pull
  requests, are applied on top of `v24`, stripped of comments the same way, one commit each with a
  `Ported-from: KytyPS5 <commit>` line per upstream commit:

  | KytyPS5 commit | Pull request | Change |
  |---|---|---|
  | `174ec912` | — | `V_MSAD_U8` is translated |
  | `2f26870e` | — | the device enables `VK_KHR_compute_shader_derivatives`, whose capabilities the translator emits |
  | `a2ea94e9` | — | no swapchain is created while the window's surface has no area |
  | `50c0c066`, `18809cb0`, `15c8c618` | — | the VideoOut output transfer applies to linear formats, unpremultiplied PQ overlays are clamped, the HDR attribute check is part of the `ATTRIBUTE` default; `50c0c066` as Kyty's `main` resolves it on `v24` |
  | `2086171a` | — | `pm4Handlers.cpp` only: a flip block on an async compute queue (`R = 0`) is recognised |
  | `85e775f0` | #2062 | `RenderContext` builds the image heap before the pipeline cache |
  | `8d2e4376` | #2245 | `LOAD_*_REG_INDEX` honour their `data_format` field |
  | `bd2cffb2` | #2363 | write-only R16G16 signed storage images go through their unsigned view |
  | `3e2e67c9`, `44016a0a` | #2006 | block-compressed textures get no DCC metadata, in `DeriveTextureDesc` |
  | `d8fcae8e` | #1963 | `BASE_ARRAY` is ignored for 1D and 2D non-array views |
  | `bc553dca` | #1733 | depth-compare samples get a sampler variant of their own |
  | `222d8301` | #1405 | a lone last dword of a command buffer is consumed |
  | `7ee598b4`, `f26b1d7e` | #1403 | a compute dispatch's scalar reads may overlap the start of its own sentinel write range (`FindBuffers`); `a9d34f70`'s `CommitBindings` check is not in `v24` |
  | `d6ad3c2b` | #1802 | a buffer allocation refused by VMA's budget is retried once without the budget limit |

  `Presenter::PrepareFrame` takes the VideoOut pixel format since `50c0c066`: the shim passes the display buffer's
  `pixelFormat`.
- Nothing else is changed in the imported files. Adaptations to AnyPS5 go in the shim, or under
  `#if defined(ANYPS5_KYTY)` in `core/kyty/`. `ANYPS5_KYTY=1` is defined for every target of a
  `-DANYPS5_GPU_BACKEND=kyty` build and never otherwise.
- `graphics/` calls four functions of SDL3 (`SDL_LoadObject`, `SDL_LoadFunction`, `SDL_UnloadObject` to open the
  DLSS and FSR libraries when the user supplies them, `SDL_GetBasePath` to find shipped pipeline manifests). AnyPS5 has
  SDL2, so `core/libs/prx/libSceAgcDriver/KytyShim/compat/SDL3/` gives `kyty_gpu` these four over `dlopen` and
  `/proc/self/exe`; nothing else of SDL3 is declared there.
- `pipelineManifest.cpp` calls `stbi_zlib_decode_buffer`, whose implementation upstream compiles in
  `src/libs/libJpegDec.cpp`, which is not imported; AnyPS5 compiles `stb_image` as `static` in its own decoders, so
  `kyty_gpu` compiles one `STB_IMAGE_IMPLEMENTATION` unit of its own (generated `stb_image.c`).

## Build

Kyty builds with Clang only (GCC rejects its Vulkan-Hpp aggregate assignments and `__builtin_memcpy_inline`).
`core/kyty/CMakeLists.txt` is therefore a project of its own, which `core/kyty/cmake/KytyBackend.cmake` builds with
`clang` and `clang++` as the external project `kyty_gpu` when `-DANYPS5_GPU_BACKEND=kyty` is set; the rest of AnyPS5 stays
on GCC. Clang is given `--gcc-install-dir` of the GCC that builds AnyPS5, so both sides use the same libstdc++ headers.
The sub-build uses upstream's compiler options (`-march=x86-64-v3 -ffp-contract=off -fno-rtti -fno-exceptions -g`,
`-fexceptions` for `common/`, `descriptors.cpp` and `pipelineCache.cpp`), Tracy on demand, and its own SPIRV-Tools and
glslang (with the optimizer, for `glslangValidator -Os` on upstream's utility shaders). `Kyty::gpu` is the interface
target that links the resulting archives into AnyPS5. `core/kyty/KytyGpuSources.cmake` lists the imported sources left
out of `kyty_gpu`, and `core/kyty/tools/kyty_frontier.py` lists the symbols `kyty_gpu` still needs from the shim.

### Windows

On Windows AnyPS5 is built with MinGW-w64 GCC and `kyty_gpu` with the `clang` and `clang++` of an LLVM release found on
`PATH`, for the `x86_64-w64-windows-gnu` target, with `--sysroot` set to the MinGW-w64 installation of that GCC
(`ANYPS5_KYTY_SYSROOT`). Three options make its objects link with GCC's: `-femulated-tls` (GCC's thread-local storage
model on MinGW, which libstdc++ uses), `-fno-zero-initialized-in-bss` (GCC puts every inline variable in a `.data$`
section, and GNU ld only merges sections of one name) and `-mno-red-zone`. The shim is compiled with
`-fno-extern-tls-init`, and `MemoryTracker::s_deferred` is declared without `inline` for GCC there, because GCC on
MinGW emits the initialisation function of an inline `thread_local` variable once per translation unit. ThinLTO is off:
lld cannot pre-link COFF objects with `-r`. `pipelineHelpers.cpp` is compiled with `-fexceptions`, as its Windows part
uses `try`.

The second view of guest memory (`DirectMemoryEnableView`) is a reserved range. Anonymous guest memory in the arena is
mapped from pagefile-backed sections of 1 GiB, each created and mapped into the view when memory is first committed in
its range, and each direct memory backing is mapped at its physical offset after the guest range. A section page cannot
be given back to the system, so a page that held data is zeroed when it is unmapped and stays committed. Access
violations reach `KytyShim::HandleHostFault` from libc's vectored exception handler through
`GuestArenaSetGpuFaultHandler`. KytyPS5's pipeline helper processes are not used (`PipelineHelperProcesses` returns 0):
a helper is the executable started again with `--pipeline-helper`, and here the executable is the converted title.

## Third-party code

The dependencies of `graphics/` and `common/` that AnyPS5 did not have are submodules at the revisions Kyty
`v24` pins:

| Submodule | Revision | Licence |
|---|---|---|
| `3rdparty/fmt` | `11ddbcb7898d2d3445d431a54814367b21dee6ad` | MIT-style ({fmt} licence) |
| `3rdparty/xxHash` | `e573d4d2aaeaba0f3e5a0a9a54144a1f2b4b56e7` | BSD-2-Clause |
| `3rdparty/magic_enum` | `1384769c66bd16ec9bb1353f45fe8ec8ccc12dbd` (v0.9.8) | MIT |
| `3rdparty/xbyak` | `44a72f369268f7d552650891b296693e91db86bb` | BSD-3-Clause |
| `3rdparty/zydis` | `120e0e705f8e3b507dc49377ac2879979f0d545c` (with `dependencies/zycore`) | MIT |

`3rdparty/ffx-api` holds the four AMD FidelityFX API headers Kyty `v24` vendors in `3rdparty/ffx-api`
(FidelityFX-SDK v2.4.0, `Kits/FidelityFX`, unmodified, MIT licence in `3rdparty/ffx-api/LICENSE.txt`); `fsr.cpp` only
declares the API with them and loads AMD's libraries at run time.

`zydis` is used by the KytyPS5 loader (`x64InstructionEmulator.cpp`, `guestInstructionPatcher.cpp`), not by
`graphics/` or `common/`: `kyty_gpu` does not build it. Vulkan-Headers, VulkanMemoryAllocator, SPIRV-Headers,
SPIRV-Tools, glslang, tracy, stb, zstd and ZArchive are AnyPS5's own submodules, built a second time by the Clang
sub-build.

## Updating

```
python3 core/kyty/tools/kyty_import.py --ref <Kyty clone> --tag <tag> --dest .
```

rewrites the folders above from the given tag and reports any file whose tokens changed. Each commit that brings
Kyty code carries a `Ported-from: Kyty <folder>@<commit> (GPL-2.0)` line.

## Settings of the kyty build

A build with `-DANYPS5_GPU_BACKEND=kyty` reads its settings from environment variables, once per process. The tables below list every `KYTY_*` name the code under `core/` reads, one row each, and `tools/check_kyty_settings.py` fails when a name read in `core/` has no row here or a row names something the code no longer reads. Three kinds of reader exist:

- `KytyShim::StartRuntime()` (`core/libs/prx/libSceAgcDriver/KytyShim/src/Config.cpp`) loads KytyPS5's `ConfigOptions` at their defaults, then reads the variables named after KytyPS5's command-line options. An empty value counts as unset; an invalid value stops the emulator with a message naming the variable. Rows of this kind say "stops" in their values.
- Code of `core/kyty/`, from Kyty or ours, reads `KYTY_*` names with `std::getenv` or with a small helper that takes the name as a string literal. Each is read once; most accept anything and fall back to the default, as their values column says.
- `common/emulatorConfig.cpp` reads the thread and pipeline variables itself.

A name built at run time from pieces is not supported, because nothing can find it: pass the whole name as a literal.

How to read the columns:

- **Default** is what the code does with the variable unset, in this build; where it differs from Kyty the effect says so.
- **Category**: `perf` is a performance switch (the default says whether it is on), `compat` fixes or works around wrong output, `option` is a plain user option, `debug` is for debugging and measurement, `experimental` is unfinished, unproven or known to hurt, `inert` has no effect in this build because the code that reads it is not built or never runs.
- **Platform** is `all` unless a part of the effect exists on one system only; `—` for `inert`.
- **Origin** is `Kyty import` for what the import of `40ddebbe` brought with its name, `Kyty option` for a field of KytyPS5's `ConfigOptions` that `8a29429250` made readable from the environment, and an AnyKyty commit for what this fork added or changed.

### Window, display and upscaling

AnyPS5 opens the window, so Kyty's window options do nothing; `ANYPS5_FULLSCREEN`, `ANYPS5_DISPLAY_INDEX` and the others in [the AnyPS5 settings below](#anyps5-settings-that-an-anykyty-session-also-reads) apply.

| Variable | Values | Default | Effect | Category | Platform | Origin |
|---|---|---|---|---|---|---|
| `KYTY_FULLSCREEN` | `0` or `1`, else stops | `0` | Kyty's fullscreen option; nothing reads it in this build, `ANYPS5_FULLSCREEN` does the job | inert | — | Kyty option |
| `KYTY_HIDE_CURSOR` | `0` or `1`, else stops | `0` | Kyty's cursor hiding, in `hostInput.cpp`, which is not built | inert | — | Kyty option |
| `KYTY_BORDERLESS` | any value | unset | Borderless window flag, set in `window.cpp`, which is not built | inert | — | KytyPS5 |
| `KYTY_VR` | `0` or `1`, else stops | `0` | Kyty's OpenXR mode; `OpenXrInit` is an empty function in the shim | inert | — | Kyty option |
| `KYTY_OPENXR` | `0` | unset | Skips OpenXR start-up; read in `openxrHeadset.cpp`, which is not built | inert | — | KytyPS5 |
| `KYTY_RENDERDOC` | `0` or `1`, else stops | `0` | RenderDoc capture support, in `window.cpp` and `renderDoc.cpp`, which are not built | inert | — | Kyty option |
| `KYTY_CONSOLE_LANGUAGE` | 0 to 29, else stops | `1` | Language of Kyty's system overlay, in `systemOverlay.cpp`, which is not built; the title's language comes from AnyPS5 | inert | — | Kyty option |
| `KYTY_TROPHY_NOTIFICATIONS` | `0` or `1`, else stops | `1` | Trophy toasts of Kyty's system overlay; no code that is built reads the option | inert | — | Kyty option |
| `KYTY_TROPHY_DURATION` | whole number, else stops | `6` | Seconds a trophy toast shows (5 to 10), in `systemOverlay.cpp`, which is not built | inert | — | Kyty option |
| `KYTY_HDR` | `0` or `1`, else stops | `0` | Asks for an HDR10 swapchain format; without one the log says `HDR10 surface unavailable; using SDR output` | option | all | Kyty option |
| `KYTY_STRETCH` | `0` or `1`, else stops | `0` | Fills the whole window with the game's image instead of fitting its aspect ratio | option | all | Kyty option |
| `KYTY_SCREEN_WIDTH` | 1 or more, else stops | `1280` | Initial width of Kyty's window context; AnyPS5's window replaces it when it attaches, before any swapchain exists, so neither the resolution the title sees nor the render size depends on it (`APS5_OUTPUT_RESOLUTION` sets the former) | inert | — | Kyty option |
| `KYTY_SCREEN_HEIGHT` | 1 or more, else stops | `720` | Initial height of the same | inert | — | Kyty option |
| `KYTY_PRESENT_MODE` | `Fifo`, `Mailbox`, `Immediate`, any case, else stops | `Mailbox` | Vulkan present mode of the swapchain; `ANYPS5_PRESENT_MODE` is not read with this backend | option | all | Kyty option |
| `KYTY_GPU` | whole number, else stops | `-1` | Index of the Vulkan physical device; `-1`, or an index with no device (the log says so), lets Kyty choose; `ANYPS5_GPU` is not read with this backend | option | all | Kyty option |
| `KYTY_VBLANK_FREQUENCY` | whole number, clamped to 30 to 360, else stops | unset | Vblank rate when `ANYPS5_VBLANK_HZ` is unset or empty; it also becomes Kyty's own vblank frequency. Unset keeps the output's 59.94 Hz and 119.88 Hz | option | all | Kyty option; `bffccd0fbf` |
| `KYTY_PSSR_UPSCALER` | `Linear`, `DLSS`, `FSR`, any case, else stops | `Linear` | Upscaler of the PSSR path; `DLSS` and `FSR` load the vendor's library from the folder below through `SDL_LoadObject` | option | all | Kyty option |
| `KYTY_TRINITY` | `0` or `1`, else stops | `0` | Kyty's PS5 Pro mode (`pro_enabled`), which only the DLSS and FSR upscalers read | option | all | Kyty option |
| `KYTY_DLSS_FOLDER` | path | empty | Folder that holds the DLSS library | option | all | Kyty option |
| `KYTY_FSR_FOLDER` | path | empty | Folder that holds the FSR library | option | all | Kyty option |
| `KYTY_DLSS_PRESET` | `A` to `O`, else stops | unset | DLSS preset letter, 1 to 15 inside Kyty; unset keeps its 0 | option | all | Kyty option |

### GPU threads and queues

| Variable | Values | Default | Effect | Category | Platform | Origin |
|---|---|---|---|---|---|---|
| `KYTY_GPU_PIPELINE` | a number (base 0), else stops: `0` runs the whole GPU inline on one thread, `1` means every stage, other values keep the bits `1` recording, `2` command processor, `4` resolve | every stage (`7`) | Which stages of the emulated GPU run on threads of their own | perf | all | Kyty import |
| `KYTY_GPU_THREAD_CPUS` | unset, empty, `off`, or up to six logical CPUs separated by commas: command processor, execution, recording, resolve, host copy, resolve helper; else stops | unset | Unset or empty: on Linux each stage keeps to the CPUs of the largest last-level cache, on Windows KytyPS5's `{4, 6, 8, 10, 2, 0}`. `off`: no placement. A list pins the stages; stages past its end keep KytyPS5's value. See [below](#details) | perf | all | Kyty import; Linux default `c06a623c12` |
| `KYTY_GPU_THREAD_PRIORITY` | `0` | unset | `0` keeps the stage threads at their normal priority; otherwise they are raised, which only happens on Windows (`RaiseCurrentPriority` does nothing on Linux) | perf | Windows | Kyty import |
| `KYTY_GPU_SPIN_US` | microseconds; text that is not a number reads as `0` | `2000` | How long a pipeline thread spins on an empty or full handoff before it waits in the kernel; `0` never spins | perf | all | Kyty import |
| `KYTY_RESOLVE_HELPER` | `1` | unset | A second thread resolves some draws ahead of their execution beside the resolve thread; it takes the sixth entry of `KYTY_GPU_THREAD_CPUS` | perf | all | Kyty import |
| `KYTY_SCOUT` | `0` to `3`, any other text reads as `3` | `3` | Look-ahead on queue 0: `0` off, `1` uses the scout thread's tickets, `2` its verifier, `3` the resolve thread evaluates each draw's tables ahead | perf | all | Kyty import |
| `KYTY_FLUSH_DRAWS` | whole number, else stops | `0` | Submits the open command buffer every this many draws; `0` never does it on a count | perf | all | Kyty option |
| `KYTY_SUSPEND_POINT_DEPTH` | 1 to 64, else stops | `8` | Suspend points the GPU may have in flight; `1` is KytyPS5's behaviour. See [below](#details) | perf | all | `b370e0778c` |
| `KYTY_PROGRAM_LOCK` | `narrow` or `wide`, else stops | `narrow` | Scope of the program-cache lock; `wide` is upstream's scope and endless spin. See [below](#details) | perf | all | `6d40e04a28` |
| `KYTY_LOCK_STATS` | `0` or `1`, else stops | `0` | Every period, `[lock-stats]` lines on stderr: wait and hold times of the lock per thread and per acquisition site, and the guest reads of `GetProgram` inside and outside the lock | debug | all | `6d40e04a28` |
| `KYTY_LOCK_STATS_PERIOD` | whole seconds, 1 to 3600, else stops | `5` | Seconds between two `[lock-stats]` reports | debug | all | `6d40e04a28` |

### Draws, tables, native records and shaders

The table draws are GPU-driven draws of a shader pair's table programs; the native XPR records are their older fallback, off by default. The sub-switches of the records matter only with `KYTY_NATIVE_XPR=1`, except where the effect says otherwise.

| Variable | Values | Default | Effect | Category | Platform | Origin |
|---|---|---|---|---|---|---|
| `KYTY_DRAW_TEMPLATES` | `0`, `1`, `2`; other text reads as `1` | `2` (KytyPS5: off) | Draw templates: `1` turns them on, `2` also keys them by the draw's registers and takes the pipeline and render targets from them | perf | all | Kyty import; on by default `50d7c13842` |
| `KYTY_DRAW_BUFFERS` | `1`; other text is off | on (KytyPS5: off) | Copies the small read-only buffer ranges of a draw to the stream buffer together before its stages bind them | perf | all | Kyty import; on by default `50d7c13842` |
| `KYTY_TABLE_XPR` | `0`, `1`, `2` (read in base 0) | `2` | Table draws: `0` off, `1` on, `2` also continues runs of clean draws, which the command processor's `DrawStateObserver` finds | perf | all | Kyty import |
| `KYTY_TABLE_DISPATCH` | `0`, `1` | `0`; `1` for Demon's Souls (`PPSA01340` to `PPSA01342`) | Dispatches of a shader pair that learned its table program go through the table | perf | all | Kyty import |
| `KYTY_TABLE_INDIRECT` | number | `1` | Read and never used: the indirect-table code is compiled out (`KYTY_CX_INDIRECT` is 0) | inert | — | Kyty import |
| `KYTY_TABLE_STORE_BUDGET` | number; `0` no limit | `32` | Table stores a frame asks for; the others are deferred to the normal path | perf | all | Kyty import |
| `KYTY_TABLE_SET_PROOFS` | `0`, `1`, `2` | `1` | Checks of a table's image sets: `0` checks every use in full, `1` skips a use whose proof still holds, `2` does and counts the uses a proof held for that did not pass | perf | all | Kyty import |
| `KYTY_TABLE_MRU` | bit mask | `7` | Bit `1` a larger pair cache, `2` variant hints, `4` reuse of the previous draw's set when its words match | perf | all | Kyty import |
| `KYTY_TABLE_TARGET_READ` | number | `0` | A render target the shaders sample is held with the shader access added, as the normal path transits it | experimental | all | Kyty import |
| `KYTY_TABLE_GATHER_MEMO` | number | `1` | Memoises the buffer descriptors a table gathers | perf | all | Kyty import |
| `KYTY_TABLE_MEMO` | number | `1` | Remembers by shader pair and state key what a table draw drew with last | perf | all | Kyty import |
| `KYTY_TABLE_MEMO_VERIFY` | number | `0` | Every Nth memo hit is checked against the full path, and a difference is counted | debug | all | Kyty import |
| `KYTY_TABLE_SERIALS` | `0`, `1`, `2` | `0` | Decides the target and set checks by sums of image serials: `1` skips a check when the sums did not move, `2` runs the checks and reports those the sums would have skipped | experimental | all | Kyty import |
| `KYTY_TABLE_STRIDES` | first character `0` is off | on | Table programs read an unswizzled buffer of any stride from the block, where otherwise they specialise on the stride | perf | all | Kyty import |
| `KYTY_DRAW_READ_AFTER_WRITE` | number | `1` | A draw whose image set binds one image as texture and as written storage transits it at every draw, after the last draw's write | compat | all | Kyty import |
| `KYTY_STATE_KEY_CACHE` | first character `0` is off | on | Hashes the draw-state registers again only for another snapshot of them; also used by table draws | perf | all | Kyty import |
| `KYTY_ASYNC_XPR_PIPELINES` | first character `0` is off | on | Creates the pipeline of a record or table variant on a background thread; `0` does it on the execution thread; also used by table draws | perf | all | Kyty import |
| `KYTY_NATIVE_XPR` | first character `1` is on | off | Native XPR records: an indexed indirect draw is replayed from a record of its object. Off because upstream saw Demon's Souls' walls drawn dark and another title slower with it | experimental | all | Kyty import |
| `KYTY_NATIVE_XPR_RELOCATE` | first character `0` is off | on | Relocated records, which rebind buffers whose bases moved | experimental | all | Kyty import |
| `KYTY_NATIVE_XPR_INSTANCES` | first character `1` is on | off | One relocated record per instance of a mesh; upstream's Demon's Souls crashed in the driver with it | experimental | all | Kyty import |
| `KYTY_NATIVE_XPR_DIRECT` | first character `0` is off | on | Direct indexed draws use records; `0` sends them to the normal path | experimental | all | Kyty import |
| `KYTY_NATIVE_XPR_VB` | first character `0` is off | on | Draws with vertex buffers are stored as records; `0` does not store them | experimental | all | Kyty import |
| `KYTY_NATIVE_IMAGE_PROOF` | first character `0` is off | on | A record's sampled images are checked again only when their own registration changed; `0` checks them whenever any image was registered or unregistered | experimental | all | Kyty import |
| `KYTY_NATIVE_XPR_STORE_BUDGET` | number; `0` no limit | `256` | Record stores asked for a frame | experimental | all | Kyty import |
| `KYTY_NATIVE_XPR_KEEP_FRAMES` | number, at least `2` | `600` | Frames an unused record stays | experimental | all | Kyty import |
| `KYTY_NATIVE_XPR_PREDICT` | first character `0` is off | on | Prefetches the record the next draw used last time | experimental | all | Kyty import |
| `KYTY_NATIVE_XPR_STATS` | first character `1` is on | off | Prints record counters every 300 frames | debug | all | Kyty import |
| `KYTY_XPR_RESOLVE_SKIP` | first character `0` is off | on | Skips the resolution ahead of a draw whose pair the last native try drew; `0` resolves them all | experimental | all | Kyty import |
| `KYTY_BDA_REGIONS` | number, `0` off | `1` for Demon's Souls, else `0` | Synchronises device-address buffers by region requests with epoch proofs instead of a pass over every mapped range | perf | all | Kyty import |
| `KYTY_BDA_DIRTY_REGIONS` | number, `0` off | `1` | With `KYTY_BDA_REGIONS`, a bounded preparation visits only the regions whose dirty bit is set | perf | all | Kyty import |
| `KYTY_UNIFORM_BUFFERS` | first character `0` is off | on | Shaders read constants at small immediate offsets through uniform buffer views (needs `uniformBufferStandardLayout`); `0` reads them through the storage buffer | perf | all | Kyty import |
| `KYTY_NATIVE_F16` | first character `0` is off | on | Shaders compute F16 natively (needs `shaderFloat16`); `0` does it per lane in F32 | perf | all | Kyty import |
| `KYTY_TESSELLATION` | `0` or `1`, else stops | `1` | Draws patch primitives with tessellation; `0` makes the draw path refuse them | option | all | Kyty option |
| `KYTY_OCCLUSION_QUERIES` | `0` or `1`, else stops | `1` | Host occlusion queries, which need the device's `hostQueryReset`; `0` turns them off | option | all | Kyty option |
| `KYTY_RELAXED_SHADER_FLOAT` | `0` or `1`, else stops | `0` | Relaxes the shader recompiler's float rules (min and max, trigonometric functions) for speed | experimental | all | Kyty option |

### Memory, readbacks, flips and labels

| Variable | Values | Default | Effect | Category | Platform | Origin |
|---|---|---|---|---|---|---|
| `KYTY_READBACK_LINEAR_IMAGES` | `0` or `1`, else stops | `1` | Copies a GPU-written linear image back to guest memory at the next suspend point | compat | all | Kyty option |
| `KYTY_READBACK_TILED_IMAGES` | `0` or `1`, else stops | `0` | The same for tiled images, which is slow | compat | all | Kyty option |
| `KYTY_READBACK_STORAGE_IMAGES` | `0` or `1`, else stops | `0` | Copies every storage image back on its first write; see `KYTY_WRITE_BACK_IMAGES` | compat | all | Kyty option |
| `KYTY_WRITE_BACK_IMAGES` | `0` or `1`, empty counts as unset, else stops | on | Copies the queued readbacks back at each `ACQUIRE_MEM` or `RELEASE_MEM` with GL2 write-back and at each flip; `0` is KytyPS5's behaviour. See [below](#details) | compat | all | `9d957fd81e` |
| `KYTY_TILE_VIA_VRAM` | `0` or `1`, empty counts as unset, else stops | on | Tiles a tiled image's readback in device-local memory, then copies it once to the host; `0` is KytyPS5's behaviour. See [below](#details) | perf | all | `08c666a175` |
| `KYTY_SYNC_RAW_IMAGE_BUFFERS` | `0` or `1`, else stops | `0` | Obtains a raw buffer that a shader reads the way a formatted (texel) buffer is obtained | compat | all | Kyty option |
| `KYTY_COPY_FEEDBACK` | `0` is off | on | After a small GPU copy into guest memory, snapshots the destination so that a later guest read is served from the snapshot instead of a readback | perf | all | Kyty import |
| `KYTY_ASYNC_REPROTECT` | `0` is off | on | The write protection of the pages an upload marks clean is applied by the host copy thread, ahead of the copy | perf | all | Kyty import |
| `KYTY_ASYNC_IMAGE_STAGING` | `0` or `1`; empty counts as unset | on for Demon's Souls, else off | The staging copies of image uploads run on the host copy thread | perf | all | Kyty import |
| `KYTY_ENGINE_FREE_BUFFERS` | `0` off, `2` drops GPU-written bytes, else on | on | Buffers over a block the game freed and kept mapped go as for an unmap. Its only entry, `NoteGameFree`, has no caller in this build: it belongs to Demon's Souls' page-free hook, which is not imported | inert | — | Kyty import |
| `KYTY_EARLY_FLIP` | `0` or `1`, empty counts as unset, else stops | on | Completes a guest flip once its command buffer is submitted, on a thread of its own; `0` is KytyPS5's behaviour. See [below](#details) | perf | all | `499dfc3f39` |
| `KYTY_LABELS_AFTER_DUMPS` | `0` or `1`, empty counts as unset, else stops | on | A deferred `RELEASE_MEM` label waits for the newest pending occlusion dump instead of its own command buffer; `0` is KytyPS5's behaviour. See [below](#details) | perf | all | `ef88cc3285` |

### Pipelines and shader store

| Variable | Values | Default | Effect | Category | Platform | Origin |
|---|---|---|---|---|---|---|
| `KYTY_ASYNC_PIPELINES` | `0` or `1`, else stops | `1` | For a draw that only changes the picture, and is not a small colour target, the pipeline is created on a background thread and the draw is skipped until it is ready; `0` always waits for it | perf | all | Kyty option |
| `KYTY_PIPELINE_LIBRARIES` | `0` is off | on | Links graphics pipelines from per-stage pipeline libraries (`VK_EXT_graphics_pipeline_library`) | perf | all | `76827a9140` |
| `KYTY_PIPELINE_LIBRARIES_AHEAD` | `0` is off | on | Prepares the stage libraries a draw will need ahead of it, while libraries were used in the last 5 seconds | perf | all | `76827a9140` |
| `KYTY_PIPELINE_STALL_LOG_MS` | milliseconds; text that is not a number reads as `0`, which logs every one | `100` | Logs a `[kyty-gpl]` line for a pipeline stall or shader translation that takes at least this long. If explicitly set, also logs completed `vkCreateComputePipelines` calls at or above the same threshold, with elapsed time, shader hash, module ID, SPIR-V word count, lookup or create mode, and Vulkan result | debug | all | `76827a9140` |
| `KYTY_PIPELINE_MANIFEST` | `0` or `1`, else stops | `1` | Warms the pipelines listed in the title's manifest (shipped with the assets, or recorded in an earlier session) at start | perf | all | Kyty option |
| `KYTY_RECORD_PIPELINE_MANIFEST` | `0` or `1`, else stops | `0` | Records the pipelines of the session as a manifest even when the manifest is off; the shader store is not used while it records | debug | all | Kyty option |
| `KYTY_STRICT_PIPELINE_MANIFEST` | `0` or `1`, else stops | `0` (the build's `KYTY_STRICT_PIPELINE_MANIFEST_DEFAULT`) | A pipeline of a shipped manifest that fails to build stops the emulator instead of being dropped | debug | all | Kyty option |
| `KYTY_PIPELINE_PRECOMPILE` | `Ask`, `Wait`, `Play`, any case, else stops | `Ask` | Whether the game stops while the pipelines of the manifest compile. Stopping needs the helper processes, which this build never starts, and `GuestPause::Supported()`, which the shim answers with `false`; the notice that asks is not drawn either, so every value plays at once | inert | — | Kyty option |
| `KYTY_PIPELINE_HELPERS` | number | unset | Number of helper processes that compile pipelines. They are never started in this build: `PipelineHelperProcesses` returns 0 on Linux and when `ANYPS5_KYTY` is defined | inert | — | Kyty import |
| `KYTY_SHADER_CACHE` | `0` or `1`, else stops | `0` | Keeps recompiled shaders in `_PipelineCache/<title id>.shaders` and warms them up at start; a Release build only, and not while `KYTY_GRAPHICS_DEBUG_DUMP` or the manifest recording is on | perf | all | Kyty option |
| `KYTY_SHADER_CACHE_STAMP` | integer in base 0, else ignored | the stamp the build computes from the shader recompiler's sources | Replaces the stamp written in the shader store's header; `0` turns the store off | debug | all | `4749ffc567` |
| `KYTY_SHADER_CACHE_THREADS` | whole number, `0` or text is ignored | `Common::ParallelThreads()`, at most the number of jobs | Threads that warm the shader store up | perf | all | `4749ffc567` |
| `KYTY_SHADER_CACHE_SHARED` | `1` shared, any other value per-thread | shared, except per-thread with NVIDIA's driver | One shared Vulkan pipeline cache for the warm-up threads, or one cache per thread | perf | all | `4749ffc567` |

### Ray tracing

| Variable | Values | Default | Effect | Category | Platform | Origin |
|---|---|---|---|---|---|---|
| `KYTY_NATIVE_RT` | `0` or `1`, else stops | unset | Unset: native ray tracing when the device has `VK_KHR_acceleration_structure`, `VK_KHR_ray_query` and `VK_KHR_deferred_host_operations` with their features, emulated otherwise. `1`: native, and a stop when the device cannot. `0`: emulated. KytyPS5's own default is off | perf | all | Kyty option; on by default `8a29429250` |
| `KYTY_RT_STATS` | `0` or `1`, else stops | `0` | With native ray tracing, shaders count their queries and an `[RT]` summary of the last 10 seconds is logged | debug | all | Kyty option |
| `KYTY_RT_ARENA_MB` | MiB, empty counts as unset | a quarter of the largest device-local heap, 512 MiB to 4 GiB | Limit of the memory the native BLASes take; `0` means no limit | perf | all | Kyty import |

### Logs, dumps, validation and profiling

| Variable | Values | Default | Effect | Category | Platform | Origin |
|---|---|---|---|---|---|---|
| `KYTY_PRINTF_DIRECTION` | `Silent`, `Console`, `File`, any case, else stops | `Silent` | Where Kyty's log (`LOGF`) goes: nowhere, stderr beside AnyPS5's own lines, or `KYTY_PRINTF_OUTPUT_FILE` | debug | all | Kyty option |
| `KYTY_PRINTF_OUTPUT_FILE` | path | `_kyty.txt` | File of the `File` direction | debug | all | Kyty option |
| `KYTY_SESSION_LOG` | path | empty | Session log file, opened at start with a header | debug | all | Kyty option |
| `KYTY_RELAUNCHED` | any value | unset | Makes the log files append instead of truncate. KytyPS5's launcher sets it when it starts the emulator again; nothing sets it here | debug | all | KytyPS5 |
| `KYTY_SHADER_LOG_DIRECTION` | `Silent`, `Console`, `File`, any case, else stops | `Silent` | Anything but `Silent` makes the recompiler dump its IR; the shader dumps go under `KYTY_SHADER_LOG_FOLDER` | debug | all | Kyty option |
| `KYTY_SHADER_LOG_FOLDER` | path | `_Shaders` | Folder of the shader dumps | debug | all | Kyty option |
| `KYTY_COMMAND_BUFFER_DUMP` | `0` or `1`, else stops | `0` | Writes each command buffer the title submits to a file under `KYTY_COMMAND_BUFFER_DUMP_FOLDER`, named by frame and number | debug | all | Kyty option |
| `KYTY_COMMAND_BUFFER_DUMP_FOLDER` | path | `_Buffers` | Folder of the command buffer dumps | debug | all | Kyty option |
| `KYTY_GRAPHICS_DEBUG_DUMP` | `0` or `1`, else stops | `0` | Emits debug info in the shaders and turns the shader store off | debug | all | Kyty option |
| `KYTY_PROFILE` | `0` or `1`, else stops | `0` | Starts the Tracy profiler, which listens on `127.0.0.1:8086` | debug | all | Kyty option |
| `KYTY_VULKAN_VALIDATION` | `0` or `1`, else stops | `0` | Enables the Khronos validation layer | debug | all | Kyty option |
| `KYTY_GPU_ASSISTED_VALIDATION` | `0` or `1`, else stops | `0` | GPU-assisted validation; it turns `KYTY_VULKAN_VALIDATION` on | debug | all | Kyty option |
| `KYTY_SHADER_VALIDATION` | `0` or `1`, else stops | `0` | Runs the SPIRV-Tools validator on each recompiled module and logs what it finds | debug | all | Kyty option |
| `KYTY_SPIRV_DEBUG_PRINTF` | `0` or `1`, else stops | `0` | Enables `debugPrintfEXT` in the generated shaders | debug | all | Kyty option |
| `KYTY_CX_OBSERVER` | any value | unset | Counts what dirtied the registers between draws, by first cause | debug | all | Kyty import |
| `KYTY_PERF_OVERLAY` | `0` or `1`, else stops | `0` | Switches Kyty's debug counters on; the overlay that draws them is not built, and `PerformanceOverlayEnabled()` is `false` in the shim | debug | all | Kyty option |

### AnyPS5 settings that an AnyKyty session also reads

The window, the flip and vblank pacing, the guest memory and the thread priority of the presenter are AnyPS5's, so their `ANYPS5_*` and `APS5_*` settings apply with the Kyty backend. Each has its row in [ENVIRONMENT](ENVIRONMENT.md); this table says what it does here. Settings read only by AnyPS5's own GPU driver (`core/libs/prx/libSceAgcDriver/Execution/`, which this build does not compile) have no effect: `ANYPS5_GPU` and `ANYPS5_PRESENT_MODE` are replaced by `KYTY_GPU` and `KYTY_PRESENT_MODE`.

| Variable | Values | Default | Effect with the Kyty backend | Category | Platform | Origin |
|---|---|---|---|---|---|---|
| `ANYPS5_VBLANK_HZ` | whole number, 30 to 360, else stops | unset | Vblank rate; unset or empty takes `KYTY_VBLANK_FREQUENCY`, and the value also becomes Kyty's vblank frequency, so `ANYPS5_VBLANK_HZ=240` lifts the 60 FPS cap of titles that pace on vblank | option | all | `bffccd0fbf` |
| `APS5_UNPACED_FLIPS` | `0`, `1` or empty, else stops | `0` | Benchmark aid: each flip completes once presented, not at the next vblank, so a title that renders faster than the vblank presents as fast as it can; set `KYTY_PRESENT_MODE=Immediate` or `Mailbox` too to present above the display's rate | debug | all | `b3691289ef` |
| `ANYPS5_KYTY_PRESTART` | `0` or `1`, empty counts as unset, else stops | `1` | Starts the Kyty GPU on a host thread as soon as the video output is created; `0` starts it after the window exists | perf | all | `e465445589` |
| `ANYPS5_KYTY_SPARSE_READS` | `0` or `1`, empty counts as unset, else stops | `1` | The Kyty GPU asks the memory files where the data of a 64 KiB or larger range is (`SEEK_DATA`) and writes zeros for the never-written parts without touching them; `0` reads every byte | perf | Linux | `21c1eae177` |
| `APS5_NO_DIRECT_GUEST_COPY` | presence | unset | On Linux x86-64 the checked guest copies read with plain loads whose faults the signal handlers turn into a failed copy; set, every checked read is a `process_vm_readv` system call again | debug | Linux | `004c1c4ede` |
| `ANYPS5_FULLSCREEN` | `0`, `1`, `windowed`, `borderless` | `windowed` | Opens the display window fullscreen at the desktop resolution; the window AnyKyty presents into is AnyPS5's, so this replaces `KYTY_FULLSCREEN` | option | all | AnyPS5 |
| `ANYPS5_DISPLAY_INDEX` | index of a connected display, else stops | `0` | Display that the window opens on | option | all | AnyPS5 |
| `ANYPS5_FPS_LIMIT` | 0.5 to 1000000 with at most three decimals, else stops | `0` (no limit) | Most frames per second the flips present | option | all | AnyPS5 |
| `APS5_OUTPUT_RESOLUTION` | `<width>x<height>`, else the output status calls throw | unset: follows the display | Output resolution `sceVideoOutGetOutputStatus` reports to the title, 1080p class under 3840 by 2160 and 2160p class otherwise; it does not change the render size | option | all | AnyPS5 |
| `ANYPS5_VIDEO_OUT_MAX_WIDTH`, `ANYPS5_VIDEO_OUT_MAX_HEIGHT` | positive integers, else output-mode queries throw | unset | Host display hint: set below 3840 (width) or 2160 (height), an output mode that AnyPS5 does not know is reported as unsupported, so that the title falls back to the host resolution | option | all | AnyPS5 |
| `APS5_SYNC_FLIP` | presence | unset | Debug aid: the flip worker waits until the presenter is done with each flip | debug | all | AnyPS5 |
| `APS5_THREAD_PRIORITY` | `off`, `high`, `rt` | `off` | Raises the presenter thread above the title's threads; Kyty's own stage threads follow `KYTY_GPU_THREAD_PRIORITY` instead | perf | all | `a9c132a629` |
| `APS5_THREAD_NICE`, `APS5_THREAD_RT`, `APS5_THREAD_RTTIME_US` | see [ENVIRONMENT](ENVIRONMENT.md) | see there | Tune that raised priority on Linux | perf | Linux | `a9c132a629` |

### Details

The settings below change how a part of the driver works, so each has its reasoning and measurements here; the tables above say what to set.

KytyPS5's log (`LOGF`) is silent by default; `KYTY_PRINTF_DIRECTION=Console` sends it to stderr, beside AnyPS5's own lines.

GPU stage threads on Linux: KytyPS5 places them only on Windows (`{4, 6, 8, 10, 2, 0}` counts SMT siblings as adjacent
logical processors, and its core reservation uses Windows CPU sets). Under `ANYPS5_KYTY` on Linux, with
`KYTY_GPU_THREAD_CPUS` unset or empty, each stage thread (command processor, execution, recording, resolve, host copy,
resolve helper) keeps to the logical CPUs that share the largest last-level cache (`/sys/devices/system/cpu`: on a Ryzen
9 7950X3D, CPUs 0-7 and 16-23, the 3D V-Cache CCD), intersected with its current affinity. No core is reserved, and the
threads it creates afterwards inherit that set. Measured on Subnautica: +7 % to +22 % frames per second, 25-35 % less CPU
time per draw on each stage (`notes/au-dela-p4-placement.md`). `KYTY_GPU_THREAD_CPUS=off` turns it off. An explicit list
pins each stage to that Linux logical CPU, without core reservation; threads a stage creates afterwards inherit its CPU.

Images the game reads after the GPU writes them: KytyPS5 copies a GPU-written linear image back to guest memory at the
next suspend point, unless `KYTY_READBACK_STORAGE_IMAGES=1`. A game that waits for the label of a `RELEASE_MEM` with GL2
write-back and then reads the image reads stale memory: in TCG Card Shop Simulator (PPSA36301) a 64×64 RGBA32F image is
read that way and the street stays almost black. Under `ANYPS5_KYTY`, the write-back half of that setting is on by
default: at each `ACQUIRE_MEM` or `RELEASE_MEM` with GL2 write-back and at each flip, the images already queued for
readback are copied back, and their pages stay read-protected until the copy lands. No other image is copied: the copy
of every storage image on its first write still needs `KYTY_READBACK_STORAGE_IMAGES=1`. `KYTY_WRITE_BACK_IMAGES=0`
restores KytyPS5's behaviour; any value other than empty, `0` or `1` stops the emulator with a message naming the
variable.

Readback of tiled images: KytyPS5 copies a GPU-written tiled image back to guest memory
(`TextureCache::DownloadImageMemory`; under VRAM pressure, the texture cache's garbage collector does it for each image
it evicts) by retiling it with a compute shader straight into the host-visible readback buffer. The shader writes in
tiled order, a few bytes at a time across PCIe: about 0.75 s for a 3840×2160 render target on an RTX 4080 SUPER, while
the game waits. Under `ANYPS5_KYTY`, when the readback buffer is not device-local, `TileManager` copies the readback
range into its device-local scratch buffer, tiles there, and copies the range back with one `vkCmdCopyBuffer`; the bytes
the shader does not write keep the guest value they had, as before. If the scratch buffer is not device-local either,
the shader writes to the readback buffer as before. Stray with 8 GiB of the 16 GiB held by another process: worst frame
5.46 s → 0.57 s, readbacks over 100 ms 9 → 0; with a second Stray on the GPU: 6.74 s → 0.67 s; the same images
(`notes/kyty-tile-via-vram.md`). `KYTY_TILE_VIA_VRAM=0` restores KytyPS5's behaviour; any value other than empty, `0` or
`1` stops the emulator with a message naming the variable.

Guest flips: KytyPS5 completes a flip (`VideoOutDriver::CompleteFlip`, which makes the request ready for presentation;
the title's flip event follows) on its priority thread, once the GPU has finished the command buffer that holds the
presentation copy. The title then waits for the end of the GPU's work before it submits its next frame. Under
`ANYPS5_KYTY`, `CommandScheduler::DeferFlipCompletion` completes the flip on a thread of its own, outside the
scheduler's locks, once that command buffer has been submitted to the Vulkan queue and the GPU has finished the command
buffer of the previous flip, with the priority operations queued before it. The present blit is submitted after the copy
on the same queue and its barrier waits for the copy, so the presented image does not change; the title's next frame
overlaps the end of the GPU's work on this one, with at most two frames in flight. The display buffer the flip frees is
the one shown before, whose copy the GPU has finished, and a title that renders into a display buffer first waits for it
on the GPU (rendering wait packet). Measured on Subnautica, in two sessions of 17 paired periods of 10 s: +1.7 % and
+4.8 % frames per second, the same images (`notes/au-dela-p9-bascule.md`). `KYTY_EARLY_FLIP=0` restores KytyPS5's behaviour; any value other than empty, `0` or
`1` stops the emulator with a message naming the variable.

Suspend points: KytyPS5 makes `sceAgcSuspendPoint` (`GuestGpu::SuspendPoint`) acquire a binary semaphore that its
priority thread releases once the GPU has finished the work queued before the previous suspend point, so a title can
never queue work more than one suspend point ahead of the GPU. On the console, and in the AnyPS5 driver
(`Driver::SuspendPoint`), a suspend point only marks a place where the system may suspend the title, and it returns at
once. Under `ANYPS5_KYTY` the semaphore counts up to `KYTY_SUSPEND_POINT_DEPTH` suspend points in flight, 8 by default,
and a suspend point that follows a flip also waits until the GPU has finished the work before the previous such suspend
point, so a title never runs more than one frame ahead of the GPU: the AnyPS5 driver keeps the same bound by holding a
submission that carries a flip until the earlier flip's work is done. Balatro calls it once a frame, after a submission
that ends with a flip, and has no other fence before it rewrites its transient vertex memory: with eight frames in flight
it rewrote vertex buffer tables the command processor had not read yet, and stopped at launch on a vertex buffer
descriptor with `ADD_TID` set (`notes/balatro-regression.md`). Stray calls it after each
submission, about five times a frame: with one in flight it could not submit a frame before the GPU had finished the
previous one, and the stages sat idle for the GPU's tail every frame. The depth has to cover a whole frame of suspend
points: on Stray at 4,240 draws a frame, depths 1, 4, 8 and 16 give 56.2, 56.5, 71.6 and 71.9 frames per second, the same
images; Subnautica gives 208.6 and 212.7 at depths 1 and 8 (`notes/stray-perf-cpu.md`). `KYTY_SUSPEND_POINT_DEPTH=1`
restores KytyPS5's behaviour; a value outside 1 to 64 stops the emulator with a message naming the variable.

Deferred labels: while an occlusion-counter dump waits for the end of its command buffer, or while a queue has a deferred
label, KytyPS5 defers each `RELEASE_MEM` label to the end of the command buffer it was recorded into, on the same
first-in first-out priority queue as the dump publications, so that labels and occlusion counters become visible in the
order the GPU wrote them. That order only needs a label to wait for the dumps older than it, not for its own command
buffer: when nothing is pending, KytyPS5 already writes a label as soon as the execute thread reaches it, and the buffer
and texture caches keep the data the GPU wrote before it. Under `ANYPS5_KYTY`, with `KYTY_LABELS_AFTER_DUMPS` at 1, the
default, a deferred label waits for the command buffer of the newest pending dump instead of its own, still behind every
older priority operation, and a priority operation never takes an older tick than the one queued before it, which
`WaitPriorityOperations` relies on. While any queue has a deferred label, every queue defers its labels: otherwise a
queue with none pending writes its label at once, ahead of an older label of another queue that the game may already have
waited for, and Stray recycles that older label's memory before its write lands (a crash in its small-block allocator
at address 1). Unreal Engine 4 keeps occlusion queries pending almost all the time, and the command processor waits about
once a frame for a deferred label: on Stray at 4,240 draws a frame, the execute thread goes from 81 to 95 percent of the
frame and the frame rate from 71.0 to 80.1 and 80.9 frames per second, the same images (`notes/stray-label-flush.md`).
`KYTY_LABELS_AFTER_DUMPS=0` restores KytyPS5's behaviour; another value stops the emulator with a message naming the
variable.

Blocked waits on deferred writes: KytyPS5's command processor submits the open command buffer after a pass over a
submission that made progress, but not after one that yields to another queue, nor when a queue resumes straight into a
wait that still fails. If the value that queue waits for is a deferred write recorded into the open command buffer, and
no other submission comes, nothing submits that buffer and the title hangs. Under `ANYPS5_KYTY`, a queue that blocks
without progress on a range with a pending deferred write submits the open command buffer, and submits it again only when
a newer deferred write lands on that range while it still waits, whatever `KYTY_LABELS_AFTER_DUMPS` says; deferring the
labels of every queue makes the case more likely in titles without occlusion queries.

`m_program_lock`, the spin lock of `graphics/host_gpu/renderer/pipeline/pipelineCache.cpp` that guards the program
cache, has a narrower scope than upstream's in this build (`programLock.cpp`). The resolve thread always runs upstream's
unlocked ahead mode, which upstream only uses with `KYTY_RESOLVE_HELPER=1`, and the execute thread releases the lock
while it materializes the resources of a cached program, which can wait for the GPU when it reads GPU-written buffers.
A waiter spins briefly, then yields. `KYTY_PROGRAM_LOCK`, `KYTY_LOCK_STATS` and `KYTY_LOCK_STATS_PERIOD` are read once.

KytyPS5's `native_rt` is off by default and its launcher's recommended settings turn it on; under `ANYPS5_KYTY` the
shim turns it on unless `KYTY_NATIVE_RT=0`, and Kyty enables the ray query extensions only on a device that has them.
Once the device exists, `KytyShim::ReportNativeRayTracing()` prints one `[kyty] native ray tracing` line on stderr:
on, off by `KYTY_NATIVE_RT=0`, or unavailable with the missing extension or feature named, in which case ray tracing
runs emulated as with `0`; with `KYTY_NATIVE_RT=1` that last case stops the emulator instead. `KYTY_RT_STATS` takes
effect whenever native ray tracing is on. `KYTY_RT_ARENA_MB` (MiB) replaces the limit of the memory the native BLASes
take, a quarter of the largest device-local heap clamped to 512 MiB to 4 GiB; it is read by
`graphics/host_gpu/renderer/rayTracingSlots.cpp`.

### Adding a setting

A pull request that makes the code under `core/` read a new `KYTY_*` name adds its row to the tables above, with its default taken from the code, and one that stops reading a name removes the row. `python3 tools/check_kyty_settings.py` lists what is missing or stale; the `conventions` job of [the AnyKyty workflow](../../.github/workflows/anykyty.yml) runs it. A new `ANYPS5_*` or `APS5_*` name also needs its row in [ENVIRONMENT](ENVIRONMENT.md).
