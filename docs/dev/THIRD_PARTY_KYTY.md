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

`KytyShim::StartRuntime()` (`core/libs/prx/libSceAgcDriver/KytyShim/src/Config.cpp`) loads KytyPS5's `ConfigOptions` at
their defaults, then reads these environment variables, named after KytyPS5's command-line options. An invalid value
stops the emulator with a message naming the variable.

| Variables | Values |
|---|---|
| `KYTY_FULLSCREEN`, `KYTY_HDR`, `KYTY_HIDE_CURSOR`, `KYTY_STRETCH`, `KYTY_VR`, `KYTY_TRINITY`, `KYTY_VULKAN_VALIDATION`, `KYTY_GPU_ASSISTED_VALIDATION`, `KYTY_SHADER_VALIDATION`, `KYTY_SHADER_CACHE`, `KYTY_PIPELINE_MANIFEST`, `KYTY_RECORD_PIPELINE_MANIFEST`, `KYTY_STRICT_PIPELINE_MANIFEST`, `KYTY_TESSELLATION`, `KYTY_OCCLUSION_QUERIES`, `KYTY_RELAXED_SHADER_FLOAT`, `KYTY_ASYNC_PIPELINES`, `KYTY_RT_STATS`, `KYTY_COMMAND_BUFFER_DUMP`, `KYTY_GRAPHICS_DEBUG_DUMP`, `KYTY_PROFILE`, `KYTY_SPIRV_DEBUG_PRINTF`, `KYTY_RENDERDOC`, `KYTY_READBACK_LINEAR_IMAGES`, `KYTY_READBACK_TILED_IMAGES`, `KYTY_READBACK_STORAGE_IMAGES`, `KYTY_SYNC_RAW_IMAGE_BUFFERS`, `KYTY_PERF_OVERLAY`, `KYTY_TROPHY_NOTIFICATIONS` | `0` or `1` |
| `KYTY_NATIVE_RT` | unset: native ray tracing when the Vulkan device has `VK_KHR_acceleration_structure`, `VK_KHR_ray_query` and `VK_KHR_deferred_host_operations` with their features, emulated otherwise; `1`: native, and a stop with a message when the device cannot; `0`: emulated |
| `KYTY_SCREEN_WIDTH`, `KYTY_SCREEN_HEIGHT` | 1 or more |
| `KYTY_VBLANK_FREQUENCY`, `KYTY_FLUSH_DRAWS`, `KYTY_TROPHY_DURATION` | whole number |
| `KYTY_CONSOLE_LANGUAGE` | 0 to 29 |
| `KYTY_GPU` | GPU index, -1 for the default |
| `KYTY_DLSS_PRESET` | `A` to `O` |
| `KYTY_PRESENT_MODE` | `Fifo`, `Mailbox`, `Immediate` |
| `KYTY_PSSR_UPSCALER` | `Linear`, `DLSS`, `FSR` |
| `KYTY_PIPELINE_PRECOMPILE` | `Ask`, `Wait`, `Play` |
| `KYTY_SHADER_LOG_DIRECTION`, `KYTY_PRINTF_DIRECTION` | `Silent` (default), `Console`, `File` |
| `KYTY_DLSS_FOLDER`, `KYTY_FSR_FOLDER`, `KYTY_SHADER_LOG_FOLDER`, `KYTY_COMMAND_BUFFER_DUMP_FOLDER`, `KYTY_PRINTF_OUTPUT_FILE`, `KYTY_SESSION_LOG` | path |

KytyPS5's log (`LOGF`) is silent by default; `KYTY_PRINTF_DIRECTION=Console` sends it to stderr, beside AnyPS5's own
lines. `KYTY_GPU_PIPELINE`, `KYTY_GPU_THREAD_CPUS`, `KYTY_GPU_THREAD_PRIORITY` and `KYTY_RESOLVE_HELPER` are read by
`common/emulatorConfig.cpp` itself.

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
A waiter spins briefly, then yields. Three variables, read once:

| Variables | Values |
|---|---|
| `KYTY_PROGRAM_LOCK` | `narrow` (default) or `wide`: upstream's scope and upstream's endless spin |
| `KYTY_LOCK_STATS` | `0` (default) or `1`: every period, `[lock-stats]` lines on stderr with the wait and hold times of the lock per thread and per acquisition site, the time other threads waited on each site, and the guest reads of `GetProgram` inside and outside the lock |
| `KYTY_LOCK_STATS_PERIOD` | seconds between two reports, 1 to 3600, 5 by default |

KytyPS5's `native_rt` is off by default and its launcher's recommended settings turn it on; under `ANYPS5_KYTY` the
shim turns it on unless `KYTY_NATIVE_RT=0`, and Kyty enables the ray query extensions only on a device that has them.
Once the device exists, `KytyShim::ReportNativeRayTracing()` prints one `[kyty] native ray tracing` line on stderr:
on, off by `KYTY_NATIVE_RT=0`, or unavailable with the missing extension or feature named, in which case ray tracing
runs emulated as with `0`; with `KYTY_NATIVE_RT=1` that last case stops the emulator instead. `KYTY_RT_STATS` takes
effect whenever native ray tracing is on. `KYTY_RT_ARENA_MB` (MiB) replaces the limit of the memory the native BLASes
take, a quarter of the largest device-local heap clamped to 512 MiB to 4 GiB; it is read by
`graphics/host_gpu/renderer/rayTracingSlots.cpp`.
