# Kyty code in AnyPS5

`core/kyty/` and `core/libs/prx/libSceAgcDriver/KytyShim/include/` hold the Kyty core. `core/kyty` is maintained in this repository; it derives from [KytyPS5](https://github.com/KytyPS5/KytyPS5) (GPL-2.0).
It is the GPU driver of AnyKyty, the only one: the historical AnyPS5 driver (the `Execution/` and `Graphics/` folders of
`libSceAgcDriver`) and its shader recompiler (`core/shader`) were removed, and the build takes no GPU driver option.

## Licence

The Kyty core is distributed under the GNU General Public License version 2, copied in
[core/kyty/LICENSE](../../core/kyty/LICENSE). The original Kyty code by InoriRus was published under the MIT licence, copied
in [core/kyty/LICENSES/Kyty-MIT.txt](../../core/kyty/LICENSES/Kyty-MIT.txt). Two files carry a notice of their own, kept
verbatim at their top:

| File | Notice |
|---|---|
| `core/kyty/common/alignment.h` | 2014 Jannik Vogel, CC0-1.0 |
| `core/kyty/graphics/host_gpu/shaders/fault_buffer_process.comp` | 2024 shadPS4 Emulator Project, GPL-2.0-or-later |

## Folders

| AnyPS5 path | Path in the imported tree | Files |
|---|---|---|
| `core/kyty/graphics/` | `src/graphics/` | every file |
| `core/kyty/common/` | `src/common/` | every file except `CMakeLists.txt` |
| `core/kyty/cmake/` | `src/` | `embed_spirv.cmake`, `cmake_config.h.in`, `kytyGitVersion.h.in` |
| `core/libs/prx/libSceAgcDriver/KytyShim/include/kernel/` | `src/kernel/` | `eventQueue.h`, `fileSystem.h`, `memory.h`, `pthread.h` |
| `core/libs/prx/libSceAgcDriver/KytyShim/include/libs/` | `src/libs/` | `agc.h`, `controller.h`, `errno.h`, `hmd2.h`, `libs.h` |
| `core/libs/prx/libSceAgcDriver/KytyShim/include/loader/` | `src/loader/` | `demonsSoulsPageFree.h`, `demonsSoulsTitle.h`, `guestOodle.h`, `systemContent.h`, `timer.h` |

Outside those folders, code adapted from the Kyty core is built into AnyKyty as well:

| AnyPS5 path | Path in the imported tree | What |
|---|---|---|
| `core/libs/prx/libSceAgc/Kyty/` | `src/libs/agc.cpp` | `AgcGetFusedShaderSize`, `AgcFuseShaderHalves_0200`, `AgcFuseShaderHalves`, `AgcCreatePrimState`, `AgcUpdatePrimState`, `AgcGetGsOversubscription`, `AgcCreateInterpolantMapping`, `AgcCreateInterpolantMapping2`, `AgcGetRegisterDefaults2`, `AgcGetRegisterDefaults2Internal`, as AnyPS5's `sceAgc…` exports of the same NIDs; a missing register throws instead of dereferencing null |

The headers under `KytyShim/include/` are the contract of the shim: they are the declarations of the KytyPS5 kernel,
library and loader functions that `graphics/` and `common/` call, and the shim implements them on top of AnyPS5.
`libs/dialog.h` is left out: no source of `kyty_gpu` includes it, and its `ErrorDialogOpen` and `ErrorDialogClose` would
be second declarations of AnyPS5 exports for `core/libs/tests/test_public_headers.py`, which reads every header under
`core/` without namespaces.

`core/kyty/cmake/PatchSpirvTools.cmake` is not from KytyPS5: it is AnyPS5's patch of the bundled SPIRV-Tools, which `core/kyty/CMakeLists.txt` includes, moved here from the removed `core/shader/recompiler`.

## Changes from the import

- Every comment is removed, except the two notices above. `core/kyty/tools/kyty_import.py` does it with a C++ lexer
  that keeps string, character and raw string literals and digit separators intact, and checks each file: the token
  stream that `g++ -fpreprocessed` produces is the same before and after, preprocessor line ends included.
- Fixes made in [KytyPS5](https://github.com/KytyPS5/KytyPS5) after the imported revision, merged on its `main` or open as pull
  requests, are applied on top of it, stripped of comments the same way, one commit each with a
  `Ported-from: KytyPS5 <commit>` line per upstream commit:

  | KytyPS5 commit | Pull request | Change |
  |---|---|---|
  | `174ec912` | — | `V_MSAD_U8` is translated |
  | `2f26870e` | — | the device enables `VK_KHR_compute_shader_derivatives`, whose capabilities the translator emits |
  | `a2ea94e9` | — | no swapchain is created while the window's surface has no area |
  | `50c0c066`, `18809cb0`, `15c8c618` | — | the VideoOut output transfer applies to linear formats, unpremultiplied PQ overlays are clamped, the HDR attribute check is part of the `ATTRIBUTE` default; `50c0c066` adapted to the imported revision |
  | `2086171a` | — | `pm4Handlers.cpp` only: a flip block on an async compute queue (`R = 0`) is recognised |
  | `85e775f0` | #2062 | `RenderContext` builds the image heap before the pipeline cache |
  | `8d2e4376` | #2245 | `LOAD_*_REG_INDEX` honour their `data_format` field |
  | `bd2cffb2` | #2363 | write-only R16G16 signed storage images go through their unsigned view |
  | `3e2e67c9`, `44016a0a` | #2006 | block-compressed textures get no DCC metadata, in `DeriveTextureDesc` |
  | `d8fcae8e` | #1963 | `BASE_ARRAY` is ignored for 1D and 2D non-array views |
  | `bc553dca` | #1733 | depth-compare samples get a sampler variant of their own |
  | `222d8301` | #1405 | a lone last dword of a command buffer is consumed |
  | `7ee598b4`, `f26b1d7e` | #1403 | a compute dispatch's scalar reads may overlap the start of its own sentinel write range (`FindBuffers`); `a9d34f70`'s `CommitBindings` check is not in the imported revision |
  | `d6ad3c2b` | #1802 | a buffer allocation refused by VMA's budget is retried once without the budget limit |
  | `fc705b4b` | — | `DmaData` no longer drops writes whose destination's low 32 bits are `0x3022c` |
  | `f216821b` | — | an `INDIRECT_BUFFER` target's two low bits are cleared before its commands are fetched |
  | `4cfdc862` | — | `FindImage` keeps a view's mip and layer across later overlap candidates |
  | `f130c02b` | — | `WRITE_DATA` to the GDS destination writes the GDS buffer; adapted to the deferred `WriteData` (one GPU-ordered callback) |
  | `046bf882` | — | the presenter copies only layer 0 of a flipped surface's cached image |
  | `8d0ebb8f` | — | the SRT walker evaluates `SMin32` |
  | `46213dfd` | — | shorter compiled SRT walkers (biased scratch register, bases formed once per walk, folded constant offsets, inline tests of values already evaluated, plain marks of later blocks); ported by hand onto the walker's page cache, flat words and ahead loads, kept after an in-run A/B (+4.0 % on Ghost of Yotei) |
  | `f6e3373d` | #2407 | `AssociateStencil` tells the image heap before it rewrites an image as a stencil plane |

  `Presenter::PrepareFrame` takes the VideoOut pixel format since `50c0c066`: the shim passes the display buffer's
  `pixelFormat`.
- Nothing else is changed in the imported files. Adaptations to AnyPS5 go in the shim, or under
  `#if defined(ANYPS5_KYTY)` in `core/kyty/`. `ANYPS5_KYTY=1` is defined for every target.
- `graphics/` calls four functions of SDL3 (`SDL_LoadObject`, `SDL_LoadFunction`, `SDL_UnloadObject` to open the
  DLSS and FSR libraries when the user supplies them, `SDL_GetBasePath` to find shipped pipeline manifests). AnyPS5 has
  SDL2, so `core/libs/prx/libSceAgcDriver/KytyShim/compat/SDL3/` gives `kyty_gpu` these four over `dlopen` and
  `/proc/self/exe`; nothing else of SDL3 is declared there.
- `pipelineManifest.cpp` calls `stbi_zlib_decode_buffer`, whose implementation upstream compiles in
  `src/libs/libJpegDec.cpp`, which is not imported; AnyPS5 compiles `stb_image` as `static` in its own decoders, so
  `kyty_gpu` compiles one `STB_IMAGE_IMPLEMENTATION` unit of its own (generated `stb_image.c`).

## Build

The Kyty core builds with Clang only (GCC rejects its Vulkan-Hpp aggregate assignments and `__builtin_memcpy_inline`).
`core/kyty/CMakeLists.txt` is therefore a project of its own, which `core/kyty/cmake/KytyBackend.cmake` builds with
`clang` and `clang++` as the external project `kyty_gpu` in every full build; the rest of AnyPS5 stays
on GCC. Clang is given `--gcc-install-dir` of the GCC that builds AnyPS5, so both sides use the same libstdc++ headers.
The sub-build uses upstream's compiler options (`-march=x86-64-v3 -ffp-contract=off -fno-rtti -fno-exceptions -g`,
`-fexceptions` for `common/`, `descriptors.cpp` and `pipelineCache.cpp`), Tracy on demand, and its own SPIRV-Tools and
glslang (with the optimizer, for `glslangValidator -Os` on upstream's utility shaders). `Kyty::gpu` is the interface
target that links the resulting archives into AnyPS5. `kyty_gpu` builds every source under `graphics/` and `common/`:
the imported sources it had no use for (the window, input, overlays, OpenXR, RenderDoc, trophies, localization and the
upstream log) are not in the tree. `core/kyty/tools/kyty_frontier.py` lists the symbols `kyty_gpu` still needs from the
shim.
`kyty_gpu` is compiled with the profile of `core/kyty/pgo/` by default ([profile-guided optimization](KYTY_PGO.md)).

### Windows

On Windows AnyPS5 is built with MinGW-w64 GCC and `kyty_gpu` with the `clang` and `clang++` of an LLVM release found on
`PATH`, for the `x86_64-w64-windows-gnu` target, with `--sysroot` set to the MinGW-w64 installation of that GCC
(`ANYPS5_KYTY_SYSROOT`). Three options make its objects link with GCC's: `-femulated-tls` (GCC's thread-local storage
model on MinGW, which libstdc++ uses), `-fno-zero-initialized-in-bss` (GCC puts every inline variable in a `.data$`
section, and GNU ld only merges sections of one name) and `-mno-red-zone`. The shim is compiled with
`-fno-extern-tls-init`, and `MemoryTracker::s_deferred` is declared without `inline` for GCC there, because GCC on
MinGW emits the initialisation function of an inline `thread_local` variable once per translation unit. ThinLTO is off:
lld cannot pre-link COFF objects with `-r`.

The second view of guest memory (`DirectMemoryEnableView`) is a reserved range. Anonymous guest memory in the arena is
mapped from pagefile-backed sections of 1 GiB, each created and mapped into the view when memory is first committed in
its range, and each direct memory backing is mapped at its physical offset after the guest range. A section page cannot
be given back to the system, so a page that held data is zeroed when it is unmapped and stays committed. Access
violations reach `KytyShim::HandleHostFault` from libc's vectored exception handler through
`GuestArenaSetGpuFaultHandler`. KytyPS5's pipeline helper processes are not imported: a helper is the executable started
again with `--pipeline-helper`, and here the executable is the converted title.

## Third-party code

The dependencies of `graphics/` and `common/` that AnyPS5 did not have are submodules at the revisions the imported
tree pinned:

| Submodule | Revision | Licence |
|---|---|---|
| `3rdparty/fmt` | `11ddbcb7898d2d3445d431a54814367b21dee6ad` | MIT-style ({fmt} licence) |
| `3rdparty/xxHash` | `e573d4d2aaeaba0f3e5a0a9a54144a1f2b4b56e7` | BSD-2-Clause |
| `3rdparty/magic_enum` | `1384769c66bd16ec9bb1353f45fe8ec8ccc12dbd` (v0.9.8) | MIT |
| `3rdparty/xbyak` | `44a72f369268f7d552650891b296693e91db86bb` | BSD-3-Clause |
| `3rdparty/zydis` | `120e0e705f8e3b507dc49377ac2879979f0d545c` (with `dependencies/zycore`) | MIT |

`3rdparty/ffx-api` holds the four AMD FidelityFX API headers the imported tree vendored in `3rdparty/ffx-api`
(FidelityFX-SDK v2.4.0, `Kits/FidelityFX`, unmodified, MIT licence in `3rdparty/ffx-api/LICENSE.txt`); `fsr.cpp` only
declares the API with them and loads AMD's libraries at run time.

`zydis` is used by the KytyPS5 loader (`x64InstructionEmulator.cpp`, `guestInstructionPatcher.cpp`), not by
`graphics/` or `common/`: `kyty_gpu` does not build it. Vulkan-Headers, VulkanMemoryAllocator, SPIRV-Headers,
SPIRV-Tools, glslang, tracy, stb, zstd and ZArchive are AnyPS5's own submodules, built a second time by the Clang
sub-build.

## Import tool

`core/kyty/tools/kyty_import.py` is historical: it made the first import of the folders above from a Kyty source tree,
without comments, and reported any file whose tokens changed. `core/kyty` is maintained in this repository; fixes taken
from [KytyPS5](https://github.com/KytyPS5/KytyPS5) carry a `Ported-from: KytyPS5 <commit>` line.

## Settings of the kyty build

A build reads its settings from environment variables, once per process. The tables below list every `KYTY_*` name the code under `core/` reads, one row each, and `tools/check_kyty_settings.py` fails when a name read in `core/` has no row here or a row names something the code no longer reads. Three kinds of reader exist:

- `KytyShim::StartRuntime()` (`core/libs/prx/libSceAgcDriver/KytyShim/src/Config.cpp`) loads KytyPS5's `ConfigOptions` at their defaults, then reads the variables named after KytyPS5's command-line options. An empty value counts as unset; an invalid value stops the emulator with a message naming the variable. Rows of this kind say "stops" in their values.
- Code of `core/kyty/`, imported or ours, reads `KYTY_*` names with `std::getenv` or with a small helper that takes the name as a string literal. Each is read once; most accept anything and fall back to the default, as their values column says.
- `common/emulatorConfig.cpp` reads the thread and pipeline variables itself.

A name built at run time from pieces is not supported, because nothing can find it: pass the whole name as a literal.

How to read the columns:

- **Default** is what the code does with the variable unset, in this build; where it differs from the imported Kyty core the effect says so.
- **Category**: `perf` is a performance switch (the default says whether it is on), `compat` fixes or works around wrong output, `option` is a plain user option, `debug` is for debugging and measurement, `experimental` is unfinished, unproven or known to hurt, `inert` has no effect in this build because the code that reads it is not built or never runs.
- **Platform** is `all` unless a part of the effect exists on one system only; `—` for `inert`.
- **Origin** is `Kyty import` for what the import of the Kyty core brought with its name, `Kyty option` for a field of KytyPS5's `ConfigOptions` that `8a29429250` made readable from the environment, and an AnyKyty commit for what this fork added or changed.

### Window, display and upscaling

AnyPS5 opens the window, so Kyty's window options do nothing; `ANYPS5_FULLSCREEN`, `ANYPS5_DISPLAY_INDEX` and the others in [the AnyPS5 settings below](#anyps5-settings-that-an-anykyty-session-also-reads) apply. This build does not read KytyPS5's `KYTY_FULLSCREEN`, `KYTY_HIDE_CURSOR`, `KYTY_VR`, `KYTY_RENDERDOC`, `KYTY_SCREEN_WIDTH`, `KYTY_SCREEN_HEIGHT`, `KYTY_CONSOLE_LANGUAGE`, `KYTY_TROPHY_NOTIFICATIONS` or `KYTY_TROPHY_DURATION`: the code they drive is not built, and the initial window size gives way to AnyPS5's window before any swapchain exists, so setting one changes nothing. KytyPS5's `KYTY_GPU`, an index into the list of Vulkan devices, is not read either: `ANYPS5_GPU` picks the device by a part of its name, as in AnyPS5's own driver.

| Variable | Values | Default | Effect | Category | Platform | Origin |
|---|---|---|---|---|---|---|
| `KYTY_HDR` | `0` or `1`, else stops | `0` | Asks for an HDR10 swapchain format; without one the log says `HDR10 surface unavailable; using SDR output` | option | all | Kyty option |
| `KYTY_STRETCH` | `0` or `1`, else stops | `0` | Fills the whole window with the game's image instead of fitting its aspect ratio | option | all | Kyty option |
| `KYTY_PRESENT_MODE` | `Fifo`, `Mailbox`, `Immediate`, any case, else stops | `Mailbox` | Vulkan present mode of the swapchain | option | all | Kyty option |
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
| `KYTY_GPU_THREAD_CPUS` | unset, empty, `off`, or up to five logical CPUs separated by commas: command processor, execution, recording, resolve, host copy; else stops; on Windows a CPU the machine does not have also stops | unset | Unset or empty: each stage keeps to the CPUs of the largest last-level cache, on Windows pinned to one core of it each. `off`: no placement. A list pins the stages; stages past its end keep KytyPS5's `{4, 6, 8, 10, 2}`. See [below](#details) | perf | all | Kyty import; Linux default `c06a623c12` |
| `KYTY_GPU_SPIN_US` | microseconds; text that is not a number reads as `0` | `2000` | How long a pipeline thread spins on an empty or full handoff before it waits in the kernel; `0` never spins | perf | all | Kyty import |
| `KYTY_FAST_TLS` | `0` turns it off, any other value or unset leaves it on | on | Serves the emulated TLS of `libSceAgcDriver.prx`'s own `thread_local` variables from one native TLS slot instead of libgcc's `__emutls_get_address`, whose `pthread_getspecific` takes a spin lock on every access and whose storage a thread frees before its `thread_local` destructors run, which corrupts the heap when the thread exits (#116); a thread's slot storage is freed after those destructors; variables of other modules keep libgcc's | perf | Windows | AnyKyty |
| `KYTY_REPLAY_GPU_PAGE_READS` | `0` turns it off, any other value or unset leaves it on | on | A guest read that faults on a GPU-owned page, on bytes no GPU write is near, is replayed on the backing view instead of waiting for a readback; instructions that write memory or read the shifted register are never replayed; `0` waits for the readback as before; on Windows the replay ends on the single step exception of the vectored handler | perf | all | AnyKyty |
| `KYTY_REPLAY_AFTER_READBACK` | `0` turns it off, any other value or unset leaves it on | on | A guest read that faults on bytes the GPU wrote no longer waits for the execution thread to release the page after the copy; the copied bytes are read by replaying the instruction on the backing view, and the release is posted without waiting; an instruction the replay cannot serve waits for the release as before; `0` always waits for the release | perf | all | AnyKyty |
| `KYTY_HOT_PAGE_WRITE_WATCH` | `1` turns it on, any other value or unset leaves it off | off | Linux: a watched page the guest writes twice within a second stops faulting; it stays writable under asynchronous userfaultfd write protection, and its writes are collected with `PAGEMAP_SCAN` when a submission starts, when a memory wait or semaphore passes, before a CPU flip, and before the GPU writes a range that holds it; a page idle for 4 s, or whose mapping changes, goes back to `mprotect`; `[kyty-hot-pages]` sums it up every 10 s; off by default because the collections cost more than the faults they remove in a city scene, where memory waits pass thousands of times a second | perf | Linux | AnyKyty |
| `KYTY_COMMAND_MEMORY_IN_PLACE` | `1` turns it on, anything else leaves it off | off | Experimental: before labels waited for their own tick here, the GPU stopped while an open-world AAA title loaded (a compute program looping on memory served in place); since, one session of that title ran without a stop but 8 % slower (the execute thread drains the GPU for labels it cannot defer). A buffer of the cache that touches command memory (the 64 KiB blocks the command processor reads and the command buffer segments `libSceAgc` writes, as marked for the write fault blocks, see [Details](#details)) and holds no GPU-written page is a `VkBuffer` bound to the guest's direct memory itself, imported with `VK_EXT_external_memory_host` in chunks of 256 MiB of the second view: the GPU reads it where the guest writes it, its pages are never write protected nor uploaded, and a command buffer the guest refills no longer faults. A buffer already in video memory moves in place at the next frame after a guest write fault in command memory. While any buffer is in place, the host follows the GPU instead of running ahead of it: `RELEASE_MEM` labels are written once the GPU reached them, flips complete once their frame is done (no early flip) and wait for the previous frame (one frame in flight), every submission ends with a barrier that makes GPU writes visible to the host, `WRITE_DATA`, fills and copies into such memory run on the GPU in order, and a command processor read of a block the GPU writes in flight waits for that write. Direct memory only (the guest file is remapped by the guest); textures keep their own copies. `[kyty-command-memory]` sums up every 10 s the buffers served in place, the conversions, the refusals, the pinned chunks, the pages handed to and from tracking, the GPU writes noted and the reads that waited, the labels written at completion and the flips that waited | perf | all | AnyKyty |
| `KYTY_MIRROR_CHECK` | `0` or `1`, else stops | `0` | On a device that binds sparse buffers, the buffer cache keeps its buffers as windows of one sparse buffer that mirrors the guest memory (offset and device address follow the guest address), whose 64 KiB blocks are bound by a thread of their own before the submission that first uses them, which waits for the bind on the GPU. A range used from the mirror while one of its blocks is not bound (a binding, an upload, a readback) is counted in the `[kyty-mirror]` summary every 10 s; `1` stops at the first one with its address. `vkCmdUpdateBuffer` into the mirror always stops | debug | all | AnyKyty |
| `KYTY_SUSPEND_POINT_DEPTH` | 1 to 64, else stops | `8` | Suspend points the GPU may have in flight; `1` is KytyPS5's behaviour. See [below](#details) | perf | all | `b370e0778c` |
| `KYTY_PROGRAM_LOCK` | `narrow` or `wide`, else stops | `narrow` | Scope of the program-cache lock; `wide` is upstream's scope and endless spin. See [below](#details) | perf | all | `6d40e04a28` |
| `KYTY_LOCK_STATS` | `0` or `1`, else stops | `0` | Every period, `[lock-stats]` lines on stderr: wait and hold times of the lock per thread and per acquisition site, and the guest reads of `GetProgram` inside and outside the lock | debug | all | `6d40e04a28` |
| `KYTY_LOCK_STATS_PERIOD` | whole seconds, 1 to 3600, else stops | `5` | Seconds between two `[lock-stats]` reports | debug | all | `6d40e04a28` |

### Draws, tables and shaders

The table draws are GPU-driven draws of a shader pair's table programs.

| Variable | Values | Default | Effect | Category | Platform | Origin |
|---|---|---|---|---|---|---|
| `KYTY_DRAW_TEMPLATES` | `0`, `1`, `2`; other text reads as `1` | `2` (KytyPS5: off) | Draw templates: `1` turns them on, `2` also keys them by the draw's registers and takes the pipeline and render targets from them | perf | all | Kyty import; on by default `50d7c13842` |
| `KYTY_DRAW_BUFFERS` | `1`; other text is off | on (KytyPS5: off) | Copies the small read-only buffer ranges of a draw to the stream buffer together before its stages bind them | perf | all | Kyty import; on by default `50d7c13842` |
| `KYTY_TABLE_XPR` | `0`, `1`, `2` (read in base 0) | `2` | Table draws: `0` off, `1` on, `2` also continues runs of clean draws, which the command processor's `DrawStateObserver` finds | perf | all | Kyty import |
| `KYTY_TABLE_DESCRIPTOR_WRITES` | `0` or `1`; empty counts as unset, else stops | `1` when the device's `VkPhysicalDeviceDriverProperties::driverID` is `VK_DRIVER_ID_MESA_RADV`, else `0` | A draw whose image tables are reused but whose ring set is not current fills each table with descriptor writes from the table's host copy; a slot whose image view is not proven alive is still copied. `0` copies each table's whole capacity from its persistent set; on RADV those copies took most of the execute thread's time in a CPU profile, on NVIDIA the writes were slower | perf | all | AnyKyty |
| `KYTY_TABLE_RESOLUTIONS` | `1` to `4096`; empty counts as unset, else stops | `256` | Entries of the cache that resolves the image tables of the normal draw path, one per table key, least recently used evicted. Evicting or renewing an entry invalidates only the table states that use it (one generation per entry), and the elements of all entries are kept under the 8 × 65,536 of the old cache by clearing the least recently used entries. Kyty's cache kept 8 entries and one generation for all, so creating, evicting or clearing any entry made every table state resolve its tables again; a tester's title rebuilt its table states 1,355 times every 10 s that way (#99). The switch back to it, `KYTY_TABLE_RESOLUTION_LEGACY`, was removed | perf | all | AnyKyty |
| `KYTY_TABLE_MEMO_VERIFY` | number | `0` | Every Nth memo hit is checked against the full path, and a difference is counted | debug | all | Kyty import |
| `KYTY_TESSELLATION` | `0` or `1`, else stops | `1` | Draws patch primitives with tessellation; `0` drops them before their programs are looked up or prepared | option | all | Kyty option |
| `KYTY_OCCLUSION_QUERIES` | `0` or `1`, else stops | `1` | Host occlusion queries, which need the device's `hostQueryReset`; `0` turns them off | option | all | Kyty option |
| `KYTY_RELAXED_SHADER_FLOAT` | `0` or `1`, else stops | `0` | Relaxes the shader recompiler's float rules (min and max, trigonometric functions) for speed | experimental | all | Kyty option |
| `KYTY_WAVE_SPLIT_CHECK` | dispatch period, `0` turns it off | `0` | On GPUs without wave64 compute, a wave64 compute program whose workgroup is whole waves (64 to 1024 threads) runs each wave as two 32-wide subgroups that exchange cross-lane values through workgroup memory, and only a program the split refuses runs in lane pairs (every instruction twice). The halves of a workgroup of several waves wait on each other through a counter in workgroup memory, which only pays when the split gives each lane twice the registers and the program needs them: such a workgroup is split only with at most 512 threads and a loop that carries at least 48 values, or when its lanes store different values to one workgroup memory word (the split makes the last lane win, as on RDNA, which lane pairs do not), and otherwise stays in lane pairs. With this setting, one direct dispatch in N of every split program is played twice: split module, outputs (buffers and images) saved and previous contents restored, then lane pair module, whose outputs stay; every other check plays the lane pair module twice instead, as a witness of what the program leaves to the order of its atomics. A compute pass counts the output dwords that differ, and a `[kyty-split-check]` line gives the first check and the first witness of each program and every check or witness that differs, with the first differing dword, the output it falls in and both values; one in 16 N once a program passed 4 checks. Programs whose outputs cannot be replayed, as for `KYTY_RT_CHECK`, or whose lane pair module has as many SPIR-V words as the split one are not checked (one line says why). The lane pair module is translated by the background compile workers: a check waits until it is ready | debug | compute | AnyKyty |
| `KYTY_FALLBACKS` | `0` or `1`, else stops | `1` on Linux, `0` on Windows | Where the emulator would stop on something Kyty does not handle and a safe way on exists, it takes it and logs one `[HACK]` line per source (see [Details](#details)); `0` stops as before | compat | all | AnyKyty |

### Memory, readbacks, flips and labels

| Variable | Values | Default | Effect | Category | Platform | Origin |
|---|---|---|---|---|---|---|
| `KYTY_DIRECT_MEMORY_PAGES` | `private`, `populate` or `demand`, else stops | `private` where it works, else `populate` | How the host backs direct memory, so a game's first write to fresh direct memory does not fault once per 4 KiB page on its own thread: `private` puts it in a file on a tmpfs mounted with `huge=within_size` in a private user and mount namespace (2 MiB pages, memory resident only where written), `populate` keeps the memfd and pre-reads each new mapping while a quarter of host memory stays available afterwards (resident memory grows by everything mapped), `demand` is the old 4 KiB demand paging. Unset tries `private` and falls back to `populate` where the namespace or the mount is refused, for example by Ubuntu's AppArmor policy or `shmem_enabled=deny`, while `private` set explicitly stops there instead; the log names the backing | perf | Linux | AnyKyty |
| `KYTY_WRITE_BACK_IMAGES` | `0` or `1`, empty counts as unset, else stops | on | Copies the queued readbacks back at each `ACQUIRE_MEM` or `RELEASE_MEM` with GL2 write-back and at each flip; `0` is KytyPS5's behaviour. See [below](#details) | compat | all | `9d957fd81e` |
| `KYTY_URGENT_WRITEBACK_WAIT` | `0` turns it off, any other value or unset leaves it on | on | A guest thread that faults on a page whose texture write-back is still in the open command buffer sends the flush on the urgent queue and waits for that submission itself, instead of queueing the flush behind every operation the command processor has queued; `0` sends it through the ordinary command queue | perf | all | AnyKyty |
| `KYTY_TILE_VIA_VRAM` | `0` or `1`, empty counts as unset, else stops | on | Tiles a tiled image's readback in device-local memory, then copies it once to the host; `0` is KytyPS5's behaviour. See [below](#details) | perf | all | `08c666a175` |
| `KYTY_COMPARE_WRITE_BACK` | `0` or `1`, empty counts as unset, else stops | on | An image readback writes guest memory only where the GPU result differs from the bytes read when the readback was recorded and the guest bytes still hold that read; `0` writes the image's whole footprint, KytyPS5's behaviour. See [below](#details) | compat | all | AnyKyty |
| `KYTY_EARLY_FLIP` | `0` or `1`, empty counts as unset, else stops | on | Completes a guest flip once its command buffer is submitted, on a thread of its own; `0` is KytyPS5's behaviour. See [below](#details) | perf | all | `499dfc3f39` |
| `KYTY_LABELS_AFTER_DUMPS` | `0` or `1`, empty counts as unset, else stops | on | A deferred `RELEASE_MEM` label waits for the newest pending occlusion dump instead of its own command buffer; `0` is KytyPS5's behaviour. See [below](#details) | perf | all | `ef88cc3285` |
| `KYTY_LABELS_AROUND_GPU_WRITES` | `0` or `1`, empty counts as unset, else stops | on | A `RELEASE_MEM` label that arrives while other labels are deferred, on a page the GPU wrote elsewhere but none of whose own bytes it wrote, is deferred behind them instead of draining the GPU; `0` drains. See [below](#details) | perf | all | AnyKyty |

### Pipelines and shader store

This build does not read KytyPS5's `KYTY_PIPELINE_PRECOMPILE`: stopping the game while the manifest's pipelines compile needs the helper processes and a way to stop the guest threads, and this build has neither, so the game never stops for them.

| Variable | Values | Default | Effect | Category | Platform | Origin |
|---|---|---|---|---|---|---|
| `KYTY_ASYNC_PIPELINES` | `0`, `1` or `2`, else stops | `1` | `1`: for a draw that only changes the picture, and is not a small colour target, the pipeline is created on a background thread and the draw is skipped until it is ready, after waiting 20 ms for it. `2`: when every target of the draw was drawn in one of the two frames before, the draw is also skipped if it goes to a small colour target, is counted by an occlusion query (the query then reads visible), has a small pipeline or has no depth and small vertex programs, and a new pipeline is not waited for: it is taken from the driver cache or linked from ready stage libraries at once, else the draw is skipped until the background thread has built it; a draw into a target that was not drawn just before is handled as with `1`. With both, draws whose programs write memory or whose targets are linear, and dispatches, wait. `0` always waits | perf | all | Kyty option |
| `KYTY_PIPELINE_LIBRARIES` | `0` is off | on | Links graphics pipelines from per-stage pipeline libraries (`VK_EXT_graphics_pipeline_library`) | perf | all | `76827a9140` |
| `KYTY_PIPELINE_CHAIN_THREADS` | whole number, clamped to 1 to 16 | `4` | Threads of the pipeline chain above; a value above 16 runs 16 threads and logs one `[kyty-gpl]` line | perf | all | `51c397b379` |
| `KYTY_SHARE_EMPTY_BUFFERS` | `0` is off | on | A shader buffer slot whose V# has `num_records = 0` does not make a shader variant of its own. Its reads return zero and its writes are dropped, as on the console, and that holds for the module built for the slot bound: a raw slot of a compute shader that reads its stride at run time and a scalar slot take the bound form, and a draw or dispatch with such a slot uses a module already built for the other slots when every formatted component it reads gives the same out-of-range value. A module built for such a draw takes the bound form of a slot from a module already built. Manifests recorded before this setting read their empty forms back the same way. `0` builds one variant per combination of empty and bound slots | perf | all | AnyKyty |
| `KYTY_GENERIC_COMPUTE_VARIANT` | `0` is off | on | A dispatch of a compute shader that already has a module, whose descriptor tables of images or samplers take a new shape (other slots with several descriptors, more candidates, a new numeric class in the image heap), runs at once on a module already built for a wider shape: the search tables and heap words of the dispatch are copied into that module's layout, and every key reaches the same descriptor, so the result is the same. A module whose pipeline is ready is preferred. When no module can take the shape, the module built is the union of the shapes already built for the shader, with room to grow, instead of the shape of the dispatch alone. `0` builds and waits for one module per shape. `KYTY_AB` can change it at a flip; its arms differ only for shapes first seen during a phase (see [Details](#details)) | perf | all | AnyKyty |
| `KYTY_E2_VSHARP_MIRROR` | `0` is off, `1` is on | off | Temporary, for the in-run A/B of step E2 of simnJS/AnyPS5#168; removed once the A/B decides. A compute shader reads a buffer that it reads with vector loads, never writes, reads without a format, and whose V# it takes from user data or SRT words through the guest mirror instead of a bound descriptor: the shader reads at the mirror's device address plus the view's guest address plus the offset of each access, and reads nothing when the view does not lie inside the mirror. The CPU still resolves the V# and obtains the same range from the buffer cache as for a bound buffer, from the mirror (never a copy in the stream buffer), so its bytes are in the mirror at the guest address before the dispatch, and passes the mirror's device address and size and the view's guest address, size, stride and whether its out-of-range mode makes every vector read zero in the shader data. Without a mirror, with `KYTY_COMMAND_MEMORY_IN_PLACE=1`, or for a V# whose range does not lie inside the mirror, the buffer keeps the bound path; a mirrored view whose cache buffer is not a window of the mirror stops. Bounds, alignment and out-of-range reads are those of the bound buffer, including the bytes inside the range that a read partly past its end returns under `robustBufferAccess2`. The stride, the zero-stride out-of-range form and the empty-scalar form of such a buffer leave the module key. Buffers read only by scalar loads, written, atomic, formatted, typed or indirect keep the bound path, and so does a buffer whose V# base, at that dispatch, is not a multiple of the widest element the shader reads from it (the module key then names the bound form). `KYTY_AB` can change it at a flip (see [Details](#details)) | perf | all | AnyKyty |
| `KYTY_E2_GPU_FORMED` | `0` is off, `1` is on | off | Temporary, for the in-run A/B of slice 2 of step E2 of simnJS/AnyPS5#168; removed once the A/B decides. Needs the guest mirror. A compute buffer slot whose V# comes from a direct four-word source in guest memory, loaded only to form that V# (no other walked value uses its words), and that the shader reads and writes raw, without atomics, formats, typed access or an indirect table, can be formed in the shader: when the ahead walk of a direct dispatch finds those words written by the GPU and not yet readable, it records their guest address instead of refusing the dispatch, and the shader loads the four words from the mirror at that address and forms the view itself, with the same bounds, stride and out-of-range rules as the bound buffer. Once a slot was formed this way, its program keeps the formed form for that slot, and a dispatch whose words the CPU could read passes them to the same module instead (a flag word in the shader data picks one per dispatch), so a slot that turns clean or dirty does not make a module variant; a V# the formed form does not cover (a base that is not a multiple of 4, or of the element of byte or short accesses, `num_records` of 0xffffffff, swizzle or `add_tid`, a written size that is not a multiple of 4) keeps the bound form. A coherent 64-bit pair at an address that is not a multiple of 8 runs as two dword accesses, as on the bound buffer. For every dispatch that forms a view the elected lane of each wave appends one 16-byte entry per formed view to the dispatch's segment of a view log (base, size as the bound buffer would have, read, written, form miss, slot). Stage 1 of the slice: the CPU still reads the V# words when it records the dispatch and binds and marks the formed view like any written buffer, so the game stays exact without a consumer of the log; a V# that the formed form does not cover then makes the dispatch resolve again on the CPU. `KYTY_AB` can change it at a flip (see [Details](#details)) | perf | all | AnyKyty |
| `KYTY_E2_GPU_FORMED_ALL` | `1` is on | off | With `KYTY_E2_GPU_FORMED`, every walk of a direct dispatch forms every eligible V# in the shader, whether or not its words are GPU-written; for tests | debug | all | AnyKyty |
| `KYTY_E2_VIEWLOG_VERIFY` | `1` is on | off | With `KYTY_E2_GPU_FORMED`, reads each view log segment back once the GPU has finished its dispatch and compares the views the shader logged with the ranges the CPU bound and marked for it: `[kyty-e2f] view log checked` for the first 64 segments, `missing` and `extra` for each difference, and the counts in the `[kyty-e2f]` line every 10 s | debug | all | AnyKyty |
| `KYTY_GENERIC_COMPUTE_SPECIALIZE` | `0` is off | on | With `KYTY_GENERIC_COMPUTE_VARIANT`, a dispatch run on a wider module also queues its own module on the background compile threads; once that module and its pipeline are ready, the dispatch uses it instead. `0` keeps the wider module | perf | all | AnyKyty |
| `KYTY_PIPELINE_STALL_LOG_MS` | milliseconds; text that is not a number reads as `0`, which logs every one | `100` | Logs a `[kyty-gpl]` line for a pipeline stall or shader translation that takes at least this long. If explicitly set, also logs completed `vkCreateComputePipelines` calls at or above the same threshold, with elapsed time, shader hash, module ID, SPIR-V word count, lookup or create mode, and Vulkan result, and `[kyty-frames]` lines: presentation gaps over 250 ms, and every 10 s the counts of gaps over 50 ms, 250 ms and 1 s. A graphics or compute stall line ends with `why=`, the reason it was not skipped | debug | all | `76827a9140` |
| `KYTY_PGO_DUMP_SECONDS` | seconds; `0` or unset is off | off | In an instrumented build (`-DKYTY_PGO_GENERATE=ON`), writes the profile counters every this many seconds, since a converted game does not run its exit handlers | debug | all | `a1b17740ba` |
| `KYTY_PGO_WRITE_SECONDS` | whole number of seconds; `0` writes only at exit | `60` | In an instrumented `kyty_gpu.prx` (`-DANYPS5_KYTY_DLL=ON -DANYPS5_KYTY_DLL_PGO=GEN`), writes the profile counters to `kyty_gpu-<pid>-<seconds>s.profraw` every this many seconds, since a converted game does not run its exit handlers | debug | Windows | AnyKyty |
| `KYTY_PGO_DIR` | directory | the build's `ANYPS5_KYTY_DLL_PGO_DIR` | In an instrumented `kyty_gpu.prx`, the directory the `.profraw` files go to | debug | Windows | AnyKyty |
| `KYTY_PIPELINE_MANIFEST` | `0` or `1`, else stops | `1` | Warms the pipelines listed in the title's manifest (shipped with the assets, or recorded in an earlier session) at start | perf | all | Kyty option |
| `KYTY_RECORD_PIPELINE_MANIFEST` | `0` or `1`, else stops | `0` | Records the pipelines of the session as a manifest even when the manifest is off; the shader store is not used while it records | debug | all | Kyty option |
| `KYTY_STRICT_PIPELINE_MANIFEST` | `0` or `1`, else stops | `0` (the build's `KYTY_STRICT_PIPELINE_MANIFEST_DEFAULT`) | A pipeline of a shipped manifest that fails to build stops the emulator instead of being dropped | debug | all | Kyty option |
| `KYTY_SHADER_CACHE` | `0` or `1`, else stops | `1` (KytyPS5: `0`) | Keeps recompiled shaders in `_PipelineCache/<title id>.shaders` and warms them up at start; a Release build only, and not while `KYTY_GRAPHICS_DEBUG_DUMP` or the manifest recording is on. A second start loads them instead of translating the shaders again: on a 16-core Ryzen with an RTX 5080, one title loaded 900 modules and 314 pipelines in 534 ms where it had translated 610 modules in 2.4 to 3.0 s, with the same frame rate and the same images. The first start of a title gains nothing | perf | all | Kyty option; on by default in AnyKyty |
| `KYTY_SHADER_CACHE_STAMP` | integer in base 0, else ignored | the stamp the build computes from the shader recompiler's sources | Replaces the stamp written in the shader store's header; `0` turns the store off | debug | all | `4749ffc567` |
| `KYTY_SHADER_CACHE_SHARED` | `1` shared, any other value per-thread | shared, except per-thread with NVIDIA's driver | One shared Vulkan pipeline cache for the warm-up threads, or one cache per thread | perf | all | `4749ffc567` |
| `KYTY_SHADER_VARIANT_LOG` | `1` enables it | off | Logs one `[kyty-variant] build` record per successful shader module build, including ahead and manifest compilation. Cache hits and stored-module restores produce no record. The module ID and separate static-state, buffer, image, sampler and linked-code fingerprints identify changing inputs without dumping shader code. `mirrored=` gives the number of buffers the module reads by mirror address (`KYTY_E2_VSHARP_MIRROR`), `formed=` how many of them it can form in the shader (`KYTY_E2_GPU_FORMED`). The record includes the compiled push-data start and the requested cursor as separate values. When the build also produces a plain pixel module, `plain_module` gives its ID and `plain_spirv_words` gives its SPIR-V word count. With `KYTY_GENERIC_COMPUTE_VARIANT`, `[kyty-variant] serve` names the wider module a compute dispatch runs on (once per shape and module), `widen` the shape built instead of the dispatch's own, and `specialize` a module queued for the background compile threads | debug | all | AnyKyty |
| `KYTY_FUZZ_PAD` | any value | unset | Read by the shader recompiler harness (`KytyShim/fuzz/ShaderRecompilerRun.hpp`), which only the libFuzzer targets and `kyty_shim_recompiler_robustness` link, never the emulator: pads its copy of the shader with 16 zero words, so that a decoder read past the last word stays inside the buffer and the fuzzer goes on to the defects behind it | debug | Linux | AnyKyty |
| `KYTY_FUZZ_SPIRV` | `abort`, `off`, any other text | unset, which does not validate | Same harness: runs `spirv-val` (Vulkan 1.3) on the SPIR-V the recompiler produced; `abort` stops on a module it refuses, any other text except `off` logs one `SPIRV-INVALID` line per distinct message | debug | Linux | AnyKyty |
| `KYTY_FUZZ_SPIRV_DIR` | directory | unset | Same harness, with `KYTY_FUZZ_SPIRV` set to log: writes the input of each distinct refused module there as `invalid-<n>.bin` | debug | Linux | AnyKyty |

### Ray tracing

| Variable | Values | Default | Effect | Category | Platform | Origin |
|---|---|---|---|---|---|---|
| `KYTY_NATIVE_RT` | `0` or `1`, else stops | unset | Unset: native ray tracing when the device has `VK_KHR_acceleration_structure`, `VK_KHR_ray_query` and `VK_KHR_deferred_host_operations` with their features, emulated otherwise. `1`: native, and a stop when the device cannot. `0`: emulated. KytyPS5's own default is off | perf | all | Kyty option; on by default `8a29429250` |
| `KYTY_RT_STATS` | `0` or `1`, else stops | `0` | With native ray tracing, shaders count their queries and an `[RT]` summary of the last 10 seconds is logged | debug | all | Kyty option |
| `KYTY_RT_ORACLE` | `0` to `15`, else stops | `1` | With native ray tracing, bit 1: the BVH intersections of compute walks no form recognizes answer from a host ray query while the guest walk code runs as written; bit 2: the walks a form recognizes are served the same way instead of their rewrite; bit 4: no form is tried; bit 8, with bit 1: the walks of pixel shaders are served too, their draws then build the host TLAS before their rendering pass. Bit 8 is off by default: `KYTY_RT_CHECK` replays dispatches, not draws | perf | all | AnyKyty |
| `KYTY_RT_CHECK` | dispatch period, `0` turns it off | `0` | With native ray tracing, one dispatch in N of every program served natively (`ShaderModule::guest_traversal` false, `bvh` and `ray_query`) is played twice: native variant, outputs saved and previous contents restored, then emulated variant, whose outputs stay; a compute pass counts the output dwords that differ and one `[RT] check` line per check is logged; one in 16 N once a program passed 4 checks. Programs that write through device addresses, GDS, self-sized buffers, indirect write tables or image tables are not replayed (one line says why). The emulated variant is translated by the background compile workers: a check waits until it is ready. Decisions go to `_PipelineCache/<title>.rt` and are reused while the check is on and the recompiler stamp, host stamp, effective oracle bits and PPM tolerance all match. A fallback holds for its session only | debug | compute | AnyKyty |
| `KYTY_RT_CHECK_PPM` | 0 to 1000000 | `1000` | Fraction of differing output dwords, in parts per million, above which a check fails | debug | compute | AnyKyty |
| `KYTY_RT_CHECK_STRIKES` | 1 to 1000 | `3` | Failed checks in a row after which the program uses its emulated variant for the rest of the session (one `[RT] check … falls back` line) | debug | compute | AnyKyty |
| `KYTY_RT_ARENA_MB` | MiB, empty counts as unset | a quarter of the largest device-local heap, 512 MiB to 4 GiB | Limit of the memory the native BLASes take; `0` means no limit | perf | all | Kyty import |
| `KYTY_RT_RESULTS` | `0` or `1`, else stops | `0` | With `KYTY_RT_ORACLE` on, bit 32 of the oracle setting: in the default mode too, a walk whose results the compiler derives by playing the walk on synthetic BVHs (`ir/passes/RayQueryResults.cpp`) is traced by one host ray query and its results are substituted after the loop, without running the guest walk code. `KYTY_RT_ORACLE=3` and `5` already do it | perf | compute | AnyKyty |

### Logs, dumps, validation and profiling

| Variable | Values | Default | Effect | Category | Platform | Origin |
|---|---|---|---|---|---|---|
| `KYTY_PRINTF_DIRECTION` | `Silent`, `Console`, `File`, any case, else stops | `Silent` | Where Kyty's log (`LOGF`) goes: nowhere, stderr beside AnyPS5's own lines, or `KYTY_PRINTF_OUTPUT_FILE` | debug | all | Kyty option |
| `KYTY_PRINTF_OUTPUT_FILE` | path | `_kyty.txt` | File of the `File` direction | debug | all | Kyty option |
| `KYTY_SESSION_LOG` | path | empty | Session log file, opened at start with a header | debug | all | Kyty option |
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
| `KYTY_AB` | `<seconds>:<setting>[+<setting>...]`, else stops | unset | In-run A/B: alternates phase A (the listed settings as the environment sets them) and phase B (the listed settings turned over) at flips, each phase at least `<seconds>` long (1 or more). The execution thread changes them right after a flip's submission, between two submissions. It prints `[kyty-ab]` lines: both arms at the start, each phase's begin and its settled point, and at the end of a phase its settled part: presents, img/s, longest frame, frames of 50 ms or more, with the wall clock. A setting that cannot change while the title runs and a name listed twice stop at start with a message that lists the settings it can change: `KYTY_GENERIC_COMPUTE_VARIANT`, `KYTY_E2_VSHARP_MIRROR`, `KYTY_E2_GPU_FORMED` (see [Details](#details)). `python3 tools/kyty_ab_report.py <session>` gives each arm | debug | all | AnyKyty |
| `KYTY_AB_SETTLE` | seconds, shorter than the phase, else stops | `2` | Seconds into a phase after which `KYTY_AB` counts it: its settled line, then its presents and img/s until the next phase | debug | all | AnyKyty |
| `KYTY_AB_START` | flip count from 1, or `<n>s`, else stops | `0` | Flip at which `KYTY_AB` starts, or `<n>s` for `<n>` seconds after the first flip | debug | all | AnyKyty |

### AnyPS5 settings that an AnyKyty session also reads

The window, the flip and vblank pacing, the guest memory and the thread priority of the presenter are AnyPS5's, so their `ANYPS5_*` and `APS5_*` settings apply with the Kyty backend. Each has its row in [ENVIRONMENT](ENVIRONMENT.md); this table says what it does here. Settings read only by AnyPS5's own GPU driver (`core/libs/prx/libSceAgcDriver/Execution/`, which this build does not compile) have no effect: `ANYPS5_PRESENT_MODE` is replaced by `KYTY_PRESENT_MODE`. `ANYPS5_GPU` is read by this build too.

| Variable | Values | Default | Effect with the Kyty backend | Category | Platform | Origin |
|---|---|---|---|---|---|---|
| `ANYPS5_VBLANK_HZ` | whole number, 30 to 360, else stops | unset | Vblank rate; unset or empty takes `KYTY_VBLANK_FREQUENCY`, and the value also becomes Kyty's vblank frequency, so `ANYPS5_VBLANK_HZ=240` lifts the 60 FPS cap of titles that pace on vblank | option | all | `bffccd0fbf` |
| `APS5_UNPACED_FLIPS` | `0`, `1` or empty, else stops | `0` | Benchmark aid: each flip completes once presented, not at the next vblank, so a title that renders faster than the vblank presents as fast as it can; set `KYTY_PRESENT_MODE=Immediate` or `Mailbox` too to present above the display's rate | debug | all | `b3691289ef` |
| `ANYPS5_GPU` | part of a Vulkan device name, any case; empty counts as unset | unset | Runs on the first device whose name contains the text and that the renderer accepts; it wins over the headset's device. Unset, the renderer picks a discrete GPU when it has one. A text that matches no device stops the game with a message that lists the devices; a match the renderer rejects stops it with the reasons. It replaces `KYTY_GPU`, an index that is no longer read | option | all | AnyPS5 |
| `ANYPS5_KYTY_PRESTART` | `0` or `1`, empty counts as unset, else stops | `1` | Starts the Kyty GPU on a host thread as soon as the video output is created; `0` starts it after the window exists | perf | all | `e465445589` |
| `ANYPS5_KYTY_SPARSE_READS` | `0` or `1`, empty counts as unset, else stops | `1` | The Kyty GPU asks the memory files where the data of a 64 KiB or larger range is (`SEEK_DATA`) and writes zeros for the never-written parts without touching them; `0` reads every byte | perf | Linux | `21c1eae177` |
| `APS5_NO_DIRECT_GUEST_COPY` | presence | unset | On Linux x86-64 the checked guest copies read with plain loads whose faults the signal handlers turn into a failed copy; set, every checked read is a `process_vm_readv` system call again | debug | Linux | `004c1c4ede` |
| `ANYPS5_FULLSCREEN` | `0`, `1`, `windowed`, `borderless` | `windowed` | Opens the display window fullscreen at the desktop resolution; the window AnyKyty presents into is AnyPS5's, so this replaces `KYTY_FULLSCREEN` | option | all | AnyPS5 |
| `ANYPS5_DISPLAY_INDEX` | index of a connected display, else stops | `0` | Display that the window opens on | option | all | AnyPS5 |
| `ANYPS5_FPS_LIMIT` | 0.5 to 1000000 with at most three decimals, else stops | `0` (no limit) | Most frames per second the flips present | option | all | AnyPS5 |
| `APS5_OUTPUT_RESOLUTION` | `<width>x<height>`, else the output status calls throw | unset: follows the display | Output resolution `sceVideoOutGetOutputStatus` reports to the title, 1080p class under 3840 by 2160 and 2160p class otherwise; it does not change the render size | option | all | AnyPS5 |
| `ANYPS5_VIDEO_OUT_MAX_WIDTH`, `ANYPS5_VIDEO_OUT_MAX_HEIGHT` | positive integers, else output-mode queries throw | unset | Host display hint: set below 3840 (width) or 2160 (height), an output mode that AnyPS5 does not know is reported as unsupported, so that the title falls back to the host resolution | option | all | AnyPS5 |
| `ANYPS5_UPSCALE` | `fsr`, `cmaa2`, `rcas`, unset or empty, else stops | unset | Scales the presentation with FSR 1 (EASU and RCAS) when the window is larger than the display buffer, applies Intel CMAA2 morphological anti-aliasing under `cmaa2` at 1:1, or 1:1 native RCAS under `rcas`; unset or empty keeps the bilinear blit | option | all | AnyPS5 |
| `ANYPS5_FSR_SHARPNESS` | number, 0.0 to 2.0, else stops | `0.2` | RCAS sharpness in stops under `ANYPS5_UPSCALE=fsr` or `rcas`, from 0.0 (sharpest) to 2.0 (weakest) | option | all | AnyPS5 |
| `APS5_SYNC_FLIP` | presence | unset | Debug aid: the flip worker waits until the presenter is done with each flip | debug | all | AnyPS5 |
| `APS5_THREAD_PRIORITY` | `off`, `high`, `rt` | `off` | Raises the presenter thread above the title's threads; Kyty's own stage threads are always raised, which only happens on Windows | perf | all | `a9c132a629` |
| `APS5_THREAD_NICE`, `APS5_THREAD_RT`, `APS5_THREAD_RTTIME_US` | see [ENVIRONMENT](ENVIRONMENT.md) | see there | Tune that raised priority on Linux | perf | Linux | `a9c132a629` |

### Details

The settings below change how a part of the driver works, so each has its reasoning and measurements here; the tables above say what to set.

KytyPS5's log (`LOGF`) is silent by default; `KYTY_PRINTF_DIRECTION=Console` sends it to stderr, beside AnyPS5's own lines.

GPU stage threads: KytyPS5 places them only on Windows, on logical CPUs `{4, 6, 8, 10, 2}` whatever the CPU (it counts
SMT siblings as adjacent logical processors, and its core reservation uses Windows CPU sets). Under `ANYPS5_KYTY`, with
`KYTY_GPU_THREAD_CPUS` unset or empty, each stage thread (command processor, execution, recording, resolve, host copy)
stays within the logical CPUs that share the largest last-level cache, intersected with its current affinity. On Linux
(`/sys/devices/system/cpu`; on a Ryzen 9 7950X3D, CPUs 0-7 and 16-23, the 3D V-Cache CCD) it keeps to that whole set. On
Windows (`GetLogicalProcessorInformationEx`, `RelationCache` and `RelationProcessorCore`, within the thread's processor
group) each stage is pinned to one logical CPU of its own core in that set: performance cores first (highest
`EfficiencyClass`), the core holding CPU 0 last, never two stages on SMT siblings of one core. With fewer such cores than
stages, the threads keep to the whole set instead. No core is reserved, and the threads a stage creates afterwards
inherit its CPUs. Measured on Subnautica (Linux): +7 % to +22 % frames per second, 25-35 % less CPU time per draw on each
stage (`notes/au-dela-p4-placement.md`). On an i9-11900H (one L3), leaving the stages free over all CPUs cost 2.4 % in
Ghost of Yōtei against pinning them to separate cores, hence the per-core pinning on Windows. `KYTY_GPU_THREAD_CPUS=off` turns it off. An explicit list pins each stage to that
logical CPU: on Linux without core reservation, so threads a stage creates afterwards inherit its CPU; on Windows with
KytyPS5's core reservation, and a listed CPU the machine does not have stops with a message naming the variable.

Images the game reads after the GPU writes them: KytyPS5 copies a GPU-written linear image back to guest memory at the
next suspend point. A game that waits for the label of a `RELEASE_MEM` with GL2
write-back and then reads the image reads stale memory: in TCG Card Shop Simulator (PPSA36301) a 64×64 RGBA32F image is
read that way and the street stays almost black. Under `ANYPS5_KYTY`, the write-back half of that setting is on by
default: at each `ACQUIRE_MEM` or `RELEASE_MEM` with GL2 write-back and at each flip, the images already queued for
readback are copied back, and their pages stay read-protected until the copy lands. No other image is copied: the copy
of every storage image on its first write is not done. `KYTY_WRITE_BACK_IMAGES=0`
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

Image readbacks write only the bytes the GPU produced: `TextureCache::DownloadImageMemory` reads the image's whole guest
footprint when it records the readback, tiles the image into that copy, and KytyPS5 then writes the whole footprint back
when the GPU is done. Every byte the game, or the write-back of another image, changed in between is put back to the old
read: block padding, an unused tail block, mips and slices the GPU never wrote, memory reused for other data. The console
writes only the texels a command produces (`notes/re-overwrite-audit.md`). Under `ANYPS5_KYTY` the readback keeps the bytes
it read, and when the GPU is done it rereads guest memory and writes, in 64-byte steps, only the bytes where the GPU result
differs from the first read and guest memory still holds that read; it skips the rest and reports
`image write-back kept bytes the game wrote after the read` when it left a byte the game changed. Cost: one copy of the
footprint when the readback is recorded and one read and compare when it lands, with reused host buffers. Both
`KYTY_TILE_VIA_VRAM` paths go through it (`notes/tile-via-vram.md`, issues #30 and #108). `KYTY_COMPARE_WRITE_BACK=0`
restores KytyPS5's behaviour; any value other than empty, `0` or `1` stops the emulator with a message naming the variable.

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

Idle submit and the walk's page reads: on an open-world AAA title at 4K with native ray tracing, the GPU is idle about
5 ms of a 19 ms frame, in some 40 gaps between command buffers. Most gaps open while the execution thread builds the next
command buffer: early in the frame it builds compute work more slowly than the GPU runs it, and about nine times a frame it
waited for the GPU to finish all its work. Those waits came from the shader resource table walk: it read descriptor tables
that share a 16 KiB page with data shaders write through device addresses, the write bitmap marks the whole page, the walk's
load faulted, and the fault handler copied the page back after the GPU finished everything. A probe that compared the bytes
before and after each such copy found them unchanged in every read in game; reading around page marks without the copy
removed those waits, but left the pages protected, so every later load of the walk and of the resolve thread faulted, and
the frame rate dropped by 5 %: that path was not kept. Two smaller changes were. The shim installs the walk's
guest-memory provider (`SetSrtWalkerGuestMemory`), which the import carried but nothing installed: on the execution thread a
load from a page that may fault goes through the same around-GPU-writes read as the fault handler, without the signal, and a
load it cannot serve still faults. The execution thread submits the open command buffer after a draw or a dispatch when the GPU
has finished everything else, so it starts on the new work while the execution thread goes on; while the GPU still runs, it
reads the semaphore value the scheduler already knows and asks the driver at most once every 25 µs. Measured with the
setting switched every 10 s inside one session (14 periods in game, neighbouring periods compared), on the open-world title:
+1.6 % frames per second for the walk's provider, +4.1 % for the idle submit (from 47 to 60 submissions a frame); on Stray,
whose execution thread is the wall, both within ±1 % (the idle submit takes it from 114 to 181 submissions a frame). Same
images. Neither change has a switch since these measurements (the idle submit was `KYTY_IDLE_SUBMIT`, `820414454e`).

Guest write faults: a guest thread's write fault takes the lock of its 4 MiB tracking region once, to note the fault, open
the block or mark the page written by the CPU, instead of once to note it and once more to mark the page; a fault on a page
the tracking already sees written by the CPU and not by the GPU (and, in a marked block, on a block already open, or the
rest of the block the locked path would mark) takes no lock at all: another watcher, or another thread that opened the page
first, made it fault; it still advances the CPU write epochs of the region and marks the region for device address reads.
Pages the GPU wrote keep the readback path. `[kyty-write-blocks]` counts the faults that found their page already open and
the faults in command memory that an image or table also watched; `[kyty-locks]` sums up every 10 s the contended
acquisitions of the tracking regions, the page manager regions, the texture cache and the other `TrackingSpinLock`s and the
time spent waiting for them. Its switch, `KYTY_WRITE_FAULT_FAST_PATH`, was removed after an in-run A/B on an open-world
title: equal frame rate (139.15 against 139.27 img/s), contended acquisitions 307 against 551 every 10 s.

Write fault blocks: the command processor marks the 64 KiB blocks of every submission and indirect buffer it reads, and
`libSceAgc` marks the whole segment of a command buffer it writes packets into (packets fill it upwards, embedded data
downwards from its end), in lock-free tables the fault handler reads, cleared when the memory is unmapped or freed. A guest
thread's write fault on a page the GPU did not write, in a marked block, marks every page of the block that the GPU did not
write as written by the CPU, with one protection change, instead of the faulted page alone: a command buffer the guest fills
page by page, in either direction, faults once until its next upload instead of once per page. Faults outside marked blocks
keep the page path. `[kyty-write-blocks]` sums up every 10 s the faults, the blocks opened, the clean pages marked with them,
the buffer upload bytes, the faults by owner (first match: a command buffer the command processor read, elsewhere in a
command buffer segment, a block with a page the GPU wrote, an image or texture, a buffer of the cache, other), the bytes of
opened pages handed to uploads, the uploads of buffers the GPU writes, the copies made again because the guest wrote a page
during the copy, and the uploads that kept the tracking region locks through the copy. Its switch, `KYTY_WRITE_FAULT_BLOCKS`,
was set to `0` by testers while the blocks also covered memory other than command buffers, which cost the game's threads;
`da2947c57d` keeps the blocks to command memory, and the switch was removed since a fix keeps no switch.

Image tables in the image heap: a sampled-float bindless texture table (an array of 32-byte image descriptors in memory)
is served through the image heap, which mirrors the table memory, so the shader and pipeline layout no longer depend on how
many textures the table holds; a table the heap cannot serve is bound as a per-draw descriptor array whose capacity is baked
into the shader, which rebuilds the shader when the table grows past a power of two. Kyty's option `KYTY_IMAGE_TABLE_HEAP=0`
bound every table that way; testers set it while the heap stopped a title on a texture without a tiled block layout, which `8de3821d41`
fixed, and the option was removed since a fix keeps no switch.

Stencil fills: a buffer fill of a stencil plane becomes a clear of the depth image's stencil aspect only while that aspect
holds the plane; a fill of another plane runs as the game's own fill, so it no longer wipes the plane the depth image holds.
This fixed the water of a title (#103); it had a switch, `KYTY_FIX_STENCIL_FILL_HELD_PLANE`, removed since a fix keeps no
switch.

Flattened SRT words left to the execution thread: an SRT walk done ahead of execution on the resolve thread does not stop at
a load from a page holding GPU-written bytes when the walker uses the loaded dword only as a flattened SRT word. The walk
records the address and the slot, and the execution thread fills the word when it takes the resolution, as it already does
for every flattened SRT word read ahead: a draw reads it on the CPU, a dispatch copies GPU-written bytes on the GPU in
command order unless the renderer also reads that slot on the CPU (address bases, device-address read spans, any ray query
program), which it then reads on the CPU. Native XPR relocation does not take a resolution with such words. A
`[kyty-flat-defer]` line counts these resolutions every 10 s. Measured by an in-run A/B on a title at 12 img/s: +2.0 %
(+0.17 img/s with drift cancelled, standard error 0.20, same scene); a check mode then compared 714,094 deferred
resolutions of that title with full ones without a mismatch, and Stray deferred none. The setting and its check mode were
removed after these measurements. The idea of GPU-read SRT words is Erlou's (Kyty `740ce47c`).

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

A label that cannot be deferred is written at once, and while another label is deferred, the execute thread first waits
for every command buffer and priority operation before it (a full drain), so that labels stay in order. KytyPS5 only
defers a label whose page the cache does not hold as written by the GPU: a label sharing its page with a few bytes a shader
writes drains the GPU each time another label is pending. Under `ANYPS5_KYTY`, with `KYTY_LABELS_AROUND_GPU_WRITES` at 1,
the default, such a label is then deferred behind the pending ones when none of its own bytes holds data the GPU wrote and
the backing view of guest memory holds it: its completion writes it there without a fault, as the immediate write does
after the drain; with no label pending, it is still written at once. A title that
keeps occlusion queries or labels pending and places a label next to a GPU-written counter (a barrier table, for example)
drained the GPU about once a frame, 5.7 ms each, 2.78 s every 10 s on a heavy city scene; the line `[kyty-labels]`, every
10 s on stderr, counts the labels deferred this way and the drains that remain. `KYTY_LABELS_AROUND_GPU_WRITES=0` restores
KytyPS5's behaviour; another value stops the emulator with a message naming the variable.

Why a label drains: when `[kyty-labels]` is printed, the line `[kyty-label-drains]` follows it with the labels of the
period and their distinct addresses, how many were deferred to the end of a tick, written at once and written after a
drain, and the first reason, in this order, that kept each drained label from being deferred: not a deferable kind (the
GDS read, an invalid range); its own bytes lie under a GPU write to memory served in place that the GPU has not finished
(the written ranges are kept byte for byte next to the 64 KiB blocks); its own bytes hold data the GPU wrote in the
cache; `KYTY_LABELS_AROUND_GPU_WRITES=0`; the backing view does not hold it; only its 64 KiB block is under an in-place
GPU write in flight; its page is GPU-written in the cache and no label is pending. Then: how many drained labels lie
under an in-place write in flight and how many only share its block, the drain waits in eight classes of duration (count
and milliseconds), and the GPU writes to memory served in place that were noted over a completion write still pending.
The line `[kyty-label-covers]` then splits the drained labels that lie under an in-place GPU write in flight by what
noted that write: a `WRITE_DATA` done as a GPU copy, a `DMA_DATA` fill or copy, a buffer bound for writing (its whole
bound range), a texel buffer filled from an image; and says whether the newest such write was noted in the label's own
tick and whether one covers exactly the label's bytes.

Labels deferred to their own tick in place: while memory is served in place (`KYTY_COMMAND_MEMORY_IN_PLACE=1`), two of these
reasons no longer drain. A label whose bytes no in-place GPU write in flight covers, on a page the cache does not hold
as GPU-written, is deferred like any label on a clean page; one on a GPU-written page of the cache whose own bytes are
clean and that the backing view holds is deferred around the GPU writes as with `KYTY_LABELS_AROUND_GPU_WRITES`, also when
no label is pending. Both go on the first-in first-out priority queue at their own tick, so they are written after the GPU
finished everything recorded before them, after every older label, and before a later label that drains. A label whose
own bytes are under in-place writes in flight is deferred the same way when all of them are `WRITE_DATA` copies or
`DMA_DATA` fills and copies: those were recorded before the label, so the GPU has done them when the label's tick ends,
and a later one to the same bytes drains first as below. A label under a buffer bound for writing or a texel buffer
filled from an image, or whose bytes the cache holds as GPU-written, still drains. Until such a label is
written, a `WRITE_DATA`, a reference clock copy or a `DMA_DATA` to its bytes, and a read of them the command processor
decides on (`WAIT_REG_MEM`, predication, a conditional execution), first drains the GPU as the label did before;
a fill or copy the renderer would do on the CPU goes to the GPU; and a GPU write to memory served in place noted over its
bytes stops the emulator with a message, because the label would land after a write that comes later in the stream.

Blocked waits on deferred writes: KytyPS5's command processor submits the open command buffer after a pass over a
submission that made progress, but not after one that yields to another queue, nor when a queue resumes straight into a
wait that still fails. If the value that queue waits for is a deferred write recorded into the open command buffer, and
no other submission comes, nothing submits that buffer and the title hangs. Under `ANYPS5_KYTY`, a queue that blocks
without progress on a range with a pending deferred write submits the open command buffer, and submits it again only when
a newer deferred write lands on that range while it still waits, whatever `KYTY_LABELS_AFTER_DUMPS` says; deferring the
labels of every queue makes the case more likely in titles without occlusion queries.

`m_program_lock`, the spin lock of `graphics/host_gpu/renderer/pipeline/pipelineCache.cpp` that guards the program
cache, has a narrower scope than upstream's in this build (`programLock.cpp`). The resolve thread always runs upstream's
unlocked ahead mode, which upstream only uses with its resolve helper thread, and the execute thread releases the lock
while it materializes the resources of a cached program, which can wait for the GPU when it reads GPU-written buffers.
A waiter spins briefly, then yields. `KYTY_PROGRAM_LOCK`, `KYTY_LOCK_STATS` and `KYTY_LOCK_STATS_PERIOD` are read once.

KytyPS5's `native_rt` is off by default and its launcher's recommended settings turn it on; under `ANYPS5_KYTY` the
shim turns it on unless `KYTY_NATIVE_RT=0`, and Kyty enables the ray query extensions only on a device that has them.
Once the device exists, `KytyShim::ReportNativeRayTracing()` prints one `[kyty] native ray tracing` line on stderr:
on, off by `KYTY_NATIVE_RT=0`, or unavailable with the missing extension or feature named, in which case ray tracing
runs emulated as with `0`; with `KYTY_NATIVE_RT=1` that last case stops the emulator instead. `KYTY_RT_STATS` takes
effect whenever native ray tracing is on, and so does `KYTY_RT_ORACLE`, which adds a `[kyty] BVH intersections of … answer
from host ray queries` line and enters the key of every compiled compute program. `KYTY_RT_ARENA_MB` (MiB) replaces the limit of the memory the native BLASes
take, a quarter of the largest device-local heap clamped to 512 MiB to 4 GiB; it is read by
`graphics/host_gpu/renderer/rayTracingSlots.cpp`.

With native ray tracing on, a BVH intersection that neither a form nor a host ray query serves runs empty: its box
nodes have no child, so the walk ends and its rays miss, while its triangle test stays exact. The program logs one
`[RT] cs 0x…: N of M BVH intersections run empty` line per session, its `[RT]` summaries count its dispatches, and with
`KYTY_RT_CHECK` it gets an `empty` row in `<title>.rt`. A pixel shader's BVH walk without bit 8 of `KYTY_RT_ORACLE`
stays emulated (`[RT] ps 0x…: pixel walk kept on the guest traversal`), and so do the lanes a form site leaves to the
guest walk for a leaf test the host does not run (`[RT] cs 0x…: site N leaves … lanes to the guest walk`). The emulated walk serves every BVH walk only
with `KYTY_NATIVE_RT=0` or on a GPU without ray queries, and the reference variant of `KYTY_RT_CHECK`. A host TLAS
walk that finds BLASes the host has not built yet, TLAS memory it has not cached yet or a full host structure still
publishes its context: the rays of that dispatch miss what it left out, and the `[RT]` summary counts these walks and
the instances they left out. `kyty_rt_probe` (a target of every `kyty` build, written to `build/tests/`; build it with `cmake --build build --target kyty_rt_probe`) recompiles such a shader offline, from its raw machine code, as the driver does with native ray tracing on:
`kyty_rt_probe [--wave 32|64] [--group X Y Z] [--scratch dwords] [--lds dwords] [--dump] [--spirv out.spv] <shader.bin>...`
prints, per file, its BVH intersections, how many of them trace with ray queries, and the TLAS source and instance
record layout derived for them, and exits with 0 only when every intersection does. `--dump` adds the IR, `--off`
compiles as with `KYTY_NATIVE_RT=0`, `--oracle 0-15` as with `KYTY_RT_ORACLE`, `--stats` as with `KYTY_RT_STATS=1`, and
`--subgroup 32` translates a wave of 64 for a GPU with subgroups of 32, as the driver does there, `--robust` emits as for a device
with `robustBufferAccess2` and a storage access alignment of 1 (no explicit bounds checks on storage buffers, as the driver emits
on such a device), `--stage pixel` translates a
pixel shader (`--inputs N` interpolated inputs, 8 by default) instead of a compute one; the line then ends with
`oracle=N lane=M empty=E`, the intersections that answer from a host ray query, those of them that do it lane by lane,
and those that run empty.

To probe directly from the game's local pipeline manifest code cache, use
`kyty_rt_probe --cache _PipelineCache/<title>_<version>.code --hash 0x<shader-hash>`.
Use the compute shader hash from the log. The probe verifies the selected record and reads the cache without
writing to it. Supply `--wave`, `--group` and the other shader settings as needed. The cache stores the program bytes.

To list every ray tracing program of a game at once, use
`kyty_rt_probe --cache _PipelineCache/<title>_<version>.code --all`. The probe indexes the code cache, reads the manifest beside it
(`<title>_<version>.kpm`, and `<title>_<version>.shared.kpm` when it exists; `--kpm <file>` names another one) and translates, as
the driver does at start, each module whose code has `image_bvh_intersect_ray` instructions, with the stage, wave size, workgroup
size, LDS and scratch the record holds. A program the game registered but no manifest record names is translated as a compute shader
with a default 8x8x1 workgroup, 8192 LDS dwords and 16384 scratch dwords, at wave sizes 64 and 32 (`source=code`); a graphics shader
with BVH instructions and no record is listed as `no_record`. Nothing is written, and stderr stays empty. The options that depend on
the GPU are `--subgroup 32|64` (the host subgroup size; 32 by default), `--lds-limit <bytes>` (49152 by default) and `--oracle`
(the value of `KYTY_RT_ORACLE`; 1, the default of the build, here), `--no-validate` skips the SPIR-V validation, and `--version`
prints the commit of the build. The output has one line per program, `key=value` pairs after the hash:

```text
rt-survey 1 build=<commit> oracle=1 subgroup=32 lds_limit=49152
note <text>
program <hash> [back=<hash>] stage=cs wave=64 group=8,8,1 lds=0 scratch=0 modules=3 source=manifest status=ok bvh=12 native=0 oracle=12 lane=0 empty=0 emulated=0 spirv=valid
summary records=N manifest_modules=N manifest_skipped=N code_only=N programs=N intersections=N missing_code=N unreadable=N undecodable=N truncated=0 seconds=S
```

`bvh` is the number of BVH intersections left in the translated program, and `native`, `oracle`, `lane`, `empty` and `emulated`
split it: served by a recognized walk form, answered by a host ray query for the whole wave, answered by one lane at a time,
run empty (the rays miss), and left to the guest walk. `status` is `ok`, `calls` or `callable` (the program calls functions whose
code is only in the game's memory), `spec_mismatch` (no record of the program fits its resources), `stop` (the translator
stopped; `phase` and a `stop` text, scrubbed of hashes and paths, say where) or `no_record`. `spirv` is `valid`, `invalid`
(Vulkan 1.3 validation with the driver's options) or `none`. `missing_code` counts the manifest modules whose code the cache does not
hold, `undecodable` the programs the decoder stopped on, and `truncated` is 1 when the cache ends in a half-written record. The exit
code is 0 when the survey ran, 2 when the cache cannot be read and 64 for a wrong command line.
[`tools/rt_survey.py`](../../tools/rt_survey.py) runs the probe and writes the report that players attach to the survey issue; see
[the ray tracing survey](../user/ANYKYTY.md#ray-tracing-survey).

Fallbacks instead of stops (`KYTY_FALLBACKS`, on by default on Linux; off on Windows, where the C++ exception that carries a fallback is not caught across the guest libc yet): KytyPS5 stops the emulator (`EXIT`, `EXIT_IF`,
`EXIT_NOT_IMPLEMENTED`) on anything it does not handle. Where the stop concerns one draw or one dispatch and nothing else
depends on it, the build skips that draw or dispatch instead and writes one `[HACK]` line the first time per shader,
naming the stop and its source line: a shader the recompiler cannot decode, translate or emit, and a shader stage whose
inputs cannot be prepared. A failed shader is remembered (the whole program when no variant of it ever compiled,
otherwise that variant), so it is not translated again for each draw; the other variants of the program keep drawing.
A background compile that stops is dropped, and the draw compiles it again. Work done ahead of execution is never
required, so its stops are caught whatever `KYTY_FALLBACKS` says, wherever a stop can be caught (Linux): a compile
queued by the resolution ahead or by a pipeline chain, and the preparation of a shader stage ahead. A compile ahead that
stops is kept for its variant (its resources and push data) and never tried again ahead; the execution thread
compiles what its draw needs as before, so a stop it hits itself still follows `KYTY_FALLBACKS`. A variant whose buffer
descriptors no program of its stage compiles (`ADD_TID` outside compute, a reserved `dst_sel`) is refused before
compiling and kept the same way, and a program with 64 failed variants is no longer compiled ahead. Where a stop cannot
be caught (Windows), a variant of a resolution ahead that left a GPU-produced load unread is not compiled ahead.
`[kyty-spec-compile]` counts these compiles every 10 s while they change. A pipeline manifest record that stops while
it is replayed is dropped, as with exceptions enabled. The command processor ignores a write to a register it does not
know (direct `SET_*_REG` packets and indirect register lists) and a GPU event type it does not know, with one `[HACK]`
line per register or event; the packet lengths come from the headers, so the rest of the stream is read as before. The
unknown registers are entries of the register tables, so the loops that apply known registers are unchanged. A fallback
only replaces a stop: nothing changes for a shader that compiles or a register Kyty knows. `KYTY_FALLBACKS=0` restores
every stop, except a write of 0 to context register `0x24a`, which Kyty already ignored in indirect lists and now also
ignores in a direct `SET_CONTEXT_REG`.

`CONTEXT_CONTROL` (PM4 opcode `0x28`) with register loading and shadowing off is accepted as the no-op it is on the
console; with either on, it still stops with the two control words.

Blending into 11_11_10 UNorm color targets: KytyPS5 writes a format 30 target through its R32_UINT backing, the pixel
shader packing the three channels, so the host cannot blend it and pipeline creation stops on any enabled blend. Under
`ANYPS5_KYTY` a replace blend (ONE, ZERO, ADD) is a plain export. Any other blend, and a write mask that leaves a
channel out, is done by the pixel shader: the draw binds the target as an R32_UINT storage image instead of an
attachment, and after the shader's own code, inside a fragment shader interlock critical section
(`PixelInterlockOrderedEXT`, so overlapping fragments of one draw merge in primitive order), the shader reads the stored
word and decodes each field as k / (2^n - 1). It clamps the source color and alpha to [0, 1] when
`CB_COLOR_INFO.BLEND_CLAMP` is set (NaN to 0), evaluates the color equation in fp32 without contraction, with a
destination alpha of 1 and MIN and MAX ignoring the factors as in Vulkan, clamps the result to [0, 1] and encodes it
with the target's `ROUND_MODE` (truncate, or round half up) as the plain export does, keeps the fields the write mask
leaves out, and stores the word. Such a shader runs with early fragment tests, and each such draw gets a render pass
instance of its own. Constant and second-source factors, multisampled, layered and volume targets, devices without
`fragmentShaderPixelInterlock`, pixel shaders that export depth or coverage or use alpha to mask, a shader that kills
pixels under late tests of a draw that writes depth or stencil, a shader that writes memory under late tests, and a
target that the draw also reads as a shader resource stop the emulator with the reason. The console's blend precision
and rounding for this format are not measured; `kyty_shim_packed_unorm_blend` runs the emitted blend arithmetic on each
Vulkan device of the host and compares every result bit for bit with a CPU model of the same arithmetic, out-of-range
and NaN sources included.

In-run A/B (`KYTY_AB`): one session gives both arms on the same scene, where two sessions, one per arm, never play the same
scene. The phases follow `APS5_AB` (same syntax, same `_SETTLE` and `_START`); phase B turns over every listed setting. A
setting can be listed only when it is read at its point of use from an atomic flag (a relaxed load, as cheap as the
static value it replaces) and when its two values may follow each other between two submissions while the other threads
keep running: each use reads it once, and work done under one value stays right under the other. The settings it can
change:

- `KYTY_GENERIC_COMPUTE_VARIANT`: read where a dispatch picks its module and where a build widens a shape; a dispatch that
  sees it change between two of its reads takes the path of the off value. Its effect belongs to the first time a shape
  is seen: a shape served on a wider module while it is on gets its own module compiled in the background, and a shape
  compiled while it is off stays built. Once the scene's shapes are known, both arms run the same modules; only the shapes
  first met during a phase differ, which its longest frame and its frames of 50 ms or more show. Compare its arms from a
  cold shader store, from the start of the game, not in a scene played before.
- `KYTY_E2_VSHARP_MIRROR`: read once where a dispatch's V# are turned into a module key. The mirrored form is
  part of that key, so the modules of both forms live side by side in one session and in the shader store, and a dispatch
  always runs the form its key names, whichever phase executes it. Without the variable arm A binds and arm B mirrors;
  with `KYTY_E2_VSHARP_MIRROR=1` it is the reverse. A phase that first meets a program compiles its form then, so
  compare from a warm store that holds both forms, or count the first phases of each arm as warm-up.
- `KYTY_E2_GPU_FORMED`: read where an ahead walk of a direct dispatch starts and where a dispatch's V# are turned into a
  module key. A dispatch whose words were left to the shader runs the formed module whichever phase records it, and a
  slot formed once stays formed for its program while the setting is on; in phase A the slot takes its module of the
  CPU-resolved form again. Compare from a warm store that holds both forms.

Refused, with every name not listed above. A setting that changes how shaders are compiled cannot follow a flip: the
modules already built keep their form, and the shader store records the profile once per session, so a phase would mix
both forms and store them under one profile. A setting read on the command processor thread cannot follow a flip of the
execution thread either: that thread reads several submissions ahead of execution, so its work for the next frame would already be
done under the old value.

`python3 tools/kyty_ab_report.py <planb/runs/<name> or a log>` reads the `[kyty-ab]` lines and, from a session folder,
`vkcount.log`, `threads.jsonl` and `phases.jsonl`. It keeps the phases whose settled part starts after the session's
`measure` step (`--from`), sets aside a phase in which a `threads.jsonl` window has a CPU-weighted wait above 10 %
(`--wait-max`: a disturbed window, where threads wait to run), leaves `--skip` more seconds out of each settled part,
and prints per arm the mean and standard deviation of img/s, the vkcount img/s and draws per image (more than 10 % apart
and the scenes do not compare), the longest frame, B − A and the difference of each B phase with the mean of its two A
neighbours, which cancels a linear drift, and the mean of each number of every periodic `[kyty-*] summary` line that
lies wholly inside a kept settled part, as `A|B`. `--json` writes the same in a file.

### Adding a setting

A pull request that makes the code under `core/` read a new `KYTY_*` name adds its row to the tables above, with its default taken from the code, and one that stops reading a name removes the row. `python3 tools/check_kyty_settings.py` lists what is missing or stale; the `conventions` job of [the AnyKyty workflow](../../.github/workflows/anykyty.yml) runs it. A new `ANYPS5_*` or `APS5_*` name also needs its row in [ENVIRONMENT](ENVIRONMENT.md).
