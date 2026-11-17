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

The headers under `KytyShim/include/` are the contract of the shim: they are the declarations of the KytyPS5 kernel,
library and loader functions that `graphics/` and `common/` call, and the shim implements them on top of AnyPS5.
`libs/dialog.h` is left out: only `graphics/presentation/systemOverlay.cpp`, which `kyty_gpu` does not build, includes
it, and its `ErrorDialogOpen` and `ErrorDialogClose` would be second declarations of AnyPS5 exports for
`core/libs/tests/test_public_headers.py`, which reads every header under `core/` without namespaces.

## Changes from upstream

- Every comment is removed, except the two notices above. `core/kyty/tools/kyty_import.py` does it with a C++ lexer
  that keeps string, character and raw string literals and digit separators intact, and checks each file: the token
  stream that `g++ -fpreprocessed` produces is the same before and after, preprocessor line ends included.
- Nothing else is changed in the imported files. Adaptations to AnyPS5 go in the shim, or under
  `#if defined(ANYPS5_KYTY)` in `core/kyty/`. `ANYPS5_KYTY=1` is defined for every target of a
  `-DANYPS5_GPU_BACKEND=kyty` build and never otherwise.

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
