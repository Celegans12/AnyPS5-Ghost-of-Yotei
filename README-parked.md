# Parked: KytyPS5's GPU backend inside AnyPS5 (Ghost of Yōtei)

This branch is an experiment from 8 to 13 November 2026. It puts the GPU backend of KytyPS5-Yotei behind a small C
interface in AnyPS5, so that AnyPS5 keeps its own HLE, kernel, window and input while Kyty's command processor,
caches, shader recompiler and presenter draw the frames. It was built for one game, Ghost of Yōtei (PPSA26344), on one
machine.

**It is parked and unsupported.** Work stopped on 13 November in the middle of its correctness phase. Nobody is
maintaining it, it will not be rebased, and `yotei/best` does not contain it. It is published so that others can
build it, run it and compare. Nothing here was offered to upstream AnyPS5 or to KytyPS5.

## Where the code comes from

- `core/kyty_gpu/` is derived from **KytyPS5-Yotei, version 0.1.5**: https://github.com/Erlou93/KytyPS5-Yotei, tag
  `v0.1.5`, commit `529c5acd3a3e8663830cf4f5e79c8ee262955a7a`. The first commit of the import (`700fb8edd91`) holds
  464 unmodified files of that tag. The commits after it, prefixed `kyty_gpu:`, are our changes.
- **It does not include anything KytyPS5-Yotei changed after 0.1.5.** The fork released 0.1.6 on 11 November (its
  releases page, read on 14 November); none of that is here.
- Licence: `core/kyty_gpu/LICENSE` is the GNU General Public License, version 2, as shipped by KytyPS5-Yotei, and
  `core/kyty_gpu/NOTICE` names it as version 2 only. KytyPS5 is based on Kyty by InoriRus under the MIT License;
  that notice is `core/kyty_gpu/LICENSES/Kyty-MIT.txt`. AnyPS5 itself is GPL version 2 (`LICENSE`).
- `core/kyty_gpu/NOTICE` lists every file that differs from the base commit (42 changed, 28 added) with the date of
  its last change.
- One inconsistency we left as it is, because this branch is published unchanged: two source comments
  (`core/kyty_gpu/src/aps5/memoryShim.cpp` and `videoOutBridge.cpp`) and a few commit messages call the Kyty code
  "GPL-2.0-or-later". The licence file and the NOTICE are what count.
- `core/kyty_gpu/3rdparty/cpuinfo` is a copy Kyty keeps in its tree (BSD-2-Clause, its own `LICENSE`). The other
  third-party libraries are git submodules.
- The AnyPS5 side (the seam header, the adapter, a null backend and the direct-memory changes under `core/libs/`)
  is ours.

## Building (Windows only)

You need what AnyPS5 needs already (`docs/dev/BUILD.md`), plus:

- **llvm-mingw 20261104, ucrt, x86_64**, unpacked anywhere. Only the new backend is built with it. Other versions
  were not tried.
- **The Vulkan SDK**, for `glslangValidator` (we used 1.4.363.0).
- A git clone with submodules, not a source zip.

```
git submodule update --init --recursive
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DAPS5_LLVM_MINGW_ROOT=<your llvm-mingw directory>
cmake --build build --target kyty_gpu
cmake --build build --target libs relinker
```

- The backend is a separate DLL, `kyty_gpu.prx`, built from `core/kyty_gpu` and copied beside the other libraries.
- `-DAPS5_KYTY_GPU=OFF` builds AnyPS5 without it. `LLVM_MINGW_ROOT` in the environment works instead of the CMake
  option. `-DAPS5_KYTY_PGO=OFF` builds without the shipped profile (`core/kyty_gpu/pgo/kyty_gpu.profdata`).
- These commands were run from a clean worktree on our machine only, with the submodules and FFmpeg taken from local
  copies. A cold build on another machine was never tried.
- **Linux:** the backend's build files stop on anything but Windows. From reading the code, AnyPS5's own Linux build
  is probably broken on this branch too (`KytyAdapter.cpp` uses Windows calls without a guard). Nobody built it.

## Selecting the backend

Start the game with the environment variable `APS5_GPU_BACKEND=kyty`. Without it the old backend runs and the DLL is
not loaded. `APS5_GPU_BACKEND=null` selects a test backend that draws nothing.

The first start compiles pipelines: 1,940 of them in about 9 minutes on our machine with an empty cache. The error
output shows `[kyty] backend 'kyty' created at its first use` near the start and a `[present] N presents over 10 s`
line every ten seconds. The backend's own switches are environment variables starting with `KYTY_`; their defaults
are in `core/kyty_gpu/src/common/envDefaults.h`.

## What we measured

Everything below is from one laptop (i9-11900H, RTX 3080 Laptop, Windows 11, NVIDIA), at 1080p, in the river
village, as presents per 10 seconds. "T1" is our own check: 60 frames of a fixed start sequence compared byte for
byte with a reference made by the same backend.

| date | build | result |
|---|---|---|
| 8 Nov | first working build (out-of-tree DLL) | 60.85 and 61.99 in two 1,500-second runs; the old backend read 9.25 with the same recipe; Kyty standalone read 65.0 in the matching scene |
| 10 Nov | integration build, new NVIDIA driver | 80.1 with the large GPU read window off, 84.1 with it on (+6.1 %) |
| 11 Nov | source in this tree, commit `c6fd8e3d9a3` | T1 match; a 900-second run at 84.41, no fatal error; private memory at most 11,685 MiB |
| 11 Nov | the same sources as this branch's tip, as an out-of-tree DLL | T1 match; one run at 79.74 after a Windows update. The comparison runs against the previous build were not done |
| 11 Nov | this branch's tip (`49ad229055d`), built in a clean worktree | T1 match. Its world run and its speed comparison were not done |

The reports under `docs/yotei/kyty-backend/` have the runs behind each number.

## What works and what does not

- Works on our machine: start, intro, loading a save and the river village, at the speeds above.
  The silver sheen and black rocks that the old backend shows in this scene are not there.
- Not part of the DLL: Kyty's window, host input, overlays, RenderDoc hook, performance sampler and VideoOut HLE.
  AnyPS5 does those; the overlay calls answer "off" (`core/kyty_gpu/src/aps5/presentationStubs.cpp`).
- Unfinished: the correctness phase. Kyty's stencil/HTILE and fast-clear workarounds and its shader loop limit are
  still in; an exact fast-clear variant, a guard for one direct-memory case and one more speed lever sit on side
  branches that were never merged and are not published here. No two-hour soak was run.
- Only one game, one game version, one machine, one GPU vendor. No AMD or Intel GPU was tried.

## The write-ups

`docs/yotei/kyty-backend/` holds the working reports of those days, as written at the time, with a few
lines removed that concerned unrelated work (marked `[redacted]`). They are lab notes: they name local paths, run
tags and work folders that are not in this repository, and `2026-11-11-v4-intree.md` was never finished (its `TODO`
markers are original). Start with `2026-11-08-v4-seam.md`, `2026-11-08-v4-g2.md` and `2026-11-11-v4-intree.md`;
`2026-11-08-tried-ledger.md` lists what was tried on the old backend before this.

You need your own legally obtained copy of the game. No game files are in this repository.
