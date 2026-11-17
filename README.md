# AnyKyty

**AnyKyty is this fork's branch: AnyPS5 running the Kyty graphics core as a second GPU backend.** It is experimental (v0). The upstream AnyPS5 README follows [below](#about).

## What changes

AnyPS5 keeps everything it does today: the relinker, `libkernel`, every system library, the SDL window, input and audio, and `libSceAgc`, the AGC library games call. Only the GPU driver behind the AGC interface is swapped at configure time:

| | `-DANYPS5_GPU_BACKEND=anyps5` (default) | `-DANYPS5_GPU_BACKEND=kyty` (AnyKyty) |
|---|---|---|
| PM4 command processing | AnyPS5 driver (decode, finishers, committer, encoder) | Kyty command processor (`kytyGpu`) |
| Draw preparation | AnyPS5 committer | Kyty resolve and execute threads, with the ordered resolution and compiled SRT walk |
| Vulkan rendering and caches | AnyPS5 renderer | Kyty renderer, buffer, texture and pipeline caches, recorder thread |
| Shader recompiler | AnyPS5 recompiler (`core/shader/recompiler`) | Kyty recompiler (`core/kyty/graphics/shader/recompiler`) |
| GPU view of game memory | AnyPS5 imports and write tracking | Kyty model: every guest page has a second view, writes are tracked by page faults |
| Presentation | AnyPS5 presentation | Kyty presenter and swapchain, on AnyPS5's window; AnyPS5's VideoOut still reserves, paces and presents the flips |

The default build is AnyPS5 as before: it compiles none of the Kyty files. The Kyty core lives in `core/kyty/`, taken from Kyty at `40ddebbe`; the glue between it and AnyPS5 (memory, faults, entry points, presentation, events) lives in `core/libs/prx/libSceAgcDriver/KytyShim/`. Changes to shared AnyPS5 files are under `ANYPS5_KYTY`, except the optional `ANYPS5_VBLANK_HZ` setting, which leaves behaviour unchanged when it is not set. Kyty's general shortcuts are kept as they are, for example skipping a draw while its pipeline compiles in the background. Its title-specific hooks (Demon's Souls page freeing, ASTRO BOT, Oodle) are not ported. AnyKyty also turns on by default two Kyty settings that native Kyty only enables in its launcher's Experiment mode, draw templates and grouped draw buffers (`KYTY_DRAW_TEMPLATES=2`, `KYTY_DRAW_BUFFERS=1`); setting either variable keeps the user's value. Native ray tracing, which native Kyty's launcher turns on in its recommended settings, is on by default when the GPU supports it (`KYTY_NATIVE_RT`, see the runtime settings below).

## Why

On the same PC and scene, Kyty spent about a third of AnyPS5's CPU time per draw and a fifth of its GPU time per frame. Reworking the AnyPS5 driver towards that design piece by piece did not close the gap, so the whole graphics core was transplanted instead.

## Measurements

Subnautica (PPSA02453), Creative mode, inside the lifepod. Ryzen 9 7950X3D, RTX 4080 SUPER, Ubuntu 26.04, 17 Nov 2026. Busiest stage = the thread with the highest CPU cost per draw call.

Capped at 60 Hz:

| | FPS | Busiest stage | GPU per frame |
|---|---|---|---|
| AnyPS5, default backend | 45.7 | 15.5 µs per draw (committer) | 12.2 ms |
| AnyKyty (`e6dfd952b5`, before the fixes below) | 59.8 | 6.7 to 7.7 µs per draw (`kytyGpuExecute`) | 6.5 ms |
| Kyty, native | 59.4 | 5.7 µs per draw | 5.7 ms |

Uncapped (`ANYPS5_VBLANK_HZ=360`, native Kyty with `KYTY_VBLANK_FREQUENCY=360`), two sessions each. Both sides ran with Kyty's default settings (its Original mode): these measurements predate the Experiment settings above.

| | FPS | Execute | Resolve | Record | GPU per frame |
|---|---|---|---|---|---|
| AnyKyty (`6155614613`) | 159.0 (157.2, 160.8) | 5.73 µs per draw | 5.14 µs | 3.54 µs | 3.9 ms |
| Kyty, native | 112.4 (106.3, 118.6) | 7.23 µs | 6.26 µs | 4.00 µs | 5.0 ms |

At a comparable draw count (911 against 863 per frame), every Kyty stage is 12 to 21 % lighter per draw on AnyKyty and the GPU 22 % lighter per frame. The latest gains come from the glue: inside the Kyty stages, the memory shim now sits in the ThinLTO object and TLS is initial-exec; on the game threads, guest memory copies no longer go through system calls and submissions no longer `msync`, which leaves more of the CPU to the stages. Native Kyty's presentation thread is replaced here by AnyPS5's VideoOut. Why AnyKyty now beats native Kyty is not fully attributed yet. Subnautica loads a slightly different world on each launch, and per-draw costs vary by about ±20 % between sessions; both sessions show the same trend. Loading the game from the Creative menu takes 3.5 s on AnyKyty, 4.0 s on native Kyty and 8.3 to 13.9 s on the default backend, with warm caches.

## Status

- Tested in game on Linux: Subnautica, Balatro and TCG Card Shop Simulator.
- Known issue, inherited from Kyty: the street of TCG Card Shop Simulator stays almost black, as it does on native Kyty. Rendering differences against the default backend are being collected and will be fixed after performance work.
- The Kyty backend has only been built and run on Linux. Windows is untested.

## Building AnyKyty

Follow [BUILD](docs/dev/BUILD.md), then configure with:

```
cmake -B build -DANYPS5_GPU_BACKEND=kyty
```

The Kyty core does not build with GCC, so it is compiled by Clang in a sub-project, against the same libstdc++ as the rest of AnyPS5. It needs `clang`, `clang++` and an `ld.lld` of the same major version (tested with 21). It is built with ThinLTO and pre-linked into one object by default; `-DANYPS5_KYTY_THIN_LTO=OFF` turns that off.

Runtime settings: `ANYPS5_VBLANK_HZ` (30 to 360) sets the vblank rate, so `ANYPS5_VBLANK_HZ=240` lifts the 60 FPS cap of games that pace on vblank. The `KYTY_*` settings of Kyty are read from the environment, for example `KYTY_VBLANK_FREQUENCY` and `KYTY_PRESENT_MODE`.

Native ray tracing (Kyty's `--native-rt`) is on by default when the Vulkan device has `VK_KHR_acceleration_structure`, `VK_KHR_ray_query` and `VK_KHR_deferred_host_operations`: the ray tracing passes whose BVH walk Kyty recognizes trace with the GPU's ray tracing hardware, and the other ones run emulated in shaders as before. A `[kyty] native ray tracing` line on the console says which way it went; on a GPU without these extensions it names what is missing and ray tracing runs emulated. `KYTY_NATIVE_RT=0` emulates all ray tracing; `KYTY_NATIVE_RT=1` stops with a message when the GPU cannot trace natively. `KYTY_RT_STATS=1` prints a `[RT]` summary every 10 s of the work native ray tracing leaves to the game. The acceleration structures Kyty builds take at most a quarter of the video memory, between 512 MiB and 4 GiB (2 GiB on an 8 GB card), allocated as needed; `KYTY_RT_ARENA_MB` sets another limit.

## Branches

- `AnyKyty` (default): AnyPS5 with the Kyty backend.
- `feat/agc-draw-block`: the performance work on AnyPS5's own driver.
- `main`: a copy of upstream AnyPS5.

## Credits and licence

The graphics core: `core/kyty` is maintained in this repository; it derives from [KytyPS5](https://github.com/KytyPS5/KytyPS5) (GPL-2.0); everything else is [AnyPS5](https://github.com/boykopovar/AnyPS5). Kyty is GPL-2.0 like AnyPS5; the original Kyty code by InoriRus is MIT. File by file origin, licences and the pinned commit: [docs/dev/THIRD_PARTY_KYTY.md](docs/dev/THIRD_PARTY_KYTY.md).

---

**English** | [Português (Brasil)](docs/user/README.pt-BR.md) | [Español](docs/user/README.es.md) | [Català](docs/user/README.ca.md) | [Italiano](docs/user/README.it.md) | [简体中文](docs/user/README.zh-CN.md) | [Русский](docs/user/README.ru.md) | [Bahasa Indonesia](docs/user/README.id.md) | [Polski](docs/user/README.pl.md) | [日本語](docs/user/README.ja.md)

# About

Tool for automatic executables porting to Linux and Windows.

The relinker-only build runs on macOS, including Apple silicon.

[**Web converter**](https://boykopovar.github.io/AnyPS5/) - files stay on your device.

![anyps5_collage1](https://github.com/user-attachments/assets/ed2f0b2d-33d3-46d7-ab88-767ca872fc99)

Includes a [relinker](core/relinker) that converts executable to the target system's native format and implementations of [system prx libraries](core/libs/prx) suitable for dynamic linking. No emulation or separate runtime process.

[**Discord**](https://discord.gg/x9naWVzC6A), [Usage](docs/user/USAGE.md), [Build instructions](docs/dev/BUILD.md), [Architecture](docs/dev/ARCHITECTURE.md), [Technical debt of the project](docs/dev/TechnicalDebt.md), [code style conventions](docs/dev/CONVENTIONS.md), [contributing](CONTRIBUTING.md)

## Status

[![libraries](https://boykopovar.github.io/AnyPS5/badge-libraries.svg)](https://boykopovar.github.io/AnyPS5/progress.html) [![RDNA2 reference shader forms decoded](https://boykopovar.github.io/AnyPS5/badge-shaders.svg)](https://boykopovar.github.io/AnyPS5/progress.html) [![gfx1013 reference shader forms decoded](https://boykopovar.github.io/AnyPS5/badge-shaders-gfx1013.svg)](https://boykopovar.github.io/AnyPS5/progress.html) [![implemented in 7 days](https://boykopovar.github.io/AnyPS5/badge-week.svg)](https://boykopovar.github.io/AnyPS5/history.html)

[![progress map](https://boykopovar.github.io/AnyPS5/progress.svg)](https://boykopovar.github.io/AnyPS5/progress.html)

[Progress history](https://boykopovar.github.io/AnyPS5/history.html)

<sub>* System libraries: percentage of the functions known to the project so far (declared in [core/libs/prx](core/libs/prx)), not of every PS5 system function. The total grows as more functions are declared.</sub>

Unsupported or unexpected states throw a C++ exception, usually `std::runtime_error`. An exception that nothing catches prints its `what()` to stderr and ends the process.

The [shader recompiler](core/shader/recompiler/Recompiler.cpp) successfully produces SPIR-V (validated via [Spirv-Tools](3rdparty/SPIRV-Tools) when built with `ANYPS5_ENABLE_SPIRV_TOOLS`).

## Compatibility

Per-game status and test reports live in the [Game bring-up](https://github.com/boykopovar/AnyPS5/discussions/categories/game-bring-up) discussions, one thread per title.

## Input mapping

SDL-mapped game controllers are supported, including analog sticks and triggers. Keyboard and mouse controls can be configured with an `anyps5-input.ini` file. See [input mapping](docs/user/INPUT_MAPPING.md) for the supported devices and configuration format.

## Disclaimer

This project is intended for interoperability, research, preservation, and compatibility purposes. It does not include, distribute, or require copyrighted software, firmware, cryptographic keys, or proprietary libraries. Users are responsible for ensuring that any binaries used with this project are obtained and used in accordance with applicable laws and their respective license terms.

AnyPS5 is an independent project and is not affiliated with, authorized or endorsed by Sony Interactive Entertainment. "PlayStation" and "PS5" are trademarks of Sony Interactive Entertainment Inc.

## License

This project is licensed under the GNU General Public License version 2 only.
