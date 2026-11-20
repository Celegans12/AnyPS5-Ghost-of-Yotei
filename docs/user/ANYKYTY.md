# AnyKyty

**No setting needed.** Convert your game and start it: every default is the fastest and most stable choice measured so far, and a few options only change how the game is shown ([Settings](#settings)).

**AnyKyty is [AnyPS5](https://github.com/boykopovar/AnyPS5) running the Kyty graphics core.** Everything else stays AnyPS5: the relinker, `libkernel`, the system libraries, the window, input and audio. Only the GPU driver behind the AGC interface changes, selected at configure time with `-DANYPS5_GPU_BACKEND=kyty`.

It exists for speed. On the same PC and scene, AnyKyty renders **63 % more frames per second than native Kyty**, with 38 to 53 % less CPU time per draw call on each of its stages.

```text
Subnautica, uncapped, Ryzen 9 7950X3D + RTX 4080 SUPER, Linux

AnyKyty (this fork)        ████████████████████████████████████  182.9 FPS
Kyty, native               ██████████████████████                112.4 FPS
```

- **What you get:** the Kyty renderer's speed and shader coverage inside AnyPS5, with extra performance work on top (listed [below](#what-makes-it-faster)).
- **Where it runs:** Linux (main target) and Windows (builds and runs, less tested).
- **Status:** experimental. There are no binary releases: [build it from source](#building).
- **Help wanted:** pull requests, game reports and measurements on other hardware are welcome, see [contributing](#contributing).

## Contents

- [Performance](#performance)
- [What makes it faster](#what-makes-it-faster)
- [Games](#games)
- [Building](#building)
- [Running a game](#running-a-game)
- [Ray tracing survey](#ray-tracing-survey)
- [Identify shaders in a GPU hang dump](#identify-shaders-in-a-gpu-hang-dump)
- [Settings](#settings)
- [Contributing](#contributing)
- [Credits and licence](#credits-and-licence)

## Performance

All numbers come from one machine: Ryzen 9 7950X3D, RTX 4080 SUPER (NVIDIA driver 595.99.02), Ubuntu 26.04, measured on 17 and 18 Nov 2026. The scene is Subnautica (PPSA02453), Creative mode, inside the lifepod, with warm shader and pipeline caches. Both sides ran uncapped: `ANYPS5_VBLANK_HZ=360` on AnyKyty, `--vblank-frequency 360` on native Kyty. Subnautica uses dynamic resolution, between about 1920×1080 and 2189×1232 in the sessions we inspected.

| | Frames per second |
|---|---|
| **AnyKyty** (`a0a5a0d4dc`) | **182.9** (two sessions: 177.2 and 188.7) |
| Kyty, native | 112.4 (two sessions, 17 Nov) |

CPU time per draw call, from the same sessions (a thread per stage; lower is better):

| Stage | AnyKyty | Native Kyty v24 | Change |
|---|---|---|---|
| Execute (`kytyGpuExecute`) | 3.64 µs | 7.23 µs | −50 % |
| Resolve (`kytyGpuResolve`) | 3.08 µs | 6.26 µs | −51 % |
| Record (`kytyGpuRecord`) | 2.49 µs | 4.00 µs | −38 % |
| Command processor (`kytyGpu`) | 1.59 µs | 3.40 µs | −53 % |
| GPU time per frame | 4.28 ms | 4.95 ms | −14 % |
| Draw calls per frame | 882 | 863 | |

How to read these numbers:

- The two sides were not run back to back. Native Kyty was measured on 17 Nov, AnyKyty on 18 Nov, same PC, same scene, same settings. Subnautica loads a slightly different world on each launch, and the frame rate of the same binary varies by about ±10 % between sessions. The gap is much larger than that, but treat the exact percentages as approximate.
- Capped at 60 Hz (17 Nov, before the gains below), AnyPS5's own driver reached 45.7 FPS on this scene, while the Kyty core reached 59.4 FPS natively and 59.8 FPS in AnyKyty, with about half the GPU time per frame: 5.7 and 6.5 ms against 12.2 ms.
- AnyKyty measured 159.0 FPS on 17 Nov (`6155614613`) and 182.9 FPS on 18 Nov. The main change in between is the thread placement described below.
- Everything was measured on NVIDIA hardware. Reports from AMD and Intel GPUs and from other CPUs are welcome: see [contributing](#contributing).

## What makes it faster

Native Kyty already sets the baseline. These are the changes in AnyKyty beyond it, each measured in game unless noted:

- **Threads stay on the V-Cache.** On Linux, native Kyty does not pin its GPU threads: they float over every CPU and both chiplets. AnyKyty keeps each stage on the CPUs that share the largest L3 cache, which is the 3D V-Cache chiplet of a Ryzen X3D. On a 7950X3D: +7 % to +22 % frames per second depending on the scene, with 25 to 35 % less CPU time per draw on every stage. Measured on that one CPU.
- **The game gets its flip back earlier.** A frame is handed back as soon as its presentation copy is submitted instead of after the GPU finished it, so the game starts the next frame while the GPU ends this one. Images are identical. +1.7 % and +4.8 % frames per second in two sessions of 17 paired periods.
- **No multi-second freezes when VRAM runs short.** When the texture cache evicts GPU-written images, native Kyty writes them back to guest memory by retiling them straight into host memory across PCIe, about 0.75 s per 4K render target. AnyKyty tiles in VRAM and copies once. Stray with 8 GiB of VRAM taken by another program: worst frame 5.5 s → 0.6 s, same images.
- **Image readbacks no longer undo the game's writes.** Native Kyty writes a GPU-written image back over its whole memory footprint from a copy read earlier, which can put back bytes the game wrote in between. AnyKyty writes only the bytes the GPU changed and that the game left alone.
- **A narrower lock in the resolve stage.** The resolve thread no longer spins on the program-cache lock while the execute thread waits for the GPU inside it.
- **Faster loading.** Reading guest memory that the game never wrote used to make the kernel allocate and zero hundreds of thousands of pages; AnyKyty skips those holes and writes the zeros itself (233,000 page faults per load before, 3,300 to 4,400 after, about 1.5 GiB less resident memory). The GPU also starts while the window is being created. Subnautica loads its Creative game in about 2.2 s on AnyKyty against 3.6 to 4.0 s on native Kyty and 8 to 14 s on AnyPS5's default driver (measured in separate sessions, so indicative).
- **Cheaper glue between AnyPS5 and the Kyty core.** The memory shim is compiled into the same ThinLTO object as the core, thread-local variables use the initial-exec model, guest memory copies skip system calls, and submissions no longer `msync`.
- **Kyty's optional features on by default.** Draw templates and grouped draw buffers, which native Kyty only enables in its launcher's Experiment mode (−4.1 % on Execute time per draw), and native ray tracing when the GPU supports it.

Each of these has a switch to turn it off, for comparing and for bug reports; they are listed in [AnyKyty settings for developers](../dev/ANYKYTY_SETTINGS.md) and none is needed to play. Why AnyKyty ends up faster than native Kyty is not fully attributed yet: part of it is the list above, part is that AnyPS5's video output replaces Kyty's presentation thread. The notes behind these numbers are in [the third-party document](../dev/THIRD_PARTY_KYTY.md).

## Games

Games are not shipped or provided; bring your own legally dumped title.

**Run by the maintainer on Linux with AnyKyty:**

| Game | Title ID | Notes |
|---|---|---|
| Subnautica | PPSA02453 | The benchmark scene above |
| Stray | PPSA02101 | Plays; first launch stutters while shaders and pipelines compile ([#33](https://github.com/simnJS/AnyPS5/issues/33)) |
| TCG Card Shop Simulator | PPSA36301 | Street rendering fixed in AnyKyty (native Kyty shows it almost black) |
| Balatro | | Runs at the vblank rate: 354 FPS with `ANYPS5_VBLANK_HZ=360` |

**Reported by contributors (merged pull requests and open issues):**

| Game | Title ID | State |
|---|---|---|
| An open-world AAA title | | Linux: in game; two crashes found in long sessions are fixed in [#34](https://github.com/simnJS/AnyPS5/pull/34) and [#35](https://github.com/simnJS/AnyPS5/pull/35). Windows (NVIDIA): reaches the prologue |
| METAL GEAR SOLID 4 | PPSA18194 | Runs in game; an exit hang is fixed in [#23](https://github.com/simnJS/AnyPS5/pull/23); memory tracking costs about 20 % of the execute stage ([#24](https://github.com/simnJS/AnyPS5/issues/24)) |
| Planet Zoo | PPSA18026 | Windows: its stops are fixed ([#25](https://github.com/simnJS/AnyPS5/pull/25), [#28](https://github.com/simnJS/AnyPS5/pull/28), [#41](https://github.com/simnJS/AnyPS5/pull/41)); it still draws most of its 3D scene wrongly ([#30](https://github.com/simnJS/AnyPS5/issues/30)) |
| LEGO Builders Journey | PPSA05496 | The first launch runs; the second stops in the PM4 parser ([#21](https://github.com/simnJS/AnyPS5/issues/21)) |
| 007 First Light | PPSA11386 | Menus run at 35 to 45 FPS; merged hull programs run ([#31](https://github.com/simnJS/AnyPS5/issues/31)) and the first cutscene plays at about 29 FPS on an RX 9070 XT with RADV ([#32](https://github.com/simnJS/AnyPS5/issues/32)); gameplay after it hangs the GPU on RADV ([#98](https://github.com/simnJS/AnyPS5/issues/98)) |

Other titles have not been tried on AnyKyty by us. A report of any title, working or not, helps: see [contributing](#contributing).

## Building

AnyKyty is the `AnyKyty` branch of this fork, its default branch. The default `-DANYPS5_GPU_BACKEND=anyps5` builds AnyPS5 as before and compiles none of the Kyty files. The Kyty core does not build with GCC: it is compiled by Clang in a sub-project, against the same libstdc++ as the rest of AnyPS5. The commands below are the ones of the [`AnyKyty` workflow](../../.github/workflows/anykyty.yml), which builds every pull request.

### Linux

Needs Git, CMake 3.21 or newer, GCC 14, Ninja and LLVM 21 (`clang`, `clang++`, `ld.lld` of the same major version, on `PATH`):

```sh
sudo apt-get install -y gcc-14 g++-14 ninja-build pkg-config ccache \
  libx11-dev libxext-dev libxrandr-dev libxkbcommon-dev libwayland-dev libegl-dev libdecor-0-dev \
  libxcursor-dev libxi-dev libxfixes-dev libxinerama-dev libxss-dev libxxf86vm-dev libudev-dev libdrm-dev libgbm-dev \
  libasound2-dev libpulse-dev libpipewire-0.3-dev libvulkan1 libvulkan-dev mesa-vulkan-drivers \
  clang-21 lld-21 llvm-21
```

On Ubuntu 24.04 the CI gets the LLVM 21 packages from [apt.llvm.org](https://apt.llvm.org) (`deb http://apt.llvm.org/noble/ llvm-toolchain-noble-21 main`) and puts `/usr/lib/llvm-21/bin` on `PATH`. Then:

```sh
git clone --recurse-submodules https://github.com/simnJS/AnyPS5.git
cd AnyPS5
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 \
  -DANYPS5_GPU_BACKEND=kyty -DANYPS5_KYTY_JOBS="$(nproc)"
cmake --build build --target kyty_gpu libs relinker --parallel "$(nproc)"
```

`-DANYPS5_KYTY_THIN_LTO=OFF` turns off ThinLTO and the pre-link of the Kyty core, which is on by default. The Kyty core is also optimized with the profile kept in `core/kyty/pgo/` (profile-guided optimization, which needs the `llvm-profdata` of the same LLVM beside `clang`); `-DANYPS5_KYTY_PGO=` builds it without, and [the PGO document](../dev/KYTY_PGO.md) explains how to record the profile again. The CI also builds and runs the Kyty shim tests on Linux; the exact commands are in the `linux` job of the workflow.

### Windows

AnyPS5 is built with MinGW-w64 GCC as usual, and `clang` and `clang++` of an LLVM release compile the Kyty core for `x86_64-w64-windows-gnu` against the headers and libstdc++ of that GCC (`ANYPS5_KYTY_SYSROOT`, by default the directory above the `bin` of `g++`). The CI uses:

- [WinLibs](https://github.com/brechtsanders/winlibs_mingw/releases/tag/15.2.0posix-14.0.0-ucrt-r7) GCC 15.2.0, `winlibs-x86_64-posix-seh-gcc-15.2.0-mingw-w64ucrt-14.0.0-r7.zip`;
- [LLVM 21.1.8](https://github.com/llvm/llvm-project/releases/tag/llvmorg-21.1.8) for Windows: `clang`, `clang++`, `llvm-ar`, `llvm-ranlib`, `llvm-profdata` and its `lib/clang` directory;
- Git for Windows with Bash, Python 3, CMake and Ninja.

Put the WinLibs and LLVM `bin` directories first on `PATH`, then, from `cmd.exe` (in PowerShell write `$env:NUMBER_OF_PROCESSORS` for `%NUMBER_OF_PROCESSORS%`):

```bat
git clone --recurse-submodules https://github.com/simnJS/AnyPS5.git
cd AnyPS5
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DANYPS5_GPU_BACKEND=kyty -DANYPS5_KYTY_JOBS=%NUMBER_OF_PROCESSORS%
cmake --build build --target kyty_gpu libs relinker --parallel %NUMBER_OF_PROCESSORS%
```

ThinLTO is off on Windows, since lld cannot pre-link Windows objects, and Kyty's pipeline helper processes are not used. The runtime DLLs of the toolchain go beside the libraries, as described in [Windows setup and troubleshooting](../dev/BUILD.md#windows-setup-and-troubleshooting).

## Running a game

The build produces the relinker (in `build/core/relinker/`) and the system libraries (`build/core/libs/libs/*.prx`). Convert the game, put the libraries in the `libs/` folder beside the output, and start it:

```sh
relinker source/input.elf app.elf
./app.elf
```

No environment variable is needed to start it. The relinker built with AnyKyty gives the functions that page tracking needs a frame on its own for a Windows output (`--no-frame-red-zone` turns that off), and a title that asks for an add-on is told it is installed (see [Add-ons](USAGE.md#add-ons)). On Windows, convert with `relinker --windows source/input.elf app.exe`, and add `--to-intel` on an Intel CPU.

The full procedure, with module folders and Windows output, is in [usage](USAGE.md) and [game testing](../dev/GAME_TESTING.md). Games that pace on vblank run at 60 FPS; to lift that cap see `ANYPS5_VBLANK_HZ` in the [settings](#settings).

## Ray tracing survey

A game that uses ray tracing walks its acceleration structure in compute shaders, and AnyKyty serves each of those programs in one of five ways: by a recognized walk form (`native`), by a host ray query that answers for the whole wave (`oracle`) or one lane at a time (`lane`), by running the walk empty so its rays miss (`empty`), or by leaving the game's own walk code to run in the shader (`emulated`). Which one a program gets decides how fast and how right the effect is, and it differs from game to game. The survey lists, for every ray tracing program in the shader cache of your game, which way AnyKyty serves it, so that the coverage can be measured on many titles. It reads the cache of programs the game registered, translates them again offline as the driver does, and sends nothing anywhere.

The survey describes what AnyKyty does on a GPU with Vulkan ray queries (NVIDIA RTX, AMD RX 6000 and newer, Intel Arc); on any other GPU it emulates all ray tracing and there is nothing to survey. The report holds no game code and no file name of the game: for each program, only its fingerprint (a hash), its stage, wave size and workgroup size, the number of BVH intersections, how many of them are served each way and whether the translated SPIR-V validates; and the AnyKyty commit, your GPU and driver, and your operating system. Home folders, user names and host names are filtered out, and the command refuses to write a report that still holds a path of your machine. Read both files before you share them anyway.

1. **Update and build the probe.** Pull the current branch and its submodules and build the probe with the rest, from the folder you built AnyKyty in:

   ```sh
   git pull
   git submodule update --init --recursive
   cmake --build build --target kyty_gpu libs relinker kyty_rt_probe --parallel "$(nproc)"
   ```

   On Windows use `%NUMBER_OF_PROCESSORS%` (or `$env:NUMBER_OF_PROCESSORS` in PowerShell). The probe is a small program, `build/tests/kyty_rt_probe`, and is also built by a plain `cmake --build build`.
2. **Play for 5 to 10 minutes** with the game's ray tracing option on, in places where it is visible. The shaders a game registers are kept in `_PipelineCache/` in the folder you start the game from, so quit the game normally when you are done.
3. **Run the survey** from that folder, or name it with `--game`:

   ```sh
   python3 /path/to/AnyPS5/tools/rt_survey.py
   ```

   The script needs Python 3 and nothing else. It finds the probe in `build*/tests/` of the AnyKyty folder it sits in (or `--probe`, or `KYTY_RT_PROBE`, or `PATH`) and the newest `_PipelineCache/<title>_<version>.code` (or `--cache`, or `--title` to pick a title), and reads your GPU and the commit from the `anyps5-report.json` the game wrote when it ended (or from a log with `--log`, or `vulkaninfo`, or `nvidia-smi`). It writes `rt-survey-<title>-<version>.json` and `rt-survey-<title>-<version>.txt` in the current folder (`--out-dir`) and prints the text, which lists the programs that are not fully served first. A program that calls functions (many Unreal Engine ray tracing shaders do) is listed as `calls`: the probe sees only its entry code, so its intersections are counted but not classified.
4. **Share it** in the ray tracing survey issue of this repository: paste the text and attach the JSON file.

On a GPU with 64-wide subgroups (AMD) pass nothing: the script reads it from the log, or from the vendor name. `--subgroup 32|64`, `--lds-limit <bytes>` and `--oracle <n>` set the GPU side and the `KYTY_RT_ORACLE` value by hand. `kyty_rt_probe --cache <file>.code --all` is the command underneath; [the probe reference](../dev/THIRD_PARTY_KYTY.md#ray-tracing) lists its options and the format of its lines.

An AI agent can do these steps for you: the survey issue holds a prompt to paste into Claude Code or another coding agent.

## Identify shaders in a GPU hang dump

Build the matcher with a C++20 compiler (MinGW on Windows).

```sh
git submodule update --init 3rdparty/fmt 3rdparty/magic_enum 3rdparty/SPIRV-Headers 3rdparty/Vulkan-Headers 3rdparty/VulkanMemoryAllocator 3rdparty/xxHash 3rdparty/zstd
cmake -S core/kyty/tools -B build-spirv-match -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-spirv-match --target kyty_spirv_match
build-spirv-match/kyty_spirv_match _PipelineCache/<title>.shaders <vertex>.spv <fragment>.spv
```

On Windows, put MinGW and Ninja on `PATH`, add `-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++` to the configure command, and run `kyty_spirv_match.exe`.

Use the compiled `.shaders` store from the run that produced the dump. The matcher compares the exact bytes of each supplied `.spv` file against the standard and plain variants in that store. It prints the input number, stage, guest shader hash and Kyty module ID for each match. You can share those lines while keeping the shader files and cache local.

The command exits with `1` if an input has no exact match and `2` if an input or store is invalid or incomplete. It reads one store record at a time and accepts up to 64 MiB of decoded data per record or input shader. It opens no Vulkan device and leaves the inputs untouched.

## Settings

None is needed: a game converted with the relinker starts and runs on the defaults, which are the fastest and most stable choices measured so far. The options below are choices about how the game is shown or run on your machine, not tuning. They are environment variables, read once when the game starts; an invalid value stops the game with a message that names the variable.

| Variable | Values | Effect |
|---|---|---|
| `ANYPS5_FULLSCREEN` | `0`, `1`, `windowed`, `borderless` | Opens the window fullscreen |
| `ANYPS5_DISPLAY_INDEX` | index of a connected display, `0` by default | Display the window opens on |
| `APS5_OUTPUT_RESOLUTION` | `<width>x<height>` | Output resolution reported to the game; unset, it follows the display. A size below 3840x2160 reports a sub-4K display, and a game that adapts to the display then renders fewer pixels. On a weaker GPU with a 4K display, `2560x1440` is worth a try |
| `KYTY_GPU` | GPU index, `-1` for AnyKyty's choice (default) | Picks the Vulkan device |
| `KYTY_HDR` | `0`, `1` | Asks for an HDR10 output when the display offers it |
| `KYTY_STRETCH` | `0`, `1` | Fills the window instead of keeping the game's aspect ratio |
| `ANYPS5_UPSCALE` | `fsr`, `cmaa2`, `rcas`, unset | Scales the picture with FSR 1 when the window is larger than the game's image, applies CMAA2 anti-aliasing, or only sharpens. `ANYPS5_FSR_SHARPNESS` sets the sharpness |
| `ANYPS5_VBLANK_HZ` | 30 to 360 | Vblank rate. A game that paces on vblank runs at the display's rate (about 60 FPS); raising it, for example to `240`, lifts that cap |
| `ANYPS5_FPS_LIMIT` | 0.5 or more, `0` for no limit | Most frames per second the game presents |
| `KYTY_PRESENT_MODE` | `Fifo`, `Mailbox`, `Immediate` | Vulkan present mode (`Mailbox` by default): `Fifo` waits for the display, `Immediate` may tear |
| `KYTY_ASYNC_PIPELINES` | `0`, `1` (default), `2` | What a draw does while its pipeline is still compiling: with `1` it waits up to 20 ms and is then skipped, so an object can be missing for a few frames the first time it shows; `2` skips more draws and waits for no new pipeline, for fewer stutters on a first run; `0` always waits, which avoids missing objects and stalls longer on a first run |
| `KYTY_NATIVE_RT` | unset (default), `0`, `1` | Unset: native ray tracing when the GPU supports it. `0`: ray tracing emulated in shaders, for a GPU or a driver that struggles with it. `1`: native, and a stop with a message when the GPU cannot |

Add-ons need no setting either: a title that asks for one is told it is installed. To refuse them, put an empty `anyps5-entitlements.ini` beside the converted executable ([Add-ons](USAGE.md#add-ons)).

Everything else, the speed switches, the ray tracing controls, the logs and the benchmark aids, is for developers and bug reports: see [AnyKyty settings for developers](../dev/ANYKYTY_SETTINGS.md). The host report that a bug report needs prints at start without any variable. [The full reference](../dev/THIRD_PARTY_KYTY.md#settings-of-the-kyty-build) lists every `KYTY_*` setting, one row each, with its default and its category.

## Contributing

AnyKyty is a young fork and there is a lot to do: speed, rendering differences, game compatibility, Windows, other GPUs. Contributions of every size are welcome.

**Send a pull request**

1. Fork [simnJS/AnyPS5](https://github.com/simnJS/AnyPS5) and branch from `AnyKyty`. Open the pull request against `AnyKyty`: the [contributing guide](../../CONTRIBUTING.md) and the pull request template are AnyPS5's and say `main`, which is the upstream branch.
2. The CI runs on every pull request: the conventions check, a Linux build of the default backend, a Linux build of the Kyty backend with the shim tests, a Linux build of the Kyty core under ThreadSanitizer and another under AddressSanitizer that run the shim tests needing no GPU, and a Windows build of the Kyty backend. The two sanitizer jobs also run every night. Changes to documentation only skip it. See the [workflow](../../.github/workflows/anykyty.yml).
3. Write commits and the pull request title as [Conventional Commits](https://www.conventionalcommits.org), for example `fix(kyty): ...`, `perf(kyty): ...` or `feat(kyty): ...`. The check rejects other subjects. You can run it before pushing: `python3 tools/check_conventions.py --base origin/AnyKyty`.
4. Follow the [coding conventions](../dev/CONVENTIONS.md): no comments in code, no images or notes in the repository (attach them to the pull request), and a row in [ENVIRONMENT](../dev/ENVIRONMENT.md) for every new `APS5_*` or `ANYPS5_*` variable.
5. Where the code goes: the Kyty core in `core/kyty/` is maintained in this repository without comments, so changes to it sit under `#if defined(ANYPS5_KYTY)`; the glue between it and AnyPS5 is in `core/libs/prx/libSceAgcDriver/KytyShim/`. The default backend must keep building unchanged.
6. Fill in the pull request template; the conventions check enforces it. For a performance change, give the numbers of the base and of your change from the same machine and scene, with the settings, as the [performance report](../../.github/ISSUE_TEMPLATE/performance.yml) form asks.

**Report a game**

Open an [issue](https://github.com/simnJS/AnyPS5/issues/new/choose) or start a [discussion](https://github.com/simnJS/AnyPS5/discussions), whether the game works or not: title ID and version, operating system, GPU and driver, the commit you built, the settings you used, and the log of the failure. [Game testing](../dev/GAME_TESTING.md) says what to capture. Reports from AMD and Intel GPUs, from Windows and from CPUs without a V-Cache are the ones we have the fewest of.

To compare steady-state timings, build both backends from the same revision and use the same machine, scene, resolution, settings and cache state. Give each run a new `APS5_VIDEOOUT_TELEMETRY_DIR` and set `APS5_VIDEOOUT_FLIP_TIMING=FIRST:COUNT`. Choose the first flip after reaching the scene and use the same count on both backends. Exit each run after its window is complete.

```sh
python tools/videoout_flip_compare.py default/videoout.jsonl kyty/videoout.jsonl
```

The comparison checks the reported build revision and every flip in each declared window. It reports mean completed-flip FPS and the median and nearest-rank p99 of the intervals between flips. `--json` prints the same metrics for other tools. The drop count covers all telemetry events, including pad and caption updates. The report measures completed VideoOut flips. Scene equivalence, build cleanliness, unique rendered frames and display timing need separate evidence.

For a baseline and candidate build, record the same first flip and count in separate runs. Pass each full revision from
`git rev-parse HEAD` in the corresponding build checkout. Use the same revision twice to compare two settings of one build.

```sh
python tools/videoout_flip_compare.py baseline/videoout.jsonl candidate/videoout.jsonl \
  --baseline-revision BASELINE_COMMIT --candidate-revision CANDIDATE_COMMIT
```

This mode checks each revision against its telemetry, rejects byte-identical inputs and reports the candidate-minus-baseline
differences. `--json` also includes the SHA-256 of each telemetry file.

To inspect a stutter, record completed flips and a `KYTY_SESSION_LOG` file from the same Kyty run. Use a new telemetry directory and log file, and set `KYTY_PIPELINE_STALL_LOG_MS` low enough to capture the events of interest. Exit after the flip window completes.

```sh
python tools/kyty_pipeline_stalls.py kyty/videoout.jsonl kyty/session.log --top 10
```

The report lists the longest completed-flip gaps by `process_us` and shows the number of recorded events and the largest timestamp overlap in each category. It uses the same-run nanosecond steady clock for the join. A dash marks gaps with no recorded overlap in that category. Background work can appear alongside a gap, and unfinished or below-threshold events can be absent. Use `--json` for shader hashes, wait routes, and Vulkan results.

**Open issues** where help is wanted ([all issues](https://github.com/simnJS/AnyPS5/issues)):

- [#98](https://github.com/simnJS/AnyPS5/issues/98) 007 First Light hangs the GPU on RADV (gfx ring timeout, `ErrorDeviceLost`);
- [#33](https://github.com/simnJS/AnyPS5/issues/33) shader and pipeline compile stalls on first play: precompile pipeline parts with `VK_EXT_graphics_pipeline_library` (labelled `help wanted`);
- [#32](https://github.com/simnJS/AnyPS5/issues/32) texture cache recreates and re-uploads aliased full-screen images every frame (007 First Light cutscene: 1 to 3 FPS before the image-heap texture tables, about 29 FPS on RADV now);
- [#30](https://github.com/simnJS/AnyPS5/issues/30) Planet Zoo draws most of its 3D scene wrongly or not at all (its start-up stop is fixed);
- [#24](https://github.com/simnJS/AnyPS5/issues/24) memory tracking takes about 20 % of `kytyGpuExecute` in METAL GEAR SOLID 4;
- [#21](https://github.com/simnJS/AnyPS5/issues/21) LEGO Builders Journey rewrites submitted command buffers while they are still queued.

Pull requests from outside contributors are already merged: [#17](https://github.com/simnJS/AnyPS5/pull/17), [#18](https://github.com/simnJS/AnyPS5/pull/18), [#19](https://github.com/simnJS/AnyPS5/pull/19), [#20](https://github.com/simnJS/AnyPS5/pull/20), [#22](https://github.com/simnJS/AnyPS5/pull/22), [#23](https://github.com/simnJS/AnyPS5/pull/23), [#25](https://github.com/simnJS/AnyPS5/pull/25), [#26](https://github.com/simnJS/AnyPS5/pull/26), [#28](https://github.com/simnJS/AnyPS5/pull/28), [#29](https://github.com/simnJS/AnyPS5/pull/29), [#34](https://github.com/simnJS/AnyPS5/pull/34), [#35](https://github.com/simnJS/AnyPS5/pull/35) and [#36](https://github.com/simnJS/AnyPS5/pull/36).

## Credits and licence

The graphics core: `core/kyty` is maintained in this repository; it derives from [KytyPS5](https://github.com/KytyPS5/KytyPS5) (GPL-2.0), itself built on Kyty by InoriRus. Everything else is [AnyPS5](https://github.com/boykopovar/AnyPS5).

AnyPS5 and the Kyty core are distributed under the GNU General Public License version 2 ([LICENSE](../../LICENSE), [core/kyty/LICENSE](../../core/kyty/LICENSE)); the original Kyty code is MIT ([core/kyty/LICENSES/Kyty-MIT.txt](../../core/kyty/LICENSES/Kyty-MIT.txt)). File by file origin and licences: [Kyty code in AnyPS5](../dev/THIRD_PARTY_KYTY.md).

Back to the [README](../../README.md).
