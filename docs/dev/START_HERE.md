# Start here: contributing to AnyKyty

AnyKyty is this fork's default branch: AnyPS5 running PS5 titles with the Kyty graphics core. This page is what a new contributor needs: how to build, run, test and measure, the rules, and how a pull request gets in. Open tasks are listed in the pinned "Start here" issue, by label (see [Find work](#find-work)).

## Contents

- [Where things are](#where-things-are)
- [Build](#build)
- [Run a title](#run-a-title)
- [Run the tests](#run-the-tests)
- [Measure](#measure)
- [Rules](#rules)
- [How a pull request is integrated](#how-a-pull-request-is-integrated)
- [Find work](#find-work)
- [What CONTRIBUTING.md says that does not apply here](#what-contributingmd-says-that-does-not-apply-here)

## Where things are

| Path | What it is |
|---|---|
| `core/kyty/` | The Kyty graphics core: command processor, the Resolve, Execute and Record stages, caches, shader recompiler (`graphics/shader/recompiler`), ray tracing. Maintained here, derived from KytyPS5 (GPL-2.0) |
| `core/libs/prx/libSceAgcDriver/KytyShim/` | The glue between Kyty and AnyPS5: memory, faults, entry points, presentation, settings (`src/Config.cpp`), and most tests (`tests/`) |
| `core/libs/prx/` | AnyPS5's reimplemented system libraries (`libc`, `libkernel`, `libSceAgc`, `libSceVideoOut`…) |
| `core/relinker/` | The relinker, which converts a PS5 executable into a native Linux or Windows program |
| `tools/` | Checks, measurement and analysis scripts |
| `docs/dev/`, `docs/user/` | Documentation; [ANYKYTY.md](../user/ANYKYTY.md) is the user page, [ANYKYTY_SETTINGS.md](ANYKYTY_SETTINGS.md) the settings for developers, [THIRD_PARTY_KYTY.md](THIRD_PARTY_KYTY.md) the full reference of the Kyty core and its settings |

A frame goes through the stages in order, one thread each: the command processor decodes the title's command buffers, Resolve prepares resources ahead, Execute records each draw and dispatch (the CPU limit on GPU-driven titles), Record submits to Vulkan.

## Build

There is one build: the Kyty core is compiled by Clang in a sub-project, the rest of AnyPS5 by GCC on Linux or MinGW-w64 GCC on Windows. The commands below are those of the [AnyKyty workflow](../../.github/workflows/anykyty.yml). On the 4-core CI runner a first build takes about 20 minutes; with `ccache` the next ones take a few.

### Linux

Git, CMake 3.21 or newer, Ninja, GCC 14 and LLVM 21 (`clang`, `clang++`, `ld.lld` and `llvm-profdata` of the same major version, on `PATH`). The package list for Ubuntu is in [the AnyKyty page](../user/ANYKYTY.md#linux); on Ubuntu 24.04 LLVM 21 comes from [apt.llvm.org](https://apt.llvm.org).

```sh
git clone --recurse-submodules https://github.com/simnJS/AnyPS5.git
cd AnyPS5
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DANYPS5_KYTY_JOBS="$(nproc)"
cmake --build build --target kyty_gpu libs relinker --parallel "$(nproc)"
```

`-DANYPS5_KYTY_THIN_LTO=OFF` skips ThinLTO and the pre-link of the Kyty core, which makes builds of the core faster while you iterate.

### Windows

[WinLibs](https://github.com/brechtsanders/winlibs_mingw/releases/tag/15.2.0posix-14.0.0-ucrt-r7) GCC 15.2.0 (UCRT), [LLVM 21.1.8](https://github.com/llvm/llvm-project/releases/tag/llvmorg-21.1.8) (`clang`, `clang++`, `llvm-ar`, `llvm-ranlib`, `llvm-profdata` and its `lib/clang` directory), Git for Windows with Bash, Python 3, CMake and Ninja. Put the WinLibs and LLVM `bin` directories first on `PATH`, then from `cmd.exe`:

```bat
git clone --recurse-submodules https://github.com/simnJS/AnyPS5.git
cd AnyPS5
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DANYPS5_KYTY_JOBS=%NUMBER_OF_PROCESSORS%
cmake --build build --target kyty_gpu libs relinker --parallel %NUMBER_OF_PROCESSORS%
```

The toolchain's runtime DLLs go beside the libraries ([Windows setup and troubleshooting](BUILD.md#windows-setup-and-troubleshooting)).

## Run a title

Bring your own legally dumped PS5 title; games are never provided. PS4 titles (GNM) do not run: the Kyty driver stops at start with `PS4 (GNM) titles are not supported by the Kyty GPU driver`. The relinker takes the extracted, decrypted executable of the title and its bundled modules ([where to put the files](../user/USAGE.md#where-to-put-the-files)). Convert it, put the libraries (`build/core/libs/libs/*.prx`) in a `libs/` folder beside the output, and start it:

```sh
build/core/relinker/relinker source/input.elf app.elf
./app.elf
```

On Windows: `relinker --windows source/input.elf app.exe`, plus `--to-intel` on an Intel CPU. Module folders, add-ons and the other options are in [usage](../user/USAGE.md). No setting is needed: the defaults are the fastest and most stable choices measured. The host report (OS, CPU, GPU, driver) prints at start; `anyps5-report.json` is written beside the game when it exits.

## Run the tests

Build the shim test executables and run them as the CI does (the exact list is the "Run the shim tests" step of the [AnyKyty workflow](../../.github/workflows/anykyty.yml)):

```sh
mapfile -t tests < <(ninja -C build -t targets all | grep -oE '^(kyty_shim_[A-Za-z0-9_]+_tests|guest_gpu_fault_tests|guest_host_write_filter_tests|agc_kyty_shaders_tests):' | tr -d ':' | sort -u)
cmake --build build --target "${tests[@]}" --parallel "$(nproc)"
sudo sysctl -w vm.unprivileged_userfaultfd=1 vm.max_map_count=2147483642
ctest --test-dir build -R '^(kyty_shim_.*|guest_gpu_fault|guest_host_write_filter|agc_kyty_shaders)$' --output-on-failure --timeout 300
```

The GPU tests need a Vulkan GPU; without one they report a skip. For a change to the shader recompiler or the renderer, also run the tests you touched with `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` on the base and on your change, and check that no new `VUID-` error appears.

Before pushing:

```sh
python3 tools/check_conventions.py --base origin/AnyKyty
python3 tools/check_kyty_settings.py
```

The first checks commit subjects, comments in code, file types and paths, links in Markdown; the second checks that every `KYTY_*` setting the code reads has its row in [THIRD_PARTY_KYTY.md](THIRD_PARTY_KYTY.md#settings-of-the-kyty-build) and its name in the list of `KytyShim/src/Settings.cpp`. A new test goes in `KytyShim/tests/` with its registration in that folder's `CMakeLists.txt`, and finishes in 30 s.

## Measure

Numbers are taken with the default settings, warm caches (play the scene once, then restart the title) and the same scene on both sides. Say which settings you changed in any number you publish.

**A change that can be switched while the title runs: an in-run A/B.** `KYTY_AB` alternates two arms in one session, so both play the same scene:

1. Put your change behind a temporary switch: an entry in `Switch` (`core/kyty/common/runtimeSwitches.h`) and in `Entries` (`runtimeSwitches.cpp`), read where it is used with `Common::RuntimeSwitches::On(Switch::YourSwitch)`, and its row in [THIRD_PARTY_KYTY.md](THIRD_PARTY_KYTY.md#settings-of-the-kyty-build). A switch can flip only if work done under one value stays right under the other ([details](THIRD_PARTY_KYTY.md#details)).
2. Run the scene: `KYTY_AB=40:KYTY_YOUR_SWITCH KYTY_AB_SETTLE=8 ./app.elf > session.log 2>&1`, for at least five phases of each arm.
3. `python3 tools/kyty_ab_report.py session.log` prints each arm's frame rate, the difference corrected for drift, and the `[kyty-*]` counters. Paste it into the pull request.
4. Once the numbers decide, the switch goes: the winning behaviour stays without a setting. AnyKyty keeps no setting that is off by default.

**A change that cannot be switched while running** (shader compile forms, the command processor thread, thread placement): two sessions, same scene, with the flip telemetry. Give each run a new `APS5_VIDEOOUT_TELEMETRY_DIR` and the same `APS5_VIDEOOUT_FLIP_TIMING=FIRST:COUNT`, then `python3 tools/videoout_flip_compare.py base/videoout.jsonl change/videoout.jsonl --baseline-revision <sha> --candidate-revision <sha>` ([more](../user/ANYKYTY.md#contributing)).

**Stutters**: `KYTY_SESSION_LOG=<file>` with the flip telemetry, then `python3 tools/kyty_pipeline_stalls.py videoout.jsonl <file> --top 10`.

You do not need every game: the maintainers run one title per engine family (Unity, Unreal Engine 4, an in-house engine designed for the PS5, an in-house engine of the PS4 era) on each integration.

## Rules

- **One topic per pull request**, against `AnyKyty`. Follow-ups go in a new pull request.
- **Conventional Commits** for commit subjects and the pull request title: `fix(kyty): …`, `perf(kyty): …`, `feat(kyty): …`, `test(kyty): …`, `docs: …`, `ci: …`.
- **No comments in code**, in `core/kyty` as elsewhere, except the licence notice of code taken from another project. Explanations go in the pull request.
- **Nothing specific to one title**: no title IDs, shader hashes, per-title branches or data taken from a game in the code or the tests. Fix the class of problem. Naming the title you tested in the pull request text is fine.
- **Defaults are the fastest and most stable measured.** A new behaviour lands on by default once measured; a temporary switch lives only as long as its A/B.
- **Every setting is documented**: a `KYTY_*` name in [THIRD_PARTY_KYTY.md](THIRD_PARTY_KYTY.md#settings-of-the-kyty-build), an `APS5_*` or `ANYPS5_*` name in [ENVIRONMENT](ENVIRONMENT.md).
- **No notes, logs or images in the repository**: attach them to the pull request. A new Markdown file goes under `docs/dev/` or `docs/user/`.
- **AI-assisted work is accepted.** Say so on the template's `AI-assisted` line; you answer for every line.
- **Shader semantics**: say where a behaviour comes from (a hardware measurement, the ISA document, Mesa or LLVM).
- Keep changes to code shared with upstream AnyPS5 (outside `core/kyty` and `KytyShim`) small: they conflict at every synchronisation.
- Code taken from KytyPS5 keeps its licence header; AnyKyty is GPL-2.0 ([licences](THIRD_PARTY_KYTY.md)).

The [coding conventions](CONVENTIONS.md) give the naming and formatting rules.

## How a pull request is integrated

1. **Open it against `AnyKyty`** and fill in the template. It is AnyPS5's and says `main`: read `AnyKyty`.
2. **The CI runs** on every push ([workflow](../../.github/workflows/anykyty.yml)): the conventions and settings checks, a Linux build with the shim tests, the CPU tests under ThreadSanitizer and AddressSanitizer, and a Windows build with the shim tests. It must be green. A change to documentation only skips the builds.
3. **A maintainer reviews it**: the diff, both checks, the builds and tests on Linux. A change on a path every title uses (presentation, memory, faults) may get a session on a title.
4. **Larger changes go through a `test/<name>` branch** first, sometimes grouped with others: the maintainers run one title per engine family, and testers with AMD, Intel or Windows are asked to run theirs.
5. **The maintainer pushes your commits to `AnyKyty`**, with you as the author, and closes the pull request with "merged as `<sha>`". GitHub shows it as closed, not merged: that is expected.

## Find work

- The pinned "Start here" issue: the roadmap and the open tasks.
- By label: [`good first issue`](https://github.com/simnJS/AnyPS5/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22), [`medium`](https://github.com/simnJS/AnyPS5/issues?q=is%3Aissue+is%3Aopen+label%3Amedium), [`hard`](https://github.com/simnJS/AnyPS5/issues?q=is%3Aissue+is%3Aopen+label%3Ahard); the `area: …` labels (shaders, gpu-memory, execute, rendering, ray-tracing, windows, tools-ci, docs) and the `needs: …` labels (AMD, Windows, game, measurement).
- Architecture v2, the main performance work: [#168](https://github.com/simnJS/AnyPS5/issues/168). Steps are claimed there.
- Without code: test a title ([#141](https://github.com/simnJS/AnyPS5/issues/141)), or survey its ray tracing ([#111](https://github.com/simnJS/AnyPS5/issues/111)).

Comment on an issue before you start, so two people don't do the same work. Questions go to [Discussions](https://github.com/simnJS/AnyPS5/discussions/categories/q-a).

## What CONTRIBUTING.md says that does not apply here

[CONTRIBUTING.md](../../CONTRIBUTING.md) is upstream AnyPS5's. Its rules for code, commits and documentation apply. Its instructions for AnyPS5's own GPU driver and shader recompiler (`core/shader`) do not, since both were removed from `AnyKyty`; the validation-layer check they describe still applies to the Kyty recompiler and renderer, as above. Its description of the pull request process does not apply either, because it is about upstream's `main`: the limit on open pull requests, the review queue, `stale` and `already in main` labels, the manually started `Build` workflow, `tools/pr_overlap.py`, the trace-based performance report form and the Game bring-up discussions of the upstream repository.
