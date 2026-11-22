# Profile-guided optimization of the Kyty core

The Kyty core (`kyty_gpu`, built by Clang, see [Kyty code in AnyPS5](THIRD_PARTY_KYTY.md#build)) is compiled with an
LLVM instrumentation profile (`-fprofile-use`): Clang then lays out, inlines and promotes indirect calls after the branches and
call targets the profile counted while titles ran. The profile is an input of the build like any source file, passed by one
option, so the same step applies whatever builds the core.

The repository keeps one, `core/kyty/pgo/kyty_gpu.proftext`, and the kyty build uses it by default. Its first line says what
recorded it. Measured on Linux against the same commit without a profile, one pass per arm, the same scene and the same machine
for both arms of a pair, with the profile recorded from two frame replays and a minute in game of Stray, Subnautica and an
open-world AAA title in both of its graphics modes: Stray runs at 84.25 frames per second instead of 78.04 (+7.96 %), and the
busiest thread of the driver, the execution thread, spends 14.5 % less CPU per draw; the AAA title in its high fidelity mode runs at
52.83 instead of 49.03 (+7.73 %), 5.7 % less CPU per draw on that thread.

## Building with the profile

| Option | Meaning |
|---|---|
| `ANYPS5_KYTY_PGO` | Profile that optimizes `kyty_gpu`: text (`.proftext`) or indexed (`.profdata`), absolute or relative to the source tree. `core/kyty/pgo/kyty_gpu.proftext` by default; empty for none. |
| `ANYPS5_KYTY_PGO_GENERATE` | `ON` builds an instrumented `kyty_gpu` that records a profile, and reads none (Linux only). |

The profile covers `kyty_gpu` and the libraries pre-linked with it (fmt, xxHash, Tracy, ZArchive, zstd), not SPIRV-Tools, whose
static libraries also build `glslangValidator` during the build. A text profile is converted at configure time by the
`llvm-profdata` installed beside the compiler (`clang++ -print-prog-name=llvm-profdata`), which must be of the same LLVM major,
so the indexed format is always one the compiler reads; an indexed profile must come from the same LLVM major. Either way the
indexed profile the compiler reads is `build/core/kyty/pgo/kyty_gpu-<hash>.profdata`, named after its content: a new profile
changes the compile command, so every object of the core is compiled again and ccache cannot return one compiled with the
previous profile (ThinLTO refuses to link modules whose profile summaries differ). When the file is missing or no matching
`llvm-profdata` is found, the configure step warns and the core is built without the profile: a missing profile costs speed,
never the build.

Static and anonymous-namespace functions are named in the profile by their file name, without its directory
(`-mllvm -static-func-full-module-prefix=false`, on both the instrumented and the optimized build), so a profile recorded in one
checkout applies in any other, on Linux or Windows. Functions compiled differently on Windows (`_WIN32` paths) do not match their
Linux counts and are compiled as if they had no profile.

## When the profile is out of date

Each function's counts carry a hash of its control flow. A function whose code changed since the profile was recorded no longer
matches it: Clang drops its counts with a `-Wbackend-plugin` warning ("function control flow change detected"), or a
`-Wprofile-instr-out-of-date` summary per file, and compiles it without the profile. Functions that did not change keep their
counts. An old profile therefore loses its effect gradually; it never breaks the build.

The text profile starts with comment lines that record what trained it: `# kyty-pgo-trained <commit> <date> <note>`, then one
`# kyty-pgo-source <sha1> <path>` per source of the core (`core/kyty/graphics`, `core/kyty/common`, and the two memory files of the
shim built with it), the SHA-1 of the file with carriage returns removed. The configure step compares them with the sources it
builds and prints a CMake warning that counts the changed and new files when they differ. `python3 tools/kyty_pgo.py check` lists
them, and exits with 1.

A changed file only costs speed where a function's control flow changed, and the compiler's warnings are the only place that says
which functions. `python3 tools/kyty_pgo.py report <build log>` reads them: it lists every function that lost its counts, its
weight (the highest counter of the function in the profile) and its rank among the functions the profile counts, as Markdown. The
Linux Kyty job of CI runs it on its build log and writes the table into the job summary. It adds a warning annotation to the run,
never a failure (contributors have no title to record a profile with), when one of the 10 hottest functions lost its counts or when
the functions that did hold 1 % of the profile's weight (`--top`, `--share`). A warning means the profile is to be recorded again
by whoever has the titles, as below, after the series of changes that caused it.

## Recording the profile

1. Configure a separate build directory with `-DANYPS5_KYTY_PGO_GENERATE=ON`, and build `libs`
   and `agc_frame_replay`.
2. Run what the profile should represent with `LLVM_PROFILE_FILE=<directory>/kyty-%p.profraw` in the environment, one file per
   process:
   - frame captures with `agc_frame_replay <capture> --repeat <n>` (see [BUILD](BUILD.md)); the replay writes its profile when it
     exits;
   - titles, played in game. A title does not run the exit handlers of the process when it quits, so set
     `KYTY_PGO_DUMP_SECONDS=10` as well: the instrumented build then appends the counts of the last 10 seconds to the file every
     10 seconds and starts counting again, so the merged profile holds the whole run but its last seconds. Stop the title at least
     that long after the scene to record.
3. Merge: `python3 tools/kyty_pgo.py merge --note "<what ran>" <directory>...` writes `core/kyty/pgo/kyty_gpu.proftext`, a sparse
   text profile (functions that never ran are left out and keep the default optimization).

## Recording again after a series of changes to the hot path

The hot functions are those of the command processor, the resource tracking and the pipeline cache (`ResolveGraphicsPrograms`,
`FindBuffers`, `Tracker::Run`): a change to one of them drops its counts whatever the rest does. Run this after each series of
changes to them, on the commit that carries the series:

1. Build the instrumented directory (`-DANYPS5_KYTY_PGO_GENERATE=ON`) and run the training set, each with its own
   `LLVM_PROFILE_FILE=<directory>/<name>/kyty-%p.profraw` and `KYTY_PGO_DUMP_SECONDS=10`: the frame replays (`sub-creative` four
   passes, `sub-long-15` two), then one minute in game of Stray, Subnautica and an open-world AAA title in each of its two graphics modes.
2. `python3 tools/kyty_pgo.py merge --note "<what ran>" <directory>`: on a tree without uncommitted changes to the Kyty sources, so
   that the header names the commit.
3. Build the default directory again, with the build output kept (`cmake --build <build> ... 2>&1 | tee build.log`), then
   `python3 tools/kyty_pgo.py report build.log` must list no function and `python3 tools/kyty_pgo.py check` must exit 0.
4. Commit the profile alone, with the commit it was recorded on in the message.

`KYTY_PGO_DUMP_SECONDS` is read only by the instrumented build (`core/kyty/pgo/profileDump.cpp`, compiled into `kyty_gpu` only with
`ANYPS5_KYTY_PGO_GENERATE`).
