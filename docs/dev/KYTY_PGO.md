# Profile-guided optimization of the Kyty core

The Kyty core (`kyty_gpu`, built by Clang, see [Kyty code in AnyPS5](THIRD_PARTY_KYTY.md#build)) can be compiled with an
LLVM instrumentation profile (`-fprofile-use`): Clang then lays out, inlines and promotes indirect calls after the branches and
call targets the profile counted while titles ran. The profile is an input of the build like any source file, passed by one
option, so the same step applies whatever builds the core.

## Building with the profile

| Option | Meaning |
|---|---|
| `ANYPS5_KYTY_PGO` | Profile that optimizes `kyty_gpu`: text (`.proftext`) or indexed (`.profdata`), absolute or relative to the source tree. Empty for none. |
| `ANYPS5_KYTY_PGO_GENERATE` | `ON` builds an instrumented `kyty_gpu` that records a profile (Linux only). Excludes `ANYPS5_KYTY_PGO`. |

The profile covers `kyty_gpu` and the libraries pre-linked with it (fmt, xxHash, Tracy, ZArchive, zstd), not SPIRV-Tools, whose
static libraries also build `glslangValidator` during the build. A text profile is converted at configure time by the
`llvm-profdata` of the compiler's LLVM major (`llvm-profdata-<major>`, then `llvm-profdata` next to the compiler or on the path), into `build/core/kyty/pgo/kyty_gpu.profdata`, so the indexed
format is always one the compiler reads. An indexed profile is used as is and must come from the same LLVM major. When the file
is missing or no matching `llvm-profdata` is found, the configure step warns and the core is built without the profile: a missing
profile costs speed, never the build. Objects depend on the profile, so changing it rebuilds `kyty_gpu`.

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
them, and exits with 1, for CI or before a release. Record the profile again when the warning names files on the paths that run
every frame (command processor, renderer), or before a release.

## Recording the profile

1. Configure a separate build directory with `-DANYPS5_KYTY_PGO_GENERATE=ON` (and `-DANYPS5_GPU_BACKEND=kyty`), and build `libs`
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

`KYTY_PGO_DUMP_SECONDS` is read only by the instrumented build (`core/kyty/pgo/profileDump.cpp`, compiled into `kyty_gpu` only with
`ANYPS5_KYTY_PGO_GENERATE`).
