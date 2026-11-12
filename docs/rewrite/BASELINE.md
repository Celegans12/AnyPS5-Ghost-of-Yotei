# Baseline: upstream versus rewrite/direct (2026-11-12)

Same machine (RTX 5080, Ryzen 7 9800X3D, Windows 11, 3840x2160 window), same bench script (`scripts\bench.ps1`), one discarded warm-up run then one measured run per build and title, shader cache per build.

- **Baseline** = upstream `1b485ff745` (the merge base) plus the unmerged game fixes: #15327 (AoM indirect record hang), #15611 (table entries filled with ones), local clock id 20 EINVAL carry (AoM needs it). Worktree `wt-upstream`, libs `libs\baseline`. No Divinity #16025, no performance PRs (#15659, #17225).
- **Current** = `rewrite/direct` at `d66234ab8c` (direct path on by default, plus the sparse-settle fix), libs `libs\current`, worktree `wt-rewrite`. `APS5_NO_DIRECT=1` restores the existing path. `libs\pre-sparse` is the build at `0e2a022c37` (without sparse-settle).

## Frame time (last 3 intervals of the measured run, median p50)

| Title | Scene | Baseline | Current | Change |
|---|---|---|---|---|
| LEGO Builder's Journey | first level, idle (110 s) | 56.0 ms, 165 presents per 10 s | 47.6 ms, 210 per 10 s | about 15 % faster |
| FF Resonance | burning town (100 s) | 33.4 ms, 298 per 10 s | 32.5 to 33.5 ms, 298 to 308 per 10 s | within noise |
| AoM: Retold, title screen | intervals 3 to 7 of a 135 s window | about 1,190 ms, 9 per 10 s | 206 to 218 ms, 46 to 47 per 10 s | about 5.5x faster |
| AoM: Retold, scene after pad press | intervals 8 to 12 | not reached | 149 to 175 ms, 54 to 67 per 10 s | n/a |
| 007 First Light, startup | 300 s, no gameplay scene | 1,630 to 1,670 ms, 5 to 7 per 10 s; stops presenting at about 40 s (228 presents) on the SAVING screen | 408 to 480 ms, 21 to 25 per 10 s; reaches the title screen ("Press X to play") | about 3.5x faster, and the baseline stalls |

007 is startup behaviour only (shader compiles, loading screens, title screen), not gameplay performance. Without the sparse-settle fix (`libs\pre-sparse`) 007 ran at 1,420 to 1,530 ms.

## Peak memory (working set / private commit, GiB)

| Title | Baseline | Current |
|---|---|---|
| LEGO | 4.8 / 10.4 | 5.9 / 11.3 |
| FF Resonance | 4.3 / 6.7 | 4.5 / 7.2 |
| AoM | 16.8 / 18.8 | 18.9 / 21.8 |
| 007 | 15.5 / 26.3 | 17.4 / 28.9 |

AoM and 007 are over the 16 GiB footprint target on both builds; the current build adds about 2 GiB to each.

## Where each game stands (current build)

| Title | State | Current blocker | Roadmap position (DESIGN.md phases) |
|---|---|---|---|
| LEGO | Plays, idle first level at 47.6 ms (GPU track) | GPU time, not CPU. 96 guest-context exceptions (signal 0x1e) in every build, so not a regression; unsupported shader type 8 (18 shaders) | Phase 0 done; direct path (phase 4 draw side) on by default; phase 5 GPU stage not started |
| FF Resonance | Plays, 33 ms (CPU-bound, regression title) | Worker CPU per frame (capture, bindings) and coherence | Direct draws done; phase 6 per-draw CPU and phase 2 coherence are the levers; no change from the rewrite yet |
| AoM | Title and menu work. Title is GPU-bound at about 215 ms; scene after pad press about 150 to 175 ms | Pixel shader GPU time from host-import buffer reads (4 programs); 86 "deferred write-back failed: invalid format" image-table warnings; 18.9 GiB working set | Phase 4 section B (device-local copies of clean buffer ranges) is the next step; phase 1 relinker audit in parallel |
| 007 | Reaches the title screen, about 2 fps | Startup CPU/shader cost; no repeatable gameplay scene yet; 17.4 GiB working set, 28.9 GiB commit | Not a focus title; needs a gameplay scene before phase work counts |

## Caveats

- One measured run per build and title. Differences under about 3 ms (LEGO, FF) are noise.
- The AoM baseline has no table performance work, so its number says what direct tables buy, not what a tuned existing path would do.
- `APS5_NO_DIRECT=1` is not identical to upstream: the compute push-constant rule in `BindingAllocator.cpp` and the stencil copy code stay on.
- Runs: `runs\<title>\cmp-<baseline|current|sparse>-<warm|m1>` (the `sparse` runs are the current numbers above); logs `logs\cmp-*.txt`.
