# Tracing

Set a switch before starting the emulator. Unless noted, any value enables it. Most output goes to the emulator log or standard error. Performance captures are written to files as described in [Build](BUILD.md#performance-trace). Traces can be large and can affect timing.

## Kernel and libraries

| Switch | Output |
| --- | --- |
| `APS5_TRACE_AFFINITY` | Thread role, host thread ID, requested CPU mask and whether pinning succeeded. |
| `APS5_TRACE_AJM` | Submitted audio jobs, decoded output and unimplemented AJM calls. |
| `APS5_TRACE_APR` | APR file reads, including path, offset, size and destination. |
| `APS5_TRACE_AUDIOOUT` | Audio output port parameters when a port is opened. |
| `APS5_TRACE_AUDIOOUT2` | AudioOut2 context and port parameters, attributes, pushes, queue polls and periodic queue summaries. |
| `APS5_TRACE_CALLS` | On Windows, Linux and macOS, records calls to the imports selected by this comma-separated library list (or `all`); crash and abort reports include the last 64 calls and most-called imports. The executable must be converted with `--trace-imports`; see [Usage](../user/USAGE.md#tracing-system-library-calls). |
| `APS5_TRACE_CALLS_FILE` | With `APS5_TRACE_CALLS`, a file rewritten every 10 seconds with every traced import called so far, most called first: number of calls, library and import name. |
| `APS5_TRACE_EXIT` | Guest `exit` and `_Exit` calls and process-exit backtraces. |
| `APS5_TRACE_FIBER` | Fiber initialization, switches and returns. |
| `APS5_TRACE_MEMORY` | Direct-memory allocation, mapping and protection operations. |
| `APS5_TRACE_SSE4A` | Each emulated SSE4A instruction once per guest instruction address (Windows). |
| `APS5_TRACE_SYNC` | Event-flag and semaphore operations and their wait and wake results. |
| `APS5_TRACE_TIMERS` | Timing measurements for the host wait primitives used by the emulator. |
| `APS5_TRACE_UDS` | Universal Data System events posted to a context, and the trophy title data it cannot find. |
| `APS5_TRACE_USLEEP` | Every 2,000 `sceKernelUsleep` calls, the call sites that sleep most, with their call counts and requested delays. Calls made through `sceNetUsleep` and by the `APS5_TRACE_TIMERS` measurement on Windows are counted too, under a libSceNet or libkernel address. Guest `usleep` calls are not counted. |
| `APS5_TRACE_VIDEODEC` | Video decoder API calls and decode results. |
| `APS5_TRACE_WAITS` | Blocking waits aggregated by guest call site, including timeout counts and longest waits, every three seconds. |

To inspect a guest address in a saved `APS5_TRACE_MEMORY` log, run
`python3 tools/memory_trace_address.py <log> --address 0x... [--length <bytes>]`. It lists recorded mapping,
remapping, protection, reservation and unmapping events whose ranges overlap the query, in log order. These are trace
records, not confirmation that each attempted operation succeeded; no matching event does not prove the address was
unmapped.

## GPU and memory

| Switch | Output |
| --- | --- |
| `APS5_TRACE_LABEL` | A matching `RELEASE_MEM` when recorded, and nearby `RELEASE_MEM`, `WRITE_DATA`, `COPY_DATA`, `DMA_DATA` and `WAIT_REG_MEM` packets when executed. Set the value to the guest address in hexadecimal. |

## Performance captures

See [Build](BUILD.md#performance-trace) for the trace format, summary fields and capture limits.

| Switch | Output |
| --- | --- |
| `APS5_TRACE_FRAMES=<start>:<count>` | Records a window of presents by zero-based frame number. |
| `APS5_TRACE_SECONDS=<start>:<length>` | Records a window of presents by elapsed seconds after the first present. Cannot be set with `APS5_TRACE_FRAMES`. |
| `APS5_TRACE_SUMMARY_EVERY=<seconds>` | Writes periodic summary lines without recording trace events. Can be combined with either window switch. |
| `KYTY_AB=<seconds>:<setting>[+<setting>...]` | The same with the Kyty backend, for the settings it can change at a flip; prints `[kyty-ab]` lines, and `python3 tools/kyty_ab_report.py <session>` gives each arm. See [the Kyty settings](THIRD_PARTY_KYTY.md#details). |
| `APS5_VIDEOOUT_TELEMETRY_DIR=<path>` | Writes the presents and the pad changes of a run to `<path>/videoout.jsonl`; see [Usage](../user/USAGE.md#video-out-and-pad-telemetry). |

## Shader registers

`APS5_TRACE_SHADER_REGS=all` prints register snapshots for every shader. Set it to a shader code address in hexadecimal to print one shader's registers.

## Running titles repeatably

`python tools/title_run.py --games <directory> [ID ...]` relinks and runs extracted title packages found under `--games` (a directory holding `eboot.bin` and `sce_sys/param.json`) and writes `summary.json` and a fixed-format `summary.md` under `--out` (default `build/title-runs/<commit>-<time>-<unique>/`). Each title runs `--runs` times (default 2): the first with empty AnyPS5 and Mesa or NVIDIA shader caches, the next ones keeping them. AMD's Windows `VkCache` is not isolated. On Windows the title's windows open behind every other window, without focus and without being minimized, and can be brought forward from the taskbar; elsewhere run it under a virtual display if that matters. A run ends as `relink-failed`, `crashed`, `exited`, `stalled` (no present in the last 20 s, or too few presents for a frame to be dumped), `frame-unreadable` (the last dumped frame is truncated or not a 24- or 32-bit BMP), `blank` (the last frame is one colour) or `rendering`; the first, middle and last dumped frames are kept. A screen that presents only when its view changes stops after a few presents and is reported `stalled` although the title still answers input; pressing a button through `tools/pad_input.py` with `APS5_PAD_INPUT_FILE` set tells it apart from a stuck title. `--compare <summary.json>` lists regressions, improvements and missing titles against an earlier run, limited to the IDs given when there are any. Titles are prepared in `<out>/work/<ID>/`, the same path on every run, so a firewall prompt for a title that opens a port appears once; answering Cancel does not affect the run. An invocation locks `<out>/work/<ID>.lock` for each title it runs before touching anything, and refuses to start while another invocation with the same `--out` holds one of them. Per-title relink flags, run length and environment live in `tools/titles.json`. It also compares the last run of each title that both summaries hold by numbers: the FPS of the trace summary, the `averageFps` and `frameMs` percentiles (p50, p90 and p99) of `anyps5-report.json`, the `compiled`, `failed` and `cacheHits` shader counts, and the functions in `missing.functions` and the instructions in `shaders.unsupported`, each split into only before, only now and in both. A value the report does not hold reads `not recorded` on that side; the lists are partial when `truncated` is above zero. `--json` prints that comparison as JSON instead of the table. `--smoke` runs the titles marked `"smoke": true` in `tools/titles.json` instead of named IDs: a start-up check for changes to the relinker, the module loader or libc start-up, which can break every title at second 0 (#7665). The set is DriftCheck (PPSA98766, the smallest import table), DOOM (PPSA99666), Jelly5 (PPSA99505), SymphonyStation5 (PPSA17641), PS5SX2 (PPSA99203, `--trap-sdk-syscall-stubs` and the most libc imports) and VLC (PPSA85300, `--registry`). A `--module-dir` directory among a title's relink flags is staged for the relinker like `sce_module/`, so a Unity title lists `--module-dir Media/Modules --module-dir Media/Plugins` as [GAME_TESTING.md](GAME_TESTING.md) describes. A title's `pad` entry holds a controller timeline in the [`tools/pad_timeline.py` format](../user/INPUT_MAPPING.md#controller-timelines) (`events`, and optionally `repeat` and `gap_ms`); each run sets `APS5_PAD_INPUT_FILE` to `pad-input.txt` in the work directory and queues the timeline `after_s` seconds after start (default 5), since commands already in the file when the title first reads its controller are ignored. Each run reports `pad_queued_s`, or null when the title ended first. The timeline follows wall time, not frames, so the title's path differs between runs.

Keep `--out` outside extracted packages and inputs, including configured `--content` source files, outside `<out>/work/`; the runner checks for overlapping paths and aliases before clearing work. Title selection does not exclude other discovered packages from these checks. Module directories must be relative paths inside the package (`.` is allowed). On Linux, `/proc/self/fdinfo/` must be readable, and work directories cannot cross mount boundaries.

A title's `files` in `tools/titles.json` adds files to its prepared folders before each run. Each key is a path under `app0/`, `data/` or `download0/`, and each value is either `{"text": ...}`, written as given, or `{"content": <path>}`, copied from the directory passed with `--content`. Content is data that cannot be committed, such as a game's own data files. A title whose content is missing still runs, and `summary.md` lists it under `Ran without content`. vkQuake (PPSA99010) reads `quake/pak0.pak`, which is `id1/pak0.pak` from id Software's shareware `quake106.zip` (MD5 `5906e5998fc3d896ddaf5e6a62e03abb`). Its entry also turns off ray-traced shadows: their first compute pipelines take longer to build than the title waits for the GPU (#6118). Afterglow (PPSA47770) runs with `APS5_ASYNC_PIPELINE_WORDS=1000000`: the pipeline of a 95,468-byte pixel shader, about 2.5 million SPIR-V words, takes longer to build on the draw than the title waits for the submission to retire, and the title then exits with `submission retirement failed` on every start. SudokuZen (PPSA50001) and Surround Sound Studio (PPSA99051) run with `APS5_ASYNC_PIPELINE_WORDS=100000` for the same reason: on the cold run the pipeline of their 21,652-byte pixel shader, about 337,000 SPIR-V words, is built on the draw, and both exit with `submission retirement failed` before their first present. VLC (PPSA85300) reads `vlc/testsrc2-720p.mp4`, made with `ffmpeg -f lavfi -i testsrc2=size=1280x720:rate=30 -f lavfi -i sine=frequency=440:sample_rate=48000 -t 20 -c:v libx264 -profile:v high -pix_fmt yuv420p -c:a aac -b:a 128k -movflags +faststart testsrc2-720p.mp4`, skips its first-run setup through `prefs.txt`, and plays the clip from its library with one Cross press. Ruffle Flash (PPSA68091) reads `ruffle/testsrc2.swf`, made with `ffmpeg -f lavfi -i testsrc2=size=640x360:rate=24 -t 20 -c:v flv1 -q:v 5 -an testsrc2.swf` (Sorenson H.263 frames; ffmpeg's JPEG-frame SWF stays on its first frame in Ruffle), and plays it from its library with one Cross press; its built-in demo is an empty 18-byte SWF.

`SDL_AUDIODRIVER` is inherited and defaults to `dummy` (silent); set it to `wasapi` or another SDL driver to exercise a title's audio path.

## Guest exceptions delivered under host locks

On Linux, `sceKernelRaiseException` signals the target thread with `SIGUSR1`. The guest handler runs at once only where the thread cannot hold a host lock of the emulator: guest code, a libkernel or guest lock wait, a host system call that waits (a contended host mutex excepted), or the return of `sched_yield`. Anywhere else in host code the delivery is set aside: it runs when the host call returns into guest code, whose return address is redirected to a delivery stub (`Aps5ReturnDeliveryStub`, which unwinders step through), or when the signal, sent again until then, finds the thread in such a place; after 100 ms it runs wherever the thread is. A handler that parks its thread, as garbage collectors that stop the world do, keeps every host lock that thread holds until it is resumed, so a lock still held at a delivery points at a host wait taken under an emulator lock, or a thread kept 100 ms in host code. Host locks that a guest thread takes for a short section can also be held under the libkernel delivery deferral (`TimedWait::DeferredDeliveryLock`, or `GuestDeliveryHold_nid_no_patch` and `GuestDeliveryRelease_nid_no_patch` from another library), which delivers the exception once the lock is released.

To list the locks a title still holds when its handlers run, build the checker and preload it into the title:

```
g++ -O2 -std=c++20 -shared -fPIC -o libdeliverylockcheck.so tools/delivery-lock-check/check.cpp -ldl
LD_PRELOAD=$PWD/libdeliverylockcheck.so DELIVERY_LOCK_CHECK_OUT=/tmp/title <emulator command>
python3 tools/delivery-lock-check/symbolize.py <standard error> /tmp/title.maps --libs <build>/core/libs/libs
```

It tracks the POSIX mutexes, read-write locks and spin locks each thread holds, wraps the `SIGUSR1` handler and writes to standard error, for every 64 deliveries, how many ran without a tracked lock, ran with locks held, or were deferred until a lock was released, with the call site and address of each lock seen. Before the handler runs, it also writes a `held` line for a delivery that arrives while the thread holds tracked locks, so a handler that never returns, as in a hang, still leaves one, and a `waiting` line, with the threads that hold the lock and where they took it, for a delivery that arrives while the thread waits for a contended lock. `symbolize.py` names the sites and locks from the maps file the checker writes at the first delivery. Locks built on atomics, such as the Kyty `TrackingSpinLock`, and glibc's internal locks are not tracked. The checker slows every lock down and is for diagnosis only.

## CSV exports

- `tools/performance_csv.py <baseline> <candidate>` exports a performance comparison as CSV. Empty metric cells mean the value was not reported; `--require-matching-context` fails when the two captures differ in machine or scene, and `--spreadsheet-safe` escapes cells that spreadsheets would evaluate.
- `tools/trace_intervals_csv.py <summary>` exports the trace interval summaries as CSV. `--require-complete` requires at least one interval, a consistent capture identity, no dropped events, and each interval to start at the previous interval's end.
- `tools/ab_split_csv.py <log> [--program P] [--class C]` exports the existing in-run A/B comparison of a log as CSV.
