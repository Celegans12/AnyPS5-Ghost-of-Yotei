# AnyKyty settings for developers

Nothing on this page is needed to play. A game converted with the relinker starts and runs on AnyKyty without any environment variable, and the defaults are the fastest and most stable choices measured so far. The few options a player may want (window, display, GPU, output resolution, how a missing pipeline is handled, ray tracing) are in [the AnyKyty page](../user/ANYKYTY.md#settings).

What follows exists to measure, to compare and to report a problem. Set one of these to compare two behaviours, to take a measurement, or to work around a fault while it is being fixed, and say which ones you set in any number or report you publish: a figure taken with other settings than the defaults does not tell what a player gets. Read once when the game starts; most refuse a value they cannot read, with a message that names the variable, and a few take any value and fall back to their default. [The full reference](THIRD_PARTY_KYTY.md#settings-of-the-kyty-build) lists every `KYTY_*` setting, one row each, with its values, default, effect, category and origin; `python3 tools/check_kyty_settings.py` fails when a name read in `core/` has no row there.

## Display and pacing

| Variable | Values | Effect |
|---|---|---|
| `KYTY_VBLANK_FREQUENCY` | whole number, clamped to 30 to 360 | Kyty's name for `ANYPS5_VBLANK_HZ`, used when that variable is unset |
| `APS5_UNPACED_FLIPS` | `0`, `1` | Benchmark aid: a flip completes once presented instead of at the next vblank, so a game that renders faster than the vblank is not held to it. `KYTY_PRESENT_MODE` still decides what the display shows |

## Speed

On by default unless the row says otherwise.

| Variable | Values | Effect |
|---|---|---|
| `KYTY_GPU_THREAD_CPUS` | unset, `off`, or a comma-separated list of logical CPUs | Unset: GPU stage threads keep to the CPUs sharing the largest L3 cache (Linux). `off`: no placement. A list pins the stages, in the order command processor, execute, record, resolve, host copy |
| `KYTY_EARLY_FLIP` | `0`, `1` (default) | Hand flips back to the game as soon as the presentation copy is submitted |
| `KYTY_SUSPEND_POINT_DEPTH` | 1 to 64, `8` by default | Suspend points the GPU may have in flight; `1` is Kyty's behaviour |
| `KYTY_LABELS_AFTER_DUMPS` | `0`, `1` (default) | Deferred labels wait for the newest pending occlusion dump instead of their own command buffer |
| `KYTY_LABELS_AROUND_GPU_WRITES` | `0`, `1` (default) | A label next to bytes the GPU writes is deferred behind the pending labels instead of draining the GPU |
| `KYTY_TABLE_DESCRIPTOR_WRITES` | `0`, `1`; default `1` on RADV, `0` on other drivers | Fill a draw's bindless texture tables with descriptor writes instead of descriptor copies, which are slow on RADV. `0` copies them and `1` writes them on any driver |
| `KYTY_TABLE_RESOLUTIONS` | 1 to 4096, `256` by default | Size of the cache of resolved bindless texture tables. Evicting a table only makes the draws that use it resolve their tables again |
| `KYTY_TABLE_RESOLUTION_LEGACY` | `0` (default), `1` | `1` restores Kyty's 8-entry cache, where any eviction makes every draw with tables resolve them again |
| `KYTY_DRAW_TEMPLATES`, `KYTY_DRAW_BUFFERS` | `2` and `1` by default; `0` turns off | Kyty's draw templates and grouped draw-buffer copies, on by default in AnyKyty |
| `KYTY_WRITE_BACK_IMAGES` | `0`, `1` (default) | Copy GPU-written images back at the write-backs the game requests |
| `KYTY_TILE_VIA_VRAM` | `0`, `1` (default) | Read tiled images back through VRAM: one copy to host memory instead of scattered writes across PCIe |
| `KYTY_COMPARE_WRITE_BACK` | `0`, `1` (default) | An image read back to game memory writes only the bytes the GPU changed, never bytes the game wrote since the read |
| `KYTY_FAST_TLS` | `0`, `1` (default) | Windows: the driver's own `thread_local` variables come from one native TLS slot instead of libgcc's emulated TLS, which is slower and frees them before their destructors run when a thread exits |
| `KYTY_PROGRAM_LOCK` | `narrow` (default), `wide` | Scope of the program-cache lock. `wide` is Kyty's |
| `KYTY_PIPELINE_LIBRARIES` | `0`, `1` (default) | Link graphics pipelines from per-stage pipeline libraries (`VK_EXT_graphics_pipeline_library`) |
| `KYTY_ASYNC_PIPELINES` | `0`, `1` (default in this build), `2` | Create the pipeline of a draw that only changes the picture in the background; the draw is skipped until it is ready, so an object or effect can be missing for a few frames when it first appears. `2` skips more draws and does not wait for a new pipeline, but only into targets the game redraws every frame: fewer shader compilation stutters on a first run. `1` skips fewer draws and waits up to 20 ms for each new pipeline; `0` always waits |
| `KYTY_SHADER_CACHE` | `0`, `1` (default) | Keep recompiled shaders on disk and warm them up at the next start (Release builds). `0` translates them again at each start. A nonempty `ANYPS5_NO_SHADER_CACHE` value other than `0` forces the store off. |
| `ANYPS5_KYTY_PRESTART` | `0`, `1` (default) | `0` starts the Kyty GPU after the window exists instead of during its creation |

## Comparing two settings in one run

`KYTY_AB` alternates two arms inside one session, so both play the same scene. Phase A keeps the listed settings as set,
phase B turns them over, at a flip, between two submissions. Only `KYTY_FLAT_DEFER`, `KYTY_GENERIC_COMPUTE_VARIANT`,
`KYTY_TRACKER_LAZY_REGIONS` and `KYTY_WRITTEN_UPLOAD_UNLOCKED` can change while the title runs; any other name stops at
start with the list ([why](THIRD_PARTY_KYTY.md#details)).

| Variable | Values | Effect |
|---|---|---|
| `KYTY_AB` | `<seconds>:<setting>[+<setting>...]` | Phases of at least `<seconds>`; `[kyty-ab]` lines give each phase's presents and img/s |
| `KYTY_AB_SETTLE` | seconds, `2` by default | Start of the counted part of each phase |
| `KYTY_AB_START` | flip count, or `<n>s` | When the first phase begins |

For example `KYTY_AB=40:KYTY_FLAT_DEFER KYTY_AB_SETTLE=8`, then `python3 tools/kyty_ab_report.py <session folder or log>`
for each arm's img/s, draws per image and `[kyty-*]` counters.

## Ray tracing

| Variable | Values | Effect |
|---|---|---|
| `KYTY_NATIVE_RT` | unset, `0`, `1` | Unset: native ray tracing when the GPU supports it. `0`: ray tracing emulated in shaders. `1`: native, and a stop with a message when the GPU cannot |
| `KYTY_RT_STATS` | `0`, `1` | Print a ray tracing summary every 10 s |
| `KYTY_RT_ORACLE` | `0` to `15` | `1`: the ray tracing passes whose BVH walk no form recognizes still trace natively, each BVH intersection answered from a host ray query (the default); `0` runs them empty (their rays miss); `9` also serves the BVH walks of pixel shaders, which stay emulated by default |
| `KYTY_RT_CHECK` | `0`, N | `0`: off. N: one dispatch in N of every program traced natively is also played with the emulated walk and both outputs are compared; `1` checks every dispatch (debugging) |
| `KYTY_RT_CHECK_PPM` | parts per million | Fraction of output dwords that may differ before a check counts as failed (default 1000) |
| `KYTY_RT_CHECK_STRIKES` | count | Failed checks in a row after which a program walks the emulated BVH for the rest of the session (default 3) |
| `KYTY_RT_ARENA_MB` | MiB | Limit of the memory the native ray tracing structures take |
| `KYTY_RT_RESULTS` | `0` (default), `1` | `1`: with `KYTY_RT_ORACLE=1`, a BVH walk whose results the shader compiler derives is traced by one host ray query, without running the guest walk code |

## Reporting a problem

With the Kyty backend the host report (OS, CPU, memory, GPU and driver) prints once at start without any variable; `ANYPS5_HOST_REPORT=0` turns it off. See [Host report](../user/USAGE.md#host-report).

| Variable | Values | Effect |
|---|---|---|
| `KYTY_PRINTF_DIRECTION` | `Silent` (default), `Console`, `File` | Where Kyty's log goes |
| `KYTY_SESSION_LOG` | path | Writes a session log file |
| `KYTY_PIPELINE_STALL_LOG_MS` | milliseconds, `100` by default | Logs `[kyty-gpl]` lines for pipeline stalls and shader translations that take this long |
| `KYTY_SHADER_VARIANT_LOG` | `1` enables it | Logs shader builds with a module ID and cache-key fingerprints. Compare records for the same stage and shader hash across repeated runs |
| `KYTY_VULKAN_VALIDATION` | `0`, `1` | Turns the Khronos validation layer on |
| `KYTY_LOCK_STATS` | `0`, `1` | Prints how long threads wait on the program-cache lock |

## What AnyKyty does not read

AnyKyty does not read Kyty's window options (`KYTY_FULLSCREEN`, `KYTY_HIDE_CURSOR`, `KYTY_VR`, `KYTY_RENDERDOC`, `KYTY_SCREEN_WIDTH`, `KYTY_SCREEN_HEIGHT`), its trophy and language options (`KYTY_TROPHY_NOTIFICATIONS`, `KYTY_TROPHY_DURATION`, `KYTY_CONSOLE_LANGUAGE`) or `KYTY_PIPELINE_PRECOMPILE`: setting them changes nothing. A few other switches have no effect or are not meant for play: the full reference says which, in its category column.
