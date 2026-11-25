# Environment variables

The code reads environment names through `getenv`, `IsEnvironmentSet`, `kibSetting`, `MaskFromEnvironment`, `EnvInt`, `GetEnvironmentVariableA`, `GetEnvironmentVariableW` and `_wgetenv` with a wide literal, `configured`, `PipelineSwitch`, `environmentCount`, `EnvironmentNumber`, `MebibytesFromEnvironment` and `addWideString` (the wide names the Windows entry stub reads). This table lists every `APS5_*` and `ANYPS5_*` name it reads, its kind, the value used when it is unset and what it does, one table per library. Regenerate the names with `python3 tools/env_reference.py --table`; the Conventions check fails when a name the code reads has no row here, or a row's name is no longer read.

### core/libs

| Name | Kind | Source | Default | Description |
|------|------|--------|---------|-------------|
| `ANYPS5_GUEST_ENV_TEST_4C27` | string | `core/libs/tests/GuestEnvironment.cpp` |  | test variable that the guest environment test sets, copies and removes; not a product setting |
| `ANYPS5_JOB_MANAGER_CASE` | string | `core/libs/tests/JobManagerReference.cpp` |  | selects the CPU fixture run by `tools/run_job_manager_reference.py` |
| `ANYPS5_JOB_MANAGER_HOST` | presence | `core/libs/tests/JobManagerReference.cpp` |  | runs the reference adapter against the host fallback instead of the loaded guest module |
| `ANYPS5_LOADEXEC_PARENT` | string | `core/libs/tests/GuestLoadExecHost.cpp` |  | variable the Windows entry stub reads; its wide name is built in `core/relinker/elfpatcher/src/windows/WindowsEntryStubBuilder.cpp` |
| `ANYPS5_KERNEL_SYS_EXEC_MODE` | string | `core/libs/tests/GuestExecHost.cpp` |  | selects `_execve` (`ve`) or `_execvpe` (`vpe`) in the exec replacement test fixture; unset tests libkernel `execve` |
| `ANYPS5_LOADEXEC_TEST_ROOT` | string | `core/libs/tests/GuestLoadExecHost.cpp` |  | directory the load-exec host test reads its guest executable from |
| `ANYPS5_POSIX_SPAWN_ROLE` | string | `core/libs/tests/GuestPosixSpawn.cpp` |  | role of the test's own copy spawned as a `posix_spawn` child, which exits or sleeps on it and otherwise fails |
| `ANYPS5_TEST_QUOTED_ARGV0` | presence | `core/libs/tests/GuestLoadExecHost.cpp` |  | makes the load-exec test's first argument one with quotes, a backslash and non-ASCII text |
| `APS5_EBOOT` | string | `core/libs/prx/libSceSystemService/Export.cpp` |  | host path of the executable launched for /app0/eboot.bin; unset or empty means the running executable |
| `APS5_LOAD_EXEC_STANDARD_ERROR` | string | `core/libs/tests/GuestSystemServiceLoadExecStandardError.cpp` |  | marker that the standard-error load-exec test sets in its child process to choose a mode, `list` or `none`; not a product setting |
| `APS5_SKIP_TIMING_TESTS` | string | `core/libs/tests/PreciseWait.cpp` |  | treats the timing tests' runner as preempted, so they skip without probing the clock; an empty value or one starting with 0 leaves the probe on |
| `APS5_USBD_DEVICE` | string | `core/libs/tests/GuestUsbd.cpp` |  | selects the USB device a test drives, as vendor:product in hex |

### core/relinker

| Name | Kind | Source | Default | Description |
|------|------|--------|---------|-------------|
| `ANYPS5_AUTORUN_NO_PAUSE` | string | `core/relinker/cli/src/Autorun.cpp` | | exactly `1` skips the exit prompt and stdin read after autorun; unset, empty and other values keep the prompt |
| `ANYPS5_GUEST_MODULE_ARGS` | string | `core/relinker/elfpatcher/tests/GuestModuleArgsHost.cpp` |  | path of the file the module-argument fixture appends its events to |

### libSceAgc

| Name | Kind | Source | Default | Description |
|------|------|--------|---------|-------------|
| `APS5_TRACE_LABEL` | integer | `core/libs/prx/libSceAgc/Cb/src/Sync.cpp` | 0 | hex guest address; logs a matching RELEASE_MEM when recorded, and nearby RELEASE_MEM, WRITE_DATA, COPY_DATA, DMA_DATA and WAIT_REG_MEM packets when executed |

### libSceAgcDriver

| Name | Kind | Source | Default | Description |
|------|------|--------|---------|-------------|
| `ANYPS5_FPS_LIMIT` | string | `core/libs/prx/libSceAgcDriver/Execution/include/FrameLimit.hpp` | 0 | the most frames per second the window presents, from 0.5 to 1000000 with at most three decimals; 0, empty or unset is no limit, and any other value is refused |
| `ANYPS5_GPU` | string | `core/libs/prx/libSceAgcDriver/KytyShim/src/Config.cpp`, `core/libs/prx/libSceAgcDriver/KytyShim/tests/HalfFallback.cpp` | automatic | case-insensitive substring of the Vulkan device name to select; unset or empty keeps automatic selection; the software-half GPU test also honors this filter |
| `ANYPS5_FSR_SHARPNESS` | string | `core/kyty/graphics/presentation/window/swapchain.cpp` | 0.2 | RCAS sharpness in stops under `ANYPS5_UPSCALE=fsr` or `rcas`, from 0.0 (sharpest) to 2.0 (weakest); a value outside that range or not a number is refused |
| `ANYPS5_HOST_REPORT` | string | `core/libs/prx/libSceAgcDriver/KytyShim/src/EntryGpu.cpp` | 0 | prints the host report to stderr once when the Vulkan device is chosen: 1 prints it; 0, empty or unset prints nothing (with the Kyty backend, unset prints it and 0 or empty turns it off), and any other value stops the start with an error naming the variable |
| `ANYPS5_KYTY_PRESTART` | string | `core/libs/prx/libSceAgcDriver/KytyShim/src/EntryGpu.cpp` | 1 | when the Kyty GPU starts: 1, empty or unset starts it on a host thread as soon as the video output is created, while the window is created; 0 starts it after the window exists, as before; any other value stops the game when its video output starts, with an error that names the variable |
| `ANYPS5_KYTY_SPARSE_READS` | string | `core/libs/prx/libSceAgcDriver/KytyShim/src/MemoryBacking.cpp` | 1 | how the Kyty GPU reads guest memory of 64 KiB or more: 1, empty or unset asks the memory files where their data is (`SEEK_DATA`) and writes zeros for the never-written parts without touching them; 0 reads every byte, as before, which fills those parts with real zeroed pages; any other value stops the game when the guest memory view is enabled, with an error that names the variable. Linux only: on Windows the variable is not read, the view is always copied whole as with 0, and `DirectMemoryViewFiles_nid_no_patch` returns false because the view has no file descriptors there |
| `ANYPS5_REQUIRE_VULKAN` | presence | `core/libs/prx/libSceAgcDriver/KytyShim/tests/BdaLoadForms.cpp` |  | fails a test instead of skipping when no usable Vulkan device is present |
| `ANYPS5_UPSCALE` | string | `core/kyty/graphics/presentation/window/swapchain.cpp` |  | fsr scales the presentation with FSR 1 (EASU and RCAS) when the window is larger than the display buffer, cmaa2 applies Intel CMAA2 morphological anti-aliasing at 1:1 native resolution, rcas applies 1:1 native RCAS sharpening; unset or empty keeps the bilinear scaler, and any other value is refused |
| `APS5_FORCE_SDR` | presence | `core/libs/prx/libSceAgcDriver/Execution/src/HostHdr.cpp` | unset | suppresses advertised HDR availability while retaining detected host capabilities; uses the existing SDR presentation path |
| `APS5_FRAME_REPLAY_ATTEMPT` | integer | `core/libs/prx/libSceAgcDriver/tools/FrameReplay.cpp` | 0 | number of times `agc_frame_replay` has started itself again on Windows because a host region sat in a captured GPU-visible range; set by the replay for the process it starts, which stops relaunching after 8 |
| `APS5_HEADLESS_PIPELINE_CACHE` | presence | `core/libs/prx/libSceAgcDriver/KytyShim/src/VideoOut.cpp` |  | loads the pipeline cache file on a device without a window too; `agc_frame_replay` sets it by default and clears it with `--cold-pipelines`, including inherited values |
| `APS5_NO_GPU_WRITE_MARK_EXPIRY` | string | `core/libs/prx/libc/include/WindowsPageTrack.hpp` | 0 | with `APS5_ARM_READ_IMPORTS=1`, `1` keeps a 64 KiB block that the AGC driver announced a GPU write into unprotected until it is unmapped, as before the mark could expire; unset, empty or `0` lets the mark expire once every batch opened before the last announcement into the block has finished on the GPU and that announcement is two frames old, which the `[collects]` line of `APS5_PIPELINE_STATS` counts as blocks whose mark expired; any other value fails |
| `APS5_SYNC_FLIP` | presence | `core/libs/prx/libSceVideoOut/src/VideoOutDriver.cpp` |  | makes flips synchronous (no frames in flight) |
| `APS5_THREAD_NICE` | integer | `core/libs/prx/libSceAgcDriver/Execution/src/ThreadPriority.cpp` | -10 | nice value the raised queue workers and presenter ask rtkit for (clamped to -20..19) |
| `APS5_THREAD_PRIORITY` | string | `core/libs/prx/libSceAgcDriver/Execution/src/ThreadPriority.cpp` | off | raises the queue workers and presenter above the title's threads: `off`, `high` or `rt`; on macOS `high` is the user-interactive QoS class and `rt` a Mach time-constraint policy (period 1 ms, computation 0.5 ms) |
| `APS5_THREAD_RT` | integer | `core/libs/prx/libSceAgcDriver/Execution/src/ThreadPriority.cpp` | 10 | realtime priority the raised threads ask rtkit for (clamped to 1..99) |
| `APS5_THREAD_RTTIME_US` | integer | `core/libs/prx/libSceAgcDriver/Execution/src/ThreadPriority.cpp` | 200000 | hard RLIMIT_RTTIME the realtime plan lowers to, in microseconds |
| `APS5_TITLE_EVERY_FRAME` | presence | `core/libs/prx/libSceVideoOut/src/DisplayWindow.cpp` |  | outside Windows, sets the window caption (title, FPS and frame number) at every flip, as on Windows, instead of at most once a second; each caption change can rebuild client-side decorations (libdecor-gtk under Wayland, 3 to 4 ms on the presenter, which then signals the flip late) |
| `APS5_TRACE_ALL_PHASES` | string | `core/libs/prx/libSceAgcDriver/Execution/include/PerformanceTrace.hpp` |  | `1` adds `phase_totals` to every summary line: each recorded scope and stage with its call count, summed milliseconds and milliseconds per present, so stages folded out of `top_phases_ms_per_frame` stay visible |
| `APS5_TRACE_BENCH` | string | `core/libs/prx/libSceAgcDriver/Execution/include/PerformanceTrace.hpp` |  | `1` with `APS5_TRACE_SUMMARY_EVERY` also rewrites `aps5-bench-<commit>-<pid>.json.gz` with the frame times of the last 240 intervals (older ones are counted as dropped); any other value, or `1` without `APS5_TRACE_SUMMARY_EVERY`, fails the first present |
| `APS5_TRACE_FRAMES` | string | `core/libs/prx/libSceAgcDriver/Execution/include/PerformanceTrace.hpp` |  | records `<count>` presents from present `<start>` (whole numbers, `start:count`) |
| `APS5_TRACE_SECONDS` | string | `core/libs/prx/libSceAgcDriver/Execution/include/PerformanceTrace.hpp` |  | records from the first present `<start>` seconds after the first present for `<length>` seconds (whole numbers, `start:length`) |
| `APS5_TRACE_TRIGGER_FILE` | native Unicode path on Windows, UTF-8 path elsewhere | `core/libs/prx/libSceAgcDriver/Execution/include/PerformanceTrace.hpp` | unset | additionally waits for a regular file before opening the frames/seconds trace window; the original start remains a lower bound, and the full duration begins when the window opens. Use a unique initially absent path, then create the file after verifying the scene. Use a local filesystem path: the presenter checks file metadata under the trace mutex, so slow or network metadata access can stall presentation. Checks occur on presents at most once per 250 ms while waiting; an existing file opens at the original lower bound. The file is not removed, read, or checked again after opening. Requires `APS5_TRACE_FRAMES` or `APS5_TRACE_SECONDS`; empty paths, existing nonregular files and filesystem errors are rejected. |
| `APS5_TRACE_SUMMARY_EVERY` | integer | `core/libs/prx/libSceAgcDriver/Execution/include/PerformanceTrace.hpp` |  | seconds between periodic trace summaries, emitted on the next present |

### libSceAjm.native

| Name | Kind | Source | Default | Description |
|------|------|--------|---------|-------------|
| `APS5_TRACE_AJM` | presence | `core/libs/prx/libSceAjm.native/src/Ajm.cpp` |  | logs every job the title submits and what the library writes back |

### libSceAudioIn

| Name | Kind | Source | Default | Description |
|------|------|--------|---------|-------------|
| `ANYPS5_AUDIO_IN_DEVICE` | string | `core/libs/prx/libSceAudioIn/Export.cpp` | default | exact SDL capture-device name; unset, empty or `default` selects SDL's default; missing, duplicate or unopenable selected devices make AudioIn open return `0x8026010F`; read each time a port opens |

### libSceAudioOut

| Name | Kind | Source | Default | Description |
|------|------|--------|---------|-------------|
| `APS5_PAD_SPEAKER_FALLBACK_DB` | string | `core/libs/prx/libSceAudioOut/src/AudioOut2PadMix.cpp` | 0 | gain in dB for AudioOut2 controller-speaker ports folded into the main mix when no controller speaker is present; accepts a number from -60 to 0 or mute/off/-inf, and the default 0 leaves the mix unchanged; any other value, including an out-of-range number, stops the title with an error naming the variable and the value |
| `ANYPS5_MUTE_ON_FOCUS_LOSS` | string | `core/libs/prx/libScePad/src/HostVolumeState.cpp` | 0 | `1` mutes the host audio while the display window has no focus and restores it when the focus returns; `0` or unset leaves the audio alone; any other value, including an empty one, stops the game with an error naming the variable |
| `ANYPS5_VOLUME` | string | `core/libs/prx/libScePad/src/HostVolumeState.cpp` | 100 | initial host output volume, a whole number from 0 to 100, changed afterwards by the volume keys; 100 leaves the audio bit-identical to the title's output; any other value, including an empty one, stops the game with an error naming the variable |
| `APS5_TRACE_AUDIOOUT` | presence | `core/libs/prx/libSceAudioOut/src/AudioOut.cpp` |  | logs each audio-out port opened with its type, sample count, frequency and format |
| `APS5_TRACE_AUDIOOUT2` | presence | `core/libs/prx/libSceAudioOut/src/AudioOut2Context.cpp` |  | prints the title's AudioOut2 context and port parameters and stream activity |

### libSceFiber

| Name | Kind | Source | Default | Description |
|------|------|--------|---------|-------------|
| `APS5_TRACE_FIBER` | presence | `core/libs/prx/libSceFiber/Export.cpp` |  | logs fiber initialization, switches and returns to the thread |

### libSceFont

| Name | Kind | Source | Default | Description |
|------|------|--------|---------|-------------|
| `ANYPS5_SYSTEM_FONTS` | string | `core/libs/prx/libSceFont/src/SystemFont.cpp` |  | directory holding the system fonts, defaulting to anyps5-fonts beside the executable |

### libSceGnmDriver

| Name | Kind | Source | Default | Description |
|------|------|--------|---------|-------------|
| `APS5_GNM_SUBMIT_WAIT_IDLE` | presence | `core/libs/prx/libSceGnmDriver/Export.cpp` |  | makes each command buffer submission after the first wait for the work already queued, so the GPU trails the title by at most one submission |

### libSceIme

| Name | Kind | Source | Default | Description |
|------|------|--------|---------|-------------|
| `ANYPS5_PSN_DIALOGS` | string | `core/libs/prx/libSceIme/src/PsnDialogNotice.cpp` | silent | when a dialog that needs PlayStation Network opens (profile, friend list, store, invitations, Facebook link, authorized apps), `show` displays a host notice that the feature is not available; `silent`, unset or empty shows nothing, as before; any other value stops the title at its first such dialog with an error naming the variable |

### libSceNpEntitlementAccess

| Name | Kind | Source | Default | Description |
|------|------|--------|---------|-------------|
| `ANYPS5_ENTITLEMENTS` | string | `core/libs/prx/libSceNpEntitlementAccess/EntitlementsIni.hpp` |  | path of the entitlements ini listing owned add-ons, defaulting to anyps5-entitlements.ini beside the executable; with the Kyty backend and neither the variable nor that file, every add-on a title asks about is reported as installed |

### libSceNpTrophy2

| Name | Kind | Source | Default | Description |
|------|------|--------|---------|-------------|
| `APS5_TRACE_UDS` | presence | `core/libs/prx/libSceNpTrophy2/src/TitleTrophies.cpp` |  | prints the UDS events contexts post and the stat changes trophy rules apply, and the trophy files a lookup does not find (read in NpUniversalDataSystem/Event and NpTrophy2/TitleTrophies) |

### libSceNetCtl

| Name | Kind | Source | Default | Description |
|------|------|--------|---------|-------------|
| `APS5_NET_DISCONNECTED` | presence | `core/libs/prx/libSceNetCtl/Export.cpp` |  | libSceNetCtl and the libSceNet route, DNS-server and interface queries report no connection; libSceHttp2 sends (`sceHttp2SendRequest`, and `sceHttp2SendRequestAsync` through its completion) fail with network error 0x80431063 without connecting; sockets are not affected |

### libScePad

| Name | Kind | Source | Default | Description |
|------|------|--------|---------|-------------|
| `ANYPS5_AUDIO_IN_MUTE_ON_FOCUS_LOSS` | string | `core/libs/prx/libScePad/src/HostVolumeState.cpp` | 0 | `1` returns silence to the guest while the display window has no focus and resumes input when it returns; SDL capture stays open, and queued samples are cleared across focus changes; `0` or unset leaves input unchanged; any other value, including an empty one, stops the game with an error naming the variable |
| `ANYPS5_RUMBLE` | integer | `core/libs/prx/libScePad/include/PadRumbleIntensity.hpp` | 100 | host controller vibration strength as a whole percentage from 0 to 100, read once when the pad input starts and again by the audio-out controller sound card path; 100 or unset leaves vibration exactly as the title asks, 0 turns motor rumble, impulse-trigger rumble, controller-sound-card haptics and Bluetooth DualSense haptics off, in between scales the motor amplitudes with rounding and the haptic samples linearly; DualSense adaptive trigger effects are not scaled; any other value (empty, signed, fractional, above 100) stops the game with an error naming the variable |
| `ANYPS5_RUMBLE_ON_FOCUS_LOSS` | string | `core/libs/prx/libScePad/include/PadRumbleIntensity.hpp` | 1 | `1` stops the motor rumble and drops new pulses while the display window has no focus, sending one zero-intensity request when focus is lost and resuming with the next pulse when it returns; `0` keeps rumble on without focus; any other value, including an empty one, stops the game at start with an error naming the variable; the light bar and the pad state are not changed |
| `ANYPS5_PAD_IDENTIFY` | string | `core/libs/prx/libScePad/include/PadIdentify.hpp` | off | which controller-identification pulse to send when a controller is bound to a player slot: `off` sends none, `connect` sends the pattern once per binding (player N gets N short pulses), `always` also repeats it every 30 seconds while the title is not rumbling; the pulse is scaled by `ANYPS5_RUMBLE`, never overrides a title's rumble, and any other value (empty, uppercase, a list) stops the game with an error naming the variable |
| `ANYPS5_PAD_PLAYER_1_SERIAL` | string | `core/libs/prx/libScePad/include/ControllerAssignment.hpp` |  | SDL controller serial assigned to guest player 1; an empty or unset value leaves the slot in sequential assignment |
| `ANYPS5_PAD_PLAYER_2_SERIAL` | string | `core/libs/prx/libScePad/include/ControllerAssignment.hpp` |  | SDL controller serial assigned to guest player 2; an empty or unset value leaves the slot in sequential assignment |
| `ANYPS5_PAD_PLAYER_3_SERIAL` | string | `core/libs/prx/libScePad/include/ControllerAssignment.hpp` |  | SDL controller serial assigned to guest player 3; an empty or unset value leaves the slot in sequential assignment |
| `ANYPS5_PAD_PLAYER_4_SERIAL` | string | `core/libs/prx/libScePad/include/ControllerAssignment.hpp` |  | SDL controller serial assigned to guest player 4; an empty or unset value leaves the slot in sequential assignment |
| `ANYPS5_PAD_LOG_CONTROLLER_SERIAL` | string | `core/libs/prx/libSceVideoOut/src/PadInput.cpp` | 0 | `1` logs each SDL controller's serial for configuring player assignment; other values leave serial logging off |
| `APS5_PAD_INPUT_FILE` | string | `core/libs/prx/libScePad/src/PadState.cpp` |  | path of the file polled for injected pad button and stick commands |
| `APS5_PAD_RECORD` | string | `core/libs/prx/libScePad/src/PadState.cpp` |  | path of a file that receives every change of the buttons, sticks and analog triggers the title reads from controller port 0, in milliseconds from its first read; a path that cannot be written stops the title at that read with an error naming the variable |
| `APS5_PAD_REPLAY` | string | `core/libs/prx/libScePad/src/PadState.cpp` |  | path of an `APS5_PAD_RECORD` file whose states replace the buttons, sticks and analog triggers the title reads from controller port 0, timed from its first read; a file that cannot be read or is not a recording stops the title at that read with an error naming the variable |
| `APS5_PAD_SCRIPT` | string | `core/libs/prx/libScePad/src/PadState.cpp` |  | path of a text file of timed pad inputs injected into user 0 and OR-ed with real input, one event per line: `<start_ms> <button>[+<button>...] [<hold_ms>]` (hold defaults to 150), `[<start_ms>] repeat <every_ms> <count> <button>[+<button>...] [<hold_ms>]`, or `stick <left\|right> <x> <y> <start_ms> <hold_ms>` with x and y 0 to 255; buttons are cross circle square triangle options touchpad l1 r1 l2 r2 l3 r3 up down left right; times count from the title's first pad read; `#` starts a comment; port 0 stays connected without a controller and `ANYPS5_PAD_PORT0_STATE` is not read, input is not gated on window focus, each fired event is logged to stderr as `[pad-script] t=<ms> <buttons> hold <ms>` and a malformed line stops the title with an error naming the file and line |
| `APS5_PAD_SCRIPT_LIVE` | string | `core/libs/prx/libScePad/src/PadState.cpp` |  | path of a file polled while the title reads the pad (at most every 250 ms, by modification time or size); lines appended after startup, in the same format as `APS5_PAD_SCRIPT` with `start_ms` meaning a delay from the moment the line is read, are scheduled then (for example `echo "0 cross" >> file`); a malformed appended line, or one that would leave more than 20000 appended events pending, is reported on stderr with its line number in the file and skipped; a poll that cannot read the file is retried at the next one; also enables the scripted pad (port 0 stays connected and `ANYPS5_PAD_PORT0_STATE` is not read) |
| `APS5_PAD_SCRIPT_LOOP_MS` | integer | `core/libs/prx/libScePad/src/PadState.cpp` | 0 | with `APS5_PAD_SCRIPT`, restarts the script this many milliseconds after its last event ends, pressing nothing in between; 0 or unset plays it once; a value that is not a non-negative integer stops the title with an error naming the variable |

### libScePlayGo

| Name | Kind | Source | Default | Description |
|------|------|--------|---------|-------------|
| `APS5_PLAYGO_UNLISTED_CHUNKS` | string | `core/libs/prx/libScePlayGo/Export.cpp` |  | answer for a chunk ID the playgo manifest does not list: installed, absent, or a comma-separated list of installed chunk ids; unset throws instead of answering |

### libSceSaveData.native

| Name | Kind | Source | Default | Description |
|------|------|--------|---------|-------------|
| `APS5_SAVEDATA_TRACE` | presence | `core/libs/prx/libSceSaveData.native/Export.cpp` | 3 | logs every save-data operation instead of the first three at each logging site |
| `ANYPS5_FREE_SPACE` | string | `core/libs/prx/libSceSaveData.native/Export.cpp` | fixed | `fixed` keeps the save sizes and room the title sees today; `host` refuses to create a save whose blocks the save's volume cannot hold (`SCE_SAVE_DATA_ERROR_NO_SPACE_FS`, with the missing blocks in `required_blocks`) and bounds a save's free blocks by the room its volume has; empty or any other value stops the game with an error naming the variable; on Windows an empty value is the same as unset, because `cmd` and PowerShell remove the variable |

### libSceSystemService

| Name | Kind | Source | Default | Description |
|------|------|--------|---------|-------------|
| `ANYPS5_DATE_FORMAT` | string | `core/libs/prx/libSceSystemService/DateTimeLocale.hpp` | 1 | console date format returned by sceSystemServiceParamGetInt parameter 2: 0 year, month, day; 1 day, month, year (the default); 2 month, day, year; `auto` (any letter case) maps the host's short date pattern once per process and falls back to 1 with one stderr line; any other value throws |
| `ANYPS5_LANGUAGE` | integer | `core/libs/prx/libSceSystemService/SystemService.hpp` | 1 | console language number from 0 to 30 returned by sceSystemServiceParamGetInt, defaulting to English US; `auto` (any letter case) maps the host's locale to a console language once per process and falls back to English US with one stderr line; any other value throws |
| `ANYPS5_NOTICE_SKIP_PERSIST` | string | `core/libs/prx/libSceSystemService/NoticeScreenSkipStore.hpp` | 1 | `1`, empty or unset keeps the notice screen skip flag a title raises with `sceSystemServiceSetNoticeScreenSkipFlag` from one launch to the next, in `anyps5-notice-skip/<title ID>` in the directory the title runs from (a file holding `1`, written when the flag is raised; delete it to lower the flag), so the title's next `sceSystemServiceGetNoticeScreenSkipFlag` reports it raised from the start, as the console does for a title that raised it; `0` keeps it only for the running process, always lowered at launch, and reads and writes no file; any other value throws. A record that cannot be written leaves the flag raised for the running process and prints one stderr line |
| `ANYPS5_OPEN_URLS` | string | `core/libs/prx/libSceSystemService/HostUrlPolicy.hpp` |  | `never`, `ask` or `always`: whether `sceSystemServiceLaunchWebBrowser` opens a valid http or https address in the host's default browser; unset, empty and `never` keep the call failing, `ask` shows the address in a confirmation box first, `always` opens it without asking; any other value throws |
| `ANYPS5_TIME_FORMAT` | string | `core/libs/prx/libSceSystemService/DateTimeLocale.hpp` | 1 | console time format returned by sceSystemServiceParamGetInt parameter 3: 0 for 12-hour, 1 for 24-hour (the default); `auto` (any letter case) maps the host's time pattern once per process and falls back to 1 with one stderr line; any other value throws |

### libSceUsbStorage

| Name | Kind | Source | Default | Description |
|------|------|--------|---------|-------------|
| `APS5_USB_STORAGE_PASSTHROUGH` | presence | `core/libs/prx/libSceUsbStorage/src/HostVolumes.cpp` |  | passes the FAT and exFAT volumes the host mounted from USB mass-storage devices through to the guest, read-only where the host mounted them read-only; unset, no host volume is scanned and the guest sees no USB storage device |

### libSceVideoOut

| Name | Kind | Source | Default | Description |
|------|------|--------|---------|-------------|
| `ANYPS5_DISPLAY_INDEX` | integer | `core/libs/prx/libSceVideoOut/src/DisplayWindow.cpp` | 0 | SDL display index used for the initial window size and centered position; invalid or unavailable indices fail window creation; selection does not guarantee the window will not activate |
| `ANYPS5_FULLSCREEN` | string | `core/libs/prx/libSceVideoOut/src/DisplayWindow.cpp` | windowed | `1` or `borderless` opens the display window fullscreen at the desktop resolution, `0` or `windowed` (or empty) keeps it windowed; any other non-empty value opens it fullscreen as before and prints a warning naming the value; F11 switches in either case |
| `ANYPS5_WINDOW_MODE` | string | `core/libs/prx/libSceVideoOut/src/DisplayWindow.cpp` | windowed | `borderless` starts the display window borderless, sized to its display, unless `ANYPS5_FULLSCREEN` opens it fullscreen; any other value, or unset, starts it as a normal window; F2 switches borderless on and off in either case |
| `ANYPS5_INPUT_CONFIG` | string | `core/libs/prx/libSceVideoOut/src/InputMapping.cpp` |  | path of the input mapping file that binds host keys and mouse to guest pad actions |
| `ANYPS5_AUDIO_DEVICE` | string | `core/libs/prx/libSceAudioOut/src/AudioOut.cpp` |  | the exact name of a host output device to play the title's audio on; a name that is not listed is refused with the list of output devices in the log, and the default device is used |
| `ANYPS5_MOUSE_GYRO` | string | `core/libs/prx/libSceVideoOut/src/PadInput.cpp` |  | `1` turns the mouse's movement into the pad's gyro while the mouse is captured (mouse mode): counts per second times 0.001 rad, so a drag of 1000 counts in a second is 1 rad/s; any other value leaves the gyro alone |
| `ANYPS5_MOUSE_TOUCHPAD` | string | `core/libs/prx/libSceVideoOut/src/PadInput.cpp` |  | `1` makes a press of the host's left mouse button touch the pad's touchpad at the cursor, scaled to the 1920x1080 touchpad, and a drag move the touch point; only while the mouse is not captured (not in mouse mode); any other value leaves the mouse off the touchpad |
| `ANYPS5_STORAGE_FOLDER` | string | `core/libs/prx/libSceUsbStorage/src/HostVolumes.cpp` |  | a host folder that appears to the guest as a FAT USB volume at the next free `/mnt/usbN` mount point; the folder must exist and be a directory, otherwise one line is logged and nothing is mounted; with the variable unset nothing changes |
| `ANYPS5_TIME_FORMAT` | string | `core/libs/prx/libc/src/Time.cpp` | `c` | `host` takes the date and time strings of `%c`, `%x` and `%X` from the host's `LC_TIME` locale (glibc only); any other value keeps the C locale strings |
| `ANYPS5_INPUT_WATCH` | integer | `core/libs/prx/libSceVideoOut/src/PadInput.cpp` | 0 | `1` checks the input mapping file and the title's profile about once a second on a background thread and reloads them after a change settles; any other value, including unset, leaves reloading to the `ReloadInputMapping` action (F1) |
| `ANYPS5_IME_COPY` | string | `core/libs/prx/libSceIme/Export.cpp` | `0` | `1` copies the text of an IME session to the host clipboard when the title confirms it (Enter), as UTF-8 through SDL; an empty text is not copied. This is an AnyPS5 choice, not a console behaviour. Any other value, unset or empty copies nothing. Apple builds never copy |
| `ANYPS5_KEEP_AWAKE` | string | `core/libs/prx/libSceVideoOut/src/VideoOutDriver.cpp` | 1 | `0` sets `SDL_VIDEO_ALLOW_SCREENSAVER=1` before SDL's video subsystem starts, so the host screen saver can run; any other value, including unset, leaves SDL's default, which disables the screen saver while video is up |
| `ANYPS5_MSG_DIALOG_HOST` | string | `core/libs/prx/libSceMsgDialog/Export.cpp` | `0` | `1` shows each user message dialog (`sceMsgDialogOpen` with a text and a button type) as a native host message box instead of the in-game dialog; the call waits until the box is answered, then reports the button. Any other value, unset or empty keeps the in-game dialog. Apple builds always keep the in-game dialog. If the box cannot be shown, the in-game dialog is used |
| `ANYPS5_PAD_PORT0_STATE` | string | `core/libs/prx/libSceVideoOut/include/Port0Presence.hpp` | default | `default` keeps port 0 connected and user 0 logged in as before; `controller` makes port 0 and user 0 follow the controller bound to slot 0, a change committed after 500 ms, and a withdrawn port 0 reads zeroed and disconnected; any other value, including an empty one, stops the game when video output starts with an error naming the variable |
| `ANYPS5_REQUIRE_DISPLAY` | presence | `core/libs/prx/libSceVideoOut/tests/HostFullscreen.cpp` |  | makes video-out tests fail instead of skipping when no display or Vulkan device is available |
| `ANYPS5_VBLANK_HZ` | integer | `core/libs/prx/libSceVideoOut/include/VblankSchedule.hpp` |  | vblanks per second of 59.94 Hz outputs, a whole number from 30 to 360; 119.88 Hz outputs then get twice that, at most 360; flips, flip rates, vblank events, `sceVideoOutWaitVblank` and the vblank status follow it, so a title that paces on them runs up to that rate; the refresh rate reported to the title is unchanged; an unset or empty value takes `KYTY_VBLANK_FREQUENCY`, clamped to 30 to 360 as Kyty does, and the value also becomes Kyty's vblank frequency; unset or empty keeps 59.94 Hz and 119.88 Hz, and any other value stops the game when its video output starts, with an error that names the variable |
| `ANYPS5_VIDEO_OUT_MAX_HEIGHT` | integer | `core/libs/prx/libSceVideoOut/src/Output.cpp` |  | host display height hint: when set to a positive integer below 2160, an unknown output mode is reported as unsupported so the title can fall back to the host resolution; a malformed or zero value makes output-mode queries throw; buffer attributes are left untouched |
| `ANYPS5_VIDEO_OUT_MAX_WIDTH` | integer | `core/libs/prx/libSceVideoOut/src/Output.cpp` |  | host display width hint: when set to a positive integer below 3840, an unknown output mode is reported as unsupported so the title can fall back to the host resolution; a malformed or zero value makes output-mode queries throw; buffer attributes are left untouched |
| `APS5_NO_PAD_INPUT` | presence | `core/libs/prx/libSceVideoOut/src/PadInput.cpp` |  | ignores keyboard and mouse input so a measurement run is not advanced by a stray press |
| `APS5_NO_WGI` | presence | `core/libs/prx/libSceVideoOut/src/VideoOutDriver.cpp` |  | turns off SDL's Windows.Gaming.Input joystick backend at default hint priority, on Windows hosts where it faults a few seconds into every run |
| `APS5_OUTPUT_RESOLUTION` | string | `core/libs/prx/libSceVideoOut/src/VideoOutDriver.cpp` |  | `<width>x<height>` in pixels that replaces the host display's size when `sceVideoOutGetOutputStatus` and `sceVideoOutGetOutputStatusWithVrr` report the output resolution: 1 for a size below 3840 wide and 2160 high, 2 otherwise, as for the display; unset or empty follows the display, and any other value makes those calls throw |
| `APS5_UNPACED_FLIPS` | string | `core/libs/prx/libSceVideoOut/include/VblankSchedule.hpp` | 0 | benchmark aid: `1` completes each flip as soon as it is presented instead of at the next vblank, whatever the flip rate, so a title that renders faster than the vblank presents as fast as it can; the swapchain keeps the present mode `ANYPS5_PRESENT_MODE` selects (`fifo` by default, still held to the display's refresh), so set it to `immediate` or `mailbox` as well to present above it; flips keep their order, status, labels and events, while vblank events, `sceVideoOutWaitVblank` and the vblank status keep the output's cadence (`ANYPS5_VBLANK_HZ` included), so a title that times its simulation on flips runs faster than real time; `ANYPS5_FPS_LIMIT` still applies; any value other than `1`, `0` or empty stops the start |
| `APS5_VIDEOOUT_FLIP_TIMING` | string | `core/libs/prx/libSceVideoOut/src/Telemetry.cpp` |  | record completed main-output flips in `videoout.jsonl` for `FIRST:COUNT`; requires a telemetry directory, a positive first flip and a count from 2 to 100000; unset or empty disables flip timing |
| `APS5_VIDEOOUT_TELEMETRY_DIR` | string | `core/libs/prx/libSceVideoOut/src/Telemetry.cpp` |  | directory where videoout.jsonl is appended with the pad and present telemetry lines; nothing is opened when unset or empty |

### libSceVideodec2

| Name | Kind | Source | Default | Description |
|------|------|--------|---------|-------------|
| `APS5_TRACE_VIDEODEC` | presence | `core/libs/prx/libSceVideodec2/Export.cpp` |  | logs decoder memory queries, queue allocation and the first few decodes |

### libc

| Name | Kind | Source | Default | Description |
|------|------|--------|---------|-------------|
| `ANYPS5_LOG_FILE` | string | `core/libs/prx/libc/src/LogFilePlatform.cpp` |  | path of a file that also receives everything the converted executable writes to standard output and standard error, appended line by line, while the terminal output goes on; relative paths resolve against the working directory; an empty value is the same as unset; a file that cannot be opened stops the executable at load with status 1 and an error naming the variable, the path and the operating system error |
| `APS5_ALLOW_UNFRAMED_FAULTS` | string | `core/libs/prx/libc/include/RedZoneGate.hpp` | 0 | on Windows, any value other than empty or `0` keeps page tracking (`APS5_PAGE_TRACK`) and read traps (`APS5_STAGING_READ_TRAP`) on although the executable or a loaded guest module lacks the red-zone framing mark that `--frame-red-zone` writes; a fault on a guest thread can then overwrite the values a leaf function keeps below RSP, so it is for measurements on a title not converted again yet; it also lets the title go on after a fault served inside a range the relinker listed as unframed, which stops it otherwise. Such a title stops at the first page protection unless this is set |
| `APS5_ARM_READ_IMPORTS` | string | `core/libs/prx/libc/include/WindowsPageTrack.hpp` | 0 | with `APS5_PAGE_TRACK=1`, `1` lets page tracking protect a 64 KiB block that the GPU imported, as long as the AGC driver announced no GPU write into it: the import opens the block, the block may be protected again afterwards, and a GPU write the driver records or stamps there opens it and bars it until every batch opened before that announcement has finished on the GPU and two frames have passed (`APS5_NO_GPU_WRITE_MARK_EXPIRY=1` bars it until it is unmapped), and a block of a range the driver stopped watching stays barred until the driver watches it again; with `APS5_VERIFY_PAGE_TRACK=1` a protected imported block whose bytes change without a fault is counted as a divergence and barred until it is unmapped instead of stopping the run; unset, empty or `0` keeps every imported block unprotected, any other value fails |
| `APS5_COLLECT_REFUSALS` | presence | `core/libs/prx/libc/include/WindowsMappings.hpp` |  | on Windows, counts the collects of guest memory and the collects refused, by cause (no-access view, compared page, unwatched resident view, outside private allocations, uncommitted private memory), by role of the asking thread (game, committer, finisher, queue worker, walker, presenter) and by 64 KiB block; at the first 1024th collect after 10 s it prints to stderr a `[collect-refusals]` line with the collects, the reserved holes `APS5_PARTIAL_COLLECT` skipped since start, each cause with its count per role, the count of refused blocks and the 16 most refused blocks with their causes and views (resident, shared, compared, aliased, no-access; private or not); nothing else changes |
| `APS5_CRASH_INDIRECT_MEMORY` | presence | `core/libs/prx/libc/src/specifics/windows/CrashReport.cpp` |  | adds to the Windows crash report up to 64 bytes from each pointer in the dump of the words the registers point at, as `[[reg+off]] @addr` lines; off when unset or empty |
| `APS5_CRASH_STACK_BYTES` | integer | `core/libs/prx/libc/src/specifics/windows/CrashReport.cpp` | 0 | bytes of the faulting thread's stack from rsp, decimal or 0x hexadecimal and at most 1048576, that the Windows crash report prints with the data at the stack words pointing outside the stack; 0 prints none |
| `APS5_FAULT_STORM` | integer | `core/libs/prx/libc/src/specifics/windows/FaultStorm.cpp` |  | on Windows, starts the red-zone fault storm: a thread stops a thread that runs guest code (the executable or a `.guest.prx` module), sets its trap flag and resumes it, so each following instruction raises a single step delivered on the guest stack; the value is the landings per second, `1` meaning 2000, and 0 or a value that does not start with a digit leaves it off. Every step fills the 128 bytes below RSP with `0xa5`, which a leaf that keeps values there without a frame reads back. A thread whose debug registers hold an enabled breakpoint, as when a guest signal waits for it to come back to guest code, gets no trap flag, and a burst ends at the first step that finds one; both are counted in the 10 s line, since a step taken with that breakpoint armed is handled by nobody and stops the process with `0x80000004` |
| `APS5_FAULT_STORM_BURST` | integer | `core/libs/prx/libc/src/specifics/windows/FaultStorm.cpp` | 256 | with `APS5_FAULT_STORM`, instructions stepped in a row at each landing, at least 1 |
| `APS5_FAULT_STORM_CLOBBER_RANGE` | string | `core/libs/prx/libc/src/specifics/windows/FaultStorm.cpp` |  | with `APS5_FAULT_STORM`, `<image>:<begin>-<end>` with hexadecimal offsets into the named loaded image (`game.exe`, `libc.prx.guest.prx`): the storm fills the bytes below RSP only at steps inside that range, to find the function that loses its values |
| `APS5_FAULT_STORM_GUEST_CLOBBER` | presence | `core/libs/prx/libc/src/specifics/windows/FaultStorm.cpp` |  | with `APS5_FAULT_STORM`, fills the bytes below RSP only at steps in guest code, not at steps in the host libraries the guest calls |
| `APS5_FAULT_STORM_NO_CLOBBER` | presence | `core/libs/prx/libc/src/specifics/windows/FaultStorm.cpp` |  | with `APS5_FAULT_STORM`, leaves the bytes below RSP to the exception frame Windows writes; on a processor with AVX-512 that frame reaches them only while the thread has ZMM16 to ZMM31 state in use |
| `APS5_GUEST_COPY_SYSCALL` | presence | `core/libs/prx/libc/src/specifics/linux/GuestFaultCopy.cpp` |  | on Linux x86-64, checked guest copies use `process_vm_readv`/`process_vm_writev` instead of a copy whose faults a signal handler catches; read once, at the first copy |
| `APS5_HANG_DUMP_SECS` | integer | `core/libs/prx/libc/src/specifics/windows/CrashReport.cpp` |  | on Windows, whole seconds after which a watchdog dumps all threads and aborts; 0 or a value that is not a decimal number is ignored |
| `APS5_HEAP_CACHE_MIB` | integer | `core/libs/prx/libc/src/GuestHeap.cpp` | 4096 | size in MiB of the cache of freed large blocks before the oldest are decommitted |
| `APS5_HELPER_AFFINITY_MASK` | integer | `core/libs/prx/libc/include/CpuTopology.hpp` |  | hex core mask that helper threads (deferred release, mirror refresh) are pinned to |
| `APS5_LOG_TIMESTAMPS` | presence | `core/libs/prx/libc/src/LogFile.cpp` |  | prefixes each log line with `[+S.UUUUUU s]`, the seconds with microsecond digits since the clock started at the first stamped line |
| `APS5_MAPPINGS_PTHREAD_LOCK` | presence | `core/libs/prx/libc/include/WindowsMappings.hpp` |  | guards the Windows guest mapping tables with a `std::mutex` instead of a slim reader/writer lock taken exclusively; read once, when the tables are created |
| `APS5_NO_DIRECT_GUEST_COPY` | presence | `core/libs/prx/libc/src/specifics/linux/GuardedCopy.cpp` |  | on Linux x86-64, turns off what is on by default there: the checked guest copies (`GuestCopyIn_nid_no_patch`, `GuestCopyInUnsynced_nid_no_patch`, `GuestCopyOut_nid_no_patch`, the driver's readability probe) copy with plain loads in one routine whose faults the SIGSEGV and SIGBUS handlers of libc and libkernel turn into a failed copy; set, every checked read is a `process_vm_readv` system call again |
| `APS5_NO_HOST_REGION_INDEX` | presence | `core/libs/prx/libc/src/GuestArena.cpp` |  | tests a range against every host region of the arena interval in turn instead of searching the sorted, merged region index built at startup, on Windows |
| `APS5_NO_KNOWN_ZERO_FILTER` | presence | `core/libs/prx/libc/include/WindowsMappings.hpp` |  | takes the mapping lock for every known-zero forget (`GuestArenaForgetKnownZero_nid_no_patch`) instead of returning without it when the per-block counts show no known-zero page in the 64 KiB blocks of the range (ranges of up to 256 MiB) |
| `APS5_NO_PARTIAL_COLLECT` | presence | `core/libs/prx/libc/include/WindowsMappings.hpp` |  | turns off the default by which, on Windows, a collect of guest writes that meets address space outside every view and private allocation skips it, up to the next view or private allocation, when it is only reserved (neither mapped nor an image), instead of being refused as outside private allocations; the skipped holes are remembered until the mappings change, `APS5_VERIFY_DIRECT_COLLECT` throws when a remembered hole holds memory again, and the `[collect-refusals]` line of `APS5_COLLECT_REFUSALS` counts the holes skipped since start |
| `APS5_NO_PROCMAP_QUERY` | presence | `core/libs/prx/libc/include/specifics/linux/HostMapping.hpp` |  | skips the /proc/self/maps PROCMAP_QUERY ioctl and scans the maps file instead |
| `APS5_NO_RANGED_HOST_WRITES` | presence | `core/libs/prx/libc/include/WindowsMappings.hpp` |  | makes the opening and the closing of a host write or host import of guest memory wait for every collect that is outside the mapping lock, and keeps collects from leaving the lock meanwhile, instead of waiting only for those over the 64 KiB blocks of the written range |
| `APS5_NO_SSE4A_EMULATION` | presence | `core/libs/prx/libc/src/specifics/windows/CrashReport.cpp` |  | disables emulation of AMD-only SSE4a instructions, making such a fault fatal |
| `APS5_NO_UNLOCKED_COLLECT` | presence | `core/libs/prx/libc/include/WindowsMappings.hpp` |  | keeps the mapping lock across the `GetWriteWatch` call of a private-memory collect instead of releasing it for the call while the collected blocks stay reserved for that collect |
| `APS5_NO_UNLOCKED_HOST_WRITES` | presence | `core/libs/prx/libc/include/WindowsMappings.hpp` |  | takes the mapping lock at the opening and at the closing of every host write of guest memory (`GuestArena::HostWrite`, `GuestCopyOut_nid_no_patch`) instead of opening without it a write of at most 64 KiB whose pages hold no shared view, no read-only view, no owned-memory proof, no known-zero page and, with `APS5_PAGE_TRACK=1`, no protected block |
| `APS5_NO_UNLOCKED_MAPPING_READS` | presence | `core/libs/prx/libc/include/WindowsMappings.hpp` |  | takes the mapping lock for every owned-memory proof check (`GuestArenaOwnedUnchanged_nid_no_patch`), keeping it across the `GetWriteWatch` call of each 64 KiB block, for every logical protection query (`GuestArenaProtection_nid_postfix`) and for every list of compared pages (`GuestArenaComparedPages_nid_no_patch`, `GuestArenaTakeUnreportedPages_nid_no_patch`), instead of answering without the lock from the published owned words, view protections and compared marks of the views (an empty list while no page was ever marked compared; a list of unreported pages that would take some still goes under the lock) |
| `APS5_NO_WRITE_WATCH` | presence | `core/libs/prx/libc/src/GuestArena.cpp` |  | disables write watching so guest buffers are fully re-read on every use |
| `APS5_PAGE_TRACK` | string | `core/libs/prx/libc/include/WindowsPageTrack.hpp` | 0 | on Windows, `1` tracks writes to cold 64 KiB blocks of private and resident guest memory by page protection, unless the executable or a loaded guest module lacks the red-zone framing mark of `--frame-red-zone` (one `[red-zone]` line names the image; `APS5_ALLOW_UNFRAMED_FAULTS`): a block that a collect found clean in an earlier frame is made read-only, answers later collects without `GetWriteWatch` while it stays clean, and goes back to the write watch at its first write fault; pinned, host-written, imported and reprotected blocks are never protected; a reader of guest memory can also have a block protected at once (`GuestPageTrackArm`), whatever its collects found, unless it faulted twice in the frame or rests after faults in three of eight frames; the `[collects]` line of `APS5_PIPELINE_STATS` counts the blocks, faults and answers; unset, empty or `0` keeps the write watch alone, any other value fails |
| `APS5_PIN_WAIT_MS` | integer | `core/libs/prx/libc/src/GuestAllocations.cpp` | 60000 | milliseconds to wait for a pinned allocation lease before failing |
| `APS5_REPORT` | string | `core/libs/prx/libc/src/RunReport.cpp` | 1 | `0` turns the run report (`anyps5-report.json`) off and `1` turns it on, including for a process that is not a converted game; the `--aps5-report=on\|off` launch flag wins over it; any other value is ignored with a warning |
| `APS5_TRACE_AFFINITY` | presence | `core/libs/prx/libc/include/CpuTopology.hpp` |  | prints a line naming the role, tid and mask of every thread pinning |
| `APS5_TRACE_CALLS` | string | `core/libs/prx/libc/src/specifics/x86_64/CallTrace.cpp` |  | comma-separated libraries (or all) whose guest calls are traced; an empty import-library label uses the defining host module filename when it can be found |
| `APS5_TRACE_CALLS_FILE` | string | `core/libs/prx/libc/src/specifics/x86_64/CallTrace.cpp` |  | file rewritten every 10 seconds with each traced import called so far and its number of calls |
| `APS5_TRACE_EXIT` | presence | `core/libs/prx/libc/src/Process.cpp` |  | prints a backtrace of the caller of exit, _Exit and the process-exit path |
| `APS5_TRACE_SSE4A` | presence | `core/libs/prx/libc/src/specifics/windows/CrashReport.cpp` |  | prints each emulated SSE4a instruction once per rip |
| `APS5_VERIFY_DIRECT_COLLECT` | presence | `core/libs/prx/libc/include/WindowsMappings.hpp` |  | queries the regions of each private-memory collect read from the allocation map and throws when that collect succeeded over memory that is no longer committed private; with `APS5_PARTIAL_COLLECT`, also throws when a reserved hole a collect skipped holds memory again |
| `APS5_VERIFY_MAPPING_READS` | presence | `core/libs/prx/libc/include/WindowsMappings.hpp` |  | unless `APS5_NO_UNLOCKED_MAPPING_READS` is set, checks under the mapping lock every list of compared pages (`GuestArenaComparedPages_nid_no_patch`) and every logical protection (`GuestArenaProtection_nid_postfix`) answered without it, prints a `mapping reads:` line for each difference, without throwing, and, at the first check after 10 s, a `[mapping-reads]` line with the answers that matched, those that differed and those not compared because a mapping changed in between |
| `APS5_VERIFY_PAGE_TRACK` | string | `core/libs/prx/libc/include/WindowsPageTrack.hpp` | 0 | `1` turns `APS5_PAGE_TRACK` on and checks every collect that a protected block answers as clean: the write watch of the block must report no page and its bytes must equal those it had when it was protected, else the collect throws and the `[collects]` line counts a divergence |
| `APS5_WATCH_ADDR` | integer | `core/libs/prx/libc/src/specifics/windows/CrashReport.cpp` |  | absolute hex address whose writes are reported after its page is committed |
| `APS5_WATCH_WRITE` | integer | `core/libs/prx/libc/src/specifics/windows/CrashReport.cpp` |  | guest ELF vaddr whose 8-byte writes are reported by single-stepping the protected page |
| `APS5_WRITE_WATCH_FAULT_AHEAD` | integer | `core/libs/prx/libc/src/GuestWriteWatch.cpp` | 32 | on macOS, at most this many pages, up to the end of the faulting page's 64 KiB block, that a write fault unprotects and records as written when the two pages before it were written since the last collection; 0 or 1 unprotects only the faulting page |
| `APS5_WRITE_WATCH_ORACLE` | string | `core/libs/prx/libc/src/GuestWriteWatch.cpp` |  | on Linux, any value but `verify` makes the write watch ask nothing of the system: the host program (`agc_frame_replay`) reports every guest write it makes, before and after it, and only the reported pages read as written, which also works where `userfaultfd` is refused (a Docker container); `verify` keeps the system's tracking and looks every page it finds written up in the host's reports |

### libkernel

| Name | Kind | Source | Default | Description |
|------|------|--------|---------|-------------|
| `ANYPS5_LOADEXEC_ARGV_SKIP` | string | `core/libs/prx/libkernel/System/src/Process.cpp` |  | on Windows, set to `1` to drop the first process argument and clear the variable |
| `APS5_COARSE_TIMED_WAITS` | presence | `core/libs/prx/libkernel/Time/TimedWait.cpp` |  | on Windows, rounds timed waits up to the scheduler tick |
| `APS5_DIRECT_MEMORY_FROM_FLEXIBLE` | presence | `core/libs/prx/libkernel/DirectMemory/Export.cpp` |  | reports 12736 MiB less the configured flexible-memory size as the direct-memory size, and stops direct-memory allocation there, instead of 13824 MiB |
| `APS5_GAME_MODE` | string | `core/libs/prx/libkernel/System/src/HostPerformanceProcess.cpp` | off | asks the host for its game or performance mode while the title runs, read when libkernel loads: `off`, `on` or `auto`; on Linux it requests Feral GameMode (`libgamemode.so.0`, loaded with dlopen) and on macOS it holds a user-initiated, latency-critical `NSProcessInfo` activity; `on` logs a missing library or backend once, `auto` stays silent about it, and a refused request is logged once in both; nothing here is fatal, and any other value stops the title with an error naming the variable |
| `APS5_JOB_AFFINITY` | presence | `core/libs/prx/libkernel/Pthread/src/Thread.cpp` |  | on Windows, pins the title's job worker threads to the efficiency cores of a hybrid CPU |
| `APS5_JOB_AFFINITY_MASK` | integer | `core/libs/prx/libkernel/Pthread/src/Thread.cpp` |  | on Windows, hex core mask that job worker threads are pinned to |
| `APS5_MIMALLOC` | presence | `core/libs/prx/libkernel/System/src/MimallocRestart.cpp` |  | makes the process allocate through mimalloc instead of the system allocator: on Windows the entry stub (`core/relinker/elfpatcher/src/windows/WindowsEntryStubBuilder.cpp`) loads `mimalloc.dll`, on macOS libkernel restarts the title with `libmimalloc.dylib` in `DYLD_INSERT_LIBRARIES` and on Linux with `libmimalloc.so` in `LD_PRELOAD`; on macOS and Linux both variables stay set after the restart, so processes the title starts inherit them |
| `APS5_MIMALLOC_RESTARTED` | presence | `core/libs/prx/libkernel/System/src/MimallocRestart.cpp` |  | on macOS and Linux, set by libkernel for the restart that loads mimalloc and removed at start-up, so a restart that did not load it is not repeated |
| `APS5_NO_DIRECT_MEMORY_SECTION_POOL` | presence | `core/libs/prx/libkernel/DirectMemory/DirectMemory.cpp` |  | on Windows, creates a section for every direct-memory allocation and closes it at release instead of reusing released sections that never held guest data |
| `APS5_NO_JOB_AFFINITY` | presence | `core/libs/prx/libkernel/Pthread/src/Thread.cpp` |  | on Windows, disables pinning of job worker threads |
| `APS5_NO_SLEEP_WAKE` | presence | `core/libs/prx/libkernel/Time/SleepWake.cpp` |  | keeps every sleep of 1 ms or less in full instead of ending it at a GPU batch, batch completion action or flip completion |
| `APS5_NO_TIMER_RESOLUTION` | presence | `core/libs/prx/libkernel/Time/Time.cpp` |  | keeps the default 15.6 ms scheduler tick instead of raising it to 0.5 ms on Windows |
| `APS5_TIME_SCALE` | string | `core/libs/prx/libkernel/Time/Time.cpp` | 1 | factor applied to the guest monotonic clocks and TSC; sleeps and host JobManager worker spin intervals still take real time; fractional factors work, and a value that is not above 0 uses 1 |
| `APS5_TRACE_APR` | presence | `core/libs/prx/libkernel/Apr/src/Apr.cpp` |  | logs every APR file read with its path, offset, size and destination |
| `APS5_TRACE_EQUEUE` | presence | `core/libs/prx/libkernel/Equeue/Equeue.cpp` |  | logs when an equeue drops the oldest pending events past its 64-event cap, rate-limited to the first drop and each power-of-two of the per-queue cumulative total |
| `APS5_TRACE_MEMORY` | presence | `core/libs/prx/libkernel/DirectMemory/DirectMemory.cpp` |  | logs direct-memory map, remap, protect, unmap and physical allocation operations |
| `APS5_TRACE_SYNC` | presence | `core/libs/prx/libkernel/EventFlag/src/EventFlag.cpp` |  | logs event-flag and semaphore operations |
| `APS5_TRACE_TIMERS` | presence | `core/libs/prx/libkernel/Time/Time.cpp` |  | on Windows, runs a timed-wait selftest at startup and prints mean, p50 and max latencies |
| `APS5_TRACE_USLEEP` | presence | `core/libs/prx/libkernel/Time/Time.cpp` |  | reports every 2000 `sceKernelUsleep` calls the call sites that sleep most, including calls through `sceNetUsleep` and the `APS5_TRACE_TIMERS` measurement on Windows; guest `usleep` calls are not counted |
| `APS5_TRACE_WAITS` | presence | `core/libs/prx/libkernel/Time/include/Time.hpp` |  | aggregates blocking waits per guest call site and prints the longest-waiting sites every 3 s |
