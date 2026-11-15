# libSceAgcDriver

## add Opt-in draw paths

- [Staging read trap](../../../core/libs/prx/libSceAgcDriver/Graphics/src/Recorder.cpp) (libSceAgcDriver) - with `APS5_STAGING_READ_TRAP` (Windows), a thread that holds the GPU lock and touches a trapped page without the flush hook is not served: the fault is reported and ends the process. A `VirtualQuery` made directly, not through `GuestPageTrackQuery`, sees the page as inaccessible, and a kernel read or write that meets a page trapped again after the hook released it fails with `ERROR_NOACCESS`. A full sync and any draw pass land every kept copy-back, and a landed one outdates its resident shadow. Only written ranges made of whole 4 KiB pages of shared views with one guest mapping are guarded; a range imported in place is not. Never run; the only check is `APS5_VERIFY_STAGING_READ_TRAP`.

# libc

## add Functional

- [Read traps on shared views](../../../core/libs/prx/libc/include/WindowsMappings.hpp) (libc) - `GuestArenaTrapReads` removes access from the guest view of shared pages until a resolver registered by the AGC driver released them. A fault raised while the faulting thread holds the mapping lock deadlocks, as a write fault on an armed shared page does. A view unmapped while its resolver refused is dropped trapped. Windows only; `GuestWriteWatchTrapReads` serves the macOS label traps.
