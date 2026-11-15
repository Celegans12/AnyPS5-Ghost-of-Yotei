# libSceAgcDriver

## add Opt-in draw paths

- [Staging read trap](../../../core/libs/prx/libSceAgcDriver/Graphics/src/Recorder.cpp) (libSceAgcDriver) - with `APS5_STAGING_READ_TRAP` (Windows), a guarded staging copy-back is kept on the GPU past label stores and submits and the guest view of its pages has no access. A thread that holds the GPU lock and touches such a page without the flush hook is not served: the fault is reported and ends the process, because the batch cannot be submitted from an arbitrary point of a recording. A `VirtualQuery` made directly, not through `GuestPageTrackQuery`, reports the page as inaccessible. A kernel read or write that starts after the hook released the page and after a new result trapped it again fails with `ERROR_NOACCESS`. A full sync lands every kept copy-back, and draws land all of them at their pass, so a title that reads no staged range still copies back once per draw pass that follows a dispatch. A landed copy-back outdates its resident shadow, so the next use copies the range in again. Only written ranges made of whole 4 KiB pages of shared views with one guest mapping are guarded; a range imported in place is not. Never run; the only check is `APS5_VERIFY_STAGING_READ_TRAP`.

# libc

## add Functional

- [Read traps on shared views](../../../core/libs/prx/libc/include/WindowsMappings.hpp) (libc) - `GuestArenaTrapReads` removes access from the guest view of shared pages until a resolver registered by the AGC driver released them. A fault raised while the faulting thread holds the mapping lock deadlocks, as a write fault on an armed shared page does. A view unmapped while trapped is dropped without calling the resolver again when the resolver refused. Not available on Linux and macOS, where `GuestWriteWatchTrapReads` serves the macOS label traps only.
