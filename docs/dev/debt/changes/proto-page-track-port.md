# libc

## add Functional

- On Windows [page tracking](../../../core/libs/prx/libc/include/WindowsPageTrack.hpp) (libc, `APS5_PAGE_TRACK=1`, off by default) write-protects cold 64 KiB blocks of private and resident guest memory. A kernel write into such a block outside `GuestArena::HostWrite` fails with `ERROR_NOACCESS`; a read still pending when its block is protected lands unseen, as it already escapes the write watch. Only blocks inside one host allocation are protected, so memory committed in 16 KiB granules stays on `GetWriteWatch`. Frames are counted by the AGC driver; without it nothing is protected. Measured on the replay bench on an earlier base only, not on this one and not in a running title.
