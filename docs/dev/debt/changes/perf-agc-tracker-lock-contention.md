# libc

## add Functional

- [Known-zero forget filter](../../../core/libs/prx/libc/include/WindowsMappings.hpp) (libc) - `GuestArenaForgetKnownZero_nid_no_patch` returns without the mapping lock only when no 64 KiB block of the range holds a known-zero page, by a count per block kept in tables of 4 GiB allocated on first use and never freed. A range in a block that still holds another known-zero page takes the lock as before, as does a range of more than 4096 blocks or one above the 47-bit address range the tables cover. Measured only on replays of captured frames. `APS5_NO_KNOWN_ZERO_FILTER=1` takes the lock for every forget. Closes when: measured in a title.

# libSceAgcDriver

## add Functional

- [Settled tracker generation](../../../core/libs/prx/libSceAgcDriver/Execution/src/GuestMemory.cpp) (libSceAgcDriver) - `SettledGeneration` reads the generation the last finished tracker update published and does not wait for an update in progress, so it can be older than the value `TrackerGeneration` would return after that update: a store the update is stamping reads as made after the value, as a store collected later always does. Only the draw bindings' unwritten-snapshot memo and `StorageTexture::markLayersPending` read it; every other `TrackerGeneration` caller still waits behind a walk. Covered by `agc_collect_memo_generation` and `APS5_VERIFY_DRAW_SNAPSHOTS` on replays of captured frames, not measured in a title. `APS5_NO_SETTLED_GENERATION=1` restores the wait. Closes when: measured in a title.
