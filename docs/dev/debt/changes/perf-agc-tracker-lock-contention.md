# libc

## add Functional

- [Known-zero forget filter](../../../core/libs/prx/libc/include/WindowsMappings.hpp) (libc) - `GuestArenaForgetKnownZero_nid_no_patch` returns without the mapping lock only when no 64 KiB block of the range holds a known-zero page, by a count per block kept in tables of 4 GiB allocated on first use and never freed. A range in a block that still holds another known-zero page takes the lock as before, as does a range of more than 4096 blocks or one above the 47-bit address range the tables cover. Measured only on replays of captured frames. `APS5_NO_KNOWN_ZERO_FILTER=1` takes the lock for every forget. Closes when: measured in a title.
