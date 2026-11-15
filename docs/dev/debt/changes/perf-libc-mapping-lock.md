# libc

## add Functional

- [Mapping reads without the lock](../../../core/libs/prx/libc/include/WindowsMappings.hpp) (libc) - `GuestArenaOwnedUnchanged_nid_no_patch` and `GuestArenaProtection_nid_postfix` answer without the mapping lock from two tables of atomics written under it, allocated on first use and never freed. A proof check that finds written pages still takes the lock for its block; a check of several blocks holds for the moment its first block was tested. A protection query made while a mapping changes answers the state before or after the change instead of waiting for it. `GuestArenaArmOwned`, `GuestArenaKnownZero`, pins, host writes and the fault handlers keep the lock across their system calls and wait for every collect outside it. Measured in no title. `APS5_NO_UNLOCKED_MAPPING_READS=1` takes the lock for both. Closes when: measured in a title.
