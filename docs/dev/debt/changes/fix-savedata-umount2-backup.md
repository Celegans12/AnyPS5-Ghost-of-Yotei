# libSceSaveData.native

## add Functional

- [sceSaveDataUmount2](../../../core/libs/prx/libSceSaveData.native/Export.cpp) (libSceSaveData.native) - backs up only when bit 16 (`SAVE_DATA_UMOUNT_MODE_BACKUP`, `0x10000`) is set, as KytyPS5 (PS5, `src/libs/libSaveData.cpp`) does: the copy is synchronous and queues and delivers a `SAVE_DATA_EVENT_TYPE_UMOUNT_BACKUP` (`1`) event like `sceSaveDataUmountWithBackup`, without telling a forced backup (`SAVE_DATA_UMOUNT_MODE_BACKUP_FORCE`, `0x1`) from an ordinary one or skipping unmodified saves. Modes `0` and `1` (the latter passed by a PS5 title before its title screen) unmount without a backup or an event. Any other bit is reported once on stderr and ignored; what those bits do on a console is not known. Unlike `sceSaveDataUmountWithBackup`, `sceSaveDataUmount2` does not require `sceSaveDataInitialize3` before returning `SAVE_DATA_ERROR_NOT_MOUNTED`. Closes when: measured on a PS5. Labels: console, tested.
