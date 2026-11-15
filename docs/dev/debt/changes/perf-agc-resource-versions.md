# libSceAgcDriver

## add Opt-in draw paths

- [Resource write versions](../../../core/libs/prx/libSceAgcDriver/Execution/src/GuestMemory.cpp) - a draw proof is refused only when a write version of its own ranges moves (swept once per collect epoch, so a CPU store is seen at the next one), by a non-draw commit that lists no writes (seals, ends of pipe, submission ends and completions excepted), by a write-back of a build that stores through BDA, or by a move of the depth surfaces, unwatch or host import serials; a GPU-lock hold of another thread no longer refuses it. While it holds, the multisample-target assertions of the full check are not repeated. In-place regions see pending-registry changes and named writes, not CPU or driver stores. A template or target with a range outside the write watch keeps the global stamp. Measured on replays of one title only. `APS5_NO_RESOURCE_VERSIONS=1` restores the global stamp.
