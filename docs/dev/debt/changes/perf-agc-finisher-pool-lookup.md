# libSceAgcDriver

## add Opt-in draw paths

- [Draw lookup on the finisher pool](../../../core/libs/prx/libSceAgcDriver/Execution/src/Driver/Draw/Draw.cpp) (`APS5_FINISHER_LOOKUP` with `APS5_FINISHER_THREADS` above 1) - opt-in, written without a build: not compiled, and not run in its tests, on a replay of captured frames or in a title. The lookup of a draw runs beside the jobs queued before it, so a lookup that reads a range one of them writes waits for its turn and for the committer, and which variant of an entry a hit promotes, and which hit moves an entry in the eviction order, depend on the order the lookups finish in. `APS5_VERIFY_FINISHER_LOOKUP` compares and counts and does not fail the driver. Closes when: measured in a title.
