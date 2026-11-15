# libSceAgcDriver

## add Opt-in draw paths

- [Draw lookup on the finisher](../../../core/libs/prx/libSceAgcDriver/Execution/src/Driver/Draw/Draw.cpp) (`APS5_FINISHER_LOOKUP`) - opt-in, written without a build: not compiled, and not run in its tests, on a replay of captured frames or in a title. The draw-cache lookup of a pipelined direct draw runs at the head of the draw's finisher job, so it reads guest memory later than the queue worker would, behind every ordered item queued before the draw. A draw one of whose programs stores to memory is queued without the ranges it writes even when the variant the lookup then finds writes its targets only, which drains the finisher at the next packet that reads. Left on the queue worker: draws without a register key, indirect draws, every draw with `APS5_FINISHER_THREADS` above 1, and dispatches. `APS5_VERIFY_FINISHER_LOOKUP` compares and counts and does not fail the driver. Closes when: measured in a title.
