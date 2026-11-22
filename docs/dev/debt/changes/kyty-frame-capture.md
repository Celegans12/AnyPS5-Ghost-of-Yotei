# libSceAgcDriver

## add Functional

- [Kyty frame capture](../../../core/libs/prx/libSceAgcDriver/Execution/src/FrameRecorder.cpp) requires the linked build and stable guest mappings. It stops when a GPU queue stays parked or when it cannot read a coherent range, including GPU-owned images without a guest copy. The frame trigger recognizes CPU flips and the flip packets detected in a graphics submission. It does not count a flip reached later through a REWIND tail. A 64-bit page checksum collision could hide a memory change. Closes when the recorder supports exact delta comparison, coherent GPU image snapshots, active mapping changes, deferred flips and the `kyty_gpu.prx` build.
