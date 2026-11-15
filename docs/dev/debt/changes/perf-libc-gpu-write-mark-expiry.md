# libc

## add Functional

- [GPU write marks that expire](../../../core/libs/prx/libc/include/WindowsPageTrack.hpp) (libc, `APS5_ARM_READ_IMPORTS=1`, off by default) - a block is protected again once every GPU flight begun before the last GPU write announced into it has ended and two frames passed (`APS5_NO_GPU_WRITE_MARK_EXPIRY=1` turns this off). The proof covers only writes carried by a flight: `Recorder` batches and `CommandBatch` are, another command buffer is not unless its code begins and ends a flight. An announcement made before its batch opens relies on the GPU mutex to keep other batches from finishing in between. A block of a range the driver stopped watching stays unprotected until `Rewatch` covers the whole block, so imports that are never settled never come back. The two frames only avoid re-protecting a block the GPU writes every frame or two. Unmapping part of a block forgets the marks of the whole block. Not built or run.
