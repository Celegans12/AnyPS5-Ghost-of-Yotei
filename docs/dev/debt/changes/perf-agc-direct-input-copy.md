# libSceAgcDriver

## add Opt-in draw paths

- [Dirty-first input copies](../../../core/libs/prx/libSceAgcDriver/Graphics/src/BufferCache.cpp) (`APS5_NO_DIRECT_INPUT_COPY`) - on by default with `APS5_DIRTY_FIRST_INPUTS` and written without a build: not compiled, not run in its test, not measured in a title. Only Windows copies in place; Linux and macOS keep one system call per copy. The guard recovers an access violation and an in-page error of its own loads only: a guard page inside a range the guest arena owns raises another exception and is not recovered, on the ground that no code of the libraries sets one there. A range that touches a host region of the arena, or lies outside it, takes the system copy. A copy that faults is tried again through the system copy, so an unreadable range costs a fault and a system call, and it is refused as before. Closes when: built, its test passes and a title is measured with it and with `APS5_VERIFY_DIRTY_FIRST_INPUTS`.
