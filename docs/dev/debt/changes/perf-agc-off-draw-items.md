# libSceAgcDriver

## add Opt-in draw paths

- [Label marks per batch](../../../core/libs/prx/libSceAgcDriver/Execution/src/GuestMemory.cpp) (`APS5_NO_CORE_OFF_DRAW` turns it off) - on by default with the submit policy. A CPU write to a label's 64 KiB block made between two labels of one batch is no longer collected at the second label: it stays in the write watch until the next collect of its page, which stamps it as a CPU write with a later generation than the label's. Readers that collect their range before they trust a stamp see it; a reader that relied on the collect made at the label does not, and none was searched for beyond the tracker test. Only labels stored on the GPU and noted in the label table take the path; labels stored behind a completion and larger stores collect as before. Closes when: a title is measured with `APS5_VERIFY_LABEL_MARKS`, whose count of collects that found a write says how often the difference exists.
