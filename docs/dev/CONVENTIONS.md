# Coding Conventions

### C++ Naming Conventions

- Interface names (classes containing only pure virtual methods with no implementation) start with `I`, as in C#.
- Class names and public method names use PascalCase.
- Private method and field names use camelCase.
- Function argument names use camelCase.
- Template parameter names follow the C# style: `TKey`, `TValue`, `TIterator`.

---

Comments in code can only be added to indicate areas of [technical debt](TechnicalDebt.md) (only by a human), the end of `#endif`, the end of namespace, and the licence and copyright notice of code adapted from an upstream source, as described in [AGENTS.md](../../AGENTS.md).

[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)

There's no need to add image files to the repository. You can add images to the [gist](https://gist.github.com/boykopovar/0e53f2e1426f29ecd41e3b51540b8a90) comments and paste a link into the md file.

### Host writes into guest memory

Host code that writes into guest memory through the kernel or a native library must hold a `GuestArena::HostWrite` ([GuestArena.hpp](../../core/libs/prx/libc/include/GuestArena.hpp)) over the destination range for the whole write. This covers file reads (`ReadFile`, `read`, `pread`), socket receives, `DeviceIoControl` and other calls that fill a caller's buffer.

The write watch can leave guest pages read-only between collections: every watched page on macOS, and on Windows the pages of shared views (shared memory, shared file mappings, and direct memory mapped at more than one guest address), which it tracks by write protection (see the write-watch and Windows direct-memory backing entries in [debt/libc.md](debt/libc.md)). A kernel-mode write into such a page doesn't fault into the watch. It fails instead, with `EFAULT` on macOS and `ERROR_NOACCESS` on Windows. `HostWrite` opens the pages and makes sure the write is recorded.

- A write that completes later on another thread (overlapped I/O, completion ports, asynchronous reads) is bracketed where the bytes land, not where the request was issued.
- Ordinary CPU stores from host C++ code need no bracket: the write watch records them (through a fault where the page is protected).
- A GPU import of guest memory uses `HostImportAccess`, not `HostWrite`.
- Third-party code that writes guest memory on its own (driver threads, library worker threads) can't be bracketed. List it in that library's debt file.

`python3 tools/guest_pointer_audit.py [library ...]` lists exported functions that read or write a guest pointer parameter directly (`->`, `[]`, `*`, or a raw `mem*`, `str*` or `std::copy` call) instead of through `GuestCopyIn_nid_no_patch`, `GuestCopyOut_nid_no_patch` or the `CopyGuest*` helpers in [General.hpp](../../core/libs/prx/libc/include/General.hpp), which report unmapped or unwritable memory instead of faulting. The error such an export returns is decided per function; `--json` prints the findings for scripts. It is a report, not a gate: it always exits 0, and it sees only accesses in the export's own body, so a pointer passed to a helper or copied to a local is not followed and an empty report does not prove a library safe.

### Page tracking tokens

With `APS5_PAGE_TRACK=1` on Windows, a reader that keeps a copy of guest memory can hold tokens of the page tracking table: `GuestPageTrackArm`, `GuestPageTrackRearm`, `GuestPageTrackTokens`, `GuestPageTrackCleanSince` and `GuestPageTrackReasons` in [GuestPageTrack.hpp](../../core/libs/prx/libc/include/GuestPageTrack.hpp), wrapped for the AGC driver as `ArmBlocks`, `RearmWritten`, `ArmedTokens`, `ArmedCleanSince` and `UnarmedReason` in [GuestMemory.hpp](../../core/libs/prx/libSceAgcDriver/Execution/include/GuestMemory.hpp). One token stands for one 64 KiB block. Every call takes one entry per block the range touches, entry 0 for the block that holds the range's address, and refuses any other count.

- A token is the word of an armed block: never 0, always below 2^63. A block that is not armed gets 0.
- A token holds while its block has stayed armed since the token was returned. A CPU write to the block faults first and ends it. So do a hold (`HostWrite`, a pin), a host import, a bar (`GuestPageTrackKeepOpen`, a protection other than read-write), any other protection change through the arena, and the unmapping of the block.
- A token that ended never holds again, and no later arming of the same address returns it: each arming advances a serial kept in the word, and the serial survives the unmapping.
- The reader takes the token first and reads the bytes after. A write that landed before the block was protected is in those bytes; a write after the token ends it. `GuestPageTrackRearm` names the 4 KiB pages written since a token only while no collect took them; otherwise it reports every page.
- A token says nothing about a kernel write that was pending when the block was protected, nor, by itself, about GPU writes, which go around the page tables. An imported block is therefore never armed, unless `APS5_ARM_READ_IMPORTS=1`: then it is armed while the GPU only reads it, and driver code that lets the GPU write guest memory must call `GuestMemory::AnnounceGpuWrite` over the range before a later reader can see the write. `Recorder::noteWrite`, the tracker's marks (`MarkWritten`, `MarkPossiblyWritten`) and its loss of coverage (`Unwatch`) already do; a GPU write that reaches guest memory by another path has to announce itself. The announcement opens the blocks, ends their tokens and bars them until they are unmapped.
- Arming protects the block first and resets its write watch after; the pages that reset reports stay pending for the collect that covers them, so the write tracker of the driver loses none.
- A request arms a block whatever its collects found, without the wait of 1 to 64 frames that a collect observes. Two limits remain, counted in the frames the AGC driver reports: a block that faulted twice in a frame is not armed again on request before the next frame, and a block that faulted in three of its last eight frames rests for 64 frames. `GuestPageTrackReasons` tells both apart, with the frames left, so that a reader can stop asking and treat the block as rewritten every frame.

### Host-only test executables

A host-only test executable (one that links no guest library) whose sources inherit the `-fno-asynchronous-unwind-tables` policy of `core/libs/CMakeLists.txt`, and that is built for MinGW, must call `configure_host_test_executable(<target>)` (root `CMakeLists.txt`) after `add_test_executable`. On MinGW only, it adds `-fasynchronous-unwind-tables` to the executable's compile options and `-static -static-libgcc -static-libstdc++` to its link options, and nothing else.

The guest `libc` provides `__gxx_personality_v0`; a host-only link doesn't get it. On a DWARF-exception MinGW toolchain an object with a landing pad then fails to link. Test executables defined in the root `CMakeLists.txt` do not inherit that policy, so they do not need the helper; nor do test executables that link a guest library, or non-MinGW hosts.
