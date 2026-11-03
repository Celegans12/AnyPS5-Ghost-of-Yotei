#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_WORKERAFFINITY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_WORKERAFFINITY_HPP

#include <cstdint>

namespace AgcDriver::DriverDetail {

std::uint64_t WorkerAffinityMask();

void PinWorkerThread(const char* role);

// Thread pinning (Windows, non-hybrid CPUs with at least 6 physical cores; APS5_THREAD_PINNING=1, an opt-in
// APS5_AB switch): the queue-0 worker, the draw committer and the draw preparer each run alone on one
// logical CPU of its own physical core (the cores at 1/4, 1/2 and 3/4 of the core list; their SMT siblings
// stay idle), and every other thread of the process on the remaining cores. They hand work to each other
// every draw, so sharing a core with each other or with game threads, or migrating, slows the frame.
// Turning it off restores each thread's mask from before its first placement.
enum class CriticalThread { Queue0Worker, Committer, Preparer };

// Records the calling thread as `role`; the placement itself is applied by FollowThreadPinning.
void NoteCriticalThread(CriticalThread role);

// The queue-0 worker at a present: applies a change of the switch at once, and while pinning places every
// thread again at most once a second (new threads start on the process mask).
void FollowThreadPinning();

}

#endif
