#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_FAULTSTORM_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_FAULTSTORM_HPP

#ifdef _WIN32
#include <windows.h>

// Red-zone fault storm (rewrite phase 1 test). APS5_FAULT_STORM=<steps per second> (1 means 2000)
// starts a thread that stops a random thread running guest code, sets its trap flag and resumes
// it: the next instruction raises a single step, and Windows writes the exception frame, with the
// thread's AVX state, below its rsp. A leaf keeping values in the red zone then loses them unless
// the relinker framed it. Handle() clears the injected trap flag and continues.
namespace FaultStorm {

void Start();
bool Handle(EXCEPTION_POINTERS* info);

}
#endif

#endif
