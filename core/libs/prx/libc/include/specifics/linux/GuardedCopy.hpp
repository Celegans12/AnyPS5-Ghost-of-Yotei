#ifndef PRX_LIBC_INCLUDE_SPECIFICS_LINUX_GUARDEDCOPY_HPP
#define PRX_LIBC_INCLUDE_SPECIFICS_LINUX_GUARDEDCOPY_HPP

#include <csignal>
#include <cstddef>

extern "C" bool GuestCopyRecoverFault_nid_no_patch(int signal, siginfo_t* info, void* context);

namespace GuardedCopy {
bool Ready();
bool Copy(void* destination, const void* source, std::size_t bytes);
bool FaultsCaught();
}

#endif
