#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_HOSTFAULT_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_HOSTFAULT_HPP

#ifdef _WIN32
#include <cstdint>

using LibcHostFaultHandler = bool (*)(void* exceptionPointers) noexcept;

extern "C" void LibcSetHostFaultHandler_nid_no_patch(LibcHostFaultHandler handler) noexcept;
extern "C" std::uintptr_t LibcUnframedFaultSite_nid_no_patch(const void* exceptionPointers) noexcept;
extern "C" void LibcRefuseUnframedFault_nid_no_patch(std::uintptr_t site) noexcept;
#endif

#endif
