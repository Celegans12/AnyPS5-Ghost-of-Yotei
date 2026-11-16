#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_FAULTSTORM_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_FAULTSTORM_HPP

#ifdef _WIN32
#include <windows.h>

namespace FaultStorm {

void Start();
bool Handle(EXCEPTION_POINTERS* info);

}
#endif

#endif
