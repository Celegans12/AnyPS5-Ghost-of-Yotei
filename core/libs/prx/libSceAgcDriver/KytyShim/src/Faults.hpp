#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_SRC_FAULTS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_SRC_FAULTS_HPP

#ifndef _WIN32
#include <csignal>
#endif

namespace KytyShim {

void InstallFaultChain();
void RemoveFaultChain();
#ifdef _WIN32
bool HandleHostFault(void* exceptionPointers);
#else
bool HandleHostFault(int signal, siginfo_t* info, void* context);
#endif

}

#endif
