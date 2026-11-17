#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_SRC_FAULTS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_SRC_FAULTS_HPP

#include <csignal>

namespace KytyShim {

void InstallFaultChain();
void RemoveFaultChain();
bool HandleHostFault(int signal, siginfo_t* info, void* context);

}

#endif
