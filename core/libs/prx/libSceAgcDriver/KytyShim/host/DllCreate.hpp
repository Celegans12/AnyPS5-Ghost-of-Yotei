#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_HOST_DLLCREATE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_HOST_DLLCREATE_HPP

#include "prx/libSceAgcDriver/KytyShim/include/KytyDll.h"

#include <stdexcept>
#include <string>

namespace KytyShim {

inline const KytyDllApi* CreateDll(KytyDllCreateFunction create, const KytyDllHost& host) {
    const KytyDllApi* created = nullptr;
    const KytyDllStatus status = create(&host, &created);
    if (status == KYTY_DLL_OK && KytyDllApiMatches(created)) return created;
    const std::string ours = "ABI version " + std::to_string(KYTY_DLL_ABI_VERSION) + " (" + std::to_string(sizeof(KytyDllHost)) + "-byte host table, " + std::to_string(sizeof(KytyDllApi)) + "-byte API table)";
    if (status == KYTY_DLL_ABI_MISMATCH) throw std::runtime_error("libSceAgcDriver: kyty_gpu.prx was built for another KytyDll.h: it refused libSceAgcDriver's " + ours);
    if (status != KYTY_DLL_OK) throw std::runtime_error("libSceAgcDriver: kyty_gpu.prx did not start (status " + std::to_string(status) + ")");
    if (created == nullptr) throw std::runtime_error("libSceAgcDriver: kyty_gpu.prx returned no API table");
    throw std::runtime_error("libSceAgcDriver: kyty_gpu.prx was built for another KytyDll.h: its API table has ABI version " + std::to_string(created->abi_version) + " and " + std::to_string(created->size) + " bytes, libSceAgcDriver needs " + ours);
}

}

#endif
