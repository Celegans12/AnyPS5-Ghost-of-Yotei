#include "prx/libc/include/GpuBackendSwitch.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

bool ReadGpuBackendSwitch() {
    const char* value = std::getenv("APS5_GPU_BACKEND");
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "anyps5") == 0) return false;
    if (std::strcmp(value, "kyty") == 0) {
        std::fprintf(stderr, "[gpu] backend: kyty (APS5_GPU_BACKEND=kyty, read once at start)\n");
        return true;
    }
    std::fprintf(stderr, "[gpu] APS5_GPU_BACKEND=%s names no backend (anyps5 or kyty)\n", value);
    throw std::invalid_argument(std::string("APS5_GPU_BACKEND names no backend: ") + value);
}

const bool g_gpuBackendRead = (LibcGpuBackendKyty_nid_no_patch(), true);

}

extern "C" bool LibcGpuBackendKyty_nid_no_patch() {
    static const bool kyty = ReadGpuBackendSwitch();
    return kyty;
}
