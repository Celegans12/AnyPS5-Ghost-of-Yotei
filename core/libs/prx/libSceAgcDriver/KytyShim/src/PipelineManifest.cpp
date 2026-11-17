#include "kernel/fileSystem.h"
#include "loader/guestOodle.h"
#include "prx/libc/include/General.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <span>
#include <stop_token>
#include <string>

namespace Libs::LibKernel::FileSystem {

std::filesystem::path GetRealFilename(const std::string& mounted_file_name) {
    return ResolveDiskPath_nid_no_patch(mounted_file_name.c_str());
}

}

namespace Loader::GuestOodle {

bool Decompress(std::span<const uint8_t>, std::span<uint8_t>, const std::stop_token&) {
    static std::atomic<bool> told{false};
    if (!told.exchange(true)) {
        std::fprintf(stderr, "[kyty] Loader::GuestOodle::Decompress: AnyPS5 has no Oodle decompressor, so the pipeline "
                             "manifest scan of the game files (pipelineManifest.cpp) skips every Oodle-compressed "
                             "shader group\n");
        std::fflush(stderr);
    }
    return false;
}

}
