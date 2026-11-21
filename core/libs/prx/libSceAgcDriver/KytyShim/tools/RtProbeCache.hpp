#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_TOOLS_RTPROBECACHE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_TOOLS_RTPROBECACHE_HPP

#include <cstdint>
#include <ios>
#include <iosfwd>
#include <vector>

namespace AgcDriver {

struct CachedShaderCode {
    std::vector<std::uint32_t> code{};
    const char* error = nullptr;
};

struct CachedShaderEntry {
    std::uint64_t hash = 0;
    std::uint32_t words = 0;
    std::uint32_t compressedBytes = 0;
    std::streamoff payload = 0;
};

struct CachedShaderIndex {
    std::vector<CachedShaderEntry> entries{};
    bool truncated = false;
    const char* error = nullptr;
};

CachedShaderCode ReadCachedShaderCode(std::istream& cache, std::uint64_t hash);
CachedShaderIndex IndexCachedShaderCode(std::istream& cache);
CachedShaderCode ReadIndexedShaderCode(std::istream& cache, const CachedShaderEntry& entry);

}

#endif
