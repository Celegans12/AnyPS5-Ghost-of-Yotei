#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_TOOLS_RTPROBECACHE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_TOOLS_RTPROBECACHE_HPP

#include <cstdint>
#include <iosfwd>
#include <vector>

namespace AgcDriver {

struct CachedShaderCode {
    std::vector<std::uint32_t> code{};
    const char* error = nullptr;
};

CachedShaderCode ReadCachedShaderCode(std::istream& cache, std::uint64_t hash);

} // namespace AgcDriver

#endif
