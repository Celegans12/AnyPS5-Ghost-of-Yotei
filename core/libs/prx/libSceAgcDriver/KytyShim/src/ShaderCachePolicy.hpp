#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_SRC_SHADERCACHEPOLICY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_SRC_SHADERCACHEPOLICY_HPP

#include <string_view>

namespace KytyShim {

inline constexpr bool ResolveShaderCacheEnabled(bool enabled, const char* disabled) {
    return enabled && (disabled == nullptr || *disabled == '\0' || std::string_view(disabled) == "0");
}

}

#endif
