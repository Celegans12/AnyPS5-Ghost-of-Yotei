#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_SRC_SYSTEMCONTENT_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_SRC_SYSTEMCONTENT_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace KytyShim::SystemContent {

struct Param {
    std::string name;
    bool isText = false;
    std::string text;
    std::int32_t number = 0;
};

std::optional<std::vector<Param>> ParseParamJson(std::string_view content);

}

#endif
