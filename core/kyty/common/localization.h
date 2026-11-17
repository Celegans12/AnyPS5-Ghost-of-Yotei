#pragma once

#include <string_view>

namespace Common::Localization {

[[nodiscard]] std::string_view LocaleName(int console_language);

[[nodiscard]] std::string_view Get(int console_language, std::string_view key);

}
