#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_SRC_SETTINGS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_SRC_SETTINGS_HPP

#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace KytyShim::Settings {

[[nodiscard]] bool Read(std::string_view name);
[[nodiscard]] std::vector<std::string> UnreadLines(const std::vector<std::string>& environment);
[[nodiscard]] std::vector<std::string> EnvironmentEntries();
void ReportUnread(std::FILE* out);

}

#endif
