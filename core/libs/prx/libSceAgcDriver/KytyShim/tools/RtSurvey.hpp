#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_TOOLS_RTSURVEY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_TOOLS_RTSURVEY_HPP

#include <cstdint>
#include <string>

namespace AgcDriver {

struct SurveySettings {
    std::string cache;
    std::string manifest;
    std::uint32_t oracle = 1;
    std::uint32_t subgroup = 32;
    std::uint32_t ldsLimitBytes = 49152;
    bool validate = true;
    std::string build;
};

int RunSurvey(const SurveySettings& settings);

} // namespace AgcDriver

#endif
