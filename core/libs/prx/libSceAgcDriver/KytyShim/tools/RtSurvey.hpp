#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_TOOLS_RTSURVEY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_TOOLS_RTSURVEY_HPP

#include <cstdint>
#include <string>
#include <string_view>

namespace AgcDriver {

class SurveyLogScope {
public:
    using Observer = void (*)(void*, std::string_view) noexcept;

    SurveyLogScope(Observer observer, void* context) noexcept;
    ~SurveyLogScope();
    SurveyLogScope(const SurveyLogScope&) = delete;
    SurveyLogScope& operator=(const SurveyLogScope&) = delete;

    static bool Active() noexcept;
    static void Observe(std::string_view text) noexcept;

private:
    static thread_local const SurveyLogScope* current_;
    const SurveyLogScope* previous_;
    Observer observer_;
    void* context_;
};

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

}

#endif
