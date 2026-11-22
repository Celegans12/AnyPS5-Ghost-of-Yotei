#include "prx/libSceAgcDriver/KytyShim/src/Settings.hpp"

#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace {

using KytyShim::Settings::EnvironmentEntries;
using KytyShim::Settings::Read;
using KytyShim::Settings::ReportUnread;
using KytyShim::Settings::UnreadLines;

int failures = 0;

void Expect(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAILED: %s\n", what);
    ++failures;
}

std::size_t Count(const std::string& text, std::string_view fragment) {
    std::size_t count = 0;
    for (auto at = text.find(fragment); at != std::string::npos; at = text.find(fragment, at + fragment.size())) ++count;
    return count;
}

void Names() {
    Expect(Read("KYTY_HDR"), "a shim option is read");
    Expect(Read("KYTY_AB") && Read("KYTY_AB_START") && Read("KYTY_AB_SETTLE"), "the in-run A/B settings are read");
    Expect(Read("KYTY_GENERIC_COMPUTE_VARIANT"), "the runtime switch is read");
    Expect(Read("KYTY_DIRECT_MEMORY_PAGES"), "a name libkernel reads is read");
    Expect(Read("KYTY_PGO_DIR"), "a name kyty_gpu.prx reads is read");
    Expect(!Read("KYTY_FLAT_DEFER"), "a removed switch is not read");
    Expect(!Read("KYTY_FULLSCREEN"), "a Kyty option AnyKyty ignores is not read");
    Expect(!Read("KYTY_"), "the bare prefix is not read");
    Expect(!Read("kyty_hdr"), "names are compared exactly");
}

void Lines() {
    const auto lines = UnreadLines({"PATH=/bin", "KYTY_HDR=1", "KYTY_NATIVERT=0", "KYTY_FLAT_DEFER=1", "KYTY_FULLSCREEN", "KYTY_AB=40:KYTY_GENERIC_COMPUTE_VARIANT", "KYTY_FLAT_DEFER=0", "APS5_KYTY_X=1"});
    const std::vector<std::string> expected = {
        "[kyty] KYTY_FLAT_DEFER is set but not read by this build\n",
        "[kyty] KYTY_FULLSCREEN is set but not read by this build\n",
        "[kyty] KYTY_NATIVERT is set but not read by this build\n",
    };
    Expect(lines == expected, "each unread KYTY_* name gets one line, in name order");
    Expect(UnreadLines({"KYTY_HDR=1", "KYTY_SHADER_CACHE=0", "KYTY_AB=40:KYTY_GENERIC_COMPUTE_VARIANT", "ANYPS5_GPU=RTX"}).empty(), "known names print nothing");
    Expect(UnreadLines({}).empty(), "an empty environment prints nothing");
}

void Environment() {
    bool unknown = false;
    bool known = false;
    for (const auto& entry : EnvironmentEntries()) {
        unknown = unknown || entry == "KYTY_SETTINGS_TEST_UNREAD=1";
        known = known || entry == "KYTY_HDR=0";
    }
    Expect(unknown && known, "the environment of the test holds both variables");
    std::FILE* file = std::tmpfile();
    Expect(file != nullptr, "a temporary file opens");
    if (file == nullptr) return;
    ReportUnread(file);
    ReportUnread(file);
    std::rewind(file);
    std::string text;
    for (int c = std::fgetc(file); c != EOF; c = std::fgetc(file)) text.push_back(static_cast<char>(c));
    std::fclose(file);
    Expect(Count(text, "[kyty] KYTY_SETTINGS_TEST_UNREAD is set but not read by this build\n") == 1, "the unread variable is reported once");
    Expect(Count(text, "KYTY_HDR") == 0, "the read variable is not reported");
}

}

int main(int argc, char** argv) {
    if (argc > 1 && std::string_view(argv[1]) == "environment") {
        Environment();
        if (failures != 0) return 1;
        std::puts("kyty_shim_settings_environment: ok");
        return 0;
    }
    Names();
    Lines();
    if (failures != 0) return 1;
    std::puts("kyty_shim_settings: ok");
    return 0;
}
