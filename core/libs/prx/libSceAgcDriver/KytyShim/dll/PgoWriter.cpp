#if defined(KYTY_PGO_GENERATE)

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

#include <windows.h>

extern "C" {
int __llvm_profile_write_file(void);
void __llvm_profile_set_filename(const char* name);
}

namespace KytyShim {

void StartPgoWriter() {
    const char* seconds_text = std::getenv("KYTY_PGO_WRITE_SECONDS");
    const auto seconds = seconds_text != nullptr && seconds_text[0] != '\0' ? static_cast<std::uint32_t>(std::strtoul(seconds_text, nullptr, 10)) : 60u;
    const char* dir_text = std::getenv("KYTY_PGO_DIR");
    const std::string dir = dir_text != nullptr && dir_text[0] != '\0' ? dir_text : KYTY_PGO_DEFAULT_DIR;
    const auto pid = std::to_string(GetCurrentProcessId());
    static std::string exit_name;
    exit_name = dir + "/kyty_gpu-" + pid + "-exit.profraw";
    __llvm_profile_set_filename(exit_name.c_str());
    std::fprintf(stderr, "[kyty] PGO: instrumented build; counters written to %s/kyty_gpu-%s-<seconds>s.profraw every %u s\n", dir.c_str(), pid.c_str(), seconds);
    if (seconds == 0) return;
    std::thread([dir, pid, seconds] {
        const auto start = std::chrono::steady_clock::now();
        for (std::uint64_t n = 1;; ++n) {
            std::this_thread::sleep_until(start + std::chrono::seconds(seconds * n));
            const auto name = dir + "/kyty_gpu-" + pid + "-" + std::to_string(seconds * n) + "s.profraw";
            __llvm_profile_set_filename(name.c_str());
            const int result = __llvm_profile_write_file();
            __llvm_profile_set_filename(exit_name.c_str());
            std::fprintf(stderr, "[kyty] PGO: counters written to %s (rc %d)\n", name.c_str(), result);
        }
    }).detach();
}

}

#else

namespace KytyShim {

void StartPgoWriter() {}

}

#endif
