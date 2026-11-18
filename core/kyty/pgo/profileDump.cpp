#include <chrono>
#include <cstdlib>
#include <thread>

extern "C" int __llvm_profile_write_file(void);
extern "C" void __llvm_profile_reset_counters(void);

extern "C" const int kyty_pgo_dump_anchor = 1;

namespace {

struct ProfileDump {
    ProfileDump() {
        const char* value = std::getenv("KYTY_PGO_DUMP_SECONDS");
        const long seconds = value != nullptr ? std::strtol(value, nullptr, 10) : 0;
        if (seconds <= 0) {
            return;
        }
        std::thread([seconds] {
            for (;;) {
                std::this_thread::sleep_for(std::chrono::seconds(seconds));
                if (__llvm_profile_write_file() == 0) {
                    __llvm_profile_reset_counters();
                }
            }
        }).detach();
    }
};

const ProfileDump g_profile_dump;

}
