#include <cstdint>
#include "SceTypes.hpp"
#include "SceShaders.hpp"
#include "prx/libc/include/General.hpp"
#include "prx/libSceAgc/Command/include/RegisterDefaults.hpp"
#include "prx/libSceAgc/Misc/include/RegisterDefaults.hpp"

namespace {

constexpr std::uint32_t RegisterDefaultsMaxVersion = 13;

void* get_register_defaults(std::uint32_t version, bool internal, const char* fn) {
    return Agc::Command::GetRegisterDefaults(version <= RegisterDefaultsMaxVersion ? version : RegisterDefaultsMaxVersion, internal, fn);
}

}

extern "C" {

void* APS5_VABI sceAgcGetRegisterDefaults2(std::uint32_t version) {
    return get_register_defaults(version, false, "sceAgcGetRegisterDefaults2");
}

void* APS5_VABI sceAgcGetRegisterDefaults2Internal(std::uint32_t version) {
    return get_register_defaults(version, true, "sceAgcGetRegisterDefaults2Internal");
}

}
