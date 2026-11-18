#include "prx/libSceSystemService/include/SystemService.hpp"
#include "prx/libSceSystemService/SystemService.hpp"
#include "SceTypes.hpp"
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>

namespace {

void Require(bool value) { if (!value) std::abort(); }

const std::filesystem::path Record = std::filesystem::path("anyps5-notice-skip") / "PPSA00000";

bool Raised() {
    bool value = false;
    Require(sceSystemServiceGetNoticeScreenSkipFlag(&value) == SYSTEM_SERVICE_OK);
    return value;
}

std::string RecordText() {
    std::ifstream stream(Record, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

template<typename TCall>
bool Throws(TCall call) {
    try {
        call();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

void CheckInProcess() {
    bool value = true;
    Require(sceSystemServiceGetNoticeScreenSkipFlag(&value) == SYSTEM_SERVICE_OK && !value);
    Require(sceSystemServiceGetNoticeScreenSkipFlag(nullptr) == SYSTEM_SERVICE_ERROR_PARAMETER);
    Require(sceSystemServiceDisableNoticeScreenSkipFlagAutoSet() == SYSTEM_SERVICE_OK);
    Require(sceSystemServiceSetNoticeScreenSkipFlag() == SYSTEM_SERVICE_OK);
    value = false;
    Require(sceSystemServiceGetNoticeScreenSkipFlag(&value) == SYSTEM_SERVICE_OK && value);
    Require(sceSystemServiceSetNoticeScreenSkipFlag() == SYSTEM_SERVICE_OK);
    Require(sceSystemServiceGetNoticeScreenSkipFlag(&value) == SYSTEM_SERVICE_OK && value);
    auto* const inaccessible = reinterpret_cast<bool*>(0x10);
    Require(sceSystemServiceGetNoticeScreenSkipFlag(inaccessible) == SYSTEM_SERVICE_ERROR_PARAMETER);
    Require(sceSystemServiceGetNoticeScreenSkipFlag(&value) == SYSTEM_SERVICE_OK && value);
}

void CheckFirstLaunch() {
    Require(!Raised());
    Require(!std::filesystem::exists(Record));
    Require(sceSystemServiceSetNoticeScreenSkipFlag() == SYSTEM_SERVICE_OK);
    Require(Raised());
    Require(RecordText() == "1");
    Require(!std::filesystem::exists(std::filesystem::path(Record).concat(".tmp")));
    Require(sceSystemServiceSetNoticeScreenSkipFlag() == SYSTEM_SERVICE_OK);
    Require(Raised() && RecordText() == "1");
}

void CheckRelaunch() {
    Require(std::filesystem::exists(Record));
    Require(Raised());
}

void CheckDisabled(bool recordPresent) {
    Require(std::filesystem::exists(Record) == recordPresent);
    Require(!Raised());
    Require(sceSystemServiceSetNoticeScreenSkipFlag() == SYSTEM_SERVICE_OK);
    Require(Raised());
    Require(std::filesystem::exists(Record) == recordPresent);
    Require(!std::filesystem::exists("anyps5-notice-skip") || recordPresent);
}

void CheckUnreadableRecord() {
    Require(!Raised());
    Require(sceSystemServiceSetNoticeScreenSkipFlag() == SYSTEM_SERVICE_OK);
    Require(Raised());
    Require(!std::filesystem::exists(Record));
}

void CheckInvalidSetting() {
    bool value = false;
    Require(Throws([&] { static_cast<void>(sceSystemServiceGetNoticeScreenSkipFlag(&value)); }));
    Require(Throws([&] { static_cast<void>(sceSystemServiceSetNoticeScreenSkipFlag()); }));
    Require(!std::filesystem::exists("anyps5-notice-skip"));
}

}

int main(int argc, char** argv) {
    const std::string_view mode = argc > 1 ? argv[1] : "process";
    if (mode == "process") CheckInProcess();
    else if (mode == "first") CheckFirstLaunch();
    else if (mode == "relaunch") CheckRelaunch();
    else if (mode == "disabled-kept") CheckDisabled(true);
    else if (mode == "disabled-fresh") CheckDisabled(false);
    else if (mode == "unwritable") CheckUnreadableRecord();
    else if (mode == "invalid") CheckInvalidSetting();
    else return 2;
}
