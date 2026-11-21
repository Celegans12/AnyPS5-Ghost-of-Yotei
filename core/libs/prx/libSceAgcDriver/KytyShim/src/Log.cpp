#include "common/logging/log.h"

#ifdef ANYPS5_KYTY_RT_PROBE
#include "../tools/RtSurvey.hpp"
#endif
#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "kytyGitVersion.h"

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <mutex>
#include <string>
#include <string_view>
#include <system_error>

#ifdef ANYPS5_KYTY_RT_PROBE
namespace AgcDriver {

thread_local const SurveyLogScope* SurveyLogScope::current_ = nullptr;

SurveyLogScope::SurveyLogScope(Observer observer, void* context) noexcept
    : previous_(current_), observer_(observer), context_(context) {
    current_ = this;
}

SurveyLogScope::~SurveyLogScope() {
    current_ = previous_;
}

bool SurveyLogScope::Active() noexcept {
    return current_ != nullptr;
}

void SurveyLogScope::Observe(std::string_view text) noexcept {
    if (current_ != nullptr) current_->observer_(current_->context_, text);
}

}
#endif

namespace Log {
namespace {

enum class Sink { Standard, Silent, File };

std::mutex g_mutex;
std::atomic<bool> g_initialized{false};
std::atomic<Sink> g_sink{Sink::Standard};
std::FILE* g_file = nullptr;
std::FILE* g_session = nullptr;

void WriteStandard(std::string_view text) {
    std::fwrite(text.data(), 1, text.size(), stderr);
}

std::FILE* Open(const std::filesystem::path& path, const char* what) {
    std::error_code error;
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path(), error);
#ifdef _WIN32
    std::FILE* file = _wfopen(path.c_str(), L"wb");
#else
    std::FILE* file = std::fopen(path.c_str(), "wb");
#endif
    if (file == nullptr) EXIT("KytyShim: cannot open the %s %s (%s)\n", what, path.string().c_str(), std::strerror(errno));
    std::setvbuf(file, nullptr, _IOFBF, static_cast<std::size_t>(64) * 1024);
    return file;
}

void FlushLocked() {
    if (g_file != nullptr) std::fflush(g_file);
    if (g_session != nullptr) std::fflush(g_session);
    std::fflush(stderr);
}

}

void Initialize() {
    const auto direction = Config::GetPrintfDirection();
    const auto path = Config::GetPrintfOutputFile();
    const auto session = Config::GetSessionLogFile();
    std::lock_guard lock(g_mutex);
    const bool again = g_initialized.load(std::memory_order_relaxed);
    FlushLocked();
    for (std::FILE** file : {&g_file, &g_session}) {
        if (*file != nullptr) std::fclose(*file);
        *file = nullptr;
    }
    Sink sink = Sink::Silent;
    if (direction == Config::LogDirection::Console) {
        sink = Sink::Standard;
    } else if (direction == Config::LogDirection::File && !path.empty()) {
        g_file = Open(path, "printf log");
        sink = Sink::File;
    }
    if (!session.empty()) {
        g_session = Open(session, "session log");
        const std::time_t now = std::time(nullptr);
        std::tm local{};
#ifdef _WIN32
        localtime_s(&local, &now);
#else
        localtime_r(&now, &local);
#endif
        char started[32] = "";
        std::strftime(started, sizeof(started), "%Y-%m-%d %H:%M:%S", &local);
        const auto header = fmt::format("--- Session {} ---\n{}\n", started, KYTY_BUILD_LABEL);
        std::fwrite(header.data(), 1, header.size(), g_session);
        std::fflush(g_session);
    }
    g_sink.store(sink, std::memory_order_relaxed);
    g_initialized.store(true, std::memory_order_release);
    if (!again) std::atexit(Flush);
}

void Flush() {
    std::lock_guard lock(g_mutex);
    FlushLocked();
}

bool IsSilent() {
#ifdef ANYPS5_KYTY_RT_PROBE
    if (AgcDriver::SurveyLogScope::Active()) return false;
#endif
    return g_initialized.load(std::memory_order_acquire) && g_sink.load(std::memory_order_relaxed) == Sink::Silent;
}

void Write(fmt::text_style style, std::string_view text) {
#ifdef ANYPS5_KYTY_RT_PROBE
    AgcDriver::SurveyLogScope::Observe(text);
#endif
    static_cast<void>(style);
    if (!g_initialized.load(std::memory_order_acquire)) {
        WriteStandard(text);
        return;
    }
    switch (g_sink.load(std::memory_order_relaxed)) {
        case Sink::Standard: WriteStandard(text); return;
        case Sink::Silent: return;
        case Sink::File: {
            std::lock_guard lock(g_mutex);
            if (g_file != nullptr) std::fwrite(text.data(), 1, text.size(), g_file);
            return;
        }
    }
}

void WriteToConsoleAndLog(std::string_view text) {
    if (g_initialized.load(std::memory_order_acquire) && g_sink.load(std::memory_order_relaxed) != Sink::Standard) WriteStandard(text);
    Write({}, text);
    WriteToSessionLog(text);
    Flush();
}

void WriteToSessionLog(std::string_view text) {
    std::lock_guard lock(g_mutex);
    if (g_session != nullptr) std::fwrite(text.data(), 1, text.size(), g_session);
}

}
