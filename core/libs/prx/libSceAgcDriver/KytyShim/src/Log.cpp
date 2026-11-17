#include "common/logging/log.h"

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
    const bool relaunched = std::getenv(RelaunchedVariable) != nullptr;
    std::FILE* file = std::fopen(path.c_str(), relaunched ? "ab" : "wb");
    if (file == nullptr) EXIT("KytyShim: cannot open the %s %s (%s)\n", what, path.c_str(), std::strerror(errno));
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
        localtime_r(&now, &local);
        char started[32] = "";
        std::strftime(started, sizeof(started), "%Y-%m-%d %H:%M:%S", &local);
        const auto header = fmt::format("--- {} {} ---\n{}\n", std::getenv(RelaunchedVariable) != nullptr ? "Relaunched" : "Session", started, KYTY_BUILD_LABEL);
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
    return g_initialized.load(std::memory_order_acquire) && g_sink.load(std::memory_order_relaxed) == Sink::Silent;
}

void Write(fmt::text_style style, std::string_view text) {
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
