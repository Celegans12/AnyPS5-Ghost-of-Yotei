#pragma once

#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

inline bool ParseNoticeScreenSkipPersist(const char* configured, std::string_view caller) {
    if (configured == nullptr || configured[0] == '\0') return true;
    const std::string_view text(configured);
    if (text == "1") return true;
    if (text == "0") return false;
    throw std::runtime_error(std::string(caller) + ": ANYPS5_NOTICE_SKIP_PERSIST must be 0 or 1, not \"" + std::string(text) + "\"");
}

inline bool NoticeScreenSkipPersistFromEnvironment(std::string_view caller) {
    return ParseNoticeScreenSkipPersist(std::getenv("ANYPS5_NOTICE_SKIP_PERSIST"), caller);
}

inline bool NoticeScreenSkipRecordableTitleId(std::string_view titleId) {
    if (titleId.empty() || titleId.size() > 16) return false;
    for (const char character : titleId) {
        if (std::isalnum(static_cast<unsigned char>(character)) == 0) return false;
    }
    return true;
}

class NoticeScreenSkipStore {
public:
    explicit NoticeScreenSkipStore(std::filesystem::path recordFile) : record(std::move(recordFile)), raised(!record.empty() && ReadRecord(record)) {}

    bool Raised() const { return raised.load(std::memory_order_relaxed); }

    void Raise() {
        if (raised.exchange(true, std::memory_order_relaxed) || record.empty()) return;
        WriteRecord(record);
    }

private:
    static bool ReadRecord(const std::filesystem::path& path) {
        std::ifstream stream(path, std::ios::binary);
        char value = 0;
        return stream.get(value) && value == '1';
    }

    static void WriteRecord(const std::filesystem::path& path) {
        std::error_code error;
        std::filesystem::create_directories(path.parent_path(), error);
        std::filesystem::path pending = path;
        pending += ".tmp";
        if (!error) {
            std::ofstream stream(pending, std::ios::binary | std::ios::trunc);
            stream.put('1');
            stream.flush();
            if (!stream) error = std::make_error_code(std::errc::io_error);
        }
        if (!error) std::filesystem::rename(pending, path, error);
        if (error) {
            std::error_code ignored;
            std::filesystem::remove(pending, ignored);
            std::fprintf(stderr, "libSceSystemService: the notice screen skip flag was not kept in %s: %s\n", path.string().c_str(), error.message().c_str());
        }
    }

    std::filesystem::path record;
    std::atomic<bool> raised;
};
