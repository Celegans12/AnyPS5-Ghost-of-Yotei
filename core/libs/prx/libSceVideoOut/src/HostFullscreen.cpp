#include "prx/libSceVideoOut/include/HostFullscreen.hpp"
#include <charconv>
#include <cstring>
#include <string_view>

HostFullscreenMode ParseHostFullscreenMode(const char* value, bool* unrecognized) {
    if (unrecognized != nullptr) *unrecognized = false;
    if (value == nullptr || *value == '\0') return HostFullscreenMode::Windowed;
    if (std::strcmp(value, "0") == 0 || std::strcmp(value, "windowed") == 0) return HostFullscreenMode::Windowed;
    if (std::strcmp(value, "1") != 0 && std::strcmp(value, "borderless") != 0 && unrecognized != nullptr) *unrecognized = true;
    return HostFullscreenMode::Borderless;
}

std::optional<HostWindowSize> ParseHostWindowSize(const char* value) {
    if (value == nullptr) return std::nullopt;
    const std::string_view text(value);
    const auto separator = text.find('x');
    if (separator == std::string_view::npos) return std::nullopt;
    const auto dimension = [](std::string_view part) -> int {
        if (part.empty() || part.front() < '0' || part.front() > '9') return 0;
        int number = 0;
        const auto [end, error] = std::from_chars(part.data(), part.data() + part.size(), number);
        if (error != std::errc() || end != part.data() + part.size() || number > HostWindowSizeMaximum) return 0;
        return number;
    };
    const int width = dimension(text.substr(0, separator));
    const int height = dimension(text.substr(separator + 1));
    if (width <= 0 || height <= 0) return std::nullopt;
    return HostWindowSize{width, height};
}

bool WindowModeStartsBorderless(const char* value) {
    return value != nullptr && std::strcmp(value, "borderless") == 0;
}

std::string HostFullscreenWarning(const char* value) {
    std::string sanitized;
    if (value != nullptr) {
        for (const char* p = value; *p != '\0'; ++p) {
            const unsigned char ch = static_cast<unsigned char>(*p);
            if (ch == '\r') {
                sanitized += "\\r";
            } else if (ch == '\n') {
                sanitized += "\\n";
            } else if (ch == '\t') {
                sanitized += "\\t";
            } else if (ch < 0x20 || ch == 0x7f) {
                static constexpr char hex[] = "0123456789abcdef";
                sanitized += "\\x";
                sanitized += hex[ch >> 4];
                sanitized += hex[ch & 0x0f];
            } else {
                sanitized += *p;
            }
        }
    }
    return "DisplayWindow: ANYPS5_FULLSCREEN='" + sanitized + "' is not 0, 1, windowed or borderless; treating it as borderless";
}

std::optional<HostWindowRect> BorderlessCoverRect(HostWindowRect display) {
    if (display.width <= 0 || display.height <= 0) return std::nullopt;
    return display;
}

HostFullscreenState::HostFullscreenState(IHostFullscreenWindow& window) : window(window) {}

void HostFullscreenState::Adopt(HostFullscreenMode mode, HostWindowSize size) {
    windowedSize = size;
    if (mode == HostFullscreenMode::Borderless && !window.IsFullscreen()) window.SetFullscreen(true);
}

void HostFullscreenState::Toggle() {
    if (window.IsFullscreen()) {
        window.SetFullscreen(false);
        if (windowedSize.width > 0 && windowedSize.height > 0 && !(window.Size() == windowedSize)) window.SetSize(windowedSize);
        return;
    }
    const HostWindowSize current = window.Size();
    window.SetFullscreen(true);
    if (current.width > 0 && current.height > 0) windowedSize = current;
}

void HostFullscreenState::Refresh(bool titleReadsMouse) {
    const bool hide = window.IsFullscreen() && !titleReadsMouse;
    if (hide == cursorHidden) return;
    window.ShowCursor(!hide);
    cursorHidden = hide;
}

void HostFullscreenState::Release() noexcept {
    if (!cursorHidden) return;
    cursorHidden = false;
    try {
        window.ShowCursor(true);
    } catch (...) {
    }
}
