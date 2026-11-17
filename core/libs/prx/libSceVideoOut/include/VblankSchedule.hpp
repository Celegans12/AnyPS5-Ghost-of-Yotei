#ifndef CORE_LIBS_PRX_LIBSCEVIDEOOUT_INCLUDE_VBLANKSCHEDULE_HPP
#define CORE_LIBS_PRX_LIBSCEVIDEOOUT_INCLUDE_VBLANKSCHEDULE_HPP

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace VideoOutVblank {

constexpr std::uint32_t MIN_VBLANK_HZ = 30;
constexpr std::uint32_t MAX_VBLANK_HZ = 360;

inline bool ParseWholeNumber(std::string_view text, std::uint32_t& out) {
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), out);
    return error == std::errc{} && end == text.data() + text.size();
}

inline std::optional<std::uint32_t> ParseVblankHz(const char* text) {
    if (text == nullptr || *text == '\0') return std::nullopt;
    const std::string_view value(text);
    std::uint32_t hertz = 0;
    if (!ParseWholeNumber(value, hertz) || hertz < MIN_VBLANK_HZ || hertz > MAX_VBLANK_HZ) {
        throw std::runtime_error("ANYPS5_VBLANK_HZ must be a whole number of hertz from 30 to 360, not '" + std::string(value) + "'");
    }
    return hertz;
}

inline std::optional<std::uint32_t> ParseKytyVblankFrequency(const char* text) {
    if (text == nullptr || *text == '\0') return std::nullopt;
    const std::string_view value(text);
    std::uint32_t hertz = 0;
    if (!ParseWholeNumber(value, hertz)) throw std::runtime_error("KYTY_VBLANK_FREQUENCY must be a whole number of hertz, not '" + std::string(value) + "'");
    return std::clamp(hertz, MIN_VBLANK_HZ, MAX_VBLANK_HZ);
}

inline std::optional<std::uint32_t> VblankHzFromEnvironment(const char* anyps5, const char* kyty) {
    if (const auto hertz = ParseVblankHz(anyps5)) return hertz;
    return ParseKytyVblankFrequency(kyty);
}

inline std::optional<std::uint32_t> ConfiguredVblankHz() {
#if defined(ANYPS5_KYTY)
    static const auto configured = VblankHzFromEnvironment(std::getenv("ANYPS5_VBLANK_HZ"), std::getenv("KYTY_VBLANK_FREQUENCY"));
#else
    static const auto configured = ParseVblankHz(std::getenv("ANYPS5_VBLANK_HZ"));
#endif
    return configured;
}

inline std::uint32_t HighVblankHz(std::uint32_t hertz) {
    return std::min(hertz * 2, MAX_VBLANK_HZ);
}

class VblankSchedule {
public:
    struct Due {
        bool standard = false;
        bool high = false;
    };

    explicit VblankSchedule(std::optional<std::uint32_t> hertz)
        : standard(hertz ? Stream{1, *hertz} : Stream{1001, 60000}),
          high(hertz ? Stream{1, HighVblankHz(*hertz)} : Stream{1001, 120000}) {}

    std::int64_t NextNanos() const {
        return std::min(standard.Next(), high.Next());
    }

    Due Advance() {
        const auto next = NextNanos();
        const Due due{standard.Next() == next, high.Next() == next};
        if (due.standard) ++standard.index;
        if (due.high) ++high.index;
        return due;
    }

private:
    struct Stream {
        std::uint64_t seconds;
        std::uint64_t parts;
        std::uint64_t index = 1;

        std::int64_t Next() const {
            constexpr std::uint64_t NanosPerSecond = 1'000'000'000;
            const auto whole = index / parts;
            const auto rest = index % parts;
            return static_cast<std::int64_t>(whole * seconds * NanosPerSecond + rest * seconds * NanosPerSecond / parts);
        }
    };

    Stream standard;
    Stream high;
};

}

#endif
