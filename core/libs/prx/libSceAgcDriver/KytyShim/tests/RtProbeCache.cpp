#include "RtProbeCache.hpp"

#define XXH_INLINE_ALL
#include "xxhash.h"
#include "zstd.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

void Require(bool condition, const char* message) {
    if (condition) return;
    std::fprintf(stderr, "kyty_shim_rt_probe_cache: %s\n", message);
    std::abort();
}

template <typename T>
void Append(std::string& cache, T value) {
    cache.append(reinterpret_cast<const char*>(&value), sizeof(value));
}

std::string Record(std::uint64_t hash, std::span<const std::uint32_t> code) {
    std::string compressed(ZSTD_compressBound(code.size_bytes()), '\0');
    const auto size = ZSTD_compress(compressed.data(), compressed.size(), code.data(), code.size_bytes(), 3);
    Require(!ZSTD_isError(size), "could not compress fixture");
    std::string record;
    Append(record, hash);
    Append(record, static_cast<std::uint32_t>(code.size()));
    Append(record, static_cast<std::uint32_t>(size));
    record.append(compressed.data(), size);
    return record;
}

std::vector<std::uint32_t> Read(const std::string& cache, std::uint64_t hash) {
    std::istringstream input(cache, std::ios::binary);
    return AgcDriver::ReadCachedShaderCode(input, hash);
}

void ExpectError(const std::string& cache, std::uint64_t hash, std::string_view expected) {
    try {
        static_cast<void>(Read(cache, hash));
        Require(false, "accepted invalid cache input");
    } catch (const std::runtime_error& error) {
        Require(std::string_view(error.what()).find(expected) != std::string_view::npos, "unexpected diagnostic");
    }
}

} // namespace

int main() {
    constexpr std::array shader{0xbf800000u, 0xbf810000u};
    constexpr std::array other{0xbf810000u};
    constexpr std::array metadata{0xffffffffu, 2u};
    const auto hash = XXH3_64bits(shader.data(), sizeof(shader));
    const auto otherHash = XXH3_64bits(other.data(), sizeof(other));
    const auto prefix = "KTYCODE1" + Record(0, metadata) + Record(otherHash, other);
    const auto record = Record(hash, shader);
    const auto cache = prefix + record;
    Require(Read(cache, hash) == std::vector(shader.begin(), shader.end()), "selected shader bytes differ");
    Require(Read(cache, otherHash) == std::vector(other.begin(), other.end()), "earlier shader bytes differ");

    ExpectError(cache, hash ^ 1, "not found");
    ExpectError("KTYCODE2" + record, hash, "expected a KTYCODE1");
    ExpectError(prefix + record.substr(0, 15), hash, "truncated manifest code record");
    ExpectError(cache.substr(0, cache.size() - 1), hash, "truncated manifest code payload");
    ExpectError(prefix + Record(hash, other), hash, "hash mismatch");

    auto badFrame = record;
    badFrame[16] ^= 1;
    ExpectError(prefix + badFrame, hash, "invalid Zstd frame");
    auto tooLarge = record;
    const auto wordCount = std::numeric_limits<std::uint32_t>::max();
    tooLarge.replace(8, sizeof(wordCount), reinterpret_cast<const char*>(&wordCount), sizeof(wordCount));
    ExpectError(prefix + tooLarge, hash, "size exceeds");
}
