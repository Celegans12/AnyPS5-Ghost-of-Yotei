#include "RtProbeCache.hpp"

#define XXH_INLINE_ALL
#include "xxhash.h"
#include "zstd.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <sstream>
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

AgcDriver::CachedShaderCode Read(const std::string& cache, std::uint64_t hash) {
    std::istringstream input(cache, std::ios::binary);
    return AgcDriver::ReadCachedShaderCode(input, hash);
}

void ExpectError(const std::string& cache, std::uint64_t hash, std::string_view expected) {
    const auto result = Read(cache, hash);
    Require(result.error != nullptr, "accepted invalid cache input");
    Require(expected == result.error, "unexpected diagnostic");
    Require(result.code.empty(), "invalid cache input returned shader words");
}

void WriteExportFixture(const std::filesystem::path& directory) {
    constexpr std::array shader{0xbf810000u};
    const auto hash = XXH3_64bits(shader.data(), sizeof(shader));
    const auto cache = "KTYCODE1" + Record(hash, shader);
    std::ofstream rawFile(directory / "shader.bin", std::ios::binary);
    rawFile.write(reinterpret_cast<const char*>(shader.data()), sizeof(shader));
    rawFile.close();
    Require(static_cast<bool>(rawFile), "could not write raw export fixture");
    std::ofstream cacheFile(directory / "manifest.spv.guest.spv", std::ios::binary);
    cacheFile.write(cache.data(), static_cast<std::streamsize>(cache.size()));
    cacheFile.close();
    Require(static_cast<bool>(cacheFile), "could not write cache export fixture");
    std::printf("%016llx\n", static_cast<unsigned long long>(hash));
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::string_view(argv[1]) == "write-export-fixture") {
        WriteExportFixture(argv[2]);
        return 0;
    }
    Require(argc == 1, "unexpected test arguments");
    constexpr std::array shader{0xbf800000u, 0xbf810000u};
    constexpr std::array other{0xbf810000u};
    constexpr std::array metadata{0xffffffffu, 2u};
    const auto hash = XXH3_64bits(shader.data(), sizeof(shader));
    const auto otherHash = XXH3_64bits(other.data(), sizeof(other));
    const auto prefix = "KTYCODE1" + Record(0, metadata) + Record(otherHash, other);
    const auto record = Record(hash, shader);
    const auto cache = prefix + record;
    const auto selected = Read(cache, hash);
    Require(!selected.error && selected.code == std::vector(shader.begin(), shader.end()), "selected shader bytes differ");
    const auto earlier = Read(cache, otherHash);
    Require(!earlier.error && earlier.code == std::vector(other.begin(), other.end()), "earlier shader bytes differ");

    ExpectError(cache, 0, "shader hash must be nonzero");
    ExpectError("KTYCODE1", hash, "shader hash not found in the manifest cache");
    ExpectError(cache, hash ^ 1, "shader hash not found in the manifest cache");
    ExpectError("KTYCODE2" + record, hash, "expected a KTYCODE1 manifest code cache");
    ExpectError(prefix + record.substr(0, 15), hash, "truncated manifest code record");
    ExpectError(cache.substr(0, cache.size() - 1), hash, "truncated manifest code payload");
    ExpectError(prefix + Record(hash, other), hash, "cached shader hash mismatch");

    std::string emptyRecord;
    Append(emptyRecord, hash);
    Append(emptyRecord, std::uint32_t{0});
    Append(emptyRecord, std::uint32_t{0});
    ExpectError(prefix + emptyRecord, hash, "empty manifest code record");

    auto badFrame = record;
    badFrame[16] ^= 1;
    ExpectError(prefix + badFrame, hash, "invalid Zstd frame in cached shader");
    auto tooLarge = record;
    const auto wordCount = std::numeric_limits<std::uint32_t>::max();
    tooLarge.replace(8, sizeof(wordCount), reinterpret_cast<const char*>(&wordCount), sizeof(wordCount));
    ExpectError(prefix + tooLarge, hash, "cached shader size exceeds its registration format");
}
