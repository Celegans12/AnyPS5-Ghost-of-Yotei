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

AgcDriver::CachedShaderIndex Index(const std::string& cache) {
    std::istringstream input(cache, std::ios::binary);
    return AgcDriver::IndexCachedShaderCode(input);
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

std::string HexText(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(value));
    return text;
}

void WriteText(const std::filesystem::path& path, const std::string& content) {
    std::ofstream file(path, std::ios::binary);
    file.write(content.data(), static_cast<std::streamsize>(content.size()));
    file.close();
    Require(static_cast<bool>(file), "could not write a survey fixture file");
}

void WriteSurveyFixture(const std::filesystem::path& directory) {
    constexpr std::array driven{0xf19c8f01u, 0x00000008u, 0xe0700000u, 0x80000000u, 0xbf810000u};
    constexpr std::array cacheOnly{0xf19c8f01u, 0x00000010u, 0xe0700000u, 0x80000000u, 0xbf810000u};
    constexpr std::array stopping{0x7e100500u, 0xf4080100u, 0x10000000u, 0xf19c8f01u, 0x00000018u, 0xe0600000u, 0x80010000u, 0xbf810000u};
    constexpr std::array plain{0xbf810000u};
    constexpr std::array absent{0xbf800000u, 0xbf810000u};
    constexpr std::array metadata{0xffffffffu, 2u};
    const auto drivenHash = XXH3_64bits(driven.data(), sizeof(driven));
    const auto cacheOnlyHash = XXH3_64bits(cacheOnly.data(), sizeof(cacheOnly));
    const auto stoppingHash = XXH3_64bits(stopping.data(), sizeof(stopping));
    const auto plainHash = XXH3_64bits(plain.data(), sizeof(plain));
    const auto absentHash = XXH3_64bits(absent.data(), sizeof(absent));
    const auto cache = "KTYCODE1" + Record(0, metadata) + Record(drivenHash, driven) + Record(plainHash, plain) +
                       Record(stoppingHash, stopping) + Record(cacheOnlyHash, cacheOnly);
    WriteText(directory / "survey.code", cache);
    WriteText(directory / "survey.kpm",
              "# kyty pipeline manifest v1\n"
              "M 000000000001 cs code=" + HexText(drivenHash) + " ud=11 t=32,1,1 wave=32 tid=1 wgr=11 gx sb=-\n"
              "M 000000000002 cs code=" + HexText(plainHash) + " ud=11 t=64,1,1 tid=1 wgr=11 gx sb=-\n"
              "M 000000000003 cs code=" + HexText(absentHash) + " ud=11 t=64,1,1 tid=1 wgr=11 gx sb=-\n");
    std::printf("%s %s %s %s\n", HexText(drivenHash).c_str(), HexText(cacheOnlyHash).c_str(), HexText(plainHash).c_str(),
                HexText(stoppingHash).c_str());
}

void CheckIndex() {
    constexpr std::array shader{0xbf800000u, 0xbf810000u};
    constexpr std::array other{0xbf810000u};
    constexpr std::array metadata{0xffffffffu, 2u};
    const auto hash = XXH3_64bits(shader.data(), sizeof(shader));
    const auto otherHash = XXH3_64bits(other.data(), sizeof(other));
    const auto prefix = "KTYCODE1" + Record(0, metadata) + Record(otherHash, other);
    const auto cache = prefix + Record(hash, shader);

    const auto index = Index(cache);
    Require(!index.error && !index.truncated, "a whole cache was refused");
    Require(index.entries.size() == 2, "the index must skip the metadata record");
    Require(index.entries[0].hash == otherHash && index.entries[1].hash == hash, "the index lost the record order");
    std::istringstream input(cache, std::ios::binary);
    const auto second = AgcDriver::ReadIndexedShaderCode(input, index.entries[1]);
    Require(!second.error && second.code == std::vector(shader.begin(), shader.end()), "indexed shader bytes differ");
    const auto first = AgcDriver::ReadIndexedShaderCode(input, index.entries[0]);
    Require(!first.error && first.code == std::vector(other.begin(), other.end()), "an earlier indexed shader differs");

    const auto cut = Index(cache.substr(0, cache.size() - 1));
    Require(!cut.error && cut.truncated && cut.entries.size() == 1, "a cut payload must end the index with the whole records");
    const auto half = Index(prefix + Record(hash, shader).substr(0, 15));
    Require(!half.error && half.truncated && half.entries.size() == 1, "a cut record header must end the index");
    Require(Index("KTYCODE2" + Record(hash, shader)).error != nullptr, "a wrong magic was accepted");
    Require(Index("").error != nullptr, "an empty file was accepted");
    const auto none = Index("KTYCODE1");
    Require(!none.error && !none.truncated && none.entries.empty(), "a cache without record was refused");

    auto damaged = prefix + Record(hash, shader);
    damaged[damaged.size() - 3] ^= 0x55;
    const auto damagedIndex = Index(damaged);
    Require(damagedIndex.entries.size() == 2, "a damaged payload must still be indexed");
    std::istringstream damagedInput(damaged, std::ios::binary);
    Require(AgcDriver::ReadIndexedShaderCode(damagedInput, damagedIndex.entries[1]).error != nullptr, "a damaged payload was decoded");
}

}

int main(int argc, char** argv) {
    if (argc == 3 && std::string_view(argv[1]) == "write-export-fixture") {
        WriteExportFixture(argv[2]);
        return 0;
    }
    if (argc == 3 && std::string_view(argv[1]) == "write-survey-fixture") {
        WriteSurveyFixture(argv[2]);
        return 0;
    }
    Require(argc == 1, "unexpected test arguments");
    CheckIndex();
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
