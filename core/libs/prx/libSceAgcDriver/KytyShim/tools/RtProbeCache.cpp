#include "RtProbeCache.hpp"

#define XXH_INLINE_ALL
#include "xxhash.h"
#include "zstd.h"

#include <bit>
#include <istream>
#include <limits>
#include <string_view>
#include <utility>

namespace AgcDriver {
namespace {

constexpr std::string_view CodeMagic = "KTYCODE1";

struct CodeRecord {
    std::uint64_t hash;
    std::uint32_t words;
    std::uint32_t compressedBytes;
};

static_assert(sizeof(CodeRecord) == 16 && std::endian::native == std::endian::little);

CachedShaderCode DecodeCachedShader(std::istream& cache, const CodeRecord& record) {
    constexpr auto MaxShaderWords = std::numeric_limits<std::uint32_t>::max() / sizeof(std::uint32_t);
    if (record.words > MaxShaderWords) return {.error = "cached shader size exceeds its registration format"};
    const auto bytes = static_cast<std::size_t>(record.words) * sizeof(std::uint32_t);
    if (record.compressedBytes > ZSTD_compressBound(bytes)) {
        return {.error = "invalid compressed size in cached shader"};
    }

    std::vector<char> compressed(record.compressedBytes);
    if (!cache.read(compressed.data(), static_cast<std::streamsize>(compressed.size()))) {
        return {.error = "truncated cached shader payload"};
    }

    if (ZSTD_getFrameContentSize(compressed.data(), compressed.size()) != bytes ||
        ZSTD_findFrameCompressedSize(compressed.data(), compressed.size()) != compressed.size()) {
        return {.error = "invalid Zstd frame in cached shader"};
    }
    std::vector<std::uint32_t> code(record.words);
    if (ZSTD_decompress(code.data(), bytes, compressed.data(), compressed.size()) != bytes) {
        return {.error = "cannot decompress cached shader"};
    }
    if (XXH3_64bits(code.data(), bytes) != record.hash) {
        return {.error = "cached shader hash mismatch"};
    }
    return {.code = std::move(code)};
}

}

CachedShaderIndex IndexCachedShaderCode(std::istream& cache) {
    cache.clear();
    cache.seekg(0, std::ios::end);
    auto remaining = cache.tellg() - std::streampos{0};
    cache.seekg(0, std::ios::beg);

    char magic[CodeMagic.size()];
    if (!cache.read(magic, sizeof(magic)) || std::string_view(magic, sizeof(magic)) != CodeMagic) {
        return {.error = "expected a KTYCODE1 manifest code cache"};
    }
    remaining -= sizeof(magic);
    CachedShaderIndex index;
    std::streamoff at = sizeof(magic);
    while (remaining != 0) {
        CodeRecord record{};
        if (remaining < static_cast<std::streamoff>(sizeof(record)) ||
            !cache.read(reinterpret_cast<char*>(&record), sizeof(record))) {
            index.truncated = true;
            break;
        }
        remaining -= sizeof(record);
        at += sizeof(record);
        if (record.words == 0 || record.compressedBytes == 0 || record.compressedBytes > remaining) {
            index.truncated = true;
            break;
        }
        if (record.hash != 0) {
            index.entries.push_back({record.hash, record.words, record.compressedBytes, at});
        }
        cache.seekg(record.compressedBytes, std::ios::cur);
        if (!cache) {
            index.truncated = true;
            break;
        }
        remaining -= record.compressedBytes;
        at += record.compressedBytes;
    }
    return index;
}

CachedShaderCode ReadIndexedShaderCode(std::istream& cache, const CachedShaderEntry& entry) {
    cache.clear();
    cache.seekg(entry.payload, std::ios::beg);
    if (!cache) return {.error = "cannot seek to the cached shader payload"};
    return DecodeCachedShader(cache, {entry.hash, entry.words, entry.compressedBytes});
}

CachedShaderCode ReadCachedShaderCode(std::istream& cache, std::uint64_t hash) {
    if (hash == 0) return {.error = "shader hash must be nonzero"};
    cache.seekg(0, std::ios::end);
    auto remaining = cache.tellg() - std::streampos{0};
    cache.seekg(0, std::ios::beg);

    char magic[CodeMagic.size()];
    if (!cache.read(magic, sizeof(magic)) || std::string_view(magic, sizeof(magic)) != CodeMagic) {
        return {.error = "expected a KTYCODE1 manifest code cache"};
    }
    remaining -= sizeof(magic);
    while (remaining != 0) {
        CodeRecord record{};
        if (remaining < static_cast<std::streamoff>(sizeof(record)) ||
            !cache.read(reinterpret_cast<char*>(&record), sizeof(record))) {
            return {.error = "truncated manifest code record"};
        }
        remaining -= sizeof(record);
        if (record.words == 0 || record.compressedBytes == 0) return {.error = "empty manifest code record"};
        if (record.compressedBytes > remaining) return {.error = "truncated manifest code payload"};
        if (record.hash == hash) return DecodeCachedShader(cache, record);
        cache.seekg(record.compressedBytes, std::ios::cur);
        if (!cache) return {.error = "cannot skip manifest code payload"};
        remaining -= record.compressedBytes;
    }
    return {.error = "shader hash not found in the manifest cache"};
}

}
