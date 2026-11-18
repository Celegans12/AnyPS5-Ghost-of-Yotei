#include "RtProbeCache.hpp"

#define XXH_INLINE_ALL
#include "xxhash.h"
#include "zstd.h"

#include <bit>
#include <istream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace AgcDriver {
namespace {

constexpr std::string_view CodeMagic = "KTYCODE1";

struct CodeRecord {
    std::uint64_t hash;
    std::uint32_t words;
    std::uint32_t compressedBytes;
};

static_assert(sizeof(CodeRecord) == 16 && std::endian::native == std::endian::little);

std::vector<std::uint32_t> DecodeCachedShader(std::istream& cache, const CodeRecord& record) {
    constexpr auto MaxShaderWords = std::numeric_limits<std::uint32_t>::max() / sizeof(std::uint32_t);
    if (record.words > MaxShaderWords) throw std::runtime_error("cached shader size exceeds its registration format");
    const auto bytes = static_cast<std::size_t>(record.words) * sizeof(std::uint32_t);
    if (record.compressedBytes > ZSTD_compressBound(bytes)) {
        throw std::runtime_error("invalid compressed size in cached shader");
    }

    std::vector<char> compressed(record.compressedBytes);
    if (!cache.read(compressed.data(), static_cast<std::streamsize>(compressed.size()))) {
        throw std::runtime_error("truncated cached shader payload");
    }

    if (ZSTD_getFrameContentSize(compressed.data(), compressed.size()) != bytes ||
        ZSTD_findFrameCompressedSize(compressed.data(), compressed.size()) != compressed.size()) {
        throw std::runtime_error("invalid Zstd frame in cached shader");
    }
    std::vector<std::uint32_t> code(record.words);
    if (ZSTD_decompress(code.data(), bytes, compressed.data(), compressed.size()) != bytes) {
        throw std::runtime_error("cannot decompress cached shader");
    }
    if (XXH3_64bits(code.data(), bytes) != record.hash) {
        throw std::runtime_error("cached shader hash mismatch");
    }
    return code;
}

} // namespace

std::vector<std::uint32_t> ReadCachedShaderCode(std::istream& cache, std::uint64_t hash) {
    if (hash == 0) throw std::invalid_argument("shader hash must be nonzero");
    cache.seekg(0, std::ios::end);
    auto remaining = cache.tellg() - std::streampos{0};
    cache.seekg(0, std::ios::beg);

    char magic[CodeMagic.size()];
    if (!cache.read(magic, sizeof(magic)) || std::string_view(magic, sizeof(magic)) != CodeMagic) {
        throw std::runtime_error("expected a KTYCODE1 manifest code cache");
    }
    remaining -= sizeof(magic);
    while (remaining != 0) {
        CodeRecord record{};
        if (remaining < static_cast<std::streamoff>(sizeof(record)) ||
            !cache.read(reinterpret_cast<char*>(&record), sizeof(record))) {
            throw std::runtime_error("truncated manifest code record");
        }
        remaining -= sizeof(record);
        if (record.words == 0 || record.compressedBytes == 0) throw std::runtime_error("empty manifest code record");
        if (record.compressedBytes > remaining) throw std::runtime_error("truncated manifest code payload");
        if (record.hash == hash) return DecodeCachedShader(cache, record);
        cache.seekg(record.compressedBytes, std::ios::cur);
        if (!cache) throw std::runtime_error("cannot skip manifest code payload");
        remaining -= record.compressedBytes;
    }
    throw std::runtime_error("shader hash not found in the manifest cache");
}

} // namespace AgcDriver
