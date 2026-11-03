#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Dispatch/DispatchCache.hpp"
#include <array>
#include <cstdio>
#include <stdexcept>
#include <vector>

using AgcDriver::DriverDetail::DispatchVariant;
using AgcDriver::DriverDetail::FreeDispatchUserWords;
using AgcDriver::DriverDetail::MatchDispatchUserWords;
using AgcDriver::DriverDetail::MixDispatchUserWords;
using AgcDriver::DriverDetail::UserWordsMatch;
using Runs = std::vector<std::pair<std::uint64_t, std::uint64_t>>;

namespace {

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::uint64_t fullKey(const std::vector<std::uint32_t>& userData) {
    std::uint64_t key = 0xcbf29ce484222325ull;
    for (const auto word : userData) {
        key ^= word;
        key *= 0x100000001b3ull;
    }
    return key;
}

void testFreeWords() {
    ShaderRecompiler::SrtUserDataUse use;
    use.asBase = 0x1u;
    use.asValue = 0x6u;
    check(FreeDispatchUserWords(use, 0x8u, 0x1u, 6) == 0x31u, "a base-only word and the words nothing reads are free");
    check(FreeDispatchUserWords(use, 0x9u, 0x1u, 6) == 0x30u, "a base word the shader also reads stays in the key");
    check(FreeDispatchUserWords(use, 0x8u, 0x0u, 6) == 0x30u, "a base word whose reads were not tagged stays in the key");
    use.asValue = 0x7u;
    check(FreeDispatchUserWords(use, 0x8u, 0x1u, 6) == 0x30u, "a base word also used as a value stays in the key");
    check(FreeDispatchUserWords({}, 0x0u, 0x0u, 6) == 0u, "a walk that recorded no user-data use frees no word");
}

void testKeys() {
    const std::vector<std::uint32_t> words{0x1460c5d0u, 0x10u, 0x200u, 0x7u};
    check(MixDispatchUserWords(0xcbf29ce484222325ull, words, 0) == fullKey(words), "with no free word the key is the full key");
    check(MixDispatchUserWords(0xcbf29ce484222325ull, words, 0x1u) != fullKey(words), "a free key never equals the full key of the same words");
    auto moved = words;
    moved[0] = 0x1460c630u;
    check(MixDispatchUserWords(0xcbf29ce484222325ull, words, 0x1u) == MixDispatchUserWords(0xcbf29ce484222325ull, moved, 0x1u), "a free word that moved changed the key");
    moved[0] = 0x1460c631u;
    check(MixDispatchUserWords(0xcbf29ce484222325ull, words, 0x1u) != MixDispatchUserWords(0xcbf29ce484222325ull, moved, 0x1u), "a free word with other low bits kept the key");
    moved[0] = 0u;
    check(MixDispatchUserWords(0xcbf29ce484222325ull, words, 0x1u) != MixDispatchUserWords(0xcbf29ce484222325ull, moved, 0x1u), "a free word that became 0 kept the key");
    moved = words;
    moved[1] = 0x11u;
    check(MixDispatchUserWords(0xcbf29ce484222325ull, words, 0x1u) != MixDispatchUserWords(0xcbf29ce484222325ull, moved, 0x1u), "a word in the key changed and the key did not");
    check(MixDispatchUserWords(0xcbf29ce484222325ull, words, 0x1u) != MixDispatchUserWords(0xcbf29ce484222325ull, words, 0x3u), "two shapes gave one key");
}

void testMatch() {
    alignas(16) static std::array<std::uint32_t, 8> table{1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u};
    alignas(16) static std::array<std::uint32_t, 4> fixed{9u, 10u, 11u, 12u};
    const auto tableAddress = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(table.data()));
    const auto fixedAddress = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(fixed.data()));
    const std::array<ShaderRecompiler::MemoryRegion, 2> regions{{{tableAddress, std::as_bytes(std::span(table))}, {fixedAddress, std::as_bytes(std::span(fixed))}}};
    ShaderRecompiler::SrtUserDataUse use;
    use.asBase = 0x1u;
    use.asValue = 0x2u;
    use.reads = {{tableAddress, 0}, {tableAddress + 4u, 0}, {tableAddress + 8u, 0}, {fixedAddress, -1}};
    DispatchVariant variant;
    AgcDriver::DriverDetail::Driver::tagVariantRuns(variant, regions, use, true, 0);
    check(variant.relocatableWords == 0x1u, "the base word was not tagged relocatable");
    const auto low = static_cast<std::uint32_t>(tableAddress);
    variant.userData = {low, 0x55u, 0x77u};
    variant.exactUserWords = ~FreeDispatchUserWords(use, 0x0u, variant.relocatableWords, 3);
    Runs runs;
    check(MatchDispatchUserWords(variant, std::vector<std::uint32_t>{low, 0x55u, 0x77u}, runs) == UserWordsMatch::Same, "equal words did not match");
    check(MatchDispatchUserWords(variant, std::vector<std::uint32_t>{low, 0x55u, 0x99u}, runs) == UserWordsMatch::Same, "a word nothing reads decided the match");
    check(MatchDispatchUserWords(variant, std::vector<std::uint32_t>{low, 0x56u, 0x77u}, runs) == UserWordsMatch::Differs, "a value word that changed matched");
    check(MatchDispatchUserWords(variant, std::vector<std::uint32_t>{low, 0x55u}, runs) == UserWordsMatch::Differs, "another user-data count matched");
    check(MatchDispatchUserWords(variant, std::vector<std::uint32_t>{low + 2u, 0x55u, 0x77u}, runs) == UserWordsMatch::Differs, "a base moved by a non-dword step matched");
    check(MatchDispatchUserWords(variant, std::vector<std::uint32_t>{0u, 0x55u, 0x77u}, runs) == UserWordsMatch::Differs, "a base that became 0 matched");
    for (const std::uint32_t step : {0x60u, 0xffffffa0u}) {
        runs.clear();
        check(MatchDispatchUserWords(variant, std::vector<std::uint32_t>{low + step, 0x55u, 0x77u}, runs) == UserWordsMatch::Moved, "a moved base did not match as moved");
        check(runs.size() == variant.runs.size(), "the moved runs are not the variant's runs");
        const auto delta = static_cast<std::uint64_t>((low + step) & ~3u) - static_cast<std::uint64_t>(low & ~3u);
        std::size_t shifted = 0;
        for (std::size_t run = 0; run < runs.size(); ++run) {
            const bool moves = run < variant.runBases.size() && variant.runBases[run] == 0;
            const auto expected = moves ? std::make_pair(variant.runs[run].first + delta, variant.runs[run].second + delta) : variant.runs[run];
            check(runs[run] == expected, moves ? "a run read through the base did not move with it" : "a run read without the base moved");
            shifted += moves ? 1u : 0u;
        }
        check(shifted == 1u, "the base's three words are not one moved run");
    }
    variant.userData.clear();
    check(MatchDispatchUserWords(variant, std::vector<std::uint32_t>{low, 0x55u, 0x77u}, runs) == UserWordsMatch::Differs, "a variant without recorded user data matched under a free key");
}

}

int main() {
    try {
        testFreeWords();
        testKeys();
        testMatch();
        std::puts("Dispatch free keys passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
