#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/MovedRead.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

namespace AgcDriver::DriverDetail {

struct DriverTestAccess {
    static bool MovedRunsReadEqual(const DispatchVariant& variant, std::span<const UserWordMove> moves) { return Driver::movedRunsReadEqual(variant, moves); }
};

}

namespace {

using AgcDriver::DriverDetail::CopyMoved;
using AgcDriver::DriverDetail::DispatchVariant;
using AgcDriver::DriverDetail::DriverTestAccess;
using AgcDriver::DriverDetail::MovedReads;
using AgcDriver::DriverDetail::UserWordMove;
using AgcDriver::GuestMemory::Compare;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("moved reads: " + message);
}

Compare Copy(std::uintptr_t address, std::span<std::uint32_t> out) {
    return CopyMoved(address, std::as_writable_bytes(out));
}

void HostMemoryIsCopiedThroughTheCheckedCopy() {
    alignas(16) static std::array<std::uint32_t, 4> host{0x11u, 0x22u, 0x33u, 0x44u};
    std::array<std::uint32_t, 4> out{};
    const auto before = MovedReads();
    Require(Copy(reinterpret_cast<std::uintptr_t>(host.data()), out) == Compare::Equal && out == host, "host memory outside the guest arena was not copied");
    const auto after = MovedReads();
    Require(after.checked == before.checked + 1 && after.guarded == before.guarded, "host memory outside the guest arena was not read through the checked copy");
    Require(Copy(0, out) == Compare::Unmapped, "a null address was copied");
}

#ifdef _WIN32
void ArenaRunsAreCopiedInPlace(bool guarded) {
    constexpr std::size_t Granule = 65536;
    constexpr std::size_t Page = 4096;
    auto* block = static_cast<std::uint32_t*>(GuestArena::GuestArenaAllocate_nid_postfix(2 * Granule, Granule));
    Require(block != nullptr, "cannot reserve two arena granules");
    GuestArena::GuestArenaCommit_nid_postfix(block, Granule, PAGE_READWRITE, Granule);
    const auto base = reinterpret_cast<std::uintptr_t>(block);
    const auto words = Granule / sizeof(std::uint32_t);
    for (std::size_t word = 0; word < words; ++word) block[word] = static_cast<std::uint32_t>(0xa0000000u + word);
    MEMORY_BASIC_INFORMATION beyond{};
    Require(VirtualQuery(reinterpret_cast<const void*>(base + Granule), &beyond, sizeof(beyond)) == sizeof(beyond) && beyond.State != MEM_COMMIT, "the granule after the committed one is committed, so the case cannot straddle uncommitted memory");

    std::array<std::uint32_t, 4> out{};
    auto before = MovedReads();
    Require(Copy(base + 64, out) == Compare::Equal && std::memcmp(out.data(), block + 16, sizeof(out)) == 0, "a run inside committed arena memory was not copied");
    auto after = MovedReads();
    Require(guarded ? after.guarded == before.guarded + 1 && after.checked == before.checked : after.checked == before.checked + 1 && after.guarded == before.guarded, std::string("a run inside committed arena memory was not read ") + (guarded ? "under the read guard" : "through the checked copy, with the guarded reads off"));

    before = MovedReads();
    Require(Copy(base + Granule - 8, out) == Compare::Unmapped, "a run straddling the end of committed memory into an uncommitted page was reported copied");
    Require(Copy(base + Granule, out) == Compare::Unmapped, "a run in uncommitted arena memory was reported copied");
    after = MovedReads();
    Require(after.guarded == before.guarded && after.checked == before.checked + 2, "a run the read guard could not copy did not fall back to the checked copy");
    Require(Copy(base + Granule - 16, out) == Compare::Equal && std::memcmp(out.data(), block + words - 4, sizeof(out)) == 0, "the last run of committed memory was not copied to its end");

    DWORD previous = 0;
    Require(VirtualProtect(block + 2 * Page / sizeof(std::uint32_t), Page, PAGE_NOACCESS, &previous) != 0, "cannot protect a page of the fixture");
    Require(Copy(base + 2 * Page - 8, out) == Compare::Unmapped && Copy(base + 3 * Page - 8, out) == Compare::Unmapped, "a run straddling a page without access was reported copied");
    Require(VirtualProtect(block + 2 * Page / sizeof(std::uint32_t), Page, previous, &previous) != 0, "cannot restore the protected page");
    Require(Copy(base + 2 * Page - 8, out) == Compare::Equal, "a run over the restored page was not copied");

    const std::size_t kept = 256, same = 1024, other = 2048;
    for (std::size_t word = 0; word < 4; ++word) {
        block[same + word] = block[kept + word];
        block[words - 2 + word % 2] = block[kept + word % 2];
    }
    DispatchVariant variant;
    variant.runs = {{base + kept * sizeof(std::uint32_t), base + (kept + 4) * sizeof(std::uint32_t)}};
    variant.words.assign(block + kept, block + kept + 4);
    variant.runBases = {0};
    variant.relocatableWords = 1u;
    const auto low = [&](std::size_t word) { return static_cast<std::uint32_t>(base + word * sizeof(std::uint32_t)); };
    const auto moved = [&](std::size_t word) {
        const std::array<UserWordMove, 1> moves{{{0u, low(kept), low(word)}}};
        return DriverTestAccess::MovedRunsReadEqual(variant, moves);
    };
    Require(DriverTestAccess::MovedRunsReadEqual(variant, {}), "an unmoved run with its captured words did not compare equal");
    Require(moved(same), "a run moved to memory holding the same words did not compare equal");
    Require(!moved(other), "a run moved to memory holding other words compared equal");
    Require(!moved(words - 2), "a run moved across the end of committed memory, whose mapped words equal the captured ones, compared equal");
    Require(!moved(words + 64), "a run moved into uncommitted memory compared equal");
}
#endif

}

int main() {
    try {
        const bool guarded = std::getenv("APS5_NO_GUARDED_MOVED_READS") == nullptr;
        HostMemoryIsCopiedThroughTheCheckedCopy();
#ifdef _WIN32
        ArenaRunsAreCopiedInPlace(guarded);
#endif
        const auto counts = MovedReads();
        std::printf("moved reads: %llu runs copied under the read guard, %llu through the checked copy (guarded reads %s); runs reaching uncommitted or protected memory were refused either way\n", static_cast<unsigned long long>(counts.guarded), static_cast<unsigned long long>(counts.checked), guarded ? "on" : "off");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
