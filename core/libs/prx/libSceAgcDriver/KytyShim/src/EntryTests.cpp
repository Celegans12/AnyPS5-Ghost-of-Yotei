#include "prx/libSceAgcDriver/KytyShim/src/EntryPure.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

int failures = 0;

void Expect(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAILED: %s\n", what);
    ++failures;
}

void ExpectThrow(const std::function<void()>& action, const char* fragment, const char* what) {
    try {
        action();
    } catch (const std::runtime_error& error) {
        if (fragment == nullptr || std::strstr(error.what(), fragment) != nullptr) return;
        std::fprintf(stderr, "FAILED: %s: wrong message \"%s\"\n", what, error.what());
        ++failures;
        return;
    }
    std::fprintf(stderr, "FAILED: %s: no exception\n", what);
    ++failures;
}

std::array<std::uint32_t, 8> Release(std::uint32_t interruptSelect, std::uint32_t context) {
    return {0xc0064900u, 0x00000514u, interruptSelect << 24u, 0x10000u, 0u, 1u, 0u, context};
}

void Append(std::vector<std::uint32_t>& words, std::span<const std::uint32_t> packet) {
    words.insert(words.end(), packet.begin(), packet.end());
}

void PacketCodes() {
    Expect(KytyShim::Entry::FlipCustomCode == 0x17u, "flip header lands on Kyty custom code 0x17");
    Expect(KytyShim::Entry::RenderingWaitCustomCode == 0x06u, "rendering wait header lands on Kyty custom code 0x06");
    Expect(KytyShim::Entry::PacketWords(AgcDriver::FlipPacketHeader) == AgcDriver::FlipPacketWords, "flip packet length from its header");
    Expect(KytyShim::Entry::PacketWords(AgcDriver::RenderingWaitPacketHeader) == AgcDriver::RenderingWaitPacketWords, "rendering wait packet length from its header");
    Expect(KytyShim::Entry::PacketWords(0x80000000u) == 1u, "type 2 filler is one word");
    Expect(KytyShim::Entry::PacketWords(0xffff1000u) == 1u, "0xffff1000 filler is one word");
    Expect(KytyShim::Entry::PacketWords(0xc0064900u) == 8u, "RELEASE_MEM is eight words");
}

void FlipRoundTrip() {
    const AgcDriver::FlipInfo info{7u, -1, 2u, -0x123456789abcdefLL};
    const auto packet = KytyShim::Entry::EncodeFlipPacket(info);
    Expect(packet[0] == AgcDriver::FlipPacketHeader, "encoded flip starts with our header");
    const auto decoded = KytyShim::Entry::DecodeFlipPacket(packet);
    Expect(decoded.has_value(), "encoded flip decodes");
    if (decoded) {
        Expect(decoded->handle == 7u && decoded->index == -1 && decoded->mode == 2u && decoded->argument == -0x123456789abcdefLL, "flip fields survive the round trip");
    }
    auto wrong = packet;
    wrong[0] = AgcDriver::RenderingWaitPacketHeader;
    Expect(!KytyShim::Entry::DecodeFlipPacket(wrong).has_value(), "flip decode refuses another header");
    Expect(!KytyShim::Entry::DecodeFlipPacket(std::span<const std::uint32_t>(packet.data(), 5)).has_value(), "flip decode refuses a truncated packet");
    const std::array<std::uint32_t, 6> written{0xc004105cu, 1u, 3u, 1u, 0x89abcdefu, 0x01234567u};
    const auto fromDcb = KytyShim::Entry::DecodeFlipPacket(written);
    Expect(fromDcb && fromDcb->handle == 1u && fromDcb->index == 3 && fromDcb->mode == 1u && fromDcb->argument == 0x0123456789abcdefLL, "flip written as sceAgcDcbSetFlip writes it");
}

void RenderingWait() {
    const std::array<std::uint32_t, 4> packet{AgcDriver::RenderingWaitPacketHeader, 9u, 2u, 0u};
    const auto target = KytyShim::Entry::DecodeRenderingWaitPacket(packet);
    Expect(target && target->handle == 9u && target->index == 2u && target->mode == 0u, "rendering wait written as sceAgcDriverWaitUntilSafeForRendering writes it");
    Expect(!KytyShim::Entry::DecodeRenderingWaitPacket(std::span<const std::uint32_t>(packet.data(), 3)).has_value(), "rendering wait decode refuses a truncated packet");
    auto wrong = packet;
    wrong[0] = AgcDriver::FlipPacketHeader;
    Expect(!KytyShim::Entry::DecodeRenderingWaitPacket(wrong).has_value(), "rendering wait decode refuses another header");
}

void EopRelease() {
    const auto eop = Release(2u, 0x04000001u);
    const auto token = KytyShim::Entry::EopReleaseToken(eop);
    Expect(token && *token == 0x04000001u, "end-of-pipe release with an interrupt carries its token");
    Expect(!KytyShim::Entry::EopReleaseToken(Release(0u, 0x04000001u)).has_value(), "release without an interrupt is not an EOP flip");
    auto shortRelease = eop;
    shortRelease[0] = 0xc0054900u;
    Expect(!KytyShim::Entry::EopReleaseToken(shortRelease).has_value(), "release of another length is not an EOP flip");
}

void EopPoints() {
    const std::array<std::uint32_t, 2> nop{0xc0001000u, 0u};
    std::vector<std::uint32_t> words;
    Append(words, nop);
    Append(words, Release(2u, 0x04000001u));
    const auto firstEnd = words.size();
    Append(words, nop);
    Append(words, Release(2u, 0x04000002u));
    Append(words, Release(2u, 0x04000003u));
    const auto thirdEnd = words.size();
    Append(words, nop);
    const std::array<std::uint32_t, 2> pending{0x04000003u, 0x04000001u};
    const auto points = KytyShim::Entry::FindEopFlipPoints(words, pending);
    Expect(points.size() == 2, "two pending EOP flips found");
    if (points.size() == 2) {
        Expect(points[0].token == 0x04000001u && points[0].end == firstEnd, "first EOP flip splits right after its release");
        Expect(points[1].token == 0x04000003u && points[1].end == thirdEnd, "second EOP flip splits right after its release");
    }
    Expect(KytyShim::Entry::FindEopFlipPoints(words, {}).empty(), "no pending token, no split");
    std::vector<std::uint32_t> repeated;
    Append(repeated, Release(2u, 0x04000001u));
    Append(repeated, Release(2u, 0x04000001u));
    ExpectThrow([&] { static_cast<void>(KytyShim::Entry::FindEopFlipPoints(repeated, pending)); }, "repeated", "a token released twice in one submission fails");
}

void EopInIndirectBuffer() {
    std::vector<std::uint32_t> inner;
    Append(inner, std::array<std::uint32_t, 2>{0xc0001000u, 0u});
    Append(inner, Release(2u, 0x04000005u));
    const auto address = reinterpret_cast<std::uintptr_t>(inner.data());
    std::vector<std::uint32_t> outer{0xc0023f00u, static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u), static_cast<std::uint32_t>(inner.size())};
    const std::array<std::uint32_t, 1> pending{0x04000005u};
    const auto found = KytyShim::Entry::FindIndirectEopToken(outer, pending);
    Expect(found && *found == 0x04000005u, "EOP release inside an indirect buffer is detected");
    Expect(KytyShim::Entry::FindEopFlipPoints(outer, pending).empty(), "EOP release inside an indirect buffer is not split at the top level");
    Expect(!KytyShim::Entry::FindIndirectEopToken(inner, pending).has_value(), "a top-level release is not reported as indirect");
    const std::array<std::uint32_t, 1> other{0x04000006u};
    Expect(!KytyShim::Entry::FindIndirectEopToken(outer, other).has_value(), "a release of another token is ignored");
}

void DisplayBuffers() {
    AgcDriver::DisplayBuffer linear{0x100000000ull, 0x8000000000000000ull, 1920u, 1080u, 1u, 0u, 0u, 0u, 65536u};
    Expect(KytyShim::Entry::DisplayBufferBytes(linear) == 1920ull * 1080ull * 4ull, "linear 1080p BGRA8 size");
    linear.pitchInPixel = 2048u;
    Expect(KytyShim::Entry::DisplayBufferBytes(linear) == 2048ull * 1080ull * 4ull, "linear pitch is honoured");
    AgcDriver::DisplayBuffer tiled{0x100000000ull, 0x8000000022000000ull, 1920u, 1080u, 0u, 0u, 0u, 0u, 65536u};
    Expect(KytyShim::Entry::DisplayBufferBytes(tiled) == 15ull * 9ull * 65536ull, "64 KiB tiled 1080p RGBA8 size");
    AgcDriver::DisplayBuffer half{0x100000000ull, 0xc001000622000000ull, 1920u, 1080u, 0u, 0u, 0u, 0u, 65536u};
    Expect(KytyShim::Entry::DisplayBufferBytes(half) == 15ull * 17ull * 65536ull, "64 KiB tiled 1080p half-float size");
    AgcDriver::DisplayBuffer gcn{0x100000000ull, 0x8000000000000000ull, 1920u, 1080u, AgcDriver::DisplayTilingGcn, 1920u, 0u, 0u, 65536u};
    ExpectThrow([&] { static_cast<void>(KytyShim::Entry::DisplayBufferBytes(gcn)); }, "PS4", "PS4 tiled display buffers stop with a clear message");
    AgcDriver::DisplayBuffer badFormat = linear;
    badFormat.pixelFormat = 0x1234u;
    ExpectThrow([&] { static_cast<void>(KytyShim::Entry::DisplayBufferBytes(badFormat)); }, "unsupported display pixel format", "unknown pixel format fails");
    AgcDriver::DisplayBuffer misaligned = linear;
    misaligned.address += 4096u;
    ExpectThrow([&] { static_cast<void>(KytyShim::Entry::DisplayBufferBytes(misaligned)); }, "64 KiB alignment", "misaligned display buffer fails");
}

void MappedRanges() {
#ifdef _WIN32
    SYSTEM_INFO systemInfo{};
    GetSystemInfo(&systemInfo);
    const auto page = static_cast<std::size_t>(systemInfo.dwPageSize);
    void* mapped = VirtualAlloc(nullptr, page * 3, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    const auto page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    void* mapped = mmap(nullptr, page * 3, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapped == MAP_FAILED) mapped = nullptr;
#endif
    Expect(mapped != nullptr, "test pages mapped");
    if (mapped == nullptr) return;
    auto* bytes = static_cast<std::uint8_t*>(mapped);
    KytyShim::Entry::CheckMappedRange(bytes + 8, page * 3 - 8, 8);
#ifdef _WIN32
    DWORD previousProtection{};
    Expect(VirtualProtect(bytes + page, page, PAGE_NOACCESS, &previousProtection) != 0, "middle page protected");
#else
    mprotect(bytes + page, page, PROT_NONE);
#endif
    KytyShim::Entry::CheckMappedRange(bytes + page, page, 4);
#ifdef _WIN32
    Expect(VirtualFree(bytes + page * 2, page, MEM_DECOMMIT) != 0, "last page decommitted");
#else
    munmap(bytes + page * 2, page);
#endif
    ExpectThrow([&] { KytyShim::Entry::CheckMappedRange(bytes + page, page * 2, 4); }, "not mapped", "a range reaching an unmapped page fails");
    ExpectThrow([&] { KytyShim::Entry::CheckMappedRange(bytes + 2, 4, 4); }, "misaligned", "a misaligned range fails");
    ExpectThrow([&] { KytyShim::Entry::CheckMappedRange(nullptr, 4, 4); }, "null", "a null range fails");
    ExpectThrow([&] { KytyShim::Entry::CheckMappedRange(bytes, 4, 0); }, "alignment", "a zero alignment fails");
#ifdef _WIN32
    Expect(VirtualFree(mapped, 0, MEM_RELEASE) != 0, "test pages released");
#else
    munmap(bytes, page * 2);
#endif
}

}

int main() {
    PacketCodes();
    FlipRoundTrip();
    RenderingWait();
    EopRelease();
    EopPoints();
    EopInIndirectBuffer();
    DisplayBuffers();
    MappedRanges();
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("kyty shim entry: all checks passed\n");
    return 0;
}
