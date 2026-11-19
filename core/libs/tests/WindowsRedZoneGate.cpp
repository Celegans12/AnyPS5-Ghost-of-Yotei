#define NOMINMAX
#include "prx/libc/include/WindowsMappings.hpp"
#include "prx/libc/include/WindowsPageTrack.hpp"
#include "prx/libc/include/HostFault.hpp"
#include "prx/libc/include/RedZoneGate.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <source_location>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Mappings = GuestArena::WindowsMappings;
using Track = GuestArena::WindowsPageTrack;
constexpr std::size_t PageBytes = 0x4000;
constexpr std::size_t SubpageBytes = 0x1000;
constexpr std::size_t Pages = 8;
constexpr std::size_t Bytes = Pages * PageBytes;
constexpr std::size_t Blocks = 4;
constexpr wchar_t HostPath[] = L"C:\\title\\libs\\libSceFiber.prx";
constexpr wchar_t GuestPath[] = L"C:\\title\\app0\\Media\\Modules\\Il2CppUserAssemblies.prx.GUEST.prx";

void Require(bool value, std::source_location location = std::source_location::current()) {
    if (!value) throw std::runtime_error("Windows red-zone gate regression at line " + std::to_string(location.line()));
}

std::size_t Length(const wchar_t* text) {
    std::size_t length = 0;
    while (text[length] != L'\0') ++length;
    return length;
}

std::array<unsigned char, RedZoneGate::HeaderBytes> Header(bool relinked, bool marked) {
    std::array<unsigned char, RedZoneGate::HeaderBytes> header{};
    header[0] = 'M';
    header[1] = 'Z';
    header[0x3c] = 0x80;
    if (!relinked) std::memcpy(header.data() + 0x4e, "This program cannot be run in DOS mode.", 39);
    if (marked) std::memcpy(header.data() + RedZoneGate::MarkOffset, RedZoneGate::Mark, sizeof(RedZoneGate::Mark));
    return header;
}

std::vector<unsigned char> RangesImage(std::initializer_list<std::pair<std::uint32_t, std::uint32_t>> ranges, std::uint32_t tableRva = 0x100, std::size_t bytes = 0x1000) {
    std::vector<unsigned char> image(bytes);
    const auto header = Header(true, false);
    std::copy(header.begin(), header.end(), image.begin());
    std::memcpy(image.data() + RedZoneGate::MarkOffset, RedZoneGate::RangesMark, sizeof(RedZoneGate::RangesMark));
    const auto count = static_cast<std::uint32_t>(ranges.size());
    std::memcpy(image.data() + RedZoneGate::TableRvaOffset, &tableRva, sizeof(tableRva));
    std::memcpy(image.data() + RedZoneGate::TableCountOffset, &count, sizeof(count));
    std::size_t at = tableRva;
    for (const auto& [begin, end] : ranges) {
        if (at + RedZoneGate::TableEntryBytes > bytes) break;
        std::memcpy(image.data() + at, &begin, sizeof(begin));
        std::memcpy(image.data() + at + sizeof(begin), &end, sizeof(end));
        at += RedZoneGate::TableEntryBytes;
    }
    return image;
}

bool Refuse(std::uintptr_t, std::uintptr_t) {
    return false;
}

DWORD HostProtection(const unsigned char* address) {
    MEMORY_BASIC_INFORMATION memory{};
    Require(VirtualQuery(address, &memory, sizeof(memory)) == sizeof(memory));
    return memory.Protect;
}

void CheckClassification() {
    using RedZoneGate::Classify;
    using RedZoneGate::Image;
    const auto host = Header(false, false);
    const auto unframed = Header(true, false);
    const auto framed = Header(true, true);
    const auto markedHost = Header(false, true);
    Require(Classify(host.data(), host.size(), false) == Image::Host);
    Require(Classify(host.data(), host.size(), true) == Image::Unframed);
    Require(Classify(markedHost.data(), markedHost.size(), false) == Image::Host);
    Require(Classify(markedHost.data(), markedHost.size(), true) == Image::Framed);
    Require(Classify(unframed.data(), unframed.size(), false) == Image::Unframed);
    Require(Classify(unframed.data(), unframed.size(), true) == Image::Unframed);
    Require(Classify(framed.data(), framed.size(), false) == Image::Framed);
    Require(Classify(framed.data(), framed.size(), true) == Image::Framed);
    Require(Classify(nullptr, 0, false) == Image::Host);
    Require(Classify(nullptr, 0, true) == Image::Unframed);
    Require(Classify(framed.data(), RedZoneGate::HeaderBytes - 1, true) == Image::Unframed);
    auto elsewhere = Header(true, true);
    elsewhere[0x3c] = 0x40;
    Require(Classify(elsewhere.data(), elsewhere.size(), false) == Image::Host);
    auto partial = Header(true, true);
    partial[RedZoneGate::MarkOffset + 7] = '0';
    Require(Classify(partial.data(), partial.size(), false) == Image::Unframed);
    Require(RedZoneGate::GuestName(GuestPath, Length(GuestPath)));
    Require(!RedZoneGate::GuestName(HostPath, Length(HostPath)));
    Require(!RedZoneGate::GuestName(L".prx", 4) && !RedZoneGate::GuestName(nullptr, 0));
}

void CheckRanges() {
    using RedZoneGate::Classify;
    using RedZoneGate::Image;
    using RedZoneGate::ValidTable;
    static const auto image = RangesImage({{0x200, 0x240}, {0x300, 0x310}});
    Require(Classify(image.data(), image.size(), false) == Image::FramedExcept);
    Require(Classify(image.data(), image.size(), true) == Image::FramedExcept);
    auto notRelinked = image;
    std::memcpy(notRelinked.data() + 0x4e, "This program cannot be run in DOS mode.", 39);
    Require(Classify(notRelinked.data(), notRelinked.size(), false) == Image::Host);
    std::uint32_t rva = 0;
    std::uint32_t count = 0;
    Require(ValidTable(image.data(), image.size(), rva, count) && rva == 0x100 && count == 2);
    Require(!ValidTable(image.data(), RedZoneGate::HeaderBytes - 1, rva, count));
    Require(!ValidTable(RangesImage({}).data(), 0x1000, rva, count));
    Require(!ValidTable(RangesImage({{0x300, 0x310}, {0x200, 0x240}}).data(), 0x1000, rva, count));
    Require(!ValidTable(RangesImage({{0x200, 0x240}, {0x230, 0x250}}).data(), 0x1000, rva, count));
    Require(!ValidTable(RangesImage({{0x240, 0x240}}).data(), 0x1000, rva, count));
    Require(!ValidTable(RangesImage({{0x200, 0x1001}}).data(), 0x1000, rva, count));
    Require(!ValidTable(RangesImage({{0x200, 0x240}}, 0x102).data(), 0x1000, rva, count));
    Require(!ValidTable(RangesImage({{0x200, 0x240}}, 0x40).data(), 0x1000, rva, count));
    Require(!ValidTable(RangesImage({{0x200, 0x240}}, 0x1000).data(), 0x1000, rva, count));
    auto tooMany = RangesImage({{0x200, 0x240}});
    const std::uint32_t manyEntries = 0x1000;
    std::memcpy(tooMany.data() + RedZoneGate::TableCountOffset, &manyEntries, sizeof(manyEntries));
    Require(!ValidTable(tooMany.data(), tooMany.size(), rva, count));

    RedZoneGate::Note(GuestPath, Length(GuestPath), image.data(), image.size());
    RedZoneGate::Note(GuestPath, Length(GuestPath), image.data(), image.size());
    const auto base = reinterpret_cast<std::uintptr_t>(image.data());
    RedZoneGate::Range range;
    Require(RedZoneGate::InUnframedRange(base + 0x200, &range) && range.begin == base + 0x200 && range.end == base + 0x240);
    Require(std::strcmp(range.image, "Il2CppUserAssemblies.prx.GUEST.prx") == 0);
    Require(RedZoneGate::InUnframedRange(base + 0x23f) && !RedZoneGate::InUnframedRange(base + 0x240));
    Require(!RedZoneGate::InUnframedRange(base + 0x1ff) && !RedZoneGate::InUnframedRange(base + 0x2ff));
    Require(RedZoneGate::InUnframedRange(base + 0x300, &range) && range.begin == base + 0x300 && range.end == base + 0x310);
    Require(RedZoneGate::InUnframedRange(base + 0x30f) && !RedZoneGate::InUnframedRange(base + 0x310));
    Require(!RedZoneGate::InUnframedRange(base - 1) && !RedZoneGate::InUnframedRange(base + image.size()));
    Require(RedZoneGate::Open());
}

void CheckFaultSite() {
    static const auto image = RangesImage({{0x200, 0x240}, {0x300, 0x310}});
    RedZoneGate::Note(GuestPath, Length(GuestPath), image.data(), image.size());
    const auto base = reinterpret_cast<std::uintptr_t>(image.data());
    alignas(16) CONTEXT context{};
    EXCEPTION_RECORD record{};
    EXCEPTION_POINTERS pointers{&record, &context};
    context.Rip = base + 0x210;
    for (const DWORD code : {static_cast<DWORD>(EXCEPTION_ACCESS_VIOLATION), static_cast<DWORD>(STATUS_GUARD_PAGE_VIOLATION), static_cast<DWORD>(EXCEPTION_SINGLE_STEP)}) {
        record.ExceptionCode = code;
        Require(RedZoneGate::FaultSite(&pointers) == base + 0x210);
    }
    for (const DWORD code : {static_cast<DWORD>(EXCEPTION_BREAKPOINT), static_cast<DWORD>(EXCEPTION_INT_DIVIDE_BY_ZERO), static_cast<DWORD>(EXCEPTION_ILLEGAL_INSTRUCTION)}) {
        record.ExceptionCode = code;
        Require(RedZoneGate::FaultSite(&pointers) == 0);
    }
    record.ExceptionCode = EXCEPTION_ACCESS_VIOLATION;
    context.Rip = base + 0x240;
    Require(RedZoneGate::FaultSite(&pointers) == 0);
    context.Rip = base + 0x1ff;
    Require(RedZoneGate::FaultSite(&pointers) == 0);
    context.Rip = base + 0x30f;
    Require(RedZoneGate::FaultSite(&pointers) == base + 0x30f);
    context.Rip = 0;
    Require(RedZoneGate::FaultSite(&pointers) == 0);
    Require(RedZoneGate::FaultSite(nullptr) == 0);
    EXCEPTION_POINTERS partial{&record, nullptr};
    Require(RedZoneGate::FaultSite(&partial) == 0);
    partial = {nullptr, &context};
    Require(RedZoneGate::FaultSite(&partial) == 0);

    context.Rip = base + 0x2f0;
    Require(RedZoneGate::FaultSite(&pointers) == 0);
    context.Rip = base + 0x23f;
    const auto before = RedZoneGate::FaultSite(&pointers);
    Require(before == base + 0x23f);
    context.Rip = base + 0x240;
    Require(RedZoneGate::FaultSite(&pointers) == 0);
    Require(LibcUnframedFaultSite_nid_no_patch(nullptr) == 0);
    LibcRefuseUnframedFault_nid_no_patch(0);
}

void CheckGate() {
    auto& mappings = Mappings::Get();
    auto& track = Track::Get();
    Require(RedZoneGate::Open());
    Require(track.Enabled());

    auto* memory = static_cast<unsigned char*>(VirtualAlloc(nullptr, Blocks * Track::BlockBytes, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE));
    Require(memory != nullptr);
    const auto first = reinterpret_cast<std::uintptr_t>(memory);
    Require(first % Track::BlockBytes == 0);

    const auto section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE, 0, static_cast<DWORD>(Bytes), nullptr);
    auto* base = static_cast<unsigned char*>(mappings.Reserve(nullptr, Bytes));
    Require(section != nullptr && base != nullptr);
    const auto start = reinterpret_cast<std::uintptr_t>(base);
    mappings.Map(base, Bytes, section, 0, PAGE_READWRITE);
    std::memset(base, 0x11, Bytes);
    auto* alias = static_cast<unsigned char*>(mappings.MapAlias(start, Bytes));
    Require(alias != nullptr && alias != base);
    mappings.SetTrapResolver(&Refuse);

    const auto host = Header(false, false);
    const auto framed = Header(true, true);
    RedZoneGate::Note(HostPath, Length(HostPath), host.data(), host.size());
    RedZoneGate::Note(GuestPath, Length(GuestPath), framed.data(), framed.size());
    Require(RedZoneGate::Open());
    Require(track.Arm(first) && track.Clean(first) && HostProtection(memory) == PAGE_READONLY);
    Require(mappings.TrapReads(start + SubpageBytes, SubpageBytes));
    Require(mappings.ReadsTrapped(start + SubpageBytes, 1) && HostProtection(base + SubpageBytes) == PAGE_NOACCESS);

    const auto unframed = Header(true, false);
    RedZoneGate::Note(GuestPath, Length(GuestPath), unframed.data(), unframed.size());
    Require(!RedZoneGate::Open() && !RedZoneGate::Open());
    Require(std::strcmp(RedZoneGate::State().name, "Il2CppUserAssemblies.prx.GUEST.prx") == 0);

    Require(!track.Arm(first + Track::BlockBytes));
    Require(!track.Clean(first + Track::BlockBytes) && HostProtection(memory + Track::BlockBytes) == PAGE_READWRITE);
    memory[Track::BlockBytes] = 7;
    Require(memory[Track::BlockBytes] == 7);
    Require(!mappings.TrapReads(start + 2 * PageBytes, SubpageBytes));
    Require(!mappings.ReadsTrapped(start + 2 * PageBytes, SubpageBytes) && HostProtection(base + 2 * PageBytes) == PAGE_READWRITE);
    Require(base[2 * PageBytes] == 0x11);
    Require(track.Clean(first) && mappings.ReadsTrapped(start + SubpageBytes, 1));

    mappings.ReleaseReads(start, Bytes);
    Require(!mappings.ReadsTrapped(start, Bytes));
    mappings.Reset(base, Bytes);
    mappings.SetTrapResolver(nullptr);
    mappings.UnmapAlias(alias);
    Require(VirtualFree(base, 0, MEM_RELEASE));
    mappings.ForgetSection(section);
    CloseHandle(section);
    track.Forget(first, first + Blocks * Track::BlockBytes);
    Require(HostProtection(memory) == PAGE_READWRITE);
    Require(VirtualFree(memory, 0, MEM_RELEASE));
}

}

int main() {
    _putenv_s("APS5_PAGE_TRACK", "1");
    _putenv_s("APS5_ALLOW_UNFRAMED_FAULTS", "");
    try {
        CheckClassification();
        CheckRanges();
        CheckFaultSite();
        CheckGate();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    std::printf("Windows red-zone gate passed\n");
    return 0;
}
