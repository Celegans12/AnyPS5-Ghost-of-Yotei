#define NOMINMAX
#include "prx/libc/include/WindowsMappings.hpp"
#include "prx/libc/include/WindowsPageTrack.hpp"
#include "prx/libc/include/RedZoneGate.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <source_location>
#include <stdexcept>
#include <string>

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
        CheckGate();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    std::printf("Windows red-zone gate passed\n");
    return 0;
}
