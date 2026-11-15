#define NOMINMAX
#include "prx/libc/include/WindowsMappings.hpp"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <source_location>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Mappings = GuestArena::WindowsMappings;
constexpr std::size_t PageBytes = 0x4000;
constexpr std::size_t SubpageBytes = 0x1000;
constexpr std::size_t Pages = 8;
constexpr std::size_t Bytes = Pages * PageBytes;
constexpr unsigned char Initial = 0x11;
constexpr unsigned char Published = 0x5a;

std::atomic<unsigned> resolved{0};
std::atomic<bool> refuse{false};
std::atomic<std::uintptr_t> resolvedBegin{0};
std::atomic<unsigned char*> writableAlias{nullptr};
std::atomic<std::uintptr_t> guestBase{0};

void Require(bool value, std::source_location location = std::source_location::current()) {
    if (!value) throw std::runtime_error("Windows read trap regression at line " + std::to_string(location.line()));
}

bool Resolve(std::uintptr_t begin, std::uintptr_t end) {
    resolved.fetch_add(1);
    resolvedBegin.store(begin);
    if (refuse.load()) return false;
    std::memset(writableAlias.load() + (begin - guestBase.load()), Published, end - begin);
    Mappings::Get().ReleaseReads(begin, end - begin);
    return true;
}

LONG CALLBACK Handler(EXCEPTION_POINTERS* info) {
    const auto* record = info->ExceptionRecord;
    if (record->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || record->NumberParameters < 2) return EXCEPTION_CONTINUE_SEARCH;
    const auto address = static_cast<std::uintptr_t>(record->ExceptionInformation[1]);
    if (Mappings::Get().HandleTrap(address)) return EXCEPTION_CONTINUE_EXECUTION;
    if (record->ExceptionInformation[0] == 1 && Mappings::Get().HandleWrite(address)) return EXCEPTION_CONTINUE_EXECUTION;
    return EXCEPTION_CONTINUE_SEARCH;
}

DWORD HostProtection(const unsigned char* address) {
    MEMORY_BASIC_INFORMATION memory{};
    Require(VirtualQuery(address, &memory, sizeof(memory)) == sizeof(memory));
    return memory.Protect;
}

unsigned char ReadByte(const unsigned char* address) {
    return *static_cast<const volatile unsigned char*>(address);
}

void WriteByte(unsigned char* address, unsigned char value) {
    *static_cast<volatile unsigned char*>(address) = value;
}

std::vector<void*> Collected(unsigned char* base) {
    std::vector<void*> pages(Bytes / SubpageBytes);
    std::size_t count = pages.size();
    Require(Mappings::Get().Collect(reinterpret_cast<std::uintptr_t>(base), Bytes, pages.data(), &count, true));
    pages.resize(count);
    return pages;
}

void CheckReadTraps() {
    auto& mappings = Mappings::Get();
    const auto section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE, 0, static_cast<DWORD>(Bytes), nullptr);
    auto* base = static_cast<unsigned char*>(mappings.Reserve(nullptr, Bytes));
    Require(section != nullptr && base != nullptr);
    const auto start = reinterpret_cast<std::uintptr_t>(base);
    mappings.Map(base, Bytes, section, 0, PAGE_READWRITE);
    std::memset(base, Initial, Bytes);
    auto* alias = static_cast<unsigned char*>(mappings.MapAlias(start, Bytes));
    Require(alias != nullptr && alias != base);
    writableAlias.store(alias);
    guestBase.store(start);

    Require(!mappings.TrapReads(start + SubpageBytes, 2 * PageBytes));
    mappings.SetTrapResolver(&Resolve);
    Require(!mappings.TrapReads(start + 1, SubpageBytes));
    Require(!mappings.TrapReads(start + SubpageBytes, SubpageBytes + 1));
    Require(!mappings.TrapReads(start + Bytes, PageBytes));
    Require(!mappings.ReadsTrapped(start, Bytes));

    Require(mappings.TrapReads(start + SubpageBytes, 2 * PageBytes));
    Require(mappings.TrapReads(start + SubpageBytes, 2 * PageBytes));
    Require(mappings.ReadsTrapped(start + SubpageBytes, 1) && !mappings.ReadsTrapped(start, SubpageBytes));
    Require(HostProtection(base + SubpageBytes) == PAGE_NOACCESS);
    MEMORY_BASIC_INFORMATION described{};
    Require(mappings.QueryOpened(start + SubpageBytes + 7, &described));
    Require(described.Protect == PAGE_READWRITE && described.RegionSize == SubpageBytes && described.BaseAddress == base + SubpageBytes);
    std::uint32_t logical = 0;
    Require(mappings.Protection(start + SubpageBytes, &logical) && logical == PAGE_READWRITE);

    Require(ReadByte(base) == Initial && resolved.load() == 0);
    alias[SubpageBytes + 9] = 0x77;
    Require(resolved.load() == 0 && HostProtection(base + SubpageBytes) == PAGE_NOACCESS);

    Require(ReadByte(base + SubpageBytes + 5) == Published);
    Require(resolved.load() == 1 && resolvedBegin.load() == start + SubpageBytes);
    Require(!mappings.ReadsTrapped(start + SubpageBytes, SubpageBytes) && mappings.ReadsTrapped(start + 2 * SubpageBytes, SubpageBytes));
    Require(ReadByte(base + SubpageBytes + 6) == Published && resolved.load() == 1);

    WriteByte(base + 2 * SubpageBytes, 9);
    Require(resolved.load() == 2 && ReadByte(base + 2 * SubpageBytes) == 9 && ReadByte(base + 2 * SubpageBytes + 1) == Published);

    const auto first = Collected(base);
    Require(!first.empty());
    Require(HostProtection(base + 3 * SubpageBytes) == PAGE_NOACCESS && mappings.ReadsTrapped(start + 3 * SubpageBytes, SubpageBytes));
    mappings.ReleaseReads(start + SubpageBytes, 2 * PageBytes);
    Require(!mappings.ReadsTrapped(start, Bytes));
    Require(ReadByte(base + 3 * SubpageBytes) == Initial && resolved.load() == 2);
    static_cast<void>(Collected(base));
    WriteByte(base + 3 * SubpageBytes, 4);
    const auto written = Collected(base);
    Require(std::find(written.begin(), written.end(), static_cast<void*>(base + 3 * SubpageBytes)) != written.end());
    Require(resolved.load() == 2);

    Require(mappings.TrapReads(start + 4 * PageBytes, PageBytes));
    refuse.store(true);
    Require(!mappings.BeginHostWrite(start + 4 * PageBytes + 3, 16));
    Require(resolved.load() == 3 && mappings.ReadsTrapped(start + 4 * PageBytes, PageBytes));
    refuse.store(false);
    Require(mappings.BeginHostWrite(start + 4 * PageBytes + 3, 16));
    Require(!mappings.ReadsTrapped(start + 4 * PageBytes, PageBytes) && HostProtection(base + 4 * PageBytes) != PAGE_NOACCESS);
    Require(!mappings.TrapReads(start + 4 * PageBytes, PageBytes));
    mappings.EndHostWrite(start + 4 * PageBytes + 3, 16);
    Require(ReadByte(base + 4 * PageBytes) == Published);

    mappings.Pin(start + 5 * PageBytes, PageBytes);
    Require(!mappings.TrapReads(start + 5 * PageBytes, PageBytes));
    mappings.Unpin(start + 5 * PageBytes, PageBytes);
    Require(mappings.TrapReads(start + 5 * PageBytes, PageBytes));
    mappings.Pin(start + 5 * PageBytes, PageBytes);
    Require(!mappings.ReadsTrapped(start + 5 * PageBytes, PageBytes) && ReadByte(base + 5 * PageBytes) == Published);
    mappings.Unpin(start + 5 * PageBytes, PageBytes);

    Require(mappings.TrapReads(start + 6 * PageBytes, PageBytes));
    const auto before = resolved.load();
    mappings.Reset(base, Bytes);
    Require(resolved.load() == before + 1 && !mappings.ReadsTrapped(start, Bytes));
    Require(alias[6 * PageBytes] == Published && alias[7 * PageBytes] == Initial);

    mappings.SetTrapResolver(nullptr);
    writableAlias.store(nullptr);
    mappings.UnmapAlias(alias);
    Require(VirtualFree(base, 0, MEM_RELEASE));
    mappings.ForgetSection(section);
    CloseHandle(section);
}

}

int main() {
    const auto handler = AddVectoredExceptionHandler(1, Handler);
    try {
        Require(handler != nullptr);
        CheckReadTraps();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    RemoveVectoredExceptionHandler(handler);
    std::printf("Windows read traps passed\n");
    return 0;
}
