#define NOMINMAX
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestDirty.hpp"
#include <windows.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <functional>
#include <thread>

namespace {

constexpr std::size_t Page = 4096;
constexpr std::size_t Block = 65536;
std::atomic<unsigned> handled{0};

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

LONG CALLBACK HandleWrite(EXCEPTION_POINTERS* exception) {
    const auto* record = exception->ExceptionRecord;
    if (record->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || record->NumberParameters < 2 || record->ExceptionInformation[0] != 1) return EXCEPTION_CONTINUE_SEARCH;
    if (!GuestArena::GuestDirtyHandleWrite_nid_postfix(record->ExceptionInformation[1]) && !GuestArena::GuestArenaHandleWrite_nid_postfix(record->ExceptionInformation[1])) return EXCEPTION_CONTINUE_SEARCH;
    ++handled;
    return EXCEPTION_CONTINUE_EXECUTION;
}

struct Memory {
    unsigned char* bytes = static_cast<unsigned char*>(VirtualAlloc(nullptr, 4 * Block, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    Memory() { Require(bytes != nullptr, "cannot allocate test memory"); }
    ~Memory() {
        GuestArena::GuestDirtyMark_nid_postfix(reinterpret_cast<std::uintptr_t>(bytes), 4 * Block);
        VirtualFree(bytes, 0, MEM_RELEASE);
    }
    std::uintptr_t At(std::size_t offset) const { return reinterpret_cast<std::uintptr_t>(bytes) + offset; }
};

DWORD Protection(std::uintptr_t address) {
    MEMORY_BASIC_INFORMATION memory{};
    Require(VirtualQuery(reinterpret_cast<const void*>(address), &memory, sizeof(memory)) == sizeof(memory), "VirtualQuery failed");
    return memory.Protect;
}

void ArmedWriteFaultsOnce() {
    Memory memory;
    const auto token = GuestArena::GuestDirtyArm_nid_postfix(memory.At(0), Block);
    Require(token != 0, "a private read-write block was not armed");
    Require(Protection(memory.At(0)) == PAGE_READONLY, "an armed page is not write-protected");
    Require(GuestArena::GuestDirtyCleanSince_nid_postfix(memory.At(0), Block, token), "an armed, unwritten block is not clean");
    Require(GuestArena::GuestDirtyArmed_nid_postfix(memory.At(Page)), "an armed page reads as unarmed");
    const auto before = handled.load();
    static_cast<volatile unsigned char*>(memory.bytes)[3 * Page + 7] = 42;
    Require(handled.load() == before + 1, "a store into an armed page did not fault exactly once");
    Require(memory.bytes[3 * Page + 7] == 42, "the store did not land after the fault");
    Require(!GuestArena::GuestDirtyCleanSince_nid_postfix(memory.At(0), Block, token), "a written block is still clean");
    Require(!GuestArena::GuestDirtyArmed_nid_postfix(memory.At(0)), "a fault left another page of its block armed");
    Require(Protection(memory.At(0)) == PAGE_READWRITE, "a fault left the block protected");
    static_cast<volatile unsigned char*>(memory.bytes)[5] = 1;
    Require(handled.load() == before + 1, "a store after the fault faulted again");
}

void NeighbourBlockStaysClean() {
    Memory memory;
    const auto token = GuestArena::GuestDirtyArm_nid_postfix(memory.At(0), 2 * Block);
    Require(token != 0, "two blocks were not armed");
    static_cast<volatile unsigned char*>(memory.bytes)[Block + 1] = 9;
    Require(GuestArena::GuestDirtyCleanSince_nid_postfix(memory.At(0), Block, token), "a store into the second block dirtied the first");
    Require(!GuestArena::GuestDirtyCleanSince_nid_postfix(memory.At(Block), Block, token), "the written block is clean");
    Require(!GuestArena::GuestDirtyCleanSince_nid_postfix(memory.At(0), 2 * Block, token), "a range over a written block is clean");
}

void MarkDisarmsWithoutFault() {
    Memory memory;
    const auto token = GuestArena::GuestDirtyArm_nid_postfix(memory.At(0), Block);
    Require(token != 0, "the block was not armed");
    GuestArena::GuestDirtyMark_nid_postfix(memory.At(Page), 16);
    Require(!GuestArena::GuestDirtyCleanSince_nid_postfix(memory.At(0), Block, token), "a marked block is clean");
    Require(Protection(memory.At(0)) == PAGE_READWRITE, "a mark left the block protected");
    const auto before = handled.load();
    static_cast<volatile unsigned char*>(memory.bytes)[Page] = 3;
    Require(handled.load() == before, "a store after a mark faulted");
    const auto again = GuestArena::GuestDirtyArm_nid_postfix(memory.At(0), Block);
    Require(again > token && GuestArena::GuestDirtyCleanSince_nid_postfix(memory.At(0), Block, again), "a re-armed block is not clean for its new token");
    Require(!GuestArena::GuestDirtyCleanSince_nid_postfix(memory.At(0), Block, token), "a re-armed block is clean for the token before the write");
}

void PartlyArmedRangeIsNotClean() {
    Memory memory;
    const auto token = GuestArena::GuestDirtyArm_nid_postfix(memory.At(0), Page);
    Require(token != 0, "one page was not armed");
    Require(!GuestArena::GuestDirtyCleanSince_nid_postfix(memory.At(0), 2 * Page, token), "a range with an unarmed page is clean");
    const auto wider = GuestArena::GuestDirtyArm_nid_postfix(memory.At(0), 2 * Page);
    Require(wider != 0 && GuestArena::GuestDirtyCleanSince_nid_postfix(memory.At(0), 2 * Page, wider), "arming the rest of a partly armed range failed");
}

void RefusesWhatItCannotOwn() {
    Memory memory;
    DWORD previous = 0;
    Require(VirtualProtect(memory.bytes + Block, Page, PAGE_READONLY, &previous) != 0, "cannot make a read-only page");
    Require(GuestArena::GuestDirtyArm_nid_postfix(memory.At(Block), Page) == 0, "a page the guest made read-only was armed");
    Require(GuestArena::GuestDirtyArm_nid_postfix(memory.At(Block), Page) == 0, "a refused page was armed on the second try");
    Require(VirtualProtect(memory.bytes + Block, Page, PAGE_READWRITE, &previous) != 0, "cannot make the page writable again");
    GuestArena::GuestDirtyMappingChange_nid_postfix(memory.At(Block), Page);
    Require(GuestArena::GuestDirtyArm_nid_postfix(memory.At(Block), Page) != 0, "a page made writable again after a mapping change was not armed");
    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, static_cast<DWORD>(Block), nullptr);
    Require(section != nullptr, "cannot create a section");
    auto* view = static_cast<unsigned char*>(MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, Block));
    Require(view != nullptr, "cannot map a section");
    Require(GuestArena::GuestDirtyArm_nid_postfix(reinterpret_cast<std::uintptr_t>(view), Block) == 0, "a mapped (aliasable) view was armed");
    UnmapViewOfFile(view);
    CloseHandle(section);
}

void ProtectionChangeBlocksArming() {
    Memory memory;
    const auto token = GuestArena::GuestDirtyArm_nid_postfix(memory.At(0), Block);
    Require(token != 0, "the block was not armed");
    GuestArena::GuestDirtyBeginProtection_nid_postfix(memory.At(0), Block);
    Require(!GuestArena::GuestDirtyCleanSince_nid_postfix(memory.At(0), Block, token), "a protection change left the block clean");
    Require(GuestArena::GuestDirtyArm_nid_postfix(memory.At(0), Block) == 0, "a block was armed during a protection change");
    GuestArena::GuestDirtyEndProtection_nid_postfix(memory.At(0), Block);
    Require(GuestArena::GuestDirtyArm_nid_postfix(memory.At(0), Block) != 0, "a block was not armed after the protection change");
}

void WriteWatchSeesFaultedStores() {
    auto* bytes = static_cast<unsigned char*>(VirtualAlloc(nullptr, Block, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE));
    Require(bytes != nullptr, "cannot allocate write-watched memory");
    ResetWriteWatch(bytes, Block);
    const auto token = GuestArena::GuestDirtyArm_nid_postfix(reinterpret_cast<std::uintptr_t>(bytes), Block);
    Require(token != 0, "write-watched memory was not armed");
    void* pages[16]{};
    ULONG_PTR count = 16;
    DWORD granularity = 0;
    Require(GetWriteWatch(0, bytes, Block, pages, &count, &granularity) == 0 && count == 0, "arming reported a write to the write watch");
    static_cast<volatile unsigned char*>(bytes)[2 * Page] = 5;
    count = 16;
    Require(GetWriteWatch(0, bytes, Block, pages, &count, &granularity) == 0 && count == 1 && pages[0] == bytes + 2 * Page, "a store that faulted is missing from the write watch");
    Require(!GuestArena::GuestDirtyCleanSince_nid_postfix(reinterpret_cast<std::uintptr_t>(bytes), Block, token), "the written block is clean");
    GuestArena::GuestDirtyMark_nid_postfix(reinterpret_cast<std::uintptr_t>(bytes), Block);
    VirtualFree(bytes, 0, MEM_RELEASE);
}

void StoreUnderOwnBlockLock() {
    Memory memory;
    const auto token = GuestArena::GuestDirtyArm_nid_postfix(memory.At(0), Block);
    Require(token != 0, "the block was not armed");
    const auto before = handled.load();
    GuestArena::GuestDirtyRunLocked_nid_no_patch(memory.At(0), [](void* target) { static_cast<volatile unsigned char*>(target)[Page + 3] = 7; }, memory.bytes);
    Require(handled.load() == before + 1, "a store under the block's own lock was not served by the handler");
    Require(memory.bytes[Page + 3] == 7, "the store under the block's own lock did not land");
    Require(!GuestArena::GuestDirtyCleanSince_nid_postfix(memory.At(0), Block, token), "the block written under its own lock is clean");
}

void GpuBumpKeepsProtection() {
    Memory memory;
    const auto token = GuestArena::GuestDirtyArm_nid_postfix(memory.At(0), Block);
    Require(token != 0, "the block was not armed");
    GuestArena::GuestDirtyBump_nid_postfix(memory.At(Page), 4);
    Require(!GuestArena::GuestDirtyCleanSince_nid_postfix(memory.At(0), Block, token), "a GPU-written block is clean for the old token");
    Require(Protection(memory.At(0)) == PAGE_READONLY && GuestArena::GuestDirtyArmed_nid_postfix(memory.At(0)), "a GPU write opened the block");
    GuestArena::GuestDirtyCounts counts;
    GuestArena::GuestDirtyTakeCounts_nid_postfix(&counts);
    const auto again = GuestArena::GuestDirtyArm_nid_postfix(memory.At(0), Block);
    GuestArena::GuestDirtyTakeCounts_nid_postfix(&counts);
    Require(again > token && GuestArena::GuestDirtyCleanSince_nid_postfix(memory.At(0), Block, again), "re-arming after a GPU write is not clean");
    Require(counts.protectCalls == 0, "re-arming a still-armed block changed a protection");
    const auto before = handled.load();
    static_cast<volatile unsigned char*>(memory.bytes)[2 * Page] = 4;
    Require(handled.load() == before + 1, "a CPU store after a GPU write did not fault");
}

void BatchedQueries() {
    Memory memory;
    const auto token = GuestArena::GuestDirtyArm_nid_postfix(memory.At(0), 2 * Block);
    Require(token != 0, "the blocks were not armed");
    const GuestArena::GuestDirtyQuery both[] = {{memory.At(0), Page, token}, {memory.At(Block), Page, token}};
    Require(GuestArena::GuestDirtyCleanSinceAll_nid_postfix(both, 2), "two clean ranges are not clean together");
    static_cast<volatile unsigned char*>(memory.bytes)[Block] = 1;
    Require(!GuestArena::GuestDirtyCleanSinceAll_nid_postfix(both, 2), "a batch with a written range is clean");
    Require(!GuestArena::GuestDirtyCleanSinceAll_nid_postfix(both, 0), "an empty batch is clean");
}

void AliasWriteDirtiesOtherView() {
    constexpr std::size_t Bytes = 65536;
    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE | SEC_RESERVE, 0, static_cast<DWORD>(Bytes), nullptr);
    Require(section != nullptr, "cannot create a section");
    auto* first = static_cast<unsigned char*>(GuestArena::GuestArenaAllocate_nid_postfix(Bytes, Bytes));
    auto* second = static_cast<unsigned char*>(GuestArena::GuestArenaAllocate_nid_postfix(Bytes, Bytes));
    Require(first != nullptr && second != nullptr, "cannot allocate two arena ranges");
    GuestArena::GuestArenaMap_nid_postfix(first, Bytes, section, 0, PAGE_READWRITE);
    GuestArena::GuestArenaMap_nid_postfix(second, Bytes, section, 0, PAGE_READWRITE);
    static_cast<volatile unsigned char*>(first)[0] = 1;
    const auto token = GuestArena::GuestDirtyArm_nid_postfix(reinterpret_cast<std::uintptr_t>(first), Bytes);
    Require(token != 0, "a mapped view was not armed");
    Require(GuestArena::GuestDirtyCleanSince_nid_postfix(reinterpret_cast<std::uintptr_t>(first), Bytes, token), "an armed view is not clean");
    static_cast<volatile unsigned char*>(second)[3 * Page + 1] = 9;
    Require(first[3 * Page + 1] == 9, "the alias write is not visible through the first view");
    Require(!GuestArena::GuestDirtyCleanSince_nid_postfix(reinterpret_cast<std::uintptr_t>(first), Bytes, token), "a write through view B left view A clean");
    const auto again = GuestArena::GuestDirtyArm_nid_postfix(reinterpret_cast<std::uintptr_t>(first), Bytes);
    Require(again != 0 && GuestArena::GuestDirtyCleanSince_nid_postfix(reinterpret_cast<std::uintptr_t>(first), Bytes, again), "a re-armed view is not clean");
    static_cast<volatile unsigned char*>(first)[5] = 1;
    Require(!GuestArena::GuestDirtyCleanSince_nid_postfix(reinterpret_cast<std::uintptr_t>(first), Bytes, again), "a write through view A left it clean");
    const auto armed = GuestArena::GuestDirtyArm_nid_postfix(reinterpret_cast<std::uintptr_t>(first), Bytes);
    Require(armed != 0, "the view was not re-armed before the host write");
    {
        const GuestArena::HostWrite open(second + Page, 16);
        std::memset(second + Page, 7, 16);
    }
    Require(!GuestArena::GuestDirtyCleanSince_nid_postfix(reinterpret_cast<std::uintptr_t>(first), Bytes, armed), "a host write through view B left view A clean");
    const auto remapped = GuestArena::GuestDirtyArm_nid_postfix(reinterpret_cast<std::uintptr_t>(first), Bytes);
    Require(remapped != 0, "the view was not re-armed before the unmap");
    GuestArena::GuestArenaRelease_nid_postfix(second, Bytes);
    GuestArena::GuestArenaRelease_nid_postfix(first, Bytes);
    Require(!GuestArena::GuestDirtyCleanSince_nid_postfix(reinterpret_cast<std::uintptr_t>(first), Bytes, remapped), "an unmapped view is still clean");
    CloseHandle(section);
}

void HostWriteOpensArmedPages() {
    Memory memory;
    char path[MAX_PATH]{};
    char directory[MAX_PATH]{};
    Require(GetTempPathA(MAX_PATH, directory) != 0 && GetTempFileNameA(directory, "gdt", 0, path) != 0, "cannot make a temporary file");
    {
        HANDLE file = CreateFileA(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        Require(file != INVALID_HANDLE_VALUE, "cannot create the temporary file");
        const std::string payload(Page, 'x');
        DWORD written = 0;
        Require(WriteFile(file, payload.data(), static_cast<DWORD>(payload.size()), &written, nullptr) && written == payload.size(), "cannot fill the temporary file");
        CloseHandle(file);
    }
    HANDLE file = CreateFileA(path, GENERIC_READ, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    Require(file != INVALID_HANDLE_VALUE, "cannot open the temporary file");
    const auto token = GuestArena::GuestDirtyArm_nid_postfix(memory.At(0), Block);
    Require(token != 0, "the block was not armed");
    DWORD read = 0;
    SetFilePointer(file, 0, nullptr, FILE_BEGIN);
    const bool unopened = ReadFile(file, memory.bytes + Page, static_cast<DWORD>(Page), &read, nullptr) != 0;
    const DWORD unopenedError = unopened ? 0 : GetLastError();
    Require(!unopened && unopenedError == ERROR_NOACCESS, "a kernel-mode read into an armed page did not fail with ERROR_NOACCESS (the hazard the host-write bracket exists for)");
    SetFilePointer(file, 0, nullptr, FILE_BEGIN);
    {
        const GuestArena::HostWrite open(memory.bytes + Page, Page);
        Require(ReadFile(file, memory.bytes + Page, static_cast<DWORD>(Page), &read, nullptr) != 0 && read == Page, "a kernel-mode read inside a host-write bracket failed");
        Require(GuestArena::GuestDirtyArm_nid_postfix(memory.At(0), Block) == 0, "a block was armed while a host write was open");
    }
    CloseHandle(file);
    DeleteFileA(path);
    Require(memory.bytes[Page] == 'x', "the host write did not land");
    Require(!GuestArena::GuestDirtyCleanSince_nid_postfix(memory.At(0), Block, token), "a host write left the block clean");
}

std::uint64_t RenewAfter(const char* name, const std::function<void(Memory&)>& between) {
    Memory memory;
    Require(GuestArena::GuestDirtyArm_nid_postfix(memory.At(0), 2 * Block) != 0, name);
    const auto window = GuestArena::GuestDirtyOpenWindow_nid_postfix(memory.At(0), 2 * Block);
    Require(window != 0, "no window opened");
    GuestArena::GuestDirtyBump_nid_postfix(memory.At(2 * Page), 16);
    between(memory);
    GuestArena::GuestDirtyCloseWindow_nid_postfix();
    const auto token = GuestArena::GuestDirtyRenew_nid_postfix(memory.At(0), 2 * Block, window);
    if (token != 0) Require(GuestArena::GuestDirtyCleanSince_nid_postfix(memory.At(0), 2 * Block, token) && Protection(memory.At(Page)) == PAGE_READONLY, "a renewed token is not clean or left the block unprotected");
    return token;
}

void WindowOwnWritesRenew() {
    Require(RenewAfter("own GPU write", [](Memory&) {}) != 0, "the image's own GPU write inside its window refused the renewal");
    Require(RenewAfter("own CPU store", [](Memory& memory) { memory.bytes[5] = 1; }) == 0, "an own CPU store opened its block (others' stores there are invisible) and the token was still renewed");
    Require(RenewAfter("own driver store", [](Memory& memory) { GuestArena::GuestDirtyMark_nid_postfix(memory.At(Page), 16); }) == 0, "an own driver store opened its block and the token was still renewed");
}

void WindowForeignWritesRefuseRenewal() {
    const auto onThread = [](auto&& body) {
        std::thread other(body);
        other.join();
    };
    Require(RenewAfter("foreign store", [&](Memory& memory) {
        onThread([&] { memory.bytes[Block + 3 * Page] = 2; });
    }) == 0, "a CPU store by another thread into a still-armed block inside the window renewed the token");
    Require(RenewAfter("foreign mark", [&](Memory& memory) { onThread([&] { GuestArena::GuestDirtyMark_nid_postfix(memory.At(4 * Page), 8); }); }) == 0, "a driver store by another thread inside the window renewed the token");
    Require(RenewAfter("foreign bump", [&](Memory& memory) { onThread([&] { GuestArena::GuestDirtyBump_nid_postfix(memory.At(5 * Page), 8); }); }) == 0, "a GPU queue write recorded by another thread inside the window renewed the token");
    Require(RenewAfter("host write", [&](Memory& memory) {
        const GuestArena::HostWrite open(memory.bytes + 6 * Page, 16);
    }) == 0, "a host write inside the window renewed the token");
    Require(RenewAfter("host import", [&](Memory& memory) {
        const GuestArena::HostImportAccess open(memory.bytes, Block);
    }) == 0, "a host import inside the window renewed the token");
    Require(RenewAfter("mapping change", [&](Memory& memory) { GuestArena::GuestDirtyMappingChange_nid_postfix(memory.At(7 * Page), Page); }) == 0, "a mapping change inside the window renewed the token");
    Require(RenewAfter("protection change", [&](Memory& memory) {
        GuestArena::GuestDirtyBeginProtection_nid_postfix(memory.At(8 * Page), Page);
        GuestArena::GuestDirtyEndProtection_nid_postfix(memory.At(8 * Page), Page);
    }) == 0, "a guest protection change inside the window renewed the token");
}

void HostImportOpensArmedPages() {
    Memory memory;
    const auto token = GuestArena::GuestDirtyArm_nid_postfix(memory.At(0), Block);
    Require(token != 0 && Protection(memory.At(Page)) == PAGE_READONLY, "the block was not armed");
    {
        const GuestArena::HostImportAccess open(memory.bytes, Block);
        Require(Protection(memory.At(0)) == PAGE_READWRITE && Protection(memory.At(Block - Page)) == PAGE_READWRITE, "an armed page stayed read-only under a host import's check");
        Require(GuestArena::GuestDirtyArm_nid_postfix(memory.At(0), Block) == 0, "a block was armed while a host import was open");
    }
    Require(!GuestArena::GuestDirtyCleanSince_nid_postfix(memory.At(0), Block, token), "a host import left the block clean");
    Require(GuestArena::GuestDirtyArm_nid_postfix(memory.At(0), Block) != 0, "the block was not armable after the import closed");
}

}

void ArmCost() {
    constexpr std::size_t Bytes = std::size_t{64} << 20;
    auto* bytes = static_cast<unsigned char*>(VirtualAlloc(nullptr, Bytes, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE));
    Require(bytes != nullptr, "cannot allocate the cost buffer");
    for (std::size_t offset = 0; offset < Bytes; offset += Page) bytes[offset] = 1;
    const auto time = [](auto&& body) {
        const auto start = std::chrono::steady_clock::now();
        body();
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
    };
    DWORD previous = 0;
    const auto rawWhole = time([&] { VirtualProtect(bytes, Bytes, PAGE_READONLY, &previous); });
    const auto rawBack = time([&] { VirtualProtect(bytes, Bytes, PAGE_READWRITE, &previous); });
    const auto rawPages = time([&] { for (std::size_t offset = 0; offset < Bytes; offset += Block) VirtualProtect(bytes + offset, Block, PAGE_READONLY, &previous); });
    VirtualProtect(bytes, Bytes, PAGE_READWRITE, &previous);
    const auto arm = time([&] { Require(GuestArena::GuestDirtyArm_nid_postfix(reinterpret_cast<std::uintptr_t>(bytes), Bytes) != 0, "the cost buffer was not armed"); });
    const auto rearm = time([&] { Require(GuestArena::GuestDirtyArm_nid_postfix(reinterpret_cast<std::uintptr_t>(bytes), Bytes) != 0, "the cost buffer was not re-armed"); });
    const auto mark = time([&] { GuestArena::GuestDirtyMark_nid_postfix(reinterpret_cast<std::uintptr_t>(bytes), Bytes); });
    std::printf("COST 64 MiB (16384 pages): VirtualProtect whole RO %.0f us, back %.0f us, per 64 KiB %.0f us (%.2f us/call); Arm %.0f us (%.2f us/page), re-Arm (all armed) %.0f us, Mark %.0f us\n", rawWhole, rawBack, rawPages, rawPages / (Bytes / Block), arm, arm / (Bytes / Page), rearm, mark);
    VirtualFree(bytes, 0, MEM_RELEASE);
}

int main() {
    GuestArena::GuestDirtyForceEnable_nid_no_patch(true);
    void* handler = AddVectoredExceptionHandler(1, HandleWrite);
    Require(handler != nullptr, "cannot install the fault handler");
    const std::pair<const char*, void (*)()> tests[] = {
        {"armed write faults once", ArmedWriteFaultsOnce},
        {"neighbour block stays clean", NeighbourBlockStaysClean},
        {"mark disarms without a fault", MarkDisarmsWithoutFault},
        {"partly armed range is not clean", PartlyArmedRangeIsNotClean},
        {"refuses what it cannot own", RefusesWhatItCannotOwn},
        {"protection change blocks arming", ProtectionChangeBlocksArming},
        {"write watch sees faulted stores", WriteWatchSeesFaultedStores},
        {"store under the block's own lock", StoreUnderOwnBlockLock},
        {"batched queries", BatchedQueries},
        {"GPU bump keeps protection", GpuBumpKeepsProtection},
        {"alias write dirties the other view", AliasWriteDirtiesOtherView},
        {"host write opens armed pages", HostWriteOpensArmedPages},
        {"host import opens armed pages", HostImportOpensArmedPages},
        {"window: own GPU writes renew, own CPU stores do not", WindowOwnWritesRenew},
        {"window: other writers refuse the renewal", WindowForeignWritesRefuseRenewal},
        {"arm cost", ArmCost},
    };
    int failed = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "PASS " << name << "\n";
        } catch (const std::exception& error) {
            std::cout << "FAIL " << name << ": " << error.what() << "\n";
            ++failed;
        }
    }
    RemoveVectoredExceptionHandler(handler);
    return failed == 0 ? 0 : 1;
}
