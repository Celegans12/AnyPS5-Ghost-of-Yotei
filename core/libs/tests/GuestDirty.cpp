#define NOMINMAX
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestDirty.hpp"
#include <windows.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>

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
    if (!GuestArena::GuestDirtyHandleWrite_nid_postfix(record->ExceptionInformation[1])) return EXCEPTION_CONTINUE_SEARCH;
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
        {"host write opens armed pages", HostWriteOpensArmedPages},
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
