#define NOMINMAX
#include "prx/libc/include/WindowsMappings.hpp"
#include "prx/libc/include/WindowsPageTrack.hpp"
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {
using Mappings = GuestArena::WindowsMappings;
using Track = GuestArena::WindowsPageTrack;
using State = Track::State;
constexpr std::size_t PageBytes = 4096;
constexpr std::size_t BlockBytes = Track::BlockBytes;
constexpr std::size_t BlockPages = BlockBytes / PageBytes;
constexpr std::size_t RunBytes = 0x200000;
std::atomic<std::uint64_t> handled{0};
std::atomic<std::uint64_t> refused{0};

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

LONG CALLBACK HandleFault(EXCEPTION_POINTERS* exception) {
    const auto* record = exception->ExceptionRecord;
    if (record->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || record->NumberParameters < 2 || record->ExceptionInformation[0] != 1) return EXCEPTION_CONTINUE_SEARCH;
    if (Track::Get().HandleWrite(record->ExceptionInformation[1])) {
        handled.fetch_add(1, std::memory_order_relaxed);
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    refused.fetch_add(1, std::memory_order_relaxed);
    return EXCEPTION_CONTINUE_SEARCH;
}

class Raw {
public:
    explicit Raw(std::size_t size) : bytes(size) {
        data = static_cast<unsigned char*>(VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE));
        Require(data != nullptr, "cannot allocate write-watched test memory");
        for (std::size_t at = 0; at < bytes; at += PageBytes) data[at] = 1;
        Require(ResetWriteWatch(data, bytes) == 0, "cannot reset the write watch of test memory");
    }
    ~Raw() {
        Track::Get().Forget(Address(), Address() + bytes);
        VirtualFree(data, 0, MEM_RELEASE);
    }
    Raw(const Raw&) = delete;
    Raw& operator=(const Raw&) = delete;

    std::uintptr_t Address() const { return reinterpret_cast<std::uintptr_t>(data); }
    std::uintptr_t Block(std::size_t index) const { return Address() + index * BlockBytes; }
    void Write(std::size_t block, std::size_t page, unsigned char value) const { static_cast<volatile unsigned char*>(data)[block * BlockBytes + page * PageBytes + 3] = value; }
    DWORD Protection(std::size_t block) const {
        MEMORY_BASIC_INFORMATION memory{};
        Require(VirtualQuery(data + block * BlockBytes, &memory, sizeof(memory)) == sizeof(memory), "cannot query test memory");
        return memory.Protect;
    }
    std::set<std::size_t> Watched(std::size_t block, bool reset) const {
        std::array<void*, BlockPages> written{};
        ULONG_PTR count = written.size();
        DWORD granularity = 0;
        Require(GetWriteWatch(reset ? WRITE_WATCH_FLAG_RESET : 0, data + block * BlockBytes, BlockBytes, written.data(), &count, &granularity) == 0, "GetWriteWatch failed");
        std::set<std::size_t> pages;
        for (ULONG_PTR index = 0; index < count; ++index) pages.insert(static_cast<std::size_t>(static_cast<unsigned char*>(written[index]) - (data + block * BlockBytes)) / PageBytes);
        return pages;
    }

    unsigned char* data = nullptr;
    std::size_t bytes;
};

void CheckFaults() {
    auto& track = Track::Get();
    Raw memory(4 * BlockBytes);
    Require(track.StateOf(memory.Block(1)) == State::Unarmed && !track.Clean(memory.Block(1)), "a block starts armed");
    Require(track.Arm(memory.Block(1)), "cannot arm a block");
    Require(track.Clean(memory.Block(1)) && memory.Protection(1) == PAGE_READONLY && memory.Protection(0) == PAGE_READWRITE && memory.Protection(2) == PAGE_READWRITE, "arming did not write-protect exactly its block");
    Require(!track.Arm(memory.Block(1)), "an armed block was armed twice");
    Require(track.Counted().armed == 1, "the armed block is not counted");
    const auto before = handled.load();
    memory.Write(1, 5, 7);
    Require(handled.load() - before == 1 && memory.data[BlockBytes + 5 * PageBytes + 3] == 7, "the write to an armed block was not replayed after one fault");
    Require(track.StateOf(memory.Block(1)) == State::Written && track.Faults(memory.Block(1)) == 1 && memory.Protection(1) == PAGE_READWRITE, "the fault did not mark the block written and reopen it");
    Require(track.Counted().armed == 0 && track.Counted().faults >= 1, "the fault is not counted");
    memory.Write(1, 6, 8);
    Require(handled.load() - before == 1, "a reopened block faulted again");
    Require(memory.Watched(1, true) == (std::set<std::size_t>{5, 6}), "the write watch lost a write of a reopened block");
    Require(track.TakeWritten(memory.Block(1)) && !track.TakeWritten(memory.Block(1)) && track.StateOf(memory.Block(1)) == State::Unarmed, "a written block is not taken once");
    Require(track.HandleWrite(memory.Block(1) + 9), "a late fault on a reopened block was refused");
    DWORD previous;
    Require(VirtualProtect(memory.data + 2 * BlockBytes, BlockBytes, PAGE_READONLY, &previous) != 0, "cannot protect a block by hand");
    Require(!track.HandleWrite(memory.Block(2) + 9), "a fault on a read-only block that was never armed was handled");
    Require(VirtualProtect(memory.data + 2 * BlockBytes, BlockBytes, PAGE_READWRITE, &previous) != 0, "cannot reopen a block by hand");
    Require(track.Arm(memory.Block(2)), "cannot arm a second block");
    track.Open(memory.Block(2), memory.Block(2) + 1);
    Require(track.StateOf(memory.Block(2)) == State::Unarmed && memory.Protection(2) == PAGE_READWRITE && track.Faults(memory.Block(2)) == 0, "opening an armed block did not leave it clean and writable");
}

void CheckQuery() {
    auto& track = Track::Get();
    Raw memory(8 * BlockBytes);
    DWORD previous;
    Require(VirtualProtect(memory.data + 3 * BlockBytes, BlockBytes, PAGE_READONLY, &previous) != 0, "cannot protect a block by hand");
    Require(track.Arm(memory.Block(2)) && track.Arm(memory.Block(4)) && track.Arm(memory.Block(5)), "cannot arm blocks");
    MEMORY_BASIC_INFORMATION memoryInfo{};
    Require(track.Query(memory.Block(2) + 100, &memoryInfo) && memoryInfo.Protect == PAGE_READWRITE && reinterpret_cast<std::uintptr_t>(memoryInfo.BaseAddress) == memory.Block(2) && memoryInfo.RegionSize == BlockBytes, "an armed block is not reported read-write over exactly its extent");
    Require(track.Query(memory.Block(3) + 100, &memoryInfo) && memoryInfo.Protect == PAGE_READONLY && reinterpret_cast<std::uintptr_t>(memoryInfo.BaseAddress) == memory.Block(3) && memoryInfo.RegionSize == BlockBytes, "a read-only block between two armed blocks is not reported read-only over exactly its extent");
    Require(track.Query(memory.Block(5) + 100, &memoryInfo) && memoryInfo.Protect == PAGE_READWRITE && reinterpret_cast<std::uintptr_t>(memoryInfo.BaseAddress) == memory.Block(5) && memoryInfo.RegionSize == BlockBytes, "the second of two armed blocks is not reported alone");
    Require(track.Query(memory.Block(0) + 100, &memoryInfo) && memoryInfo.Protect == PAGE_READWRITE && reinterpret_cast<std::uintptr_t>(memoryInfo.BaseAddress) == memory.Block(0) && memoryInfo.RegionSize == 2 * BlockBytes, "open blocks are not reported as the host reports them");
    Require(track.Protected(memory.Block(4)) && !track.Protected(memory.Block(3)) && !track.Protected(memory.Block(6)), "the armed blocks are not the protected ones");
    Require(VirtualProtect(memory.data + 3 * BlockBytes, BlockBytes, PAGE_READWRITE, &previous) != 0, "cannot reopen a block by hand");
}

void CheckHolds() {
    auto& track = Track::Get();
    Raw memory(4 * BlockBytes);
    Require(track.Armable(memory.Block(0)), "a fresh block cannot be armed");
    Require(track.Arm(memory.Block(0)) && track.Arm(memory.Block(1)), "cannot arm blocks");
    track.Hold(memory.Block(0) + 100, memory.Block(0) + 200);
    Require(!track.Armable(memory.Block(0)) && track.StateOf(memory.Block(0)) == State::Unarmed && memory.Protection(0) == PAGE_READWRITE, "a hold did not reopen its block");
    Require(track.Clean(memory.Block(1)) && track.Armable(memory.Block(1)), "a hold touched a neighbour");
    track.Hold(memory.Block(0), memory.Block(2));
    Require(!track.Armable(memory.Block(1)) && memory.Protection(1) == PAGE_READWRITE, "a hold over two blocks left one armed");
    track.Release(memory.Block(0), memory.Block(2));
    Require(!track.Armable(memory.Block(0)) && track.Armable(memory.Block(1)), "releasing one of two holds released both, or kept the other block held");
    track.Release(memory.Block(0) + 100, memory.Block(0) + 200);
    Require(track.Armable(memory.Block(0)), "a block stays held after its last hold");
    Require(track.Arm(memory.Block(2)), "cannot arm a block");
    track.Import(memory.Block(2), memory.Block(2) + PageBytes);
    Require(!track.Armable(memory.Block(2)) && memory.Protection(2) == PAGE_READWRITE, "an import did not reopen its block for good");
    Require(track.Arm(memory.Block(3)), "cannot arm a block");
    track.Bar(memory.Block(3), memory.Block(3) + PageBytes);
    Require(!track.Armable(memory.Block(3)) && memory.Protection(3) == PAGE_READWRITE, "a barred block stays armed");
    track.Forget(memory.Block(2), memory.Block(3) + BlockBytes);
    Require(track.Armable(memory.Block(2)) && track.Armable(memory.Block(3)), "forgetting a range kept its imports or bars");
}

class Guest {
public:
    Guest(bool resident, std::size_t size) : bytes(size) {
        auto& mappings = Mappings::Get();
        reservation = static_cast<unsigned char*>(mappings.Reserve(nullptr, bytes + RunBytes));
        Require(reservation != nullptr, "cannot reserve guest test memory");
        data = reinterpret_cast<unsigned char*>((reinterpret_cast<std::uintptr_t>(reservation) + RunBytes - 1) & ~(RunBytes - 1));
        if (resident) {
            section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_RESERVE, 0, static_cast<DWORD>(bytes), nullptr);
            Require(section != nullptr, "cannot reserve the backing of resident test memory");
            mappings.ForgetSection(section);
            mappings.Map(data, bytes, section, 0, PAGE_READWRITE, true, true);
        } else {
            mappings.Commit(data, bytes, PAGE_READWRITE, bytes, true);
        }
        for (std::size_t at = 0; at < bytes; at += PageBytes) data[at] = 1;
    }
    ~Guest() {
        if (!released) Mappings::Get().Reset(data, bytes);
        if (section != nullptr) {
            Mappings::Get().ForgetSection(section);
            CloseHandle(section);
        }
    }
    Guest(const Guest&) = delete;
    Guest& operator=(const Guest&) = delete;

    std::uintptr_t Address() const { return reinterpret_cast<std::uintptr_t>(data); }
    std::uintptr_t Block(std::size_t index) const { return Address() + index * BlockBytes; }
    void Write(std::size_t block, std::size_t page, unsigned char value) const { static_cast<volatile unsigned char*>(data)[block * BlockBytes + page * PageBytes + 3] = value; }
    DWORD Protection(std::size_t block, std::size_t page = 0) const {
        MEMORY_BASIC_INFORMATION memory{};
        Require(VirtualQuery(data + block * BlockBytes + page * PageBytes, &memory, sizeof(memory)) == sizeof(memory), "cannot query guest test memory");
        return memory.Protect;
    }
    std::set<std::size_t> Collect(std::size_t firstPage, std::size_t pages, bool clear = true) const {
        std::set<std::size_t> written;
        auto cursor = Address() + firstPage * PageBytes;
        const auto end = cursor + pages * PageBytes;
        while (cursor < end) {
            std::array<void*, 64> found{};
            std::size_t count = found.size();
            Require(Mappings::Get().Collect(cursor, end - cursor, found.data(), &count, clear), "collecting guest test memory failed");
            for (std::size_t index = 0; index < count; ++index) written.insert((reinterpret_cast<std::uintptr_t>(found[index]) - Address()) / PageBytes);
            if (count < found.size()) break;
            cursor = reinterpret_cast<std::uintptr_t>(found[count - 1]) + PageBytes;
        }
        return written;
    }
    void Release() {
        Mappings::Get().Reset(data, bytes);
        released = true;
    }

    unsigned char* data = nullptr;
    std::size_t bytes;

private:
    unsigned char* reservation = nullptr;
    HANDLE section = nullptr;
    bool released = false;
};

struct TemporaryFile {
    std::wstring path;
    HANDLE handle = INVALID_HANDLE_VALUE;

    TemporaryFile() {
        wchar_t directory[MAX_PATH];
        wchar_t name[MAX_PATH];
        Require(GetTempPathW(MAX_PATH, directory) != 0 && GetTempFileNameW(directory, L"aps", 0, name) != 0, "cannot name a temporary file");
        path = name;
        const HANDLE writer = CreateFileW(name, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        Require(writer != INVALID_HANDLE_VALUE, "cannot create a temporary file");
        std::array<unsigned char, PageBytes> content;
        content.fill(0xa5);
        DWORD written = 0;
        Require(WriteFile(writer, content.data(), static_cast<DWORD>(content.size()), &written, nullptr) != 0 && written == content.size(), "cannot fill a temporary file");
        CloseHandle(writer);
        handle = CreateFileW(name, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        Require(handle != INVALID_HANDLE_VALUE, "cannot reopen a temporary file");
    }
    ~TemporaryFile() {
        CloseHandle(handle);
        DeleteFileW(path.c_str());
    }
    bool Read(void* destination, DWORD* error) const {
        SetFilePointer(handle, 0, nullptr, FILE_BEGIN);
        DWORD read = 0;
        SetLastError(0);
        const bool done = ReadFile(handle, destination, PageBytes, &read, nullptr) != 0 && read == PageBytes;
        *error = done ? 0 : GetLastError();
        return done;
    }
};

void CheckGuards(bool resident) {
    auto& track = Track::Get();
    auto& mappings = Mappings::Get();
    const std::string kind = resident ? "resident" : "private";
    Guest memory(resident, 32 * BlockBytes);
    const auto arm = [&](std::size_t block) {
        Require(track.Armable(memory.Block(block)) && track.Arm(memory.Block(block)) && memory.Protection(block) == PAGE_READONLY, kind + ": cannot arm block " + std::to_string(block));
    };
    const auto open = [&](std::size_t block) { return track.StateOf(memory.Block(block)) == State::Unarmed && memory.Protection(block) == PAGE_READWRITE; };

    arm(0);
    mappings.Pin(memory.Block(0) + 0x2000, 0x3000);
    Require(open(0) && !track.Armable(memory.Block(0)), kind + ": pinning a guest stack left its block armed or armable");
    mappings.Unpin(memory.Block(0) + 0x2000, 0x3000);
    Require(track.Armable(memory.Block(0)), kind + ": an unpinned block stays unarmable");

    TemporaryFile file;
    DWORD error = 0;
    arm(1);
    Require(!file.Read(memory.data + BlockBytes, &error) && error == ERROR_NOACCESS && memory.data[BlockBytes + 8] == 0, kind + ": a kernel write into an armed block without a host write bracket did not fail with ERROR_NOACCESS");
    Require(mappings.BeginHostWrite(memory.Block(1), PageBytes), kind + ": a host write on an armed block was refused");
    Require(open(1) && !track.Armable(memory.Block(1)), kind + ": a host write left its block armed or armable");
    Require(file.Read(memory.data + BlockBytes, &error) && memory.data[BlockBytes + 8] == 0xa5, kind + ": a kernel write under a host write bracket failed");
    mappings.EndHostWrite(memory.Block(1), PageBytes);
    Require(track.Armable(memory.Block(1)), kind + ": a block stays unarmable after its host write");

    arm(2);
    Require(mappings.BeginHostWrite(memory.Block(2), BlockBytes, false), kind + ": a host import of an armed block was refused");
    Require(open(2), kind + ": a host import left its block armed");
    mappings.EndHostWrite(memory.Block(2), BlockBytes, false);
    Require(!track.Armable(memory.Block(2)) && track.Armable(memory.Block(3)), kind + ": an imported block can be armed again, or its neighbour cannot");

    arm(4);
    Require(mappings.Protect(memory.Block(4) + PageBytes, PageBytes, PAGE_READONLY, true), kind + ": protecting guest memory failed");
    Require(track.StateOf(memory.Block(4)) == State::Unarmed && memory.Protection(4, 0) == PAGE_READWRITE && memory.Protection(4, 1) == PAGE_READONLY && memory.Protection(4, 2) == PAGE_READWRITE, kind + ": a guest protection change did not reopen the block and protect exactly its range");
    Require(!track.HandleWrite(memory.Block(4) + PageBytes + 5), kind + ": a write to memory the guest made read-only was taken for a tracked write");
    Require(mappings.Protect(memory.Block(4) + PageBytes, PageBytes, PAGE_READWRITE, true) && memory.Protection(4, 1) == PAGE_READWRITE, kind + ": restoring the guest protection failed");
    Require(!track.Armable(memory.Block(4)), kind + ": a block the guest gave another protection can be armed again");
    arm(18);
    Require(mappings.Protect(memory.Block(18), BlockBytes, PAGE_READWRITE, true) && open(18) && track.Armable(memory.Block(18)), kind + ": a guest protection change to read-write left the block armed or barred it");

    if (!resident) {
        arm(5);
        mappings.Commit(memory.data + 5 * BlockBytes, BlockBytes, PAGE_READONLY, BlockBytes, true);
        Require(track.StateOf(memory.Block(5)) == State::Unarmed && memory.Protection(5) == PAGE_READONLY && !track.HandleWrite(memory.Block(5) + 5), kind + ": committing over an armed block with a new protection kept it tracked");
        mappings.Commit(memory.data + 5 * BlockBytes, BlockBytes, PAGE_READWRITE, BlockBytes, true);
        Require(!track.Armable(memory.Block(5)), kind + ": a block committed again with another protection can be armed");
    }

    arm(6);
    mappings.SetProtection(memory.Block(6), PageBytes, PAGE_READWRITE);
    Require(open(6), kind + ": an announced protection change left the block armed");

    arm(8);
    arm(10);
    MEMORY_BASIC_INFORMATION info{};
    Require(mappings.QueryOpened(memory.Block(9), &info) && info.Protect == PAGE_READWRITE && open(8) && open(10), kind + ": the query the guest sees did not reopen the armed blocks of its allocation");
    Require(reinterpret_cast<std::uintptr_t>(info.BaseAddress) + info.RegionSize >= memory.Block(11), kind + ": the query the guest sees stops at a block that was armed");

    arm(12);
    mappings.KeepOpen(memory.Block(12), PageBytes);
    Require(open(12) && !track.Armable(memory.Block(12)), kind + ": a range the kernel layer commits itself can still be armed");

    arm(14);
    std::array<std::uint64_t, 1> words{};
    Require(mappings.ArmOwned(memory.Block(14) + 0x100, 0x200, words.data(), words.size(), Mappings::OwnedPrivate), kind + ": an owned proof was refused because its block is armed");
    Require(mappings.OwnedUnchanged(memory.Block(14) + 0x100, 0x200, words.data(), words.size()), kind + ": an owned proof over an armed clean block does not hold");
    const auto before = handled.load();
    memory.Write(14, 0, 9);
    Require(handled.load() - before == 1 && !mappings.OwnedUnchanged(memory.Block(14) + 0x100, 0x200, words.data(), words.size()), kind + ": an owned proof survived a write to its armed block");

    arm(16);
    arm(17);
    memory.Release();
    Require(track.StateOf(memory.Block(16)) == State::Unarmed && track.StateOf(memory.Block(17)) == State::Unarmed && track.Counted().armed == 0, kind + ": released guest memory kept armed blocks");
}

std::set<std::size_t> Pages(std::size_t block, std::initializer_list<std::size_t> pages) {
    std::set<std::size_t> result;
    for (const auto page : pages) result.insert(block * BlockPages + page);
    return result;
}

void Settle(const Guest& memory, std::size_t blocks) {
    auto& track = Track::Get();
    memory.Collect(0, blocks * BlockPages);
    track.AdvancePeriod();
    memory.Collect(0, blocks * BlockPages);
    track.AdvancePeriod();
}

void CheckCollect(bool resident) {
    constexpr std::size_t Blocks = 32;
    auto& track = Track::Get();
    auto& mappings = Mappings::Get();
    const std::string kind = resident ? "resident" : "private";
    Guest memory(resident, Blocks * BlockBytes);
    const auto all = [&] { return memory.Collect(0, Blocks * BlockPages); };
    const auto calls = [&] { return mappings.CollectWatchCalls(); };
    const std::set<std::size_t> none;
    Require(all().size() == Blocks * BlockPages, kind + ": the first collect did not report every written page");
    Require(all() == none && track.Counted().armed == 0, kind + ": a block was armed in the frame it was written in");
    track.AdvancePeriod();
    Require(all() == none && track.Counted().armed == 0, kind + ": a block was armed before a whole frame found it clean");
    track.AdvancePeriod();
    auto before = calls();
    Require(all() == none && track.Counted().armed == Blocks && calls() - before == Blocks, kind + ": cold blocks were not armed, one reset of the write watch each");
    before = calls();
    const auto quiet = track.Counted().quiet;
    Require(all() == none && calls() == before && track.Counted().quiet > quiet, kind + ": a collect of armed clean blocks made a system call");
    Require(memory.Collect(3 * BlockPages + 2, 5, false) == none && calls() == before, kind + ": a collect without reset of an armed clean block made a system call");

    auto faults = handled.load();
    memory.Write(5, 3, 7);
    memory.Write(5, 11, 8);
    Require(handled.load() - faults == 1, kind + ": two writes to one armed block did not fault once");
    before = calls();
    Require(all() == Pages(5, {3, 11}) && calls() - before == 1, kind + ": the collect after a fault did not report exactly the written pages with one system call");
    Require(track.StateOf(memory.Block(5)) == State::Unarmed && track.Counted().armed == Blocks - 1, kind + ": a block written in this frame was armed again at once");
    Require(all() == none, kind + ": a written block is reported twice");
    track.AdvancePeriod();
    Require(all() == none && track.Counted().armed == Blocks - 1, kind + ": a written block was armed before a later frame found it clean");
    track.AdvancePeriod();
    Require(all() == none && track.Counted().armed == Blocks, kind + ": a block that stayed clean for a frame was not armed again");

    memory.Write(7, 2, 1);
    memory.Write(7, 9, 2);
    Require(memory.Collect(7 * BlockPages, 4) == Pages(7, {2}), kind + ": a collect of part of a written block did not report the page inside it");
    Require(all() == Pages(7, {9}), kind + ": a page outside an earlier partial collect was lost");
    Settle(memory, Blocks);
    Require(all() == none && track.Counted().armed == Blocks, kind + ": the blocks were not armed again");

    Require(mappings.BeginHostWrite(memory.Block(9) + 12 * PageBytes, 16), kind + ": a host write was refused");
    memory.Write(9, 12, 5);
    mappings.EndHostWrite(memory.Block(9) + 12 * PageBytes, 16);
    Require(track.StateOf(memory.Block(9)) == State::Unarmed, kind + ": a host write did not reopen its block");
    before = calls();
    Require(memory.Collect(9 * BlockPages, 4) == none && track.Clean(memory.Block(9)) && calls() - before == 1, kind + ": a partial collect did not arm the cold block it touched");
    before = calls();
    Require(memory.Collect(9 * BlockPages + 8, 8) == Pages(9, {12}) && calls() == before && track.Clean(memory.Block(9)), kind + ": a page the arming found outside the collected range was not reported later without a system call");
    Require(all() == none, kind + ": a page found by an arming was reported twice");

    memory.Write(11, 0, 1);
    for (std::size_t page = 0; page < BlockPages; ++page) memory.Write(12, page, 2);
    std::array<void*, 4> few{};
    std::size_t count = few.size();
    Require(mappings.Collect(memory.Block(11), 2 * BlockBytes, few.data(), &count, true) && count == few.size() && few[0] == memory.data + 11 * BlockBytes, kind + ": a collect into a short buffer did not fill it in address order");
    std::set<std::size_t> rest = memory.Collect(11 * BlockPages + (reinterpret_cast<std::uintptr_t>(few[3]) - memory.Block(11)) / PageBytes + 1, 2 * BlockPages - ((reinterpret_cast<std::uintptr_t>(few[3]) - memory.Block(11)) / PageBytes + 1));
    Require(rest.size() == BlockPages - 3, kind + ": pages past a short buffer were lost or reported twice");

    mappings.Pin(memory.Block(21), BlockBytes);
    Settle(memory, Blocks);
    all();
    Require(track.StateOf(memory.Block(21)) == State::Unarmed && track.Clean(memory.Block(22)), kind + ": a pinned block was armed, or its neighbour was not");
    mappings.Unpin(memory.Block(21), BlockBytes);
    track.AdvancePeriod();
    all();
    Require(track.Clean(memory.Block(21)), kind + ": an unpinned block was not armed");

    Require(mappings.BeginHostWrite(memory.Block(20), BlockBytes, false), kind + ": a host import was refused");
    mappings.EndHostWrite(memory.Block(20), BlockBytes, false);
    Settle(memory, Blocks);
    all();
    Require(track.StateOf(memory.Block(20)) == State::Unarmed, kind + ": an imported block was armed");
    Require(track.Clean(memory.Block(19)), kind + ": the neighbour of an imported block was not armed");
}

void CheckCollectRace(bool resident) {
    constexpr std::size_t Writers = 8;
    constexpr std::size_t Blocks = 32;
    constexpr std::size_t Pages = Blocks * BlockPages;
    constexpr std::size_t Trials = 1800;
    constexpr std::size_t Burst = 6;
    auto& track = Track::Get();
    Guest memory(resident, Blocks * BlockBytes);
    const auto counter = [&](std::size_t page) { return reinterpret_cast<std::uint64_t*>(memory.data + page * PageBytes); };
    for (std::size_t page = 0; page < Pages; ++page) *counter(page) = 0;
    std::vector<std::uint64_t> shadow(Pages, 0);
    std::atomic<std::size_t> trial{0};
    std::atomic<std::size_t> finished{0};
    std::atomic<bool> stop{false};
    std::vector<std::thread> writers;
    for (std::size_t writer = 0; writer < Writers; ++writer) {
        writers.emplace_back([&, writer] {
            std::mt19937 random(static_cast<unsigned>(writer) * 104729u + 11u);
            for (std::size_t current = 1;; ++current) {
                while (trial.load(std::memory_order_acquire) != current) {
                    if (stop.load(std::memory_order_acquire)) return;
                    YieldProcessor();
                }
                for (std::size_t write = 0; write < Burst && current % 6 == 0; ++write) {
                    const auto page = (random() % (Pages / Writers)) * Writers + writer;
                    auto* value = static_cast<volatile std::uint64_t*>(counter(page));
                    *value = *value + 1;
                }
                finished.fetch_add(1, std::memory_order_release);
            }
        });
    }
    const auto collect = [&] {
        for (const auto page : memory.Collect(0, Pages)) shadow[page] = *static_cast<volatile std::uint64_t*>(counter(page));
    };
    std::size_t lost = 0;
    const auto before = track.Counted();
    for (std::size_t current = 1; current <= Trials; ++current) {
        track.AdvancePeriod();
        trial.store(current, std::memory_order_release);
        for (std::size_t pass = 0; pass < 8; ++pass) collect();
        while (finished.load(std::memory_order_acquire) != current * Writers) collect();
        collect();
        for (std::size_t page = 0; page < Pages; ++page) {
            if (shadow[page] != *counter(page)) ++lost;
        }
    }
    stop.store(true, std::memory_order_release);
    for (auto& writer : writers) writer.join();
    const auto after = track.Counted();
    std::printf("%s memory, %zu writers against collects: %zu trials, %llu arms, %llu faults, %llu collects without a system call, %zu last writes lost\n", resident ? "resident" : "private", Writers, Trials, static_cast<unsigned long long>(after.arms - before.arms), static_cast<unsigned long long>(after.faults - before.faults), static_cast<unsigned long long>(after.quiet - before.quiet), lost);
    Require(lost == 0, "a write was lost by a collect over armed blocks");
    Require(after.arms != before.arms && after.faults != before.faults, "the race never armed a block or never faulted");
    Require(refused.load() == 0, "a fault on tracked memory was refused");
}

GuestArena::PageTrackReason ReasonOf(std::uint64_t reason) {
    return static_cast<GuestArena::PageTrackReason>(reason & GuestArena::PageTrackReasonMask);
}

std::uint64_t FramesOf(std::uint64_t reason) {
    return reason >> GuestArena::PageTrackReasonFramesShift;
}

void CheckDemand(bool resident) {
    constexpr std::size_t Blocks = 8;
    using Reason = GuestArena::PageTrackReason;
    auto& track = Track::Get();
    auto& mappings = Mappings::Get();
    const std::string kind = resident ? "resident" : "private";
    Guest memory(resident, Blocks * BlockBytes);
    const auto all = [&] { return memory.Collect(0, Blocks * BlockPages); };
    const std::set<std::size_t> none;
    std::array<std::uint64_t, Blocks> tokens{};
    std::array<std::uint64_t, Blocks> reasons{};
    Require(mappings.ArmBlocks(memory.Address(), Blocks * BlockBytes, tokens.data(), reasons.data(), Blocks), kind + ": a request did not arm every block of an allocation");
    for (std::size_t block = 0; block < Blocks; ++block) {
        Require(tokens[block] != 0 && ReasonOf(reasons[block]) == Reason::Armed && track.Clean(memory.Block(block)) && memory.Protection(block) == PAGE_READONLY, kind + ": a requested block is not armed with a token");
    }
    Require(track.CleanSince(memory.Address(), Blocks * BlockBytes, tokens.data(), Blocks), kind + ": fresh tokens do not hold");
    Require(all().size() == Blocks * BlockPages, kind + ": the pages written before a requested arming were lost for the collect");
    Require(all() == none && track.CleanSince(memory.Address(), Blocks * BlockBytes, tokens.data(), Blocks), kind + ": a collect of blocks armed on request reported pages twice or broke their tokens");

    const auto faults = handled.load();
    memory.Write(2, 5, 9);
    memory.Write(2, 6, 9);
    Require(handled.load() - faults == 1 && track.StateOf(memory.Block(2)) == State::Written, kind + ": two writes to a block armed on request did not fault once");
    Require(!track.CleanSince(memory.Address(), Blocks * BlockBytes, tokens.data(), Blocks) && !track.CleanSince(memory.Block(2), BlockBytes, &tokens[2], 1) && track.CleanSince(memory.Block(3), BlockBytes, &tokens[3], 1), kind + ": a write did not break the token of its block alone");
    std::uint64_t again = 0;
    Require(mappings.ArmBlocks(memory.Block(2), BlockBytes, &again, nullptr, 1) && again != 0 && again != tokens[2] && track.Clean(memory.Block(2)), kind + ": a written block was not armed again at once on request");
    Require(all() == Pages(2, {5, 6}), kind + ": the pages written before a second requested arming were lost for the collect");
    Require(track.CleanSince(memory.Block(2), BlockBytes, &again, 1), kind + ": a collect broke the token of a block armed on request");
    std::uint64_t same = 0;
    Require(mappings.ArmBlocks(memory.Block(2), BlockBytes, &same, nullptr, 1) && same == again, kind + ": a request on an armed block did not return its token");

    std::array<std::uint64_t, 2> pair{};
    Require(!mappings.ArmBlocks(memory.Block(4) + 100, BlockBytes, pair.data(), nullptr, 1) && pair[0] == 0, kind + ": a request with the wrong number of blocks was accepted");
    Require(mappings.ArmBlocks(memory.Block(4) + 100, BlockBytes, pair.data(), nullptr, 2) && pair[0] == tokens[4] && pair[1] == tokens[5], kind + ": a range over two blocks did not get both tokens, the first for the block of its address");

    std::uint64_t hot = 0;
    std::uint64_t reason = 0;
    for (unsigned frame = 0; frame < Track::HotFrames; ++frame) {
        track.AdvancePeriod();
        Require(mappings.ArmBlocks(memory.Block(6), BlockBytes, &hot, &reason, 1) && hot != 0, kind + ": a block that faulted in fewer than three frames was refused");
        memory.Write(6, 1, static_cast<unsigned char>(frame + 1));
    }
    track.AdvancePeriod();
    Require(!mappings.ArmBlocks(memory.Block(6), BlockBytes, &hot, &reason, 1) && hot == 0 && ReasonOf(reason) == Reason::Resting && FramesOf(reason) == Track::RestFrames && memory.Protection(6) == PAGE_READWRITE, kind + ": a block that faulted in three frames was not put to rest for requests");
    std::uint64_t seen = 0;
    Require(track.Reasons(memory.Block(6), BlockBytes, &seen, 1) && ReasonOf(seen) == Reason::Resting && FramesOf(seen) == Track::RestFrames, kind + ": the table does not say that a block rests, or for how long");
    Require(track.Reasons(memory.Block(5), BlockBytes, &seen, 1) && ReasonOf(seen) == Reason::Armed, kind + ": the table does not say that a block is armed");
    for (std::uint64_t frame = 0; frame + 1 < Track::RestFrames; ++frame) track.AdvancePeriod();
    Require(!mappings.ArmBlocks(memory.Block(6), BlockBytes, &hot, &reason, 1) && ReasonOf(reason) == Reason::Resting && FramesOf(reason) == 1, kind + ": a resting block was armed on request before its rest ended");
    track.AdvancePeriod();
    Require(mappings.ArmBlocks(memory.Block(6), BlockBytes, &hot, &reason, 1) && hot != 0 && ReasonOf(reason) == Reason::Armed, kind + ": a block was not armed on request after its rest");
    Require(all() == Pages(6, {1}), kind + ": the writes of a resting block were lost for the collect");

    mappings.Pin(memory.Block(7), BlockBytes);
    Require(!mappings.ArmBlocks(memory.Block(7), BlockBytes, &hot, &reason, 1) && hot == 0 && ReasonOf(reason) == Reason::Held && memory.Protection(7) == PAGE_READWRITE, kind + ": a pinned block was armed on request");
    Require(track.Reasons(memory.Block(7), BlockBytes, &seen, 1) && ReasonOf(seen) == Reason::Held, kind + ": the table does not say that a block is held");
    mappings.Unpin(memory.Block(7), BlockBytes);
    Require(track.Reasons(memory.Block(7), BlockBytes, &seen, 1) && ReasonOf(seen) == Reason::Armable, kind + ": the table still refuses a block after its last hold");
    Require(mappings.ArmBlocks(memory.Block(7), BlockBytes, &hot, &reason, 1) && hot != 0, kind + ": an unpinned block was not armed on request");
    Require(!mappings.ArmBlocks(memory.Block(Blocks), BlockBytes, &hot, &reason, 1) && hot == 0 && ReasonOf(reason) == Reason::Edge, kind + ": a block outside the allocation was armed on request");
    Require(track.Reasons(memory.Block(Blocks), BlockBytes, &seen, 1) && ReasonOf(seen) == Reason::Edge, kind + ": the table forgot that a block is outside a whole allocation");
    std::array<std::uint64_t, Blocks> every{};
    Require(track.Reasons(memory.Address(), Blocks * BlockBytes, every.data(), Blocks) && !track.Reasons(memory.Address(), Blocks * BlockBytes, every.data(), Blocks - 1) && ReasonOf(every[0]) == Reason::Disabled, kind + ": the reasons of a range are not given per block, or a wrong block count was accepted");

    track.AdvancePeriod();
    memory.Write(3, 0, 1);
    Require(mappings.ArmBlocks(memory.Block(3), BlockBytes, &hot, &reason, 1) && hot != 0, kind + ": a block was not armed again on request after its first fault of a frame");
    memory.Write(3, 0, 2);
    Require(!mappings.ArmBlocks(memory.Block(3), BlockBytes, &hot, &reason, 1) && hot == 0 && ReasonOf(reason) == Reason::Faulted && FramesOf(reason) == 1 && memory.Protection(3) == PAGE_READWRITE, kind + ": a block that faulted twice in a frame was armed again on request in that frame");
    Require(track.Reasons(memory.Block(3), BlockBytes, &seen, 1) && ReasonOf(seen) == Reason::Faulted, kind + ": the table does not say that a block faulted twice in this frame");
    track.AdvancePeriod();
    Require(mappings.ArmBlocks(memory.Block(3), BlockBytes, &hot, &reason, 1) && hot != 0, kind + ": a block that faulted twice in a frame was not armed on request in the next frame");
}

void CheckRearm(bool resident) {
    constexpr std::size_t Blocks = 6;
    constexpr std::uint16_t Every = 0xffffu;
    auto& track = Track::Get();
    auto& mappings = Mappings::Get();
    const std::string kind = resident ? "resident" : "private";
    Guest memory(resident, Blocks * BlockBytes);
    const auto all = [&] { return memory.Collect(0, Blocks * BlockPages); };
    const auto bit = [](std::size_t page) { return static_cast<std::uint16_t>(1u << page); };
    const std::set<std::size_t> none;
    all();
    std::array<std::uint64_t, Blocks> tokens{};
    std::array<std::uint16_t, Blocks> written{};
    Require(mappings.RearmBlocks(memory.Address(), Blocks * BlockBytes, tokens.data(), written.data(), nullptr, Blocks), kind + ": a rearm without tokens did not arm every block");
    for (std::size_t block = 0; block < Blocks; ++block) Require(tokens[block] != 0 && written[block] == Every && track.Clean(memory.Block(block)), kind + ": a block armed without a token held did not report every page");
    const auto first = tokens;
    Require(mappings.RearmBlocks(memory.Address(), Blocks * BlockBytes, tokens.data(), written.data(), nullptr, Blocks) && tokens == first, kind + ": a rearm of clean blocks changed their tokens");
    for (std::size_t block = 0; block < Blocks; ++block) Require(written[block] == 0, kind + ": a clean block reported written pages");

    memory.Write(1, 3, 1);
    memory.Write(1, 7, 1);
    memory.Write(2, 15, 1);
    Require(mappings.RearmBlocks(memory.Address(), Blocks * BlockBytes, tokens.data(), written.data(), nullptr, Blocks), kind + ": written blocks were not armed again");
    Require(written[0] == 0 && written[3] == 0 && tokens[0] == first[0] && tokens[3] == first[3], kind + ": the neighbours of written blocks lost their tokens or reported pages");
    Require(written[1] == (bit(3) | bit(7)) && written[2] == bit(15) && tokens[1] != first[1] && tokens[2] != first[2], kind + ": the pages written since the token were not reported at the rearm");
    Require(track.Clean(memory.Block(1)) && track.Clean(memory.Block(2)) && memory.Protection(1) == PAGE_READONLY, kind + ": a block was not armed again by the rearm");
    auto expected = Pages(1, {3, 7});
    expected.insert(2 * BlockPages + 15);
    Require(all() == expected, kind + ": the pages reported at a rearm were taken from the collect");
    Require(all() == none && track.CleanSince(memory.Address(), Blocks * BlockBytes, tokens.data(), Blocks), kind + ": a collect after a rearm reported pages twice or broke the tokens");

    track.AdvancePeriod();
    memory.Write(1, 4, 2);
    Require(mappings.RearmBlocks(memory.Block(1), BlockBytes, &tokens[1], &written[1], nullptr, 1) && written[1] == bit(4), kind + ": a second rearm reported pages of the first");

    memory.Write(0, 1, 3);
    expected = Pages(0, {1});
    expected.insert(BlockPages + 4);
    Require(all() == expected, kind + ": the collect lost a page written to a block armed on request");
    Require(mappings.RearmBlocks(memory.Block(0), BlockBytes, &tokens[0], &written[0], nullptr, 1) && written[0] == Every && tokens[0] != first[0], kind + ": a block whose pages a collect took first was not reported whole");

    std::uint64_t outdated = first[2];
    std::uint16_t mask = 0;
    Require(mappings.RearmBlocks(memory.Block(2), BlockBytes, &outdated, &mask, nullptr, 1) && outdated == tokens[2] && mask == Every, kind + ": an outdated token was answered with a page mask");

    memory.Write(4, 9, 4);
    std::array<std::uint64_t, 1> words{};
    Require(mappings.ArmOwned(memory.Block(4) + 0x100, 0x200, words.data(), words.size(), Mappings::OwnedPrivate), kind + ": an owned proof of a written block was refused");
    Require(mappings.RearmBlocks(memory.Block(4), BlockBytes, &tokens[4], &written[4], nullptr, 1) && written[4] == bit(9), kind + ": the pages an owned proof took from the write watch were missing at the rearm");
    Require(all() == Pages(4, {9}), kind + ": a page an owned proof and a rearm both saw was lost for the collect");

    mappings.Pin(memory.Block(5), BlockBytes);
    std::uint64_t reason = 0;
    Require(!mappings.RearmBlocks(memory.Block(5), BlockBytes, &tokens[5], &written[5], &reason, 1) && tokens[5] == 0 && written[5] == Every && ReasonOf(reason) == GuestArena::PageTrackReason::Held, kind + ": a pinned block was armed again or reported a page mask");
    mappings.Unpin(memory.Block(5), BlockBytes);
}

void CheckTokens(bool resident) {
    constexpr std::size_t Blocks = 10;
    using Reason = GuestArena::PageTrackReason;
    auto& track = Track::Get();
    auto& mappings = Mappings::Get();
    const std::string kind = resident ? "resident" : "private";
    Guest memory(resident, Blocks * BlockBytes);
    memory.Collect(0, Blocks * BlockPages);
    const auto arm = [&](std::size_t block) {
        std::uint64_t token = 0;
        Require(mappings.ArmBlocks(memory.Block(block), BlockBytes, &token, nullptr, 1) && token != 0 && token < Track::TokenLimit, kind + ": a block was not armed with a token below 2^63");
        return token;
    };
    const auto holds = [&](std::size_t block, std::uint64_t token) { return track.CleanSince(memory.Block(block), BlockBytes, &token, 1); };
    const auto denied = [&](std::size_t block) {
        std::uint64_t token = 1;
        std::uint64_t reason = 0;
        Require(!mappings.ArmBlocks(memory.Block(block), BlockBytes, &token, &reason, 1) && token == 0, kind + ": a block that cannot be armed got a token");
        return ReasonOf(reason);
    };

    auto token = arm(0);
    Require(holds(0, token) && !holds(0, 0) && !holds(0, token + 8), kind + ": a fresh token does not hold, or another word does");
    mappings.Pin(memory.Block(0) + 100, 8);
    Require(!holds(0, token) && denied(0) == Reason::Held, kind + ": a pin did not end the token of its block");
    mappings.Unpin(memory.Block(0) + 100, 8);
    const auto after = arm(0);
    Require(after > token && !holds(0, token) && holds(0, after), kind + ": an ended token holds again, or an arming returned it");

    token = arm(1);
    Require(mappings.BeginHostWrite(memory.Block(1) + 12 * PageBytes, 16), kind + ": a host write was refused");
    Require(!holds(1, token) && denied(1) == Reason::Held, kind + ": a host write did not end the token of its block");
    mappings.EndHostWrite(memory.Block(1) + 12 * PageBytes, 16);
    Require(!holds(1, token) && arm(1) > token, kind + ": the end of a host write revived a token");

    token = arm(2);
    Require(mappings.BeginHostWrite(memory.Block(2), BlockBytes, false), kind + ": a host import was refused");
    mappings.EndHostWrite(memory.Block(2), BlockBytes, false);
    Require(!holds(2, token) && denied(2) == Reason::Imported, kind + ": an import did not end the token of its block for good");

    token = arm(3);
    mappings.KeepOpen(memory.Block(3), PageBytes);
    Require(!holds(3, token) && denied(3) == Reason::Barred, kind + ": a bar did not end the token of its block for good");

    token = arm(4);
    Require(mappings.Protect(memory.Block(4), PageBytes, PAGE_READWRITE, true), kind + ": a protection change was refused");
    Require(!holds(4, token) && arm(4) > token, kind + ": a protection change did not end the token of its block");

    auto last = arm(5);
    for (unsigned cycle = 0; cycle < 64; ++cycle) {
        track.Open(memory.Block(5), memory.Block(5) + 1);
        Require(!holds(5, last), kind + ": opening a block did not end its token");
        const auto next = arm(5);
        Require(next > last && !holds(5, last), kind + ": an arming returned an earlier token");
        last = next;
    }

    const auto six = arm(6);
    std::array<std::uint64_t, 2> words{7, 7};
    Require(!track.Tokens(memory.Block(6) + 100, BlockBytes, words.data(), 2) && words[0] == six && words[1] == 0, kind + ": the tokens of a range are not indexed from the block of its address, or an unarmed block got one");
    words = {7, 7};
    Require(!track.Tokens(memory.Block(6) + 100, BlockBytes, words.data(), 1) && words[0] == 0 && words[1] == 7, kind + ": a wrong block count was answered with tokens");
    const auto seven = arm(7);
    Require(track.Tokens(memory.Block(6) + 100, BlockBytes, words.data(), 2) && words[0] == six && words[1] == seven && track.CleanSince(memory.Block(6) + 100, BlockBytes, words.data(), 2), kind + ": the tokens of two armed blocks are not those their armings returned");

    const auto eight = arm(8);
    const auto address = memory.Block(8);
    memory.Release();
    Require(!track.CleanSince(address, BlockBytes, &eight, 1), kind + ": unmapping did not end the token of a block");
    Require(Track::SameArming(track.Word(address), eight), kind + ": unmapping lost the serial of a block, so a later arming could return its token");
    std::uint64_t none = 1;
    std::uint64_t reason = 0;
    Require(!mappings.ArmBlocks(address, BlockBytes, &none, &reason, 1) && none == 0 && ReasonOf(reason) == Reason::Edge, kind + ": a released block was armed");
}

void CheckDemandRace(bool resident) {
    constexpr std::size_t Writers = 8;
    constexpr std::size_t Blocks = 16;
    constexpr std::size_t Pages = Blocks * BlockPages;
    constexpr std::size_t Trials = 600;
    constexpr std::size_t Burst = 6;
    auto& track = Track::Get();
    auto& mappings = Mappings::Get();
    Guest memory(resident, Blocks * BlockBytes);
    const auto counter = [&](std::size_t page) { return reinterpret_cast<std::uint64_t*>(memory.data + page * PageBytes); };
    const auto value = [&](std::size_t page) { return *static_cast<volatile std::uint64_t*>(counter(page)); };
    for (std::size_t page = 0; page < Pages; ++page) *counter(page) = 0;
    std::vector<std::uint64_t> copy(Pages, 0);
    std::vector<std::uint64_t> collected(Pages, 0);
    std::array<std::uint64_t, Blocks> tokens{};
    std::atomic<std::size_t> trial{0};
    std::atomic<std::size_t> finished{0};
    std::atomic<bool> stop{false};
    std::vector<std::thread> writers;
    for (std::size_t writer = 0; writer < Writers; ++writer) {
        writers.emplace_back([&, writer] {
            std::mt19937 random(static_cast<unsigned>(writer) * 104729u + 17u);
            for (std::size_t current = 1;; ++current) {
                while (trial.load(std::memory_order_acquire) != current) {
                    if (stop.load(std::memory_order_acquire)) return;
                    YieldProcessor();
                }
                for (std::size_t write = 0; write < Burst && current % 3 == 0; ++write) {
                    const auto page = (random() % (Pages / Writers)) * Writers + writer;
                    auto* slot = static_cast<volatile std::uint64_t*>(counter(page));
                    *slot = *slot + 1;
                }
                finished.fetch_add(1, std::memory_order_release);
            }
        });
    }
    const auto refresh = [&] {
        for (std::size_t block = 0; block < Blocks; ++block) {
            if (tokens[block] != 0 && track.CleanSince(memory.Block(block), BlockBytes, &tokens[block], 1)) continue;
            std::uint64_t token = 0;
            mappings.ArmBlocks(memory.Block(block), BlockBytes, &token, nullptr, 1);
            for (std::size_t page = block * BlockPages; page < (block + 1) * BlockPages; ++page) copy[page] = value(page);
            tokens[block] = token;
        }
    };
    const auto collect = [&] {
        for (const auto page : memory.Collect(0, Pages)) collected[page] = value(page);
    };
    std::size_t stale = 0;
    std::size_t lost = 0;
    std::size_t trusted = 0;
    const auto before = track.Counted();
    for (std::size_t current = 1; current <= Trials; ++current) {
        track.AdvancePeriod();
        trial.store(current, std::memory_order_release);
        for (std::size_t pass = 0; pass < 8; ++pass) {
            refresh();
            collect();
        }
        while (finished.load(std::memory_order_acquire) != current * Writers) {
            refresh();
            collect();
        }
        collect();
        for (std::size_t block = 0; block < Blocks; ++block) {
            if (tokens[block] == 0 || !track.CleanSince(memory.Block(block), BlockBytes, &tokens[block], 1)) continue;
            ++trusted;
            for (std::size_t page = block * BlockPages; page < (block + 1) * BlockPages; ++page) {
                if (copy[page] != *counter(page)) ++stale;
            }
        }
        for (std::size_t page = 0; page < Pages; ++page) {
            if (collected[page] != *counter(page)) ++lost;
        }
    }
    stop.store(true, std::memory_order_release);
    for (auto& writer : writers) writer.join();
    const auto after = track.Counted();
    std::printf("%s memory, %zu writers against requested armings: %zu trials, %llu arms, %llu faults, %zu blocks trusted by their token, %zu stale copies, %zu writes lost for the collect\n", resident ? "resident" : "private", Writers, Trials, static_cast<unsigned long long>(after.arms - before.arms), static_cast<unsigned long long>(after.faults - before.faults), trusted, stale, lost);
    Require(stale == 0, "a copy taken after a token differs from memory while the token holds");
    Require(lost == 0, "a write was lost for the collect under requested armings");
    Require(after.arms != before.arms && after.faults != before.faults && trusted != 0, "the race never armed a block, never faulted or never trusted a token");
    Require(refused.load() == 0, "a fault on tracked memory was refused");
}

void CheckVerify(bool resident) {
    constexpr std::size_t Blocks = 8;
    auto& track = Track::Get();
    Require(track.Verifying(), "APS5_VERIFY_PAGE_TRACK=1 did not turn verification on");
    const std::string kind = resident ? "resident" : "private";
    Guest memory(resident, Blocks * BlockBytes);
    const auto all = [&] { return memory.Collect(0, Blocks * BlockPages); };
    const auto diverged = track.Counted().divergences;
    Settle(memory, Blocks);
    Require(all().empty() && track.Counted().armed == Blocks, kind + ": the blocks were not armed");
    memory.Write(2, 1, 4);
    Require(all() == Pages(2, {1}) && all().empty() && track.Counted().divergences == diverged, kind + ": verified collects diverged on ordinary writes");
    const auto expectDivergence = [&](const char* what) {
        const auto before = track.Counted().divergences;
        bool thrown = false;
        try {
            all();
        } catch (const std::runtime_error&) {
            thrown = true;
        }
        Require(thrown && track.Counted().divergences == before + 1, kind + ": " + what);
        track.Open(memory.Block(4), memory.Block(4) + 1);
        Settle(memory, Blocks);
        all();
    };
    DWORD previous;
    Require(VirtualProtect(memory.data + 4 * BlockBytes, BlockBytes, PAGE_READWRITE, &previous) != 0, "cannot reopen a block behind the tracker");
    memory.Write(4, 6, 9);
    Require(VirtualProtect(memory.data + 4 * BlockBytes, BlockBytes, PAGE_READONLY, &previous) != 0, "cannot protect a block behind the tracker");
    expectDivergence("a write that reached an armed block without a fault was not reported");
    Require(track.Clean(memory.Block(4)), kind + ": the block was not armed again after the divergence");
    Require(VirtualProtect(memory.data + 4 * BlockBytes, BlockBytes, PAGE_READWRITE, &previous) != 0, "cannot reopen a block behind the tracker");
    memory.Write(4, 6, 10);
    Require(ResetWriteWatch(memory.data + 4 * BlockBytes, BlockBytes) == 0, "cannot reset the write watch behind the tracker");
    Require(VirtualProtect(memory.data + 4 * BlockBytes, BlockBytes, PAGE_READONLY, &previous) != 0, "cannot protect a block behind the tracker");
    expectDivergence("bytes that changed in an armed block without the write watch seeing them were not reported");
}

void CheckHot(bool resident) {
    constexpr std::size_t Blocks = 8;
    auto& track = Track::Get();
    const std::string kind = resident ? "resident" : "private";
    Guest memory(resident, Blocks * BlockBytes);
    const auto all = [&] { return memory.Collect(0, Blocks * BlockPages); };
    Settle(memory, Blocks);
    Require(all().empty() && track.Counted().armed == Blocks, kind + ": the blocks were not armed");
    const auto rests = track.Counted().rests;
    std::size_t writes = 0;
    for (std::size_t frame = 0; frame < 8; ++frame) {
        all();
        if (track.Clean(memory.Block(3))) {
            memory.Write(3, 0, 1);
            ++writes;
        }
        all();
        track.AdvancePeriod();
    }
    Require(writes == Track::HotFrames && track.Faults(memory.Block(3)) == Track::HotFrames && track.Counted().rests == rests + 1, kind + ": a block written as soon as it is armed did not fault in 3 of 8 frames and rest, after waits of 1, 2 and 4 frames");
    for (std::size_t frame = 0; frame < Track::RestFrames - 8; ++frame) {
        all();
        Require(!track.Clean(memory.Block(3)), kind + ": a resting block was armed again");
        track.AdvancePeriod();
    }
    Require(track.Clean(memory.Block(2)) && track.Clean(memory.Block(4)), kind + ": the neighbours of a resting block were not kept armed");
    for (std::size_t frame = 0; frame < 16; ++frame) {
        all();
        track.AdvancePeriod();
    }
    all();
    Require(track.Clean(memory.Block(3)), kind + ": a block was not armed again after its rest");
    memory.Write(3, 0, 2);
    all();
    Require(track.Faults(memory.Block(3)) == Track::HotFrames + 1, kind + ": a block armed after its rest did not fault on a write");

    const auto before = track.Faults(memory.Block(5));
    for (std::size_t frame = 0; frame < 96; ++frame) {
        all();
        if (frame % 3 == 0) memory.Write(5, 1, 2);
        all();
        track.AdvancePeriod();
    }
    Require(track.Faults(memory.Block(5)) - before <= 3, kind + ": a block written every third frame kept being armed");

    const auto slow = track.Faults(memory.Block(6));
    for (std::size_t frame = 0; frame < 160; ++frame) {
        all();
        if (frame % 40 == 39) memory.Write(6, 1, 3);
        all();
        track.AdvancePeriod();
    }
    Require(track.Faults(memory.Block(6)) - slow == 4, kind + ": a block written every fortieth frame was not armed again for each of its writes");
}

void CheckRace() {
    constexpr std::size_t Writers = 8;
    constexpr std::size_t Blocks = 32;
    constexpr std::size_t Pages = Blocks * BlockPages;
    constexpr std::size_t Trials = 1200;
    constexpr std::size_t Burst = 5;
    auto& track = Track::Get();
    Raw memory(Blocks * BlockBytes);
    const auto counter = [&](std::size_t page) { return reinterpret_cast<std::uint64_t*>(memory.data + page * PageBytes); };
    for (std::size_t page = 0; page < Pages; ++page) *counter(page) = 0;
    Require(ResetWriteWatch(memory.data, memory.bytes) == 0, "cannot reset the write watch");
    std::vector<std::uint64_t> shadow(Pages, 0);
    std::atomic<std::size_t> trial{0};
    std::atomic<std::size_t> finished{0};
    std::atomic<bool> stop{false};
    std::vector<std::thread> writers;
    for (std::size_t writer = 0; writer < Writers; ++writer) {
        writers.emplace_back([&, writer] {
            std::mt19937 random(static_cast<unsigned>(writer) * 7919u + 3u);
            for (std::size_t current = 1;; ++current) {
                while (trial.load(std::memory_order_acquire) != current) {
                    if (stop.load(std::memory_order_acquire)) return;
                    YieldProcessor();
                }
                for (std::size_t write = 0; write < Burst; ++write) {
                    const auto page = (random() % (Pages / Writers)) * Writers + writer;
                    auto* value = static_cast<volatile std::uint64_t*>(counter(page));
                    *value = *value + 1;
                }
                finished.fetch_add(1, std::memory_order_release);
            }
        });
    }
    std::uint64_t clean = 0;
    std::uint64_t collected = 0;
    const auto collect = [&] {
        for (std::size_t block = 0; block < Blocks; ++block) {
            if (track.Clean(memory.Block(block))) {
                ++clean;
                continue;
            }
            track.TakeWritten(memory.Block(block));
            track.Arm(memory.Block(block));
            ++collected;
            for (const auto page : memory.Watched(block, true)) shadow[block * BlockPages + page] = *static_cast<volatile std::uint64_t*>(counter(block * BlockPages + page));
        }
    };
    std::size_t lost = 0;
    const auto faultsBefore = track.Counted().faults;
    for (std::size_t current = 1; current <= Trials; ++current) {
        trial.store(current, std::memory_order_release);
        collect();
        while (finished.load(std::memory_order_acquire) != current * Writers) collect();
        collect();
        for (std::size_t page = 0; page < Pages; ++page) {
            if (shadow[page] != *counter(page)) ++lost;
        }
    }
    stop.store(true, std::memory_order_release);
    for (auto& writer : writers) writer.join();
    std::printf("race of %zu writers against a collector that arms, resets the write watch, then reads: %zu trials, %llu faults, %llu collects answered by the armed state, %llu by GetWriteWatch, %zu last writes lost\n", Writers, Trials, static_cast<unsigned long long>(track.Counted().faults - faultsBefore), static_cast<unsigned long long>(clean), static_cast<unsigned long long>(collected), lost);
    Require(lost == 0, "a write to an armed block was lost");
    Require(refused.load() == 0, "a fault on tracked memory was refused");
}

}

int main(int argc, char** argv) {
    const std::string_view mode = argc > 1 ? argv[1] : "";
    _putenv_s(mode.starts_with("verify") ? "APS5_VERIFY_PAGE_TRACK" : "APS5_PAGE_TRACK", "1");
    const auto handler = AddVectoredExceptionHandler(1, HandleFault);
    int result = 0;
    try {
        Require(Track::Get().Enabled(), "APS5_PAGE_TRACK=1 did not enable page tracking");
        if (mode == "table") {
            CheckFaults();
            CheckQuery();
            CheckHolds();
        } else if (mode == "race") {
            CheckRace();
        } else if (mode == "guards") {
            CheckGuards(false);
            CheckGuards(true);
        } else if (mode == "collect") {
            CheckCollect(false);
            CheckCollect(true);
        } else if (mode == "hot") {
            CheckHot(false);
            CheckHot(true);
        } else if (mode == "collect-race") {
            CheckCollectRace(false);
            CheckCollectRace(true);
        } else if (mode == "demand") {
            CheckDemand(false);
            CheckDemand(true);
        } else if (mode == "rearm") {
            CheckRearm(false);
            CheckRearm(true);
        } else if (mode == "tokens") {
            CheckTokens(false);
            CheckTokens(true);
        } else if (mode == "demand-race") {
            CheckDemandRace(false);
            CheckDemandRace(true);
        } else if (mode == "verify-demand") {
            CheckDemand(false);
            CheckDemand(true);
            CheckRearm(false);
            CheckRearm(true);
            CheckTokens(false);
            CheckTokens(true);
            CheckDemandRace(false);
            CheckDemandRace(true);
            Require(Track::Get().Counted().divergences == 0, "verified collects diverged under requested armings");
        } else if (mode == "verify" || mode == "verify-race") {
            if (mode == "verify") {
                CheckVerify(false);
                CheckVerify(true);
            } else {
                CheckCollectRace(false);
                CheckCollectRace(true);
                Require(Track::Get().Counted().divergences == 0, "verified collects diverged under concurrent writers");
            }
        } else {
            std::fprintf(stderr, "usage: %s table|race|guards|collect|hot|collect-race|demand|rearm|tokens|demand-race|verify|verify-race|verify-demand\n", argv[0]);
            result = 2;
        }
        if (result == 0) std::printf("page tracking %.*s tests passed\n", static_cast<int>(mode.size()), mode.data());
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        result = 1;
    }
    RemoveVectoredExceptionHandler(handler);
    return result;
}
