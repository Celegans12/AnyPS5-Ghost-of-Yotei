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
    arm(4);
    Require(mappings.Protect(memory.Block(4), BlockBytes, PAGE_READWRITE, true) && open(4), kind + ": a guest protection change to read-write left the block armed");

    if (!resident) {
        arm(5);
        mappings.Commit(memory.data + 5 * BlockBytes, BlockBytes, PAGE_READONLY, BlockBytes, true);
        Require(track.StateOf(memory.Block(5)) == State::Unarmed && memory.Protection(5) == PAGE_READONLY && !track.HandleWrite(memory.Block(5) + 5), kind + ": committing over an armed block with a new protection kept it tracked");
        mappings.Commit(memory.data + 5 * BlockBytes, BlockBytes, PAGE_READWRITE, BlockBytes, true);
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
    constexpr std::size_t Trials = 600;
    constexpr std::size_t Burst = 3;
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
                for (std::size_t write = 0; write < Burst; ++write) {
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
        collect();
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
    const auto frame = [&](bool write) {
        if (write) memory.Write(3, 0, 1);
        all();
        track.AdvancePeriod();
    };
    for (std::size_t round = 0; round < 3; ++round) {
        all();
        Require(track.Clean(memory.Block(3)), kind + ": the block was not armed before round " + std::to_string(round));
        frame(true);
        frame(false);
    }
    Require(track.Faults(memory.Block(3)) == Track::HotFrames && track.Counted().rests == rests + 1, kind + ": a block that faulted in 3 of 8 frames was not put to rest");
    for (std::size_t later = 0; later < 20; ++later) frame(later % 2 == 0);
    Require(track.Faults(memory.Block(3)) == Track::HotFrames && !track.Clean(memory.Block(3)) && track.Clean(memory.Block(2)) && track.Clean(memory.Block(4)), kind + ": a resting block was armed again, or its neighbours were not kept armed");
    for (std::size_t later = 0; later < Track::RestFrames; ++later) frame(false);
    all();
    Require(track.Clean(memory.Block(3)), kind + ": a block was not armed again after its rest");
    frame(true);
    Require(track.Faults(memory.Block(3)) == Track::HotFrames + 1, kind + ": a block armed after its rest did not fault on a write");

    const auto before = track.Faults(memory.Block(5));
    for (std::size_t round = 0; round < 12; ++round) {
        frame(false);
        frame(false);
        frame(false);
        frame(false);
        memory.Write(5, 1, 2);
    }
    all();
    Require(track.Faults(memory.Block(5)) - before >= 10 && track.Counted().rests == rests + 1, kind + ": a block written once in four frames was put to rest or stopped being armed");
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
    _putenv_s(mode == "verify" || mode == "verify-race" ? "APS5_VERIFY_PAGE_TRACK" : "APS5_PAGE_TRACK", "1");
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
            std::fprintf(stderr, "usage: %s table|race|guards|collect|hot|collect-race|verify|verify-race\n", argv[0]);
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
