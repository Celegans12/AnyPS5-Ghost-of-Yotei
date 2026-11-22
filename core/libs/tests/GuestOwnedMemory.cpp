#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/WindowsMappings.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <source_location>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static void Require(bool value, std::source_location location = std::source_location::current()) {
    if (!value) throw std::runtime_error("owned guest memory regression at " + std::to_string(location.line()));
}

#ifdef _WIN32
using Mappings = GuestArena::WindowsMappings;
constexpr std::size_t blockBytes = Mappings::ownedBlockBytes;

struct Memory {
    Mappings& mappings = Mappings::Get();
    unsigned char* base;
    std::size_t bytes;
    HANDLE section = nullptr;

    explicit Memory(std::size_t bytes = blockBytes, bool resident = true, bool watched = true) : bytes(bytes) {
        base = static_cast<unsigned char*>(mappings.Reserve(nullptr, bytes));
        Require(base != nullptr);
        if (resident) {
            section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE | SEC_RESERVE, 0, static_cast<DWORD>(bytes), nullptr);
            Require(section != nullptr);
            mappings.Map(base, bytes, section, 0, PAGE_READWRITE, watched, true);
        } else mappings.Commit(base, bytes, PAGE_READWRITE, bytes, watched);
    }

    ~Memory() {
        mappings.Reset(base, bytes);
        VirtualFree(base, 0, MEM_RELEASE);
        if (section != nullptr) {
            mappings.ForgetSection(section);
            CloseHandle(section);
        }
    }

    std::uintptr_t Address() const { return reinterpret_cast<std::uintptr_t>(base); }

    bool Arm(std::uint64_t* words, unsigned flags = 0) { return mappings.ArmOwned(Address(), bytes, words, bytes / blockBytes, flags); }
    bool Same(const std::uint64_t* words) { return mappings.OwnedUnchanged(Address(), bytes, words, bytes / blockBytes); }

    std::vector<void*> Collect(bool clear, Mappings::Collection collection = Mappings::Collection::All, std::size_t capacity = 0) {
        std::vector<void*> pages(capacity == 0 ? bytes / 4096 : capacity);
        auto count = pages.size();
        Require(mappings.Collect(Address(), bytes, pages.data(), &count, clear, collection));
        pages.resize(count);
        return pages;
    }
};

static void CheckWrites(bool resident) {
    Memory memory(blockBytes, resident);
    const unsigned flags = resident ? 0 : Mappings::OwnedPrivate;
    std::uint64_t word = 0;
    MEMORY_BASIC_INFORMATION before{};
    Require(VirtualQuery(memory.base, &before, sizeof(before)) == sizeof(before));
    Require(memory.Arm(&word, flags));
    Require(word != 0 && memory.Same(&word));
    MEMORY_BASIC_INFORMATION after{};
    Require(VirtualQuery(memory.base, &after, sizeof(after)) == sizeof(after));
    Require(before.Protect == after.Protect && after.Protect == PAGE_READWRITE);
    memory.base[4095] = 37;
    Require(!memory.Same(&word));
    const auto old = word;
    Require(memory.Arm(&word, flags));
    Require(word != old && memory.Same(&word) && !memory.Same(&old));
    memory.base[blockBytes - 1] = 81;
    Require(!memory.Same(&word));
    Require(memory.Arm(&word, flags));
    Require(memory.mappings.OwnedUnchanged(memory.Address() + 4095, blockBytes - 8190, &word, 1));
    Require(memory.base[4095] == 37 && memory.base[blockBytes - 1] == 81);
}

static void CheckPending(bool resident) {
    Memory memory(2 * 1024 * 1024, resident);
    memory.Collect(true);
    for (std::size_t at = 0; at < memory.bytes; at += 4096) memory.base[at] = 17;
    std::vector<std::uint64_t> words(memory.bytes / blockBytes);
    Require(memory.Arm(words.data(), resident ? 0 : Mappings::OwnedPrivate));
    Require(memory.Same(words.data()));
    const auto peek = memory.Collect(false, Mappings::Collection::All, 3);
    Require(peek.size() == 3 && memory.Collect(false, Mappings::Collection::All, 3) == peek);
    Require(memory.Same(words.data()));
    std::vector<bool> seen(memory.bytes / 4096);
    std::size_t total = 0;
    for (;;) {
        const auto pages = memory.Collect(true, Mappings::Collection::All, 3);
        if (pages.empty()) break;
        for (auto* page : pages) {
            const auto index = (reinterpret_cast<std::uintptr_t>(page) - memory.Address()) / 4096;
            Require(index < seen.size() && !seen[index]);
            seen[index] = true;
            ++total;
        }
    }
    Require(total == seen.size() && !memory.Same(words.data()));
    Require(memory.Arm(words.data(), resident ? 0 : Mappings::OwnedPrivate));
    memory.base[7 * 4096] = 23;
    Require(!memory.Same(words.data()));
    Require(memory.Collect(true).size() == 1);
}

static void CheckPendingOverlap(bool resident) {
    Memory memory(2 * blockBytes, resident);
    memory.Collect(true);
    const auto pageCount = memory.bytes / 4096;
    for (std::size_t index = 0; index < pageCount; index += 2) memory.base[index * 4096] = 1;
    std::vector<std::uint64_t> words(memory.bytes / blockBytes);
    Require(memory.Arm(words.data(), resident ? 0 : Mappings::OwnedPrivate));
    for (std::size_t index = 0; index < pageCount; index += 3) memory.base[index * 4096] = 2;
    std::vector<void*> expected;
    for (std::size_t index = 0; index < pageCount; ++index) {
        if (index % 2 == 0 || index % 3 == 0) expected.push_back(memory.base + index * 4096);
    }
    Require(memory.Collect(false) == expected);
    Require(memory.Collect(false) == expected);
    std::vector<void*> collected;
    for (;;) {
        const auto pages = memory.Collect(true, Mappings::Collection::All, 3);
        if (pages.empty()) break;
        collected.insert(collected.end(), pages.begin(), pages.end());
    }
    Require(collected == expected);
    Require(memory.Collect(false).empty());
}

static void CheckFresh() {
    Memory memory;
    memory.base[4096] = 9;
    std::uint64_t word = 0;
    Require(memory.Arm(&word));
    Require(memory.Collect(false, Mappings::Collection::Armed).empty());
    Require(memory.Collect(true, Mappings::Collection::Fresh).size() == blockBytes / 4096);
    Require(memory.Collect(true, Mappings::Collection::Armed).empty());
    Require(memory.Arm(&word));
    memory.base[8192] = 11;
    Require(memory.Arm(&word));
    Require(memory.Collect(false, Mappings::Collection::Fresh).empty());
    Require(memory.Collect(true, Mappings::Collection::Armed).size() == 1);
}

static void CheckEviction() {
    Memory memory;
    memory.base[7] = 73;
    std::uint64_t word = 0;
    Require(memory.Arm(&word));
    const auto old = word;
    memory.mappings.Reset(memory.base, memory.bytes);
    Require(!memory.Same(&word));
    memory.mappings.Map(memory.base, memory.bytes, memory.section, 0, PAGE_READWRITE, true, true);
    Require(memory.base[7] == 73 && memory.Arm(&word) && word != old);
    Require(!memory.Same(&old));
    auto* alias = static_cast<unsigned char*>(memory.mappings.MapAlias(memory.Address(), memory.bytes));
    Require(alias != nullptr && !memory.Same(&word) && !memory.Arm(&word));
    alias[19] = 41;
    Require(memory.base[19] == 41);
    memory.mappings.UnmapAlias(alias);
    Require(!memory.Arm(&word));
}

static void CheckAccess(bool resident) {
    Memory memory(blockBytes, resident);
    const auto flags = resident ? 0 : Mappings::OwnedPrivate;
    std::uint64_t word = 0;
    Require(memory.Arm(&word, flags));
    memory.mappings.Pin(memory.Address() + 31, 9);
    memory.mappings.Pin(memory.Address() + 31, 9);
    Require(!memory.Same(&word) && !memory.Arm(&word, flags));
    memory.mappings.Unpin(memory.Address() + 31, 9);
    Require(!memory.Arm(&word, flags));
    memory.mappings.Unpin(memory.Address() + 31, 9);
    Require(memory.Arm(&word, flags));
    for (const bool store : {true, false}) {
        Require(memory.mappings.BeginHostWrite(memory.Address() + 4096, 1, store));
        Require(!memory.Same(&word) && !memory.Arm(&word, flags));
        memory.base[4096] = 59;
        memory.mappings.EndHostWrite(memory.Address() + 4096, 1, store);
        Require(memory.Arm(&word, flags));
    }
    memory.mappings.SetProtection(memory.Address(), memory.bytes, PAGE_READONLY);
    Require(!memory.Same(&word));
    DWORD old = 0;
    Require(VirtualProtect(memory.base, memory.bytes, PAGE_READONLY, &old));
    Require(!memory.Arm(&word, flags));
    memory.mappings.SetProtection(memory.Address(), memory.bytes, PAGE_READWRITE);
    Require(!memory.Arm(&word, flags));
    Require(VirtualProtect(memory.base, memory.bytes, PAGE_READWRITE, &old));
    Require(memory.Arm(&word, flags));
    memory.mappings.Commit(memory.base, memory.bytes, PAGE_READWRITE, memory.bytes, true);
    Require(!memory.Same(&word));
}

static void CheckRefusals() {
    Memory memory(2 * blockBytes, false);
    std::array<std::uint64_t, 2> words{7, 9};
    Require(!memory.Arm(words.data()));
    Require(!memory.mappings.ArmOwned(memory.Address(), 0, words.data(), 0, Mappings::OwnedPrivate));
    Require(!memory.mappings.ArmOwned(memory.Address(), memory.bytes, nullptr, 2, Mappings::OwnedPrivate));
    Require(!memory.mappings.ArmOwned(memory.Address(), memory.bytes, words.data(), 1, Mappings::OwnedPrivate));
    Require(!memory.mappings.ArmOwned(memory.Address(), memory.bytes, words.data(), 2, 2));
    Require(!memory.mappings.ArmOwned(std::numeric_limits<std::uintptr_t>::max() - 7, 16, words.data(), 2));
    Require(words[0] == 7 && words[1] == 9);
    Require(memory.mappings.ArmOwned(memory.Address(), blockBytes, words.data(), 1, Mappings::OwnedPrivate));
    const auto old = words[0];
    DWORD previous;
    Require(VirtualProtect(memory.base + blockBytes, 4096, PAGE_READONLY, &previous));
    Require(!memory.Arm(words.data(), Mappings::OwnedPrivate));
    Require(memory.mappings.OwnedUnchanged(memory.Address(), blockBytes, &old, 1));
    Require(VirtualProtect(memory.base + blockBytes, 4096, PAGE_READWRITE, &previous));
    Require(memory.mappings.ArmOwned(memory.Address() + blockBytes - 1, 2, words.data(), 2, Mappings::OwnedPrivate));
    memory.base[blockBytes] = 5;
    Require(!memory.mappings.OwnedUnchanged(memory.Address() + blockBytes - 1, 2, words.data(), 2));
    Memory untracked(blockBytes, false, false);
    Require(!untracked.Arm(words.data(), Mappings::OwnedPrivate));
    Memory unwatched(blockBytes, true, false);
    Require(!unwatched.Arm(words.data()));
    memory.mappings.Reset(memory.base, memory.bytes);
    Require(!memory.Arm(words.data(), Mappings::OwnedPrivate));
}

static void CheckWriterAndCollector() {
    Memory memory;
    memory.Collect(true);
    std::atomic<unsigned> request{0};
    std::atomic<unsigned> done{0};
    const auto wait = [](const std::atomic<unsigned>& value, unsigned expected) {
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (value.load(std::memory_order_acquire) != expected) {
            if (std::chrono::steady_clock::now() >= until) return false;
            std::this_thread::yield();
        }
        return true;
    };
    std::thread writer([&] {
        for (unsigned step = 1; step <= 64; ++step) {
            if (!wait(request, step)) return;
            memory.base[4096] = static_cast<unsigned char>(step);
            if ((step & 1) != 0) memory.Collect(true);
            done.store(step, std::memory_order_release);
        }
    });
    bool exact = true;
    for (unsigned step = 1; step <= 64; ++step) {
        std::uint64_t word = 0;
        exact &= memory.Arm(&word) && memory.Same(&word);
        request.store(step, std::memory_order_release);
        if (!wait(done, step)) {
            exact = false;
            break;
        }
        exact &= !memory.Same(&word);
    }
    writer.join();
    Require(exact);
}

static void CheckUnlockedReads() {
    const bool unlocked = std::getenv("APS5_NO_UNLOCKED_MAPPING_READS") == nullptr;
    const auto other = static_cast<std::size_t>(Mappings::LockSite::Other);
    Memory memory(2 * blockBytes);
    auto& mappings = memory.mappings;
    const auto taken = [&] { return mappings.LockCounted().sites[other].acquisitions; };
    const auto answered = [&] { return mappings.LockCounted().unlocked; };
    std::array<std::uint64_t, 2> words{};
    Require(memory.Arm(words.data()));
    auto locks = taken();
    auto answers = answered();
    Require(memory.Same(words.data()));
    Require(taken() == locks + (unlocked ? 0u : 1u) && answered() == answers + (unlocked ? 1u : 0u));
    const std::array<std::uint64_t, 2> stale{words[0], words[1] + 1};
    const std::array<std::uint64_t, 2> blank{};
    locks = taken();
    Require(!memory.Same(stale.data()) && !memory.Same(blank.data()));
    Require(taken() == locks + (unlocked ? 0u : 2u));
    Require(memory.Same(words.data()));
    memory.base[blockBytes + 8192] = 3;
    locks = taken();
    answers = answered();
    Require(!memory.Same(words.data()));
    Require(taken() == locks + 1 && answered() == answers);
    Require(mappings.OwnedUnchanged(memory.Address(), blockBytes, words.data(), 1));
    locks = taken();
    Require(!mappings.OwnedUnchanged(memory.Address() + blockBytes, blockBytes, &words[1], 1));
    Require(taken() == locks + (unlocked ? 0u : 1u));
    const auto old = words[0];
    Require(memory.Arm(words.data()) && words[0] != old);
    Require(!mappings.OwnedUnchanged(memory.Address(), blockBytes, &old, 1) && memory.Same(words.data()));
    memory.Collect(true);
    Require(!memory.Same(words.data()));
    Require(memory.Arm(words.data()) && memory.Same(words.data()));
    mappings.Pin(memory.Address(), blockBytes);
    Require(!memory.Same(words.data()) && mappings.OwnedUnchanged(memory.Address() + blockBytes, blockBytes, &words[1], 1));
    mappings.Unpin(memory.Address(), blockBytes);
    const auto address = memory.Address();
    mappings.Reset(memory.base, memory.bytes);
    Require(!mappings.OwnedUnchanged(address + blockBytes, blockBytes, &words[1], 1));
}

static void CheckUnlockedProtection() {
    const bool unlocked = std::getenv("APS5_NO_UNLOCKED_MAPPING_READS") == nullptr;
    const auto other = static_cast<std::size_t>(Mappings::LockSite::Other);
    auto& mappings = Mappings::Get();
    const auto taken = [&] { return mappings.LockCounted().sites[other].acquisitions; };
    const auto answered = [&] { return mappings.LockCounted().unlocked; };
    const bool counted = std::getenv("APS5_VERIFY_MAPPING_READS") == nullptr;
    std::uint32_t protection = 0;
    std::uintptr_t released = 0;
    {
        Memory memory(2 * blockBytes);
        released = memory.Address();
        auto locks = taken();
        auto answers = answered();
        Require(mappings.Protection(memory.Address() + blockBytes + 4097, &protection) && protection == PAGE_READWRITE);
        Require(!counted || (taken() == locks + (unlocked ? 0u : 1u) && answered() == answers + (unlocked ? 1u : 0u)));
        mappings.SetProtection(memory.Address(), blockBytes, PAGE_READONLY);
        Require(mappings.Protection(memory.Address() + blockBytes - 1, &protection) && protection == PAGE_READONLY);
        Require(mappings.Protection(memory.Address() + blockBytes, &protection) && protection == PAGE_READWRITE);
        mappings.SetProtection(memory.Address(), blockBytes, PAGE_READWRITE);
        Require(mappings.Protection(memory.Address(), &protection) && protection == PAGE_READWRITE);
    }
    protection = 7;
    auto locks = taken();
    auto answers = answered();
    Require(!mappings.Protection(released, &protection) && protection == 7);
    Require(!counted || (taken() == locks + (unlocked ? 0u : 1u) && answered() == answers + (unlocked ? 1u : 0u)));
    Memory plain(blockBytes, false);
    Require(!mappings.Protection(plain.Address(), &protection) && protection == 7);
    Require(!mappings.Protection(std::uintptr_t{1} << 47, &protection) && protection == 7);
    std::array<void*, 4> compared{};
    std::size_t count = compared.size();
    locks = taken();
    mappings.ComparedPages(plain.Address(), blockBytes, compared.data(), &count, false);
    Require(count == 0 && (!counted || taken() == locks + (unlocked ? 0u : 1u)));
    count = compared.size();
    mappings.ComparedPages(plain.Address(), blockBytes, compared.data(), &count, true);
    Require(count == 0);
    Memory again(blockBytes);
    Require(mappings.Protection(again.Address() + blockBytes - 1, &protection) && protection == PAGE_READWRITE);
}

struct Alias {
    Mappings& mappings = Mappings::Get();
    unsigned char* base;
    std::size_t bytes;

    explicit Alias(const Memory& memory) : bytes(memory.bytes) {
        base = static_cast<unsigned char*>(mappings.Reserve(nullptr, bytes));
        Require(base != nullptr);
        mappings.Map(base, bytes, memory.section, 0, PAGE_READWRITE);
    }

    ~Alias() {
        mappings.Reset(base, bytes);
        VirtualFree(base, 0, MEM_RELEASE);
    }

    std::uintptr_t Address() const { return reinterpret_cast<std::uintptr_t>(base); }
};

static std::vector<void*> ComparedPages(std::uintptr_t address, std::size_t bytes, bool unreported, std::size_t capacity = 64) {
    std::vector<void*> pages(capacity);
    auto count = pages.size();
    Mappings::Get().ComparedPages(address, bytes, pages.data(), &count, unreported);
    Require(count <= capacity);
    pages.resize(count);
    return pages;
}

static void CheckUnlockedComparedPages(bool counted) {
    const bool unlocked = std::getenv("APS5_NO_UNLOCKED_MAPPING_READS") == nullptr;
    const auto other = static_cast<std::size_t>(Mappings::LockSite::Other);
    auto& mappings = Mappings::Get();
    const auto taken = [&] { return mappings.LockCounted().sites[other].acquisitions; };
    const auto answered = [&] { return mappings.LockCounted().unlocked; };
    const auto listedUnlocked = [&](std::uint64_t locks, std::uint64_t answers) { return !counted || (taken() == locks + (unlocked ? 0u : 1u) && answered() == answers + (unlocked ? 1u : 0u)); };
    constexpr std::size_t page = 0x4000;
    std::uintptr_t released = 0;
    std::uintptr_t unmapped = 0;
    {
        Memory memory;
        Memory plain(blockBytes, false);
        released = memory.Address();
        const auto at = [&](std::size_t offset) { return reinterpret_cast<void*>(memory.Address() + offset); };
        {
            Alias alias(memory);
            unmapped = alias.Address();
            Require(ComparedPages(memory.Address(), blockBytes, false).empty() && ComparedPages(alias.Address(), blockBytes, false).empty());
            mappings.CompareWhenShared(memory.Address() + page, 2 * page);
            auto locks = taken();
            auto answers = answered();
            const auto compared = ComparedPages(memory.Address(), blockBytes, false);
            Require(listedUnlocked(locks, answers));
            Require(compared.size() == 8 && compared.front() == at(page) && compared.back() == at(3 * page - 4096));
            Require(ComparedPages(alias.Address(), blockBytes, false).size() == 8);
            Require(ComparedPages(memory.Address() + page + 4097, 8191, false) == std::vector<void*>({at(page + 4096), at(page + 8192)}));
            Require(ComparedPages(memory.Address(), blockBytes, false, 3) == std::vector<void*>({at(page), at(page + 4096), at(page + 8192)}));
            Require(ComparedPages(memory.Address(), blockBytes, false, 0).empty());
            locks = taken();
            answers = answered();
            Require(ComparedPages(plain.Address(), blockBytes, false).empty() && ComparedPages(memory.Address(), page, true).empty());
            Require(!counted || (taken() == locks + (unlocked ? 0u : 2u) && answered() == answers + (unlocked ? 2u : 0u)));
            locks = taken();
            Require(ComparedPages(memory.Address(), blockBytes, true, 3) == std::vector<void*>({at(page), at(page + 4096), at(page + 8192)}));
            Require(ComparedPages(memory.Address(), blockBytes, true).size() == 5);
            Require(!counted || taken() == locks + 2);
            locks = taken();
            answers = answered();
            Require(ComparedPages(memory.Address(), blockBytes, true).empty());
            Require(listedUnlocked(locks, answers));
            Require(ComparedPages(alias.Address(), blockBytes, true).size() == 8 && ComparedPages(alias.Address(), blockBytes, true).empty());
            Require(ComparedPages(memory.Address(), blockBytes, false) == compared);
            mappings.SetProtection(memory.Address() + page, page, PAGE_READONLY);
            std::uint32_t protection = 0;
            Require(mappings.Protection(memory.Address() + page, &protection) && protection == PAGE_READONLY);
            Require(ComparedPages(memory.Address(), blockBytes, false) == compared);
            mappings.SetProtection(memory.Address() + page, page, PAGE_READWRITE);
            Require(mappings.Protection(memory.Address() + page, &protection) && protection == PAGE_READWRITE);
        }
        Require(ComparedPages(unmapped, blockBytes, false).empty());
        Require(ComparedPages(memory.Address(), blockBytes, false).size() == 8);
    }
    const auto locks = taken();
    const auto answers = answered();
    Require(ComparedPages(released, blockBytes, false).empty());
    Require(listedUnlocked(locks, answers));
}

static void CheckComparedPageRaces() {
    constexpr std::size_t page = 0x4000;
    Memory memory;
    Memory steady;
    Alias kept(steady);
    auto& mappings = memory.mappings;
    mappings.CompareWhenShared(steady.Address(), 2 * page);
    std::atomic<bool> stop{false};
    std::thread mapper([&] {
        for (unsigned round = 0; !stop.load(std::memory_order_acquire); ++round) {
            {
                Alias alias(memory);
                mappings.CompareWhenShared(memory.Address() + (round % 4) * page, page);
                mappings.SetProtection(alias.Address(), page, round % 2 == 0 ? PAGE_READONLY : PAGE_READWRITE);
                ComparedPages(alias.Address(), blockBytes, true, 5);
            }
            mappings.Reset(memory.base, memory.bytes);
            mappings.Map(memory.base, memory.bytes, memory.section, 0, PAGE_READWRITE, true, true);
        }
    });
    bool exact = true;
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(1000);
    for (unsigned round = 0; round % 64 != 0 || std::chrono::steady_clock::now() < until; ++round) {
        ComparedPages(memory.Address(), blockBytes, round % 3 == 0);
        std::uint32_t protection = 0;
        mappings.Protection(memory.Address() + (round % 4) * page, &protection);
        exact &= ComparedPages(steady.Address(), blockBytes, false).size() == 8;
        exact &= ComparedPages(steady.Address() + page, blockBytes, false, 2).size() == 2;
    }
    stop.store(true, std::memory_order_release);
    mapper.join();
    Require(exact);
}

static void CheckVerifiedMappingReads() {
    auto& mappings = Mappings::Get();
    std::uint64_t matched = 0;
    std::uint64_t differing = 0;
    std::uint64_t skipped = 0;
    mappings.MappingReadChecks(&matched, &differing, &skipped);
    const auto before = matched;
    Mappings::VerifyMappingReads(true);
    CheckUnlockedComparedPages(false);
    CheckComparedPageRaces();
    Mappings::VerifyMappingReads(std::getenv("APS5_VERIFY_MAPPING_READS") != nullptr);
    mappings.MappingReadChecks(&matched, &differing, &skipped);
    Require(differing == 0);
    Require((matched != before) == (std::getenv("APS5_NO_UNLOCKED_MAPPING_READS") == nullptr));
}

static void CheckReservedHoles() {
    const bool partial = std::getenv("APS5_NO_PARTIAL_COLLECT") == nullptr;
    constexpr std::size_t page = 0x4000;
    auto& mappings = Mappings::Get();
    const auto bytes = 2 * blockBytes;
    auto* base = static_cast<unsigned char*>(mappings.Reserve(nullptr, bytes));
    Require(base != nullptr);
    const auto address = reinterpret_cast<std::uintptr_t>(base);
    mappings.Commit(base + page, 2 * page, PAGE_READWRITE, 2 * page, true);
    const auto collect = [&](std::uintptr_t from, std::size_t size, bool clear, std::vector<void*>* written) {
        written->assign(bytes / 4096, nullptr);
        auto count = written->size();
        const bool collected = mappings.Collect(from, size, written->data(), &count, clear);
        written->resize(collected ? count : 0);
        return collected;
    };
    std::vector<void*> written;
    Require(collect(address + page, 2 * page, true, &written));
    base[page + 4096] = 5;
    base[3 * page - 1] = 6;
    for (const bool clear : {false, true}) {
        Require(collect(address, bytes, clear, &written) == partial);
        Require(!partial || written == std::vector<void*>({base + page + 4096, base + 3 * page - 4096}));
    }
    Require(collect(address + page, 2 * page, true, &written) && written.size() == (partial ? 0u : 2u));
    Require(collect(address, page, true, &written) == partial && written.empty());
    Require(collect(address + 3 * page + 1, blockBytes, false, &written) == partial && written.empty());
    mappings.Commit(base + blockBytes, page, PAGE_READWRITE, page, true);
    Require(collect(address, bytes, true, &written) == partial);
    base[blockBytes + 4096] = 7;
    Require(collect(address, bytes, true, &written) == partial);
    Require(!partial || written == std::vector<void*>({base + blockBytes + 4096}));
    Memory resident;
    Require(collect(resident.Address(), resident.bytes, true, &written));
    mappings.Reset(base, bytes);
    Require(collect(address, bytes, true, &written) == partial && written.empty());
    VirtualFree(base, 0, MEM_RELEASE);
}

static void CheckRangedHostWrites() {
    const auto other = static_cast<std::size_t>(Mappings::LockSite::Other);
    Memory memory(blockBytes, false);
    Memory collected(4 * blockBytes, false);
    auto& mappings = memory.mappings;
    memory.Collect(true);
    collected.Collect(true);
    std::atomic<bool> stop{false};
    std::atomic<bool> failed{false};
    std::thread collector([&] {
        std::array<void*, 64> pages{};
        while (!stop.load(std::memory_order_acquire)) {
            collected.base[blockBytes] = 1;
            std::size_t count = pages.size();
            if (!mappings.Collect(collected.Address(), collected.bytes, pages.data(), &count, true) || count != 1) failed.store(true, std::memory_order_release);
        }
    });
    const auto before = mappings.LockCounted().sites[other].acquisitions;
    constexpr unsigned rounds = 4000;
    bool exact = true;
    for (unsigned round = 0; round < rounds; ++round) {
        const auto offset = (round % 15) * 4096 + 7;
        std::uint64_t word = 0;
        exact &= mappings.BeginHostWrite(memory.Address() + offset, 4096);
        exact &= !memory.Arm(&word, Mappings::OwnedPrivate);
        memory.base[offset] = static_cast<unsigned char>(round);
        mappings.EndHostWrite(memory.Address() + offset, 4096);
        exact &= memory.Arm(&word, Mappings::OwnedPrivate);
    }
    const auto after = mappings.LockCounted().sites[other].acquisitions;
    stop.store(true, std::memory_order_release);
    collector.join();
    Require(exact);
    Require(!failed.load(std::memory_order_acquire));
    Require(after - before == 4u * rounds);
    const auto pages = memory.Collect(true);
    Require(!pages.empty());
}

static void CheckUnlockedProofRaces() {
    Memory memory;
    memory.Collect(true);
    std::atomic<std::uint64_t> armed{0};
    std::atomic<bool> written{false};
    std::atomic<bool> stop{false};
    std::atomic<bool> missed{false};
    std::thread reader([&] {
        while (!stop.load(std::memory_order_acquire)) {
            const auto word = armed.load(std::memory_order_acquire);
            const bool after = written.load(std::memory_order_acquire);
            if (word != 0 && memory.Same(&word) && after) missed.store(true, std::memory_order_release);
        }
    });
    bool exact = true;
    for (unsigned round = 1; round <= 2000 && exact; ++round) {
        written.store(false, std::memory_order_release);
        std::uint64_t word = 0;
        exact &= memory.Arm(&word);
        armed.store(word, std::memory_order_release);
        memory.base[(round % 16) * 4096] = static_cast<unsigned char>(round);
        written.store(true, std::memory_order_release);
        std::array<void*, 16> pages{};
        std::size_t count = pages.size();
        exact &= memory.mappings.Collect(memory.Address(), memory.bytes, pages.data(), &count, true) && count == 1;
        exact &= !memory.Same(&word);
    }
    stop.store(true, std::memory_order_release);
    reader.join();
    Require(exact);
    Require(!missed.load(std::memory_order_acquire));
}

static void CheckUnlockedHostWrites(bool resident) {
    const bool unlocked = std::getenv("APS5_NO_UNLOCKED_HOST_WRITES") == nullptr;
    const auto other = static_cast<std::size_t>(Mappings::LockSite::Other);
    const unsigned flags = resident ? 0 : Mappings::OwnedPrivate;
    Memory memory(2 * blockBytes, resident);
    auto& mappings = memory.mappings;
    const auto taken = [&] { return mappings.LockCounted().sites[other].acquisitions; };
    const auto opened = [&] { return mappings.LockCounted().unlockedHostWrites; };
    const auto target = memory.Address() + 4096 + 7;
    auto token = mappings.OpenHostWrite(target, 4);
    Require(token > 1);
    mappings.CloseHostWrite(target, 4, token);
    memory.Collect(true);
    auto locks = taken();
    const auto writes = opened();
    token = mappings.OpenHostWrite(target, 4);
    Require(token != 0 && (token == 1) == unlocked);
    Require(taken() == locks + (unlocked ? 0u : 1u) && opened() == writes + (unlocked ? 1u : 0u));
    std::array<std::uint64_t, 2> words{};
    Require(!memory.Arm(words.data(), flags));
    memory.base[4096 + 7] = 29;
    locks = taken();
    mappings.CloseHostWrite(target, 4, token);
    Require(taken() == locks + (unlocked ? 0u : 1u));
    const auto written = memory.Collect(true);
    Require(written.size() == 1 && written[0] == memory.base + 4096);
    Require(memory.Arm(words.data(), flags));
    token = mappings.OpenHostWrite(target, 4);
    Require(token > 1 && !memory.Same(words.data()));
    mappings.CloseHostWrite(target, 4, token);
    token = mappings.OpenHostWrite(target + 8192, 4);
    Require(token > 1);
    mappings.CloseHostWrite(target + 8192, 4, token);
    token = mappings.OpenHostWrite(memory.Address() + 4096, blockBytes + 1);
    Require(token > 1);
    mappings.CloseHostWrite(memory.Address() + 4096, blockBytes + 1, token);
    token = mappings.OpenHostWrite(target, 0);
    Require(token > 1);
    mappings.CloseHostWrite(target, 0, token);
    token = mappings.OpenHostWrite(target, 4);
    Require(token != 0 && (token == 1) == unlocked);
    mappings.CloseHostWrite(target, 4, token);
    Require(memory.Arm(words.data(), flags));
    if (!resident) return;
    mappings.SetProtection(memory.Address(), memory.bytes, PAGE_READONLY);
    Require(mappings.OpenHostWrite(target, 4) == 0);
    mappings.SetProtection(memory.Address(), memory.bytes, PAGE_READWRITE);
    token = mappings.OpenHostWrite(target, 4);
    Require(token != 0 && (token == 1) == unlocked);
    Require(mappings.MapAlias(memory.Address(), memory.bytes) == nullptr);
    std::uint32_t protection = 0;
    Require(mappings.Protection(target, &protection) && protection == PAGE_READWRITE);
    mappings.CloseHostWrite(target, 4, token);
    auto* alias = static_cast<unsigned char*>(mappings.MapAlias(memory.Address(), memory.bytes));
    Require(alias != nullptr);
    token = mappings.OpenHostWrite(target, 4);
    Require(token > 1);
    memory.base[4096 + 7] = 31;
    mappings.CloseHostWrite(target, 4, token);
    Require(alias[4096 + 7] == 31);
    mappings.UnmapAlias(alias);
}

static void CheckUnlockedHostWriteCollected(bool resident) {
    const bool unlocked = std::getenv("APS5_NO_UNLOCKED_HOST_WRITES") == nullptr;
    Memory memory(4 * blockBytes, resident);
    auto& mappings = memory.mappings;
    auto* const block = memory.base + blockBytes;
    auto* const page = block + 4096;
    const auto target = reinterpret_cast<std::uintptr_t>(page) + 7;
    auto token = mappings.OpenHostWrite(target, 4);
    Require(token > 1);
    mappings.CloseHostWrite(target, 4, token);
    memory.Collect(true);
    Require(memory.Collect(true).empty());
    token = mappings.OpenHostWrite(target, 4);
    Require(token != 0 && (token == 1) == unlocked);
    const auto open = memory.Collect(true);
    bool found = false;
    for (std::size_t index = 0; index < open.size(); ++index) {
        auto* const reported = static_cast<unsigned char*>(open[index]);
        Require(reported >= block && reported < block + blockBytes);
        Require(index == 0 || open[index - 1] < open[index]);
        found = found || reported == page;
    }
    Require(found);
    page[7] = 41;
    mappings.CloseHostWrite(target, 4, token);
    const auto written = memory.Collect(true);
    Require(written.size() == 1 && written[0] == page);
    Require(memory.Collect(true).empty());
}

static void CheckHostWriteTokens() {
    Memory memory;
    auto& mappings = memory.mappings;
    const auto target = memory.Address() + 8192;
    const auto token = mappings.OpenHostWrite(target, 8);
    Require(token > 1);
    mappings.Reset(memory.base, memory.bytes);
    mappings.Map(memory.base, memory.bytes, memory.section, 0, PAGE_READWRITE, true, true);
    mappings.CloseHostWrite(target, 8, token);
    auto* alias = mappings.MapAlias(memory.Address(), memory.bytes);
    Require(alias != nullptr);
    mappings.UnmapAlias(alias);
}

static void CheckUnlockedHostWriteRaces() {
    const bool unlocked = std::getenv("APS5_NO_UNLOCKED_HOST_WRITES") == nullptr;
    constexpr std::size_t pages = 4 * blockBytes / 4096;
    Memory memory(4 * blockBytes, false);
    Memory churn;
    auto& mappings = memory.mappings;
    for (std::size_t page = 0; page < pages; ++page) {
        const auto token = mappings.OpenHostWrite(memory.Address() + page * 4096, 4096);
        Require(token != 0);
        mappings.CloseHostWrite(memory.Address() + page * 4096, 4096, token);
    }
    memory.Collect(true);
    const auto before = mappings.LockCounted().unlockedHostWrites;
    std::array<std::atomic<bool>, pages> owed{};
    std::atomic<bool> stop{false};
    std::atomic<bool> failed{false};
    std::atomic<unsigned> writing{2};
    std::thread collector([&] {
        std::array<void*, pages> found{};
        while (!stop.load(std::memory_order_acquire)) {
            std::size_t count = found.size();
            if (!mappings.Collect(memory.Address(), memory.bytes, found.data(), &count, true)) failed.store(true, std::memory_order_release);
            for (std::size_t index = 0; index < count; ++index) owed[(reinterpret_cast<std::uintptr_t>(found[index]) - memory.Address()) / 4096].store(false);
        }
    });
    std::thread mapper([&] {
        while (!stop.load(std::memory_order_acquire)) {
            mappings.Reset(churn.base, churn.bytes);
            mappings.Map(churn.base, churn.bytes, churn.section, 0, PAGE_READWRITE, true, true);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(1500);
    const auto write = [&](unsigned id) {
        for (unsigned round = 0; !failed.load(std::memory_order_acquire); ++round) {
            if (round % 64 == 0 && std::chrono::steady_clock::now() >= until) break;
            const auto page = (round * 7 + id * 13) % pages;
            const auto offset = page * 4096 + 64 + id;
            const auto token = mappings.OpenHostWrite(memory.Address() + offset, 1);
            if (token == 0) {
                failed.store(true, std::memory_order_release);
                break;
            }
            owed[page].store(true);
            memory.base[offset] = static_cast<unsigned char>(round);
            mappings.CloseHostWrite(memory.Address() + offset, 1, token);
            if (round % 16 != 0) continue;
            const auto spare = churn.Address() + (round / 16 % 16) * 4096;
            const auto held = mappings.OpenHostWrite(spare, 8);
            mappings.CloseHostWrite(spare, 8, held);
        }
        writing.fetch_sub(1);
    };
    std::thread first(write, 0u);
    std::thread second(write, 1u);
    std::array<std::uint64_t, 4> words{};
    while (writing.load() != 0) {
        memory.Arm(words.data(), Mappings::OwnedPrivate);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    first.join();
    second.join();
    stop.store(true, std::memory_order_release);
    collector.join();
    mapper.join();
    Require(!failed.load(std::memory_order_acquire));
    std::vector<bool> reported(pages);
    for (auto* page : memory.Collect(true)) reported[(reinterpret_cast<std::uintptr_t>(page) - memory.Address()) / 4096] = true;
    for (std::size_t page = 0; page < pages; ++page) Require(!owed[page].load() || reported[page]);
    Require(memory.Arm(words.data(), Mappings::OwnedPrivate));
    const auto opened = mappings.LockCounted().unlockedHostWrites - before;
    Require(unlocked ? opened != 0 : opened == 0);
    auto* alias = mappings.MapAlias(churn.Address(), churn.bytes);
    Require(alias != nullptr);
    mappings.UnmapAlias(alias);
}
#endif

static void CheckExports() {
    std::uint64_t word = 0;
    Require(!GuestArena::GuestArenaArmOwned_nid_no_patch(0, 0, &word, 1, 0));
    Require(!GuestArena::GuestArenaOwnedUnchanged_nid_no_patch(0, 0, &word, 1));
#ifdef _WIN32
    auto* pointer = GuestArena::GuestArenaAllocate_nid_postfix(blockBytes, blockBytes);
    Require(pointer != nullptr);
    GuestArena::GuestArenaCommit_nid_postfix(pointer, blockBytes, PAGE_READWRITE, blockBytes);
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    Require(GuestArena::GuestArenaArmOwned_nid_no_patch(address, blockBytes, &word, 1, Mappings::OwnedPrivate));
    Require(GuestArena::GuestArenaOwnedUnchanged_nid_no_patch(address, blockBytes, &word, 1));
    static_cast<unsigned char*>(pointer)[11] = 77;
    Require(!GuestArena::GuestArenaOwnedUnchanged_nid_no_patch(address, blockBytes, &word, 1));
    GuestArena::GuestArenaReset_nid_postfix(pointer, blockBytes);
    GuestArena::GuestArenaRelease_nid_postfix(pointer, blockBytes);
#else
    Require(!GuestArena::GuestArenaArmOwned_nid_no_patch(0x100000, 65536, &word, 1, 1));
    Require(!GuestArena::GuestArenaOwnedUnchanged_nid_no_patch(0x100000, 65536, &word, 1));
#endif
}

int main() {
    try {
#ifdef _WIN32
        for (const bool resident : {true, false}) {
            CheckWrites(resident);
            CheckPending(resident);
            CheckPendingOverlap(resident);
            CheckAccess(resident);
        }
        CheckFresh();
        CheckEviction();
        CheckRefusals();
        CheckWriterAndCollector();
        CheckUnlockedReads();
        CheckUnlockedProtection();
        CheckUnlockedComparedPages(std::getenv("APS5_VERIFY_MAPPING_READS") == nullptr);
        CheckComparedPageRaces();
        CheckVerifiedMappingReads();
        CheckReservedHoles();
        CheckUnlockedProofRaces();
        CheckRangedHostWrites();
        CheckUnlockedHostWrites(true);
        CheckUnlockedHostWrites(false);
        CheckUnlockedHostWriteCollected(true);
        CheckUnlockedHostWriteCollected(false);
        CheckHostWriteTokens();
        CheckUnlockedHostWriteRaces();
#endif
        CheckExports();
        std::puts("owned guest memory checks passed");
    } catch (const std::exception& failure) {
        std::fprintf(stderr, "%s\n", failure.what());
        return 1;
    }
    return 0;
}
