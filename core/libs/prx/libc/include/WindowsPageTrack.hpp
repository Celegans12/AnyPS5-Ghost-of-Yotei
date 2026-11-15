#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_WINDOWSPAGETRACK_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_WINDOWSPAGETRACK_HPP

#ifdef _WIN32
#include "prx/libc/include/GuestArena.hpp"
#include <windows.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace GuestArena {

class WindowsPageTrack {
public:
    static constexpr std::size_t BlockBytes = 0x10000;
    static constexpr unsigned HistoryFrames = 8;
    static constexpr unsigned HotFrames = 3;
    static constexpr std::uint64_t RestFrames = 64;
    static constexpr unsigned LongestWait = 64;
    static constexpr std::uint32_t WorthAnswers = 16;
    static constexpr std::uint64_t TokenLimit = std::uint64_t{1} << 63;
    static constexpr std::uint32_t RequestFaults = 2;
    enum class State : std::uint64_t { Unarmed = 0, Arming = 1, Armed = 2, Opening = 3, Written = 4 };
    struct Block {
        std::atomic<std::uint64_t> word{0};
        std::atomic<std::uint32_t> faults{0};
        std::atomic<std::uint16_t> holds{0};
        std::atomic<std::uint8_t> flags{0};
        std::uint8_t history = 0;
        std::uint32_t cleanPeriod = 0;
        std::uint32_t writtenPeriod = 0;
        std::atomic<std::uint32_t> faultPeriod{0};
        std::atomic<std::uint32_t> restUntil{0};
        std::uint32_t answers = 0;
        std::uint32_t wait = 0;
        std::atomic<std::uint32_t> frameFaults{0};
        std::atomic<std::uint8_t> edge{0};
    };
    struct Counts {
        std::uint64_t armed = 0;
        std::uint64_t arms = 0;
        std::uint64_t faults = 0;
        std::uint64_t quiet = 0;
        std::uint64_t rests = 0;
        std::uint64_t divergences = 0;
        std::array<std::uint64_t, 4> left{};
        std::uint64_t importArms = 0;
        std::uint64_t gpuMarks = 0;
    };
    enum class Left : std::size_t { Edge, Held, Imported, Waiting };

    static WindowsPageTrack& Get() {
        static auto* track = new WindowsPageTrack;
        return *track;
    }

    bool Enabled() const {
        return enabled;
    }

    bool Verifying() const {
        return verifying;
    }

    bool ArmsImports() const {
        return armImports;
    }

    bool ReadImported(std::uintptr_t block) const {
        const auto* entry = find(block);
        return entry != nullptr && (entry->flags & ReadImport) != 0;
    }

    bool MarkedGpuWrites(std::uintptr_t first, std::uintptr_t end) const {
        if (!armImports || end <= first) return true;
        for (auto block = first & ~(BlockBytes - 1); block < end; block += BlockBytes) {
            const auto* entry = find(block);
            if (entry == nullptr || (entry->flags & GpuWrites) == 0) return false;
        }
        return true;
    }

    void GpuWrite(std::uintptr_t first, std::uintptr_t end) {
        if (!armImports || end <= first) return;
        forEach(first, end, true, [&](std::uintptr_t block, Block& entry) {
            if ((entry.flags & GpuWrites) == 0) gpuMarks.fetch_add(1, std::memory_order_relaxed);
            entry.flags |= GpuWrites;
            open(block, entry);
        });
    }

    bool HandleWrite(std::uintptr_t address) {
        if (!enabled) return false;
        auto* block = find(address);
        if (block == nullptr) return false;
        const auto base = address & ~(BlockBytes - 1);
        for (unsigned spins = 0;; ++spins) {
            auto word = block->word.load(std::memory_order_acquire);
            const auto state = stateOf(word);
            if (state == State::Armed) {
                if (!block->word.compare_exchange_weak(word, with(word, State::Opening), std::memory_order_acq_rel, std::memory_order_acquire)) continue;
                transitions.fetch_add(1, std::memory_order_acq_rel);
                DWORD previous;
                const bool opened = VirtualProtect(reinterpret_cast<void*>(base), BlockBytes, PAGE_READWRITE, &previous) != 0;
                block->faults.fetch_add(1, std::memory_order_relaxed);
                faults.fetch_add(1, std::memory_order_relaxed);
                armed.fetch_sub(1, std::memory_order_relaxed);
                block->word.store(with(word, State::Written), std::memory_order_release);
                transitions.fetch_add(1, std::memory_order_acq_rel);
                return opened;
            }
            if (state == State::Arming || state == State::Opening) {
                if (spins % 64 == 63) SwitchToThread();
                else YieldProcessor();
                continue;
            }
            MEMORY_BASIC_INFORMATION memory{};
            if (VirtualQuery(reinterpret_cast<const void*>(address), &memory, sizeof(memory)) == sizeof(memory) && memory.State == MEM_COMMIT && writable(memory.Protect)) return true;
            if (block->word.load(std::memory_order_acquire) != word) continue;
            return false;
        }
    }

    bool Protected(std::uintptr_t address) const {
        if (!enabled) return false;
        const auto* block = find(address);
        return block != nullptr && guarded(stateOf(block->word.load(std::memory_order_acquire)));
    }

    bool Query(std::uintptr_t address, MEMORY_BASIC_INFORMATION* memory) const {
        for (;;) {
            const auto before = transitions.load(std::memory_order_acquire);
            if (VirtualQuery(reinterpret_cast<const void*>(address), memory, sizeof(*memory)) != sizeof(*memory)) return false;
            if (!enabled || memory->State != MEM_COMMIT || memory->Protect != PAGE_READONLY) return true;
            if (armed.load(std::memory_order_acquire) == 0) {
                if (transitions.load(std::memory_order_acquire) != before) continue;
                return true;
            }
            const auto regionBase = reinterpret_cast<std::uintptr_t>(memory->BaseAddress);
            const auto regionEnd = regionBase + memory->RegionSize;
            const auto block = address & ~(BlockBytes - 1);
            auto base = regionBase;
            auto end = regionEnd;
            const bool inside = Protected(address);
            if (inside) {
                base = std::max(regionBase, block);
                end = std::min(regionEnd, block + BlockBytes);
            } else {
                for (auto next = block + BlockBytes; next < regionEnd; next += BlockBytes) {
                    if (!Protected(next)) continue;
                    end = next;
                    break;
                }
                for (auto previous = block; previous > regionBase;) {
                    previous -= BlockBytes;
                    if (!Protected(previous)) continue;
                    base = previous + BlockBytes;
                    break;
                }
            }
            if (transitions.load(std::memory_order_acquire) != before) continue;
            memory->BaseAddress = reinterpret_cast<void*>(base);
            memory->RegionSize = end - base;
            if (inside) memory->Protect = PAGE_READWRITE;
            return true;
        }
    }

    Block* Peek(std::uintptr_t block) const {
        return find(block);
    }

    Block& Touch(std::uintptr_t block) {
        return at(block);
    }

    static State StateOfWord(std::uint64_t word) {
        return stateOf(word);
    }

    static State StateOf(const Block& entry) {
        return stateOf(entry.word.load(std::memory_order_acquire));
    }

    State StateOf(std::uintptr_t block) const {
        const auto* entry = find(block);
        return entry == nullptr ? State::Unarmed : StateOf(*entry);
    }

    bool Clean(std::uintptr_t block) const {
        return StateOf(block) == State::Armed;
    }

    bool Armable(std::uintptr_t block) const {
        if (!enabled) return false;
        const auto* entry = find(block);
        return entry == nullptr || (entry->holds == 0 && (entry->flags & Blocking) == 0);
    }

    bool Arm(std::uintptr_t block) {
        auto& entry = at(block);
        const auto word = entry.word.load(std::memory_order_acquire);
        const auto state = stateOf(word);
        if (state != State::Unarmed && state != State::Written) return false;
        if (word >= TokenLimit - SerialStep) return false;
        const auto next = word + SerialStep;
        entry.word.store(with(next, State::Arming), std::memory_order_release);
        transitions.fetch_add(1, std::memory_order_acq_rel);
        armed.fetch_add(1, std::memory_order_acq_rel);
        DWORD previous;
        if (!VirtualProtect(reinterpret_cast<void*>(block), BlockBytes, PAGE_READONLY, &previous)) {
            armed.fetch_sub(1, std::memory_order_acq_rel);
            entry.word.store(with(next, state), std::memory_order_release);
            transitions.fetch_add(1, std::memory_order_acq_rel);
            return false;
        }
        arms.fetch_add(1, std::memory_order_relaxed);
        if ((entry.flags & ReadImport) != 0) importArms.fetch_add(1, std::memory_order_relaxed);
        entry.word.store(with(next, State::Armed), std::memory_order_release);
        return true;
    }

    static bool TakeWritten(Block& entry) {
        const auto word = entry.word.load(std::memory_order_acquire);
        if (stateOf(word) != State::Written) return false;
        entry.word.store(with(word, State::Unarmed), std::memory_order_release);
        return true;
    }

    bool TakeWritten(std::uintptr_t block) {
        auto* entry = find(block);
        return entry != nullptr && TakeWritten(*entry);
    }

    std::uint64_t Word(std::uintptr_t block) const {
        const auto* entry = find(block);
        return entry == nullptr ? 0 : entry->word.load(std::memory_order_acquire);
    }

    std::uint64_t ArmedWord(std::uintptr_t block) const {
        return with(Word(block), State::Armed);
    }

    static bool SameArming(std::uint64_t word, std::uint64_t token) {
        return (word & ~StateMask) == (token & ~StateMask);
    }

    static std::uint64_t Reason(PageTrackReason reason, std::uint64_t frames = 0) {
        return static_cast<std::uint64_t>(reason) | (frames << PageTrackReasonFramesShift);
    }

    static std::uint64_t Refusal(const Block& entry, std::uint32_t now) {
        if (entry.holds != 0) return Reason(PageTrackReason::Held);
        const std::uint8_t flags = entry.flags;
        if ((flags & GpuWrites) != 0) return Reason(PageTrackReason::GpuWritten);
        if ((flags & Imported) != 0) return Reason(PageTrackReason::Imported);
        if ((flags & Barred) != 0) return Reason(PageTrackReason::Barred);
        const std::uint32_t rest = entry.restUntil;
        if (now < rest) return Reason(PageTrackReason::Resting, rest - now);
        const std::uint32_t last = entry.faultPeriod;
        if (last == now && entry.frameFaults.load(std::memory_order_relaxed) >= RequestFaults) return Reason(PageTrackReason::Faulted, 1);
        return Reason(PageTrackReason::Armable);
    }

    void NoteEdge(std::uintptr_t block, bool edge) {
        at(block).edge.store(edge ? 1 : 0, std::memory_order_relaxed);
    }

    bool Reasons(std::uintptr_t address, std::size_t bytes, std::uint64_t* reasons, std::size_t count) const {
        if (reasons == nullptr) return false;
        std::fill_n(reasons, count, Reason(PageTrackReason::Disabled));
        if (!enabled || !Spans(address, bytes, count)) return false;
        const auto now = Period();
        auto block = address & ~(BlockBytes - 1);
        for (std::size_t index = 0; index < count; ++index, block += BlockBytes) {
            const auto* entry = find(block);
            if (entry == nullptr) {
                reasons[index] = Reason(PageTrackReason::Armable);
                continue;
            }
            const auto state = StateOf(*entry);
            if (state == State::Armed) reasons[index] = Reason(PageTrackReason::Armed);
            else if (state == State::Arming || state == State::Opening) reasons[index] = Reason(PageTrackReason::Busy);
            else if (const auto refusal = Refusal(*entry, now); refusal != Reason(PageTrackReason::Armable)) reasons[index] = refusal;
            else reasons[index] = Reason(entry->edge.load(std::memory_order_relaxed) != 0 ? PageTrackReason::Edge : PageTrackReason::Armable);
        }
        return true;
    }

    static bool Spans(std::uintptr_t address, std::size_t bytes, std::size_t count) {
        if (bytes == 0 || address >= (std::uintptr_t{1} << 47) || bytes > (std::uintptr_t{1} << 47) - address) return false;
        return count == (address + bytes - 1) / BlockBytes - address / BlockBytes + 1;
    }

    bool Tokens(std::uintptr_t address, std::size_t bytes, std::uint64_t* tokens, std::size_t count) const {
        if (tokens == nullptr) return false;
        std::fill_n(tokens, count, std::uint64_t{0});
        if (!enabled || !Spans(address, bytes, count)) return false;
        bool armed = true;
        auto block = address & ~(BlockBytes - 1);
        for (std::size_t index = 0; index < count; ++index, block += BlockBytes) {
            const auto word = Word(block);
            if (stateOf(word) == State::Armed) tokens[index] = word;
            else armed = false;
        }
        return armed;
    }

    bool CleanSince(std::uintptr_t address, std::size_t bytes, const std::uint64_t* tokens, std::size_t count) const {
        if (!enabled || tokens == nullptr || !Spans(address, bytes, count)) return false;
        auto block = address & ~(BlockBytes - 1);
        for (std::size_t index = 0; index < count; ++index, block += BlockBytes) {
            if (tokens[index] == 0 || Word(block) != tokens[index]) return false;
        }
        return true;
    }

    std::uint32_t Period() const {
        return period.load(std::memory_order_relaxed);
    }

    void AdvancePeriod() {
        if (!enabled) return;
        auto current = period.load(std::memory_order_relaxed);
        while (current < UINT32_MAX - RestFrames && !period.compare_exchange_weak(current, current + 1, std::memory_order_relaxed)) {}
    }

    static bool Free(const Block& entry) {
        return entry.holds == 0 && (entry.flags & Blocking) == 0;
    }

    static void NoteClean(Block& entry, std::uint32_t now) {
        if (entry.cleanPeriod != now) entry.cleanPeriod = now;
    }

    static void NoteWritten(Block& entry, std::uint32_t now) {
        if (entry.writtenPeriod != now) entry.writtenPeriod = now;
    }

    void NoteClean(std::uintptr_t block) {
        NoteClean(at(block), Period());
    }

    void NoteWritten(std::uintptr_t block) {
        NoteWritten(at(block), Period());
    }

    static std::uint32_t Wait(const Block& entry) {
        return std::max<std::uint32_t>(1, entry.wait);
    }

    static void NoteArmed(Block& entry) {
        entry.answers = 0;
    }

    static void NoteAnswer(Block& entry) {
        if (entry.answers < WorthAnswers) ++entry.answers;
    }

    void NoteFault(Block& entry, std::uint32_t now) {
        entry.writtenPeriod = now;
        const auto wait = Wait(entry);
        entry.wait = entry.answers >= WorthAnswers ? std::max<std::uint32_t>(1, wait / 2) : std::min<std::uint32_t>(LongestWait, wait * 2);
        const std::uint32_t last = entry.faultPeriod;
        if (last == now) {
            entry.frameFaults.store(entry.frameFaults.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
            return;
        }
        entry.frameFaults.store(1, std::memory_order_relaxed);
        const auto gap = now - last;
        entry.history = static_cast<std::uint8_t>((gap >= HistoryFrames ? 0u : static_cast<unsigned>(entry.history) << gap) | 1u);
        entry.faultPeriod = now;
        if (static_cast<unsigned>(__builtin_popcount(entry.history)) < HotFrames) return;
        entry.history = 0;
        entry.restUntil = now + static_cast<std::uint32_t>(RestFrames);
        rests.fetch_add(1, std::memory_order_relaxed);
    }

    void NoteFault(std::uintptr_t block) {
        NoteFault(at(block), Period());
    }

    static bool Cold(const Block& entry, std::uint32_t now) {
        return Free(entry) && entry.cleanPeriod > entry.writtenPeriod && now > entry.cleanPeriod && now >= entry.restUntil && now - entry.writtenPeriod >= Wait(entry);
    }

    bool Cold(std::uintptr_t block) const {
        const auto* entry = find(block);
        return entry != nullptr && Cold(*entry, Period());
    }

    void SetDigest(std::uintptr_t block, std::uint64_t digest) {
        digests[block] = digest;
    }

    std::uint64_t Digest(std::uintptr_t block) const {
        const auto found = digests.find(block);
        return found == digests.end() ? 0 : found->second;
    }

    static_assert(sizeof(Block) == 48, "a block of the page tracking table stays within a cache line");

    void NoteDivergence() {
        divergences.fetch_add(1, std::memory_order_relaxed);
    }

    std::uint32_t Faults(std::uintptr_t block) const {
        const auto* entry = find(block);
        return entry == nullptr ? 0 : entry->faults.load(std::memory_order_relaxed);
    }

    void Open(std::uintptr_t first, std::uintptr_t end) {
        if (!enabled || end <= first) return;
        forEach(first, end, false, [&](std::uintptr_t block, Block& entry) { open(block, entry); });
    }

    void Hold(std::uintptr_t first, std::uintptr_t end) {
        if (!enabled || end <= first) return;
        forEach(first, end, true, [&](std::uintptr_t block, Block& entry) {
            if (entry.holds == UINT16_MAX) throw std::runtime_error("page tracking: too many holds on one block of guest memory");
            ++entry.holds;
            open(block, entry);
        });
    }

    void Release(std::uintptr_t first, std::uintptr_t end) {
        if (!enabled || end <= first) return;
        forEach(first, end, false, [](std::uintptr_t, Block& entry) {
            if (entry.holds != 0) --entry.holds;
        });
    }

    void Import(std::uintptr_t first, std::uintptr_t end) {
        mark(first, end, armImports ? ReadImport : Imported);
    }

    void Bar(std::uintptr_t first, std::uintptr_t end) {
        mark(first, end, Barred);
    }

    void Forget(std::uintptr_t first, std::uintptr_t end) {
        if (!enabled || end <= first) return;
        forEach(first, end, false, [&](std::uintptr_t block, Block& entry) {
            open(block, entry);
            entry.word.store(with(entry.word.load(std::memory_order_acquire), State::Unarmed), std::memory_order_release);
            entry.flags = 0;
            entry.edge = 0;
            entry.frameFaults = 0;
            entry.cleanPeriod = 0;
            entry.writtenPeriod = 0;
            entry.faultPeriod = 0;
            entry.restUntil = 0;
            entry.history = 0;
            entry.answers = 0;
            entry.wait = 0;
        });
    }

    void NoteQuiet(std::uint64_t answers) {
        quiet.fetch_add(answers, std::memory_order_relaxed);
    }

    void NoteLeft(const std::array<std::uint64_t, 4>& blocks) {
        for (std::size_t reason = 0; reason < left.size(); ++reason) {
            if (blocks[reason] != 0) left[reason].store(left[reason].load(std::memory_order_relaxed) + blocks[reason], std::memory_order_relaxed);
        }
    }

    static Left LeftBecause(const Block& entry) {
        return entry.holds != 0 ? Left::Held : (entry.flags & Blocking) != 0 ? Left::Imported : Left::Waiting;
    }

    Counts Counted() const {
        Counts counted{armed.load(std::memory_order_relaxed), arms.load(std::memory_order_relaxed), faults.load(std::memory_order_relaxed), quiet.load(std::memory_order_relaxed), rests.load(std::memory_order_relaxed), divergences.load(std::memory_order_relaxed), {}};
        for (std::size_t reason = 0; reason < left.size(); ++reason) counted.left[reason] = left[reason].load(std::memory_order_relaxed);
        counted.importArms = importArms.load(std::memory_order_relaxed);
        counted.gpuMarks = gpuMarks.load(std::memory_order_relaxed);
        return counted;
    }

private:
    static constexpr std::uint64_t StateMask = 7;
    static constexpr std::uint64_t SerialStep = 8;
    static constexpr std::uint8_t Imported = 1;
    static constexpr std::uint8_t Barred = 2;
    static constexpr std::uint8_t GpuWrites = 4;
    static constexpr std::uint8_t ReadImport = 8;
    static constexpr std::uint8_t Blocking = Imported | Barred | GpuWrites;
    static constexpr unsigned ChunkShift = 30;
    static constexpr std::size_t ChunkBlocks = (std::size_t{1} << ChunkShift) / BlockBytes;
    static constexpr std::size_t Chunks = std::size_t{1} << (47 - ChunkShift);

    WindowsPageTrack() {
        verifying = set("APS5_VERIFY_PAGE_TRACK", std::getenv("APS5_VERIFY_PAGE_TRACK"));
        enabled = set("APS5_PAGE_TRACK", std::getenv("APS5_PAGE_TRACK")) || verifying;
        armImports = set("APS5_ARM_READ_IMPORTS", std::getenv("APS5_ARM_READ_IMPORTS")) && enabled;
        if (enabled) chunks = std::make_unique<std::atomic<Block*>[]>(Chunks);
    }

    static bool set(const char* name, const char* value) {
        if (value == nullptr || value[0] == '\0' || (value[0] == '0' && value[1] == '\0')) return false;
        if (value[0] == '1' && value[1] == '\0') return true;
        char text[96];
        std::snprintf(text, sizeof(text), "%s must be unset, 0 or 1", name);
        throw std::runtime_error(text);
    }

    static State stateOf(std::uint64_t word) {
        return static_cast<State>(word & StateMask);
    }

    static std::uint64_t with(std::uint64_t word, State state) {
        return (word & ~StateMask) | static_cast<std::uint64_t>(state);
    }

    static bool guarded(State state) {
        return state == State::Arming || state == State::Armed || state == State::Opening;
    }

    static bool writable(DWORD protection) {
        protection &= 0xffu;
        return protection == PAGE_READWRITE || protection == PAGE_EXECUTE_READWRITE || protection == PAGE_WRITECOPY || protection == PAGE_EXECUTE_WRITECOPY;
    }

    Block* find(std::uintptr_t address) const {
        const auto index = address >> ChunkShift;
        if (!enabled || index >= Chunks) return nullptr;
        auto* chunk = chunks[index].load(std::memory_order_acquire);
        return chunk == nullptr ? nullptr : chunk + ((address >> 16) & (ChunkBlocks - 1));
    }

    Block& at(std::uintptr_t address) {
        const auto index = address >> ChunkShift;
        if (!enabled) throw std::runtime_error("page tracking is not enabled");
        if (index >= Chunks) throw std::runtime_error("page tracking: guest memory above the user address space");
        auto* chunk = chunks[index].load(std::memory_order_acquire);
        if (chunk == nullptr) {
            chunk = new Block[ChunkBlocks]();
            chunks[index].store(chunk, std::memory_order_release);
        }
        return chunk[(address >> 16) & (ChunkBlocks - 1)];
    }

    template <class TVisit>
    void forEach(std::uintptr_t first, std::uintptr_t end, bool create, TVisit&& visit) {
        const auto last = (end - 1) & ~(BlockBytes - 1);
        for (auto block = first & ~(BlockBytes - 1);; block += BlockBytes) {
            if (!create && (block >> ChunkShift) < Chunks && chunks[block >> ChunkShift].load(std::memory_order_acquire) == nullptr) {
                const auto next = ((block >> ChunkShift) + 1) << ChunkShift;
                if (next == 0 || next > last) return;
                block = next - BlockBytes;
                continue;
            }
            if ((block >> ChunkShift) >= Chunks) {
                if (create) throw std::runtime_error("page tracking: guest memory above the user address space");
                return;
            }
            visit(block, at(block));
            if (block == last) return;
        }
    }

    void open(std::uintptr_t block, Block& entry) {
        for (unsigned spins = 0;; ++spins) {
            auto word = entry.word.load(std::memory_order_acquire);
            const auto state = stateOf(word);
            if (state == State::Armed) {
                if (!entry.word.compare_exchange_weak(word, with(word, State::Opening), std::memory_order_acq_rel, std::memory_order_acquire)) continue;
                transitions.fetch_add(1, std::memory_order_acq_rel);
                DWORD previous;
                const bool opened = VirtualProtect(reinterpret_cast<void*>(block), BlockBytes, PAGE_READWRITE, &previous) != 0;
                const auto error = GetLastError();
                armed.fetch_sub(1, std::memory_order_relaxed);
                entry.word.store(with(word, State::Unarmed), std::memory_order_release);
                transitions.fetch_add(1, std::memory_order_acq_rel);
                if (!opened) {
                    char text[128];
                    std::snprintf(text, sizeof(text), "page tracking: cannot reopen the armed block 0x%llx (error %lu)", static_cast<unsigned long long>(block), error);
                    throw std::runtime_error(text);
                }
                return;
            }
            if (state != State::Opening) return;
            if (spins % 64 == 63) SwitchToThread();
            else YieldProcessor();
        }
    }

    void mark(std::uintptr_t first, std::uintptr_t end, std::uint8_t flag) {
        if (!enabled || end <= first) return;
        forEach(first, end, true, [&](std::uintptr_t block, Block& entry) {
            entry.flags |= flag;
            open(block, entry);
        });
    }

    bool enabled = false;
    bool verifying = false;
    bool armImports = false;
    std::unique_ptr<std::atomic<Block*>[]> chunks;
    mutable std::atomic<std::uint64_t> transitions{0};
    std::atomic<std::uint64_t> armed{0};
    std::atomic<std::uint64_t> arms{0};
    std::atomic<std::uint64_t> faults{0};
    std::atomic<std::uint64_t> quiet{0};
    std::atomic<std::uint64_t> rests{0};
    std::atomic<std::uint64_t> divergences{0};
    std::atomic<std::uint64_t> importArms{0};
    std::atomic<std::uint64_t> gpuMarks{0};
    std::array<std::atomic<std::uint64_t>, 4> left{};
    std::atomic<std::uint32_t> period{1};
    std::unordered_map<std::uintptr_t, std::uint64_t> digests;
};

}
#endif

#endif
