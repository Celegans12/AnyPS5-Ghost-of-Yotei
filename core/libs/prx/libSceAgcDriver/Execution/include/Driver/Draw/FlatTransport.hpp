#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_FLATTRANSPORT_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_FLATTRANSPORT_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace AgcDriver::DriverDetail {

class PendingLedger {
public:
    using Range = std::pair<std::uint64_t, std::uint64_t>;
    static constexpr std::size_t InlineWrites = 6;
    static constexpr std::size_t InlineReads = 12;
    static constexpr std::size_t InlineLabel = 16;
    static constexpr std::uint64_t Every = std::numeric_limits<std::uint64_t>::max();
    enum class Held : std::uint8_t { None, Value, Unknown };
    enum class Stored : std::uint8_t { None, Stored, Unknown };
    struct Entry {
        std::span<const Range> writes;
        std::span<const Range> reads;
        std::uint64_t labelAddress = 0;
        std::span<const std::byte> label;
        std::uint64_t key = 0;
        std::uint64_t shape = 0;
        bool unknown = false;
        bool store = false;
    };
    struct Snapshot {
        std::uint64_t retired = 0;
        std::size_t items = 0;
        std::vector<Range> writes;
    };
    struct Counts {
        std::uint64_t published = 0;
        std::uint64_t wide = 0;
        std::uint64_t queries = 0;
        std::uint64_t lockedQueries = 0;
        std::uint64_t scanned = 0;
    };

    PendingLedger(std::size_t depth, bool verifiesAnswers, bool countsQueries)
        : capacity(std::bit_ceil(std::max<std::size_t>(depth, 1))), mask(capacity - 1), verifies(verifiesAnswers), counted(countsQueries), records(std::make_unique<Record[]>(capacity)) {}

    PendingLedger(const PendingLedger&) = delete;
    PendingLedger& operator=(const PendingLedger&) = delete;

    std::size_t Capacity() const { return capacity; }
    std::uint64_t Published() const { return published.load(); }
    std::uint64_t Retired() const { return retired.load(); }
    bool Holds(std::uint64_t position) const { return records[position & mask].stamp.load(std::memory_order_acquire) == position + 1; }

    void Publish(const Entry& entry) {
        std::unique_lock lock(referenceMutex, std::defer_lock);
        if (verifies) lock.lock();
        const auto position = published.load(std::memory_order_relaxed);
        if (position - retired.load() >= capacity) throw std::logic_error("flat transport: an element was published over one that has not been retired");
        Record& record = records[position & mask];
        record.stamp.store(0, std::memory_order_release);
        const bool wide = entry.writes.size() > InlineWrites || entry.reads.size() > InlineReads || entry.label.size() > InlineLabel;
        std::uint64_t low = Every;
        std::uint64_t high = 0;
        for (const auto& [begin, end] : entry.writes) {
            low = std::min(low, begin);
            high = std::max(high, end);
        }
        std::uint64_t layout = (entry.unknown ? UnknownBit : 0) | (entry.store ? StoreBit : 0);
        if (wide) {
            layout |= WideBit;
            std::lock_guard wideLock(wideMutex);
            wides.push_back(Wide{position, {entry.writes.begin(), entry.writes.end()}, {entry.reads.begin(), entry.reads.end()}, {entry.label.begin(), entry.label.end()}});
        } else {
            layout |= static_cast<std::uint64_t>(entry.writes.size()) | (static_cast<std::uint64_t>(entry.reads.size()) << ReadCountShift) | (static_cast<std::uint64_t>(entry.label.size()) << LabelSizeShift);
            for (std::size_t index = 0; index < entry.writes.size(); ++index) {
                record.writes[2 * index].store(entry.writes[index].first, std::memory_order_release);
                record.writes[2 * index + 1].store(entry.writes[index].second, std::memory_order_release);
            }
            for (std::size_t index = 0; index < entry.reads.size(); ++index) {
                record.reads[2 * index].store(entry.reads[index].first, std::memory_order_release);
                record.reads[2 * index + 1].store(entry.reads[index].second, std::memory_order_release);
            }
            std::array<std::uint64_t, LabelWords> words{};
            if (!entry.label.empty()) std::memcpy(words.data(), entry.label.data(), entry.label.size());
            for (std::size_t index = 0; index < LabelWords; ++index) record.label[index].store(words[index], std::memory_order_release);
        }
        record.low.store(low, std::memory_order_release);
        record.high.store(high, std::memory_order_release);
        record.labelAddress.store(entry.labelAddress, std::memory_order_release);
        record.key.store(entry.key, std::memory_order_release);
        record.shape.store(entry.shape, std::memory_order_release);
        record.released.store(0, std::memory_order_release);
        record.layout.store(layout, std::memory_order_release);
        record.stamp.store(position + 1, std::memory_order_release);
        if (verifies) listed.push_back(Listed{position, {entry.writes.begin(), entry.writes.end()}, {entry.reads.begin(), entry.reads.end()}, entry.labelAddress, {entry.label.begin(), entry.label.end()}, entry.key, entry.shape, entry.unknown, entry.store, false});
        if (counted) {
            publishedCount.fetch_add(1, std::memory_order_relaxed);
            if (wide) wideCount.fetch_add(1, std::memory_order_relaxed);
        }
        published.store(position + 1);
    }

    void Retire() {
        std::unique_lock lock(referenceMutex, std::defer_lock);
        if (verifies) lock.lock();
        const auto position = retired.load(std::memory_order_relaxed);
        if (position == published.load()) throw std::logic_error("flat transport: an element was retired from an empty queue");
        if ((records[position & mask].layout.load(std::memory_order_acquire) & WideBit) != 0) {
            std::lock_guard wideLock(wideMutex);
            std::erase_if(wides, [position](const Wide& wide) { return wide.position == position; });
        }
        if (verifies) {
            if (listed.empty() || listed.front().position != position) disagree("the element at its head");
            listed.pop_front();
        }
        retired.store(position + 1);
    }

    void ReleaseReads() {
        std::unique_lock lock(referenceMutex, std::defer_lock);
        if (verifies) lock.lock();
        const auto position = retired.load(std::memory_order_relaxed);
        if (position == published.load()) return;
        records[position & mask].released.store(1, std::memory_order_release);
        if (verifies && !listed.empty()) listed.front().released = true;
    }

    bool Overlaps(std::span<const Range> ranges, std::uint64_t before = Every) const {
        std::unique_lock lock(referenceMutex, std::defer_lock);
        if (verifies) lock.lock();
        Seen seen;
        const bool answer = overlapsFlat(ranges, before, seen);
        tally(seen);
        if (verifies && answer != overlapsListed(ranges, before)) disagree("whether a queued write overlaps a range");
        return answer;
    }

    std::optional<bool> OverlapsUnretired(std::uint64_t retiredBefore, std::span<const Range> ranges) const {
        std::unique_lock lock(referenceMutex, std::defer_lock);
        if (verifies) lock.lock();
        Seen seen;
        std::optional<bool> answer;
        if (retired.load() == retiredBefore) {
            const bool overlaps = overlapsFlat(ranges, Every, seen);
            if (retired.load() == retiredBefore) answer = overlaps;
        }
        tally(seen);
        if (verifies) {
            std::optional<bool> expected;
            if (retired.load() == retiredBefore) expected = overlapsListed(ranges, Every);
            if (expected != answer) disagree("whether a write overlaps a range while nothing was retired");
        }
        return answer;
    }

    bool Reads(std::uint64_t address, std::uint64_t end) const {
        std::unique_lock lock(referenceMutex, std::defer_lock);
        if (verifies) lock.lock();
        Seen seen;
        bool answer = false;
        const auto first = retired.load();
        const auto last = published.load();
        for (auto position = first; position < last && !answer; ++position) {
            if (!load(position, ReadsPart, seen) || seen.released) continue;
            const auto reads = seen.Reads();
            answer = std::any_of(reads.begin(), reads.end(), [&](const Range& read) { return address < read.second && read.first < end; });
        }
        tally(seen);
        if (verifies) {
            const bool expected = std::any_of(listed.begin(), listed.end(), [&](const Listed& item) {
                return !item.released && std::any_of(item.reads.begin(), item.reads.end(), [&](const Range& read) { return address < read.second && read.first < end; });
            });
            if (expected != answer) disagree("whether a queued element reads a range");
        }
        return answer;
    }

    bool Keyed(std::uint64_t key, std::uint64_t shape) const {
        std::unique_lock lock(referenceMutex, std::defer_lock);
        if (verifies) lock.lock();
        Seen seen;
        bool answer = false;
        const auto first = retired.load();
        const auto last = published.load();
        for (auto position = first; position < last && !answer; ++position) {
            if (!load(position, KeysPart, seen)) continue;
            answer = seen.key == key || (shape != 0 && seen.shape == shape);
        }
        tally(seen);
        if (verifies) {
            const bool expected = std::any_of(listed.begin(), listed.end(), [&](const Listed& item) { return item.key == key || (shape != 0 && item.shape == shape); });
            if (expected != answer) disagree("whether a queued element has a draw key or shape");
        }
        return answer;
    }

    Held Label(std::uint64_t address, std::size_t bytes, std::uint64_t before, std::uint64_t& value) const {
        std::unique_lock lock(referenceMutex, std::defer_lock);
        if (verifies) lock.lock();
        Seen seen;
        std::uint64_t found = 0;
        const auto answer = labelFlat(address, bytes, before, found, seen);
        tally(seen);
        if (verifies) {
            std::uint64_t expectedValue = 0;
            const auto expected = labelListed(address, bytes, before, expectedValue);
            if (expected != answer || (answer == Held::Value && expectedValue != found)) disagree("the value a queued write gives a label");
        }
        if (answer == Held::Value) value = found;
        return answer;
    }

    Stored StoredBytes(std::uint64_t address, std::size_t bytes, std::span<std::byte> out) const {
        std::unique_lock lock(referenceMutex, std::defer_lock);
        if (verifies) lock.lock();
        Seen seen;
        const auto answer = storedFlat(address, bytes, out, seen);
        tally(seen);
        if (verifies) {
            std::vector<std::byte> expectedBytes(out.size());
            const auto expected = storedListed(address, bytes, expectedBytes);
            const bool copied = answer == Stored::Stored && !out.empty();
            if (expected != answer || (copied && !std::equal(out.begin(), out.end(), expectedBytes.begin()))) disagree("the bytes a queued store writes");
        }
        return answer;
    }

    Snapshot Writes() const {
        std::unique_lock lock(referenceMutex, std::defer_lock);
        if (verifies) lock.lock();
        Seen seen;
        Snapshot shot;
        for (;;) {
            shot.retired = retired.load();
            const auto last = published.load();
            shot.writes.clear();
            bool whole = true;
            for (auto position = shot.retired; position < last && whole; ++position) {
                whole = load(position, WritesPart, seen);
                if (!whole) break;
                const auto writes = seen.Writes();
                shot.writes.insert(shot.writes.end(), writes.begin(), writes.end());
            }
            if (whole && retired.load() == shot.retired) {
                shot.items = static_cast<std::size_t>(last - shot.retired);
                break;
            }
            std::this_thread::yield();
        }
        tally(seen);
        if (verifies) {
            std::vector<Range> expected;
            for (const auto& item : listed) expected.insert(expected.end(), item.writes.begin(), item.writes.end());
            if (shot.items != listed.size() || shot.writes != expected) disagree("the writes of the queued elements");
        }
        return shot;
    }

    Counts Counted() const {
        const auto load = [](const std::atomic<std::uint64_t>& counter) { return counter.load(std::memory_order_relaxed); };
        return {load(publishedCount), load(wideCount), load(queryCount), load(lockedQueryCount), load(scannedCount)};
    }

private:
    static constexpr std::size_t LabelWords = InlineLabel / sizeof(std::uint64_t);
    static constexpr unsigned ReadCountShift = 8;
    static constexpr unsigned LabelSizeShift = 16;
    static constexpr std::uint64_t CountMask = 0xff;
    static constexpr std::uint64_t UnknownBit = std::uint64_t{1} << 24u;
    static constexpr std::uint64_t StoreBit = std::uint64_t{1} << 25u;
    static constexpr std::uint64_t WideBit = std::uint64_t{1} << 26u;
    enum Part : unsigned { BoundsPart = 0, WritesPart = 1, ReadsPart = 2, LabelPart = 4, KeysPart = 8 };

    struct alignas(64) Record {
        std::atomic<std::uint64_t> stamp{0};
        std::atomic<std::uint64_t> layout{0};
        std::atomic<std::uint64_t> low{0};
        std::atomic<std::uint64_t> high{0};
        std::array<std::atomic<std::uint64_t>, 2 * InlineWrites> writes{};
        std::atomic<std::uint64_t> labelAddress{0};
        std::array<std::atomic<std::uint64_t>, LabelWords> label{};
        std::atomic<std::uint64_t> key{0};
        std::atomic<std::uint64_t> shape{0};
        std::atomic<std::uint64_t> released{0};
        std::array<std::atomic<std::uint64_t>, 2 * InlineReads> reads{};
    };

    struct Wide {
        std::uint64_t position = 0;
        std::vector<Range> writes;
        std::vector<Range> reads;
        std::vector<std::byte> label;
    };

    struct Listed {
        std::uint64_t position = 0;
        std::vector<Range> writes;
        std::vector<Range> reads;
        std::uint64_t labelAddress = 0;
        std::vector<std::byte> label;
        std::uint64_t key = 0;
        std::uint64_t shape = 0;
        bool unknown = false;
        bool store = false;
        bool released = false;
    };

    struct Seen {
        std::uint64_t layout = 0;
        std::uint64_t low = 0;
        std::uint64_t high = 0;
        std::uint64_t labelAddress = 0;
        std::uint64_t key = 0;
        std::uint64_t shape = 0;
        std::uint64_t scanned = 0;
        std::uint64_t locked = 0;
        bool released = false;
        std::array<Range, InlineWrites> inlineWrites;
        std::array<Range, InlineReads> inlineReads;
        std::array<std::byte, InlineLabel> inlineLabel;
        std::vector<Range> wideWrites;
        std::vector<Range> wideReads;
        std::vector<std::byte> wideLabel;

        bool Unknown() const { return (layout & UnknownBit) != 0; }
        bool Store() const { return (layout & StoreBit) != 0; }
        bool IsWide() const { return (layout & WideBit) != 0; }
        std::span<const Range> Writes() const {
            if (IsWide()) return wideWrites;
            return {inlineWrites.data(), std::min<std::size_t>(static_cast<std::size_t>(layout & CountMask), InlineWrites)};
        }
        std::span<const Range> Reads() const {
            if (IsWide()) return wideReads;
            return {inlineReads.data(), std::min<std::size_t>(static_cast<std::size_t>((layout >> ReadCountShift) & CountMask), InlineReads)};
        }
        std::span<const std::byte> Label() const {
            if (IsWide()) return wideLabel;
            return {inlineLabel.data(), std::min<std::size_t>(static_cast<std::size_t>((layout >> LabelSizeShift) & CountMask), InlineLabel)};
        }
        bool Bounded(std::uint64_t begin, std::uint64_t end) const { return begin < high && low < end; }
        bool Writes(std::uint64_t begin, std::uint64_t end) const {
            const auto writes = Writes();
            return std::any_of(writes.begin(), writes.end(), [&](const Range& write) { return begin < write.second && write.first < end; });
        }
    };

    [[noreturn]] static void disagree(const char* what) {
        throw std::runtime_error(std::string("draw pipeline: the flat records of a queue and the queue they describe disagree on ") + what + " (APS5_VERIFY_PIPELINE_ORDER)");
    }

    void tally(const Seen& seen) const {
        if (!counted) return;
        queryCount.fetch_add(1, std::memory_order_relaxed);
        scannedCount.fetch_add(seen.scanned, std::memory_order_relaxed);
        if (seen.locked != 0) lockedQueryCount.fetch_add(1, std::memory_order_relaxed);
    }

    bool load(std::uint64_t position, unsigned parts, Seen& seen) const {
        const Record& record = records[position & mask];
        if (record.stamp.load(std::memory_order_acquire) != position + 1) return false;
        seen.layout = record.layout.load(std::memory_order_acquire);
        seen.low = record.low.load(std::memory_order_acquire);
        seen.high = record.high.load(std::memory_order_acquire);
        const bool wide = seen.IsWide();
        if ((parts & WritesPart) != 0 && !wide) {
            const auto count = seen.Writes().size();
            for (std::size_t index = 0; index < count; ++index) seen.inlineWrites[index] = {record.writes[2 * index].load(std::memory_order_acquire), record.writes[2 * index + 1].load(std::memory_order_acquire)};
        }
        if ((parts & ReadsPart) != 0) {
            seen.released = record.released.load(std::memory_order_acquire) != 0;
            const auto count = wide ? std::size_t{0} : seen.Reads().size();
            for (std::size_t index = 0; index < count; ++index) seen.inlineReads[index] = {record.reads[2 * index].load(std::memory_order_acquire), record.reads[2 * index + 1].load(std::memory_order_acquire)};
        }
        if ((parts & LabelPart) != 0) {
            seen.labelAddress = record.labelAddress.load(std::memory_order_acquire);
            std::array<std::uint64_t, LabelWords> words;
            for (std::size_t index = 0; index < LabelWords; ++index) words[index] = record.label[index].load(std::memory_order_acquire);
            std::memcpy(seen.inlineLabel.data(), words.data(), InlineLabel);
        }
        if ((parts & KeysPart) != 0) {
            seen.key = record.key.load(std::memory_order_acquire);
            seen.shape = record.shape.load(std::memory_order_acquire);
        }
        if (record.stamp.load(std::memory_order_acquire) != position + 1) return false;
        ++seen.scanned;
        if (!wide || (parts & (WritesPart | ReadsPart | LabelPart)) == 0) return true;
        ++seen.locked;
        std::lock_guard wideLock(wideMutex);
        const auto found = std::find_if(wides.begin(), wides.end(), [position](const Wide& held) { return held.position == position; });
        if (found == wides.end()) return false;
        if ((parts & WritesPart) != 0) seen.wideWrites = found->writes;
        if ((parts & ReadsPart) != 0) seen.wideReads = found->reads;
        if ((parts & LabelPart) != 0) seen.wideLabel = found->label;
        return true;
    }

    bool overlapsFlat(std::span<const Range> ranges, std::uint64_t before, Seen& seen) const {
        const auto first = retired.load();
        const auto last = std::min(published.load(), before);
        for (auto position = first; position < last; ++position) {
            if (!load(position, BoundsPart, seen)) continue;
            if (seen.Unknown()) return true;
            if (std::none_of(ranges.begin(), ranges.end(), [&](const Range& range) { return seen.Bounded(range.first, range.second); })) continue;
            if (!load(position, WritesPart, seen)) continue;
            if (std::any_of(ranges.begin(), ranges.end(), [&](const Range& range) { return seen.Writes(range.first, range.second); })) return true;
        }
        return false;
    }

    bool overlapsListed(std::span<const Range> ranges, std::uint64_t before) const {
        for (const auto& item : listed) {
            if (item.position >= before) break;
            if (item.unknown) return true;
            for (const auto& [first, last] : item.writes) {
                for (const auto& [begin, end] : ranges) {
                    if (begin < last && first < end) return true;
                }
            }
        }
        return false;
    }

    Held labelFlat(std::uint64_t address, std::size_t bytes, std::uint64_t before, std::uint64_t& value, Seen& seen) const {
        const auto end = address + bytes;
        const auto first = retired.load();
        const auto last = std::min(published.load(), before);
        for (auto position = last; position > first;) {
            --position;
            if (!load(position, BoundsPart, seen)) continue;
            if (seen.Unknown()) return Held::Unknown;
            if (!seen.Bounded(address, end)) continue;
            if (!load(position, WritesPart | LabelPart, seen)) continue;
            if (!seen.Writes(address, end)) continue;
            const auto label = seen.Label();
            if (label.empty() || bytes > sizeof(value) || address < seen.labelAddress || end > seen.labelAddress + label.size()) return Held::Unknown;
            value = 0;
            std::memcpy(&value, label.data() + (address - seen.labelAddress), bytes);
            return Held::Value;
        }
        return Held::None;
    }

    Held labelListed(std::uint64_t address, std::size_t bytes, std::uint64_t before, std::uint64_t& value) const {
        const auto end = address + bytes;
        for (auto item = listed.rbegin(); item != listed.rend(); ++item) {
            if (item->position >= before) continue;
            if (item->unknown) return Held::Unknown;
            const bool overlaps = std::any_of(item->writes.begin(), item->writes.end(), [&](const Range& range) { return address < range.second && range.first < end; });
            if (!overlaps) continue;
            if (item->label.empty() || bytes > sizeof(value) || address < item->labelAddress || end > item->labelAddress + item->label.size()) return Held::Unknown;
            value = 0;
            std::memcpy(&value, item->label.data() + (address - item->labelAddress), bytes);
            return Held::Value;
        }
        return Held::None;
    }

    Stored storedFlat(std::uint64_t address, std::size_t bytes, std::span<std::byte> out, Seen& seen) const {
        const auto end = address + bytes;
        auto answer = Stored::None;
        const auto first = retired.load();
        const auto last = published.load();
        for (auto position = last; position > first;) {
            --position;
            if (!load(position, BoundsPart, seen) || !seen.Bounded(address, end)) continue;
            if (!load(position, WritesPart | LabelPart, seen) || !seen.Writes(address, end)) continue;
            if (!seen.Store()) return Stored::Unknown;
            if (out.empty()) {
                answer = Stored::Stored;
                continue;
            }
            const auto label = seen.Label();
            if (out.size() != bytes || address < seen.labelAddress || end > seen.labelAddress + label.size()) return Stored::Unknown;
            std::memcpy(out.data(), label.data() + (address - seen.labelAddress), bytes);
            return Stored::Stored;
        }
        return answer;
    }

    Stored storedListed(std::uint64_t address, std::size_t bytes, std::span<std::byte> out) const {
        const auto end = address + bytes;
        auto answer = Stored::None;
        for (auto item = listed.rbegin(); item != listed.rend(); ++item) {
            if (std::none_of(item->writes.begin(), item->writes.end(), [&](const Range& range) { return address < range.second && range.first < end; })) continue;
            if (!item->store) return Stored::Unknown;
            if (out.empty()) {
                answer = Stored::Stored;
                continue;
            }
            if (out.size() != bytes || address < item->labelAddress || end > item->labelAddress + item->label.size()) return Stored::Unknown;
            std::memcpy(out.data(), item->label.data() + (address - item->labelAddress), bytes);
            return Stored::Stored;
        }
        return answer;
    }

    const std::size_t capacity;
    const std::uint64_t mask;
    const bool verifies;
    const bool counted;
    const std::unique_ptr<Record[]> records;
    alignas(64) std::atomic<std::uint64_t> published{0};
    alignas(64) std::atomic<std::uint64_t> retired{0};
    mutable std::mutex wideMutex;
    std::vector<Wide> wides;
    mutable std::mutex referenceMutex;
    std::deque<Listed> listed;
    mutable std::atomic<std::uint64_t> publishedCount{0};
    mutable std::atomic<std::uint64_t> wideCount{0};
    mutable std::atomic<std::uint64_t> queryCount{0};
    mutable std::atomic<std::uint64_t> lockedQueryCount{0};
    mutable std::atomic<std::uint64_t> scannedCount{0};
};

template<typename TPayload>
class FlatQueue {
public:
    struct Counts {
        std::uint64_t parks = 0;
        std::uint64_t wakes = 0;
        std::uint64_t progressWakes = 0;
        std::uint64_t collisions = 0;
    };

    class Turn {
    public:
        explicit Turn(FlatQueue& taken) : queue(taken) { Retake(); }
        ~Turn() {
            if (held) queue.turn.store(false, std::memory_order_release);
        }
        Turn(const Turn&) = delete;
        Turn& operator=(const Turn&) = delete;

        void Release() {
            held = false;
            queue.turn.store(false, std::memory_order_release);
        }

        void Retake() {
            while (queue.turn.exchange(true, std::memory_order_acquire)) {
                queue.collisions.fetch_add(1, std::memory_order_relaxed);
                while (queue.turn.load(std::memory_order_relaxed)) std::this_thread::yield();
            }
            held = true;
        }

    private:
        FlatQueue& queue;
        bool held = false;
    };

    FlatQueue(std::size_t queueDepth, std::size_t wakeBatch, bool verifiesOrder, bool countsQueries)
        : ledger(queueDepth, verifiesOrder, countsQueries), slots(std::make_unique<Slot[]>(ledger.Capacity())), mask(ledger.Capacity() - 1), depth(std::max<std::size_t>(queueDepth, 1)), batch(std::clamp<std::size_t>(wakeBatch, 1, std::max<std::size_t>(queueDepth, 1))), verifies(verifiesOrder) {}

    FlatQueue(const FlatQueue&) = delete;
    FlatQueue& operator=(const FlatQueue&) = delete;

    PendingLedger& Ledger() { return ledger; }
    const PendingLedger& Ledger() const { return ledger; }
    std::size_t Depth() const { return depth; }
    std::size_t Batch() const { return batch; }
    bool Closed() const { return closed.load(); }

    std::size_t Queued() const {
        const auto first = ledger.Retired();
        return static_cast<std::size_t>(ledger.Published() - first);
    }

    bool Full() const { return Queued() >= depth; }

    TPayload& Stage() {
        reclaim();
        return slots[ledger.Published() & mask].payload;
    }

    void Commit(const PendingLedger::Entry& entry) {
        const auto position = ledger.Published();
        slots[position & mask].position = position;
        ledger.Publish(entry);
        if (parked.load() == 0) return;
        parked.store(0);
        parked.notify_one();
        wakes.fetch_add(1, std::memory_order_relaxed);
    }

    void Reclaim() {
        Turn held(*this);
        reclaim();
    }

    void AwaitRetired(std::uint64_t target) {
        for (;;) {
            const auto seen = progress.load();
            auto armed = wakeAt.load();
            while (target < armed && !wakeAt.compare_exchange_weak(armed, target)) {
            }
            if (ledger.Retired() >= target) return;
            progress.wait(seen);
        }
    }

    void AwaitRoom() {
        const auto last = ledger.Published();
        AwaitRetired(last + batch > depth ? last + batch - depth : 0);
    }

    void AwaitEmpty() {
        for (;;) {
            const auto target = ledger.Published();
            if (ledger.Retired() >= target) return;
            AwaitRetired(target);
        }
    }

    TPayload* Front() {
        const auto position = ledger.Retired();
        if (ledger.Published() == position) return nullptr;
        Slot& slot = slots[position & mask];
        if (verifies && (slot.position != position || !ledger.Holds(position))) throw std::runtime_error("draw pipeline: the slot at the head of a flat queue does not hold the element published at its position (APS5_VERIFY_PIPELINE_ORDER)");
        return &slot.payload;
    }

    void Park() {
        parked.store(1);
        if (ledger.Published() == ledger.Retired() && !closed.load()) {
            parks.fetch_add(1, std::memory_order_relaxed);
            parked.wait(1);
        }
        parked.store(0);
    }

    void Retire() {
        ledger.Retire();
        if (ledger.Retired() < wakeAt.load()) return;
        wakeAt.store(Never);
        progress.fetch_add(1);
        progress.notify_all();
        progressWakes.fetch_add(1, std::memory_order_relaxed);
    }

    void Close() {
        {
            Turn held(*this);
            closed.store(true);
        }
        parked.store(0);
        parked.notify_all();
    }

    Counts Counted() const {
        const auto load = [](const std::atomic<std::uint64_t>& counter) { return counter.load(std::memory_order_relaxed); };
        return {load(parks), load(wakes), load(progressWakes), load(collisions)};
    }

private:
    static constexpr std::uint64_t Never = std::numeric_limits<std::uint64_t>::max();

    struct Slot {
        TPayload payload{};
        std::uint64_t position = 0;
    };

    void reclaim() {
        const auto done = ledger.Retired();
        for (; reclaimed < done; ++reclaimed) slots[reclaimed & mask].payload = TPayload{};
    }

    PendingLedger ledger;
    const std::unique_ptr<Slot[]> slots;
    const std::uint64_t mask;
    const std::size_t depth;
    const std::size_t batch;
    const bool verifies;
    std::uint64_t reclaimed = 0;
    alignas(64) std::atomic<bool> turn{false};
    std::atomic<bool> closed{false};
    std::atomic<std::uint64_t> collisions{0};
    std::atomic<std::uint64_t> wakes{0};
    alignas(64) std::atomic<std::uint32_t> parked{0};
    std::atomic<std::uint64_t> parks{0};
    std::atomic<std::uint64_t> progressWakes{0};
    alignas(64) std::atomic<std::uint64_t> wakeAt{Never};
    std::atomic<std::uint32_t> progress{0};
};

}

#endif
