#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/WriteEvidence.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PendingPages.hpp"
#include "prx/libc/include/HostThreadSlot.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

namespace AgcDriver::DriverDetail {

namespace {

enum class BlockSite : std::uint8_t { Validate, Query, Span, Shadow, Count };
constexpr std::array<const char*, static_cast<std::size_t>(BlockSite::Count)> blockSiteNames{"validate", "query", "span", "shadow"};
constexpr std::uint64_t WriteBlockBytes = 64 * 1024;
constexpr std::uint64_t WritePageBytes = 4 * 1024;
constexpr std::size_t ViewCallerSlots = 32;
constexpr std::uint64_t BlockReportEvery = 1024;

struct BlockProbe {
    bool overlaps = false;
    bool blocks = false;
    bool pages = false;
    bool pipelineBlocks = false;
    bool recorderBlocks = false;
};

struct alignas(64) BlockSiteCounts {
    std::atomic<std::uint64_t> reads{0};
    std::atomic<std::uint64_t> overlaps{0};
    std::atomic<std::uint64_t> blocks{0};
    std::atomic<std::uint64_t> pages{0};
    std::atomic<std::uint64_t> pipelineBlocks{0};
    std::atomic<std::uint64_t> recorderBlocks{0};
    std::atomic<std::uint64_t> spanned{0};
};

struct BlockSiteTaken {
    std::uint64_t reads = 0;
    std::uint64_t overlaps = 0;
    std::uint64_t blocks = 0;
    std::uint64_t pages = 0;
    std::uint64_t pipelineBlocks = 0;
    std::uint64_t recorderBlocks = 0;
    std::uint64_t spanned = 0;
};

struct ViewCaller {
    std::atomic<std::uintptr_t> caller{0};
    BlockSiteCounts counts;
};

std::array<BlockSiteCounts, static_cast<std::size_t>(BlockSite::Count)> blockSites;
std::array<ViewCaller, ViewCallerSlots> viewCallers;
BlockSiteCounts otherViewCallers;
std::atomic<std::uint64_t> blockNotes{0};
std::atomic<std::int64_t> blockReportDue{0};
std::atomic<std::int64_t> blockReportFrom{0};

struct OuterBlockSiteTag;

bool writeBlocksStats() {
    static const bool enabled = std::getenv("APS5_WRITE_BLOCKS_STATS") != nullptr;
    return enabled;
}

class OuterBlockSite {
public:
    explicit OuterBlockSite(bool active) : active(active), previous(active && HostThreadSlot<bool, OuterBlockSiteTag>::Get()) {
        if (active) HostThreadSlot<bool, OuterBlockSiteTag>::Set(true);
    }
    ~OuterBlockSite() {
        if (active) HostThreadSlot<bool, OuterBlockSiteTag>::Set(previous);
    }
    OuterBlockSite(const OuterBlockSite&) = delete;
    OuterBlockSite& operator=(const OuterBlockSite&) = delete;

private:
    bool active;
    bool previous;
};

std::uint64_t lastReadByte(std::uint64_t address, std::size_t bytes) {
    const auto count = static_cast<std::uint64_t>(bytes);
    return count - 1 > std::numeric_limits<std::uint64_t>::max() - address ? std::numeric_limits<std::uint64_t>::max() : address + count - 1;
}

std::pair<std::uint64_t, std::size_t> widenedRead(std::uint64_t address, std::size_t bytes, std::uint64_t grain) {
    const auto first = address & ~(grain - 1);
    const auto span = (lastReadByte(address, bytes) | (grain - 1)) - first;
    return {first, static_cast<std::size_t>(span == std::numeric_limits<std::uint64_t>::max() ? span : span + 1)};
}

bool viewOverlaps(const PendingView& view, bool legacy, std::uint64_t address, std::size_t bytes) {
    return legacy ? Graphics::Recorder::SnapshotWriteOverlaps(address, bytes) : Graphics::Recorder::SnapshotOverlaps(view.snapshot.get(), address, bytes);
}

template<typename TOverlaps>
void probeQueue(BlockProbe& probe, bool& queueBlocks, std::uint64_t address, std::size_t bytes, bool exact, TOverlaps&& overlaps) {
    const auto [blockAddress, blockBytes] = widenedRead(address, bytes, WriteBlockBytes);
    const bool block = exact || overlaps(blockAddress, blockBytes);
    const auto [pageAddress, pageBytes] = widenedRead(address, bytes, WritePageBytes);
    const bool page = exact || (block && overlaps(pageAddress, pageBytes));
    probe.overlaps = probe.overlaps || exact;
    probe.blocks = probe.blocks || block;
    probe.pages = probe.pages || page;
    queueBlocks = queueBlocks || block;
}

void probeRecorder(BlockProbe& probe, const PendingView& view, bool legacy, std::uint64_t address, std::size_t bytes, bool exact) {
    probeQueue(probe, probe.recorderBlocks, address, bytes, exact, [&](std::uint64_t first, std::size_t count) { return viewOverlaps(view, legacy, first, count); });
}

double blockShare(std::uint64_t part, std::uint64_t whole) {
    return whole == 0 ? 0.0 : 100.0 * static_cast<double>(part) / static_cast<double>(whole);
}

BlockSiteTaken takeBlockSite(BlockSiteCounts& counts) {
    return {counts.reads.exchange(0, std::memory_order_relaxed), counts.overlaps.exchange(0, std::memory_order_relaxed), counts.blocks.exchange(0, std::memory_order_relaxed), counts.pages.exchange(0, std::memory_order_relaxed), counts.pipelineBlocks.exchange(0, std::memory_order_relaxed), counts.recorderBlocks.exchange(0, std::memory_order_relaxed), counts.spanned.exchange(0, std::memory_order_relaxed)};
}

void addBlockSite(BlockSiteTaken& total, const BlockSiteTaken& taken) {
    total.reads += taken.reads;
    total.overlaps += taken.overlaps;
    total.blocks += taken.blocks;
    total.pages += taken.pages;
    total.pipelineBlocks += taken.pipelineBlocks;
    total.recorderBlocks += taken.recorderBlocks;
    total.spanned += taken.spanned;
}

void appendBlockSite(std::string& text, const char* name, const BlockSiteTaken& taken, bool queues) {
    char piece[512];
    if (queues) {
        std::snprintf(piece, sizeof(piece), "; %s %llu reads, overlap %.2f %%, block %.2f %% (pipeline %.2f %%, recorder %.2f %%), page %.2f %%, %.2f blocks/read", name, static_cast<unsigned long long>(taken.reads), blockShare(taken.overlaps, taken.reads), blockShare(taken.blocks, taken.reads), blockShare(taken.pipelineBlocks, taken.reads), blockShare(taken.recorderBlocks, taken.reads), blockShare(taken.pages, taken.reads), taken.reads == 0 ? 0.0 : static_cast<double>(taken.spanned) / static_cast<double>(taken.reads));
    } else {
        std::snprintf(piece, sizeof(piece), "; %s %llu reads, overlap %.2f %%, block %.2f %%, page %.2f %%, %.2f blocks/read", name, static_cast<unsigned long long>(taken.reads), blockShare(taken.overlaps, taken.reads), blockShare(taken.blocks, taken.reads), blockShare(taken.pages, taken.reads), taken.reads == 0 ? 0.0 : static_cast<double>(taken.spanned) / static_cast<double>(taken.reads));
    }
    text += piece;
}

void reportWriteBlocks() {
    const auto now = static_cast<std::int64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto period = static_cast<std::int64_t>(std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::seconds(10)).count());
    auto due = blockReportDue.load(std::memory_order_relaxed);
    if (due == 0) {
        if (blockReportDue.compare_exchange_strong(due, now + period, std::memory_order_relaxed)) blockReportFrom.store(now, std::memory_order_relaxed);
        return;
    }
    if (now < due || !blockReportDue.compare_exchange_strong(due, now + period, std::memory_order_relaxed)) return;
    const auto from = blockReportFrom.exchange(now, std::memory_order_relaxed);
    const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::duration(now - from)).count();
    const auto anchor = reinterpret_cast<std::uintptr_t>(&Driver::QueryPendingSpan);
    BlockSiteTaken total;
    std::string text;
    for (std::size_t site = 0; site < blockSites.size(); ++site) {
        const auto taken = takeBlockSite(blockSites[site]);
        if (taken.reads == 0) continue;
        addBlockSite(total, taken);
        appendBlockSite(text, blockSiteNames[site], taken, site == static_cast<std::size_t>(BlockSite::Query) || site == static_cast<std::size_t>(BlockSite::Span));
    }
    for (auto& slot : viewCallers) {
        const auto caller = slot.caller.load(std::memory_order_acquire);
        if (caller == 0) continue;
        const auto taken = takeBlockSite(slot.counts);
        if (taken.reads == 0) continue;
        addBlockSite(total, taken);
        char name[48];
        std::snprintf(name, sizeof(name), "view@%c0x%llx", caller >= anchor ? '+' : '-', static_cast<unsigned long long>(caller >= anchor ? caller - anchor : anchor - caller));
        appendBlockSite(text, name, taken, false);
    }
    const auto other = takeBlockSite(otherViewCallers);
    if (other.reads != 0) {
        addBlockSite(total, other);
        appendBlockSite(text, "view@other", other, false);
    }
    if (total.reads == 0) return;
    ProfilePrint_nid_no_patch("[write-blocks] (%.1f s, view callers as offsets from Driver::QueryPendingSpan): all %llu reads, overlap %.2f %%, block %.2f %%, page %.2f %%%s\n", seconds, static_cast<unsigned long long>(total.reads), blockShare(total.overlaps, total.reads), blockShare(total.blocks, total.reads), blockShare(total.pages, total.reads), text.c_str());
}

void noteBlockRead(BlockSiteCounts& counts, const BlockProbe& probe, std::uint64_t address, std::size_t bytes) {
    counts.reads.fetch_add(1, std::memory_order_relaxed);
    if (probe.overlaps) counts.overlaps.fetch_add(1, std::memory_order_relaxed);
    if (probe.blocks) counts.blocks.fetch_add(1, std::memory_order_relaxed);
    if (probe.pages) counts.pages.fetch_add(1, std::memory_order_relaxed);
    if (probe.pipelineBlocks) counts.pipelineBlocks.fetch_add(1, std::memory_order_relaxed);
    if (probe.recorderBlocks) counts.recorderBlocks.fetch_add(1, std::memory_order_relaxed);
    counts.spanned.fetch_add((lastReadByte(address, bytes) >> 16u) - (address >> 16u) + 1, std::memory_order_relaxed);
    if (blockNotes.fetch_add(1, std::memory_order_relaxed) % BlockReportEvery == 0) reportWriteBlocks();
}

BlockSiteCounts& viewCallerCounts(std::uintptr_t caller) {
    if (caller == 0) return otherViewCallers;
    const auto start = static_cast<std::size_t>((static_cast<std::uint64_t>(caller) * 0x9e3779b97f4a7c15ull) >> 59u);
    for (std::size_t probe = 0; probe < ViewCallerSlots; ++probe) {
        auto& slot = viewCallers[(start + probe) % ViewCallerSlots];
        auto held = slot.caller.load(std::memory_order_acquire);
        if (held == 0 && slot.caller.compare_exchange_strong(held, caller, std::memory_order_acq_rel)) return slot.counts;
        if (held == caller) return slot.counts;
    }
    return otherViewCallers;
}

void noteViewRead(std::uintptr_t caller, const PendingView& view, bool legacy, std::uint64_t address, std::size_t bytes, bool exact) {
    BlockProbe probe;
    probeRecorder(probe, view, legacy, address, bytes, exact);
    noteBlockRead(viewCallerCounts(caller), probe, address, bytes);
}

void noteValidateRead(const PendingView& view, bool legacy, std::uint64_t address, std::size_t bytes, bool exact) {
    if (HostThreadSlot<bool, OuterBlockSiteTag>::Get()) return;
    BlockProbe probe;
    probeRecorder(probe, view, legacy, address, bytes, exact);
    noteBlockRead(blockSites[static_cast<std::size_t>(BlockSite::Validate)], probe, address, bytes);
}

void noteShadowRead(std::uint64_t address, std::size_t bytes, bool exact) {
    BlockProbe probe;
    bool shadowBlocks = false;
    probeQueue(probe, shadowBlocks, address, bytes, exact, [](std::uint64_t first, std::size_t count) { return Graphics::AnyShadowedOverlaps(first, count); });
    noteBlockRead(blockSites[static_cast<std::size_t>(BlockSite::Shadow)], probe, address, bytes);
}

void noteQueuedRead(BlockSite site, DrawPipeline* pipeline, bool queued, bool legacy, std::uint64_t address, std::size_t bytes) {
    BlockProbe probe;
    if (pipeline != nullptr) probeQueue(probe, probe.pipelineBlocks, address, bytes, queued, [pipeline](std::uint64_t first, std::size_t count) { return pipeline->Overlaps(first, count); });
    PendingView view;
    view.Load();
    probeRecorder(probe, view, legacy, address, bytes, viewOverlaps(view, legacy, address, bytes));
    noteBlockRead(blockSites[static_cast<std::size_t>(site)], probe, address, bytes);
}

}

SampledReadScope::SampledReadScope(std::atomic<std::uint64_t>& reads) : previous(Driver::sampledRead()) {
    Driver::setSampledRead(Driver::writeEvidenceEnabled() && reads.fetch_add(1, std::memory_order_relaxed) % Driver::writeEvidenceSampleEvery() == 0);
}

SampledReadScope::~SampledReadScope() { Driver::setSampledRead(previous); }

void PendingView::Load() {
    if (Driver::validateLegacy()) return;
    const auto current = Graphics::Recorder::PublishGeneration();
    if (loaded && current == generation) return;
    snapshot = Graphics::Recorder::PendingWriteSnapshot();
    generation = current;
    loaded = true;
}

bool PendingView::Overlaps(std::uint64_t address, std::size_t bytes) const {
    const bool legacy = Driver::validateLegacy();
    const bool overlaps = viewOverlaps(*this, legacy, address, bytes);
    if (writeBlocksStats() && bytes != 0) noteViewRead(reinterpret_cast<std::uintptr_t>(__builtin_return_address(0)), *this, legacy, address, bytes, overlaps);
    return overlaps;
}

bool PendingView::AnyOverlaps(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges) {
    using Graphics::PendingPages::Layer;
    if (!writeBlocksStats() && Graphics::PendingPages::Rejects(Layer::Recorder, ranges)) return false;
    const bool verify = Graphics::PendingPages::Verifying();
    const auto sequence = verify ? Graphics::PendingPages::Sequence(Layer::Recorder) : 0;
    const bool clear = verify && Graphics::PendingPages::Clear(Layer::Recorder, ranges);
    Load();
    const auto found = std::find_if(ranges.begin(), ranges.end(), [&](const std::pair<std::uint64_t, std::uint64_t>& range) { return Overlaps(range.first, static_cast<std::size_t>(range.second - range.first)); });
    const bool overlaps = found != ranges.end();
    if (verify) Graphics::PendingPages::Compare(Layer::Recorder, sequence, clear, overlaps, overlaps ? found->first : 0, overlaps ? static_cast<std::size_t>(found->second - found->first) : 0);
    return overlaps;
}

namespace {

struct SampledReadTag;

}

bool Driver::sampledRead() {
    return HostThreadSlot<bool, SampledReadTag>::Get();
}

void Driver::setSampledRead(bool sampled) {
    HostThreadSlot<bool, SampledReadTag>::Set(sampled);
}

bool Driver::writeEvidenceEnabled() {
    static const bool enabled = std::getenv("APS5_NO_WRITE_EVIDENCE") == nullptr;
    return enabled;
}

bool Driver::writerKeyedEvidence() {
    static const bool keyed = std::getenv("APS5_NO_WRITER_KEYED_EVIDENCE") == nullptr;
    return keyed;
}

bool Driver::storedKnownValues() {
    static const bool enabled = std::getenv("APS5_NO_STORED_KNOWN_VALUES") == nullptr;
    return enabled;
}

bool Driver::foreignWriters() {
    static const bool noted = std::getenv("APS5_NO_FOREIGN_WRITERS") == nullptr;
    return noted;
}

bool Driver::observationGuard() {
    static const bool guard = std::getenv("APS5_NO_OBSERVATION_GUARD") == nullptr;
    return guard;
}

bool Driver::siteSampling() {
    static const bool perSite = std::getenv("APS5_NO_SITE_SAMPLING") == nullptr;
    return perSite;
}

std::uint64_t Driver::hookWaits() {
    return Graphics::Recorder::ThreadHookWaits();
}

ShaderMemory::HookWaitCounter Driver::hookWaitCounter() {
    return observationGuard() ? &hookWaits : nullptr;
}

bool Driver::validateSkipEnabled() {
    static const bool enabled = std::getenv("APS5_NO_VALIDATE_SKIP") == nullptr;
    return enabled;
}

bool Driver::validateLegacy() {
    static const bool legacy = std::getenv("APS5_VALIDATE_LEGACY") != nullptr;
    return legacy;
}

bool Driver::gateRetry() {
    static const bool retry = std::getenv("APS5_NO_GATE_RETRY") == nullptr && !validateLegacy();
    return retry;
}

bool Driver::validateSkipVerify() {
    static const bool verify = std::getenv("APS5_VALIDATE_SYNC_SKIP") != nullptr;
    return verify;
}

std::uint32_t Driver::writeEvidenceAfter() {
    static const std::uint32_t value = [] { const char* text = std::getenv("APS5_WRITE_EVIDENCE_AFTER"); return text ? static_cast<std::uint32_t>(std::atoi(text)) : 4u; }();
    return value;
}

std::uint32_t Driver::writeEvidenceSampleEvery() {
    static const std::uint32_t value = [] { const char* text = std::getenv("APS5_WRITE_EVIDENCE_SAMPLE"); const auto parsed = text ? std::atoi(text) : 16; return parsed > 0 ? static_cast<std::uint32_t>(parsed) : 16u; }();
    return value;
}

std::uint64_t Driver::writeEvidenceMaxBytes() {
    static const std::uint64_t value = [] { const char* text = std::getenv("APS5_WRITE_EVIDENCE_MAX_KIB"); return (text ? std::strtoull(text, nullptr, 10) : 16ull) << 10u; }();
    return value;
}

bool Driver::traceCapSync() {
    static const bool trace = std::getenv("APS5_TRACE_CAPSYNC") != nullptr;
    return trace;
}

void Driver::observeDword(std::uint64_t address, bool unchanged) {
    if (!writeEvidenceEnabled()) return;
    (unchanged ? observedUnchanged : observedChanged).fetch_add(1, std::memory_order_relaxed);
    std::lock_guard lock(writtenBuffersMutex);
    if (dwordEvidence.size() >= DwordEvidenceEntries && !dwordEvidence.contains(address)) dwordEvidence.clear();
    auto& evidence = dwordEvidence[address];
    if (writerKeyedEvidence()) {
        const auto writer = newestWriterLocked(address, address + 4);
        const std::uint64_t program = writer ? writer->program : 0, begin = writer ? writer->begin : 0, end = writer ? writer->end : 0;
        if (program != evidence.program || begin != evidence.begin || end != evidence.end) evidence = DwordEvidence{0, 0, program, begin, end};
    }
    if (!unchanged) {
        evidence.streak = 0;
        ++evidence.changed;
    } else if (evidence.streak < (1u << 30u)) {
        ++evidence.streak;
    }
}

void Driver::observeRange(std::uint64_t address, std::span<const std::byte> before) {
    if (before.empty() || !GuestMemory::Accessible(reinterpret_cast<const void*>(address), before.size())) return;
    const auto* now = reinterpret_cast<const std::byte*>(address);
    for (std::size_t offset = 0; offset + 4 <= before.size(); offset += 4) {
        observeDword(address + offset, std::memcmp(before.data() + offset, now + offset, 4) == 0);
    }
}

void Driver::observePendingWrite(std::uint64_t address, bool unchanged) {
    Get().observeDword(address, unchanged);
}

bool Driver::knownValueCurrent(const WrittenBuffer& writer) {
    const auto bytes = static_cast<std::size_t>(writer.end - writer.begin);
    GuestMemory::CollectWrites(writer.begin, bytes);
    return GuestMemory::UnchangedSince(writer.begin, bytes, writer.generation);
}

ShaderMemory::PendingWrite Driver::classifyPendingWrite(std::uint64_t address, std::size_t bytes, std::uint64_t ValidateCounters::*& reason, const PendingView& pending, std::span<std::byte> known) {
    using Policy = ShaderMemory::PendingWrite;
    const bool legacy = validateLegacy();
    const bool stats = writeBlocksStats() && bytes != 0;
    const bool overlaps = viewOverlaps(pending, legacy, address, bytes);
    if (stats) noteValidateRead(pending, legacy, address, bytes, overlaps);
    if (!overlaps) return Policy::None;

    const bool shadowed = Graphics::AnyShadowedOverlaps(address, bytes);
    if (stats) noteShadowRead(address, bytes, shadowed);
    if (shadowed) {
        reason = &ValidateCounters::syncedShadow;
        return Policy::Sync;
    }
    if (storedKnownValues() && !known.empty() && known.size() == bytes && Graphics::Recorder::LookupStoredBytes(address, known)) return knownValueVerify() ? Policy::VerifyKnownValue : Policy::KnownValue;
    reason = &ValidateCounters::syncedOff;
    if (!writeEvidenceEnabled()) return Policy::Sync;
    const auto writer = newestWriter(address, address + bytes);
    if (!writer) {
        reason = &ValidateCounters::syncedNoWriter;
        return Policy::Sync;
    }
    if (writer->program == 0) {
        reason = &ValidateCounters::syncedForeign;
        return Policy::Sync;
    }
    if (writer->end - writer->begin > writeEvidenceMaxBytes() || writer->begin < 4 || viewOverlaps(pending, legacy, writer->begin - 4, 4) || viewOverlaps(pending, legacy, writer->end, 4)) {
        reason = &ValidateCounters::syncedLargeRange;
        return Policy::Sync;
    }
    const auto first = address & ~3ull;
    const auto limit = address + bytes;
    if (Graphics::Recorder::WideLabelIn(first, static_cast<std::size_t>(limit - first))) {
        reason = &ValidateCounters::syncedLabel;
        return Policy::Sync;
    }
    for (std::uint64_t dword = first; dword < limit; dword += 4) {
        if (Graphics::Recorder::LookupLabelValue(dword, 4, 0).has_value()) {
            reason = &ValidateCounters::syncedLabel;
            return Policy::Sync;
        }
    }
    if (writer->value && writer->begin <= address && limit <= writer->end && (known.empty() || known.size() == bytes) && knownValueCurrent(*writer)) {
        if (!known.empty()) {
            std::memcpy(known.data(), writer->value->data() + (address - writer->begin), bytes);
            knownValueReads.fetch_add(1, std::memory_order_relaxed);
        }
        return knownValueVerify() ? Policy::VerifyKnownValue : Policy::KnownValue;
    }
    {
        std::lock_guard lock(writtenBuffersMutex);
        for (std::uint64_t dword = first; dword < limit; dword += 4) {
            const auto found = dwordEvidence.find(dword);
            if (found == dwordEvidence.end() || found->second.streak < writeEvidenceAfter()) {
                reason = &ValidateCounters::syncedEvidence;
                return Policy::Sync;
            }
            if (writerKeyedEvidence() && (found->second.program != writer->program || found->second.begin != writer->begin || found->second.end != writer->end)) {
                reason = &ValidateCounters::syncedWriterChanged;
                return Policy::Sync;
            }
        }
    }
    if (sampledRead()) {
        reason = &ValidateCounters::syncedSample;
        return Policy::Sync;
    }
    return validateSkipVerify() ? Policy::VerifyRaw : Policy::Raw;
}

ShaderMemory::PendingWrite Driver::queryPendingWrite(std::uint64_t address, std::size_t bytes, std::span<std::byte> known) {
    using Policy = ShaderMemory::PendingWrite;
    const bool stats = writeBlocksStats() && bytes != 0;
    auto queued = DrawPipeline::QueuedWrite::None;
    DrawPipeline* pipeline = nullptr;
    if (DrawPipeline::Active()) {
        auto& queue = DrawPipeline::Queue0();
        if (queue.Busy()) {
            pipeline = &queue;
            queued = queue.StoredBytes(address, bytes, known.size() == bytes ? known : std::span<std::byte>{});
        }
    }
    if (stats) noteQueuedRead(BlockSite::Query, pipeline, queued != DrawPipeline::QueuedWrite::None, validateLegacy(), address, bytes);
    if (queued == DrawPipeline::QueuedWrite::Stored && known.size() == bytes) {
        if (!knownValueVerify()) return Policy::KnownValue;
        pipeline->Drain(DrawPipeline::DrainReason::Capture);
        return Policy::VerifyKnownValue;
    }
    if (queued == DrawPipeline::QueuedWrite::Unknown) pipeline->Drain(DrawPipeline::DrainReason::Capture);
    std::uint64_t ValidateCounters::*reason = nullptr;
    PendingView pending;
    pending.Load();
    const OuterBlockSite outer(stats);
    const auto policy = Get().classifyPendingWrite(address, bytes, reason, pending, known);
    return queued == DrawPipeline::QueuedWrite::Stored && policy == Policy::None ? Policy::KnownValue : policy;
}

ShaderMemory::PendingWrite Driver::QueryPendingSpan(std::uint64_t address, std::size_t bytes, std::span<std::byte> known) {
    const bool stats = writeBlocksStats() && bytes != 0;
    DrawPipeline* pipeline = nullptr;
    bool queued = false;
    if (DrawPipeline::Active()) {
        auto& queue = DrawPipeline::Queue0();
        if (queue.Busy()) {
            pipeline = &queue;
            queued = queue.Overlaps(address, bytes);
        }
    }
    if (stats) noteQueuedRead(BlockSite::Span, pipeline, queued, validateLegacy(), address, bytes);
    if (queued) return ShaderMemory::PendingWrite::Sync;
    std::uint64_t ValidateCounters::*reason = nullptr;
    PendingView pending;
    pending.Load();
    const OuterBlockSite outer(stats);
    return Get().classifyPendingWrite(address, bytes, reason, pending, known);
}

}
