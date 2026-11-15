#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/CaptureTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ShaderMemory.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include <limits>
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "Optimization/ResourceMaterializer.hpp"
#include "Optimization/SrtWalker.hpp"
#include "CompiledVariant.hpp"
#include "DirectPath.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DirectSlots.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DirectHeap.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include <chrono>
#include <deque>
#include <optional>
#include <cstdio>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <unordered_map>
#include <array>
#include <memory>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <span>
#include <sstream>
#include <stdexcept>
#include <exception>
#include <typeinfo>

namespace AgcDriver::DriverDetail {

namespace {

bool directEvalRead(void*, std::uint64_t address, std::uint32_t* value) {
    const auto* pointer = reinterpret_cast<const std::byte*>(address);
    if (!GuestMemory::Accessible(pointer, sizeof(std::uint32_t), false)) return false;
    std::memcpy(value, pointer, sizeof(std::uint32_t));
    return true;
}

bool directEvalAccessible(void*, std::uint64_t address, std::uint64_t bytes) {
    return GuestMemory::Accessible(reinterpret_cast<const void*>(address), static_cast<std::size_t>(bytes), false);
}

bool directEvalReadWords(void*, std::uint64_t address, std::uint32_t* values, std::uint64_t count) {
    if (count == 0u) return true;
    const auto bytes = count * sizeof(std::uint32_t);
    const auto* pointer = reinterpret_cast<const std::byte*>(address);
    if (!GuestMemory::Accessible(pointer, static_cast<std::size_t>(bytes), false)) return false;
    std::memcpy(values, pointer, static_cast<std::size_t>(bytes));
    return true;
}

// Write stamps for the capture's table column memos (ResourceMaterializer ColumnMemos): a column
// whose guest range is unchanged since its stamp is taken from the memo instead of read again,
// with the old path's rule (ShaderMemory::Vouched). The direct capture reads live guest memory,
// so supplying the memo's words to it needs nothing. APS5_DIRECT_NO_TABLE_STAMPS=1 reads every column.
ShaderRecompiler::SrtWriteStamp directEvalStamp(void*, std::uint64_t address, std::uint64_t bytes) {
    if (bytes == 0u || address > std::numeric_limits<std::uint64_t>::max() - bytes) return {};
    const auto layout = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    const auto generation = GuestMemory::CollectWrites(address, static_cast<std::size_t>(bytes));
    if (generation == 0u) return {};
    return {generation, layout};
}

bool directEvalUnchanged(void*, std::uint64_t address, std::uint64_t bytes, const ShaderRecompiler::SrtWriteStamp& stamp) {
    return ShaderMemory::Vouched(nullptr, address, bytes, stamp);
}

bool directEvalSupply(void*, const std::shared_ptr<const ShaderRecompiler::SrtGuestWords>& words) {
    return words != nullptr;
}

void directEvalStamps(ShaderRecompiler::SrtRuntime& runtime) {
    static const bool disabled = std::getenv("APS5_DIRECT_NO_TABLE_STAMPS") != nullptr;
    runtime.readWords = &directEvalReadWords;
    if (disabled) return;
    runtime.stampWrites = &directEvalStamp;
    runtime.unchangedSince = &directEvalUnchanged;
    runtime.supplyWords = &directEvalSupply;
}

struct DirectStage {
    std::shared_ptr<const ShaderRecompiler::ResourceCapture> capture;
    std::shared_ptr<const ShaderRecompiler::RecompileResult> result;
};

// A result nobody else holds (a memo donor's patched copy, or this function's own table copy) is
// completed in place; a shared one (a result memo entry) is copied. Results are created non-const.
// APS5_DIRECT_COPY_RESULTS=1 always copies.
std::shared_ptr<ShaderRecompiler::RecompileResult> ownResult(std::shared_ptr<const ShaderRecompiler::RecompileResult>& base) {
    static const bool copyAlways = std::getenv("APS5_DIRECT_COPY_RESULTS") != nullptr;
    if (!copyAlways && base.use_count() == 1) return std::const_pointer_cast<ShaderRecompiler::RecompileResult>(std::move(base));
    return std::make_shared<ShaderRecompiler::RecompileResult>(*base);
}

std::optional<DirectStage> materializeDirectStage(const ShaderRecompiler::RecompileRequest& request, VulkanDevice* device) {
    struct Totals {
        std::uint64_t direct = 0, noHeap = 0;
        std::map<std::string, std::uint64_t> fallbacks;
        std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
        double us = 0;
        // The stage's parts: the capture, the materialized result, the table resolution.
        double captureUs = 0, materializeUs = 0, resolveUs = 0;
        std::uint64_t tableStages = 0;
    };
    static std::mutex mutex;
    static Totals totals;
    double captureUs = 0, materializeUs = 0, resolveUs = 0;
    bool tableStage = false;
    static std::map<std::pair<std::uint64_t, int>, std::shared_ptr<const ShaderRecompiler::SourceHandle>> handles;
    static std::deque<std::pair<std::chrono::steady_clock::time_point, std::vector<std::shared_ptr<void>>>> kept;
    const auto started = std::chrono::steady_clock::now();
    const auto report = [&](const char* fallback) {
        std::lock_guard lock(mutex);
        if (fallback != nullptr) ++totals.fallbacks[fallback];
        totals.us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count();
        totals.captureUs += captureUs;
        totals.materializeUs += materializeUs;
        totals.resolveUs += resolveUs;
        totals.tableStages += tableStage ? 1u : 0u;
        const auto now = std::chrono::steady_clock::now();
        while (!kept.empty() && now - kept.front().first > std::chrono::seconds(2)) kept.pop_front();
        if (now - totals.last < std::chrono::seconds(10)) return;
        const auto calls = totals.direct + totals.noHeap;
        std::string line = "[direct-draw] 10 s: " + std::to_string(totals.direct) + " direct stages with heap slots, " + std::to_string(totals.noHeap) + " direct without, avg " + std::to_string(calls != 0 ? totals.us / static_cast<double>(calls) : 0.0) + " us; fallbacks:";
        for (const auto& [text, count] : totals.fallbacks) line += " [" + std::to_string(count) + "x] " + text + ";";
        char parts[200];
        std::snprintf(parts, sizeof(parts), " parts per stage (us): capture %.2f, materialize %.2f, table resolve %.2f; %llu table stages", calls != 0 ? totals.captureUs / static_cast<double>(calls) : 0.0, calls != 0 ? totals.materializeUs / static_cast<double>(calls) : 0.0, calls != 0 ? totals.resolveUs / static_cast<double>(calls) : 0.0, static_cast<unsigned long long>(totals.tableStages));
        line += parts;
        std::fprintf(stderr, "%s\n", line.c_str());
        totals = {};
        totals.last = now;
    };
    if (device == nullptr) {
        report("no device");
        return std::nullopt;
    }
    auto direct = request;
    direct.target.directResources = true;
    direct.target.directFoldSrt = DirectFoldSrt();
    const auto key = std::pair{request.shader.codeAddress, static_cast<int>(request.shader.stage)};
    std::shared_ptr<const ShaderRecompiler::SourceHandle> handle;
    {
        std::lock_guard lock(mutex);
        if (const auto found = handles.find(key); found != handles.end()) handle = found->second;
    }
    try {
        if (handle == nullptr) {
            handle = ShaderRecompiler::PrepareShader(direct, nullptr);
            std::lock_guard lock(mutex);
            handles[key] = handle;
        }
        ShaderRecompiler::SrtRuntime runtime;
        runtime.userData = request.context.userData;
        runtime.shaderBase = request.shader.codeAddress;
        runtime.readMemory = &directEvalRead;
        runtime.readSpecializationMemory = &directEvalRead;
        runtime.accessible = &directEvalAccessible;
        directEvalStamps(runtime);
        const auto captureStarted = std::chrono::steady_clock::now();
        auto capture = ShaderRecompiler::CaptureResources(direct, runtime, *handle);
        captureUs = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - captureStarted).count();
        tableStage = !capture->snapshot.tables.tables.empty();
        if (!capture->snapshot.tables.tables.empty()) {
            // Debug aid: APS5_DIRECT_TABLE_PROBE=1 reports the image tables that keep stages off
            // the direct path: per 10 s, stages, columns per stage, records, bytes, and how many
            // distinct column contents (base plus descriptor words) were seen.
            static const bool tableProbe = std::getenv("APS5_DIRECT_TABLE_PROBE") != nullptr;
            if (tableProbe) {
                struct TableTotals {
                    std::uint64_t stages = 0, columns = 0, records = 0, nested = 0, bytes = 0, maxRecords = 0, keys = 0, words = 0, readRanges = 0, codes = 0, pointers = 0, outside = 0;
                    std::set<std::uint64_t> contents, bases;
                    std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
                };
                static std::mutex tableMutex;
                static TableTotals table;
                std::lock_guard tableLock(tableMutex);
                ++table.stages;
                table.nested += capture->snapshot.tables.nested.size();
                for (const auto& column : capture->snapshot.tables.tables) {
                    ++table.columns;
                    table.records += column.records;
                    table.bytes += column.size;
                    table.maxRecords = std::max<std::uint64_t>(table.maxRecords, column.records);
                    std::uint64_t hash = 0xcbf29ce484222325ull ^ column.base;
                    for (const auto& word : column.words) {
                        for (std::uint32_t at = 0; at < word.dwordCount; ++at) hash = (hash ^ word.dwords[at]) * 0x100000001b3ull;
                    }
                    table.contents.insert(hash);
                    table.keys += column.keys;
                    table.words += column.words.size();
                    table.readRanges += column.readRanges.size();
                    table.codes += column.codes.size();
                    table.pointers += column.pointers.size();
                    if (column.outside) ++table.outside;
                    // Once per base: how far the table's memory is populated (32-byte blocks with
                    // any non-zero dword, up to 64 MiB or the first inaccessible page).
                    static std::set<std::uint64_t> scanned;
                    if (scanned.insert(column.base).second) {
                        std::uint64_t accessible = 0, nonZero = 0, lastNonZero = 0;
                        for (std::uint64_t offset = 0; offset < (64ull << 20u); offset += 32u) {
                            const auto* at = reinterpret_cast<const std::uint32_t*>(column.base + offset);
                            if ((offset & 4095u) == 0u && !GuestMemory::Accessible(at, 4096u)) break;
                            accessible = offset + 32u;
                            bool any = false;
                            for (int dword = 0; dword < 8; ++dword) any = any || at[dword] != 0u;
                            if (any) {
                                ++nonZero;
                                lastNonZero = offset;
                            }
                        }
                        std::fprintf(stderr, "[direct-tables] base 0x%llx: %llu bytes accessible, %llu non-zero 32-byte blocks, last at +0x%llx\n", static_cast<unsigned long long>(column.base), static_cast<unsigned long long>(accessible), static_cast<unsigned long long>(nonZero), static_cast<unsigned long long>(lastNonZero));
                        // Where the column's textures live: decoded guest ranges merged with gaps
                        // under 2 MiB, so one collect could cover them.
                        std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
                        std::uint64_t decoded = 0, failed = 0, nulls = 0;
                        for (const auto& word : column.words) {
                            if (word.dwordCount < 8u) continue;
                            const std::span<const std::uint32_t> view(word.dwords.data(), 8u);
                            if (Graphics::IsNullSampledTextureResource(view)) {
                                ++nulls;
                                continue;
                            }
                            try {
                                const auto resource = Graphics::DecodeTextureResource(view);
                                const auto bytes = Graphics::DescribeSurface(resource).guestBytes;
                                if (resource.baseAddress != 0u && bytes != 0u) ranges.emplace_back(resource.baseAddress, resource.baseAddress + bytes);
                                ++decoded;
                            } catch (const std::exception&) {
                                ++failed;
                            }
                        }
                        std::sort(ranges.begin(), ranges.end());
                        std::vector<std::pair<std::uint64_t, std::uint64_t>> runs;
                        std::uint64_t textureBytes = 0;
                        for (const auto& [begin, end] : ranges) {
                            textureBytes += end - begin;
                            if (!runs.empty() && begin <= runs.back().second + (2ull << 20u)) runs.back().second = std::max(runs.back().second, end);
                            else runs.emplace_back(begin, end);
                        }
                        std::uint64_t runBytes = 0;
                        for (const auto& [begin, end] : runs) runBytes += end - begin;
                        std::fprintf(stderr, "[direct-tables] base 0x%llx textures: %llu decoded, %llu null, %llu failed; %.1f MiB of textures in %zu runs covering %.1f MiB (0x%llx..0x%llx)\n", static_cast<unsigned long long>(column.base), static_cast<unsigned long long>(decoded), static_cast<unsigned long long>(nulls), static_cast<unsigned long long>(failed), static_cast<double>(textureBytes) / (1 << 20), runs.size(), static_cast<double>(runBytes) / (1 << 20), static_cast<unsigned long long>(runs.empty() ? 0 : runs.front().first), static_cast<unsigned long long>(runs.empty() ? 0 : runs.back().second));
                    }
                    table.bases.insert(column.base);
                }
                if (const auto now = std::chrono::steady_clock::now(); now - table.last >= std::chrono::seconds(10)) {
                    std::fprintf(stderr, "[direct-tables] 10 s: %llu stages, %.2f columns per stage, %.1f records per column (max %llu), %.0f bytes per column, %llu nested; %zu distinct column contents over %zu bases\n", static_cast<unsigned long long>(table.stages), table.stages ? static_cast<double>(table.columns) / table.stages : 0.0, table.columns ? static_cast<double>(table.records) / table.columns : 0.0, static_cast<unsigned long long>(table.maxRecords), table.columns ? static_cast<double>(table.bytes) / table.columns : 0.0, static_cast<unsigned long long>(table.nested), table.contents.size(), table.bases.size());
                    if (table.columns != 0) std::fprintf(stderr, "[direct-tables] per column: %.2f keys, %.2f captured words, %.2f read ranges, %.2f codes, %.2f pointers; %llu outside\n", static_cast<double>(table.keys) / table.columns, static_cast<double>(table.words) / table.columns, static_cast<double>(table.readRanges) / table.columns, static_cast<double>(table.codes) / table.columns, static_cast<double>(table.pointers) / table.columns, static_cast<unsigned long long>(table.outside));
                    table = {};
                }
            }
            // Table stages go direct by default: their elements become heap slots at record time
            // (ResolveDirectStages). With APS5_NO_DIRECT=1 they keep the existing path.
            static const bool directTables = ShaderRecompiler::DirectPath::Enabled();
            if (!directTables || !capture->snapshot.tables.nested.empty()) {
                report(directTables ? "nested image tables" : "image tables");
                return std::nullopt;
            }
        }
        const auto materializeStarted = std::chrono::steady_clock::now();
        auto base = ShaderRecompiler::MaterializeShader(direct, *capture, *handle);
        materializeUs = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - materializeStarted).count();
        if (!capture->snapshot.tables.tables.empty()) {
            const auto& info = handle->artifact->info.info;
            const auto resolveStarted = std::chrono::steady_clock::now();
            auto resolvedTables = ShaderRecompiler::ResourceMaterializer::ResolveImageTablesShared(info, capture->snapshot, handle->artifact->artifact.variantId);
            resolveUs = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - resolveStarted).count();
            // The same materialized result with the same resolved tables gives the same stage: reuse
            // it instead of copying the result (and its table map) again.
            struct TableStageMemo {
                std::weak_ptr<const ShaderRecompiler::RecompileResult> base;
                std::weak_ptr<const ShaderRecompiler::ResolvedImageTables> tables;
                std::shared_ptr<const ShaderRecompiler::RecompileResult> stage;
            };
            thread_local std::array<TableStageMemo, 64> tableStages;
            auto& slot = tableStages[(reinterpret_cast<std::uintptr_t>(base.get()) >> 4u) % tableStages.size()];
            // Only table-only stages: slot requests come from this capture's image words, which the
            // materialized result does not pin.
            const bool tableOnly = base->directSlotCount == 0u;
            if (tableOnly && slot.stage != nullptr && slot.base.lock() == base && slot.tables.lock() == resolvedTables) {
                base = slot.stage;
                {
                    std::lock_guard lock(mutex);
                    ++totals.direct;
                }
                report(nullptr);
                return DirectStage{std::move(capture), std::move(base)};
            }
            const std::weak_ptr<const ShaderRecompiler::RecompileResult> materialized = base;
            auto withTables = ownResult(base);
            withTables->directTables = std::move(resolvedTables);
            if (std::ranges::any_of(withTables->directTables->elements, [](const ShaderRecompiler::ImageTableElement& element) { return ShaderRecompiler::ImageBindingResourceClass(element.kind) != ShaderRecompiler::ImageResourceClass::Sampled; })) {
                report("storage image table");
                return std::nullopt;
            }
            withTables->directSamplerTables.assign(capture->snapshot.tables.tables.size(), 0u);
            for (const auto& sampler : info.samplers) {
                if (sampler.table != ShaderRecompiler::NoTable && sampler.table < withTables->directSamplerTables.size()) withTables->directSamplerTables[sampler.table] = 1u;
            }
            base = std::move(withTables);
            if (tableOnly) {
                slot = {materialized, base->directTables, base};
                {
                    std::lock_guard lock(mutex);
                    ++totals.direct;
                }
                report(nullptr);
                return DirectStage{std::move(capture), std::move(base)};
            }
        }
        if (base->directSlotCount == 0u && base->directTables == nullptr) {
            {
                std::lock_guard lock(mutex);
                ++totals.noHeap;
            }
            report(nullptr);
            return DirectStage{std::move(capture), std::move(base)};
        }
        std::vector<std::uint32_t> requests;
        DirectSlotStatistics statistics;
        std::string reason;
        if (base->directSlotCount != 0u && !BuildDirectSlotRequests(handle->artifact->info, capture->snapshot, requests, statistics, reason)) {
            // A plain local: report() copies the text. (A thread_local string is destroyed at thread
            // exit after its storage was freed, which corrupts the heap.)
            const std::string shortReason = reason.substr(0, 80);
            report(shortReason.c_str());
            return std::nullopt;
        }
        bool pushed = true;
        for (const auto& binding : base->bindings) pushed = pushed && binding.role != ShaderRecompiler::DescriptorRole::ShaderData;
        if (pushed && base->pushConstants.size() < (static_cast<std::size_t>(base->directSlotDword) + base->directSlotCount) * sizeof(std::uint32_t)) {
            report("slot words outside the push data");
            return std::nullopt;
        }
        auto patched = ownResult(base);
        patched->directSlotRequests = std::move(requests);
        {
            std::lock_guard lock(mutex);
            ++totals.direct;
        }
        report(nullptr);
        return DirectStage{std::move(capture), std::move(patched)};
    } catch (const std::exception& error) {
        std::string lastError = std::string("exception: ") + error.what();
        if (lastError.size() > 100) lastError.resize(100);
        report(lastError.c_str());
        return std::nullopt;
    }
}

void probeDirectEvaluation(const ShaderRecompiler::RecompileRequest& request, VulkanDevice* device) {
    struct Totals {
        std::uint64_t calls = 0, failed = 0, images = 0, samplers = 0, tables = 0, prepares = 0;
        std::uint64_t lookups = 0, lookupFailures = 0;
        double us = 0, lookupUs = 0;
        std::map<std::string, std::uint64_t> reasons;
        std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
    };
    static std::mutex mutex;
    static Totals totals;
    static std::map<std::pair<std::uint64_t, int>, std::shared_ptr<const ShaderRecompiler::SourceHandle>> handles;
    auto direct = request;
    direct.target.directResources = true;
    const auto key = std::pair{request.shader.codeAddress, static_cast<int>(request.shader.stage)};
    std::shared_ptr<const ShaderRecompiler::SourceHandle> handle;
    bool prepared = false;
    std::string failure;
    {
        std::lock_guard lock(mutex);
        if (const auto found = handles.find(key); found != handles.end()) handle = found->second;
    }
    if (handle == nullptr) {
        try {
            handle = ShaderRecompiler::PrepareShader(direct, nullptr);
            prepared = true;
            std::lock_guard lock(mutex);
            handles[key] = handle;
        } catch (const std::exception& error) {
            failure = std::string("prepare: ") + error.what();
        }
    }
    double us = 0, lookupUs = 0;
    std::size_t images = 0, samplers = 0, tables = 0, lookups = 0, lookupFailures = 0;
    static const bool textureProbe = std::getenv("APS5_DIRECT_TEX_PROBE") != nullptr;
    if (handle != nullptr) {
        ShaderRecompiler::SrtRuntime runtime;
        runtime.userData = request.context.userData;
        runtime.shaderBase = request.shader.codeAddress;
        runtime.readMemory = &directEvalRead;
        runtime.readSpecializationMemory = &directEvalRead;
        runtime.accessible = &directEvalAccessible;
        const auto started = std::chrono::steady_clock::now();
        try {
            const auto capture = ShaderRecompiler::CaptureResources(direct, runtime, *handle);
            images = capture->snapshot.images.size();
            samplers = capture->snapshot.samplers.size();
            tables = capture->snapshot.tables.tables.size();
            us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count();
            static const bool materializeProbe = std::getenv("APS5_DIRECT_MAT_PROBE") != nullptr;
            static std::atomic<std::uint64_t> materialized{0}, materializeFailed{0};
            static std::atomic<double> materializeUs{0};
            if (materializeProbe && tables == 0u) {
                const auto materializeStarted = std::chrono::steady_clock::now();
                static const bool fullMaterialize = std::getenv("APS5_DIRECT_FULL_PROBE") != nullptr;
                try {
                    if (fullMaterialize) static_cast<void>(ShaderRecompiler::MaterializeShader(direct, *capture, *handle));
                    else static_cast<void>(ShaderRecompiler::DirectStageSpecialization(direct, *capture, *handle));
                    ++materialized;
                } catch (const std::exception& error) {
                    ++materializeFailed;
                    static std::mutex reasonMutex;
                    static std::map<std::string, std::uint64_t> reasons;
                    std::string text = error.what();
                    if (text.size() > 160) text.resize(160);
                    std::lock_guard reasonLock(reasonMutex);
                    if (++reasons[text] == 1u && reasons.size() <= 12u) std::fprintf(stderr, "[direct-eval] materialize failure: %s\n", text.c_str());
                }
                materializeUs = materializeUs.load() + std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - materializeStarted).count();
                if ((materialized.load() + materializeFailed.load()) % 100000u == 0u) std::fprintf(stderr, "[direct-eval] materialize: %llu done, %llu failed, avg %.2f us\n", static_cast<unsigned long long>(materialized.load()), static_cast<unsigned long long>(materializeFailed.load()), materialized.load() != 0 ? materializeUs.load() / static_cast<double>(materialized.load()) : 0.0);
            }
            if (textureProbe && device != nullptr && !capture->snapshot.images.empty()) {
                const auto context = device->graphicsContext();
                std::lock_guard gpu(GuestMemory::GpuMutex());
                const auto lookupStarted = std::chrono::steady_clock::now();
                static std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint64_t> shapes;
                static std::uint64_t shapeReports = 0;
                struct MemoEntry {
                    std::array<std::uint32_t, 8> words;
                    std::uint64_t epoch;
                    std::shared_ptr<Graphics::Texture> texture;
                };
                static std::unordered_map<std::uint64_t, MemoEntry> memo;
                static std::uint64_t memoHits = 0, memoMisses = 0, memoReports = 0;
                static const bool useMemo = std::getenv("APS5_DIRECT_TEX_MEMO") != nullptr;
                const auto epoch = GuestMemory::ThreadCollectEpoch();
                for (const auto& image : capture->snapshot.images) {
                    ++shapes[{image.dwordCount, image.availableDwordMask}];
                    if (image.dwordCount < 8u) continue;
                    ++lookups;
                    std::uint64_t hash = 1469598103934665603ull;
                    for (const auto word : image.dwords) hash = (hash ^ word) * 1099511628211ull;
                    if (useMemo) {
                        if (const auto found = memo.find(hash); found != memo.end() && found->second.epoch == epoch && found->second.words == image.dwords) {
                            ++memoHits;
                            continue;
                        }
                        ++memoMisses;
                    }
                    try {
                        auto texture = Graphics::CachedSampledTexture(context, std::span<const std::uint32_t>(image.dwords.data(), 8u), VK_IMAGE_VIEW_TYPE_2D);
                        if (useMemo) memo[hash] = {image.dwords, epoch, std::move(texture)};
                    } catch (const std::exception&) {
                        ++lookupFailures;
                    }
                }
                if (useMemo && ++memoReports % 100000u == 1u) {
                    std::fprintf(stderr, "[direct-eval] texture memo: %llu hits, %llu misses, %zu entries\n", static_cast<unsigned long long>(memoHits), static_cast<unsigned long long>(memoMisses), memo.size());
                    if (memo.size() > 65536u) memo.clear();
                }
                lookupUs = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - lookupStarted).count();
                if (++shapeReports % 200000u == 1u) {
                    std::string text = "[direct-eval] image shapes (dwords/mask: count):";
                    for (const auto& [shape, count] : shapes) text += " " + std::to_string(shape.first) + "/0x" + std::to_string(shape.second) + ": " + std::to_string(count);
                    std::fprintf(stderr, "%s\n", text.c_str());
                }
            }
        } catch (const std::exception& error) {
            failure = std::string("capture: ") + error.what();
        }
    }
    std::lock_guard lock(mutex);
    ++totals.calls;
    totals.prepares += prepared ? 1u : 0u;
    if (!failure.empty()) {
        ++totals.failed;
        if (failure.size() > 100) failure.resize(100);
        ++totals.reasons[failure];
    } else {
        totals.us += us;
        totals.images += images;
        totals.samplers += samplers;
        totals.tables += tables;
        totals.lookups += lookups;
        totals.lookupFailures += lookupFailures;
        totals.lookupUs += lookupUs;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - totals.last < std::chrono::seconds(10)) return;
    totals.last = now;
    const auto ok = totals.calls - totals.failed;
    std::string line = "[direct-eval] 10 s: " + std::to_string(totals.calls) + " stage evaluations, " + std::to_string(totals.failed) + " failed, " + std::to_string(totals.prepares) + " prepared; avg " + std::to_string(ok != 0 ? totals.us / static_cast<double>(ok) : 0.0) + " us; images " + std::to_string(totals.images) + ", samplers " + std::to_string(totals.samplers) + ", table columns " + std::to_string(totals.tables) + "; texture lookups " + std::to_string(totals.lookups) + " (" + std::to_string(totals.lookupFailures) + " failed), avg " + std::to_string(totals.lookups != 0 ? totals.lookupUs / static_cast<double>(totals.lookups) : 0.0) + " us each;";
    std::vector<std::pair<std::uint64_t, std::string>> top;
    for (const auto& [text, count] : totals.reasons) top.emplace_back(count, text);
    std::sort(top.rbegin(), top.rend());
    for (std::size_t index = 0; index < top.size() && index < 3; ++index) line += " [" + std::to_string(top[index].first) + "x] " + top[index].second + ";";
    std::fprintf(stderr, "%s\n", line.c_str());
    const auto last = totals.last;
    totals = {};
    totals.last = last;
}

void TraceMaterializationFailure(std::uint64_t token, const Submission& submission, std::size_t packetOffset, std::size_t stage, const char* phase, const std::exception_ptr& failure) {
    if (token == 0) return;
    const char* type = "unknown";
    std::string reason = "unknown exception";
    try {
        if (failure != nullptr) std::rethrow_exception(failure);
    } catch (const std::exception& error) {
        type = typeid(error).name();
        reason = error.what();
    } catch (...) {
    }
    try {
        AgcDriver::CaptureTrace::Log("draw-materialization-failed token=%llu submission=%llu queue=%x offset=%zu stage=%zu phase=%s exception=%s reason=%.256s",
                static_cast<unsigned long long>(token), static_cast<unsigned long long>(submission.serial), submission.queue, packetOffset, stage, phase, type, reason.c_str());
    } catch (...) {
    }
}

void TraceShaderRequestFailure(std::uint64_t token, const Submission& submission, std::size_t packetOffset, const DrawProgram& program, const std::exception_ptr& failure) noexcept {
    if (token == 0) return;
    try {
        std::string reason = "unknown exception";
        const char* type = "unknown";
        try {
            if (failure != nullptr) std::rethrow_exception(failure);
        } catch (const std::exception& error) {
            type = typeid(error).name();
            reason = error.what();
        } catch (...) {
        }
        AgcDriver::CaptureTrace::LogBestEffort("draw-shader-request-failed token=%llu submission=%llu queue=%x offset=%zu snapshot=0x%llx code=%llx codeOffset=%zu stage=%u type=%u reason=%.256s",
                static_cast<unsigned long long>(token), static_cast<unsigned long long>(submission.serial), submission.queue, packetOffset,
                static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(program.snapshot.get())),
                static_cast<unsigned long long>(program.snapshot->codeAddress), program.codeOffset,
                static_cast<unsigned>(program.binary.stage), static_cast<unsigned>(program.snapshot->type), reason.c_str());
    } catch (...) {
    }
}

}

std::shared_ptr<const ShaderRecompiler::RecompileResult> Driver::materializeDrawStage(std::size_t i, std::uint32_t pushOffset, const QueueState& queue, const Submission& submission, std::size_t packetOffset, std::uint64_t traceToken, const std::vector<DrawProgram>& programs, const Graphics::State& graphics, const ShaderRecompiler::ShaderPixelStageInfo& pixel, const std::vector<std::optional<ShaderRecompiler::ShaderVertexStageInfo>>& vertexInfos, std::vector<ShaderRecompiler::MemoryRegion>& memory, const std::vector<ShaderRecompiler::LinkedProgram>& linked, const Pm4::DrawParameters& drawParameters, const std::shared_ptr<VulkanDevice>& localDevice, ShaderMemory& shaderMemory, std::vector<StageCapture>& stageCaptures, std::vector<bool>& recompiled, bool drawHit, const std::vector<std::shared_ptr<DispatchVariant>>& matched, const std::vector<std::vector<ShaderRecompiler::MemoryRegion>>& matchedRegions, bool profile, std::uint64_t dumpTarget, std::uint64_t dumpSlot1, std::uint64_t& captures, DrawPhaseTiming& phaseTiming, std::array<double, DrawDriverPhaseCount>& phaseMs, std::string& rejected, const std::optional<ShaderRecompiler::MeshBarycentricLayout>& meshBarycentrics) try {
    PerformanceTimer timing("Shader.DrawStage");
    using Stage = ShaderRecompiler::ShaderStage;
    phaseTiming.Phase(DrawRowVectors);
    const auto& program = programs[i];
    const auto waveSize = Graphics::StageWaveSize(graphics.stages, program.binary.stage);
    ShaderRecompiler::RecompileRequest request{
        program.binary,
        {waveSize, program.firstUserSgpr, program.userData, std::nullopt, program.binary.stage == Stage::Fragment ? std::optional(pixel) : std::nullopt, vertexInfos[i], memory, RegisteredFloatMode(*program.snapshot)},
        localDevice->Target(),
        {0, 0, pushOffset, Graphics::StagePushLimit(graphics.stages, pushOffset, graphics.depthRemap.mode != 0u) - pushOffset % Graphics::PipelinePushSlotBytes},
        ShaderRecompiler::GraphicsCompileContext{program.firstUserSgpr, linked, Graphics::StageMeshConfiguration(graphics.stages, program.binary.stage), graphics.stages.tessellation, {drawParameters.indexAddress, drawParameters.indexCount, drawParameters.indexSize, drawParameters.instanceCount}, program.binary.stage == Stage::Mesh ? meshBarycentrics : std::nullopt}
    };
    if (graphics.depthRemap.mode != 0u && pushOffset < Graphics::PipelinePushSlotBytes) {
        request.layout.pushConstantSizeBytes = std::min(request.layout.pushConstantSizeBytes, ShaderRecompiler::DepthRemapPushOffsetBytes - pushOffset);
    }
    static const bool directEvalProbe = std::getenv("APS5_DIRECT_EVAL_PROBE") != nullptr;
    if (directEvalProbe) probeDirectEvaluation(request, localDevice.get());
    static const bool directDraw = ShaderRecompiler::DirectPath::Enabled();
    const bool plainDraw = !graphics.rectList && !graphics.stages.tessellation && !graphics.stages.mesh && (program.binary.stage == Stage::Vertex || program.binary.stage == Stage::Fragment);
    if (directDraw && plainDraw) {
        if (auto direct = materializeDirectStage(request, localDevice.get())) {
            auto& stageCapture = stageCaptures[i];
            stageCapture.forgetSerial = GuestMemory::ForgetSerial();
            stageCapture.pushOffset = pushOffset;
            stageCapture.capture = direct->capture;
            stageCapture.compiled = direct->result;
            stageCapture.userDataMask = ~std::uint64_t{0};
            stageCapture.registeredViews = std::make_shared<const std::vector<ShaderRecompiler::RegisteredImageView>>();
            recompiled[i] = true;
            // Debug aid: APS5_DIRECT_DUMP_SPIRV=<dir> writes each direct draw stage's SPIR-V once as
            // draw_<code address>.spv (the first module seen for that code).
            if (static const char* dumpDir = std::getenv("APS5_DIRECT_DUMP_SPIRV"); dumpDir != nullptr) {
                static std::mutex dumpMutex;
                static std::set<std::uint64_t> dumped;
                std::lock_guard dumpLock(dumpMutex);
                if (dumped.insert(program.binary.codeAddress).second) {
                    char path[512];
                    std::snprintf(path, sizeof(path), "%s/draw_%llx.spv", dumpDir, static_cast<unsigned long long>(program.binary.codeAddress));
                    if (std::FILE* file = std::fopen(path, "wb")) {
                        std::fwrite(direct->result->spirv.data(), sizeof(std::uint32_t), direct->result->spirv.size(), file);
                        std::fclose(file);
                    }
                }
            }
            return stageCapture.compiled;
        }
    }
    const auto waitedBefore = traceCapSync() || profile ? Graphics::Recorder::ThreadWaitedMs() : 0.0;
    auto invocation = [&] {
        try {
            return InvocationFor(*program.snapshot, program.codeOffset, request);
        } catch (...) {
            TraceShaderRequestFailure(traceToken, submission, packetOffset, program, std::current_exception());
            throw;
        }
    }();
    timing.Mark("prepared_invocation");
    auto& stageCapture = stageCaptures[i];
    stageCapture.forgetSerial = GuestMemory::ForgetSerial();
    stageCapture.pushOffset = pushOffset;
    const auto named = [&](const std::exception& error, std::span<const ShaderRecompiler::MemoryRegion> regions) {
        static const bool dumpShaders = std::getenv("APS5_DUMP_SHADERS") != nullptr;
        std::string dump;
        if (dumpShaders) {
            try {
                auto failed = request;
                failed.context.memory = regions;
                dump = dumpRequest(program.binary.codeAddress, failed, "_failed");
            } catch (...) {}
        }
        std::ostringstream where;
        where << "shader 0x" << std::hex << program.binary.codeAddress << " stage " << std::dec << static_cast<unsigned>(program.binary.stage);
        if (!dump.empty()) where << " (" << dump << ")";
        where << ": ";
        return std::runtime_error(where.str() + error.what());
    };
    const auto capture = [&] {
        const SampledReadScope sampling(evidenceReads);
        try {
            return CaptureInvocation(shaderMemory, *program.snapshot, program.codeOffset, request, invocation);
        } catch (const std::exception& error) {
            throw named(error, shaderMemory.Regions());
        }
    }();

    stageCapture.regions = shaderMemory.TakeRecentRegions();
    stageCapture.supplied = shaderMemory.SuppliedBlocks();
    stageCapture.userDataUse = capture->userDataUse;
    stageCapture.registeredViews = std::make_shared<const std::vector<ShaderRecompiler::RegisteredImageView>>(capture->snapshot.registeredImages);
    recompiled[i] = true;
    memory = shaderMemory.Regions();

    for (std::size_t j = 0; j < programs.size(); ++j) {
        if (matched[j] != nullptr && !recompiled[j] && (drawHit || j < i)) memory.insert(memory.end(), matchedRegions[j].begin(), matchedRegions[j].end());
    }
    request.context.memory = memory;
    static const bool dumpGraphics = std::getenv("APS5_DUMP_GRAPHICS_SHADERS") != nullptr;
    if (dumpGraphics) static_cast<void>(dumpRequest(program.binary.codeAddress, request));
    if (traceCapSync()) traceCapture("draw-capture", program.binary.codeAddress, submission.queue, memory, Graphics::Recorder::ThreadWaitedMs() - waitedBefore);
    if (profile) {
        ++captures;
        phaseTiming.Phase(DrawRowCapture);

            const auto waited = std::min(Graphics::Recorder::ThreadWaitedMs() - waitedBefore, phaseMs[DrawRowCapture]);
            phaseMs[DrawRowCapture] -= waited;
            phaseMs[DrawRowCaptureHookWaits] += waited;
        }
        if (dumpTarget != 0) {

        const auto slot0 = (static_cast<std::uint64_t>(readRegister(queue.context, 0x390)) << 40u) | (static_cast<std::uint64_t>(readRegister(queue.context, 0x318)) << 8u);
        if ((graphics.hasColorTarget && graphics.color.address == dumpTarget) || slot0 == dumpTarget || program.binary.codeAddress == dumpTarget) static_cast<void>(dumpRequest(program.binary.codeAddress, request));
        }
        if (dumpSlot1 != 0) {
        const auto value = [&](std::uint32_t offset) -> std::uint64_t { const auto it = queue.context.find(offset); return it == queue.context.end() ? 0u : it->second; };
        const auto slot1 = (value(0x391) << 40u) | (value(0x327) << 8u);
        if (slot1 == dumpSlot1) {
            static_cast<void>(dumpRequest(program.binary.codeAddress, request));
            if (std::FILE* file = std::fopen("draw_slot1.regs", "w")) {
                for (const auto& [offset, value] : queue.context) std::fprintf(file, "context %x %08x\n", offset, value);
                for (const auto& [offset, value] : queue.userConfig) std::fprintf(file, "uconfig %x %08x\n", offset, value);
                for (const auto& [offset, value] : queue.shader) std::fprintf(file, "shader %x %08x\n", offset, value);
                std::fclose(file);
            }
        }
        }

        phaseTiming.Phase(DrawRowCapture);

    timing.Mark("capture_resources");
    try {
        stageCapture.compiled = invocation.Materialize(*capture);
        stageCapture.capture = capture;
    } catch (const std::exception& error) {
        TraceMaterializationFailure(traceToken, submission, packetOffset, i, "capture-or-materialize", std::current_exception());
        throw named(error, request.context.memory);
    }
    timing.Mark("materialize");
    const auto walked = capture->userDataUse.asValue | capture->userDataUse.asBase;
    auto mask = capture->plan != nullptr && capture->plan->srtPlanComplete ? walked | stageCapture.compiled->userDataRegisterReads : ~std::uint64_t{0};
    if (vertexInfos[i]) {
        for (const auto reg : {static_cast<std::uint32_t>(vertexInfos[i]->fetchAttribReg), static_cast<std::uint32_t>(vertexInfos[i]->fetchBufferReg)}) {
            mask |= reg + 1u < 64u ? std::uint64_t{3} << reg : ~std::uint64_t{0};
        }
    }
    if (program.snapshot && program.snapshot->userDataTailRegister) {
        const auto reg = *program.snapshot->userDataTailRegister;
        if (reg >= program.firstUserSgpr) {
            const auto slot = reg - program.firstUserSgpr;
            mask |= slot + 1u < 64u ? std::uint64_t{3} << slot : ~std::uint64_t{0};
        } else {
            mask = ~std::uint64_t{0};
        }
    }
    if (program.snapshot && StageLinksUserDataCallees(*program.snapshot, program.codeOffset)) {
        mask = ~std::uint64_t{0};
    }
    stageCapture.userDataMask = mask;
    phaseTiming.Phase(DrawRowRecompile);
    return stageCapture.compiled;
} catch (const std::exception& error) {
    if (!SkipRefusedDraws()) throw;
    throw RefusedDraw(programs[i].binary.codeAddress, error.what());
}

void Driver::recordStageDataPositions(DispatchVariant& variant, const ShaderRecompiler::ResourceCapture& capture, const std::vector<Graphics::DecodeRead>& decodeReads) {
    const auto& compiled = *variant.compiled;
    for (std::size_t b = 0; b < compiled.bindings.size(); ++b) {
        if (compiled.bindings[b].role != ShaderRecompiler::DescriptorRole::FlattenedSrt) continue;
        variant.flatBinding = static_cast<std::uint32_t>(b);
        static_cast<void>(DataWordPositions(variant.runs, capture.readTrace.leaves, capture.readTrace.otherReads, variant.words, compiled.bindings[b].guestDescriptor, variant.dataPositions, variant.dataSlots));
        break;
    }
    if (variant.dataPositions.empty() || decodeReads.empty()) return;
    std::vector<bool> decoded(variant.words.size(), false);
    std::size_t offset = 0;
    for (const auto& [begin, end] : variant.runs) {
        for (const auto& read : decodeReads) {
            const auto from = std::max(begin, read.address);
            const auto to = std::min(end, read.address + read.size);
            if (from >= to) continue;
            for (auto word = (from - begin) / sizeof(std::uint32_t); word < (to - begin + sizeof(std::uint32_t) - 1) / sizeof(std::uint32_t); ++word) decoded[offset + static_cast<std::size_t>(word)] = true;
        }
        offset += static_cast<std::size_t>((end - begin) / sizeof(std::uint32_t));
    }
    std::size_t kept = 0;
    for (std::size_t k = 0; k < variant.dataPositions.size(); ++k) {
        if (decoded[variant.dataPositions[k]]) continue;
        variant.dataPositions[kept] = variant.dataPositions[k];
        variant.dataSlots[kept] = variant.dataSlots[k];
        ++kept;
    }
    variant.dataPositions.resize(kept);
    variant.dataSlots.resize(kept);
}

void Driver::cacheDrawStages(bool useDrawEntries, bool drawHit, const Pm4::DrawParameters& drawParameters, const std::optional<Graphics::IndirectDrawPath>& indirectCpu, const std::vector<DrawProgram>& programs, const std::vector<StageCapture>& stageCaptures, const std::vector<std::optional<ShaderRecompiler::ShaderVertexStageInfo>>& vertexInfos, const std::vector<std::vector<Graphics::DecodeRead>>& decodeReads, bool verifyHit, const std::vector<std::shared_ptr<DispatchVariant>>& matched, const std::vector<StageDataHit>& stageData, std::vector<std::shared_ptr<DispatchVariant>>& fresh, std::uint64_t drawKey, bool registerKey, const std::shared_ptr<const DrawDecode>& decode, DrawPhaseTiming& phaseTiming, const std::vector<bool>& reused, const std::shared_ptr<DrawEntry>& entry, std::uint64_t shapeKey, bool shapeNeedsEntry) {
    const bool cacheable = useDrawEntries && !drawHit && (drawCacheIndirect() || !(drawParameters.indirect && indirectCpu));
    const bool admitted = cacheable && (verifyHit || admitDrawKey(drawKey, registerKey && drawDataHits() ? shapeKey : 0));
    const bool replaceShape = cacheable && !admitted && shapeNeedsEntry;
    if (admitted || replaceShape) {
        phaseTiming.Phase(DrawRowVectors);
        std::uint64_t unstable = 0, mismatches = 0, differingSameRuns = 0, differingWords = 0, differingRunsChanged = 0;
        for (std::size_t i = 0; i < programs.size(); ++i) {
            const auto& stageCapture = stageCaptures[i];
            if (reused[i]) {
                fresh[i] = matched[i];
                continue;
            }
            if (stageCapture.compiled == nullptr || !CacheableResult(*stageCapture.compiled)) continue;
            auto variant = std::make_shared<DispatchVariant>();
            variant->compiled = stageCapture.compiled;
            variant->registeredViews = stageCapture.registeredViews;
            if (programs[i].binary.stage == ShaderRecompiler::ShaderStage::Mesh) {
                std::copy_n(programs[i].userData.begin() + ShaderRecompiler::MeshIndexBufferUserWord, variant->meshIndexDescriptor.size(), variant->meshIndexDescriptor.begin());
            }
            variant->shader = programs[i].snapshot;
            variant->forgetSerial = stageCapture.forgetSerial;
            variant->pushOffset = stageCapture.pushOffset;
            variant->userDataMask = stageCapture.userDataMask;
            if (vertexInfos[i]) variant->vertexInfo = std::make_shared<const ShaderRecompiler::ShaderVertexStageInfo>(*vertexInfos[i]);

            std::vector<ShaderRecompiler::MemoryRegion> regions(stageCapture.regions.begin(), stageCapture.regions.end());
            for (const auto& read : decodeReads[i]) regions.push_back({read.address, read.Bytes()});
            std::stable_sort(regions.begin(), regions.end(), [](const ShaderRecompiler::MemoryRegion& a, const ShaderRecompiler::MemoryRegion& b) { return a.guestAddress < b.guestAddress; });
            const bool relocatable = decodeReads[i].empty() && (!vertexInfos[i] || vertexInfos[i]->resourcesNum == 0);
            tagVariantRuns(*variant, regions, stageCapture.userDataUse, relocatable, programs[i].hiddenUserWords, stageCapture.supplied);
            if (relocatable) tagUserValueWords(*variant, programs[i]);
            if (drawStageDataHits() && stageCapture.capture != nullptr && !stageCapture.capture->readTrace.leaves.empty()) recordStageDataPositions(*variant, *stageCapture.capture, decodeReads[i]);
            if (traceDrawCache() && entry != nullptr && matched[i] == nullptr && i < entry->stages.size()) {
                const auto& variants = entry->stages[i];
                const auto front = std::find_if(variants.begin(), variants.end(), [&](const std::shared_ptr<DispatchVariant>& kept) { return kept->pushOffset == variant->pushOffset; });
                if (front != variants.end() && (*front)->runs == variant->runs && (*front)->words.size() == variant->words.size()) {
                    ++differingSameRuns;
                    for (std::size_t w = 0; w < variant->words.size(); ++w) differingWords += (*front)->words[w] != variant->words[w] ? 1u : 0u;
                } else if (front != variants.end()) {
                    ++differingRunsChanged;
                }
            }
            const auto sameContent = [](const DispatchVariant& a, const std::vector<std::uint32_t>& words, const DispatchVariant& b) {
                const auto flatten = [](const DispatchVariant& source, const std::vector<std::uint32_t>& values) {
                    std::vector<std::pair<std::uint64_t, std::uint32_t>> flat;
                    std::size_t word = 0;
                    for (const auto& [begin, end] : source.runs) {
                        for (auto address = begin; address < end && word < values.size(); address += sizeof(std::uint32_t)) flat.emplace_back(address, values[word++]);
                    }
                    std::sort(flat.begin(), flat.end());
                    return flat;
                };
                return (a.runs == b.runs && words == b.words) || flatten(a, words) == flatten(b, b.words);
            };
            const auto sameForwardedWords = [](const ShaderRecompiler::RecompileResult& a, const ShaderRecompiler::RecompileResult& b, std::uint32_t hidden) {
                if (a.pushConstants.size() != b.pushConstants.size() || a.shaderDataUserWords != b.shaderDataUserWords) return false;
                for (std::size_t slot = 0; slot < a.shaderDataUserWords.size() && (slot + 1) * sizeof(std::uint32_t) <= a.pushConstants.size(); ++slot) {
                    if (a.shaderDataUserWords[slot] >= hidden && std::memcmp(a.pushConstants.data() + slot * sizeof(std::uint32_t), b.pushConstants.data() + slot * sizeof(std::uint32_t), sizeof(std::uint32_t)) != 0) return false;
                }
                return true;
            };
            if (verifyHit && matched[i] != nullptr) {
                const auto& words = stageData[i].compiled != nullptr ? stageData[i].words : matched[i]->words;
                const auto& compiled = stageData[i].compiled != nullptr ? *stageData[i].compiled : *matched[i]->compiled;
                if (!sameContent(*matched[i], words, *variant) || !sameForwardedWords(compiled, *variant->compiled, programs[i].hiddenUserWords)) {
                    ++mismatches;
                    static std::atomic<std::uint64_t> reports{0};
                    if (reports.fetch_add(1) < 20) std::fprintf(stderr, "[draw-cache] verify: stage %zu (program 0x%llx) of a hit captured differently: %zu runs / %zu words matched, %zu / %zu fresh\n", i, static_cast<unsigned long long>(programs[i].binary.codeAddress), matched[i]->runs.size(), words.size(), variant->runs.size(), variant->words.size());
                }
            }
            if (insertCompare()) {
                const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::DrawCache);
                if (!captureStable(stageCapture.regions)) {
                    ++unstable;
                    continue;
                }
            }
            fresh[i] = std::move(variant);
        }
        insertDrawEntry(drawKey, fresh, registerKey ? decode : nullptr, registerKey && drawDataHits() ? shapeKey : 0, replaceShape);
        std::uint64_t dataInserts = 0, dataPositions = 0;
        for (std::size_t i = 0; i < programs.size(); ++i) {
            if (fresh[i] == nullptr || reused[i] || fresh[i]->dataPositions.empty()) continue;
            ++dataInserts;
            dataPositions += fresh[i]->dataPositions.size();
        }
        if (unstable != 0 || mismatches != 0 || differingSameRuns != 0 || differingRunsChanged != 0 || dataInserts != 0) {
            std::lock_guard cacheLock(drawCacheMutex);
            drawEntryCounters.stageDataInserts += dataInserts;
            drawEntryCounters.stageDataPositions += dataPositions;
            drawEntryCounters.unstable += unstable;
            drawEntryCounters.verifyMismatches += mismatches;
            drawEntryCounters.differingSameRuns += differingSameRuns;
            drawEntryCounters.differingWords += differingWords;
            drawEntryCounters.differingRunsChanged += differingRunsChanged;
        }
        phaseTiming.Phase(DrawRowKeyLookupValidate);
    }
}

}
