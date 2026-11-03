#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "Optimization/ResourceProgram.hpp"
#include <cstdlib>
#include <cstring>
#include <map>

namespace AgcDriver::DriverDetail {

void Driver::verifyDataHit(const ShaderSnapshot& snapshot, std::size_t codeOffset, ShaderRecompiler::RecompileRequest request, std::span<const ShaderRecompiler::MemoryRegion> memory, std::uint64_t address, const DispatchVariant& variant, std::span<const std::uint32_t> liveWords, const ShaderRecompiler::RecompileResult& patched) {
    const auto fail = [&](const char* what, std::size_t position, std::size_t slot) {
        std::fprintf(stderr, "[dispatch-cache] APS5_VERIFY_DATA_HITS: %s disagrees (program 0x%llx, position %zu, slot %zu)\n", what, static_cast<unsigned long long>(address), position, slot);
        std::fflush(stderr);
        std::abort();
    };
    auto verifyMemory = std::make_shared<ShaderMemory>(memory, &queryPendingWrite, &observePendingWrite, hookWaitCounter(), &QueryPendingSpan);
    auto invocation = InvocationFor(snapshot, codeOffset, request);
    const auto capture = CaptureInvocation(*verifyMemory, snapshot, codeOffset, request, invocation);
    const auto regions = verifyMemory->Regions();
    request.context.memory = regions;
    const auto result = invocation.Materialize(*capture);
    std::size_t position = 0, run = 0;
    for (const auto& region : regions) {
        const bool registered = std::any_of(memory.begin(), memory.end(), [&](const auto& known) { return region.guestAddress >= known.guestAddress && region.guestAddress < known.guestAddress + known.bytes.size(); });
        if (registered) continue;
        if (run >= variant.runs.size() || variant.runs[run].first != region.guestAddress || variant.runs[run].second != region.guestAddress + region.bytes.size()) fail("the read set", position, run);
        const auto count = region.bytes.size() / sizeof(std::uint32_t);
        if (position + count > liveWords.size() || std::memcmp(region.bytes.data(), liveWords.data() + position, region.bytes.size()) != 0) fail("the captured bytes", position, run);
        position += count;
        ++run;
    }
    if (run != variant.runs.size()) fail("the read set", position, run);
    if (result->variantId != patched.variantId) fail("the variant", 0, 0);
    if (result->bindings.size() != patched.bindings.size() || result->pushConstants != patched.pushConstants) fail("the bindings", 0, 0);
    for (std::size_t b = 0; b < result->bindings.size(); ++b) {
        const auto& fresh = result->bindings[b];
        const auto& reused = patched.bindings[b];
        if (fresh.kind != reused.kind || fresh.role != reused.role || fresh.binding != reused.binding || fresh.count != reused.count || fresh.guestDescriptor != reused.guestDescriptor) fail("the bindings", b, 0);
    }
    std::lock_guard cacheLock(dispatchCacheMutex);
    ++entryCounters.dataVerified;
}

void Driver::verifyFreeKeyHit(const ShaderSnapshot& snapshot, std::size_t codeOffset, ShaderRecompiler::RecompileRequest request, std::span<const ShaderRecompiler::MemoryRegion> memory, std::uint64_t address, std::span<const ShaderRecompiler::MemoryRegion> captured, const ShaderRecompiler::RecompileResult& reused) {
    const auto fail = [&](const char* what, std::uint64_t where) {
        std::fprintf(stderr, "[dispatch-cache] APS5_VERIFY_FREE_DISPATCH_KEYS: %s disagrees (program 0x%llx, at 0x%llx)\n", what, static_cast<unsigned long long>(address), static_cast<unsigned long long>(where));
        std::fflush(stderr);
        std::abort();
    };
    const auto words = [&](std::span<const ShaderRecompiler::MemoryRegion> regions) {
        std::map<std::uint64_t, std::uint32_t> mapped;
        for (const auto& region : regions) {
            const bool registered = std::any_of(memory.begin(), memory.end(), [&](const auto& known) { return region.guestAddress >= known.guestAddress && region.guestAddress < known.guestAddress + known.bytes.size(); });
            if (registered) continue;
            for (std::size_t at = 0; at + sizeof(std::uint32_t) <= region.bytes.size(); at += sizeof(std::uint32_t)) {
                std::uint32_t word = 0;
                std::memcpy(&word, region.bytes.data() + at, sizeof(word));
                const auto [slot, inserted] = mapped.emplace(region.guestAddress + at, word);
                if (!inserted && slot->second != word) fail("a word read twice", region.guestAddress + at);
            }
        }
        return mapped;
    };
    auto verifyMemory = std::make_shared<ShaderMemory>(memory, &queryPendingWrite, &observePendingWrite, hookWaitCounter(), &QueryPendingSpan);
    auto invocation = InvocationFor(snapshot, codeOffset, request);
    const auto capture = CaptureInvocation(*verifyMemory, snapshot, codeOffset, request, invocation);
    const auto regions = verifyMemory->Regions();
    request.context.memory = regions;
    const auto result = invocation.Materialize(*capture);
    const auto fresh = words(regions);
    const auto hit = words(captured);
    for (const auto& [at, word] : fresh) {
        const auto found = hit.find(at);
        if (found == hit.end()) fail("the read set", at);
        if (found->second != word) fail("the captured bytes", at);
    }
    if (fresh.size() != hit.size()) fail("the read set", 0);
    if (result->variantId != reused.variantId || result->specialization != reused.specialization) fail("the variant", 0);
    if (result->bindings.size() != reused.bindings.size() || result->pushConstants != reused.pushConstants) fail("the bindings", 0);
    for (std::size_t b = 0; b < result->bindings.size(); ++b) {
        const auto& lhs = result->bindings[b];
        const auto& rhs = reused.bindings[b];
        if (lhs.kind != rhs.kind || lhs.role != rhs.role || lhs.binding != rhs.binding || lhs.count != rhs.count || lhs.guestDescriptor != rhs.guestDescriptor) fail("the bindings", b);
    }
    std::lock_guard cacheLock(dispatchCacheMutex);
    ++entryCounters.freeVerified;
}

}
