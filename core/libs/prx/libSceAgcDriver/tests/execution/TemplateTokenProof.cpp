#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace AgcDriver {

struct VulkanDeviceTestAccess {
    static Graphics::Context ContextOf(const VulkanDevice& device) { return device.graphicsContext(); }
};

namespace Graphics {

bool TakeTemplateTokens(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges, std::span<const std::uint32_t> words, std::vector<GuestMemory::ContentToken>& tokens, std::uint64_t& taken);
bool TemplateTokensHold(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges, std::span<const std::uint32_t> words, std::span<const GuestMemory::ContentToken> tokens, std::uint64_t taken);
std::array<std::uint64_t, 7> TemplateTokenProofTotals();

}

}

namespace {

namespace GuestMemory = AgcDriver::GuestMemory;
using AgcDriver::Graphics::CompiledShader;
using AgcDriver::Graphics::ShaderResources;
using ShaderRecompiler::ShaderStage;

constexpr std::uint64_t Block = 65536;
constexpr std::uint64_t Page = 4096;
constexpr std::uint64_t ArenaBlocks = 16;
constexpr std::uint32_t Lanes = 64;
constexpr std::uint32_t Textures = 24;
constexpr std::uint32_t Stride = 32;
constexpr std::size_t SurfaceBytes = 256;
constexpr std::uint32_t Format8888UNorm = 56;
constexpr std::uint32_t Type2D = 9;
constexpr std::size_t SrtOffset = 4096;
constexpr std::size_t InputOffset = 8192;
constexpr std::size_t OutputOffset = 12288;
constexpr std::size_t HeapOffset = 65536;
constexpr std::size_t BlockBytes = (HeapOffset + std::size_t{Textures} * Stride + 65535u) & ~std::size_t{65535};
constexpr std::size_t SurfaceSpacing = 2 * Block;
constexpr std::size_t SurfaceBlockBytes = std::size_t{Textures} * SurfaceSpacing;

constexpr std::array<std::uint32_t, 25> Code{
    0xf4080100u, 0xfa000000u, 0xf4080200u, 0xfa000010u, 0xf4080300u, 0xfa000020u, 0xf4080700u, 0xfa000030u,
    0x34080082u, 0x7e200500u, 0x8f108510u, 0xf42c0502u, 0x20000000u, 0xbf8cc07fu,
    0xe0301000u, 0x80030504u, 0xf09c0f08u, 0x00450000u, 0xbf8c3f70u, 0x4a000105u,
    0xe0701000u, 0x80070004u, 0xbf810000u, 0u, 0u};

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::atomic<std::uint64_t> osWalks{0};

void Hook(int point) {
    if (point == 1) osWalks.fetch_add(1);
}

std::uint8_t* AllocateWatched(std::size_t bytes, bool gpu) {
#ifdef _WIN32
    auto* block = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(bytes, Block));
    Require(block != nullptr, "the guest arena allocation failed");
    GuestArena::GuestArenaCommit_nid_postfix(block, bytes, PAGE_READWRITE, bytes);
#else
    void* raw = mmap(nullptr, bytes + Block, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    Require(raw != MAP_FAILED, "cannot map the watched block");
    const auto begin = reinterpret_cast<std::uintptr_t>(raw);
    const auto aligned = (begin + Block - 1) & ~static_cast<std::uintptr_t>(Block - 1);
    if (aligned != begin) munmap(raw, aligned - begin);
    if (aligned + bytes != begin + bytes + Block) munmap(reinterpret_cast<void*>(aligned + bytes), begin + Block - aligned);
    auto* block = reinterpret_cast<std::uint8_t*>(aligned);
    GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, bytes);
#endif
    std::memset(block, 0, bytes);
    GuestAllocations::Mutation().Add(block, bytes, true, true, gpu);
    GuestMemory::CollectWritesUncached(reinterpret_cast<std::uintptr_t>(block), bytes);
    return block;
}

void Store(std::uint64_t address) {
    auto* byte = reinterpret_cast<volatile std::uint8_t*>(static_cast<std::uintptr_t>(address));
    *byte = static_cast<std::uint8_t>(*byte + 1);
}

struct Template {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    std::array<std::uint32_t, 16> words{};
    std::vector<GuestMemory::ContentToken> tokens;
    std::uint64_t taken = 0;

    void Take() {
        for (const auto& [begin, end] : ranges) Require(GuestMemory::CollectWrites(begin, static_cast<std::size_t>(end - begin)) != 0, "a collect of a template range failed");
        Require(AgcDriver::Graphics::TakeTemplateTokens(ranges, words, tokens, taken), "a template over collected ranges got no content tokens");
        Require(Holds(), "a template token proof just taken does not hold");
    }

    bool Holds() const { return AgcDriver::Graphics::TemplateTokensHold(ranges, words, tokens, taken); }
};

void RangesProveATemplate(std::uint64_t base) {
    const std::pair<std::uint64_t, std::uint64_t> surface{base + 2 * Block + Page, base + 2 * Block + 4 * Page};
    const std::pair<std::uint64_t, std::uint64_t> keys{base + 5 * Block, base + 5 * Block + Page};
    const std::pair<std::uint64_t, std::uint64_t> storage{base + 8 * Block, base + 10 * Block};
    Template proof;
    proof.ranges = {surface, keys, storage};
    for (std::size_t word = 0; word < proof.words.size(); ++word) proof.words[word] = static_cast<std::uint32_t>(0x1000u + word);
    GuestMemory::BumpCollectEpoch();
    proof.Take();
    GuestMemory::BumpCollectEpoch();
    auto walks = osWalks.load();
    Require(proof.Holds(), "a template whose ranges nothing wrote was refused in the next epoch");
    Require(osWalks.load() == walks, "the token proof of an unwritten template walked the write watch");
    auto other = proof;
    other.words[5] ^= 1u;
    Require(!other.Holds(), "a template was proved for other descriptor words");
    std::puts("an unwritten template is proved by its tokens without a sweep");

    Store(surface.first + 7);
    GuestMemory::BumpCollectEpoch();
    Require(!proof.Holds(), "a template was reused after a store into one of its ranges");
    proof.Take();
    GuestMemory::BumpCollectEpoch();
    Require(proof.Holds(), "a template retaken after the store did not hold in the next epoch");
    Store(storage.second + Block);
    GuestMemory::BumpCollectEpoch();
    Require(proof.Holds(), "a store outside the template's blocks refused the template");
    std::puts("a store into a range refuses the template");

    std::array<std::byte, 256> fill{};
    fill.fill(std::byte{0x55});
    GuestMemory::StoreOwnBytes(keys.first, fill);
    Require(!proof.Holds(), "a template was reused after a fill of its DCC keys stored by the driver");
    proof.Take();
    GuestMemory::NoteGpuPending(keys.first, static_cast<std::size_t>(keys.second - keys.first));
    Require(!proof.Holds(), "a template was reused after a fill of its DCC keys recorded for the GPU");
    proof.Take();
    GuestMemory::MarkWritten(keys.first, static_cast<std::size_t>(keys.second - keys.first));
    Require(!proof.Holds(), "a template was reused after the GPU reported the fill of its DCC keys");
    proof.Take();
    GuestMemory::BumpCollectEpoch();
    Require(proof.Holds(), "a template retaken after the DCC fill did not hold in the next epoch");
    std::puts("a fill of the DCC keys refuses the template");

    GuestMemory::MarkWritten(storage.first + Block + 64, 64);
    Require(!proof.Holds(), "a template was reused after a write of its storage image");
    proof.Take();
    GuestMemory::BreakProofs(GuestMemory::ProofBreak::Commit);
    Require(!proof.Holds(), "a template was reused after a commit that declared no ranges");
    std::puts("a storage image write refuses the template");
}

class GuestBlock {
public:
    GuestBlock(std::size_t bytes, bool gpu) : block(AllocateWatched(bytes, gpu)) {}
    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;

    std::uint64_t At(std::size_t offset) const { return reinterpret_cast<std::uint64_t>(block + offset); }
    std::uint8_t* Bytes(std::size_t offset) const { return block + offset; }
    std::uint32_t* Words(std::size_t offset) const { return reinterpret_cast<std::uint32_t*>(block + offset); }

private:
    std::uint8_t* block = nullptr;
};

std::array<std::uint32_t, 4> BufferDescriptor(std::uint64_t address, std::uint32_t stride, std::uint32_t records) {
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (stride << 16u), records, stride == 0u ? 0x31016facu : 0x01016facu};
}

void Prepare(const GuestBlock& block, const GuestBlock& surfaces) {
    std::memcpy(block.Words(0), Code.data(), sizeof(Code));
    for (std::uint32_t texture = 0; texture < Textures; ++texture) {
        const auto offset = static_cast<std::size_t>(texture) * SurfaceSpacing;
        std::memset(surfaces.Bytes(offset), static_cast<int>(texture & 0xffu), SurfaceBytes);
        const auto surface = surfaces.At(offset);
        const std::array<std::uint32_t, 8> words{static_cast<std::uint32_t>(surface >> 8u), static_cast<std::uint32_t>((surface >> 40u) & 0xffu) | (Format8888UNorm << 20u) | (3u << 30u), 1u | (7u << 14u), 0xfacu | (Type2D << 28u), 0u, 0u, 0u, 0u};
        std::memcpy(block.Words(HeapOffset + static_cast<std::size_t>(texture) * Stride), words.data(), sizeof(words));
    }
    auto* srt = block.Words(SrtOffset);
    const auto heap = BufferDescriptor(block.At(HeapOffset), Stride, Textures);
    const auto input = BufferDescriptor(block.At(InputOffset), 0u, Lanes * 4u);
    const auto output = BufferDescriptor(block.At(OutputOffset), 0u, Lanes * 4u);
    std::copy(heap.begin(), heap.end(), srt);
    std::copy(input.begin(), input.end(), srt + 8);
    std::copy(output.begin(), output.end(), srt + 12);
    GuestMemory::CollectWritesUncached(block.At(0), BlockBytes);
    GuestMemory::CollectWritesUncached(surfaces.At(0), SurfaceBlockBytes);
}

ShaderRecompiler::RecompileResult Compile(const AgcDriver::VulkanDevice& device, const GuestBlock& block) {
    const auto srt = block.At(SrtOffset);
    const std::array<std::uint32_t, 2> userData{static_cast<std::uint32_t>(srt), static_cast<std::uint32_t>(srt >> 32u)};
    const std::span<const std::uint32_t> code(block.Words(0), Code.size());
    const std::array<ShaderRecompiler::MemoryRegion, 3> memory{{
        {block.At(0), std::as_bytes(code)},
        {srt, std::as_bytes(std::span<const std::uint32_t>(block.Words(SrtOffset), 64))},
        {block.At(HeapOffset), std::as_bytes(std::span<const std::uint32_t>(block.Words(HeapOffset), static_cast<std::size_t>(Textures) * Stride / 4u))},
    }};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Lanes, 1, 1}, 0u, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, block.At(0), code, 0, {}},
        {Lanes, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    return ShaderRecompiler::Recompile(request);
}

struct Proved {
    bool current = false;
    ShaderResources::ProofReport report;
    std::uint64_t byToken = 0;
    std::uint64_t byTracker = 0;
    std::uint64_t walks = 0;
};

Proved Prove(ShaderResources& resources, const CompiledShader& shader) {
    Proved proved;
    const auto before = AgcDriver::Graphics::TemplateTokenProofTotals();
    const auto walks = osWalks.load();
    proved.current = resources.Revalidate(shader, &proved.report);
    const auto after = AgcDriver::Graphics::TemplateTokenProofTotals();
    proved.byToken = after[0] + after[1] - before[0] - before[1];
    proved.byTracker = after[2] - before[2];
    proved.walks = osWalks.load() - walks;
    return proved;
}

void TemplateProvesByTokens(const AgcDriver::VulkanDevice& device) {
    GuestBlock block(BlockBytes, true);
    GuestBlock surfaces(SurfaceBlockBytes, false);
    Prepare(block, surfaces);
    const auto result = Compile(device, block);
    const auto table = std::find_if(result.bindings.begin(), result.bindings.end(), [](const auto& binding) { return binding.role == ShaderRecompiler::DescriptorRole::ImageTable; });
    Require(table != result.bindings.end() && table->count == Textures, "the program does not bind an image table of " + std::to_string(Textures) + " textures");
    const auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(device);
    const CompiledShader shader{ShaderStage::Compute, &result, 0};
    std::lock_guard gpu(GuestMemory::GpuMutex());
    GuestMemory::BumpCollectEpoch();
    auto resources = std::make_shared<ShaderResources>(context, shader);
    Require(resources->Reusable() && resources->SampledTextures().size() == Textures, "the set over " + std::to_string(Textures) + " textures is not reusable");
    auto proved = Prove(*resources, shader);
    Require(proved.current && proved.report.path == ShaderResources::ProofPath::Fast, "the template did not take the fast proof after its build");
    proved = Prove(*resources, shader);
    Require(proved.current && proved.report.path == ShaderResources::ProofPath::Fast && proved.byToken == Textures && proved.byTracker == 0, "the second proof of an unwritten template proved " + std::to_string(proved.byToken) + " of its " + std::to_string(Textures) + " surfaces by their tokens");
    GuestMemory::BumpCollectEpoch();
    proved = Prove(*resources, shader);
    Require(proved.current && proved.report.path == ShaderResources::ProofPath::Fast && proved.byToken == Textures && proved.byTracker == 0, "the proof of an unwritten template in the next epoch proved " + std::to_string(proved.byToken) + " of its " + std::to_string(Textures) + " surfaces by their tokens");
    Require(proved.walks == 0, "the token proof of an unwritten template walked the write watch");
    std::puts("an unwritten template is proved by its tokens in the next epoch");

    const auto target = static_cast<std::size_t>(Textures / 3u) * SurfaceSpacing;
    auto* first = reinterpret_cast<volatile std::uint8_t*>(surfaces.Bytes(target));
    const std::uint8_t saved = *first;
    *first = saved;
    GuestMemory::BumpCollectEpoch();
    proved = Prove(*resources, shader);
    Require(proved.report.path == ShaderResources::ProofPath::Full, "a store over one surface left the template on the fast proof");
    Require(proved.byToken == Textures - 1u && proved.byTracker == 1, "a store over one surface was answered by the tokens of " + std::to_string(proved.byToken) + " surfaces and the write tracker for " + std::to_string(proved.byTracker));
    if (proved.current) {
        proved = Prove(*resources, shader);
        Require(proved.current && proved.report.path == ShaderResources::ProofPath::Fast, "the template did not return to the fast proof after the full walk");
        GuestMemory::BumpCollectEpoch();
        proved = Prove(*resources, shader);
        Require(proved.current && proved.byToken == Textures && proved.byTracker == 0, "the template did not return to its tokens after the full walk");
    }
    *first = static_cast<std::uint8_t>(saved ^ 0xffu);
    GuestMemory::BumpCollectEpoch();
    proved = Prove(*resources, shader);
    Require(!proved.current && proved.report.path == ShaderResources::ProofPath::Full, "a template whose surface changed was proved current");
    std::puts("a written surface refuses the template");
}

}

int main(int argc, char** argv) {
    const bool verify = argc == 2 && std::string(argv[1]) == "--verify";
#ifdef _WIN32
    _putenv_s("APS5_EPOCH_SWEEP", "1");
    _putenv_s("APS5_TEMPLATE_TOKEN_PROOF", "1");
    if (verify) _putenv_s("APS5_VERIFY_TEMPLATE_TOKEN_PROOF", "1");
    _putenv_s("APS5_TEMPLATE_AUDIT_MS", "0");
    _putenv_s("APS5_TABLE_INDEX", "0");
#else
    setenv("APS5_EPOCH_SWEEP", "1", 1);
    setenv("APS5_TEMPLATE_TOKEN_PROOF", "1", 1);
    if (verify) setenv("APS5_VERIFY_TEMPLATE_TOKEN_PROOF", "1", 1);
    setenv("APS5_TEMPLATE_AUDIT_MS", "0", 1);
    setenv("APS5_TABLE_INDEX", "0", 1);
#endif
    try {
        if (!GuestMemory::WriteWatched()) {
            std::cout << "template token proof tests skipped: guest memory is not write watched\n";
            return 77;
        }
        const auto base = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(AllocateWatched(static_cast<std::size_t>(ArenaBlocks * Block), true)));
        if (!GuestMemory::Watched(base, static_cast<std::size_t>(ArenaBlocks * Block))) {
            std::cout << "template token proof tests skipped: the test block is not write watched\n";
            return 77;
        }
        Require(GuestMemory::EpochSweeps(), "the epoch sweep switch was not read");
        GuestMemory::SetObservationHookForTest(&Hook);
        RangesProveATemplate(base);
        const auto device = OpenVulkanTestDevice();
        if (!device) {
            std::puts("template token proof tests passed without a device: the template part was skipped");
            return 0;
        }
        TemplateProvesByTokens(*device);
        std::puts("template token proof tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
