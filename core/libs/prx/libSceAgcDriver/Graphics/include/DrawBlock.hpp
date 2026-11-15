#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DRAWBLOCK_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DRAWBLOCK_HPP

#include "prx/libSceAgcDriver/Core/include/BlockRing.hpp"
#include "prx/libSceAgcDriver/Core/include/BlockWriter.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics {

class Recorder;

}

namespace AgcDriver::Core {

enum class DrawBlockRefusal : std::uint32_t { None, Address, Layout, Count };

struct DrawBlockCounts {
    std::uint64_t draws = 0;
    std::uint64_t stages = 0;
    std::uint64_t sectionBytes = 0;
    std::uint64_t copies = 0;
    std::uint64_t copyBytes = 0;
    std::uint64_t apart = 0;
    std::uint64_t apartBytes = 0;
    std::uint64_t absent = 0;
    std::uint64_t synced = 0;
    std::uint64_t unreadable = 0;
    std::uint64_t mixed = 0;
    std::uint64_t mirrored = 0;
    std::uint64_t mirroredBytes = 0;
    std::uint64_t imported = 0;
    std::uint64_t importedBytes = 0;
    std::uint64_t verified = 0;
    std::uint64_t mismatches = 0;
    std::uint64_t classic = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(DrawBlockRefusal::Count)> refused{};
    BlockRing::Counts ring;
};

const char* DrawBlockRefusalName(DrawBlockRefusal refusal);

class DrawBlockHost {
public:
    static DrawBlockHost& For(const Graphics::Context& context);
    static void Release(VkDevice device);
    static DrawBlockCounts Counted();
    static std::string Report();
    static void NoteClassic();
    static void NoteRefused(DrawBlockRefusal refusal);
    static void NoteVerified(std::size_t mismatches);

    explicit DrawBlockHost(const Graphics::Context& context);
    [[nodiscard]] bool Addressable() const { return context.bufferDeviceAddress; }
    ~DrawBlockHost();
    DrawBlockHost(const DrawBlockHost&) = delete;
    DrawBlockHost& operator=(const DrawBlockHost&) = delete;

    struct Range {
        std::uint64_t deviceAddress = 0;
        const std::byte* bytes = nullptr;
        std::size_t size = 0;
    };

    class Draw {
    public:
        Draw(DrawBlockHost& host, Graphics::Recorder* recorder, std::vector<std::shared_ptr<void>>* kept, bool shadowed, std::vector<std::pair<std::uint64_t, std::uint64_t>>* reads);
        ~Draw();
        Draw(const Draw&) = delete;
        Draw& operator=(const Draw&) = delete;
        [[nodiscard]] std::uint32_t* Section(std::size_t bytes, std::uint64_t& deviceAddress);
        [[nodiscard]] bool Copy(std::uint64_t address, std::size_t bytes, std::size_t padded, std::uint64_t& deviceAddress, bool& placed);
        void Served(std::size_t stages, bool mixed);
        void Absent();
        [[nodiscard]] std::span<const Range> Ranges() const { return ranges; }
        static bool Read(void* context, std::uint64_t address, std::uint32_t* out, std::uint32_t count);
        static bool Resolve(void* context, const BufferRequest& request, BufferAnswer& answer);
        static std::span<const std::byte> BytesAt(void* context, std::uint64_t deviceAddress, std::uint32_t bytes);

    private:
        bool hold(const RingSlice& slice);
        bool inPlace(std::uint64_t address, std::size_t bytes, std::uint64_t& deviceAddress);
        std::byte* take(std::size_t bytes, std::uint64_t& deviceAddress, bool& ringed);
        DrawBlockHost& host;
        Graphics::Recorder* recorder = nullptr;
        std::vector<std::shared_ptr<void>>* kept = nullptr;
        std::vector<std::pair<std::uint64_t, std::uint64_t>>* reads = nullptr;
        bool shadowed = false;
        bool served = false;
        std::array<RingSlice, 8> held{};
        std::size_t heldCount = 0;
        std::vector<Range> ranges;
    };

private:
    static RingChunk allocate(void* context, std::size_t bytes);
    static bool executed(void* context, std::uint64_t serial);
    Graphics::Context context;
    Graphics::Recorder* recorder = nullptr;
    BlockRing ring;
};

[[nodiscard]] bool ReadGuestBytes(std::uint64_t address, std::span<std::byte> out);

}

namespace AgcDriver::Graphics {

struct ColorTarget;

struct DrawBlockSections {
    std::array<std::uint64_t, 4> address{};
    std::array<std::uint32_t, 4> pushByte{};
    std::uint32_t count = 0;
    void Patch(std::array<std::byte, PipelinePushConstantBytes>& block) const;
};

[[nodiscard]] bool UsesDrawBlocks(std::span<const CompiledShader> shaders);
enum class DrawBlockClass : std::uint32_t { Classic, Mixed, Block, Count };
[[nodiscard]] DrawBlockClass DrawBlockClassOf(std::span<const CompiledShader> shaders);
void NoteDrawBlockCommit(DrawBlockClass kind, std::uint64_t nanoseconds);
void WriteDrawBlocks(const Context& context, Recorder* recorder, std::span<const CompiledShader> shaders, const ColorTarget* target, DrawBlockSections& sections, std::vector<std::shared_ptr<void>>* kept, std::vector<std::pair<std::uint64_t, std::uint64_t>>* reads);

}

#endif
