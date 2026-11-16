#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
#endif
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#include <SPIRV/GlslangToSpv.h>
#include <glslang/Public/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#define DGC_INSTANCE_FUNCTIONS(X) \
    X(vkDestroyInstance) \
    X(vkEnumeratePhysicalDevices) \
    X(vkGetPhysicalDeviceProperties2) \
    X(vkGetPhysicalDeviceFeatures2) \
    X(vkGetPhysicalDeviceQueueFamilyProperties) \
    X(vkGetPhysicalDeviceMemoryProperties) \
    X(vkEnumerateDeviceExtensionProperties) \
    X(vkCreateDevice) \
    X(vkGetDeviceProcAddr)

#define DGC_DEVICE_FUNCTIONS(X) \
    X(vkDestroyDevice) \
    X(vkGetDeviceQueue) \
    X(vkDeviceWaitIdle) \
    X(vkCreateBuffer) \
    X(vkDestroyBuffer) \
    X(vkGetBufferMemoryRequirements) \
    X(vkBindBufferMemory) \
    X(vkGetBufferDeviceAddress) \
    X(vkAllocateMemory) \
    X(vkFreeMemory) \
    X(vkMapMemory) \
    X(vkCreateImage) \
    X(vkDestroyImage) \
    X(vkGetImageMemoryRequirements) \
    X(vkBindImageMemory) \
    X(vkCreateImageView) \
    X(vkDestroyImageView) \
    X(vkCreateShaderModule) \
    X(vkDestroyShaderModule) \
    X(vkCreatePipelineLayout) \
    X(vkDestroyPipelineLayout) \
    X(vkCreateGraphicsPipelines) \
    X(vkDestroyPipeline) \
    X(vkCreateCommandPool) \
    X(vkDestroyCommandPool) \
    X(vkResetCommandPool) \
    X(vkAllocateCommandBuffers) \
    X(vkBeginCommandBuffer) \
    X(vkEndCommandBuffer) \
    X(vkCreateFence) \
    X(vkDestroyFence) \
    X(vkWaitForFences) \
    X(vkResetFences) \
    X(vkQueueSubmit) \
    X(vkCreateQueryPool) \
    X(vkDestroyQueryPool) \
    X(vkGetQueryPoolResults) \
    X(vkCmdResetQueryPool) \
    X(vkCmdWriteTimestamp) \
    X(vkCmdPipelineBarrier2) \
    X(vkCmdBeginRendering) \
    X(vkCmdEndRendering) \
    X(vkCmdSetViewport) \
    X(vkCmdSetScissor) \
    X(vkCmdBindPipeline) \
    X(vkCmdPushConstants) \
    X(vkCmdBindIndexBuffer) \
    X(vkCmdDrawIndexed) \
    X(vkCmdDrawIndexedIndirectCount) \
    X(vkCmdCopyImageToBuffer)

#define DGC_GENERATED_FUNCTIONS(X) \
    X(vkGetGeneratedCommandsMemoryRequirementsEXT) \
    X(vkCmdPreprocessGeneratedCommandsEXT) \
    X(vkCmdExecuteGeneratedCommandsEXT) \
    X(vkCreateIndirectCommandsLayoutEXT) \
    X(vkDestroyIndirectCommandsLayoutEXT) \
    X(vkCreateIndirectExecutionSetEXT) \
    X(vkDestroyIndirectExecutionSetEXT) \
    X(vkUpdateIndirectExecutionSetPipelineEXT)

namespace {

#define DGC_DECLARE(name) PFN_##name name = nullptr;

struct Api {
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
    PFN_vkCreateInstance vkCreateInstance = nullptr;
    DGC_INSTANCE_FUNCTIONS(DGC_DECLARE)
    DGC_DEVICE_FUNCTIONS(DGC_DECLARE)
    DGC_GENERATED_FUNCTIONS(DGC_DECLARE)
};

#undef DGC_DECLARE

Api vk;

using Clock = std::chrono::steady_clock;

constexpr VkFormat ColorFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat DepthFormat = VK_FORMAT_D32_SFLOAT;
constexpr VkShaderStageFlags GraphicsStages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
constexpr std::uint32_t TimestampCount = 4;
constexpr std::array<std::uint32_t, 4> Grids{2, 4, 8, 16};

void Check(VkResult result, std::string_view what) {
    if (result != VK_SUCCESS) throw std::runtime_error(std::string(what) + " returned " + std::to_string(static_cast<int>(result)));
}

enum class Arm { Inline, IndirectRuns, IndirectSorted, Generated, GeneratedPreprocessed };

constexpr std::array<Arm, 5> AllArms{Arm::Inline, Arm::IndirectRuns, Arm::IndirectSorted, Arm::Generated, Arm::GeneratedPreprocessed};

std::string_view Name(Arm arm) {
    switch (arm) {
    case Arm::Inline: return "inline";
    case Arm::IndirectRuns: return "indirect-runs";
    case Arm::IndirectSorted: return "indirect-sorted";
    case Arm::Generated: return "dgc";
    case Arm::GeneratedPreprocessed: return "dgc-pre";
    }
    return "?";
}

bool UsesGenerated(Arm arm) {
    return arm == Arm::Generated || arm == Arm::GeneratedPreprocessed;
}

struct Options {
    std::vector<std::uint32_t> draws{3000, 6000};
    std::vector<Arm> arms{AllArms.begin(), AllArms.end()};
    std::uint32_t pipelines = 16;
    std::uint32_t frames = 300;
    std::uint32_t warmup = 50;
    double switchRate = 0.45;
    std::uint32_t pollute = 0;
    std::uint32_t width = 1920;
    std::uint32_t height = 1080;
    std::uint64_t seed = 1;
    int cpu = 2;
    std::string device;
    std::filesystem::path json;
    bool separateState = false;
    bool fullBarrier = false;
};

std::uint64_t ParseNumber(std::string_view text, std::string_view option) {
    if (text.empty()) throw std::runtime_error("empty value for " + std::string(option));
    std::uint64_t value = 0;
    for (const char character : text) {
        if (character < '0' || character > '9') throw std::runtime_error("not a number for " + std::string(option) + ": " + std::string(text));
        value = value * 10 + static_cast<std::uint64_t>(character - '0');
    }
    return value;
}

std::vector<std::string_view> Split(std::string_view text) {
    std::vector<std::string_view> parts;
    while (true) {
        const auto comma = text.find(',');
        parts.push_back(text.substr(0, comma));
        if (comma == std::string_view::npos) break;
        text.remove_prefix(comma + 1);
    }
    return parts;
}

Options Parse(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (index + 1 >= argc) throw std::runtime_error("missing value for " + std::string(argument));
        const std::string_view value(argv[++index]);
        if (argument == "--draws") {
            options.draws.clear();
            for (const auto part : Split(value)) {
                const auto count = ParseNumber(part, argument);
                if (count == 0 || count > 1000000) throw std::runtime_error("--draws takes 1 to 1000000");
                options.draws.push_back(static_cast<std::uint32_t>(count));
            }
        } else if (argument == "--arms") {
            options.arms.clear();
            for (const auto part : Split(value)) {
                const auto found = std::find_if(AllArms.begin(), AllArms.end(), [&](Arm arm) { return Name(arm) == part; });
                if (found == AllArms.end()) throw std::runtime_error("unknown arm: " + std::string(part));
                options.arms.push_back(*found);
            }
        } else if (argument == "--pipelines") {
            options.pipelines = static_cast<std::uint32_t>(ParseNumber(value, argument));
            if (options.pipelines < 2 || options.pipelines > 1024) throw std::runtime_error("--pipelines takes 2 to 1024");
        } else if (argument == "--frames") {
            options.frames = static_cast<std::uint32_t>(ParseNumber(value, argument));
            if (options.frames == 0) throw std::runtime_error("--frames must be positive");
        } else if (argument == "--warmup") {
            options.warmup = static_cast<std::uint32_t>(ParseNumber(value, argument));
        } else if (argument == "--switch-percent") {
            const auto percent = ParseNumber(value, argument);
            if (percent > 100) throw std::runtime_error("--switch-percent takes 0 to 100");
            options.switchRate = static_cast<double>(percent) / 100.0;
        } else if (argument == "--pollute") {
            options.pollute = static_cast<std::uint32_t>(ParseNumber(value, argument));
            if (options.pollute > 16u * 1024u * 1024u) throw std::runtime_error("--pollute takes at most 16 MiB per draw");
        } else if (argument == "--width") {
            options.width = static_cast<std::uint32_t>(ParseNumber(value, argument));
        } else if (argument == "--height") {
            options.height = static_cast<std::uint32_t>(ParseNumber(value, argument));
        } else if (argument == "--seed") {
            options.seed = ParseNumber(value, argument);
        } else if (argument == "--cpu") {
            options.cpu = value == "none" ? -1 : static_cast<int>(ParseNumber(value, argument));
        } else if (argument == "--device") {
            options.device = value;
        } else if (argument == "--json") {
            options.json = value;
        } else if (argument == "--pre-state") {
            if (value != "same" && value != "separate") throw std::runtime_error("--pre-state takes same or separate");
            options.separateState = value == "separate";
        } else if (argument == "--pre-barrier") {
            if (value != "exact" && value != "full") throw std::runtime_error("--pre-barrier takes exact or full");
            options.fullBarrier = value == "full";
        } else {
            throw std::runtime_error("unknown option: " + std::string(argument));
        }
    }
    if (options.width == 0 || options.height == 0 || options.width > 8192 || options.height > 8192) throw std::runtime_error("--width and --height take 1 to 8192");
    if (options.arms.empty()) throw std::runtime_error("--arms is empty");
    return options;
}

struct Random {
    std::uint64_t state;

    std::uint64_t Next() {
        std::uint64_t value = (state += 0x9e3779b97f4a7c15ull);
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
        return value ^ (value >> 31);
    }

    double Unit() {
        return static_cast<double>(Next() >> 11) * 0x1.0p-53;
    }

    float UnitFloat() {
        return static_cast<float>(Unit());
    }

    std::uint32_t Below(std::uint32_t bound) {
        return static_cast<std::uint32_t>(Next() % bound);
    }
};

struct Mesh {
    std::uint32_t grid;
    std::uint32_t firstIndex;
    std::uint32_t indexCount;
};

struct DrawItem {
    std::uint32_t pipeline;
    std::uint32_t mesh;
};

struct alignas(16) Block {
    float rect[4];
    float params[4];
    float color[4];
    float extra[13][4];
};

static_assert(sizeof(Block) == 256);

struct Sequence {
    std::uint32_t pipeline;
    std::uint32_t pad;
    std::uint64_t address;
    VkDrawIndexedIndirectCommand draw;
    std::uint32_t tail;
};

static_assert(sizeof(Sequence) == 40);

struct TablePush {
    std::uint64_t base;
    std::uint32_t first;
    std::uint32_t pad;
};

static_assert(sizeof(TablePush) == 16);

struct Scene {
    std::vector<DrawItem> items;
    std::vector<Block> blocks;
    std::uint32_t changes = 0;
    std::uint32_t runs = 0;
    std::uint32_t usedPipelines = 0;
};

Scene BuildScene(std::uint32_t count, const Options& options) {
    Random random{options.seed * 0x100000001b3ull + count};
    Scene scene;
    scene.items.resize(count);
    scene.blocks.resize(count);
    std::vector<std::uint32_t> order(count);
    std::iota(order.begin(), order.end(), 0u);
    for (std::uint32_t index = count; index > 1; --index) std::swap(order[index - 1], order[random.Below(index)]);
    std::uint32_t pipeline = random.Below(options.pipelines);
    std::vector<bool> used(options.pipelines, false);
    for (std::uint32_t index = 0; index < count; ++index) {
        if (index == 0) {
            scene.runs = 1;
        } else if (random.Unit() < options.switchRate) {
            auto next = random.Below(options.pipelines - 1);
            if (next >= pipeline) ++next;
            pipeline = next;
            ++scene.changes;
            ++scene.runs;
        }
        used[pipeline] = true;
        const auto mesh = random.Below(static_cast<std::uint32_t>(Grids.size()));
        scene.items[index] = {pipeline, mesh};
        auto& block = scene.blocks[index];
        const float width = 0.02f + 0.08f * random.UnitFloat();
        const float height = 0.02f + 0.08f * random.UnitFloat();
        const float left = -1.0f + (2.0f - width) * random.UnitFloat();
        const float top = -1.0f + (2.0f - height) * random.UnitFloat();
        block.rect[0] = left;
        block.rect[1] = top;
        block.rect[2] = left + width;
        block.rect[3] = top + height;
        block.params[0] = static_cast<float>(order[index] + 1) / static_cast<float>(count + 2);
        block.params[1] = static_cast<float>(Grids[mesh]);
        block.params[2] = 0.0f;
        block.params[3] = 0.0f;
        for (std::uint32_t lane = 0; lane < 3; ++lane) block.color[lane] = random.UnitFloat();
        block.color[3] = 1.0f;
        for (auto& row : block.extra) {
            for (auto& lane : row) lane = random.UnitFloat();
        }
    }
    scene.usedPipelines = static_cast<std::uint32_t>(std::count(used.begin(), used.end(), true));
    return scene;
}

std::vector<std::uint32_t> BuildIndices(std::vector<Mesh>& meshes) {
    std::vector<std::uint32_t> indices;
    for (const auto grid : Grids) {
        Mesh mesh{grid, static_cast<std::uint32_t>(indices.size()), 0};
        const auto row = grid + 1;
        for (std::uint32_t y = 0; y < grid; ++y) {
            for (std::uint32_t x = 0; x < grid; ++x) {
                const auto corner = y * row + x;
                indices.insert(indices.end(), {corner, corner + row, corner + 1, corner + 1, corner + row, corner + row + 1});
            }
        }
        mesh.indexCount = static_cast<std::uint32_t>(indices.size()) - mesh.firstIndex;
        meshes.push_back(mesh);
    }
    return indices;
}

const char* const VertexSource = R"(#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer Block { vec4 rect; vec4 params; vec4 color; vec4 extra[13]; };
layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer Table { uint64_t entries[]; };
layout(push_constant) uniform Push { uint64_t base; uint first; } pc;
layout(location = 0) flat out uvec2 address;
layout(location = 1) out vec2 uv;
void main() {
#if DGC_TABLE
    uint64_t value = Table(pc.base).entries[pc.first + uint(gl_DrawID)];
#else
    uint64_t value = pc.base;
#endif
    Block block = Block(value);
    uint grid = uint(block.params.y);
    uint row = grid + 1u;
    uint vertex = uint(gl_VertexIndex);
    vec2 local = vec2(float(vertex % row), float(vertex / row)) / float(grid);
    gl_Position = vec4(mix(block.rect.xy, block.rect.zw, local), block.params.x, 1.0);
    address = uvec2(uint(value), uint(value >> 32));
    uv = local;
}
)";

const char* const FragmentSource = R"(#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(constant_id = 0) const uint PIPELINE = 0u;
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer Block { vec4 rect; vec4 params; vec4 color; vec4 extra[13]; };
layout(location = 0) flat in uvec2 address;
layout(location = 1) in vec2 uv;
layout(location = 0) out vec4 color;
void main() {
    Block block = Block(uint64_t(address.x) | (uint64_t(address.y) << 32));
    vec4 sum = block.color;
    for (uint index = 0u; index < 13u; ++index) sum += block.extra[index] * (float((index + PIPELINE) & 7u) * 0.03125);
    float k = float(PIPELINE) * 0.0625;
    color = vec4(fract(sum.rgb + vec3(k, 0.5 * k, 1.0 - k) + vec3(uv * 0.25, 0.0)), 1.0);
}
)";

std::vector<std::uint32_t> Compile(const std::string& source, EShLanguage stage) {
    glslang::TShader shader(stage);
    const char* text = source.c_str();
    shader.setStrings(&text, 1);
    shader.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientVulkan, 100);
    shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_3);
    shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_6);
    const auto messages = static_cast<EShMessages>(EShMsgSpvRules | EShMsgVulkanRules);
    if (!shader.parse(GetDefaultResources(), 460, false, messages)) throw std::runtime_error(std::string("glslang parse failed: ") + shader.getInfoLog());
    glslang::TProgram program;
    program.addShader(&shader);
    if (!program.link(messages)) throw std::runtime_error(std::string("glslang link failed: ") + program.getInfoLog());
    std::vector<std::uint32_t> spirv;
    glslang::SpvOptions spirvOptions{};
    spirvOptions.validate = true;
    glslang::GlslangToSpv(*program.getIntermediate(stage), spirv, &spirvOptions);
    if (spirv.empty()) throw std::runtime_error("glslang produced no SPIR-V");
    return spirv;
}

std::string VertexVariant(bool table) {
    std::string source(VertexSource);
    const auto line = source.find('\n');
    source.insert(line + 1, table ? "#define DGC_TABLE 1\n" : "#define DGC_TABLE 0\n");
    return source;
}

double Percentile(std::vector<double> values, double fraction) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double position = fraction * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<std::size_t>(position);
    const auto upper = std::min(lower + 1, values.size() - 1);
    return values[lower] + (values[upper] - values[lower]) * (position - static_cast<double>(lower));
}

struct Summary {
    double median = 0.0;
    double low = 0.0;
    double high = 0.0;
    double minimum = 0.0;
};

Summary Summarize(const std::vector<double>& values) {
    if (values.empty()) return {};
    return {Percentile(values, 0.5), Percentile(values, 0.1), Percentile(values, 0.9), *std::min_element(values.begin(), values.end())};
}

struct Sample {
    double cpuMicro = 0.0;
    double recordMicro = 0.0;
    double submitMicro = 0.0;
    double gpuMilli = 0.0;
    double preprocessMilli = 0.0;
};

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    std::byte* mapped = nullptr;
    VkDeviceAddress address = 0;
    VkDeviceSize size = 0;
    bool deviceLocal = false;
};

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
};

struct ArmResult {
    Arm arm;
    Summary cpu;
    Summary record;
    Summary submit;
    Summary gpu;
    Summary preprocess;
    std::uint64_t commands = 0;
    std::uint64_t differing = 0;
    std::uint64_t covered = 0;
    std::uint64_t hash = 0;
};

struct CountResult {
    std::uint32_t draws = 0;
    std::uint32_t changes = 0;
    std::uint32_t runs = 0;
    std::uint32_t usedPipelines = 0;
    double polluteMicro = 0.0;
    VkDeviceSize preprocessBytes = 0;
    VkDeviceSize preprocessExplicitBytes = 0;
    std::vector<ArmResult> arms;
};

std::uint64_t Fnv(const std::vector<std::uint32_t>& pixels) {
    std::uint64_t hash = 0xcbf29ce484222325ull;
    for (const auto pixel : pixels) {
        for (int shift = 0; shift < 32; shift += 8) {
            hash ^= (pixel >> shift) & 0xffu;
            hash *= 0x100000001b3ull;
        }
    }
    return hash;
}

std::string Quote(std::string_view text) {
    std::ostringstream result;
    result << '"';
    for (const unsigned char character : text) {
        if (character == '\\' || character == '"') result << '\\' << character;
        else if (character < 0x20u) result << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned>(character) << std::dec;
        else result << character;
    }
    result << '"';
    return result.str();
}

class Probe {
public:
    explicit Probe(const Options& options) : options_(options) {
        maxDraws_ = *std::max_element(options_.draws.begin(), options_.draws.end());
        generated_ = std::any_of(options_.arms.begin(), options_.arms.end(), UsesGenerated);
        CreateInstance();
        PickDevice();
        CreateDevice();
        CreateResources();
        CreatePipelines();
        if (generated_) CreateGenerated();
        scratch_.assign(options_.pollute == 0 ? 0 : 64u * 1024u * 1024u, 0);
    }

    ~Probe() {
        if (device_ != VK_NULL_HANDLE) {
            vk.vkDeviceWaitIdle(device_);
            if (generated_) {
                for (const auto layout : {implicitLayout_, explicitLayout_}) {
                    if (layout != VK_NULL_HANDLE) vk.vkDestroyIndirectCommandsLayoutEXT(device_, layout, nullptr);
                }
                if (executionSet_ != VK_NULL_HANDLE) vk.vkDestroyIndirectExecutionSetEXT(device_, executionSet_, nullptr);
            }
            for (const auto pipeline : pipelines_) vk.vkDestroyPipeline(device_, pipeline, nullptr);
            if (pipelineLayout_ != VK_NULL_HANDLE) vk.vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
            for (const auto module : modules_) vk.vkDestroyShaderModule(device_, module, nullptr);
            for (auto* buffer : {&blocks_, &indices_, &table_, &indirect_, &counts_, &sequences_, &preprocess_, &preprocessExplicit_, &readback_}) {
                if (buffer->buffer != VK_NULL_HANDLE) vk.vkDestroyBuffer(device_, buffer->buffer, nullptr);
                if (buffer->memory != VK_NULL_HANDLE) vk.vkFreeMemory(device_, buffer->memory, nullptr);
            }
            for (auto* image : {&color_, &depth_}) {
                if (image->view != VK_NULL_HANDLE) vk.vkDestroyImageView(device_, image->view, nullptr);
                if (image->image != VK_NULL_HANDLE) vk.vkDestroyImage(device_, image->image, nullptr);
                if (image->memory != VK_NULL_HANDLE) vk.vkFreeMemory(device_, image->memory, nullptr);
            }
            if (queries_ != VK_NULL_HANDLE) vk.vkDestroyQueryPool(device_, queries_, nullptr);
            if (fence_ != VK_NULL_HANDLE) vk.vkDestroyFence(device_, fence_, nullptr);
            if (pool_ != VK_NULL_HANDLE) vk.vkDestroyCommandPool(device_, pool_, nullptr);
            vk.vkDestroyDevice(device_, nullptr);
        }
        if (instance_ != VK_NULL_HANDLE) vk.vkDestroyInstance(instance_, nullptr);
    }

    Probe(const Probe&) = delete;
    Probe& operator=(const Probe&) = delete;

    std::vector<CountResult> Run() {
        std::vector<CountResult> results;
        for (const auto count : options_.draws) results.push_back(RunCount(count));
        return results;
    }

    void Print(const std::vector<CountResult>& results, std::ostream& output) const {
        output << "dgc-probe device=" << Quote(deviceName_) << " driver=" << Quote(driverInfo_)
               << " pipelines=" << options_.pipelines << " frames=" << options_.frames << " warmup=" << options_.warmup
               << " switch=" << options_.switchRate << " pollute=" << options_.pollute << " cpu=" << options_.cpu
               << " size=" << options_.width << 'x' << options_.height
               << " pre_state=" << (options_.separateState ? "separate" : "same") << " pre_barrier=" << (options_.fullBarrier ? "full" : "exact")
               << " host_buffers=" << (sequences_.deviceLocal || blocks_.deviceLocal ? "device-local" : "system") << '\n';
        for (const auto& result : results) {
            output << "\ndraws=" << result.draws << " pipeline_changes_per_draw=" << std::fixed << std::setprecision(3)
                   << static_cast<double>(result.changes) / result.draws << " runs=" << result.runs << " used_pipelines=" << result.usedPipelines
                   << " pollute_us_per_draw=" << result.polluteMicro;
            if (generated_) output << " preprocess_bytes=" << result.preprocessBytes << " preprocess_explicit_bytes=" << result.preprocessExplicitBytes;
            output << '\n';
            output << std::left << std::setw(16) << "arm" << std::right << std::setw(12) << "cpu_us/draw" << std::setw(13) << "rec_us/draw"
                   << std::setw(11) << "submit_us" << std::setw(10) << "gpu_ms" << std::setw(18) << "gpu_p10-p90" << std::setw(10) << "pre_ms"
                   << std::setw(10) << "commands" << std::setw(11) << "diff_px" << std::setw(11) << "covered" << "  hash\n";
            for (const auto& arm : result.arms) {
                std::ostringstream spread;
                spread << std::fixed << std::setprecision(3) << arm.gpu.low << '-' << arm.gpu.high;
                output << std::left << std::setw(16) << Name(arm.arm) << std::right << std::fixed << std::setprecision(4)
                       << std::setw(12) << arm.cpu.median << std::setw(13) << arm.record.median << std::setprecision(1)
                       << std::setw(11) << arm.submit.median << std::setprecision(3) << std::setw(10) << arm.gpu.median
                       << std::setw(18) << spread.str() << std::setw(10) << arm.preprocess.median << std::setw(10) << arm.commands
                       << std::setw(11) << arm.differing << std::setw(11) << arm.covered << "  " << std::hex << std::setw(16)
                       << std::setfill('0') << arm.hash << std::setfill(' ') << std::dec << '\n';
            }
        }
    }

    std::string Json(const std::vector<CountResult>& results) const {
        std::ostringstream output;
        output << std::setprecision(9);
        const auto summary = [&](const Summary& value) {
            std::ostringstream text;
            text << std::setprecision(9) << "{\"median\":" << value.median << ",\"p10\":" << value.low << ",\"p90\":" << value.high << ",\"min\":" << value.minimum << '}';
            return text.str();
        };
        output << "{\"schema_version\":1,\"tool\":\"agc_dgc_probe\",\"device\":" << Quote(deviceName_) << ",\"driver\":" << Quote(driverInfo_)
               << ",\"pipelines\":" << options_.pipelines << ",\"frames\":" << options_.frames << ",\"warmup\":" << options_.warmup
               << ",\"switch_rate\":" << options_.switchRate << ",\"pollute_bytes_per_draw\":" << options_.pollute << ",\"cpu\":" << options_.cpu
               << ",\"width\":" << options_.width << ",\"height\":" << options_.height << ",\"seed\":" << options_.seed
               << ",\"pre_state\":" << (options_.separateState ? "\"separate\"" : "\"same\"") << ",\"pre_barrier\":" << (options_.fullBarrier ? "\"full\"" : "\"exact\"")
               << ",\"host_buffers_device_local\":" << ((sequences_.deviceLocal || blocks_.deviceLocal) ? "true" : "false") << ",\"results\":[";
        for (std::size_t index = 0; index < results.size(); ++index) {
            const auto& result = results[index];
            output << (index == 0 ? "" : ",") << "{\"draws\":" << result.draws << ",\"pipeline_changes\":" << result.changes << ",\"runs\":" << result.runs
                   << ",\"used_pipelines\":" << result.usedPipelines << ",\"pollute_us_per_draw\":" << result.polluteMicro
                   << ",\"preprocess_bytes\":" << result.preprocessBytes << ",\"preprocess_explicit_bytes\":" << result.preprocessExplicitBytes << ",\"arms\":[";
            for (std::size_t armIndex = 0; armIndex < result.arms.size(); ++armIndex) {
                const auto& arm = result.arms[armIndex];
                output << (armIndex == 0 ? "" : ",") << "{\"arm\":" << Quote(Name(arm.arm)) << ",\"cpu_us_per_draw\":" << summary(arm.cpu)
                       << ",\"record_us_per_draw\":" << summary(arm.record) << ",\"submit_us\":" << summary(arm.submit)
                       << ",\"gpu_ms\":" << summary(arm.gpu) << ",\"preprocess_ms\":" << summary(arm.preprocess) << ",\"commands\":" << arm.commands
                       << ",\"differing_pixels\":" << arm.differing << ",\"covered_pixels\":" << arm.covered << ",\"image_fnv1a64\":\"" << std::hex
                       << std::setw(16) << std::setfill('0') << arm.hash << std::setfill(' ') << std::dec << "\"}";
            }
            output << "]}";
        }
        output << "]}\n";
        return output.str();
    }

private:
    void CreateInstance() {
        VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        application.pApplicationName = "agc_dgc_probe";
        application.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        info.pApplicationInfo = &application;
        Check(vk.vkCreateInstance(&info, nullptr, &instance_), "vkCreateInstance");
#define DGC_LOAD_INSTANCE(name) \
        vk.name = reinterpret_cast<PFN_##name>(vk.vkGetInstanceProcAddr(instance_, #name)); \
        if (vk.name == nullptr) throw std::runtime_error("missing " #name);
        DGC_INSTANCE_FUNCTIONS(DGC_LOAD_INSTANCE)
#undef DGC_LOAD_INSTANCE
    }

    std::vector<std::string> Extensions(VkPhysicalDevice device) const {
        std::uint32_t count = 0;
        Check(vk.vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr), "vkEnumerateDeviceExtensionProperties");
        std::vector<VkExtensionProperties> properties(count);
        Check(vk.vkEnumerateDeviceExtensionProperties(device, nullptr, &count, properties.data()), "vkEnumerateDeviceExtensionProperties");
        std::vector<std::string> names;
        for (const auto& property : properties) names.emplace_back(property.extensionName);
        return names;
    }

    void PickDevice() {
        std::uint32_t count = 0;
        Check(vk.vkEnumeratePhysicalDevices(instance_, &count, nullptr), "vkEnumeratePhysicalDevices");
        std::vector<VkPhysicalDevice> devices(count);
        Check(vk.vkEnumeratePhysicalDevices(instance_, &count, devices.data()), "vkEnumeratePhysicalDevices");
        std::string rejected;
        for (const auto candidate : devices) {
            VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            vk.vkGetPhysicalDeviceProperties2(candidate, &properties);
            const std::string name(properties.properties.deviceName);
            const auto extensions = Extensions(candidate);
            const auto has = [&](std::string_view extension) { return std::find(extensions.begin(), extensions.end(), extension) != extensions.end(); };
            std::string reason;
            if (!options_.device.empty() && name.find(options_.device) == std::string::npos) reason = "name";
            else if (properties.properties.apiVersion < VK_API_VERSION_1_3) reason = "Vulkan 1.3";
            else if (!has(VK_KHR_MAINTENANCE_5_EXTENSION_NAME)) reason = VK_KHR_MAINTENANCE_5_EXTENSION_NAME;
            else if (generated_ && !has(VK_EXT_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME)) reason = VK_EXT_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME;
            if (!reason.empty()) {
                rejected += " " + name + " (" + reason + ")";
                continue;
            }
            if (physical_ == VK_NULL_HANDLE || properties.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                physical_ = candidate;
                if (properties.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) break;
            }
        }
        if (physical_ == VK_NULL_HANDLE) throw std::runtime_error("no suitable device:" + rejected);
        VkPhysicalDeviceDriverProperties driver{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        VkPhysicalDeviceDeviceGeneratedCommandsPropertiesEXT generatedProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_PROPERTIES_EXT};
        if (generated_) driver.pNext = &generatedProperties;
        VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &driver};
        vk.vkGetPhysicalDeviceProperties2(physical_, &properties);
        deviceName_ = properties.properties.deviceName;
        driverInfo_ = std::string(driver.driverName) + " " + driver.driverInfo;
        timestampPeriod_ = properties.properties.limits.timestampPeriod;
        vk.vkGetPhysicalDeviceMemoryProperties(physical_, &memory_);
        if (generated_) {
            if (generatedProperties.maxIndirectPipelineCount < options_.pipelines) throw std::runtime_error("maxIndirectPipelineCount below --pipelines");
            if (generatedProperties.maxIndirectSequenceCount < maxDraws_) throw std::runtime_error("maxIndirectSequenceCount below --draws");
            if (generatedProperties.maxIndirectCommandsIndirectStride < sizeof(Sequence)) throw std::runtime_error("maxIndirectCommandsIndirectStride below the sequence stride");
            if (generatedProperties.maxIndirectCommandsTokenOffset < offsetof(Sequence, draw)) throw std::runtime_error("maxIndirectCommandsTokenOffset below the draw token offset");
            if ((generatedProperties.supportedIndirectCommandsShaderStagesPipelineBinding & GraphicsStages) != GraphicsStages) throw std::runtime_error("device cannot switch vertex and fragment pipelines from generated commands");
            if ((generatedProperties.supportedIndirectCommandsShaderStages & GraphicsStages) != GraphicsStages) throw std::runtime_error("device cannot generate graphics commands");
        }
        std::uint32_t families = 0;
        vk.vkGetPhysicalDeviceQueueFamilyProperties(physical_, &families, nullptr);
        std::vector<VkQueueFamilyProperties> familyProperties(families);
        vk.vkGetPhysicalDeviceQueueFamilyProperties(physical_, &families, familyProperties.data());
        for (std::uint32_t index = 0; index < families; ++index) {
            if ((familyProperties[index].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0 && familyProperties[index].timestampValidBits != 0) {
                family_ = index;
                timestampMask_ = familyProperties[index].timestampValidBits >= 64 ? ~0ull : ((1ull << familyProperties[index].timestampValidBits) - 1);
                return;
            }
        }
        throw std::runtime_error("no graphics queue with timestamps");
    }

    void CreateDevice() {
        VkPhysicalDeviceDeviceGeneratedCommandsFeaturesEXT generatedSupport{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES_EXT};
        VkPhysicalDeviceMaintenance5Features maintenanceSupport{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES, generated_ ? &generatedSupport : nullptr};
        VkPhysicalDeviceVulkan13Features support13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &maintenanceSupport};
        VkPhysicalDeviceVulkan12Features support12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &support13};
        VkPhysicalDeviceVulkan11Features support11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &support12};
        VkPhysicalDeviceFeatures2 support{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &support11};
        vk.vkGetPhysicalDeviceFeatures2(physical_, &support);
        const std::array<std::pair<VkBool32, const char*>, 8> required{{
            {support.features.shaderInt64, "shaderInt64"},
            {support11.shaderDrawParameters, "shaderDrawParameters"},
            {support12.bufferDeviceAddress, "bufferDeviceAddress"},
            {support12.drawIndirectCount, "drawIndirectCount"},
            {support13.dynamicRendering, "dynamicRendering"},
            {support13.synchronization2, "synchronization2"},
            {maintenanceSupport.maintenance5, "maintenance5"},
            {generated_ ? generatedSupport.deviceGeneratedCommands : VK_TRUE, "deviceGeneratedCommands"},
        }};
        for (const auto& [supported, name] : required) {
            if (supported != VK_TRUE) throw std::runtime_error(std::string("missing feature ") + name);
        }
        VkPhysicalDeviceDeviceGeneratedCommandsFeaturesEXT generatedFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES_EXT};
        generatedFeatures.deviceGeneratedCommands = VK_TRUE;
        VkPhysicalDeviceMaintenance5Features maintenanceFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES, generated_ ? &generatedFeatures : nullptr};
        maintenanceFeatures.maintenance5 = VK_TRUE;
        VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &maintenanceFeatures};
        features13.dynamicRendering = VK_TRUE;
        features13.synchronization2 = VK_TRUE;
        VkPhysicalDeviceVulkan12Features features12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &features13};
        features12.bufferDeviceAddress = VK_TRUE;
        features12.drawIndirectCount = VK_TRUE;
        VkPhysicalDeviceVulkan11Features features11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &features12};
        features11.shaderDrawParameters = VK_TRUE;
        VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &features11};
        features.features.shaderInt64 = VK_TRUE;
        const float priority = 1.0f;
        VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue.queueFamilyIndex = family_;
        queue.queueCount = 1;
        queue.pQueuePriorities = &priority;
        std::vector<const char*> extensions{VK_KHR_MAINTENANCE_5_EXTENSION_NAME};
        if (generated_) extensions.push_back(VK_EXT_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME);
        VkDeviceCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &features};
        info.queueCreateInfoCount = 1;
        info.pQueueCreateInfos = &queue;
        info.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
        info.ppEnabledExtensionNames = extensions.data();
        Check(vk.vkCreateDevice(physical_, &info, nullptr, &device_), "vkCreateDevice");
#define DGC_LOAD_DEVICE(name) \
        vk.name = reinterpret_cast<PFN_##name>(vk.vkGetDeviceProcAddr(device_, #name)); \
        if (vk.name == nullptr) throw std::runtime_error("missing " #name);
        DGC_DEVICE_FUNCTIONS(DGC_LOAD_DEVICE)
        if (generated_) {
            DGC_GENERATED_FUNCTIONS(DGC_LOAD_DEVICE)
        }
#undef DGC_LOAD_DEVICE
        vk.vkGetDeviceQueue(device_, family_, 0, &queue_);
    }

    std::uint32_t MemoryType(std::uint32_t bits, VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred) const {
        for (const auto wanted : {required | preferred, required}) {
            for (std::uint32_t index = 0; index < memory_.memoryTypeCount; ++index) {
                if ((bits & (1u << index)) != 0 && (memory_.memoryTypes[index].propertyFlags & wanted) == wanted) return index;
            }
        }
        throw std::runtime_error("no memory type for the requested properties");
    }

    Buffer CreateBuffer(VkDeviceSize size, VkBufferUsageFlags2 usage, bool host, VkMemoryPropertyFlags preferred, std::uint32_t allowedTypes = ~0u) {
        Buffer result;
        result.size = size;
        VkBufferUsageFlags2CreateInfo usageInfo{VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO};
        usageInfo.usage = usage;
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &usageInfo};
        info.size = size;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        Check(vk.vkCreateBuffer(device_, &info, nullptr, &result.buffer), "vkCreateBuffer");
        VkMemoryRequirements requirements{};
        vk.vkGetBufferMemoryRequirements(device_, result.buffer, &requirements);
        const auto required = host ? static_cast<VkMemoryPropertyFlags>(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                                   : static_cast<VkMemoryPropertyFlags>(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        const auto type = MemoryType(requirements.memoryTypeBits & allowedTypes, required, preferred);
        result.deviceLocal = (memory_.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &flags};
        allocate.allocationSize = requirements.size;
        allocate.memoryTypeIndex = type;
        Check(vk.vkAllocateMemory(device_, &allocate, nullptr, &result.memory), "vkAllocateMemory");
        Check(vk.vkBindBufferMemory(device_, result.buffer, result.memory, 0), "vkBindBufferMemory");
        if (host) {
            void* mapped = nullptr;
            Check(vk.vkMapMemory(device_, result.memory, 0, VK_WHOLE_SIZE, 0, &mapped), "vkMapMemory");
            result.mapped = static_cast<std::byte*>(mapped);
        }
        if ((usage & VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT) != 0) {
            VkBufferDeviceAddressInfo addressInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            addressInfo.buffer = result.buffer;
            result.address = vk.vkGetBufferDeviceAddress(device_, &addressInfo);
        }
        return result;
    }

    Image CreateImage(VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect) {
        Image result;
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = format;
        info.extent = {options_.width, options_.height, 1};
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        Check(vk.vkCreateImage(device_, &info, nullptr, &result.image), "vkCreateImage");
        VkMemoryRequirements requirements{};
        vk.vkGetImageMemoryRequirements(device_, result.image, &requirements);
        VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocate.allocationSize = requirements.size;
        allocate.memoryTypeIndex = MemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
        Check(vk.vkAllocateMemory(device_, &allocate, nullptr, &result.memory), "vkAllocateMemory");
        Check(vk.vkBindImageMemory(device_, result.image, result.memory, 0), "vkBindImageMemory");
        VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view.image = result.image;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = format;
        view.subresourceRange = {aspect, 0, 1, 0, 1};
        Check(vk.vkCreateImageView(device_, &view, nullptr, &result.view), "vkCreateImageView");
        return result;
    }

    void CreateResources() {
        const VkMemoryPropertyFlags local = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        blocks_ = CreateBuffer(VkDeviceSize{maxDraws_} * sizeof(Block), VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT, true, local);
        const auto indexData = BuildIndices(meshes_);
        indices_ = CreateBuffer(indexData.size() * sizeof(std::uint32_t), VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT, true, local);
        std::memcpy(indices_.mapped, indexData.data(), indexData.size() * sizeof(std::uint32_t));
        table_ = CreateBuffer(VkDeviceSize{maxDraws_} * sizeof(std::uint64_t), VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT, true, local);
        indirect_ = CreateBuffer(VkDeviceSize{maxDraws_} * sizeof(VkDrawIndexedIndirectCommand), VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT, true, local);
        counts_ = CreateBuffer(VkDeviceSize{std::max(maxDraws_, options_.pipelines)} * sizeof(std::uint32_t), VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT, true, local);
        if (generated_) sequences_ = CreateBuffer(VkDeviceSize{maxDraws_} * sizeof(Sequence), VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT, true, local);
        readback_ = CreateBuffer(VkDeviceSize{options_.width} * options_.height * 4, VK_BUFFER_USAGE_2_TRANSFER_DST_BIT, true, VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        color_ = CreateImage(ColorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        depth_ = CreateImage(DepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
        VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        poolInfo.queueFamilyIndex = family_;
        Check(vk.vkCreateCommandPool(device_, &poolInfo, nullptr, &pool_), "vkCreateCommandPool");
        VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocate.commandPool = pool_;
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1;
        Check(vk.vkAllocateCommandBuffers(device_, &allocate, &command_), "vkAllocateCommandBuffers");
        Check(vk.vkAllocateCommandBuffers(device_, &allocate, &state_), "vkAllocateCommandBuffers");
        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        Check(vk.vkCreateFence(device_, &fenceInfo, nullptr, &fence_), "vkCreateFence");
        VkQueryPoolCreateInfo queryInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        queryInfo.queryCount = TimestampCount;
        Check(vk.vkCreateQueryPool(device_, &queryInfo, nullptr, &queries_), "vkCreateQueryPool");
    }

    VkShaderModule CreateModule(const std::vector<std::uint32_t>& spirv) {
        VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        info.codeSize = spirv.size() * sizeof(std::uint32_t);
        info.pCode = spirv.data();
        VkShaderModule module = VK_NULL_HANDLE;
        Check(vk.vkCreateShaderModule(device_, &info, nullptr, &module), "vkCreateShaderModule");
        modules_.push_back(module);
        return module;
    }

    VkPipeline CreatePipeline(VkShaderModule vertex, VkShaderModule fragment, std::uint32_t index, bool bindable) {
        const VkSpecializationMapEntry entry{0, 0, sizeof(std::uint32_t)};
        const VkSpecializationInfo specialization{1, &entry, sizeof(std::uint32_t), &index};
        const std::array<VkPipelineShaderStageCreateInfo, 2> stages{{
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertex, "main", nullptr},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragment, "main", &specialization},
        }};
        VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport.viewportCount = 1;
        viewport.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depth.depthTestEnable = VK_TRUE;
        depth.depthWriteEnable = VK_TRUE;
        depth.depthCompareOp = VK_COMPARE_OP_LESS;
        VkPipelineColorBlendAttachmentState attachment{};
        attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blend.attachmentCount = 1;
        blend.pAttachments = &attachment;
        const std::array<VkDynamicState, 2> dynamicStates{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size());
        dynamic.pDynamicStates = dynamicStates.data();
        const VkFormat colorFormat = ColorFormat;
        VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachmentFormats = &colorFormat;
        rendering.depthAttachmentFormat = DepthFormat;
        VkPipelineCreateFlags2CreateInfo flags{VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO, &rendering};
        flags.flags = VK_PIPELINE_CREATE_2_INDIRECT_BINDABLE_BIT_EXT;
        VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        info.pNext = bindable ? static_cast<const void*>(&flags) : static_cast<const void*>(&rendering);
        info.stageCount = static_cast<std::uint32_t>(stages.size());
        info.pStages = stages.data();
        info.pVertexInputState = &vertexInput;
        info.pInputAssemblyState = &assembly;
        info.pViewportState = &viewport;
        info.pRasterizationState = &raster;
        info.pMultisampleState = &multisample;
        info.pDepthStencilState = &depth;
        info.pColorBlendState = &blend;
        info.pDynamicState = &dynamic;
        info.layout = pipelineLayout_;
        VkPipeline pipeline = VK_NULL_HANDLE;
        Check(vk.vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline), "vkCreateGraphicsPipelines");
        pipelines_.push_back(pipeline);
        return pipeline;
    }

    void CreatePipelines() {
        if (!glslang::InitializeProcess()) throw std::runtime_error("glslang initialization failed");
        const auto vertexDirect = CreateModule(Compile(VertexVariant(false), EShLangVertex));
        const auto vertexTable = CreateModule(Compile(VertexVariant(true), EShLangVertex));
        const auto fragment = CreateModule(Compile(FragmentSource, EShLangFragment));
        glslang::FinalizeProcess();
        const VkPushConstantRange range{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(TablePush)};
        VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout.pushConstantRangeCount = 1;
        layout.pPushConstantRanges = &range;
        Check(vk.vkCreatePipelineLayout(device_, &layout, nullptr, &pipelineLayout_), "vkCreatePipelineLayout");
        for (std::uint32_t index = 0; index < options_.pipelines; ++index) {
            direct_.push_back(CreatePipeline(vertexDirect, fragment, index, false));
            tabled_.push_back(CreatePipeline(vertexTable, fragment, index, false));
            if (generated_) bindable_.push_back(CreatePipeline(vertexDirect, fragment, index, true));
        }
    }

    VkIndirectCommandsLayoutEXT CreateLayout(VkIndirectCommandsLayoutUsageFlagsEXT usage) {
        const VkIndirectCommandsExecutionSetTokenEXT setToken{VK_INDIRECT_EXECUTION_SET_INFO_TYPE_PIPELINES_EXT, GraphicsStages};
        const VkIndirectCommandsPushConstantTokenEXT pushToken{{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(std::uint64_t)}};
        std::array<VkIndirectCommandsLayoutTokenEXT, 3> tokens{};
        for (auto& token : tokens) token.sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT;
        tokens[0].type = VK_INDIRECT_COMMANDS_TOKEN_TYPE_EXECUTION_SET_EXT;
        tokens[0].data.pExecutionSet = &setToken;
        tokens[0].offset = offsetof(Sequence, pipeline);
        tokens[1].type = VK_INDIRECT_COMMANDS_TOKEN_TYPE_PUSH_CONSTANT_EXT;
        tokens[1].data.pPushConstant = &pushToken;
        tokens[1].offset = offsetof(Sequence, address);
        tokens[2].type = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_INDEXED_EXT;
        tokens[2].offset = offsetof(Sequence, draw);
        VkIndirectCommandsLayoutCreateInfoEXT info{VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_CREATE_INFO_EXT};
        info.flags = usage;
        info.shaderStages = GraphicsStages;
        info.indirectStride = sizeof(Sequence);
        info.pipelineLayout = pipelineLayout_;
        info.tokenCount = static_cast<std::uint32_t>(tokens.size());
        info.pTokens = tokens.data();
        VkIndirectCommandsLayoutEXT layout = VK_NULL_HANDLE;
        Check(vk.vkCreateIndirectCommandsLayoutEXT(device_, &info, nullptr, &layout), "vkCreateIndirectCommandsLayoutEXT");
        return layout;
    }

    VkMemoryRequirements GeneratedRequirements(VkIndirectCommandsLayoutEXT layout, std::uint32_t count) const {
        VkGeneratedCommandsMemoryRequirementsInfoEXT info{VK_STRUCTURE_TYPE_GENERATED_COMMANDS_MEMORY_REQUIREMENTS_INFO_EXT};
        info.indirectExecutionSet = executionSet_;
        info.indirectCommandsLayout = layout;
        info.maxSequenceCount = count;
        VkMemoryRequirements2 requirements{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
        vk.vkGetGeneratedCommandsMemoryRequirementsEXT(device_, &info, &requirements);
        return requirements.memoryRequirements;
    }

    Buffer CreatePreprocess(VkIndirectCommandsLayoutEXT layout) {
        const auto requirements = GeneratedRequirements(layout, maxDraws_);
        if (requirements.size == 0) return {};
        return CreateBuffer(requirements.size, VK_BUFFER_USAGE_2_PREPROCESS_BUFFER_BIT_EXT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT, false, 0, requirements.memoryTypeBits);
    }

    void CreateGenerated() {
        VkIndirectExecutionSetPipelineInfoEXT pipelineInfo{VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_PIPELINE_INFO_EXT};
        pipelineInfo.initialPipeline = bindable_.front();
        pipelineInfo.maxPipelineCount = options_.pipelines;
        VkIndirectExecutionSetCreateInfoEXT setInfo{VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_CREATE_INFO_EXT};
        setInfo.type = VK_INDIRECT_EXECUTION_SET_INFO_TYPE_PIPELINES_EXT;
        setInfo.info.pPipelineInfo = &pipelineInfo;
        Check(vk.vkCreateIndirectExecutionSetEXT(device_, &setInfo, nullptr, &executionSet_), "vkCreateIndirectExecutionSetEXT");
        std::vector<VkWriteIndirectExecutionSetPipelineEXT> writes;
        for (std::uint32_t index = 1; index < options_.pipelines; ++index) {
            writes.push_back({VK_STRUCTURE_TYPE_WRITE_INDIRECT_EXECUTION_SET_PIPELINE_EXT, nullptr, index, bindable_[index]});
        }
        vk.vkUpdateIndirectExecutionSetPipelineEXT(device_, executionSet_, static_cast<std::uint32_t>(writes.size()), writes.data());
        implicitLayout_ = CreateLayout(0);
        explicitLayout_ = CreateLayout(VK_INDIRECT_COMMANDS_LAYOUT_USAGE_EXPLICIT_PREPROCESS_BIT_EXT);
        preprocess_ = CreatePreprocess(implicitLayout_);
        preprocessExplicit_ = CreatePreprocess(explicitLayout_);
    }

    void Pollute() {
        if (options_.pollute == 0) return;
        auto* bytes = scratch_.data();
        const auto size = scratch_.size();
        for (std::uint32_t done = 0; done < options_.pollute; done += 64) {
            bytes[cursor_] = static_cast<std::uint8_t>(bytes[cursor_] + 1);
            cursor_ += 64;
            if (cursor_ >= size) cursor_ = 0;
        }
    }

    double PolluteMicro(std::uint32_t count) {
        if (options_.pollute == 0) return 0.0;
        std::vector<double> values;
        for (int repeat = 0; repeat < 21; ++repeat) {
            const auto start = Clock::now();
            for (std::uint32_t index = 0; index < count; ++index) Pollute();
            values.push_back(std::chrono::duration<double, std::micro>(Clock::now() - start).count() / count);
        }
        return Percentile(values, 0.5);
    }

    void SetState(VkCommandBuffer buffer) {
        const VkViewport viewport{0.0f, 0.0f, static_cast<float>(options_.width), static_cast<float>(options_.height), 0.0f, 1.0f};
        const VkRect2D scissor{{0, 0}, {options_.width, options_.height}};
        vk.vkCmdSetViewport(buffer, 0, 1, &viewport);
        vk.vkCmdSetScissor(buffer, 0, 1, &scissor);
        vk.vkCmdBindIndexBuffer(buffer, indices_.buffer, 0, VK_INDEX_TYPE_UINT32);
    }

    std::uint64_t RecordInline(const Scene& scene) {
        std::uint64_t commands = 0;
        std::uint32_t bound = ~0u;
        const auto count = static_cast<std::uint32_t>(scene.items.size());
        for (std::uint32_t index = 0; index < count; ++index) {
            const auto& item = scene.items[index];
            Pollute();
            if (item.pipeline != bound) {
                vk.vkCmdBindPipeline(command_, VK_PIPELINE_BIND_POINT_GRAPHICS, direct_[item.pipeline]);
                bound = item.pipeline;
                ++commands;
            }
            const std::uint64_t address = blocks_.address + VkDeviceSize{index} * sizeof(Block);
            vk.vkCmdPushConstants(command_, pipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(address), &address);
            const auto& mesh = meshes_[item.mesh];
            vk.vkCmdDrawIndexed(command_, mesh.indexCount, 1, mesh.firstIndex, 0, 0);
            commands += 2;
        }
        return commands;
    }

    std::uint64_t RecordRuns(const Scene& scene) {
        std::uint64_t commands = 0;
        auto* commandsOut = reinterpret_cast<VkDrawIndexedIndirectCommand*>(indirect_.mapped);
        auto* tableOut = reinterpret_cast<std::uint64_t*>(table_.mapped);
        auto* countsOut = reinterpret_cast<std::uint32_t*>(counts_.mapped);
        const auto count = static_cast<std::uint32_t>(scene.items.size());
        std::uint32_t run = 0;
        std::uint32_t first = 0;
        while (first < count) {
            const auto pipeline = scene.items[first].pipeline;
            std::uint32_t end = first;
            while (end < count && scene.items[end].pipeline == pipeline) {
                Pollute();
                const auto& mesh = meshes_[scene.items[end].mesh];
                commandsOut[end] = {mesh.indexCount, 1, mesh.firstIndex, 0, 0};
                tableOut[end] = blocks_.address + VkDeviceSize{end} * sizeof(Block);
                ++end;
            }
            countsOut[run] = end - first;
            vk.vkCmdBindPipeline(command_, VK_PIPELINE_BIND_POINT_GRAPHICS, tabled_[pipeline]);
            const TablePush push{table_.address, first, 0};
            vk.vkCmdPushConstants(command_, pipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), &push);
            vk.vkCmdDrawIndexedIndirectCount(command_, indirect_.buffer, VkDeviceSize{first} * sizeof(VkDrawIndexedIndirectCommand), counts_.buffer,
                                             VkDeviceSize{run} * sizeof(std::uint32_t), end - first, sizeof(VkDrawIndexedIndirectCommand));
            commands += 3;
            ++run;
            first = end;
        }
        return commands;
    }

    std::uint64_t RecordSorted(const Scene& scene) {
        std::uint64_t commands = 0;
        auto* commandsOut = reinterpret_cast<VkDrawIndexedIndirectCommand*>(indirect_.mapped);
        auto* tableOut = reinterpret_cast<std::uint64_t*>(table_.mapped);
        auto* countsOut = reinterpret_cast<std::uint32_t*>(counts_.mapped);
        const auto count = static_cast<std::uint32_t>(scene.items.size());
        groupCount_.assign(options_.pipelines, 0);
        for (const auto& item : scene.items) ++groupCount_[item.pipeline];
        groupStart_.assign(options_.pipelines, 0);
        std::uint32_t offset = 0;
        for (std::uint32_t pipeline = 0; pipeline < options_.pipelines; ++pipeline) {
            groupStart_[pipeline] = offset;
            offset += groupCount_[pipeline];
        }
        groupNext_ = groupStart_;
        for (std::uint32_t index = 0; index < count; ++index) {
            Pollute();
            const auto& item = scene.items[index];
            const auto slot = groupNext_[item.pipeline]++;
            const auto& mesh = meshes_[item.mesh];
            commandsOut[slot] = {mesh.indexCount, 1, mesh.firstIndex, 0, 0};
            tableOut[slot] = blocks_.address + VkDeviceSize{index} * sizeof(Block);
        }
        for (std::uint32_t pipeline = 0; pipeline < options_.pipelines; ++pipeline) {
            if (groupCount_[pipeline] == 0) continue;
            countsOut[pipeline] = groupCount_[pipeline];
            vk.vkCmdBindPipeline(command_, VK_PIPELINE_BIND_POINT_GRAPHICS, tabled_[pipeline]);
            const TablePush push{table_.address, groupStart_[pipeline], 0};
            vk.vkCmdPushConstants(command_, pipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), &push);
            vk.vkCmdDrawIndexedIndirectCount(command_, indirect_.buffer, VkDeviceSize{groupStart_[pipeline]} * sizeof(VkDrawIndexedIndirectCommand), counts_.buffer,
                                             VkDeviceSize{pipeline} * sizeof(std::uint32_t), groupCount_[pipeline], sizeof(VkDrawIndexedIndirectCommand));
            commands += 3;
        }
        return commands;
    }

    void WriteSequences(const Scene& scene) {
        auto* out = reinterpret_cast<Sequence*>(sequences_.mapped);
        const auto count = static_cast<std::uint32_t>(scene.items.size());
        for (std::uint32_t index = 0; index < count; ++index) {
            Pollute();
            const auto& item = scene.items[index];
            const auto& mesh = meshes_[item.mesh];
            const Sequence sequence{item.pipeline, 0, blocks_.address + VkDeviceSize{index} * sizeof(Block), {mesh.indexCount, 1, mesh.firstIndex, 0, 0}, 0};
            std::memcpy(out + index, &sequence, sizeof(sequence));
        }
    }

    VkGeneratedCommandsInfoEXT GeneratedInfo(std::uint32_t count, bool preprocessed) const {
        const auto& buffer = preprocessed ? preprocessExplicit_ : preprocess_;
        VkGeneratedCommandsInfoEXT info{VK_STRUCTURE_TYPE_GENERATED_COMMANDS_INFO_EXT};
        info.shaderStages = GraphicsStages;
        info.indirectExecutionSet = executionSet_;
        info.indirectCommandsLayout = preprocessed ? explicitLayout_ : implicitLayout_;
        info.indirectAddress = sequences_.address;
        info.indirectAddressSize = VkDeviceSize{count} * sizeof(Sequence);
        info.preprocessAddress = buffer.address;
        info.preprocessSize = buffer.size;
        info.maxSequenceCount = count;
        return info;
    }

    void BindGeneratedStart(VkCommandBuffer buffer) {
        const std::uint64_t zero = 0;
        vk.vkCmdBindPipeline(buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, bindable_.front());
        vk.vkCmdPushConstants(buffer, pipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(zero), &zero);
    }

    Sample Frame(Arm arm, const Scene& scene, std::uint64_t& commands, bool readback) {
        const auto count = static_cast<std::uint32_t>(scene.items.size());
        Sample sample;
        const auto start = Clock::now();
        Check(vk.vkResetCommandPool(device_, pool_, 0), "vkResetCommandPool");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        Check(vk.vkBeginCommandBuffer(command_, &begin), "vkBeginCommandBuffer");
        vk.vkCmdResetQueryPool(command_, queries_, 0, TimestampCount);
        commands = 0;
        VkGeneratedCommandsInfoEXT generated{};
        if (UsesGenerated(arm)) {
            WriteSequences(scene);
            generated = GeneratedInfo(count, arm == Arm::GeneratedPreprocessed);
        }
        if (arm == Arm::GeneratedPreprocessed) {
            BindGeneratedStart(command_);
            SetState(command_);
            if (options_.separateState) {
                Check(vk.vkBeginCommandBuffer(state_, &begin), "vkBeginCommandBuffer");
                BindGeneratedStart(state_);
                SetState(state_);
            }
            vk.vkCmdWriteTimestamp(command_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries_, 2);
            vk.vkCmdPreprocessGeneratedCommandsEXT(command_, &generated, options_.separateState ? state_ : command_);
            if (options_.separateState) Check(vk.vkEndCommandBuffer(state_), "vkEndCommandBuffer");
            VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
            barrier.srcStageMask = options_.fullBarrier ? VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT : VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_EXT;
            barrier.srcAccessMask = options_.fullBarrier ? VK_ACCESS_2_MEMORY_WRITE_BIT : VK_ACCESS_2_COMMAND_PREPROCESS_WRITE_BIT_EXT;
            barrier.dstStageMask = options_.fullBarrier ? VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT : VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
            barrier.dstAccessMask = options_.fullBarrier ? (VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT) : VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
            VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dependency.memoryBarrierCount = 1;
            dependency.pMemoryBarriers = &barrier;
            vk.vkCmdPipelineBarrier2(command_, &dependency);
            vk.vkCmdWriteTimestamp(command_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries_, 3);
            commands += 9;
        }
        std::array<VkImageMemoryBarrier2, 2> attachments{};
        attachments[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        attachments[0].srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_COPY_BIT;
        attachments[0].dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        attachments[0].dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        attachments[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        attachments[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        attachments[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        attachments[0].image = color_.image;
        attachments[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        attachments[1] = attachments[0];
        attachments[1].srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        attachments[1].dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        attachments[1].dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        attachments[1].newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        attachments[1].image = depth_.image;
        attachments[1].subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        VkDependencyInfo attachmentDependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        attachmentDependency.imageMemoryBarrierCount = static_cast<std::uint32_t>(attachments.size());
        attachmentDependency.pImageMemoryBarriers = attachments.data();
        vk.vkCmdPipelineBarrier2(command_, &attachmentDependency);
        vk.vkCmdWriteTimestamp(command_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries_, 0);
        VkRenderingAttachmentInfo colorAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        colorAttachment.imageView = color_.view;
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.clearValue.color = {{0.0f, 0.0f, 0.0f, 0.0f}};
        VkRenderingAttachmentInfo depthAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        depthAttachment.imageView = depth_.view;
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.clearValue.depthStencil = {1.0f, 0};
        VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
        rendering.renderArea = {{0, 0}, {options_.width, options_.height}};
        rendering.layerCount = 1;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachments = &colorAttachment;
        rendering.pDepthAttachment = &depthAttachment;
        vk.vkCmdBeginRendering(command_, &rendering);
        if (arm != Arm::GeneratedPreprocessed) SetState(command_);
        commands += 5;
        switch (arm) {
        case Arm::Inline:
            commands += RecordInline(scene);
            break;
        case Arm::IndirectRuns:
            commands += RecordRuns(scene);
            break;
        case Arm::IndirectSorted:
            commands += RecordSorted(scene);
            break;
        case Arm::Generated:
            BindGeneratedStart(command_);
            vk.vkCmdExecuteGeneratedCommandsEXT(command_, VK_FALSE, &generated);
            commands += 3;
            break;
        case Arm::GeneratedPreprocessed:
            vk.vkCmdExecuteGeneratedCommandsEXT(command_, VK_TRUE, &generated);
            commands += 1;
            break;
        }
        vk.vkCmdEndRendering(command_);
        vk.vkCmdWriteTimestamp(command_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries_, 1);
        if (readback) RecordReadback();
        Check(vk.vkEndCommandBuffer(command_), "vkEndCommandBuffer");
        const auto recorded = Clock::now();
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command_;
        Check(vk.vkQueueSubmit(queue_, 1, &submit, fence_), "vkQueueSubmit");
        const auto submitted = Clock::now();
        Check(vk.vkWaitForFences(device_, 1, &fence_, VK_TRUE, 10'000'000'000ull), "vkWaitForFences");
        Check(vk.vkResetFences(device_, 1, &fence_), "vkResetFences");
        std::array<std::uint64_t, TimestampCount> stamps{};
        const std::uint32_t used = arm == Arm::GeneratedPreprocessed ? 4u : 2u;
        Check(vk.vkGetQueryPoolResults(device_, queries_, 0, used, used * sizeof(std::uint64_t), stamps.data(), sizeof(std::uint64_t),
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT), "vkGetQueryPoolResults");
        const auto ticks = [&](std::uint32_t from, std::uint32_t to) {
            return static_cast<double>((stamps[to] - stamps[from]) & timestampMask_) * timestampPeriod_ / 1.0e6;
        };
        sample.recordMicro = std::chrono::duration<double, std::micro>(recorded - start).count();
        sample.submitMicro = std::chrono::duration<double, std::micro>(submitted - recorded).count();
        sample.cpuMicro = std::chrono::duration<double, std::micro>(submitted - start).count();
        sample.gpuMilli = ticks(0, 1);
        if (arm == Arm::GeneratedPreprocessed) {
            sample.preprocessMilli = ticks(2, 3);
            sample.gpuMilli += sample.preprocessMilli;
        }
        return sample;
    }

    void RecordReadback() {
        VkImageMemoryBarrier2 toTransfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        toTransfer.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        toTransfer.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        toTransfer.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        toTransfer.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        toTransfer.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransfer.image = color_.image;
        toTransfer.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkDependencyInfo before{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        before.imageMemoryBarrierCount = 1;
        before.pImageMemoryBarriers = &toTransfer;
        vk.vkCmdPipelineBarrier2(command_, &before);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {options_.width, options_.height, 1};
        vk.vkCmdCopyImageToBuffer(command_, color_.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback_.buffer, 1, &region);
        VkBufferMemoryBarrier2 toHost{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        toHost.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        toHost.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        toHost.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        toHost.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
        toHost.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toHost.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toHost.buffer = readback_.buffer;
        toHost.size = VK_WHOLE_SIZE;
        VkDependencyInfo after{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        after.bufferMemoryBarrierCount = 1;
        after.pBufferMemoryBarriers = &toHost;
        vk.vkCmdPipelineBarrier2(command_, &after);
    }

    CountResult RunCount(std::uint32_t count) {
        const auto scene = BuildScene(count, options_);
        std::memcpy(blocks_.mapped, scene.blocks.data(), scene.blocks.size() * sizeof(Block));
        CountResult result;
        result.draws = count;
        result.changes = scene.changes;
        result.runs = scene.runs;
        result.usedPipelines = scene.usedPipelines;
        result.polluteMicro = PolluteMicro(count);
        if (generated_) {
            result.preprocessBytes = GeneratedRequirements(implicitLayout_, count).size;
            result.preprocessExplicitBytes = GeneratedRequirements(explicitLayout_, count).size;
        }
        const auto armCount = options_.arms.size();
        std::vector<std::vector<Sample>> samples(armCount);
        std::vector<std::uint64_t> commands(armCount, 0);
        for (std::uint32_t frame = 0; frame < options_.warmup + options_.frames; ++frame) {
            for (std::size_t step = 0; step < armCount; ++step) {
                const auto armIndex = (step + frame) % armCount;
                const auto sample = Frame(options_.arms[armIndex], scene, commands[armIndex], false);
                if (frame >= options_.warmup) samples[armIndex].push_back(sample);
            }
        }
        std::vector<std::uint32_t> reference;
        for (std::size_t armIndex = 0; armIndex < armCount; ++armIndex) {
            std::uint64_t ignored = 0;
            Frame(options_.arms[armIndex], scene, ignored, true);
            std::vector<std::uint32_t> pixels(std::size_t{options_.width} * options_.height);
            std::memcpy(pixels.data(), readback_.mapped, pixels.size() * sizeof(std::uint32_t));
            ArmResult arm;
            arm.arm = options_.arms[armIndex];
            arm.commands = commands[armIndex];
            arm.hash = Fnv(pixels);
            arm.covered = static_cast<std::uint64_t>(std::count_if(pixels.begin(), pixels.end(), [](std::uint32_t pixel) { return (pixel >> 24) != 0; }));
            if (reference.empty()) reference = pixels;
            for (std::size_t pixel = 0; pixel < pixels.size(); ++pixel) arm.differing += pixels[pixel] != reference[pixel] ? 1u : 0u;
            const auto select = [&](auto member, double scale) {
                std::vector<double> values;
                for (const auto& sample : samples[armIndex]) values.push_back(sample.*member * scale);
                return Summarize(values);
            };
            arm.cpu = select(&Sample::cpuMicro, 1.0 / count);
            arm.record = select(&Sample::recordMicro, 1.0 / count);
            arm.submit = select(&Sample::submitMicro, 1.0);
            arm.gpu = select(&Sample::gpuMilli, 1.0);
            arm.preprocess = select(&Sample::preprocessMilli, 1.0);
            result.arms.push_back(arm);
        }
        return result;
    }

    const Options& options_;
    std::uint32_t maxDraws_ = 0;
    bool generated_ = false;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    std::uint32_t family_ = 0;
    float timestampPeriod_ = 1.0f;
    std::uint64_t timestampMask_ = ~0ull;
    VkPhysicalDeviceMemoryProperties memory_{};
    std::string deviceName_;
    std::string driverInfo_;
    std::vector<Mesh> meshes_;
    Buffer blocks_;
    Buffer indices_;
    Buffer table_;
    Buffer indirect_;
    Buffer counts_;
    Buffer sequences_;
    Buffer preprocess_;
    Buffer preprocessExplicit_;
    Buffer readback_;
    Image color_;
    Image depth_;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_ = VK_NULL_HANDLE;
    VkCommandBuffer state_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    VkQueryPool queries_ = VK_NULL_HANDLE;
    std::vector<VkShaderModule> modules_;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    std::vector<VkPipeline> pipelines_;
    std::vector<VkPipeline> direct_;
    std::vector<VkPipeline> tabled_;
    std::vector<VkPipeline> bindable_;
    VkIndirectExecutionSetEXT executionSet_ = VK_NULL_HANDLE;
    VkIndirectCommandsLayoutEXT implicitLayout_ = VK_NULL_HANDLE;
    VkIndirectCommandsLayoutEXT explicitLayout_ = VK_NULL_HANDLE;
    std::vector<std::uint32_t> groupCount_;
    std::vector<std::uint32_t> groupStart_;
    std::vector<std::uint32_t> groupNext_;
    std::vector<std::uint8_t> scratch_;
    std::size_t cursor_ = 0;
};

void LoadVulkan() {
#ifdef _WIN32
    const HMODULE library = LoadLibraryW(L"vulkan-1.dll");
    if (library == nullptr) throw std::runtime_error("vulkan-1.dll not found");
    vk.vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(reinterpret_cast<void*>(GetProcAddress(library, "vkGetInstanceProcAddr")));
#else
    void* library = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) throw std::runtime_error("libvulkan.so.1 not found");
    vk.vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(library, "vkGetInstanceProcAddr"));
#endif
    if (vk.vkGetInstanceProcAddr == nullptr) throw std::runtime_error("vkGetInstanceProcAddr not exported by the Vulkan loader");
    vk.vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(vk.vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance"));
    if (vk.vkCreateInstance == nullptr) throw std::runtime_error("missing vkCreateInstance");
}

void PinThread(int cpu) {
    if (cpu < 0) return;
#ifdef _WIN32
    if (cpu >= 64 || SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR{1} << cpu) == 0) throw std::runtime_error("cannot pin to cpu " + std::to_string(cpu));
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) throw std::runtime_error("cannot pin to cpu " + std::to_string(cpu));
#endif
}

}

int main(int argc, char** argv) {
    if (argc == 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
        std::cout << "agc_dgc_probe [--draws 3000,6000] [--arms inline,indirect-runs,indirect-sorted,dgc,dgc-pre] [--pipelines 16]\n"
                     "              [--frames 300] [--warmup 50] [--switch-percent 45] [--pollute BYTES] [--width 1920] [--height 1080]\n"
                     "              [--seed 1] [--cpu 2|none] [--device NAME] [--json FILE] [--pre-state same|separate]\n"
                     "              [--pre-barrier exact|full]\n"
                     "Records the same synthetic frame (one 256-byte block per draw, read by address) through inline commands,\n"
                     "vkCmdDrawIndexedIndirectCount per run of one pipeline or per pipeline, and VK_EXT_device_generated_commands\n"
                     "with a pipeline switch per sequence, then prints CPU microseconds per draw (recording and submission),\n"
                     "GPU milliseconds per frame (timestamps) and the pixels that differ from the first arm.\n";
        return 0;
    }
    try {
        const auto options = Parse(argc, argv);
        PinThread(options.cpu);
        LoadVulkan();
        std::vector<CountResult> results;
        std::string json;
        {
            Probe probe(options);
            results = probe.Run();
            probe.Print(results, std::cout);
            json = probe.Json(results);
        }
        if (!options.json.empty()) {
            std::ofstream output(options.json, std::ios::binary);
            output << json;
            output.close();
            if (!output) throw std::runtime_error("could not write " + options.json.string());
        }
        for (const auto& result : results) {
            for (const auto& arm : result.arms) {
                if (arm.differing != 0) {
                    std::cerr << "dgc probe: arm " << Name(arm.arm) << " differs from " << Name(result.arms.front().arm) << " on " << arm.differing << " pixels at " << result.draws << " draws\n";
                    return 2;
                }
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "dgc probe: " << error.what() << '\n';
        return 1;
    }
}
