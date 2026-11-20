#include "prx/libSceAgcDriver/KytyShim/src/Runtime.hpp"

#include "common/assert.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/pipeline/imageHeap.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/window.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <spirv-tools/libspirv.hpp>
#include <string_view>

namespace {

using namespace Libs::Graphics;

struct ReplayResources {
    GraphicContext graphics;
    ShaderModule module;
    PipelineCache::Pipeline pipeline;
    vk::PipelineCache cache;

    ~ReplayResources() {
        if (graphics.device) {
            const auto device = graphics.device;
            device.destroyPipeline(pipeline.pipeline, nullptr);
            device.destroyPipelineLayout(pipeline.pipeline_layout, nullptr);
            device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout, nullptr);
            device.destroyDescriptorSetLayout(graphics.image_heap_layout, nullptr);
            device.destroyPipelineCache(cache, nullptr);
            device.destroyShaderModule(module.module, nullptr);
            device.destroy(nullptr);
        }
        if (graphics.instance) graphics.instance.destroy(nullptr);
    }
};

int ReplayComputePipeline(const std::filesystem::path& store, uint64_t moduleId,
                          const std::filesystem::path& output) {
    std::error_code fileError;
    if (!std::filesystem::is_regular_file(store, fileError)) {
        std::fputs("Expected a regular Kyty shader store file\n", stderr);
        return 2;
    }
    std::ifstream input(store, std::ios::binary);
    if (!input) {
        std::fputs("Could not open Kyty shader store\n", stderr);
        return 2;
    }

    Common::SoftExitScope softExit;
    KytyShim::StartRuntime();
    ReplayResources resources;
    auto& graphics = resources.graphics;
    std::string error;
    if (!CreateHeadlessVulkan(graphics, nullptr, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 3;
    }
    KytyShim::ReportNativeRayTracing(graphics);
    auto& module = resources.module;
    std::vector<uint32_t> spirv;
    if (!LoadStoredComputeModule(input, ComputeShaderStoreHeader(graphics), moduleId, module, spirv, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 2;
    }
    spvtools::SpirvTools validator(SPV_ENV_VULKAN_1_3);
    validator.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
        std::fprintf(stderr, "%s\n", message);
    });
    spvtools::ValidatorOptions validatorOptions;
    validatorOptions.SetUniformBufferStandardLayout(true);
    if (!validator.Validate(spirv.data(), spirv.size(), validatorOptions)) {
        std::fputs("The selected compute shader contains invalid SPIR-V\n", stderr);
        return 2;
    }

    const vk::PipelineCacheCreateInfo cacheInfo {};
    if (graphics.device.createPipelineCache(&cacheInfo, nullptr, &resources.cache) != vk::Result::eSuccess) {
        std::fputs("Could not create an empty Vulkan pipeline cache\n", stderr);
        return 3;
    }
    if (module.info.info.uses_image_heap) graphics.image_heap_layout = ImageHeap::CreateLayout(graphics.device);
    module.module = CompileSPV(spirv, graphics.device);
    std::chrono::nanoseconds elapsed {};
    const auto result = CreateComputePipeline(graphics, resources.pipeline, module, resources.cache, false, &elapsed);
    const auto gpu = GetVulkanDeviceIdentity(graphics);
    const char* diskCache = std::getenv("__GL_SHADER_DISK_CACHE");
    const auto policy = diskCache == nullptr ? std::string_view {} : std::string_view(diskCache);
    const auto setting = diskCache == nullptr ? "unset" : policy == "0" || policy == "1" ? diskCache : "other";
#ifdef _WIN32
    auto* report = _wfopen(output.c_str(), L"wbx");
#else
    auto* report = std::fopen(output.c_str(), "wbx");
#endif
    if (report == nullptr) {
        std::fputs("Could not create a new replay report\n", stderr);
        return 2;
    }
    const auto written = std::fprintf(report,
        "{\"schema\":1,\"module_id\":\"%" PRIu64 "\",\"spirv_words\":%u,"
        "\"vendor_id\":%u,\"device_id\":%u,\"driver_version\":%u,"
        "\"driver_create_ns\":%" PRId64 ",\"vk_result\":%d,"
        "\"vulkan_cache\":\"empty\",\"nvidia_disk_cache_env\":\"%s\"}\n",
        module.id, module.spirv_words, gpu.vendor_id, gpu.device_id, gpu.driver_version,
        static_cast<int64_t>(elapsed.count()), static_cast<int>(result), setting);
    if (std::fclose(report) != 0 || written < 0) {
        std::fputs("Could not write the replay report\n", stderr);
        return 2;
    }
    return result == vk::Result::eSuccess && resources.pipeline.pipeline ? 0 : 3;
}


}

extern "C" int AgcDriverKytyReplayComputePipeline_nid_postfix(
    const std::filesystem::path::value_type* store, uint64_t moduleId,
    const std::filesystem::path::value_type* output) {
    if (store == nullptr || *store == 0 || moduleId == 0 || output == nullptr || *output == 0) return 64;
    try {
        return ReplayComputePipeline(store, moduleId, output);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 3;
    }
}
