#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_UPSCALER_FSR_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_UPSCALER_FSR_H_

#include "graphics/host_gpu/renderer/upscaler/dlss.h"

struct ffxCreateContextDescUpscale;
struct ffxDispatchDescUpscale;

namespace Libs::Graphics::Fsr {

struct Context;

[[nodiscard]] bool Reads(const Pssr::Request& request, const Dlss::Inputs& inputs);
[[nodiscard]] std::optional<Dlss::Creation> Choose(const Pssr::Request& request,
                                                   const Dlss::Inputs&  inputs);
void SetCreation(ffxCreateContextDescUpscale& desc, const Dlss::Creation& creation);
void SetDispatch(ffxDispatchDescUpscale& desc, const Pssr::Request& request,
                 const Dlss::Inputs& inputs, bool reset, float frame_time);

void Load();
void Init(const GraphicContext& graphics, PFN_vkGetDeviceProcAddr get_device_proc_addr);
void               Shutdown();
[[nodiscard]] bool Enabled();

[[nodiscard]] Context* CreateContext(const Dlss::Creation& creation);
[[nodiscard]] bool Evaluate(vk::CommandBuffer command, Context* context,
                            const Pssr::Request& request, const Dlss::Inputs& inputs, bool reset);
void               ReleaseContext(Context* context);

}

#endif
