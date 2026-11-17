#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_UPSCALER_UPSCALER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_UPSCALER_UPSCALER_H_

#include "graphics/host_gpu/renderer/upscaler/dlss.h"

#include <memory>

namespace Libs::Graphics::Upscaler {

class Feature {
public:
	virtual ~Feature()                                                    = default;
	[[nodiscard]] virtual bool Evaluate(vk::CommandBuffer command, const Pssr::Request& request,
	                                    Dlss::Inputs& inputs, bool reset) = 0;
};

[[nodiscard]] bool                          Enabled();
[[nodiscard]] std::optional<Dlss::Creation> Choose(const Pssr::Request& request,
                                                   const Dlss::Inputs&  inputs);
[[nodiscard]] std::unique_ptr<Feature> Create(vk::CommandBuffer     command,
                                              const Dlss::Creation& creation);

}

#endif
