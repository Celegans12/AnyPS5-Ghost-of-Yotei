#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_WINDOWINTERNAL_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_WINDOWINTERNAL_H_

#if defined(ANYPS5_KYTY)
struct SDL_Window;
struct SDL_WindowEvent;
struct SDL_DisplayEvent;
#else
#include <SDL3/SDL.h>
#endif

#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace Libs::Graphics {

class Presenter;
class RenderContext;

struct SurfaceCapabilities {
	vk::SurfaceCapabilitiesKHR        capabilities {};
	std::vector<vk::SurfaceFormatKHR> formats;
	std::vector<vk::PresentModeKHR>   present_modes;
};

struct WindowLoopState {
#if !defined(ANYPS5_KYTY)
	SDL_Event             event {};
#endif
	bool                  need_exit = false;
	std::atomic<uint64_t> presented_frames {0};
};

struct WindowContext {
	WindowContext();
	~WindowContext();
	KYTY_CLASS_NO_COPY(WindowContext);

	[[nodiscard]] static vk::PhysicalDeviceVulkan11Features RequiredVulkan11Features() noexcept;
	[[nodiscard]] static vk::PhysicalDeviceVulkan12Features RequiredVulkan12Features() noexcept;
	[[nodiscard]] static vk::PhysicalDeviceVulkan13Features RequiredVulkan13Features() noexcept;
	[[nodiscard]] static uint32_t InitialWindowFlags(bool fullscreen) noexcept;
	void                                                    CreateVulkan();
	void                                                    RecreateSurface();
	void                                                    RefreshSurfaceCapabilities();
	void                                                    UpdateIcon();
	void UpdateTitle(uint64_t frame_num, uint64_t host_frames, double current_fps);
	void                                                    Resize(int width, int height);
	void ProcessWindowEvent(const SDL_WindowEvent& event);
	void ProcessDisplayEvent(const SDL_DisplayEvent& event);
	void ProcessEvent(double time_seconds);
	void Run();

	GraphicContext                 graphic_ctx;
	SDL_Window*                    window        = nullptr;
	vk::SurfaceKHR                 surface       = nullptr;
	SurfaceCapabilities            surface_capabilities;
	std::unique_ptr<RenderContext> render_context;
	std::unique_ptr<Presenter>     presenter;
	WindowLoopState                loop;

	Common::Mutex mutex;
#if defined(ANYPS5_KYTY)
	PFN_vkGetInstanceProcAddr host_get_instance_proc_addr = nullptr;
	std::vector<const char*>  host_extensions;
	void*                     host_context = nullptr;
	VkSurfaceKHR (*host_create_surface)(void* context, VkInstance instance) = nullptr;
#endif
};

}

#endif
