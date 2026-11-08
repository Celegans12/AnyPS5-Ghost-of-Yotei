#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_PRESENTER_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_PRESENTER_H_

#include "common/common.h"

#include <memory>
#include <span>

namespace Libs::Graphics {

class CommandBuffer;
class RenderContext;
struct ImageInfo;
struct WindowContext;

class Presenter final {
public:
	struct Frame;
	struct Layer {
		Frame* frame;
		int    bus;
		bool   premultiplied_alpha;
	};

	explicit Presenter(WindowContext& window);
	~Presenter();
	KYTY_CLASS_NO_COPY(Presenter);

	[[nodiscard]] Frame&         PrepareFrame(CommandBuffer& command, const ImageInfo& info,
	                                          bool hdr = false, bool headset = false);
	[[nodiscard]] Frame&         PrepareBlankFrame(uint32_t width, uint32_t height, bool opaque,
	                                               CommandBuffer* producer = nullptr);
	[[nodiscard]] bool           PresentLastFrame();
	[[nodiscard]] bool           IsGuestPaused() const noexcept;
	[[nodiscard]] bool           IsHdrSupported() const noexcept;
	[[nodiscard]] bool           NeedsSystemOverlayRefresh() const noexcept;
	[[nodiscard]] RenderContext& Renderer() const noexcept;
	void                         Present(Frame& frame);
	void                         Present(std::span<const Layer> layers);
	void                         ClearLayer(int bus);
	void                         Discard(Frame& frame);
	// AnyPS5's frame dumps (kyty_gpu.dll): the pixels of a prepared frame whose producer finished, read back through the
	// present queue before it is presented, in the frame's format (a VkFormat of 4 or 8 bytes per pixel; others: false).
	using FrameReader = void (*)(void* context, const void* pixels, uint32_t width, uint32_t height, uint32_t row_bytes,
	                             uint32_t vk_format);
	[[nodiscard]] bool ReadFrame(Frame& frame, FrameReader reader, void* context);
	// The VkFormat of a prepared frame.
	[[nodiscard]] static uint32_t FrameFormat(const Frame& frame) noexcept;

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_PRESENTER_H_
