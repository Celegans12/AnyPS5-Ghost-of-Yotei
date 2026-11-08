// Kyty's VideoOutDriver in kyty_gpu.dll, over AnyPS5's VideoOut (the host table's flip_* calls). It replaces the driver
// part of Kyty's src/graphics/presentation/videoOut.cpp; the display-buffer description (DisplayImageInfo) follows
// BufferAttributeGroup::ImageInfo of that file (KytyPS5-Yotei v0.1.5, GPL-2.0-or-later).
//
// The flip is recorded where it is in the command stream, as on standalone Kyty: SubmitFlipFromGpu reserves the flip at
// AnyPS5 and records the copy of the display buffer into a presenter frame; CompleteFlip (after that GPU work) queues the
// frame and tells AnyPS5 the flip is ready; AnyPS5's presenter thread, at its vblank, asks for the frame (Present).

#include "aps5/aps5VideoOut.h"

#include "aps5/aps5Host.h"
#include "aps5/aps5Window.h"
#include "common/assert.h"
#include "common/threads.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/presentation/presenter.h"
#include "graphics/presentation/videoOut.h"
#include "libs/errno.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace Libs::VideoOut {

namespace {

constexpr int VIDEO_OUT_BUFFER_INDEX_BLANK = -1;

struct Request {
	Aps5GpuFlip                         flip {};
	uint64_t                            token   = 0;
	uint64_t                            address = 0;
	Graphics::Presenter::Frame*         frame   = nullptr;
	bool                                prepared = false;
};

struct ReadyFrame {
	Graphics::Presenter::Frame* frame   = nullptr;
	uint64_t                    address = 0;
	int32_t                     index   = 0;
};

} // namespace

struct VideoOutDriver::Impl {
	Impl(uint32_t width, uint32_t height, Graphics::Presenter& presenter)
	    : renderer(presenter.Renderer()), presenter(presenter), width(width), height(height) {}

	Graphics::RenderContext& renderer;
	Graphics::Presenter&     presenter;
	uint32_t                 width;
	uint32_t                 height;

	std::mutex                             requests_mutex;
	std::unordered_map<uint64_t, Request> requests;
	uint64_t                               next_id = 1;

	std::mutex             order_mutex;
	std::mutex             ready_mutex;
	std::deque<ReadyFrame> ready;

	std::atomic<uint64_t> flips {0};
	std::atomic<uint64_t> presents {0};
	std::atomic<uint64_t> mismatches {0};
	std::atomic<uint64_t> empty_presents {0};
	std::atomic<uint64_t> dump_failures {0};

	uint64_t Reserve(const Aps5GpuFlip& flip, int* status) {
		const auto& host  = Aps5::Host();
		uint64_t    token = 0;
		const int   result = host.flip_reserve(host.context, &flip, &token);
		if (result == APS5_GPU_FLIP_QUEUE_FULL) {
			*status = VIDEO_OUT_ERROR_FLIP_QUEUE_FULL;
			return 0;
		}
		if (result != APS5_GPU_OK) {
			Aps5::Fatal("VideoOut: AnyPS5 refused the flip of handle %u index %d (status %d)", flip.handle, flip.index,
			            result);
		}
		std::scoped_lock lock(requests_mutex);
		const auto       id = next_id++;
		requests.emplace(id, Request {flip, token, 0, nullptr, false});
		*status = OK;
		return id;
	}

	void Prepare(uint64_t request_id, Graphics::CommandBuffer& buffer) {
		Aps5GpuFlip flip {};
		{
			std::scoped_lock lock(requests_mutex);
			const auto       it = requests.find(request_id);
			if (it == requests.end() || it->second.prepared) {
				Aps5::Fatal("VideoOut: cannot prepare flip request %" PRIu64, request_id);
			}
			flip = it->second.flip;
		}
		Graphics::Presenter::Frame* frame   = nullptr;
		uint64_t                    address = 0;
		Aps5GpuDisplayBuffer        display {};
		const auto&                 host = Aps5::Host();
		if (flip.index >= 0 && host.display_buffer(host.context, flip.handle, flip.index, &display) == APS5_GPU_OK) {
			const auto info = DisplayImageInfo(display);
			frame           = &presenter.PrepareFrame(buffer, info, Graphics::IsHdrVideoOutPixelFormat(display.pixel_format), false);
			address         = display.address;
		} else {
			// A blank or black flip, or a buffer index with no registered buffer (AnyPS5 presents those as a clear).
			frame = &presenter.PrepareBlankFrame(width, height, flip.index != VIDEO_OUT_BUFFER_INDEX_BLANK, &buffer);
		}
		std::scoped_lock lock(requests_mutex);
		auto&            request = requests.at(request_id);
		request.frame            = frame;
		request.address          = address;
		request.prepared         = true;
	}

	void Complete(uint64_t request_id) {
		Request request;
		{
			std::scoped_lock lock(requests_mutex);
			const auto       it = requests.find(request_id);
			if (it == requests.end() || !it->second.prepared) {
				Aps5::Fatal("VideoOut: completed flip request %" PRIu64 " has no prepared frame", request_id);
			}
			request = it->second;
			requests.erase(it);
		}
		// The frame is queued before AnyPS5 learns the flip is ready, and in the order AnyPS5 queues its flips.
		std::scoped_lock order(order_mutex);
		{
			std::scoped_lock lock(ready_mutex);
			ready.push_back({request.frame, request.address, request.flip.index});
		}
		flips.fetch_add(1, std::memory_order_relaxed);
		const auto& host = Aps5::Host();
		host.flip_ready(host.context, request.token);
	}

	// BufferAttributeGroup::ImageInfo of Kyty's videoOut.cpp, from AnyPS5's description of the registered buffer.
	static Graphics::ImageInfo DisplayImageInfo(const Aps5GpuDisplayBuffer& display) {
		const bool compressed  = display.dcc_address != 0;
		const auto compression = Graphics::ClassifyVideoOutCompression(compressed, display.dcc_address, display.dcc_control,
		                                                               display.dcc_clear_color);
		if (display.width == 0 || display.height == 0 || display.width > 16384 || display.height > 16384 ||
		    display.tiling_mode != 0 || display.address == 0 || compression == Graphics::VideoOutCompression::Unsupported) {
			Aps5::Fatal("VideoOut: unsupported display buffer 0x%" PRIx64 " %ux%u tiling %u, DCC 0x%" PRIx64
			            " control 0x%x clear 0x%" PRIx64,
			            display.address, display.width, display.height, display.tiling_mode, display.dcc_address,
			            display.dcc_control, display.dcc_clear_color);
		}
		Graphics::VideoOutPixelFormatInfo pixel_format {};
		if (!Graphics::DecodeVideoOutPixelFormat(display.pixel_format, pixel_format)) {
			Aps5::Fatal("VideoOut: unsupported display pixel format 0x%016" PRIx64, display.pixel_format);
		}
		const auto          tile_mode = Graphics::Prospero::TileMode::kRenderTarget;
		Graphics::ImageInfo info {};
		info.data            = {display.address, 0};
		info.pixel_format    = pixel_format.format;
		info.guest_format    = pixel_format.guest_format;
		info.type            = Graphics::Prospero::ImageType::kColor2D;
		info.extent          = vk::Extent3D {display.width, display.height, 1};
		info.resources       = {1, 1};
		info.bytes_per_block = pixel_format.bytes_per_element;
		info.samples         = 1;
		info.tile_mode       = tile_mode;
		info.bgra16          = pixel_format.bgra16;
		info.UpdateSize();
		if (info.data.size == 0 || info.tiling.block.block_size != 65536 || (display.address & 65535u) != 0) {
			Aps5::Fatal("VideoOut: invalid display buffer footprint or alignment at 0x%" PRIx64, display.address);
		}
		if (compression != Graphics::VideoOutCompression::Uncompressed) {
			Graphics::TileSizeAlign dcc_size {};
			if (!Graphics::TileGetDccSize(display.width, display.height, 1, pixel_format.bytes_per_element, 1, tile_mode,
			                              dcc_size)) {
				Aps5::Fatal("VideoOut: invalid display buffer DCC footprint at 0x%" PRIx64, display.address);
			}
			info.metadata.range       = {display.dcc_address, dcc_size.size};
			info.metadata.kind        = Graphics::ImageMetadataKind::Dcc;
			info.metadata.control     = display.dcc_control;
			info.metadata.compression = compression;
		}
		Graphics::ImageOps::Validate(info);
		if (!Graphics::IsSupportedVideoOutFormat(info)) {
			Aps5::Fatal("VideoOut: unsupported normalized display format at 0x%" PRIx64, display.address);
		}
		return info;
	}
};

VideoOutDriver::VideoOutDriver(uint32_t width, uint32_t height, Graphics::Presenter& presenter)
    : m_impl(std::make_unique<Impl>(width, height, presenter)) {}

VideoOutDriver::~VideoOutDriver() = default;

VideoOutDriver::Impl& VideoOutDriver::State() noexcept {
	return *m_impl;
}

int VideoOutDriver::SubmitFlipFromGpu(Graphics::CommandBuffer& buffer, int handle, int index, int flip_mode, int64_t flip_arg,
                                      uint64_t& request_id) {
	EXIT_IF(buffer.IsInvalid());
	const Aps5GpuFlip flip {static_cast<uint32_t>(handle), index, static_cast<uint32_t>(flip_mode), 0, flip_arg};
	int               status = OK;
	request_id               = m_impl->Reserve(flip, &status);
	if (status != OK) {
		return status;
	}
	m_impl->Prepare(request_id, buffer);
	return OK;
}

void VideoOutDriver::SubmitFlipPreparation(uint64_t request_id) {
	m_impl->renderer.GetGpu().SubmitFlipPreparation(request_id);
}

void VideoOutDriver::PrepareFlip(uint64_t request_id, Graphics::CommandBuffer& buffer) {
	m_impl->Prepare(request_id, buffer);
}

void VideoOutDriver::CompleteFlip(uint64_t request_id) {
	m_impl->Complete(request_id);
}

void VideoOutDriver::WaitForSubmitSlot(int handle) {
	const auto& host = Aps5::Host();
	host.flip_wait_room(host.context, static_cast<uint32_t>(handle));
}

void VideoOutDriver::WaitFlipDone(int handle, int index) {
	const auto& host = Aps5::Host();
	host.rendering_wait(host.context, static_cast<uint32_t>(handle), index);
}

// KYTY_UPSTREAM_FLIP_SUSPEND=1 only (off): AnyPS5's rendering wait blocks until the buffer is safe, so it is safe after it.
bool VideoOutDriver::IsSafeForRendering(int handle, int index) {
	WaitFlipDone(handle, index);
	return true;
}

namespace {
std::unique_ptr<VideoOutDriver> g_driver;
} // namespace

} // namespace Libs::VideoOut

namespace Aps5 {

Libs::VideoOut::VideoOutDriver& VideoOutCreate(Libs::Graphics::Presenter& presenter, uint32_t width, uint32_t height) {
	EXIT_IF(Libs::VideoOut::g_driver != nullptr);
	Libs::VideoOut::g_driver = std::make_unique<Libs::VideoOut::VideoOutDriver>(width, height, presenter);
	return *Libs::VideoOut::g_driver;
}

int VideoOutSubmitCpuFlip(const Aps5GpuFlip& flip) {
	EXIT_IF(Libs::VideoOut::g_driver == nullptr);
	auto& driver = *Libs::VideoOut::g_driver;
	for (;;) {
		int        status = OK;
		const auto id     = driver.State().Reserve(flip, &status);
		if (status == OK) {
			driver.SubmitFlipPreparation(id);
			return APS5_GPU_OK;
		}
		driver.WaitForSubmitSlot(static_cast<int>(flip.handle));
	}
}

namespace {

struct DumpContext {
	int32_t index = -1;
	bool    done  = false;
};

float HalfToFloat(uint16_t half) {
	const uint32_t sign     = (half >> 15u) & 1u;
	const uint32_t exponent = (half >> 10u) & 0x1fu;
	const uint32_t mantissa = half & 0x3ffu;
	float          value    = 0.0f;
	if (exponent == 0) {
		value = std::ldexp(static_cast<float>(mantissa), -24);
	} else if (exponent == 31) {
		value = mantissa == 0 ? INFINITY : 0.0f;
	} else {
		value = std::ldexp(static_cast<float>(mantissa | 0x400u), static_cast<int>(exponent) - 25);
	}
	return sign != 0 ? -value : value;
}

uint8_t Unit8(float value) {
	return static_cast<uint8_t>(std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f));
}

// The frame in BGRA8, as AnyPS5's frame dumps want it: 8-bit formats as they are (channels swapped for RGBA), 10-bit
// channels by their top 8 bits, 16-bit ones scaled (a float channel clamped to [0, 1]).
void DumpFrame(void* context, const void* pixels, uint32_t width, uint32_t height, uint32_t row_bytes, uint32_t vk_format) {
	auto*                dump   = static_cast<DumpContext*>(context);
	const auto           format = static_cast<VkFormat>(vk_format);
	std::vector<uint8_t> out(static_cast<size_t>(width) * height * 4u);
	for (uint32_t y = 0; y < height; y++) {
		const auto* row = static_cast<const uint8_t*>(pixels) + static_cast<size_t>(y) * row_bytes;
		auto*       to  = out.data() + static_cast<size_t>(y) * width * 4u;
		for (uint32_t x = 0; x < width; x++, to += 4) {
			uint8_t r = 0;
			uint8_t g = 0;
			uint8_t b = 0;
			uint8_t a = 255;
			if (format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB) {
				b = row[x * 4u + 0u], g = row[x * 4u + 1u], r = row[x * 4u + 2u], a = row[x * 4u + 3u];
			} else if (format == VK_FORMAT_R8G8B8A8_UNORM || format == VK_FORMAT_R8G8B8A8_SRGB) {
				r = row[x * 4u + 0u], g = row[x * 4u + 1u], b = row[x * 4u + 2u], a = row[x * 4u + 3u];
			} else if (format == VK_FORMAT_A2R10G10B10_UNORM_PACK32 || format == VK_FORMAT_A2B10G10R10_UNORM_PACK32) {
				uint32_t v = 0;
				std::memcpy(&v, row + x * 4u, sizeof(v));
				const auto low  = static_cast<uint8_t>((v >> 2u) & 0xffu);
				const auto mid  = static_cast<uint8_t>((v >> 12u) & 0xffu);
				const auto high = static_cast<uint8_t>((v >> 22u) & 0xffu);
				g               = mid;
				r               = format == VK_FORMAT_A2R10G10B10_UNORM_PACK32 ? high : low;
				b               = format == VK_FORMAT_A2R10G10B10_UNORM_PACK32 ? low : high;
				a               = static_cast<uint8_t>(((v >> 30u) & 3u) * 85u);
			} else {
				uint16_t c[4] {};
				std::memcpy(c, row + x * 8u, sizeof(c));
				if (format == VK_FORMAT_R16G16B16A16_SFLOAT) {
					r = Unit8(HalfToFloat(c[0])), g = Unit8(HalfToFloat(c[1])), b = Unit8(HalfToFloat(c[2])),
					a = Unit8(HalfToFloat(c[3]));
				} else {
					r = static_cast<uint8_t>(c[0] >> 8u), g = static_cast<uint8_t>(c[1] >> 8u),
					b = static_cast<uint8_t>(c[2] >> 8u), a = static_cast<uint8_t>(c[3] >> 8u);
				}
			}
			to[0] = b, to[1] = g, to[2] = r, to[3] = a;
		}
	}
	const auto& host = Host();
	host.frame_ready(host.context, dump->index, out.data(), width, height, width * 4u);
	dump->done = true;
}

} // namespace

int VideoOutPresent(const Aps5GpuWindow* window, const Aps5GpuDisplayBuffer* buffer, int32_t opaque, int32_t dump_index,
                    void (*presented)(void* context), void* presented_context) {
	EXIT_IF(Libs::VideoOut::g_driver == nullptr);
	auto&                          impl = Libs::VideoOut::g_driver->State();
	Libs::VideoOut::ReadyFrame     next {};
	bool                           have = false;
	{
		std::scoped_lock lock(impl.ready_mutex);
		if (!impl.ready.empty()) {
			next = impl.ready.front();
			impl.ready.pop_front();
			have = true;
		}
	}
	if (have && (buffer != nullptr ? buffer->address : 0) != next.address &&
	    impl.mismatches.fetch_add(1, std::memory_order_relaxed) < 8) {
		Log("videoout: presented frame of flip index %d (buffer 0x%llx) for AnyPS5's present of buffer 0x%llx", next.index,
		    static_cast<unsigned long long>(next.address),
		    static_cast<unsigned long long>(buffer != nullptr ? buffer->address : 0));
	}
	if (!have) {
		if (impl.empty_presents.fetch_add(1, std::memory_order_relaxed) < 8) {
			Log("videoout: present with no prepared flip; showing a %s frame", opaque != 0 ? "black" : "clear");
		}
		next.frame = &impl.presenter.PrepareBlankFrame(window != nullptr ? window->width : impl.width,
		                                               window != nullptr ? window->height : impl.height, opaque != 0, nullptr);
	}
	WindowRefreshSize();
	if (dump_index >= 0) {
		DumpContext dump {dump_index, false};
		if (!impl.presenter.ReadFrame(*next.frame, &DumpFrame, &dump) &&
		    impl.dump_failures.fetch_add(1, std::memory_order_relaxed) < 4) {
			Log("videoout: frame dump %d not written: frame format %u cannot be read back", dump_index,
			    Libs::Graphics::Presenter::FrameFormat(*next.frame));
		}
	}
	impl.presenter.Present(*next.frame);
	impl.presents.fetch_add(1, std::memory_order_relaxed);
	presented(presented_context);
	return APS5_GPU_OK;
}

} // namespace Aps5
