#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_DEBUGSAMPLE_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_DEBUGSAMPLE_H_

#include "common/envRead.h"
#include "common/onceValue.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/image/image.h"

#include <atomic>
#include <cstdlib>
#include <mutex>
#include <utility>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

// KYTY_TRACE_FLIPS diagnostic: copies a 16x16 grid of texels of an image to host memory and,
// at the next sample point (after the GPU is idle), prints how many are nonzero.
struct DebugSample {
	vk::Buffer    buffer;
	VmaAllocation allocation = nullptr;
	void*         mapped     = nullptr;
	uint32_t      texel_bytes = 0;
	std::string   tag;
	std::string   dump_path;  // whole-image dump instead of grid statistics
	uint64_t      dump_size = 0;
	uint32_t      generation = 0;  // presented-frame count when queued
};
inline std::vector<DebugSample>& DebugSamples() {
	static std::vector<DebugSample> samples;
	return samples;
}
inline std::atomic<uint32_t>& DebugPresentCount();
// A sample's copy is recorded in the command buffer that is open when it is queued. That buffer
// may still be unsubmitted at the next present, so a sample is read one present later (after the
// buffer was submitted and the device went idle); reading it earlier returned zeros or stale data.
inline void DebugPushSample(DebugSample&& sample) {
	sample.generation = DebugPresentCount().load();
	DebugSamples().push_back(std::move(sample));
}

// KYTY_CAPTURE_BUDGET_MB (default 2048): bytes of whole-image dumps queued until the next read;
// later dumps are skipped and counted. An unbounded capture copied ~14 GiB within one frame and
// ended in a lost device. 0 = no image dumps besides the presented frame (a listing capture).
inline uint64_t& DebugDumpQueuedBytes() {
	static uint64_t bytes = 0;
	return bytes;
}
inline uint32_t& DebugDumpSkipped() {
	static uint32_t skipped = 0;
	return skipped;
}
inline bool DebugDumpWithinBudget(uint64_t bytes) {
	static const uint64_t budget = [] {
		const char* value = std::getenv("KYTY_CAPTURE_BUDGET_MB");
		return (value != nullptr ? std::strtoull(value, nullptr, 10) : 2048ull) << 20u;
	}();
	auto& queued = DebugDumpQueuedBytes();
	if (queued + bytes > budget) {
		DebugDumpSkipped()++;
		return false;
	}
	queued += bytes;
	return true;
}

inline void DebugReadSamples(GraphicContext& graphics) {
	if (DebugDumpSkipped() != 0u) {
		std::fprintf(stderr, "CAPTURE_BUDGET skipped=%u dumps queued_bytes=%llu\n", DebugDumpSkipped(),
		             static_cast<unsigned long long>(DebugDumpQueuedBytes()));
	}
	DebugDumpQueuedBytes() = 0;
	DebugDumpSkipped()     = 0;
	if (DebugSamples().empty()) return;
	const auto               present = DebugPresentCount().load();
	std::vector<DebugSample> pending;
	std::vector<DebugSample> ready;
	for (auto& sample: DebugSamples()) {
		(sample.generation + 1u < present ? ready : pending).push_back(std::move(sample));
	}
	DebugSamples() = std::move(pending);
	if (ready.empty()) return;
	// vkDeviceWaitIdle uses every queue: the recording thread must have made its pending submissions first (it submits
	// the render scheduler's buffers through its worker; Vulkan validation 17.10: Threading-MultipleThreads-
	// Write on the VkQueue, DeviceLost in test runs that dump frames).
	CommandRecorder::SyncQueue();
	(void)graphics.device.waitIdle();
	for (auto& sample: ready) {
		const auto* bytes = static_cast<const uint8_t*>(sample.mapped);
		if (!sample.dump_path.empty()) {
			if (FILE* file = std::fopen(sample.dump_path.c_str(), "wb")) {
				std::fwrite(bytes, 1, sample.dump_size, file);
				std::fclose(file);
				std::fprintf(stderr, "FRAME_DUMP %s\n", sample.dump_path.c_str());
			}
			vmaDestroyBuffer(graphics.allocator, sample.buffer, sample.allocation);
			continue;
		}
		uint32_t nonzero = 0;
		uint64_t sum     = 0;
		for (uint32_t t = 0; t < 256u; ++t) {
			bool any = false;
			for (uint32_t b = 0; b < sample.texel_bytes; ++b) {
				any |= bytes[t * sample.texel_bytes + b] != 0;
				sum += bytes[t * sample.texel_bytes + b];
			}
			nonzero += any ? 1u : 0u;
		}
		uint32_t first = 0;
		std::memcpy(&first, bytes + 128u * sample.texel_bytes, std::min(4u, sample.texel_bytes));
		std::fprintf(stderr, "SAMPLE %s nonzero=%u/256 mean_byte=%.2f center=0x%08x\n", sample.tag.c_str(),
		             nonzero, static_cast<double>(sum) / (256.0 * sample.texel_bytes), first);
		vmaDestroyBuffer(graphics.allocator, sample.buffer, sample.allocation);
	}
	std::fflush(stderr);
}

inline void DebugQueueSample(GraphicContext& graphics, vk::CommandBuffer command, Image& image,
                      const std::string& tag) {
	const auto format = image.backing.format;
	const auto name   = vk::to_string(format);
	if (image.backing.image == nullptr || image.backing.samples != 1u ||
	    name.find("Block") != std::string::npos) {
		std::fprintf(stderr, "SAMPLE %s skipped (format %s)\n", tag.c_str(), name.c_str());
		return;
	}
	vk::ImageAspectFlags aspect = !name.empty() && name[0] == 'D' ? vk::ImageAspectFlagBits::eDepth
	                              : name == "S8Uint"              ? vk::ImageAspectFlagBits::eStencil
	                                                              : vk::ImageAspectFlagBits::eColor;
	uint32_t texel_bytes = image.info.bytes_per_block;
	if (aspect & vk::ImageAspectFlagBits::eDepth) {
		aspect      = vk::ImageAspectFlagBits::eDepth;
		texel_bytes = format == vk::Format::eD16Unorm || format == vk::Format::eD16UnormS8Uint ? 2u : 4u;
	} else if (aspect & vk::ImageAspectFlagBits::eStencil) {
		aspect      = vk::ImageAspectFlagBits::eStencil;
		texel_bytes = 1u;
	}
	if (texel_bytes == 0u || texel_bytes > 16u) return;
	vk::BufferCreateInfo buffer_info {};
	buffer_info.size  = 256u * texel_bytes;
	buffer_info.usage = vk::BufferUsageFlagBits::eTransferDst;
	VmaAllocationCreateInfo allocation_info {};
	allocation_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
	                        VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
	allocation_info.usage = VMA_MEMORY_USAGE_AUTO;
	VmaAllocationInfo result {};
	VkBuffer          native = VK_NULL_HANDLE;
	DebugSample       sample;
	if (vmaCreateBuffer(graphics.allocator, reinterpret_cast<const VkBufferCreateInfo*>(&buffer_info),
	                    &allocation_info, &native, &sample.allocation, &result) != VK_SUCCESS) {
		return;
	}
	sample.buffer      = native;
	sample.mapped      = result.pMappedData;
	sample.texel_bytes = texel_bytes;
	sample.tag         = tag + " fmt=" + name + " extent=" + std::to_string(image.backing.extent.width) + "x" +
	             std::to_string(image.backing.extent.height);
	image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {}, command);
	std::vector<vk::BufferImageCopy> copies;
	const auto w = image.backing.extent.width;
	const auto h = image.backing.extent.height;
	for (uint32_t y = 0; y < 16u; ++y) {
		for (uint32_t x = 0; x < 16u; ++x) {
			vk::BufferImageCopy copy {};
			copy.bufferOffset     = (y * 16u + x) * texel_bytes;
			copy.imageSubresource = {aspect, 0, 0, 1};
			copy.imageOffset      = vk::Offset3D {static_cast<int32_t>((x * 2u + 1u) * w / 32u),
			                                      static_cast<int32_t>((y * 2u + 1u) * h / 32u), 0};
			copy.imageExtent      = vk::Extent3D {1, 1, 1};
			copies.push_back(copy);
		}
	}
	command.copyImageToBuffer(image.backing.image, vk::ImageLayout::eTransferSrcOptimal, sample.buffer,
	                          static_cast<uint32_t>(copies.size()), copies.data());
	DebugPushSample(std::move(sample));
}

// Capture frame: a GPU buffer range copied to a host-visible buffer, written to `path` at present.
inline void DebugQueueBufferDump(GraphicContext& graphics, vk::CommandBuffer command, vk::Buffer source,
                                 uint64_t offset, uint64_t size, const std::string& path) {
	vk::BufferCreateInfo buffer_info {};
	buffer_info.size  = size;
	buffer_info.usage = vk::BufferUsageFlagBits::eTransferDst;
	VmaAllocationCreateInfo allocation_info {};
	allocation_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
	allocation_info.usage = VMA_MEMORY_USAGE_AUTO;
	VmaAllocationInfo result {};
	VkBuffer          native = VK_NULL_HANDLE;
	DebugSample       sample;
	if (vmaCreateBuffer(graphics.allocator, reinterpret_cast<const VkBufferCreateInfo*>(&buffer_info),
	                    &allocation_info, &native, &sample.allocation, &result) != VK_SUCCESS) {
		return;
	}
	sample.buffer    = native;
	sample.mapped    = result.pMappedData;
	sample.dump_size = size;
	sample.dump_path = path;
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {}, 1,
	                        &barrier, 0, nullptr, 0, nullptr);
	const vk::BufferCopy copy {offset, 0, size};
	command.copyBuffer(source, sample.buffer, 1, &copy);
	DebugPushSample(std::move(sample));
}

// KYTY_TRACE_FRAME_DIR: the whole presented image (mip 0, layer 0) as raw texels.
inline void DebugQueueFullDump(GraphicContext& graphics, vk::CommandBuffer command, Image& image,
                        const std::string& path, bool budgeted = true) {
	uint32_t   texel_bytes = image.info.bytes_per_block;
	const auto w           = image.backing.extent.width;
	const auto h           = image.backing.extent.height;
	// 3D images dump every slice (slices follow each other in the file).
	const auto slices      = std::max(1u, image.backing.extent.depth);
	// 2D array images dump every layer (at most 256), one after another.
	const auto layers      = slices > 1u ? 1u : std::min(256u, std::max(1u, image.info.resources.layers));
	if (image.backing.samples != 1u) return;
	// Color images copy the color aspect; depth images copy only the depth aspect.
	const auto format_name = vk::to_string(image.backing.format);
	if (format_name.empty() || format_name == "S8Uint") {
		return;
	}
	// Block-compressed 2D images (BCn: 4x4 texel blocks) dump their raw blocks, one layer, no mips.
	const bool blocks = format_name.find("Block") != std::string::npos;
	if (blocks && (slices > 1u || layers > 1u || format_name.find("Bc") != 0)) {
		return;
	}
	const bool depth = format_name[0] == 'D';
	if (depth) {
		texel_bytes = image.backing.format == vk::Format::eD16Unorm ||
		                      image.backing.format == vk::Format::eD16UnormS8Uint
		                  ? 2u
		                  : 4u;
	}
	if (texel_bytes == 0u || texel_bytes > 16u) return;
	vk::BufferCreateInfo buffer_info {};
	buffer_info.size  = blocks ? uint64_t {(w + 3u) / 4u} * ((h + 3u) / 4u) * texel_bytes
	                           : uint64_t {w} * h * texel_bytes * (depth ? 1u : slices) * layers;
	buffer_info.usage = vk::BufferUsageFlagBits::eTransferDst;
	if (budgeted && !DebugDumpWithinBudget(buffer_info.size)) return;
	VmaAllocationCreateInfo allocation_info {};
	allocation_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
	allocation_info.usage = VMA_MEMORY_USAGE_AUTO;
	VmaAllocationInfo result {};
	VkBuffer          native = VK_NULL_HANDLE;
	DebugSample       sample;
	if (vmaCreateBuffer(graphics.allocator, reinterpret_cast<const VkBufferCreateInfo*>(&buffer_info),
	                    &allocation_info, &native, &sample.allocation, &result) != VK_SUCCESS) {
		return;
	}
	sample.buffer    = native;
	sample.mapped    = result.pMappedData;
	sample.dump_size = buffer_info.size;
	sample.dump_path = path + "_" + std::to_string(w) + "x" + std::to_string(h) +
	                   (!depth && slices > 1u ? "x" + std::to_string(slices) : std::string()) +
	                   (layers > 1u ? "L" + std::to_string(layers) : std::string()) + "_" +
	                   vk::to_string(image.backing.format) + ".raw";
	image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {}, command);
	vk::BufferImageCopy copy {};
	copy.imageSubresource = {depth ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor, 0, 0, layers};
	copy.imageExtent      = vk::Extent3D {w, h, depth ? 1u : slices};
	command.copyImageToBuffer(image.backing.image, vk::ImageLayout::eTransferSrcOptimal, sample.buffer, 1, &copy);
	DebugPushSample(std::move(sample));
	// The stencil aspect of a combined depth/stencil image goes to its own file (1 byte per texel).
	if (depth && format_name.find("S8") != std::string::npos) {
		vk::BufferCreateInfo stencil_info {};
		stencil_info.size  = uint64_t {w} * h;
		stencil_info.usage = vk::BufferUsageFlagBits::eTransferDst;
		if (budgeted && !DebugDumpWithinBudget(stencil_info.size)) return;
		VmaAllocationInfo stencil_result {};
		VkBuffer          stencil_native = VK_NULL_HANDLE;
		DebugSample       stencil;
		if (vmaCreateBuffer(graphics.allocator, reinterpret_cast<const VkBufferCreateInfo*>(&stencil_info),
		                    &allocation_info, &stencil_native, &stencil.allocation, &stencil_result) != VK_SUCCESS) {
			return;
		}
		stencil.buffer    = stencil_native;
		stencil.mapped    = stencil_result.pMappedData;
		stencil.dump_size = stencil_info.size;
		stencil.dump_path = path + "_" + std::to_string(w) + "x" + std::to_string(h) + "_S8Stencil.raw";
		vk::BufferImageCopy stencil_copy {};
		stencil_copy.imageSubresource = {vk::ImageAspectFlagBits::eStencil, 0, 0, 1};
		stencil_copy.imageExtent      = vk::Extent3D {w, h, 1};
		command.copyImageToBuffer(image.backing.image, vk::ImageLayout::eTransferSrcOptimal, stencil.buffer, 1,
		                          &stencil_copy);
		DebugPushSample(std::move(stencil));
	}
}
// Set by ResolveTexture when a storage binding of a depth-associated image is nulled, so the
// caller can name the shader (DIAG_DEPTH_STORAGE_NULL).
inline bool& DebugDepthStorageNulled() {
	thread_local bool nulled = false;
	return nulled;
}

// The presenter's sample point (KYTY_TRACE_DEPTH_ADDR counts draws per point).
inline std::atomic<uint32_t>& DebugArm() {
	static std::atomic<uint32_t> arm {0};
	return arm;
}

// KYTY_TRACE_RT_DUMP=<presented frame numbers, comma separated> (with KYTY_TRACE_FRAME_DIR): every
// render target drawn during such a frame is dumped whole at its present, and the shader pairs
// that drew into it are listed (RT_DUMP lines).
struct DebugRtRecord {
	uint64_t address = 0;
	uint32_t width   = 0;
	uint32_t height  = 0;
	bool     depth   = false;
	uint32_t draws   = 0;
	std::vector<std::pair<uint64_t, uint64_t>> shaders;
	std::vector<std::string>                   markers;  // parallel to shaders
};
// Game debug markers (PM4 push/pop) as a stack; RT dumps label draws with the innermost two.
inline std::mutex& DebugMarkerMutex() {
	static std::mutex mutex;
	return mutex;
}
// The command stream (queue) whose PM4 is executing: the graphics and compute queues interleave on the GPU
// thread, so each keeps its own marker stack (callers hold DebugMarkerMutex).
inline const void*& DebugMarkerStream() {
	static thread_local const void* stream = nullptr;
	return stream;
}
inline std::vector<std::string>& DebugMarkerStack() {
	static std::unordered_map<const void*, std::vector<std::string>> stacks;
	return stacks[DebugMarkerStream()];
}
inline void DebugMarkerPush(const char* name) {
	std::scoped_lock lock(DebugMarkerMutex());
	auto&            stack = DebugMarkerStack();
	if (stack.size() < 64u) stack.emplace_back(name);
}
inline void DebugMarkerPop() {
	std::scoped_lock lock(DebugMarkerMutex());
	auto&            stack = DebugMarkerStack();
	if (!stack.empty()) stack.pop_back();
}
// Shader whose storage buffers are being bound (the writer the logs name).
inline std::atomic<uint64_t>& DebugWriterHash() {
	static std::atomic<uint64_t> hash {0};
	return hash;
}

// KYTY_TRACE_PERF: the renderer lock between the presentation thread and the GPU thread ("slow runs", 19.10).
struct RenderLockStats {
	std::atomic<uint64_t> present_wait_ns {0}, present_hold_ns {0}, present_hold_max_ns {0}, presents {0};
	std::atomic<uint64_t> present_submit_ns {0};
	std::atomic<uint64_t> draw_wait_ns {0}, draw_wait_max_ns {0}, draws {0};
};
inline RenderLockStats& GetRenderLockStats() {
	static RenderLockStats s;
	return s;
}
inline void NoteMax(std::atomic<uint64_t>& max, uint64_t v) {
	uint64_t cur = max.load(std::memory_order_relaxed);
	while (v > cur && !max.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {
	}
}

inline std::string DebugMarkerPath() {
	std::scoped_lock lock(DebugMarkerMutex());
	const auto&      stack = DebugMarkerStack();
	if (stack.empty()) return "-";
	return stack.size() == 1u ? stack.back() : stack[stack.size() - 2u] + ">" + stack.back();
}
// Compute dispatches since the last present (marker, hash, written image/buffer addresses).
struct DebugCsRecord {
	std::string marker;
	uint64_t    hash = 0;
	std::string writes;
	std::string reads;
	uint32_t    count = 0;
};
inline std::vector<DebugCsRecord>& DebugCsFrame() {
	static std::vector<DebugCsRecord> records;
	return records;
}
// Zero-instance indirect draws per marker path since the last present.
inline std::vector<std::pair<std::string, uint32_t>>& DebugZeroDraws() {
	static std::vector<std::pair<std::string, uint32_t>> zero;
	return zero;
}
inline std::atomic<uint32_t>& DebugPresentCount() {
	static std::atomic<uint32_t> count {0};
	return count;
}

// KYTY_TEST_PAD_RECORD / KYTY_TEST_PAD_REPLAY (test tool): steady-clock nanoseconds of the moment gameplay started (the first
// of 8 presented frames in a row with more than 3000 draws, set once by the execution thread); 0 until then. Recording and
// replaying pad input count time from here, so a different loading time does not shift the route.
inline std::atomic<int64_t>& GameplayStartNs() {
	static std::atomic<int64_t> ns {0};
	return ns;
}

// KYTY_TRACE_FRAME_RANGE=<first>-<last>: every presented frame in the range is dumped.
inline bool DebugFrameRange() {
	static constinit Common::OnceObject<std::pair<uint32_t, uint32_t>> range_once;
	const std::pair<uint32_t, uint32_t>& range = range_once.Get([] { return Common::EnvRead::FrameRange(); });
	const auto frame = DebugPresentCount().load();
	return frame >= range.first && frame <= range.second;
}
// Stencil/water trace window: frames 1100-1102, or the 3 frames after the file named by
// KYTY_TRACE_STENCIL_TRIGGER appears (the file is deleted when the window opens).
inline bool DebugStencilWindow() {
	static std::atomic<uint32_t> start {1100u};
	static std::atomic<uint32_t> checked {0u};
	static const char* const     trigger = std::getenv("KYTY_TRACE_STENCIL_TRIGGER");
	const auto frame = DebugPresentCount().load(std::memory_order_relaxed);
	// Without a trigger file the window is fixed: no per-call exchange (it ran on every draw).
	if (trigger != nullptr && checked.exchange(frame) != frame) {
		{
			if (FILE* file = std::fopen(trigger, "rb")) {
				std::fclose(file);
				std::remove(trigger);
				start.store(frame + 1u);
				std::fprintf(stderr, "STENCIL_TRIGGER frames %u-%u\n", frame + 1u, frame + 3u);
			}
		}
	}
	const auto first = start.load();
	return frame >= first && frame < first + 3u;
}
inline std::mutex& DebugRtMutex() {
	static std::mutex mutex;
	return mutex;
}
inline std::vector<DebugRtRecord>& DebugRtFrame() {
	static std::vector<DebugRtRecord> records;
	return records;
}
inline const std::vector<uint32_t>& DebugRtDumpFrames() {
	static constinit Common::OnceObject<std::vector<uint32_t>> frames_once;
	const std::vector<uint32_t>& frames = frames_once.Get([] {
		std::vector<uint32_t> result;
		const char*           value = std::getenv("KYTY_TRACE_RT_DUMP");
		if (value == nullptr) return result;
		std::string text(value);
		size_t      start = 0;
		while (start < text.size()) {
			const auto end = text.find(',', start);
			result.push_back(static_cast<uint32_t>(
			    std::strtoul(text.substr(start, end == std::string::npos ? std::string::npos : end - start).c_str(),
			                 nullptr, 10)));
			if (end == std::string::npos) break;
			start = end + 1;
		}
		return result;
	});
	return frames;
}
// KYTY_TRACE_RT_DUMP_TIMES=<seconds,...>: like KYTY_TRACE_RT_DUMP, the first presented frame after each of those seconds
// since the process started (the test runner's clock), saved as rttime<seconds>_presented: the same moment of play in
// every run, whatever number of frames the loading screen presented before it.
inline const std::vector<uint32_t>& DebugRtDumpTimes() {
	static constinit Common::OnceObject<std::vector<uint32_t>> times_once;
	const std::vector<uint32_t>& times = times_once.Get([] {
		std::vector<uint32_t> result;
		const char*           value = std::getenv("KYTY_TRACE_RT_DUMP_TIMES");
		if (value == nullptr) return result;
		std::string text(value);
		size_t      start = 0;
		while (start < text.size()) {
			const auto end = text.find(',', start);
			result.push_back(static_cast<uint32_t>(
			    std::strtoul(text.substr(start, end == std::string::npos ? std::string::npos : end - start).c_str(),
			                 nullptr, 10)));
			if (end == std::string::npos) break;
			start = end + 1;
		}
		std::ranges::sort(result);
		return result;
	});
	return times;
}
// KYTY_TRACE_RT_DUMP_TRIGGER=<file>: creating that file dumps the next presented frame (the file
// is deleted), so a frame can be captured during a long run.
inline const char* DebugRtTriggerPath() {
	static constinit Common::OnceObject<const char*> path_once;
	const char* const path = path_once.Get([] { return std::getenv("KYTY_TRACE_RT_DUMP_TRIGGER"); });
	return path;
}
// Set when the trigger file is seen: the following frame dumps every image written by compute
// dispatches (with their order), and its present dumps the render targets.
inline std::atomic<bool>& DebugCaptureFrame() {
	static std::atomic<bool> capture {false};
	return capture;
}
inline std::atomic<uint32_t>& DebugCaptureIndex() {
	static std::atomic<uint32_t> index {0};
	return index;
}
inline bool DebugRtEnabled() {
	return !DebugRtDumpFrames().empty() || !DebugRtDumpTimes().empty() || DebugRtTriggerPath() != nullptr;
}
// Set at each present (swapchain.cpp) for the frames that follow: one of them may be dumped. The per-draw and
// per-dispatch dump records (draws per target, dispatches, zero-instance draws) are kept only then; kept for every
// frame they cost the GPU thread several ms per frame in every test run that takes a screenshot.
inline std::atomic<bool>& DebugRtCollect() {
	static std::atomic<bool> collect {DebugRtEnabled()};
	return collect;
}
inline bool DebugRtCollecting() {
	return DebugRtEnabled() && DebugRtCollect().load(std::memory_order_relaxed);
}
// Game debug markers are tracked for RT dumps and for KYTY_TRACE_VK_LABELS (labels then carry the pass name).
inline bool DebugMarkersEnabled() {
	static constinit Common::OnceValue<bool> labels_once;
	const bool labels = labels_once.Get([] { return std::getenv("KYTY_TRACE_VK_LABELS") != nullptr; });
	return labels || DebugRtEnabled();
}
inline void DebugRtRecordDraw(uint64_t address, uint32_t width, uint32_t height, bool depth, uint64_t vs,
                              uint64_t ps) {
	std::scoped_lock lock(DebugRtMutex());
	auto&            records = DebugRtFrame();
	auto it = std::ranges::find_if(records, [&](const DebugRtRecord& r) {
		return r.address == address && r.width == width && r.height == height && r.depth == depth;
	});
	if (it == records.end()) {
		if (records.size() >= 256u) return;
		records.push_back({address, width, height, depth, 0, {}});
		it = records.end() - 1;
	}
	it->draws++;
	const auto pair = std::make_pair(vs, ps);
	if (std::ranges::find(it->shaders, pair) == it->shaders.end() && it->shaders.size() < 512u) {
		it->shaders.push_back(pair);
		it->markers.push_back(DebugMarkerPath());
	}
}

} // namespace Libs::Graphics

#endif
