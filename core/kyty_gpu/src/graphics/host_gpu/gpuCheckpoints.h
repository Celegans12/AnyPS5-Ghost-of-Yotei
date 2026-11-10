#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GPUCHECKPOINTS_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GPUCHECKPOINTS_H_

#include "common/envRead.h"
#include "common/onceValue.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/host_gpu/renderer/gpuTimer.h"

#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

// KYTY_TRACE_GPU_CHECKPOINTS=1 (diagnostic, default off): with VK_NV_device_diagnostic_checkpoints
// each dispatch/draw records the shader hashes it runs; after a lost device the last markers the
// GPU reached are printed to stderr.
namespace Libs::Graphics::GpuCheckpoints {

[[nodiscard]] inline bool Requested() {
	static constinit Common::OnceValue<bool> requested_once;
	const bool requested = requested_once.Get([] {
		const char* value = std::getenv("KYTY_TRACE_GPU_CHECKPOINTS");
		return value != nullptr && value[0] == '1';
	});
	return requested;
}

inline std::atomic_bool& EnabledFlag() {
	static std::atomic_bool enabled = false;
	return enabled;
}

// VK_EXT_device_fault (enabled together with the checkpoints when available): after a lost
// device the driver's fault report (faulting addresses, vendor fault codes) is printed too.
inline std::atomic_bool& FaultEnabledFlag() {
	static std::atomic_bool enabled = false;
	return enabled;
}

inline vk::Device& FaultDevice() {
	static vk::Device device = nullptr;
	return device;
}

inline void DumpDeviceFault() {
	if (!FaultEnabledFlag().load() || FaultDevice() == nullptr) return;
	auto&                  device = FaultDevice();
	vk::DeviceFaultCountsEXT counts {};
	if (device.getFaultInfoEXT(&counts, nullptr) != vk::Result::eSuccess) {
		std::fprintf(stderr, "GPU_FAULT query failed\n");
		return;
	}
	std::vector<vk::DeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
	std::vector<vk::DeviceFaultVendorInfoEXT>  vendors(counts.vendorInfoCount);
	vk::DeviceFaultInfoEXT                     info {};
	info.pAddressInfos     = addresses.data();
	info.pVendorInfos      = vendors.data();
	counts.vendorBinarySize = 0;
	const auto result      = device.getFaultInfoEXT(&counts, &info);
	std::fprintf(stderr, "GPU_FAULT result=%d description=\"%s\" addresses=%u vendor_infos=%u\n",
	             static_cast<int>(result), info.description.data(), counts.addressInfoCount, counts.vendorInfoCount);
	for (uint32_t i = 0; i < counts.addressInfoCount; ++i) {
		std::fprintf(stderr, "GPU_FAULT address type=%s address=0x%016" PRIx64 " precision=0x%" PRIx64 "\n",
		             vk::to_string(addresses[i].addressType).c_str(),
		             static_cast<uint64_t>(addresses[i].reportedAddress),
		             static_cast<uint64_t>(addresses[i].addressPrecision));
	}
	for (uint32_t i = 0; i < counts.vendorInfoCount; ++i) {
		std::fprintf(stderr, "GPU_FAULT vendor \"%s\" code=0x%016" PRIx64 " data=0x%016" PRIx64 "\n",
		             vendors[i].description.data(), static_cast<uint64_t>(vendors[i].vendorFaultCode),
		             static_cast<uint64_t>(vendors[i].vendorFaultData));
	}
	std::fflush(stderr);
}

[[nodiscard]] inline bool Enabled() { return EnabledFlag().load(std::memory_order_relaxed); }

// The last recorded commands (CPU side), printed with the checkpoints after a lost device.
struct RecentCommands {
	std::mutex            mutex;
	uint64_t              count = 0;
	std::vector<uint64_t> hashes = std::vector<uint64_t>(64, 0);
};

inline RecentCommands& Recent() {
	static RecentCommands recent;
	return recent;
}

// KYTY_TRACE_GPU_TIME: while a FeedLabel is alive on this thread, emulator-internal markers (0xFEED...) take its label,
// so the GPU time of a composite operation (copies, tiler passes and uploads it records) counts under one name.
inline thread_local uint64_t t_feed_label = 0;

// Markers are opaque pointer-sized values; the shader hash is the marker.
inline void Mark(vk::CommandBuffer buffer, uint64_t hash) {
	if ((hash >> 48u) == 0xFEEDu && t_feed_label != 0u) hash = t_feed_label;
	// KYTY_TRACE_GPU_TIME=1/3: emulator-internal operations (0xFEED<component><line> markers) also
	// label the GPU time that follows them.
	if ((hash >> 48u) == 0xFEEDu && buffer != nullptr && GpuTimer::Enabled()) {
		GpuTimer::Mark(buffer, 'e', hash & 0xffffffffull, 0);
	}
	if (!Enabled() || buffer == nullptr) return;
	buffer.setCheckpointNV(reinterpret_cast<const void*>(static_cast<uintptr_t>(hash)));
	auto&            recent = Recent();
	std::scoped_lock lock(recent.mutex);
	recent.hashes[recent.count++ % recent.hashes.size()] = hash;
}

// The scope of a composite emulator-internal operation (see t_feed_label): marks its label once, and the markers
// recorded inside it carry the label too.
struct FeedLabel {
	FeedLabel(vk::CommandBuffer buffer, uint64_t label): previous(t_feed_label) {
		t_feed_label = 0;
		Mark(buffer, label);
		t_feed_label = label;
	}
	~FeedLabel() { t_feed_label = previous; }
	FeedLabel(const FeedLabel&)            = delete;
	FeedLabel& operator=(const FeedLabel&) = delete;
	uint64_t previous;
};

// KYTY_TRACE_FLIPS: guest addresses of the presented surfaces, so writers can be named.
inline std::atomic<uint64_t>& FlipAddress(uint32_t slot) {
	static std::atomic<uint64_t> addresses[4] {};
	return addresses[slot & 3u];
}

struct ChainState {
	std::mutex            mutex;
	std::vector<uint64_t> tracked;  // inputs of traced writers
	std::vector<uint64_t> seen;     // writer hash ^ target
};

inline ChainState& Chain() {
	static ChainState state;
	return state;
}

// A presented surface, or an image read by a traced producer of one.
[[nodiscard]] inline bool IsFlipAddress(uint64_t address) {
	// Flip addresses (and so the traced chain) are only recorded with KYTY_TRACE_FLIPS: without it
	// skip the chain lock taken for every color target of every draw.
	static constinit Common::OnceValue<bool> flips_traced_once;
	const bool flips_traced = flips_traced_once.Get([&] { return Common::EnvRead::Flips(); });
	if (!flips_traced || address == 0) return false;
	for (uint32_t i = 0; i < 4u; ++i)
		if (FlipAddress(i).load(std::memory_order_relaxed) == address) return true;
	auto& chain = Chain();
	std::scoped_lock lock(chain.mutex);
	for (auto a: chain.tracked)
		if (a == address) return true;
	return false;
}

struct ChainRead {
	uint64_t address;
	uint32_t format, width, height, tile;
};

// Logs a shader writing a traced image (each writer/target pair once) and the images it reads;
// those reads become traced, so their producers are named next.
inline void TraceFlipWriter(const char* kind, uint64_t hash, uint64_t address,
                            const std::vector<ChainRead>& reads = {}) {
	auto& chain = Chain();
	std::scoped_lock lock(chain.mutex);
	const auto key = hash ^ address;
	for (auto k: chain.seen)
		if (k == key) return;
	if (chain.seen.size() >= 96u) return;
	chain.seen.push_back(key);
	std::fprintf(stderr, "FLIP_WRITER %s hash=0x%016" PRIx64 " target=0x%" PRIx64 " reads=%u\n", kind, hash,
	             address, static_cast<uint32_t>(reads.size()));
	for (const auto& r: reads) {
		std::fprintf(stderr, "  read addr=0x%" PRIx64 " fmt=%u extent=%ux%u tile=%u\n", r.address, r.format,
		             r.width, r.height, r.tile);
		bool known = r.address == 0;
		for (auto a: chain.tracked) known |= a == r.address;
		if (!known && chain.tracked.size() < 128u) chain.tracked.push_back(r.address);
	}
	std::fflush(stderr);
}

inline void Dump(vk::Queue queue) {
	DumpDeviceFault();
	if (!Enabled() || queue == nullptr) return;
	uint32_t count = 0;
	queue.getCheckpointDataNV(&count, nullptr);
	std::vector<vk::CheckpointDataNV> data(count);
	queue.getCheckpointDataNV(&count, data.data());
	std::fprintf(stderr, "GPU_CHECKPOINTS count=%u\n", count);
	for (uint32_t i = 0; i < count; ++i) {
		std::fprintf(stderr, "GPU_CHECKPOINT stage=0x%08x marker=0x%016" PRIx64 "\n",
		             static_cast<uint32_t>(data[i].stage),
		             static_cast<uint64_t>(reinterpret_cast<uintptr_t>(data[i].pCheckpointMarker)));
	}
	{
		auto&            recent = Recent();
		std::scoped_lock lock(recent.mutex);
		const auto       size  = static_cast<uint64_t>(recent.hashes.size());
		const auto       first = recent.count > size ? recent.count - size : 0;
		std::fprintf(stderr, "GPU_RECENT total=%llu (oldest first)\n", static_cast<unsigned long long>(recent.count));
		for (uint64_t i = first; i < recent.count; ++i) {
			std::fprintf(stderr, "GPU_RECENT #%llu 0x%016" PRIx64 "\n", static_cast<unsigned long long>(i),
			             recent.hashes[i % size]);
		}
	}
	std::fflush(stderr);
}

} // namespace Libs::Graphics::GpuCheckpoints

#endif
