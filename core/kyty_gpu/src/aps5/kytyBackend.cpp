// kyty_gpu.dll: Kyty's GPU backend behind AnyPS5's plain C seam (GpuBackend.h, Aps5GpuBackendApi). AnyPS5 keeps the HLE
// (libSceAgc, libSceVideoOut, the event queues, the window and its input) and calls in here with the guest's submissions,
// flips, mapping changes and faults; Kyty's command processor, caches, shader recompiler and presenter do the rest.
// Only Aps5GpuBackendCreate is exported, and only the plain C tables cross: no STL type, no exception, no C++ object.

#include "aps5/GpuBackend.h"
#include "aps5/aps5Host.h"
#include "aps5/aps5Memory.h"
#include "aps5/aps5VideoOut.h"
#include "aps5/aps5Window.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/threads.h"
#include "common/virtualMemory.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/presentation/presenter.h"
#include "graphics/presentation/window/windowInternal.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep

struct Aps5GpuBackend {
	const Aps5GpuHost* host = nullptr;
};

namespace Aps5 {

namespace Graphics = Libs::Graphics;

namespace {

struct WindowState {
	std::mutex               mutex;
	void*                    context = nullptr;
	std::vector<std::string> extensions;
	uint64_t (*create_surface)(void* context, void* vk_instance)                  = nullptr;
	void (*drawable_size)(void* context, uint32_t* width, uint32_t* height)       = nullptr;
	uint32_t                 width  = 0;
	uint32_t                 height = 0;
};

struct Counts {
	std::atomic<uint64_t> submits {0};
	std::atomic<uint64_t> cpu_flips {0};
	std::atomic<uint64_t> shaders {0};
	std::atomic<uint64_t> maps {0};
	std::atomic<uint64_t> unmaps {0};
	std::atomic<uint64_t> protects {0};
	std::atomic<uint64_t> host_writes {0};
	std::atomic<uint64_t> host_reads {0};
	std::atomic<uint64_t> faults {0};
	std::atomic<uint64_t> faults_handled {0};
	std::atomic<uint64_t> guest_faults {0};
	std::atomic<uint64_t> presents {0};
};

struct Backend {
	Aps5GpuBackend                                 handle;
	std::once_flag                                 started;
	std::once_flag                                 shaders_ready;
	std::mutex                                     device_mutex;
	std::condition_variable                        device_cv;
	std::atomic<bool>                              device_ready {false};
	std::unique_ptr<Graphics::WindowContext>       window_context;
	std::atomic<Graphics::RenderContext*>          renderer {nullptr};
	std::mutex                                     map_mutex;
	WindowState                                    window;
	HMODULE                                        vulkan = nullptr;
	Counts                                         counts;
	std::mutex                                     report_mutex;
	std::chrono::steady_clock::time_point          last_report = std::chrono::steady_clock::now();
};

Backend* g_backend = nullptr;

Backend& B() {
	return *g_backend;
}

void FatalHook(const char* report, int status) {
	Fatal("Kyty stopped (status %d): %s", status, report != nullptr && report[0] != '\0' ? report : "see the log above");
}

// Kyty reads its settings with getenv as it first needs each one. AnyPS5's runs pass only APS5_* variables, so
// APS5_KYTY_<NAME>=<value> becomes KYTY_<NAME>=<value> here; then the AnyPS5 defaults below fill what is still unset,
// before any Kyty code reads them.
void ApplySettings() {
	std::vector<std::pair<std::string, std::string>> forwarded;
	for (char** entry = _environ; entry != nullptr && *entry != nullptr; ++entry) {
		const std::string text(*entry);
		if (text.rfind("APS5_KYTY_", 0) != 0) continue;
		const auto equals = text.find('=');
		if (equals == std::string::npos) continue;
		forwarded.emplace_back(text.substr(5, equals - 5), text.substr(equals + 1));
	}
	for (const auto& [name, value]: forwarded) {
		_putenv_s(name.c_str(), value.c_str());
		Log("setting %s=%s (from APS5_%s)", name.c_str(), value.c_str(), name.c_str());
	}
	struct Default {
		const char* name;
		const char* value;
		const char* why;
	};
	// PLAN v4 section 5: Kyty's work-skipping shortcuts off, or on their exact variant (report 2026-11-08-v4-kyty-m1b).
	static constexpr Default defaults[] = {
	    {"KYTY_PERF_TABLE_ASYNC", "0", "draws wait for their pipeline instead of being skipped"},
	    {"KYTY_UPSTREAM_GPU_INDIRECT_RECORDS", "1", "GPU indirect packets leave NUM_INSTANCES as the hardware does"},
	};
	for (const auto& entry: defaults) {
		if (std::getenv(entry.name) != nullptr) {
			Log("setting %s=%s (environment)", entry.name, std::getenv(entry.name));
			continue;
		}
		_putenv_s(entry.name, entry.value);
		Log("setting %s=%s (AnyPS5 default: %s)", entry.name, entry.value, entry.why);
	}
	if (std::getenv("KYTY_PIPELINE_CACHE_DIR") == nullptr) {
		const char* shader_cache = std::getenv("ANYPS5_SHADER_CACHE_DIR");
		const auto  dir = shader_cache != nullptr && shader_cache[0] != '\0' ? std::string(shader_cache) + "\\kyty"
		                                                                     : std::string("_PipelineCache");
		_putenv_s("KYTY_PIPELINE_CACHE_DIR", dir.c_str());
	}
	Log("pipeline cache, shader journals and BDA granule list: %s", std::getenv("KYTY_PIPELINE_CACHE_DIR"));
}

Config::ConfigOptions ReleaseConfig() {
	// Kyty's release defaults (ConfigOptions), at the 1080p output the AnyPS5 runs use; readback of small tiled images
	// on (the exact variant: the guest CPU then sees what the GPU wrote into them).
	Config::ConfigOptions options;
	options.screen_width          = 1920;
	options.screen_height         = 1080;
	options.splash_enabled        = false;
	options.readback_tiled_images = true;
	return options;
}

void ReportMaybe() {
	auto&      b   = B();
	const auto now = std::chrono::steady_clock::now();
	{
		std::unique_lock lock(b.report_mutex, std::try_to_lock);
		if (!lock.owns_lock() || now - b.last_report < std::chrono::seconds(10)) return;
		b.last_report = now;
	}
	const auto memory = GetMemoryStats();
	const auto load   = [](const std::atomic<uint64_t>& value) { return static_cast<unsigned long long>(value.load()); };
	Log("kyty backend: submits %llu, cpu flips %llu, presents %llu, shaders %llu; mappings %llu (direct %llu MiB, private "
	    "%llu MiB, %llu found by lookup), map %llu unmap %llu protect %llu, host writes %llu reads %llu; faults %llu "
	    "(%llu served, %llu on guest threads)",
	    load(b.counts.submits), load(b.counts.cpu_flips), load(b.counts.presents), load(b.counts.shaders),
	    static_cast<unsigned long long>(memory.entries), static_cast<unsigned long long>(memory.direct_bytes >> 20u),
	    static_cast<unsigned long long>(memory.private_bytes >> 20u), static_cast<unsigned long long>(memory.discovered),
	    load(b.counts.maps), load(b.counts.unmaps), load(b.counts.protects), load(b.counts.host_writes),
	    load(b.counts.host_reads), load(b.counts.faults), load(b.counts.faults_handled), load(b.counts.guest_faults));
}

int Start(Aps5GpuBackend* /*backend*/) {
	bool ok = true;
	std::call_once(B().started, [&] {
		Common::SetFatalHook(&FatalHook);
		std::call_once(B().shaders_ready, [] { Graphics::ShaderInit(); });
		ApplySettings();
		Config::Initialize();
		Config::Load(ReleaseConfig());
		Log::Initialize();
		Common::VirtualMemory::Init();
		const auto& host  = Host();
		uint8_t*    base  = nullptr;
		uint64_t    bytes = 0;
		if (host.backing_view(host.context, &base, &bytes) != APS5_GPU_OK || base == nullptr || bytes == 0) {
			Log("AnyPS5 has no direct memory backing view (APS5_GPU_BACKEND=kyty must put direct memory in kyty mode)");
			ok = false;
			return;
		}
		MemoryInitialize(base, bytes);
		Log("started: direct memory backing view %p+0x%llx; Kyty v0.1.5 + v4/base + v4/m1b, clang, libc++", base,
		    static_cast<unsigned long long>(bytes));
	});
	return ok ? APS5_GPU_OK : APS5_GPU_ERROR;
}

// The device, the renderer, the presenter and Kyty's GPU threads, once AnyPS5's window exists (the surface decides the
// device). This thread becomes the one Kyty calls its main thread (the renderer's constructors check it).
void CreateDevice() {
	auto& b = B();
	Common::InitializeThreads();
	auto context                       = std::make_unique<Graphics::WindowContext>();
	context->graphic_ctx.screen_width  = b.window.width;
	context->graphic_ctx.screen_height = b.window.height;
	context->CreateVulkan();
	auto& presenter = *context->presenter;
	auto* renderer  = context->render_context.get();
	auto& video_out = VideoOutCreate(presenter, b.window.width, b.window.height);
	renderer->InitializeGpu(&video_out);
	b.window_context = std::move(context);
	{
		// What AnyPS5 mapped before the renderer existed (Kyty's kernel maps GPU ranges only once the renderer exists,
		// but its renderer starts before the game runs; here the game has mapped memory by the time its window opens).
		std::scoped_lock lock(b.map_mutex);
		uint64_t         ranges = 0;
		uint64_t         bytes  = 0;
		MemoryForEachMapping([&](uint64_t address, uint64_t size) {
			if (!Graphics::GuestRange {address, size}.Valid()) return;
			renderer->MapMemory(address, size);
			ranges++;
			bytes += size;
		});
		b.renderer.store(renderer, std::memory_order_release);
		Log("renderer started: %llu mapped ranges (%llu MiB) handed to it", static_cast<unsigned long long>(ranges),
		    static_cast<unsigned long long>(bytes >> 20u));
	}
	{
		std::scoped_lock lock(b.device_mutex);
		b.device_ready.store(true, std::memory_order_release);
	}
	b.device_cv.notify_all();
}

Graphics::RenderContext& WaitDevice(const char* what) {
	auto& b = B();
	if (!b.device_ready.load(std::memory_order_acquire)) {
		std::unique_lock lock(b.device_mutex);
		if (!b.device_cv.wait_for(lock, std::chrono::seconds(300), [&] { return b.device_ready.load(); })) {
			Fatal("%s before AnyPS5 attached its window (the device is made with the window's surface)", what);
		}
	}
	return *b.renderer.load(std::memory_order_acquire);
}

Graphics::RenderContext* Renderer() {
	return B().renderer.load(std::memory_order_acquire);
}

void Shutdown(Aps5GpuBackend* /*backend*/) {
	ReportMaybe();
	if (auto* renderer = Renderer(); renderer != nullptr) {
		Common::LockGuard lock(renderer->GetMutex());
		renderer->GetPipelineCache().Save();
		Log("shutdown: pipeline cache saved");
	}
}

void Destroy(Aps5GpuBackend* /*backend*/) {}

constexpr uint32_t GraphicsQueue     = 0;
constexpr uint32_t ComputeQueueFirst = 0x20;
constexpr uint32_t ComputeQueueEnd   = 0x58;

int Submit(Aps5GpuBackend* /*backend*/, uint32_t queue, const uint32_t* dwords, uint32_t count) {
	auto& renderer = WaitDevice("a submission");
	if (B().counts.submits.fetch_add(1, std::memory_order_relaxed) == 0) {
		Log("first submission: queue 0x%x, %u dwords", queue, count);
	}
	const std::span<const uint32_t> commands {dwords, count};
	if (queue == GraphicsQueue) {
		renderer.GetGpu().Submit(commands, {});
	} else if (queue >= ComputeQueueFirst && queue < ComputeQueueEnd) {
		renderer.GetGpu().SubmitCompute(queue, commands);
	} else {
		Log("submission on unknown queue 0x%x refused", queue);
		return APS5_GPU_ERROR;
	}
	return APS5_GPU_OK;
}

int SubmitFlip(Aps5GpuBackend* /*backend*/, const Aps5GpuFlip* flip) {
	(void)WaitDevice("a CPU flip");
	if (B().counts.cpu_flips.fetch_add(1, std::memory_order_relaxed) == 0) {
		Log("first CPU flip: handle %u index %d mode %u", flip->handle, flip->index, flip->mode);
	}
	return VideoOutSubmitCpuFlip(*flip);
}

void SuspendPoint(Aps5GpuBackend* /*backend*/) {
	if (auto* renderer = Renderer(); renderer != nullptr) renderer->GetGpu().SuspendPoint();
}

void WaitIdle(Aps5GpuBackend* /*backend*/) {
	if (auto* renderer = Renderer(); renderer != nullptr) renderer->GetGpu().WaitForIdle();
}

int RegisterShader(Aps5GpuBackend* /*backend*/, const Aps5GpuShaderInfo* shader) {
	std::call_once(B().shaders_ready, [] { Graphics::ShaderInit(); });
	Graphics::ShaderMappedData data;
	data.type                = static_cast<Graphics::Prospero::ShaderBinaryType>(shader->type);
	data.user_data           = static_cast<Graphics::ShaderUserData*>(const_cast<void*>(shader->user_data));
	data.input_semantics     = static_cast<Graphics::ShaderSemantic*>(const_cast<void*>(shader->input_semantics));
	data.num_input_semantics = shader->num_input_semantics;
	data.code_size_bytes     = shader->code_size_bytes;
	data.scratch_size_dwords = shader->scratch_size_dwords;
	Graphics::ShaderMapUserData(shader->code, data);
	B().counts.shaders.fetch_add(1, std::memory_order_relaxed);
	return APS5_GPU_OK;
}

void MemoryUnmap(Aps5GpuBackend* /*backend*/, uint64_t address, uint64_t bytes) {
	auto&            b = B();
	std::scoped_lock lock(b.map_mutex);
	b.counts.unmaps.fetch_add(1, std::memory_order_relaxed);
	if (auto* renderer = Renderer(); renderer != nullptr && Graphics::GuestRange {address, bytes}.Valid()) {
		renderer->UnmapMemory(address, bytes);
	}
	MemoryUnmapped(address, bytes);
}

void MemoryMap(Aps5GpuBackend* /*backend*/, uint64_t address, uint64_t bytes, uint32_t /*protection*/, uint64_t direct_offset) {
	auto&            b = B();
	std::scoped_lock lock(b.map_mutex);
	b.counts.maps.fetch_add(1, std::memory_order_relaxed);
	MemoryMapped(address, bytes, direct_offset);
	if (auto* renderer = Renderer(); renderer != nullptr && Graphics::GuestRange {address, bytes}.Valid()) {
		renderer->MapMemory(address, bytes);
	}
}

// A guest mprotect: AnyPS5's protection replaced the tracker's on the range. Its bytes are made CPU-dirty (GPU-written
// bytes downloaded first), so a guest store after the change reaches the GPU's copies at their next use, and the
// tracker protects the pages again then.
void MemoryProtect(Aps5GpuBackend* /*backend*/, uint64_t address, uint64_t bytes, uint32_t /*protection*/) {
	B().counts.protects.fetch_add(1, std::memory_order_relaxed);
	if (auto* renderer = Renderer(); renderer != nullptr && Graphics::GuestRange {address, bytes}.Valid()) {
		(void)renderer->InvalidateMemory(address, bytes);
	}
}

// AnyPS5 writes guest memory in kernel mode (a file read): as Kyty's own kernel does before such a call.
void HostWriteBegin(Aps5GpuBackend* /*backend*/, uint64_t address, uint64_t bytes) {
	B().counts.host_writes.fetch_add(1, std::memory_order_relaxed);
	if (auto* renderer = Renderer(); renderer != nullptr && Graphics::GuestRange {address, bytes}.Valid()) {
		(void)renderer->InvalidateMemory(address, bytes);
	}
}

void HostWriteEnd(Aps5GpuBackend* /*backend*/, uint64_t /*address*/, uint64_t /*bytes*/) {}

// AnyPS5 reads guest memory in kernel mode (a file write), where a page the tracker made no-access fails the call: each
// such page is served as a read fault would be (GPU-written bytes downloaded, the image read watch lifted).
void HostRead(Aps5GpuBackend* /*backend*/, uint64_t address, uint64_t bytes) {
	B().counts.host_reads.fetch_add(1, std::memory_order_relaxed);
	auto* renderer = Renderer();
	if (renderer == nullptr || bytes == 0 || !Graphics::GuestRange {address, bytes}.Valid() ||
	    !renderer->IsMapped(address, bytes)) {
		return;
	}
	constexpr uint64_t PageSize = 4096;
	const auto         end      = address + bytes;
	for (auto page = address & ~(PageSize - 1u); page < end; page += PageSize) {
		const auto at = std::max(page, address);
		(void)renderer->HandleFault(Graphics::PageFaultAccess::Read, at);
	}
}

int HandleFault(Aps5GpuBackend* /*backend*/, uint64_t address, uint32_t access) {
	auto* renderer = Renderer();
	if (renderer == nullptr) {
		return APS5_GPU_FAULT_NOT_MINE;
	}
	auto& b = B();
	b.counts.faults.fetch_add(1, std::memory_order_relaxed);
	const auto kind = access == APS5_GPU_FAULT_WRITE     ? Graphics::PageFaultAccess::Write
	                  : access == APS5_GPU_FAULT_EXECUTE ? Graphics::PageFaultAccess::Execute
	                                                     : Graphics::PageFaultAccess::Read;
	// As Kyty's exception handler: the faulting instruction runs again when this returns.
	const Graphics::RetriedFaultScope retried_fault;
	const Graphics::FaultSiteScope    fault_site(0);
	if (!Libs::LibKernel::Memory::HandleGpuFault(kind, address)) {
		return APS5_GPU_FAULT_NOT_MINE;
	}
	b.counts.faults_handled.fetch_add(1, std::memory_order_relaxed);
	if (!Graphics::GuestGpu::IsGpuThread() && b.counts.guest_faults.fetch_add(1, std::memory_order_relaxed) == 0) {
		Log("first tracker fault served on a thread of AnyPS5's: address 0x%llx, %s",
		    static_cast<unsigned long long>(address), kind == Graphics::PageFaultAccess::Write ? "write" : "read");
	}
	return APS5_GPU_FAULT_HANDLED;
}

int AttachWindow(Aps5GpuBackend* /*backend*/, const Aps5GpuWindow* window) {
	auto& b = B();
	{
		std::scoped_lock lock(b.window.mutex);
		b.window.context = window->context;
		b.window.extensions.clear();
		for (uint32_t i = 0; i < window->instance_extension_count; i++) {
			b.window.extensions.emplace_back(window->instance_extensions[i]);
		}
		b.window.create_surface = window->create_surface;
		b.window.drawable_size  = window->drawable_size;
		b.window.width          = window->width;
		b.window.height         = window->height;
	}
	if (!b.device_ready.load(std::memory_order_acquire)) {
		Log("window attached: %ux%u, %u instance extensions; creating the device", window->width, window->height,
		    window->instance_extension_count);
		CreateDevice();
	}
	return APS5_GPU_OK;
}

void ReleaseWindow(Aps5GpuBackend* /*backend*/, void* /*window_context*/) {}

int Present(Aps5GpuBackend* /*backend*/, const Aps5GpuWindow* window, const Aps5GpuDisplayBuffer* buffer, int32_t opaque,
            int32_t dump_index, void (*presented)(void* context), void* presented_context) {
	if (!B().device_ready.load(std::memory_order_acquire)) {
		presented(presented_context);
		return APS5_GPU_OK;
	}
	B().counts.presents.fetch_add(1, std::memory_order_relaxed);
	const int result = VideoOutPresent(window, buffer, opaque, dump_index, presented, presented_context);
	ReportMaybe();
	return result;
}

const Aps5GpuBackendApi KytyApi {
    APS5_GPU_BACKEND_ABI_VERSION,
    sizeof(Aps5GpuBackendApi),
    "kyty",
    &Start,
    &Shutdown,
    &Destroy,
    &Submit,
    &SubmitFlip,
    &SuspendPoint,
    &WaitIdle,
    &RegisterShader,
    &MemoryUnmap,
    &MemoryMap,
    &MemoryProtect,
    &HostWriteBegin,
    &HostWriteEnd,
    &HostRead,
    &HandleFault,
    &AttachWindow,
    &ReleaseWindow,
    &Present,
};

} // namespace

PFN_vkGetInstanceProcAddr WindowVulkanLoader() {
	auto& b = B();
	if (b.vulkan == nullptr) {
		b.vulkan = LoadLibraryW(L"vulkan-1.dll");
	}
	return b.vulkan != nullptr ? reinterpret_cast<PFN_vkGetInstanceProcAddr>(
	                                 reinterpret_cast<void*>(GetProcAddress(b.vulkan, "vkGetInstanceProcAddr")))
	                           : nullptr;
}

std::vector<const char*> WindowInstanceExtensions() {
	auto&                    b = B();
	std::scoped_lock         lock(b.window.mutex);
	std::vector<const char*> names;
	for (const auto& name: b.window.extensions) {
		names.push_back(name.c_str());
	}
	return names;
}

VkSurfaceKHR WindowCreateSurface(VkInstance instance) {
	auto&                                          b = B();
	uint64_t (*create)(void*, void*)                 = nullptr;
	void* context                                    = nullptr;
	{
		std::scoped_lock lock(b.window.mutex);
		create  = b.window.create_surface;
		context = b.window.context;
	}
	if (create == nullptr) {
		return VK_NULL_HANDLE;
	}
	return reinterpret_cast<VkSurfaceKHR>(create(context, instance));
}

void WindowRefreshSize() {
	auto& b = B();
	void (*size)(void*, uint32_t*, uint32_t*) = nullptr;
	void* context                             = nullptr;
	{
		std::scoped_lock lock(b.window.mutex);
		size    = b.window.drawable_size;
		context = b.window.context;
	}
	if (size == nullptr || b.window_context == nullptr) {
		return;
	}
	uint32_t width  = 0;
	uint32_t height = 0;
	size(context, &width, &height);
	if (width == 0 || height == 0) {
		return;
	}
	auto&             window = *b.window_context;
	Common::LockGuard lock(window.mutex);
	window.graphic_ctx.screen_width  = width;
	window.graphic_ctx.screen_height = height;
}

} // namespace Aps5

extern "C" __declspec(dllexport) int Aps5GpuBackendCreate(const Aps5GpuHost* host, const Aps5GpuBackendApi** api,
                                                          Aps5GpuBackend** backend) {
	if (host == nullptr || api == nullptr || backend == nullptr) {
		return APS5_GPU_ERROR;
	}
	if (host->abi_version != APS5_GPU_BACKEND_ABI_VERSION || host->size < sizeof(Aps5GpuHost)) {
		return APS5_GPU_ABI_MISMATCH;
	}
	if (Aps5::g_backend != nullptr) {
		return APS5_GPU_ERROR;
	}
	Aps5::SetHost(host);
	Aps5::g_backend         = new Aps5::Backend;
	Aps5::g_backend->handle = Aps5GpuBackend {host};
	*api                    = &Aps5::KytyApi;
	*backend                = &Aps5::g_backend->handle;
	return APS5_GPU_OK;
}
