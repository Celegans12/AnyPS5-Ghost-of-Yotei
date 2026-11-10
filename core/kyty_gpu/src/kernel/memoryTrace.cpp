#include "kernel/memoryTrace.h"

#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/debugSample.h"
#include "graphics/host_gpu/renderer/perfCounters.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

// The logs of the resource walker's reads (memory.cpp calls them behind one test each; off unless their names are set).
namespace Libs::LibKernel::Memory {

// KYTY_TRACE_SRT_CLASSES (diagnostic, default off): the GPU thread's SRT reads, their waits for GPU writes and the
// Adopt work by what the word is for (shader-only or needed by the CPU) and by address region; one SRT_CLASSES line
// per ~5 s with per-frame averages. Decides whether letting the shaders read the shader-only words pays.
// KYTY_TRACE_SRT_CLASSES: waits per (shader, class) since the last report, for the SRT_CLASS_TOP line (the steady-state
// counterpart of the first-600 SRT_CLASS_WAIT log). Guarded: an adopt verify may wait off the GPU thread.
static std::mutex g_srt_wait_shaders_mutex;
struct SrtWaitByShader {
	uint64_t    ns    = 0;
	uint64_t    count = 0;
	std::string marker;
};
static std::map<std::pair<uint64_t, uint32_t>, SrtWaitByShader>& SrtWaitShaders() {
	static std::map<std::pair<uint64_t, uint32_t>, SrtWaitByShader> waits;
	return waits;
}

static void SrtClassesReport() {
	namespace Ir = Graphics::ShaderRecompiler::IR;
	static uint64_t last_ns    = 0;
	static uint32_t last_frame = 0;
	const uint64_t  now        = SrtClassesNowNs();
	const uint32_t  frame      = Graphics::DebugPresentCount().load();
	if (last_ns == 0) {
		last_ns    = now;
		last_frame = frame;
		Ir::SrtClasses::Get() = {};
		return;
	}
	if (now - last_ns < 5000000000ull || frame == last_frame) return;
	const double frames = static_cast<double>(frame - last_frame);
	last_ns             = now;
	last_frame          = frame;
	auto&                    c         = Ir::SrtClasses::Get();
	static const char* const classes[] = {"none", "cpu", "shader", "mixed", "unknown"};
	static const char* const regions[] = {"ring", "data", "r50", "other"};
	std::fprintf(stderr, "SRT_CLASSES frame=%u frames=%.0f per_frame", frame, frames);
	for (uint32_t phase = 0; phase < 2u; ++phase) {
		std::fprintf(stderr, " %s:", phase == 0u ? "walk" : "adopt_verify");
		for (uint32_t cls = 0; cls < Ir::SrtClasses::Classes; ++cls) {
			for (uint32_t region = 0; region < Ir::SrtClasses::Regions; ++region) {
				if (c.reads[phase][cls][region] == 0u && c.wait_ns[phase][cls][region] == 0u) continue;
				std::fprintf(stderr, " [%s.%s n=%.0f kb=%.1f wait_ms=%.3f waits=%.2f]", classes[cls], regions[region],
				             static_cast<double>(c.reads[phase][cls][region]) / frames,
				             static_cast<double>(c.bytes[phase][cls][region]) / 1024.0 / frames,
				             static_cast<double>(c.wait_ns[phase][cls][region]) / 1e6 / frames,
				             static_cast<double>(c.waits[phase][cls][region]) / frames);
			}
		}
	}
	std::fprintf(stderr, " adopted:");
	for (uint32_t cls = 0; cls < Ir::SrtClasses::Classes; ++cls) {
		for (uint32_t region = 0; region < Ir::SrtClasses::Regions; ++region) {
			if (c.adopted[cls][region] == 0u) continue;
			std::fprintf(stderr, " [%s.%s n=%.0f kb=%.1f]", classes[cls], regions[region],
			             static_cast<double>(c.adopted[cls][region]) / frames,
			             static_cast<double>(c.adopted_bytes[cls][region]) / 1024.0 / frames);
		}
	}
	std::fprintf(stderr, " rejected:");
	for (uint32_t cls = 0; cls <= Ir::SrtClasses::Classes; ++cls) {
		for (uint32_t region = 0; region < Ir::SrtClasses::Regions; ++region) {
			if (c.rejected[cls][region] == 0u) continue;
			std::fprintf(stderr, " [%s.%s n=%.1f]", classes[cls], regions[region],
			             static_cast<double>(c.rejected[cls][region]) / frames);
		}
	}
	std::fprintf(stderr, "\n");
	c = {};
	// The shaders whose walks waited longest in this report's frames (every wait, not only the >= 200 us ones).
	std::vector<std::pair<std::pair<uint64_t, uint32_t>, SrtWaitByShader>> top;
	{
		std::lock_guard lock(g_srt_wait_shaders_mutex);
		auto&           shaders = SrtWaitShaders();
		top.assign(shaders.begin(), shaders.end());
		shaders.clear();
	}
	std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
	std::fprintf(stderr, "SRT_CLASS_TOP frame=%u frames=%.0f", frame, frames);
	for (size_t i = 0; i < top.size() && i < 10u; ++i) {
		std::fprintf(stderr, " [%016llx %s ms=%.3f n=%.1f %s]", static_cast<unsigned long long>(top[i].first.first),
		             classes[top[i].first.second & 3u], static_cast<double>(top[i].second.ns) / 1e6 / frames,
		             static_cast<double>(top[i].second.count) / frames, top[i].second.marker.c_str());
	}
	std::fprintf(stderr, "\n");
}

void SrtClassesNoteRead(uint64_t vaddr, uint64_t size) {
	namespace Ir = Graphics::ShaderRecompiler::IR;
	auto&          c      = Ir::SrtClasses::Get();
	const uint32_t phase  = Ir::t_srt_read_phase & 1u;
	const uint32_t cls    = Ir::t_srt_read_class & 3u;
	const uint32_t region = Ir::SrtClasses::Region(vaddr);
	c.reads[phase][cls][region]++;
	c.bytes[phase][cls][region] += size;
	static uint32_t calls = 0;
	if ((++calls & 0xfffu) == 0u) SrtClassesReport();
}

// A wait of a GPU-thread read, counted by class and region; a long one (>= 200 us) is also logged with the shader and
// the game's marker path (first 600: SRT_CLASS_WAIT).
void SrtClassesWait(const char* where, uint32_t cls, uint64_t vaddr, uint64_t ns) {
	namespace Ir = Graphics::ShaderRecompiler::IR;
	Ir::SrtClasses::AddWait(cls, vaddr, ns);
	{
		const bool      gpu_thread = Graphics::GuestGpu::IsGpuThread();
		std::lock_guard lock(g_srt_wait_shaders_mutex);
		auto&           shader = SrtWaitShaders()[{Graphics::PerfMaterializeHash(), cls & 3u}];
		shader.ns += ns;
		shader.count++;
		if (shader.marker.empty() && gpu_thread) shader.marker = Graphics::DebugMarkerPath().substr(0, 80);
	}
	static uint32_t printed = 0;
	if (ns < 200000u || printed >= 600u) return;
	++printed;
	std::fprintf(stderr, "SRT_CLASS_WAIT frame=%u where=%s phase=%u cls=%u addr=0x%llx us=%llu hash=%016llx marker=%s\n",
	             Graphics::DebugPresentCount().load(), where, static_cast<uint32_t>(Ir::t_srt_read_phase), cls,
	             static_cast<unsigned long long>(vaddr), static_cast<unsigned long long>(ns / 1000u),
	             static_cast<unsigned long long>(Graphics::PerfMaterializeHash()), Graphics::DebugMarkerPath().c_str());
}

} // namespace Libs::LibKernel::Memory
