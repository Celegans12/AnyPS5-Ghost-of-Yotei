#pragma once

#include "common/envRead.h"
#include "common/onceValue.h"
#include "graphics/host_gpu/renderer/dispatchCensus.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>

// The compute dispatch paths' diagnostic tools (renderCompute.cpp calls them; every one is off unless its name is set):
// logs and dumps, the capture frame, and the experiments that change a dispatch's result. Each *On() is the tool's gate,
// read once: the only cost of a tool that is not set.
namespace Libs::Graphics {

// KYTY_TRACE_DISPATCH_CENSUS=1 (diagnostic, GPU thread): DispatchDirect's count and time by compute shader address, the top
// 16 by time every ~5 s (PERF dispatch_census lines: per frame count, ms, and us per dispatch).
struct DispatchCensusTimer {
	static bool Enabled() {
		static constinit Common::OnceValue<bool> on_once;
		return on_once.Get([] { return Common::EnvRead::DispatchCensus(); });
	}
	// 0 nothing known yet, 1 writes buffers or addresses (BDA / record-keyed ranges), 2 writes only images, 3 writes nothing.
	void SetClass(uint32_t cls) { m_class = cls; }
	// Phase boundaries of the full path (dispatch_phases lines): 1 program lookup start, 2 its end, 3 pipeline lookup
	// start, 4 bindings prepared, 5 buffers and images resolved, 6 bindings committed (barriers included); the tail runs
	// to the end. A dispatch that returns early counts only in the totals.
	static constexpr uint32_t kMarks = 7;
	void Mark(uint32_t i) {
		if (m_on) m_marks[i] = std::chrono::steady_clock::now();
	}
	explicit DispatchCensusTimer(uint64_t address, bool async_compute = false, bool indirect = false)
	    : m_address(address), m_kind((async_compute ? 2u : 0u) | (indirect ? 1u : 0u)), m_on(Enabled()) {
		if (m_on) Start();
	}
	~DispatchCensusTimer() {
		if (m_on) Finish();
	}
	// The dispatch's recipe key and what it writes.
	void NoteDispatch(const ShaderComputeInputInfo& input_info, uint32_t mode);

private:
	void Start();
	void Finish();
	// The recipe key of a dispatch (0: none) and where it was seen.
	void NoteRecipe(uint64_t key);
	uint64_t                              m_address;
	uint32_t                              m_kind;
	uint32_t                              m_class = 0;
	bool                                  m_on;
	std::chrono::steady_clock::time_point m_start {};
	std::array<std::chrono::steady_clock::time_point, kMarks> m_marks {};
};

} // namespace Libs::Graphics

namespace Libs::Graphics::ComputeTrace {

#define ENV_GATE(function, name)                                                                                             \
	[[nodiscard]] inline bool function() {                                                                                   \
		static constinit Common::OnceValue<bool> on_once;                                                                    \
		return on_once.Get([] { return std::getenv(name) != nullptr; });                                                     \
	}
ENV_GATE(GuestWatchOn, "KYTY_TRACE_GUEST_WATCH")
ENV_GATE(ProxyPhaseOn, "KYTY_TRACE_PROXY_PHASE")
ENV_GATE(CsImageDumpOn, "KYTY_TRACE_CS_IMAGE_DUMP")
ENV_GATE(ObjCensusOn, "KYTY_TRACE_OBJ_CENSUS")
ENV_GATE(GuestPokeOn, "KYTY_DIAG_GUEST_POKE")
#undef ENV_GATE
[[nodiscard]] inline bool ProxyCountsOn() {
	static constinit Common::OnceValue<bool> on_once;
	return on_once.Get([] { return Common::EnvRead::ProxyCounts(); });
}

// KYTY_TRACE_CS_WRITER_ADDR (0: not set).
[[nodiscard]] inline uint64_t CsWriterAddr() {
	static constinit Common::OnceValue<uint64_t> writer_addr_once;
	return writer_addr_once.Get([] { return Common::EnvRead::CsWriterAddr(); });
}

using Program   = ShaderRecompiler::IR::CompiledShaderInfo;
using Resources = ShaderRecompiler::IR::ResourceSnapshot;

// Logs and dumps.
namespace GuestWatch {
void Poll(RenderContext& context, uint64_t next_hash);
} // namespace GuestWatch
namespace ObjCensus {
void Poll();
} // namespace ObjCensus
namespace GuestPoke {
void Poll();
} // namespace GuestPoke
void CsImageDumpTrace(RenderContext& context, vk::CommandBuffer vk_buffer, const Program& program, PreparedBindings& bindings,
                      const Resources* resources = nullptr);
void CsImageDumpAfter(RenderContext& context, vk::CommandBuffer vk_buffer, const Program& program, PreparedBindings& bindings);
void ProxyCountsDump(RenderContext& context, vk::CommandBuffer vk_buffer, const Program& program, const Resources* resources);
void ProxyPhaseNote(uint64_t shader_hash, const Resources* resources);
void NoteRtDispatch(const Program& program, const Resources& resources);
void LogDispatch(RenderContext& context, uint64_t cs_address, const ShaderComputeInputInfo& input_info, uint32_t thread_group_x,
                 uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode);
void CsWriterTrace(uint64_t writer_addr, const Program& program, const Resources& resources, uint32_t thread_group_x,
                   uint32_t thread_group_y, uint32_t thread_group_z);

// The capture frame.
void CapturePreDirect(RenderContext& context, CommandBuffer& buffer, vk::CommandBuffer vk_buffer, const Program& program,
                      const Resources& resources);
void CapturePreIndirect(RenderContext& context, CommandBuffer& buffer, vk::CommandBuffer vk_buffer, const Program& program,
                        const Resources& resources);
void CaptureAfterDirect(RenderContext& context, CommandBuffer& buffer, vk::CommandBuffer vk_buffer, const Program& program,
                        const Resources& resources, uint32_t thread_group_x, uint32_t thread_group_y, uint32_t thread_group_z);
void CaptureAfterIndirect(RenderContext& context, CommandBuffer& buffer, vk::CommandBuffer vk_buffer, const Program& program,
                          const Resources& resources, uint64_t args_addr, const Buffer& args_buffer, uint64_t args_offset);

} // namespace Libs::Graphics::ComputeTrace
