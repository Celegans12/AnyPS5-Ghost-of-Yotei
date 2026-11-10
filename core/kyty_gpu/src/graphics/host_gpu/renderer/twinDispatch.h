#pragma once

#include "common/onceValue.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <functional>

namespace Libs::Graphics {
class RenderContext;
struct PreparedBindings;
struct ShaderComputeInputInfo;
} // namespace Libs::Graphics

// KYTY_DIAG_TWIN_DISPATCH=<hex shader hash>[:<n>[:<hex variant bits>]] (diagnostic, default off; measurement only):
// an exact check, on the GPU, of a change of a compute shader's whole module.
// At most one dispatch (direct or indirect) of the shader per n presented frames (default 8) runs twice: first "B", the
// same dispatch with its program looked up with the variant bits toggled in ShaderComputeInputInfo::compute_variant
// (default 2 = ComputeVariantPairOff: the two-lanes module against the paired one; 4 =
// ComputeVariantPairSpin: spin exchanges against the barrier ones), then "A", today's program,
// whose results stay. Before B every storage image the dispatch binds (each candidate of a GPU-selected table is a binding
// of its own; all storage bindings are taken, read-only ones compare equal), its written storage buffers and the GDS it
// binds are copied aside (device memory); after B its results are copied out and the originals copied back, so A runs on
// what it would have run on; A's results are copied out too. Once the GPU finished, B's and A's copies are compared word
// by word: one "PERF twin_dispatch" line per twin (running totals twins / words / differ, the first difference ever, and
// this twin's own counts). Accepted only with differ=0; a run with the same module on both sides (the shader left out of
// the pair list) tells whether the shader itself is deterministic. B runs with A's bound descriptors and push data, so
// both programs must lay them out alike, or the twin is skipped ("PERF twin_dispatch ... skipped: <reason>"), as are
// programs that write through addresses or record-keyed ranges, images not in the general layout, depth, compressed or
// multisampled storage images, table-mode dispatches and more than 1 GiB to copy. Every copy is ordered by barriers and
// the commands after the dispatch wait for the last one; the GPU thread only records (no wait). Nothing runs while unset.
namespace Libs::Graphics::TwinDispatch {

// The B program and pipeline of a selected dispatch (program null: not selected).
struct Twin {
	const ShaderRecompiler::IR::CompiledShaderInfo* program  = nullptr;
	const PipelineCache::Pipeline*                  pipeline = nullptr;
};
[[nodiscard]] bool Requested();
[[nodiscard]] inline bool Enabled() {
	static constinit Common::OnceValue<bool> enabled_once;
	return enabled_once.Get([] { return Requested(); });
}
// `info`: the dispatch's inputs as they were before its own lookup. Looked up with the table mode off (today's
// descriptors), before anything of the dispatch is recorded (a program miss compiles here).
Twin LookUp(RenderContext& context, const HW::ComputeShaderInfo& regs, const HW::ShaderRegisters& sh, uint64_t hash,
            ShaderComputeInputInfo& info);
// Records the originals' copy, B (`dispatch` records its dispatch command), B's results' copy and the restore; A is
// recorded next by the caller (which binds A's pipeline again). False: no twin (not selected or skipped).
bool Begin(RenderContext& context, vk::CommandBuffer command, const Twin& twin,
           const ShaderRecompiler::IR::CompiledShaderInfo& program, const PipelineCache::Pipeline& pipeline,
           const PreparedBindings& bindings, const ShaderRecompiler::IR::ResourceSnapshot& resources,
           const std::function<void()>& dispatch);
// After A's dispatch command: A's results' copy, then the comparison once the GPU finished (priority thread).
void End(RenderContext& context, vk::CommandBuffer command, uint64_t hash);

} // namespace Libs::Graphics::TwinDispatch
