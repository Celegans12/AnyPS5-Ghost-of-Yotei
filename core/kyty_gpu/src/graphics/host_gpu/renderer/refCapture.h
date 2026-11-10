#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_REFCAPTURE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_REFCAPTURE_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

namespace Libs::Graphics {

class RenderContext;
struct PreparedBindings;
struct ShaderComputeInputInfo;
namespace HW {
struct ComputeShaderInfo;
} // namespace HW
namespace ShaderRecompiler::IR {
struct CompiledShaderInfo;
struct ResourceSnapshot;
} // namespace ShaderRecompiler::IR

// KYTY_TRACE_REF_CAPTURE=<cs hash hex>[,<hash>...][:<first presented frame>] (diagnostic; directory from
// KYTY_TRACE_REF_DIR, else KYTY_TRACE_FRAME_DIR): one direct or indirect dispatch of that compute shader is captured
// for the offline RDNA2 reference interpreter (tools/refinterp): registers, user data, shader code (read like
// the recompiler does: backing store first, then the host mapping for shaders embedded in a module image), the
// guest memory reachable from the user-data pointers (crawled pages, GPU-written bytes synced first),
// and every bound buffer and image as the GPU sees them right before the dispatch, plus the written
// ones right after it. Nothing of the dispatch itself changes, except that the copies are fenced in: the
// deferred compute access barrier and a full memory barrier are recorded
// before them, and a read -> write barrier after them. Buffers are copied from the VkBuffer/offset the dispatch
// is bound to (meta.txt: backing=cache|stream bound_offset=, guest_nz= / host_nz= of the CPU views), indexed by
// the shader's program buffer index like the descriptor in meta.txt.
namespace RefCapture {

// True for the one dispatch to capture (first match at or after the frame); claims it.
bool Claim(uint64_t shader_hash);
// Reads the guest memory of the claimed dispatch (shader code, pages reachable from the user data). Call it right
// after a successful Claim, before anything of the dispatch is recorded: syncing GPU-written pages submits the
// command buffer being recorded, which loses whatever was already recorded for the dispatch (descriptor binds) and
// every capture copy recorded afterwards through an earlier handle. Before/BeforeIndirect read the memory themselves
// when this was not called.
void Prefetch(const HW::ComputeShaderInfo& cs);

void Before(RenderContext& context, vk::CommandBuffer command, const HW::ComputeShaderInfo& cs,
            const ShaderComputeInputInfo& input_info, const ShaderRecompiler::IR::CompiledShaderInfo& program,
            PreparedBindings& bindings, const ShaderRecompiler::IR::ResourceSnapshot& resources, uint32_t groups_x,
            uint32_t groups_y, uint32_t groups_z);

// Same capture for an indirect dispatch. The group counts live in args_buffer at args_offset (a
// vk::DispatchIndirectCommand, possibly written by the GPU), so they are not read back while recording
// (that would submit the command buffer being recorded): a GPU copy of the 12 bytes is queued like the
// buffer dumps and lands in <capture dir>/dispatch_args.bin one present later. meta.txt gets groups=0,0,0
// and indirect_args=dispatch_args.bin; every other field is the same as for a direct dispatch. Call it
// right before bindPipeline/dispatchIndirect, then After() right after the dispatch.
void BeforeIndirect(RenderContext& context, vk::CommandBuffer command, const HW::ComputeShaderInfo& cs,
                    const ShaderComputeInputInfo& input_info, const ShaderRecompiler::IR::CompiledShaderInfo& program,
                    PreparedBindings& bindings, const ShaderRecompiler::IR::ResourceSnapshot& resources,
                    vk::Buffer args_buffer, uint64_t args_offset);

void After(RenderContext& context, vk::CommandBuffer command, const ShaderRecompiler::IR::CompiledShaderInfo& program,
           PreparedBindings& bindings, const ShaderRecompiler::IR::ResourceSnapshot& resources);

} // namespace RefCapture
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_REFCAPTURE_H_
