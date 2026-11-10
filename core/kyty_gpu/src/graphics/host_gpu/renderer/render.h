#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/pipeline/bindlessTable.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <unordered_set>
#include <array>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

namespace HW {
class Context;
class UserConfig;
class Shader;
} // namespace HW

// While set on a thread (an operation's execution, a draw's translation), the register accessors of every
// CommandBuffer return this draw's copy of the registers instead of the live state the PM4 handlers write.
struct DrawRegsOverride {
	HW::Context*    ctx  = nullptr;
	HW::UserConfig* ucfg = nullptr;
	HW::Shader*     sh   = nullptr;
};
inline thread_local DrawRegsOverride t_draw_regs;
// The draw-state observer's answer for the draw being processed (graphicsRun.cpp, drawStateObserver.h; measurement only,
// KYTY_TRACE_CLEAN_DRAWS' observer_clean column):
// nothing but user data, index and instance state changed since the previous direct draw.
inline thread_local bool t_draw_observer_clean = false;
// KYTY_TRACE_CLEAN_DRAWS: the observer's first cause of dirt for that draw (DrawStateObserver::Cause; 0 when clean).
inline thread_local uint32_t t_draw_observer_cause = 0;

// Which register structures a PM4 packet may write (graphicsRun.cpp RegistersWrittenBy, for DrawStateObserver).
inline constexpr uint32_t RegsWrittenCtx  = 1u;
inline constexpr uint32_t RegsWrittenUcfg = 2u;
inline constexpr uint32_t RegsWrittenSh   = 4u;
inline constexpr uint32_t RegsWrittenAll  = 7u;



struct GraphicContext;
struct TableRefreshBatch;
struct ShaderBufferResource;
struct ShaderComputeInputInfo;
struct RenderDepthInfo;
struct RenderColorInfo;
struct RenderDepthInfo;
// D6 step K5: a draw's color and depth target descriptions, for the scalar-read check (RenderExecutor::m_commit_targets).
struct CommitTargets {
	const RenderColorInfo* colors      = nullptr;
	uint32_t               color_count = 0;
	const RenderDepthInfo* depth       = nullptr;
};
struct DrawCallInfo;
struct DrawEmitInfo;
struct DrawIndexBufferSource;
struct DrawIndirectSource;
struct DrawPlan;
struct DrawSlot;
struct DrawRenderState;
class RenderContext;
class CommandScheduler;
struct RenderExecutorTestAccess;

enum class CommandBufferDebugOp : uint32_t {
	DispatchDirect,
	DrawIndex,
	DrawIndexAuto,
	DispatchIndirect = 9,
	Unknown,
};

enum class DrawOffsetSource : uint8_t {
	DrawState,
	IndirectArgs,
};

// Nonzero address: the GPU reads count Vulkan indirect draw records, stride bytes apart.
struct DrawIndirectArgs {
	uint64_t address = 0;
	uint32_t count   = 0;
	uint32_t stride  = 0;
};

struct DrawIndexArgs {
	uint32_t         index_count                = 0;
	const void*      index_addr                 = nullptr;
	uint32_t         instance_count             = 0;
	uint32_t         index_type_and_size        = 0;
	int32_t          base_vertex                = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
	// With indirect records, index_count is INDEX_BUFFER_SIZE.
	DrawIndirectArgs indirect;
};

struct DrawAutoArgs {
	uint32_t         vertex_count               = 0;
	uint32_t         instance_count             = 0;
	uint32_t         first_vertex               = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
	DrawIndirectArgs indirect;
};

// GPU indirect draws: an indirect draw packet the GPU executes with the guest argument
// records (PS5 layouts equal VkDraw[Indexed]IndirectCommand) and count as they are in memory.
struct DrawIndirectGpuArgs {
	uint64_t               args_addr           = 0;
	uint64_t               count_addr          = 0; // 0: exactly max_count draws
	uint32_t               max_count           = 0;
	uint32_t               stride              = 0;
	uint64_t               index_base_addr     = 0;
	uint32_t               index_buffer_size   = 0; // in indices (INDEX_BUFFER_SIZE)
	uint32_t               index_type_and_size = 0;
	bool                   indexed             = false;
	ShaderIndirectDrawInfo indirect_draw;
};

struct SubmitInfo {
	static constexpr uint32_t MaxSemaphores = 3;

	std::array<vk::Semaphore, MaxSemaphores>          wait_semaphores {};
	std::array<uint64_t, MaxSemaphores>               wait_ticks {};
	std::array<vk::PipelineStageFlags, MaxSemaphores> wait_stages {};
	std::array<vk::Semaphore, MaxSemaphores>          signal_semaphores {};
	std::array<uint64_t, MaxSemaphores>               signal_ticks {};
	uint32_t                                          num_wait_semaphores   = 0;
	uint32_t                                          num_signal_semaphores = 0;

	void AddWait(vk::Semaphore semaphore, uint64_t tick = 1,
	             vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eAllCommands) {
		EXIT_IF(semaphore == nullptr || num_wait_semaphores >= MaxSemaphores);
		wait_semaphores[num_wait_semaphores] = semaphore;
		wait_ticks[num_wait_semaphores]      = tick;
		wait_stages[num_wait_semaphores++]   = stage;
	}

	void AddSignal(vk::Semaphore semaphore, uint64_t tick = 1) {
		EXIT_IF(semaphore == nullptr || num_signal_semaphores >= MaxSemaphores);
		signal_semaphores[num_signal_semaphores] = semaphore;
		signal_ticks[num_signal_semaphores++]    = tick;
	}
};

// The record packets (renderDraw.cpp): `buffer`'s GraphicsState was forgotten by the recording side; the recording
// thread forgets its copy at this point of the stream (a packet), when its draws' dynamic state can be packets.
void NoteGraphicsStateInvalidated(vk::CommandBuffer buffer) noexcept;

class CommandBuffer {
public:
	~CommandBuffer() = default;

	KYTY_CLASS_NO_COPY(CommandBuffer);

	[[nodiscard]] bool IsInvalid() const;

	void SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0 = 0, uint32_t arg1 = 0,
	                  uint32_t arg2 = 0, uint32_t arg3 = 0, uint64_t arg4 = 0);
	void BeginRendering(const RenderState& state) const;
	void EndRendering() const;
	[[nodiscard]] bool IsRendering() const {
		return m_rendering;
	}

	// KYTY_FIX_DRAW_STATE_FILTER: the graphics pipeline and dynamic state draws last recorded into
	// this command buffer, so that a draw skips the commands that would set the same values again.
	// Vulkan (Pipelines, "Dynamic State"): dynamic state "persists for the lifetime of the command
	// buffer, or until modified by another dynamic state setting command or made invalid by binding
	// a pipeline in which that state is statically specified"; a pipeline that declares the state
	// dynamic leaves it undisturbed. BeginRendering/EndRendering and compute pipeline binds do not
	// touch it. Every graphics pipeline a draw binds declares the same dynamic states, except color
	// write enables when it has no color attachments (pipeline/shaders.cpp). Whatever else binds a
	// graphics pipeline or records dynamic state into this command buffer (the MSAA depth blit, the
	// present and its overlays) must call InvalidateGraphicsState; a new command buffer does.
	struct GraphicsState {
		static constexpr uint32_t ViewportSlots = 16;

		enum Kind : uint32_t {
			Viewports,
			Scissors,
			LineWidth,
			BlendConstants,
			DepthTest,
			DepthWrite,
			DepthCompare,
			DepthBoundsTest,
			DepthBounds,
			DepthBiasEnable,
			DepthBias,
			StencilTest,
			StencilFront,
			StencilBack,
			ColorWrite,
			FeedbackLoop,
			PipelineBind,
			KindCount
		};

		[[nodiscard]] bool Has(Kind kind) const noexcept { return (known & (1u << kind)) != 0; }
		void               Set(Kind kind) noexcept { known |= 1u << kind; }
		void               Forget(Kind kind) noexcept { known &= ~(1u << kind); }

		uint32_t                                known = 0; // One bit per Kind: the value below is held.
		vk::Pipeline                            pipeline;  // With PipelineBind: the bound graphics pipeline.
		uint32_t                                viewport_count = 0;
		std::array<vk::Viewport, ViewportSlots> viewports {};
		uint32_t                                scissor_count = 0;
		std::array<vk::Rect2D, ViewportSlots>   scissors {};
		float                                   line_width = 0.0f;
		std::array<float, 4>                    blend_constants {};
		vk::Bool32                              depth_test        = VK_FALSE;
		vk::Bool32                              depth_write       = VK_FALSE;
		vk::CompareOp                           depth_compare     = vk::CompareOp::eNever;
		vk::Bool32                              depth_bounds_test = VK_FALSE;
		std::array<float, 2>                    depth_bounds {};
		vk::Bool32                              depth_bias_enable = VK_FALSE;
		std::array<float, 3>                    depth_bias {};
		vk::Bool32                              stencil_test = VK_FALSE;
		std::array<vk::StencilOpState, 2>       stencil {};
		uint32_t                                color_write_count = 0;
		std::array<vk::Bool32, RENDER_COLOR_ATTACHMENTS_MAX> color_write {};
		vk::ImageAspectFlags                    feedback;
		// The record packets: which copy of this state the draws of the command buffer use (renderDraw.cpp): 0 none
		// yet, 1 this one (the recording side's), 2 the recording thread's (the draws' dynamic-state packets). A draw that
		// switches forgets both copies first.
		uint8_t                                 keeper = 0;
	};
	[[nodiscard]] GraphicsState& GetGraphicsState() const noexcept { return m_graphics_state; }
	// Forgets what the command buffer holds; with the record packets the recording thread's copy too (context.cpp).
	void InvalidateGraphicsState() const noexcept;

	[[nodiscard]] vk::CommandBuffer Handle() const;
	[[nodiscard]] GraphicContext&   GetGraphics() const noexcept { return m_graphics; }
	[[nodiscard]] RenderContext&    GetContext() const noexcept { return m_context; }
	[[nodiscard]] HW::Context&      GetRegisters() const noexcept {
		if (t_draw_regs.ctx != nullptr) return *t_draw_regs.ctx;
		return *m_registers;
	}
	[[nodiscard]] HW::UserConfig&   GetUserConfig() const noexcept {
		if (t_draw_regs.ucfg != nullptr) return *t_draw_regs.ucfg;
		return *m_user_config;
	}
	[[nodiscard]] HW::Shader&       GetShaders() const noexcept {
		if (t_draw_regs.sh != nullptr) return *t_draw_regs.sh;
		return *m_shaders;
	}

private:
	explicit CommandBuffer(CommandScheduler& scheduler);
	void Bind(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders) noexcept {
		m_registers   = &registers;
		m_user_config = &user_config;
		m_shaders     = &shaders;
	}

	void Begin();
	void End() const;

	RenderContext&      m_context;
	GraphicContext&     m_graphics;
	vk::CommandBuffer   m_buffer          = nullptr;
	uint32_t            m_debug_op        = 0;
	uint64_t            m_debug_submit_id = 0;
	uint32_t            m_debug_arg0      = 0;
	uint32_t            m_debug_arg1      = 0;
	uint32_t            m_debug_arg2      = 0;
	uint32_t            m_debug_arg3      = 0;
	uint64_t            m_debug_arg4      = 0;
	mutable RenderState m_render_state;
	mutable bool        m_rendering   = false;
	mutable GraphicsState m_graphics_state;
	HW::Context*        m_registers   = nullptr;
	HW::UserConfig*     m_user_config = nullptr;
	HW::Shader*         m_shaders     = nullptr;

	friend class CommandScheduler;
};

class RenderExecutor {
public:
	explicit RenderExecutor(RenderContext& context): m_context(context) {}
	~RenderExecutor();
	// The resolve thread, ahead of the operation's execution: the programs of its draw or dispatch, from its register
	// snapshots, which stay valid until it executed; null when there is nothing to resolve. Based on
	// TheCruZ/KytyPS5-GTA daaf45201 (render.h), GPL-2.0.
	struct ResolvedDraw;
	[[nodiscard]] ResolvedDraw* ResolveDrawAhead(const HW::Context& context, const HW::UserConfig& user_config,
	                                             const HW::Shader& shaders, const ShaderIndirectDrawInfo& indirect_draw);
	[[nodiscard]] ResolvedDraw* ResolveDispatchAhead(const HW::Shader& shaders, uint32_t groups_x, uint32_t groups_y,
	                                                 uint32_t groups_z, uint32_t mode, bool indirect, bool async_compute);
	// The execution thread, around the operation. Release gives the resolution back.
	void        UseResolvedDraw(ResolvedDraw* resolved) noexcept;
	static void ReleaseResolvedDraw(ResolvedDraw* resolved) noexcept;
	// KYTY_DIAG_GPU_BURN_NS / KYTY_DIAG_GPU_BURN_AT (measurement only): GPU-thread spin at one of these places.
	// InvalidateCaller: the guest thread whose write fault reads GPU-modified bytes back, not the GPU thread.
	// Readback and Pm4 (measurement only): a GPU-thread guest read of GPU-written data, a PM4 packet of the GPU thread.
	enum class GpuBurnSite : uint32_t { None, Indirect, Dispatch, DispatchAsync, DispatchGfx, Readback, Pm4, InvalidateCaller, DirectWait, Draw };
	static void GpuBurnAt(GpuBurnSite site);
	// The site KYTY_DIAG_GPU_BURN_AT selects (read once): lets a hot loop decide once whether to call GpuBurnAt at all.
	[[nodiscard]] static GpuBurnSite GpuBurnSelected();
	KYTY_CLASS_NO_COPY(RenderExecutor);

	void DispatchDirect(uint64_t submit_id, CommandBuffer& buffer, uint32_t thread_group_x,
	                    uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode, bool async_compute);
	void DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer, uint64_t args_addr,
	                      uint32_t mode, bool async_compute);

	void PrepareBindings(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	void PrepareBindingsToday(const ShaderStageRuntime& runtime, PreparedBindings& prepared); // KYTY_DIAG_BIND_REPLAY
	void                           FindBuffers(std::span<PreparedBindings* const> stages);
	void                           RebindBuffers(PreparedBindings& bindings);
	void                           RebindImages(PreparedBindings& bindings);
	void CommitBindings(CommandBuffer& buffer, vk::PipelineBindPoint pipeline_bind_point,
	                    const PipelineCache::Pipeline&     pipeline,
	                    std::span<PreparedBindings* const> bindings);
	// KYTY_BINDLESS: register the draw's heaps, patch their regions, and once per frame resolve
	// the keys shaders flagged as pending.
	void PrepareBindlessHeaps(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	void ResolveBindlessRequests();
	bool ValidateBindlessHeap(BindlessTable::Heap& heap); // true when the guest rewrote entries
	void PrefillBindlessHeap(BindlessTable::Heap& heap, uint32_t words);
	void ReleaseUnusedBindlessSlots(uint64_t frame);
	void RefreshBindlessDescriptor(BindlessTable::Heap& heap, const std::array<uint32_t, 8>& words,
	                               BindlessTable::DescriptorSlot& cached);
	// immediate: resolved while preparing a draw, not from feedback (the key trace names the origin).
	bool ResolveBindlessKey(BindlessTable::Heap& heap, uint32_t entry, bool immediate = false);
	// The bindless tables (bindlessTable.h): the heap of a draw as the table it is, translated before the draw;
	// false only for an empty use (nothing to translate: the heap path takes it as before); a table it cannot take stops
	// the emulator with the reason (bindless E2c).
	[[nodiscard]] bool PrepareBindlessTable(uint64_t shader_hash, const ShaderRecompiler::IR::BindlessHeapUse& use,
	                                        const ShaderRecompiler::IR::ImageResource& resource,
	                                        PreparedBindings&                          prepared);
	// Every field of every record of m_table_words (the table's first `covered` bytes) into the table's values and
	// region; false when the arena has no room. shader_hash: the reader, for the BINDLESS_TABLE layout log.
	[[nodiscard]] bool TranslateBindlessTable(BindlessTable::Table& table, uint32_t covered, uint64_t hash,
	                                          uint64_t frame, uint64_t shader_hash);
	// The table slot of one descriptor, resolving its texture now when no table slot shows it yet; 0 = placeholder.
	[[nodiscard]] uint32_t ResolveTableDescriptor(BindlessTable::Table& table, const std::array<uint32_t, 8>& words,
	                                              uint64_t frame);
	// Looks the table's textures up again once per presented frame (every draw for images the GPU writes), as the
	// heap path does for its resolved keys, and lists their images for the draw.
	void RefreshTableSlots(BindlessTable::Table& table, uint64_t frame);
	// Refresh events: appends the table's images a draw's layout walk must see (all of them after the set changed, else
	// those logged in g_bindless_layout_log since and the GPU-written ones).
	void ListTableImagesByEvents(BindlessTable::Table& table, std::vector<ImageId>& list);
	[[nodiscard]] bool RefreshTableSlot(BindlessTable::Table& table, uint32_t slot, uint64_t frame,
	                                    TableRefreshBatch& batch); // true: remapped
	// Once per presented frame: stale tables and unused table slots go.
	void PrepareTablesFrame(uint64_t frame);
	// KYTY_PERF_BINDLESS_CLASSES: registers an integer class for a float resource of one field whose records (the
	// table's last translation) hold integer T#s; its next materialization types it.
	void LearnBindlessClass(uint64_t shader_hash, const ShaderRecompiler::IR::BindlessHeapUse& use,
	                        const ShaderRecompiler::IR::ImageResource& resource, const BindlessTable::Table& table);

private:
	friend struct DrawPlan; // S2 (renderDrawPlan.h): a draw's plan holds its GraphicsBindings
	friend struct DrawSlot; // S2 step B.4b (renderDraw.cpp): everything of one draw
	// Upstream's image heap (imageHeap.cpp) resolves its elements with ResolveTexture; it is never in use here
	// (RenderContext::ImageHeapEnabled), but compiles.
	friend class ImageHeap;
	// False, with nothing drawn, when the draw's indirect records need CPU counts (upstream c513a4a13, 949de19bb,
	// 86fed10b8): args.indirect with a nonzero address. Our path does not take GPU-read records (the GPU indirect path
	// and the CPU expansion in graphicsRun.cpp do), so such a call always returns false; every other call draws and
	// returns true.
	bool DrawIndex(uint64_t submit_id, CommandBuffer& buffer, const DrawIndexArgs& args);
	bool DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args);
	// nullptr: the draw ran (or was skipped) on the GPU path. Otherwise the reason it is unsupported,
	// returned before any draw command is recorded; the caller expands it on the CPU instead.
	[[nodiscard]] const char* DrawIndirectGpu(uint64_t submit_id, CommandBuffer& buffer,
	                                          const DrawIndirectGpuArgs& args);

	struct GraphicsBindings {
		std::array<PreparedBindings, 3> vertex;
		std::optional<PreparedBindings> pixel;
	};

	// compare_funcs: DEPTH_COMPARE_FUNC bits (1 << func) of the S#s that compare-sample the image, for the null
	// image of a depth-compare binding (descriptors.cpp NullCompareBinding); 0 when unknown.
	[[nodiscard]] TextureBinding ResolveTexture(const ShaderRecompiler::IR::ImageResource& resource,
	                                            const ShaderRecompiler::IR::DescriptorValue& value,
	                                            uint32_t compare_funcs = 0);
	// ResolveTexture's texture cache lookup and checks for a description (fresh or cached).
	[[nodiscard]] TextureBinding FindResolvedTexture(const ShaderRecompiler::IR::ImageResource& resource,
	                                                 const ShaderTextureResource&               descriptor,
	                                                 TextureCache::ImageDesc& desc, bool shader_conversion,
	                                                 vk::Format pixel_format, vk::Format view_format);
	void PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
	                             std::span<RenderColorInfo> colors);
	// PrepareGraphicsBindings' parts in their order - FindBuffers and PrepareBda, RebindImages and the render targets,
	// RebindBuffers.
	void PrepareGraphicsBuffersHead(std::span<PreparedBindings* const> stages);
	void PrepareGraphicsImages(std::span<PreparedBindings* const> stages, std::span<RenderColorInfo> colors);
	void PrepareGraphicsBuffersTail(std::span<PreparedBindings* const> stages);
	void ResolveRenderColorTarget(CommandBuffer& buffer, RenderColorInfo& target,
	                              uint32_t render_target_slice_offset, uint32_t render_target_slot,
	                              bool ignore_target_mask = false, bool exact_format = false);
	void ResolveRenderDepthTarget(CommandBuffer& buffer, RenderDepthInfo& target);
	[[nodiscard]] bool DepthStencilCopy(CommandBuffer& buffer);
	[[nodiscard]] bool CopyDepthToColor(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                   const DrawEmitInfo& emit, DrawRenderState& state);
	[[nodiscard]] bool PrepareDrawRenderState(CommandBuffer& buffer,
	                                          const DrawCallInfo& draw,
	                                          uint32_t            render_target_slice_offset,
	                                          DrawRenderState& state);
	void ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer, const DrawCallInfo& draw,
	                         DrawRenderState& state, vk::PrimitiveTopology topology,
	                         const DrawEmitInfo& emit, const DrawIndexBufferSource& index_source,
	                         bool                      primitive_restart_enable,
	                         const DrawIndirectSource* indirect = nullptr);
	// S1: the part of a draw after its "cannot touch guest memory" point,
	// from what ExecutePreparedDraw prepared (the plan).
	void CommitPreparedDraw(DrawPlan& plan);
	// The steps the draw entries share. DrawEntryStart: DrawIndex and DrawAuto.
	// ColorOperationInstead: a metadata, copy or resolve operation took the place of the draw (true: it ends here).
	void DrawEntryStart(CommandBuffer& buffer);
	[[nodiscard]] bool ColorOperationInstead(CommandBuffer& buffer, uint32_t render_target_slice_offset, const char* drop_reason);
	// The slot of the draw being handled (never null: every draw entry takes one before its plan).
	DrawSlot* NextDrawSlot();
	uint64_t  m_async_skipped_draws = 0; // draws not issued while a worker creates their pipeline
	struct ResolvedDrawRing;
	[[nodiscard]] ResolvedDraw& NextResolvedDraw();
	// The draw state the resolve thread made for the operation's draw, its selection adopted (null: the draw selects
	// itself): only a draw of the operation's own registers and indirect layout, once.
	[[nodiscard]] DrawRenderState* TakeResolvedState(const ShaderIndirectDrawInfo& indirect_draw);
	// The same for a direct dispatch: `input_info` and `program` as GetComputeProgram would leave them (false: it runs).
	[[nodiscard]] bool TakeResolvedDispatch(ShaderComputeInputInfo& input_info, ShaderProgram& program);
	ResolvedDraw* m_resolved_draw  = nullptr; // the operation's
	bool          m_state_resolved = false;   // the draw being prepared adopted its state
	std::unique_ptr<ResolvedDrawRing, void (*)(ResolvedDrawRing*)> m_resolved_ring {nullptr, nullptr};
	DrawSlot* m_draw_slot    = nullptr; // allocated on first use
	DrawSlot* m_current_slot = nullptr;
	// S2 step B.4f: the bindless in-draw marks the commit being made covers (SIZE_MAX: all).
	size_t    m_commit_bindless_heaps  = SIZE_MAX;
	size_t    m_commit_bindless_tables = SIZE_MAX;
	uint64_t  m_commit_bindless_plan   = 0; // the committed plan's serial (0: not a plan)
	[[nodiscard]] RenderState AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
	                                               uint32_t color_count, RenderDepthInfo& depth,
	                                               vk::ImageAspectFlags& feedback_aspects,
	                                               std::span<PreparedBindings* const> stages = {});
	[[nodiscard]] bool        ResolveColorTargets(CommandBuffer& buffer,
	                                              uint32_t render_target_slice_offset);
	[[nodiscard]] bool        ConsumeMetadataColorOperation(CommandBuffer& buffer,
	                                                       uint32_t render_target_slice_offset);
	void                      BindImage(ImageId id, bool storage);
	void                      BindRenderTarget(ImageId id);
	// The plan flags (S2 step B.3): the image binding flags of the draw being prepared stay in its plan (not in the images)
	// until its commit applies them.
	struct PlanImageFlags {
		std::vector<std::pair<ImageId, ImageBinding>> flags;
		std::vector<ImageId>                          bound;
		uint64_t                                      hash = 0;
	};
	void                      BeginDrawPlan();
	static PlanImageFlags&    SlotFlags(DrawSlot& slot);
	void                      ApplyPlanFlags();
	void                      ReleasePlanFlags(PlanImageFlags& plan);
	ImageBinding&             PlanFlags(PlanImageFlags& plan, ImageId id);
	// D6 step K5 (the scalar-read check): the draw's own target descriptions while CommitPreparedDraw commits it
	// (null otherwise, e.g. for dispatches): the scalar-read check tests reads against the bytes they describe.
	const CommitTargets* m_commit_targets = nullptr;
	[[nodiscard]] PlanImageFlags* PrepFlags() const;
	void                          SetPrepFlags(PlanImageFlags* flags);
	void                      ClearImageBinding(Image& image, ImageId id);
	PlanImageFlags*               m_prep_flags       = nullptr; // the draw being prepared
	PlanImageFlags*               m_applied_flags    = nullptr; // the draw committed (released by ResetBindings)
	bool                          m_texture_plan_transfer_set = false;
	void                      ResetBindings();
	void                      UpdateComputeMetaClears(const ShaderComputeInputInfo& input,
	                                                  const CommandBuffer&          buffer);
	[[nodiscard]] bool TryConsumeComputeImageClear(const ShaderComputeInputInfo& input,
	                                              CommandBuffer& command, uint32_t group_x,
	                                              uint32_t group_y, uint32_t group_z, uint32_t mode);

	RenderContext&                        m_context;
	uint64_t                              m_bindless_frame = UINT64_MAX;
	std::vector<uint32_t>                 m_bindless_requests;
		static inline thread_local std::vector<uint32_t> m_bindless_srt;
	// The last bindless image lists CommitBindings walked, with the epochs of that walk.
	struct BindlessLayoutMemo {
		std::vector<ImageId> ids;
		uint64_t             registry = 0;
		uint64_t             layouts  = 0;
	};
	std::array<BindlessLayoutMemo, 8> m_bindless_layout_memos {};
	uint32_t                          m_bindless_layout_memo_next = 0;
	// A running hash of this draw's image binds (BindImage / BindRenderTarget: id and order), reset by ResetBindings.
	uint64_t                          m_bound_hash = 0;
	// Uniform const buffers (ShaderIR.h): words padded to whole uvec4 elements before an upload read as a uniform buffer.
	// The soft-resolve flag below: set and reset around a soft resolve; read by its own image slots and table-set keys.
	static inline thread_local std::vector<uint32_t> m_uniform_words;
	static inline thread_local bool                  m_soft_texture_resolve = false;
	bool                                  m_bindless_cached      = false;
	uint64_t                              m_bindless_draw             = 0;
	std::vector<uint32_t>                 m_bindless_words;
	std::unordered_map<uint64_t, uint64_t> m_bindless_refreshed; // image -> last refreshed frame
	uint64_t                              m_bindless_stats_frame = 0;
	std::vector<BindlessTable::Heap*>     m_bindless_draw_heaps; // heaps of the stage being prepared
	// The bytes of the table being checked, the fields of the use being prepared, the
	// translation stamp (TableSlotInfo::stamp) and the frames of the per-frame work.
	std::vector<uint32_t>                 m_table_words;
	std::vector<BindlessTable::TableField> m_table_fields;
	uint32_t                              m_table_stamp         = 0;
	std::vector<ImageId>                  m_table_touch; // refresh events: unchanged slots' images (RefreshTableSlots)
	// The slot journal: a translated table's slots before it (TranslateBindlessTable).
	std::vector<std::pair<uint32_t, uint32_t>> m_table_old_slots;
	uint64_t                              m_table_frame         = 0;
	uint64_t                              m_table_release_frame = 0;
	PreparedBindings                     m_compute_bindings;
	// KYTY_WORKAROUND_STENCIL_STORAGE_SHADOW: storage writes to a stencil plane go to an R8_UINT shadow
	// that is filled from the stencil before the dispatch and copied back after it.
	struct StencilShadow {
		ImageId shadow;
		ImageId depth;
	};
	std::vector<StencilShadow> m_stencil_shadows;
	std::unique_ptr<Buffer>    m_stencil_copy_buffer;
	void CopyStencilShadow(vk::CommandBuffer command, const StencilShadow& pair, bool to_stencil);
	void FlushStencilShadows(vk::CommandBuffer command);
	void CopyStencilToAlias(ImageId depth_id, ImageId target_id);
	std::vector<ImageId>                  m_bound_images;
	std::vector<vk::DescriptorImageInfo>  m_descriptor_images;
	std::vector<vk::WriteDescriptorSet>   m_descriptor_writes;

	friend class CommandProcessor;
	friend struct RenderExecutorTestAccess;
};

[[nodiscard]] bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                            uint32_t group_y, uint32_t group_z, uint32_t mode,
                                            ShaderBufferResource& descriptor,
                                            uint32_t& packed_clear, uint64_t& size);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_ */
