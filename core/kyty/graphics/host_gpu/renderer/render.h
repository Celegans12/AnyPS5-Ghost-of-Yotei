#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/drawTemplates.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/tableResolutionCache.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/upscaler/upscaler.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <deque>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace HW {
class Context;
class UserConfig;
class Shader;
}

struct GraphicContext;
struct DrawCallInfo;
struct DrawEmitInfo;
struct DrawIndexBufferSource;
struct DynamicState;
class RenderContext;
class CommandScheduler;
struct RenderExecutorTestAccess;
struct IndirectThreadPass;
class RayTracingCheck;

struct TextureDescDerivation {
	TextureCache::ImageDesc desc;
	bool                    shader_conversion = false;
	vk::Format              pixel_format      = vk::Format::eUndefined;
	vk::Format              view_format       = vk::Format::eUndefined;
	uint64_t                size              = 0;
	std::string unsupported;
	bool        unknown_tiling = false;
	struct Found {
		ImageId                 id;
		TextureCache::ImageDesc desc;
		vk::ImageView           view            = nullptr;
		uint64_t                image_set_epoch = 0;
		ImageLookupState        state;
		bool                    valid = false;
	};
	Found found;
};

enum class DrawOffsetSource : uint8_t {
	DrawState,
	IndirectArgs,
};

struct DrawIndirectArgs {
	uint64_t              address = 0;
	uint32_t              count   = 0;
	uint32_t              stride  = 0;
	ShaderIndirectOffsets offsets;
	uint64_t              count_address = 0;
};

struct DrawIndexArgs {
	uint32_t         index_count         = 0;
	const void*      index_addr          = nullptr;
	uint32_t         instance_count      = 0;
	uint32_t         index_type_and_size = 0;
	int32_t          base_vertex         = 0;
	uint32_t         first_instance      = 0;
	DrawOffsetSource offset_source       = DrawOffsetSource::DrawState;
	DrawIndirectArgs indirect;
};

struct DrawAutoArgs {
	uint32_t         vertex_count   = 0;
	uint32_t         instance_count = 0;
	uint32_t         first_vertex   = 0;
	uint32_t         first_instance = 0;
	DrawOffsetSource offset_source  = DrawOffsetSource::DrawState;
	DrawIndirectArgs indirect;
};

struct StateKeyCache {
	uint64_t context_id = 0, user_config_id = 0, hash = 0;
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

class CommandBuffer {
public:
	~CommandBuffer() = default;

	KYTY_CLASS_NO_COPY(CommandBuffer);

	[[nodiscard]] bool IsInvalid() const noexcept {
		return m_stream != nullptr ? !m_open : m_buffer == nullptr;
	}

	void BeginRendering(const RenderState& state) const;
	void EndRendering() const;
	[[nodiscard]] bool IsRendering() const noexcept { return m_rendering; }
	void DeferShaderWriteBarrier(vk::PipelineStageFlags source_stages) const noexcept {
		m_pending_shader_writes |= source_stages;
	}
	[[nodiscard]] bool HasPendingShaderWrites() const noexcept {
		return static_cast<bool>(m_pending_shader_writes);
	}

	[[nodiscard]] CommandRecorder Handle() const;
	[[nodiscard]] GraphicContext&   GetGraphics() const noexcept { return m_graphics; }
	[[nodiscard]] RenderContext&    GetContext() const noexcept { return m_context; }
	[[nodiscard]] const HW::Context&    GetRegisters() const noexcept { return *m_registers; }
	[[nodiscard]] const HW::UserConfig& GetUserConfig() const noexcept { return *m_user_config; }
	[[nodiscard]] const HW::Shader&     GetShaders() const noexcept { return *m_shaders; }

	void InvalidateDynamicState() const;

private:
	CommandBuffer(CommandScheduler& scheduler, CommandStream* stream);
	void Bind(const HW::Context& registers, const HW::UserConfig& user_config,
	          const HW::Shader& shaders) noexcept {
		m_registers   = &registers;
		m_user_config = &user_config;
		m_shaders     = &shaders;
	}

	void Begin();
	void End() const;

	CommandScheduler&   m_scheduler;
	RenderContext&      m_context;
	GraphicContext&     m_graphics;
	vk::CommandBuffer   m_buffer = nullptr;
	CommandStream*      m_stream = nullptr;
	bool                m_open   = false;
	mutable RenderState m_render_state;
	mutable bool        m_rendering   = false;
	mutable vk::PipelineStageFlags m_pending_shader_writes;
	const HW::Context*    m_registers   = nullptr;
	const HW::UserConfig* m_user_config = nullptr;
	const HW::Shader*     m_shaders     = nullptr;

	friend class CommandScheduler;
};

namespace Pssr {
struct Request;
}

class RenderExecutor {
public:
	explicit RenderExecutor(RenderContext& context): m_context(context) { ConfigureTableResolutions(); }
	~RenderExecutor();
	KYTY_CLASS_NO_COPY(RenderExecutor);

	void DispatchDirect(CommandBuffer& buffer, uint32_t thread_group_x, uint32_t thread_group_y,
	                    uint32_t thread_group_z, uint32_t mode, bool async_compute);
	void DispatchIndirect(CommandBuffer& buffer, uint64_t args_addr, uint32_t mode,
	                      bool async_compute);
	void ProcessPssr(CommandBuffer& buffer, const Pssr::Request& request);
	void DispatchIndirectThreads(CommandBuffer& buffer, uint64_t args_addr, uint32_t mode,
	                             bool async_compute);
	RayTracingCheck& RtCheck();

	struct ResolvedDraw;
	[[nodiscard]] ResolvedDraw* ResolveDrawAhead(const HW::Context&    context,
	                                             const HW::UserConfig& user_config,
	                                             const HW::Shader&     shaders,
	                                             const ShaderIndirectOffsets& indirect_offsets = {},
	                                             bool indirect_draw = false, uint64_t context_id = 0,
	                                             uint64_t user_config_id = 0);
	[[nodiscard]] ResolvedDraw* ResolveDispatchAhead(const HW::Context& context,
	                                                 const HW::Shader& shaders,
	                                                 const uint32_t (&groups)[3], uint32_t mode,
	                                                 bool async_compute);
	[[nodiscard]] ResolvedDraw* ResolveIndirectDispatchAhead(const HW::Context& context,
	                                                         const HW::Shader&  shaders,
	                                                         uint32_t           mode, uint64_t args_addr,
	                                                         bool               async_compute);
	bool TakeResolvedDispatch(ShaderComputeInputInfo& input_info, const ShaderModule*& module);
#if defined(ANYPS5_KYTY)
	void CheckDeferredDraw(CommandBuffer& buffer, const DrawRenderState& state);
	void CheckDeferredDispatch(const HW::ComputeShaderInfo& regs, const ShaderComputeInputInfo& pristine,
	                           const ShaderComputeInputInfo& input_info, const ShaderModule* module);
#endif
	void UseResolvedDraw(ResolvedDraw* resolved) noexcept {
		m_resolved_draw = resolved != nullptr ? WaitResolvedDraw(resolved) : nullptr;
	}
	static ResolvedDraw* WaitResolvedDraw(ResolvedDraw* resolved) noexcept;
	static void ReleaseResolvedDraw(ResolvedDraw* resolved) noexcept;
	enum class AheadNote : uint8_t { NotAttempted, Prepared, NotPrepared, TableSkip };
	enum class ExecuteCause : uint8_t { TableSkip, Revalidation, NotPrepared, Again, Other, Count };
	void BeginAheadOperation(ResolvedDraw* resolved, AheadNote note) noexcept;
	void EndAheadOperation() noexcept;
	void NoteTableServed() noexcept { m_ahead_table = true; }
	static void NoteAhead(bool dispatch, AheadNote note, uint64_t ns) noexcept;
	void LogResolveSummary();
	void UseContextId(uint64_t context_id, uint64_t user_config_id = 0) noexcept {
		m_context_id     = context_id;
		m_user_config_id = user_config_id;
	}
	void BeginGuestOperation(uint32_t queue, bool dispatch, bool draw, bool guest_sync);
	void FlushDeferredDispatchBarrier();
	void InvalidateRenderTargetMemo() noexcept;
	[[nodiscard]] uint64_t DrawWindow() const noexcept { return m_draw_window; }

	void PrepareBindings(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	void FindBuffers(std::span<PreparedBindings* const> stages, bool find_buffers = true);
	void                           RebindBuffers(PreparedBindings& bindings);
	void                           RebindImages(PreparedBindings& bindings);
	void CommitBindings(CommandBuffer& buffer, vk::PipelineBindPoint pipeline_bind_point,
	                    const PipelineCache::Pipeline&     pipeline,
	                    std::span<PreparedBindings* const> bindings);
	[[nodiscard]] TextureBinding ResolveTexture(const ShaderRecompiler::IR::ImageResource&   resource,
	                                            const ShaderRecompiler::IR::DescriptorValue& value,
	                                            bool        table_candidate = false,
	                                            GuestRange* examined        = nullptr);

	void NativeXprEndDraw() noexcept {
		m_table_store = false;
	}
	[[nodiscard]] bool TableStoreRequested() const noexcept { return m_table_store; }
	[[nodiscard]] bool TableLikely(const HW::Shader& shaders) const noexcept;
	[[nodiscard]] uint32_t TableHint(const HW::Shader& shaders) const noexcept;
	[[nodiscard]] bool TableDrewDraws() const noexcept { return m_table_drew_draws.load(std::memory_order_relaxed); }
	void SetTableTicket(const void* ticket) noexcept { m_table_ticket = ticket; }
	[[nodiscard]] bool TableXprTry(CommandBuffer& buffer, uint64_t args_addr, uint32_t count, uint32_t stride,
	                               uint64_t index_base, uint64_t index_bytes, vk::IndexType index_type,
	                               const DrawIndexArgs* direct, bool clean,
	                               const ShaderIndirectOffsets& offsets = {}, const uint32_t* run_offsets = nullptr);

private:
	struct NativeXprCache;
	struct NativeXprDrawState;
	NativeXprCache&   NativeXprState();
	vk::DescriptorSet NativeXprAllocateSet(vk::DescriptorSetLayout layout, vk::DescriptorPool& pool_out);
	void              NativeXprRetireSet(vk::DescriptorPool pool, vk::DescriptorSet set);
	void              NativeXprCollect();
	bool              NativeXprSweepStates(NativeXprCache& cache, uint64_t frame);
	void NativeXprCaptureState(const DrawRenderState& state, const RenderState& rendering, uint64_t state_key,
	                           const DynamicState& dynamic_state);
	bool NativeXprEmitTargets(const NativeXprDrawState& draw_state, CommandRecorder vk_buffer);
	bool NativeXprTargetsCurrent(const NativeXprDrawState& draw_state);
	void NativeXprEmitDynamic(NativeXprCache& cache, CommandRecorder vk_buffer, const NativeXprDrawState& draw_state,
	                          bool same_command, bool query_bias);

	bool DrawIndex(CommandBuffer& buffer, const DrawIndexArgs& args);
	bool DrawAuto(CommandBuffer& buffer, const DrawAutoArgs& args);

	struct GraphicsBindings {
		std::array<PreparedBindings, 3> vertex;
		PreparedBindings                pixel;
	};

	struct TableElementResolution {
		uint64_t       prepared_stamp = 0;
		uint32_t       prepared_index = 0;
		uint32_t       version_slot  = UINT32_MAX;
		bool           permanent     = false;
		bool           volatile_dcc  = false;
		bool           backing_checkable = false;
		bool           backing_readable  = false;
		uint64_t       image_epoch   = 0;
		uint64_t       backing_epoch = 0;
		uint64_t       checked_epoch = 0;
		uint64_t       clear_memo    = 0;
		GuestRange     range;
		GuestRange     metadata_range;
		TableViewMemo  view;
		TextureBinding texture;
	};
	struct TableDescriptorHash {
		size_t operator()(const std::array<uint32_t, 8>& dwords) const noexcept;
	};
	using TableElements =
	    std::unordered_map<std::array<uint32_t, 8>, TableElementResolution, TableDescriptorHash>;
	struct TableSlot {
		std::array<uint32_t, 8> dwords {};
		TableElementResolution* element = nullptr;
		uint32_t                version_slot = UINT32_MAX;
	};
	struct TableResolution {
		ShaderRecompiler::IR::ImageResource root;
		uint64_t                            last_use   = 0;
		uint64_t                            generation = 0;
		TableElements                       elements;
		std::vector<TableSlot>              slots;
	};
	struct TableResolutionKeyHash {
		size_t operator()(const ShaderRecompiler::IR::ImageResource& root) const noexcept;
	};
	using TableResolutions =
	    TableResolutionCache<TableResolution, ShaderRecompiler::IR::ImageResource, TableResolutionKeyHash>;
	struct TableStats {
		uint64_t bindings     = 0;
		uint64_t prepared     = 0;
		uint64_t reused       = 0;
		uint64_t unfinished   = 0;
		uint64_t renewed      = 0;
		uint64_t layout       = 0;
		uint64_t left_renewed = 0;
		uint64_t left_size    = 0;
		uint64_t resets       = 0;
		uint64_t window_ns    = 0;
		bool     idle_logged  = false;
	};
	void ConfigureTableResolutions();
	void RetireTableResolution(TableResolution& table);
	void TrimTableResolutions();
	void LogTableSummary();

	[[nodiscard]] TableElementResolution&
	ResolveTableElement(TableResolution& table, const ShaderRecompiler::IR::DescriptorValue& value,
	                    uint64_t stamp, TableElementResolution* known = nullptr);
	[[nodiscard]] TableResolution& FindTableResolution(const ShaderRecompiler::IR::ImageResource& root);
	[[nodiscard]] bool TableElementBackingReadable(const TableElementResolution& element);
	void TableElementChanged(const TableElementResolution& element) {
		if (element.version_slot < m_table_element_versions.size()) {
			m_table_element_versions[element.version_slot]++;
		}
	}

	struct TableBindingState {
		struct Range {
			ShaderRecompiler::IR::ImageResource root;
			TableResolution*                    resolution = nullptr;
			uint32_t                            table      = 0;
			uint32_t                            capacity   = 0;
			uint64_t                            slots_id   = 0;
			uint64_t                            generation = 0;
		};
		struct Element {
			TableElementResolution*               resolution = nullptr;
			TableResolution*                      table      = nullptr;
			ShaderRecompiler::IR::DescriptorValue source;
			uint32_t                              root = 0;
			ImageId                               pinned;
			bool always            = false;
			bool unchecked_backing = false;
			GuestRange span_range;
			GuestRange span_metadata;
		};
		struct PageSpan {
			uint64_t first   = 0;
			uint64_t last    = 0;
			uint32_t element = 0;
		};
		std::vector<Range>                   ranges;
		std::vector<ShaderRecompiler::IR::DescriptorValue> values;
		std::vector<Element>                 elements;
		std::vector<uint32_t>                slots;
		std::vector<vk::DescriptorImageInfo> infos;
		std::vector<vk::DescriptorImageInfo> slot_infos;
		std::vector<uint32_t>                element_slot_begin;
		std::vector<uint32_t>                element_slots;
		std::vector<uint32_t>                element_of_version;
		std::vector<PageSpan>                spans;
		uint64_t                             max_span = 0;
		std::vector<uint32_t>                always;
		std::vector<uint32_t>                unchecked_backing;
		bool                                 compute = false;
		std::vector<uint64_t>                marks;
		uint64_t                             mark_stamp = 0;
		struct Checked {
			static constexpr uint32_t None      = UINT32_MAX;
			static constexpr uint32_t NullImage = UINT32_MAX - 1;
			uint32_t                  image           = None;
			uint32_t                  image_version   = 0;
			uint32_t                  element         = 0;
			uint32_t                  element_version = 0;
		};
		std::vector<Checked>                 checked;
		uint64_t                             generation = 0;
		uint64_t                             set_epoch  = 0;
		uint64_t                             backing_epoch = 0;
		bool                                 valid = false;
		vk::DescriptorSet                    slot_set;
		bool                                 slot_set_current = false;
		struct RingSet {
			vk::DescriptorSet       set;
			vk::DescriptorSetLayout layout;
			uint64_t                tick    = 0;
			uint64_t                version = 0;
		};
		std::vector<RingSet>                 ring;
		uint64_t                             slot_version   = 1;
		uint64_t                             slot_log_floor = 1;
		std::vector<std::pair<uint64_t, uint32_t>> slot_log;
	};
	[[nodiscard]] TableBindingState::RingSet* AcquireTableRingSet(TableBindingState&      state,
	                                                              vk::DescriptorSetLayout layout);
	[[nodiscard]] bool ProveTableSlotsLive(const TableBindingState& state);
	[[nodiscard]] bool ReuseTableBindings(const ShaderRecompiler::IR::CompiledShaderInfo& program,
	                                      const ShaderRecompiler::IR::ResourceSnapshot&   snapshot,
	                                      TableBindingState& state, uint64_t stamp,
	                                      PreparedBindings& prepared);
	void MarkTableChanges(TableBindingState& state);
	void MarkTableElement(TableBindingState& state, uint32_t element);
	void MarkTablePages(TableBindingState& state, uint64_t address, uint64_t size);
	void RebuildTableSpans(TableBindingState& state);
	static void AddTableSpans(TableBindingState& state, uint32_t element);
	void MergeTableSpans(TableBindingState& state, size_t sorted);
	static void IndexTableSlots(TableBindingState& state);
	void PatchTableSlots(const ShaderRecompiler::IR::CompiledShaderInfo& program,
	                     const ShaderRecompiler::IR::ResourceSnapshot&   snapshot,
	                     TableBindingState& state, uint64_t stamp);
	void DropUnusedTableElements(TableBindingState& state);
	void FinishTableBindings(TableBindingState& state, const PreparedBindings& prepared);
	void UnpinTableBindings(TableBindingState& state);
	void RebindReusedTables(PreparedBindings& prepared);

	struct TextureDescKey {
		std::array<uint32_t, 8> dwords {};
		std::array<uint32_t, 5> view {};
		bool                    operator==(const TextureDescKey&) const = default;
	};
	struct TextureDescKeyHash {
		size_t operator()(const TextureDescKey& key) const noexcept;
	};
	[[nodiscard]] static TextureDescKey
	MakeTextureDescKey(const ShaderRecompiler::IR::ImageResource&   resource,
	                   const ShaderRecompiler::IR::DescriptorValue& value);
	[[nodiscard]] bool ReuseTextureLookup(TextureDescDerivation& derived, TextureBinding& binding);
	static constexpr size_t MaxTextureDescs = size_t {1} << 16u;
	std::unordered_map<TextureDescKey, TextureDescDerivation, TextureDescKeyHash> m_texture_descs;
	bool m_clear_texture_descs = false;
	uint64_t m_texture_descs_generation = 1;

	uint64_t                           m_draw_resource_key = 0;
	std::unique_ptr<DrawTemplateTable> m_draw_templates;
	struct DrawBufferView {
		uint64_t                 address = 0;
		uint64_t                 size    = 0;
		uint32_t                 flags   = 0;
		uint32_t                 offset  = 0;
		vk::DescriptorBufferInfo view;
	};
	std::array<DrawBufferView, 16> m_draw_buffer_views {};
	uint32_t                       m_draw_buffer_view_count = 0;
	bool                           m_draw_buffer_views_on   = false;
	struct DrawStreamRead {
		uint64_t address = 0;
		uint64_t size    = 0;
		uint64_t offset  = UINT64_MAX;
	};
	std::array<DrawStreamRead, 32> m_draw_stream_reads {};
	uint32_t                       m_draw_stream_read_count = 0;
	void                           StreamDrawReads(std::span<PreparedBindings* const> stages);
	bool ResolveDrawPrograms(ResolvedDraw& draw, const HW::Context& context, const HW::UserConfig& user_config,
	                         const HW::Shader& shaders, const ShaderIndirectOffsets& indirect_offsets,
	                         bool indirect_draw, uint64_t context_id, uint64_t user_config_id);
	ResolvedDraw& NextResolvedDraw();
	[[nodiscard]] static bool DrawTemplateEligible(std::span<const ShaderStageRuntime* const> stages);
	[[nodiscard]] DrawTemplate* FindDrawTemplate(uint64_t key,
	                                             std::span<const ShaderStageRuntime* const> stages);
	[[nodiscard]] bool DrawTemplateRefused(uint64_t key) const;
	void PrepareBindingsFromTemplate(const ShaderStageRuntime& runtime,
	                                 const DrawTemplateStage& stage, PreparedBindings& prepared);
	[[nodiscard]] bool CheckTemplateImages(PreparedBindings& prepared);
	void StoreDrawTemplate(uint64_t key, std::span<PreparedBindings* const> stages);
	[[nodiscard]] static bool DrawTemplateStagesMatch(const DrawTemplate&                        found,
	                                                  std::span<const ShaderStageRuntime* const> stages);
	[[nodiscard]] bool TakeTemplateTargets(const DrawTemplate& found);
	void               StoreTemplateTargets(DrawTemplate& stored, const RenderState& rendering,
	                                        vk::ImageAspectFlags feedback_aspects);
	DrawTemplate*                                              m_commit_capture    = nullptr;
	bool                                                       m_commit_capture_ok = false;
	std::vector<std::pair<uint8_t, uint8_t>>                   m_commit_capture_kinds;
	std::unordered_map<uint64_t, vk::DescriptorUpdateTemplate> m_commit_templates;
	std::vector<uint8_t>                                       m_commit_data;
	[[nodiscard]] static bool TemplateCommitsSupported();
	void StoreTemplateCommit(DrawTemplate& stored, const PipelineCache::Pipeline& pipeline);
	void CommitTemplateBindings(CommandBuffer& buffer, const PipelineCache::Pipeline& pipeline,
	                            std::span<PreparedBindings* const> stages, const DrawTemplate& found);
	[[nodiscard]] uint64_t TemplateRangeEpoch(uint64_t address, uint64_t size);
	static constexpr uint32_t TemplatePageBuckets = 1u << 16u;
	std::vector<uint64_t>     m_template_page_epochs;
	uint64_t                  m_template_pages_epoch = 0;
	uint64_t                  m_template_pages_floor = 0;

	[[nodiscard]] vk::Sampler NativeSampler(const ShaderRecompiler::IR::CompiledShaderInfo& program,
	                                        uint32_t                                        index,
	                                        const ShaderRecompiler::IR::DescriptorValue&    value);
	struct SamplerMemoEntry {
		std::array<uint32_t, 4> fields {};
		bool                    integer_border = false;
		vk::Sampler             sampler        = nullptr;
	};
	std::array<SamplerMemoEntry, 256> m_sampler_memo {};

	[[nodiscard]] TextureBinding
	ResolveTableTexture(const ShaderRecompiler::IR::ImageResource&   root,
	                    const ShaderRecompiler::IR::DescriptorValue& value,
	                    GuestRange*                                  examined = nullptr);
	void PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
	                             std::span<RenderColorInfo> colors);
	void ResolveRenderColorTarget(CommandBuffer& buffer, RenderColorInfo& target,
	                              uint32_t render_target_slot, bool ignore_target_mask = false,
	                              bool exact_format = false);
	void ResolveRenderDepthTarget(CommandBuffer& buffer, RenderDepthInfo& target);
	[[nodiscard]] bool DepthStencilCopy(CommandBuffer& buffer);
	[[nodiscard]] bool CopyDepthToColor(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                    const DrawEmitInfo& emit, const DrawRenderState& state);
	struct DrawRenderStorage;
	struct ColorTargetMemo;
	struct DepthTargetMemo;
	struct RenderTargetMemo;
	struct ResolvedDrawRing;
	void               CaptureRenderTargetKey(const HW::Context& registers, uint32_t mrt_mask);
	[[nodiscard]] bool RenderTargetMemoCurrent();
	[[nodiscard]] bool TryReuseRenderTargets(const HW::Context& registers, uint32_t mrt_mask);
	void StoreRenderTargetMemo(const RenderColorInfo* colors, uint32_t color_count,
	                           const RenderDepthInfo& depth, const vk::ImageView* views,
	                           vk::ImageView depth_view, ImageId stencil);
	[[nodiscard]] DrawRenderState& AcquireDrawRenderState(bool  tessellation,
	                                                      bool& shaders_resolved,
	                                                      const ShaderIndirectOffsets& indirect_offsets,
	                                                      bool indirect_draw);
	[[nodiscard]] bool PrepareDrawRenderState(CommandBuffer& buffer, bool shaders_resolved,
	                                          const DrawCallInfo& draw, DrawRenderState& state);
	void               ExecutePreparedDraw(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                       const DrawRenderState& state, const DrawEmitInfo& emit,
	                                       const DrawIndexBufferSource& index_source);
	[[nodiscard]] RenderState AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
	                                               uint32_t color_count, RenderDepthInfo& depth,
	                                               vk::ImageAspectFlags& feedback_aspects,
	                                               std::span<PreparedBindings* const> stages = {});
#if defined(ANYPS5_KYTY)
	void PreparePackedTargets(CommandBuffer& buffer, PreparedBindings& pixel,
	                          const ShaderPixelInputInfo& info);
#endif
	[[nodiscard]] bool        ResolveColorTargets(CommandBuffer& buffer);
	[[nodiscard]] bool        ConsumeMetadataColorOperation(CommandBuffer& buffer);
	void                      BindImage(ImageId id, bool storage);
	void                      BindRenderTarget(ImageId id);
	void                      ResetBindings();
	[[nodiscard]] bool        TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
	                                                     const CommandBuffer&          buffer);
	[[nodiscard]] bool TryConsumeComputeFill(const ShaderComputeInputInfo& input,
	                                        CommandBuffer& command, uint32_t group_x,
	                                        uint32_t group_y, uint32_t group_z, uint32_t mode);
	[[nodiscard]] bool ConsumeReplicate(const ShaderBufferResource& destination, const ShaderBufferResource& source,
	                                    uint64_t count, uint64_t pattern, uint32_t group_x);
	[[nodiscard]] bool TryLinearCopy(const HW::ComputeShaderInfo& cs, uint32_t group_x, uint32_t group_y,
	                                 uint32_t group_z, uint32_t mode);
	void CreatePssrPipeline();
	struct UpscalerFeature;
	[[nodiscard]] UpscalerFeature* UpscalerFeatureFor(const Pssr::Request& request,
	                                                  const Dlss::Inputs& inputs, bool& reset);
	[[nodiscard]] static bool EvaluateUpscaler(vk::CommandBuffer command, UpscalerFeature& feature,
	                                           const Pssr::Request& request, Dlss::Inputs& inputs,
	                                           bool reset);
	static void ReleaseUpscalerFeature(UpscalerFeature* feature);

	struct UpscalerContext {
		uint64_t         context;
		Dlss::Creation   creation;
		UpscalerFeature* feature;
	};

	RenderContext&                        m_context;
	GraphicsBindings                     m_graphics_bindings;
	DrawTargets                           m_targets;
	DrawTargets*                          m_draw_targets = &m_targets;
	PreparedBindings                     m_compute_bindings;
	std::shared_ptr<RayTracingCheck>     m_rt_check;
	std::vector<ImageId>                  m_bound_images;
	std::vector<vk::DescriptorImageInfo>  m_descriptor_images;
	std::vector<vk::WriteDescriptorSet>   m_descriptor_writes;
	std::vector<vk::CopyDescriptorSet>    m_descriptor_copies;
	std::vector<vk::WriteDescriptorSet>   m_slot_set_writes;
	std::vector<ImageTableBinding>        m_table_fills;
	std::vector<vk::WriteDescriptorSet>   m_table_writes;
	std::vector<uint8_t>                  m_table_element_live;
	std::vector<uint8_t>                  m_table_slot_live;
	vk::DescriptorSetLayout               m_pssr_descriptor_layout = nullptr;
	vk::PipelineLayout                    m_pssr_pipeline_layout   = nullptr;
	vk::Pipeline                          m_pssr_pipeline          = nullptr;
	std::vector<UpscalerContext>          m_upscaler_contexts;
	TableResolutions                      m_table_resolutions;
	std::deque<TableElements>             m_retired_table_elements;
	size_t                                m_table_element_count = 0;
	TableStats                            m_table_stats;
	uint64_t                              m_table_prepare_stamp   = 0;
	std::vector<vk::DescriptorImageInfo>  m_table_image_infos;
	std::unordered_map<const void*, TableBindingState> m_table_states;
	std::vector<std::pair<uint64_t, uint64_t>>         m_table_changes;
	std::vector<uint32_t>                              m_table_marked;
	std::vector<TableBindingState::PageSpan>           m_table_spans;
	std::vector<uint32_t>                              m_table_remap;
	struct ChangedTableSlot {
		uint32_t slot  = 0;
		uint32_t range = 0;
		uint32_t index = 0;
	};
	std::vector<ChangedTableSlot>                      m_table_changed_slots;
	std::vector<uint8_t>                               m_table_element_flags;
	std::vector<uint32_t>                              m_table_moved_slots;
	std::vector<uint32_t>                              m_table_checks;
	std::vector<uint8_t>                               m_table_flags;
	std::vector<uint32_t>                              m_table_element_versions;
	std::vector<uint32_t>                              m_free_table_element_versions;
	std::vector<uint64_t>                              m_bound_image_marks;
	uint64_t                                           m_bound_image_stamp = 0;
	uint32_t                                           m_shader_write_queue      = 0;
	bool                                               m_shader_write_deferred   = false;
	uint32_t                                           m_guest_queue             = 0;
	bool                                               m_deferred_barrier        = false;
	uint32_t                                           m_deferred_barrier_queue  = 0;
	bool                                               m_dispatch_chained        = false;
	uint64_t                                           m_linear_copy_shader      = 0;
	uint64_t                                           m_linear_copy_generation  = 0;
	uint32_t                                           m_linear_copy_uses        = 0;
	std::array<uint64_t, 4>                            m_replicated {};
	std::unique_ptr<DrawRenderStorage, void (*)(DrawRenderStorage*)> m_draw_state {nullptr,
	                                                                               nullptr};
	std::unique_ptr<ResolvedDrawRing, void (*)(ResolvedDrawRing*)> m_resolved_ring {nullptr,
	                                                                                nullptr};
	ResolvedDraw*                                                  m_resolved_draw = nullptr;
	[[nodiscard]] ExecuteCause MissingAheadCause() const noexcept;
	[[nodiscard]] bool         RevalidateAhead(ResolvedDraw& resolved, bool dispatch);
	const ShaderModule* ResolveComputeHere(const HW::ComputeShaderInfo& regs, ShaderComputeInputInfo& input_info);
	AheadNote    m_ahead_note     = AheadNote::NotAttempted;
	ExecuteCause m_execute_cause  = ExecuteCause::Other;
	bool         m_ahead_present  = false;
	bool         m_ahead_dispatch = false;
	bool         m_ahead_taken    = false;
	bool         m_ahead_table    = false;
	uint32_t     m_revalidations  = 0;
	uint64_t     m_resolve_frames = 0;
#if defined(ANYPS5_KYTY)
	ResolvedDraw* m_flat_defer_check = nullptr;
#endif
	uint64_t                                                       m_context_id    = 0;
	uint64_t                                                       m_user_config_id = 0;
	StateKeyCache                                                  m_state_key_cache;
	uint64_t m_checked_context_id = 0;
	std::unique_ptr<ColorTargetMemo, void (*)(ColorTargetMemo*)> m_color_target_memo {nullptr,
	                                                                                  nullptr};
	std::unique_ptr<DepthTargetMemo, void (*)(DepthTargetMemo*)> m_depth_target_memo {nullptr,
	                                                                                  nullptr};
	std::unique_ptr<RenderTargetMemo, void (*)(RenderTargetMemo*)> m_render_target_memo {nullptr,
	                                                                                     nullptr};
	bool     m_render_targets_reused = false;
	bool     m_targets_from_memo     = false;
#if defined(ANYPS5_KYTY)
	uint32_t m_packed_target_slots   = 0;
#endif
	uint64_t m_draw_window           = 0;
	std::unique_ptr<IndirectThreadPass, void (*)(IndirectThreadPass*)> m_indirect_thread_pass {
	    nullptr, nullptr};
	std::unique_ptr<NativeXprCache, void (*)(NativeXprCache*)> m_native_xpr {nullptr, nullptr};
	struct TableXpr;
	struct TablePair;
	struct TableVariant;
	struct TableImageSet;
	struct TableFlatStage;
	enum class TableResult { Drawn, Native, Store, Deferred };
	std::unique_ptr<TableXpr, void (*)(TableXpr*)> m_table_xpr {nullptr, nullptr};
	bool m_table_store = false;
	const void* m_table_ticket = nullptr;
	TableXpr&   OwnTable();
	[[nodiscard]] static constexpr uint32_t Role() noexcept { return 0; }
	static constexpr uint32_t               Roles = 1;
	[[nodiscard]] RenderExecutor&           Catalog() noexcept { return *this; }
	[[nodiscard]] TableXpr*                 CatalogTable() noexcept { return m_table_xpr.get(); }
	struct CatalogLock {
		CatalogLock() noexcept {}
		~CatalogLock() {}
	};
	[[nodiscard]] static CatalogLock CatalogWrite() noexcept { return {}; }
	[[nodiscard]] static CatalogLock CatalogRead() noexcept { return {}; }
	void TableMemoRetired();
	void DropStaleCatalog() {}
	void ForgetStaleCatalog() {}
	void TableForgetStates();
	void TableSweepPairs();
public:
	struct NativeXprDirectDraw {
		uint64_t index_address  = 0;
		uint32_t index_count    = 0;
		uint32_t instance_count = 1;
		int32_t  base_vertex    = 0;
		bool     indexed        = true;
		bool     indirect       = false;
	};
private:
	TableResult TableTry(CommandBuffer& buffer, std::span<const uint64_t> commands, uint64_t index_base,
	                     uint64_t index_bytes, vk::IndexType index_type, const NativeXprDirectDraw* direct);
	bool        TableContinue(CommandBuffer& buffer, std::span<const uint64_t> commands, uint64_t index_base,
	                          uint64_t index_bytes, vk::IndexType index_type, const NativeXprDirectDraw* direct);
	TableResult TableDraw(CommandBuffer& buffer, TablePair& pair, TableVariant& variant,
	                      const NativeXprDrawState& draw_state, std::span<const uint64_t> commands,
	                      uint64_t index_base, uint64_t index_bytes, vk::IndexType index_type,
	                      const NativeXprDirectDraw* direct);
	void        TableReset(TablePair& pair);
	void        TableRuns(TablePair& pair);
	void        TablePack(TablePair& pair, const std::array<TableFlatStage, 2>& flat_forms);
	bool        TableEvaluate(TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data);
	void        TableGather(const TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data);
	TableImageSet* TableSet(CommandBuffer& buffer, TablePair& pair, const TableVariant* variant,
	                        const NativeXprDrawState* draw_state);
	TableImageSet* TableResolveSet(CommandBuffer& buffer, TablePair& pair, const TableVariant* variant,
	                               const NativeXprDrawState* draw_state, std::span<const uint32_t> words);
	bool        TableValidateSet(TableImageSet& set);
	bool        TableTransit(const TableImageSet& set, CommandRecorder vk_buffer, const NativeXprDrawState* draw_state);
	void        TablePrepareBda(const TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data);
	bool        TableBlocks(TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data,
	                        std::array<vk::DeviceAddress, 2>& blocks, bool* writes);
	void TableStore(CommandBuffer& buffer, const DrawRenderState& state, vk::PrimitiveTopology topology,
	                bool primitive_restart_enable, const RenderState& rendering, const DynamicState& dynamic_state,
	                bool storable);
	std::array<std::atomic<uint64_t>, 16384> m_table_drawn {};
	std::atomic<bool>                        m_table_drew_draws {false};
	std::array<std::atomic<uint64_t>, 16384> m_table_mixed_until {};
	std::array<uint64_t, 4096> m_table_refused {};
	std::array<uint64_t, 4096> m_table_refused_generation {};
	bool                       m_table_try_refused = false;

	friend class CommandProcessor;
	friend struct RenderExecutorTestAccess;
};

[[nodiscard]] bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                            uint32_t group_y, uint32_t group_z, uint32_t mode,
                                            ShaderBufferResource& descriptor,
                                            uint32_t& packed_clear, uint64_t& size);

}

#endif
