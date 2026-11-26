#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/file.h"
#include "common/threadSafety.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineStoreTypes.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/rectListShader.h"
#include "graphics/shader/shader.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <functional>
#include <iosfwd>
#include <span>
#if defined(ANYPS5_KYTY)
#include <source_location>
#endif
#include <stop_token>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <vector>

namespace Libs::Graphics {

struct KYTY_CAPABILITY("programs lock") ProgramSpinLock {
	std::atomic_flag flag;
};

enum class AheadRefusal : uint8_t {
	None,
	Read,
	Stage,
	Compiling,
	Shape,
	Calls,
	ImageHeap,
	Materialize,
	Variant,
	GpuVariant,
	Count
};

enum class StaleRead : uint8_t {
	None,
	Failed,
	ZeroTest,
	CpuWrite,
	GpuWrite,
	Strict,
	AroundGpuWrite,
	Count
};

class GuestReadLog {
public:
	void Clear() {
		m_entries.clear();
		m_bytes.clear();
		m_flat_loads.clear();
		m_words.clear();
		m_refresh.clear();
		m_stage       = 0;
		m_plan        = nullptr;
		m_failed      = false;
		m_refusal     = AheadRefusal::None;
		m_stale       = StaleRead::None;
		m_gpu_written = false;
		m_around      = false;
#if defined(ANYPS5_KYTY)
		m_deferred_words     = 0;
		m_deferred_gpu_words = 0;
#endif
	}
	bool               Read(uint64_t address, void* data, uint64_t size, bool strict);
	void               RecordLoad(uint64_t address, uint32_t value, uint32_t flat_slot);
	void               RecordLoads(std::span<const uint64_t> loads);
	void SetStage(uint32_t stage, const ShaderRecompiler::IR::ResourcePlan* plan = nullptr) {
		m_stage = stage;
		m_plan  = plan;
	}
	void Fail(AheadRefusal why = AheadRefusal::Read) {
		if (!m_failed) m_refusal = why;
		m_failed = true;
	}
	[[nodiscard]] bool         Failed() const { return m_failed; }
	[[nodiscard]] AheadRefusal Refusal() const { return m_refusal; }
	[[nodiscard]] StaleRead    Stale() const { return m_stale; }
	[[nodiscard]] bool MayHaveReadGpuWrites() const { return m_gpu_written; }
	[[nodiscard]] bool ReadAroundGpuWrites() const { return m_around; }
	void               NoteAroundLoads(std::span<uint64_t> addresses);
	[[nodiscard]] bool StillValid() const;
	[[nodiscard]] bool HasDescriptorRefresh() const { return !m_refresh.empty(); }
	void RefreshDescriptorWords(std::span<ShaderRecompiler::IR::ResourceSnapshot>          resources,
	                            std::span<const ShaderRecompiler::IR::ResourcePlan* const> plans) const;
#if defined(ANYPS5_KYTY)
	void RefreshFlatLoads(std::span<ShaderRecompiler::IR::ResourceSnapshot> resources, bool gpu_copies = false,
	                      const ShaderRecompiler::IR::CompiledShaderInfo* program = nullptr) const;
	void                   NoteDeferredLoads(std::span<uint64_t> addresses);
	[[nodiscard]] uint32_t DeferredWords() const { return m_deferred_words; }
	[[nodiscard]] uint32_t DeferredGpuWords() const { return m_deferred_gpu_words; }
#else
	void RefreshFlatLoads(std::span<ShaderRecompiler::IR::ResourceSnapshot> resources,
	                      bool gpu_copies = false) const;
#endif

	[[nodiscard]] bool     DwordAligned() const;
	[[nodiscard]] size_t   WordCount() const { return m_bytes.size() / 4u; }
	[[nodiscard]] uint32_t WordAt(size_t index) const;
	void                   SetWord(size_t index, uint32_t value);
	[[nodiscard]] bool     SameReads(const GuestReadLog& other) const;
	[[nodiscard]] bool     CollectChanges(std::vector<std::pair<uint32_t, uint32_t>>& changed) const;
	[[nodiscard]] bool     FirstMismatch(uint32_t& index, uint64_t& address, uint32_t& size, uint32_t& old_value,
	                                     uint32_t& new_value) const;

private:
	static constexpr uint64_t StrictCheckBytes = 64;
	struct Entry {
		uint64_t address = 0;
		uint32_t size    = 0;
		uint32_t offset  = 0;
		bool     strict  = false;
		bool     zero_test = false;
		bool     around    = false;
		uint64_t watched = 0;
		uint32_t words_begin = 0;
		uint32_t words_end   = 0;
	};
	struct Word {
		uint32_t offset = 0;
		uint32_t stage  = 0;
		uint32_t source = 0;
		uint32_t word   = 0;
		uint32_t mask   = 0;
	};
	void ReadNow(const Entry& entry, uint8_t* bytes) const;
	[[nodiscard]] bool OnlyRefreshableBitsChanged(const Entry& entry) const;
	struct FlatLoad {
		uint64_t address = 0;
		uint32_t stage   = 0;
		uint32_t slot    = 0;
		uint32_t count   = 1;
		bool     unread  = false;
#if defined(ANYPS5_KYTY)
		bool deferred = false;
#endif
	};
	void                  Append(uint64_t address, const void* data, uint32_t size, bool strict,
	                             bool zero_test = false);
	void                  AddFlatLoad(uint64_t address, uint32_t flat_slot);
	[[nodiscard]] bool ZeroTestHolds(const Entry& entry) const;
	std::vector<Entry>    m_entries;
	std::vector<uint8_t>  m_bytes;
	std::vector<FlatLoad> m_flat_loads;
	std::vector<Word>     m_words;
	mutable std::vector<uint32_t> m_refresh;
	struct CheckedPage {
		uint64_t       page    = UINT64_MAX;
		bool           owned   = false;
		const uint8_t* backing = nullptr;
	};
	mutable std::array<CheckedPage, 8> m_checked {};
	uint32_t              m_stage       = 0;
	const ShaderRecompiler::IR::ResourcePlan* m_plan = nullptr;
	bool                  m_failed      = false;
	AheadRefusal          m_refusal     = AheadRefusal::None;
	mutable StaleRead     m_stale       = StaleRead::None;
	bool                  m_gpu_written = false;
	bool                  m_around      = false;
#if defined(ANYPS5_KYTY)
	uint32_t         m_deferred_words     = 0;
	mutable uint32_t m_deferred_gpu_words = 0;
#endif
};

struct ProgramResolution {
	static constexpr uint32_t MaxStages = 4;

	std::array<ShaderRecompiler::IR::ResourceSnapshot, MaxStages>       resources;
	std::array<ShaderRecompiler::IR::ResourceSpecialization, MaxStages> specializations;
	std::array<const ShaderRecompiler::IR::ResourcePlan*, MaxStages>    plans {};
	GuestReadLog                                                        reads;
	uint32_t                                                            stages = 0;
	bool                                                                refreshed_words = false;

	void Reset() {
		reads.Clear();
		stages          = 0;
		refreshed_words = false;
	}
#if defined(ANYPS5_KYTY)
	[[nodiscard]] bool Revalidate(bool gpu_flat_loads = false,
	                              const ShaderRecompiler::IR::CompiledShaderInfo* program = nullptr) {
		if (!reads.StillValid()) {
			return false;
		}
		refreshed_words = reads.HasDescriptorRefresh();
		reads.RefreshDescriptorWords(std::span(resources).first(stages), std::span(plans).first(stages));
		reads.RefreshFlatLoads(std::span(resources).first(stages), gpu_flat_loads, program);
		return true;
	}
#else
	[[nodiscard]] bool Revalidate(bool gpu_flat_loads = false) {
		if (!reads.StillValid()) {
			return false;
		}
		refreshed_words = reads.HasDescriptorRefresh();
		reads.RefreshDescriptorWords(std::span(resources).first(stages), std::span(plans).first(stages));
		reads.RefreshFlatLoads(std::span(resources).first(stages), gpu_flat_loads);
		return true;
	}
#endif
};

struct GraphicContext;
#if defined(ANYPS5_KYTY)
class RenderContext;
#endif
struct ShaderParams;
struct ManifestModule;
struct PipelineManifest;
class CommandBuffer;

namespace HW {
class Context;
class UserConfig;
class Shader;
}

namespace ShaderRecompiler {
struct CompileOptions;
struct ShaderSource;
struct TranslateResult;
}

namespace ShaderRecompiler::IR {
class ImageHeapSource;
}

namespace ShaderRecompiler::Spirv {
struct Profile;
}

#pragma pack(push, 1)

struct PipelineStaticParameters {
	bool                       negative_one_to_one      = false;
	bool                       depth_clip_enable        = true;
	vk::PrimitiveTopology      topology                 = vk::PrimitiveTopology::ePointList;
	uint32_t                   patch_control_points     = 0;
	bool                       primitive_restart_enable = false;
	uint32_t                   samples                  = 1;
	bool                       sample_shading_enable    = false;
	bool                       alpha_to_coverage_enable = false;
	uint32_t                   color_mask[RENDER_COLOR_ATTACHMENTS_MAX]           = {};
	bool                       provoking_vtx_last                                 = false;
	vk::PolygonMode            polygon_mode                                       = vk::PolygonMode::eFill;
	uint8_t                    color_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	uint8_t                    alpha_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	bool                       separate_alpha_blend[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	bool                       blend_enable[RENDER_COLOR_ATTACHMENTS_MAX]         = {};
#if defined(ANYPS5_KYTY)
	uint32_t                   merged_hull                                        = 0;
#endif
};

struct PipelineRenderingState {
	std::array<vk::Format, RENDER_COLOR_ATTACHMENTS_MAX> color_formats {};
	vk::Format                                           depth_format   = vk::Format::eUndefined;
	vk::Format                                           stencil_format = vk::Format::eUndefined;
	uint32_t                                             color_count    = 0;
};

struct PipelineVertexInputState {
	struct Binding {
		uint32_t stride   = 0;
		uint8_t  instance = 0;
	};
	struct Attribute {
		uint32_t offset  = 0;
		uint8_t  binding = 0;
	};

	std::array<Binding, ShaderVertexInputInfo::RES_MAX>   bindings {};
	std::array<Attribute, ShaderVertexInputInfo::RES_MAX> attributes {};
	uint8_t                                               binding_count   = 0;
	uint8_t                                               attribute_count = 0;
};

struct alignas(4) GraphicsPipelineKey {
	std::array<uint64_t, 4>  stage_ids {};
	PipelineRenderingState   rendering;
	PipelineVertexInputState vertex_input;
	PipelineStaticParameters static_params;

	bool operator==(const GraphicsPipelineKey& other) const;
};

#pragma pack(pop)

static_assert(std::has_unique_object_representations_v<GraphicsPipelineKey>);

struct ManifestFloatMode {
	uint8_t float_mode = 0;
	bool    ieee       = false;
	bool    dx10_clamp = false;
	bool    recorded   = false;

	bool operator==(const ManifestFloatMode& other) const = default;
};

using ManifestFloatModes = std::array<ManifestFloatMode, 4>;

#if defined(ANYPS5_KYTY)
namespace HW {
struct ModeControl;
}

[[nodiscard]] vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back);

class PipelineLibraries;

struct OptimizedGraphicsPipeline {
	GraphicsPipelineKey key {};
	bool                native   = false;
	vk::Pipeline        pipeline = nullptr;
};

[[nodiscard]] PipelineStaticParameters LibraryStaticParameters(const HW::Context&    context,
                                                               const HW::UserConfig& user_config,
                                                               bool                  tessellation,
                                                               const ShaderTessellationInputInfo& tess,
                                                               bool ps_active, bool ps_sample_shading);
#endif

[[nodiscard]] std::shared_ptr<const ShaderModule> MakePlainModule(const ShaderModule&   module,
                                                                  std::span<const uint32_t> spirv,
                                                                  vk::Device            device);

[[nodiscard]] bool SmallColorTargetDraw(const HW::Context& context);
[[nodiscard]] bool RecordsNoMipStats(const ShaderStageRuntime& stage);

struct DrawTargets {
	RenderDepthInfo depth;
	RenderColorInfo colors[RENDER_COLOR_ATTACHMENTS_MAX];
	uint32_t        color_count = 0;
};

struct DrawRenderState {
	uint32_t                             mrt_mask = 0;
	vk::PrimitiveTopology                topology = vk::PrimitiveTopology::ePointList;
	bool                                 primitive_restart_enable = false;
	bool                                 ps_active                = false;
	bool                                 ps_plain                 = false;
	std::array<ShaderVertexInputInfo, 3> vertex_info;
	ShaderPixelInputInfo                 ps_input_info;
	ShaderIndirectOffsets                indirect_offsets;
	bool                                 indirect_draw = false;

	[[nodiscard]] std::span<const ShaderVertexInputInfo> VertexStages() const {
		return {vertex_info.data(), vertex_info[1].stage ? 3u : 1u};
	}
};

class ShaderStoreWriter {
public:
	ShaderStoreWriter();
	void Open(const std::filesystem::path& path, std::span<const uint8_t> header, uint64_t kept);
	explicit operator bool() const { return !m_file.IsInvalid(); }
	void Append(std::vector<uint8_t>&& record);
	void Post(Common::UniqueFunction<void>&& job);
	void Finish() { m_thread = {}; }

private:
	friend struct PipelineCacheTestAccess;

	Common::File                             m_file;
	std::mutex                               m_mutex;
	std::condition_variable_any              m_ready;
	std::queue<Common::UniqueFunction<void>> m_jobs;
	std::jthread m_thread;
};

class PrecompileMean {
public:
	void Add(double seconds) {
		m_samples++;
		m_mean = m_samples <= Window ? m_mean + (seconds - m_mean) / m_samples
		                             : m_mean + (seconds - m_mean) / Window;
	}
	[[nodiscard]] double Get() const { return m_samples >= MinSamples ? m_mean : 0.0; }

private:
	static constexpr uint32_t MinSamples = 8;
	static constexpr uint32_t Window     = 32;
	double                    m_mean     = 0.0;
	uint32_t                  m_samples  = 0;
};

class PipelineCache {
public:
	PipelineCache(GraphicContext& graphics, ShaderRecompiler::IR::ImageHeapSource* image_heap,
	              const std::filesystem::path& shader_store = {});
	~PipelineCache();
	KYTY_CLASS_NO_COPY(PipelineCache);
	void Save();

	struct Pipeline {
		vk::PipelineLayout      pipeline_layout       = nullptr;
		vk::Pipeline            pipeline              = nullptr;
		vk::DescriptorSetLayout descriptor_set_layout = nullptr;
		vk::ShaderStageFlags    push_stages;
		bool                    uses_push_descriptors = false;
		bool                    uses_image_heap       = false;
		bool                    use_noted             = false;
		bool                    native_bindings       = false;
		ManifestFloatModes      float_modes {};
		uint64_t                float_modes_generation = 0;
	};

	void GetGraphicsPrograms(const CommandBuffer& command, DrawRenderState& state);
	bool GetGraphicsProgramsAhead(const HW::Context& context, const HW::UserConfig& user_config,
	                              const HW::Shader& shaders, DrawRenderState& state,
	                              ProgramResolution& ahead);
	const ShaderModule* GetComputeProgram(const HW::ComputeShaderInfo& regs,
	                                      ShaderComputeInputInfo&      input_info,
	                                      bool                         guest_traversal = false);
	const ShaderModule* GetGuestComputeProgramQueued(const HW::ComputeShaderInfo& regs,
	                                                 ShaderComputeInputInfo& input_info, bool& queued);
	const ShaderModule* GetLanePairsComputeProgramQueued(const HW::ComputeShaderInfo& regs,
	                                                     ShaderComputeInputInfo& input_info, bool& queued);
	static void NoteAheadGuestWrite(uint64_t sequence, uint64_t address,
	                                std::span<const uint32_t> data, bool one_address);
	static void RetireAheadGuestWrites(uint64_t executed);
	[[nodiscard]] static uint64_t AheadWritesSequence();
	static void SyncAheadWrites(uint64_t sequence);
	static void BeginAheadThread(uint32_t slot);
#if defined(ANYPS5_KYTY)
	[[nodiscard]] static bool NarrowProgramLock();
	static void BeginExecuteThread();
#endif
	const ShaderModule* GetComputeProgramAhead(const HW::ComputeShaderInfo& regs,
	                                           ShaderComputeInputInfo&      input_info,
	                                           ProgramResolution&           ahead);

	Pipeline* GetGraphicsPipeline(const CommandBuffer& command, const DrawRenderState& state,
	                              const DrawTargets& targets, bool allow_async = false,
	                              bool native_bindings = false);
#if defined(ANYPS5_KYTY)
	[[nodiscard]] bool PreparesGraphicsLibraries() const;
	void SetRenderContext(RenderContext* context) { m_render = context; }
	enum class AsyncDraw : uint8_t {
		Off,
		WritesMemory,
		LinearTarget,
		Occlusion,
		SmallTarget,
		Eligible,
	};
	void PrepareGraphicsLibraries(const HW::Context& context, const HW::UserConfig& user_config,
	                              const DrawRenderState& state);
#endif
	Pipeline& GetComputePipeline(const ShaderComputeInputInfo& input_info);
	Pipeline* TryGetComputePipeline(const ShaderComputeInputInfo& input_info);
	[[nodiscard]] const std::shared_ptr<const std::atomic<bool>>& PipelineWaiting() const {
		return m_pipeline_waiting;
	}

	enum class TableStatus : uint8_t {
		Ready,
		Pending,
		Refused,
	};
	struct TableProgram {
		const ShaderModule* module = nullptr;
		TableStatus         status = TableStatus::Refused;
		const std::string*  refusal = nullptr;
		std::shared_ptr<const std::atomic<bool>> waiting;
	};
	std::pair<const ShaderRecompiler::IR::ResourcePlan*, const ShaderRecompiler::IR::ResourceSpecialization*>
	ProgramPlanOf(const ShaderModule& module) const;
	bool SpecializeDirectResources(const ShaderRecompiler::IR::ResourcePlan&     plan,
	                               ShaderRecompiler::IR::ResourceSnapshot&       snapshot,
	                               ShaderRecompiler::IR::ResourceSpecialization& specialization);
	TableProgram GetVertexProgramTable(const ShaderParams& params, const ShaderVertexInputInfo& input_info,
	                                   const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	                                   uint32_t& push_data_cursor);
	TableProgram GetPixelProgramTable(const ShaderParams& params, const ShaderPixelInputInfo& input_info,
	                                  const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	                                  uint32_t& push_data_cursor);

private:
	friend struct PipelineCacheTestAccess;

	struct Program {
		std::unique_ptr<ShaderRecompiler::ShaderSource> call_source;
		ShaderRecompiler::IR::ResourcePlan           resource_plan;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		std::deque<ShaderModule>                     modules;
		bool                                         skip_dispatch = false;
		std::array<size_t, 2>                        last_module {SIZE_MAX, SIZE_MAX};
#if defined(ANYPS5_KYTY)
		struct FailedVariant {
			ShaderRecompiler::IR::ResourceSpecialization specialization;
			uint32_t                                     push_data_cursor = 0;
			std::vector<uint32_t>                        function_code;
		};
		static constexpr size_t    SpeculativeFailureCap = 64;
		std::vector<FailedVariant> failed_variants;
		std::vector<FailedVariant> speculative_failures;
		size_t                     served_module = SIZE_MAX;
		uint64_t                   served_logged = 0;

		[[nodiscard]] bool Failed(const ShaderRecompiler::IR::ResourceSpecialization& candidate,
		                          uint32_t push_data_cursor) const;
		void               NoteFailed(uint32_t push_data_cursor);
		[[nodiscard]] bool SpeculativeFailed(const ShaderRecompiler::IR::ResourceSpecialization& candidate,
		                                     uint32_t push_data_cursor) const;
		void               NoteSpeculativeFailure(const ShaderRecompiler::IR::ResourceSpecialization& specialization,
		                                          uint32_t push_data_cursor);
		enum class Speculation : uint8_t { Compile, Kept, Refused, Uncontained };
		[[nodiscard]] Speculation JudgeSpeculativeCompile(ShaderType stage,
		                                                  const ShaderRecompiler::IR::ResourceSpecialization& candidate,
		                                                  uint32_t push_data_cursor, bool unread);
		ShaderModule*      ShareModule(const ShaderRecompiler::IR::ResourceSpecialization& requested,
		                               const ShaderRecompiler::IR::ResourceSnapshot&       snapshot,
		                               uint32_t push_data_cursor, bool calls);
		void               PromoteEmptyBuffers(const ShaderRecompiler::IR::ResourceSnapshot& snapshot,
		                                       ShaderRecompiler::IR::ResourceSpecialization& specialization,
		                                       uint32_t push_data_cursor) const;
#endif

		[[nodiscard]] std::span<const uint32_t> FunctionCode() const;
		bool RefreshCallSource(const ShaderRecompiler::IR::SrtRuntime& runtime);
		template <bool HasCalls>
		ShaderModule* FindModule(const ShaderRecompiler::IR::SrtRuntime& runtime,
		                         uint32_t push_data_cursor, bool& failed);
#if defined(ANYPS5_KYTY)
		template <bool HasCalls>
		ShaderModule* MatchModule(uint32_t push_data_cursor);
#endif
	};

	using ProgramKey = ShaderProgramKey;

	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const { return key.hash; }
	};

	struct GraphicsPipelineKeyHash {
		std::size_t operator()(const GraphicsPipelineKey& key) const;
	};

	using ModuleMap = std::unordered_map<uint64_t, const ShaderModule*>;

	GraphicContext&                        m_graphics;
	ShaderRecompiler::IR::ImageHeapSource* m_image_heap;
	std::unordered_map<ProgramKey, Program, ProgramKeyHash> m_program_cache KYTY_GUARDED_BY(m_program_lock);
	std::array<uint32_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> m_compiled {};
	ProgramKey                             m_lookup_key KYTY_GUARDED_BY(m_program_lock);
	bool                                   m_guest_traversal KYTY_GUARDED_BY(m_program_lock) = false;
	bool                                   m_lane_pairs KYTY_GUARDED_BY(m_program_lock)      = false;
	bool                                   m_queue_only KYTY_GUARDED_BY(m_program_lock)      = false;
	bool                                   m_queued KYTY_GUARDED_BY(m_program_lock)          = false;
	vk::PipelineCache             m_driver_cache = nullptr;
	std::filesystem::path         m_driver_cache_path;
	std::mutex                    m_driver_cache_mutex;
	std::chrono::steady_clock::time_point m_next_driver_cache_save;
	std::chrono::steady_clock::time_point m_driver_cache_posted;
	bool                          m_driver_cache_dirty = false;
	std::unordered_map<GraphicsPipelineKey, Pipeline, GraphicsPipelineKeyHash> m_graphics_pipelines;
	std::unordered_map<GraphicsPipelineKey, Pipeline, GraphicsPipelineKeyHash> m_native_graphics_pipelines;
	std::unordered_map<uint64_t, Pipeline>                                     m_compute_pipelines;
	std::unordered_map<uint64_t, ShaderTessellationInputInfo>                  m_tess_layouts KYTY_GUARDED_BY(m_program_lock);
	GraphicsPipelineKey m_last_graphics_key {};
	Pipeline*           m_last_graphics_pipeline = nullptr;
	ProgramSpinLock m_program_lock;

	struct CompileJob;
	static constexpr uint32_t CompileWorkers = 6;
	std::unordered_map<ProgramKey, uint32_t, ProgramKeyHash> m_in_flight KYTY_GUARDED_BY(m_program_lock);
#if defined(ANYPS5_KYTY)
	std::unordered_set<ProgramKey, ProgramKeyHash> m_speculative_failed_programs KYTY_GUARDED_BY(m_program_lock);
	std::atomic<uint64_t> m_speculative_failed {0}, m_speculative_refused {0}, m_speculative_skipped {0};
	std::array<uint64_t, 3> m_speculative_logged {};
	[[nodiscard]] bool TakesSpeculativeCompile(Program& program, const ProgramKey& key,
	                                           const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	                                           uint32_t push_data_cursor, bool unread) KYTY_REQUIRES(m_program_lock);
#endif
	std::mutex                                               m_job_mutex;
	std::condition_variable                                  m_job_available;
	std::deque<std::unique_ptr<CompileJob>>                  m_jobs;
	bool                                                     m_stopping = false;
	std::vector<std::jthread>                                m_workers;

	ShaderStoreWriter m_store;
	std::jthread      m_driver_cache_timer;

	bool ResolveGraphicsPrograms(const HW::Context& context, const HW::UserConfig& user_config,
	                             const HW::Shader& shaders, DrawRenderState& state,
	                             ProgramResolution* ahead);
	template <typename InputInfo>
	const ShaderModule* GetProgram(const ShaderParams& params, InputInfo& input_info,
	                               uint32_t& push_data_cursor) KYTY_REQUIRES(m_program_lock);
	template <typename InputInfo>
	const ShaderModule* GetAhead(const ShaderParams& params, InputInfo& input_info,
	                             uint32_t& push_data_cursor, ProgramResolution& ahead)
	    KYTY_NO_THREAD_SAFETY_ANALYSIS;
	ShaderModule& CompileModule(Program& program, const ProgramKey& key,
	                            ShaderRecompiler::CompileOptions  options,
	                            ShaderRecompiler::TranslateResult translated,
	                            uint32_t push_data_cursor, std::span<const uint32_t> code);
	ShaderModule BuildModule(const ProgramKey& key, const ShaderRecompiler::CompileOptions& options,
	                         ShaderRecompiler::TranslateResult                   translated,
	                         const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	                         std::span<const uint32_t> function_code, uint32_t push_data_cursor,
	                         std::vector<uint32_t>& spirv, std::vector<uint32_t>& plain_spirv,
	                         std::string* table_refusal = nullptr) const;
	void AddedModule(const ProgramKey& key, const Program& program, const ShaderModule& module,
	                 const std::vector<uint32_t>& spirv, const std::vector<uint32_t>& plain_spirv,
	                 const ShaderRecompiler::CompileOptions& options,
	                 std::span<const uint32_t> code, uint32_t push_data_cursor);
	void DestroyModule(const ShaderModule& module) const;
#if defined(ANYPS5_KYTY)
	void LockPrograms(std::source_location where = std::source_location::current()) KYTY_ACQUIRE(m_program_lock);
#else
	void LockPrograms() KYTY_ACQUIRE(m_program_lock);
#endif
	void UnlockPrograms() KYTY_RELEASE(m_program_lock);
	void AssumeProgramsConfined() noexcept KYTY_ASSERT_CAPABILITY(m_program_lock) {}
	const ShaderModule* GetComputeProgramQueued(const HW::ComputeShaderInfo& regs, ShaderComputeInputInfo& input_info,
	                                            bool& queued, const char* source, bool guest_traversal,
	                                            bool lane_pairs);
	template <typename InputInfo>
	void QueueCompile(const ProgramKey& key, const ShaderParams& params, const InputInfo& input_info,
	                  uint32_t                                            push_data_cursor,
	                  const ShaderRecompiler::IR::ResourceSpecialization* specialization,
	                  bool                                                guest_traversal = false,
	                  bool                                                speculative     = false,
	                  bool                                                lane_pairs      = false)
	    KYTY_REQUIRES(m_program_lock);
	bool WaitForCompile(const ProgramKey& key) KYTY_REQUIRES(m_program_lock);
	template <typename InputInfo>
	void RunCompileJob(CompileJob& job, InputInfo& input_info);
	void CompileWorker();
	void StopWorkers();
#if defined(ANYPS5_KYTY)
	uint32_t      ComputePipelineState(const ShaderModule& module);
	ShaderModule* ServeComputeVariant(Program& program, uint32_t push_data_cursor, const ShaderModule* exact,
	                                  uint32_t least_state);
	void          SpecializeComputeVariant(const ProgramKey& key, Program& program, const ShaderParams& params,
	                                       const ShaderComputeInputInfo&                       input_info,
	                                       const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	                                       uint32_t push_data_cursor) KYTY_REQUIRES(m_program_lock);
	void          WidenComputeVariant(const ProgramKey& key, Program& program, const ShaderParams& params,
	                                  const ShaderComputeInputInfo& input_info, uint32_t push_data_cursor,
	                                  std::optional<std::pair<ShaderRecompiler::IR::ResourceSpecialization,
	                                                          ShaderRecompiler::IR::ResourceSnapshot>>& natural)
	    KYTY_REQUIRES(m_program_lock);
#endif

	struct TableVariant {
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		uint32_t                                     push_data_cursor = 0;
		bool                                         done             = false;
		std::shared_ptr<std::atomic<bool>>           waiting = std::make_shared<std::atomic<bool>>(false);
		std::optional<ShaderModule>                  module;
		std::string                                  refusal;
	};
	std::unordered_map<ProgramKey, std::deque<TableVariant>, ProgramKeyHash> m_table_programs KYTY_GUARDED_BY(m_program_lock);
	template <typename InputInfo>
	TableProgram GetTableProgram(const ShaderParams& params, const InputInfo& input_info,
	                             const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	                             uint32_t push_data_cursor);
	template <typename InputInfo>
	void RunTableJob(CompileJob& job, InputInfo& input_info);

	struct AsyncPipelineJob {
		bool                               compute = false;
		bool                               native  = false;
		GraphicsPipelineKey                key {};
		std::array<const ShaderModule*, 4> stages {};
		const ShaderModule*                module = nullptr;
	};
	std::unordered_set<GraphicsPipelineKey, GraphicsPipelineKeyHash> m_async_pending_graphics;
	std::unordered_set<uint64_t>                                     m_async_pending_compute;
	std::unordered_set<GraphicsPipelineKey, GraphicsPipelineKeyHash> m_async_failed_graphics;
	std::unordered_map<GraphicsPipelineKey, Pipeline, GraphicsPipelineKeyHash> m_async_ready;
	struct AsyncNativePipeline {
		std::shared_ptr<std::atomic<bool>> done = std::make_shared<std::atomic<bool>>(false);
		Pipeline                           pipeline;
	};
	std::unordered_map<GraphicsPipelineKey, AsyncNativePipeline, GraphicsPipelineKeyHash> m_async_native;
	std::shared_ptr<const std::atomic<bool>>                                   m_pipeline_waiting;
	std::unordered_map<uint64_t, Pipeline>                                     m_async_ready_compute;
	std::unordered_set<uint64_t>                                               m_async_asked_compute;
	std::mutex                                                       m_async_mutex;
	std::condition_variable                                          m_async_available;
	std::deque<AsyncPipelineJob>                                     m_async_jobs;
	std::deque<AsyncPipelineJob>                                     m_async_sibling_jobs;
	void QueueSiblingPipeline(const GraphicsPipelineKey& key, const std::array<const ShaderModule*, 4>& stages,
	                          const ShaderModule* pixel);
	bool                                                             m_async_stopping = false;
	std::vector<std::jthread>                                        m_async_threads;
	bool                                                             m_async_handoff = false;
	std::unordered_set<GraphicsPipelineKey, GraphicsPipelineKeyHash> m_claimed_graphics;
	std::unordered_set<uint64_t>                                     m_claimed_compute;

	std::jthread m_precompile;
	void         StartPrecompile();
	void         Precompile(const std::stop_token& stop);

	struct PipelineStall {
		uint32_t                              threads = 0;
		std::chrono::steady_clock::time_point since;
		std::chrono::steady_clock::time_point ended;
	};
	class StallScope {
	public:
		explicit StallScope(PipelineCache& cache);
#if defined(ANYPS5_KYTY)
		StallScope(PipelineCache& cache, const char* what, uint64_t hash, const char* why = nullptr);
		StallScope(PipelineCache& cache, const char* what, const ShaderModule& module);
#endif
		~StallScope();
		KYTY_CLASS_NO_COPY(StallScope);

	private:
		PipelineCache& m_cache;
#if defined(ANYPS5_KYTY)
		std::chrono::steady_clock::time_point m_started;
		const char*                           m_what = nullptr;
		uint64_t                              m_hash = 0;
		const char*                           m_why  = nullptr;
#endif
	};
	mutable std::mutex m_stall_mutex;
	PipelineStall      m_stall;
	[[nodiscard]] PipelineStall GetPipelineStall() const;
#if defined(ANYPS5_KYTY)
	struct StallStats {
		uint64_t count      = 0;
		uint64_t total_ns   = 0;
		uint64_t max_ns     = 0;
		uint64_t over_100ms = 0;
		uint64_t over_500ms = 0;
		uint64_t pipelines  = 0;
	};
	StallStats            m_stall_stats;
	StallStats            m_stall_logged;
	std::atomic<uint64_t> m_pipelines_created {0};
	void                  LogStallStats(bool force);
#endif

#if defined(ANYPS5_KYTY)
	std::unique_ptr<PipelineLibraries>     m_libraries;
	std::vector<vk::Pipeline>              m_retired_pipelines;
	std::vector<OptimizedGraphicsPipeline> m_optimized_deferred;
	uint64_t                               m_optimized_seen = 0;
	size_t                                 m_optimized_scanned = 0;
	[[nodiscard]] AsyncDraw HeldDraw(const CommandBuffer& command, const DrawRenderState& state,
	                                 const DrawTargets& targets) const;
	[[nodiscard]] bool LiveTargets(const DrawTargets& targets) const;
	RenderContext* m_render = nullptr;
	bool CreateGraphicsPipelineNow(Pipeline& pipeline, const GraphicsPipelineKey& key,
	                               const std::array<const ShaderModule*, 4>& stages);
	bool CreateGraphicsPipelineFast(Pipeline& pipeline, const GraphicsPipelineKey& key,
	                                const std::array<const ShaderModule*, 4>& stages);
	void ApplyOptimizedPipelines();

	struct ChainJob;
	std::mutex                              m_chain_mutex;
	std::condition_variable                 m_chain_available;
	std::deque<std::unique_ptr<ChainJob>>   m_chain_jobs;
	bool                                    m_chain_stopping = false;
	std::vector<std::jthread>               m_chain_threads;
	std::array<std::atomic<uint64_t>, 5> m_async_skipped {};
	std::atomic<uint64_t> m_async_first_waits {0}, m_async_first_wait_ns {0}, m_async_first_ready {0},
	    m_async_probes {0}, m_async_probe_ns {0}, m_async_created_now {0};
	std::atomic<uint64_t> m_chain_queued {0}, m_chain_dropped {0}, m_chain_prepared {0}, m_chain_failed {0},
	    m_chain_compiled {0}, m_chain_ns {0};
	void NoteFirstWait(std::chrono::steady_clock::time_point started, bool ready);
	void QueueChain(std::unique_ptr<ChainJob>&& job);
	void ChainWorker(uint32_t index);
	void StopChains();
	void RunChain(ChainJob& job);
	template <typename InputInfo>
	const ShaderModule* ChainModule(const ShaderParams& params, InputInfo& input_info, uint32_t& push_data_cursor,
	                                ShaderRecompiler::IR::ResourceSnapshot& resources);
#endif
	void QueueAsyncPipeline(const AsyncPipelineJob& job);
	template <typename Matches>
	bool TakeQueuedAsyncPipeline(Matches&& matches);
	void AsyncPipelineWorker();
	void StopAsyncPipelines();
	void CreateComputePipelineAhead(const ShaderModule& module);
#if defined(ANYPS5_KYTY)
	void RequestComputePipelineAhead(const ShaderModule& module);
#endif
	void ClaimPipeline(const GraphicsPipelineKey& key);
	void ClaimComputePipeline(uint64_t module_id);
	bool InitializeDriverCache();
	void CheckpointDriverCache(bool slow_creation = false);
	void NoteCreatedPipeline(std::chrono::steady_clock::time_point started);
	void PostDueDriverCache();
	void WriteDriverCache();

	void WarmUp(const std::filesystem::path& path, bool driver_cache_restored);
	size_t RestoreStoreRecords(std::span<const uint8_t> bytes, std::span<const uint8_t> header);
	struct StoreRecord;
	bool ReadRecord(std::span<const uint8_t> frame, StoreRecord& record) const;
	bool AddRecord(StoreRecord& record) KYTY_REQUIRES(m_program_lock);
	void StoreModule(const ProgramKey& key, const Program& program, const ShaderModule& module,
	                 const std::vector<uint32_t>& spirv, const std::vector<uint32_t>& plain_spirv);
	void StorePipeline(const GraphicsPipelineKey& key);
	bool ResolveTessellationLayout(const HW::ShaderRegisters& sh, const ShaderParams& ls,
	                               const ShaderParams& hs, uint64_t address, bool lookup_only,
	                               ShaderTessellationInputInfo& tess) KYTY_REQUIRES(m_program_lock);
	void StoreLayout(uint64_t key, const ShaderTessellationInputInfo& layout);
	[[nodiscard]] ModuleMap MapModules() const KYTY_REQUIRES(m_program_lock);
	bool CreateGraphicsPipeline(Pipeline& pipeline, const GraphicsPipelineKey& key,
	                            const ModuleMap& modules);

	std::unordered_set<uint64_t> m_baked_stride_programs;
	[[nodiscard]] bool RuntimeStride(uint64_t program_hash) const {
		return !m_baked_stride_programs.contains(program_hash);
	}
	void DecideRuntimeStrides(std::span<PipelineManifest*> manifests);

	struct ManifestState;
	struct ManifestStateDeleter {
		void operator()(ManifestState* state) const;
	};
	std::unique_ptr<ManifestState, ManifestStateDeleter> m_manifest;
	bool                           m_manifest_warm = false;

	void OpenManifest();
	void CloseManifest();
	void RecordModule(const ProgramKey& key, const ShaderRecompiler::CompileOptions& options,
	                  std::span<const uint32_t> code, uint32_t push_data_cursor,
	                  const ShaderModule& module);
	bool RecordPipeline(const GraphicsPipelineKey& key, const ManifestFloatModes& float_modes);
	void NoteFloatModes(Pipeline& pipeline, const GraphicsPipelineKey& key, const HW::Shader& shaders);
	void RecordLayout(uint64_t key, const ShaderTessellationInputInfo& layout);
	static void OnShaderMapped(void* userdata, uint64_t hash, std::span<const uint32_t> code);
	bool ProvideCode(uint64_t hash, std::span<const uint32_t> code, bool from_guest);
	void CodeStoreThread(const std::stop_token& stop);
	bool ScanGameFiles(const std::stop_token&                                          stop,
	                   const std::function<void(uint64_t, std::span<const uint32_t>)>& append);
	void ManifestWorker(const std::stop_token& stop);
	std::pair<uint32_t, uint32_t> AddManifestRecords(PipelineManifest& loaded);
	void ReloadSharedManifest();
	void BuildManifestModule(uint32_t index, std::span<const uint32_t> code,
	                         std::span<const uint32_t> back_code, bool guest_traversal = false);
	void CreateManifestPipeline(uint32_t index);
	void CreateManifestComputePipeline(const ShaderModule& module, bool from_assets);
	void SetComputeLdsStorage(ShaderComputeInputInfo& input) const;
	void SetHostInput(ManifestModule& module) const;
	const ShaderModule* FindCompiledModule(
	    const ProgramKey& key, const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	    uint32_t push_data_cursor) const KYTY_REQUIRES(m_program_lock);
	void ManifestProgress(bool force);
	void SaveManifestRecording();
	struct ManifestBacklog {
		uint32_t modules         = 0;
		uint32_t compute_modules = 0;
		uint32_t pipelines       = 0;
		uint32_t running         = 0;
		bool     scanning        = false;
		uint64_t modules_done    = 0;
		uint64_t pipelines_done  = 0;
		uint32_t threads         = 0;
		double   module_seconds   = 0.0;
		double   pipeline_seconds = 0.0;
	};
	[[nodiscard]] ManifestBacklog GetManifestBacklog() const;

	struct UseOrder;
	struct UseOrderDeleter {
		void operator()(UseOrder* order) const;
	};
	std::unique_ptr<UseOrder, UseOrderDeleter> m_use_order;
	void OpenUseOrder(const std::filesystem::path& path);
	void NoteUse(Pipeline& pipeline, std::span<const ShaderModule* const> stages);
	static constexpr uint32_t NoRank     = UINT32_MAX;
	static constexpr uint32_t MappedRank = NoRank - 1;
	[[nodiscard]] uint32_t    UseRank(uint64_t program_hash) const;
};

void AppendNativeRayTracingKey(ShaderType stage, std::vector<uint32_t>& key, bool native = true);

std::string PipelineCacheTitleId();
std::filesystem::path ShaderStorePath();
uint64_t ShaderStoreBuildStamp();
std::string StoreMismatch(std::span<const uint8_t> bytes, std::span<const uint8_t> header, uint64_t current_stamp);
std::vector<uint8_t> StoreHeader(const GraphicContext& graphics, uint64_t stamp);
ShaderRecompiler::Spirv::Profile SpirvProfile(const GraphicContext& graphics);

void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const GraphicsPipelineKey&                 key,
                            const std::array<const ShaderModule*, 4>& stages,
                            vk::PipelineCache                          driver_cache);
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderModule& module, vk::PipelineCache driver_cache);
vk::Result CreateComputePipeline(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                                 const ShaderModule& module, vk::PipelineCache driver_cache,
                                 bool lookup, std::chrono::nanoseconds* driver_time = nullptr);
bool FindComputePipeline(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                         const ShaderModule& module, vk::PipelineCache cache);
struct DriverCreations {
	uint32_t running = 0;
	uint32_t slow    = 0;
	double   longest = 0.0;
};
[[nodiscard]] DriverCreations GetDriverCreations(double slow_seconds);
void DescribeDriverPipelineCreation(std::string& report);

[[nodiscard]] bool EncodeModuleRecord(ShaderType stage, uint64_t hash, uint32_t user_data_count,
                                      const std::vector<uint32_t>&              static_state,
                                      const ShaderRecompiler::IR::ResourcePlan& plan,
                                      const ShaderModule& module, const std::vector<uint32_t>& spirv,
                                      const std::vector<uint32_t>& plain_spirv,
                                      std::vector<uint8_t>&        record);

#if defined(ANYPS5_KYTY)
[[nodiscard]] std::vector<uint8_t> ComputeShaderStoreHeader(const GraphicContext& graphics);
bool LoadStoredComputeModule(std::istream& input, std::span<const uint8_t> expected_header,
                             uint64_t id, ShaderModule& module, std::vector<uint32_t>& spirv,
                             std::string& error);
#endif

}

#endif
