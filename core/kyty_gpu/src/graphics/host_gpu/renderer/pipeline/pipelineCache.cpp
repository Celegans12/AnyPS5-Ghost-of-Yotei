#include "graphics/host_gpu/renderer/perfCounters.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCacheTrace.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/envRead.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/lookupMemo.h"
#include "common/onceValue.h"
#include "common/profiler.h"
#include "common/startupTrace.h"
#include "common/threads.h"
#include "common/shaderEnv.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/debugSample.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/pipeline/blendMapping.h"
#include "graphics/host_gpu/renderer/pipeline/shaderPrecompile.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/stageFLookup.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/SrtTape.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "kytyGitVersion.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <thread>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <spirv-tools/libspirv.hpp>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {
bool ReadShaderGuestMemory(void*, uint64_t address, std::span<uint32_t> values);
// The scratch slot of MaterializeResourcesAhead on the resolve thread (ir/ShaderIR.h).
constexpr uint32_t ResolveScratchSlot = 1;
// A stage the resolve thread resolved (see ResolvedStages in pipelineCache.h).
struct ResolvedStage {
	ShaderType                                   stage      = ShaderType::Unknown;
	const ShaderRecompiler::IR::ResourcePlan*    plan       = nullptr;
	void*                                        entry      = nullptr; // the program (ProgramCache::SourceEntry)
	void*                                        permutation = nullptr; // ProgramCache::Permutation, null: none found ahead
	ShaderStageRuntime*                          out_stage   = nullptr; // where the stage's runtime went
	ShaderProgram*                               out_program = nullptr;
	uint64_t                                     shader_base = 0;
	std::array<uint32_t, 40>                     user_data {};
	uint32_t                                     user_data_count = 0;
	std::array<uint32_t, 3>                      workgroup_counts {}; // a compute stage's runtime
	bool                                         ok         = false;
	bool                                         taken      = false;
	ShaderRecompiler::IR::ResourceSnapshot       resources;
	ShaderRecompiler::IR::ResourceSpecialization specialization;
	std::vector<uint8_t>                         null_share_kinds;
	StageF::ResolveRecordPtr                     record; // FindBuffers' sources, found ahead
};
// Why the execution thread materialized a stage itself.
enum class ResolveMiss : uint32_t { NoResolution, Refused, Stale, NoMatch, Taken, Count };
// PERF_RESOLVE_STAGES: stages made, without a program (which never compiles ahead) and refused by reason; taken, whole
// logs read differently, not found at the draw; the materializations left to the execution thread by cause (draws and
// dispatches apart), with their time.
struct ResolveStats {
	std::atomic<uint64_t> made {0}, no_program {0}, taken {0}, stale {0}, missing {0}, adopted {0}, not_adopted {0};
	std::array<std::atomic<uint64_t>, static_cast<size_t>(ShaderRecompiler::IR::AheadRefusal::Count)> refused {};
	std::array<std::array<std::atomic<uint64_t>, static_cast<size_t>(ResolveMiss::Count)>, 2> walks {}, walk_ns {};
	std::array<std::atomic<uint64_t>, 16> missing_by_stage {};
} g_resolve_stats;
} // namespace

struct ResolvedStages {
	enum class Reads : uint8_t { Unchecked, Hold, Changed };

	std::array<ResolvedStage, 4>  stages;
	uint32_t                      count      = 0;
	uint32_t                      push_start = 0; // the push data cursor before the first stage
	bool                          adoptable  = false; // the execution thread may take the selection as a whole
	Reads                         reads      = Reads::Unchecked;
	ShaderRecompiler::IR::ReadLog log; // everything the operation's stages read, each read once

	void Reset() {
		count       = 0;
		push_start  = 0;
		adoptable   = false;
		reads       = Reads::Unchecked;
		log.Clear();
	}
};

ResolvedStages* NewResolvedStages() {
	return new ResolvedStages;
}
void DeleteResolvedStages(ResolvedStages* stages) {
	delete stages;
}
[[nodiscard]] bool ResolvedStagesEmpty(const ResolvedStages& stages) noexcept {
	return stages.count == 0;
}

namespace {
// Every read the resolve thread made for the operation reads the same through this thread's readers now (which download
// GPU-dirty bytes and wait for pending readbacks, as the operation's own walk would): operations before this one and the
// guest's own threads write guest memory after those reads. Asked once per operation, each read once.
bool ResolvedReadsHold(ResolvedStages& results) {
	using Reads = ResolvedStages::Reads;
	if (results.reads == Reads::Unchecked) {
		const DispatchSubTimer reads_timer(DispatchSub::AdoptReads); // KYTY_TRACE_DISPATCH_CENSUS
		const ShaderRecompiler::IR::SrtRuntime runtime {.read_specialization_memory = ReadShaderGuestMemory};
		results.reads = ShaderRecompiler::IR::ReadsUnchangedFast(results.log, runtime) ? Reads::Hold : Reads::Changed;
		if (results.reads == Reads::Changed) g_resolve_stats.stale.fetch_add(1, std::memory_order_relaxed);
	}
	return results.reads == Reads::Hold;
}

// The result of `stage` for this call, when the resolve thread made one with the same key and its reads hold: moved into
// the outputs.
ResolveMiss TakeResolvedStage(ResolvedStages& results, ShaderType stage, const ShaderRecompiler::IR::ResourcePlan& plan,
                              const ShaderRecompiler::IR::SrtRuntime& runtime, ShaderRecompiler::IR::ResourceSnapshot& out,
                              ShaderRecompiler::IR::ResourceSpecialization& out_specialization,
                              StageF::ResolveRecordPtr& out_record) {
	const auto user_data = runtime.user_data;
	for (uint32_t i = 0; i < results.count; i++) {
		auto& r = results.stages[i];
		if (r.taken || r.stage != stage || r.plan != &plan || r.shader_base != runtime.shader_base ||
		    r.user_data_count != user_data.size() || !std::equal(user_data.begin(), user_data.end(), r.user_data.begin()) ||
		    !std::equal(runtime.workgroup_counts.begin(), runtime.workgroup_counts.end(), r.workgroup_counts.begin())) {
			continue;
		}
		r.taken = true;
		if (!r.ok) return ResolveMiss::Refused;
		if (!ResolvedReadsHold(results)) return ResolveMiss::Stale;
		std::swap(out, r.resources);
		std::swap(out_specialization, r.specialization);
		ShaderRecompiler::IR::ApplyNullFormattedShare(plan, r.null_share_kinds, out_specialization); // in draw order
		std::swap(out_record, r.record);
		g_resolve_stats.taken.fetch_add(1, std::memory_order_relaxed);
		return ResolveMiss::Taken;
	}
	g_resolve_stats.missing.fetch_add(1, std::memory_order_relaxed);
	g_resolve_stats.missing_by_stage[static_cast<uint32_t>(stage) & 15u].fetch_add(1, std::memory_order_relaxed);
	return ResolveMiss::NoMatch;
}

} // namespace

ShaderRecompiler::Spirv::Profile ShaderCompileProfile(const GraphicContext& graphics) {
	// Upstream 460a196fc (maintenance8: dynamic sample texel offsets); native min/max on AMD (NMin/NMax give the guest's
	// NaN and signed-zero results); the device clock (77a8f3075); robustBufferAccess2 to the byte (ee034be4b; the
	// emitter's compute stages rely on it, step K2).
	return {.maintenance8           = graphics.maintenance8_enabled,
	        .native_float_min_max   = graphics.physical_device_properties.vendorID == 0x1002u,
	        .device_clock_period_ps = graphics.device_clock_period_ps,
	        .robust_buffer_access   = graphics.robust_buffer_access_enabled};
}

namespace {

uint8_t RemapSourceAlphaFactor(uint8_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Alpha);
		case Prospero::BlendFactor::kOneMinusSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
		default: return factor;
	}
}

vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	// CxPrimitiveSetup::PolygonMode disables both per-face modes when it is zero.
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	EXIT_NOT_IMPLEMENTED(mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	// Vulkan has one polygon mode. A culled face does not constrain that mode.
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default: EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
	}
}

std::string DriverCacheSignature(const vk::PhysicalDeviceProperties& properties) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(VK_UUID_SIZE * 2, '0');
	for (size_t i = 0; i < VK_UUID_SIZE; i++) {
		uuid[i * 2]     = hex[properties.pipelineCacheUUID[i] >> 4u];
		uuid[i * 2 + 1] = hex[properties.pipelineCacheUUID[i] & 0xfu];
	}
	// KYTY_PIPELINE_CACHE_ANY_REVISION=1: keep the cache across emulator builds.
	// The driver validates each entry against the pipeline it is asked for, so entries from
	// another build are only unused, never misapplied.
	// Default from Common::EnvDefaults; =0 turns it off.
	const char* any_revision = Common::ShaderEnv::Read("KYTY_PIPELINE_CACHE_ANY_REVISION");
	const bool  any          = any_revision != nullptr && any_revision[0] == '1';
	return fmt::format("KytyPC1:{}:{:08x}:{:08x}:{:08x}:{}\n", any ? "any" : KYTY_GIT_REVISION,
	                   properties.vendorID, properties.deviceID, properties.driverVersion, uuid);
}

std::string PipelineCacheTitleId() {
	std::string title_id;
	if ((!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) &&
	    (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) || title_id.empty())) {
		return {};
	}
	if (!std::ranges::all_of(title_id, [](unsigned char c) {
		    return std::isalnum(c) != 0 || c == '-' || c == '_';
	    })) {
		return {};
	}
	return title_id;
}



template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

// True when a user SGPR the vertex variant reads from SPIR-V draw
// parameters also feeds the descriptor, SRT or condition values the host evaluates per draw.
bool IndirectDrawSgprsFeedResources(const ShaderRecompiler::IR::ResourcePlan& plan,
                                    const ShaderIndirectDrawInfo&             indirect) {
	if (!indirect.enabled) {
		return false;
	}
	for (const auto& inst: plan.value_storage) {
		if (inst.GetOpcode() != ShaderRecompiler::IR::ValueOpcode::GetUserData ||
		    inst.NumArgs() != 1 || inst.Arg(0).GetType() != ShaderRecompiler::IR::Type::ScalarReg) {
			continue;
		}
		const auto reg = ShaderRecompiler::IR::RegIndex(inst.Arg(0).ScalarRegister());
		if (reg < plan.user_data_base) {
			continue;
		}
		const auto index = reg - plan.user_data_base;
		if (index == indirect.base_vertex || index == indirect.base_instance ||
		    index == indirect.draw_index) {
			return true;
		}
	}
	return false;
}

bool ReadShaderGuestMemory(void*, uint64_t address, std::span<uint32_t> values) {
	// Scalar and unformatted buffer dependencies use the same backing as native raw loads.
	// Image synchronization belongs to formatted buffer bindings, not these reads.
	return !values.empty() &&
	       Libs::LibKernel::Memory::TryReadBufferBacking(address, values.data(), values.size_bytes());
}

double MillisecondsSince(std::chrono::steady_clock::time_point begin) {
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
}

// The driver cache file passed 4 GiB once (4.83 GB): Common::File reads and writes 32-bit sizes, so the
// payload goes in 1 GiB pieces. Returns the bytes done (short on an error).
static constexpr uint64_t PipelineCacheFileChunk = uint64_t {1} << 30u;

static uint64_t ReadFileChunked(Common::File& file, uint8_t* data, uint64_t size) {
	uint64_t done = 0;
	while (done < size) {
		const auto piece = static_cast<uint32_t>(std::min(size - done, PipelineCacheFileChunk));
		uint32_t   read  = 0;
		file.Read(data + done, piece, &read);
		done += read;
		if (read != piece) break;
	}
	return done;
}

static uint64_t WriteFileChunked(Common::File& file, const uint8_t* data, uint64_t size) {
	uint64_t done = 0;
	while (done < size) {
		const auto piece   = static_cast<uint32_t>(std::min(size - done, PipelineCacheFileChunk));
		uint32_t   written = 0;
		file.Write(data + done, piece, &written);
		done += written;
		if (written != piece) break;
	}
	return done;
}

// The shader reuse: the one stage-input field the recompiler reads beyond the static key (the SPIR-V emitter
// puts a compute shader's LDS in a storage buffer when it does not fit the device's shared memory).
template <typename InputInfo>
bool ReuseLdsStorage(const InputInfo& input_info) {
	if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
		return input_info.lds_storage;
	} else {
		return false;
	}
}

} // namespace

std::size_t PipelineCache::GraphicsPipelineKeyHash::operator()(const GraphicsPipelineKey& key) const {
	std::size_t hash = 0;
	PipelineKeyHash::Mix(hash, key.rendering.color_count);
	for (uint32_t i = 0; i < key.rendering.color_count; i++) {
		PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.color_formats[i]));
	}
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.depth_format));
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.stencil_format));
	for (const auto id: key.vertex_shader_ids) {
		PipelineKeyHash::Mix(hash, id);
	}
	PipelineKeyHash::Mix(hash, key.ps_shader_id);
	PipelineKeyHash::Mix(hash, key.vertex_input.binding_count);
	for (uint32_t i = 0; i < key.vertex_input.binding_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].stride);
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].instance);
	}
	PipelineKeyHash::Mix(hash, key.vertex_input.attribute_count);
	for (uint32_t i = 0; i < key.vertex_input.attribute_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].offset);
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].binding);
	}
	PipelineKeyHash::Mix(hash, XXH3_64bits(&key.static_params, sizeof(key.static_params)));
	return hash;
}

struct PipelineCache::ProgramCache {
	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		uint32_t              code_size       = 0;
		std::vector<uint32_t> static_state;

		bool operator==(const ProgramKey&) const = default;
	};

	struct Permutation {
		std::vector<uint32_t>                       function_code;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::IR::CompiledShaderInfo     program;
		ShaderProgram                                handle;
	};

	// The specialization of the last permutation (null: none yet), for the logs of a new one.
	template <typename Permutations>
	static const ShaderRecompiler::IR::ResourceSpecialization* LastSpecialization(const Permutations& permutations) {
		return permutations.empty() ? nullptr : &permutations.back().specialization;
	}

	struct SourceEntry {
		SourceEntry(ShaderRecompiler::IR::ResourcePlan plan,
		            std::unique_ptr<ShaderRecompiler::ShaderSource> call_source)
		    : call_source(std::move(call_source)), resource_plan(std::move(plan)) {}

		std::span<const uint32_t> FunctionCode() const {
			return call_source
			    ? std::span<const uint32_t>(call_source->linked->code).subspan(call_source->code.size())
			    : std::span<const uint32_t>{};
		}

		bool RefreshCallSource(const ShaderRecompiler::IR::SrtRuntime& runtime) {
			const auto revision = call_source->revision;
			ShaderRecompiler::RefreshShaderSource(*call_source, runtime);
			resource_plan.source_reads = call_source->reads;
			return revision != call_source->revision;
		}

		// The permutation for the current materialization (call MaterializeResources first): push-data start,
		// specialization and, for programs with calls, the linked function code must match.
		// common/lookupMemo.h: the last matching permutation is tried first. At most one permutation matches (one is
		// compiled only when none does), so the scan below would return the same one; the index stays valid (the deque
		// only grows).
		template <bool HasCalls>
		Permutation* FindPermutation(uint32_t push_data_cursor,
		                             const ShaderRecompiler::IR::ResourceSpecialization& specialization) {
			const auto matches = [&](const Permutation& candidate) {
				const auto& layout = candidate.program.bindings;
				if (!layout.PushStartMatches(push_data_cursor) || candidate.specialization != specialization) {
					return false;
				}
				if constexpr (HasCalls) {
					if (!std::ranges::equal(candidate.function_code, FunctionCode())) return false;
				}
				return true;
			};
			if (last_permutation < permutations.size() && matches(permutations[last_permutation])) {
				Common::LookupMemo::NoteHit(Common::LookupMemo::Kind::Permutation);
				return &permutations[last_permutation];
			}
			for (size_t index = 0; index < permutations.size(); index++) {
				auto& candidate = permutations[index];
				if (!matches(candidate)) continue;
				last_permutation = index;
				return &candidate;
			}
			return nullptr;
		}
		size_t last_permutation = SIZE_MAX; // common/lookupMemo.h
		// The resolve thread, programs_mutex held shared: FindPermutation for a program without a linked call; it leaves the
		// execution thread's memo alone.
		Permutation* FindPermutationAhead(uint32_t push_data_cursor,
		                                  const ShaderRecompiler::IR::ResourceSpecialization& specialization) {
			const auto matches = [&](const Permutation& candidate) {
				return candidate.program.bindings.PushStartMatches(push_data_cursor) && candidate.specialization == specialization;
			};
			if (ahead_permutation < permutations.size() && matches(permutations[ahead_permutation])) {
				return &permutations[ahead_permutation];
			}
			for (size_t index = 0; index < permutations.size(); index++) {
				if (!matches(permutations[index])) continue;
				ahead_permutation = index;
				return &permutations[index];
			}
			return nullptr;
		}
		size_t ahead_permutation = SIZE_MAX;

		std::unique_ptr<ShaderRecompiler::ShaderSource> call_source;
		ShaderRecompiler::IR::ResourcePlan           resource_plan;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		// FindBuffers' sources for the snapshot (valid while the last call took the resolve thread's result).
		StageF::ResolveRecordPtr                     stagef_record;
		ShaderRecompiler::IR::ResourceSnapshot& Resources() { return resources; }
		StageF::ResolveRecordPtr&               StagefRecord() { return stagef_record; }
		// A deque keeps every permutation in place: input_info.stage and the pipelines point into it (a vector
		// moved them all at its 9th permutation).
		std::deque<Permutation>                     permutations;
		bool                                        indirect_draw_resource_sgprs = false;
	};

	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const {
			std::size_t hash = static_cast<std::size_t>(key.stage);
			PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash));
			if constexpr (sizeof(std::size_t) < sizeof(uint64_t)) {
				PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash >> 32u));
			}
			PipelineKeyHash::Mix(hash, key.user_data_count);
			PipelineKeyHash::Mix(hash, key.code_size);
			PipelineKeyHash::Mix(hash, key.static_state.size());
			// Bucket same-shape static variants by source. ProgramKey equality performs the one
			// exact state comparison needed on a stable hit without hashing the full state first.
			return hash;
		}
	};

	static constexpr std::size_t MaxStaticKeyWords = 32 + ShaderVertexInputInfo::RES_MAX * 6;

	// The shader reuse: `reused` is CompileProgram's output for exactly these inputs, made by a background compile
	// (`translated` is then empty).
	Permutation CompilePermutation(const char*                                  stage_name,
	                               const ShaderRecompiler::CompileOptions&      options,
	                               ShaderRecompiler::TranslateResult            translated,
	                               ShaderRecompiler::IR::ResourceSpecialization specialization,
	                               uint32_t push_data_start_dword,
	                               std::span<const uint32_t> function_code,
	                               PrecompileReused* reused) {
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::ShadersCompiled,
		                                         Common::DebugCounters::Counter::ShaderCompileNs,
		                                         Common::DebugCounters::Peak::ShaderCompileNs);
		const auto compile_begin = std::chrono::steady_clock::now();
		auto result = reused != nullptr ? ShaderRecompiler::CompileResult {}
		                                : ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                                                   specialization, push_data_start_dword);
		if (reused != nullptr) {
			result.spirv = std::move(reused->spirv);
		}
		auto program = reused != nullptr ? std::move(reused->info) : std::move(result.program).TakeCompiledInfo();
		last_compile_ms  = MillisecondsSince(compile_begin);
		last_spirv_words = result.spirv.size();
		if (precompiler != nullptr) {
			last_spirv_hash = XXH3_64bits(result.spirv.data(), result.spirv.size() * sizeof(uint32_t));
		}
		if (std::FILE* f = Common::EnvRead::ShaderStagesFile()) { std::fprintf(f, "STAGE emitted 0x%016llx%s\n", static_cast<unsigned long long>(options.shader_hash), fmt::format(" words={}", result.spirv.size()).c_str()); std::fclose(f); }
		PipelineTrace::DumpModule(stage_name, options.shader_hash, result.spirv);
		PipelineTrace::ReplaceSpirv(options.shader_hash, result.spirv);
		if (!PipelineTrace::ValidateShaderSpirv(options.dump_label, options.shader_hash, result.spirv,
		                         options.profile->maintenance8)) {
			PipelineTrace::DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);
			EXIT("%s failed hash=0x%016" PRIx64 ": SPIR-V validation failed\n", options.dump_label,
			     options.shader_hash);
		}
		PipelineTrace::DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);
		PipelineTrace::DumpSpirvText(stage_name, options.shader_hash, result.spirv);

		if (std::FILE* f = Common::EnvRead::ShaderStagesFile()) { std::fprintf(f, "STAGE module 0x%016llx%s\n", static_cast<unsigned long long>(options.shader_hash), ""); std::fclose(f); }
		const auto module_begin = std::chrono::steady_clock::now();
		const auto module = CompileSPV(result.spirv, device);
		last_module_ms = MillisecondsSince(module_begin);
		if (std::FILE* f = Common::EnvRead::ShaderStagesFile()) { std::fprintf(f, "STAGE module-done 0x%016llx%s\n", static_cast<unsigned long long>(options.shader_hash), ""); std::fclose(f); }
		EXIT_IF(module == nullptr);
		if (options.dump_ir) {
			LOGF("%s SPIR-V words=%" PRIu64 " wave_size=%u\n", options.dump_label,
			     static_cast<uint64_t>(result.spirv.size()), options.wave_size);
		}
		return {
		    .function_code  = {function_code.begin(), function_code.end()},
		    .specialization = std::move(specialization),
		    .program        = std::move(program),
		    .handle         = {.id = ++next_shader_id, .module = module},
		};
	}

	template <typename InputInfo>
	ShaderProgram Get(const ShaderParams& params, InputInfo& input_info,
	                  uint32_t& push_data_cursor) {
		ShaderType stage;
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage = input_info.logical_stage;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage = ShaderType::Pixel;
		} else {
			static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
			stage = ShaderType::Compute;
		}

		const auto user_data = std::span(params.user_data).first(params.user_data_count);
		lookup_key.stage           = stage;
		lookup_key.hash            = params.hash;
		lookup_key.user_data_count = params.user_data_count;
		lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
		// A plain block, not a lambda: a lambda inside this template would make `entry` a dependent type.
		decltype(programs)::iterator entry;
		{
			KYTY_PROFILER_BLOCK("ProgramCache::Lookup");
			const DispatchSubTimer key_timer(std::is_same_v<InputInfo, ShaderComputeInputInfo> ? DispatchSub::KeyFind
			                                                                                   : DispatchSub::GLookup);
			BuildStageStaticKey(input_info, lookup_key.static_state);
			entry = programs.find(lookup_key);
		}
		ShaderRecompiler::IR::SrtRuntime             runtime {
		    .user_data                  = user_data,
		    .shader_base                = params.Base(),
		    .read_specialization_memory = ReadShaderGuestMemory,
		    .image_heap                 = image_heap,
		};
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			runtime.workgroup_counts = input_info.workgroup_counts;
			runtime.indirect_args_address = input_info.indirect_args_address;
		}
		// A program with a linked call: its function code changed since the cached translation.
		bool source_changed = false;
		if (entry != programs.end()) {
			auto& cached = entry->second;
			auto& out_resources      = cached.Resources();
			auto& out_specialization = cached.specialization;
			auto& out_record         = cached.StagefRecord();
			if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
				input_info.indirect_draw_resource_sgprs = entry->second.indirect_draw_resource_sgprs;
			}
			if (cached.call_source) {
				source_changed = cached.RefreshCallSource(runtime);
			}
			Permutation* permutation = nullptr;
			// The resolve thread's lookups: the resolve thread's record of the snapshot when this call adopted it.
			const StageF::ResolveRecord* stagef_record = nullptr;
			if (!source_changed) {
				PerfMaterializeHash() = params.hash;
				// KYTY_TRACE_DISPATCH_CENSUS: the stage's result taken and re-checked.
				std::optional<DispatchSubTimer> adopt_timer;
				adopt_timer.emplace(std::is_same_v<InputInfo, ShaderComputeInputInfo> ? DispatchSub::Adopt
				                                                                         : DispatchSub::GTake);
				// The stage the resolve thread materialized for this operation.
				const auto take = t_resolved_stages != nullptr
				                      ? TakeResolvedStage(*t_resolved_stages, stage, entry->second.resource_plan, runtime,
				                                          out_resources, out_specialization, out_record)
				                      : ResolveMiss::NoResolution;
				const bool resolved = take == ResolveMiss::Taken;
				if (resolved) {
					stagef_record = out_record.get(); // its ResolveRecord describes out_resources
				}

				adopt_timer.reset();
				std::optional<DispatchSubTimer> mr_timer; // KYTY_TRACE_DRAW_CENSUS: graphics materialization
				if constexpr (!std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
					if (t_dispatch_sub_on) {
						t_dispatch_sub_ns[static_cast<size_t>(resolved ? DispatchSub::GPathResolved : DispatchSub::GPathMr)]++;
					}
					if (!resolved) mr_timer.emplace(DispatchSub::GMr);
				}
				if (!resolved) {
					KYTY_PROFILER_BLOCK("MaterializeResources");
					const auto walk_begin = std::chrono::steady_clock::now(); // PERF_RESOLVE_STAGES
					EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(entry->second.resource_plan, runtime, out_resources,
					                                                    out_specialization));
					const size_t kind = std::is_same_v<InputInfo, ShaderComputeInputInfo> ? 1u : 0u;
					g_resolve_stats.walks[kind][static_cast<size_t>(take)].fetch_add(1, std::memory_order_relaxed);
					g_resolve_stats.walk_ns[kind][static_cast<size_t>(take)].fetch_add(
					    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
					                              std::chrono::steady_clock::now() - walk_begin)
					                              .count()),
					    std::memory_order_relaxed);
				}
				if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) { // KYTY_TRACE_DISPATCH_CENSUS
					t_compute_program_path = resolved ? 3u : 2u;
				}
				KYTY_PROFILER_BLOCK("ProgramCache::FindPermutation");
				{
					const DispatchSubTimer permutation_timer(std::is_same_v<InputInfo, ShaderComputeInputInfo>
					                                             ? DispatchSub::Permutation
					                                             : DispatchSub::GPerm);
					permutation = cached.call_source ? cached.FindPermutation<true>(push_data_cursor, out_specialization)
					                                 : cached.FindPermutation<false>(push_data_cursor, out_specialization);
				}
			}
			if (permutation != nullptr) {
				input_info.stage = {.program       = &permutation->program,
				                    .resources     = &out_resources,
				                    .stagef_record = stagef_record};
				permutation->program.bindings.AdvancePushData(push_data_cursor);
				return permutation->handle;
			}
		}

		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) t_compute_program_path = 4u; // KYTY_TRACE_DISPATCH_CENSUS
		ShaderStageInputInfo stage_input {};
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage_input.vertex = &input_info;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage_input.pixel = &input_info;
		} else {
			stage_input.compute = &input_info;
		}
		const char* label = nullptr;
		const char* stage_name = nullptr;
		switch (stage) {
			case ShaderType::Vertex: label = "ShaderRecompiler VS"; stage_name = "vs"; break;
			case ShaderType::Mesh: label = "ShaderRecompiler MS"; stage_name = "ms"; break;
			case ShaderType::Local: label = "ShaderRecompiler LS"; stage_name = "ls"; break;
			case ShaderType::TessellationControl: label = "ShaderRecompiler HS"; stage_name = "hs"; break;
			case ShaderType::TessellationEvaluation: label = "ShaderRecompiler DS"; stage_name = "ds"; break;
			case ShaderType::Pixel: label = "ShaderRecompiler PS"; stage_name = "ps"; break;
			case ShaderType::Compute: label = "ShaderRecompiler CS"; stage_name = "cs"; break;
			default: EXIT("invalid pipeline shader stage\n");
		}
		ShaderRecompiler::CompileOptions options;
		options.stage       = stage;
		options.profile     = &profile;
		options.shader_hash = params.hash;
		options.user_data   = user_data;
		options.back_code      = params.back_code;
		options.dump_ir     = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
		options.early_dump  = options.dump_ir;
		options.emit_debug_info = Config::GraphicsDebugDumpEnabled();
		options.dump_label  = label;
		options.input_info  = stage_input;
		// User-data base and wave size by stage (KYTY_SHADER_PRECOMPILE's background compiles set theirs with it).
		SetStageCompileOptions(options);
		PipelineTrace::DumpShaderOriginal(stage_name, options.shader_hash, params.code);
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			if (Common::EnvRead::ShaderCodeDir() != nullptr) {
				std::fprintf(stderr, "CS_INFO hash=0x%016llx wave=%u wg=%ux%ux%u lds_dwords=%u user_sgprs=%d "
				             "tid=%d tgid=%d%d%d tg_size=%d\n",
				             static_cast<unsigned long long>(options.shader_hash), input_info.wave_size,
				             input_info.threads_num[0], input_info.threads_num[1], input_info.threads_num[2],
				             input_info.lds_size_dwords, input_info.workgroup_register, input_info.thread_ids_num,
				             input_info.group_id[0] ? 1 : 0, input_info.group_id[1] ? 1 : 0,
				             input_info.group_id[2] ? 1 : 0, input_info.tg_size_en ? 1 : 0);
			}
		}
		if (std::FILE* f = Common::EnvRead::ShaderStagesFile()) { std::fprintf(f, "STAGE translate 0x%016llx\n", static_cast<unsigned long long>(options.shader_hash)); std::fclose(f); }
		// A program with a scalar leaf call (SWAPPC) translates its source linked with the called function.
		ShaderRecompiler::ShaderSource source;
		const bool first_use       = entry == programs.end();
		const auto translate_begin = std::chrono::steady_clock::now();
		// The shader reuse: what a background compile of the same inputs kept (shaderPrecompile.h). A program
		// with a linked call is never kept (its callee is read from guest memory).
		const bool reuse = precompiler != nullptr && (first_use || !entry->second.call_source) &&
		                   (stage == ShaderType::Compute || ShaderPrecompiler::GraphicsEnabled()) && ShaderReuseEnabled();
		PrecompileReuseIdentity reuse_identity;
		if (reuse) {
			reuse_identity = {.stage           = stage,
			                  .shader_hash     = params.hash,
			                  .user_data_count = params.user_data_count,
			                  .code            = params.code,
			                  .back_code       = params.back_code,
			                  .static_key      = lookup_key.static_state,
			                  .lds_storage     = ReuseLdsStorage(input_info)};
		}
		// A first use takes the resource plan of a stored translation of the program; the translation is then left
		// until a permutation is not stored.
		std::optional<ShaderRecompiler::IR::ResourcePlan> reused_plan;
		if (reuse && first_use) {
			precompiler->TakeReusedPlan(reuse_identity, reused_plan);
		}
		std::optional<ShaderRecompiler::TranslateResult> translated;
		double                                           translate_ms = 0.0;
		const auto translate = [&] {
			const auto begin = std::chrono::steady_clock::now();
			KYTY_PROFILER_BLOCK("PipelineCache::Translate");
			if (entry == programs.end()) source = ShaderRecompiler::PrepareShaderSource(params.code, options);
			// A GPU-selected call target the host cannot resolve is not translated (Get stops the emulator below).
			if (entry == programs.end() && source.unsupported_call_pc != UINT32_MAX) return;
			auto* current = entry == programs.end() ? &source : entry->second.call_source.get();
			auto& result  = translated.emplace();
			if (current == nullptr) {
				result = ShaderRecompiler::TranslateProgram(params.code, options);
			} else if (!current->calls.empty()) {
				if (entry == programs.end()) ShaderRecompiler::RefreshShaderSource(*current, runtime);
				result = ShaderRecompiler::TranslateProgram(current->linked->decoded, options);
				result.program.source_reads = current->reads;
			} else {
				result = ShaderRecompiler::TranslateProgram(current->decoded, options);
			}
			translate_ms += MillisecondsSince(begin);
		};
		// A new permutation of a known program (same call source) is translated only when the store has no compile of it
		// (below).
		const bool defer_translation = reuse && !first_use && !source_changed;
		if (!reused_plan.has_value() && !defer_translation) {
			translate();
		}
		if (entry == programs.end() && source.unsupported_call_pc != UINT32_MAX) {
			// Upstream cad6aafdc skips such a dispatch ("TEMPORARY COMPUTE DISPATCH SKIP", rendering incomplete); not taken
			// (a skip is a workaround, SESJA-SYNC1): the emulator stops as before the merge.
			EXIT("shader 0x%016" PRIx64 ": call at pc 0x%08x has an unavailable scalar target (GPU-selected callable)\n",
			     params.hash, source.unsupported_call_pc);
		}
		const auto   materialize_begin = std::chrono::steady_clock::now();
		if (std::FILE* f = Common::EnvRead::ShaderStagesFile()) { std::fprintf(f, "STAGE extract 0x%016llx\n", static_cast<unsigned long long>(options.shader_hash)); std::fclose(f); }
		if (entry == programs.end()) {
			// The stored plan (of the same translation), else this thread's own.
			auto plan = [&] {
				if (reused_plan.has_value()) return std::move(*reused_plan);
				return ShaderRecompiler::IR::ExtractResourcePlan(translated->program);
			}();
			auto retained = !source.calls.empty() ? std::make_unique<ShaderRecompiler::ShaderSource>(std::move(source)) : nullptr;
			{
				// The resolve thread reads `programs` from its own thread; the insert (which may rehash) excludes it.
				std::unique_lock insert_lock(programs_mutex);
				entry = programs.try_emplace(lookup_key, std::move(plan), std::move(retained)).first;
				if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
					entry->second.indirect_draw_resource_sgprs = IndirectDrawSgprsFeedResources(
					    entry->second.resource_plan, input_info.indirect_draw);
				}
			}
			if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
				input_info.indirect_draw_resource_sgprs = entry->second.indirect_draw_resource_sgprs;
			}
			// The null share journal: the journal's last live values of the program's formatted buffers, before its
			// first materialization (a program with a linked call is never recorded).
			if (precompiler != nullptr && !entry->second.call_source) {
				const PrecompileReuseIdentity identity = {.stage           = stage,
				                                          .shader_hash     = params.hash,
				                                          .user_data_count = params.user_data_count,
				                                          .code            = params.code,
				                                          .back_code       = params.back_code,
				                                          .static_key      = lookup_key.static_state,
				                                          .lds_storage     = ReuseLdsStorage(input_info)};
				std::vector<std::array<uint32_t, 2>> journal_live;
				std::vector<uint8_t>                 journal_multi;
				if (precompiler->NullShareJournal(identity, entry->second.resource_plan.info.buffers.size(), journal_live,
				                                  journal_multi)) {
					ShaderRecompiler::IR::SeedNullFormattedShare(entry->second.resource_plan, journal_live, journal_multi);
				}
			}
			PerfMaterializeHash() = params.hash;
			KYTY_PROFILER_BLOCK("MaterializeResources");
			if (!ShaderRecompiler::IR::MaterializeResources(entry->second.resource_plan, runtime,
			                                                entry->second.Resources(), entry->second.specialization)) {
				const auto& failed = entry->second.resource_plan;
				const auto  indirect =
				    std::ranges::count_if(failed.descriptor_sources, [](const auto& source) {
					    return source.indirect_descriptor.has_value();
				    });
				EXIT("shader resource materialization failed: hash=0x%016llx stage=%u tracking_complete=%d "
				     "specialization_memory=%d capture_reads=%d indirect_descriptors=%d\n",
				     static_cast<unsigned long long>(params.hash), static_cast<uint32_t>(stage),
				     failed.resource_tracking_complete ? 1 : 0, failed.requires_specialization_memory ? 1 : 0,
				     failed.capture_specialization_reads ? 1 : 0, static_cast<int>(indirect));
			}
		} else if (source_changed) {
			// The called function changed: a new plan, and a permutation of the same linked code may already exist.
			entry->second.resource_plan = ShaderRecompiler::IR::ExtractResourcePlan(translated->program);
			EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(entry->second.resource_plan, runtime,
			                                                     entry->second.Resources(), entry->second.specialization));
			if (auto* permutation = entry->second.FindPermutation<true>(push_data_cursor, entry->second.specialization)) {
				input_info.stage = {.program = &permutation->program, .resources = &entry->second.Resources()};
				permutation->program.bindings.AdvancePushData(push_data_cursor);
				return permutation->handle;
			}
		}
		const double materialize_ms = MillisecondsSince(materialize_begin);
		if (std::FILE* f = Common::EnvRead::ShaderStagesFile()) { std::fprintf(f, "STAGE compile 0x%016llx\n", static_cast<unsigned long long>(options.shader_hash)); std::fclose(f); }
		PipelineTrace::TracePermutation(options.shader_hash, LastSpecialization(entry->second.permutations),
		                                entry->second.permutations.size(), entry->second.specialization);
		// KYTY_SHADER_PRECOMPILE: a background compile of this compute permutation is waited for (its driver pipeline
		// is then cached); one not started yet is left to this thread. A graphics permutation is recorded for the
		// background compiles of the pipelines made from it (KYTY_SHADER_PRECOMPILE_GRAPHICS).
		uint64_t       precompile_key     = 0;
		const uint32_t compile_push_start = push_data_cursor;
		bool           queued_dropped     = false;
		if (precompiler != nullptr && !entry->second.call_source &&
		    (stage == ShaderType::Compute || ShaderPrecompiler::GraphicsEnabled())) {
			precompile_key = PrecompileKey(stage, params.hash, params.user_data_count,
			                               static_cast<uint32_t>(params.code.size()), lookup_key.static_state,
			                               push_data_cursor, entry->second.specialization);
			if (stage == ShaderType::Compute) {
				queued_dropped = precompiler->BeforeCompile(precompile_key);
			}
		}
		// The stored compile of exactly this permutation (after the wait above, a running compute job's is stored too),
		// taken instead of CompileProgram. Without one, a program whose plan came from the store is translated now.
		PrecompileReused reused;
		const bool       use_reused = reuse && precompile_key != 0 &&
		                        precompiler->TakeReusedCompile(precompile_key, reuse_identity, compile_push_start,
		                                                       entry->second.specialization, queued_dropped, reused);
		const bool skip_translation = use_reused && !translated.has_value();
		if (!use_reused && !translated.has_value()) {
			translate();
		}
		ShaderRecompiler::IR::NoteNullShareVariant(entry->second.resource_plan,
		                                           entry->second.specialization); // the null share journal
		{
			auto compiled = CompilePermutation(
			    stage_name, options,
			    translated.has_value() ? std::move(*translated) : ShaderRecompiler::TranslateResult {},
			    entry->second.specialization, push_data_cursor, entry->second.FunctionCode(),
			    use_reused ? &reused : nullptr);
			std::unique_lock append_lock(programs_mutex); // the resolve thread searches the list under it
			entry->second.permutations.push_back(std::move(compiled));
		}
		if (use_reused) {
			ShaderReuseNoteUsed(reused.compile_ms + (skip_translation ? reused.translate_ms : 0.0));
		}
		const auto& permutation = entry->second.permutations.back();
		input_info.stage = {.program = &permutation.program, .resources = &entry->second.Resources()};
		permutation.program.bindings.AdvancePushData(push_data_cursor);
		if (precompile_key != 0) {
			RecordPrecompileStage(stage, params, input_info, compile_push_start, permutation, precompile_key);
		}
		// KYTY_TRACE_PERF: a shader preparation over 50 ms, split by step (the GPU thread waits for all of it).
		if (PerfEnabled()) {
			const double total_ms = MillisecondsSince(translate_begin);
			if (total_ms >= 50.0) {
				std::fprintf(stderr,
				             "SHADER_TIME hash=0x%016llx stage=%s permutation=%zu first_use=%d total=%.0fms "
				             "translate=%.0f materialize=%.0f compile=%.0f module=%.0f spirv_words=%zu\n",
				             static_cast<unsigned long long>(options.shader_hash), stage_name,
				             entry->second.permutations.size(), first_use ? 1 : 0, total_ms, translate_ms,
				             materialize_ms, last_compile_ms, last_module_ms, last_spirv_words);
				std::fflush(stderr);
			}
		}
		// KYTY_TRACE_STARTUP: every permutation, split as SHADER_TIME splits it.
		if (Common::StartupTrace::Enabled()) {
			using Common::StartupTrace::Counter;
			Common::StartupTrace::AddMs(Counter::Shader, MillisecondsSince(translate_begin));
			Common::StartupTrace::AddMs(Counter::ShaderTranslate, translate_ms);
			Common::StartupTrace::AddMs(Counter::ShaderCompile, materialize_ms + last_compile_ms + last_module_ms);
		}

		std::array<size_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> counts {};
		for (const auto& [key, source]: programs) {
			counts[static_cast<size_t>(key.stage)] += source.permutations.size();
		}
		// Guest geometry shaders are compiled through the host mesh stage.
		std::printf("Shaders: VS %zu | PS %zu | CS %zu | GS %zu | LS %zu | HS %zu | TES %zu\n",
		            counts[static_cast<size_t>(ShaderType::Vertex)],
		            counts[static_cast<size_t>(ShaderType::Pixel)],
		            counts[static_cast<size_t>(ShaderType::Compute)],
		            counts[static_cast<size_t>(ShaderType::Mesh)],
		            counts[static_cast<size_t>(ShaderType::Local)],
		            counts[static_cast<size_t>(ShaderType::TessellationControl)],
		            counts[static_cast<size_t>(ShaderType::TessellationEvaluation)]);
		return permutation.handle;
	}

	// ---- KYTY_SHADER_PRECOMPILE (GPU thread) ----

	// The inputs of a permutation this thread just compiled, for the background compiles. The code must still be
	// the code the shader's hash names (the guest can write other code to a shader's address).
	template <typename InputInfo>
	void RecordPrecompileStage(ShaderType stage, const ShaderParams& params, const InputInfo& input_info,
	                           uint32_t push_data_start, const Permutation& permutation, uint64_t key) {
		PrecompileStageRecord record;
		record.stage           = stage;
		record.shader_hash     = params.hash;
		const auto user_data   = std::span(params.user_data).first(params.user_data_count);
		record.user_data.assign(user_data.begin(), user_data.end());
		record.code.assign(params.code.begin(), params.code.end());
		record.back_code.assign(params.back_code.begin(), params.back_code.end());
		if (PrecompileCodeHash(record.code, record.back_code) != params.hash) {
			static std::atomic_uint logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 16u) {
				std::fprintf(stderr, "SHADER_PRECOMPILE not recorded: hash=0x%016llx (the code at its address changed)\n",
				             static_cast<unsigned long long>(params.hash));
			}
			return;
		}
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			record.vertex       = input_info;
			record.vertex.stage = {};
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			record.pixel       = input_info;
			record.pixel.stage = {};
		} else {
			record.compute       = input_info;
			record.compute.stage = {};
		}
		record.static_key      = lookup_key.static_state;
		record.push_data_start = push_data_start;
		record.specialization  = permutation.specialization;
		record.spirv_hash      = last_spirv_hash;
		record.revision        = ShaderPrecompiler::Revision();
		precompiler->Record(record);
		// Compute ones too: GetComputePipeline marks the key warm once the pipeline exists.
		precompile_stages[permutation.handle.id] = {.key = key, .spirv_hash = last_spirv_hash};
	}

	// The record keys of a graphics pipeline's stages; false unless every stage was recorded.
	bool PrecompileStageKeys(const GraphicsPrograms& programs, bool ps_active, PrecompileGraphicsRecord& record) const {
		const auto find = [this](const ShaderProgram& program, uint64_t& key) {
			const auto found = precompile_stages.find(program.id);
			if (found == precompile_stages.end()) {
				return false;
			}
			key = found->second.key;
			return true;
		};
		for (uint32_t i = 0; i < programs.VertexStageCount(); i++) {
			if (!find(programs.vertex[i], record.vertex[i])) {
				return false;
			}
		}
		return !ps_active || find(programs.pixel, record.pixel);
	}

	// ---- The resolve thread ----

	// The stage of `params`, ahead of its execution: the program of the exact key, the materialization with the logged
	// reader into `results`, the permutation. It never compiles: without the program (or with a linked call, which reads its
	// function code from guest memory) the stage is left to the execution thread. `push_data_cursor` is the resolution's
	// own. Based on TheCruZ/KytyPS5-GTA daaf45201 (pipelineCache.cpp), GPL-2.0.
	template <typename InputInfo>
	void Resolve(RenderContext& renderer, const ShaderParams& params, InputInfo& input_info, uint32_t& push_data_cursor,
	             ResolvedStages& results, ShaderProgram& out_program) {
		ShaderType stage;
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage = input_info.logical_stage;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage = ShaderType::Pixel;
		} else {
			stage = ShaderType::Compute;
		}
		ahead_key.stage           = stage;
		ahead_key.hash            = params.hash;
		ahead_key.user_data_count = params.user_data_count;
		ahead_key.code_size       = static_cast<uint32_t>(params.code.size());
		BuildStageStaticKey(input_info, ahead_key.static_state);
		SourceEntry* entry = nullptr;
		{
			std::shared_lock lock(programs_mutex);
			if (const auto found = programs.find(ahead_key); found != programs.end()) entry = &found->second;
		}
		if (entry == nullptr || entry->call_source || results.count >= results.stages.size()) {
			g_resolve_stats.no_program.fetch_add(1, std::memory_order_relaxed);
			results.adoptable = false;
			return;
		}
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			input_info.indirect_draw_resource_sgprs = entry->indirect_draw_resource_sgprs;
		}
		auto& r           = results.stages[results.count++];
		r.stage           = stage;
		r.plan            = &entry->resource_plan;
		r.entry           = entry;
		r.permutation     = nullptr;
		r.out_stage       = &input_info.stage;
		r.out_program     = &out_program;
		r.shader_base     = params.Base();
		r.user_data_count = params.user_data_count;
		std::copy_n(params.user_data.begin(), params.user_data_count, r.user_data.begin());
		r.taken = false;
		ShaderRecompiler::IR::SrtRuntime runtime {
		    .user_data                  = std::span<const uint32_t>(r.user_data.data(), r.user_data_count),
		    .shader_base                = r.shader_base,
		    .read_specialization_memory = &ShaderRecompiler::IR::AheadReadMemory,
		};
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			std::ranges::copy(input_info.workgroup_counts, r.workgroup_counts.begin());
			runtime.workgroup_counts = std::span<const uint32_t>(r.workgroup_counts.data(), 3);
		}
		auto refusal = ShaderRecompiler::IR::AheadRefusal::None;
		r.ok = ShaderRecompiler::IR::MaterializeResourcesAhead(*r.plan, runtime, r.resources, r.specialization, results.log,
		                                                       &refusal);
		if (!r.ok) {
			g_resolve_stats.refused[static_cast<size_t>(refusal)].fetch_add(1, std::memory_order_relaxed);
			results.adoptable = false;
			return;
		}
		r.null_share_kinds = r.plan->scratch[ResolveScratchSlot].null_share_kinds;
		StageF::BuildResolveRecord(renderer, r.resources, r.record); // FindBuffers' sources for this snapshot
		g_resolve_stats.made.fetch_add(1, std::memory_order_relaxed);
		Permutation* permutation = nullptr;
		{
			std::shared_lock lock(programs_mutex);
			permutation = entry->FindPermutationAhead(push_data_cursor, r.specialization);
		}
		r.permutation    = permutation;
		input_info.stage = {.program       = permutation != nullptr ? &permutation->program : nullptr,
		                    .resources     = &r.resources,
		                    .stagef_record = r.record.get()};
		if (permutation != nullptr) {
			out_program = permutation->handle;
			permutation->program.bindings.AdvancePushData(push_data_cursor);
		}
	}

	// The execution thread, the reads hold: finishes the selection as the calls of Get would, in their order: the null
	// share in draw order, the permutations of stages the resolve thread did not find one for (or whose specialization the
	// share changed), the push cursor through them. False: a stage needs a compile, Get takes it from here.
	bool Adopt(ResolvedStages& results) {
		uint32_t cursor = results.push_start;
		for (uint32_t i = 0; i < results.count; i++) {
			auto& r     = results.stages[i];
			auto& entry = *static_cast<SourceEntry*>(r.entry);
			ShaderRecompiler::IR::ApplyNullFormattedShare(*r.plan, r.null_share_kinds, r.specialization);
			auto*      permutation = static_cast<Permutation*>(r.permutation);
			const bool shared      = std::ranges::find(r.null_share_kinds, uint8_t {1}) != r.null_share_kinds.end();
			if (permutation == nullptr || shared || !permutation->program.bindings.PushStartMatches(cursor)) {
				permutation = entry.FindPermutation<false>(cursor, r.specialization);
				if (permutation == nullptr) return false;
				r.out_stage->program = &permutation->program;
				*r.out_program       = permutation->handle;
			}
			permutation->program.bindings.AdvancePushData(cursor);
		}
		for (uint32_t i = 0; i < results.count; i++) {
			auto& r = results.stages[i];
			r.taken = true;
		}
		g_resolve_stats.taken.fetch_add(results.count, std::memory_order_relaxed);
		return true;
	}

	ProgramCache(GraphicContext& graphics, ShaderRecompiler::IR::ImageHeapSource* image_heap)
	    : device(graphics.device), graphics(graphics), image_heap(image_heap), profile(ShaderCompileProfile(graphics)) {
		lookup_key.static_state.reserve(MaxStaticKeyWords);
		ahead_key.static_state.reserve(MaxStaticKeyWords);
	}
	~ProgramCache() {
		for (const auto& [key, entry]: programs) {
			(void)key;
			for (const auto& permutation: entry.permutations) {
				graphics.device.destroyShaderModule(permutation.handle.module, nullptr);
			}
		}
	}

	std::unordered_map<ProgramKey, SourceEntry, ProgramKeyHash> programs;
	ProgramKey                                                  lookup_key;
	ProgramKey                                                  ahead_key; // the resolve thread's
	vk::Device                                                  device;
	GraphicContext&                                             graphics;
	// Upstream's image heap source (27dc7c281): always null here (RenderContext m_image_heap, SESJA-SYNC1).
	ShaderRecompiler::IR::ImageHeapSource*                      image_heap;
	// Upstream 460a196fc / 286481889: maintenance8 dynamic sample offsets, native float min/max on AMD.
	const ShaderRecompiler::Spirv::Profile                      profile;
	uint64_t                                                    next_shader_id = 0;
	// CompilePermutation's last split, for SHADER_TIME (KYTY_TRACE_PERF).
	double last_compile_ms  = 0.0;
	double last_module_ms   = 0.0;
	size_t last_spirv_words = 0;
	uint64_t last_spirv_hash = 0; // XXH3 of the module, recorded for KYTY_SHADER_PRECOMPILE
	ShaderPrecompiler* precompiler = nullptr; // KYTY_SHADER_PRECOMPILE (owned by PipelineCache)
	// KYTY_SHADER_PRECOMPILE: the record of each permutation this thread compiled and recorded, by program id (GPU
	// thread only).
	struct PrecompileStage {
		uint64_t key        = 0;
		uint64_t spirv_hash = 0;
	};
	std::unordered_map<uint64_t, PrecompileStage> precompile_stages;
	// Inserts into `programs` and appends to a program's permutations take it exclusively (execution thread), the resolve
	// thread's finds and permutation searches shared. The execution thread's own reads are unlocked: it is the only writer.
	std::shared_mutex                                           programs_mutex;
};

// KYTY_PERF_TABLE_ASYNC: one queued pipeline creation (GetGraphicsPipelineAsync). Its inputs are copies (the draw's
// state is the next draw's): the stages' program infos are the program cache's (kept for the session), their resource
// snapshots are not read by the creation (CreatePipelineInternal only checks a stage has one: `resources` points at
// `no_resources`). Written by a worker until `done`, then read by the cache's thread.
struct PipelineCache::TableCompileJob {
	PipelineRenderingState                 rendering;
	PipelineVertexInputState               vertex_input;
	PipelineStaticParameters               static_params;
	std::array<ShaderVertexInputInfo, 3>   vertex_info {};
	uint32_t                               vertex_count = 1;
	ShaderPixelInputInfo                   ps_input_info {};
	bool                                   ps_active = false;
	GraphicsPrograms                       programs;
	ShaderRecompiler::IR::ResourceSnapshot no_resources;
	// KYTY_SHADER_PRECOMPILE: the pipeline's record (made from recorded stages), recorded once the cache takes it.
	PrecompileGraphicsRecord               precompile;
	bool                                   precompile_record = false;
	std::unique_ptr<Pipeline>              pipeline = std::make_unique<Pipeline>();
	bool                                   failed   = false; // the driver refused it (nothing of it is left)
	bool                                   failed_counted = false; // (the cache's thread: PERF table_async)
	double                                 ms       = 0.0;   // the worker's time on it
	std::atomic<bool>                      done {false};
};

// KYTY_PERF_TABLE_ASYNC: the threads that create queued pipelines beside the game (below normal priority,
// labelled and given up on as background compiles: EnterBackgroundCompileThread). Based on chenxiao07/KytyPS5 v20261020b
// (src/graphics/host_gpu/renderer/pipeline/pipelineCache.cpp, CompileWorkers), GPL-2.0.
class PipelineCache::TableCompileWorkers {
public:
	explicit TableCompileWorkers(uint32_t count) {
		for (uint32_t i = 0; i < count; i++) {
			m_threads.emplace_back([this, i] {
				EnterBackgroundCompileThread(fmt::format("Thread_TableCompile{}", i).c_str());
				Run();
			});
		}
	}
	~TableCompileWorkers() { Stop(); }
	KYTY_CLASS_NO_COPY(TableCompileWorkers);

	void Push(std::function<void()> job) {
		{
			std::lock_guard lock(m_mutex);
			m_jobs.push_back(std::move(job));
		}
		m_wake.notify_one();
	}
	// Joins the threads (idempotent): a running job finishes, the queued ones are dropped (their creations never finish;
	// their draws stay on today's path).
	void Stop() {
		{
			std::lock_guard lock(m_mutex);
			m_stop = true;
		}
		m_wake.notify_all();
		for (auto& thread: m_threads) {
			if (thread.joinable()) thread.join();
		}
	}
	[[nodiscard]] size_t Queued() {
		std::lock_guard lock(m_mutex);
		return m_jobs.size();
	}

private:
	void Run() {
		for (;;) {
			std::function<void()> job;
			{
				std::unique_lock lock(m_mutex);
				m_wake.wait(lock, [this] { return m_stop || !m_jobs.empty(); });
				if (m_stop) return;
				job = std::move(m_jobs.front());
				m_jobs.pop_front();
			}
			job();
		}
	}

	std::mutex                        m_mutex;
	std::condition_variable           m_wake;
	std::deque<std::function<void()>> m_jobs;
	bool                              m_stop = false;
	std::vector<std::thread>          m_threads;
};

PipelineCache::PipelineCache(GraphicContext&                         graphics,
                             ShaderRecompiler::IR::ImageHeapSource* image_heap)
    : m_graphics(graphics), m_program_cache(std::make_unique<ProgramCache>(graphics, image_heap)) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	InitializeDriverCache();
	if (m_driver_cache != nullptr && ShaderPrecompiler::Enabled()) {
		// KYTY_PERF_BINDLESS_CLASSES changes how many image resources a shader has (a resource per table field): a
		// journal record replayed under the other setting would not fit the program translated now. Each mode keeps
		// journals and a warm list of its own (the driver cache is shared: unchanged shaders keep their pipelines).
		const auto classes = ShaderRecompiler::IR::BindlessClassesRequested() ? ShaderRecompiler::IR::BindlessClassesMode()
		                                                                       : 0u;
		const auto suffix  = classes != 0u ? fmt::format(".classes{}", classes) : std::string();
		auto journal = m_driver_cache_path;
		journal.replace_extension(suffix + ".cs.journal");
		auto graphics_journal = m_driver_cache_path;
		graphics_journal.replace_extension(suffix + ".gfx.journal");
		auto warm = m_driver_cache_path;
		// ".warm"; KYTY_SHADER_PRECOMPILE_WARM_PRUNE: ".warm2" (the list by build).
		warm.replace_extension(suffix + PrecompileWarmList::FileExtension());
		m_warm = std::make_shared<PrecompileWarmList>(
		    warm, DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties()), m_driver_cache_loaded_bytes);
		if (const auto& out = Config::GetPrecompileExportHints(); Config::PrecompileOnly() && !out.empty()) {
			// --export-hints: the journals as a hints file; nothing is compiled.
			std::string summary;
			m_hints_export = true;
			m_hints_failed = !PrecompileHintsExport(journal, graphics_journal, out, summary);
			m_hints_line   = fmt::format("PRECOMPILE_HINTS kind=export status={} {}", m_hints_failed ? "failed" : "ok",
			                             summary);
			return;
		}
		if (const auto& hints = Config::GetPrecompileImportHints(); Config::PrecompileOnly() && !hints.empty()) {
			// --import-hints: the records found are added to the journals before the precompiler loads them.
			std::string summary;
			m_hints_failed = !PrecompileHintsImport(m_graphics, hints, Config::GetGameDir(), journal, graphics_journal,
			                                        summary);
			m_hints_line   = fmt::format("PRECOMPILE_HINTS kind=import status={} {}", m_hints_failed ? "failed" : "ok",
			                             summary);
		}
		m_precompiler = std::make_unique<ShaderPrecompiler>(m_graphics, m_driver_cache, journal, graphics_journal,
		                                                    m_precompiled_since_save, m_warm.get());
		m_program_cache->precompiler = m_precompiler.get();
	}
}

// The renderer is complete once the GPU thread asks for a pipeline (the bindless table is built after this cache).
void PipelineCache::StartBackgroundCompiles() {
	if (m_precompiler != nullptr) m_precompiler->Start();
}

void PipelineCache::StopBackgroundCompiles() {
	// KYTY_PERF_TABLE_ASYNC: its workers first (one may wait for a precompiler job: ShaderPrecompiler::BeforeCompile);
	// nothing is queued after this.
	if (m_table_workers != nullptr) m_table_workers->Stop();
	m_table_workers_stopped = true;
	if (m_precompiler != nullptr) m_precompiler->Stop();
}

// --precompile-only: one PRECOMPILE_* line on stdout (the launcher reads them), flushed at once.
static void PrecompileOnlyLine(const std::string& text) {
	std::fputs(text.c_str(), stdout);
	std::fputc('\n', stdout);
	std::fflush(stdout);
}

// PRECOMPILE_PROGRESS: jobs through (done) of those queued at the start (total), by kind; pipelines the driver really
// compiled; seconds since the start and to the end. The workers take compute and graphics jobs in proportion
// (ShaderPrecompiler's order), so the share of jobs done stands for the share of the time; eta_s is -1 until 10 s and
// one job have passed.
static std::string PrecompileOnlyProgressLine(const char* phase, const PrecompileProgress& progress, double elapsed_s) {
	const uint32_t total          = progress.queued;
	const uint32_t done           = std::min(progress.finished, total);
	const uint32_t graphics_total = std::min(progress.queued_graphics, total);
	const uint32_t graphics_done  = std::min(progress.finished_graphics, done);
	int64_t        eta_s          = -1;
	if (done >= total) {
		eta_s = 0;
	} else if (done != 0 && elapsed_s >= 10.0) {
		eta_s = static_cast<int64_t>(elapsed_s * static_cast<double>(total - done) / static_cast<double>(done) + 0.5);
	}
	return fmt::format("PRECOMPILE_PROGRESS phase={} done={} total={} compute={}/{} graphics={}/{} compiled={} "
	                   "threads={} elapsed_s={} eta_s={}",
	                   phase, done, total, done - graphics_done, total - graphics_total, graphics_done, graphics_total,
	                   progress.compiled, progress.threads, static_cast<uint64_t>(elapsed_s), eta_s);
}

int PipelineCache::RunPrecompileOnly(uint32_t threads, const std::atomic<bool>& stop, Common::Mutex& save_mutex) {
	using Clock = std::chrono::steady_clock;
	if (!m_hints_line.empty()) {
		PrecompileOnlyLine(m_hints_line);
		std::fprintf(stderr, "%s\n", m_hints_line.c_str());
	}
	if (m_hints_export) {
		PrecompileOnlyLine(fmt::format("PRECOMPILE_DONE status={}", m_hints_failed ? "export_failed" : "exported"));
		return m_hints_failed ? 5 : 0;
	}
	if (m_hints_failed) {
		// An import that could not be used (another game, mode or emulator layout): nothing was added; the recorded
		// shaders are precompiled as without it.
		PrecompileOnlyLine("PRECOMPILE_HINTS_IGNORED");
	}
	if (m_precompiler == nullptr) {
		// No driver cache (a debug, dirty or unknown build, or no title id), or KYTY_SHADER_PRECOMPILE=0.
		PrecompileOnlyLine(fmt::format("PRECOMPILE_DONE status=unavailable reason={}",
		                               m_driver_cache == nullptr ? "no_pipeline_cache" : "precompile_off"));
		return 2;
	}
	const auto begin   = Clock::now();
	const auto elapsed = [&begin] { return std::chrono::duration<double>(Clock::now() - begin).count(); };
	m_precompiler->Start(threads);
	auto progress = GetPrecompileProgress();
	// cached: left out, their pipelines are in the saved driver cache already (warm list); suspects: left out for this
	// build (a compile of theirs ended a run); old_journals: journal files of another record layout or version, not read
	// (an older emulator wrote them; the game starts them again, this mode leaves them as they are).
	PrecompileOnlyLine(fmt::format(
	    "PRECOMPILE_START total={} compute={} graphics={} cached={} suspects={} threads={} old_journals={}",
	    progress.queued, progress.queued - progress.queued_graphics, progress.queued_graphics, progress.warm,
	    progress.suspects, progress.threads, progress.other_layout_journals));
	PrecompileOnlyLine(PrecompileOnlyProgressLine("compile", progress, 0.0));
	// The periodic save (as SaveIfDue while the precompiler is busy): a run that is killed loses at most this much work.
	constexpr auto save_interval = std::chrono::minutes(6);
	auto           last_line     = begin;
	auto           last_save     = begin;
	bool           stopping      = false;
	while (m_precompiler->Busy()) {
		std::this_thread::sleep_for(std::chrono::milliseconds(250));
		if (!stopping && stop.load(std::memory_order_acquire)) {
			// The running compiles finish (a large compute pipeline can take the driver ~45 s); queued jobs stay.
			stopping = true;
			m_precompiler->RequestStop();
			last_line = Clock::time_point {}; // a line now
		}
		const auto now = Clock::now();
		if (now - last_line >= std::chrono::seconds(1)) {
			last_line = now;
			PrecompileOnlyLine(
			    PrecompileOnlyProgressLine(stopping ? "stopping" : "compile", GetPrecompileProgress(), elapsed()));
		}
		if (!stopping && now - last_save >= save_interval) {
			last_save = now;
			Common::LockGuard lock(save_mutex);
			Save(true);
			PrecompileOnlyLine(fmt::format("PRECOMPILE_SAVE kind=periodic elapsed_s={}", static_cast<uint64_t>(elapsed())));
		}
	}
	m_precompiler->Stop(); // joins the workers
	progress = GetPrecompileProgress();
	// A stop also ends graphics jobs between their stages (counted through, their pipelines not made): the next run
	// queues them again.
	const bool complete = !stopping && progress.finished >= progress.queued;
	PrecompileOnlyLine(PrecompileOnlyProgressLine("save", progress, elapsed()));
	bool saved = true;
	if (progress.queued != 0) {
		// The driver cache, then the warm list with every pipeline created since the last save.
		Common::LockGuard lock(save_mutex);
		saved = Save(false);
	}
	PrecompileOnlyLine(fmt::format("PRECOMPILE_DONE status={} done={} total={} compiled={} elapsed_s={} saved={}",
	                               complete ? "complete" : "stopped", std::min(progress.finished, progress.queued),
	                               progress.queued, progress.compiled, static_cast<uint64_t>(elapsed()), saved ? 1 : 0));
	if (!saved) return 4;
	return complete ? 0 : 3;
}

PipelineCache::HostCensus PipelineCache::TakeHostCensus() const {
	HostCensus census;
	// Every module the GPU thread compiled is kept until the cache is destroyed (~ProgramCache).
	census.shader_modules     = m_program_cache->next_shader_id;
	census.graphics_pipelines = m_graphics_pipelines.size();
	census.compute_pipelines  = m_compute_pipelines.size();
	census.cache_loaded       = m_cache_loaded_bytes;
	census.cache_saved        = m_cache_saved_bytes->load(std::memory_order_relaxed);
	return census;
}

PipelineCache::~PipelineCache() {
	// KYTY_PERF_TABLE_ASYNC: its workers first (they use the precompiler, the driver cache and the programs' modules);
	// the pipelines they made that no draw took yet are destroyed with the others.
	m_table_workers.reset();
	for (auto& [key, job]: m_table_pending) {
		if (job->done.load(std::memory_order_acquire) && !job->failed && job->pipeline != nullptr) {
			m_graphics_pipelines.emplace(key, std::move(job->pipeline));
		}
	}
	m_table_pending.clear();
	// Background compiles first: they use the driver cache the save below writes out.
	m_program_cache->precompiler = nullptr;
	m_precompiler.reset();
	Save();
	auto destroy = [this](const auto& pipelines) {
		for (const auto& [key, pipeline]: pipelines) {
			(void)key;
			m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr);
			if (pipeline->push_template != nullptr) {
				m_graphics.device.destroyDescriptorUpdateTemplate(pipeline->push_template, nullptr);
			}
			m_graphics.device.destroyPipelineLayout(pipeline->pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline->descriptor_set_layout, nullptr);
		}
	};
	if (m_save_thread.joinable()) {
		m_save_thread.join();
	}
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	if (m_driver_cache != nullptr) {
		m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	}
}

void PipelineCache::InitializeDriverCache() {
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return;
	}
	if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (non-Release build)");
		return;
	}
	// (Upstream d58aefde4 drops these checks with its build-independent KytyPC2 signature; ours keeps the revision in
	// the signature unless KYTY_PIPELINE_CACHE_ANY_REVISION, see DriverCacheSignature.)
	const std::string_view git_hash     = KYTY_GIT_HASH;
	const std::string_view git_revision = KYTY_GIT_REVISION;
	if (git_hash == "unknown" || git_revision == "unknown") {
		PipelineCacheLog("Vulkan pipeline cache: disabled (unknown git revision)");
		return;
	}
	if (git_hash.ends_with("-dirty")) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (dirty build)");
		return;
	}

	m_driver_cache_path     = std::filesystem::path("_PipelineCache") / (title_id + ".bin");
	const auto path         = Common::PathToString(m_driver_cache_path);
	const bool cache_exists = Common::File::IsFileExisting(m_driver_cache_path);
	if (cache_exists) {
		PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initializing {}", path);
	}
	std::vector<uint8_t> initial_data;
	// KYTY_TRACE_STARTUP: the load's read, hash and vkCreatePipelineCache times (one STARTUP_PCACHE line).
	const auto load_begin = std::chrono::steady_clock::now();
	double     read_ms    = 0.0;
	double     hash_ms    = 0.0;
	if (cache_exists) {
		Common::File file(m_driver_cache_path, Common::File::Mode::Read);
		const auto   file_size = file.IsInvalid() ? 0 : file.Size();
		const auto   signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
		if (file_size >= signature.size() + sizeof(uint64_t)) {
			std::string cached_signature(signature.size(), '\0');
			uint64_t    payload_hash = 0;
			initial_data.resize(file_size - signature.size() - sizeof(payload_hash));
			uint32_t signature_read = 0;
			uint32_t hash_read      = 0;
			file.Read(cached_signature.data(), static_cast<uint32_t>(cached_signature.size()),
			          &signature_read);
			file.Read(&payload_hash, sizeof(payload_hash), &hash_read);
			const uint64_t payload_read = ReadFileChunked(file, initial_data.data(), initial_data.size());
			file.Close();
			read_ms                    = MillisecondsSince(load_begin);
			const auto payload_matches = [&] {
				const auto hash_begin = std::chrono::steady_clock::now();
				const bool same       = XXH3_64bits(initial_data.data(), initial_data.size()) == payload_hash;
				hash_ms               = MillisecondsSince(hash_begin);
				return same;
			};
			if (signature_read != cached_signature.size() || hash_read != sizeof(payload_hash) ||
			    payload_read != initial_data.size() || cached_signature != signature || !payload_matches()) {
				initial_data.clear();
				PipelineCacheLog(
				    "Vulkan pipeline cache: invalidating {} (driver, emulator, or data mismatch)",
				    path);
			}
		} else {
			file.Close();
			PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid file size)", path);
		}
	}

	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize = initial_data.size();
	create.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();
	const auto create_begin = std::chrono::steady_clock::now();
	auto result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	if (Common::StartupTrace::Enabled()) {
		std::fprintf(stderr, "STARTUP_PCACHE bytes=%zu read_ms=%.0f hash_ms=%.0f create_ms=%.0f total_ms=%.0f\n",
		             initial_data.size(), read_ms, hash_ms, MillisecondsSince(create_begin),
		             MillisecondsSince(load_begin));
		std::fflush(stderr);
	}
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		create.initialDataSize = 0;
		create.pInitialData    = nullptr;
		result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: disabled ({})", vk::to_string(result));
		m_driver_cache = nullptr;
		return;
	}
	if (!initial_data.empty()) {
		m_driver_cache_loaded_bytes = initial_data.size();
		PipelineCacheLog("Vulkan pipeline cache: loaded {} bytes from {}", initial_data.size(),
		                 path);
		m_cache_loaded_bytes = initial_data.size();
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}
}

// The driver cache is otherwise written only at shutdown; save it every two minutes so a
// session that ends abnormally keeps its compiled pipelines.
void PipelineCache::SaveIfDue() {
	if (m_driver_cache == nullptr) {
		return;
	}
	const auto now = std::chrono::steady_clock::now();
	if (m_last_save.time_since_epoch().count() == 0) {
		m_last_save = now;
		return;
	}
	if (now - m_last_save < std::chrono::minutes(2)) {
		return;
	}
	// KYTY_SHADER_PRECOMPILE: while background compiles still add entries, a later save keeps more, so the save waits
	// six minutes instead of two. It still runs: a session whose background compiles never pause (a new area, the first
	// cutscenes) would otherwise save nothing until a normal shutdown, and an abnormal end would lose every pipeline
	// compiled in it.
	if (m_precompiler != nullptr && m_precompiler->Busy()) {
		if (now - m_last_save < std::chrono::minutes(6)) {
			return;
		}
		std::fprintf(stderr, "PIPELINE_CACHE saving while the precompiler is busy\n");
	}
	m_last_save = now;
	// Nothing new to keep: the periodic save rewrote ~1.5 GB on the GPU thread (a 1.5 s hitch).
	const auto precompiled = m_precompiled_since_save.exchange(0);
	if (m_pipelines_since_save == 0 && precompiled == 0) {
		return;
	}
	Save(true);
}

bool PipelineCache::Save(bool background) {
	if (m_driver_cache == nullptr) {
		return false;
	}
	if (m_save_thread.joinable()) {
		m_save_thread.join();
	}
	m_pipelines_since_save = 0;

	// Fetch, hash, write and rename on a background thread for periodic saves (vkGetPipelineCacheData
	// may run while other threads create pipelines with the cache; serializing ~1.7 GB took ~1.5 s).
	// The warm list: the pipelines created before this point are in the data fetched below.
	std::vector<uint64_t> warm_entries;
	if (m_warm != nullptr) warm_entries = m_warm->TakeMarked();
	auto write = [device = m_graphics.device, cache = m_driver_cache, path = m_driver_cache_path,
	              signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties()), warm = m_warm,
	              warm_entries = std::move(warm_entries), cache_saved_bytes = m_cache_saved_bytes]() mutable {
		uint64_t   saved_bytes = 0;
		const auto save_begin  = std::chrono::steady_clock::now(); // KYTY_TRACE_STARTUP
		double     fetch_ms    = 0.0;
		const bool written     = [&] {
			size_t               size = 0;
			vk::Result           result;
			std::vector<uint8_t> payload;
			for (uint32_t attempt = 0; attempt < 3; attempt++) {
				size   = 0;
				result = device.getPipelineCacheData(cache, &size, nullptr);
				if (result != vk::Result::eSuccess || size == 0) {
					break;
				}
				// Headroom: other threads (background compiles, the GPU thread) may add entries in between.
				payload.resize(size + (size_t {128} << 20u));
				size   = payload.size();
				result = device.getPipelineCacheData(cache, &size, payload.data());
				if (result != vk::Result::eIncomplete) {
					break;
				}
			}
			if (result != vk::Result::eSuccess || size == 0) {
				PipelineCacheLog("Vulkan pipeline cache: save failed ({}, {} bytes)",
				                 vk::to_string(result), size);
				return false;
			}
			payload.resize(size);
			fetch_ms                = MillisecondsSince(save_begin);
			auto       prefix       = signature;
			const auto payload_hash = XXH3_64bits(payload.data(), payload.size());
			prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
			if (!Common::File::CreateDirectories(path.parent_path())) {
				PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
				return false;
			}
			auto temp_path = path;
			temp_path += ".tmp";
			Common::File file;
			uint32_t     prefix_written  = 0;
			uint64_t     payload_written = 0;
			if (file.Create(temp_path)) {
				file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
				payload_written = WriteFileChunked(file, payload.data(), payload.size());
			}
			const bool flushed = !file.IsInvalid() && file.Flush();
			file.Close();
			if (prefix_written != prefix.size() || payload_written != payload.size() || !flushed ||
			    !Common::File::RenameFile(temp_path, path)) {
				PipelineCacheLog("Vulkan pipeline cache: failed to write {}",
				                 Common::PathToString(path));
				return false;
			}
			PipelineCacheLog("Vulkan pipeline cache: saved {} bytes to {}", payload.size(),
			                 Common::PathToString(path));
			if (Common::StartupTrace::Enabled()) {
				std::fprintf(stderr, "STARTUP_PCACHE_SAVE bytes=%zu fetch_ms=%.0f total_ms=%.0f\n", payload.size(),
				             fetch_ms, MillisecondsSince(save_begin));
				std::fflush(stderr);
			}
			saved_bytes = payload.size();
			cache_saved_bytes->store(payload.size(), std::memory_order_relaxed);
			return true;
		}();
		if (warm != nullptr) warm->Commit(std::move(warm_entries), written, saved_bytes);
		return written;
	};
	if (background) {
		m_save_thread = std::thread(std::move(write));
		return true;
	}
	// The destructor destroys the cache; periodic saves keep using it.
	return write();
}

// The ObjPrimID write: emulates the hardware ObjPrimID write. A draw with
// VGT_DRAW_PAYLOAD_CNTL.OBJPRIM_ID_EN writes VGT_OBJECT_ID (+ the primitive index when PA_CL_OBJPRIM_ID_CNTL
// .ADD_PIPED_PRIM_ID is set) into an enabled 32-bit colour slot its pixel shader does not export. Yotei's depth-only
// pass draws its 536 culled proxy sets so, into the ID image CS 17444e6a ("Proxy Set Vcull") reads; without it the
// image stays empty.
// The slot of the ObjPrimID write: slot 7 when it qualifies (Yotei binds its ID image to slots 1..7 and clears it
// through slot 7; one attachment is enough, the same image twice is not valid in Vulkan), else the first that does.
// A slot qualifies when CB_TARGET_MASK enables it, a colour target is set (base != 0, COLOR_32) and the pixel shader
// does not export to it. 0xff: none (logged once).
static uint8_t ObjprimIdSlot(const HW::Context& context, const ShaderPixelInputInfo& pixel_info) {
	if ((context.GetDrawPayloadControl() & 0x1u) == 0) return 0xffu;
	uint32_t exported = 0;
	for (uint32_t index = 0; index < std::size(pixel_info.target_output_mode); index++) {
		if (pixel_info.target_output_mode[index] == 0) continue;
		const auto slot = ShaderPixelExportTarget(pixel_info.target_shader_mask, index);
		if (slot < 8u) exported |= 1u << slot;
	}
	const auto usable = [&](uint32_t slot) {
		const auto& rt = context.GetRenderTarget(slot);
		return render_target_mask_slot(context.GetRenderTargetMask(), slot) != 0 && (exported & (1u << slot)) == 0 &&
		       rt.base.addr != 0 && rt.info.format == Prospero::ChannelLayout::k32;
	};
	uint8_t chosen = 0xffu;
	if (usable(7)) {
		chosen = 7;
	} else {
		for (uint32_t slot = 0; slot < 8u && chosen == 0xffu; slot++) {
			if (usable(slot)) chosen = static_cast<uint8_t>(slot);
		}
	}
	static std::atomic_uint logged {0};
	if (logged.fetch_add(1) < 4u) {
		std::string layout;
		for (uint32_t slot = 0; slot < 8u; slot++) {
			const auto& rt = context.GetRenderTarget(slot);
			layout += fmt::format(" [{} base=0x{:x} fmt={} type={} mask={:x}{}]", slot, rt.base.addr,
			                      static_cast<uint32_t>(rt.info.format), static_cast<uint32_t>(rt.info.channel_type),
			                      render_target_mask_slot(context.GetRenderTargetMask(), slot),
			                      (exported & (1u << slot)) != 0 ? " export" : "");
		}
		std::fprintf(stderr, "FIX_OBJPRIM_ID payload=0x%x objprim=0x%x slot=%d%s\n", context.GetDrawPayloadControl(),
		             context.GetObjprimIdControl(), chosen == 0xffu ? -1 : static_cast<int>(chosen), layout.c_str());
	}
	return chosen;
}

// The program inputs of a draw (the part of GetGraphicsPrograms before the program lookups); the cursor before the
// first stage in *push_data_cursor.
static void GraphicsProgramInputs(const GraphicContext& graphics, const HW::VertexShaderInfo& vertex_regs,
                                  const HW::PixelShaderInfo& pixel_regs, const HW::ShaderRegisters& sh,
                                  const HW::Context& context, const HW::UserConfig& user_config,
                                  std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
                                  bool pixel_active, std::array<ShaderVertexInputInfo, 3>& vertex_info,
                                  ShaderPixelInputInfo& pixel_info, const ShaderIndirectDrawInfo& indirect_draw,
                                  std::array<ShaderParams, 3>& vertex_params, ShaderParams& pixel_params,
                                  uint32_t* push_data_cursor);

static void GraphicsProgramInputs(const GraphicContext& graphics, const HW::VertexShaderInfo& vertex_regs,
                                  const HW::PixelShaderInfo& pixel_regs, const HW::ShaderRegisters& sh,
                                  const HW::Context& context, const HW::UserConfig& user_config,
                                  std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
                                  bool pixel_active, std::array<ShaderVertexInputInfo, 3>& vertex_info,
                                  ShaderPixelInputInfo& pixel_info, const ShaderIndirectDrawInfo& indirect_draw,
                                  std::array<ShaderParams, 3>& vertex_params, ShaderParams& pixel_params,
                                  uint32_t* push_data_cursor_out) {
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	if (tess_active) {
		vertex_params = PrepareTessellationPrograms(vertex_regs, context, vertex_info);
	} else {
		vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
		if (vertex_info[0].logical_stage == ShaderType::Vertex) {
			vertex_info[0].indirect_draw = indirect_draw;
		}
	}
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	// Upstream f7226ca7d writes the PS5 object/primitive ID to RT7 from the pixel shader (HasObjectPrimitiveIdOutput,
	// ShaderPixelInputInfo::object_primitive_id, an object-ID push dword). Ours is the ObjPrimID write (objprim_slot,
	// objprim_push_dword below), so upstream's output stays off: object_primitive_id is never set.
	constexpr bool object_primitive_id = false;
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(!graphics.mesh_shader_enabled);
		auto& mesh              = vertex_info[0].mesh;
		mesh.host_subgroup_size = graphics.subgroup_size;
		const auto& limits      = graphics.mesh_shader_properties;
		const auto  logical_threads =
		    mesh.threads_num[0] * mesh.threads_num[1] * mesh.threads_num[2];
		const auto host_threads = ((logical_threads + mesh.wave_size - 1u) / mesh.wave_size) *
		                          std::min(mesh.host_subgroup_size, mesh.wave_size);
		if (host_threads > limits.maxMeshWorkGroupInvocations ||
		    host_threads > limits.maxMeshWorkGroupSize[0] ||
		    mesh.max_vertices > limits.maxMeshOutputVertices ||
		    mesh.max_primitives > limits.maxMeshOutputPrimitives ||
		    mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			EXIT("mesh shader exceeds host limits: threads=%u vertices=%u primitives=%u LDS=%u\n",
			     host_threads, mesh.max_vertices, mesh.max_primitives, mesh.lds_size_dwords);
		}
	}
	if (pixel_active) {
		pixel_params      = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
		pixel_info.object_primitive_id = object_primitive_id;
		// Upstream b4406b046: COLOR_11_11_10 UNORM targets get packed exports with exact rounding (their backing is
		// remapped in TextureGetRenderTargetFormat). Only without KYTY_FIX_RT_11_11_10_FLOAT (default on, M48), which
		// takes these targets as packed unsigned floats instead.
		for (uint32_t export_index = 0; export_index < 8u && !Prospero::Diagnostic111110UnormAsFloat(); export_index++) {
			const auto slot = ShaderPixelExportTarget(pixel_info.target_shader_mask, export_index);
			if (slot >= 8u || pixel_info.target_output_mode[export_index] == 0u) continue;
			const auto& rt = context.GetRenderTarget(slot);
			const auto target_mask = render_target_mask_slot(context.GetRenderTargetMask(), slot);
			if (rt.base.addr == 0 || target_mask == 0u) continue;
			const auto format = Prospero::ResolveRenderTargetFormat(rt.info.format, rt.info.channel_type);
			if (format.buffer_format != Prospero::BufferFormat::k11_11_10UNorm) continue;
			const auto mapping = pixel_info.target_export_mapping[export_index];
			const auto shader_mask = render_target_mask_slot(pixel_info.target_shader_mask, slot);
			// A blend whose colour equation is ONE * src + ZERO * dst writes the source unchanged (the format has no
			// alpha): a plain export.
			const auto& blend_control = context.GetBlendControl(slot);
			const bool  replace       = blend_control.color_srcblend == 1u && blend_control.color_destblend == 0u &&
			                            blend_control.color_comb_fcn == 0u;
			const bool  blend         = blend_control.enable && !rt.info.blend_bypass && !replace;
			if (blend || (mapping.ApplyMask(target_mask) & 7u) != 7u ||
			    (mapping.ApplyMask(shader_mask) & 7u) != 7u ||
			    pixel_info.target_output_mode[export_index] != 9u) {
				EXIT("unsupported packed UNorm render-target export: slot=%u export=%u mode=%u "
				     "target_mask=0x%x shader_mask=0x%x mapping=0x%02x blend=%u src=%u dst=%u fcn=%u\n",
				     slot, export_index, pixel_info.target_output_mode[export_index], target_mask,
				     shader_mask, mapping.packed, static_cast<uint32_t>(blend), blend_control.color_srcblend,
				     blend_control.color_destblend, blend_control.color_comb_fcn);
			}
			pixel_info.target_packed_unorm_mask |= 1u << export_index;
			pixel_info.target_round_mode |= static_cast<uint8_t>(rt.info.round_mode) << export_index;
		}
		if (!mesh_active && !tess_active && Prospero::IsRectList(user_config.GetPrimType())) {
			pixel_info.parameter_mode = ShaderPixelParameterMode::Rectangle;
		} else if (context.GetModeControl().provoking_vtx_last) {
			pixel_info.parameter_mode = ShaderPixelParameterMode::LastVertex;
		} else {
			pixel_info.parameter_mode = ShaderPixelParameterMode::FirstVertex;
		}
		const auto& blend = context.GetBlendControl(0);
		pixel_info.dual_source_blending =
		    blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		    (BlendFactorIsDualSource(blend.color_srcblend) ||
		     BlendFactorIsDualSource(blend.color_destblend) ||
		     (blend.separate_alpha_blend && (BlendFactorIsDualSource(blend.alpha_srcblend) ||
		                                     BlendFactorIsDualSource(blend.alpha_destblend))));
		if (pixel_info.dual_source_blending) {
			// MRT1 supplies the second blend source for target 0.
			pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
			pixel_info.target_export_mapping[1] = pixel_info.target_export_mapping[0];
		} else if (blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		           pixel_info.target_output_mode[0] != 0 && pixel_info.target_output_mode[0] != 7 &&
		           std::all_of(std::begin(pixel_info.target_output_mode) + 1,
		                       std::end(pixel_info.target_output_mode),
		                       [](uint8_t mode) { return mode == 0; })) {
			switch (ClassifyBlendMapping(blend, pixel_info.target_export_mapping[0])) {
				case BlendMappingSupport::SourceAlpha:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlpha;
					break;
				case BlendMappingSupport::SourceAlphaOne:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaOne;
					break;
				case BlendMappingSupport::SourceAlphaZero:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaZero;
					break;
				default: break;
			}
			if (pixel_info.alpha_blend_source != ShaderAlphaBlendSource::None) {
				pixel_info.dual_source_blending     = true;
				pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
				pixel_info.target_export_mapping[1] = {};
			}
		}
	}
	auto&       last_stage   = vertex_info[tess_active ? 2u : 0u];
	const auto& clip_control = context.GetClipControl();
	if (clip_control.clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		const auto& limits   = graphics.GetPhysicalDeviceProperties().limits;
		auto&       clip     = last_stage.clip_space;
		clip.scale[0]        = viewport.xscale;
		clip.scale[1]        = viewport.yscale;
		clip.offset[0]       = viewport.xoffset;
		clip.offset[1]       = viewport.yoffset;
		clip.half_extent[0] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
		clip.half_extent[1] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		clip.enabled = true;
	} else if (clip_control.user_clip_planes != 0) {
		// CLIP_DISABLE also disables user planes. How they combine with shader clip and cull
		// distances is unverified; draw push constants only reach VS/mesh and PS, and the
		// generated RECTLIST stages do not forward clip distances.
		// (Upstream feed075a9 stops the emulator on these combinations; they drew without the user planes before the
		// merge and still do, logged once.)
		if ((last_stage.pa_cl_vs_out_cntl & 0xffffu) != 0 || tess_active ||
		    Prospero::IsRectList(user_config.GetPrimType())) {
			static std::atomic_bool logged {false};
			if (!logged.exchange(true)) {
				std::fprintf(stderr, "UCP: user clip planes 0x%x not applied (vs_out_cntl=0x%x tess=%d rectlist=%d)\n",
				             clip_control.user_clip_planes, last_stage.pa_cl_vs_out_cntl, tess_active ? 1 : 0,
				             Prospero::IsRectList(user_config.GetPrimType()) ? 1 : 0);
			}
		} else {
			last_stage.user_clip_planes = clip_control.user_clip_planes;
		}
	}
	// The draw's own push dwords: the mesh draw parameters, then (upstream feed075a9) the user clip planes from
	// PushData::UserClipPlaneDword, four dwords each; upstream's object-ID dword is not used (object_primitive_id).
	uint32_t push_data_cursor = ShaderRecompiler::IR::PushData::GraphicsDrawDwords(mesh_active, object_primitive_id,
	                                                                               last_stage.user_clip_planes);
	// The pixel stage's data (and its ObjPrimID dword) at the end of the block, the vertex stages' from the draw's dwords
	// up to it (PushData::FromEndCursor): a vertex stage's start does not depend on the pixel stage's data size, so a VS
	// keeps one permutation across the pixel shaders it is drawn with (25.10: 22 -> 1 new VS permutations in 300 s at the
	// meadow; with the pixel data first, each pixel shader's size gave the VS another start and another translation).
	uint32_t end = ShaderRecompiler::IR::PushData::DwordCount;
	if (pixel_active) {
		// The ObjPrimID write: the per-draw VGT_OBJECT_ID takes the last push dword, after the pixel stage's data.
		pixel_info.objprim_slot = ObjprimIdSlot(context, pixel_info);
		if (pixel_info.HasObjprimOutput()) {
			pixel_info.objprim_add_prim_id = (context.GetObjprimIdControl() & 0x2u) != 0;
			end--;
			pixel_info.objprim_push_dword = static_cast<uint8_t>(end);
		}
	}
	*push_data_cursor_out =
	    pixel_active ? ShaderRecompiler::IR::PushData::FromEndCursor(push_data_cursor, end) : push_data_cursor;
}

PipelineCache::GraphicsPrograms PipelineCache::GetGraphicsPrograms(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info,
    const ShaderIndirectDrawInfo& indirect_draw) {
	std::array<ShaderParams, 3> vertex_params;
	ShaderParams                pixel_params;
	uint32_t                    push_data_cursor = 0;
	GraphicsProgramInputs(m_graphics, vertex_regs, pixel_regs, sh, context, user_config, target_export_mapping, pixel_active,
	                      vertex_info, pixel_info, indirect_draw, vertex_params, pixel_params, &push_data_cursor);
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	GraphicsPrograms  result;
	KYTY_PROFILER_BLOCK("GraphicsPrograms::Get");
	if (pixel_active) {
		result.pixel = m_program_cache->Get(pixel_params, pixel_info, push_data_cursor);
	}
	for (uint32_t i = 0; i < (tess_active ? 3u : 1u); i++) {
		result.vertex[i] = m_program_cache->Get(vertex_params[i], vertex_info[i], push_data_cursor);
	}
	return result;
}

ShaderProgram PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                               const HW::ShaderRegisters&   sh,
                                               ShaderComputeInputInfo&      input_info) {
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	std::optional<DispatchSubTimer> prepare_timer(std::in_place, DispatchSub::PrepareProgram); // KYTY_TRACE_DISPATCH_CENSUS
	const auto        params      = PrepareProgram(regs, sh, input_info);
	prepare_timer.reset();
	input_info.lds_storage = input_info.lds_size_dwords * 4u >
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	input_info.compute_variant = 0;
	uint32_t          push_data_cursor = 0;
	return m_program_cache->Get(params, input_info, push_data_cursor);
}

// KYTY_DIAG_TWIN_DISPATCH (diagnostic): as GetComputeProgram, the variant bits `variant_xor` toggled (a key of its own: the
// program and its resource snapshot are not the dispatch's own).
ShaderProgram PipelineCache::GetComputeProgramTwin(const HW::ComputeShaderInfo& regs, const HW::ShaderRegisters& sh,
                                                   ShaderComputeInputInfo& input_info, uint8_t variant_xor) {
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto params             = PrepareProgram(regs, sh, input_info);
	input_info.lds_storage        = input_info.lds_size_dwords * 4u >
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	input_info.compute_variant = variant_xor;
	uint32_t push_data_cursor  = 0;
	return m_program_cache->Get(params, input_info, push_data_cursor);
}

// Every 64k stages, on the resolve thread.
static void ReportResolveStats(uint32_t count) {
	static uint64_t stages = 0;
	const uint64_t  before = stages;
	stages += count + 1u;
	if ((before >> 16u) == (stages >> 16u) || !Common::EnvRead::PerfLines()) return;
	using Ir          = ShaderRecompiler::IR::AheadRefusal;
	auto&       s     = g_resolve_stats;
	const auto  total = [](const std::atomic<uint64_t>& a) -> unsigned long long { return a.load(std::memory_order_relaxed); };
	const auto  by    = [&](Ir reason) { return total(s.refused[static_cast<size_t>(reason)]); };
	std::string text;
	for (size_t i = 0; i < s.missing_by_stage.size(); i++) {
		if (const uint64_t n = total(s.missing_by_stage[i]); n != 0) {
			text += " " + std::to_string(i) + ":" + std::to_string(n);
		}
	}
	for (size_t kind = 0; kind < 2u; kind++) {
		static constexpr const char* causes[] = {"none", "refused", "stale", "no_match"};
		text += kind == 0u ? "] walks_draw[" : "] walks_dispatch[";
		for (size_t cause = 0; cause < std::size(causes); cause++) {
			text += fmt::format(" {}={}/{}ms", causes[cause], total(s.walks[kind][cause]), total(s.walk_ns[kind][cause]) / 1000000u);
		}
	}
	std::fprintf(stderr,
	             "PERF_RESOLVE_STAGES made=%llu no_program=%llu refused[gpu_dirty=%llu backing=%llu cap=%llu flag=%llu "
	             "no_tape=%llu other=%llu] taken=%llu stale=%llu missing=%llu adopted=%llu not_adopted=%llu [%s]\n",
	             total(s.made), total(s.no_program), by(Ir::GpuDirty), by(Ir::Backing), by(Ir::Cap), by(Ir::Flag), by(Ir::NoTape),
	             by(Ir::Other), total(s.taken), total(s.stale), total(s.missing), total(s.adopted), total(s.not_adopted),
	             text.c_str());
}

// As RenderExecutor::DispatchDirect and DispatchIndirect up to GetComputeProgram: the same workgroup counts, the same
// ShaderComputeInputInfo, the same key.
void PipelineCache::ResolveDispatchPrograms(RenderContext& renderer, const HW::ComputeShaderInfo& regs, uint32_t groups_x,
                                            uint32_t groups_y, uint32_t groups_z, uint32_t mode, bool indirect,
                                            bool async_compute, ShaderComputeInputInfo& input_info, ShaderProgram& program,
                                            ResolvedStages& results) {
	constexpr uint32_t DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS = 1u << 5u;
	results.Reset();
	// The real call returns before materializing for empty dispatches and null shaders, and PrepareProgram exits when the
	// shader is not mapped or has more user SGPRs than exist.
	if ((!indirect && (groups_x == 0 || groups_y == 0 || groups_z == 0)) || regs.cs_regs.data_addr == 0 ||
	    regs.cs_regs.user_sgpr > HW::UserSgprInfo::SGPRS_MAX || !ShaderLookupMapped(regs.cs_regs.data_addr, nullptr, nullptr)) {
		return;
	}
	const bool use_thread_dimensions      = !indirect && (mode & DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	input_info                            = {};
	input_info.dispatch_thread_dimensions = use_thread_dimensions;
	input_info.async_compute              = async_compute;
	input_info.workgroup_counts[0]        = indirect ? 0u : groups_x;
	input_info.workgroup_counts[1]        = indirect ? 0u : groups_y;
	input_info.workgroup_counts[2]        = indirect ? 0u : groups_z;
	if (use_thread_dimensions) {
		const uint32_t group_sizes[] = {regs.cs_regs.num_thread_x, regs.cs_regs.num_thread_y, regs.cs_regs.num_thread_z};
		for (uint32_t axis = 0; axis < 3u; ++axis) {
			const auto size    = std::max(group_sizes[axis], 1u);
			const auto threads = input_info.workgroup_counts[axis];
			input_info.workgroup_counts[axis] = threads / size + (threads % size != 0u);
		}
	}
	// As GetComputeProgram; the compute preparation does not read the shader registers.
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	static const HW::ShaderRegisters no_shader_registers {};
	const auto                       params = PrepareProgram(regs, no_shader_registers, input_info);
	input_info.lds_storage     = input_info.lds_size_dwords * 4u > m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	input_info.compute_variant = 0;
	results.adoptable          = !indirect;
	uint32_t push_data_cursor  = 0;
	m_program_cache->Resolve(renderer, params, input_info, push_data_cursor, results, program);
	ReportResolveStats(results.count);
}

// The fetch tables a vertex program's key depends on, logged with the operation's reads.
static bool ReadProgramInputAhead(void* userdata, uint64_t address, void* data, uint64_t size) {
	return ShaderRecompiler::IR::ReadGuestAhead(static_cast<ResolvedStages*>(userdata)->log, address, data, size);
}

// Based on TheCruZ/KytyPS5-GTA daaf45201 (pipelineCache.cpp), GPL-2.0.
void PipelineCache::ResolveGraphicsPrograms(RenderContext& renderer, const HW::VertexShaderInfo& vertex_regs,
                                            const HW::PixelShaderInfo& pixel_regs, const HW::ShaderRegisters& sh,
                                            const HW::Context& context, const HW::UserConfig& user_config,
                                            std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
                                            bool pixel_active, std::array<ShaderVertexInputInfo, 3>& vertex_info,
                                            ShaderPixelInputInfo& pixel_info, const ShaderIndirectDrawInfo& indirect_draw,
                                            GraphicsPrograms& programs, ResolvedStages& results) {
	results.Reset();
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	const auto mapped      = [](uint64_t base) { return base == 0 || ShaderLookupMapped(base, nullptr, nullptr); };
	// PrepareProgram stops the emulator at a shader that is not mapped: the execution thread's call decides (a draw that
	// never gets there may leave one in the registers).
	if (vertex_regs.es_regs.data_addr == 0 || !mapped(vertex_regs.es_regs.data_addr) || !mapped(vertex_regs.gs_regs.data_addr) ||
	    (pixel_active && (pixel_regs.ps_regs.data_addr == 0 || !mapped(pixel_regs.ps_regs.data_addr))) ||
	    (tess_active && (!mapped(vertex_regs.ls_regs.data_addr) || !mapped(vertex_regs.hs_regs.data_addr)))) {
		return;
	}
	std::array<ShaderParams, 3> vertex_params;
	ShaderParams                pixel_params;
	uint32_t                    push_data_cursor = 0;
	{
		const ShaderGuestReaderScope reader(&ReadProgramInputAhead, &results);
		GraphicsProgramInputs(m_graphics, vertex_regs, pixel_regs, sh, context, user_config, target_export_mapping, pixel_active,
		                      vertex_info, pixel_info, indirect_draw, vertex_params, pixel_params, &push_data_cursor);
		if (reader.Refused()) {
			results.log.Clear();
			return;
		}
	}
	results.push_start = push_data_cursor;
	// As in daaf45201: tessellated draws select on the execution thread (their stages still come from here).
	results.adoptable = !tess_active;
	if (pixel_active) {
		m_program_cache->Resolve(renderer, pixel_params, pixel_info, push_data_cursor, results, programs.pixel);
	}
	for (uint32_t i = 0; i < (tess_active ? 3u : 1u); i++) {
		m_program_cache->Resolve(renderer, vertex_params[i], vertex_info[i], push_data_cursor, results, programs.vertex[i]);
	}
	ReportResolveStats(results.count);
}

bool PipelineCache::AdoptResolvedPrograms(ResolvedStages& results) {
	if (!results.adoptable || !ResolvedReadsHold(results)) {
		g_resolve_stats.not_adopted.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	const bool adopted = m_program_cache->Adopt(results);
	(adopted ? g_resolve_stats.adopted : g_resolve_stats.not_adopted).fetch_add(1, std::memory_order_relaxed);
	return adopted;
}

bool PipelineStaticParameters::operator==(const PipelineStaticParameters& other) const noexcept {
	return std::memcmp(this, &other, sizeof(*this)) == 0;
}

PipelineCache::GraphicsPipelineKey PipelineCache::MakeGraphicsPipelineKey(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs) {
	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;

	EXIT_IF(colors.size() > RENDER_COLOR_ATTACHMENTS_MAX);
	EXIT_IF(!vertex_program);
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(ps_active && !pixel_program);
	const auto color_count = static_cast<uint32_t>(colors.size());

	auto&             ctx = command.GetRegisters();

	const HW::ModeControl& mc = ctx.GetModeControl();

	const auto ps_id = ps_active ? pixel_program.id : 0;

	GraphicsPipelineKey key {};
	for (uint32_t i = 0; i < programs.vertex.size(); i++) {
		key.vertex_shader_ids[i] = programs.vertex[i].id;
	}
	key.ps_shader_id            = ps_id;
	auto& static_params         = key.static_params;
	auto& rendering             = key.rendering;
	rendering.color_count       = 0;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		const auto slot = colors[i].target_slot;
		EXIT_IF(slot >= RENDER_COLOR_ATTACHMENTS_MAX);
		rendering.color_count = std::max(rendering.color_count, slot + 1);
		EXIT_IF(!colors[i].image_id || colors[i].desc.view_info.format == vk::Format::eUndefined);
		static_params.color_mask[slot] = colors[i].export_mapping.ApplyMask(
		    render_target_mask_slot(ctx.GetRenderTargetMask(), colors[i].target_slot));
		// Upstream b4406b046: a packed 11-11-10 UNorm export writes one 32-bit channel (not with
		// KYTY_FIX_RT_11_11_10_FLOAT, see GraphicsProgramInputs).
		if (colors[i].desc.info.guest_format == Prospero::BufferFormat::k11_11_10UNorm &&
		    !Prospero::Diagnostic111110UnormAsFloat()) {
			static_params.color_mask[slot] = 1u;
		}
		rendering.color_formats[slot] = colors[i].desc.view_info.format;
		if (attachment_samples == 0) {
			attachment_samples = colors[i].desc.info.samples;
		} else if (attachment_samples != colors[i].desc.info.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, colors[i].desc.info.samples);
		}
		const auto& rt                        = ctx.GetRenderTarget(colors[i].target_slot);
		const auto& bc                        = ctx.GetBlendControl(colors[i].target_slot);
		auto alpha_source = ShaderAlphaBlendSource::None;
		if (slot == 0 && ps_input_info != nullptr) {
			alpha_source = ps_input_info->alpha_blend_source;
		}
		static_params.blend_enable[slot] = bc.enable && !rt.info.blend_bypass;
		if (static_params.blend_enable[slot] && alpha_source == ShaderAlphaBlendSource::None &&
		    ClassifyBlendMapping(bc, colors[i].export_mapping) != BlendMappingSupport::Direct) {
			static_params.blend_enable[slot] = false;
			static std::atomic_bool warned = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Warning: blending disabled for unsupported color mapping "
				    "(slot={} mapping=0x{:02x} color={}/{} alpha={}/{} separate={}).\n",
				    slot, colors[i].export_mapping.packed, bc.color_srcblend, bc.color_destblend,
				    bc.alpha_srcblend, bc.alpha_destblend, bc.separate_alpha_blend ? 1 : 0));
			}
		}
		if (static_params.blend_enable[slot]) {
			auto blend = bc;
			switch (alpha_source) {
				case ShaderAlphaBlendSource::SourceAlpha:
					blend.color_srcblend  = RemapSourceAlphaFactor(blend.color_srcblend);
					blend.color_destblend = RemapSourceAlphaFactor(blend.color_destblend);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::SourceAlphaOne:
				case ShaderAlphaBlendSource::SourceAlphaZero:
					// The second source carries the mapped source factor; its alpha stays logical Sa.
					blend.color_srcblend = static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Color);
					blend.color_destblend =
					    static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::None: break;
			}
			static_params.color_srcblend[slot]       = blend.color_srcblend;
			static_params.color_comb_fcn[slot]       = blend.color_comb_fcn;
			static_params.color_destblend[slot]      = blend.color_destblend;
			static_params.separate_alpha_blend[slot] = blend.separate_alpha_blend;
			if (blend.separate_alpha_blend) {
				static_params.alpha_srcblend[slot]  = blend.alpha_srcblend;
				static_params.alpha_comb_fcn[slot]  = blend.alpha_comb_fcn;
				static_params.alpha_destblend[slot] = blend.alpha_destblend;
			}
		}
	}
	const bool with_depth =
	    depth.desc.view_info.format != vk::Format::eUndefined && static_cast<bool>(depth.image_id);
	if (with_depth) {
		const auto aspects       = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		rendering.depth_format   = aspects & vk::ImageAspectFlagBits::eDepth
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		rendering.stencil_format = aspects & vk::ImageAspectFlagBits::eStencil
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		if (attachment_samples == 0) {
			attachment_samples = depth.desc.info.samples;
		} else if (attachment_samples != depth.desc.info.samples) {
			EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
			     depth.desc.info.samples);
		}
	}
	if (color_count == 0 && !with_depth) {
		attachment_samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
		EXIT_IF(!static_cast<bool>(
		    m_graphics.GetPhysicalDeviceProperties().limits.framebufferNoAttachmentsSampleCounts &
		    vulkan_sample_count(attachment_samples)));
	}
	EXIT_IF(attachment_samples == 0 ||
	        vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {});

	if (ps_active && depth.depth_test_enable && ps_input_info->ps_execute_on_noop) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("Pipeline: temporary: accepting EXEC_ON_NOOP with depth test enabled\n");
		}
	}

	const auto& clip_control               = ctx.GetClipControl();
	static_params.negative_one_to_one      = !clip_control.dx_clip_space;
	static_params.depth_clip_enable        = clip_control.IsZClipEnabled();
	static_params.topology                 = topology;
	static_params.primitive_restart_enable = primitive_restart_enable;
	static_params.samples                  = attachment_samples;
	static_params.sample_shading_enable =
	    ps_active && attachment_samples > 1 && ps_input_info->ps_sample_shading;
	if (static_params.sample_shading_enable && !m_graphics.sample_rate_shading_enabled) {
		EXIT("Pipeline: sample-rate shading is required but unsupported by the host\n");
	}
	static_params.alpha_to_coverage_enable = ps_active && ctx.GetShaderRegisters().AlphaToMask();
	const bool rect_list = Prospero::IsRectList(command.GetUserConfig().GetPrimType());
	static_params.cull_back  = !rect_list && mc.cull_back;
	static_params.cull_front = !rect_list && mc.cull_front;
	static_params.face       = mc.face;
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	static_params.polygon_mode =
	    ResolvePolygonMode(mc, static_params.cull_front, static_params.cull_back);

	if (vs_input_info.stage.program->stage != ShaderType::Mesh) {
		EXIT_IF(vs_input_info.buffers_num < 0 ||
		        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		        vs_input_info.resources_num < 0 ||
		        vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX);
		key.vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
		key.vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
		for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
			const auto& buffer = vs_input_info.buffers[binding];
			key.vertex_input.bindings[binding] = {.stride   = buffer.stride,
			                                      .instance = buffer.fetch_index != 0};
		}
		for (int attribute = 0; attribute < vs_input_info.resources_num; attribute++) {
			const auto binding = vs_input_info.resources_dst[attribute].buffer_index;
			EXIT_IF(binding < 0 || binding >= vs_input_info.buffers_num);
			key.vertex_input.attributes[attribute] = {
			    .offset = static_cast<uint32_t>(vs_input_info.resources[attribute].Base48() -
			                                    vs_input_info.buffers[binding].addr),
			    .binding = static_cast<uint8_t>(binding),
			};
		}
	}

	return key;
}

void PipelineCache::PipelineHousekeeping() {
	StartBackgroundCompiles();
	SaveIfDue();
}

void PipelineCache::RecreateGraphicsPipeline(Pipeline& pipeline, const GraphicsPipelineKey& key,
                                             std::span<const ShaderVertexInputInfo> vertex_info,
                                             const ShaderPixelInputInfo* ps_input_info, const GraphicsPrograms& programs) {
	// EvictIdlePipelines gave its objects back: created again from the same inputs.
	CreatePipelineInternal(m_graphics, pipeline, key.rendering, key.vertex_input, vertex_info, ps_input_info, programs,
	                       key.static_params, m_driver_cache);
	EXIT_NOT_IMPLEMENTED(pipeline.pipeline == nullptr || pipeline.pipeline_layout == nullptr);
	m_recreated++;
}

PipelineCache::Pipeline& PipelineCache::GetGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs) {
	PipelineHousekeeping();
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);
	GraphicsPipelineKey key = MakeGraphicsPipelineKey(colors, depth, vertex_info, command, ps_input_info, topology,
	                                                  primitive_restart_enable, programs);
	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	const bool  ps_active      = ps_input_info != nullptr;
	const auto  vs_id          = vertex_program.id;
	const auto  ps_id          = ps_active ? pixel_program.id : 0;
	const auto& static_params  = key.static_params;
	const auto& rendering      = key.rendering;

	if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
		auto& found     = *iter->second;
		found.last_used = static_cast<uint32_t>(DebugPresentCount().load());
		if (found.pipeline == nullptr) RecreateGraphicsPipeline(found, key, vertex_info, ps_input_info, programs);
		return found;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(vs_input_info);
		if (ps_active) {
			ShaderDbgDumpInputInfo(*ps_input_info);
		}
		LOGF("PipelineTrace: shader modules VS=%" PRIu64 " module=%p PS=%" PRIu64 " module=%p\n",
		     vs_id, static_cast<void*>(vertex_program.module), ps_id,
		     static_cast<void*>(pixel_program.module));
	}

	auto cached = std::make_unique<Pipeline>();
	// KYTY_SHADER_PRECOMPILE: a background compile of this pipeline is waited for (the driver then has it), one not
	// started yet is dropped; a pipeline made from recorded stages is recorded once it exists.
	PrecompileGraphicsRecord precompile;
	const bool precompile_record = m_precompiler != nullptr && ShaderPrecompiler::GraphicsEnabled() &&
	                               m_program_cache->PrecompileStageKeys(programs, ps_active, precompile);
	if (precompile_record) {
		precompile.rendering     = rendering;
		precompile.vertex_input  = key.vertex_input;
		precompile.static_params = static_params;
		precompile.revision      = ShaderPrecompiler::Revision();
		m_precompiler->BeforeCompile(precompile.Key());
	}
	LogPipelineTrace("CreatePipelineInternal begin", vs_id, ps_id);
	{
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::PipelinesCreated,
		                                         Common::DebugCounters::Counter::PipelineCreateNs,
		                                         Common::DebugCounters::Peak::PipelineCreateNs);
		CreatePipelineInternal(m_graphics, *cached, rendering, key.vertex_input, vertex_info,
		                       ps_input_info, programs, static_params, m_driver_cache);
	}
	m_pipelines_since_save++;
	LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);
	if (precompile_record) {
		m_precompiler->RecordPipeline(precompile);
		if (m_warm != nullptr) m_warm->Mark(precompile.Key(), true);
	}

	cached->last_used     = static_cast<uint32_t>(DebugPresentCount().load());
	auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
	EXIT_IF(!inserted);
	// (Upstream 959bee479 checkpoints the driver cache here; ours saves it with SaveIfDue, m_pipelines_since_save.)

	return *iter->second;
}

namespace {
// KYTY_PERF_TABLE_ASYNC (the cache's thread): draws given a ready pipeline, draws without one (fallback: queued now,
// still compiling, or refused), creations queued, taken and refused, and the workers' time on the taken ones.
struct TableAsyncStats {
	uint64_t draws = 0, fallback = 0, queued = 0, ready = 0, failed = 0;
	double   worker_ms = 0.0;
	uint64_t calls     = 0; // (TableAsyncReport's)
	std::chrono::steady_clock::time_point window {};
} g_table_async;

uint32_t TableAsyncThreads() {
	static const uint32_t threads = [] {
		const char* value = std::getenv("KYTY_PERF_TABLE_ASYNC_THREADS");
		return value == nullptr ? 4u : static_cast<uint32_t>(std::clamp(std::atoi(value), 1, 16));
	}();
	return threads;
}

void TableAsyncReport(TableAsyncStats& s, size_t pending, size_t queued) {
	if (!PerfEnabled()) return;
	// The clock is read at every 128th call only (a QueryPerformanceCounter call per table-mode draw was ~0.14 ms a frame
	// of the GPU thread with KYTY_TRACE_PERF=1 at the torii gate, run 20261021-091655).
	if ((++s.calls & 127u) != 0u && s.window != std::chrono::steady_clock::time_point {}) return;
	const auto now = std::chrono::steady_clock::now();
	if (s.window == std::chrono::steady_clock::time_point {}) {
		s.window = now;
		return;
	}
	if (now - s.window < std::chrono::seconds(5)) return;
	s.window = now;
	std::fprintf(stderr,
	             "PERF table_async draws=%llu fallback=%llu queued=%llu ready=%llu failed=%llu pending=%zu waiting=%zu "
	             "worker_ms=%.0f\n",
	             static_cast<unsigned long long>(s.draws), static_cast<unsigned long long>(s.fallback),
	             static_cast<unsigned long long>(s.queued), static_cast<unsigned long long>(s.ready),
	             static_cast<unsigned long long>(s.failed), pending, queued, s.worker_ms);
}
} // namespace

bool PipelineCache::TableAsyncEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PERF_TABLE_ASYNC");
		const bool  on    = value == nullptr || value[0] != '0'; // default on (=0: created on the calling thread)
		if (!on) std::fprintf(stderr, "PERF_TABLE_ASYNC off: graphics pipelines are created by the draw\n");
		return on;
	}();
	return enabled;
}

PipelineCache::Pipeline* PipelineCache::GetGraphicsPipelineAsync(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs, bool* failed) {
	PipelineHousekeeping();
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(GfxAsync)", profiler::colors::DeepOrangeA200);
	auto& stats = g_table_async;
	TableAsyncReport(stats, m_table_pending.size(), m_table_workers != nullptr ? m_table_workers->Queued() : 0u);
	GraphicsPipelineKey key = MakeGraphicsPipelineKey(colors, depth, vertex_info, command, ps_input_info, topology,
	                                                  primitive_restart_enable, programs);
	if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
		auto& found     = *iter->second;
		found.last_used = static_cast<uint32_t>(DebugPresentCount().load());
		if (found.pipeline == nullptr) RecreateGraphicsPipeline(found, key, vertex_info, ps_input_info, programs);
		stats.draws++;
		return &found;
	}
	if (auto pending = m_table_pending.find(key); pending != m_table_pending.end()) {
		auto& job = *pending->second;
		if (!job.done.load(std::memory_order_acquire) || job.failed) {
			if (job.failed && !job.failed_counted) {
				job.failed_counted = true;
				stats.failed++;
			}
			if (failed != nullptr) *failed = job.failed;
			stats.fallback++;
			return nullptr;
		}
		// Finished: taken as GetGraphicsPipeline keeps the pipeline it creates (recorded by its worker already).
		m_pipelines_since_save++;
		stats.ready++;
		stats.worker_ms += job.ms;
		auto finished       = std::move(job.pipeline);
		finished->last_used = static_cast<uint32_t>(DebugPresentCount().load());
		m_table_pending.erase(pending);
		auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(finished));
		EXIT_IF(!inserted);
		stats.draws++;
		return iter->second.get();
	}
	stats.fallback++;
	if (m_table_workers_stopped) return nullptr; // (shutting down: nothing more is queued)
	if (m_table_workers == nullptr) {
		m_table_workers = std::make_unique<TableCompileWorkers>(TableAsyncThreads());
		std::fprintf(stderr, "PERF_TABLE_ASYNC workers=%u\n", TableAsyncThreads());
	}
	const bool ps_active = ps_input_info != nullptr;
	auto       job       = std::make_shared<TableCompileJob>();
	job->rendering       = key.rendering;
	job->vertex_input    = key.vertex_input;
	job->static_params   = key.static_params;
	job->vertex_count    = static_cast<uint32_t>(vertex_info.size());
	EXIT_IF(job->vertex_count == 0 || job->vertex_count > job->vertex_info.size());
	for (uint32_t i = 0; i < job->vertex_count; i++) {
		job->vertex_info[i]                     = vertex_info[i];
		job->vertex_info[i].stage.resources     = &job->no_resources;
		job->vertex_info[i].stage.stagef_record = nullptr;
	}
	job->ps_active = ps_active;
	if (ps_active) {
		job->ps_input_info                     = *ps_input_info;
		job->ps_input_info.stage.resources     = &job->no_resources;
		job->ps_input_info.stage.stagef_record = nullptr;
	}
	job->programs = programs;
	// KYTY_SHADER_PRECOMPILE: as GetGraphicsPipeline (the worker waits for a background compile of it).
	job->precompile_record = m_precompiler != nullptr && ShaderPrecompiler::GraphicsEnabled() &&
	                         m_program_cache->PrecompileStageKeys(programs, ps_active, job->precompile);
	if (job->precompile_record) {
		job->precompile.rendering     = key.rendering;
		job->precompile.vertex_input  = key.vertex_input;
		job->precompile.static_params = key.static_params;
		job->precompile.revision      = ShaderPrecompiler::Revision();
	}
	m_table_pending.emplace(std::move(key), job);
	stats.queued++;
	m_table_workers->Push([this, job] {
		if (job->precompile_record) (void)m_precompiler->BeforeCompile(job->precompile.Key());
		const auto begin = std::chrono::steady_clock::now();
		CreatePipelineInternal(m_graphics, *job->pipeline, job->rendering, job->vertex_input,
		                       std::span<const ShaderVertexInputInfo>(job->vertex_info.data(), job->vertex_count),
		                       job->ps_active ? &job->ps_input_info : nullptr, job->programs, job->static_params,
		                       m_driver_cache);
		auto& made = *job->pipeline;
		if (made.pipeline == nullptr || made.pipeline_layout == nullptr) {
			// A background creation gives up where the GPU thread would exit (CreatePipelineInternal): what it left is
			// destroyed, and the draws of this key stay on today's path.
			if (made.pipeline != nullptr) m_graphics.device.destroyPipeline(made.pipeline, nullptr);
			if (made.pipeline_layout != nullptr) m_graphics.device.destroyPipelineLayout(made.pipeline_layout, nullptr);
			if (made.descriptor_set_layout != nullptr) {
				m_graphics.device.destroyDescriptorSetLayout(made.descriptor_set_layout, nullptr);
			}
			made        = Pipeline {};
			job->failed = true;
			std::fprintf(stderr, "PERF_TABLE_ASYNC refused: vs=0x%016llx ps=0x%016llx (its draws take today's path)\n",
			             static_cast<unsigned long long>(job->vertex_info[0].stage.program->shader_hash),
			             static_cast<unsigned long long>(job->ps_active ? job->ps_input_info.stage.program->shader_hash
			                                                            : 0ull));
		} else if (job->precompile_record) {
			// Recorded when made, not when a draw takes it: a pipeline no later draw took (219 in a 40-minute session)
			// left its recorded stages without a pipeline record, so the next start never compiled them.
			m_precompiler->RecordPipeline(job->precompile);
			if (m_warm != nullptr) m_warm->Mark(job->precompile.Key(), true);
		}
		job->ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
		job->done.store(true, std::memory_order_release);
	});
	return nullptr;
}

PipelineCache::Pipeline&
PipelineCache::GetComputePipeline(const ShaderComputeInputInfo& input_info,
                                  const ShaderProgram&          compute_program) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);
	PipelineHousekeeping();

	EXIT_IF(!compute_program);

	if (auto iter = m_compute_pipelines.find(compute_program.id);
	    iter != m_compute_pipelines.end()) {
		auto& found     = *iter->second;
		found.last_used = static_cast<uint32_t>(DebugPresentCount().load());
		if (found.pipeline == nullptr) RecreateComputePipeline(found, input_info, compute_program);
		return found;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(input_info);
	}

	auto cached = std::make_unique<Pipeline>();
	{
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::PipelinesCreated,
		                                         Common::DebugCounters::Counter::PipelineCreateNs,
		                                         Common::DebugCounters::Peak::PipelineCreateNs);
		CreatePipelineInternal(m_graphics, *cached, input_info, compute_program.module, m_driver_cache);
	}
	m_pipelines_since_save++;

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);
	// The warm list: a recorded permutation's pipeline is in the driver cache now.
	if (m_warm != nullptr) {
		if (const auto found = m_program_cache->precompile_stages.find(compute_program.id);
		    found != m_program_cache->precompile_stages.end()) {
			m_warm->Mark(found->second.key, false);
		}
	}

	cached->last_used     = static_cast<uint32_t>(DebugPresentCount().load());
	auto [iter, inserted] = m_compute_pipelines.emplace(compute_program.id, std::move(cached));
	EXIT_IF(!inserted);
	// (Upstream 959bee479 checkpoints the driver cache here; ours saves it with SaveIfDue, m_pipelines_since_save.)

	return *iter->second;
}

void PipelineCache::RecreateComputePipeline(Pipeline& pipeline, const ShaderComputeInputInfo& input_info,
                                            const ShaderProgram& compute_program) {
	// EvictIdlePipelines gave its objects back: created again from the same inputs.
	CreatePipelineInternal(m_graphics, pipeline, input_info, compute_program.module, m_driver_cache);
	EXIT_NOT_IMPLEMENTED(pipeline.pipeline == nullptr || pipeline.pipeline_layout == nullptr);
	m_recreated++;
}

namespace {
// About 5 minutes at 30 fps: an area the player left. A pipeline created again costs a driver-cache lookup (~1-5 ms).
// KYTY_TEST_PIPELINE_IDLE_FRAMES=<n> (test only): another idle span, to run the re-creation path often in a short run.
uint32_t PipelineIdleFrames() {
	static const uint32_t frames = [] {
		const char* value = std::getenv("KYTY_TEST_PIPELINE_IDLE_FRAMES");
		return value != nullptr ? static_cast<uint32_t>(std::max(1, std::atoi(value))) : 9000u;
	}();
	return frames;
}
// The sweep walks both maps at most every this many frames (or the idle span, if shorter).
constexpr uint32_t PipelineSweepFrames = 300;
} // namespace

void PipelineCache::EvictIdlePipelines(CommandScheduler& scheduler) {
	const auto now = static_cast<uint32_t>(DebugPresentCount().load());
	if (now - m_last_eviction < std::min(PipelineSweepFrames, PipelineIdleFrames())) return;
	m_last_eviction = now;
	struct Retired {
		vk::Pipeline                 pipeline;
		vk::PipelineLayout           layout;
		vk::DescriptorSetLayout      set_layout;
		vk::DescriptorUpdateTemplate push_template;
	};
	std::vector<Retired> retired;
	size_t               live  = 0;
	const auto           sweep = [&](auto& map) {
        for (auto& [key, pointer]: map) {
            auto& p = *pointer;
            if (p.pipeline == nullptr) continue;
            if (now - p.last_used < PipelineIdleFrames()) {
                live++;
                continue;
            }
            retired.push_back({p.pipeline, p.pipeline_layout, p.descriptor_set_layout, p.push_template});
            const auto last_used = p.last_used;
            p                    = Pipeline {};
            p.last_used          = last_used;
        }
	};
	sweep(m_graphics_pipelines);
	sweep(m_compute_pipelines);
	if (retired.empty()) return;
	m_evicted += retired.size();
	if (PerfEnabled()) {
		std::fprintf(stderr, "PERF pipeline_evict frame=%u evicted=%zu total_evicted=%llu recreated=%llu live=%zu\n", now,
		             retired.size(), static_cast<unsigned long long>(m_evicted),
		             static_cast<unsigned long long>(m_recreated), live);
	}
	// Recorded commands of the open tick may still name them (recordPackets.h): destroyed once the GPU finished it.
	scheduler.DeferOperation([device = m_graphics.device, retired = std::move(retired)] {
		for (const auto& r: retired) {
			device.destroyPipeline(r.pipeline, nullptr);
			if (r.push_template != nullptr) device.destroyDescriptorUpdateTemplate(r.push_template, nullptr);
			if (r.layout != nullptr) device.destroyPipelineLayout(r.layout, nullptr);
			if (r.set_layout != nullptr) device.destroyDescriptorSetLayout(r.set_layout, nullptr);
		}
	});
}
} // namespace Libs::Graphics
