#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERPRECOMPILE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERPRECOMPILE_H_

#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;

namespace ShaderRecompiler {
struct CompileOptions;
} // namespace ShaderRecompiler

// KYTY_SHADER_PRECOMPILE=1: every compute permutation the GPU thread compiles is recorded once in
// _PipelineCache/<title>.cs.journal with every input of its translation and compile (shader code, user data, stage
// inputs, push-data start, resource specialization). Once the renderer is complete, the recorded permutations are
// translated and compiled again on background threads (KYTY_SHADER_PRECOMPILE_THREADS, default 4, below normal
// priority) and their compute pipelines created through the Vulkan pipeline cache, then destroyed: the driver has
// the pipeline before the game asks. Emulator changes alter the SPIR-V of every shader, and NVIDIA takes up to ~45 s
// per large wave64 compute pipeline; without this every one of them compiles again on the GPU thread in play.
// Nothing the GPU thread renders changes: it still translates, compiles and creates its own pipeline (which then
// hits the driver cache). A job running for the permutation the GPU thread is about to compile is waited for (its
// worker raised to normal priority); a queued one is dropped and compiled by the GPU thread as before.
// A record that was compiling when a run ended without a clean quit is a suspect: skipped for the rest of that
// emulator build (.suspects), so a record that crashes the recompiler cannot stop every start.
// Graphics pipelines likewise (KYTY_SHADER_PRECOMPILE_GRAPHICS=0 leaves them out): every stage permutation a draw
// compiles (vertex, mesh, LS/HS/DS, pixel) is recorded in <title>.gfx.journal, and every graphics pipeline the GPU
// thread creates from recorded stages as their record keys plus its key's state (attachment formats, vertex input,
// static parameters). A worker compiles each stage once (the jobs using it share the module) and creates the
// pipeline with the GPU thread's CreatePipelineInternal. The journal's header names the GPU: the records of another
// GPU are dropped. A pipeline this device cannot make (a missing feature, a value the pipeline code exits on) is
// skipped. A stage being compiled is in flight too: one that crashes is a suspect, and so is every pipeline using it.
struct PrecompileStageRecord {
	ShaderType                                   stage       = ShaderType::Compute;
	uint64_t                                     shader_hash = 0;
	std::vector<uint32_t>                        user_data;
	std::vector<uint32_t>                        code;
	std::vector<uint32_t>                        back_code; // the GS back half of a merged mesh program, else empty
	// The input of `stage` (vertex: the vertex, mesh, LS, HS and DS stages); their `stage` members are not recorded.
	ShaderVertexInputInfo                        vertex;
	ShaderPixelInputInfo                         pixel;
	ShaderComputeInputInfo                       compute;
	// The record's table-mode bits (KYTY_PERF_TABLE_BUFFERS / project D's table mode, removed; the journal keeps
	// their place): not 0, the record was made for a program no lookup asks for any more and is not prepared.
	uint8_t                                      table_mode = 0;
	std::vector<uint32_t>                        static_key; // BuildStageStaticKey(input), part of Key()
	uint32_t                                     push_data_start = 0;
	ShaderRecompiler::IR::ResourceSpecialization specialization;
	uint64_t                                     spirv_hash = 0; // the GPU thread's module (XXH3 of its words)
	uint64_t                                     revision   = 0; // emulator build that recorded it

	// Identity of the permutation: the program cache key (stage, hash, user data count, code size, static key)
	// plus push-data start and specialization.
	[[nodiscard]] uint64_t Key() const;
};

// A graphics pipeline the GPU thread created (PipelineCache::GetGraphicsPipeline): its stages' record keys and
// the state of its pipeline key.
struct PrecompileGraphicsRecord {
	// vertex[0]: the vertex, mesh or LS stage; vertex[1], vertex[2]: the HS and DS stages of a tessellated draw
	// (else 0). pixel: 0 without a pixel stage.
	std::array<uint64_t, 3>  vertex {};
	uint64_t                 pixel = 0;
	PipelineRenderingState   rendering;
	PipelineVertexInputState vertex_input;
	PipelineStaticParameters static_params;
	uint64_t                 revision = 0; // emulator build that recorded it

	[[nodiscard]] uint32_t VertexStageCount() const { return vertex[1] != 0 ? 3u : 1u; }
	// Identity of the pipeline: its stage keys and state.
	[[nodiscard]] uint64_t Key() const;
};

// A compute key leaves the stage out (the compute journal's records were keyed so).
[[nodiscard]] uint64_t PrecompileKey(ShaderType stage, uint64_t shader_hash, uint32_t user_data_count,
                                     uint32_t code_words, std::span<const uint32_t> static_key,
                                     uint32_t push_data_start,
                                     const ShaderRecompiler::IR::ResourceSpecialization& specialization);

// The shader hash of this code as shader.cpp makes it (XXH3 of the code; a merged GS front half: XXH3 of the two
// halves' hashes). A permutation is recorded only when its code still has the hash it was looked up by.
[[nodiscard]] uint64_t PrecompileCodeHash(std::span<const uint32_t> code, std::span<const uint32_t> back_code);

// --export-hints <file> (PipelineCache, with --precompile-only): the shader journals of this mode written as a hints file
// without the game's code (shaderPrecompile.cpp, before PrecompileHintsExport's body: the format). False when there is
// no journal or the file cannot be written; `summary` holds the counts or the reason.
bool PrecompileHintsExport(const std::filesystem::path& journal, const std::filesystem::path& graphics_journal,
                           const std::filesystem::path& out, std::string& summary);
// --import-hints <file>: the code of each hint is looked for in the files of `game_dir` (the game folder of --game); the
// records found are added to the journals (this device's), which the precompiler then loads. False when the file is no
// hints file, is for another emulator layout, another shader mode or another game.
bool PrecompileHintsImport(const GraphicContext& graphics, const std::filesystem::path& hints,
                           const std::filesystem::path& game_dir, const std::filesystem::path& journal,
                           const std::filesystem::path& graphics_journal, std::string& summary);

// The user-data base and wave size of a stage's recompiler options, from options.stage and options.input_info, as
// the program cache sets them (ProgramCache::Get); the background compiles use the same.
void SetStageCompileOptions(ShaderRecompiler::CompileOptions& options);

// True on a ShaderPrecompiler worker (its pipeline logs are labelled as background compiles).
[[nodiscard]] bool IsBackgroundCompileThread();
// Makes the calling thread one (PipelineCache's table compile workers, KYTY_PERF_TABLE_ASYNC): named `name`, below
// normal priority, its pipeline creations labelled and given up on as a ShaderPrecompiler worker's (CreatePipelineInternal
// leaves the pipeline null where the GPU thread would exit).
void EnterBackgroundCompileThread(const char* name);

// The shader reuse (needs KYTY_SHADER_PRECOMPILE=1, off with --precompile-only): the background compiles keep what they
// translate and compile instead of throwing it away (once a start re-translated ~3,600 recorded permutations
// on the workers, then the GPU thread translated every one of them again when the game first used it, ~1,900 in
// the first in-game frames). For each recorded permutation a worker keeps the SPIR-V and the compiled program info
// (CompileProgram's output) and the program's resource plan (ExtractResourcePlan of its translation). The GPU
// thread's ProgramCache::Get takes them instead of translating and compiling when what it is about to compile is
// the same, word for word, as what the worker compiled (PrecompileReuseIdentity plus push-data start and
// specialization); everything after CompileProgram (validation, dumps, KYTY_DIAG_SPIRV_REPLACE, the module) runs
// as before. Anything else, and a permutation whose worker has not finished, is translated by the GPU thread: it never
// waits for a translation (a running compute job is still waited for, as before).
// Taken results are freed; the store is bounded by KYTY_PERF_SHADER_REUSE_MB (default 2048 MiB): beyond it the
// workers keep nothing more. Warm-list jobs (pipeline already in the driver cache) are
// translated for the store but make no pipeline.
// Counters: "PERF shader_reuse" (KYTY_TRACE_PERF) and reuse= in the STARTUP lines (KYTY_TRACE_STARTUP).
[[nodiscard]] bool ShaderReuseEnabled();

// What decides the translation of a program, as the GPU thread compiles it (spans into its live state) and as a
// background compile of a record compiled it. The recompiler reads of the stage input only what its static key
// (BuildStageStaticKey) holds, plus a compute input's lds_storage; of the user data only the count; the rest is the
// code (and a merged stage's back half), the stage and the shader hash. Process-wide switches are the same on both
// threads (one process, read once).
struct PrecompileReuseIdentity {
	ShaderType                stage           = ShaderType::Unknown;
	uint64_t                  shader_hash     = 0;
	uint32_t                  user_data_count = 0;
	std::span<const uint32_t> code;
	std::span<const uint32_t> back_code;
	std::span<const uint32_t> static_key;
	bool                      lds_storage = false; // compute only
};

// A stored compile, taken by the GPU thread: CompileProgram's SPIR-V and program info, and what the worker spent.
struct PrecompileReused {
	std::vector<uint32_t>                    spirv;
	ShaderRecompiler::IR::CompiledShaderInfo info;
	double                                   translate_ms = 0.0;
	double                                   compile_ms   = 0.0;
};

// The shader reuse (GPU thread): a stored result was used; saved_ms is what the worker spent on what the GPU
// thread skipped (the estimate of the time saved).
void ShaderReuseNoteUsed(double saved_ms);

// The warm list: the recorded compute permutations
// and graphics pipelines whose driver pipelines the saved driver cache (_PipelineCache/<title>.bin) already holds
// for this emulator build and these values of the settings that change what a record makes (Common::ShaderEnv::Names,
// common/shaderEnv.h; no other KYTY_ variable), in _PipelineCache/<title>.warm. A start leaves their background
// compiles out: they would only translate the shaders again to hit the driver cache (before: every start
// re-translated ~670 compute permutations, minutes of 4 cores while the game ran). What the GPU thread does is
// unchanged. An entry is kept only after a driver-cache save that followed its pipeline's creation; the list is
// dropped when the driver cache did not load from its file, was written for another driver or GPU, or holds fewer
// bytes than at the list's last save (another run started it again).
// KYTY_SHADER_PRECOMPILE_WARM_PRUNE (default 0: one list of every build's entries in <title>.warm, as before): an entry
// hashes its record key with the emulator build (ShaderPrecompiler::Revision) and the switches, so only the running
// build's entries can ever match, yet the list kept every build's entries for good (39 MB, 4,881,189
// entries, ~8.5k more per build, all loaded at every start and written at every save). =1: the list is kept by build
// in <title>.warm2 (the old <title>.warm is left to older emulators, untouched): the running build's entries and those
// of the two builds saved before it (two builds alternating on one pipeline cache stay warm for both); older builds'
// entries are dropped at load. Contains answers as before: a dropped entry never matches a lookup of this build.
// =2: as 1, and every lookup is also checked against all other builds' entries of the file (kept and dropped) and
// every entry of the old <title>.warm: SHADER_PRECOMPILE_WARM_PRUNE verify lines count the lookups and the hits, which
// must be 0 (on a build's first run with the old list in place; a run of this build with the switch off puts its own
// entries there). The switch is not in the environment fingerprint (it changes no pipeline).
class PrecompileWarmList {
public:
	PrecompileWarmList(std::filesystem::path path, std::string cache_signature, uint64_t loaded_cache_bytes);

	// KYTY_SHADER_PRECOMPILE_WARM_PRUNE: 0, 1 or 2.
	[[nodiscard]] static uint32_t PruneMode();
	// The list's file extension: ".warm" (PruneMode 0) or ".warm2".
	[[nodiscard]] static const char* FileExtension();
	// A background compile or the GPU thread created the pipeline of this record key through the driver cache.
	void Mark(uint64_t key, bool graphics);
	// The saved driver cache holds its pipeline (for this build and these switches).
	[[nodiscard]] bool Contains(uint64_t key, bool graphics) const;
	// A driver-cache save, before it fetches the cache's data: the entries marked so far.
	[[nodiscard]] std::vector<uint64_t> TakeMarked();
	// After that save: `written` (the cache file holds their pipelines now) keeps them and rewrites the list with
	// the saved payload size; otherwise they wait for the next save.
	void Commit(std::vector<uint64_t> entries, bool written, uint64_t cache_payload_bytes);

private:
	[[nodiscard]] static uint64_t Entry(uint64_t key, bool graphics);
	void WriteLocked(uint64_t cache_payload_bytes) const;

	std::filesystem::path        m_path;
	std::string                  m_signature; // the driver cache's (DriverCacheSignature)
	mutable std::mutex           m_mutex;
	std::unordered_set<uint64_t> m_saved;  // in the list file, so in the saved driver cache (PruneMode: of this build)
	std::unordered_set<uint64_t> m_marked; // created since the last TakeMarked
	// PruneMode: the other builds' entries kept (most recently saved first), written back after this build's.
	std::vector<std::pair<uint64_t, std::vector<uint64_t>>> m_other_builds;
	// PruneMode 2: every other build's entry of the file (kept and dropped), and the lookups checked against them.
	std::unordered_set<uint64_t> m_verify_other;
	mutable uint64_t             m_verify_lookups = 0;
	mutable uint64_t             m_verify_hits    = 0;
};

// --precompile-only (PipelineCache::RunPrecompileOnly): the background
// precompile's progress, readable from any thread.
struct PrecompileProgress {
	bool     started  = false; // ShaderPrecompiler::Start ran
	bool     done     = false; // its last worker finished (or nothing was queued)
	uint32_t queued   = 0;     // jobs left to compile at the start
	uint32_t threads  = 0;
	uint32_t compiled = 0;     // pipelines the driver really compiled (10 ms or more) so far
	uint32_t seconds  = 0;     // from the start to the end (done)
	// Of `queued` the graphics jobs; jobs finished so far (by a worker, or dropped because the GPU thread compiles
	// them itself) and of them graphics. finished reaches queued when every job is through.
	uint32_t queued_graphics   = 0;
	uint32_t finished          = 0;
	uint32_t finished_graphics = 0;
	// Left out at load: jobs whose pipelines the saved driver cache holds already (warm list; without
	// The shader reuse they are not queued), and suspects of this build.
	uint32_t warm     = 0;
	uint32_t suspects = 0;
	// Journal files (compute, graphics) of another record layout or version: none of their records was read.
	uint32_t other_layout_journals = 0;
};
[[nodiscard]] PrecompileProgress GetPrecompileProgress();

class ShaderPrecompiler {
public:
	// `warm` (may be null, owned by the caller): jobs whose pipelines it holds are dropped at load; created ones are
	// marked in it.
	ShaderPrecompiler(GraphicContext& graphics, vk::PipelineCache driver_cache, std::filesystem::path journal,
	                  std::filesystem::path graphics_journal, std::atomic<uint32_t>& created,
	                  PrecompileWarmList* warm);
	~ShaderPrecompiler();
	ShaderPrecompiler(const ShaderPrecompiler&)            = delete;
	ShaderPrecompiler& operator=(const ShaderPrecompiler&) = delete;

	[[nodiscard]] static bool Enabled();
	// KYTY_SHADER_PRECOMPILE_GRAPHICS=0: graphics pipelines are neither recorded nor compiled (default: they are,
	// with KYTY_SHADER_PRECOMPILE=1).
	[[nodiscard]] static bool GraphicsEnabled();
	// The emulator build, as recorded in the records' revision.
	[[nodiscard]] static uint64_t Revision();

	// Starts the workers once. Call it when the renderer is complete (pipelines may need the bindless layout).
	// `threads`: 0 = KYTY_SHADER_PRECOMPILE_THREADS (default 6, at most 16); otherwise that many (1..64,
	// --precompile-threads).
	void Start(uint32_t threads = 0);
	// Tells the workers to stop after their running jobs, without waiting (Busy() turns false when they have).
	void RequestStop();
	// Stops and joins the workers (idempotent); before anything they use is destroyed.
	void Stop();
	// True while workers still have jobs (the driver cache keeps growing).
	[[nodiscard]] bool Busy() const;

	// GPU thread, before it compiles the compute permutation or creates the graphics pipeline with this key: waits
	// for a background job compiling it; a job not started yet is dropped (the GPU thread compiles it now). A
	// graphics job is waited for only while the driver compiles its pipeline: one still compiling stages (the GPU
	// thread has its own) leaves the pipeline to the GPU thread. True when it dropped a job not started yet.
	// (Also a PipelineCache table compile worker, KYTY_PERF_TABLE_ASYNC, before it creates the pipeline of a queued draw:
	// it waits there instead of the GPU thread.)
	bool BeforeCompile(uint64_t key);

	// The shader reuse (GPU thread). The first use of a program: the resource plan of a stored translation of
	// it (false: none; the GPU thread translates). Every stored plan of the program is released either way.
	bool TakeReusedPlan(const PrecompileReuseIdentity& identity, std::optional<ShaderRecompiler::IR::ResourcePlan>& plan);
	// Before the GPU thread compiles the permutation with record key `key` (PrecompileKey): the stored compile of
	// exactly these inputs, moved out and freed (false: none; a worker that finishes it later keeps nothing).
	// `queued_dropped`: BeforeCompile just dropped its job (a miss counts as not finished).
	bool TakeReusedCompile(uint64_t key, const PrecompileReuseIdentity& identity, uint32_t push_data_start,
	                       const ShaderRecompiler::IR::ResourceSpecialization& specialization, bool queued_dropped,
	                       PrecompileReused& out);
	// The null share journal: the last live {format, swizzle} of each buffer
	// binding of the program in its loaded journal records (compute or graphics stage), in file order: a binding whose
	// V# was live (run-time stride, a format) in a record takes that record's values. Resized to `buffers` ({0, 0}:
	// never live in a record); `multi`: 1 where the records had more than one live value (the later one is kept). False
	// when no record is of this program.
	bool NullShareJournal(const PrecompileReuseIdentity& identity, size_t buffers,
	                      std::vector<std::array<uint32_t, 2>>& live, std::vector<uint8_t>& multi) const;

	// GPU thread, after it compiled a permutation: appended to its journal unless already recorded (a compute one
	// is a job of the next start; a graphics one is compiled for the recorded pipelines that use it).
	void Record(const PrecompileStageRecord& record);
	// GPU thread, after it created a graphics pipeline from recorded stages: appended unless already recorded.
	void RecordPipeline(const PrecompileGraphicsRecord& record);

private:
	enum class JobState : uint8_t { Queued, Running, Done, Dropped };
	enum class StageState : uint8_t { Idle, Compiling, Ready, Failed };
	struct Job {
		uint64_t key      = 0;
		uint32_t record   = 0; // index into m_compute_records or, graphics, m_graphics_records
		bool     graphics = false;
		JobState state    = JobState::Queued;
		uint32_t worker   = 0; // the worker compiling it (Running)
		// Graphics: the driver compiles the pipeline now (the GPU thread waits for that, not for the stages it has
		// compiled itself); or the GPU thread came first and makes the pipeline itself (the worker leaves it).
		bool     creating    = false;
		bool     skip_create = false;
		// The shader reuse: its pipeline is in the saved driver cache (warm list); the job only translates
		// and compiles its permutations for the reuse store.
		bool     warm        = false;
		// The shader reuse: `key` is a recorded graphics stage no recorded pipeline uses; the job compiles it for
		// the reuse store only (after every other job, not in m_job_index).
		bool     stage_only  = false;
	};
	// A recorded graphics stage: compiled by the first job that needs it, its module destroyed after the last one.
	struct Stage {
		PrecompileStageRecord                    record;
		StageState                               state  = StageState::Idle;
		uint32_t                                 users  = 0; // queued and running jobs that use it
		vk::ShaderModule                         module = nullptr;
		ShaderRecompiler::IR::CompiledShaderInfo info;
	};

	// The shader reuse: a recorded permutation's stored compile (and, until its program's first use, its
	// program's resource plan). `record` is one of the loaded records (fixed once loaded, as long as this object).
	struct ReuseEntry {
		const PrecompileStageRecord*                       record  = nullptr;
		uint64_t                                           program = 0; // PrecompileProgramKey of the record
		std::optional<ShaderRecompiler::IR::ResourcePlan>  plan;
		std::vector<uint32_t>                              spirv;
		ShaderRecompiler::IR::CompiledShaderInfo           info;
		size_t                                             bytes      = 0; // spirv and info (estimate)
		size_t                                             plan_bytes = 0; // the plan (estimate)
		double                                             translate_ms = 0.0;
		double                                             compile_ms   = 0.0;
	};
	enum class ReuseMiss : uint8_t { Unrecorded, Pending, Dropped, Differs, Count };

	[[nodiscard]] std::unordered_set<uint64_t> LoadSuspects();
	void LoadCompute(const std::unordered_set<uint64_t>& skipped, std::vector<Job>& jobs);
	// `stage_jobs`: the shader reuse, the stage-only jobs (queued after all others).
	void LoadGraphics(const std::unordered_set<uint64_t>& skipped, std::vector<Job>& jobs, std::vector<Job>& stage_jobs);
	void Worker(uint32_t index);
	// `key`: the record's key. `need_module`: false for a warm job's permutation (no pipeline is made from it).
	bool CompileModule(const PrecompileStageRecord& record, uint64_t key, bool need_module, vk::ShaderModule& module,
	                   ShaderRecompiler::IR::CompiledShaderInfo& info, size_t& spirv_words);
	void Compile(const PrecompileStageRecord& record, uint64_t key, bool warm);
	// The shader reuse (workers). `any_size`: wanted also with the store at its budget.
	[[nodiscard]] bool ReuseWanted(uint64_t key, bool any_size = false) const;
	[[nodiscard]] bool ReusePlanWanted(uint64_t program) const;
	[[nodiscard]] bool AnyStageWanted(const PrecompileGraphicsRecord& record);
	void StoreReuse(const PrecompileStageRecord& record, uint64_t key, uint64_t program,
	                std::optional<ShaderRecompiler::IR::ResourcePlan> plan, std::vector<uint32_t> spirv,
	                const ShaderRecompiler::IR::CompiledShaderInfo& info, double translate_ms, double compile_ms);
	[[nodiscard]] ReuseMiss ReuseMissReason(uint64_t key, bool queued_dropped);
	void CompileGraphics(Job& job);
	void CompileStageOnly(uint64_t key);
	const Stage* AcquireStage(uint64_t key);
	void ReleaseStagesLocked(const PrecompileGraphicsRecord& record, std::vector<vk::ShaderModule>& unused);
	void CollectModulesLocked(std::vector<vk::ShaderModule>& modules);
	void DestroyModules(std::vector<vk::ShaderModule>& modules);
	void WriteInflightLocked();
	void WriteJournalLocked();
	void WriteGraphicsJournalLocked();
	static void OnQuickExit();

	GraphicContext&        m_graphics;
	vk::PipelineCache      m_driver_cache;
	std::filesystem::path  m_journal_path;
	std::filesystem::path  m_graphics_journal_path;
	std::filesystem::path  m_inflight_path;
	std::filesystem::path  m_suspects_path;
	std::atomic<uint32_t>& m_created;
	PrecompileWarmList*    m_warm = nullptr;

	mutable std::mutex                     m_mutex;
	std::condition_variable                m_done; // a job or a stage compile finished
	std::vector<PrecompileStageRecord>     m_compute_records;  // the compute journal's records (fixed once loaded)
	std::vector<PrecompileGraphicsRecord>  m_graphics_records; // the graphics journal's pipelines (fixed once loaded)
	std::unordered_map<uint64_t, Stage>    m_stages;           // the graphics journal's stages (fixed once loaded)
	std::vector<uint64_t>                  m_stage_order;      // their keys in file order
	std::vector<Job>                       m_jobs; // fixed once workers start
	std::unordered_map<uint64_t, size_t>   m_job_index;
	// The null share journal: the loaded records (compute and graphics stages, file order) by program key
	// (ProgramKey of the record's identity); fixed once loaded.
	std::unordered_map<uint64_t, std::vector<const PrecompileStageRecord*>> m_program_records;
	std::unordered_set<uint64_t>           m_recorded;          // in the compute journal
	std::unordered_set<uint64_t>           m_recorded_graphics; // in the graphics journal (stages and pipelines)
	std::unordered_set<uint64_t>           m_inflight;
	std::vector<vk::ShaderModule>          m_unused_modules; // released by dropped jobs, destroyed by a worker
	size_t                                 m_next      = 0;
	bool                                   m_stop      = false;
	bool                                   m_started   = false;
	bool                                   m_quitting  = false; // clean quit: the in-flight list is final
	std::FILE*                             m_journal   = nullptr;
	std::FILE*                             m_graphics_journal = nullptr;
	std::vector<std::thread>               m_workers;
	std::chrono::steady_clock::time_point  m_start {};
	uint32_t                               m_thread_count = 0;
	std::atomic<uint32_t>                  m_finished {0};
	std::atomic<uint32_t>                  m_compiled {0};          // pipelines the driver really compiled
	std::atomic<uint32_t>                  m_compiled_graphics {0}; // of them graphics
	std::atomic<uint32_t>                  m_skipped_graphics {0};  // graphics jobs not made (stage, device, state)
	// Records this build cannot make with this run's settings (recorded under other ones): their resource layout or an
	// image conversion format (CompileModule); each is skipped instead of ending the process.
	std::atomic<uint32_t>                  m_rejected {0};

	// The shader reuse: the store (a leaf lock: nothing else is locked while it is held).
	mutable std::mutex                          m_reuse_mutex;
	std::unordered_map<uint64_t, ReuseEntry>    m_reuse;                // by record key
	std::unordered_multimap<uint64_t, uint64_t> m_reuse_programs;       // program key -> record keys holding a plan
	std::unordered_set<uint64_t>                m_reuse_taken;          // record keys the GPU thread compiled
	std::unordered_set<uint64_t>                m_reuse_programs_taken; // programs the GPU thread has (no plan needed)
	size_t                                      m_reuse_bytes = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERPRECOMPILE_H_
