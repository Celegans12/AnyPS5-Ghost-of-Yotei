#include "graphics/host_gpu/renderer/pipeline/pipelineManifest.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineShaderOptions.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/emulatorConfig.h"
#if defined(ANYPS5_KYTY)
#include "common/fallback.h"
#endif
#include "common/file.h"
#include "common/hackNotice.h"
#include "common/logging/log.h"
#include "common/parallelFor.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/host_gpu/renderer/rayTracing.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/fileSystem.h"
#include "loader/guestOodle.h"
#include "loader/systemContent.h"

#include <SDL3/SDL.h>
#include <stb_image.h>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <functional>
#include <cstring>
#include <deque>
#include <exception>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <ranges>
#include <set>
#include <sstream>
#include <stop_token>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <xxhash.h>
#include <zstd.h>

namespace Libs::Graphics {

namespace {

using ManifestText::ParseNumber;
using ManifestText::Split;

std::string SafeFileName(std::string name) {
	for (auto& c: name) {
		if (std::isalnum(static_cast<unsigned char>(c)) == 0 && c != '-' && c != '_' && c != '.') {
			c = '_';
		}
	}
	return name;
}

template <typename... Args>
void ManifestLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

template <class Body>
bool RunDroppable(std::atomic<uint32_t>& dropped, bool from_assets, uint64_t shader_hash,
                  uint64_t where, std::string_view what, Body&& body) {
#if defined(__cpp_exceptions)
	if (from_assets && Config::PipelineManifestStrict()) {
		try {
			body();
		} catch (const std::exception& error) {
			EXIT("Pipeline manifest: %s failed: %s\n", std::string(what).c_str(), error.what());
		} catch (...) {
			EXIT("Pipeline manifest: %s failed: unknown exception\n", std::string(what).c_str());
		}
		return true;
	}
	std::string error_text;
	try {
		Common::SoftExitScope soft;
		body();
		return true;
	} catch (const std::exception& error) {
		error_text = error.what();
	} catch (...) {
		error_text = "unknown exception";
	}
	ManifestLog("Pipeline manifest: dropping {}: {}", what, error_text);
	Common::HackNotice::Report("pipeline manifest record dropped after a fatal error",
	                           "RunDroppable", shader_hash, where,
	                           fmt::format("{}: {}", what, error_text));
	dropped.fetch_add(1, std::memory_order_relaxed);
	return false;
#else
#if defined(ANYPS5_KYTY)
	if (!from_assets || !Config::PipelineManifestStrict()) {
		if (Common::Fallback::Run("pipeline manifest record dropped after a fatal error", "RunDroppable",
		                          shader_hash, where, body)) {
			return true;
		}
		ManifestLog("Pipeline manifest: dropping {}", what);
		dropped.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
#endif
	body();
	return true;
#endif
}

double Seconds(std::chrono::steady_clock::duration duration) {
	return std::chrono::duration<double>(duration).count();
}

double Milliseconds(uint64_t nanoseconds) {
	return static_cast<double>(nanoseconds) / 1'000'000.0;
}

std::vector<uint8_t> ReadFile(const std::filesystem::path& path) {
	std::vector<uint8_t> bytes;
	Common::File         file;
	if (!Common::File::IsFileExisting(path) || !file.Open(path, Common::File::Mode::Read)) {
		return bytes;
	}
	bytes.resize(file.Size());
	uint32_t read = 0;
	file.Read(bytes.data(), static_cast<uint32_t>(bytes.size()), &read);
	bytes.resize(read);
	return bytes;
}

std::vector<uint32_t> StaticKey(const ManifestModule& module) {
	std::vector<uint32_t> key;
	std::visit([&](const auto& input) { BuildStageStaticKey(input, key); }, module.input);
	return key;
}

struct FloatModeMerge {
	bool changed = false;
	bool mixed   = false;
};

FloatModeMerge MergeFloatModes(ManifestFloatModes& known, const ManifestFloatModes& seen) {
	FloatModeMerge merge;
	for (size_t i = 0; i < known.size(); i++) {
		if (!seen[i].recorded) {
			continue;
		}
		if (!known[i].recorded) {
			known[i]      = seen[i];
			merge.changed = true;
		} else if (known[i] != seen[i]) {
			merge.mixed = true;
		}
	}
	return merge;
}

}

struct PipelineCache::ManifestState {
	std::filesystem::path record_path;
	std::filesystem::path shared_path;
	std::jthread          watcher;
	uintmax_t                             shared_size = 0;
	std::filesystem::file_time_type       shared_time {};
	std::unordered_map<uint64_t, uint32_t> warm_index;
	std::unordered_set<uint64_t>          warm_pipeline_ids;
	std::unordered_set<uint64_t>          warm_layout_keys;

	std::mutex mutex;
	std::deque<ManifestModule>   warm_modules;
	std::deque<ManifestPipeline> warm_pipelines;
	PipelineManifest                       recorded;
	std::unordered_map<uint64_t, uint32_t> module_index;
	std::unordered_map<uint64_t, uint32_t> pipeline_index;
	std::unordered_set<uint32_t>           mixed_float_modes;
	std::unordered_set<size_t>             unlinked_pipelines;
	std::atomic<uint64_t>                  module_generation {1};
	std::unordered_set<uint64_t>           layout_keys;
	std::unordered_map<uint64_t, std::pair<uint32_t, bool>> module_records;
	bool                                                    dirty           = false;
	uint32_t                                                added_modules   = 0;
	uint32_t                                                added_pipelines = 0;

	struct Code {
		std::shared_ptr<const std::vector<uint32_t>> words;
		std::vector<uint32_t>                        waiting;
		std::vector<uint32_t>                        early;
		uint32_t                                     readers = 0;
		bool                                         mapped  = false;
	};
	std::unordered_map<uint64_t, Code> code;
	std::vector<uint8_t>               missing_code;
	std::vector<const ShaderModule*>   modules;
	std::vector<uint8_t>               module_failed;
	std::vector<std::vector<uint32_t>> module_pipelines;
	std::vector<uint8_t>               missing_modules;
	std::vector<uint32_t>              module_rank;
	std::deque<uint32_t>               module_jobs;
	std::deque<uint32_t>               pipeline_jobs;
	std::condition_variable_any        available;
	uint32_t                           running = 0;

	std::chrono::steady_clock::time_point start;
	std::chrono::steady_clock::time_point last_log;
	bool                                  logged_idle      = true;
	uint32_t                              built            = 0;
	uint32_t                              found            = 0;
	uint32_t                              failed           = 0;
	uint32_t                              pipelines        = 0;
	uint32_t                              pipelines_known  = 0;
	uint32_t                              pipelines_failed = 0;
	uint32_t                              compute          = 0;
	std::atomic<uint32_t>                 dropped {0};
	uint64_t                              build_ns         = 0;
	uint64_t                              pipeline_ns      = 0;
	PrecompileMean                        module_time;
	PrecompileMean                        pipeline_time;

	std::vector<std::jthread> threads;
	std::jthread              saver;

	std::filesystem::path                                   code_path;
	std::mutex                                              code_mutex;
	std::unordered_set<uint64_t>                            code_seen;
	std::deque<std::pair<uint64_t, std::vector<uint32_t>>> code_queue;
	std::condition_variable_any                             code_available;
	std::jthread                                            code_thread;
	std::atomic<bool>                                       scanning {true};

	void QueueModule(uint32_t index) {
		module_jobs.insert(std::upper_bound(module_jobs.begin(), module_jobs.end(), module_rank[index],
		                                    [this](uint32_t value, uint32_t job) {
			                                    return value < module_rank[job];
		                                    }),
		                   index);
	}

	[[nodiscard]] uint32_t PipelineRank(uint32_t index) const {
		uint32_t rank = 0;
		for (const auto stage: warm_pipelines[index].stages) {
			if (stage != ManifestPipeline::NoStage) {
				rank = std::max(rank, module_rank[stage]);
			}
		}
		return rank;
	}

	void QueuePipeline(uint32_t index) {
		const auto words = [this](uint32_t pipeline) {
			uint64_t total = 0;
			for (const auto stage: warm_pipelines[pipeline].stages) {
				if (stage != ManifestPipeline::NoStage && modules[stage] != nullptr) {
					total += modules[stage]->spirv_words;
				}
			}
			return total;
		};
		const auto key = std::pair {PipelineRank(index), words(index)};
		pipeline_jobs.insert(
		    std::upper_bound(pipeline_jobs.begin(), pipeline_jobs.end(), key,
		                     [&](const std::pair<uint32_t, uint64_t>& value, uint32_t job) {
			                     const auto rank = PipelineRank(job);
			                     return value.first != rank ? value.first < rank
			                                                : value.second > words(job);
		                     }),
		    index);
	}
};

void PipelineCache::ManifestStateDeleter::operator()(ManifestState* state) const {
	delete state;
}

struct PipelineCache::UseOrder {
	static constexpr std::string_view Header  = "# kyty shader use order v1";
	static constexpr uintmax_t        MaxSize = 4u << 20u;

	std::unordered_map<uint64_t, uint32_t> rank;

	std::filesystem::path        path;
	std::mutex                   mutex;
	std::unordered_set<uint64_t> noted;
	std::string                  pending;
	bool                         session_started = false;
	std::mutex                   mapped_mutex;
	std::unordered_set<uint64_t> mapped;
	std::jthread                 writer;

	void Write() {
		std::string lines;
		{
			std::scoped_lock lock(mutex);
			lines.swap(pending);
		}
		if (lines.empty()) {
			return;
		}
		Common::File::CreateDirectories(path.parent_path());
		std::ofstream file(path, std::ios::binary | std::ios::app);
		if (!session_started) {
			if (file.tellp() == std::streampos(0)) {
				file << Header << '\n';
			}
			file << "S\n";
			session_started = true;
		}
		file << lines;
	}
};

void PipelineCache::UseOrderDeleter::operator()(UseOrder* order) const {
	order->writer = {};
	delete order;
}

void PipelineCache::OpenUseOrder(const std::filesystem::path& path) {
	m_use_order.reset(new UseOrder());
	auto& order = *m_use_order;
	order.path  = path;
	std::vector<std::vector<uint64_t>> sessions;
	const auto                         bytes = ReadFile(path);
	const std::string_view text(reinterpret_cast<const char*>(bytes.data()), bytes.size());
	for (const auto line: Split(text, '\n')) {
		if (line == "S") {
			sessions.emplace_back();
		} else if (uint64_t hash = 0;
		           !sessions.empty() && line.size() == 16 && ParseNumber(line, hash, 16)) {
			sessions.back().push_back(hash);
		}
	}
	std::vector<uint64_t> programs;
	for (const auto& session: sessions | std::views::reverse) {
		for (const auto hash: session) {
			if (order.rank.try_emplace(hash, static_cast<uint32_t>(programs.size())).second) {
				programs.push_back(hash);
			}
		}
	}
	if (bytes.size() > UseOrder::MaxSize) {
		std::string out = fmt::format("{}\nS\n", UseOrder::Header);
		for (const auto hash: programs) {
			out += fmt::format("{:016x}\n", hash);
		}
		auto temporary = path;
		temporary += ".tmp";
		{
			std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
			file << out;
		}
		std::error_code error;
		std::filesystem::rename(temporary, path, error);
	}
	if (!programs.empty()) {
		ManifestLog("Pipeline manifest: the {} shaders used in {} earlier sessions go first ({})",
		            programs.size(), sessions.size(), Common::PathToString(path));
	}
	order.writer = std::jthread([&order](const std::stop_token& stop) {
		constexpr auto Period = std::chrono::seconds(2);
		auto           next   = std::chrono::steady_clock::now() + Period;
		while (!stop.stop_requested()) {
			std::this_thread::sleep_for(std::chrono::milliseconds(250));
			if (std::chrono::steady_clock::now() >= next) {
				order.Write();
				next = std::chrono::steady_clock::now() + Period;
			}
		}
		order.Write();
	});
}

void PipelineCache::NoteUse(Pipeline& pipeline, std::span<const ShaderModule* const> stages) {
	pipeline.use_noted = true;
	if (m_use_order == nullptr) {
		return;
	}
	auto&            order = *m_use_order;
	std::scoped_lock lock(order.mutex);
	for (const auto* stage: stages) {
		if (stage != nullptr && order.noted.insert(stage->info.shader_hash).second) {
			order.pending += fmt::format("{:016x}\n", stage->info.shader_hash);
		}
	}
}

uint32_t PipelineCache::UseRank(uint64_t program_hash) const {
	if (m_use_order == nullptr) {
		return NoRank;
	}
	if (const auto found = m_use_order->rank.find(program_hash); found != m_use_order->rank.end()) {
		return found->second;
	}
	std::scoped_lock lock(m_use_order->mapped_mutex);
	return m_use_order->mapped.contains(program_hash) ? MappedRank : NoRank;
}

void PipelineCache::DecideRuntimeStrides(std::span<PipelineManifest*> manifests) {
	std::unordered_set<uint64_t> listed;
	std::unordered_set<uint64_t> runtime;
	for (const auto* manifest: manifests) {
		for (const auto& module: manifest->modules) {
			if (module.stage != ShaderType::Compute) {
				continue;
			}
			const auto hash = module.ProgramHash();
			listed.insert(hash);
			if (std::ranges::any_of(module.specialization.buffers, [](const auto& buffer) {
				    return ShaderRecompiler::IR::HasRuntimeStride(buffer.packed_stride);
			    })) {
				runtime.insert(hash);
			}
		}
	}
	for (const auto hash: listed) {
		if (!runtime.contains(hash)) {
			m_baked_stride_programs.insert(hash);
		}
	}
}

const ShaderModule* PipelineCache::FindCompiledModule(
    const ProgramKey& key, const ShaderRecompiler::IR::ResourceSpecialization& specialization,
    uint32_t push_data_cursor) const {
	const auto entry = m_program_cache.find(key);
	if (entry == m_program_cache.end()) {
		return nullptr;
	}
	for (const auto& module: entry->second.modules) {
		const auto& layout = module.info.bindings;
#if defined(ANYPS5_KYTY)
		const bool same = module.specialization == specialization ||
		                  (!module.shared_variant && ShaderRecompiler::IR::SharesEmptyBuffers() &&
		                   ShaderRecompiler::IR::SameModuleCode(entry->second.resource_plan, specialization,
		                                                        module.specialization));
#else
		const bool same = module.specialization == specialization;
#endif
		if (module.function_code.empty() && same &&
		    layout.push_data_start_dword == ShaderRecompiler::IR::PushData::StartFor(
		                                        push_data_cursor, layout.ShaderDataDwords())) {
			return &module;
		}
	}
	return nullptr;
}

void PipelineCache::OpenManifest() {
	const bool warm   = Config::PipelineManifestEnabled();
	const bool record = Config::PipelineManifestRecord() || warm;
	const auto title  = PipelineCacheTitleId();
	if ((!warm && !record) || title.empty()) {
		return;
	}
	std::string version;
	Loader::SystemContentParamSfoGetString("APP_VER", &version);
	const auto stem = SafeFileName(version.empty() ? title : title + "_" + version);
	const auto name = stem + ".kpm";
	OpenUseOrder(std::filesystem::path("_PipelineCache") / (stem + ".use"));
	std::vector<std::filesystem::path> folders;
	std::filesystem::path              assets;
	if (const char* base = SDL_GetBasePath(); base != nullptr) {
		assets = Common::PathFromUtf8(base) / "assets" / "pipelines";
		folders.push_back(assets);
	}
	folders.emplace_back("_PipelineCache");
	std::vector<std::filesystem::path> files;
	for (const auto& folder: folders) {
		if (Common::File::IsFileExisting(folder / name)) {
			files.push_back(folder / name);
		}
	}
	auto shared_name = name;
	shared_name.resize(shared_name.size() - std::string_view(".kpm").size());
	shared_name += ".shared.kpm";
	const auto shared_path = folders.back() / shared_name;
	if (files.empty()) {
		for (const auto& folder: folders) {
			std::error_code error;
			for (const auto& entry: std::filesystem::directory_iterator(folder, error)) {
				const auto file = Common::PathToString(entry.path().filename());
				if (file.starts_with(title + "_") && file.ends_with(".kpm")) {
					files.push_back(entry.path());
				}
			}
		}
	}

	if (Common::File::IsFileExisting(shared_path) &&
	    std::ranges::find(files, shared_path) == files.end()) {
		files.push_back(shared_path);
	}

	m_manifest.reset(new ManifestState());
	auto& state = *m_manifest;
	auto& all   = state.recorded;
	if (warm) {
		state.shared_path = shared_path;
		std::error_code error;
		state.shared_size = std::filesystem::file_size(shared_path, error);
		if (error) state.shared_size = 0;
		state.shared_time = std::filesystem::last_write_time(shared_path, error);
	}
	struct ReadManifest {
		PipelineManifest                      manifest;
		uint32_t                              skipped = 0;
		std::vector<std::string>              unlinked;
		bool                                  read    = false;
		std::chrono::steady_clock::duration   time {};
	};
	std::vector<ReadManifest> read_files(files.size());
	{
		std::vector<PipelineManifest*> manifests;
		for (size_t i = 0; i < files.size(); i++) {
			auto&      file  = read_files[i];
			const auto start = std::chrono::steady_clock::now();
			file.read = LoadManifest(ReadFile(files[i]), file.manifest, &file.skipped, &file.unlinked);
			file.time = std::chrono::steady_clock::now() - start;
			if (file.read) {
				manifests.push_back(&file.manifest);
			}
		}
		DecideRuntimeStrides(manifests);
	}
	for (const auto& path: files) {
		auto&      file     = read_files[static_cast<size_t>(&path - files.data())];
		auto&      loaded   = file.manifest;
		const auto skipped  = file.skipped;
		const auto start    = std::chrono::steady_clock::now() - file.time;
		const auto& unlinked = file.unlinked;
		if (!file.read) {
			ManifestLog("Pipeline manifest: ignoring {} (another format)",
			            Common::PathToString(path));
			continue;
		}
		if (Common::PathToString(path.filename()).ends_with(".shared.kpm")) {
			loaded.tessellation_layouts.clear();
		}
		if (!assets.empty() && path.parent_path() == assets) {
			for (auto& module: loaded.modules) {
				module.from_assets = true;
			}
			for (auto& pipeline: loaded.pipelines) {
				pipeline.from_assets = true;
			}
		}
		std::vector<uint32_t> indices(loaded.modules.size());
		for (size_t i = 0; i < loaded.modules.size(); i++) {
			auto& module = loaded.modules[i];
			SetHostInput(module);
			const auto [found, inserted] =
			    state.module_index.try_emplace(ManifestModuleIdentity(module, StaticKey(module)),
			                                   static_cast<uint32_t>(all.modules.size()));
			if (inserted) {
				all.modules.push_back(std::move(module));
			}
			indices[i] = found->second;
		}
		for (auto& pipeline: loaded.pipelines) {
			for (auto& stage: pipeline.stages) {
				stage = stage == ManifestPipeline::NoStage ? stage : indices[stage];
			}
			const auto [found, inserted] = state.pipeline_index.try_emplace(
			    ManifestPipelineIdentity(pipeline), static_cast<uint32_t>(all.pipelines.size()));
			if (inserted) {
				all.pipelines.push_back(pipeline);
			} else if (MergeFloatModes(all.pipelines[found->second].float_modes, pipeline.float_modes)
			               .mixed) {
				state.mixed_float_modes.insert(found->second);
			}
		}
		for (const auto& layout: loaded.tessellation_layouts) {
			if (state.layout_keys.insert(layout.first).second) {
				all.tessellation_layouts.push_back(layout);
			}
		}
		const auto load_time = std::chrono::steady_clock::now() - start;
		ManifestLog("Pipeline manifest: {} modules, {} pipelines in {} (loaded in {:.1f} ms{})",
		            loaded.modules.size(), loaded.pipelines.size(), Common::PathToString(path),
		            Seconds(load_time) * 1000.0,
		            skipped != 0 ? fmt::format(", {} records left out", skipped) : std::string {});
		if (!unlinked.empty()) {
			ManifestLog("Pipeline manifest: left out {} pipelines of {} that no draw creates from "
			            "the modules they name, such as {}",
			            unlinked.size(), Common::PathToString(path), unlinked.front());
		}
	}
	if (files.empty()) {
		ManifestLog("Pipeline manifest: none for {}", name);
	}
	if (record) {
		state.record_path = folders.back() / name;
		ManifestLog("Pipeline manifest: recording into {}",
		            Common::PathToString(state.record_path));
		state.saver = std::jthread([this](const std::stop_token& stop) {
			constexpr auto Period = std::chrono::seconds(10);
			auto           next   = std::chrono::steady_clock::now() + Period;
			while (!stop.stop_requested()) {
				std::this_thread::sleep_for(std::chrono::milliseconds(250));
				if (std::chrono::steady_clock::now() >= next) {
					SaveManifestRecording();
					next = std::chrono::steady_clock::now() + Period;
				}
			}
		});
	} else if (!warm || all.modules.empty()) {
		m_manifest.reset();
		return;
	}
	if (!warm) {
		return;
	}

	for (const auto& key: m_graphics_pipelines | std::views::keys) {
		m_claimed_graphics.insert(key);
	}
	for (const auto id: m_compute_pipelines | std::views::keys) {
		m_claimed_compute.insert(id);
	}
	m_manifest_warm = true;
	state.start     = std::chrono::steady_clock::now();
	state.last_log  = state.start;
	if (record) {
		auto listed = all;
		AddManifestRecords(listed);
	} else {
		AddManifestRecords(all);
	}
	for (size_t i = 0; i < Common::ParallelThreads(); i++) {
		state.threads.emplace_back([this](const std::stop_token& stop) { ManifestWorker(stop); });
	}
	state.code_path   = folders.back() / (stem + ".code");
	state.code_thread = std::jthread([this](const std::stop_token& stop) { CodeStoreThread(stop); });
	state.watcher = std::jthread([this](const std::stop_token& stop) {
		KYTY_PROFILER_THREAD("Thread_PipelineManifestShared");
		Common::Thread::LowerCurrentPriority();
		constexpr auto Period = std::chrono::seconds(15);
		auto           next   = std::chrono::steady_clock::now() + Period;
		while (!stop.stop_requested()) {
			std::this_thread::sleep_for(std::chrono::milliseconds(250));
			if (std::chrono::steady_clock::now() >= next) {
				ReloadSharedManifest();
				next = std::chrono::steady_clock::now() + Period;
			}
		}
	});
	ShaderSetMapListener(OnShaderMapped, this);
}

std::pair<uint32_t, uint32_t> PipelineCache::AddManifestRecords(PipelineManifest& loaded) {
	auto&                 state = *m_manifest;
	std::vector<uint32_t> indices(loaded.modules.size(), ManifestPipeline::NoStage);
	std::vector<uint32_t> new_modules;
	std::vector<uint32_t> new_pipelines;
	const auto supported = [&](const ManifestModule& module) {
		return module.stage != ShaderType::Mesh || m_graphics.mesh_shader_enabled;
	};
	auto next_module = static_cast<uint32_t>(state.warm_modules.size());
	for (uint32_t i = 0; i < loaded.modules.size(); i++) {
		auto& module = loaded.modules[i];
		if (!supported(module)) continue;
		const auto [found, inserted] = state.warm_index.try_emplace(
		    ManifestModuleIdentity(module, StaticKey(module)), next_module);
		if (inserted) {
			new_modules.push_back(i);
			next_module++;
		}
		indices[i] = found->second;
	}
	for (uint32_t i = 0; i < loaded.pipelines.size(); i++) {
		auto& pipeline = loaded.pipelines[i];
		bool  dropped  = false;
		for (auto& stage: pipeline.stages) {
			if (stage != ManifestPipeline::NoStage) {
				stage   = indices[stage];
				dropped = dropped || stage == ManifestPipeline::NoStage;
			}
		}
		if (!dropped && state.warm_pipeline_ids.insert(ManifestPipelineIdentity(pipeline)).second) {
			new_pipelines.push_back(i);
		}
	}
	uint32_t new_layouts = 0;
	{
		LockPrograms();
		for (const auto& layout: loaded.tessellation_layouts) {
			if (state.warm_layout_keys.insert(layout.first).second) {
				m_tess_layouts.try_emplace(layout.first, layout.second);
				new_layouts++;
			}
		}
		UnlockPrograms();
	}
	if (new_modules.empty() && new_pipelines.empty()) {
		return {0, 0};
	}
	{
		std::scoped_lock lock(state.mutex);
		for (const auto loaded_index: new_modules) {
			const auto index = static_cast<uint32_t>(state.warm_modules.size());
			state.warm_modules.push_back(std::move(loaded.modules[loaded_index]));
			const auto& module = state.warm_modules.back();
			state.missing_code.push_back(0);
			state.module_rank.push_back(UseRank(module.ProgramHash()));
			state.modules.push_back(nullptr);
			state.module_failed.push_back(0);
			state.module_pipelines.emplace_back();
			for (const auto hash: {module.code_hash, module.back_hash}) {
				if (hash == 0) continue;
				auto& code = state.code[hash];
				code.readers++;
				if (code.mapped && code.words != nullptr) continue;
				code.mapped = false;
				code.waiting.push_back(index);
				state.missing_code[index]++;
			}
			if (state.missing_code[index] == 0) {
				state.QueueModule(index);
			}
		}
		for (const auto loaded_index: new_pipelines) {
			const auto  index    = static_cast<uint32_t>(state.warm_pipelines.size());
			const auto& pipeline = state.warm_pipelines.emplace_back(loaded.pipelines[loaded_index]);
			auto&       missing  = state.missing_modules.emplace_back(0);
			for (const auto stage: pipeline.stages) {
				if (stage == ManifestPipeline::NoStage || state.modules[stage] != nullptr) continue;
				if (state.module_failed[stage] != 0) {
					missing = UINT8_MAX;
				} else {
					state.module_pipelines[stage].push_back(index);
					missing = missing == UINT8_MAX ? missing : static_cast<uint8_t>(missing + 1);
				}
			}
			if (missing == 0) {
				state.QueuePipeline(index);
			}
		}
		state.logged_idle = false;
	}
	state.available.notify_all();
	if (state.watcher.joinable()) {
		ManifestLog("Pipeline manifest: {} new modules, {} new pipelines and {} tessellation "
		            "layouts from {}",
		            new_modules.size(), new_pipelines.size(), new_layouts,
		            Common::PathToString(state.shared_path));
	}
	return {static_cast<uint32_t>(new_modules.size()), static_cast<uint32_t>(new_pipelines.size())};
}

void PipelineCache::ReloadSharedManifest() {
	auto&           state = *m_manifest;
	std::error_code error;
	const auto      size = std::filesystem::file_size(state.shared_path, error);
	if (error) return;
	const auto time = std::filesystem::last_write_time(state.shared_path, error);
	if (error || (size == state.shared_size && time == state.shared_time)) return;
	state.shared_size = size;
	state.shared_time = time;
	constexpr uintmax_t MaxSize = 1ull << 29;
	if (size > MaxSize) {
		ManifestLog("Pipeline manifest: ignoring {} ({} bytes)",
		            Common::PathToString(state.shared_path), size);
		return;
	}
	PipelineManifest loaded;
	uint32_t         skipped = 0;
	if (!LoadManifest(ReadFile(state.shared_path), loaded, &skipped)) {
		return;
	}
	loaded.tessellation_layouts.clear();
	for (auto& module: loaded.modules) {
		SetHostInput(module);
	}
	if (AddManifestRecords(loaded) != std::pair<uint32_t, uint32_t> {0, 0}) {
		ShaderSetMapListener(OnShaderMapped, this);
	}
}

void PipelineCache::CloseManifest() {
	if (m_manifest == nullptr) {
		return;
	}
	auto& state   = *m_manifest;
	state.watcher = {};
	if (m_manifest_warm) {
		ShaderSetMapListener(nullptr, nullptr);
		state.code_thread = {};
		for (auto& thread: state.threads) {
			thread.request_stop();
		}
		state.available.notify_all();
		state.threads.clear();
		ManifestProgress(true);
	}
	state.saver = {};
	SaveManifestRecording();
}

void PipelineCache::OnShaderMapped(void* userdata, uint64_t hash, std::span<const uint32_t> code) {
	auto& cache = *static_cast<PipelineCache*>(userdata);
	auto& state = *cache.m_manifest;
	{
		std::scoped_lock lock(state.code_mutex);
		if (state.code_seen.insert(hash).second) {
			state.code_queue.emplace_back(hash, std::vector<uint32_t>(code.begin(), code.end()));
		}
	}
	state.code_available.notify_all();
	{
		std::scoped_lock lock(cache.m_use_order->mapped_mutex);
		cache.m_use_order->mapped.insert(hash);
	}
	cache.ProvideCode(hash, code, true);
}

bool PipelineCache::ProvideCode(uint64_t hash, std::span<const uint32_t> code, bool from_guest) {
	auto& state = *m_manifest;
	{
		std::scoped_lock lock(state.mutex);
		const auto       found = state.code.find(hash);
		if (found == state.code.end()) {
			return false;
		}
		auto& entry = found->second;
		const auto mapped_first = [&](uint32_t index) {
			if (state.module_rank[index] != NoRank) {
				return false;
			}
			state.module_rank[index] = MappedRank;
			return true;
		};
		if (entry.mapped) {
			for (const auto index: from_guest ? std::exchange(entry.early, {}) : std::vector<uint32_t> {}) {
				if (!mapped_first(index)) {
					continue;
				}
				if (const auto queued = std::ranges::find(state.module_jobs, index);
				    queued != state.module_jobs.end()) {
					state.module_jobs.erase(queued);
					state.QueueModule(index);
				}
			}
			return false;
		}
		entry.words  = std::make_shared<const std::vector<uint32_t>>(code.begin(), code.end());
		entry.mapped = true;
		for (const auto index: entry.waiting) {
			if (from_guest) {
				mapped_first(index);
			}
			if (--state.missing_code[index] == 0) {
				state.QueueModule(index);
			}
		}
		if (!from_guest) {
			entry.early = entry.waiting;
		}
		entry.waiting     = {};
		state.logged_idle = false;
	}
	state.available.notify_all();
	return true;
}

namespace {

constexpr uint64_t CodeStoreMagic = 0x3145444F4359544Bull;
constexpr uint32_t ScanMark       = 0xffffffffu;
constexpr uint32_t ScanVersion    = 2;
struct CodeRecord {
	uint64_t hash;
	uint32_t words;
	uint32_t compressed_bytes;
};

bool ReadCodeRecord(const CodeRecord& record, const uint8_t* compressed, std::vector<uint32_t>& words) {
	words.resize(record.words);
	const auto bytes = words.size() * sizeof(uint32_t);
	return ZSTD_getFrameContentSize(compressed, record.compressed_bytes) == bytes &&
	       ZSTD_decompress(words.data(), bytes, compressed, record.compressed_bytes) == bytes;
}

template <typename Found>
uint32_t FindShaderPrograms(std::span<const uint8_t> data, Found&& found) {
	constexpr std::string_view Magic      = "barefoot";
	constexpr size_t           FooterSize = 0x30;
	const std::string_view     text(reinterpret_cast<const char*>(data.data()), data.size());
	uint32_t                   count = 0;
	for (size_t at = text.find(Magic); at != std::string_view::npos; at = text.find(Magic, at + 1)) {
		if (data.size() - at < FooterSize) {
			break;
		}
		std::array<uint32_t, 8> footer {};
		std::memcpy(footer.data(), data.data() + at, sizeof(footer));
		const auto program = Common::AlignUp(uint64_t {footer[5]}, 16) + footer[6];
		const auto before  = Common::AlignUp(program, 16) + Common::AlignUp(uint64_t {footer[7]}, 16);
		if (footer[4] != 1 || footer[5] == 0 || program % 4 != 0 || before > at) {
			continue;
		}
		count++;
		found(data.subspan(at - before, program));
	}
	return count;
}

bool Lz4Decompress(std::span<const uint8_t> src, std::span<uint8_t> dst) {
	size_t     s      = 0;
	size_t     d      = 0;
	const auto length = [&](size_t& value) {
		for (uint8_t more = 255; more == 255;) {
			if (s == src.size()) {
				return false;
			}
			more = src[s++];
			value += more;
		}
		return true;
	};
	while (s < src.size()) {
		const auto token    = src[s++];
		size_t     literals = token >> 4u;
		if ((literals == 15 && !length(literals)) || literals > src.size() - s || literals > dst.size() - d) {
			return false;
		}
		std::memcpy(dst.data() + d, src.data() + s, literals);
		s += literals;
		d += literals;
		if (s == src.size()) {
			break;
		}
		if (src.size() - s < 2) {
			return false;
		}
		const size_t offset = src[s] | (size_t {src[s + 1]} << 8u);
		s += 2;
		size_t match = token & 15u;
		if (offset == 0 || offset > d || (match == 15 && !length(match)) || match + 4 > dst.size() - d) {
			return false;
		}
		for (const auto end = d + match + 4; d < end; d++) {
			dst[d] = dst[d - offset];
		}
	}
	return d == dst.size();
}

template <typename Found>
void FindCompressedShaders(std::span<const uint8_t> data, std::vector<uint8_t>& shader, Found&& found) {
	constexpr std::string_view Magic      = "barefoot";
	constexpr size_t           FooterSize = 0x30;
	constexpr uint32_t         MaxSize    = 4u << 20u;
	constexpr uint64_t         MaxStored  = 256u << 10u;
	const std::string_view     text(reinterpret_cast<const char*>(data.data()), data.size());
	const auto                 word = [&](uint64_t at) {
		uint32_t value = 0;
		std::memcpy(&value, data.data() + at, sizeof(value));
		return value;
	};
	uint64_t done = 0;
	for (size_t at = text.find(Magic); at != std::string_view::npos; at = text.find(Magic, at + 1)) {
		if (at < done) {
			continue;
		}
		if (data.size() - at >= FooterSize) {
			const auto program = Common::AlignUp(uint64_t {word(at + 20)}, 16) + word(at + 24);
			const auto before  = program + Common::AlignUp(uint64_t {word(at + 28)}, 16);
			if (word(at + 16) == 1 && word(at + 20) != 0 && program % 4 == 0 && before <= at) {
				continue;
			}
		}
		for (uint64_t end = at + FooterSize - 4; end <= at + FooterSize + 64 && end + 5 <= data.size(); end++) {
			const auto size = word(end);
			if (data[end + 4] >= 8 || size < 64 || size > MaxSize) {
				continue;
			}
			const auto most = std::min({uint64_t {size}, end - 4, MaxStored});
			for (uint64_t stored = std::max<uint64_t>(size / 16, end - at); stored <= most; stored++) {
				if (word(end - 4 - stored) != stored) {
					continue;
				}
				shader.resize(size);
				if (Lz4Decompress(data.subspan(end - stored, stored), shader)) {
					found(std::span<const uint8_t>(shader));
					done = end;
					break;
				}
			}
			if (done == end) {
				break;
			}
		}
	}
}

struct SpanReader {
	std::span<const uint8_t> data;
	size_t                   at = 0;
	bool                     ok = true;

	bool Skip(uint64_t bytes) {
		if (data.size() - at < bytes) {
			ok = false;
			at = data.size();
		} else {
			at += bytes;
		}
		return ok;
	}
	template <typename T>
	T Get() {
		T value {};
		const auto from = at;
		if (Skip(sizeof(T))) {
			std::memcpy(&value, data.data() + from, sizeof(T));
		}
		return value;
	}
	std::string String() {
		const auto  length = Get<int32_t>();
		const auto  units  = length < 0 ? -int64_t {length} : int64_t {length};
		const auto  unit   = length < 0 ? 2u : 1u;
		const auto  from   = at;
		std::string text;
		if (units > 0 && Skip(units * unit)) {
			for (int64_t i = 0; i + 1 < units; i++) {
				text.push_back(static_cast<char>(data[from + i * unit]));
			}
		}
		return text;
	}
};

template <typename Found>
void ForEachUnrealShader(std::span<const uint8_t> archive, Found&& found) {
	SpanReader reader {archive};
	const auto array = [&](uint32_t element) {
		const auto count = reader.Get<uint32_t>();
		return reader.Skip(uint64_t {count} * element) ? count : 0;
	};
	if (reader.Get<uint32_t>() != 2) {
		return;
	}
	array(20);
	array(20);
	array(16);
	const auto table   = reader.at + 4;
	const auto shaders = array(17);
	array(16);
	array(4);
	if (!reader.ok) {
		return;
	}
	const auto code = archive.subspan(reader.at);
	for (uint32_t i = 0; i < shaders; i++) {
		SpanReader entry {archive, table + i * 17};
		const auto offset = entry.Get<uint64_t>();
		const auto stored = entry.Get<uint32_t>();
		const auto size   = entry.Get<uint32_t>();
		if (offset <= code.size() && stored <= code.size() - offset) {
			found(code.subspan(offset, stored), size);
		}
	}
}

template <typename Read, typename Archive, typename Assets>
bool ForEachUnrealEntry(uint64_t size, Read&& read, Archive&& archive, Assets&& assets) {
	constexpr uint32_t   Magic    = 0x5A6F12E1u;
	constexpr uint64_t   MaxAsset = 1u << 20u;
	std::vector<uint8_t> data;
	const auto           tail = std::min<uint64_t>(size, 1024);
	if (!read(size - tail, tail, data)) {
		return false;
	}
	size_t magic = 0;
	for (size_t at = data.size() >= 24 ? data.size() - 24 : 0; at > 17 && magic == 0; at--) {
		if (std::memcmp(data.data() + at, &Magic, 4) == 0) {
			magic = at;
		}
	}
	SpanReader footer {data, magic + 4};
	const auto version      = footer.Get<uint32_t>();
	const auto index_offset = footer.Get<uint64_t>();
	const auto index_size   = footer.Get<uint64_t>();
	footer.Skip(20);
	std::array<bool, 6> zlib {false, version < 8};
	for (size_t method = 1; version >= 8 && method < zlib.size() && footer.Skip(0) &&
	                        footer.data.size() - footer.at >= 32;
	     method++) {
		const auto* name = reinterpret_cast<const char*>(data.data() + footer.at);
		zlib[method]     = std::string_view(name, strnlen(name, 32)) == "Zlib";
		footer.Skip(32);
	}
	const bool legacy = version >= 3 && version <= 7;
	if (magic == 0 || (version >= 4 && data[magic - 1] != 0) || (!legacy && version != 10 && version != 11) ||
	    index_offset > size || index_size > size - index_offset || index_size > (256u << 20u) ||
	    !read(index_offset, index_size, data)) {
		return false;
	}
	std::vector<uint8_t> entry_data;
	std::vector<uint8_t> inflated;
	const auto visit = [&](std::string name, uint64_t offset, uint64_t length, uint64_t stored, uint32_t method,
	                       uint64_t blocks) {
		std::ranges::transform(name, name.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
		const bool is_archive = name.ends_with(".ushaderbytecode");
		const bool is_global  = name.ends_with(".bin");
		const bool is_map     = name.ends_with(".umap");
		const bool is_asset   = is_map || name.ends_with(".uexp") || name.ends_with(".uasset");
		const uint64_t header = 8 + 8 + 8 + 4 + 20 + (method != 0 ? 4 + 16 * blocks : 0) + 1 + 4;
		const bool     stored_archive = is_archive && method == 0;
		const bool     readable_asset = (is_global || (is_asset && (method != 0 || is_map || length <= MaxAsset))) &&
		                            (method == 0 || (method < zlib.size() && zlib[method]));
		if ((!stored_archive && !readable_asset) || offset > size || stored > size - offset ||
		    header > size - offset - stored || length > (256u << 20u) || !read(offset, header + stored, entry_data)) {
			return;
		}
		SpanReader copy {entry_data, 8};
		if (copy.Get<uint64_t>() != stored || copy.Get<uint64_t>() != length) {
			return;
		}
		if (method == 0) {
			const auto stored_data = std::span<const uint8_t>(entry_data).subspan(header);
			is_archive ? archive(stored_data) : assets(stored_data);
			return;
		}
		copy.Skip(4 + 20 + 4);
		inflated.resize(length);
		const uint64_t base   = version < 5 ? offset : 0;
		uint64_t       filled = 0;
		for (uint64_t block = 0; block < blocks && copy.ok; block++) {
			const auto start = copy.Get<uint64_t>() - base;
			const auto end   = copy.Get<uint64_t>() - base;
			if (start < header || end < start || end > entry_data.size() || end - start > INT32_MAX) {
				break;
			}
			const auto out = stbi_zlib_decode_buffer(
			    reinterpret_cast<char*>(inflated.data() + filled), static_cast<int>(length - filled),
			    reinterpret_cast<const char*>(entry_data.data() + start), static_cast<int>(end - start));
			if (out <= 0) {
				break;
			}
			filled += static_cast<uint64_t>(out);
		}
		assets(std::span<const uint8_t>(inflated).first(filled));
	};
	SpanReader index {data};
	index.String();
	if (legacy) {
		const auto entries = index.Get<uint32_t>();
		for (uint32_t entry = 0; entry < entries && index.ok; entry++) {
			auto           name   = index.String();
			const auto     offset = index.Get<uint64_t>();
			const auto     stored = index.Get<uint64_t>();
			const auto     length = index.Get<uint64_t>();
			const auto     method = index.Get<uint32_t>();
			uint64_t       blocks = 0;
			index.Skip(20);
			if (method != 0) {
				blocks = index.Get<uint32_t>();
				index.Skip(16 * blocks);
			}
			const bool encrypted = index.Get<uint8_t>() != 0;
			index.Skip(4);
			if (index.ok && !encrypted && method < zlib.size()) {
				visit(std::move(name), offset, length, stored, method, blocks);
			}
		}
		return index.ok;
	}
	index.Skip(4 + 8);
	if (index.Get<uint32_t>() != 0) {
		index.Skip(8 + 8 + 20);
	}
	if (index.Get<uint32_t>() == 0) {
		return false;
	}
	const auto directory_offset = index.Get<uint64_t>();
	const auto directory_size   = index.Get<uint64_t>();
	index.Skip(20);
	const auto           encoded_size = index.Get<uint32_t>();
	const auto           encoded_at   = index.at;
	std::vector<uint8_t> directory;
	if (!index.Skip(encoded_size) || directory_offset > size || directory_size > size - directory_offset ||
	    directory_size > (256u << 20u) || !read(directory_offset, directory_size, directory)) {
		return false;
	}
	const auto encoded = std::span<const uint8_t>(data).subspan(encoded_at, encoded_size);
	SpanReader names {directory};
	const auto folders = names.Get<uint32_t>();
	for (uint32_t folder = 0; folder < folders && names.ok; folder++) {
		names.String();
		const auto files = names.Get<uint32_t>();
		for (uint32_t file = 0; file < files && names.ok; file++) {
			auto       name  = names.String();
			const auto entry = names.Get<int32_t>();
			if (entry < 0) {
				continue;
			}
			SpanReader     reader {encoded, static_cast<size_t>(entry)};
			const auto     flags  = reader.Get<uint32_t>();
			const auto     method = (flags >> 23u) & 0x3fu;
			const uint64_t offset = (flags & (1u << 31u)) != 0 ? reader.Get<uint32_t>() : reader.Get<uint64_t>();
			const uint64_t length = (flags & (1u << 30u)) != 0 ? reader.Get<uint32_t>() : reader.Get<uint64_t>();
			uint64_t       stored = length;
			if (method != 0) {
				stored = (flags & (1u << 29u)) != 0 ? reader.Get<uint32_t>() : reader.Get<uint64_t>();
			}
			if (reader.ok && (flags & (1u << 22u)) == 0) {
				visit(std::move(name), offset, length, stored, method, (flags >> 6u) & 0xffffu);
			}
		}
	}
	return names.ok;
}

template <typename Found>
void FindUnityShaders(std::span<const uint8_t> data, std::vector<uint8_t>& shader, Found&& found) {
	constexpr uint32_t MaxChunks = 64;
	constexpr uint32_t MaxSize   = 64u << 20u;
	const auto         word      = [&](uint64_t at) {
		uint32_t value = 0;
		std::memcpy(&value, data.data() + at, sizeof(value));
		return value;
	};
	for (uint64_t at = 0; at + 16 <= data.size(); at += 4) {
		const auto chunks = word(at + 4);
		if (word(at) != 1 || chunks == 0 || chunks > MaxChunks) {
			continue;
		}
		const uint64_t list = 8 + 4 * uint64_t {chunks};
		const uint64_t blob = at + 3 * list;
		if (blob + 4 > data.size() || word(at + list) != 1 || word(at + list + 4) != chunks ||
		    word(at + 2 * list) != 1 || word(at + 2 * list + 4) != chunks) {
			continue;
		}
		const uint64_t blob_size = word(blob);
		if (blob_size > data.size() - blob - 4) {
			continue;
		}
		for (uint32_t chunk = 0; chunk < chunks; chunk++) {
			const uint64_t offset = word(at + 8 + 4 * chunk);
			const uint64_t stored = word(at + list + 8 + 4 * chunk);
			const auto     size   = word(at + 2 * list + 8 + 4 * chunk);
			if (stored == 0 || size < stored || size > MaxSize || offset > blob_size || stored > blob_size - offset) {
				break;
			}
			shader.resize(size);
			if (Lz4Decompress(data.subspan(blob + 4 + offset, stored), shader)) {
				found(std::span<const uint8_t>(shader));
			}
		}
	}
}

template <typename Read, typename Found>
void ForEachUnityShaderFile(uint64_t size, Read&& read, Found&& found) {
	const auto big = [](std::span<const uint8_t> data, uint64_t at, uint32_t bytes) {
		uint64_t value = 0;
		for (uint32_t i = 0; i < bytes && at + i < data.size(); i++) {
			value = (value << 8u) | data[at + i];
		}
		return value;
	};
	const auto text = [](SpanReader& reader) {
		const auto rest = reader.data.subspan(reader.at);
		const auto end  = std::ranges::find(rest, uint8_t {0});
		reader.Skip(static_cast<uint64_t>(end - rest.begin()) + 1);
	};
	const auto has_shaders = [&](std::span<const uint8_t> head) -> std::optional<bool> {
		const auto version = big(head, 8, 4);
		if (head.size() < 48 || version < 19 || version > 23 || head[16] != 0) {
			return false;
		}
		SpanReader reader {head, version >= 22 ? 48u : 20u};
		text(reader);
		reader.Skip(4);
		const bool trees = reader.Get<uint8_t>() != 0;
		const auto types = reader.Get<uint32_t>();
		for (uint32_t type = 0; type < types && reader.ok; type++) {
			const auto id = reader.Get<int32_t>();
			if (id == 48 && reader.ok) {
				return true;
			}
			reader.Skip(1 + 2 + (id == 114 ? 16 : 0) + 16);
			if (trees) {
				const auto nodes   = reader.Get<uint32_t>();
				const auto strings = reader.Get<uint32_t>();
				reader.Skip(32 * uint64_t {nodes} + strings);
				if (version >= 21) {
					reader.Skip(4 * uint64_t {reader.Get<uint32_t>()});
				}
			}
		}
		return reader.ok ? std::optional<bool>(false) : std::nullopt;
	};
	std::vector<uint8_t> head;
	std::vector<uint8_t> serialized;
	const auto           search = [&](uint64_t offset, uint64_t length) {
		for (const uint64_t most: {uint64_t {256u << 10u}, uint64_t {4u << 20u}}) {
			if (!read(offset, std::min(length, most), head)) {
				return;
			}
			const auto shaders = has_shaders(head);
			if (shaders.has_value() || length <= most) {
				if (shaders.value_or(false) && read(offset, length, serialized)) {
					found(std::span<const uint8_t>(serialized));
				}
				return;
			}
		}
	};
	if (!read(0, std::min<uint64_t>(size, 4u << 10u), head)) {
		return;
	}
	if (std::string_view(reinterpret_cast<const char*>(head.data()), std::min<size_t>(head.size(), 8)) !=
	    std::string_view("UnityFS\0", 8)) {
		search(0, size);
		return;
	}
	SpanReader reader {head, 8};
	const auto version = big(head, reader.at, 4);
	reader.Skip(4);
	text(reader);
	text(reader);
	reader.Skip(8);
	const auto table_stored = big(head, reader.at, 4);
	const auto table_size   = big(head, reader.at + 4, 4);
	const auto flags        = big(head, reader.at + 8, 4);
	reader.Skip(12);
	uint64_t table_at = version >= 7 ? Common::AlignUp(uint64_t {reader.at}, 16) : reader.at;
	uint64_t data_at  = table_at + table_stored;
	if ((flags & 0x80u) != 0) {
		data_at  = table_at;
		table_at = size - table_stored;
	}
	if ((flags & 0x200u) != 0) {
		data_at = Common::AlignUp(data_at, 16);
	}
	std::vector<uint8_t> stored;
	std::vector<uint8_t> table(table_size);
	if (!reader.ok || table_at > size || table_stored > size - table_at || table_size > (64u << 20u) ||
	    !read(table_at, table_stored, stored)) {
		return;
	}
	const auto method = flags & 0x3fu;
	if (method == 0 && table_stored == table_size) {
		table = stored;
	} else if ((method != 2 && method != 3) || !Lz4Decompress(stored, table)) {
		return;
	}
	uint64_t   at     = 16;
	const auto blocks = big(table, at, 4);
	at += 4;
	for (uint64_t block = 0; block < blocks && at + 10 <= table.size(); block++, at += 10) {
		if (big(table, at, 4) != big(table, at + 4, 4) || (big(table, at + 8, 2) & 0x3fu) != 0) {
			return;
		}
	}
	const auto files = big(table, at, 4);
	at += 4;
	for (uint64_t file = 0; file < files && at + 20 < table.size(); file++) {
		const auto offset     = big(table, at, 8);
		const auto length     = big(table, at + 8, 8);
		const auto file_flags = big(table, at + 16, 4);
		SpanReader name {table, static_cast<size_t>(at + 20)};
		text(name);
		at = name.at;
		if ((file_flags & 4u) != 0 && offset <= size - data_at && length <= size - data_at - offset) {
			search(data_at + offset, length);
		}
	}
}

template <typename Read, typename Decompress, typename Found>
bool ForEachIoStoreShaderGroup(std::span<const uint8_t> toc, Read&& read, Decompress&& decompress,
                               Found&& found) {
	constexpr std::string_view Magic("-==--==--==--==-");
	constexpr uint64_t         MaxChunk = 64u << 20u;
	if (toc.size() < 144 || std::memcmp(toc.data(), Magic.data(), Magic.size()) != 0) {
		return false;
	}
	SpanReader header {toc, Magic.size()};
	const auto version = header.Get<uint8_t>();
	header.Skip(3);
	const auto header_size   = header.Get<uint32_t>();
	const auto chunks        = header.Get<uint32_t>();
	const auto blocks        = header.Get<uint32_t>();
	const auto block_entry   = header.Get<uint32_t>();
	const auto method_count  = header.Get<uint32_t>();
	const auto method_length = header.Get<uint32_t>();
	const auto block_size    = header.Get<uint32_t>();
	header.Skip(4);
	const auto partitions = header.Get<uint32_t>();
	header.Skip(8 + 16);
	const auto flags = header.Get<uint8_t>();
	header.Skip(3);
	const auto        seeds          = header.Get<uint32_t>();
	auto              partition_size = header.Get<uint64_t>();
	const auto        unhashed       = header.Get<uint32_t>();
	constexpr uint8_t Encrypted      = 2;
	if (!header.ok || version < 3 || version > 8 || (flags & Encrypted) != 0 || block_entry != 12 ||
	    block_size == 0) {
		return false;
	}
	if (partitions <= 1 || partition_size == 0) {
		partition_size = UINT64_MAX;
	}
	SpanReader body {toc, header_size};
	const auto ids_at = body.at;
	body.Skip(uint64_t {chunks} * 12);
	const auto spans_at = body.at;
	body.Skip(uint64_t {chunks} * 10);
	body.Skip((version >= 4 ? uint64_t {seeds} * 4 : 0) +
	          (version >= 5 ? uint64_t {unhashed} * 4 : 0));
	const auto blocks_at = body.at;
	body.Skip(uint64_t {blocks} * 12);
	const auto methods_at = body.at;
	body.Skip(uint64_t {method_count} * method_length);
	if (!body.ok) {
		return false;
	}
	const auto number = [&](size_t at, uint32_t bytes, bool big_endian) {
		uint64_t value = 0;
		for (uint32_t i = 0; i < bytes; i++) {
			value |= uint64_t {toc[at + i]} << (8u * (big_endian ? bytes - 1 - i : i));
		}
		return value;
	};
	const auto id = [&](uint32_t chunk) {
		return std::string_view(reinterpret_cast<const char*>(toc.data() + ids_at + chunk * 12),
		                        12);
	};
	std::vector<uint8_t> stored;
	const auto           read_chunk = [&](uint32_t chunk, std::vector<uint8_t>& out) {
		const auto offset = number(spans_at + chunk * 10, 5, true);
		const auto length = number(spans_at + chunk * 10 + 5, 5, true);
		out.clear();
		if (length > MaxChunk) {
			return false;
		}
		const auto last = (offset + length + block_size - 1) / block_size;
		for (uint64_t block = offset / block_size; block < last;) {
			if (block >= blocks) {
				return false;
			}
			const auto entry  = blocks_at + block * 12;
			const auto at     = number(entry, 5, false);
			const auto size   = number(entry + 8, 3, false);
			const auto method = toc[entry + 11];
			if (method == 0) {
				auto end   = block;
				auto bytes = uint64_t {0};
				for (; end < last && end < blocks; end++) {
					const auto next = blocks_at + end * 12;
					if (toc[next + 11] != 0 || number(next, 5, false) != at + bytes ||
					    number(next + 5, 3, false) != number(next + 8, 3, false) ||
					    (at + bytes) / partition_size != at / partition_size) {
						break;
					}
					bytes += number(next + 8, 3, false);
				}
				if (end == block || !read(static_cast<uint32_t>(at / partition_size),
				                          at % partition_size, bytes, stored)) {
					return false;
				}
				out.insert(out.end(), stored.begin(), stored.end());
				block = end;
				continue;
			}
			if (method > method_count ||
			    !read(static_cast<uint32_t>(at / partition_size), at % partition_size,
			          number(entry + 5, 3, false), stored)) {
				return false;
			}
			const auto* name = reinterpret_cast<const char*>(toc.data() + methods_at +
			                                                 (method - 1u) * method_length);
			const auto  from = out.size();
			out.resize(from + size);
			if (!decompress(std::string_view(name, strnlen(name, method_length)),
			                std::span<const uint8_t>(stored),
			                std::span<uint8_t>(out).subspan(from))) {
				return false;
			}
			block++;
		}
		const auto skip = offset % block_size;
		if (out.size() < skip + length) {
			return false;
		}
		out.erase(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(skip));
		out.resize(length);
		return true;
	};
	constexpr uint8_t                              LibraryChunk = 8, GroupChunk = 9;
	std::unordered_map<std::string_view, uint32_t> groups;
	for (uint32_t chunk = 0; chunk < chunks; chunk++) {
		if (static_cast<uint8_t>(id(chunk)[11]) == GroupChunk) {
			groups.emplace(id(chunk), chunk);
		}
	}
	std::vector<uint8_t> library, group, shaders;
	for (uint32_t chunk = 0; chunk < chunks; chunk++) {
		if (static_cast<uint8_t>(id(chunk)[11]) != LibraryChunk || !read_chunk(chunk, library)) {
			continue;
		}
		SpanReader reader {library};
		const auto array = [&](uint32_t element) {
			const auto count = reader.Get<uint32_t>();
			const auto at    = reader.at;
			reader.Skip(uint64_t {count} * element);
			return std::pair {at, count};
		};
		if (reader.Get<uint32_t>() != 1) {
			continue;
		}
		array(20);
		array(20);
		const auto [ids, id_count] = array(12);
		array(8);
		array(8);
		const auto [entries, group_count] = array(16);
		array(4);
		if (!reader.ok || id_count != group_count) {
			continue;
		}
		for (uint32_t i = 0; i < group_count; i++) {
			const auto chunk_of = groups.find(
			    std::string_view(reinterpret_cast<const char*>(library.data() + ids + i * 12), 12));
			uint32_t size   = 0;
			uint32_t length = 0;
			std::memcpy(&size, library.data() + entries + i * 16 + 8, 4);
			std::memcpy(&length, library.data() + entries + i * 16 + 12, 4);
			if (chunk_of == groups.end() || size > MaxChunk ||
			    !read_chunk(chunk_of->second, group) || group.size() != length) {
				continue;
			}
			if (length == size) {
				found(std::span<const uint8_t>(group));
				continue;
			}
			shaders.resize(size);
			if (decompress(std::string_view(), std::span<const uint8_t>(group),
			               std::span<uint8_t>(shaders))) {
				found(std::span<const uint8_t>(shaders));
			}
		}
	}
	return true;
}

constexpr std::string_view FrostbiteSignature("\x00\xd1\xce\x01", 4);
constexpr size_t           FrostbiteSignatureSize = 0x22c;

std::span<const uint8_t> FrostbiteUnsigned(std::span<const uint8_t> data) {
	if (data.size() >= FrostbiteSignatureSize &&
	    std::memcmp(data.data(), FrostbiteSignature.data(), FrostbiteSignature.size()) == 0) {
		return data.subspan(FrostbiteSignatureSize);
	}
	return data;
}

struct DbField {
	std::string_view         name;
	uint8_t                  type = 0;
	std::span<const uint8_t> value;
};

bool ReadDbField(SpanReader& reader, DbField& field) {
	const auto type = reader.Get<uint8_t>();
	if (!reader.ok || type == 0) {
		return false;
	}
	field.type = type & 0x1fu;
	field.name = {};
	if ((type & 0x80u) == 0) {
		const auto* begin = reinterpret_cast<const char*>(reader.data.data() + reader.at);
		const auto  name  = std::string_view(begin, strnlen(begin, reader.data.size() - reader.at));
		if (!reader.Skip(name.size() + 1)) {
			return false;
		}
		field.name = name;
	}
	uint64_t size = 0;
	switch (field.type) {
		case 0x01:
		case 0x02:
		case 0x07:
		case 0x13:
			for (uint32_t shift = 0; shift < 35; shift += 7) {
				const auto byte = reader.Get<uint8_t>();
				size |= uint64_t {byte & 0x7fu} << shift;
				if ((byte & 0x80u) == 0) {
					break;
				}
			}
			break;
		case 0x06: size = 1; break;
		case 0x08:
		case 0x0b: size = 4; break;
		case 0x09:
		case 0x0c: size = 8; break;
		case 0x0f: size = 16; break;
		case 0x10: size = 20; break;
		default: return false;
	}
	const auto at = reader.at;
	if (!reader.ok || !reader.Skip(size)) {
		return false;
	}
	field.value = reader.data.subspan(at, size);
	return true;
}

template <typename F>
void ForEachDbField(std::span<const uint8_t> value, F&& f) {
	SpanReader reader {value};
	DbField    field;
	while (ReadDbField(reader, field)) {
		f(field);
	}
}

std::string_view DbString(const DbField& field) {
	return field.type == 0x07 && !field.value.empty()
	           ? std::string_view(reinterpret_cast<const char*>(field.value.data()),
	                              field.value.size() - 1)
	           : std::string_view();
}

struct FrostbiteLayout {
	std::string                               platform;
	std::vector<std::string>                  superbundles;
	std::unordered_map<uint32_t, std::string> chunks;
};

bool ReadFrostbiteLayout(std::span<const uint8_t> data, FrostbiteLayout& layout) {
	SpanReader reader {FrostbiteUnsigned(data)};
	DbField    root;
	if (!ReadDbField(reader, root) || root.type != 0x02) {
		return false;
	}
	ForEachDbField(root.value, [&](const DbField& field) {
		if (field.name == "superBundles" && field.type == 0x01) {
			ForEachDbField(field.value, [&](const DbField& superbundle) {
				ForEachDbField(superbundle.value, [&](const DbField& item) {
					if (item.name == "name" && !DbString(item).empty()) {
						layout.superbundles.emplace_back(DbString(item));
					}
				});
			});
		}
		if (field.name != "installManifest" || field.type != 0x02) {
			return;
		}
		ForEachDbField(field.value, [&](const DbField& item) {
			if (item.name == "platform") {
				layout.platform = DbString(item);
			}
			if (item.name != "installChunks" || item.type != 0x01) {
				return;
			}
			ForEachDbField(item.value, [&](const DbField& chunk) {
				std::string_view name;
				uint32_t         index = 0;
				bool             known = false;
				ForEachDbField(chunk.value, [&](const DbField& key) {
					if (key.name == "name") {
						name = DbString(key);
					} else if (key.name == "persistentIndex" && key.type == 0x08) {
						std::memcpy(&index, key.value.data(), 4);
						known = true;
					}
				});
				if (known && !name.empty()) {
					layout.chunks.emplace(index, name);
				}
			});
		});
	});
	return !layout.platform.empty() && !layout.chunks.empty();
}

struct FrostbiteEntry {
	bool     patch;
	uint32_t chunk;
	uint32_t cas;
	uint64_t offset;
	uint32_t size;
};

template <typename Read, typename Found, typename Stop>
void ForEachFrostbiteShaderEntry(const std::vector<std::vector<uint8_t>>& tocs, uint32_t quiet,
                                 Read&& read, Found&& found, Stop&& stop) {
	const auto big = [](std::span<const uint8_t> data, size_t at, uint32_t bytes) {
		uint64_t value = 0;
		for (uint32_t i = 0; i < bytes; i++) {
			value = (value << 8u) | data[at + i];
		}
		return value;
	};
	struct Bundle {
		std::span<const uint8_t> data;
		uint32_t                 entries;
	};
	std::vector<Bundle> bundles;
	for (const auto& file: tocs) {
		const auto toc = FrostbiteUnsigned(file);
		if (toc.size() < 0x3c || big(toc, 0, 4) != 0x3c) {
			continue;
		}
		const auto at    = big(toc, 4, 4);
		const auto count = big(toc, 8, 4);
		for (uint64_t i = 0; i < count && at + (i + 1) * 16 <= toc.size(); i++) {
			const auto         size   = big(toc, at + i * 16 + 4, 4);
			const auto         offset = big(toc, at + i * 16 + 8, 8);
			constexpr uint32_t InToc  = 0x40000000;
			if ((size & InToc) == 0 || offset > toc.size() ||
			    (size & 0xffffffu) > toc.size() - offset || (size & 0xffffffu) < 0x24) {
				continue;
			}
			const auto data = toc.subspan(offset, size & 0xffffffu);
			bundles.push_back({data, static_cast<uint32_t>(big(data, 12, 4))});
		}
	}
	std::ranges::stable_sort(bundles, std::ranges::greater {}, &Bundle::entries);
	std::unordered_map<uint32_t, bool> shader_type;
	std::vector<FrostbiteEntry>        entries, shaders;
	std::vector<uint8_t>               head, table, name;
	uint32_t                           silent = 0;
	for (const auto& bundle: bundles) {
		if (silent >= quiet || stop()) {
			break;
		}
		silent++;
		const auto flags_at = big(bundle.data, 8, 4);
		const auto count    = big(bundle.data, 12, 4);
		auto       at       = big(bundle.data, 16, 4);
		if (count < 2 || flags_at > bundle.data.size() || count > bundle.data.size() - flags_at) {
			continue;
		}
		entries.clear();
		FrostbiteEntry cas {};
		bool           named = false;
		for (uint64_t i = 0; i < count; i++) {
			const bool next = (bundle.data[flags_at + i] & 0x80u) != 0;
			if (at + (next ? 16 : 8) > flags_at) {
				break;
			}
			if (next) {
				cas = {bundle.data[at + 1] != 0, static_cast<uint32_t>(big(bundle.data, at + 2, 4)),
				       static_cast<uint32_t>(big(bundle.data, at + 6, 2)), 0, 0};
				named = true;
				at += 8;
			}
			if (!named) {
				break;
			}
			cas.offset = big(bundle.data, at, 4);
			cas.size   = static_cast<uint32_t>(big(bundle.data, at + 4, 4));
			entries.push_back(cas);
			at += 8;
		}
		if (entries.size() != count) {
			continue;
		}
		const auto& meta = entries[0];
		if (!read(meta, meta.offset, 0x24, head) || head.size() != 0x24 ||
		    big(head, 0, 4) + 4 != meta.size) {
			continue;
		}
		uint32_t counts[5] = {};
		std::memcpy(counts, head.data() + 8, sizeof(counts));
		const auto [total, ebx, res, chunk_count, names_at] = counts;
		if (uint64_t {ebx} + res + chunk_count != total || uint64_t {total} + 1 != count ||
		    res == 0) {
			continue;
		}
		const auto res_at = uint64_t {0x24} + uint64_t {total} * 20 + uint64_t {ebx} * 8;
		if (res_at + uint64_t {res} * 12 > meta.size ||
		    !read(meta, meta.offset + res_at, uint64_t {res} * 12, table) ||
		    table.size() != uint64_t {res} * 12) {
			continue;
		}
		shaders.clear();
		for (uint32_t i = 0; i < res; i++) {
			uint32_t type = 0;
			std::memcpy(&type, table.data() + uint64_t {res} * 8 + i * 4, 4);
			auto known = shader_type.find(type);
			if (known == shader_type.end()) {
				constexpr std::string_view Shaders("shaders/bytecode/");
				uint32_t                   name_at = 0;
				std::memcpy(&name_at, table.data() + i * 8, 4);
				const auto from = uint64_t {4} + names_at + name_at;
				known = shader_type
				            .emplace(type,
				                     from + Shaders.size() <= meta.size &&
				                         read(meta, meta.offset + from, Shaders.size(), name) &&
				                         std::ranges::equal(name, Shaders,
				                                            [](uint8_t a, char b) {
					                                            return a == static_cast<uint8_t>(b);
				                                            }))
				            .first;
			}
			if (known->second) {
				shaders.push_back(entries[1 + ebx + i]);
			}
		}
		if (!shaders.empty() && found(std::span<const FrostbiteEntry>(shaders))) {
			silent = 0;
		}
	}
}

}

bool PipelineCache::ScanGameFiles(const std::stop_token&                                          stop,
                                  const std::function<void(uint64_t, std::span<const uint32_t>)>& append) {
	auto& state = *m_manifest;
	std::filesystem::path root;
	for (uint32_t tries = 0; tries < 200 && !stop.stop_requested(); tries++) {
		root = LibKernel::FileSystem::GetRealFilename("/app0");
		if (!root.empty()) {
			break;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
	}
	if (root.empty()) {
		return false;
	}
	constexpr uint32_t SampleFiles = 32;
	constexpr uint64_t SampleBytes = 256ull << 20u;
	constexpr uint32_t Piece   = 64u << 20u;
	constexpr uint32_t Overlap = 4u << 20u;
	constexpr uint32_t ProbeBytes  = 1u << 20u;
	constexpr uint64_t ProbeStride = 32ull << 20u;
	constexpr uint32_t RunPiece    = 8u << 20u;
	constexpr uint64_t RunQuiet    = 16ull << 20u;
	constexpr uint32_t ProbeFiles = 256;
	const auto         start      = std::chrono::steady_clock::now();
	size_t             listed     = 0;
	{
		std::scoped_lock lock(state.mutex);
		listed = state.code.size();
	}
	std::unordered_set<uint64_t> found;
	std::unordered_set<uint64_t> found_listed;
	uint32_t                     provided = 0;
	uint32_t                     files    = 0;
	uint64_t                     bytes    = 0;
	std::vector<uint8_t>         data;
	std::vector<uint32_t>        code;
	const auto search = [&](std::span<const uint8_t> chunk) {
		return FindShaderPrograms(chunk, [&](std::span<const uint8_t> program) {
			const auto hash = XXH3_64bits(program.data(), program.size());
			if (!found.insert(hash).second) {
				return;
			}
			{
				std::scoped_lock lock(state.mutex);
				const auto       entry = state.code.find(hash);
				if (entry == state.code.end()) {
					return;
				}
				found_listed.insert(hash);
				if (entry->second.mapped) {
					return;
				}
			}
			code.resize(program.size() / sizeof(uint32_t));
			std::memcpy(code.data(), program.data(), program.size());
			if (ProvideCode(hash, code, false)) {
				provided++;
				append(hash, code);
			}
		});
	};
	auto       next_count      = start;
	bool       missing         = true;
	const auto nothing_missing = [&] {
		if (const auto now = std::chrono::steady_clock::now(); now >= next_count) {
			next_count = now + std::chrono::milliseconds(250);
			std::scoped_lock lock(state.mutex);
			missing = std::ranges::any_of(state.code | std::views::values,
			                              [](const auto& entry) { return !entry.mapped; });
		}
		return !missing;
	};
	struct Run {
		uint64_t end;
		uint32_t shaders;
		uint64_t silent;
	};
	const std::unique_ptr<ZSTD_DCtx, decltype(&ZSTD_freeDCtx)> frames(ZSTD_createDCtx(), &ZSTD_freeDCtx);
	std::vector<uint8_t>                                       frame;
	const auto search_frames = [&](std::span<const uint8_t> chunk) {
		constexpr std::string_view Magic("\x28\xb5\x2f\xfd", 4);
		constexpr uint64_t         MaxFrame = 64u << 20u;
		const std::string_view     text(reinterpret_cast<const char*>(chunk.data()), chunk.size());
		uint32_t                   shaders = 0;
		for (size_t at = text.find(Magic); at != std::string_view::npos && frames != nullptr;
		     at        = text.find(Magic, at + 1)) {
			const auto stored = ZSTD_findFrameCompressedSize(chunk.data() + at, chunk.size() - at);
			if (ZSTD_isError(stored) != 0) {
				continue;
			}
			const auto size = ZSTD_getFrameContentSize(chunk.data() + at, stored);
			if (size == ZSTD_CONTENTSIZE_UNKNOWN || size == ZSTD_CONTENTSIZE_ERROR || size > MaxFrame) {
				continue;
			}
			frame.resize(size);
			const auto done = ZSTD_decompressDCtx(frames.get(), frame.data(), frame.size(), chunk.data() + at, stored);
			if (ZSTD_isError(done) == 0) {
				shaders += search(std::span<const uint8_t>(frame).first(done));
				at += stored - 1;
			}
		}
		return shaders;
	};
	const auto read_run = [&](Common::File& file, uint64_t from, uint64_t reach, uint64_t quiet,
	                          uint64_t end, uint32_t piece, uint64_t quiet_after) {
		Run    run {from, 0, 0};
		size_t kept = 0;
		if (!file.Seek(from)) {
			return run;
		}
		while (run.end < end && !stop.stop_requested()) {
			const auto size = static_cast<uint32_t>(std::min<uint64_t>(end - run.end, piece));
			data.resize(kept + size);
			uint32_t read = 0;
			file.Read(data.data() + kept, size, &read);
			if (read != size) {
				break;
			}
			run.end += size;
			bytes += size;
			const auto shaders = search(data) + search_frames(data);
			run.shaders += shaders;
			run.silent = shaders != 0 ? 0 : run.silent + size;
			if (shaders != 0 && quiet_after != 0) {
				quiet = quiet_after;
			}
			if (run.end >= reach && run.silent >= quiet) {
				break;
			}
			kept = std::min<size_t>(data.size(), Overlap);
			std::memmove(data.data(), data.data() + data.size() - kept, kept);
		}
		return run;
	};
	std::vector<uint8_t> shader;
	const auto           search_archive = [&](std::span<const uint8_t> archive) {
		ForEachUnrealShader(archive, [&](std::span<const uint8_t> stored, uint32_t size) {
			if (stored.size() == size) {
				search(stored);
				return;
			}
			if (size > (16u << 20u)) {
				return;
			}
			shader.resize(size);
			if (Lz4Decompress(stored, shader)) {
				search(shader);
			}
		});
	};
	const auto read_file = [&](Common::File& file, uint64_t offset, uint64_t size, std::vector<uint8_t>& out) {
		uint32_t done = 0;
		if (size > (1u << 30u) || !file.Seek(offset)) {
			return false;
		}
		out.resize(size);
		file.Read(out.data(), static_cast<uint32_t>(size), &done);
		bytes += done;
		return done == size;
	};
	const auto search_unreal = [&](const std::filesystem::path& path, const std::string& extension) {
		Common::File file;
		if (!file.Open(path, Common::File::Mode::Read)) {
			return false;
		}
		const auto read = [&](uint64_t offset, uint64_t size, std::vector<uint8_t>& out) {
			return read_file(file, offset, size, out);
		};
		std::vector<uint8_t> archive;
		if (extension == ".pak") {
			return ForEachUnrealEntry(file.Size(), read, search_archive, [&](std::span<const uint8_t> asset) {
				search(asset);
				FindCompressedShaders(asset, archive, [&](std::span<const uint8_t> shader) { search(shader); });
			});
		}
		if (!read(0, file.Size(), archive)) {
			return false;
		}
		search_archive(archive);
		return true;
	};
	const auto search_unity = [&](const std::filesystem::path& path) {
		Common::File file;
		if (!file.Open(path, Common::File::Mode::Read)) {
			return;
		}
		const auto read = [&](uint64_t offset, uint64_t size, std::vector<uint8_t>& out) {
			return read_file(file, offset, size, out);
		};
		ForEachUnityShaderFile(file.Size(), read, [&](std::span<const uint8_t> serialized) {
			FindUnityShaders(serialized, shader, [&](std::span<const uint8_t> code) { search(code); });
		});
	};
	const auto iostore_toc = [](const std::filesystem::path& path) {
		auto toc = path;
		toc.replace_extension(".utoc");
		const auto stem = Common::PathToString(path.stem());
		const auto cut  = stem.rfind("_s");
		if (path.extension() == ".ucas" && !Common::File::IsFileExisting(toc) &&
		    cut != std::string::npos && cut + 2 < stem.size() &&
		    std::all_of(stem.begin() + static_cast<std::ptrdiff_t>(cut + 2), stem.end(),
		                [](unsigned char c) { return std::isdigit(c) != 0; })) {
			toc.replace_filename(Common::PathFromUtf8(stem.substr(0, cut) + ".utoc"));
		}
		return toc;
	};
	std::unordered_set<std::string> iostore_read;
	std::vector<uint8_t>            toc;
	const auto                      search_iostore = [&](const std::filesystem::path& toc_path) {
		Common::File toc_file;
		if (!toc_file.Open(toc_path, Common::File::Mode::Read) ||
		    !read_file(toc_file, 0, toc_file.Size(), toc)) {
			return false;
		}
		std::vector<std::unique_ptr<Common::File>> partitions;
		const auto read = [&](uint32_t partition, uint64_t offset, uint64_t size,
		                      std::vector<uint8_t>& out) {
			while (partitions.size() <= partition && partitions.size() < 64) {
				auto path = toc_path;
				path.replace_extension(".ucas");
				if (!partitions.empty()) {
					path.replace_filename(Common::PathFromUtf8(fmt::format(
					    "{}_s{}.ucas", Common::PathToString(toc_path.stem()), partitions.size())));
				}
				auto file = std::make_unique<Common::File>();
				if (!file->Open(path, Common::File::Mode::Read)) {
					file.reset();
				} else if (partitions.empty()) {
					files++;
				}
				partitions.push_back(std::move(file));
			}
			return partition < partitions.size() && partitions[partition] != nullptr &&
			       read_file(*partitions[partition], offset, size, out);
		};
		const auto named = [](std::string_view method, std::string_view name) {
			return std::ranges::equal(
			    method, name, [](char a, char b) { return std::tolower(a) == std::tolower(b); });
		};
		const auto decompress = [&](std::string_view method, std::span<const uint8_t> stored,
		                            std::span<uint8_t> out) {
			if (method.empty() ? stored.size() >= 2 && (stored[0] & 0x0fu) == 0x0c
			                   : named(method, "Oodle")) {
				return Loader::GuestOodle::Decompress(stored, out, stop);
			}
			if (named(method, "Zlib")) {
				return stbi_zlib_decode_buffer(
				           reinterpret_cast<char*>(out.data()), static_cast<int>(out.size()),
				           reinterpret_cast<const char*>(stored.data()),
				           static_cast<int>(stored.size())) == static_cast<int>(out.size());
			}
			return (method.empty() || named(method, "LZ4")) && Lz4Decompress(stored, out);
		};
		return ForEachIoStoreShaderGroup(toc, read, decompress,
		                                 [&](std::span<const uint8_t> group) { search(group); });
	};
	constexpr uint32_t FrostbiteQuiet   = 256;
	const auto         search_frostbite = [&] {
		FrostbiteLayout      layout;
		std::vector<uint8_t> text;
		for (const char* folder: {"Patch", "Data"}) {
			Common::File file;
			if (file.Open(root / folder / "layout.toc", Common::File::Mode::Read) &&
			    read_file(file, 0, file.Size(), text)) {
				files++;
				ReadFrostbiteLayout(text, layout);
			}
		}
		if (layout.chunks.empty()) {
			return false;
		}
		std::vector<std::vector<uint8_t>> tocs;
		std::unordered_set<std::string>   superbundles;
		for (const char* folder: {"Patch", "Data"}) {
			for (const auto& name: layout.superbundles) {
				Common::File file;
				if (superbundles.insert(std::string(folder) + "/" + name).second &&
				    file.Open(root / folder / Common::PathFromUtf8(name + ".toc"),
				              Common::File::Mode::Read) &&
				    read_file(file, 0, file.Size(), text)) {
					files++;
					tocs.push_back(std::move(text));
				}
			}
		}
		if (tocs.empty()) {
			return false;
		}
		using Cas = std::tuple<bool, uint32_t, uint32_t>;
		std::map<Cas, std::unique_ptr<Common::File>> cas_files;
		const auto read = [&](const FrostbiteEntry& entry, uint64_t offset, uint64_t size,
		                      std::vector<uint8_t>& out) {
			auto& file = cas_files[{entry.patch, entry.chunk, entry.cas}];
			if (file == nullptr) {
				file             = std::make_unique<Common::File>();
				const auto chunk = layout.chunks.find(entry.chunk);
				if (chunk == layout.chunks.end() ||
				    !file->Open(root / (entry.patch ? "Patch" : "Data") /
				                    Common::PathFromUtf8(layout.platform) /
				                    Common::PathFromUtf8(chunk->second) /
				                    Common::PathFromUtf8(fmt::format("cas_{:02}.cas", entry.cas)),
				                Common::File::Mode::Read)) {
					return false;
				}
				files++;
			}
			return !file->IsInvalid() && read_file(*file, offset, size, out);
		};
		std::set<std::tuple<bool, uint32_t, uint32_t, uint64_t>> seen;
		std::vector<FrostbiteEntry>                              todo;
		std::vector<uint8_t>                                     run;
		ForEachFrostbiteShaderEntry(
		    tocs, FrostbiteQuiet, read,
		    [&](std::span<const FrostbiteEntry> shaders) {
			    const auto known = found_listed.size();
			    todo.clear();
			    for (const auto& entry: shaders) {
				    if (seen.emplace(entry.patch, entry.chunk, entry.cas, entry.offset).second) {
					    todo.push_back(entry);
				    }
			    }
			    const auto order = [](const FrostbiteEntry& e) {
				    return std::tuple(e.patch, e.chunk, e.cas, e.offset);
			    };
			    std::ranges::sort(todo, {}, order);
			    for (size_t i = 0; i < todo.size() && !stop.stop_requested();) {
				    auto end  = todo[i].offset + todo[i].size;
				    auto next = i + 1;
				    for (; next < todo.size() && todo[next].patch == todo[i].patch &&
				           todo[next].chunk == todo[i].chunk && todo[next].cas == todo[i].cas &&
				           todo[next].offset == end && end - todo[i].offset < RunPiece;
				         next++) {
					    end += todo[next].size;
				    }
				    if (read(todo[i], todo[i].offset, end - todo[i].offset, run)) {
					    search(run);
				    }
				    i = next;
			    }
			    return found_listed.size() != known;
		    },
		    [&] { return stop.stop_requested() || nothing_missing(); });
		return true;
	};
	const bool frostbite = search_frostbite();
	struct Kind {
		uint32_t files  = 0;
		uint64_t bytes  = 0;
		uint32_t probed = 0;
	};
	std::unordered_map<std::string, Kind> kinds;
	std::vector<std::filesystem::path>    folders {root};
	struct Later {
		std::filesystem::path path;
		uint64_t              from;
		Kind*                 kind;
	};
	std::vector<Later> later;
	while (!folders.empty() && !stop.stop_requested() && !nothing_missing()) {
		const auto folder = std::move(folders.back());
		folders.pop_back();
		for (const auto& entry: Common::File::GetDirEntries(folder)) {
			if (stop.stop_requested() || nothing_missing()) {
				break;
			}
			if (entry.name == "." || entry.name == "..") {
				continue;
			}
			const auto path = folder / Common::PathFromUtf8(entry.name);
			if (!entry.is_file) {
				folders.push_back(path);
				continue;
			}
			auto extension = Common::PathToString(path.extension());
			std::ranges::transform(extension, extension.begin(),
			                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
			if (frostbite && (extension == ".cas" || extension == ".toc")) {
				continue;
			}
			if ((extension == ".pak" || extension == ".ushaderbytecode") && search_unreal(path, extension)) {
				continue;
			}
			if (extension == ".ucas" || extension == ".utoc") {
				const auto toc_path = iostore_toc(path);
				if (iostore_read.contains(Common::PathToString(toc_path))) {
					continue;
				}
				if (search_iostore(toc_path)) {
					iostore_read.insert(Common::PathToString(toc_path));
					continue;
				}
			}
			if (extension == ".bundle" || extension == ".unity3d" || extension == ".assets") {
				search_unity(path);
			}
			auto& kind = kinds[extension];
			if (kind.files >= SampleFiles || kind.bytes >= SampleBytes) {
				later.push_back({path, 0, &kind});
				continue;
			}
			Common::File file;
			if (!file.Open(path, Common::File::Mode::Read)) {
				continue;
			}
			files++;
			const auto size = file.Size();
			const auto run  = read_run(file, 0, 0, SampleBytes - kind.bytes, size, Piece, SampleBytes);
			if (run.shaders != 0) {
				kind.files = 0;
				kind.bytes = run.silent;
			} else {
				kind.files++;
				kind.bytes += run.end;
			}
			if (run.end < size) {
				later.push_back({path, run.end, &kind});
			}
		}
	}
	for (const auto& [path, from, kind]: later) {
		if (stop.stop_requested() || nothing_missing()) {
			break;
		}
		if (kind->probed >= ProbeFiles) {
			continue;
		}
		Common::File file;
		if (!file.Open(path, Common::File::Mode::Read)) {
			continue;
		}
		files += from == 0 ? 1 : 0;
		const auto size   = file.Size();
		const auto before = provided;
		uint64_t   read   = from;
		for (uint64_t at = from; at < size && !stop.stop_requested() && !nothing_missing();) {
			const auto window = std::min<uint64_t>(size, at + ProbeBytes);
			if (read_run(file, at, window, 0, window, ProbeBytes, 0).shaders == 0) {
				at += ProbeStride;
				continue;
			}
			const auto begin = std::max(read, at >= from + ProbeStride ? at - ProbeStride + ProbeBytes : from);
			const auto floor = read;
			const auto known = found_listed.size();
			read             = read_run(file, begin, window, RunQuiet, size, RunPiece, 0).end;
			for (uint64_t end = begin, silent = 0; found_listed.size() != known && end > floor &&
			                                       silent < RunQuiet && !stop.stop_requested();) {
				const auto back = std::max(floor, end - RunPiece);
				const auto run  = read_run(file, back, 0, 0, std::min(size, end + Overlap),
				                           RunPiece + Overlap, 0);
				silent          = run.shaders != 0 ? 0 : silent + (end - back);
				end             = back;
			}
			at = std::max(at + ProbeStride, read);
		}
		kind->probed = provided != before ? 0 : kind->probed + 1;
	}
	ManifestLog("[ShaderScan] game files: {} shaders found ({} of {} listed by the manifest, {} new), {:.1f} s "
	            "({} files, {:.0f} MB read)",
	            found.size(), found_listed.size(), listed, provided,
	            Seconds(std::chrono::steady_clock::now() - start), files,
	            static_cast<double>(bytes) / (1024.0 * 1024.0));
	return true;
}

void PipelineCache::CodeStoreThread(const std::stop_token& stop) {
	KYTY_PROFILER_THREAD("Thread_PipelineManifestCode");
	auto& state = *m_manifest;
	std::unordered_set<uint64_t> stored;
	std::unordered_set<uint64_t> not_in_files;
	uint64_t                     valid = 0;
	{
		const auto bytes = ReadFile(state.code_path);
		uint64_t   magic = 0;
		if (bytes.size() >= sizeof(magic)) {
			std::memcpy(&magic, bytes.data(), sizeof(magic));
		}
		uint32_t              provided = 0;
		std::vector<uint32_t> code;
		for (size_t at = sizeof(magic); magic == CodeStoreMagic;) {
			CodeRecord record {};
			if (bytes.size() - at < sizeof(record)) {
				break;
			}
			std::memcpy(&record, bytes.data() + at, sizeof(record));
			if (record.words == 0 || bytes.size() - at - sizeof(record) < record.compressed_bytes) {
				break;
			}
			const auto* compressed = bytes.data() + at + sizeof(record);
			at += sizeof(record) + record.compressed_bytes;
			valid = at;
			if (record.hash == 0) {
				not_in_files.clear();
				if (ReadCodeRecord(record, compressed, code) && code.size() >= 2 && code[0] == ScanMark &&
				    code[1] == ScanVersion) {
					for (size_t i = 2; i + 1 < code.size(); i += 2) {
						not_in_files.insert(code[i] | (uint64_t {code[i + 1]} << 32u));
					}
				}
				continue;
			}
			stored.insert(record.hash);
			bool wanted = false;
			{
				std::scoped_lock lock(state.mutex);
				const auto       found = state.code.find(record.hash);
				wanted = found != state.code.end() && !found->second.mapped;
			}
			if (wanted && !stop.stop_requested() && ReadCodeRecord(record, compressed, code) &&
			    XXH3_64bits(code.data(), code.size() * sizeof(uint32_t)) == record.hash) {
				provided += ProvideCode(record.hash, code, false) ? 1u : 0u;
			}
		}
		if (provided != 0) {
			ManifestLog("Pipeline manifest: the code of {} shaders comes from {} ({} shaders)", provided,
			            Common::PathToString(state.code_path), stored.size());
		}
	}
	Common::File file;
	bool         opened = false;
	const auto   append = [&](uint64_t hash, std::span<const uint32_t> code) {
		if (hash != 0 && !stored.insert(hash).second) {
			return;
		}
		if (!opened) {
			opened = true;
			Common::File::CreateDirectories(state.code_path.parent_path());
			if (valid != 0 && file.Open(state.code_path, Common::File::Mode::ReadWrite)) {
				file.Truncate(valid);
				file.Seek(valid);
			} else if (file.Create(state.code_path)) {
				file.Write(&CodeStoreMagic, sizeof(CodeStoreMagic));
			}
		}
		if (file.IsInvalid()) {
			return;
		}
		std::vector<uint8_t> bytes(sizeof(CodeRecord) + ZSTD_compressBound(code.size_bytes()));
		const auto size = ZSTD_compress(bytes.data() + sizeof(CodeRecord), bytes.size() - sizeof(CodeRecord),
		                                code.data(), code.size_bytes(), 3);
		if (ZSTD_isError(size)) {
			return;
		}
		const CodeRecord record {.hash             = hash,
		                         .words            = static_cast<uint32_t>(code.size()),
		                         .compressed_bytes = static_cast<uint32_t>(size)};
		std::memcpy(bytes.data(), &record, sizeof(record));
		file.Write(bytes.data(), static_cast<uint32_t>(sizeof(record) + size));
	};
	const auto listed_code = [&] {
		std::scoped_lock lock(state.mutex);
		return state.code.size();
	};
	const auto missing = [&] {
		std::vector<uint64_t> hashes;
		std::scoped_lock      lock(state.mutex);
		for (const auto& [hash, entry]: state.code) {
			if (!entry.mapped) {
				hashes.push_back(hash);
			}
		}
		std::ranges::sort(hashes);
		return hashes;
	};
	if (const auto still = missing();
	    std::ranges::any_of(still, [&](uint64_t hash) { return !not_in_files.contains(hash); })) {
		if (ScanGameFiles(stop, append) && !stop.stop_requested()) {
			std::vector<uint32_t> words {ScanMark, ScanVersion};
			for (const auto hash: missing()) {
				words.push_back(static_cast<uint32_t>(hash));
				words.push_back(static_cast<uint32_t>(hash >> 32u));
			}
			append(0, words);
		}
	} else if (still.empty()) {
		ManifestLog("[ShaderScan] game files: not searched (the code store has all {} shaders the manifest lists)",
		            listed_code());
	} else {
		ManifestLog("[ShaderScan] game files: not searched ({} of the {} shaders the manifest lists are missing, "
		            "and the last search did not find them)",
		            still.size(), listed_code());
	}
	state.scanning.store(false, std::memory_order_release);
	Common::Thread::LowerCurrentPriority();
	for (;;) {
		std::pair<uint64_t, std::vector<uint32_t>> job;
		{
			std::unique_lock lock(state.code_mutex);
			state.code_available.wait(lock, stop, [&] { return !state.code_queue.empty(); });
			if (state.code_queue.empty()) {
				return;
			}
			job = std::move(state.code_queue.front());
			state.code_queue.pop_front();
		}
		append(job.first, job.second);
	}
}

void PipelineCache::ManifestWorker(const std::stop_token& stop) {
	KYTY_PROFILER_THREAD("Thread_PipelineManifest");
	Common::Thread::LowerCurrentPriority();
	auto& state = *m_manifest;
	for (;;) {
		std::shared_ptr<const std::vector<uint32_t>> code;
		std::shared_ptr<const std::vector<uint32_t>> back_code;
		uint32_t                                     index    = 0;
		bool                                         pipeline = false;
		{
			std::unique_lock lock(state.mutex);
			if (!state.available.wait(lock, stop, [&] {
				    return !state.pipeline_jobs.empty() || !state.module_jobs.empty();
			    })) {
				return;
			}
			pipeline = !state.pipeline_jobs.empty() &&
			           (state.module_jobs.empty() ||
			            state.PipelineRank(state.pipeline_jobs.front()) <=
			                state.module_rank[state.module_jobs.front()]);
			auto& jobs = pipeline ? state.pipeline_jobs : state.module_jobs;
			index      = jobs.front();
			jobs.pop_front();
			if (!pipeline) {
				const auto& module = state.warm_modules[index];
				for (auto [hash, words]: {std::pair {module.code_hash, &code},
				                          std::pair {module.back_hash, &back_code}}) {
					if (hash == 0) {
						continue;
					}
					auto& entry = state.code.at(hash);
					*words      = entry.words;
					if (--entry.readers == 0) {
						entry.words = nullptr;
					}
				}
			}
			state.running++;
		}
		if (pipeline) {
			CreateManifestPipeline(index);
		} else {
			BuildManifestModule(index, *code,
			                    back_code ? std::span<const uint32_t>(*back_code)
			                              : std::span<const uint32_t> {});
		}
		{
			std::scoped_lock lock(state.mutex);
			state.running--;
		}
		ManifestProgress(false);
	}
}

void PipelineCache::BuildManifestModule(uint32_t index, std::span<const uint32_t> code,
                                        std::span<const uint32_t> back_code, bool guest_traversal) {
	auto&       state  = *m_manifest;
	const auto& record = [&]() -> const ManifestModule& {
		std::scoped_lock lock(state.mutex);
		return state.warm_modules[index];
	}();
	const auto start = std::chrono::steady_clock::now();
	auto        input  = record.input;
	const std::vector<uint32_t>      user_data(record.user_data_count);
	ShaderRecompiler::CompileOptions options;
	ProgramKey                       key {.stage           = record.stage,
	                                      .hash            = record.ProgramHash(),
	                                      .user_data_count = record.user_data_count};
	std::visit(
	    [&]<typename InputInfo>(InputInfo& info) {
		    options = StageCompileOptions(info);
		    BuildStageStaticKey(info, key.static_state);
	    },
	    input);
	const bool native = RayTracing::NativeTraversals() && !guest_traversal;
	AppendNativeRayTracingKey(options.stage, key.static_state, native);
	SetShaderCompileOptions(options, key.hash, user_data, back_code, native);
	std::vector<uint32_t> spirv;
	std::vector<uint32_t> plain_spirv;

	const auto finish = [&](const ShaderModule* module, bool built) {
		if (guest_traversal) {
			if (module != nullptr && module->info.stage == ShaderType::Compute) {
				CreateManifestComputePipeline(*module, record.from_assets);
			}
			return;
		}
		const auto elapsed =
		    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                              std::chrono::steady_clock::now() - start)
		                              .count());
		const bool recording = !state.record_path.empty();
		const auto identity  = recording ? ManifestModuleIdentity(record, StaticKey(record)) : 0u;
		{
			std::scoped_lock lock(state.mutex);
			state.modules[index] = module;
			if (module == nullptr) {
				state.module_failed[index] = 1;
				state.failed++;
			} else {
				(built ? state.built : state.found)++;
				state.build_ns += built ? elapsed : 0u;
				if (built) {
					state.module_time.Add(static_cast<double>(elapsed) / 1e9);
				}
				if (recording) {
					const auto [found, inserted] = state.module_index.try_emplace(
					    identity, static_cast<uint32_t>(state.recorded.modules.size()));
					if (inserted) {
						state.recorded.modules.push_back(record);
					}
					state.module_records[module->id] = {found->second, false};
					if (module->plain != nullptr) {
						state.module_records[module->plain->id] = {found->second, true};
					}
					state.module_generation.fetch_add(1, std::memory_order_release);
				}
			}
			for (const auto pipeline: state.module_pipelines[index]) {
				if (module == nullptr) {
					state.missing_modules[pipeline] = UINT8_MAX;
				} else if (state.missing_modules[pipeline] != UINT8_MAX &&
				           --state.missing_modules[pipeline] == 0) {
					state.QueuePipeline(pipeline);
				}
			}
		}
		state.available.notify_all();
		if (module != nullptr && module->info.stage == ShaderType::Compute) {
			CreateManifestComputePipeline(*module, record.from_assets);
		}
		if (native && record.stage == ShaderType::Compute &&
		    (module == nullptr || module->info.info.ray_query)) {
			BuildManifestModule(index, code, back_code, true);
		}
	};
	auto                specialization = record.specialization;
	const ShaderModule* compiled       = nullptr;
	LockPrograms();
	compiled = FindCompiledModule(key, specialization, record.push_data_cursor);
	UnlockPrograms();
	if (compiled != nullptr) {
		finish(compiled, false);
		return;
	}
	std::optional<ShaderRecompiler::IR::ResourcePlan> plan;
	std::optional<ShaderModule>                       module;
	const auto what = fmt::format("module {:016x}", record.code_hash);
	RunDroppable(state.dropped, record.from_assets, record.code_hash, 0, what, [&] {
		auto source = ShaderRecompiler::PrepareShaderSource(code, options);
		if (!source.calls.empty() || source.unsupported_call_pc != UINT32_MAX) {
			return;
		}
		auto translated = ShaderRecompiler::TranslateProgram(source.decoded, options);
		if (!ShaderRecompiler::IR::ResourceSpecializationFits(translated.program,
		                                                      record.specialization)) {
			return;
		}
		plan            = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
		if (record.stage == ShaderType::Compute && RuntimeStride(key.hash)) {
			ShaderRecompiler::IR::ReadStridesAtRunTime(*plan, specialization);
			LockPrograms();
			compiled = FindCompiledModule(key, specialization, record.push_data_cursor);
			UnlockPrograms();
			if (compiled != nullptr) {
				return;
			}
		}
#if defined(ANYPS5_KYTY)
		const auto recorded = specialization;
		ShaderRecompiler::IR::ShareNullBufferVariants(
		    *plan, specialization, record.stage == ShaderType::Compute && RuntimeStride(key.hash));
		if (specialization != recorded) {
			LockPrograms();
			compiled = FindCompiledModule(key, specialization, record.push_data_cursor);
			UnlockPrograms();
			if (compiled != nullptr) {
				return;
			}
		}
#endif
		module          = BuildModule(key, options, std::move(translated), specialization, {},
		                              record.push_data_cursor, spirv, plain_spirv);
	});
	if (compiled != nullptr) {
		finish(compiled, false);
		return;
	}
	if (!module) {
		finish(nullptr, false);
		return;
	}
	bool added = false;
	{
		LockPrograms();
		auto entry = m_program_cache.find(key);
		if (entry == m_program_cache.end()) {
			entry =
			    m_program_cache.try_emplace(key, Program {.resource_plan = std::move(*plan)}).first;
		}
		auto& program = entry->second;
		compiled      = FindCompiledModule(key, specialization, record.push_data_cursor);
		if (compiled != nullptr || program.skip_dispatch || program.call_source != nullptr) {
			DestroyModule(*module);
		} else {
			auto& added_module = program.modules.emplace_back(std::move(*module));
			added_module.plan  = &program.resource_plan;
			compiled           = &added_module;
			StoreModule(key, program, *compiled, spirv, plain_spirv);
			added = true;
		}
		UnlockPrograms();
	}
	finish(compiled, added);
}

void PipelineCache::CreateManifestPipeline(uint32_t index) {
	auto&                              state  = *m_manifest;
	std::array<const ShaderModule*, 4> stages {};
	const auto& record = [&]() -> const ManifestPipeline& {
		std::scoped_lock lock(state.mutex);
		return state.warm_pipelines[index];
	}();
	auto key = record.key;
	{
		std::scoped_lock lock(state.mutex);
		for (size_t i = 0; i < stages.size(); i++) {
			if (record.stages[i] != ManifestPipeline::NoStage) {
				stages[i] = state.modules[record.stages[i]];
			}
		}
	}
	if (record.plain_pixel && stages[3] != nullptr) {
		stages[3] = stages[3]->plain.get();
	}
	if (stages[0] == nullptr ||
	    (record.stages[3] != ManifestPipeline::NoStage && stages[3] == nullptr)) {
		std::scoped_lock lock(state.mutex);
		state.pipelines_failed++;
		return;
	}
	for (size_t i = 0; i < stages.size(); i++) {
		key.stage_ids[i] = stages[i] != nullptr ? stages[i]->id : 0u;
	}
	{
		std::scoped_lock lock(m_async_mutex);
		if (!m_claimed_graphics.insert(key).second) {
			std::scoped_lock manifest_lock(state.mutex);
			state.pipelines_known++;
			return;
		}
		m_async_pending_graphics.insert(key);
	}
	const auto start = std::chrono::steady_clock::now();
	Pipeline   pipeline;
	const auto key_hash = static_cast<uint64_t>(GraphicsPipelineKeyHash {}(key));
	const auto what     = fmt::format("pipeline {:016x}", key_hash);
	const bool ran = RunDroppable(state.dropped, record.from_assets, stages[0]->info.shader_hash,
	                              key_hash, what, [&] {
		CreatePipelineInternal(m_graphics, pipeline, key, stages, m_driver_cache);
	});
	const bool created = ran && pipeline.pipeline != nullptr && pipeline.pipeline_layout != nullptr;
	if (created) {
		NoteCreatedPipeline(start);
	}
	{
		std::scoped_lock lock(m_async_mutex);
		if (created) {
			m_async_ready.try_emplace(key, pipeline);
		} else if (ran) {
			m_async_failed_graphics.insert(key);
		} else {
			m_claimed_graphics.erase(key);
		}
		m_async_pending_graphics.erase(key);
	}
	const auto elapsed = std::chrono::steady_clock::now() - start;
	std::scoped_lock lock(state.mutex);
	(created ? state.pipelines : state.pipelines_failed)++;
	state.pipeline_ns += static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
	if (created) {
		state.pipeline_time.Add(std::chrono::duration<double>(elapsed).count());
	}
}

void PipelineCache::CreateManifestComputePipeline(const ShaderModule& module, bool from_assets) {
#if defined(ANYPS5_KYTY)
	std::atomic_ref<bool>(module.pipeline_requested).store(true, std::memory_order_relaxed);
#endif
	{
		std::scoped_lock lock(m_async_mutex);
		if (!m_claimed_compute.insert(module.id).second) {
			return;
		}
		m_async_pending_compute.insert(module.id);
	}
	const auto start = std::chrono::steady_clock::now();
	Pipeline   pipeline;
	auto&      state = *m_manifest;
	const auto what  = fmt::format("compute pipeline {:016x}", module.info.shader_hash);
	const bool ran   = RunDroppable(state.dropped, from_assets, module.info.shader_hash, 0, what, [&] {
		CreatePipelineInternal(m_graphics, pipeline, module, m_driver_cache);
	});
	const bool created = ran && pipeline.pipeline != nullptr && pipeline.pipeline_layout != nullptr;
	if (created) {
		NoteCreatedPipeline(start);
	}
	{
		std::scoped_lock lock(m_async_mutex);
		if (created) {
			m_async_ready_compute.try_emplace(module.id, pipeline);
		} else if (!ran) {
			m_claimed_compute.erase(module.id);
		}
		m_async_pending_compute.erase(module.id);
	}
	std::scoped_lock lock(state.mutex);
	state.compute += created ? 1u : 0u;
	state.pipeline_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                               std::chrono::steady_clock::now() - start)
	                                               .count());
}

PipelineCache::ManifestBacklog PipelineCache::GetManifestBacklog() const {
	ManifestBacklog backlog;
	if (m_manifest == nullptr) {
		return backlog;
	}
	auto&            state = *m_manifest;
	std::scoped_lock lock(state.mutex);
	backlog.modules   = static_cast<uint32_t>(state.module_jobs.size());
	backlog.pipelines = static_cast<uint32_t>(state.pipeline_jobs.size());
	backlog.running   = state.running;
	backlog.scanning  = state.scanning.load(std::memory_order_acquire);
	for (const auto index: state.module_jobs) {
		backlog.compute_modules += state.warm_modules[index].stage == ShaderType::Compute ? 1u : 0u;
	}
	backlog.modules_done     = state.built + state.found + state.failed;
	backlog.pipelines_done   = state.pipelines + state.pipelines_known + state.pipelines_failed;
	backlog.threads          = static_cast<uint32_t>(state.threads.size());
	backlog.module_seconds   = state.module_time.Get();
	backlog.pipeline_seconds = state.pipeline_time.Get();
	return backlog;
}

void PipelineCache::ManifestProgress(bool force) {
	auto&            state = *m_manifest;
	std::scoped_lock lock(state.mutex);
	const auto       now = std::chrono::steady_clock::now();
	const bool       idle =
	    state.module_jobs.empty() && state.pipeline_jobs.empty() && state.running == 0;
	if (!force &&
	    (state.logged_idle || (!idle && now - state.last_log < std::chrono::seconds(2)))) {
		return;
	}
	state.last_log   = now;
	uint32_t waiting = 0;
	for (const auto missing: state.missing_code) {
		waiting += missing != 0 ? 1u : 0u;
	}
	const auto modules   = state.warm_modules.size();
	const auto pipelines = state.warm_pipelines.size();
	if (!idle && !force) {
		ManifestLog("Pipeline manifest: {}/{} modules, {}/{} pipelines ({} modules wait for "
		            "their shaders) at {:.1f} s",
		            state.built + state.found + state.failed, modules,
		            state.pipelines + state.pipelines_known + state.pipelines_failed, pipelines,
		            waiting, Seconds(now - state.start));
		return;
	}
	state.logged_idle = idle;
	ManifestLog("Pipeline manifest: {} at {:.1f} s: {} modules compiled ({:.0f} ms), {} compiled "
	            "already, {} failed, {} waiting for their shaders; {} pipelines and {} compute "
	            "pipelines created ({:.0f} ms), {} created already, {} failed; {} records dropped "
	            "after a fatal error",
	            idle ? "idle" : "stopped", Seconds(now - state.start), state.built,
	            Milliseconds(state.build_ns), state.found, state.failed, waiting, state.pipelines,
	            state.compute, Milliseconds(state.pipeline_ns), state.pipelines_known,
	            state.pipelines_failed, state.dropped.load(std::memory_order_relaxed));
}

void PipelineCache::RecordModule(const ProgramKey&                       key,
                                 const ShaderRecompiler::CompileOptions& options,
                                 std::span<const uint32_t> code, uint32_t push_data_cursor,
                                 const ShaderModule& module) {
	if (m_manifest == nullptr || m_manifest->record_path.empty()) {
		return;
	}
	ManifestModule record {.stage            = key.stage,
	                       .code_hash        = key.hash,
	                       .user_data_count  = key.user_data_count,
	                       .push_data_cursor = push_data_cursor,
	                       .specialization   = module.specialization};
	if (!options.back_code.empty()) {
		record.code_hash = XXH3_64bits(code.data(), code.size_bytes());
		record.back_hash = XXH3_64bits(options.back_code.data(), options.back_code.size_bytes());
		if (record.ProgramHash() != key.hash) {
			return;
		}
	}
	switch (key.stage) {
		case ShaderType::Pixel: record.input = *options.input_info.pixel; break;
		case ShaderType::Compute: record.input = *options.input_info.compute; break;
		default: record.input = *options.input_info.vertex; break;
	}
	NormalizeManifestInput(record);
	auto host = record;
	SetHostInput(host);
	const auto       identity = ManifestModuleIdentity(record, StaticKey(host));
	auto&            state    = *m_manifest;
	std::scoped_lock lock(state.mutex);
	const auto [found, inserted] = state.module_index.try_emplace(
	    identity, static_cast<uint32_t>(state.recorded.modules.size()));
	if (inserted) {
		state.recorded.modules.push_back(std::move(record));
		state.added_modules++;
		state.dirty = true;
	}
	state.module_records[module.id] = {found->second, false};
	if (module.plain != nullptr) {
		state.module_records[module.plain->id] = {found->second, true};
	}
	state.module_generation.fetch_add(1, std::memory_order_release);
}

bool PipelineCache::RecordPipeline(const GraphicsPipelineKey& key,
                                   const ManifestFloatModes&  float_modes) {
	auto&            state = *m_manifest;
	ManifestPipeline record {.key = key, .float_modes = float_modes};
	record.key.stage_ids = {};
	const auto       key_hash = GraphicsPipelineKeyHash {}(key);
	std::scoped_lock lock(state.mutex);
	for (size_t i = 0; i < key.stage_ids.size(); i++) {
		if (key.stage_ids[i] == 0) {
			continue;
		}
		const auto found = state.module_records.find(key.stage_ids[i]);
		if (found == state.module_records.end() || (found->second.second && i != 3)) {
			state.unlinked_pipelines.insert(key_hash);
			return false;
		}
		record.stages[i]   = found->second.first;
		record.plain_pixel = record.plain_pixel || found->second.second;
	}
	state.unlinked_pipelines.erase(key_hash);
	const auto [found, inserted] = state.pipeline_index.try_emplace(
	    ManifestPipelineIdentity(record), static_cast<uint32_t>(state.recorded.pipelines.size()));
	if (inserted) {
		state.recorded.pipelines.push_back(record);
		state.added_pipelines++;
		state.dirty = true;
		return true;
	}
	const auto merge =
	    MergeFloatModes(state.recorded.pipelines[found->second].float_modes, float_modes);
	state.dirty = state.dirty || merge.changed;
	if (merge.mixed && state.mixed_float_modes.insert(found->second).second) {
		state.dirty = true;
	}
	return true;
}

void PipelineCache::NoteFloatModes(Pipeline& pipeline, const GraphicsPipelineKey& key,
                                   const HW::Shader& shaders) {
	if (m_manifest == nullptr || m_manifest->record_path.empty()) {
		return;
	}
	constexpr uint64_t Linked     = UINT64_MAX;
	const auto         modes      = ReadManifestFloatModes(shaders, key.stage_ids);
	const auto         generation = m_manifest->module_generation.load(std::memory_order_acquire);
	if (modes == pipeline.float_modes &&
	    (pipeline.float_modes_generation == Linked || pipeline.float_modes_generation == generation)) {
		return;
	}
	pipeline.float_modes            = modes;
	pipeline.float_modes_generation = RecordPipeline(key, modes) ? Linked : generation;
}

void PipelineCache::RecordLayout(uint64_t key, const ShaderTessellationInputInfo& layout) {
	if (m_manifest == nullptr || m_manifest->record_path.empty()) {
		return;
	}
	auto&            state = *m_manifest;
	std::scoped_lock lock(state.mutex);
	if (state.layout_keys.insert(key).second) {
		state.recorded.tessellation_layouts.emplace_back(key, layout);
		state.dirty = true;
	}
}

void PipelineCache::SaveManifestRecording() {
	auto& state = *m_manifest;
	if (state.record_path.empty()) {
		return;
	}
	PipelineManifest manifest;
	uint32_t         added_modules   = 0;
	uint32_t         added_pipelines = 0;
	size_t           mixed           = 0;
	size_t           unlinked        = 0;
	{
		std::scoped_lock lock(state.mutex);
		if (!state.dirty) {
			return;
		}
		manifest        = state.recorded;
		added_modules   = state.added_modules;
		added_pipelines = state.added_pipelines;
		mixed           = state.mixed_float_modes.size();
		unlinked        = state.unlinked_pipelines.size();
		state.dirty     = false;
	}
	const auto with_modes =
	    std::ranges::count_if(manifest.pipelines, [](const ManifestPipeline& pipeline) {
		    return pipeline.float_modes[0].recorded;
	    });
	const auto bytes     = SaveManifest(manifest);
	auto       temporary = state.record_path;
	temporary += ".tmp";
	Common::File::CreateDirectories(state.record_path.parent_path());
	{
		Common::File file;
		if (!file.Create(temporary)) {
			return;
		}
		file.Write(bytes.data(), static_cast<uint32_t>(bytes.size()));
	}
	std::error_code error;
	std::filesystem::rename(temporary, state.record_path, error);
	ManifestLog(
	    "Pipeline manifest: wrote {} ({} modules, {} pipelines, {:.1f} KB; {} modules and {} "
	    "pipelines new in this session; float modes of {} pipelines, {} of them drawn with more "
	    "than one, {} drawn pipelines whose modules the manifest does not list)",
	    Common::PathToString(state.record_path), manifest.modules.size(), manifest.pipelines.size(),
	    static_cast<double>(bytes.size()) / 1024.0, added_modules, added_pipelines, with_modes,
	    mixed, unlinked);
}

}
