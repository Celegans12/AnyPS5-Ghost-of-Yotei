#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "common/logging/log.h"
#include "common/parallelFor.h"
#include "common/stringUtils.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fmt/format.h>
#include <ranges>
#include <string_view>

namespace Libs::Graphics {

#if defined(ANYPS5_KYTY)
namespace {

size_t WarmUpThreads(size_t jobs) {
	return std::max<size_t>(1, std::min<size_t>(Common::ParallelThreads(), jobs));
}

bool WarmUpSharedCache(const GraphicContext& graphics) {
	constexpr uint32_t NvidiaVendor = 0x10deu;
	if (const char* value = std::getenv("KYTY_SHADER_CACHE_SHARED"); value != nullptr) {
		return std::string_view(value) == "1";
	}
	return graphics.GetPhysicalDeviceProperties().vendorID != NvidiaVendor;
}

}
#endif

ShaderStoreWriter::ShaderStoreWriter() {
	m_thread = std::jthread([this](std::stop_token stop) {
		std::unique_lock lock(m_mutex);
		while (m_ready.wait(lock, stop, [this] { return !m_jobs.empty(); })) {
			const auto job = std::move(m_jobs.front());
			m_jobs.pop();
			lock.unlock();
			job();
			lock.lock();
		}
	});
}

void ShaderStoreWriter::Post(Common::UniqueFunction<void>&& job) {
	{
		std::scoped_lock lock(m_mutex);
		m_jobs.push(std::move(job));
	}
	m_ready.notify_one();
}

void PipelineCache::WarmUp(const std::filesystem::path& path,
                           [[maybe_unused]] bool driver_cache_restored) {
	AssumeProgramsConfined();
	if (path.empty()) {
		return;
	}
	const auto stamp  = ShaderStoreBuildStamp();
	const auto header = StoreHeader(m_graphics, stamp);
	if (header.empty()) {
		return;
	}
	const GuestClockHold clock_hold;
	const auto           start = std::chrono::steady_clock::now();
	std::vector<uint8_t> bytes;
	bool read_store = false;
	const bool existed = Common::File::IsFileExisting(path);
	if (Common::File file; existed && file.Open(path, Common::File::Mode::Read)) {
		bytes.resize(file.Size());
		uint32_t read = 0;
		file.Read(bytes.data(), static_cast<uint32_t>(bytes.size()), &read);
		read_store = read == bytes.size();
		bytes.resize(read);
	}
	Log::WriteToConsoleAndLog(fmt::format(
	    "Shader cache: compiling shaders and pipelines from {} ({:.1f} MB) with {} threads\n",
	    Common::PathToString(path), static_cast<double>(bytes.size()) / 1'000'000.0,
	    Common::ParallelThreads()));

	const auto kept = RestoreStoreRecords(bytes, header);
	if (kept == 0 && existed) {
		const auto reason = read_store ? StoreMismatch(bytes, header, stamp) : "could not read the store";
		Log::WriteToConsoleAndLog(fmt::format("Shader cache: replacing {}: {}\n", Common::PathToString(path), reason));
	}
	m_store.Open(path, header, kept);
	const auto modules = MapModules();
#if defined(ANYPS5_KYTY)
	const bool shared = WarmUpSharedCache(m_graphics);
	const bool lookup = driver_cache_restored && !shared && m_graphics.pipeline_cache_lookup_enabled;
	std::vector<std::function<void(vk::PipelineCache)>> jobs;
	jobs.reserve(m_graphics_pipelines.size() + modules.size());
	for (auto& node: m_graphics_pipelines) {
		jobs.emplace_back([this, &modules, &node](vk::PipelineCache cache) {
			std::array<const ShaderModule*, 4> stages {};
			for (size_t i = 0; i < stages.size(); i++) {
				const auto found = modules.find(node.first.stage_ids[i]);
				if (found == modules.end()) {
					return;
				}
				stages[i] = found->second;
			}
			CreatePipelineInternal(m_graphics, node.second, node.first, stages, cache);
		});
	}
	for (const auto* module: modules | std::views::values) {
		if (module != nullptr && module->info.stage == ShaderType::Compute) {
			module->pipeline_requested = true;
			auto& pipeline = m_compute_pipelines[module->id];
			if (!lookup || !FindComputePipeline(m_graphics, pipeline, *module, m_driver_cache)) {
				jobs.emplace_back([this, module, &pipeline](vk::PipelineCache cache) {
					CreatePipelineInternal(m_graphics, pipeline, *module, cache);
				});
			}
		}
	}
	const auto                     threads = WarmUpThreads(jobs.size());
	std::vector<vk::PipelineCache> caches(threads);
	for (auto& cache: caches) {
		const vk::PipelineCacheCreateInfo create {};
		if (shared || m_graphics.device.createPipelineCache(&create, nullptr, &cache) != vk::Result::eSuccess) {
			cache = m_driver_cache;
		}
	}
	std::atomic_size_t next = 0;
	const auto         work = [&](size_t thread) {
		for (auto i = next++; i < jobs.size(); i = next++) {
			jobs[i](caches[thread]);
		}
	};
	{
		std::vector<std::jthread> helpers;
		for (size_t thread = 1; thread < threads; thread++) {
			helpers.emplace_back(work, thread);
		}
		work(0);
	}
	std::vector<vk::PipelineCache> merged;
	for (const auto cache: caches) {
		if (cache != m_driver_cache && cache != nullptr) {
			merged.push_back(cache);
		}
	}
	if (m_driver_cache != nullptr && !merged.empty() &&
	    m_graphics.device.mergePipelineCaches(m_driver_cache, static_cast<uint32_t>(merged.size()),
	                                          merged.data()) != vk::Result::eSuccess) {
		Log::WriteToConsoleAndLog("Shader cache: the per-thread pipeline caches could not be merged\n");
	}
	for (const auto cache: merged) {
		m_graphics.device.destroyPipelineCache(cache, nullptr);
	}
	Log::WriteToConsoleAndLog(fmt::format(
	    "Shader cache: {} pipelines created on {} threads, {}, stamp {:016x}\n", jobs.size(), threads,
	    shared ? "one shared pipeline cache" : "one pipeline cache per thread", stamp));
#else
	std::vector<Common::UniqueFunction<void>> jobs;
	jobs.reserve(m_graphics_pipelines.size() + modules.size());
	for (auto& node: m_graphics_pipelines) {
		jobs.emplace_back([this, &modules, &node] {
			CreateGraphicsPipeline(node.second, node.first, modules);
		});
	}
	for (const auto* module: modules | std::views::values) {
		if (module != nullptr && module->info.stage == ShaderType::Compute) {
			jobs.emplace_back([this, module, &pipeline = m_compute_pipelines[module->id]] {
				CreatePipelineInternal(m_graphics, pipeline, *module, m_driver_cache);
			});
		}
	}
	Common::ParallelFor(jobs.size(), [&](size_t i) { jobs[i](); });
#endif
	std::erase_if(m_graphics_pipelines,
	              [](const auto& entry) { return entry.second.pipeline == nullptr; });
	if (!m_graphics_pipelines.empty() || !m_compute_pipelines.empty()) {
		CheckpointDriverCache();
	}
	Log::WriteToConsoleAndLog(fmt::format(
	    "Shader cache: loaded {} programs, {} modules, {} pipelines in {} ms\n",
	    m_program_cache.size(), modules.size() - 1,
	    m_graphics_pipelines.size() + m_compute_pipelines.size(),
	    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
	                                                          start)
	        .count()));
}

PipelineCache::ModuleMap PipelineCache::MapModules() const {
	ModuleMap modules {{0, nullptr}};
	for (const auto& program: m_program_cache | std::views::values) {
		for (const auto& module: program.modules) {
			modules.emplace(module.id, &module);
			if (module.plain != nullptr) {
				modules.emplace(module.plain->id, module.plain.get());
			}
		}
	}
	return modules;
}

bool PipelineCache::CreateGraphicsPipeline(Pipeline& pipeline, const GraphicsPipelineKey& key,
                                           const ModuleMap& modules) {
	std::array<const ShaderModule*, 4> stages {};
	for (size_t i = 0; i < stages.size(); i++) {
		const auto found = modules.find(key.stage_ids[i]);
		if (found == modules.end()) {
			return false;
		}
		stages[i] = found->second;
	}
	CreatePipelineInternal(m_graphics, pipeline, key, stages, m_driver_cache);
	return true;
}

}
