#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineStoreFormat.h"

#include "common/logging/log.h"
#include "common/parallelFor.h"
#include "common/stringUtils.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fmt/format.h>
#include <mutex>
#include <ranges>
#include <string>
#include <xxhash.h>
#if defined(ANYPS5_KYTY)
#include "kytyShaderStoreStamp.h"

#include <cstdlib>
#include <string_view>
#include <thread>
#endif
#include <zstd.h>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

namespace Libs::Graphics {

namespace {

using ShaderRecompiler::IR::IoAll;
using ShaderRecompiler::IR::Reader;
using ShaderRecompiler::IR::Writer;

using namespace ShaderStoreFormat;

#if defined(ANYPS5_KYTY)
uint64_t BuildStamp() {
	static const uint64_t stamp = [] {
		if (const char* value = std::getenv("KYTY_SHADER_CACHE_STAMP"); value != nullptr) {
			char*      end    = nullptr;
			const auto parsed = std::strtoull(value, &end, 0);
			if (end != value && *end == '\0') {
				return static_cast<uint64_t>(parsed);
			}
		}
		return static_cast<uint64_t>(KYTY_SHADER_STORE_STAMP);
	}();
	return stamp;
}

size_t WarmUpThreads(size_t jobs) {
	size_t threads = Common::ParallelThreads();
	if (const char* value = std::getenv("KYTY_SHADER_CACHE_THREADS"); value != nullptr) {
		char*      end    = nullptr;
		const auto parsed = std::strtoul(value, &end, 10);
		if (end != value && *end == '\0' && parsed != 0) {
			threads = parsed;
		}
	}
	return std::max<size_t>(1, std::min(threads, jobs));
}

bool WarmUpSharedCache(const GraphicContext& graphics) {
	constexpr uint32_t NvidiaVendor = 0x10deu;
	if (const char* value = std::getenv("KYTY_SHADER_CACHE_SHARED"); value != nullptr) {
		return std::string_view(value) == "1";
	}
	return graphics.GetPhysicalDeviceProperties().vendorID != NvidiaVendor;
}
#else
uint64_t BuildStamp() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	std::array<wchar_t, 32768> name {};
	GetModuleFileNameW(nullptr, name.data(), static_cast<DWORD>(name.size()));
	const std::filesystem::path path = name.data();
#elif defined(__APPLE__)
	std::array<char, 4096> name {};
	auto size = static_cast<uint32_t>(name.size());
	_NSGetExecutablePath(name.data(), &size);
	const std::filesystem::path path = name.data();
#else
	const std::filesystem::path path = "/proc/self/exe";
#endif
	std::error_code error;
	const auto      time = std::filesystem::last_write_time(path, error).time_since_epoch().count();
	return error ? 0 : static_cast<uint64_t>(time);
}
#endif

std::vector<uint8_t> StoreHeader(const GraphicContext& graphics) {
	const auto stamp = BuildStamp();
	if (stamp == 0) {
		return {};
	}
	Writer header;
	const auto profile = SpirvProfile(graphics);
	const auto limit = graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	IoStoreHeader(header, stamp, profile, limit);
	return std::move(header.bytes);
}

}

void ShaderStoreWriter::Open(const std::filesystem::path& path, std::span<const uint8_t> header,
                             uint64_t kept) {
	Common::File::CreateDirectories(path.parent_path());
	if (kept == 0) {
		if (m_file.Create(path)) {
			m_file.Write(header.data(), static_cast<uint32_t>(header.size()));
		}
	} else if (m_file.Open(path, Common::File::Mode::ReadWrite)) {
		m_file.Truncate(kept);
		m_file.Seek(kept);
	}
}

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

void ShaderStoreWriter::Append(std::vector<uint8_t>&& record) {
	Post([this, record = std::move(record)] {
		std::vector<uint8_t> bytes(sizeof(Frame) + ZSTD_compressBound(record.size()));
		const auto size = ZSTD_compress(bytes.data() + sizeof(Frame), bytes.size() - sizeof(Frame),
		                                record.data(), record.size(), 3);
		EXIT_IF(ZSTD_isError(size));
		const Frame frame {.compressed_size = static_cast<uint32_t>(size),
		                   .raw_size        = static_cast<uint32_t>(record.size()),
		                   .checksum        = XXH3_64bits(bytes.data() + sizeof(Frame), size)};
		std::memcpy(bytes.data(), &frame, sizeof(frame));
		m_file.Write(bytes.data(), static_cast<uint32_t>(sizeof(Frame) + size));
	});
}

void ShaderStoreWriter::Post(Common::UniqueFunction<void>&& job) {
	{
		std::scoped_lock lock(m_mutex);
		m_jobs.push(std::move(job));
	}
	m_ready.notify_one();
}

struct PipelineCache::StoreRecord {
	Record                             type = Record::Module;
	ProgramKey                         key;
	ShaderRecompiler::IR::ResourcePlan plan;
	bool                               plan_loaded = false;
	ShaderModule                       module;
	GraphicsPipelineKey                pipeline {};
	uint64_t                           layout_key = 0;
	ShaderTessellationInputInfo        layout;
};

void PipelineCache::WarmUp(const std::filesystem::path& path,
                           [[maybe_unused]] bool driver_cache_restored) {
	AssumeProgramsConfined();
	if (path.empty()) {
		return;
	}
	const auto header = StoreHeader(m_graphics);
	if (header.empty()) {
		return;
	}
	const auto           start = std::chrono::steady_clock::now();
	std::vector<uint8_t> bytes;
	if (Common::File file;
	    Common::File::IsFileExisting(path) && file.Open(path, Common::File::Mode::Read)) {
		bytes.resize(file.Size());
		uint32_t read = 0;
		file.Read(bytes.data(), static_cast<uint32_t>(bytes.size()), &read);
		bytes.resize(read);
	}
	Log::WriteToConsoleAndLog(fmt::format(
	    "Shader cache: compiling shaders and pipelines from {} ({:.1f} MB) with {} threads\n",
	    Common::PathToString(path), static_cast<double>(bytes.size()) / 1'000'000.0,
	    Common::ParallelThreads()));
	auto       records = std::span<const uint8_t>(bytes);
	const bool reusable =
	    records.size() >= header.size() && std::ranges::equal(records.first(header.size()), header);
	if (reusable) {
		records = records.subspan(header.size());
		std::vector<std::span<const uint8_t>> frames;
		for (auto rest = records; rest.size() >= sizeof(Frame);) {
			Frame frame;
			std::memcpy(&frame, rest.data(), sizeof(frame));
			if (rest.size() - sizeof(frame) < frame.compressed_size) {
				break;
			}
			frames.push_back(rest.first(sizeof(frame) + frame.compressed_size));
			rest = rest.subspan(frames.back().size());
		}
		std::vector<StoreRecord> loaded(frames.size());
		std::vector<uint8_t>     read(frames.size());
		Common::ParallelFor(frames.size(), [&](size_t i) { read[i] = ReadRecord(frames[i], loaded[i]) ? 1 : 0; });
		size_t kept = 0;
		while (kept < frames.size() && read[kept] != 0 && AddRecord(loaded[kept])) {
			records = records.subspan(frames[kept].size());
			kept++;
		}
		for (size_t i = kept + 1; i < frames.size(); i++) {
			if (read[i] != 0 && loaded[i].type == Record::Module) {
				DestroyModule(loaded[i].module);
			}
		}
	}
	m_store.Open(path, header, reusable ? bytes.size() - records.size() : 0);

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
	    shared ? "one shared pipeline cache" : "one pipeline cache per thread", BuildStamp()));
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

bool PipelineCache::ReadRecord(std::span<const uint8_t> frame_bytes, StoreRecord& record) const {
	Frame frame;
	std::memcpy(&frame, frame_bytes.data(), sizeof(frame));
	const auto compressed = frame_bytes.subspan(sizeof(frame), frame.compressed_size);
	std::vector<uint8_t> raw;
	if (!DecodeShaderStoreFrame(frame, compressed, raw)) {
		return false;
	}
	Reader     reader {raw};
	const auto complete = [&] { return reader.ok && reader.bytes.empty(); };
	IoAll(reader, record.type);
	switch (record.type) {
		case Record::Module: {
			std::vector<uint8_t>  plan;
			auto&                 module = record.module;
			std::vector<uint32_t> spirv;
			std::vector<uint32_t> plain_spirv;
			IoStoreModule(reader, record.key, module, plan, spirv, plain_spirv);
			if (!complete()) {
				return false;
			}
			record.plan_loaded = ShaderRecompiler::IR::LoadResourcePlan(plan, record.plan);
			module.module      = CompileSPV(spirv, m_graphics.device);
			module.spirv_words = static_cast<uint32_t>(spirv.size());
			if (!plain_spirv.empty()) {
				module.plain = MakePlainModule(module, plain_spirv, m_graphics.device);
				if (module.plain == nullptr) {
					m_graphics.device.destroyShaderModule(module.module, nullptr);
					return false;
				}
			}
			return true;
		}
		case Record::GraphicsPipeline: {
			reader.Raw(&record.pipeline, sizeof(record.pipeline));
			return complete();
		}
		case Record::TessellationLayout: {
			IoAll(reader, record.layout_key, record.layout);
			return complete();
		}
		default: return false;
	}
}

bool PipelineCache::AddRecord(StoreRecord& record) {
	switch (record.type) {
		case Record::Module: {
			const auto [program, inserted] = m_program_cache.try_emplace(std::move(record.key));
			if (inserted) {
				if (!record.plan_loaded) {
					m_program_cache.erase(program);
					DestroyModule(record.module);
					return false;
				}
				program->second.resource_plan = std::move(record.plan);
			}
			program->second.modules.push_back(std::move(record.module));
			program->second.modules.back().plan = &program->second.resource_plan;
			break;
		}
		case Record::GraphicsPipeline: {
			m_graphics_pipelines.try_emplace(record.pipeline);
			break;
		}
		case Record::TessellationLayout: {
			m_tess_layouts.try_emplace(record.layout_key, record.layout);
			break;
		}
	}
	return true;
}

void PipelineCache::StoreModule(const ProgramKey& key, const Program& program,
                                const ShaderModule& module, const std::vector<uint32_t>& spirv,
                                const std::vector<uint32_t>& plain_spirv) {
	if (!m_store) {
		return;
	}
	std::vector<uint8_t> record;
	if (EncodeModuleRecord(key.stage, key.hash, key.user_data_count, key.static_state,
	                       program.resource_plan, module, spirv, plain_spirv, record)) {
		m_store.Append(std::move(record));
	}
}

void PipelineCache::StorePipeline(const GraphicsPipelineKey& key) {
	RecordPipeline(key);
	if (!m_store) {
		return;
	}
	Writer record;
	IoAll(record, Record::GraphicsPipeline);
	record.Raw(&key, sizeof(key));
	m_store.Append(std::move(record.bytes));
}

void PipelineCache::StoreLayout(uint64_t key, const ShaderTessellationInputInfo& layout) {
	RecordLayout(key, layout);
	if (!m_store) {
		return;
	}
	Writer record;
	IoAll(record, Record::TessellationLayout, key, layout);
	m_store.Append(std::move(record.bytes));
}

std::vector<uint8_t> EncodeComputeModule(const ShaderModule& module, const std::vector<uint32_t>& spirv) {
	Writer record;
	IoStoreModuleInfo(record, module);
	IoAll(record, spirv);
	return std::move(record.bytes);
}

bool DecodeComputeModule(std::span<const uint8_t> bytes, ShaderModule& module,
                         std::vector<uint32_t>& spirv) {
	Reader reader {bytes};
	IoStoreModuleInfo(reader, module);
	IoAll(reader, spirv);
	module.spirv_words = static_cast<uint32_t>(spirv.size());
	return reader.ok && reader.bytes.empty() && module.info.stage == ShaderType::Compute;
}

#if defined(ANYPS5_KYTY)
std::vector<uint8_t> ComputeShaderStoreHeader(const GraphicContext& graphics) {
	return StoreHeader(graphics);
}

#endif

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
