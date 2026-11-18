#if defined(ANYPS5_KYTY)

#include "graphics/host_gpu/renderer/pipeline/pipelineLibrary.h"

#include "common/cacheTopology.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <cstdlib>
#include <fmt/format.h>
#include <ranges>
#include <string_view>
#include <type_traits>

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Libs::Graphics {

namespace {

constexpr uint32_t LibraryWorkers   = 4;
constexpr uint32_t OptimizedWorkers = 2;
constexpr auto     AheadWindow      = std::chrono::seconds(5);

bool EnvironmentSwitch(const char* name) {
	const char* value = std::getenv(name);
	return value == nullptr || std::string_view(value) != "0";
}

template <typename T>
void Append(std::string& key, const T& value) {
	static_assert(std::is_trivially_copyable_v<T>);
	key.append(reinterpret_cast<const char*>(&value), sizeof(value));
}

std::string LayoutKey(const GraphicsPipelineKey& key, bool native) {
	std::string result;
	Append(result, key.stage_ids);
	Append(result, native);
	return result;
}

std::string PartKey(PipelineLibraryPart part, const GraphicsPipelineKey& key,
                    const PipelineLibraries::Stages& stages, bool native) {
	const auto& params = key.static_params;
	std::string result;
	switch (part) {
		case PipelineLibraryPart::VertexInput:
			Append(result, key.vertex_input);
			Append(result, params.topology);
			Append(result, params.primitive_restart_enable);
			for (uint32_t index = 0; index < key.vertex_input.attribute_count; index++) {
				Append(result, stages[0]->vertex_formats[index]);
			}
			break;
		case PipelineLibraryPart::PreRasterization:
			result = LayoutKey(key, native);
			Append(result, params.negative_one_to_one);
			Append(result, params.depth_clip_enable);
			Append(result, params.patch_control_points);
			Append(result, params.provoking_vtx_last);
			Append(result, params.polygon_mode);
			break;
		case PipelineLibraryPart::FragmentShader:
			result = LayoutKey(key, native);
			Append(result, params.samples);
			Append(result, params.sample_shading_enable);
			Append(result, params.alpha_to_coverage_enable);
			break;
		case PipelineLibraryPart::FragmentOutput:
			Append(result, key.rendering);
			Append(result, params);
			break;
	}
	return result;
}

uint64_t Elapsed(std::chrono::steady_clock::time_point started) {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now() - started)
	                                 .count());
}

}

bool PipelineLibrariesWanted() {
	static const bool wanted = EnvironmentSwitch("KYTY_PIPELINE_LIBRARIES");
	return wanted;
}

bool PipelineLibrariesAhead() {
	static const bool ahead = EnvironmentSwitch("KYTY_PIPELINE_LIBRARIES_AHEAD");
	return ahead;
}

const char*& PipelineRouteNote() {
	thread_local const char* note = nullptr;
	return note;
}

void PlaceBackgroundCompileThread(bool low_priority) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (low_priority) {
		Common::Thread::LowerCurrentPriority();
	}
#else
	static const std::vector<int> cache_cpus = Common::LargestCacheCpus("/sys/devices/system/cpu");
	cpu_set_t                     allowed;
	CPU_ZERO(&allowed);
	if (!cache_cpus.empty() && sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
		cpu_set_t set = allowed;
		for (const int cpu: cache_cpus) {
			if (cpu < CPU_SETSIZE) {
				CPU_CLR(cpu, &set);
			}
		}
		if (CPU_COUNT(&set) != 0 && !CPU_EQUAL(&set, &allowed)) {
			(void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
		}
	}
	if (low_priority) {
		(void)setpriority(PRIO_PROCESS, static_cast<id_t>(syscall(SYS_gettid)), 10);
	}
#endif
}

struct PipelineLibraries::Library {
	enum class State : uint8_t { Queued, Building, Ready, Failed };

	PipelineLibraryPart part = PipelineLibraryPart::VertexInput;
	GraphicsPipelineKey key {};
	Stages              stages {};
	bool                native = false;
	vk::PipelineLayout  layout = nullptr;
	State               state  = State::Queued;
	bool                ahead  = false;
	vk::Pipeline        handle = nullptr;
};

PipelineLibraries::PipelineLibraries(GraphicContext& graphics, vk::PipelineCache driver_cache,
                                     std::function<void(std::chrono::steady_clock::time_point)> on_optimized)
    : m_graphics(graphics), m_driver_cache(driver_cache), m_on_optimized(std::move(on_optimized)) {
	for (uint32_t i = 0; i < LibraryWorkers; i++) {
		m_workers.emplace_back([this] { Worker(); });
	}
	for (uint32_t i = 0; i < OptimizedWorkers; i++) {
		m_optimized_workers.emplace_back([this] { OptimizedWorker(); });
	}
	Log::WriteToConsoleAndLog(fmt::format(
	    "[kyty-gpl] graphics pipeline libraries on: {} library threads, {} background threads for the "
	    "whole pipelines (KYTY_PIPELINE_LIBRARIES=0 to turn off)\n",
	    LibraryWorkers, OptimizedWorkers));
}

PipelineLibraries::~PipelineLibraries() {
	Stop();
	for (const auto& optimized: m_optimized) {
		m_graphics.device.destroyPipeline(optimized.pipeline, nullptr);
	}
	for (auto& part: m_parts) {
		for (const auto& library: part | std::views::values) {
			m_graphics.device.destroyPipeline(library->handle, nullptr);
		}
	}
	for (const auto& layout: m_layouts | std::views::values) {
		m_graphics.device.destroyPipelineLayout(layout.pipeline_layout, nullptr);
		m_graphics.device.destroyDescriptorSetLayout(layout.descriptor_set_layout, nullptr);
	}
}

void PipelineLibraries::Stop() {
	{
		std::lock_guard lock(m_mutex);
		m_stopping = true;
	}
	m_available.notify_all();
	m_workers.clear();
	{
		std::lock_guard lock(m_optimized_mutex);
		m_optimized_stopping = true;
		m_optimized_jobs.clear();
	}
	m_optimized_available.notify_all();
	m_optimized_workers.clear();
}

bool PipelineLibraries::Eligible(const Stages& stages) {
	return stages[0] != nullptr && stages[3] != nullptr;
}

vk::PipelineLayout PipelineLibraries::LibraryLayout(const std::string& key, const GraphicsPipelineKey& pipeline_key,
                                                    const Stages& stages, bool native) {
	std::lock_guard lock(m_mutex);
	auto [layout, inserted] = m_layouts.try_emplace(key);
	if (inserted) {
		layout->second.native_bindings = native;
		CreateGraphicsPipelineLayout(m_graphics, layout->second, pipeline_key, stages);
	}
	return layout->second.pipeline_layout;
}

std::shared_ptr<PipelineLibraries::Library> PipelineLibraries::Request(
    PipelineLibraryPart part, std::string&& key, const GraphicsPipelineKey& pipeline_key, const Stages& stages,
    bool native, vk::PipelineLayout layout, bool urgent, bool ahead) {
	const bool queued = part == PipelineLibraryPart::PreRasterization || part == PipelineLibraryPart::FragmentShader;
	std::unique_lock lock(m_mutex);
	auto& slot = m_parts[static_cast<uint32_t>(part)][std::move(key)];
	if (slot == nullptr) {
		slot         = std::make_shared<Library>();
		slot->part   = part;
		slot->key    = pipeline_key;
		slot->stages = stages;
		slot->native = native;
		slot->layout = layout;
		slot->ahead  = ahead;
		if (ahead) {
			m_ahead_requested.fetch_add(1, std::memory_order_relaxed);
		}
		if (queued && !m_stopping) {
			if (urgent) {
				m_queue.push_front(slot);
			} else {
				m_queue.push_back(slot);
			}
			lock.unlock();
			m_available.notify_one();
			lock.lock();
		}
	} else if (!ahead && slot->ahead) {
		slot->ahead = false;
		m_ahead_used.fetch_add(1, std::memory_order_relaxed);
		if (urgent && slot->state == Library::State::Queued) {
			if (const auto queued = std::ranges::find(m_queue, slot); queued != m_queue.end()) {
				m_queue.erase(queued);
				m_queue.push_front(slot);
			}
		}
	}
	return slot;
}

void PipelineLibraries::Build(const std::shared_ptr<Library>& library) {
	const auto started = std::chrono::steady_clock::now();
	const auto handle  = CreateGraphicsPipelineLibrary(m_graphics, library->layout, library->native, library->key,
	                                                   library->stages, library->part);
	const auto index   = static_cast<uint32_t>(library->part);
	m_build_ns[index].fetch_add(Elapsed(started), std::memory_order_relaxed);
	m_built[index].fetch_add(1, std::memory_order_relaxed);
	{
		std::lock_guard lock(m_mutex);
		library->handle = handle;
		library->state  = handle ? Library::State::Ready : Library::State::Failed;
	}
	m_done.notify_all();
}

vk::Pipeline PipelineLibraries::Await(const std::shared_ptr<Library>& library) {
	std::unique_lock lock(m_mutex);
	if (library->state == Library::State::Queued) {
		library->state = Library::State::Building;
		lock.unlock();
		if (library->part == PipelineLibraryPart::PreRasterization ||
		    library->part == PipelineLibraryPart::FragmentShader) {
			m_claimed.fetch_add(1, std::memory_order_relaxed);
		}
		Build(library);
		lock.lock();
	}
	const auto started = std::chrono::steady_clock::now();
	m_done.wait(lock, [&] {
		return library->state == Library::State::Ready || library->state == Library::State::Failed;
	});
	m_wait_ns.fetch_add(Elapsed(started), std::memory_order_relaxed);
	return library->handle;
}

void PipelineLibraries::Worker() {
	KYTY_PROFILER_THREAD("Thread_PipelineLibrary");
	PlaceBackgroundCompileThread(false);
	for (;;) {
		std::shared_ptr<Library> library;
		{
			std::unique_lock lock(m_mutex);
			m_available.wait(lock, [&] { return m_stopping || !m_queue.empty(); });
			if (m_stopping) {
				return;
			}
			library = std::move(m_queue.front());
			m_queue.pop_front();
			if (library->state != Library::State::Queued) {
				continue;
			}
			library->state = Library::State::Building;
		}
		Build(library);
	}
}

bool PipelineLibraries::Link(PipelineCache::Pipeline& pipeline, const GraphicsPipelineKey& key, const Stages& stages) {
	if (!Eligible(stages) || pipeline.pipeline_layout == nullptr) {
		return false;
	}
	const bool native = pipeline.native_bindings;
	const auto layout = LibraryLayout(LayoutKey(key, native), key, stages, native);
	if (layout == nullptr) {
		return false;
	}
	m_last_link.store(std::chrono::steady_clock::now().time_since_epoch().count(), std::memory_order_relaxed);
	m_active.store(true, std::memory_order_relaxed);
	const auto request = [&](PipelineLibraryPart part, vk::PipelineLayout part_layout) {
		return Request(part, PartKey(part, key, stages, native), key, stages, native, part_layout, true, false);
	};
	const bool mesh              = stages[0]->info.stage == ShaderType::Mesh;
	const auto pre_rasterization = request(PipelineLibraryPart::PreRasterization, layout);
	const auto fragment_shader   = request(PipelineLibraryPart::FragmentShader, layout);
	const auto vertex_input      = mesh ? nullptr : request(PipelineLibraryPart::VertexInput, nullptr);
	const auto fragment_output   = request(PipelineLibraryPart::FragmentOutput, nullptr);
	const char* link_note = nullptr;
	{
		static constexpr const char* Notes[3][3] = {
		    {"linked fs=queued pr=queued", "linked fs=queued pr=building", "linked fs=queued pr=ready"},
		    {"linked fs=building pr=queued", "linked fs=building pr=building", "linked fs=building pr=ready"},
		    {"linked fs=ready pr=queued", "linked fs=ready pr=building", "linked fs=ready pr=ready"},
		};
		const auto state = [](const Library& library) {
			switch (library.state) {
				case Library::State::Queued: return 0;
				case Library::State::Building: return 1;
				default: return 2;
			}
		};
		std::lock_guard lock(m_mutex);
		link_note = Notes[state(*fragment_shader)][state(*pre_rasterization)];
	}
	std::array<vk::Pipeline, 4> libraries {};
	uint32_t                    count = 0;
	if (vertex_input != nullptr) {
		libraries[count++] = Await(vertex_input);
	}
	libraries[count++] = Await(fragment_output);
	libraries[count++] = Await(fragment_shader);
	libraries[count++] = Await(pre_rasterization);
	const auto linked  = std::span(libraries).first(count);
	if (std::ranges::any_of(linked, [](vk::Pipeline library) { return !library; })) {
		m_failed.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	const auto started = std::chrono::steady_clock::now();
	pipeline.pipeline  = LinkGraphicsPipeline(m_graphics, pipeline.pipeline_layout, linked);
	m_link_ns.fetch_add(Elapsed(started), std::memory_order_relaxed);
	if (!pipeline.pipeline) {
		m_failed.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	m_linked.fetch_add(1, std::memory_order_relaxed);
	PipelineRouteNote() = link_note;
	return true;
}

void PipelineLibraries::Prepare(const GraphicsPipelineKey& key, const Stages& stages, bool native) {
	if (!Eligible(stages)) {
		return;
	}
	const auto layout = LibraryLayout(LayoutKey(key, native), key, stages, native);
	if (layout == nullptr) {
		return;
	}
	for (const auto part: {PipelineLibraryPart::FragmentShader, PipelineLibraryPart::PreRasterization}) {
		(void)Request(part, PartKey(part, key, stages, native), key, stages, native, layout, false, true);
	}
}

void PipelineLibraries::BuildIfQueued(const std::shared_ptr<Library>& library) {
	{
		std::lock_guard lock(m_mutex);
		if (library->state != Library::State::Queued) {
			return;
		}
		library->state = Library::State::Building;
	}
	m_built_ahead.fetch_add(1, std::memory_order_relaxed);
	Build(library);
}

void PipelineLibraries::PrepareNow(const GraphicsPipelineKey& key, const Stages& stages, bool native) {
	if (!Eligible(stages)) {
		return;
	}
	const auto layout = LibraryLayout(LayoutKey(key, native), key, stages, native);
	if (layout == nullptr) {
		return;
	}
	const auto fragment_shader = Request(PipelineLibraryPart::FragmentShader,
	                                     PartKey(PipelineLibraryPart::FragmentShader, key, stages, native), key,
	                                     stages, native, layout, false, true);
	const auto pre_rasterization = Request(PipelineLibraryPart::PreRasterization,
	                                       PartKey(PipelineLibraryPart::PreRasterization, key, stages, native), key,
	                                       stages, native, layout, false, true);
	BuildIfQueued(fragment_shader);
	BuildIfQueued(pre_rasterization);
}

void PipelineLibraries::QueueOptimized(const PipelineCache::Pipeline& linked, const GraphicsPipelineKey& key,
                                       const Stages& stages) {
	{
		std::lock_guard lock(m_optimized_mutex);
		if (m_optimized_stopping) {
			return;
		}
		m_optimized_jobs.push_back(
		    {.key = key, .stages = stages, .native = linked.native_bindings, .layout = linked.pipeline_layout});
	}
	m_optimized_available.notify_one();
}

void PipelineLibraries::OptimizedWorker() {
	KYTY_PROFILER_THREAD("Thread_PipelineOptimized");
	PlaceBackgroundCompileThread(true);
	for (;;) {
		OptimizedJob job;
		{
			std::unique_lock lock(m_optimized_mutex);
			m_optimized_available.wait(lock, [&] { return m_optimized_stopping || !m_optimized_jobs.empty(); });
			if (m_optimized_stopping) {
				return;
			}
			job = m_optimized_jobs.front();
			m_optimized_jobs.pop_front();
		}
		PipelineCache::Pipeline layout;
		layout.native_bindings = job.native;
		layout.pipeline_layout = job.layout;
		vk::Pipeline handle    = nullptr;
		const auto   started   = std::chrono::steady_clock::now();
		const auto   result    = CreateGraphicsPipelineWhole(m_graphics, layout, job.key, job.stages, m_driver_cache,
		                                                     {}, handle);
		if (result != vk::Result::eSuccess || !handle) {
			continue;
		}
		m_optimized_ns.fetch_add(Elapsed(started), std::memory_order_relaxed);
		m_optimized_built.fetch_add(1, std::memory_order_relaxed);
		if (m_on_optimized) {
			m_on_optimized(started);
		}
		{
			std::lock_guard lock(m_optimized_mutex);
			m_optimized.push_back({.key = job.key, .native = job.native, .pipeline = handle});
		}
		m_optimized_ready.fetch_add(1, std::memory_order_release);
	}
}

void PipelineLibraries::Tick() {
	const auto last = std::chrono::steady_clock::time_point(
	    std::chrono::steady_clock::duration(m_last_link.load(std::memory_order_relaxed)));
	m_active.store(std::chrono::steady_clock::now() - last < AheadWindow, std::memory_order_relaxed);
}

std::vector<PipelineLibraries::Optimized> PipelineLibraries::TakeOptimized() {
	std::lock_guard lock(m_optimized_mutex);
	return std::exchange(m_optimized, {});
}

void PipelineLibraries::LogStats() {
	const auto avg_ms = [](uint64_t ns, uint64_t count) {
		return count != 0 ? static_cast<double>(ns) / 1e6 / static_cast<double>(count) : 0.0;
	};
	uint64_t built[4];
	uint64_t build_ns[4];
	for (uint32_t i = 0; i < 4; i++) {
		built[i]    = m_built[i].load(std::memory_order_relaxed);
		build_ns[i] = m_build_ns[i].load(std::memory_order_relaxed);
	}
	const auto linked    = m_linked.load(std::memory_order_relaxed);
	const auto optimized = m_optimized_built.load(std::memory_order_relaxed);
	Log::WriteToConsoleAndLog(fmt::format(
	    "[kyty-gpl] libraries built vi {} fo {} pr {} ({:.1f} ms avg) fs {} ({:.1f} ms avg); linked {} "
	    "({:.3f} ms avg); waited {:.3f} s; ahead requested {} used {}; claimed {}; failed {}; whole in "
	    "background {} ({:.1f} ms avg); built ahead {}\n",
	    built[0], built[3], built[1], avg_ms(build_ns[1], built[1]), built[2], avg_ms(build_ns[2], built[2]),
	    linked, avg_ms(m_link_ns.load(std::memory_order_relaxed), linked),
	    static_cast<double>(m_wait_ns.load(std::memory_order_relaxed)) / 1e9,
	    m_ahead_requested.load(std::memory_order_relaxed), m_ahead_used.load(std::memory_order_relaxed),
	    m_claimed.load(std::memory_order_relaxed), m_failed.load(std::memory_order_relaxed), optimized,
	    avg_ms(m_optimized_ns.load(std::memory_order_relaxed), optimized),
	    m_built_ahead.load(std::memory_order_relaxed)));
}

}

#endif
