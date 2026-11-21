#include "graphics/host_gpu/renderer/rayTracingCheck.h"

#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/rayTracing.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "kytyHostGpuStamp.h"
#include "kytyShaderStoreStamp.h"

#include <fmt/format.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <tuple>
#include <utility>

namespace Libs::Graphics {

namespace {

constexpr uint64_t MinRegion     = uint64_t {4} << 20u;
constexpr uint32_t QueuedAtMost  = 4;
constexpr uint32_t DeferredAtMost = 4096;
constexpr uint32_t ResultBytes   = 32;
constexpr uint32_t VerifiedAfter = 4;
constexpr uint32_t HostFullAfter = 8;
constexpr uint32_t SlowerAfter   = 16;
constexpr uint32_t CompareGroups = 1024;
constexpr size_t   KeptFractions = 4096;
constexpr uint64_t ContextServed = 52;
constexpr uint64_t ContextCounts = 24;

std::string Hash(uint64_t hash) {
	return fmt::format("0x{:016x}", hash);
}

std::string Spread(std::vector<double> values) {
	if (values.empty()) {
		return "none";
	}
	std::sort(values.begin(), values.end());
	return fmt::format("min {:.3g}, median {:.3g}, max {:.3g}", values.front(), values[values.size() / 2],
	                   values.back());
}

std::string Compact(std::vector<double> values) {
	std::sort(values.begin(), values.end());
	return fmt::format("{:.3g}/{:.3g}/{:.3g}", values.front(), values[values.size() / 2], values.back());
}

std::string VersionLine() {
	return RayTracingCheckVersionLine(KYTY_SHADER_STORE_STAMP, KYTY_HOST_GPU_STAMP,
	                                  Config::RayTracingOracle(), Config::RayTracingCheckPpm());
}

}

RayTracingCheck::RayTracingCheck(RenderContext& context): m_context(context) {}

RayTracingCheck::~RayTracingCheck() {
	DestroyCheckComparePipeline(m_context.GetGraphics().device, m_compare);
}

bool RayTracingCheck::Served(const ShaderModule& module) {
	const auto& info = module.info.info;
	return !module.guest_traversal && info.bvh && info.ray_query;
}

bool RayTracingCheck::Empty(const ShaderModule& module) {
	const auto& info = module.info.info;
	return !module.guest_traversal && info.bvh && !info.ray_query && RayTracing::NativeTraversals();
}

void RayTracingCheck::NoteEmpty(uint64_t hash) {
	std::lock_guard lock(m_mutex);
	Load();
	auto& program = m_programs[hash];
	if (program.empty) {
		return;
	}
	program.empty = true;
	Note(fmt::format("[RT] check {}: empty: no form and no host query serves its BVH walks, their rays miss; not checked",
	                 Hash(hash)));
	Save();
}

bool RayTracingCheck::FallsBack(uint64_t shader_hash) {
	if (!m_any_fallback.load(std::memory_order_relaxed)) {
		return false;
	}
	std::lock_guard lock(m_mutex);
	const auto found = m_programs.find(shader_hash);
	return found != m_programs.end() && found->second.fallback;
}

void RayTracingCheck::Load() {
	if (m_loaded) {
		return;
	}
	m_loaded = true;
	const auto title = PipelineCacheTitleId();
	if (!title.empty()) {
		m_path = std::filesystem::path("_PipelineCache") / (title + ".rt");
	}
	if (m_path.empty()) {
		return;
	}
	std::ifstream in(m_path);
	std::string   version;
	if (!in || !std::getline(in, version)) {
		return;
	}
	if (version != VersionLine()) {
		Note(fmt::format("[RT] check: ignoring decisions in {} from a different build or check configuration", m_path.string()));
		return;
	}
	uint32_t verified = 0, fallback = 0, unreplayable = 0;
	for (std::string line; std::getline(in, line);) {
		std::istringstream fields(line);
		std::string        hash_text, state;
		fields >> hash_text >> state;
		const auto hash = std::strtoull(hash_text.c_str(), nullptr, 16);
		if (hash == 0) {
			continue;
		}
		auto& program = m_programs[hash];
		for (std::string token; fields >> token;) {
			if (token.starts_with("checks=")) {
				program.checks = static_cast<uint32_t>(std::strtoul(token.c_str() + 7, nullptr, 10));
			} else if (token.starts_with("unknown=")) {
				program.unknown = static_cast<uint32_t>(std::strtoul(token.c_str() + 8, nullptr, 10));
			} else if (token.starts_with("full=")) {
				program.full = static_cast<uint32_t>(std::strtoul(token.c_str() + 5, nullptr, 10));
			} else if (token.starts_with("transient=")) {
				char* rest            = nullptr;
				program.transient     = static_cast<uint32_t>(std::strtoul(token.c_str() + 10, &rest, 10));
				program.transient_max = rest != nullptr && *rest == '/' ? std::strtod(rest + 1, nullptr) : 0.0;
			} else if (token.starts_with("reason=")) {
				std::string rest;
				std::getline(fields, rest);
				program.unreplayable = token.substr(7) + rest;
				break;
			}
		}
		if (state == "verified") {
			program.verified = true;
			program.clean    = VerifiedAfter;
			verified++;
		} else if (state == "fallback") {
			program.unreplayable.clear();
			fallback++;
		} else if (state == "unreplayable" && !program.unreplayable.empty()) {
			unreplayable++;
		} else if (state == "unjudged-host-full") {
			program.host_full = true;
			program.unreplayable.clear();
		} else if (state == "empty") {
			program.unreplayable.clear();
		} else {
			program.unreplayable.clear();
		}
	}
	if (verified + fallback + unreplayable != 0) {
		Note(fmt::format("[RT] check: {} decisions read from {} ({} verified, {} not replayable, {} fell back in an "
		                 "earlier session and are checked again)",
		                 verified + fallback + unreplayable, m_path.string(), verified, unreplayable, fallback));
	}
}

void RayTracingCheck::Save() const {
	if (m_path.empty()) {
		return;
	}
	std::vector<std::pair<uint64_t, const Program*>> rows;
	for (const auto& [hash, program]: m_programs) {
		if (program.fallback || program.verified || !program.unreplayable.empty() || program.transient != 0 ||
		    program.unknown != 0 || program.full != 0 || program.empty) {
			rows.emplace_back(hash, &program);
		}
	}
	std::sort(rows.begin(), rows.end());
	std::error_code error;
	std::filesystem::create_directories(m_path.parent_path(), error);
	auto temporary = m_path;
	temporary += ".tmp";
	{
		std::ofstream out(temporary, std::ios::trunc);
		out << VersionLine() << '\n';
		for (const auto& [hash, program]: rows) {
			const auto* state = program->fallback       ? "fallback"
			                    : program->verified     ? "verified"
			                    : !program->unreplayable.empty() ? "unreplayable"
			                    : program->host_full             ? "unjudged-host-full"
			                    : program->empty                 ? "empty"
			                                                     : "checking";
			auto        row   = fmt::format("{} {} checks={}", Hash(hash), state, program->checks);
			if (!program->fractions.empty()) {
				row += " fractions=" + Compact(program->fractions);
			}
			if (program->transient != 0) {
				row += fmt::format(" transient={}/{:.3g}", program->transient, program->transient_max);
			}
			if (program->unknown != 0) {
				row += fmt::format(" unknown={}", program->unknown);
			}
			if (program->full != 0) {
				row += fmt::format(" full={}", program->full);
			}
			if (!program->unreplayable.empty() && !program->fallback && !program->verified) {
				row += " reason=" + program->unreplayable;
			}
			out << row << '\n';
		}
	}
	std::filesystem::rename(temporary, m_path, error);
}

bool RayTracingCheck::Prepare() {
	if (m_ready) {
		return true;
	}
	auto& graphics  = m_context.GetGraphics();
	auto& scheduler = m_context.GetCommandScheduler();
	m_results = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Download, 0, AllFlags,
	                                     uint64_t {ResultBytes} * m_pending.size());

	CreateCheckComparePipeline(graphics.device, "RT check", "RT Check Compare", m_compare);
	m_ready = true;
	Note(fmt::format("[RT] check: differential check on, one dispatch in {} per program served natively (one in {} once "
	                "verified), threshold {:.3g} of the output dwords, fallback after {} checked dispatches in a row "
	                "above it",
	                Config::RayTracingCheckPeriod(), uint64_t {Config::RayTracingCheckPeriod()} * SlowerAfter,
	                Config::RayTracingCheckPpm() / 1e6, Config::RayTracingCheckStrikes()));
	return true;
}

void RayTracingCheck::Reserve(uint64_t region) {
	if (region <= m_capacity) {
		return;
	}
	auto capacity = MinRegion;
	while (capacity < region) {
		capacity <<= 1u;
	}
	auto& graphics  = m_context.GetGraphics();
	auto& scheduler = m_context.GetCommandScheduler();
	if (m_scratch != nullptr) {
		scheduler.DeferHostOperation([retired = std::move(m_scratch)]() mutable { retired.reset(); });
	}
	m_scratch  = std::make_shared<Buffer>(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 3 * capacity);
	m_capacity = capacity;
	SetVulkanObjectNameF(graphics.device, m_scratch->Handle(), "RT Check Scratch");
	Note(fmt::format("[RT] check: {} MiB of device memory for checked outputs of up to {} MiB", (3 * capacity) >> 20u,
	                 capacity >> 20u));
}

uint32_t RayTracingCheck::TakeSlot() {
	for (uint32_t slot = 0; slot < m_pending.size(); slot++) {
		if (!m_pending[slot].busy) {
			return slot;
		}
	}
	return UINT32_MAX;
}

void RayTracingCheck::Note(const std::string& line) const {
	::Log::WriteToConsoleAndLog(line + "\n");
}

void RayTracingCheck::Refuse(uint64_t hash, const std::string& reason) {
	auto& program = m_programs[hash];
	if (!program.unreplayable.empty()) {
		return;
	}
	program.unreplayable = reason;
	Note(fmt::format("[RT] check {}: not replayable ({}): no differential check, the static decision stands",
	                Hash(hash), reason));
	Save();
}

bool RayTracingCheck::Sample(const ShaderModule& module, const CommandBuffer& buffer) {
	const auto period = Config::RayTracingCheckPeriod();
	if (period == 0 || m_active) {
		return false;
	}
	if (!Served(module)) {
		if (Empty(module)) {
			NoteEmpty(module.info.shader_hash);
		}
		return false;
	}
	if (&buffer != &m_context.GetCommandScheduler().Current()) {
		return false;
	}
	const auto      hash = module.info.shader_hash;
	std::lock_guard lock(m_mutex);
	Load();
	auto&           program = m_programs[hash];
	const auto      index   = program.dispatches++;
	if (program.fallback || !program.unreplayable.empty()) {
		return false;
	}
	if (const char* reason = RayTracingCheckUnreplayable(module.info)) {
		Refuse(hash, reason);
		return false;
	}
	if (program.waiting) {
		return true;
	}
	const auto every = program.verified ? uint64_t {period} * SlowerAfter : uint64_t {period};
	return index % every == 0;
}

void RayTracingCheck::Defer(uint64_t shader_hash, bool queued) {
	std::lock_guard lock(m_mutex);
	auto&           program = m_programs[shader_hash];
	program.waiting         = true;
	program.deferred++;
	program.queued += queued ? 1 : 0;
	if (program.queued > QueuedAtMost || program.deferred > DeferredAtMost) {
		program.waiting = false;
		Refuse(shader_hash, fmt::format("the emulated variant was not ready after {} dispatches and {} background "
		                                "translations",
		                                program.deferred, program.queued));
	}
}

bool RayTracingCheck::Begin(uint64_t shader_hash) {
	std::lock_guard lock(m_mutex);
	auto&           program = m_programs[shader_hash];
	const auto      slot    = TakeSlot();
	if (slot == UINT32_MAX) {
		program.skipped++;
		program.waiting = true;
		return false;
	}
	if (!Prepare()) {
		return false;
	}
	if (program.waiting && program.deferred != 0) {
		Note(fmt::format("[RT] check {}: emulated variant ready after {} deferred dispatches", Hash(shader_hash),
		                 program.deferred));
	}
	program.waiting  = false;
	program.deferred = 0;
	program.queued   = 0;
	m_pending[slot]  = {.busy = true, .hash = shader_hash, .dwords = 0, .outputs = 0, .index = program.dispatches - 1};
	m_slot           = slot;
	m_hash           = shader_hash;
	m_active         = true;
	m_prepared       = m_context.GetRayTracing()->LastPreparation();
	return true;
}

void RayTracingCheck::Before(const PreparedBindings& bindings) {
	if (!m_active) {
		return;
	}
	std::string reason;
	if (!CollectCheckOutputs(m_context, bindings, m_outputs, reason)) {
		std::lock_guard lock(m_mutex);
		Refuse(m_hash, reason);
		m_pending[m_slot].busy = false;
		m_active               = false;
		return;
	}
	m_region = CheckOutputsBytes(m_outputs);
	Reserve(m_region);
	auto& scheduler = m_context.GetCommandScheduler();
	scheduler.EndRendering();
	const auto command = scheduler.Current().Handle();
	CheckBarrier(command);
	command.fillBuffer(m_scratch->Handle(), m_capacity, m_region, 0);
	command.fillBuffer(m_scratch->Handle(), 2 * m_capacity, m_region, 0);
	CopyCheckOutputs(m_context, command, m_scratch->Handle(), m_outputs, 0, false);
	CheckBarrier(scheduler.Current().Handle());
}

void RayTracingCheck::AfterOracle(const PreparedBindings&, bool counted) {
	if (!m_active) {
		return;
	}
	auto& scheduler = m_context.GetCommandScheduler();
	scheduler.EndRendering();
	CheckBarrier(scheduler.Current().Handle());
	const auto context = m_context.GetRayTracing()->ContextBuffer();
	counted            = counted && context;
	if (counted) {
		const vk::BufferCopy copy {ContextServed, uint64_t {m_slot} * ResultBytes + 8, ContextCounts};
		scheduler.Current().Handle().copyBuffer(context, m_results->Handle(), 1, &copy);
	}
	{
		const auto      preparation = m_context.GetRayTracing()->LastPreparation();
		std::lock_guard lock(m_mutex);
		m_pending[m_slot].counted     = counted;
		m_pending[m_slot].preparation = counted && preparation != m_prepared ? preparation : 0;
	}
	CopyCheckOutputs(m_context, scheduler.Current().Handle(), m_scratch->Handle(), m_outputs, m_capacity, false);
	CheckBarrier(scheduler.Current().Handle());
	CopyCheckOutputs(m_context, scheduler.Current().Handle(), m_scratch->Handle(), m_outputs, 0, true);
	CheckBarrier(scheduler.Current().Handle());
}

void RayTracingCheck::AfterGuest(const PreparedBindings& bindings) {
	if (!m_active) {
		return;
	}
	m_active = false;
	std::vector<CheckOutput> guest;
	std::string              reason;
	bool                     same = CollectCheckOutputs(m_context, bindings, guest, reason);
	std::vector<CheckOutput> matched;
	for (const auto& output: m_outputs) {
		const auto found = std::find_if(guest.begin(), guest.end(), [&](const CheckOutput& candidate) {
			return CheckOutputPlace(candidate) == CheckOutputPlace(output) && candidate.size >= output.size;
		});
		if (!same || found == guest.end()) {
			same = false;
			break;
		}
		auto& place = matched.emplace_back(*found);
		place.size  = output.size;
		place.at    = output.at;
	}
	if (!same) {
		std::lock_guard lock(m_mutex);
		Refuse(m_hash, reason.empty() ? std::string("the emulated variant writes other outputs") : reason);
		m_pending[m_slot].busy = false;
		return;
	}
	auto& scheduler = m_context.GetCommandScheduler();
	scheduler.EndRendering();
	CheckBarrier(scheduler.Current().Handle());
	CopyCheckOutputs(m_context, scheduler.Current().Handle(), m_scratch->Handle(), matched, 2 * m_capacity, false);
	const auto command = scheduler.Current().Handle();
	const auto offset  = uint64_t {m_slot} * ResultBytes;
	command.fillBuffer(m_results->Handle(), offset, 4, 0);
	command.fillBuffer(m_results->Handle(), offset + 4, 4, 0xffffffffu);
	CheckBarrier(command);
	const std::array<vk::DescriptorBufferInfo, 3> infos {
	    vk::DescriptorBufferInfo {m_scratch->Handle(), m_capacity, m_region},
	    vk::DescriptorBufferInfo {m_scratch->Handle(), 2 * m_capacity, m_region},
	    vk::DescriptorBufferInfo {m_results->Handle(), 0, VK_WHOLE_SIZE},
	};
	std::array<vk::WriteDescriptorSet, 3> writes {};
	for (uint32_t index = 0; index < writes.size(); index++) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}
	const CheckComparePush push {.count = static_cast<uint32_t>(m_region / 4), .slot = m_slot};
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_compare.pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_compare.pipeline_layout, 0, writes);
	command.pushConstants(m_compare.pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
	command.dispatch(std::min<uint32_t>(CompareGroups, (push.count + 255u) / 256u), 1, 1);
	CheckBarrier(command);
	{
		std::lock_guard lock(m_mutex);
		m_pending[m_slot].dwords  = push.count;
		m_pending[m_slot].outputs = static_cast<uint32_t>(matched.size());
	}
	const auto slot = m_slot;
	scheduler.DeferHostOperation([weak = weak_from_this(), slot] {
		if (const auto self = weak.lock()) {
			self->Finish(slot);
		}
	});
}

void RayTracingCheck::Finish(uint32_t slot) {
	m_results->Invalidate(uint64_t {slot} * ResultBytes, ResultBytes);
	const auto* words  = reinterpret_cast<const uint32_t*>(m_results->Mapped().data() + uint64_t {slot} * ResultBytes);
	const auto  differ   = words[0];
	const auto  first    = words[1];
	const auto  served     = words[2];
	const auto  returned   = words[3];
	const auto  resumes    = words[5];
	const auto  waves      = words[6];
	const auto  candidates = words[7];
	std::lock_guard lock(m_mutex);
	const auto pending     = m_pending[slot];
	m_pending[slot].busy   = false;
	auto&        program   = m_programs[pending.hash];
	const double fraction  = pending.dwords == 0 ? 0.0 : static_cast<double>(differ) / static_cast<double>(pending.dwords);
	const double threshold = Config::RayTracingCheckPpm() / 1e6;
	const bool   over      = fraction > threshold;
	const auto   outcome   = pending.preparation == 0 ? std::nullopt
	                                                  : m_context.GetRayTracing()->Outcome(pending.preparation);
	if (pending.preparation != 0 && !outcome) {
		program.unknown++;
		program.full_run = 0;
		Note(fmt::format("[RT] check {}: dispatch {}: {} of {} output dwords differ ({:.3g}), the host readback of this "
		                 "dispatch is gone, not judged; {} unknown so far",
		                 Hash(pending.hash), pending.index, differ, pending.dwords, fraction, program.unknown));
		Save();
		return;
	}
	if (outcome && outcome->full) {
		program.full++;
		program.full_run++;
		Note(fmt::format("[RT] check {}: dispatch {}: {} of {} output dwords differ ({:.3g}), the host structure was full "
		                 "(more instances than it holds), not judged; {} so far",
		                 Hash(pending.hash), pending.index, differ, pending.dwords, fraction, program.full));
		if (!program.host_full && program.full_run >= HostFullAfter) {
			program.host_full = true;
			Note(fmt::format("[RT] check {}: unjudged-host-full: {} checks in a row on a full host structure",
			                 Hash(pending.hash), program.full_run));
		}
		Save();
		return;
	}
	program.full_run = 0;
	if (outcome && (outcome->missing != 0 || outcome->uncached)) {
		program.transient++;
		program.transient_max = std::max(program.transient_max, fraction);
		Note(fmt::format("[RT] check {}: dispatch {}: {} of {} output dwords differ ({:.3g}), transient: the host TLAS of "
		                 "this dispatch was incomplete ({} instances without a BLAS{}), not judged; {} transient checks so "
		                 "far, worst {:.3g}",
		                 Hash(pending.hash), pending.index, differ, pending.dwords, fraction, outcome->missing,
		                 outcome->uncached ? ", TLAS memory not cached yet" : "", program.transient,
		                 program.transient_max));
		Save();
		return;
	}
	program.checks++;
	program.strikes = over ? program.strikes + 1 : 0;
	program.clean += over ? 0 : 1;
	if (program.fractions.size() < KeptFractions) {
		program.fractions.push_back(fraction);
	}
	Note(fmt::format("[RT] check {}: dispatch {}: {} of {} output dwords differ ({:.3g}, threshold {:.3g}){}, {} "
	                "output(s){}",
	                Hash(pending.hash), pending.index, differ, pending.dwords, fraction, threshold,
	                differ == 0 ? std::string() : fmt::format(", first at dword {}", first), pending.outputs,
	                pending.counted
	                    ? fmt::format("; host: {} rays served in {} waves, {} candidates, {} resume rounds, {} box calls "
	                                  "handed back to the guest walk",
	                                  served, waves, candidates, resumes, returned)
	                    : std::string()));
	if (!program.verified && !over && program.clean >= VerifiedAfter) {
		program.verified = true;
		Note(fmt::format("[RT] check {}: verified on {} checked dispatches ({}); one dispatch in {} checked from now on",
		                Hash(pending.hash), program.checks, Spread(program.fractions),
		                uint64_t {Config::RayTracingCheckPeriod()} * SlowerAfter));
		Save();
	}
	if (!program.fallback && program.strikes >= Config::RayTracingCheckStrikes()) {
		program.fallback = true;
		Note(fmt::format("[RT] check {}: falls back to the guest BVH walk for the rest of the session: {} checked "
		                "dispatches in a row above {:.3g} ({} over {} checks)",
		                Hash(pending.hash), program.strikes, threshold, Spread(program.fractions), program.checks));
		m_any_fallback.store(true, std::memory_order_relaxed);
		Save();
	}
}

}
