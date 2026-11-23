#include "graphics/host_gpu/renderer/formedViews.h"

#include <fmt/format.h>

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <string_view>

namespace Libs::Graphics {

FormedPrograms& FormedPrograms::Instance() {
	static FormedPrograms programs(WarmupNs(), WarmupUses());
	return programs;
}

bool FormedPrograms::StrictGate() {
	static const bool strict = [] {
		const char* value = std::getenv("KYTY_E2_FORMED_GATE");
		return value == nullptr || std::string_view {value} != "0";
	}();
	return strict;
}

uint64_t FormedPrograms::WarmupUses() {
	static const uint64_t warmup = [] {
		const char* value = std::getenv("KYTY_E2_FORMED_WARMUP_USES");
		if (value == nullptr || *value == '\0') return uint64_t {100000};
		char*      end  = nullptr;
		const auto uses = std::strtoull(value, &end, 10);
		return end != nullptr && *end == '\0' ? uint64_t {uses} : uint64_t {100000};
	}();
	return warmup;
}

int64_t FormedPrograms::WarmupNs() {
	static const int64_t warmup = [] {
		const char* value = std::getenv("KYTY_E2_FORMED_WARMUP_MS");
		if (value == nullptr || *value == '\0') return int64_t {10000} * 1000000;
		char*      end = nullptr;
		const auto ms  = std::strtoll(value, &end, 10);
		return end != nullptr && *end == '\0' && ms >= 0 ? int64_t {ms} * 1000000 : int64_t {10000} * 1000000;
	}();
	return warmup;
}

bool FormedPrograms::Holds(const Program& program, uint64_t base, uint64_t bytes, uint64_t version) {
	if (bytes == 0) return true;
	const auto first = base >> BlockBits;
	const auto last  = (base + bytes - 1) >> BlockBits;
	for (auto block = first; block <= last; block++) {
		const auto found = program.blocks.find(block);
		if (found == program.blocks.end() || found->second > version) return false;
	}
	return true;
}

void FormedPrograms::Seed(uint64_t program, uint64_t base, uint64_t bytes, int64_t now_ns) {
	if (bytes == 0 || base == 0) return;
	{
		std::shared_lock lock(m_mutex);
		const auto       found = m_programs.find(program);
		if (found != m_programs.end() && Holds(found->second, base, bytes, UINT64_MAX)) return;
	}
	std::unique_lock lock(m_mutex);
	auto&            entry = m_programs[program];
	const auto       first = base >> BlockBits;
	const auto       last  = (base + bytes - 1) >> BlockBits;
	uint64_t         added = 0;
	const bool formed = entry.state != FormedState::Warming;
	for (auto block = first; block <= last; block++) {
		if (!entry.blocks.emplace(block, entry.version + 1).second) continue;
		if (formed) m_blocks.insert(block);
		added++;
	}
	if (added == 0) return;
	entry.version++;
	entry.last_new = now_ns;
	entry.quiet_uses.store(0, std::memory_order_relaxed);
	m_new_blocks.fetch_add(added, std::memory_order_relaxed);
}

void FormedPrograms::NoteUse(uint64_t program) {
	std::shared_lock lock(m_mutex);
	const auto       found = m_programs.find(program);
	if (found != m_programs.end() && found->second.state == FormedState::Warming) {
		found->second.quiet_uses.fetch_add(1, std::memory_order_relaxed);
	}
}

bool FormedPrograms::Allowed(uint64_t program, int64_t now_ns) {
	{
		std::shared_lock lock(m_mutex);
		const auto       found = m_programs.find(program);
		if (found == m_programs.end()) return false;
		const auto& entry = found->second;
		if (entry.state != FormedState::Warming) return entry.state == FormedState::Active;
		if (entry.blocks.empty() || now_ns - entry.last_new < m_warmup_ns ||
		    entry.quiet_uses.load(std::memory_order_relaxed) < m_warmup_uses) {
			return false;
		}
	}
	std::unique_lock lock(m_mutex);
	auto&            entry = m_programs[program];
	if (entry.state == FormedState::Warming && !entry.blocks.empty() && now_ns - entry.last_new >= m_warmup_ns &&
	    entry.quiet_uses.load(std::memory_order_relaxed) >= m_warmup_uses) {
		entry.state = FormedState::Active;
		for (const auto& [block, added]: entry.blocks) m_blocks.insert(block);
		m_activated.fetch_add(1, std::memory_order_relaxed);
	}
	return entry.state == FormedState::Active;
}

void FormedPrograms::Demote(uint64_t program) {
	std::unique_lock lock(m_mutex);
	auto&            entry = m_programs[program];
	if (entry.state == FormedState::Demoted) return;
	entry.state = FormedState::Demoted;
	m_demoted.fetch_add(1, std::memory_order_relaxed);
}

FormedState FormedPrograms::State(uint64_t program) const {
	std::shared_lock lock(m_mutex);
	const auto       found = m_programs.find(program);
	return found == m_programs.end() ? FormedState::Warming : found->second.state;
}

bool FormedPrograms::Covers(uint64_t program, uint64_t base, uint64_t bytes, uint64_t version) const {
	std::shared_lock lock(m_mutex);
	const auto       found = m_programs.find(program);
	return found != m_programs.end() && Holds(found->second, base, bytes, version);
}

bool FormedPrograms::MayWrite(uint64_t address) const {
	std::shared_lock lock(m_mutex);
	return m_blocks.contains(address >> BlockBits);
}

uint64_t FormedPrograms::Version(uint64_t program) const {
	std::shared_lock lock(m_mutex);
	const auto       found = m_programs.find(program);
	return found == m_programs.end() ? 0 : found->second.version;
}

void FormedPrograms::CopyRanges(uint64_t program, Ranges& out) const {
	std::shared_lock lock(m_mutex);
	const auto       found = m_programs.find(program);
	if (found == m_programs.end()) {
		out.version = 0;
		out.ranges.clear();
		return;
	}
	if (out.version == found->second.version && !out.ranges.empty()) return;
	out.version = found->second.version;
	out.ranges.clear();
	for (const auto& [block, added]: found->second.blocks) {
		const auto begin = block << BlockBits;
		if (!out.ranges.empty() && out.ranges.back().second == begin) {
			out.ranges.back().second = begin + BlockSize;
		} else {
			out.ranges.emplace_back(begin, begin + BlockSize);
		}
	}
}

FormedPrograms::Counts FormedPrograms::TakeCounts() {
	return {.new_blocks = m_new_blocks.exchange(0, std::memory_order_relaxed),
	        .activated  = m_activated.exchange(0, std::memory_order_relaxed),
	        .demoted    = m_demoted.exchange(0, std::memory_order_relaxed)};
}

void ConsumeFormedSegment(const ViewLogConsumed& segment, FormedPrograms& programs, FormedSink& sink,
                          FormedConsumeStats& stats, FormedConsumeResult& result, int64_t now_ns,
                          bool history_marked) {
	stats.segments++;
	const auto demote = [&](std::string_view why, uint64_t base, uint64_t bytes, uint32_t flags) {
		result.stop = true;
		result.reports.push_back(fmt::format("{} program={:016x} tick={} slot={} base=0x{:x} bytes=0x{:x} flags=0x{:x}; "
		                                     "the program goes back to CPU-resolved V#s",
		                                     why, segment.program, segment.tick, flags >> ViewLogEntry::SlotShift, base,
		                                     bytes, flags & 0xffu));
		programs.Demote(segment.program);
	};
	if (segment.count > segment.capacity) {
		stats.overflows++;
		demote("view log overflow", 0, segment.count, 0);
	}
	bool reported_read = false;
	for (const auto& view: segment.views) {
		stats.views++;
		const auto flags   = view.Flags();
		const bool written = (flags & ViewLogEntry::Written) != 0u;
		const auto base    = view.Base();
		if ((flags & ViewLogEntry::FormMiss) != 0u) {
			stats.form_misses++;
			demote("form miss", base, view.bytes, flags);
			continue;
		}
		if (base == 0 || view.bytes == 0) continue;
		const auto bytes = sink.Clamp(base, view.bytes);
		if (bytes == 0) continue;
		const bool known    = programs.Covers(segment.program, base, bytes, segment.context);
		const bool windowed = sink.Windowed(base, bytes);
		if (!written) {
			if (known && windowed) continue;
			stats.read_misses++;
			if (!reported_read) {
				reported_read = true;
				result.reports.push_back(fmt::format("read miss program={:016x} tick={} slot={} base=0x{:x} bytes=0x{:x} "
				                                     "(history {}, window {})",
				                                     segment.program, segment.tick, flags >> ViewLogEntry::SlotShift,
				                                     base, bytes, known ? 1 : 0, windowed ? 1 : 0));
			}
			programs.Seed(segment.program, base, bytes, now_ns);
			result.binds.emplace_back(base, bytes);
			continue;
		}
		if (!known || !windowed) {
			stats.write_misses++;
			demote(known ? "write miss outside a window" : "write miss outside the history (CPU reads of it before now may have been stale)",
			       base, bytes, flags);
			programs.Seed(segment.program, base, bytes, now_ns);
			result.binds.emplace_back(base, bytes);
			if (windowed) sink.Written(base, bytes, segment.tick);
			continue;
		}
		stats.written++;
		if (history_marked) continue;
		if (sink.CpuDirty(base, bytes)) {
			stats.conflicts++;
			demote("guest write conflict", base, bytes, flags);
		}
		sink.Written(base, bytes, segment.tick);
	}
}

}
