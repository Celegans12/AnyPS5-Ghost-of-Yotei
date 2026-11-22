#include "graphics/host_gpu/renderer/viewLog.h"

#include <fmt/format.h>

#include <algorithm>
#include <cstring>
#include <utility>

namespace Libs::Graphics {

namespace {

constexpr size_t MaxReports = 32;

}

ViewLog::ViewLog(std::span<uint8_t> memory, uint64_t device_address)
    : m_memory(memory), m_device_address(device_address),
      m_units(static_cast<uint32_t>(std::min<uint64_t>(memory.size() / EntryBytes, UINT32_MAX))) {}

uint32_t& ViewLog::Count(uint32_t first) noexcept {
	return *reinterpret_cast<uint32_t*>(m_memory.data() + uint64_t {first} * EntryBytes);
}

ViewLogEntry* ViewLog::Entries(uint32_t first) noexcept {
	return reinterpret_cast<ViewLogEntry*>(m_memory.data() + (uint64_t {first} + 1u) * EntryBytes);
}

std::optional<ViewLog::Segment> ViewLog::Reserve(uint32_t entries, uint64_t tick, uint64_t& wait_tick) {
	wait_tick       = 0;
	const auto units = uint64_t {entries} + 1u;
	if (entries == 0 || units > m_units / 2u) {
		m_stats.full++;
		return std::nullopt;
	}
	const auto start = m_head + units <= m_units ? m_head : 0u;
	const auto end   = start + units;
	for (const auto& live: m_live) {
		if (live.first < end && start < uint64_t {live.first} + live.units) {
			wait_tick = live.tick;
			return std::nullopt;
		}
	}
	m_head = static_cast<uint32_t>(end);
	m_live.push_back({.first = start, .units = static_cast<uint32_t>(units), .capacity = entries, .tick = tick});
	Count(start) = 0;
	m_stats.segments++;
	m_stats.reserved += entries;
	return Segment {.device_address = m_device_address, .first = start, .capacity = entries};
}

void ViewLog::Attach(const Segment& segment, uint64_t program, std::vector<ViewLogEntry> expected) {
	for (auto live = m_live.rbegin(); live != m_live.rend(); ++live) {
		if (live->first != segment.first) continue;
		live->program  = program;
		live->attached = true;
		live->expected = std::move(expected);
		return;
	}
}

void ViewLog::Consume(uint64_t completed_tick) {
	while (!m_live.empty() && m_live.front().tick <= completed_tick) {
		if (m_live.front().attached && !m_live.front().expected.empty()) Check(m_live.front());
		m_live.pop_front();
	}
}

uint64_t ViewLog::Pending() const noexcept {
	return m_live.empty() ? 0 : m_live.front().tick;
}

void ViewLog::Inject(const Segment& segment, std::span<const ViewLogEntry> entries) {
	const auto count = static_cast<uint32_t>(entries.size());
	std::memcpy(Entries(segment.first), entries.data(),
	            std::min(count, segment.capacity) * sizeof(ViewLogEntry));
	Count(segment.first) = count;
}

void ViewLog::Check(const Live& live) {
	const auto count = Count(live.first);
	const auto used  = std::min(count, live.capacity);
	if (count > live.capacity) m_stats.overflow++;
	std::vector<ViewLogEntry> logged(Entries(live.first), Entries(live.first) + used);
	m_stats.checked++;
	m_stats.logged += used;
	for (const auto& entry: logged) {
		if ((entry.Flags() & ViewLogEntry::FormMiss) != 0u) m_stats.misses++;
	}
	const auto order = [](const ViewLogEntry& a, const ViewLogEntry& b) {
		return a.tagged_base != b.tagged_base ? a.tagged_base < b.tagged_base : a.bytes < b.bytes;
	};
	std::ranges::sort(logged, order);
	logged.erase(std::unique(logged.begin(), logged.end()), logged.end());
	m_stats.distinct += logged.size();
	auto expected = live.expected;
	std::ranges::sort(expected, order);
	expected.erase(std::unique(expected.begin(), expected.end()), expected.end());
	m_stats.expected += expected.size();
	const auto report = [&](const char* what, const ViewLogEntry& entry) {
		if (m_reports.size() >= MaxReports + 64u) return;
		m_reports.push_back(fmt::format("{} program={:016x} tick={} slot={} base=0x{:x} bytes=0x{:x} flags=0x{:x} "
		                                "count={} capacity={}",
		                                what, live.program, live.tick, entry.Slot(), entry.Base(), entry.bytes,
		                                entry.Flags() & 0xffu, count, live.capacity));
	};
	if (m_trace != 0) {
		m_trace--;
		std::string views;
		for (const auto& entry: logged) {
			views += fmt::format(" {}:0x{:x}+0x{:x}:0x{:x}", entry.Slot(), entry.Base(), entry.bytes, entry.Flags() & 0xffu);
		}
		m_reports.push_back(fmt::format("checked program={:016x} tick={} count={} capacity={} expected={} views{}",
		                                live.program, live.tick, count, live.capacity, expected.size(), views));
	}
	for (const auto& entry: expected) {
		if (std::ranges::binary_search(logged, entry, order)) {
			m_stats.matched++;
		} else {
			m_stats.missing++;
			report("missing", entry);
		}
	}
	for (const auto& entry: logged) {
		if (!std::ranges::binary_search(expected, entry, order)) {
			m_stats.extra++;
			report("extra", entry);
		}
	}
}

ViewLog::Stats ViewLog::TakeStats() noexcept {
	return std::exchange(m_stats, {});
}

std::vector<std::string> ViewLog::TakeReports() {
	return std::exchange(m_reports, {});
}

}
