#include "graphics/host_gpu/renderer/hTrace.h"

#include "graphics/guest_gpu/graphicsRun.h"
#include "kernel/pthread.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep
#endif

namespace Libs::Graphics::HTrace {

namespace {

struct Counter {
	std::atomic<uint64_t> count {0};
	std::atomic<uint64_t> us {0};
	void                  Add(int64_t us_spent) {
        count.fetch_add(1, std::memory_order_relaxed);
        us.fetch_add(static_cast<uint64_t>(std::max<int64_t>(us_spent, 0)), std::memory_order_relaxed);
	}
	std::pair<uint64_t, uint64_t> Take() {
		return {count.exchange(0, std::memory_order_relaxed), us.exchange(0, std::memory_order_relaxed)};
	}
};

struct State {
	std::atomic<int64_t>  ready_us {0};
	std::atomic<int64_t>  present_us {0};
	std::atomic<int64_t>  done_us {0};
	std::atomic<int64_t>  first_q0_after_ready {0};
	std::atomic<int64_t>  first_q0_after_done {0};
	std::atomic<int64_t>  first_any_after_ready {0};
	std::atomic<int64_t>  last_q0r {-1};
	std::atomic<int64_t>  last_anyr {-1};
	std::atomic<uint64_t> q0_submits {0};
	std::atomic<uint64_t> other_submits {0};
	std::atomic<uint64_t> flips {0};
	Counter               fault_guest;
	Counter               fault_exec;
	Counter               fault_other;
	Counter               fault_outer;
	Counter               flip_wait;
	std::atomic<uint64_t> pac_ranges {0};
	std::atomic<uint64_t> pac_changed {0};
	std::mutex                             pages_mutex;
	std::unordered_map<uint64_t, uint32_t> pages;
	int64_t                                pages_report_us = 0;
};

State& S() {
	static auto* state = new State;
	return *state;
}

void SetFirst(std::atomic<int64_t>& slot, int64_t now) {
	int64_t expected = 0;
	(void)slot.compare_exchange_strong(expected, now, std::memory_order_relaxed);
}

int64_t Since(int64_t later, int64_t earlier) {
	return later != 0 && earlier != 0 ? later - earlier : -1;
}

} // namespace

bool Enabled() {
	static const bool on = [] {
		const char* value   = std::getenv("KYTY_TRACE_H");
		const bool  enabled = value != nullptr && value[0] == '1';
		if (enabled) std::fprintf(stderr, "TRACE_H enabled\n");
		return enabled;
	}();
	return on;
}

int64_t NowUs() {
#if defined(_WIN32)
	static const int64_t frequency = [] {
		LARGE_INTEGER value {};
		QueryPerformanceFrequency(&value);
		return static_cast<int64_t>(value.QuadPart);
	}();
	LARGE_INTEGER counter {};
	QueryPerformanceCounter(&counter);
	return static_cast<int64_t>(static_cast<long double>(counter.QuadPart) * 1000000.0L / static_cast<long double>(frequency));
#else
	return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
}

void NoteSubmit(uint32_t queue) {
	if (!Enabled()) return;
	auto&      s   = S();
	const auto now = NowUs();
	if (queue == 0) {
		s.q0_submits.fetch_add(1, std::memory_order_relaxed);
		if (s.ready_us.load(std::memory_order_relaxed) != 0) SetFirst(s.first_q0_after_ready, now);
		if (s.done_us.load(std::memory_order_relaxed) != 0) SetFirst(s.first_q0_after_done, now);
	} else {
		s.other_submits.fetch_add(1, std::memory_order_relaxed);
	}
	if (s.ready_us.load(std::memory_order_relaxed) != 0) SetFirst(s.first_any_after_ready, now);
}

void NoteFlipReady() {
	if (!Enabled()) return;
	auto&      s        = S();
	const auto now      = NowUs();
	const auto previous = s.ready_us.exchange(now, std::memory_order_relaxed);
	// The previous frame's link: its ready to the first submissions after it (they came before this ready).
	s.last_q0r.store(Since(s.first_q0_after_ready.exchange(0, std::memory_order_relaxed), previous), std::memory_order_relaxed);
	s.last_anyr.store(Since(s.first_any_after_ready.exchange(0, std::memory_order_relaxed), previous), std::memory_order_relaxed);
}

void NoteFlipPresentBegin() {
	if (!Enabled()) return;
	S().present_us.store(NowUs(), std::memory_order_relaxed);
}

void NoteFlipDone() {
	if (!Enabled()) return;
	auto&      s        = S();
	const auto now      = NowUs();
	const auto ready    = s.ready_us.load(std::memory_order_relaxed);
	const auto present  = s.present_us.load(std::memory_order_relaxed);
	const auto previous = s.done_us.exchange(now, std::memory_order_relaxed);
	// q0d: the first queue-0 submission after the previous done, if it came before this one.
	const auto q0_done  = s.first_q0_after_done.exchange(0, std::memory_order_relaxed);
	const auto q0_ready = s.last_q0r.load(std::memory_order_relaxed);
	const auto any      = s.last_anyr.load(std::memory_order_relaxed);
	const auto fg       = s.fault_guest.Take();
	const auto fe       = s.fault_exec.Take();
	const auto fo       = s.fault_other.Take();
	const auto fx       = s.fault_outer.Take();
	const auto fw       = s.flip_wait.Take();
	const auto flip     = s.flips.fetch_add(1, std::memory_order_relaxed) + 1;
	std::fprintf(stderr,
	             "HT f=%llu t=%lld rd=%lld rp=%lld pd=%lld q0r=%lld q0d=%lld anyr=%lld sub=%llu/%llu fg=%llu/%llu fe=%llu/%llu "
	             "fo=%llu/%llu fx=%llu/%llu fw=%llu/%llu pac=%llu/%llu\n",
	             static_cast<unsigned long long>(flip), static_cast<long long>(now), static_cast<long long>(Since(now, ready)),
	             static_cast<long long>(Since(present, ready)), static_cast<long long>(Since(now, present)),
	             static_cast<long long>(q0_ready), static_cast<long long>(Since(q0_done, previous)),
	             static_cast<long long>(any),
	             static_cast<unsigned long long>(s.q0_submits.exchange(0, std::memory_order_relaxed)),
	             static_cast<unsigned long long>(s.other_submits.exchange(0, std::memory_order_relaxed)),
	             static_cast<unsigned long long>(fg.first), static_cast<unsigned long long>(fg.second),
	             static_cast<unsigned long long>(fe.first), static_cast<unsigned long long>(fe.second),
	             static_cast<unsigned long long>(fo.first), static_cast<unsigned long long>(fo.second),
	             static_cast<unsigned long long>(fx.first), static_cast<unsigned long long>(fx.second),
	             static_cast<unsigned long long>(fw.first), static_cast<unsigned long long>(fw.second),
	             static_cast<unsigned long long>(s.pac_changed.exchange(0, std::memory_order_relaxed)),
	             static_cast<unsigned long long>(s.pac_ranges.exchange(0, std::memory_order_relaxed)));
	std::vector<std::pair<uint64_t, uint32_t>> top;
	{
		std::scoped_lock lock(s.pages_mutex);
		if (now - s.pages_report_us >= 5000000 && !s.pages.empty()) {
			s.pages_report_us = now;
			top.assign(s.pages.begin(), s.pages.end());
			s.pages.clear();
		}
	}
	if (!top.empty()) {
		std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
		std::fprintf(stderr, "HT_PAC pages=%zu", top.size());
		for (size_t i = 0; i < top.size() && i < 12u; i++) {
			std::fprintf(stderr, " 0x%llx0000x%u", static_cast<unsigned long long>(top[i].first), top[i].second);
		}
		std::fprintf(stderr, "\n");
	}
}

void NoteFault(FaultLayer layer, bool write, int64_t start_us) {
	(void)write;
	if (!Enabled()) return;
	auto&      s     = S();
	const auto spent = NowUs() - start_us;
	if (layer == FaultLayer::Outer) {
		s.fault_outer.Add(spent);
	} else if (GuestGpu::IsGpuThread()) {
		s.fault_exec.Add(spent);
	} else if (LibKernel::PthreadSelfOrNull() != nullptr) {
		s.fault_guest.Add(spent);
	} else {
		s.fault_other.Add(spent);
	}
}

void NoteFlipWait(int64_t start_us) {
	if (!Enabled()) return;
	S().flip_wait.Add(NowUs() - start_us);
}

void NotePacRanges(uint64_t ranges) {
	if (!Enabled()) return;
	S().pac_ranges.fetch_add(ranges, std::memory_order_relaxed);
}

void NotePacChanged(uint64_t address, uint64_t size) {
	if (!Enabled()) return;
	auto& s = S();
	s.pac_changed.fetch_add(1, std::memory_order_relaxed);
	std::scoped_lock lock(s.pages_mutex);
	for (uint64_t page = address >> 16u; page <= ((address + std::max<uint64_t>(size, 1u) - 1u) >> 16u); page++) {
		s.pages[page]++;
	}
}

void NotePacSample(uint64_t address, uint64_t size, uint64_t offset, uint64_t before, uint64_t after) {
	static std::atomic<uint32_t> printed {0};
	if (!Enabled() || printed.fetch_add(1, std::memory_order_relaxed) >= 4000u) return;
	std::fprintf(stderr, "HT_PAC_SAMPLE addr=0x%llx size=0x%llx off=0x%llx before=%016llx after=%016llx\n",
	             static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
	             static_cast<unsigned long long>(offset), static_cast<unsigned long long>(before),
	             static_cast<unsigned long long>(after));
}

} // namespace Libs::Graphics::HTrace
