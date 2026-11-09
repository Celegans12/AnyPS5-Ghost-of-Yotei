#include "common/gpuPlacement.h"

#include "common/assert.h"
#include "common/liveAb.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep
#include <tlhelp32.h>
#endif

namespace Common::GpuPlacement {

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

namespace {

constexpr uint32_t LayoutCount   = 4;
constexpr uint32_t StageCount    = 4;
constexpr uint32_t JobRescanFlips = 600; // a layout with jobs: also places job threads started since it began

enum class DefaultKind : uint32_t { Reserve, All, List };

struct Layout {
	bool                                 defined = false;
	std::optional<std::array<int, 4>>    stages;
	std::optional<DefaultKind>           default_kind;
	std::vector<int>                     default_cpus;
	std::vector<int>                     barcopy;
	std::vector<int>                     jobs;
	std::optional<bool>                  prio;
	std::string                          text;
};

struct Cpu {
	int   logical = 0;
	ULONG id      = 0;
	BYTE  core    = 0;
};

struct State {
	std::mutex                         mutex;
	bool                               start_set = false;
	std::array<int, 4>                 start_stages {-1, -1, -1, -1};
	bool                               start_reserved = false;
	bool                               start_prio     = true;
	std::array<HANDLE, StageCount>     stage_threads {};
	std::vector<HANDLE>                barcopy_threads;
	std::vector<DWORD>                 placed_jobs; // threads given CPU sets by a jobs field
	std::vector<Cpu>                   cpus;
	std::array<Layout, LayoutCount + 1> layouts {}; // [0]: the start placement (no fields)
	uint32_t                           applied = 0;
	std::vector<int>                   barcopy_now; // of the applied layout
	uint64_t                           flips   = 0;
};

State& GetState() {
	static auto* state = new State; // never destroyed: threads may still run during exit
	return *state;
}

HANDLE CurrentThreadHandle() {
	HANDLE handle = nullptr;
	if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &handle, 0, FALSE,
	                     DUPLICATE_SAME_ACCESS)) {
		return nullptr;
	}
	return handle;
}

std::vector<Cpu> SystemCpus() {
	std::vector<Cpu> result;
	ULONG            length = 0;
	(void)GetSystemCpuSetInformation(nullptr, 0, &length, GetCurrentProcess(), 0);
	if (length == 0) return result;
	std::vector<uint8_t> buffer(length);
	if (!GetSystemCpuSetInformation(reinterpret_cast<SYSTEM_CPU_SET_INFORMATION*>(buffer.data()), length, &length,
	                                GetCurrentProcess(), 0)) {
		return result;
	}
	for (ULONG offset = 0; offset + sizeof(SYSTEM_CPU_SET_INFORMATION) <= length;) {
		const auto* info = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buffer.data() + offset);
		if (info->Type == CpuSetInformation && info->CpuSet.Group == 0) {
			result.push_back({info->CpuSet.LogicalProcessorIndex, info->CpuSet.Id, info->CpuSet.CoreIndex});
		}
		offset += info->Size;
	}
	return result;
}

std::vector<int> ParseCpus(std::string_view text, const char* what, const std::string& layout) {
	std::vector<int> cpus;
	size_t           start = 0;
	while (start < text.size()) {
		size_t end = text.find(',', start);
		if (end == std::string_view::npos) end = text.size();
		const std::string item(text.substr(start, end - start));
		start = end + 1;
		if (item.empty()) continue;
		char*      tail  = nullptr;
		const long first = std::strtol(item.c_str(), &tail, 10);
		long       last  = first;
		if (tail != nullptr && *tail == '-') last = std::strtol(tail + 1, &tail, 10);
		if (tail == nullptr || *tail != '\0' || first < 0 || last < first || last > 63) {
			EXIT("KYTY_PLACEMENT_LAYOUT: bad %s list in '%s'\n", what, layout.c_str());
		}
		for (long cpu = first; cpu <= last; cpu++) cpus.push_back(static_cast<int>(cpu));
	}
	if (cpus.empty()) EXIT("KYTY_PLACEMENT_LAYOUT: empty %s list in '%s'\n", what, layout.c_str());
	return cpus;
}

Layout ParseLayout(const std::string& text) {
	Layout layout;
	layout.defined = true;
	layout.text    = text;
	size_t start   = 0;
	while (start < text.size()) {
		size_t end = text.find_first_of("+/", start);
		if (end == std::string::npos) end = text.size();
		const std::string field = text.substr(start, end - start);
		start                   = end + 1;
		if (field.empty()) continue;
		const auto colon = field.find(':');
		if (colon == std::string::npos) EXIT("KYTY_PLACEMENT_LAYOUT: field '%s' has no ':' in '%s'\n", field.c_str(), text.c_str());
		const std::string      key   = field.substr(0, colon);
		const std::string_view value = std::string_view(field).substr(colon + 1);
		if (key == "stages") {
			const auto cpus = ParseCpus(value, "stages", text);
			if (cpus.size() != StageCount) EXIT("KYTY_PLACEMENT_LAYOUT: stages needs 4 processors in '%s'\n", text.c_str());
			layout.stages = std::array<int, 4> {cpus[0], cpus[1], cpus[2], cpus[3]};
		} else if (key == "default") {
			if (value == "reserve") {
				layout.default_kind = DefaultKind::Reserve;
			} else if (value == "all") {
				layout.default_kind = DefaultKind::All;
			} else {
				layout.default_kind = DefaultKind::List;
				layout.default_cpus = ParseCpus(value, "default", text);
			}
		} else if (key == "barcopy") {
			layout.barcopy = ParseCpus(value, "barcopy", text);
		} else if (key == "jobs") {
			layout.jobs = ParseCpus(value, "jobs", text);
		} else if (key == "prio") {
			if (value != "0" && value != "1") EXIT("KYTY_PLACEMENT_LAYOUT: prio must be 0 or 1 in '%s'\n", text.c_str());
			layout.prio = value == "1";
		} else {
			EXIT("KYTY_PLACEMENT_LAYOUT: unknown field '%s' in '%s'\n", key.c_str(), text.c_str());
		}
	}
	return layout;
}

uint32_t StartValue() {
	static const uint32_t value = [] {
		const char* text = std::getenv("KYTY_PERF_PLACEMENT");
		return text != nullptr ? static_cast<uint32_t>(std::strtoul(text, nullptr, 10)) : 0u;
	}();
	return value;
}

const char* JobPrefix() {
	static const std::string prefix = [] {
		const char* text = std::getenv("KYTY_PLACEMENT_JOB_PREFIX");
		return std::string(text != nullptr && text[0] != '\0' ? text : "JobWorker");
	}();
	return prefix.c_str();
}

std::optional<ULONG> IdOf(const State& state, int logical) {
	for (const auto& cpu: state.cpus) {
		if (cpu.logical == logical) return cpu.id;
	}
	return std::nullopt;
}

std::vector<ULONG> IdsOf(const State& state, const std::vector<int>& logical) {
	std::vector<ULONG> ids;
	for (const int cpu: logical) {
		if (const auto id = IdOf(state, cpu)) ids.push_back(*id);
	}
	return ids;
}

// ReserveCores' rule: every logical processor outside the physical cores of the stage processors.
std::vector<ULONG> ReserveIds(const State& state, const std::array<int, 4>& stages) {
	std::vector<BYTE> cores;
	for (const auto& cpu: state.cpus) {
		if (std::ranges::find(stages, cpu.logical) != stages.end()) cores.push_back(cpu.core);
	}
	std::vector<ULONG> ids;
	for (const auto& cpu: state.cpus) {
		if (std::ranges::find(cores, cpu.core) == cores.end()) ids.push_back(cpu.id);
	}
	return ids;
}

std::string CpuList(const std::vector<int>& cpus) {
	std::string text;
	for (const int cpu: cpus) text += (text.empty() ? "" : ",") + std::to_string(cpu);
	return text;
}

void SetThreadSets(HANDLE thread, const std::vector<ULONG>& ids) {
	(void)SetThreadSelectedCpuSets(thread, ids.empty() ? nullptr : ids.data(), static_cast<ULONG>(ids.size()));
}

std::wstring ThreadName(HANDLE thread) {
	using GetThreadDescriptionFunc = HRESULT(WINAPI*)(HANDLE, PWSTR*);
	static const auto get_description = reinterpret_cast<GetThreadDescriptionFunc>(
	    reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "GetThreadDescription")));
	if (get_description == nullptr) return {};
	PWSTR name = nullptr;
	if (FAILED(get_description(thread, &name)) || name == nullptr) return {};
	std::wstring result(name);
	LocalFree(name);
	return result;
}

// The guest's job threads: given `ids` (empty: their selection cleared). Returns how many were placed.
uint32_t PlaceJobs(State& state, const std::vector<ULONG>& ids) {
	const std::string  narrow(JobPrefix());
	const std::wstring prefix(narrow.begin(), narrow.end());
	std::vector<DWORD> placed;
	const HANDLE       snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	if (snapshot != INVALID_HANDLE_VALUE) {
		const DWORD   pid = GetCurrentProcessId();
		THREADENTRY32 entry {};
		entry.dwSize = sizeof(entry);
		for (BOOL ok = Thread32First(snapshot, &entry); ok; ok = Thread32Next(snapshot, &entry)) {
			if (entry.th32OwnerProcessID != pid) continue;
			const bool was_placed = std::ranges::find(state.placed_jobs, entry.th32ThreadID) != state.placed_jobs.end();
			if (ids.empty() && !was_placed) continue;
			const HANDLE thread = OpenThread(THREAD_QUERY_LIMITED_INFORMATION | THREAD_SET_LIMITED_INFORMATION, FALSE,
			                                 entry.th32ThreadID);
			if (thread == nullptr) continue;
			if (was_placed || ThreadName(thread).starts_with(prefix)) {
				SetThreadSets(thread, ids);
				if (!ids.empty()) placed.push_back(entry.th32ThreadID);
			}
			CloseHandle(thread);
		}
		CloseHandle(snapshot);
	}
	const auto count  = static_cast<uint32_t>(ids.empty() ? state.placed_jobs.size() : placed.size());
	state.placed_jobs = std::move(placed);
	return count;
}

void PlaceBarCopy(const State& state, HANDLE thread, size_t index) {
	if (state.barcopy_now.empty()) {
		SetThreadSets(thread, {});
		return;
	}
	const auto id = IdOf(state, state.barcopy_now[index % state.barcopy_now.size()]);
	SetThreadSets(thread, id ? std::vector<ULONG> {*id} : std::vector<ULONG> {});
}

void Apply(State& state, uint32_t n, bool rescan_only) {
	const auto  begin  = std::chrono::steady_clock::now();
	const auto& layout = state.layouts[n];
	if (rescan_only) {
		const auto placed = PlaceJobs(state, IdsOf(state, layout.jobs));
		std::fprintf(stderr, "PLACEMENT layout=%u jobs rescan: %u threads\n", n, placed);
		return;
	}
	const auto stages = layout.stages.value_or(state.start_stages);
	const bool prio   = layout.prio.value_or(state.start_prio);
	for (uint32_t stage = 0; stage < StageCount; stage++) {
		const HANDLE thread = state.stage_threads[stage];
		if (thread == nullptr) continue;
		if (stages[stage] >= 0 && stages[stage] < 64) {
			(void)SetThreadAffinityMask(thread, DWORD_PTR {1} << static_cast<uint32_t>(stages[stage]));
			(void)SetThreadIdealProcessor(thread, static_cast<DWORD>(stages[stage]));
		}
		if (stage == 0 || stage == 3) {
			(void)SetThreadPriority(thread, prio ? THREAD_PRIORITY_HIGHEST : THREAD_PRIORITY_NORMAL);
		}
	}
	const auto         kind = layout.default_kind.value_or(state.start_reserved ? DefaultKind::Reserve : DefaultKind::All);
	std::vector<ULONG> default_ids;
	if (kind == DefaultKind::Reserve) default_ids = ReserveIds(state, stages);
	if (kind == DefaultKind::List) default_ids = IdsOf(state, layout.default_cpus);
	const BOOL default_ok =
	    SetProcessDefaultCpuSets(GetCurrentProcess(), default_ids.empty() ? nullptr : default_ids.data(),
	                             static_cast<ULONG>(default_ids.size()));
	state.barcopy_now = layout.barcopy;
	for (size_t i = 0; i < state.barcopy_threads.size(); i++) PlaceBarCopy(state, state.barcopy_threads[i], i);
	const uint32_t jobs = !layout.jobs.empty() || !state.placed_jobs.empty() ? PlaceJobs(state, IdsOf(state, layout.jobs)) : 0;
	state.applied       = n;
	const auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count();
	std::string default_text = kind == DefaultKind::All ? "all" : kind == DefaultKind::Reserve ? "reserve" : CpuList(layout.default_cpus);
	std::fprintf(stderr,
	             "PLACEMENT layout=%u (%s) stages=%d,%d,%d,%d prio=%d default=%s (%zu sets%s) barcopy=%s (%zu threads) "
	             "jobs=%s (%u threads) in %lld us\n",
	             n, n == 0 ? "start" : layout.text.c_str(), stages[0], stages[1], stages[2], stages[3], prio ? 1 : 0,
	             default_text.c_str(), default_ids.size(), default_ok ? "" : ", FAILED",
	             layout.barcopy.empty() ? "default" : CpuList(layout.barcopy).c_str(), state.barcopy_threads.size(),
	             layout.jobs.empty() ? "default" : CpuList(layout.jobs).c_str(), jobs, static_cast<long long>(us));
}

void LoadLayouts(State& state) {
	state.cpus = SystemCpus();
	for (uint32_t n = 1; n <= LayoutCount; n++) {
		const auto  name = "KYTY_PLACEMENT_LAYOUT_" + std::to_string(n);
		const char* text = std::getenv(name.c_str());
		if (text == nullptr || text[0] == '\0') continue;
		state.layouts[n] = ParseLayout(text);
		std::fprintf(stderr, "PLACEMENT layout %u: %s\n", n, text);
	}
}

} // namespace

void SetStart(const std::array<int, 4>& stage_cpus, bool reserved, bool raised_priority) {
	auto&            state = GetState();
	std::scoped_lock lock(state.mutex);
	state.start_stages   = stage_cpus;
	state.start_reserved = reserved;
	state.start_prio     = raised_priority;
	state.start_set      = true;
}

void RegisterStage(uint32_t stage) {
	if (stage >= StageCount) return;
	auto&            state = GetState();
	std::scoped_lock lock(state.mutex);
	if (state.stage_threads[stage] == nullptr) state.stage_threads[stage] = CurrentThreadHandle();
}

void RegisterBarCopyThread() {
	auto&            state  = GetState();
	const HANDLE     thread = CurrentThreadHandle();
	std::scoped_lock lock(state.mutex);
	if (thread == nullptr) return;
	state.barcopy_threads.push_back(thread);
	if (!state.barcopy_now.empty()) PlaceBarCopy(state, thread, state.barcopy_threads.size() - 1);
}

void OnFrameBoundary() {
	static const bool possible = [] {
		const bool result = LiveAb::Possible(LiveAb::Switch::Placement, StartValue());
		if (result) {
			auto&            state = GetState();
			std::scoped_lock lock(state.mutex);
			LoadLayouts(state);
		}
		return result;
	}();
	if (!possible) return;
	auto&          state = GetState();
	const uint32_t want  = LiveAb::Gate(LiveAb::Switch::Placement, StartValue());
	state.flips++;
	std::scoped_lock lock(state.mutex);
	if (!state.start_set) return; // no stage thread has started yet
	if (want > LayoutCount || (want != 0 && !state.layouts[want].defined)) {
		EXIT("KYTY_PERF_PLACEMENT=%u: KYTY_PLACEMENT_LAYOUT_%u is not set\n", want, want);
	}
	if (want != state.applied) {
		Apply(state, want, false);
	} else if (want != 0 && !state.layouts[want].jobs.empty() && state.flips % JobRescanFlips == 0) {
		Apply(state, want, true);
	}
}

#else

void SetStart(const std::array<int, 4>& /*stage_cpus*/, bool /*reserved*/, bool /*raised_priority*/) {}
void RegisterStage(uint32_t /*stage*/) {}
void RegisterBarCopyThread() {}
void OnFrameBoundary() {}

#endif

} // namespace Common::GpuPlacement
