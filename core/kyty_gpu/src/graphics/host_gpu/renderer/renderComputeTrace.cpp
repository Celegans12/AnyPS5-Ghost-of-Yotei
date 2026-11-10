#include "graphics/host_gpu/renderer/renderComputeTrace.h"

#include "common/envRead.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "common/workaroundHits.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/debugSample.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/refCapture.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"
#include <xxhash.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace {
struct Row {
	uint64_t count = 0;
	uint64_t ns    = 0;
};
struct Phases {
	uint64_t count = 0, full = 0, ns = 0;
	std::array<uint64_t, DispatchCensusTimer::kMarks> phase_ns {};
	std::array<uint64_t, 5>      paths {}; // t_compute_program_path of the full ones
	std::array<uint64_t, 5>      path_ns {}; // program lookup ns by path
	std::array<uint64_t, static_cast<size_t>(DispatchSub::Count)> sub_ns {}; // the full ones' parts
	// "Recipe" census: dispatches whose program and resource snapshot (descriptors, flattened SRT, user data, group
	// counts) equal the previous dispatch of the kind, one seen earlier in this frame, or one of the previous frame.
	uint64_t same_prev = 0, same_frame = 0, same_last_frame = 0;
};
static std::array<Phases, 4>& Kinds() {
	static std::array<Phases, 4> kinds {};
	return kinds;
}
static std::array<Row, 4>& Classes() {
	static std::array<Row, 4> classes {};
	return classes;
}
static std::unordered_map<uint64_t, Row>& Rows() {
	static std::unordered_map<uint64_t, Row> rows;
	return rows;
}
} // namespace

void DispatchCensusTimer::Start() {
	m_start    = std::chrono::steady_clock::now();
	m_marks[0] = m_start;
	t_compute_program_path = 0;
	t_dispatch_sub_ns      = {};
	t_dispatch_sub_on      = true;
}

void DispatchCensusTimer::NoteRecipe(uint64_t key) {
	if (!m_on || key == 0) return;
	struct Seen {
		uint64_t key = 0;
		uint32_t frame = 0;
	};
	static std::array<Seen, 65536> seen {};
	static std::array<uint64_t, 4> last {};
	auto&          k     = Kinds()[m_kind & 3u];
	const uint32_t frame = DebugPresentCount().load();
	auto&          slot  = seen[(key ^ (key >> 29u)) & 0xffffu];
	if (last[m_kind & 3u] == key) {
		k.same_prev++;
	} else if (slot.key == key && slot.frame == frame) {
		k.same_frame++;
	} else if (slot.key == key && slot.frame + 1u == frame) {
		k.same_last_frame++;
	}
	last[m_kind & 3u] = key;
	slot              = {key, frame};
}

void DispatchCensusTimer::NoteDispatch(const ShaderComputeInputInfo& input_info, uint32_t mode) {
	const auto& program   = *input_info.stage.program;
	const auto& resources = *input_info.stage.resources;
	// The recipe key.
	XXH3_state_t* state = XXH3_createState();
	XXH3_64bits_reset(state);
	const auto add = [&](const void* data, size_t bytes) { if (bytes != 0) XXH3_64bits_update(state, data, bytes); };
	const void* program_id = &program;
	add(&program_id, sizeof(program_id));
	add(input_info.workgroup_counts, sizeof(input_info.workgroup_counts));
	add(&mode, sizeof(mode));
	for (const auto* list: {&resources.buffers, &resources.images, &resources.samplers}) {
		for (const auto& d: *list) add(d.dwords.data(), d.dwords.size() * sizeof(d.dwords[0]));
		const size_t n = list->size();
		add(&n, sizeof(n));
	}
	add(resources.flattened_srt.data(), resources.flattened_srt.size() * sizeof(uint32_t));
	add(resources.user_data.data(), resources.user_data.size() * sizeof(uint32_t));
	NoteRecipe(XXH3_64bits_digest(state) | 1u);
	XXH3_freeState(state);
	// What the dispatch writes.
	bool buffers = program.has_address_writes;
	for (const auto& b: program.info.buffers) buffers |= b.written;
	for (const auto& r: resources.indirect_ranges) buffers |= r.written;
	bool images = false;
	for (const auto& i: program.info.images) images |= i.written;
	SetClass(buffers ? 1u : images ? 2u : 3u);
}

void DispatchCensusTimer::Finish() {
	const auto now = std::chrono::steady_clock::now();
	t_dispatch_sub_on = false;
	{
		auto&      k     = Kinds()[m_kind & 3u];
		const auto ns_of = [](auto d) { return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(d).count()); };
		k.count++;
		k.ns += ns_of(now - m_start);
		bool full = true;
		for (uint32_t i = 1; i < kMarks; i++) full = full && m_marks[i] >= m_marks[i - 1];
		full = full && m_marks[kMarks - 1] != std::chrono::steady_clock::time_point {};
		if (full) {
			k.full++;
			for (uint32_t i = 0; i + 1 < kMarks; i++) k.phase_ns[i] += ns_of(m_marks[i + 1] - m_marks[i]);
			k.phase_ns[kMarks - 1] += ns_of(now - m_marks[kMarks - 1]);
			const uint32_t path = std::min<uint32_t>(t_compute_program_path, 4u);
			k.paths[path]++;
			k.path_ns[path] += ns_of(m_marks[2] - m_marks[1]);
			for (size_t i = 0; i < k.sub_ns.size(); i++) k.sub_ns[i] += t_dispatch_sub_ns[i];
		}
	}
	auto&      row = Rows()[m_address];
	Classes()[m_class & 3u].count++;
	Classes()[m_class & 3u].ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - m_start).count());
	row.count++;
	row.ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - m_start).count());
	static auto     window      = now;
	static uint32_t frame_start = DebugPresentCount().load();
	if (now - window < std::chrono::seconds(5)) return;
	const uint32_t frame  = DebugPresentCount().load();
	const double   frames = frame > frame_start ? static_cast<double>(frame - frame_start) : 1.0;
	window                = now;
	frame_start           = frame;
	std::vector<std::pair<uint64_t, Row>> rows(Rows().begin(), Rows().end());
	std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
	uint64_t total_count = 0, total_ns = 0;
	for (const auto& [a, r]: rows) {
		total_count += r.count;
		total_ns += r.ns;
	}
	std::fprintf(stderr, "PERF dispatch_census total=%.0f/frame %.2fms/frame shaders=%zu class(n/ms per frame: ?/buffer_writes/image_only/none)=",
	             static_cast<double>(total_count) / frames, static_cast<double>(total_ns) / frames / 1e6, rows.size());
	for (auto& c: Classes()) {
		std::fprintf(stderr, "%.0f/%.2f ", static_cast<double>(c.count) / frames, static_cast<double>(c.ns) / frames / 1e6);
		c = {};
	}
	std::fprintf(stderr, "\n");
	static const char* const kind_names[4] = {"gfx_direct", "gfx_indirect", "async_direct", "async_indirect"};
	for (uint32_t kk = 0; kk < 4u; kk++) {
		auto& k = Kinds()[kk];
		if (k.count == 0) continue;
		const double full = k.full != 0 ? static_cast<double>(k.full) : 1.0;
		std::fprintf(stderr,
		             "PERF dispatch_phases %s n=%.0f/frame %.2fms/frame %.2fus/dispatch full=%.0f%% us(pre=%.2f program=%.2f "
		             "meta=%.2f bind=%.2f buffers=%.2f commit=%.2f tail=%.2f) path(none/adopt/mr/worker/compile)=",
		             kind_names[kk], static_cast<double>(k.count) / frames, static_cast<double>(k.ns) / frames / 1e6,
		             static_cast<double>(k.ns) / 1e3 / static_cast<double>(k.count),
		             100.0 * static_cast<double>(k.full) / static_cast<double>(k.count), k.phase_ns[0] / full / 1e3,
		             k.phase_ns[1] / full / 1e3, k.phase_ns[2] / full / 1e3, k.phase_ns[3] / full / 1e3,
		             k.phase_ns[4] / full / 1e3, k.phase_ns[5] / full / 1e3, k.phase_ns[6] / full / 1e3);
		for (uint32_t p = 0; p < 5u; p++) {
			std::fprintf(stderr, "%.0f%%@%.2fus%s", 100.0 * static_cast<double>(k.paths[p]) / full,
			             k.paths[p] != 0 ? static_cast<double>(k.path_ns[p]) / 1e3 / static_cast<double>(k.paths[p]) : 0.0,
			             p + 1u < 5u ? "/" : "\n");
		}
		static const char* const sub_names[] = {"prepare_program", "key_find", "adopt", "permutation", "find_buffers",
		                                        "prepare_bda", "rebind_images", "shared_memory", "rebind_buffers",
		                                        "pipeline", "prepare_bindings", "commit_bindings", "record", "reset",
		                                        "adopt_reads", "bda_ranges", "obtain_written", "obtain_read", "tc_invalidate",
		                                        "srt_upload", "data_upload", "ow_batch_commit", "ow_upload_walk",
		                                        "ow_dirty_add", "ow_observer", "ow_write_tick", "tail_pre", "tail_post",
		                                        "tail_emitted", "bindless_heaps"};
		static_assert(std::size(sub_names) <= static_cast<size_t>(DispatchSub::Count)); // the draw census parts follow
		std::fprintf(stderr, "PERF dispatch_recipe %s same_prev=%.1f%% same_frame=%.1f%% same_last_frame=%.1f%%\n", kind_names[kk],
		             100.0 * static_cast<double>(k.same_prev) / static_cast<double>(k.count),
		             100.0 * static_cast<double>(k.same_frame) / static_cast<double>(k.count),
		             100.0 * static_cast<double>(k.same_last_frame) / static_cast<double>(k.count));
		std::fprintf(stderr, "PERF dispatch_parts %s us:", kind_names[kk]);
		for (size_t i = 0; i < std::size(sub_names); i++) std::fprintf(stderr, " %s=%.3f", sub_names[i], k.sub_ns[i] / full / 1e3);
		std::fprintf(stderr, "\n");
		k = {};
	}
	for (size_t i = 0; i < rows.size() && i < 16u; i++) {
		const auto& [a, r] = rows[i];
		std::fprintf(stderr, "PERF dispatch_census cs=0x%llx n=%.1f/frame %.3fms/frame %.1fus\n",
		             static_cast<unsigned long long>(a), static_cast<double>(r.count) / frames,
		             static_cast<double>(r.ns) / frames / 1e6, static_cast<double>(r.ns) / 1e3 / static_cast<double>(r.count));
	}
	Rows().clear();
}

namespace ComputeTrace {

// KYTY_TRACE_GUEST_WATCH=<hex addr>:<hex size>[,...] or *<hex pointer>+<hex offset>:<hex size> (diagnostic): before every compute dispatch the GPU thread reads
// the ranges from guest memory (the CPU view, no GPU sync) and logs each change of their contents (non-zero byte count)
// and of the buffer cache's GPU-modified state (ranges up to 8 bytes also print their value), with the frame and the previous dispatch's shader: a change of the CPU
// view comes from the guest CPU, AMPR or a readback; a GPU-modified flip without it is a GPU write.
namespace GuestWatch {
struct Range {
	uint64_t addr = 0, size = 0, hash = 0;
	uint64_t pointer = 0; // *<pointer>+<addr>: addr is an offset from the 8-byte value read at pointer each poll
	uint32_t nonzero = 0;
	bool     gpu_modified = false, seen = false;
};
static std::mutex& Mutex() {
	static std::mutex mutex;
	return mutex;
}
static std::vector<Range>& Ranges() {
	static std::vector<Range> ranges = [] {
		std::vector<Range> list;
		const char*        value = std::getenv("KYTY_TRACE_GUEST_WATCH");
		while (value != nullptr && *value != 0) {
			char*    end     = nullptr;
			uint64_t pointer = 0;
			if (*value == '*') {
				pointer = std::strtoull(value + 1, &end, 16);
				value   = end != nullptr && *end == '+' ? end + 1 : end;
			}
			const auto addr = std::strtoull(value, &end, 16);
			uint64_t   size = 0x1000;
			if (end != nullptr && *end == ':') size = std::strtoull(end + 1, &end, 16);
			list.push_back({.addr = addr, .size = std::min<uint64_t>(size, 0x40000u), .pointer = pointer});
			if (end == nullptr || *end != ',') break;
			value = end + 1;
		}
		return list;
	}();
	return ranges;
}
void Poll(RenderContext& context, uint64_t next_hash) {
	static uint64_t last_hash = 0;
	std::scoped_lock lock(Mutex());
	// Not direct memory (e.g. the game's own data segment): a guarded copy of the guest mapping.
	const auto read = [](uint64_t addr, void* data, uint64_t size) {
		return Libs::LibKernel::Memory::TryReadBacking(addr, data, size) ||
		       Libs::LibKernel::Memory::GuardedGuestCopy(data, reinterpret_cast<const void*>(addr), size);
	};
	for (auto& r: Ranges()) {
		std::vector<uint8_t> bytes(r.size);
		uint64_t             base = 0;
		if (r.pointer != 0 && (!read(r.pointer, &base, sizeof(base)) || base == 0)) continue;
		const uint64_t addr = base + r.addr;
		if (!read(addr, bytes.data(), r.size)) continue;
		uint64_t hash    = 1469598103934665603ull;
		uint32_t nonzero = 0;
		for (const auto b: bytes) {
			hash = (hash ^ b) * 1099511628211ull;
			nonzero += b != 0 ? 1u : 0u;
		}
		const bool gpu_modified = context.GetBufferCache().IsRegionGpuModified(addr, r.size);
		if (!r.seen || hash != r.hash || gpu_modified != r.gpu_modified) {
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1) < 400u) {
				uint64_t value = 0;
				if (r.size <= 8u) std::memcpy(&value, bytes.data(), r.size);
				std::fprintf(stderr,
				             "GUEST_WATCH addr=0x%llx frame=%u cpu_view_nonzero=%u (was %u)%s value=0x%llx gpu_modified=%d "
				             "after_cs=%016llx before_cs=%016llx marker=[%s]\n",
				             static_cast<unsigned long long>(addr), DebugPresentCount().load(), nonzero, r.nonzero,
				             r.seen && hash != r.hash ? " CPU_VIEW_CHANGED" : "", static_cast<unsigned long long>(value),
				             gpu_modified ? 1 : 0,
				             static_cast<unsigned long long>(last_hash), static_cast<unsigned long long>(next_hash),
				             DebugMarkerPath().c_str());
			}
			r.hash = hash, r.nonzero = nonzero, r.gpu_modified = gpu_modified, r.seen = true;
		}
	}
	last_hash = next_hash;
}
} // namespace GuestWatch

// KYTY_DIAG_GUEST_POKE=<hex addr>:<hex value> (logged APPROXIMATION, measurement only): before every compute dispatch
// the GPU thread writes the 4-byte value to the guest address (e.g. a game setting the CPU recomputes each frame, to
// see what the image does with another value). Changes what the game does; never a fix.
namespace GuestPoke {
void Poll() {
	struct Spec {
		uint64_t addr  = 0;
		uint32_t value = 0;
	};
	static const Spec spec = [] {
		Spec        result;
		const char* value = std::getenv("KYTY_DIAG_GUEST_POKE");
		if (value == nullptr) return result;
		char* end   = nullptr;
		result.addr = std::strtoull(value, &end, 16);
		if (end != nullptr && *end == ':') result.value = static_cast<uint32_t>(std::strtoul(end + 1, nullptr, 16));
		std::fprintf(stderr, "DIAG_GUEST_POKE APPROXIMATION addr=0x%llx value=0x%x\n", static_cast<unsigned long long>(result.addr),
		             result.value);
		return result;
	}();
	if (spec.addr == 0) return;
	if (!Libs::LibKernel::Memory::TryWriteBacking(spec.addr, &spec.value, sizeof(spec.value))) {
		(void)Libs::LibKernel::Memory::GuardedGuestCopy(reinterpret_cast<void*>(spec.addr), &spec.value, sizeof(spec.value));
	}
}
} // namespace GuestPoke

// KYTY_TRACE_OBJ_CENSUS=<registry>:<types>:<stride>:<head>:<next>:<flag>:<every> (hex, <every> decimal; diagnostic): every
// <every> presented frames the GPU thread walks the guest's per-type object lists - list head = 8 bytes at registry +
// type * stride + head, next object = 8 bytes at object + next - and logs per type the objects and those with bit 0 of
// the byte at object + flag set ("active"), from the CPU view without a GPU sync. One OBJ_CENSUS line per walk, types
// with objects only, as t<type>=<objects>/<active>.
namespace ObjCensus {
void Poll() {
	struct Spec {
		uint64_t registry = 0, types = 0, stride = 0x10, head = 0, next = 0, flag = 0;
		uint32_t every = 300;
	};
	static const Spec spec = [] {
		Spec        result;
		const char* value = std::getenv("KYTY_TRACE_OBJ_CENSUS");
		uint64_t    fields[7] {};
		for (int i = 0; i < 7 && value != nullptr && *value != 0; i++) {
			char* end = nullptr;
			fields[i] = std::strtoull(value, &end, i == 6 ? 10 : 16);
			value     = end != nullptr && *end == ':' ? end + 1 : nullptr;
		}
		result = {fields[0], fields[1], fields[2], fields[3], fields[4], fields[5], static_cast<uint32_t>(fields[6])};
		if (result.every == 0) result.every = 300;
		std::fprintf(stderr, "OBJ_CENSUS spec registry=0x%llx types=0x%llx stride=0x%llx head=0x%llx next=0x%llx flag=0x%llx every=%u\n",
		             static_cast<unsigned long long>(result.registry), static_cast<unsigned long long>(result.types),
		             static_cast<unsigned long long>(result.stride), static_cast<unsigned long long>(result.head),
		             static_cast<unsigned long long>(result.next), static_cast<unsigned long long>(result.flag), result.every);
		return result;
	}();
	static uint32_t last_frame = 0;
	const uint32_t  frame      = DebugPresentCount().load();
	if (spec.registry == 0 || frame == last_frame || frame % spec.every != 0) return;
	last_frame      = frame;
	const auto read = [](uint64_t addr, void* data, uint64_t size) {
		return Libs::LibKernel::Memory::TryReadBacking(addr, data, size) ||
		       Libs::LibKernel::Memory::GuardedGuestCopy(data, reinterpret_cast<const void*>(addr), size);
	};
	std::string line;
	for (uint64_t t = 0; t < spec.types; t++) {
		uint64_t object = 0;
		if (!read(spec.registry + t * spec.stride + spec.head, &object, sizeof(object))) continue;
		uint32_t objects = 0, active = 0;
		// A list that loops or runs away ends at 100000 objects.
		for (; object != 0 && objects < 100000u; objects++) {
			uint8_t flags = 0;
			if (read(object + spec.flag, &flags, 1) && (flags & 1u) != 0) active++;
			if (!read(object + spec.next, &object, sizeof(object))) object = 0;
		}
		if (objects != 0) {
			char text[48];
			std::snprintf(text, sizeof(text), " t%llx=%u/%u", static_cast<unsigned long long>(t), objects, active);
			line += text;
		}
	}
	std::fprintf(stderr, "OBJ_CENSUS frame=%u%s\n", frame, line.c_str());
	// KYTY_TRACE_OBJ_CENSUS_DETAIL=<type hex>[,...]: for the listed types also why their objects are off - per group of
	// objects with the same values, the disable-reason byte (+0xd6), the handle count (+0xde) and the parent chain
	// (+0x40, up to 6 levels) with each parent's vtable, reason byte, data type and id (+0x30) and container (+0x38):
	// OBJ_DETAIL lines.
	static const std::vector<uint64_t> detail = [] {
		std::vector<uint64_t> list;
		const char*           value = std::getenv("KYTY_TRACE_OBJ_CENSUS_DETAIL");
		while (value != nullptr && *value != 0) {
			char* end = nullptr;
			list.push_back(std::strtoull(value, &end, 16));
			value = end != nullptr && *end == ',' ? end + 1 : nullptr;
		}
		return list;
	}();
	for (const uint64_t t: detail) {
		uint64_t object = 0;
		if (!read(spec.registry + t * spec.stride + spec.head, &object, sizeof(object))) continue;
		std::map<std::string, uint32_t> groups;
		for (uint32_t n = 0; object != 0 && n < 100000u; n++) {
			uint8_t reasons = 0, handles = 0;
			read(object + 0xd6, &reasons, 1);
			read(object + 0xde, &handles, 1);
			char text[192];
			std::snprintf(text, sizeof(text), "reasons=0x%02x handles=%u parents:", reasons, handles);
			std::string key    = text;
			uint64_t    parent = 0;
			if (!read(object + 0x40, &parent, sizeof(parent))) parent = 0;
			for (int level = 0; parent != 0 && level < 6; level++) {
				uint64_t vtable = 0, data = 0, data_id = 0, container = 0;
				uint32_t data_type      = 0;
				uint8_t  parent_reasons = 0;
				read(parent, &vtable, sizeof(vtable));
				read(parent + 0xd6, &parent_reasons, 1);
				read(parent + 0x38, &container, sizeof(container));
				// The object's world-file data (+0x30): its data type (+0x10) and 64-bit object id (+0x18).
				if (read(parent + 0x30, &data, sizeof(data)) && data != 0) {
					read(data + 0x10, &data_type, sizeof(data_type));
					read(data + 0x18, &data_id, sizeof(data_id));
				}
				std::snprintf(text, sizeof(text), " [0x%llx vt=0x%llx r=0x%02x data=0x%x:%016llx in=0x%llx]",
				              static_cast<unsigned long long>(parent), static_cast<unsigned long long>(vtable), parent_reasons,
				              data_type, static_cast<unsigned long long>(data_id), static_cast<unsigned long long>(container));
				key += text;
				if (!read(parent + 0x40, &parent, sizeof(parent))) parent = 0;
			}
			groups[key]++;
			if (!read(object + spec.next, &object, sizeof(object))) object = 0;
		}
		for (const auto& [key, count]: groups) {
			std::fprintf(stderr, "OBJ_DETAIL frame=%u t%llx n=%u %s\n", frame, static_cast<unsigned long long>(t), count, key.c_str());
		}
	}
}
} // namespace ObjCensus

// KYTY_TRACE_CS_IMAGE_DUMP=<hex,...>: at the listed direct dispatches (3 per shader once 1000
// frames were presented), every bound image as the GPU sees it before the dispatch, to
// <KYTY_TRACE_FRAME_DIR>/csimg_<hash>_<n>_b<binding>_<addr>_<WxH>_<fmt>.raw, plus an IMAGE_DUMP
// line per binding (bound image id/extent/format vs the descriptor). Diagnostic only.
void CsImageDumpTrace(RenderContext& context, vk::CommandBuffer vk_buffer,
                      const ShaderRecompiler::IR::CompiledShaderInfo& program, PreparedBindings& bindings,
                      const ShaderRecompiler::IR::ResourceSnapshot* resources) {
	static constinit Common::OnceObject<std::string> list_once;
	const std::string& list = list_once.Get([] { return Common::EnvRead::CsImageDump(); });
	// KYTY_TRACE_CS_IMAGE_DUMP_FROM=<frame> (default 1000): first presented frame that dumps.
	static constinit Common::OnceValue<uint32_t> from_once;
	const uint32_t from = from_once.Get([] {
		const char* value = std::getenv("KYTY_TRACE_CS_IMAGE_DUMP_FROM");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 1000u;
	});
	// Also inside the on-demand stencil/water trace window (KYTY_TRACE_STENCIL_TRIGGER), 2 per shader.
	const bool window = DebugStencilWindow();
	if (list.empty() || (DebugPresentCount().load() < from && !window)) return;
	const char* dir = Common::EnvRead::FrameDir();
	if (dir == nullptr) return;
	char text[17];
	std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(program.shader_hash));
	if (list.find(text) == std::string::npos) return;
	static std::unordered_map<uint64_t, unsigned> counts;
	static std::unordered_map<uint64_t, unsigned> window_counts;
	unsigned n = 0;
	if (window) {
		// A new window (more than 3 frames after the last windowed dump) starts the counts again.
		static uint32_t last_window_frame = 0;
		const auto      frame             = DebugPresentCount().load();
		if (frame > last_window_frame + 3u) window_counts.clear();
		last_window_frame = frame;
		n = 100u + window_counts[program.shader_hash]++;
		if (n >= 102u) return;
	} else {
		n = counts[program.shader_hash]++;
		if (n >= 3u) return;
	}
	auto& cache = context.GetTextureCache();
	for (uint32_t i = 0; i < bindings.images.size(); i++) {
		auto& binding = bindings.images[i];
		if (!binding.image_id) continue;
		auto& image = cache.GetImage(binding.image_id);
		std::fprintf(stderr,
		             "IMAGE_DUMP frame=%u hash=0x%s n=%u binding=%u %s desc_addr=0x%llx desc=%ux%u bound_addr=0x%llx bound=%ux%u fmt=%s depth=%d stencil=%d depth_id=%d\n",
		             DebugPresentCount().load(), text, n, i, program.info.images[i].written ? "write" : "read",
		             static_cast<unsigned long long>(binding.desc.info.data.address), binding.desc.info.extent.width,
		             binding.desc.info.extent.height, static_cast<unsigned long long>(image.info.data.address),
		             image.backing.extent.width, image.backing.extent.height, vk::to_string(image.backing.format).c_str(),
		             image.info.IsDepth() ? 1 : 0, image.info.HasStencil() ? 1 : 0, image.depth_id ? 1 : 0);
		char name[200];
		std::snprintf(name, sizeof(name), "%s/csimg_%s_%u_b%u_%llx", dir, text, n, i,
		              static_cast<unsigned long long>(binding.desc.info.data.address));
		DebugQueueFullDump(context.GetGraphics(), vk_buffer, image, name, false);
		// The bindings were committed with this layout; the dump left the image in a transfer layout.
		if (binding.layout != vk::ImageLayout::eUndefined) {
			image.Transit(binding.layout,
			              program.info.images[i].written
			                  ? vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite
			                  : vk::AccessFlagBits2::eShaderRead,
			              {}, vk_buffer);
		}
	}
	// The bound buffers (<= 1 MiB each) as the GPU sees them before the dispatch.
	for (uint32_t j = 0; resources != nullptr && j < program.info.buffers.size() && j < resources->buffers.size(); j++) {
		const auto v     = DecodeNativeDescriptor<ShaderBufferResource>(resources->buffers[j]);
		const auto bytes = std::min<uint64_t>(v.GetSize(), 1ull << 20u) & ~uint64_t {3};
		std::fprintf(stderr, "BUFFER_DUMP frame=%u hash=0x%s n=%u buf=%u %s addr=0x%llx stride=%u records=%u\n",
		             DebugPresentCount().load(), text, n, j, program.info.buffers[j].written ? "rw" : "ro",
		             static_cast<unsigned long long>(v.Base48()), v.Stride(), v.NumRecords());
		if (v.Base48() == 0 || bytes == 0) continue;
		auto [source, source_offset] = context.GetBufferCache().ObtainBuffer(v.Base48(), bytes, false);
		if (source == nullptr) continue;
		char name[200];
		std::snprintf(name, sizeof(name), "%s/csbuf_%s_%u_b%u_%llx.bin", dir, text, n, j,
		              static_cast<unsigned long long>(v.Base48()));
		DebugQueueBufferDump(context.GetGraphics(), vk_buffer, source->Handle(), source_offset, bytes, name);
	}
	if (resources != nullptr && !resources->user_data.empty()) {
		std::string line;
		for (size_t k = 0; k < std::min<size_t>(16u, resources->user_data.size()); k++) line += fmt::format(" {:08x}", resources->user_data[k]);
		std::fprintf(stderr, "CS_USER_DATA frame=%u hash=0x%s n=%u%s\n", DebugPresentCount().load(), text, n, line.c_str());
		// s[0:1] usually points at the shader's constant table: its first 32 dwords (guest memory, CPU view).
		if (resources->user_data.size() >= 2u) {
			const auto table = static_cast<uint64_t>(resources->user_data[0]) |
			                   (static_cast<uint64_t>(resources->user_data[1] & 0xffffu) << 32u);
			uint32_t words[32] {};
			if (table != 0 && (table & 3u) == 0 &&
			    Libs::LibKernel::Memory::TryReadBacking(table, words, sizeof(words))) {
				std::string table_line;
				for (int k = 0; k < 32; k++) table_line += fmt::format(" {}:{:08x}", k, words[k]);
				std::fprintf(stderr, "CS_TABLE frame=%u hash=0x%s n=%u table=0x%llx%s\n", DebugPresentCount().load(), text, n,
				             static_cast<unsigned long long>(table), table_line.c_str());
			}
		}
	}
}

// KYTY_TRACE_PROXY_COUNTS=1 (diagnostic, default off; M118): right after each proxy-set cull dispatch (CS 8457) in frames
// >= 1500 with frame % 10 < 4, the per-set visible-instance counts (V# at SRT+0x20, 4 B per set) as the GPU wrote them,
// to <KYTY_TRACE_FRAME_DIR>/proxycnt_<frame>_<srt>_<sets>.bin - compared offline with the next Fill Streams count (what
// the game's CPU read).
void ProxyCountsDump(RenderContext& context, vk::CommandBuffer vk_buffer,
                     const ShaderRecompiler::IR::CompiledShaderInfo& program,
                     const ShaderRecompiler::IR::ResourceSnapshot* resources) {
	if (!ProxyCountsOn() || program.shader_hash != 0x8457901d80b91921ull || resources == nullptr || resources->user_data.size() < 2u) {
		return;
	}
	const uint32_t frame = DebugPresentCount().load();
	const char*    dir   = Common::EnvRead::FrameDir();
	if (dir == nullptr || frame < 1500u || frame % 10u >= 4u) return;
	const uint64_t srt = resources->user_data[0] | (uint64_t {resources->user_data[1]} << 32u);
	uint32_t       words[12] {};
	if (!LibKernel::Memory::TryReadBacking(srt, words, sizeof(words))) return;
	const uint64_t base  = words[8] | (uint64_t {words[9] & 0xffffu} << 32u);
	const uint32_t sets  = std::min<uint32_t>(words[10], 1u << 16u);
	const uint64_t bytes = uint64_t {sets} * 4u;
	if (base == 0 || bytes == 0) return;
	auto [source, source_offset] = context.GetBufferCache().ObtainBuffer(base, bytes, false);
	if (source == nullptr) return;
	char name[200];
	std::snprintf(name, sizeof(name), "%s/proxycnt_%u_%llx_%u.bin", dir, frame, static_cast<unsigned long long>(srt), sets);
	DebugQueueBufferDump(context.GetGraphics(), vk_buffer, source->Handle(), source_offset, bytes, name);
}

// KYTY_TRACE_PROXY_PHASE=1 (diagnostic, default off; M118, the open-world flicker after a cutscene): per presented frame
// one PROXY_PHASE line - the proxy-set cull dispatches (CS 8457), the Fill Streams dispatches (e80c5289) and all
// dispatches the GPU thread translated in it. A healthy frame has one cull; the flicker state alternates 0 and 2
// (ZNIKANIE-WIDOK-GLOWNY-ANALIZA.md appendix C). Counting only, on the GPU thread.
void ProxyPhaseNote(uint64_t shader_hash, const ShaderRecompiler::IR::ResourceSnapshot* resources) {
	static const bool on = std::getenv("KYTY_TRACE_PROXY_PHASE") != nullptr;
	if (!on) return;
	static uint32_t frame = 0, culls = 0, fills = 0, dispatches = 0;
	// The hi-Z pyramid chain the cull reads (analysis appendix D): level 0 from depth, scratch clear, reprojection into
	// the next camera, hole fill, the two mip passes.
	static constexpr std::array<uint32_t, 6> HizHashes {0x9dcf7a37u, 0x004abfcdu, 0xf00717deu, 0xb9b4f925u, 0xe584166fu, 0x32839016u};
	static std::array<uint32_t, 6>           hiz {};
	const uint32_t                           now = DebugPresentCount().load();
	if (now != frame) {
		if (frame != 0) {
			std::fprintf(stderr, "PROXY_PHASE f=%u culls=%u fills=%u cs=%u hiz=%u/%u/%u/%u/%u/%u%s\n", frame, culls, fills,
			             dispatches, hiz[0], hiz[1], hiz[2], hiz[3], hiz[4], hiz[5], culls != 1u ? " <<" : "");
		}
		frame = now;
		culls = fills = dispatches = 0;
		hiz.fill(0u);
	}
	for (size_t k = 0; k < HizHashes.size(); k++) {
		if ((shader_hash >> 32u) == HizHashes[k]) hiz[k]++;
	}
	dispatches++;
	if (shader_hash == 0x8457901d80b91921ull) {
		culls++;
		// The cull's inputs as guest memory holds them at its translation: its SRT block (user data s[0:1]), hashed in
		// 64-byte pieces, and the first 16 words as floats - a camera cut that the cull does not see yet keeps the hashes.
		if (resources != nullptr && resources->user_data.size() >= 2u) {
			const uint64_t srt = resources->user_data[0] | (uint64_t {resources->user_data[1]} << 32u);
			uint32_t       words[128] {};
			if (LibKernel::Memory::TryReadBacking(srt, words, sizeof(words))) {
				std::string line;
				for (uint32_t k = 0; k < 8u; k++) line += fmt::format(" {:016x}", XXH3_64bits(words + 16u * k, 64u));
				line += " |";
				for (uint32_t k = 0; k < 16u; k++) line += fmt::format(" {:g}", std::bit_cast<float>(words[k]));
				std::fprintf(stderr, "PROXY_CULL f=%u srt=0x%llx%s\n", now, static_cast<unsigned long long>(srt), line.c_str());
			}
		}
	}
	if ((shader_hash >> 32u) == 0xe80c5289ull) fills++;
}

void NoteRtDispatch(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                    const ShaderRecompiler::IR::ResourceSnapshot& resources) {
	std::string writes;
	std::string reads;
	char        text[64];
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		const auto r = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
		std::snprintf(text, sizeof(text), " i%llx/%u", static_cast<unsigned long long>(r.Base40()),
		              static_cast<uint32_t>(r.Format()));
		(program.info.images[i].written ? writes : reads) += text;
	}
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		const auto r = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		std::snprintf(text, sizeof(text), " b%llx", static_cast<unsigned long long>(r.Base48()));
		(program.info.buffers[i].written ? writes : reads) += text;
	}
	const auto       marker = DebugMarkerPath();
	std::scoped_lock lock(DebugRtMutex());
	auto&            records = DebugCsFrame();
	auto it = std::ranges::find_if(records, [&](const DebugCsRecord& r) {
		return r.hash == program.shader_hash && r.marker == marker;
	});
	if (it != records.end()) {
		it->count++;
	} else if (records.size() < 512u) {
		records.push_back({marker, program.shader_hash, writes.substr(0, 400), reads.substr(0, 600), 1u});
	}
}

void LogDispatch(RenderContext& context, uint64_t cs_address, const ShaderComputeInputInfo& input_info, uint32_t thread_group_x,
                 uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode) {
	const auto& program   = *input_info.stage.program;
	const auto& resources = *input_info.stage.resources;
	const auto sampled_images = std::count_if(
	    program.info.images.begin(), program.info.images.end(), [](const auto& image) {
		    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
	    });
	const uint32_t frame_num = static_cast<uint32_t>(context.GetGpu().GetFrameNum());
	LOGF("GraphicsRenderDispatchDirect: frame=%u shader=0x%016" PRIx64
	     " hash=0x%016" PRIx64 " tick=%" PRIu64 " completed_tick=%" PRIu64
	     " groups=%ux%ux%u mode=0x%08" PRIx32 " local=%ux%ux%u "
	     "buffers=%zu textures=%zu sampled=%zu storage=%zu samplers=%zu push=%u\n",
	     frame_num, cs_address, program.shader_hash,
	     context.GetCommandScheduler().CurrentTick(),
	     context.GetCommandScheduler().GetMasterSemaphore().KnownGpuTick(),
	     thread_group_x, thread_group_y, thread_group_z, mode,
	     input_info.threads_num[0], input_info.threads_num[1],
	     input_info.threads_num[2], program.info.buffers.size(), program.info.images.size(),
	     sampled_images, program.info.images.size() - sampled_images,
	     program.info.samplers.size(),
	     program.bindings.UsesPushData()
	         ? static_cast<uint32_t>(sizeof(ShaderRecompiler::IR::PushData))
	         : 0u);
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		const auto& buffer = program.info.buffers[i];
		const auto  r      = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		LOGF("  CS buffer[%u]: source=%u usage=%s addr=0x%012" PRIx64
		     " stride=%u records=%u format=%u\n",
		     i, buffer.source, buffer.written ? "read-write" : "read-only", r.Base48(),
		     r.Stride(), r.NumRecords(), r.RawFormat());
	}
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		const auto& image = program.info.images[i];
		const auto  r     = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
		LOGF("  CS texture[%u]: source=%u usage=%s sampled=%s addr=0x%010" PRIx64
		     " type=%u fmt=%u extent=%ux%u depth=%u levels=%u tile=%u\n",
		     i, image.source, image.written ? "read-write" : "read-only",
		     image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled
		         ? "true"
		         : "false",
		     r.Base40(), static_cast<uint32_t>(r.Type()), static_cast<uint32_t>(r.Format()),
		     static_cast<uint32_t>(r.Width5()) + 1u, static_cast<uint32_t>(r.Height5()) + 1u,
		     static_cast<uint32_t>(r.Depth()) + 1u,
		     r.Type() == Prospero::ImageType::kColor2DMsaa ||
		             r.Type() == Prospero::ImageType::kColor2DMsaaArray
		         ? 1u
		         : static_cast<uint32_t>(image.r128 ? r.LastLevel() : r.MaxMip()) + 1u,
		     static_cast<uint32_t>(r.TileMode()));
	}
	for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
		const auto r = DecodeNativeDescriptor<ShaderSamplerResource>(
		    resources.samplers[program.info.samplers[i].snapshot_index]);
		LOGF("  CS sampler[%u]: source=%u clamp=%u/%u/%u filter=%u/%u/%u mip=%u "
		     "lod=%u-%u bias=%d\n",
		     i, program.info.samplers[i].source, static_cast<uint32_t>(r.ClampX()),
		     static_cast<uint32_t>(r.ClampY()), static_cast<uint32_t>(r.ClampZ()),
		     static_cast<uint32_t>(r.XyMagFilter()), static_cast<uint32_t>(r.XyMinFilter()),
		     static_cast<uint32_t>(r.ZFilter()), static_cast<uint32_t>(r.MipFilter()),
		     static_cast<uint32_t>(r.MinLod()), static_cast<uint32_t>(r.MaxLod()),
		     static_cast<int32_t>(r.LodBias()));
	}
}

// KYTY_TRACE_CS_WRITER_ADDR=<hex address>: names every compute shader whose written buffer
// range covers that guest address (once per shader), with its other buffers.
void CsWriterTrace(uint64_t writer_addr, const ShaderRecompiler::IR::CompiledShaderInfo& program,
                   const ShaderRecompiler::IR::ResourceSnapshot& resources, uint32_t thread_group_x,
                   uint32_t thread_group_y, uint32_t thread_group_z) {
	static std::mutex                 writer_mutex;
	static std::vector<uint64_t>      writer_seen;
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		if (!program.info.buffers[i].written) continue;
		const auto r      = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		const auto base   = r.Base48();
		const auto stride = std::max<uint64_t>(r.Stride(), 1u);
		const auto size   = uint64_t {r.NumRecords()} * stride;
		if (writer_addr < base || writer_addr >= base + size) continue;
		std::scoped_lock lock(writer_mutex);
		if (std::ranges::find(writer_seen, program.shader_hash) != writer_seen.end()) break;
		writer_seen.push_back(program.shader_hash);
		std::fprintf(stderr, "CS_WRITER addr=0x%llx hash=0x%016llx groups=%ux%ux%u wbuf=%u base=0x%llx size=0x%llx\n",
		             static_cast<unsigned long long>(writer_addr),
		             static_cast<unsigned long long>(program.shader_hash), thread_group_x, thread_group_y,
		             thread_group_z, i, static_cast<unsigned long long>(base), static_cast<unsigned long long>(size));
		for (uint32_t j = 0; j < program.info.buffers.size(); j++) {
			const auto b = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[j]);
			std::fprintf(stderr, "CS_WRITER   buffer[%u] %s addr=0x%llx stride=%u records=%u\n", j,
			             program.info.buffers[j].written ? "rw" : "ro",
			             static_cast<unsigned long long>(b.Base48()), b.Stride(), b.NumRecords());
		}
		break;
	}
}

// Capture frame: images written by a KYTY_CAPTURE_PRE shader, dumped before the dispatch
// (then returned to the general layout the dispatch binds them with).
void CapturePreDirect(RenderContext& context, CommandBuffer& buffer, vk::CommandBuffer vk_buffer,
                      const ShaderRecompiler::IR::CompiledShaderInfo& program,
                      const ShaderRecompiler::IR::ResourceSnapshot& resources) {
	static const std::string capture_pre = Common::EnvRead::CapturePre();
	char hash_text[17];
	std::snprintf(hash_text, sizeof(hash_text), "%016llx", static_cast<unsigned long long>(program.shader_hash));
	const char* dir = Common::EnvRead::FrameDir();
	if (dir != nullptr && !capture_pre.empty() && capture_pre.find(hash_text) != std::string::npos) {
		auto& cache = buffer.GetContext().GetTextureCache();
		for (uint32_t j = 0; j < program.info.images.size(); j++) {
			if (!program.info.images[j].written) continue;
			const auto r       = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[j]);
			const auto address = r.Base40();
			const auto images  = cache.DebugImagesAt(address);
			for (size_t k = 0; k < images.size() && k < 2u; ++k) {
				char name[128];
				std::snprintf(name, sizeof(name), "/pre_%u_%s_out%u_%zu_%llx", DebugCaptureIndex().load(),
				              hash_text, j, k, static_cast<unsigned long long>(address));
				DebugQueueFullDump(context.GetGraphics(), vk_buffer, *images[k], std::string(dir) + name);
				images[k]->Transit(vk::ImageLayout::eGeneral,
				                   vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite, {},
				                   vk_buffer);
			}
		}
	}
}

// Capture frame (M120): images written by a KYTY_CAPTURE_PRE shader, dumped before its indirect dispatch too
// (pre_<index>_<hash>_out<j>_<k>_<addr>, the index the dispatch's CAPTURE_CS_INDIRECT line gets), then returned to
// the general layout the dispatch binds them with. Diagnostic only.
void CapturePreIndirect(RenderContext& context, CommandBuffer& buffer, vk::CommandBuffer vk_buffer,
                      const ShaderRecompiler::IR::CompiledShaderInfo& program,
                      const ShaderRecompiler::IR::ResourceSnapshot& resources) {
	static const std::string capture_pre = Common::EnvRead::CapturePre();
	char hash_text[17];
	std::snprintf(hash_text, sizeof(hash_text), "%016llx", static_cast<unsigned long long>(program.shader_hash));
	const char* dir = Common::EnvRead::FrameDir();
	if (dir != nullptr && !capture_pre.empty() && capture_pre.find(hash_text) != std::string::npos) {
		auto&       cache     = buffer.GetContext().GetTextureCache();
		for (uint32_t j = 0; j < program.info.images.size() && j < resources.images.size(); j++) {
			if (!program.info.images[j].written) continue;
			const auto r       = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[j]);
			const auto address = r.Base40();
			const auto images  = cache.DebugImagesAt(address);
			for (size_t k = 0; k < images.size() && k < 2u; ++k) {
				char name[128];
				std::snprintf(name, sizeof(name), "/pre_%03u_%s_out%u_%zu_%llx", DebugCaptureIndex().load(), hash_text,
				              j, k, static_cast<unsigned long long>(address));
				DebugQueueFullDump(context.GetGraphics(), vk_buffer, *images[k], std::string(dir) + name);
				images[k]->Transit(vk::ImageLayout::eGeneral,
				                   vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite, {}, vk_buffer);
			}
		}
	}
}

// KYTY_TRACE_CS_IMAGE_DUMP also dumps the images a listed compute shader wrote, right after its
// first two dispatches inside the trace window (csafter_<hash>_<n>_b<binding>_<addr>).
void CsImageDumpAfter(RenderContext& context, vk::CommandBuffer vk_buffer,
                      const ShaderRecompiler::IR::CompiledShaderInfo& program, PreparedBindings& bindings) {
	static const std::string after_list = Common::EnvRead::CsImageDump();
	const char* dir = Common::EnvRead::FrameDir();
	char        text[17];
	std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(program.shader_hash));
	static std::unordered_map<uint64_t, unsigned> after_counts;
	// A new window (more than 3 frames after the last windowed dump) starts the counts again.
	static uint32_t after_last_frame = 0;
	if (const auto frame = DebugPresentCount().load(); frame > after_last_frame + 3u) after_counts.clear();
	after_last_frame = DebugPresentCount().load();
	if (dir != nullptr && !after_list.empty() && after_list.find(text) != std::string::npos &&
	    after_counts[program.shader_hash] < 2u) {
		const auto n = after_counts[program.shader_hash]++;
		// The dispatch's writes must land before the copy.
		vk::MemoryBarrier2 barrier {};
		barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
		barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		vk::DependencyInfo dependency {};
		dependency.memoryBarrierCount = 1;
		dependency.pMemoryBarriers    = &barrier;
		vk_buffer.pipelineBarrier2(dependency);
		for (uint32_t i = 0; i < bindings.images.size() && i < program.info.images.size(); i++) {
			if (!program.info.images[i].written || !bindings.images[i].image_id) continue;
			auto& image = context.GetTextureCache().GetImage(bindings.images[i].image_id);
			char  name[200];
			std::snprintf(name, sizeof(name), "%s/csafter_%s_%u_b%u_%llx", dir, text, n, i,
			              static_cast<unsigned long long>(bindings.images[i].desc.info.data.address));
			DebugQueueFullDump(context.GetGraphics(), vk_buffer, image, name, false);
		}
	}
}

void CaptureAfterDirect(RenderContext& context, CommandBuffer& buffer, vk::CommandBuffer vk_buffer,
                        const ShaderRecompiler::IR::CompiledShaderInfo& program,
                        const ShaderRecompiler::IR::ResourceSnapshot& resources, uint32_t thread_group_x,
                        uint32_t thread_group_y, uint32_t thread_group_z) {
	const auto index = DebugCaptureIndex().fetch_add(1);
	const char* dir  = Common::EnvRead::FrameDir();
	auto&       cache = buffer.GetContext().GetTextureCache();
	std::fprintf(stderr, "CAPTURE_CS index=%u hash=0x%016llx groups=%ux%ux%u gds=%d marker=[%s]\n", index,
	             static_cast<unsigned long long>(program.shader_hash), thread_group_x, thread_group_y,
	             thread_group_z,
	             program.bindings.descriptor_counts[static_cast<size_t>(ShaderRecompiler::IR::DescriptorBindingKind::Gds)] != 0u ? 1 : 0,
	             DebugMarkerPath().c_str());
	static const std::string capture_inputs = Common::EnvRead::CaptureInputs();
	char hash_text[17];
	std::snprintf(hash_text, sizeof(hash_text), "%016llx", static_cast<unsigned long long>(program.shader_hash));
	const bool all_images = !capture_inputs.empty() && capture_inputs.find(hash_text) != std::string::npos;
	// Listed shaders: 1 KiB at the user-data pointer s[0:1] (scalar constants table), GPU-side copy.
	if (dir != nullptr && all_images && resources.user_data.size() >= 2u) {
		const auto table = static_cast<uint64_t>(resources.user_data[0]) |
		                   (static_cast<uint64_t>(resources.user_data[1] & 0xffffu) << 32u);
		std::fprintf(stderr, "CAPTURE_USERDATA index=%u hash=%s s0..s3=%08x %08x %08x %08x table=0x%llx\n", index,
		             hash_text, resources.user_data[0], resources.user_data[1],
		             resources.user_data.size() > 2u ? resources.user_data[2] : 0u,
		             resources.user_data.size() > 3u ? resources.user_data[3] : 0u,
		             static_cast<unsigned long long>(table));
		if (table != 0 && (table & 3u) == 0) {
			auto [source, source_offset] = context.GetBufferCache().ObtainBuffer(table, 1024u, false);
			if (source != nullptr) {
				char name[160];
				std::snprintf(name, sizeof(name), "%s/userdata_%03u_%s_%llx.bin", dir, index, hash_text,
				              static_cast<unsigned long long>(table));
				DebugQueueBufferDump(context.GetGraphics(), vk_buffer, source->Handle(), source_offset, 1024u,
				                     name);
			}
		}
	}
	// Listed shaders: the GPU-side contents of every bound buffer after the dispatch (<= 4 MiB each).
	for (uint32_t j = 0; dir != nullptr && all_images && j < program.info.buffers.size(); j++) {
		const auto b     = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[j]);
		const auto bytes = std::min<uint64_t>(b.GetSize(), 4ull << 20u) & ~uint64_t {3};
		std::fprintf(stderr, "CAPTURE_BUFFER index=%u hash=%s buf=%u %s addr=0x%llx stride=%u records=%u\n", index,
		             hash_text, j, program.info.buffers[j].written ? "rw" : "ro",
		             static_cast<unsigned long long>(b.Base48()), b.Stride(), b.NumRecords());
		if (b.Base48() == 0 || bytes == 0) continue;
		auto [source, source_offset] = context.GetBufferCache().ObtainBuffer(b.Base48(), bytes, false);
		if (source == nullptr) continue;
		char name[160];
		std::snprintf(name, sizeof(name), "%s/capbuf_%03u_%s_buf%u_%llx.bin", dir, index, hash_text, j,
		              static_cast<unsigned long long>(b.Base48()));
		DebugQueueBufferDump(context.GetGraphics(), vk_buffer, source->Handle(), source_offset, bytes, name);
	}
	// KYTY_CAPTURE_CS_FROM=<index>: outputs of earlier dispatches are not dumped (still listed), so
	// the dump budget reaches the late passes of a frame (post-processing).
	static const uint32_t capture_from = [] {
		const char* value = std::getenv("KYTY_CAPTURE_CS_FROM");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
	}();
	for (uint32_t j = 0; dir != nullptr && ((index >= capture_from && index < capture_from + 1200u) || all_images) &&
	                     j < program.info.images.size();
	     j++) {
		if (!program.info.images[j].written && !all_images) continue;
		const auto r       = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[j]);
		const auto address = r.Base40();
		const auto images  = cache.DebugImagesAt(address);
		if (all_images) {
			std::fprintf(stderr,
			             "CAPTURE_INPUT index=%u hash=%s img=%u %s addr=0x%llx type=%u fmt=%u %ux%ux%u null=%d "
			             "cached=%zu dwords=%08x %08x %08x %08x %08x %08x %08x %08x\n",
			             index, hash_text, j, program.info.images[j].written ? "out" : "in",
			             static_cast<unsigned long long>(address), static_cast<uint32_t>(r.Type()),
			             static_cast<uint32_t>(r.Format()), static_cast<uint32_t>(r.Width5()) + 1u,
			             static_cast<uint32_t>(r.Height5()) + 1u, static_cast<uint32_t>(r.Depth()) + 1u,
			             r.IsNull() ? 1 : 0, images.size(), r.fields[0], r.fields[1], r.fields[2], r.fields[3],
			             r.fields[4], r.fields[5], r.fields[6], r.fields[7]);
			if (program.info.images[j].written) cache.DebugDescribeRegion("capture_out", address, 1);
		}
		for (size_t k = 0; k < images.size() && k < 2u; ++k) {
			char name[128];
			std::snprintf(name, sizeof(name), "/cap_%03u_%016llx_out%u_%zu_%llx", index,
			              static_cast<unsigned long long>(program.shader_hash), j, k,
			              static_cast<unsigned long long>(address));
			DebugQueueFullDump(context.GetGraphics(), vk_buffer, *images[k], std::string(dir) + name);
		}
	}
}

// Capture frame: indirect dispatches, their GPU-side arguments and (for KYTY_CAPTURE_INPUTS
// hashes) the images they write.
void CaptureAfterIndirect(RenderContext& context, CommandBuffer& buffer, vk::CommandBuffer vk_buffer,
                          const ShaderRecompiler::IR::CompiledShaderInfo& program,
                          const ShaderRecompiler::IR::ResourceSnapshot& resources, uint64_t args_addr,
                          const Buffer& args_buffer, uint64_t args_offset) {
	const auto  index = DebugCaptureIndex().fetch_add(1);
	const char* dir   = Common::EnvRead::FrameDir();
	char        hash_text[17];
	std::snprintf(hash_text, sizeof(hash_text), "%016llx", static_cast<unsigned long long>(program.shader_hash));
	std::fprintf(stderr, "CAPTURE_CS_INDIRECT index=%u hash=0x%s args=0x%llx gds=%d marker=[%s]\n", index, hash_text,
	             static_cast<unsigned long long>(args_addr),
	             program.bindings.descriptor_counts[static_cast<size_t>(ShaderRecompiler::IR::DescriptorBindingKind::Gds)] != 0u ? 1 : 0,
	             DebugMarkerPath().c_str());
	if (dir != nullptr) {
		char name[160];
		std::snprintf(name, sizeof(name), "%s/indargs_%03u_%s.bin", dir, index, hash_text);
		DebugQueueBufferDump(context.GetGraphics(), vk_buffer, args_buffer.Handle(), args_offset,
		                     sizeof(vk::DispatchIndirectCommand), name);
		static const std::string capture_inputs = Common::EnvRead::CaptureInputs();
		if (!capture_inputs.empty() && capture_inputs.find(hash_text) != std::string::npos) {
			auto&       cache     = buffer.GetContext().GetTextureCache();
			for (uint32_t j = 0; j < program.info.images.size(); j++) {
				const auto r      = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[j]);
				const auto images = cache.DebugImagesAt(r.Base40());
				std::fprintf(stderr,
				             "CAPTURE_INPUT index=%u hash=%s img=%u %s addr=0x%llx type=%u fmt=%u %ux%ux%u null=%d "
				             "cached=%zu\n",
				             index, hash_text, j, program.info.images[j].written ? "out" : "in",
				             static_cast<unsigned long long>(r.Base40()), static_cast<uint32_t>(r.Type()),
				             static_cast<uint32_t>(r.Format()), static_cast<uint32_t>(r.Width5()) + 1u,
				             static_cast<uint32_t>(r.Height5()) + 1u, static_cast<uint32_t>(r.Depth()) + 1u,
				             r.IsNull() ? 1 : 0, images.size());
				for (size_t k = 0; k < images.size() && k < 2u; ++k) {
					std::snprintf(name, sizeof(name), "/cap_%03u_%s_out%u_%zu_%llx", index, hash_text, j, k,
					              static_cast<unsigned long long>(r.Base40()));
					DebugQueueFullDump(context.GetGraphics(), vk_buffer, *images[k], std::string(dir) + name);
				}
			}
		}
	}
}



} // namespace ComputeTrace

} // namespace Libs::Graphics
