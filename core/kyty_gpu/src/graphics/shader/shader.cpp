#include "graphics/shader/shader.h"

#include <chrono>

#include "common/assert.h"
#include "common/common.h"
#include "common/envRead.h"
#include "common/liveAb.h"
#include "common/lookupMemo.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/onceValue.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/workaroundHits.h"
#include "common/shaderEnv.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/Tessellation.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/shaderCompiler.h"
#include "graphics/shader/shaderVertexMetadata.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <string_view>
#include <atomic>
#include <bit>
#include <cstdio>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <xxhash.h>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#endif

namespace Libs::Graphics {

struct ShaderMapEntry {
	ShaderMappedData data;
	uint64_t hash;
};

static std::unique_ptr<std::unordered_map<uint64_t, ShaderMapEntry>> g_shader_map;
static std::mutex                                                      g_shader_map_mutex;
static std::unordered_map<uint64_t, std::chrono::steady_clock::time_point> g_shader_created; // under g_shader_map_mutex
// common/lookupMemo.h: bumped by every write of g_shader_map, under its mutex.
static std::atomic<uint64_t> g_shader_map_generation {1};

namespace {
// Per thread, the entry of an address as the map held it at `generation` (direct-mapped).
struct ShaderMapMemo {
	uint64_t       addr       = 0;
	uint64_t       generation = 0;
	ShaderMapEntry entry {};
};
thread_local std::array<ShaderMapMemo, 256> t_shader_map_memo {};

ShaderMapMemo& ShaderMapMemoSlot(uint64_t addr) {
	return t_shader_map_memo[(addr >> 8u) % t_shader_map_memo.size()];
}

// The memo's entry for `addr` when the map has not changed since it was taken (null otherwise).
const ShaderMapEntry* ShaderMapMemoFind(uint64_t addr) {
	const auto& memo = ShaderMapMemoSlot(addr);
	return memo.addr == addr && memo.generation == g_shader_map_generation.load(std::memory_order_acquire)
	           ? &memo.entry
	           : nullptr;
}
} // namespace
static std::unordered_map<uint64_t, ShaderTessellationInputInfo>       g_tess_layouts;
static std::mutex                                                      g_tess_layouts_mutex;

void ShaderInit() {
	EXIT_IF(g_shader_map != nullptr);

	g_shader_map = std::make_unique<std::unordered_map<uint64_t, ShaderMapEntry>>();
}

void ShaderMapUserData(uint64_t addr, const ShaderMappedData& data) {
	EXIT_IF(g_shader_map == nullptr);

	if (addr == 0 || data.code_size_bytes == 0 || data.code_size_bytes % sizeof(uint32_t) != 0) {
		EXIT("ShaderMapUserData shader=0x%016" PRIx64 " has invalid AGC shader_size=0x%08" PRIx32 "\n",
		     addr, data.code_size_bytes);
	}
	const auto hash = XXH3_64bits(reinterpret_cast<const void*>(addr), data.code_size_bytes);
	std::scoped_lock lock(g_shader_map_mutex);
	(*g_shader_map)[addr] = {data, hash};
	g_shader_map_generation.fetch_add(1, std::memory_order_release);
	// When the game first created this shader (PIPELINE_TIME: how long before its first pipeline compile).
	g_shader_created.try_emplace(hash, std::chrono::steady_clock::now());
}

double ShaderCreatedMsAgo(uint64_t hash) {
	std::scoped_lock lock(g_shader_map_mutex);
	const auto found = g_shader_created.find(hash);
	if (found == g_shader_created.end()) return -1.0;
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - found->second).count();
}

static ShaderMapEntry ShaderGetMappedData(uint64_t addr, const char* label) {
	EXIT_IF(g_shader_map == nullptr);

	if (const ShaderMapEntry* memo = ShaderMapMemoFind(addr); memo != nullptr) {
		Common::LookupMemo::NoteHit(Common::LookupMemo::Kind::Shader);
		return *memo;
	}

	std::scoped_lock lock(g_shader_map_mutex);

	if (auto iter = g_shader_map->find(addr); iter != g_shader_map->end()) {
		ShaderMapMemoSlot(addr) = {addr, g_shader_map_generation.load(std::memory_order_relaxed), iter->second};
		return iter->second;
	}

	EXIT("%s shader=0x%016" PRIx64 " is missing from ShaderMap\n", label, addr);
}

bool ShaderLookupMapped(uint64_t addr, ShaderMappedData* data, uint64_t* hash) {
	if (g_shader_map == nullptr) {
		return false;
	}

	if (const ShaderMapEntry* memo = ShaderMapMemoFind(addr); memo != nullptr) {
		Common::LookupMemo::NoteHit(Common::LookupMemo::Kind::Shader);
		if (data != nullptr) *data = memo->data;
		if (hash != nullptr) *hash = memo->hash;
		return true;
	}

	std::scoped_lock lock(g_shader_map_mutex);

	const auto iter = g_shader_map->find(addr);
	if (iter == g_shader_map->end()) {
		return false;
	}
	ShaderMapMemoSlot(addr) = {addr, g_shader_map_generation.load(std::memory_order_relaxed), iter->second};
	if (data != nullptr) {
		*data = iter->second.data;
	}
	if (hash != nullptr) {
		*hash = iter->second.hash;
	}
	return true;
}

// The hash lists of the per-dispatch checks below are compared as numbers (a list made only of 16-digit hex tokens; any
// other list keeps the text search), and the unsupported-shader verdict of a hash is remembered per thread in front of
// the shared map (a verdict depends only on the hash).
struct ShaderHashList {
	bool                  numeric = false; // every token is 16 hex digits
	std::vector<uint64_t> values;          // sorted
};

static ShaderHashList ParseShaderHashList(const std::string& text) {
	ShaderHashList list;
	list.numeric = true;
	size_t pos   = 0;
	while (pos <= text.size()) {
		const size_t end   = std::min(text.find(',', pos), text.size());
		const auto   token = std::string_view(text).substr(pos, end - pos);
		if (token.size() != 16u || !std::ranges::all_of(token, [](char c) {
			    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
		    })) {
			list.numeric = false;
			return list;
		}
		list.values.push_back(std::strtoull(std::string(token).c_str(), nullptr, 16));
		pos = end + 1u;
	}
	std::sort(list.values.begin(), list.values.end());
	return list;
}

bool ShaderComputeSkippedAsUnsupported(const HW::ComputeShaderInfo& regs) {
	static const bool enabled = [] {
		// Default from Common::EnvDefaults; workaround: D7 (skipped shaders,
		// loop caps), to be replaced by the proper implementation; =0 turns it off.
		const char* value = Common::ShaderEnv::Read("KYTY_DIAG_SKIP_UNSUPPORTED_CS");
		return value != nullptr && value[0] == '1';
	}();
	// KYTY_DIAG_SKIP_CS_HASHES=<hex,hex,...> (logged APPROXIMATION): listed compute shaders are
	// not dispatched (bisecting GPU faults).
	// Default from Common::EnvDefaults; workaround: D7 (skipped shaders,
	// loop caps), to be replaced by the proper implementation; =0 turns it off.
	static const std::string skip_list = Common::EnvRead::SkipCsHashes();
	// KYTY_DIAG_SKIP_CS_LIVE=<hex,hex,...> (logged APPROXIMATION, measurement only): the listed compute shaders are not
	// dispatched while KYTY_LIVE_AB has KYTY_DIAG_SKIP_CS_LIVE on (Switch::SkipCsLive): a same-run ceiling of what a
	// pass costs in fps (separate runs differ by +-3%).
	static const std::string live_list = [] {
		const char* value = Common::ShaderEnv::Read("KYTY_DIAG_SKIP_CS_LIVE");
		if (value != nullptr && value[0] != '\0') std::fprintf(stderr, "DIAG_SKIP_CS_LIVE APPROXIMATION list=%s\n", value);
		return value != nullptr ? std::string(value) : std::string();
	}();
	if (!live_list.empty() && regs.cs_regs.data_addr != 0 &&
	    Common::LiveAb::Gate(Common::LiveAb::Switch::SkipCsLive, 0u) != 0u) {
		const auto [live_data, live_hash] = ShaderGetMappedData(regs.cs_regs.data_addr, "ShaderComputeSkippedAsUnsupported():");
		char text[17];
		std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(live_hash));
		if (live_list.find(text) != std::string::npos) return true;
	}
	if ((!enabled && skip_list.empty()) || regs.cs_regs.data_addr == 0) return false;
	const auto [data, hash] = ShaderGetMappedData(regs.cs_regs.data_addr, "ShaderComputeSkippedAsUnsupported():");
	// Workaround W9 removed: the particle-emit CS 142383a7de1e5111 was skipped since 06.10 (a device loss then,
	// before the wave64 loop fixes; its pair b629e577 runs fine since 23.10). It runs rarely (once in all logs: storm over
	// the village, 21.10). If the GPU hangs or the device is lost right after this line, look here first.
	if (hash == 0x142383a7de1e5111ull) {
		static std::atomic_flag noted = ATOMIC_FLAG_INIT;
		if (!noted.test_and_set(std::memory_order_relaxed))
			std::fprintf(stderr, "W9_PARTICLE_EMIT hash=0x142383a7de1e5111 dispatched (skipped until 29.10; see shader.cpp)\n");
	}
	static const ShaderHashList skip_numeric = ParseShaderHashList(skip_list);
	const auto                  listed       = [&] {
		if (skip_numeric.numeric) return std::binary_search(skip_numeric.values.begin(), skip_numeric.values.end(), hash);
		char text[17];
		std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(hash));
		return skip_list.find(text) != std::string::npos;
	};
	if (!skip_list.empty() && listed()) {
		static std::atomic_uint logged = 0;
		if (logged.fetch_add(1) < 16)
			std::fprintf(stderr, "DIAG_SKIP_CS_HASHES APPROXIMATION hash=0x%016llx dispatch skipped\n",
			             static_cast<unsigned long long>(hash));
		// KYTY_TRACE_PERF: PERF workaround_hits skip_cs (per shader; the log above stops after 16 lines of all shaders).
		Common::WorkaroundHits::NoteShader(Common::WorkaroundHits::Hit::SkipCsListed, Common::WorkaroundHits::Tally::SkipCs,
		                                   hash);
		return true;
	}
	if (!enabled) return false;
	// KYTY_TRACE_PERF: PERF workaround_hits skip_unsupported (dispatches of shaders with the skip verdict).
	const auto counted = [&](bool skip_verdict) {
		if (skip_verdict) {
			Common::WorkaroundHits::NoteShader(Common::WorkaroundHits::Hit::SkipUnsupportedDispatch,
			                                   Common::WorkaroundHits::Tally::SkipUnsupported, hash);
		}
		return skip_verdict;
	};
	// This thread's last verdicts (copies of the shared map's entries).
	struct RecentVerdict {
		uint64_t hash = 0;
		bool     skip = false;
		bool     set  = false;
	};
	static thread_local std::array<RecentVerdict, 64> recent {};
	auto& slot = recent[(hash ^ (hash >> 17u)) % recent.size()];
	if (slot.set && slot.hash == hash) return counted(slot.skip);
	static std::mutex mutex;
	static std::unordered_map<uint64_t, bool> verdicts;
	std::scoped_lock lock(mutex);
	if (const auto it = verdicts.find(hash); it != verdicts.end()) {
		slot = {hash, it->second, true};
		return counted(it->second);
	}
	const auto code = std::span {reinterpret_cast<const uint32_t*>(regs.cs_regs.data_addr),
	                             data.code_size_bytes / sizeof(uint32_t)};
	ShaderRecompiler::Decoder::Program decoded;
	ShaderRecompiler::Decoder::DecodeProgram(code, decoded);
	bool skip = false;
	for (const auto& inst: decoded.instructions) {
		if (inst.opcode == ShaderRecompiler::Decoder::Opcode::UNKNOWN ||
		    inst.opcode == ShaderRecompiler::Decoder::Opcode::UNSUPPORTED) {
			std::fprintf(stderr, "DIAG_SKIP_UNSUPPORTED_CS APPROXIMATION hash=0x%016llx dispatch skipped: %s\n",
			             static_cast<unsigned long long>(hash),
			             ShaderRecompiler::Decoder::InstructionToString(inst).c_str());
			skip = true;
			break;
		}
	}
	verdicts.emplace(hash, skip);
	if (skip) Common::WorkaroundHits::Note(Common::WorkaroundHits::Hit::SkipUnsupportedShader); // KYTY_TRACE_PERF
	return counted(skip);
}

static ShaderParams GetShaderParams(uint64_t shader_addr, uint64_t hash,
	                                std::span<const uint32_t> user_data,
	                                const ShaderMappedData& data, uint32_t user_data_base = 0) {
	const auto code_words = data.code_size_bytes / sizeof(uint32_t);
	const auto code = std::span {reinterpret_cast<const uint32_t*>(shader_addr), code_words};
	ShaderParams params {
	    .code            = code,
	    .user_data_count = static_cast<uint32_t>(user_data.size()) + user_data_base,
	    .hash            = hash,
	};
	EXIT_IF(user_data.size() > HW::UserSgprInfo::SGPRS_MAX ||
	        params.user_data_count > params.user_data.size());
	std::copy(user_data.begin(), user_data.end(), params.user_data.begin() + user_data_base);
	return params;
}

// Based on TheCruZ/KytyPS5-GTA daaf45201 (shader.cpp), GPL-2.0.
static thread_local ShaderGuestReader t_guest_reader          = nullptr;
static thread_local void*             t_guest_reader_userdata = nullptr;
static thread_local bool              t_guest_refused         = false;

ShaderGuestReaderScope::ShaderGuestReaderScope(ShaderGuestReader reader, void* userdata)
    : m_previous_reader(t_guest_reader), m_previous_userdata(t_guest_reader_userdata) {
	t_guest_reader          = reader;
	t_guest_reader_userdata = userdata;
	t_guest_refused         = false;
}

bool ShaderGuestReaderScope::Refused() const noexcept {
	return t_guest_refused;
}

ShaderGuestReaderScope::~ShaderGuestReaderScope() {
	t_guest_reader          = m_previous_reader;
	t_guest_reader_userdata = m_previous_userdata;
}

static void ReadShaderGuest(const void* address, void* data, size_t size) {
	if (t_guest_reader == nullptr) {
		std::memcpy(data, address, size);
	} else if (!t_guest_reader(t_guest_reader_userdata, reinterpret_cast<uint64_t>(address), data, size)) {
		std::memset(data, 0, size);
		t_guest_refused = true;
	}
}

// Guest data this path stops at: not on the resolve thread, which reads ahead of the operation (its result is dropped, the
// execution thread's call decides).
static bool ShaderGuestRefuse() {
	if (t_guest_reader == nullptr) return false;
	t_guest_refused = true;
	return true;
}

static void ShaderDetectBuffers(ShaderVertexInputInfo& info) {
	KYTY_PROFILER_FUNCTION();

	info.buffers_num = 0;

	for (int ri = 0; ri < info.resources_num; ri++) {
		const auto& r = info.resources[ri];

		auto& destination = info.resources_dst[ri];
		int bi = 0;
		for (; bi < info.buffers_num; bi++) {
			auto& b = info.buffers[bi];

			uint64_t stride = b.stride;

			if (stride == r.Stride() &&
			    b.fetch_index == destination.fetch_index) {
				uint64_t rbase   = r.Base48();
				uint64_t base    = std::min(rbase, b.addr);
				uint64_t offset1 = rbase - base;
				uint64_t offset2 = b.addr - base;

				if (offset1 < stride && offset2 < stride) {
					EXIT_NOT_IMPLEMENTED(b.num_records != r.NumRecords() && !ShaderGuestRefuse());
					b.addr = base;
					break;
				}
			}
		}

		if (bi == info.buffers_num) {
			EXIT_NOT_IMPLEMENTED(info.buffers_num >= ShaderVertexInputInfo::RES_MAX);
			info.buffers[bi] = {.addr = r.Base48(),
			                    .stride = r.Stride(),
			                    .num_records = r.NumRecords(),
			                    .fetch_index = destination.fetch_index};
			info.buffers_num++;
		}
		destination.buffer_index = bi;
	}
}

static void ShaderApplyAttribSemantics(ShaderVertexInputInfo& info,
                                       std::span<const ShaderSemantic> input_semantics,
                                       const uint32_t* attrib, const uint32_t* buffer) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(attrib == nullptr || buffer == nullptr);

	const bool debug_dump = Config::GraphicsDebugDumpEnabled();

	for (uint32_t i = 0; i < input_semantics.size(); i++) {
		const auto& in = input_semantics[i];

		EXIT_NOT_IMPLEMENTED(in.static_vb_index == 1 || in.static_attribute == 1);

		uint32_t reg  = in.hardware_mapping;
		uint32_t size = in.size_in_elements;

		uint32_t attribute = 0;
		ReadShaderGuest(&attrib[in.semantic], &attribute, sizeof(attribute));
		if (debug_dump) {
			LOGF("reg = %u, size = %u, va[%u] = 0x%08" PRIx32 "\n", reg, size, i, attribute);
		}

		size_t   index       = attribute & 0x1fu;
		auto     format      = static_cast<Prospero::VertexAttribFormat>((attribute >> 5u) & 0x1ffu);
		uint32_t offset      = (attribute >> 14u) & 0xfffu;
		uint32_t fetch_index = (attribute >> 26u) & 0x1u;

		EXIT_NOT_IMPLEMENTED(index >= ShaderVertexInputInfo::RES_MAX);

		uint32_t sharp[4] {};
		ReadShaderGuest(&buffer[index * 4], sharp, sizeof(sharp));

		EXIT_NOT_IMPLEMENTED(info.resources_num >= ShaderVertexInputInfo::RES_MAX);

		auto& r           = info.resources[info.resources_num];
		auto& rd          = info.resources_dst[info.resources_num];
		rd.register_start = static_cast<int>(reg);
		rd.registers_num  = static_cast<int>(size);
		rd.attr_id        = static_cast<int>(in.semantic);
		rd.fetch_index    = fetch_index;
		r.fields[0]       = sharp[0];
		r.fields[1]       = sharp[1];
		r.fields[2]       = sharp[2];
		r.fields[3]       = sharp[3];
		EXIT_NOT_IMPLEMENTED((r.AddTid() || r.SwizzleEnabled()) && !ShaderGuestRefuse());
		if (format != Prospero::VertexAttribFormat::kInvalid) {
			const auto                   format_raw    = static_cast<uint32_t>(format);
			const auto                   buffer_format = format_raw >> 2u;
			const auto                   channels      = (format_raw & 3u) + 1u;
			// AGC vertex formats encode the buffer format above the two channel-count bits.
			// The fetch prolog selects X001, XY01, XYZ1, or XYZW from that count.
			r.fields[3] = (r.fields[3] & ~((0x7fu << 12u) | 0xfffu)) |
			              (buffer_format << 12u) |
			              DstSel(4, channels > 1u ? 5u : 0u, channels > 2u ? 6u : 0u,
			                     channels > 3u ? 7u : 1u);
		}
		if (offset != 0) {
			r.UpdateAddress48(r.Base48() + offset);
		}

		info.resources_num++;
	}
}

static uint32_t ShaderCalcPsSystemInputBase(const HW::ShaderRegisters& regs) {
	constexpr uint32_t ps_input_persp_sample    = 0x00000001u;
	constexpr uint32_t ps_input_persp_center    = 0x00000002u;
	constexpr uint32_t ps_input_persp_centroid  = 0x00000004u;
	constexpr uint32_t ps_input_persp_pull      = 0x00000008u;
	constexpr uint32_t ps_input_linear_sample   = 0x00000010u;
	constexpr uint32_t ps_input_linear_center   = 0x00000020u;
	constexpr uint32_t ps_input_linear_centroid = 0x00000040u;
	constexpr uint32_t ps_input_line_stipple    = 0x00000080u;
	constexpr uint32_t ps_input_pos_xy          = 0x00000300u;
	constexpr uint32_t ps_input_pos_z           = 0x00000400u;
	constexpr uint32_t ps_input_pos_w           = 0x00000800u;
	constexpr uint32_t ps_input_front_face      = 0x00001000u;
	constexpr uint32_t ps_input_ancillary       = 0x00002000u;
	constexpr uint32_t ps_input_sample_coverage = 0x00004000u;
	constexpr uint32_t ps_input_pos_fixed_pt    = 0x00008000u;
	constexpr uint32_t supported_ps_input_bits =
	    ps_input_persp_sample | ps_input_persp_center | ps_input_persp_centroid |
	    ps_input_persp_pull | ps_input_linear_sample | ps_input_linear_center |
	    ps_input_linear_centroid | ps_input_line_stipple | ps_input_pos_xy | ps_input_pos_z |
	    ps_input_pos_w | ps_input_front_face | ps_input_ancillary | ps_input_sample_coverage |
	    ps_input_pos_fixed_pt;

	EXIT_NOT_IMPLEMENTED((regs.ps_input_ena & ~supported_ps_input_bits) != 0);
	EXIT_NOT_IMPLEMENTED((regs.ps_input_addr & ~supported_ps_input_bits) != 0);
	EXIT_NOT_IMPLEMENTED(regs.ps_input_ena != regs.ps_input_addr);

	const uint32_t inputs = regs.ps_input_addr;
	uint32_t       reg    = 0;
	if ((inputs & ps_input_persp_sample) != 0) {
		reg += 2;
	}
	if ((inputs & ps_input_persp_center) != 0) {
		reg += 2;
	}
	if ((inputs & ps_input_persp_centroid) != 0) {
		reg += 2;
	}
	if ((inputs & ps_input_persp_pull) != 0) {
		reg += 3;
	}
	if ((inputs & ps_input_linear_sample) != 0) {
		reg += 2;
	}
	if ((inputs & ps_input_linear_center) != 0) {
		reg += 2;
	}
	if ((inputs & ps_input_linear_centroid) != 0) {
		reg += 2;
	}
	if ((inputs & ps_input_line_stipple) != 0) {
		reg += 1;
	}
	return reg;
}

static bool ShaderGetStaticVertexInputInfo(uint64_t shader_addr, const HW::UserSgprInfo& user_sgpr,
                                           uint32_t user_sgpr_num, const HW::ShaderRegisters& sh,
                                           const ShaderMappedData& data,
                                           ShaderVertexInputInfo&  info) {
	KYTY_PROFILER_FUNCTION();

	info = {};

	info.pa_cl_vs_out_cntl = sh.m_paClVsOutCntl;

	EXIT_NOT_IMPLEMENTED(shader_addr == 0);
	info.scratch_size_dwords = data.scratch_size_dwords;

	if (data.user_data == nullptr) {
		LOGF("ShaderGetVertexInputInfo(): no AGC user data for shader=0x%016" PRIx64
		     " user_sgpr_num=%u\n",
		     shader_addr, user_sgpr_num);
	}
	ShaderVertexMetadata metadata;
	std::string          metadata_error;
	if (!ShaderReadVertexMetadata(data, HW::UserSgprInfo::SGPRS_MAX, metadata, &metadata_error)) {
		LOGF("ShaderGetVertexInputInfo(): invalid AGC metadata shader=0x%016" PRIx64 ": %s\n",
		     shader_addr, metadata_error.c_str());
		return false;
	}

	if (metadata.vertex_attrib_reg >= 0) {
		info.fetch_external   = false;
		info.fetch_embedded   = true;
		info.fetch_attrib_reg = metadata.vertex_attrib_reg;
		info.fetch_buffer_reg = metadata.vertex_buffer_reg;

		const auto* attrib = reinterpret_cast<const uint32_t*>(
		    static_cast<uint64_t>(user_sgpr.value[metadata.vertex_attrib_reg]) |
		    (static_cast<uint64_t>(user_sgpr.value[metadata.vertex_attrib_reg + 1]) << 32u));
		const auto* buffer = reinterpret_cast<const uint32_t*>(
		    static_cast<uint64_t>(user_sgpr.value[metadata.vertex_buffer_reg]) |
		    (static_cast<uint64_t>(user_sgpr.value[metadata.vertex_buffer_reg + 1]) << 32u));

		if (attrib == nullptr || buffer == nullptr) {
			LOGF("ShaderGetVertexInputInfo(): null vertex table pointer shader=0x%016" PRIx64 "\n",
			     shader_addr);
			return false;
		}
		ShaderApplyAttribSemantics(info, metadata.input_semantics, attrib, buffer);
		ShaderDetectBuffers(info);
	}
	return true;
}

static void ShaderGetStaticInputInfoPS(
    const HW::PixelShaderInfo& regs, const HW::ShaderRegisters& sh,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
	const ShaderMappedData& data, ShaderPixelInputInfo& ps_info) {
	KYTY_PROFILER_FUNCTION();

	ps_info = {};
	ps_info.scratch_size_dwords = data.scratch_size_dwords;
	ps_info.lds_size_dwords     = static_cast<uint32_t>(regs.ps_regs.rsrc2.extra_lds_size) * 128u;

	// SPI_PS_IN_CONTROL: NUM_INTERP occupies bits 5:0 and PS_W32_EN is bit 15.
	ps_info.input_num            = sh.ps_in_control & 0x3fu;
	if ((sh.ps_in_control & 0x8000u) != 0) {
		ps_info.wave_size = 32;
	}
	EXIT_NOT_IMPLEMENTED(ps_info.input_num > std::size(ps_info.interpolator_settings));
	ps_info.ps_system_input_base = ShaderCalcPsSystemInputBase(sh);
	const uint32_t active_inputs = sh.ps_input_ena & sh.ps_input_addr;
	if ((active_inputs & 0x00000001u) != 0) {
		ps_info.ps_perspective_sample_vgpr = 0;
	}
	if ((active_inputs & 0x00000002u) != 0) {
		ps_info.ps_perspective_center_vgpr = (active_inputs & 0x00000001u) != 0 ? 2u : 0u;
	}
	if ((active_inputs & 0x00000004u) != 0) {
		ps_info.ps_perspective_centroid_vgpr = 2u * std::popcount(active_inputs & 0x3u);
	}
	for (uint32_t i = 0; i < data.num_input_semantics && i < ps_info.input_num && i < 32u; i++) {
		const auto& semantic = data.input_semantics[i];
		if (semantic.is_custom != 0 && semantic.is_f16 == 0) {
			ps_info.custom_interpolation_mask |= 1u << i;
		}
	}
	ps_info.ps_pos_x                     = (active_inputs & 0x00000100u) != 0;
	ps_info.ps_pos_y                     = (active_inputs & 0x00000200u) != 0;
	ps_info.ps_pos_z                     = (active_inputs & 0x00000400u) != 0;
	ps_info.ps_pos_w                     = (active_inputs & 0x00000800u) != 0;
	ps_info.ps_front_face                = (active_inputs & 0x00001000u) != 0;
	// SPI_BARYC_CNTL.FRONT_FACE_ALL_BITS selects integer masks instead of float +/-1.
	ps_info.ps_front_face_all_bits       = (sh.baryc_cntl & 0x01000000u) != 0;
	ps_info.ps_ancillary                 = (active_inputs & 0x00002000u) != 0;
	ps_info.ps_sample_shading            = (active_inputs & 0x00000011u) != 0;
	ps_info.ps_no_perspective            = (sh.ps_input_ena & sh.ps_input_addr & 0x00000020u) != 0;
	ps_info.ps_pixel_kill_enable         = sh.db_shader_control.shader_kill_enable;
	ps_info.ps_depth_export_enable       = sh.db_shader_control.shader_z_export_enable;
	ps_info.ps_sample_mask_export_enable = sh.db_shader_control.shader_mask_export_enable;
	// Alpha-to-mask coverage must gate depth writes, so it rules out early fragment tests.
	ps_info.ps_early_z =
	    (sh.db_shader_control.shader_z_behavior == 1 && !sh.db_shader_control.shader_kill_enable &&
	     !sh.db_shader_control.shader_z_export_enable &&
	     !sh.db_shader_control.shader_mask_export_enable && !sh.AlphaToMask());
	ps_info.ps_execute_on_noop = sh.db_shader_control.shader_execute_on_noop;

	for (uint32_t i = 0; i < ps_info.input_num; i++) {
		ps_info.interpolator_settings[i] = sh.ps_interpolator_settings[i];
	}

	ps_info.target_shader_mask = sh.m_cbShaderMask;
	for (int i = 0; i < 8; i++) {
		const auto slot = ShaderPixelExportTarget(ps_info.target_shader_mask, i);
		ps_info.target_output_mode[i]    = sh.target_output_mode[i];
		ps_info.target_export_mapping[i] = sh.target_output_mode[i] != 0 && slot < 8
		                                       ? target_export_mapping[slot]
		                                       : Prospero::ColorComponentMapping {};
	}
}

static void ShaderGetStaticInputInfoCS(const HW::ComputeShaderInfo& regs,
                                       const HW::ShaderRegisters& /*sh*/,
                                       const ShaderMappedData& data, ShaderComputeInputInfo& info) {
	const bool async_compute             = info.async_compute;
	const bool dispatch_thread_dimensions = info.dispatch_thread_dimensions;
	const auto host_subgroup_size         = info.host_subgroup_size;
	const auto workgroup_counts           = std::to_array(info.workgroup_counts);
	const auto indirect_args_address      = info.indirect_args_address;
	info                                  = {};
	info.async_compute                    = async_compute;
	info.dispatch_thread_dimensions       = dispatch_thread_dimensions;
	info.host_subgroup_size               = host_subgroup_size;
	std::ranges::copy(workgroup_counts, info.workgroup_counts);
	info.indirect_args_address            = indirect_args_address;
	info.threads_num[0]                   = regs.cs_regs.num_thread_x;
	info.threads_num[1]                   = regs.cs_regs.num_thread_y;
	info.threads_num[2]                   = regs.cs_regs.num_thread_z;
	info.lds_size_dwords                  = static_cast<uint32_t>(regs.cs_regs.lds_size) * 128u;
	info.scratch_size_dwords              = data.scratch_size_dwords;
	info.float_mode                       = regs.cs_regs.float_mode;
	info.dx10_clamp                       = regs.cs_regs.dx10_clamp;
	info.group_id[0]                      = regs.cs_regs.tgid_x_en != 0;
	info.group_id[1]                      = regs.cs_regs.tgid_y_en != 0;
	info.group_id[2]                      = regs.cs_regs.tgid_z_en != 0;
	info.wave_size                        = regs.cs_regs.wave_size;
	info.thread_ids_num      = regs.cs_regs.tidig_comp_cnt + 1;
	info.tg_size_en          = regs.cs_regs.tg_size_en != 0;

	info.workgroup_register = regs.cs_regs.user_sgpr;
}

void BuildStageStaticKey(const ShaderVertexInputInfo& info, std::vector<uint32_t>& key) {
	EXIT_IF(info.resources_num < 0 || info.resources_num > ShaderVertexInputInfo::RES_MAX);
	key.clear();
	// Bit 1: draw_base_once (draw base once) changes the vertex entry; off keeps the key as it was.
	key.push_back(static_cast<uint32_t>(info.fetch_embedded) | (info.draw_base_once ? 2u : 0u));
	key.push_back(static_cast<uint32_t>(info.fetch_attrib_reg));
	key.push_back(static_cast<uint32_t>(info.fetch_buffer_reg));
	key.push_back(info.indirect_draw.Key());
	key.push_back(info.resources_num);
	key.push_back(info.wave_size);
	key.push_back(info.scratch_size_dwords);
	key.push_back(info.lds_size_dwords);
	key.push_back(info.pa_cl_vs_out_cntl);
	key.push_back(info.user_clip_planes);
	key.push_back(static_cast<uint32_t>(info.clip_space.enabled));
	if (info.clip_space.enabled) {
		for (const float value: info.clip_space.scale) {
			key.push_back(std::bit_cast<uint32_t>(value));
		}
		for (const float value: info.clip_space.offset) {
			key.push_back(std::bit_cast<uint32_t>(value));
		}
		for (const float value: info.clip_space.half_extent) {
			key.push_back(std::bit_cast<uint32_t>(value));
		}
	}

	key.push_back(info.mesh.threads_num[0]);
	if (info.mesh.threads_num[0] != 0) {
		const auto& mesh = info.mesh;
		key.insert(key.end(), {mesh.wave_size, mesh.host_subgroup_size, mesh.lds_size_dwords,
		                       mesh.scratch_size_dwords, mesh.input_primitive,
		                       mesh.primitives_per_group, mesh.vertices_per_group,
		                       mesh.max_vertices, mesh.max_primitives, mesh.provoking_vertex,
		                       static_cast<uint32_t>(mesh.fast_launch)});
	}
	key.push_back(info.tess.input_control_points);
	if (info.tess.input_control_points != 0) {
		const auto& tess = info.tess;
		key.insert(key.end(), {tess.output_control_points, tess.ls_stride, tess.hs_stride,
		                       tess.patch_base, tess.patch_size, tess.domain, tess.partitioning,
		                       tess.output_topology});
		key.insert(key.end(), tess.carried_vgprs.begin(), tess.carried_vgprs.end());
	}

	for (int i = 0; i < info.resources_num; i++) {
		const auto& resource    = info.resources[i];
		const auto& destination = info.resources_dst[i];
		key.push_back(destination.registers_num);
		key.push_back(static_cast<uint32_t>(destination.attr_id));
		key.push_back(resource.fields[3] & 0x7ffffu); // Channels and format used by embedded fetch.
	}
}

void BuildStageStaticKey(const ShaderPixelInputInfo& info, std::vector<uint32_t>& key) {
	EXIT_IF(info.input_num > std::size(info.interpolator_settings));
	key.clear();
	key.push_back(info.scratch_size_dwords);
	key.push_back(info.lds_size_dwords);
	key.push_back(info.input_num);
	key.push_back(info.wave_size);
	key.push_back(info.ps_system_input_base);
	key.push_back(info.custom_interpolation_mask);
	key.push_back(info.ps_perspective_center_vgpr);
	key.push_back(info.ps_perspective_sample_vgpr);
	key.push_back(info.ps_perspective_centroid_vgpr);
	key.push_back(static_cast<uint32_t>(info.ps_pos_x));
	key.push_back(static_cast<uint32_t>(info.ps_pos_y));
	key.push_back(static_cast<uint32_t>(info.ps_pos_z));
	key.push_back(static_cast<uint32_t>(info.ps_pos_w));
	key.push_back(static_cast<uint32_t>(info.ps_front_face));
	key.push_back(static_cast<uint32_t>(info.ps_front_face_all_bits));
	key.push_back(static_cast<uint32_t>(info.ps_ancillary));
	key.push_back(static_cast<uint32_t>(info.ps_no_perspective));
	key.push_back(static_cast<uint32_t>(info.parameter_mode));
	key.push_back(static_cast<uint32_t>(info.ps_pixel_kill_enable));
	key.push_back(static_cast<uint32_t>(info.ps_depth_export_enable));
	key.push_back(static_cast<uint32_t>(info.ps_sample_mask_export_enable));
	key.push_back(static_cast<uint32_t>(info.ps_early_z));
	key.push_back(static_cast<uint32_t>(info.dual_source_blending));
	key.push_back(static_cast<uint32_t>(info.object_primitive_id));
	key.push_back(static_cast<uint32_t>(info.alpha_blend_source));
	key.insert(key.end(), std::begin(info.target_output_mode), std::end(info.target_output_mode));
	key.push_back(info.target_packed_unorm_mask | (uint32_t {info.target_round_mode} << 8u));
	const auto mask = info.target_shader_mask;
	key.push_back((mask | (mask >> 1u) | (mask >> 2u) | (mask >> 3u)) & 0x11111111u);
	for (uint32_t base = 0; base < info.target_export_mapping.size(); base += 4u) {
		uint32_t packed = 0;
		for (uint32_t i = 0; i < 4u; i++) {
			packed |= static_cast<uint32_t>(info.target_export_mapping[base + i].packed)
			          << (i * 8u);
		}
		key.push_back(packed);
	}
	key.insert(key.end(), std::begin(info.interpolator_settings),
	           std::begin(info.interpolator_settings) + info.input_num);
	// Appended only with an ObjPrimID output, so every other key stays as it was (input_num fixes the length above).
	if (info.HasObjprimOutput()) {
		key.push_back(0x0b1d0000u | info.objprim_slot | (static_cast<uint32_t>(info.objprim_push_dword) << 4u) |
		              (static_cast<uint32_t>(info.objprim_add_prim_id) << 12u));
	}
}

void BuildStageStaticKey(const ShaderComputeInputInfo& info, std::vector<uint32_t>& key) {
	key.clear();
	key.push_back(info.workgroup_register);
	key.push_back(info.wave_size | (static_cast<uint32_t>(info.float_mode) << 8u));
	key.push_back(info.host_subgroup_size);
	key.push_back(info.thread_ids_num);
	key.push_back(info.lds_size_dwords);
	key.push_back(info.scratch_size_dwords);
	key.push_back(static_cast<uint32_t>(info.dispatch_thread_dimensions) |
	              (static_cast<uint32_t>(info.async_compute) << 1u));
	for (int i = 0; i < 3; i++) {
		key.push_back(info.threads_num[i]);
		key.push_back(static_cast<uint32_t>(info.group_id[i]));
	}
	key.push_back(static_cast<uint32_t>(info.tg_size_en));
	// ComputeVariantPairOff (KYTY_DIAG_TWIN_DISPATCH): appended only when set (0x9a1f tells it from the above).
	if ((info.compute_variant & ComputeVariantPairOff) != 0) key.push_back(0x9a1f0000u);
	// ComputeVariantPairSpin (KYTY_DIAG_TWIN_DISPATCH): appended only when set.
	if ((info.compute_variant & ComputeVariantPairSpin) != 0) key.push_back(0x9a1f0001u);
	// KYTY_FIX_CLAMP_NAN: the saturate translation depends on DX10_CLAMP; appended only when the switch is on and the
	// bit is 0 (0xd10c tells it from the above).
	if (ShaderRecompiler::IR::ClampNanMode() != 0 && !info.dx10_clamp) key.push_back(0xd10c0000u);
}

ShaderParams PrepareProgram(const HW::VertexShaderInfo& regs, const HW::Context& context,
                            const HW::UserConfig& user_config, ShaderVertexInputInfo& info) {
	const auto& sh     = context.GetShaderRegisters();
	const auto [data, hash] = ShaderGetMappedData(regs.es_regs.data_addr, "ShaderGetInputInfoVS():");
	const bool merged = (context.GetShaderStages() & 0x20u) != 0;
	// GS_EN controls amplification, not whether an NGG shader uses a workgroup.
	const bool native_ngg = !merged && data.type == Prospero::ShaderBinaryType::kGs &&
	                        regs.gs_regs.rsrc2.lds_size != 0;
	auto        params = GetShaderParams(
	    regs.es_regs.data_addr, hash,
	    std::span<const uint32_t>(regs.gs_user_sgpr.value, regs.gs_regs.rsrc2.user_sgpr), data,
	    merged || native_ngg ? 8u : 0u);
	if (!merged && !native_ngg) {
		KYTY_PROFILER_BLOCK("ShaderGetStaticVertexInputInfo");
		if (!ShaderGetStaticVertexInputInfo(regs.es_regs.data_addr, regs.gs_user_sgpr,
		                                    regs.gs_regs.rsrc2.user_sgpr, sh, data, info)) {
			if (ShaderGuestRefuse()) return params;
			EXIT("failed to prepare vertex shader program\n");
		}
		info.wave_size = (context.GetShaderStages() & 0x00400000u) != 0 ? 32u : 64u;
		// The plain vertex stage only: the mesh path below and LS keep their entry state.
		info.draw_base_once = true;
		info.lds_size_dwords = static_cast<uint32_t>(regs.gs_regs.rsrc2.lds_size) * 128u;
		return params;
	}
	// NGG user SGPRs start at s8; a separately compiled GS back half also receives
	// its user-data pointer in s0:s1.
	info                     = {};
	info.logical_stage       = ShaderType::Mesh;
	info.pa_cl_vs_out_cntl   = sh.m_paClVsOutCntl;
	auto& mesh               = info.mesh;
	mesh.input_primitive     = static_cast<uint32_t>(user_config.GetPrimType());
	mesh.wave_size           = (context.GetShaderStages() & 0x00400000u) != 0 ? 32u : 64u;
	mesh.max_vertices        = sh.m_geMaxOutputPerSubgroup;
	mesh.provoking_vertex    = context.GetModeControl().provoking_vtx_last ? 2u : 0u;
	mesh.lds_size_dwords     = static_cast<uint32_t>(regs.gs_regs.rsrc2.lds_size) * 128u;
	mesh.scratch_size_dwords = data.scratch_size_dwords;
	const auto fast_launch = (context.GetShaderStages() >> 19u) & 3u;
	EXIT_NOT_IMPLEMENTED(fast_launch > 1u);
	mesh.fast_launch = fast_launch != 0;
	if (data.type == Prospero::ShaderBinaryType::kGsFront) {
		EXIT_IF(regs.gs_regs.data_addr == 0);
		const auto [back, back_hash] = ShaderGetMappedData(regs.gs_regs.data_addr, "ShaderGetInputInfoGS():");
		const auto back_params =
		    GetShaderParams(regs.gs_regs.data_addr, back_hash, {}, back);
		params.back_code = back_params.code;
		params.user_data[0] = static_cast<uint32_t>(regs.gs_regs.user_data_addr);
		params.user_data[1] = static_cast<uint32_t>(regs.gs_regs.user_data_addr >> 32u);
		const uint64_t hashes[] = {params.hash, back_params.hash};
		params.hash = XXH3_64bits(hashes, sizeof(hashes));
		mesh.scratch_size_dwords = std::max(mesh.scratch_size_dwords, back.scratch_size_dwords);
	}
	const auto& group = user_config.GetGeControl();
	if (mesh.fast_launch) {
		const auto& user_vgpr = user_config.GetGeUserVgprEn();
		EXIT_NOT_IMPLEMENTED(data.type != Prospero::ShaderBinaryType::kGs ||
		                     regs.gs_regs.rsrc1.gs_vgpr_component_count != 0u ||
		                     regs.gs_regs.rsrc2.es_vgpr_component_count != 1u ||
		                     user_config.GetPrimType() != Prospero::PrimitiveType::kPointList ||
		                     group.primitive_group_size != 1u || group.vertex_group_size != 1u ||
		                     user_vgpr.vgpr1 || user_vgpr.vgpr2 || user_vgpr.vgpr3 ||
		                     mesh.max_vertices != sh.m_vgtGsMaxVertOut);
	} else if (merged) {
		EXIT_NOT_IMPLEMENTED(regs.gs_regs.rsrc1.gs_vgpr_component_count != 3u ||
		                     regs.gs_regs.rsrc2.es_vgpr_component_count != 3u);
	}
	if ((user_config.GetPrimType() != Prospero::PrimitiveType::kPointList &&
	     user_config.GetPrimType() != Prospero::PrimitiveType::kLineList &&
	     user_config.GetPrimType() != Prospero::PrimitiveType::kTriFan &&
	     user_config.GetPrimType() != Prospero::PrimitiveType::kTriStrip &&
	     user_config.GetPrimType() != Prospero::PrimitiveType::kTriList) ||
	    sh.m_vgtGsOutPrimType != 2u || (!native_ngg && sh.m_vgtGsMaxVertOut < 3u) ||
	    group.vertex_group_size < mesh.InputPrimitiveSize() ||
	    mesh.max_vertices == 0u) {
		EXIT("unsupported GS assembly: input=%u output=%u vertices=%u GE=%u/%u max_output=%u\n",
		     mesh.input_primitive, sh.m_vgtGsOutPrimType, sh.m_vgtGsMaxVertOut,
		     group.primitive_group_size, group.vertex_group_size, mesh.max_vertices);
	}
	mesh.primitives_per_group = std::min({static_cast<uint32_t>(group.primitive_group_size),
	                                      mesh.InputPrimitiveCount(group.vertex_group_size),
	                                      native_ngg ? mesh.InputPrimitiveCount(mesh.max_vertices) :
	                                          mesh.max_vertices / sh.m_vgtGsMaxVertOut});
	EXIT_IF(mesh.primitives_per_group == 0u);
	mesh.vertices_per_group = mesh.InputVertexCount(mesh.primitives_per_group);
	mesh.max_primitives = mesh.primitives_per_group * (sh.m_geNggSubgrpCntl & 0x1ffu);
	const auto threads = std::max(native_ngg && !mesh.fast_launch ? mesh.vertices_per_group
	                                                            : mesh.max_vertices,
	                              mesh.max_primitives);
	mesh.threads_num[0] = ((threads + mesh.wave_size - 1u) / mesh.wave_size) * mesh.wave_size;
	mesh.threads_num[1] = mesh.threads_num[2] = 1u;
	return params;
}

// Ghost of Yotei 2.000 binds one merged LS+HS program (Agc type kHs) at the LS address and leaves the HS address
// unset, where 1.x bound the two halves sceAgcFuseShaderHalves joined (2.000 no longer imports it). The game's merged
// program is the two halves one after the other: a prologue (EXEC = all, v0 = lane index by V_MBCNT, V_CMPX lane <
// LS vertex count in s3[7:0], S_CBRANCH_EXECZ to the HS part), the LS part, then the HS part from the branch target on
// (EXEC = all, S_BARRIER, V_CMPX lane < HS thread count in s3[15:8]). It is split there into what the halves path takes:
// the front is the prologue and the LS part ending in the merged-stage handoff (S_SETPC_B64 s[6:7], where the Local
// stage ends and which AnalyzeTessellationPrograms expects); the back is the prologue's two V_MBCNT (its v0 lane index,
// as a separately compiled HS back half computes it itself) and the HS part. A program of another shape stops here.
struct MergedHullHalves {
	std::vector<uint32_t> front;
	std::vector<uint32_t> back;
	uint64_t              front_hash = 0;
	uint64_t              back_hash  = 0;
};

static const MergedHullHalves& SplitMergedHullProgram(uint64_t addr, const ShaderMappedData& data, uint64_t hash) {
	namespace Decoder = ShaderRecompiler::Decoder;
	using Decoder::Opcode;
	static std::mutex                                     mutex;
	static std::unordered_map<uint64_t, MergedHullHalves> splits; // node-based: the halves' spans stay valid
	std::scoped_lock lock(mutex);
	if (const auto found = splits.find(hash); found != splits.end()) return found->second;

	const auto code = std::span {reinterpret_cast<const uint32_t*>(addr), data.code_size_bytes / sizeof(uint32_t)};
	Decoder::Program program;
	Decoder::DecodeProgram(code, program);
	const auto& insts = program.instructions;
	const auto  fail  = [&](const char* reason) {
		EXIT("merged HS program 0x%016" PRIx64 " (hash 0x%016" PRIx64 ") cannot be split: %s\n", addr, hash, reason);
	};
	size_t i = 0;
	if (i < insts.size() && insts[i].opcode == Opcode::S_INST_PREFETCH) i++;
	const auto is_exec_all = [](const Decoder::Instruction& inst) {
		return inst.opcode == Opcode::S_MOV_B64 && inst.raw[0] == 0xbefe04c1u; // s_mov_b64 exec, -1
	};
	const auto is_lane_index = [](const Decoder::Instruction& inst, Opcode opcode) {
		return inst.opcode == opcode && inst.dst.kind == Decoder::OperandKind::Vgpr && inst.dst.reg == 0u;
	};
	if (i + 5u > insts.size() || !is_exec_all(insts[i]) || !is_lane_index(insts[i + 1u], Opcode::V_MBCNT_LO_U32_B32) ||
	    !is_lane_index(insts[i + 2u], Opcode::V_MBCNT_HI_U32_B32) || insts[i + 3u].opcode != Opcode::V_CMPX_LT_U32 ||
	    insts[i + 4u].opcode != Opcode::S_CBRANCH_EXECZ) {
		fail("no LS prologue (EXEC = all, V_MBCNT lane index, V_CMPX, S_CBRANCH_EXECZ)");
	}
	const auto& mbcnt_lo = insts[i + 1u];
	const auto& mbcnt_hi = insts[i + 2u];
	const auto  split_pc = insts[i + 4u].branch_target;
	const auto  at_split = std::find_if(insts.begin(), insts.end(), [&](const auto& inst) { return inst.pc == split_pc; });
	if (at_split == insts.end() || !is_exec_all(*at_split)) fail("the HS part does not begin with EXEC = all");
	for (const auto& inst: insts) {
		if (inst.opcode == Opcode::S_GETPC_B64 || inst.opcode == Opcode::S_SETPC_B64 ||
		    inst.opcode == Opcode::S_SWAPPC_B64 || (inst.opcode == Opcode::S_ENDPGM && inst.pc < split_pc)) {
			fail("PC-relative code or an LS-part exit");
		}
		if (Decoder::IsDirectBranch(inst.opcode) && (inst.pc < split_pc) != (inst.branch_target <= split_pc)) {
			fail("a branch crosses between the LS and HS parts");
		}
	}

	MergedHullHalves halves;
	const auto split_word = split_pc / sizeof(uint32_t);
	halves.front.assign(code.begin(), code.begin() + split_word);
	halves.front.push_back(0xbe802006u); // s_setpc_b64 s[6:7]: the merged-stage handoff of a front half
	for (const auto* inst: {&mbcnt_lo, &mbcnt_hi}) {
		const auto first = code.begin() + inst->pc / sizeof(uint32_t);
		halves.back.insert(halves.back.end(), first, first + inst->word_count);
	}
	halves.back.insert(halves.back.end(), code.begin() + split_word, code.end());
	halves.front_hash = XXH3_64bits(halves.front.data(), halves.front.size() * sizeof(uint32_t));
	halves.back_hash  = XXH3_64bits(halves.back.data(), halves.back.size() * sizeof(uint32_t));
	LOGF("merged HS program 0x%016" PRIx64 " (hash 0x%016" PRIx64 ") split at 0x%x: LS 0x%016" PRIx64 " HS 0x%016" PRIx64
	     "\n", addr, hash, split_pc, halves.front_hash, halves.back_hash);
	return splits.emplace(hash, std::move(halves)).first->second;
}

std::array<ShaderParams, 3>
PrepareTessellationPrograms(const HW::VertexShaderInfo& regs, const HW::Context& context,
                            std::array<ShaderVertexInputInfo, 3>& input_info) {
	const auto& sh        = context.GetShaderRegisters();
	const auto [local, local_hash] = ShaderGetMappedData(regs.ls_regs.data_addr, "ShaderGetInputInfoLS():");
	// One merged program (Agc type kHs) at the LS address and no HS address: split into the two halves (above).
	const MergedHullHalves* merged = regs.hs_regs.data_addr == 0 && local.type == Prospero::ShaderBinaryType::kHs
	                                     ? &SplitMergedHullProgram(regs.ls_regs.data_addr, local, local_hash)
	                                     : nullptr;
	ShaderMapEntry control_entry;
	if (merged != nullptr) {
		control_entry           = {local, merged->back_hash};
		control_entry.data.type = Prospero::ShaderBinaryType::kHsBack;
	} else {
		control_entry = ShaderGetMappedData(regs.hs_regs.data_addr, "ShaderGetInputInfoHS():");
	}
	const auto& [control, control_hash] = control_entry;
	const auto [evaluation, evaluation_hash] = ShaderGetMappedData(regs.es_regs.data_addr, "ShaderGetInputInfoTES():");
	EXIT_NOT_IMPLEMENTED((merged == nullptr && local.type != Prospero::ShaderBinaryType::kHsFront) ||
	                     control.type != Prospero::ShaderBinaryType::kHsBack ||
	                     evaluation.type != Prospero::ShaderBinaryType::kGs);
	EXIT_NOT_IMPLEMENTED((context.GetShaderStages() & 0x4u) == 0 ||
	                     (context.GetShaderStages() & 0x00600020u) != 0);
	EXIT_IF(merged == nullptr && regs.hs_regs.user_data_addr == 0);

	const auto local_users      = std::span(regs.hs_user_sgpr.value, regs.hs_regs.rsrc2.user_sgpr);
	const auto evaluation_users = std::span(regs.gs_user_sgpr.value, regs.gs_regs.rsrc2.user_sgpr);
	std::array<ShaderParams, 3> params {
	    GetShaderParams(regs.ls_regs.data_addr, local_hash, local_users, local),
	    GetShaderParams(regs.ls_regs.data_addr, control_hash, local_users, local, 8u),
	    GetShaderParams(regs.es_regs.data_addr, evaluation_hash, evaluation_users,
	                    evaluation),
	};
	if (merged != nullptr) {
		params[0].code = merged->front;
		params[0].hash = merged->front_hash;
		params[1].code = merged->back;
	} else {
		params[1] = GetShaderParams(regs.hs_regs.data_addr, control_hash, local_users, control, 8u);
		// The fused HS back half receives its separate user-data address in s0:s1.
		// RDNA2 reserves s0:s7 before the native HS user SGPRs.
		params[1].user_data[0] = static_cast<uint32_t>(regs.hs_regs.user_data_addr);
		params[1].user_data[1] = static_cast<uint32_t>(regs.hs_regs.user_data_addr >> 32u);
	}

	input_info = {};
	if (!ShaderGetStaticVertexInputInfo(regs.ls_regs.data_addr, regs.hs_user_sgpr,
	                                    regs.hs_regs.rsrc2.user_sgpr, sh, local, input_info[0])) {
		if (ShaderGuestRefuse()) return {};
		EXIT("failed to prepare local shader program\n");
	}
	input_info[0].logical_stage       = ShaderType::Local;
	input_info[1].logical_stage       = ShaderType::TessellationControl;
	input_info[1].scratch_size_dwords = control.scratch_size_dwords;
	input_info[2].logical_stage       = ShaderType::TessellationEvaluation;
	input_info[2].scratch_size_dwords = evaluation.scratch_size_dwords;
	input_info[2].pa_cl_vs_out_cntl   = sh.m_paClVsOutCntl;

	ShaderTessellationInputInfo tess {
	    .input_control_points  = (sh.m_vgtLsHsConfig >> 8u) & 0x3fu,
	    .output_control_points = (sh.m_vgtLsHsConfig >> 14u) & 0x3fu,
	    .domain                = sh.m_vgtTfParam & 0x3u,
	    .partitioning          = (sh.m_vgtTfParam >> 2u) & 0x3u,
	    .output_topology       = (sh.m_vgtTfParam >> 5u) & 0x3u,
	};
	EXIT_IF(tess.input_control_points == 0 || tess.input_control_points > 32 ||
	        tess.output_control_points == 0 || tess.output_control_points > 32);
	{
		// The tessellation layout depends only on the two programs' code (named by their registered code hashes) and the
		// VGT LS/HS and TF configuration, so each combination is analysed once (g_tess_layouts, upstream c71589124) instead
		// of decoding both programs on every tessellated draw. The key also covers the VGT fields cached in tess.
		const uint64_t key[] = {params[0].hash, control_hash,
		                        uint64_t {sh.m_vgtLsHsConfig & 0xfff00u} << 32u | sh.m_vgtTfParam};
		std::scoped_lock lock(g_tess_layouts_mutex);
		auto [entry, inserted] = g_tess_layouts.try_emplace(XXH3_64bits(key, sizeof(key)), tess);
		if (inserted) {
			ShaderRecompiler::AnalyzeTessellationPrograms(params[0].code, params[1].code,
			                                              entry->second, merged != nullptr);
		}
		tess = entry->second;
	}
	for (auto& stage: input_info) {
		stage.tess = tess;
	}
	return params;
}

ShaderParams PrepareProgram(
    const HW::PixelShaderInfo& regs, const HW::ShaderRegisters& sh,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
    ShaderPixelInputInfo&                               ps_info) {
	const auto [data, hash] = ShaderGetMappedData(regs.ps_regs.data_addr, "ShaderGetInputInfoPS():");
	ShaderGetStaticInputInfoPS(regs, sh, target_export_mapping, data, ps_info);
	return GetShaderParams(
	    regs.ps_regs.data_addr, hash,
	    std::span<const uint32_t>(regs.ps_user_sgpr.value, regs.ps_regs.rsrc2.user_sgpr), data);
}

ShaderParams PrepareProgram(const HW::ComputeShaderInfo& regs, const HW::ShaderRegisters& sh,
                            ShaderComputeInputInfo& info) {
	const auto [data, hash] = ShaderGetMappedData(regs.cs_regs.data_addr, "ShaderGetInputInfoCS():");
	ShaderGetStaticInputInfoCS(regs, sh, data, info);
	return GetShaderParams(
	    regs.cs_regs.data_addr, hash,
	    std::span<const uint32_t>(regs.cs_user_sgpr.value, regs.cs_regs.user_sgpr), data);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void ShaderDbgDumpInputInfo(const ShaderVertexInputInfo& info) {
	KYTY_PROFILER_BLOCK("ShaderDbgDumpInputInfo(Vs)");

	LOGF("ShaderDbgDumpInputInfo()\n");

	LOGF("\t fetch_external = %s\n"
	     "\t fetch_embedded = %s\n",
	     info.fetch_external ? "true" : "false", info.fetch_embedded ? "true" : "false");

	for (int i = 0; i < info.resources_num; i++) {
		LOGF("\t input %d\n", i);

		const auto& r  = info.resources[i];
		const auto& rd = info.resources_dst[i];

		LOGF("\t\t register_start   = %d\n"
		     "\t\t registers_num    = %d\n"
		     "\t\t fetch_index      = %" PRIu32 "\n"
		     "\t\t buffer_index     = %d\n",
		     rd.register_start, rd.registers_num, rd.fetch_index, rd.buffer_index);
		LOGF("\t\t fields           = %08" PRIx32 "%08" PRIx32 "%08" PRIx32 "%08" PRIx32 "\n",
		     r.fields[3], r.fields[2], r.fields[1], r.fields[0]);
		LOGF("\t\t Base()           = %" PRIx64 "\n"
		     "\t\t Stride()         = %" PRIu16 "\n"
		     "\t\t SwizzleEnabled() = %s\n"
		     "\t\t NumRecords()     = %" PRIu32 "\n"
		     "\t\t DstSelX()        = %" PRIu8 "\n"
		     "\t\t DstSelY()        = %" PRIu8 "\n"
		     "\t\t DstSelZ()        = %" PRIu8 "\n"
		     "\t\t DstSelW()        = %" PRIu8 "\n",
		     r.Base48(), r.Stride(), r.SwizzleEnabled() ? "true" : "false", r.NumRecords(),
		     r.DstSelX(), r.DstSelY(), r.DstSelZ(), r.DstSelW());
		LOGF("\t\t Format()         = %" PRIu8 "\n"
		     "\t\t OutOfBounds()    = %" PRIu8 "\n",
		     r.RawFormat(), r.OutOfBounds());
		LOGF("\t\t AddTid()         = %s\n", r.AddTid() ? "true" : "false");
	}

	for (int i = 0; i < info.buffers_num; i++) {
		LOGF("\t buffer %d\n", i);

		const auto& r = info.buffers[i];
		LOGF("\t\t addr        = %" PRIx64 "\n"
		     "\t\t stride      = %" PRIu32 "\n"
		     "\t\t num_records = %" PRIu32 "\n"
		     "\t\t fetch_index = %" PRIu32 "\n",
		     r.addr, r.stride, r.num_records, r.fetch_index);
	}
}

void ShaderDbgDumpInputInfo(const ShaderPixelInputInfo& info) {
	KYTY_PROFILER_BLOCK("ShaderDbgDumpInputInfo(Ps)");

	LOGF("ShaderDbgDumpInputInfo()\n");

	LOGF("\t input_num            = %u\n"
	     "\t ps_system_input_base = %u\n"
	     "\t custom_interpolation_mask = 0x%08" PRIx32 "\n"
	     "\t ps_perspective_center_vgpr = %" PRIu32 "\n"
	     "\t ps_perspective_sample_vgpr = %" PRIu32 "\n"
	     "\t ps_perspective_centroid_vgpr = %" PRIu32 "\n"
	     "\t ps_pos_x             = %s\n"
	     "\t ps_pos_y             = %s\n"
	     "\t ps_pos_z             = %s\n"
	     "\t ps_pos_w             = %s\n"
	     "\t ps_front_face        = %s\n"
	     "\t ps_front_face_all_bits = %s\n"
	     "\t ps_ancillary         = %s\n"
	     "\t ps_sample_shading    = %s\n"
	     "\t ps_no_perspective    = %s\n"
	     "\t ps_pixel_kill_enable = %s\n"
	     "\t ps_early_z           = %s\n"
	     "\t ps_execute_on_noop   = %s\n"
	     "\t lds_size_dwords      = %u\n",
	     info.input_num, info.ps_system_input_base, info.custom_interpolation_mask,
	     info.ps_perspective_center_vgpr, info.ps_perspective_sample_vgpr,
	     info.ps_perspective_centroid_vgpr, info.ps_pos_x ? "true" : "false",
	     info.ps_pos_y ? "true" : "false", info.ps_pos_z ? "true" : "false",
	     info.ps_pos_w ? "true" : "false", info.ps_front_face ? "true" : "false",
	     info.ps_front_face_all_bits ? "true" : "false", info.ps_ancillary ? "true" : "false",
	     info.ps_sample_shading ? "true" : "false", info.ps_no_perspective ? "true" : "false",
	     info.ps_pixel_kill_enable ? "true" : "false", info.ps_early_z ? "true" : "false",
	     info.ps_execute_on_noop ? "true" : "false", info.lds_size_dwords);

	for (uint32_t i = 0; i < info.input_num; i++) {
		LOGF("\t interpolator_settings[%u] = %u\n", i, info.interpolator_settings[i]);
	}
}

void ShaderDbgDumpInputInfo(const ShaderComputeInputInfo& info) {
	LOGF("ShaderDbgDumpInputInfo()\n");

	LOGF("\t workgroup_register = %d\n"
	     "\t thread_ids_num     = %d\n"
	     "\t wave_size          = %u\n"
	     "\t lds_size_dwords    = %u\n"
	     "\t threads_num        = {%u, %u, %u}\n"
	     "\t tg_size_en         = %s\n",
	     info.workgroup_register, info.thread_ids_num, info.wave_size, info.lds_size_dwords,
	     info.threads_num[0], info.threads_num[1], info.threads_num[2],
	     info.tg_size_en ? "true" : "false");
	LOGF("\t threadgroup_id     = {%s, %s, %s}\n", info.group_id[0] ? "true" : "false",
	     info.group_id[1] ? "true" : "false", info.group_id[2] ? "true" : "false");
}

} // namespace Libs::Graphics
