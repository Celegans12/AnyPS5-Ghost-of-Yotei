#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/pipeline/shaderPrecompile.h"

#include "common/archive.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/envDefaults.h"
#include "common/envRead.h"
#include "common/logging/log.h"
#include "common/shaderEnv.h"
#include "common/startupTrace.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/shaderCompiler.h"
#include "kytyGitVersion.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <fstream>
#include <functional>
#include <fmt/format.h>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

// Defined at the end of the file: windows.h stays out of everything above.
void LowerCurrentThreadPriority();
void RaiseThreadPriority(std::thread& thread);

// The warm list's environment fingerprint (PrecompileWarmList::Entry): XXH3 of NAME=value over the Common::ShaderEnv
// names (common/shaderEnv.h: the settings that change what a journal record makes) that have an effective value
// (EnvOrDefault: the environment, else the code default), in the list's order. No other KYTY_ variable enters: a
// report, the overlay, a live A/B, a pad script or a dispatch skip list keeps the warm list. Logged once
// (ENV_FINGERPRINT hash= names= set=: the listed names, those with a value), so two runs' logs say whether they share
// the warm list.
uint64_t EnvironmentFingerprint() {
	static const uint64_t fingerprint = [] {
		uint64_t hash = 0;
		size_t   set  = 0;
		for (const char* name: Common::ShaderEnv::Names) {
			const char* value = Common::EnvOrDefault(name);
			if (value == nullptr) continue;
			const auto     entry   = fmt::format("{}={}", name, value);
			const uint64_t words[] = {hash, XXH3_64bits(entry.data(), entry.size())};
			hash                   = XXH3_64bits(words, sizeof(words));
			set++;
		}
		std::fprintf(stderr, "ENV_FINGERPRINT hash=%016llx names=%zu set=%zu\n", static_cast<unsigned long long>(hash),
		             std::size(Common::ShaderEnv::Names), set);
		std::fflush(stderr);
		return hash;
	}();
	return fingerprint;
}

// The warm list (PrecompileWarmList): magic, version, driver-cache signature (size, bytes), the driver cache's
// payload bytes at the save, entry count, entries; then XXH3 of everything before it.
constexpr char     WarmListMagic[8] = {'K', 'Y', 'W', 'A', 'R', 'M', 'L', 'S'};
constexpr uint32_t WarmListVersion  = 1;
// KYTY_SHADER_PRECOMPILE_WARM_PRUNE (<title>.warm2): as version 1 up to the payload bytes, then the section count and
// per section the emulator build (ShaderPrecompiler::Revision), its entry count and entries (the writing build's first);
// then XXH3 of everything before it.
constexpr uint32_t WarmListBuildsVersion = 2;
// Other builds whose entries a list by build keeps besides the running build's.
constexpr size_t   WarmListOtherBuilds   = 2;
constexpr char     WarmPruneSwitchPrefix[] = "KYTY_SHADER_PRECOMPILE_WARM_PRUNE";

constexpr char     JournalMagic[8] = {'K', 'Y', 'C', 'S', 'J', 'R', 'N', 'L'};
constexpr uint32_t JournalVersion  = 1;
constexpr uint32_t RecordMagic     = 0x52534b43u; // "CKSR"
constexpr size_t   HeaderBytes     = sizeof(JournalMagic) + sizeof(uint32_t) + sizeof(uint64_t);

// The graphics journal: magic, version, schema fingerprint, device fingerprint; then stage and pipeline records.
constexpr char     GraphicsJournalMagic[8] = {'K', 'Y', 'G', 'F', 'J', 'R', 'N', 'L'};
// 2 (4.11.2026): ShaderTessellationInputInfo::carried_vgprs (the 2.000 merged LS+HS program's carried VGPRs).
// 3 (4.11.2026): the pixel inputs target_packed_unorm_mask, target_round_mode, ps_front_face_all_bits,
// object_primitive_id and lds_size_dwords (part of the stage key, missing from the records).
constexpr uint32_t GraphicsJournalVersion  = 3;
constexpr uint32_t StageRecordMagic        = 0x47545347u; // "GSTG"
constexpr uint32_t PipelineRecordMagic     = 0x50495047u; // "GPIP"
constexpr size_t   GraphicsHeaderBytes = sizeof(GraphicsJournalMagic) + sizeof(uint32_t) + 2 * sizeof(uint64_t);

thread_local bool  g_background_thread = false;
ShaderPrecompiler* g_quick_exit_target = nullptr; // the live precompiler, for the at_quick_exit handler
// GetPrecompileProgress's values (written by Start, the workers and the last worker).
struct ProgressAtomics {
	std::atomic<bool>     started {false};
	std::atomic<bool>     done {false};
	std::atomic<uint32_t> queued {0};
	std::atomic<uint32_t> threads {0};
	std::atomic<uint32_t> compiled {0};
	std::atomic<uint32_t> seconds {0};
	std::atomic<uint32_t> queued_graphics {0};
	std::atomic<uint32_t> finished {0};
	std::atomic<uint32_t> finished_graphics {0};
	std::atomic<uint32_t> warm {0};
	std::atomic<uint32_t> suspects {0};
	std::atomic<uint32_t> other_layout_journals {0};
};
ProgressAtomics g_progress;
std::once_flag     g_quick_exit_registered;

using Specialization = ShaderRecompiler::IR::ResourceSpecialization;

// Records only load into the same layouts they were written from.
uint64_t SchemaFingerprint() {
	const auto text = fmt::format("v{} B{} I{} S{} C{}", JournalVersion, sizeof(Specialization::Buffer),
	                              sizeof(Specialization::Image), sizeof(Specialization::Sampler),
	                              sizeof(ShaderComputeInputInfo));
	return XXH3_64bits(text.data(), text.size());
}

uint64_t GraphicsSchemaFingerprint() {
	const auto text = fmt::format("v{} B{} I{} S{} V{} P{} R{} K{} C{}", GraphicsJournalVersion,
	                              sizeof(Specialization::Buffer), sizeof(Specialization::Image),
	                              sizeof(Specialization::Sampler), sizeof(ShaderVertexInputInfo),
	                              sizeof(ShaderPixelInputInfo), ShaderVertexInputInfo::RES_MAX,
	                              sizeof(PipelineStaticParameters), RENDER_COLOR_ATTACHMENTS_MAX);
	return XXH3_64bits(text.data(), text.size());
}

// The GPU the graphics records are for: vendor and device. Another driver version keeps them (they are inputs; a new
// driver is when they are worth the most).
uint64_t DeviceFingerprint(const GraphicContext& graphics) {
	const auto& properties = graphics.GetPhysicalDeviceProperties();
	const auto  text       = fmt::format("{:08x}:{:08x}", properties.vendorID, properties.deviceID);
	return XXH3_64bits(text.data(), text.size());
}

// The recompiler's dump label of a stage, as ProgramCache::Get names it.
const char* StageLabel(ShaderType stage) {
	switch (stage) {
		case ShaderType::Vertex: return "ShaderRecompiler VS";
		case ShaderType::Mesh: return "ShaderRecompiler MS";
		case ShaderType::Local: return "ShaderRecompiler LS";
		case ShaderType::TessellationControl: return "ShaderRecompiler HS";
		case ShaderType::TessellationEvaluation: return "ShaderRecompiler DS";
		case ShaderType::Pixel: return "ShaderRecompiler PS";
		default: return "ShaderRecompiler CS";
	}
}

const char* StageName(ShaderType stage) {
	switch (stage) {
		case ShaderType::Vertex: return "vs";
		case ShaderType::Mesh: return "ms";
		case ShaderType::Local: return "ls";
		case ShaderType::TessellationControl: return "hs";
		case ShaderType::TessellationEvaluation: return "ds";
		case ShaderType::Pixel: return "ps";
		default: return "cs";
	}
}

// KYTY_DIAG_PRECOMPILE_FILE_MATCH=1 (diagnostic, shader project, precompile step 0): with --precompile-only,
// a FILE_MATCH line per record gives its module (hash, words) with what came from the game at run time: the static
// input key, the user data and the resource specialization (hashes and a short form of its buffers and images), so
// the journals tell how many permutations a program has and what makes them (what a precompile from the game's files
// would have to know). No module or pipeline is made.
bool FileMatchDiag() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_DIAG_PRECOMPILE_FILE_MATCH");
		return value != nullptr && value[0] == '1';
	}();
	return on;
}

uint64_t SpecializationHash(const ShaderRecompiler::IR::ResourceSpecialization& s, std::string* images_text) {
	std::vector<uint32_t> words;
	for (const auto& b: s.buffers) {
		words.insert(words.end(), {b.packed_stride, static_cast<uint32_t>(b.descriptor_format), b.descriptor_swizzle,
		                           uint32_t {b.zero_stride_oob} | uint32_t {b.scalar_empty} << 1u, b.indirect_root,
		                           b.indirect_mapping_offset, b.indirect_search_iterations});
	}
	words.push_back(0xffffffffu);
	for (const auto& i: s.images) {
		words.insert(words.end(),
		             {static_cast<uint32_t>(i.numeric_class), static_cast<uint32_t>(i.dimension), i.mip_count,
		              static_cast<uint32_t>(i.conversion_format), i.shader_swizzle, i.indirect_root,
		              i.indirect_mapping_offset, i.indirect_search_iterations,
		              uint32_t {i.cube} | uint32_t {i.fmask} << 1u | uint32_t {i.bindless} << 2u | uint32_t {i.heap} << 3u,
		              i.bindless_stride_dwords});
		if (images_text != nullptr && images_text->size() < 200u) {
			*images_text += fmt::format("{}n{}d{}c{}m{}{}{}", images_text->empty() ? "" : "/",
			                            static_cast<uint32_t>(i.numeric_class), static_cast<uint32_t>(i.dimension),
			                            static_cast<uint32_t>(i.conversion_format), i.mip_count, i.bindless ? "b" : "",
			                            i.indirect_root != ShaderRecompiler::IR::ImageResource::NoIndirectImage ? "i" : "");
		}
	}
	words.push_back(0xfffffffeu);
	for (const auto& m: s.samplers) {
		words.insert(words.end(), {m.indirect_root, m.indirect_mapping_offset, m.indirect_search_iterations});
	}
	return XXH3_64bits(words.data(), words.size() * sizeof(uint32_t));
}

void FileMatchReport(const PrecompileStageRecord& record, uint64_t game_hash, size_t game_words) {
	std::vector<uint32_t> key;
	switch (record.stage) {
		case ShaderType::Pixel: BuildStageStaticKey(record.pixel, key); break;
		case ShaderType::Compute: BuildStageStaticKey(record.compute, key); break;
		default: BuildStageStaticKey(record.vertex, key); break;
	}
	std::string buffers;
	for (const auto& b: record.specialization.buffers) {
		if (buffers.size() > 240u) break;
		buffers += fmt::format("{}f{}s{}w{:x}{}{}{}", buffers.empty() ? "" : "/", static_cast<uint32_t>(b.descriptor_format),
		                       b.packed_stride, b.descriptor_swizzle, b.zero_stride_oob ? "z" : "", b.scalar_empty ? "e" : "",
		                       b.indirect_root != ShaderRecompiler::IR::BufferResource::NoIndirectBuffer ? "i" : "");
	}
	std::string images;
	const auto  spec_hash = SpecializationHash(record.specialization, &images);
	std::fprintf(stderr,
	             "FILE_MATCH stage=%s hash=0x%016llx game=0x%016llx/%zu static=0x%016llx ud=%u/0x%016llx spec=0x%016llx "
	             "buffers=%zu images=%zu samplers=%zu buf=%s img=%s\n",
	             StageName(record.stage), static_cast<unsigned long long>(record.shader_hash),
	             static_cast<unsigned long long>(game_hash), game_words,
	             static_cast<unsigned long long>(XXH3_64bits(key.data(), key.size() * sizeof(uint32_t))),
	             static_cast<unsigned>(record.user_data.size()),
	             static_cast<unsigned long long>(XXH3_64bits(record.user_data.data(), record.user_data.size() * sizeof(uint32_t))),
	             static_cast<unsigned long long>(spec_hash), record.specialization.buffers.size(),
	             record.specialization.images.size(), record.specialization.samplers.size(),
	             buffers.empty() ? "-" : buffers.c_str(), images.empty() ? "-" : images.c_str());
}

// The graphics stages a graphics journal holds.
bool GraphicsStage(ShaderType stage) {
	switch (stage) {
		case ShaderType::Vertex:
		case ShaderType::Mesh:
		case ShaderType::Local:
		case ShaderType::TessellationControl:
		case ShaderType::TessellationEvaluation:
		case ShaderType::Pixel: return true;
		default: return false;
	}
}

// Calls fn with the record key of each stage of a graphics pipeline.
template <typename Fn>
void ForEachStage(const PrecompileGraphicsRecord& record, Fn&& fn) {
	for (uint32_t i = 0; i < record.VertexStageCount(); i++) {
		fn(record.vertex[i]);
	}
	if (record.pixel != 0) {
		fn(record.pixel);
	}
}

class Writer {
public:
	void U32(uint32_t value) { words.push_back(value); }
	void U64(uint64_t value) {
		U32(static_cast<uint32_t>(value));
		U32(static_cast<uint32_t>(value >> 32u));
	}
	void Words(std::span<const uint32_t> values) {
		U32(static_cast<uint32_t>(values.size()));
		words.insert(words.end(), values.begin(), values.end());
	}
	std::vector<uint32_t> words;
};

class Reader {
public:
	explicit Reader(std::span<const uint32_t> words): m_words(words) {}
	uint32_t U32() {
		if (m_pos >= m_words.size()) {
			ok = false;
			return 0;
		}
		return m_words[m_pos++];
	}
	uint64_t U64() {
		const uint64_t low = U32();
		return low | (static_cast<uint64_t>(U32()) << 32u);
	}
	std::vector<uint32_t> Words(uint32_t limit) {
		const auto count = U32();
		if (!ok || count > limit || m_pos + count > m_words.size()) {
			ok = false;
			return {};
		}
		std::vector<uint32_t> values(m_words.begin() + static_cast<std::ptrdiff_t>(m_pos),
		                             m_words.begin() + static_cast<std::ptrdiff_t>(m_pos + count));
		m_pos += count;
		return values;
	}
	[[nodiscard]] bool AtEnd() const { return m_pos == m_words.size(); }
	bool ok = true;

private:
	std::span<const uint32_t> m_words;
	size_t                    m_pos = 0;
};

void WriteSpecialization(Writer& w, const Specialization& s) {
	w.U32(static_cast<uint32_t>(s.buffers.size()));
	for (const auto& b: s.buffers) {
		w.U32(b.packed_stride);
		w.U32(static_cast<uint32_t>(b.descriptor_format));
		w.U32(b.descriptor_swizzle);
		w.U32(b.zero_stride_oob ? 1u : 0u);
		w.U32(b.indirect_root);
		w.U32(b.indirect_mapping_offset);
		w.U32(b.indirect_search_iterations);
	}
	w.U32(static_cast<uint32_t>(s.images.size()));
	for (const auto& i: s.images) {
		w.U32(static_cast<uint32_t>(i.numeric_class));
		w.U32(static_cast<uint32_t>(i.dimension));
		w.U32(i.mip_count);
		w.U32(static_cast<uint32_t>(i.conversion_format));
		w.U32(i.shader_swizzle);
		w.U32(i.indirect_root);
		w.U32(i.indirect_mapping_offset);
		w.U32(i.indirect_search_iterations);
		w.U32((i.cube ? 1u : 0u) | (i.fmask ? 2u : 0u) | (i.bindless ? 4u : 0u));
		w.U32(i.bindless_stride_dwords);
	}
	w.U32(static_cast<uint32_t>(s.samplers.size()));
	for (const auto& sampler: s.samplers) {
		w.U32(sampler.indirect_root);
		w.U32(sampler.indirect_mapping_offset);
		w.U32(sampler.indirect_search_iterations);
	}
}

bool ReadSpecialization(Reader& r, Specialization& s) {
	constexpr uint32_t Limit = 4096;
	const auto buffers = r.U32();
	if (!r.ok || buffers > Limit) return false;
	s.buffers.resize(buffers);
	for (auto& b: s.buffers) {
		b.packed_stride              = r.U32();
		b.descriptor_format          = static_cast<Prospero::BufferFormat>(r.U32());
		b.descriptor_swizzle         = r.U32();
		b.zero_stride_oob            = r.U32() != 0u;
		b.indirect_root              = r.U32();
		b.indirect_mapping_offset    = r.U32();
		b.indirect_search_iterations = r.U32();
	}
	const auto images = r.U32();
	if (!r.ok || images > Limit) return false;
	s.images.resize(images);
	for (auto& i: s.images) {
		i.numeric_class              = static_cast<Prospero::TextureNumericClass>(r.U32());
		i.dimension                  = static_cast<ShaderRecompiler::Decoder::ImageDimension>(r.U32());
		i.mip_count                  = r.U32();
		i.conversion_format          = static_cast<Prospero::BufferFormat>(r.U32());
		i.shader_swizzle             = r.U32();
		i.indirect_root              = r.U32();
		i.indirect_mapping_offset    = r.U32();
		i.indirect_search_iterations = r.U32();
		const auto flags             = r.U32();
		i.cube                       = (flags & 1u) != 0u;
		i.fmask                      = (flags & 2u) != 0u;
		i.bindless                   = (flags & 4u) != 0u;
		i.bindless_stride_dwords     = r.U32();
	}
	const auto samplers = r.U32();
	if (!r.ok || samplers > Limit) return false;
	s.samplers.resize(samplers);
	for (auto& sampler: s.samplers) {
		sampler.indirect_root              = r.U32();
		sampler.indirect_mapping_offset    = r.U32();
		sampler.indirect_search_iterations = r.U32();
	}
	return r.ok;
}

void WriteInput(Writer& w, const ShaderComputeInputInfo& in) {
	for (const auto value: in.threads_num) w.U32(value);
	w.U32(in.lds_size_dwords);
	w.U32(in.scratch_size_dwords);
	w.U32(in.host_subgroup_size);
	w.U32(in.wave_size);
	w.U32(in.async_compute ? 1u : 0u);
	w.U32(in.float_mode);
	for (const auto value: in.dispatch_threads_num) w.U32(value);
	for (const auto value: in.workgroup_counts) w.U32(value);
	for (const auto value: in.group_id) w.U32(value ? 1u : 0u);
	w.U32(in.dispatch_thread_dimensions ? 1u : 0u);
	w.U32(in.lds_storage ? 1u : 0u);
	w.U32(static_cast<uint32_t>(in.thread_ids_num));
	w.U32(static_cast<uint32_t>(in.workgroup_register));
	w.U32(in.tg_size_en ? 1u : 0u);
}

void ReadInput(Reader& r, ShaderComputeInputInfo& in) {
	for (auto& value: in.threads_num) value = r.U32();
	in.lds_size_dwords     = r.U32();
	in.scratch_size_dwords = r.U32();
	in.host_subgroup_size  = r.U32();
	in.wave_size           = r.U32();
	in.async_compute       = r.U32() != 0u;
	in.float_mode          = static_cast<uint8_t>(r.U32());
	for (auto& value: in.dispatch_threads_num) value = r.U32();
	for (auto& value: in.workgroup_counts) value = r.U32();
	for (auto& value: in.group_id) value = r.U32() != 0u;
	in.dispatch_thread_dimensions = r.U32() != 0u;
	in.lds_storage                = r.U32() != 0u;
	in.thread_ids_num             = static_cast<int>(r.U32());
	in.workgroup_register         = static_cast<int>(r.U32());
	in.tg_size_en                 = r.U32() != 0u;
	in.stage                      = {};
}

// Every field of the input (the arrays up to their counts) but `stage`, which is not an input. A field added to
// ShaderVertexInputInfo or ShaderPixelInputInfo must be added here and GraphicsJournalVersion raised: the replay
// would compile without it (SHADER_PRECOMPILE_MISMATCH shows that).
void WriteVertexInput(Writer& w, const ShaderVertexInputInfo& in) {
	w.U32(static_cast<uint32_t>(in.logical_stage));
	w.U32(static_cast<uint32_t>(in.resources_num));
	for (int i = 0; i < in.resources_num; i++) {
		for (const auto field: in.resources[i].fields) w.U32(field);
		const auto& destination = in.resources_dst[i];
		w.U32(static_cast<uint32_t>(destination.register_start));
		w.U32(static_cast<uint32_t>(destination.registers_num));
		w.U32(static_cast<uint32_t>(destination.attr_id));
		w.U32(destination.fetch_index);
		w.U32(static_cast<uint32_t>(destination.buffer_index));
	}
	w.U32(static_cast<uint32_t>(in.buffers_num));
	for (int i = 0; i < in.buffers_num; i++) {
		const auto& buffer = in.buffers[i];
		w.U64(buffer.addr);
		w.U32(buffer.stride);
		w.U32(buffer.num_records);
		w.U32(buffer.fetch_index);
	}
	w.U32(static_cast<uint32_t>(in.fetch_attrib_reg));
	w.U32(static_cast<uint32_t>(in.fetch_buffer_reg));
	w.U32(in.wave_size);
	w.U32(in.scratch_size_dwords);
	w.U32(in.pa_cl_vs_out_cntl);
	for (const float value: in.clip_space.scale) w.U32(std::bit_cast<uint32_t>(value));
	for (const float value: in.clip_space.offset) w.U32(std::bit_cast<uint32_t>(value));
	for (const float value: in.clip_space.half_extent) w.U32(std::bit_cast<uint32_t>(value));
	w.U32(in.clip_space.enabled ? 1u : 0u);
	const auto& mesh = in.mesh;
	for (const auto value: mesh.threads_num) w.U32(value);
	w.U32(mesh.lds_size_dwords);
	w.U32(mesh.scratch_size_dwords);
	w.U32(mesh.host_subgroup_size);
	w.U32(mesh.wave_size);
	w.U32(mesh.input_primitive);
	w.U32(mesh.primitives_per_group);
	w.U32(mesh.vertices_per_group);
	w.U32(mesh.max_vertices);
	w.U32(mesh.max_primitives);
	w.U32(mesh.provoking_vertex);
	w.U32(mesh.fast_launch ? 1u : 0u);
	const auto& tess = in.tess;
	w.U32(tess.input_control_points);
	w.U32(tess.output_control_points);
	w.U32(tess.ls_stride);
	w.U32(tess.hs_stride);
	w.U32(tess.domain);
	w.U32(tess.partitioning);
	w.U32(tess.output_topology);
	for (const auto word: tess.carried_vgprs) w.U32(word);
	w.U32(in.indirect_draw.enabled ? 1u : 0u);
	w.U32(in.indirect_draw.base_vertex);
	w.U32(in.indirect_draw.base_instance);
	w.U32(in.indirect_draw.draw_index);
	// Bit 2: draw_base_once (draw base once). A new bit of an existing word keeps the layout (and
	// GraphicsJournalVersion): records written before it read as off, as they were compiled.
	// Bits 8..15: the table-mode key of 21.10-31.10 (KYTY_PERF_TABLE_BUFFERS, project D), always 0 now; a record with
	// them set is skipped (PrecompileStageRecord::table_mode).
	w.U32((in.fetch_external ? 1u : 0u) | (in.fetch_embedded ? 2u : 0u) | (in.draw_base_once ? 4u : 0u));
}

bool ReadVertexInput(Reader& r, ShaderVertexInputInfo& in, uint8_t& table_mode) {
	in               = {};
	in.logical_stage = static_cast<ShaderType>(r.U32());
	const auto resources = r.U32();
	if (!r.ok || resources > static_cast<uint32_t>(ShaderVertexInputInfo::RES_MAX)) return false;
	in.resources_num = static_cast<int>(resources);
	for (uint32_t i = 0; i < resources; i++) {
		for (auto& field: in.resources[i].fields) field = r.U32();
		auto& destination          = in.resources_dst[i];
		destination.register_start = static_cast<int>(r.U32());
		destination.registers_num  = static_cast<int>(r.U32());
		destination.attr_id        = static_cast<int>(r.U32());
		destination.fetch_index    = r.U32();
		destination.buffer_index   = static_cast<int>(r.U32());
	}
	const auto buffers = r.U32();
	if (!r.ok || buffers > static_cast<uint32_t>(ShaderVertexInputInfo::RES_MAX)) return false;
	in.buffers_num = static_cast<int>(buffers);
	for (uint32_t i = 0; i < buffers; i++) {
		auto& buffer       = in.buffers[i];
		buffer.addr        = r.U64();
		buffer.stride      = r.U32();
		buffer.num_records = r.U32();
		buffer.fetch_index = r.U32();
	}
	in.fetch_attrib_reg    = static_cast<int>(r.U32());
	in.fetch_buffer_reg    = static_cast<int>(r.U32());
	in.wave_size           = r.U32();
	in.scratch_size_dwords = r.U32();
	in.pa_cl_vs_out_cntl   = r.U32();
	for (auto& value: in.clip_space.scale) value = std::bit_cast<float>(r.U32());
	for (auto& value: in.clip_space.offset) value = std::bit_cast<float>(r.U32());
	for (auto& value: in.clip_space.half_extent) value = std::bit_cast<float>(r.U32());
	in.clip_space.enabled = r.U32() != 0u;
	auto& mesh = in.mesh;
	for (auto& value: mesh.threads_num) value = r.U32();
	mesh.lds_size_dwords      = r.U32();
	mesh.scratch_size_dwords  = r.U32();
	mesh.host_subgroup_size   = r.U32();
	mesh.wave_size            = r.U32();
	mesh.input_primitive      = r.U32();
	mesh.primitives_per_group = r.U32();
	mesh.vertices_per_group   = r.U32();
	mesh.max_vertices         = r.U32();
	mesh.max_primitives       = r.U32();
	mesh.provoking_vertex     = r.U32();
	mesh.fast_launch          = r.U32() != 0u;
	auto& tess                 = in.tess;
	tess.input_control_points  = r.U32();
	tess.output_control_points = r.U32();
	tess.ls_stride             = r.U32();
	tess.hs_stride             = r.U32();
	tess.domain                = r.U32();
	tess.partitioning          = r.U32();
	tess.output_topology       = r.U32();
	for (auto& word: tess.carried_vgprs) word = r.U32();
	in.indirect_draw.enabled       = r.U32() != 0u;
	in.indirect_draw.base_vertex   = static_cast<uint8_t>(r.U32());
	in.indirect_draw.base_instance = static_cast<uint8_t>(r.U32());
	in.indirect_draw.draw_index    = static_cast<uint8_t>(r.U32());
	const auto fetch               = r.U32();
	in.fetch_external              = (fetch & 1u) != 0u;
	in.fetch_embedded              = (fetch & 2u) != 0u;
	in.draw_base_once              = (fetch & 4u) != 0u;
	table_mode                     = static_cast<uint8_t>((fetch >> 8u) & 0xffu);
	return r.ok;
}

void WritePixelInput(Writer& w, const ShaderPixelInputInfo& in) {
	for (const auto value: in.interpolator_settings) w.U32(value);
	w.U32(in.input_num);
	w.U32(in.wave_size);
	w.U32(in.ps_system_input_base);
	w.U32(in.custom_interpolation_mask);
	w.U32(in.ps_perspective_center_vgpr);
	w.U32(in.ps_perspective_sample_vgpr);
	w.U32(in.ps_perspective_centroid_vgpr);
	for (const auto mode: in.target_output_mode) w.U32(mode);
	w.U32(in.target_shader_mask);
	for (const auto mapping: in.target_export_mapping) w.U32(mapping.packed);
	w.U32(in.scratch_size_dwords);
	const bool flags[] = {in.ps_pos_x,          in.ps_pos_y,
	                      in.ps_pos_z,          in.ps_pos_w,
	                      in.ps_front_face,     in.ps_ancillary,
	                      in.ps_no_perspective, in.ps_pixel_kill_enable,
	                      in.ps_depth_export_enable, in.ps_sample_mask_export_enable,
	                      in.ps_sample_shading, in.dual_source_blending,
	                      in.ps_early_z,        in.ps_execute_on_noop};
	uint32_t   bits    = 0;
	for (uint32_t i = 0; i < std::size(flags); i++) {
		bits |= (flags[i] ? 1u : 0u) << i;
	}
	// The ObjPrimID write: bits 16..19 slot + 1 (0 = none), 20..25 push dword, 26 ADD_PIPED_PRIM_ID. Older records
	// have zeros there, so they still read as "no ObjPrimID output" (no version change needed).
	if (in.HasObjprimOutput()) {
		bits |= ((in.objprim_slot + 1u) << 16u) | ((static_cast<uint32_t>(in.objprim_push_dword) & 0x3fu) << 20u) |
		        (static_cast<uint32_t>(in.objprim_add_prim_id) << 26u);
	}
	// Bits 27..29: the table-mode key (as the vertex input's fetch word), always 0 now.
	w.U32(bits);
	w.U32(static_cast<uint32_t>(in.parameter_mode));
	w.U32(static_cast<uint32_t>(in.alpha_blend_source));
	// Version 3: the pixel inputs BuildStageStaticKey also keys on (records without them reloaded with zeros there, so their
	// permutations were looked up under another key and translated again every session, 4.11.2026).
	w.U32(uint32_t {in.target_packed_unorm_mask} | (uint32_t {in.target_round_mode} << 8u) |
	      (static_cast<uint32_t>(in.ps_front_face_all_bits) << 16u) | (static_cast<uint32_t>(in.object_primitive_id) << 17u));
	w.U32(in.lds_size_dwords);
}

bool ReadPixelInput(Reader& r, ShaderPixelInputInfo& in, uint8_t& table_mode) {
	in = {};
	for (auto& value: in.interpolator_settings) value = r.U32();
	in.input_num                    = r.U32();
	in.wave_size                    = r.U32();
	in.ps_system_input_base         = r.U32();
	in.custom_interpolation_mask    = r.U32();
	in.ps_perspective_center_vgpr   = r.U32();
	in.ps_perspective_sample_vgpr   = r.U32();
	in.ps_perspective_centroid_vgpr = r.U32();
	for (auto& mode: in.target_output_mode) mode = static_cast<uint8_t>(r.U32());
	in.target_shader_mask = r.U32();
	for (auto& mapping: in.target_export_mapping) mapping.packed = static_cast<uint8_t>(r.U32());
	in.scratch_size_dwords = r.U32();
	bool* const flags[] = {&in.ps_pos_x,          &in.ps_pos_y,
	                       &in.ps_pos_z,          &in.ps_pos_w,
	                       &in.ps_front_face,     &in.ps_ancillary,
	                       &in.ps_no_perspective, &in.ps_pixel_kill_enable,
	                       &in.ps_depth_export_enable, &in.ps_sample_mask_export_enable,
	                       &in.ps_sample_shading, &in.dual_source_blending,
	                       &in.ps_early_z,        &in.ps_execute_on_noop};
	const auto  bits    = r.U32();
	for (uint32_t i = 0; i < std::size(flags); i++) {
		*flags[i] = ((bits >> i) & 1u) != 0u;
	}
	if (const auto slot = (bits >> 16u) & 0xfu; slot != 0u) {
		in.objprim_slot        = static_cast<uint8_t>(slot - 1u);
		in.objprim_push_dword  = static_cast<uint8_t>((bits >> 20u) & 0x3fu);
		in.objprim_add_prim_id = ((bits >> 26u) & 1u) != 0u;
	}
	table_mode            = static_cast<uint8_t>((bits >> 27u) & 7u);
	in.parameter_mode     = static_cast<ShaderPixelParameterMode>(r.U32());
	in.alpha_blend_source = static_cast<ShaderAlphaBlendSource>(r.U32());
	const auto targets            = r.U32();
	in.target_packed_unorm_mask   = static_cast<uint8_t>(targets & 0xffu);
	in.target_round_mode          = static_cast<uint8_t>((targets >> 8u) & 0xffu);
	in.ps_front_face_all_bits     = ((targets >> 16u) & 1u) != 0u;
	in.object_primitive_id        = ((targets >> 17u) & 1u) != 0u;
	in.lds_size_dwords            = r.U32();
	return r.ok && in.input_num <= std::size(in.interpolator_settings);
}

std::vector<uint32_t> Serialize(const PrecompileStageRecord& record) {
	Writer w;
	w.U32(RecordMagic);
	w.U64(record.shader_hash);
	w.Words(record.user_data);
	w.Words(record.code);
	WriteInput(w, record.compute);
	w.U32(record.push_data_start);
	WriteSpecialization(w, record.specialization);
	w.U64(record.spirv_hash);
	w.U64(record.revision);
	return std::move(w.words);
}

// `hint`: a hints file's record, without its code (PrecompileHintsImport).
bool Deserialize(std::span<const uint32_t> words, PrecompileStageRecord& record, bool hint = false) {
	Reader r(words);
	if (r.U32() != RecordMagic) return false;
	record.stage       = ShaderType::Compute;
	record.shader_hash = r.U64();
	record.user_data   = r.Words(64);
	record.code        = r.Words(1u << 22u);
	ReadInput(r, record.compute);
	record.push_data_start = r.U32();
	if (!r.ok || !ReadSpecialization(r, record.specialization)) return false;
	record.spirv_hash = r.U64();
	record.revision   = r.U64();
	if (!r.ok || !r.AtEnd() || (record.code.empty() && !hint)) return false;
	BuildStageStaticKey(record.compute, record.static_key);
	return true;
}

// A graphics stage record (the graphics journal).
std::vector<uint32_t> SerializeStage(const PrecompileStageRecord& record) {
	Writer w;
	w.U32(StageRecordMagic);
	w.U32(static_cast<uint32_t>(record.stage));
	w.U64(record.shader_hash);
	w.Words(record.user_data);
	w.Words(record.code);
	w.Words(record.back_code);
	if (record.stage == ShaderType::Pixel) {
		WritePixelInput(w, record.pixel);
	} else {
		WriteVertexInput(w, record.vertex);
	}
	w.U32(record.push_data_start);
	WriteSpecialization(w, record.specialization);
	w.U64(record.spirv_hash);
	w.U64(record.revision);
	return std::move(w.words);
}

bool DeserializeStage(std::span<const uint32_t> words, PrecompileStageRecord& record, bool hint = false) {
	Reader r(words);
	if (r.U32() != StageRecordMagic) return false;
	record.stage = static_cast<ShaderType>(r.U32());
	if (!r.ok || !GraphicsStage(record.stage)) return false;
	record.shader_hash = r.U64();
	record.user_data   = r.Words(64);
	record.code        = r.Words(1u << 22u);
	record.back_code   = r.Words(1u << 22u);
	if (!r.ok) return false;
	if (record.stage == ShaderType::Pixel) {
		if (!ReadPixelInput(r, record.pixel, record.table_mode)) return false;
	} else if (!ReadVertexInput(r, record.vertex, record.table_mode) || record.vertex.logical_stage != record.stage) {
		// The program cache takes a vertex input's stage from its logical stage.
		return false;
	}
	record.push_data_start = r.U32();
	if (!r.ok || !ReadSpecialization(r, record.specialization)) return false;
	record.spirv_hash = r.U64();
	record.revision   = r.U64();
	if (!r.ok || !r.AtEnd() || (record.code.empty() && !hint)) return false;
	if (record.stage == ShaderType::Pixel) {
		BuildStageStaticKey(record.pixel, record.static_key);
	} else {
		BuildStageStaticKey(record.vertex, record.static_key);
	}
	return true;
}

// Whether a lookup of this run can ask for the graphics stage record: one made with a table-mode key (KYTY_PERF_TABLE_BUFFERS
// or project D's table mode, both removed) would only be translated for nothing.
bool TableKeyAsked(const PrecompileStageRecord& record) {
	return record.table_mode == 0u;
}

// The pipeline key's state of a graphics pipeline record (the arrays up to their counts), behind its stage keys.
void WriteGraphicsState(Writer& w, const PrecompileGraphicsRecord& record) {
	for (const auto key: record.vertex) w.U64(key);
	w.U64(record.pixel);
	const auto& rendering = record.rendering;
	w.U32(rendering.color_count);
	for (const auto format: rendering.color_formats) w.U32(static_cast<uint32_t>(format));
	w.U32(static_cast<uint32_t>(rendering.depth_format));
	w.U32(static_cast<uint32_t>(rendering.stencil_format));
	const auto& input = record.vertex_input;
	w.U32(input.binding_count);
	for (uint32_t i = 0; i < input.binding_count && i < input.bindings.size(); i++) {
		w.U32(input.bindings[i].stride);
		w.U32(input.bindings[i].instance ? 1u : 0u);
	}
	w.U32(input.attribute_count);
	for (uint32_t i = 0; i < input.attribute_count && i < input.attributes.size(); i++) {
		w.U32(input.attributes[i].offset);
		w.U32(input.attributes[i].binding);
	}
	// Packed (no padding) and trivially copyable: its bytes are its value, as the pipeline cache hashes it.
	std::array<uint32_t, (sizeof(PipelineStaticParameters) + 3u) / 4u> params {};
	std::memcpy(params.data(), &record.static_params, sizeof(PipelineStaticParameters));
	for (const auto word: params) w.U32(word);
}

std::vector<uint32_t> SerializePipeline(const PrecompileGraphicsRecord& record) {
	Writer w;
	w.U32(PipelineRecordMagic);
	WriteGraphicsState(w, record);
	w.U64(record.revision);
	return std::move(w.words);
}

bool DeserializePipeline(std::span<const uint32_t> words, PrecompileGraphicsRecord& record) {
	Reader r(words);
	if (r.U32() != PipelineRecordMagic) return false;
	for (auto& key: record.vertex) key = r.U64();
	record.pixel     = r.U64();
	auto& rendering  = record.rendering;
	rendering.color_count = r.U32();
	for (auto& format: rendering.color_formats) format = static_cast<vk::Format>(r.U32());
	rendering.depth_format   = static_cast<vk::Format>(r.U32());
	rendering.stencil_format = static_cast<vk::Format>(r.U32());
	auto&      input    = record.vertex_input;
	const auto bindings = r.U32();
	if (!r.ok || bindings > input.bindings.size()) return false;
	input.binding_count = static_cast<uint8_t>(bindings);
	for (uint32_t i = 0; i < bindings; i++) {
		input.bindings[i].stride   = r.U32();
		input.bindings[i].instance = r.U32() != 0u;
	}
	const auto attributes = r.U32();
	if (!r.ok || attributes > input.attributes.size()) return false;
	input.attribute_count = static_cast<uint8_t>(attributes);
	for (uint32_t i = 0; i < attributes; i++) {
		input.attributes[i].offset  = r.U32();
		input.attributes[i].binding = static_cast<uint8_t>(r.U32());
	}
	std::array<uint32_t, (sizeof(PipelineStaticParameters) + 3u) / 4u> params {};
	for (auto& word: params) word = r.U32();
	std::memcpy(&record.static_params, params.data(), sizeof(PipelineStaticParameters));
	record.revision = r.U64();
	// A tessellated pipeline has both tessellation stages.
	return r.ok && r.AtEnd() && record.vertex[0] != 0 && (record.vertex[1] == 0) == (record.vertex[2] == 0) &&
	       rendering.color_count <= RENDER_COLOR_ATTACHMENTS_MAX;
}

// A record: payload size in words, payload, XXH3 of the payload.
void WritePayload(std::FILE* file, std::span<const uint32_t> payload) {
	const auto size = static_cast<uint32_t>(payload.size());
	const auto hash = XXH3_64bits(payload.data(), payload.size_bytes());
	std::fwrite(&size, sizeof(size), 1, file);
	std::fwrite(payload.data(), sizeof(uint32_t), payload.size(), file);
	std::fwrite(&hash, sizeof(hash), 1, file);
}

// Calls fn(payload, intact) for each record of a journal's words. False when a torn tail (a run killed while
// appending) ended the list.
template <typename Fn>
bool ForEachPayload(std::span<const uint32_t> words, Fn&& fn) {
	size_t pos = 0;
	while (pos < words.size()) {
		const auto size = words[pos];
		if (size == 0 || pos + 1u + size + 2u > words.size()) {
			return false;
		}
		const std::span<const uint32_t> payload(words.data() + pos + 1u, size);
		const uint64_t stored = static_cast<uint64_t>(words[pos + 1u + size]) |
		                        (static_cast<uint64_t>(words[pos + 2u + size]) << 32u);
		pos += 3u + size;
		fn(payload, XXH3_64bits(payload.data(), payload.size_bytes()) == stored);
	}
	return true;
}

std::string HexKey(uint64_t key) {
	return fmt::format("{:016x}", key);
}

std::vector<uint8_t> ReadFileBytes(const std::filesystem::path& path) {
	std::vector<uint8_t> bytes;
	if (std::FILE* f = std::fopen(path.string().c_str(), "rb")) {
		std::vector<uint8_t> chunk(1u << 16u);
		size_t               read = 0;
		while ((read = std::fread(chunk.data(), 1, chunk.size(), f)) > 0) {
			bytes.insert(bytes.end(), chunk.begin(), chunk.begin() + static_cast<std::ptrdiff_t>(read));
		}
		std::fclose(f);
	}
	return bytes;
}

// A warm list file: false unless it is intact and was written for `signature`.
bool ParseWarmList(std::span<const uint8_t> bytes, std::string_view signature, uint64_t& cache_bytes,
                   std::vector<uint64_t>& entries) {
	if (bytes.size() < sizeof(WarmListMagic) + 2 * sizeof(uint32_t) + 3 * sizeof(uint64_t)) return false;
	uint64_t checksum = 0;
	std::memcpy(&checksum, bytes.data() + bytes.size() - sizeof(checksum), sizeof(checksum));
	const auto body = bytes.first(bytes.size() - sizeof(checksum));
	if (XXH3_64bits(body.data(), body.size()) != checksum) return false;
	size_t     offset = 0;
	const auto take   = [&](void* out, size_t size) {
		if (body.size() - offset < size) return false;
		std::memcpy(out, body.data() + offset, size);
		offset += size;
		return true;
	};
	char     magic[sizeof(WarmListMagic)] {};
	uint32_t version        = 0;
	uint32_t signature_size = 0;
	uint64_t count          = 0;
	if (!take(magic, sizeof(magic)) || std::memcmp(magic, WarmListMagic, sizeof(magic)) != 0 ||
	    !take(&version, sizeof(version)) || version != WarmListVersion ||
	    !take(&signature_size, sizeof(signature_size)) || signature_size != signature.size() ||
	    body.size() - offset < signature_size ||
	    std::memcmp(body.data() + offset, signature.data(), signature_size) != 0) {
		return false;
	}
	offset += signature_size;
	if (!take(&cache_bytes, sizeof(cache_bytes)) || !take(&count, sizeof(count)) ||
	    (body.size() - offset) % sizeof(uint64_t) != 0 || count != (body.size() - offset) / sizeof(uint64_t)) {
		return false;
	}
	entries.resize(static_cast<size_t>(count));
	std::memcpy(entries.data(), body.data() + offset, entries.size() * sizeof(uint64_t));
	return true;
}

// A warm list file by build (WarmListBuildsVersion): false unless it is intact and was written for `signature`.
// `sections`: (build, entries) in file order.
bool ParseWarmListBuilds(std::span<const uint8_t> bytes, std::string_view signature, uint64_t& cache_bytes,
                         std::vector<std::pair<uint64_t, std::vector<uint64_t>>>& sections) {
	if (bytes.size() < sizeof(WarmListMagic) + 3 * sizeof(uint32_t) + 2 * sizeof(uint64_t)) return false;
	uint64_t checksum = 0;
	std::memcpy(&checksum, bytes.data() + bytes.size() - sizeof(checksum), sizeof(checksum));
	const auto body = bytes.first(bytes.size() - sizeof(checksum));
	if (XXH3_64bits(body.data(), body.size()) != checksum) return false;
	size_t     offset = 0;
	const auto take   = [&](void* out, size_t size) {
		if (body.size() - offset < size) return false;
		std::memcpy(out, body.data() + offset, size);
		offset += size;
		return true;
	};
	char     magic[sizeof(WarmListMagic)] {};
	uint32_t version        = 0;
	uint32_t signature_size = 0;
	uint32_t section_count  = 0;
	if (!take(magic, sizeof(magic)) || std::memcmp(magic, WarmListMagic, sizeof(magic)) != 0 ||
	    !take(&version, sizeof(version)) || version != WarmListBuildsVersion ||
	    !take(&signature_size, sizeof(signature_size)) || signature_size != signature.size() ||
	    body.size() - offset < signature_size ||
	    std::memcmp(body.data() + offset, signature.data(), signature_size) != 0) {
		return false;
	}
	offset += signature_size;
	if (!take(&cache_bytes, sizeof(cache_bytes)) || !take(&section_count, sizeof(section_count))) return false;
	sections.clear();
	for (uint32_t i = 0; i < section_count; i++) {
		uint64_t revision = 0;
		uint64_t count    = 0;
		if (!take(&revision, sizeof(revision)) || !take(&count, sizeof(count)) ||
		    count > (body.size() - offset) / sizeof(uint64_t)) {
			return false;
		}
		std::vector<uint64_t> entries(static_cast<size_t>(count));
		if (!entries.empty()) {
			std::memcpy(entries.data(), body.data() + offset, entries.size() * sizeof(uint64_t));
		}
		offset += entries.size() * sizeof(uint64_t);
		sections.emplace_back(revision, std::move(entries));
	}
	return offset == body.size();
}

double MillisecondsSince(std::chrono::steady_clock::time_point begin) {
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
}

// ---- the shader reuse ----

// KYTY_PERF_SHADER_REUSE_MB (default 2048, 64..16384): what the store may hold.
size_t ReuseBudgetBytes() {
	static const size_t budget = [] {
		// Measured: 1024 dropped 906 results (39 of 100 ms or more) on the user's pipeline cache at the gate start;
		// 2048: none dropped, peak 1282 MiB, saved 32.2 -> 33.6 s.
		size_t      megabytes = 2048;
		const char* value     = std::getenv("KYTY_PERF_SHADER_REUSE_MB");
		if (value != nullptr && value[0] != '\0') {
			megabytes = std::clamp<size_t>(std::strtoull(value, nullptr, 10), 64u, 16384u);
		}
		return megabytes << 20u;
	}();
	return budget;
}

// Totals since the start (PERF shader_reuse); the store's sizes are written under its lock.
struct ReuseStats {
	std::atomic<uint64_t> plan_hits {0};   // first uses that took a stored resource plan
	std::atomic<uint64_t> plan_misses {0}; // first uses without one
	std::atomic<uint64_t> found {0};       // stored compiles of exactly the inputs (=1 used, =2 compared)
	std::atomic<uint64_t> used {0};        // =1: CompileProgram (and the translation, if none was needed) skipped
	std::atomic<uint64_t> misses[4] {};    // ReuseMiss: unrecorded, pending, dropped, differs
	// The dropped ones by cause: a stage no job compiled (no recorded pipeline used it), a stage compiled but not in
	// the store (taken, store full or failed), a compute job's result not in the store.
	std::atomic<uint64_t> dropped_by[3] {};
	std::atomic<uint64_t> stage_only_queued {0};   // stage-only jobs (stages no recorded pipeline uses)
	std::atomic<uint64_t> stage_only_compiled {0}; // the same: compiled (kept or not)
	std::atomic<uint64_t> saved_us {0};    // what the workers spent on what the GPU thread skipped
	std::atomic<uint64_t> stored {0};      // compiles kept by the workers
	std::atomic<uint64_t> late {0};        // finished after the GPU thread had compiled them itself
	std::atomic<uint64_t> full {0};        // not kept: the store was at KYTY_PERF_SHADER_REUSE_MB
	std::atomic<uint64_t> held_bytes {0};
	std::atomic<uint64_t> peak_bytes {0};
	std::atomic<uint64_t> full_start {0};  // not kept: started with the store at its budget (never offered to it)
	std::atomic<uint64_t> full_us {0};     // the workers' time on results not kept for the budget (full, full_start)
	std::atomic<uint64_t> full_big {0};    // of them those of ReuseExpensiveMs or more
};
ReuseStats g_reuse;

// A result this expensive (translate + compile on the worker) is a stall of the GPU thread when it is not kept
// (full_big): the SHADER_TIME rows of 100 ms and more.
constexpr double ReuseExpensiveMs = 100.0;

// A finished result the store did not keep for its budget (`offered`: StoreReuse refused it; otherwise its compile
// started with the store at the budget and kept nothing).
void ReuseNoteFull(double ms, bool offered) {
	(offered ? g_reuse.full : g_reuse.full_start).fetch_add(1, std::memory_order_relaxed);
	g_reuse.full_us.fetch_add(static_cast<uint64_t>(std::max(ms, 0.0) * 1000.0), std::memory_order_relaxed);
	if (ms >= ReuseExpensiveMs) g_reuse.full_big.fetch_add(1, std::memory_order_relaxed);
}

using CompiledInfo = ShaderRecompiler::IR::CompiledShaderInfo;
using ResourcePlan = ShaderRecompiler::IR::ResourcePlan;

PrecompileReuseIdentity RecordIdentity(const PrecompileStageRecord& record) {
	return {.stage           = record.stage,
	        .shader_hash     = record.shader_hash,
	        .user_data_count = static_cast<uint32_t>(record.user_data.size()),
	        .code            = record.code,
	        .back_code       = record.back_code,
	        .static_key      = record.static_key,
	        .lds_storage     = record.stage == ShaderType::Compute && record.compute.lds_storage};
}

// The program a translation is of: every field of the identity (the code and back half by size; equality compares
// them word by word).
uint64_t ProgramKey(const PrecompileReuseIdentity& identity) {
	Writer w;
	w.U32(static_cast<uint32_t>(identity.stage));
	w.U64(identity.shader_hash);
	w.U32(identity.user_data_count);
	w.U32(static_cast<uint32_t>(identity.code.size()));
	w.U32(static_cast<uint32_t>(identity.back_code.size()));
	w.Words(identity.static_key);
	w.U32(identity.lds_storage ? 1u : 0u);
	return XXH3_64bits(w.words.data(), w.words.size() * sizeof(uint32_t));
}

bool SameProgram(const PrecompileStageRecord& record, const PrecompileReuseIdentity& identity) {
	const auto own = RecordIdentity(record);
	return own.stage == identity.stage && own.shader_hash == identity.shader_hash &&
	       own.user_data_count == identity.user_data_count && own.lds_storage == identity.lds_storage &&
	       std::ranges::equal(own.code, identity.code) && std::ranges::equal(own.back_code, identity.back_code) &&
	       std::ranges::equal(own.static_key, identity.static_key);
}

// Estimates of what an entry holds (the budget is approximate; the SPIR-V is most of it).
size_t InfoBytes(const CompiledInfo& info) {
	using namespace ShaderRecompiler::IR;
	return sizeof(CompiledInfo) + info.info.buffers.size() * sizeof(BufferResource) +
	       info.info.images.size() * sizeof(ImageResource) + info.info.samplers.size() * sizeof(SamplerResource) +
	       info.info.inputs.size() * sizeof(StageInput) + info.info.outputs.size() * sizeof(StageOutput) +
	       info.info.user_data_registers.size() * sizeof(uint32_t);
}

size_t PlanBytes(const ResourcePlan& plan) {
	using namespace ShaderRecompiler::IR;
	return sizeof(ResourcePlan) + plan.value_storage.size() * (sizeof(Inst) + 2 * sizeof(void*)) +
	       plan.memory_info.size() * sizeof(MemoryInfo) + plan.descriptor_sources.size() * sizeof(DescriptorSource) +
	       plan.control_flow.size() * sizeof(ResourceBlock) + plan.srt_reads.size() * sizeof(SrtRead) +
	       plan.info.buffers.size() * sizeof(BufferResource) + plan.info.images.size() * sizeof(ImageResource);
}

// "<revision> <key>" lines (both hex), as written by WriteInflightLocked and to .suspects.
std::vector<std::pair<uint64_t, uint64_t>> ReadKeyLines(const std::filesystem::path& path) {
	std::vector<std::pair<uint64_t, uint64_t>> lines;
	if (std::FILE* f = std::fopen(path.string().c_str(), "rb")) {
		char line[96];
		while (std::fgets(line, sizeof(line), f) != nullptr) {
			char*      end      = nullptr;
			const auto revision = std::strtoull(line, &end, 16);
			const auto key      = std::strtoull(end, nullptr, 16);
			if (key != 0) lines.emplace_back(revision, key);
		}
		std::fclose(f);
	}
	return lines;
}

} // namespace

PrecompileProgress GetPrecompileProgress() {
	PrecompileProgress progress;
	progress.started  = g_progress.started.load(std::memory_order_acquire);
	progress.done     = g_progress.done.load(std::memory_order_acquire);
	progress.queued   = g_progress.queued.load(std::memory_order_relaxed);
	progress.threads  = g_progress.threads.load(std::memory_order_relaxed);
	progress.compiled = g_progress.compiled.load(std::memory_order_relaxed);
	progress.seconds  = g_progress.seconds.load(std::memory_order_relaxed);
	progress.queued_graphics   = g_progress.queued_graphics.load(std::memory_order_relaxed);
	progress.finished          = g_progress.finished.load(std::memory_order_relaxed);
	progress.finished_graphics = g_progress.finished_graphics.load(std::memory_order_relaxed);
	progress.warm              = g_progress.warm.load(std::memory_order_relaxed);
	progress.suspects          = g_progress.suspects.load(std::memory_order_relaxed);
	progress.other_layout_journals = g_progress.other_layout_journals.load(std::memory_order_relaxed);
	return progress;
}

bool IsBackgroundCompileThread() {
	return g_background_thread;
}

void EnterBackgroundCompileThread(const char* name) {
	g_background_thread = true;
	Common::Thread::SetCurrentName(name);
	LowerCurrentThreadPriority();
}

bool ShaderReuseEnabled() {
	static const bool enabled = [] {
		// --precompile-only: no GPU thread takes a stored result, and jobs whose pipelines the driver cache holds
		// already (warm list) are left out at load instead of being translated for the store.
		const bool on = ShaderPrecompiler::Enabled() && !Config::PrecompileOnly();
		if (on) {
			std::fprintf(stderr, "Shader reuse on (store up to %zu MiB, KYTY_PERF_SHADER_REUSE_MB)\n", ReuseBudgetBytes() >> 20u);
			std::fflush(stderr);
		}
		return on;
	}();
	return enabled;
}

void ShaderReuseNoteUsed(double saved_ms) {
	g_reuse.used.fetch_add(1, std::memory_order_relaxed);
	g_reuse.saved_us.fetch_add(static_cast<uint64_t>(std::max(saved_ms, 0.0) * 1000.0), std::memory_order_relaxed);
	if (Common::StartupTrace::Enabled()) {
		Common::StartupTrace::AddMs(Common::StartupTrace::Counter::ShaderReuse, saved_ms);
	}
}

void ReportShaderReusePerf() {
	if (!ShaderReuseEnabled()) return;
	const auto n = [](const std::atomic<uint64_t>& value) {
		return static_cast<unsigned long long>(value.load(std::memory_order_relaxed));
	};
	std::fprintf(stderr,
	             "PERF shader_reuse used=%llu found=%llu plan=%llu/%llu miss(unrecorded,pending,dropped,"
	             "differs)=%llu,%llu,%llu,%llu saved_ms=%llu stored=%llu late=%llu full=%llu held=%lluMiB peak=%lluMiB "
	             "full_start=%llu full_ms=%llu full_big=%llu "
	             "dropped(orphan,not_kept,compute)=%llu,%llu,%llu stage_only=%llu/%llu\n",
	             n(g_reuse.used), n(g_reuse.found), n(g_reuse.plan_hits), n(g_reuse.plan_hits) + n(g_reuse.plan_misses),
	             n(g_reuse.misses[0]), n(g_reuse.misses[1]), n(g_reuse.misses[2]), n(g_reuse.misses[3]),
	             n(g_reuse.saved_us) / 1000u, n(g_reuse.stored), n(g_reuse.late), n(g_reuse.full),
	             n(g_reuse.held_bytes) >> 20u, n(g_reuse.peak_bytes) >> 20u,
	             n(g_reuse.full_start), n(g_reuse.full_us) / 1000u, n(g_reuse.full_big), n(g_reuse.dropped_by[0]),
	             n(g_reuse.dropped_by[1]), n(g_reuse.dropped_by[2]), n(g_reuse.stage_only_compiled),
	             n(g_reuse.stage_only_queued));
}

uint64_t PrecompileKey(ShaderType stage, uint64_t shader_hash, uint32_t user_data_count, uint32_t code_words,
                       std::span<const uint32_t> static_key, uint32_t push_data_start,
                       const ShaderRecompiler::IR::ResourceSpecialization& specialization) {
	Writer w;
	w.U64(shader_hash);
	w.U32(user_data_count);
	w.U32(code_words);
	w.Words(static_key);
	w.U32(push_data_start);
	WriteSpecialization(w, specialization);
	if (stage != ShaderType::Compute) {
		w.U32(static_cast<uint32_t>(stage));
	}
	return XXH3_64bits(w.words.data(), w.words.size() * sizeof(uint32_t));
}

uint64_t PrecompileStageRecord::Key() const {
	return PrecompileKey(stage, shader_hash, static_cast<uint32_t>(user_data.size()),
	                     static_cast<uint32_t>(code.size()), static_key, push_data_start, specialization);
}

uint64_t PrecompileGraphicsRecord::Key() const {
	Writer w;
	WriteGraphicsState(w, *this);
	return XXH3_64bits(w.words.data(), w.words.size() * sizeof(uint32_t));
}

uint64_t PrecompileCodeHash(std::span<const uint32_t> code, std::span<const uint32_t> back_code) {
	const uint64_t hash = XXH3_64bits(code.data(), code.size_bytes());
	if (back_code.empty()) {
		return hash;
	}
	const uint64_t hashes[] = {hash, XXH3_64bits(back_code.data(), back_code.size_bytes())};
	return XXH3_64bits(hashes, sizeof(hashes));
}

// --export-hints / --import-hints (PrecompileHintsExport / PrecompileHintsImport). A hints file is the shader journals
// without the game's code: magic "KYHINTS1", version, the two schema fingerprints; then framed payloads (as a
// journal's: size, words, XXH3): one info payload (title id, game version, journal suffix), a compute hint per compute
// record, a stage hint per graphics stage record and the pipeline records as they are. A hint holds its record with the
// code (and a merged stage's back half) left out and the user data values zeroed (the recompiler and the record key use
// only their count; the values are addresses of one run), plus, per code block, its size in bytes and the XXH3 of its
// last 48 bytes: the block's footer, which starts with "barefoot". The import finds every footer in the files of the
// game folder, takes the bytes before it of each size a hint names with that footer hash and keeps them when their
// XXH3 is the record's shader hash; the records found are added to this device's journals.
namespace {

constexpr char     HintsMagic[8]      = {'K', 'Y', 'H', 'I', 'N', 'T', 'S', '1'};
constexpr uint32_t HintsVersion       = 1;
constexpr size_t   HintsHeaderBytes   = sizeof(HintsMagic) + sizeof(uint32_t) + 2 * sizeof(uint64_t);
constexpr uint32_t HintInfoMagic      = 0x464e4948u; // "HINF"
constexpr uint32_t HintComputeMagic   = 0x52534348u; // "HCSR"
constexpr uint32_t HintStageMagic     = 0x54534748u; // "HGST"
constexpr size_t   HintFooterBytes    = 48;
constexpr char     HintFooterText[8]  = {'b', 'a', 'r', 'e', 'f', 'o', 'o', 't'};

// A code block's footer hash: XXH3 of its last 48 bytes; 0 when the block has no footer (it cannot be found).
uint64_t HintFooterHash(std::span<const uint32_t> code) {
	if (code.size_bytes() < HintFooterBytes) return 0;
	const auto* footer = reinterpret_cast<const char*>(code.data()) + code.size_bytes() - HintFooterBytes;
	if (std::memcmp(footer, HintFooterText, sizeof(HintFooterText)) != 0) return 0;
	return XXH3_64bits(footer, HintFooterBytes) | 1u;
}

void WriteText(Writer& w, std::string_view text) {
	std::vector<uint32_t> words((text.size() + 3u) / 4u, 0u);
	std::memcpy(words.data(), text.data(), text.size());
	w.U32(static_cast<uint32_t>(text.size()));
	w.Words(words);
}

std::string ReadText(Reader& r) {
	const auto size  = r.U32();
	const auto words = r.Words(256);
	if (!r.ok || size > words.size() * 4u) {
		r.ok = false;
		return {};
	}
	return {reinterpret_cast<const char*>(words.data()), size};
}

// A journal file's records (payloads) when its header is `magic`, `version` and `fingerprint` (a graphics journal: and
// any device). False when the file is missing or of another layout.
bool ReadJournalPayloads(const std::filesystem::path& path, const char (&magic)[8], uint32_t version, uint64_t fingerprint,
                         size_t header_bytes, std::vector<std::vector<uint32_t>>& payloads, uint64_t* device = nullptr) {
	const auto bytes = ReadFileBytes(path);
	if (bytes.size() < header_bytes || std::memcmp(bytes.data(), magic, sizeof(magic)) != 0) return false;
	uint32_t file_version     = 0;
	uint64_t file_fingerprint = 0;
	std::memcpy(&file_version, bytes.data() + sizeof(magic), sizeof(file_version));
	std::memcpy(&file_fingerprint, bytes.data() + sizeof(magic) + sizeof(file_version), sizeof(file_fingerprint));
	if (file_version != version || file_fingerprint != fingerprint) return false;
	if (device != nullptr) {
		std::memcpy(device, bytes.data() + sizeof(magic) + sizeof(file_version) + sizeof(file_fingerprint), sizeof(*device));
	}
	std::vector<uint32_t> words((bytes.size() - header_bytes) / sizeof(uint32_t));
	std::memcpy(words.data(), bytes.data() + header_bytes, words.size() * sizeof(uint32_t));
	ForEachPayload(words, [&](std::span<const uint32_t> payload, bool intact) {
		if (intact) payloads.emplace_back(payload.begin(), payload.end());
	});
	return true;
}

// The journal suffix of this run's mode (PipelineCache: ".classes<n>" or empty), from the journal's file name.
std::string JournalSuffix(const std::filesystem::path& journal) {
	auto name = journal.filename().string();
	const auto dot = name.find('.');
	const auto end = name.rfind(".cs.journal");
	return dot != std::string::npos && end != std::string::npos && end > dot ? name.substr(dot, end - dot) : std::string();
}

struct HintCode {
	uint32_t bytes  = 0;
	uint64_t footer = 0;
	uint64_t hash   = 0; // XXH3 of the block
};

void WriteHintCode(Writer& w, std::span<const uint32_t> code) {
	w.U32(static_cast<uint32_t>(code.size_bytes()));
	w.U64(code.empty() ? 0u : HintFooterHash(code));
	w.U64(code.empty() ? 0u : XXH3_64bits(code.data(), code.size_bytes()));
}

HintCode ReadHintCode(Reader& r) {
	HintCode c;
	c.bytes  = r.U32();
	c.footer = r.U64();
	c.hash   = r.U64();
	return c;
}

// Calls fn(block bytes, footer hash) for every footer in the file (chunks with an overlap of `max_bytes` before each
// footer). False when the file cannot be read.
template <typename Fn>
bool ForEachFooter(const std::filesystem::path& path, size_t max_bytes, Fn&& fn) {
	// A member of a .zar game (Common::IsArchivePath) is read through the archive; a host file by its wide path (a
	// narrow fopen lost every path with characters outside the code page).
	std::unique_ptr<Common::ArchiveFile> member;
	std::ifstream                        host;
	if (Common::IsArchivePath(path)) {
		member = Common::OpenArchiveFile(path);
		if (member == nullptr) return false;
	} else {
		host.open(path, std::ios::binary);
		if (!host) return false;
	}
	const auto fread_chunk = [&](char* data, size_t size) -> size_t {
		if (member != nullptr) {
			uint32_t got = 0;
			member->Read(data, static_cast<uint32_t>(size), &got);
			return got;
		}
		host.read(data, static_cast<std::streamsize>(size));
		return static_cast<size_t>(host.gcount());
	};
	constexpr size_t     chunk = size_t {64} << 20u;
	std::vector<char>    buffer;
	size_t               kept  = 0; // bytes of the previous chunk kept in front (the overlap)
	std::vector<char>    read_buffer(chunk);
	const std::boyer_moore_horspool_searcher searcher(std::begin(HintFooterText), std::end(HintFooterText));
	for (;;) {
		const size_t got = fread_chunk(read_buffer.data(), chunk);
		if (got == 0) break;
		buffer.resize(kept + got);
		std::memcpy(buffer.data() + kept, read_buffer.data(), got);
		// Footers that end in this buffer and start after the part searched before (the first `searched_from` bytes of
		// the overlap were searched with the previous chunk, except for a footer cut by its end).
		const size_t searched_from = kept > HintFooterBytes ? kept - HintFooterBytes : 0;
		auto         it            = buffer.begin() + static_cast<std::ptrdiff_t>(searched_from);
		while (true) {
			it = std::search(it, buffer.end(), searcher);
			if (it == buffer.end()) break;
			const auto pos = static_cast<size_t>(it - buffer.begin());
			if (pos + HintFooterBytes <= buffer.size()) {
				const uint64_t footer = XXH3_64bits(buffer.data() + pos, HintFooterBytes) | 1u;
				fn(std::span<const char>(buffer.data(), pos + HintFooterBytes), footer);
			} else {
				break; // cut by the end: searched again with the next chunk
			}
			++it;
		}
		// Keep the tail for blocks that started in this chunk and footers cut by its end.
		kept = std::min(buffer.size(), max_bytes + HintFooterBytes);
		std::memmove(buffer.data(), buffer.data() + buffer.size() - kept, kept);
		buffer.resize(kept);
	}
	return true;
}

} // namespace

bool PrecompileHintsExport(const std::filesystem::path& journal, const std::filesystem::path& graphics_journal,
                           const std::filesystem::path& out, std::string& summary) {
	std::vector<std::vector<uint32_t>> compute;
	std::vector<std::vector<uint32_t>> graphics;
	const bool                         has_compute =
	    ReadJournalPayloads(journal, JournalMagic, JournalVersion, SchemaFingerprint(), HeaderBytes, compute);
	uint64_t   device       = 0;
	const bool has_graphics = ReadJournalPayloads(graphics_journal, GraphicsJournalMagic, GraphicsJournalVersion,
	                                              GraphicsSchemaFingerprint(), GraphicsHeaderBytes, graphics, &device);
	if (!has_compute && !has_graphics) {
		summary = "reason=no_journal";
		return false;
	}
	std::string title;
	std::string version;
	::Loader::SystemContentParamSfoGetString("TITLE_ID", &title);
	::Loader::SystemContentParamSfoGetString("APP_VER", &version);
	auto temp = out;
	temp += ".tmp";
	std::FILE* f = std::fopen(temp.string().c_str(), "wb");
	if (f == nullptr) {
		summary = "reason=cannot_write";
		return false;
	}
	const auto file_version = HintsVersion;
	const auto schema       = SchemaFingerprint();
	const auto gschema      = GraphicsSchemaFingerprint();
	std::fwrite(HintsMagic, 1, sizeof(HintsMagic), f);
	std::fwrite(&file_version, sizeof(file_version), 1, f);
	std::fwrite(&schema, sizeof(schema), 1, f);
	std::fwrite(&gschema, sizeof(gschema), 1, f);
	{
		Writer w;
		w.U32(HintInfoMagic);
		WriteText(w, title);
		WriteText(w, version);
		WriteText(w, JournalSuffix(journal));
		WritePayload(f, w.words);
	}
	size_t compute_hints = 0;
	size_t stage_hints   = 0;
	size_t pipelines     = 0;
	size_t no_footer     = 0;
	const auto hint = [&](PrecompileStageRecord& record, uint32_t magic) {
		if (HintFooterHash(record.code) == 0 || (!record.back_code.empty() && HintFooterHash(record.back_code) == 0)) {
			no_footer++;
			return false;
		}
		Writer w;
		w.U32(magic);
		WriteHintCode(w, record.code);
		WriteHintCode(w, record.back_code);
		record.code.clear();
		record.back_code.clear();
		std::ranges::fill(record.user_data, 0u);
		const auto nested = magic == HintComputeMagic ? Serialize(record) : SerializeStage(record);
		w.words.insert(w.words.end(), nested.begin(), nested.end());
		WritePayload(f, w.words);
		return true;
	};
	for (const auto& payload: compute) {
		PrecompileStageRecord record;
		if (Deserialize(payload, record) && hint(record, HintComputeMagic)) compute_hints++;
	}
	// The stages first, then only the pipelines whose stages all went into the file: a pipeline recorded under a stage
	// key no stage record has (an older build's) is dropped by every load (PrecompileJournal: "pipelines without a stage
	// record"), so it is left out (28.10: 1,704 of 7,301 recorded pipelines were such).
	std::unordered_set<uint64_t> stage_keys;
	for (const auto& payload: graphics) {
		if (payload[0] != StageRecordMagic) continue;
		PrecompileStageRecord record;
		if (!DeserializeStage(payload, record)) continue;
		const auto key = record.Key();
		if (hint(record, HintStageMagic)) {
			stage_hints++;
			stage_keys.insert(key);
		}
	}
	size_t dead = 0;
	for (const auto& payload: graphics) {
		if (payload[0] == StageRecordMagic) continue;
		if (PrecompileGraphicsRecord record; DeserializePipeline(payload, record)) {
			bool resolved = true;
			ForEachStage(record, [&](uint64_t stage) { resolved = resolved && stage_keys.contains(stage); });
			if (!resolved) {
				dead++;
				continue;
			}
			WritePayload(f, payload);
			pipelines++;
		}
	}
	const bool written = std::fflush(f) == 0 && std::ferror(f) == 0;
	std::fclose(f);
	std::error_code ec;
	if (written) std::filesystem::rename(temp, out, ec);
	if (!written || ec) {
		summary = "reason=cannot_write";
		return false;
	}
	summary = fmt::format("compute={} stages={} pipelines={} dead_pipelines={} no_footer={} title={} version={}",
	                      compute_hints, stage_hints, pipelines, dead, no_footer, title.empty() ? "-" : title,
	                      version.empty() ? "-" : version);
	return true;
}

bool PrecompileHintsImport(const GraphicContext& graphics, const std::filesystem::path& hints,
                           const std::filesystem::path& game_dir, const std::filesystem::path& journal,
                           const std::filesystem::path& graphics_journal, std::string& summary) {
	const auto bytes = ReadFileBytes(hints);
	if (bytes.size() < HintsHeaderBytes || std::memcmp(bytes.data(), HintsMagic, sizeof(HintsMagic)) != 0) {
		summary = "reason=not_a_hints_file";
		return false;
	}
	uint32_t version = 0;
	uint64_t schema  = 0;
	uint64_t gschema = 0;
	std::memcpy(&version, bytes.data() + sizeof(HintsMagic), sizeof(version));
	std::memcpy(&schema, bytes.data() + sizeof(HintsMagic) + sizeof(version), sizeof(schema));
	std::memcpy(&gschema, bytes.data() + sizeof(HintsMagic) + sizeof(version) + sizeof(schema), sizeof(gschema));
	if (version != HintsVersion || schema != SchemaFingerprint() || gschema != GraphicsSchemaFingerprint()) {
		summary = "reason=other_emulator_version";
		return false;
	}
	std::vector<uint32_t> words((bytes.size() - HintsHeaderBytes) / sizeof(uint32_t));
	std::memcpy(words.data(), bytes.data() + HintsHeaderBytes, words.size() * sizeof(uint32_t));

	struct Hint {
		PrecompileStageRecord record;
		HintCode              code;
		HintCode              back;
	};
	std::vector<Hint>                      stage_hints; // compute and graphics stages
	std::vector<std::vector<uint32_t>>     pipelines;
	std::string                            title;
	std::string                            game_version;
	std::string                            suffix;
	bool                                   info = false;
	ForEachPayload(words, [&](std::span<const uint32_t> payload, bool intact) {
		if (!intact) return;
		Reader r(payload);
		const auto magic = r.U32();
		if (magic == HintInfoMagic) {
			title        = ReadText(r);
			game_version = ReadText(r);
			suffix       = ReadText(r);
			info         = r.ok;
		} else if (magic == HintComputeMagic || magic == HintStageMagic) {
			Hint hint;
			hint.code = ReadHintCode(r);
			hint.back = ReadHintCode(r);
			if (!r.ok || hint.code.bytes == 0 || hint.code.bytes % 4u != 0 || hint.back.bytes % 4u != 0) return;
			const auto nested = payload.subspan(1u + 2u * 5u);
			const bool ok     = magic == HintComputeMagic ? Deserialize(nested, hint.record, true)
			                                              : DeserializeStage(nested, hint.record, true);
			if (ok) stage_hints.push_back(std::move(hint));
		} else if (magic == PipelineRecordMagic) {
			pipelines.emplace_back(payload.begin(), payload.end());
		}
	});
	if (!info) {
		summary = "reason=not_a_hints_file";
		return false;
	}
	if (suffix != JournalSuffix(journal)) {
		summary = "reason=other_shader_mode";
		return false;
	}
	std::string own_title;
	std::string own_version;
	::Loader::SystemContentParamSfoGetString("TITLE_ID", &own_title);
	::Loader::SystemContentParamSfoGetString("APP_VER", &own_version);
	if (!title.empty() && !own_title.empty() && title != own_title) {
		summary = fmt::format("reason=other_game hints_title={} game_title={}", title, own_title);
		return false;
	}
	// A hints file of another game version finds few of its shaders and reads the whole game looking for the rest (the
	// 1.512 file on 2.002: ~185 GB, players thought it hung, 3.11): nothing is scanned.
	if (!game_version.empty() && !own_version.empty() && game_version != own_version) {
		summary = fmt::format("reason=other_version hints_version={} game_version={}", game_version, own_version);
		return false;
	}

	// The code blocks wanted: footer hash -> (size, block hash).
	std::unordered_multimap<uint64_t, std::pair<uint32_t, uint64_t>> wanted;
	std::unordered_map<uint64_t, std::vector<uint32_t>>               found; // block hash -> code
	size_t                                                            max_bytes = 0;
	const auto want = [&](const HintCode& c) {
		if (c.bytes == 0) return;
		wanted.emplace(c.footer, std::pair {c.bytes, c.hash});
		max_bytes = std::max<size_t>(max_bytes, c.bytes);
	};
	for (const auto& hint: stage_hints) {
		want(hint.code);
		want(hint.back);
	}
	std::unordered_set<uint64_t> distinct_blocks;
	for (const auto& [footer, need]: wanted) distinct_blocks.insert(need.second);
	const auto begin = std::chrono::steady_clock::now();
	size_t     files = 0;
	uint64_t   read  = 0;
	std::error_code ec;
	// eboot.bin, then the shader packs (.xpps; links followed: a game folder can link its parts); done once every block is
	// found. Only those files: every block found in 1.512 and 2.002 was in them (22-25 GB of shader packs), and a hint
	// whose code the game makes at run time sent the scan through the other ~180 GB (30 min on 2.002, 3.11.2026).
	// A game run from a .zar archive: its members are listed and read through the archive (the handle keeps the archive's
	// index and block cache for the whole scan).
	const bool archive        = Common::IsArchivePath(game_dir);
	const auto archive_handle = archive ? Common::OpenArchive(game_dir) : nullptr;
	std::vector<std::filesystem::path> paths {game_dir / "eboot.bin"};
	if (archive) {
		std::vector<std::filesystem::path> dirs {game_dir};
		while (!dirs.empty()) {
			const auto dir = std::move(dirs.back());
			dirs.pop_back();
			for (const auto& entry: Common::GetArchiveDirEntries(dir)) {
				const auto member = dir / Common::PathFromUtf8(entry.name);
				if (!entry.is_file) {
					dirs.push_back(member);
				} else if (member.extension() == ".xpps") {
					paths.push_back(member);
				}
			}
		}
	} else {
		for (auto it = std::filesystem::recursive_directory_iterator(
		         game_dir, std::filesystem::directory_options::follow_directory_symlink, ec);
		     !ec && it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
			if (it->is_regular_file(ec) && it->path().extension() == ".xpps") paths.push_back(it->path());
		}
	}
	for (const auto& path: paths) {
		if (found.size() == distinct_blocks.size()) break;
		uint64_t size = 0;
		if (archive) {
			const auto info = Common::GetArchiveInfo(path);
			if (!info || !info->is_file) continue;
			size = info->size;
		} else {
			size = std::filesystem::file_size(path, ec);
			if (ec) continue;
		}
		files++;
		read += size;
		ForEachFooter(path, max_bytes, [&](std::span<const char> before, uint64_t footer) {
			const auto [first, last] = wanted.equal_range(footer);
			for (auto it = first; it != last; ++it) {
				const auto [block_bytes, block_hash] = it->second;
				if (block_bytes > before.size() || found.contains(block_hash)) continue;
				const char* start = before.data() + before.size() - block_bytes;
				if (XXH3_64bits(start, block_bytes) != block_hash) continue;
				std::vector<uint32_t> code(block_bytes / 4u);
				std::memcpy(code.data(), start, block_bytes);
				found.emplace(block_hash, std::move(code));
			}
		});
	}
	const auto scan_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();

	// The records whose code was found (and still has the shader hash it was recorded with).
	const uint32_t subgroup = graphics.SupportsComputeWave64() ? 64u : 32u;
	std::vector<PrecompileStageRecord> compute_records;
	std::vector<PrecompileStageRecord> stage_records;
	size_t                             missing = 0;
	for (auto& hint: stage_hints) {
		const auto code = found.find(hint.code.hash);
		const auto back = hint.back.bytes == 0 ? found.end() : found.find(hint.back.hash);
		if (code == found.end() || (hint.back.bytes != 0 && back == found.end())) {
			missing++;
			continue;
		}
		auto& record = hint.record;
		record.code  = code->second;
		if (back != found.end()) record.back_code = back->second;
		if (PrecompileCodeHash(record.code, record.back_code) != record.shader_hash) {
			missing++;
			continue;
		}
		if (record.stage == ShaderType::Compute) {
			// The host's subgroup size, as the program lookup sets it (GetComputeProgram).
			record.compute.host_subgroup_size = subgroup;
			BuildStageStaticKey(record.compute, record.static_key);
			compute_records.push_back(std::move(record));
		} else {
			stage_records.push_back(std::move(record));
		}
	}

	// Added to the journals: the records they do not hold yet. A journal of another layout or another GPU is kept
	// beside as <name>.old and started again.
	const auto keep_old = [&](const std::filesystem::path& path) {
		if (!std::filesystem::exists(path, ec)) return;
		auto old = path;
		old += ".old";
		std::filesystem::rename(path, old, ec);
	};
	size_t added_compute   = 0;
	size_t added_stages    = 0;
	size_t added_pipelines = 0;
	{
		std::vector<std::vector<uint32_t>> existing;
		const bool ok = ReadJournalPayloads(journal, JournalMagic, JournalVersion, SchemaFingerprint(), HeaderBytes, existing);
		std::unordered_set<uint64_t> keys;
		for (const auto& payload: existing) {
			if (PrecompileStageRecord record; Deserialize(payload, record)) keys.insert(record.Key());
		}
		if (!ok) {
			keep_old(journal);
			existing.clear();
		}
		std::filesystem::create_directories(journal.parent_path(), ec);
		auto temp = journal;
		temp += ".tmp";
		if (std::FILE* f = std::fopen(temp.string().c_str(), "wb")) {
			const auto file_version = JournalVersion;
			const auto fingerprint  = SchemaFingerprint();
			std::fwrite(JournalMagic, 1, sizeof(JournalMagic), f);
			std::fwrite(&file_version, sizeof(file_version), 1, f);
			std::fwrite(&fingerprint, sizeof(fingerprint), 1, f);
			for (const auto& payload: existing) WritePayload(f, payload);
			for (const auto& record: compute_records) {
				if (!keys.insert(record.Key()).second) continue;
				WritePayload(f, Serialize(record));
				added_compute++;
			}
			const bool written = std::fflush(f) == 0 && std::ferror(f) == 0;
			std::fclose(f);
			if (written) std::filesystem::rename(temp, journal, ec);
		}
	}
	{
		std::vector<std::vector<uint32_t>> existing;
		uint64_t                           device = 0;
		const bool ok = ReadJournalPayloads(graphics_journal, GraphicsJournalMagic, GraphicsJournalVersion,
		                                    GraphicsSchemaFingerprint(), GraphicsHeaderBytes, existing, &device) &&
		                device == DeviceFingerprint(graphics);
		if (!ok) {
			keep_old(graphics_journal);
			existing.clear();
		}
		std::unordered_set<uint64_t> keys;
		std::vector<std::vector<uint32_t>> existing_stages;
		std::vector<std::vector<uint32_t>> existing_pipelines;
		for (auto& payload: existing) {
			if (PrecompileStageRecord record; payload[0] == StageRecordMagic && DeserializeStage(payload, record)) {
				keys.insert(record.Key());
				existing_stages.push_back(std::move(payload));
			} else if (PrecompileGraphicsRecord pipeline; DeserializePipeline(payload, pipeline)) {
				keys.insert(pipeline.Key());
				existing_pipelines.push_back(std::move(payload));
			}
		}
		std::filesystem::create_directories(graphics_journal.parent_path(), ec);
		auto temp = graphics_journal;
		temp += ".tmp";
		if (std::FILE* f = std::fopen(temp.string().c_str(), "wb")) {
			const auto file_version = GraphicsJournalVersion;
			const auto fingerprint  = GraphicsSchemaFingerprint();
			const auto own_device   = DeviceFingerprint(graphics);
			std::fwrite(GraphicsJournalMagic, 1, sizeof(GraphicsJournalMagic), f);
			std::fwrite(&file_version, sizeof(file_version), 1, f);
			std::fwrite(&fingerprint, sizeof(fingerprint), 1, f);
			std::fwrite(&own_device, sizeof(own_device), 1, f);
			// The stage records before the pipeline records (as WriteGraphicsJournalLocked).
			for (const auto& payload: existing_stages) WritePayload(f, payload);
			for (const auto& record: stage_records) {
				if (!keys.insert(record.Key()).second) continue;
				WritePayload(f, SerializeStage(record));
				added_stages++;
			}
			for (const auto& payload: existing_pipelines) WritePayload(f, payload);
			for (const auto& payload: pipelines) {
				PrecompileGraphicsRecord pipeline;
				if (!DeserializePipeline(payload, pipeline) || !keys.insert(pipeline.Key()).second) continue;
				WritePayload(f, payload);
				added_pipelines++;
			}
			const bool written = std::fflush(f) == 0 && std::ferror(f) == 0;
			std::fclose(f);
			if (written) std::filesystem::rename(temp, graphics_journal, ec);
		}
	}
	summary = fmt::format("hints={} found={} missing={} added_compute={} added_stages={} added_pipelines={} files={} "
	                      "read_mb={} scan_s={:.1f} hints_version={} game_version={}",
	                      stage_hints.size(), stage_hints.size() - missing, missing, added_compute, added_stages,
	                      added_pipelines, files, read >> 20u, scan_s, game_version.empty() ? "-" : game_version,
	                      own_version.empty() ? "-" : own_version);
	return true;
}

void SetStageCompileOptions(ShaderRecompiler::CompileOptions& options) {
	switch (options.stage) {
		case ShaderType::Vertex:
		case ShaderType::Local:
		case ShaderType::TessellationEvaluation:
			options.user_data_base = 8;
			options.wave_size      = options.input_info.vertex->wave_size;
			break;
		case ShaderType::Mesh:
			options.user_data_base = 0;
			options.wave_size      = options.input_info.vertex->mesh.wave_size;
			break;
		case ShaderType::TessellationControl:
			options.user_data_base = 0;
			options.wave_size      = 64u;
			break;
		case ShaderType::Pixel: options.wave_size = options.input_info.pixel->wave_size; break;
		case ShaderType::Compute: options.wave_size = options.input_info.compute->wave_size; break;
		default: break;
	}
}

bool ShaderPrecompiler::Enabled() {
	static const bool enabled = [] {
		// Default from Common::EnvDefaults; =0 turns it off.
		const char* value = Common::EnvOrDefault("KYTY_SHADER_PRECOMPILE");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

bool ShaderPrecompiler::GraphicsEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SHADER_PRECOMPILE_GRAPHICS");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

uint64_t ShaderPrecompiler::Revision() {
	static const uint64_t revision = [] {
		const std::string_view hash = KYTY_GIT_HASH;
		return XXH3_64bits(hash.data(), hash.size());
	}();
	return revision;
}

ShaderPrecompiler::ShaderPrecompiler(GraphicContext& graphics, vk::PipelineCache driver_cache,
                                     std::filesystem::path journal, std::filesystem::path graphics_journal,
                                     std::atomic<uint32_t>& created, PrecompileWarmList* warm)
    : m_graphics(graphics), m_driver_cache(driver_cache), m_journal_path(std::move(journal)),
      m_graphics_journal_path(std::move(graphics_journal)), m_created(created), m_warm(warm) {
	// One in-flight list and one suspect list for the compute and the graphics keys.
	m_inflight_path = m_journal_path;
	m_inflight_path += ".inflight";
	m_suspects_path = m_journal_path;
	m_suspects_path += ".suspects";
	// The ENV_FINGERPRINT line in every run with the precompiler, with the warm list or without it.
	(void)EnvironmentFingerprint();
	const auto        skipped = LoadSuspects();
	std::vector<Job>  compute_jobs;
	std::vector<Job>  graphics_jobs;
	std::vector<Job>  stage_jobs;
	LoadCompute(skipped, compute_jobs);
	if (GraphicsEnabled()) {
		LoadGraphics(skipped, graphics_jobs, stage_jobs);
	}
	// The null share journal: the records are fixed now (m_stages is node-based).
	for (const auto& record: m_compute_records) {
		m_program_records[ProgramKey(RecordIdentity(record))].push_back(&record);
	}
	for (const auto key: m_stage_order) {
		const auto& record = m_stages.at(key).record;
		m_program_records[ProgramKey(RecordIdentity(record))].push_back(&record);
	}
	// Both kinds in proportion, each in its journal's order (the order an earlier run first needed them in): the
	// job whose position in its own list is earlier goes first.
	size_t compute = 0;
	size_t graphics_job = 0;
	while (compute < compute_jobs.size() || graphics_job < graphics_jobs.size()) {
		const bool take_compute = graphics_job >= graphics_jobs.size() ||
		                          (compute < compute_jobs.size() &&
		                           compute * graphics_jobs.size() <= graphics_job * compute_jobs.size());
		const auto& job = take_compute ? compute_jobs[compute++] : graphics_jobs[graphics_job++];
		m_job_index.emplace(job.key, m_jobs.size());
		m_jobs.push_back(job);
	}
	// The stage-only jobs: after the jobs of the recorded pipelines (their gain is surer). Not in m_job_index: the
	// GPU thread waits (BeforeCompile) for pipeline and compute jobs only; a stage it needs first it compiles itself.
	m_jobs.insert(m_jobs.end(), stage_jobs.begin(), stage_jobs.end());
	// std::quick_exit (the normal quit) runs no destructors: the handler marks the in-flight jobs as interrupted
	// by a quit, not by a crash.
	g_quick_exit_target = this;
	std::call_once(g_quick_exit_registered, [] { std::at_quick_exit(OnQuickExit); });
}

void ShaderPrecompiler::OnQuickExit() {
	if (ShaderPrecompiler* self = g_quick_exit_target; self != nullptr) {
		std::lock_guard lock(self->m_mutex);
		self->m_quitting = true;
		self->m_inflight.clear();
		self->WriteInflightLocked();
	}
}

void ShaderPrecompiler::Start(uint32_t requested_threads) {
	if (m_started) return;
	m_started        = true;
	m_start          = std::chrono::steady_clock::now();
	uint32_t threads = 4;
	if (requested_threads != 0) {
		// --precompile-threads (the launcher's precompile run).
		threads = std::clamp(requested_threads, 1u, 64u);
	} else if (const char* value = Common::EnvOrDefault("KYTY_SHADER_PRECOMPILE_THREADS"); value != nullptr) {
		// Default from Common::EnvDefaults: 6 threads; 4 with KYTY_ENV_DEFAULTS=0.
		threads = static_cast<uint32_t>(std::clamp(std::atoi(value), 0, 16));
	}
	// Only the jobs left to compile need workers (suspects and pipelines already in the driver cache are dropped).
	const auto queued = static_cast<uint32_t>(
	    std::ranges::count_if(m_jobs, [](const Job& job) { return job.state == JobState::Queued; }));
	const auto queued_graphics = static_cast<uint32_t>(std::ranges::count_if(
	    m_jobs, [](const Job& job) { return job.state == JobState::Queued && job.graphics; }));
	g_progress.queued_graphics.store(queued_graphics, std::memory_order_relaxed);
	threads = std::min(threads, queued);
	std::fprintf(stderr, "SHADER_PRECOMPILE journal=%s records=%zu queued=%u (compute %zu, graphics pipelines %zu of "
	                     "%zu stages) threads=%u\n",
	             m_journal_path.string().c_str(), m_jobs.size(), queued, m_compute_records.size(),
	             m_graphics_records.size(), m_stages.size(), threads);
	std::fflush(stderr);
	m_thread_count = threads;
	g_progress.queued.store(queued, std::memory_order_relaxed);
	g_progress.threads.store(threads, std::memory_order_relaxed);
	g_progress.done.store(threads == 0, std::memory_order_relaxed);
	g_progress.started.store(true, std::memory_order_release);
	for (uint32_t i = 0; i < threads; i++) {
		m_workers.emplace_back([this, i] { Worker(i); });
	}
}

void ShaderPrecompiler::RequestStop() {
	{
		std::lock_guard lock(m_mutex);
		m_stop = true;
	}
	m_done.notify_all();
}

void ShaderPrecompiler::Stop() {
	RequestStop();
	for (auto& worker: m_workers) {
		if (worker.joinable()) worker.join();
	}
}

bool ShaderPrecompiler::Busy() const {
	return m_started && m_finished.load() < m_thread_count;
}

ShaderPrecompiler::~ShaderPrecompiler() {
	Stop();
	std::vector<vk::ShaderModule> modules;
	std::lock_guard               lock(m_mutex);
	g_quick_exit_target = nullptr;
	if (m_journal != nullptr) {
		std::fclose(m_journal);
		m_journal = nullptr;
	}
	if (m_graphics_journal != nullptr) {
		std::fclose(m_graphics_journal);
		m_graphics_journal = nullptr;
	}
	m_inflight.clear();
	WriteInflightLocked();
	// The workers are gone: stage modules the last of them did not destroy.
	CollectModulesLocked(modules);
	DestroyModules(modules);
}

std::unordered_set<uint64_t> ShaderPrecompiler::LoadSuspects() {
	// Suspects: keys that were compiling when a run of this build ended without a clean quit (a crash in a
	// background compile, or one the GPU thread then hit). They stay skipped until the emulator build changes.
	const auto revision = Revision();
	std::vector<std::pair<uint64_t, uint64_t>> suspects;
	for (const auto& line: ReadKeyLines(m_suspects_path)) {
		if (line.first == revision) suspects.push_back(line);
	}
	for (const auto& line: ReadKeyLines(m_inflight_path)) {
		if (line.first == revision && std::ranges::find(suspects, line) == suspects.end()) suspects.push_back(line);
	}
	if (std::FILE* f = std::fopen(m_suspects_path.string().c_str(), "wb")) {
		for (const auto& [line_revision, key]: suspects) {
			std::fprintf(f, "%s %s\n", HexKey(line_revision).c_str(), HexKey(key).c_str());
		}
		std::fclose(f);
	}
	std::unordered_set<uint64_t> skipped;
	for (const auto& line: suspects) skipped.insert(line.second);
	return skipped;
}

void ShaderPrecompiler::LoadCompute(const std::unordered_set<uint64_t>& skipped, std::vector<Job>& jobs) {
	const auto bytes     = ReadFileBytes(m_journal_path);
	bool       header_ok = false;
	if (bytes.size() >= HeaderBytes) {
		uint32_t version     = 0;
		uint64_t fingerprint = 0;
		std::memcpy(&version, bytes.data() + sizeof(JournalMagic), sizeof(version));
		std::memcpy(&fingerprint, bytes.data() + sizeof(JournalMagic) + sizeof(version), sizeof(fingerprint));
		header_ok = std::memcmp(bytes.data(), JournalMagic, sizeof(JournalMagic)) == 0 &&
		            version == JournalVersion && fingerprint == SchemaFingerprint();
	}
	// A torn tail (a run killed while appending) ends the list; the file is then rewritten from the records that
	// loaded.
	bool                  torn     = header_ok && (bytes.size() - HeaderBytes) % sizeof(uint32_t) != 0;
	size_t                rejected = 0;
	size_t                suspects = 0;
	size_t                warm     = 0;
	std::vector<uint32_t> words;
	if (header_ok) {
		words.resize((bytes.size() - HeaderBytes) / sizeof(uint32_t));
		std::memcpy(words.data(), bytes.data() + HeaderBytes, words.size() * sizeof(uint32_t));
	}
	const bool complete = ForEachPayload(words, [&](std::span<const uint32_t> payload, bool intact) {
		PrecompileStageRecord record;
		if (!intact || !Deserialize(payload, record)) {
			rejected++;
			return;
		}
		Job job;
		job.key = record.Key();
		if (!m_recorded.insert(job.key).second) return;
		job.record = static_cast<uint32_t>(m_compute_records.size());
		if (skipped.contains(job.key)) {
			job.state = JobState::Dropped;
			suspects++;
		} else if (m_warm != nullptr && m_warm->Contains(job.key, false)) {
			// Its pipeline is in the saved driver cache already: a compile would only hit it. the shader reuse:
			// it is still translated and compiled for the reuse store, without a pipeline.
			if (ShaderReuseEnabled()) {
				job.warm = true;
			} else {
				job.state = JobState::Dropped;
			}
			warm++;
		}
		jobs.push_back(job);
		m_compute_records.push_back(std::move(record));
	});
	torn = torn || !complete;
	// A file whose header does not match (another record layout or version): none of its records is read.
	const bool other_layout = !bytes.empty() && !header_ok;
	if (other_layout) g_progress.other_layout_journals.fetch_add(1, std::memory_order_relaxed);
	g_progress.warm.fetch_add(static_cast<uint32_t>(warm), std::memory_order_relaxed);
	g_progress.suspects.fetch_add(static_cast<uint32_t>(suspects), std::memory_order_relaxed);
	std::fprintf(stderr,
	             "SHADER_PRECOMPILE journal: %zu records, %zu suspects skipped for this build, %zu already in the "
	             "driver cache%s%s%s%s\n",
	             m_compute_records.size(), suspects, warm,
	             warm != 0 && ShaderReuseEnabled() ? " (compiled for the reuse store, no pipeline)" : "",
	             rejected != 0 ? ", some records rejected" : "", torn ? ", torn tail dropped" : "",
	             other_layout ? ", the file has another record layout or version (its records dropped)" : "");

	// --precompile-only records nothing: the file is left as it is (one of another layout stays for the build that
	// wrote it; a torn tail is repaired by the next game start).
	if (Config::PrecompileOnly()) return;
	std::lock_guard lock(m_mutex);
	if (!header_ok || torn || rejected != 0) {
		WriteJournalLocked();
	} else {
		m_journal = std::fopen(m_journal_path.string().c_str(), "ab");
	}
	if (m_journal == nullptr) {
		std::fprintf(stderr, "SHADER_PRECOMPILE journal: cannot open %s for writing\n",
		             m_journal_path.string().c_str());
	}
}

void ShaderPrecompiler::LoadGraphics(const std::unordered_set<uint64_t>& skipped, std::vector<Job>& jobs,
                                     std::vector<Job>& stage_jobs) {
	const auto bytes        = ReadFileBytes(m_graphics_journal_path);
	bool       header_ok    = false;
	bool       other_device = false;
	if (bytes.size() >= GraphicsHeaderBytes) {
		uint32_t version     = 0;
		uint64_t fingerprint = 0;
		uint64_t device      = 0;
		auto     offset      = sizeof(GraphicsJournalMagic);
		std::memcpy(&version, bytes.data() + offset, sizeof(version));
		offset += sizeof(version);
		std::memcpy(&fingerprint, bytes.data() + offset, sizeof(fingerprint));
		offset += sizeof(fingerprint);
		std::memcpy(&device, bytes.data() + offset, sizeof(device));
		header_ok = std::memcmp(bytes.data(), GraphicsJournalMagic, sizeof(GraphicsJournalMagic)) == 0 &&
		            version == GraphicsJournalVersion && fingerprint == GraphicsSchemaFingerprint();
		// Another GPU's records: dropped (the file starts again for this one).
		other_device = header_ok && device != DeviceFingerprint(m_graphics);
		header_ok    = header_ok && !other_device;
	}
	bool                  torn     = header_ok && (bytes.size() - GraphicsHeaderBytes) % sizeof(uint32_t) != 0;
	size_t                rejected = 0;
	std::vector<uint32_t> words;
	if (header_ok) {
		words.resize((bytes.size() - GraphicsHeaderBytes) / sizeof(uint32_t));
		std::memcpy(words.data(), bytes.data() + GraphicsHeaderBytes, words.size() * sizeof(uint32_t));
	}
	std::vector<PrecompileGraphicsRecord> pipelines;
	const bool complete = ForEachPayload(words, [&](std::span<const uint32_t> payload, bool intact) {
		if (!intact) {
			rejected++;
		} else if (payload[0] == StageRecordMagic) {
			PrecompileStageRecord record;
			if (!DeserializeStage(payload, record)) {
				rejected++;
				return;
			}
			const auto key = record.Key();
			if (!m_recorded_graphics.insert(key).second) return;
			if (!TableKeyAsked(record)) return; // recorded with another table mode: no lookup of this run asks for it
			m_stage_order.push_back(key);
			m_stages[key].record = std::move(record);
		} else if (PrecompileGraphicsRecord record; DeserializePipeline(payload, record)) {
			pipelines.push_back(record);
		} else {
			rejected++;
		}
	});
	torn = torn || !complete;
	// The pipelines whose stages were all recorded (a stage record can be missing when its code had changed); one
	// with a suspect key or stage is skipped for this build.
	size_t unresolved = 0;
	size_t suspects   = 0;
	size_t warm       = 0;
	for (auto& record: pipelines) {
		const auto key      = record.Key();
		bool       resolved = true;
		bool       suspect  = skipped.contains(key);
		ForEachStage(record, [&](uint64_t stage) {
			resolved = resolved && m_stages.contains(stage);
			suspect  = suspect || skipped.contains(stage);
		});
		if (!resolved) {
			unresolved++;
			continue;
		}
		if (!m_recorded_graphics.insert(key).second) continue;
		Job job;
		job.key      = key;
		job.record   = static_cast<uint32_t>(m_graphics_records.size());
		job.graphics = true;
		if (suspect) {
			job.state = JobState::Dropped;
			suspects++;
		} else if (m_warm != nullptr && m_warm->Contains(key, true) && !ShaderReuseEnabled()) {
			// Its pipeline is in the saved driver cache already: its stages need no compile for it.
			job.state = JobState::Dropped;
			warm++;
		} else {
			// The shader reuse: a warm job compiles its stages for the reuse store and makes no pipeline.
			if (m_warm != nullptr && m_warm->Contains(key, true)) {
				job.warm = true;
				warm++;
			}
			ForEachStage(record, [&](uint64_t stage) { m_stages[stage].users++; });
		}
		jobs.push_back(job);
		m_graphics_records.push_back(record);
	}
	// The stages no queued job uses (no recorded pipeline names them: a table-mode pipeline no draw took, or one recorded
	// under an older stage key); the shader reuse compiles them for the store, in the journal's order
	// (village house: dropped misses 351 -> 0, =2 0 mismatches of 1963).
	size_t orphans = 0;
	for (const auto key: m_stage_order) {
		auto& stage = m_stages.at(key);
		if (stage.users != 0) continue;
		orphans++;
		if (!ShaderReuseEnabled() || skipped.contains(key)) continue;
		stage.users = 1; // the job's (a miss while it is queued is pending)
		Job job;
		job.key        = key;
		job.graphics   = true;
		job.stage_only = true;
		stage_jobs.push_back(job);
	}
	g_reuse.stage_only_queued.fetch_add(stage_jobs.size(), std::memory_order_relaxed);
	// A file whose header does not match (another record layout or version): none of its records is read.
	const bool other_layout = !bytes.empty() && !header_ok && !other_device;
	if (other_layout) g_progress.other_layout_journals.fetch_add(1, std::memory_order_relaxed);
	g_progress.warm.fetch_add(static_cast<uint32_t>(warm), std::memory_order_relaxed);
	g_progress.suspects.fetch_add(static_cast<uint32_t>(suspects), std::memory_order_relaxed);
	std::fprintf(stderr,
	             "SHADER_PRECOMPILE graphics journal: %zu pipelines, %zu stages, %zu suspects skipped for this build, %zu "
	             "already in the driver cache, %zu pipelines without a stage record, %zu stages without a pipeline "
	             "(%zu stage-only jobs)%s%s%s%s%s%s\n",
	             m_graphics_records.size(), m_stages.size(), suspects, warm, unresolved, orphans, stage_jobs.size(),
	             warm != 0 && ShaderReuseEnabled() ? " (stages compiled for the reuse store, no pipeline)" : "",
	             other_device ? ", another GPU's records dropped" : "", rejected != 0 ? ", some records rejected" : "",
	             unresolved != 0 ? ", pipelines without their stages dropped" : "", torn ? ", torn tail dropped" : "",
	             other_layout ? ", the file has another record layout or version (its records dropped)" : "");

	// --precompile-only records nothing: the file is left as it is (as the compute journal).
	if (Config::PrecompileOnly()) return;
	std::lock_guard lock(m_mutex);
	if (!header_ok || torn || rejected != 0 || unresolved != 0) {
		WriteGraphicsJournalLocked();
	} else {
		m_graphics_journal = std::fopen(m_graphics_journal_path.string().c_str(), "ab");
	}
	if (m_graphics_journal == nullptr) {
		std::fprintf(stderr, "SHADER_PRECOMPILE graphics journal: cannot open %s for writing\n",
		             m_graphics_journal_path.string().c_str());
	}
}

// Rewrites the journal from the loaded records into a temporary file that replaces it, then appends there.
void ShaderPrecompiler::WriteJournalLocked() {
	std::error_code ec;
	std::filesystem::create_directories(m_journal_path.parent_path(), ec);
	auto temp_path = m_journal_path;
	temp_path += ".tmp";
	if (std::FILE* f = std::fopen(temp_path.string().c_str(), "wb")) {
		const auto version     = JournalVersion;
		const auto fingerprint = SchemaFingerprint();
		std::fwrite(JournalMagic, 1, sizeof(JournalMagic), f);
		std::fwrite(&version, sizeof(version), 1, f);
		std::fwrite(&fingerprint, sizeof(fingerprint), 1, f);
		for (const auto& record: m_compute_records) {
			WritePayload(f, Serialize(record));
		}
		const bool written = std::fflush(f) == 0 && std::ferror(f) == 0;
		std::fclose(f);
		if (written) {
			std::filesystem::rename(temp_path, m_journal_path, ec);
		}
	}
	m_journal = std::fopen(m_journal_path.string().c_str(), "ab");
}

// The same for the graphics journal (this device's header, the stage records, then the pipeline records).
void ShaderPrecompiler::WriteGraphicsJournalLocked() {
	std::error_code ec;
	std::filesystem::create_directories(m_graphics_journal_path.parent_path(), ec);
	auto temp_path = m_graphics_journal_path;
	temp_path += ".tmp";
	if (std::FILE* f = std::fopen(temp_path.string().c_str(), "wb")) {
		const auto version     = GraphicsJournalVersion;
		const auto fingerprint = GraphicsSchemaFingerprint();
		const auto device      = DeviceFingerprint(m_graphics);
		std::fwrite(GraphicsJournalMagic, 1, sizeof(GraphicsJournalMagic), f);
		std::fwrite(&version, sizeof(version), 1, f);
		std::fwrite(&fingerprint, sizeof(fingerprint), 1, f);
		std::fwrite(&device, sizeof(device), 1, f);
		for (const auto key: m_stage_order) {
			WritePayload(f, SerializeStage(m_stages.at(key).record));
		}
		for (const auto& record: m_graphics_records) {
			WritePayload(f, SerializePipeline(record));
		}
		const bool written = std::fflush(f) == 0 && std::ferror(f) == 0;
		std::fclose(f);
		if (written) {
			std::filesystem::rename(temp_path, m_graphics_journal_path, ec);
		}
	}
	m_graphics_journal = std::fopen(m_graphics_journal_path.string().c_str(), "ab");
}

void ShaderPrecompiler::WriteInflightLocked() {
	if (std::FILE* f = std::fopen(m_inflight_path.string().c_str(), "wb")) {
		const auto revision = HexKey(Revision());
		for (const auto key: m_inflight) {
			std::fprintf(f, "%s %s\n", revision.c_str(), HexKey(key).c_str());
		}
		std::fclose(f);
	}
}

void ShaderPrecompiler::Worker(uint32_t index) {
	g_background_thread = true;
	Common::Thread::SetCurrentName(fmt::format("Thread_ShaderPrecompile{}", index).c_str());
	std::vector<vk::ShaderModule> unused;
	for (;;) {
		Job* job = nullptr;
		{
			std::lock_guard lock(m_mutex);
			// Modules the jobs the GPU thread dropped released.
			unused.swap(m_unused_modules);
			while (!m_stop && m_next < m_jobs.size() && m_jobs[m_next].state != JobState::Queued) {
				m_next++;
			}
			if (!m_stop && !m_quitting && m_next < m_jobs.size()) {
				job         = &m_jobs[m_next++];
				job->state  = JobState::Running;
				job->worker = index;
				m_inflight.insert(job->key);
				WriteInflightLocked();
			}
		}
		DestroyModules(unused);
		if (job == nullptr) break;
		// Below normal again (the GPU thread raises a worker it waits for).
		LowerCurrentThreadPriority();
		if (job->stage_only) {
			CompileStageOnly(job->key);
		} else if (job->graphics) {
			CompileGraphics(*job);
		} else {
			Compile(m_compute_records[job->record], job->key, job->warm);
		}
		{
			std::lock_guard lock(m_mutex);
			job->state = JobState::Done;
			m_inflight.erase(job->key);
			if (job->graphics && !job->stage_only) ReleaseStagesLocked(m_graphics_records[job->record], unused);
			if (!m_quitting) WriteInflightLocked();
		}
		g_progress.finished.fetch_add(1, std::memory_order_relaxed);
		if (job->graphics) g_progress.finished_graphics.fetch_add(1, std::memory_order_relaxed);
		m_done.notify_all();
		DestroyModules(unused);
	}
	if (m_finished.fetch_add(1) + 1 == m_thread_count) {
		// The last worker: no job runs any more, every stage module goes.
		{
			std::lock_guard lock(m_mutex);
			CollectModulesLocked(unused);
		}
		DestroyModules(unused);
		const auto seconds =
		    std::chrono::duration<double>(std::chrono::steady_clock::now() - m_start).count();
		g_progress.seconds.store(static_cast<uint32_t>(seconds + 0.5), std::memory_order_relaxed);
		g_progress.done.store(true, std::memory_order_release);
		std::fprintf(stderr,
		             "SHADER_PRECOMPILE done:%u pipelines compiled (%u compute, %u graphics; %u graphics skipped; %u records "
		             "rejected, made under other settings) in %.0f s\n",
		             m_compiled.load(), m_compiled.load() - m_compiled_graphics.load(), m_compiled_graphics.load(),
		             m_skipped_graphics.load(), m_rejected.load(), seconds);
		if (ShaderReuseEnabled()) {
			std::fprintf(stderr,
			             "SHADER_REUSE workers done: %llu results kept (%llu MiB held now, peak %llu MiB), %llu not kept "
			             "(store full; +%llu started full, %llu of %.0f ms or more, %llu ms of work), "
			             "%llu finished after the GPU thread compiled them; %llu taken so far\n",
			             static_cast<unsigned long long>(g_reuse.stored.load()),
			             static_cast<unsigned long long>(g_reuse.held_bytes.load() >> 20u),
			             static_cast<unsigned long long>(g_reuse.peak_bytes.load() >> 20u),
			             static_cast<unsigned long long>(g_reuse.full.load()),
			             static_cast<unsigned long long>(g_reuse.full_start.load()),
			             static_cast<unsigned long long>(g_reuse.full_big.load()), ReuseExpensiveMs,
			             static_cast<unsigned long long>(g_reuse.full_us.load() / 1000u),
			             static_cast<unsigned long long>(g_reuse.late.load()),
			             static_cast<unsigned long long>(g_reuse.found.load()));
		}
		std::fflush(stderr);
	}
}

// Translates and compiles a recorded permutation with the GPU thread's options (ProgramCache::Get): its module (when
// `need_module`) and compiled program. False when it gives no module. the shader reuse: the SPIR-V, the program
// info and (for the program's first use) the resource plan go to the store, unless the GPU thread compiled the
// permutation already.
bool ShaderPrecompiler::CompileModule(const PrecompileStageRecord& record, uint64_t key, bool need_module,
                                      vk::ShaderModule& module, ShaderRecompiler::IR::CompiledShaderInfo& info,
                                      size_t& spirv_words) {
	const Common::StartupTrace::Scope startup_scope(Common::StartupTrace::Counter::ShaderBackground);
	ShaderVertexInputInfo  vertex  = record.vertex;
	ShaderPixelInputInfo   pixel   = record.pixel;
	ShaderComputeInputInfo compute = record.compute;
	ShaderStageInputInfo   stage_input {};
	switch (record.stage) {
		case ShaderType::Pixel: stage_input.pixel = &pixel; break;
		case ShaderType::Compute: stage_input.compute = &compute; break;
		default: stage_input.vertex = &vertex; break;
	}
	ShaderRecompiler::CompileOptions options;
	options.stage       = record.stage;
	options.shader_hash = record.shader_hash;
	options.user_data   = record.user_data;
	options.back_code   = record.back_code;
	options.dump_ir     = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
	options.early_dump  = options.dump_ir;
	options.dump_label  = StageLabel(record.stage);
	options.input_info  = stage_input;
	SetStageCompileOptions(options);
	// The profile of the GPU thread's compiles (ProgramCache: maintenance8, AMD native min/max), so a module made here is the
	// one the GPU thread would make.
	static const auto profile = ShaderCompileProfile(m_graphics);
	options.profile           = &profile;

	const bool     reuse           = ShaderReuseEnabled();
	const bool     keep            = reuse && ReuseWanted(key);
	// Wanted but for the budget: not kept (counted, full_start).
	const bool     over_budget     = reuse && !keep && ReuseWanted(key, true);
	const uint64_t program         = keep ? ProgramKey(RecordIdentity(record)) : 0u;
	const auto     translate_begin = std::chrono::steady_clock::now();
	auto source = ShaderRecompiler::PrepareShaderSource(record.code, options);
	if (!source.calls.empty() || source.unsupported_call_pc != UINT32_MAX) {
		return false; // programs with a scalar call read their callee from guest memory: never recorded
	}
	auto translated = ShaderRecompiler::TranslateProgram(source.decoded, options);
	// A record whose specialization does not fit the resources this build and these switches give the program (a
	// switch that changes the resource layout, such as KYTY_PERF_BINDLESS_CLASSES, differed in the recording run)
	// would end the emulator in ApplyResourceSpecialization: such a job makes no module.
	{
		const auto& info  = translated.program.info;
		const auto& s     = record.specialization;
		bool        fits  = info.buffers.size() <= s.buffers.size() && info.images.size() <= s.images.size();
		for (size_t i = info.buffers.size(); fits && i < s.buffers.size(); i++)
			fits = s.buffers[i].indirect_root < info.buffers.size();
		for (size_t i = info.images.size(); fits && i < s.images.size(); i++)
			fits = s.images[i].indirect_root < info.images.size();
		// A record that selects among candidates for an image whose descriptor this translation does not plan as GPU-
		// selected (lane-selected descriptors differed in the recording run) cannot be emitted either.
		for (size_t i = 0; fits && i < info.images.size() && i < s.images.size(); i++) {
			if (s.images[i].indirect_root != i || s.images[i].bindless) continue;
			// The candidates beyond the root name it as their root.
			const bool selected = std::ranges::any_of(std::span(s.images).subspan(info.images.size()),
			                                          [&](const auto& image) { return image.indirect_root == i; });
			if (!selected) continue;
			const auto source = info.images[i].source;
			fits = source < translated.program.descriptor_sources.size() &&
			       translated.program.descriptor_sources[source].indirect_descriptor.has_value();
		}
		if (!fits) {
			m_rejected.fetch_add(1, std::memory_order_relaxed);
			static std::atomic_uint logged {0};
			if (logged.fetch_add(1) < 16u) {
				std::fprintf(stderr,
				             "SHADER_PRECOMPILE_LAYOUT_MISMATCH hash=0x%016llx stage=%s key=%s images=%zu/%zu buffers=%zu/%zu "
				             "(recorded with other resource-changing switches; skipped)\n",
				             static_cast<unsigned long long>(record.shader_hash), StageName(record.stage),
				             HexKey(record.Key()).c_str(), info.images.size(), s.images.size(), info.buffers.size(),
				             s.buffers.size());
				std::fflush(stderr);
			}
			return false;
		}
		// Nor one with an image conversion format the emitter cannot make under this run's format switches
		// (a format-30 record of a KYTY_FIX_RT_11_11_10_FLOAT=0 run ended every later run in ImageConversionFormat). The
		// images as ApplyResourceSpecialization makes them: the translation's fields (a candidate's are its root's) with
		// the record's conversion format. The record stays in the journal: a run with its settings makes it again.
		for (size_t i = 0; i < s.images.size(); i++) {
			const auto& image = info.images[i < info.images.size() ? i : s.images[i].indirect_root];
			if (ShaderRecompiler::Spirv::ImageConversionEmittable(image, s.images[i].conversion_format)) continue;
			m_rejected.fetch_add(1, std::memory_order_relaxed);
			static std::atomic_uint logged {0};
			if (logged.fetch_add(1) < 16u) {
				std::fprintf(stderr,
				             "SHADER_PRECOMPILE_FORMAT_MISMATCH hash=0x%016llx stage=%s key=%s image=%zu conversion=%u "
				             "(recorded with other format switches, e.g. KYTY_FIX_RT_11_11_10_FLOAT; skipped)\n",
				             static_cast<unsigned long long>(record.shader_hash), StageName(record.stage),
				             HexKey(record.Key()).c_str(), i, static_cast<uint32_t>(s.images[i].conversion_format));
				std::fflush(stderr);
			}
			return false;
		}
	}
	// The shader reuse: the plan as the GPU thread's first use of the program extracts it (from the same
	// translation, before CompileProgram); one per program.
	std::optional<ShaderRecompiler::IR::ResourcePlan> plan;
	if (keep && ReusePlanWanted(program)) {
		plan.emplace(ShaderRecompiler::IR::ExtractResourcePlan(translated.program));
	}
	const double translate_ms  = MillisecondsSince(translate_begin);
	const auto   compile_begin = std::chrono::steady_clock::now();
	auto result = ShaderRecompiler::CompileProgram(std::move(translated), options, record.specialization,
	                                               record.push_data_start);
	const double compile_ms = MillisecondsSince(compile_begin);
	spirv_words           = result.spirv.size();
	const auto spirv_hash = XXH3_64bits(result.spirv.data(), result.spirv.size() * sizeof(uint32_t));
	if (record.revision == Revision() && spirv_hash != record.spirv_hash) {
		// The same build made other SPIR-V from the recorded inputs: an input is missing from the record (or a
		// SPIR-V-changing KYTY_ switch differs from the recording run).
		std::fprintf(stderr, "SHADER_PRECOMPILE_MISMATCH hash=0x%016llx stage=%s key=%s (the game's SPIR-V differs)\n",
		             static_cast<unsigned long long>(record.shader_hash), StageName(record.stage),
		             HexKey(record.Key()).c_str());
		std::fflush(stderr);
	}
	// KYTY_TRACE_SPIRV_ALL_DIR=<dir> (diagnostic): every module made here too, named as the GPU thread's ProgramCache
	// names its own (<stage>_<hash>_<SPIR-V hash>.spv), so two --precompile-only runs compare their SPIR-V by file name
	// (the test of Common::ShaderEnv's lists).
	if (const char* dir = Common::EnvRead::SpirvAllDir(); dir != nullptr && !result.spirv.empty()) {
		const auto name = fmt::format("{}/{}_{:016x}_{:016x}.spv", dir, StageName(record.stage), record.shader_hash,
		                              spirv_hash);
		if (std::FILE* f = std::fopen(name.c_str(), "wb")) {
			std::fwrite(result.spirv.data(), sizeof(uint32_t), result.spirv.size(), f);
			std::fclose(f);
		}
	}
	if (FileMatchDiag()) { // KYTY_DIAG_PRECOMPILE_FILE_MATCH: the report, no module (so no pipeline)
		FileMatchReport(record, spirv_hash, result.spirv.size());
		return false;
	}
	if (need_module) {
		module = TryCompileSPV(result.spirv, m_graphics.device);
		if (module == nullptr) {
			return false;
		}
	}
	info = std::move(result.program).TakeCompiledInfo();
	if (keep) {
		StoreReuse(record, key, program, std::move(plan), std::move(result.spirv), info, translate_ms, compile_ms);
	} else if (over_budget) {
		ReuseNoteFull(translate_ms + compile_ms, false);
	}
	return true;
}

void ShaderPrecompiler::Compile(const PrecompileStageRecord& record, uint64_t key, bool warm) {
	// A warm job (the shader reuse) makes no pipeline: nothing to do once the store does not want it.
	if (warm && !ReuseWanted(key)) {
		return;
	}
	const auto                               begin  = std::chrono::steady_clock::now();
	vk::ShaderModule                         module = nullptr;
	ShaderRecompiler::IR::CompiledShaderInfo program;
	size_t                                   spirv_words = 0;
	if (!CompileModule(record, key, !warm, module, program, spirv_words) || warm) {
		return;
	}
	// Bindless images need the bindless table's layout; without it (KYTY_BINDLESS off) the GPU thread would not
	// make this pipeline either.
	if (m_graphics.bindless_layout == nullptr &&
	    std::ranges::any_of(program.info.images, &ShaderRecompiler::IR::ImageResource::bindless)) {
		m_graphics.device.destroyShaderModule(module, nullptr);
		return;
	}
	ShaderComputeInputInfo                       input = record.compute;
	const ShaderRecompiler::IR::ResourceSnapshot resources {};
	input.stage = {.program = &program, .resources = &resources};
	PipelineCache::Pipeline pipeline;
	const auto              create_begin = std::chrono::steady_clock::now();
	CreatePipelineInternal(m_graphics, pipeline, input, module, m_driver_cache);
	const auto create_ms =
	    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - create_begin).count();
	const bool created = pipeline.pipeline != nullptr;
	m_graphics.device.destroyPipeline(pipeline.pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(pipeline.pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout, nullptr);
	m_graphics.device.destroyShaderModule(module, nullptr);
	if (created && m_warm != nullptr) {
		m_warm->Mark(key, false);
	}
	// A driver-cache hit takes well under a millisecond; only real compiles make the cache worth saving.
	if (created && create_ms >= 10.0) {
		m_created.fetch_add(1);
		m_compiled.fetch_add(1);
		g_progress.compiled.fetch_add(1, std::memory_order_relaxed);
	}
	const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
	if (ms >= 1000.0) {
		std::fprintf(stderr, "SHADER_PRECOMPILE hash=0x%016llx %.0fms (driver %.0fms) spirv_words=%zu\n",
		             static_cast<unsigned long long>(record.shader_hash), ms, create_ms, spirv_words);
		std::fflush(stderr);
	}
}

// The compiling thread reads the shader's code from guest memory: on a page the GPU owns it faults and waits for the
// thread that owns the renderer, which may be this one (or the preparation job this is).
template <typename Predicate>
static void WaitOnDone(std::unique_lock<std::mutex>& lock, std::condition_variable& done, Predicate predicate) {
	if (!GuestGpu::WaitsServeCommands()) {
		done.wait(lock, predicate);
		return;
	}
	while (!done.wait_for(lock, std::chrono::milliseconds(1), predicate)) {
		lock.unlock();
		GuestGpu::RunCommandsWhileWaiting();
		lock.lock();
	}
}

// A recorded graphics stage, compiled by the first job that needs it (later ones wait for that compile and share the
// module); null when it gave no module. The stage key is in flight while it compiles.
const ShaderPrecompiler::Stage* ShaderPrecompiler::AcquireStage(uint64_t key) {
	std::unique_lock lock(m_mutex);
	const auto       found = m_stages.find(key);
	// Stopping (the renderer goes away): the job is left.
	if (found == m_stages.end() || m_stop || m_quitting) {
		return nullptr;
	}
	auto& stage = found->second;
	WaitOnDone(lock, m_done, [&] { return stage.state != StageState::Compiling; });
	if (stage.state == StageState::Idle) {
		stage.state = StageState::Compiling;
		m_inflight.insert(key);
		WriteInflightLocked();
		lock.unlock();
		vk::ShaderModule                         module = nullptr;
		ShaderRecompiler::IR::CompiledShaderInfo info;
		size_t                                   spirv_words = 0;
		const bool compiled = CompileModule(stage.record, key, true, module, info, spirv_words);
		lock.lock();
		stage.state  = compiled ? StageState::Ready : StageState::Failed;
		stage.module = module;
		stage.info   = std::move(info);
		m_inflight.erase(key);
		if (!m_quitting) WriteInflightLocked();
		lock.unlock();
		m_done.notify_all();
		lock.lock();
	}
	return stage.state == StageState::Ready && stage.module != nullptr ? &stage : nullptr;
}

// The shader reuse: a recorded stage no recorded pipeline uses, translated and compiled for the reuse store
// only (no module, no pipeline), as a warm job's stages; nothing once the store does not want it.
void ShaderPrecompiler::CompileStageOnly(uint64_t key) {
	Stage* stage = nullptr;
	{
		std::lock_guard lock(m_mutex);
		const auto      found = m_stages.find(key);
		if (found == m_stages.end()) return;
		stage = &found->second;
		stage->users--;
		if (stage->state != StageState::Idle || m_stop || m_quitting) return;
		stage->state = StageState::Compiling;
		m_inflight.insert(key);
		WriteInflightLocked();
	}
	bool compiled = false;
	if (ReuseWanted(key)) {
		vk::ShaderModule                         module = nullptr;
		ShaderRecompiler::IR::CompiledShaderInfo info;
		size_t                                   spirv_words = 0;
		compiled = CompileModule(stage->record, key, false, module, info, spirv_words);
		if (compiled) g_reuse.stage_only_compiled.fetch_add(1, std::memory_order_relaxed);
	}
	{
		std::lock_guard lock(m_mutex);
		// Not wanted by the store or no module: a later miss counts it as compiled but not kept, not as an orphan.
		stage->state = compiled ? StageState::Ready : StageState::Failed;
		m_inflight.erase(key);
		if (!m_quitting) WriteInflightLocked();
	}
	m_done.notify_all();
}

// A job is done or dropped: its stages lose a user, and a module no job uses any more goes to `unused`.
void ShaderPrecompiler::ReleaseStagesLocked(const PrecompileGraphicsRecord&  record,
                                            std::vector<vk::ShaderModule>& unused) {
	ForEachStage(record, [&](uint64_t key) {
		const auto found = m_stages.find(key);
		if (found == m_stages.end() || found->second.users == 0) return;
		auto& stage = found->second;
		if (--stage.users == 0 && stage.module != nullptr) {
			unused.push_back(stage.module);
			stage.module = nullptr;
			stage.info   = {};
		}
	});
}

// Every stage module (no worker compiles any more) and those released by dropped jobs.
void ShaderPrecompiler::CollectModulesLocked(std::vector<vk::ShaderModule>& modules) {
	modules.insert(modules.end(), m_unused_modules.begin(), m_unused_modules.end());
	m_unused_modules.clear();
	for (auto& entry: m_stages) {
		auto& stage = entry.second;
		if (stage.module != nullptr && stage.state != StageState::Compiling) {
			modules.push_back(stage.module);
			stage.module = nullptr;
			stage.info   = {};
		}
	}
}

void ShaderPrecompiler::DestroyModules(std::vector<vk::ShaderModule>& modules) {
	for (const auto module: modules) {
		m_graphics.device.destroyShaderModule(module, nullptr);
	}
	modules.clear();
}

void ShaderPrecompiler::CompileGraphics(Job& job) {
	const auto  begin  = std::chrono::steady_clock::now();
	const auto& record = m_graphics_records[job.record];
	// The shader reuse: a warm job makes no pipeline; it compiles its stages for the store while one wants it.
	if (job.warm && !AnyStageWanted(record)) {
		return;
	}
	// The stages (a failed one skips the pipeline); the GPU thread's input infos with the compiled programs.
	const auto                                   count = record.VertexStageCount();
	const ShaderRecompiler::IR::ResourceSnapshot resources {};
	std::array<ShaderVertexInputInfo, 3>         vertex_info {};
	ShaderPixelInputInfo                         pixel_info {};
	PipelineCache::GraphicsPrograms              programs;
	uint64_t                                     vs_hash = 0;
	uint64_t                                     ps_hash = 0;
	for (uint32_t i = 0; i < count; i++) {
		const auto* stage = AcquireStage(record.vertex[i]);
		if (stage == nullptr) {
			if (!job.warm) m_skipped_graphics.fetch_add(1);
			return;
		}
		vertex_info[i]       = stage->record.vertex;
		vertex_info[i].stage = {.program = &stage->info, .resources = &resources};
		// Any id: the pipeline code tests them and logs them.
		programs.vertex[i] = {.id = i + 1u, .module = stage->module};
		vs_hash            = i == 0 ? stage->record.shader_hash : vs_hash;
	}
	if (record.pixel != 0) {
		const auto* stage = AcquireStage(record.pixel);
		if (stage == nullptr) {
			if (!job.warm) m_skipped_graphics.fetch_add(1);
			return;
		}
		pixel_info       = stage->record.pixel;
		pixel_info.stage = {.program = &stage->info, .resources = &resources};
		programs.pixel   = {.id = 4u, .module = stage->module};
		ps_hash          = stage->record.shader_hash;
	}
	if (job.warm) {
		return; // its pipeline is in the saved driver cache
	}
	const auto  stages = std::span<const ShaderVertexInputInfo>(vertex_info.data(), count);
	const auto* pixel  = record.pixel != 0 ? &pixel_info : nullptr;
	// What the pipeline code would exit on (another device's feature set, another build's records): skipped.
	if (const char* reason = GraphicsPipelineUnsupported(m_graphics, record.rendering, record.vertex_input, stages,
	                                                     pixel, record.static_params);
	    reason != nullptr) {
		if (m_skipped_graphics.fetch_add(1) < 16u) {
			std::fprintf(stderr, "SHADER_PRECOMPILE skipped graphics vs=0x%016llx ps=0x%016llx: %s\n",
			             static_cast<unsigned long long>(vs_hash), static_cast<unsigned long long>(ps_hash), reason);
			std::fflush(stderr);
		}
		return;
	}
	{
		std::lock_guard lock(m_mutex);
		if (job.skip_create) {
			return; // the GPU thread needed it first and makes it itself
		}
		job.creating = true;
	}
	PipelineCache::Pipeline pipeline;
	const auto              create_begin = std::chrono::steady_clock::now();
	CreatePipelineInternal(m_graphics, pipeline, record.rendering, record.vertex_input, stages, pixel, programs,
	                       record.static_params, m_driver_cache);
	const auto create_ms =
	    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - create_begin).count();
	const bool created = pipeline.pipeline != nullptr;
	m_graphics.device.destroyPipeline(pipeline.pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(pipeline.pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout, nullptr);
	if (created && m_warm != nullptr) {
		m_warm->Mark(job.key, true);
	}
	if (!created) {
		m_skipped_graphics.fetch_add(1);
	} else if (create_ms >= 10.0) {
		// A driver-cache hit takes well under a millisecond; only real compiles make the cache worth saving.
		m_created.fetch_add(1);
		m_compiled.fetch_add(1);
		m_compiled_graphics.fetch_add(1);
		g_progress.compiled.fetch_add(1, std::memory_order_relaxed);
	}
	const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
	if (ms >= 1000.0) {
		std::fprintf(stderr, "SHADER_PRECOMPILE graphics vs=0x%016llx ps=0x%016llx %.0fms (driver %.0fms)\n",
		             static_cast<unsigned long long>(vs_hash), static_cast<unsigned long long>(ps_hash), ms, create_ms);
		std::fflush(stderr);
	}
}

bool ShaderPrecompiler::BeforeCompile(uint64_t key) {
	std::unique_lock lock(m_mutex);
	const auto found = m_job_index.find(key);
	if (found == m_job_index.end()) return false;
	auto& job = m_jobs[found->second];
	if (job.state == JobState::Queued) {
		job.state = JobState::Dropped;
		if (job.graphics) ReleaseStagesLocked(m_graphics_records[job.record], m_unused_modules);
		// Through (the GPU thread compiles it now): counted as finished for the progress, once Start counted it queued.
		if (g_progress.started.load(std::memory_order_acquire)) {
			g_progress.finished.fetch_add(1, std::memory_order_relaxed);
			if (job.graphics) g_progress.finished_graphics.fetch_add(1, std::memory_order_relaxed);
		}
		return true;
	}
	if (job.state == JobState::Running) {
		if (job.graphics && !job.creating) {
			// Its worker still compiles stages the GPU thread has compiled already: the GPU thread makes the
			// pipeline now, the worker leaves it.
			job.skip_create = true;
			return false;
		}
		// The GPU thread waits at normal priority: so does the worker it waits for.
		RaiseThreadPriority(m_workers[job.worker]);
		const auto begin = std::chrono::steady_clock::now();
		WaitOnDone(lock, m_done, [&] { return job.state != JobState::Running; });
		const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
		if (Common::StartupTrace::Enabled()) {
			Common::StartupTrace::AddMs(Common::StartupTrace::Counter::PrecompileWait, ms);
		}
		std::fprintf(stderr, "SHADER_PRECOMPILE waited %.0fms for %s%s\n", ms, HexKey(key).c_str(),
		             job.graphics ? " (graphics)" : "");
		std::fflush(stderr);
	}
	return false;
}

// ---- the shader reuse ----

// A worker is about to compile the permutation with record key `key`: worth keeping unless the GPU thread compiled it
// already, it is kept already, or the store is at its budget (unless `any_size`).
bool ShaderPrecompiler::ReuseWanted(uint64_t key, bool any_size) const {
	std::lock_guard lock(m_reuse_mutex);
	return !m_reuse_taken.contains(key) && !m_reuse.contains(key) && (any_size || m_reuse_bytes < ReuseBudgetBytes());
}

// A program's plan is kept once, and not after the GPU thread's first use of the program.
bool ShaderPrecompiler::ReusePlanWanted(uint64_t program) const {
	std::lock_guard lock(m_reuse_mutex);
	return !m_reuse_programs_taken.contains(program) && !m_reuse_programs.contains(program);
}

// A warm graphics job: one of its stages is still to be compiled and wanted by the store.
bool ShaderPrecompiler::AnyStageWanted(const PrecompileGraphicsRecord& record) {
	std::vector<uint64_t> idle;
	{
		std::lock_guard lock(m_mutex);
		ForEachStage(record, [&](uint64_t key) {
			const auto found = m_stages.find(key);
			if (found != m_stages.end() && found->second.state == StageState::Idle) idle.push_back(key);
		});
	}
	return std::ranges::any_of(idle, [this](uint64_t key) { return ReuseWanted(key); });
}

void ShaderPrecompiler::StoreReuse(const PrecompileStageRecord& record, uint64_t key, uint64_t program,
                                   std::optional<ShaderRecompiler::IR::ResourcePlan> plan,
                                   std::vector<uint32_t> spirv, const ShaderRecompiler::IR::CompiledShaderInfo& info,
                                   double translate_ms, double compile_ms) {
	const size_t                                      bytes      = spirv.size() * sizeof(uint32_t) + InfoBytes(info);
	size_t                                            plan_bytes = plan.has_value() ? PlanBytes(*plan) : 0u;
	std::optional<ShaderRecompiler::IR::ResourcePlan> unused_plan; // destroyed after the lock is released
	std::lock_guard                                   lock(m_reuse_mutex);
	if (m_reuse_taken.contains(key) || m_reuse.contains(key)) {
		// The GPU thread needed it before this compile finished (and compiled it itself).
		g_reuse.late.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	if (plan.has_value() && (m_reuse_programs_taken.contains(program) || m_reuse_programs.contains(program))) {
		// The program is in use already, or another permutation of it holds its plan.
		unused_plan = std::move(plan);
		plan.reset();
		plan_bytes = 0;
	}
	const size_t need  = bytes + plan_bytes;
	const double cost  = translate_ms + compile_ms;
	if (m_reuse_bytes + need > ReuseBudgetBytes()) {
		ReuseNoteFull(cost, true);
		return;
	}
	auto& entry        = m_reuse[key];
	entry.record       = &record;
	entry.program      = program;
	entry.spirv        = std::move(spirv);
	entry.info         = info;
	entry.bytes        = bytes;
	entry.plan_bytes   = plan_bytes;
	entry.translate_ms = translate_ms;
	entry.compile_ms   = compile_ms;
	if (plan.has_value()) {
		entry.plan.emplace(std::move(*plan));
		m_reuse_programs.emplace(program, key);
	}
	m_reuse_bytes += bytes + plan_bytes;
	g_reuse.stored.fetch_add(1, std::memory_order_relaxed);
	g_reuse.held_bytes.store(m_reuse_bytes, std::memory_order_relaxed);
	if (m_reuse_bytes > g_reuse.peak_bytes.load(std::memory_order_relaxed)) {
		g_reuse.peak_bytes.store(m_reuse_bytes, std::memory_order_relaxed);
	}
}

bool ShaderPrecompiler::NullShareJournal(const PrecompileReuseIdentity& identity, size_t buffers,
                                         std::vector<std::array<uint32_t, 2>>& live, std::vector<uint8_t>& multi) const {
	live.assign(buffers, {0u, 0u});
	multi.assign(buffers, 0u);
	const auto found = m_program_records.find(ProgramKey(identity));
	if (found == m_program_records.end()) return false;
	bool any = false;
	for (const auto* record: found->second) {
		if (!SameProgram(*record, identity)) continue;
		any = true;
		// As MaterializeResources sets a binding's last live value: run-time stride, not null, a format (a formatted
		// buffer; a raw one has none).
		const auto& recorded = record->specialization.buffers;
		for (size_t i = 0; i < buffers && i < recorded.size(); i++) {
			const auto& buffer = recorded[i];
			if (!buffer.zero_stride_oob && (buffer.packed_stride >> 31u) != 0u &&
			    buffer.descriptor_format != Prospero::BufferFormat::kInvalid) {
				const std::array<uint32_t, 2> value {static_cast<uint32_t>(buffer.descriptor_format), buffer.descriptor_swizzle};
				if (live[i][0] != 0u && live[i] != value) multi[i] = 1u;
				live[i] = value;
			}
		}
	}
	return any;
}

bool ShaderPrecompiler::TakeReusedPlan(const PrecompileReuseIdentity&                     identity,
                                       std::optional<ShaderRecompiler::IR::ResourcePlan>& plan) {
	const uint64_t                                 program = ProgramKey(identity);
	std::vector<ShaderRecompiler::IR::ResourcePlan> released; // destroyed after the lock is released
	bool                                           found = false;
	{
		std::lock_guard lock(m_reuse_mutex);
		m_reuse_programs_taken.insert(program);
		const auto [first, last] = m_reuse_programs.equal_range(program);
		for (auto it = first; it != last; ++it) {
			const auto entry = m_reuse.find(it->second);
			if (entry == m_reuse.end() || !entry->second.plan.has_value()) continue;
			auto& stored = entry->second;
			if (!found && SameProgram(*stored.record, identity)) {
				plan.emplace(std::move(*stored.plan));
				found = true;
			} else {
				released.push_back(std::move(*stored.plan));
			}
			stored.plan.reset();
			m_reuse_bytes -= stored.plan_bytes;
			stored.plan_bytes = 0;
		}
		m_reuse_programs.erase(program);
		g_reuse.held_bytes.store(m_reuse_bytes, std::memory_order_relaxed);
	}
	(found ? g_reuse.plan_hits : g_reuse.plan_misses).fetch_add(1, std::memory_order_relaxed);
	return found;
}

bool ShaderPrecompiler::TakeReusedCompile(uint64_t key, const PrecompileReuseIdentity& identity,
                                          uint32_t                                            push_data_start,
                                          const ShaderRecompiler::IR::ResourceSpecialization& specialization,
                                          bool queued_dropped, PrecompileReused& out) {
	std::optional<ReuseEntry> taken; // destroyed after the lock is released
	bool                      same = false;
	{
		std::lock_guard lock(m_reuse_mutex);
		// A worker that finishes this permutation later keeps nothing.
		m_reuse_taken.insert(key);
		if (const auto found = m_reuse.find(key); found != m_reuse.end()) {
			auto& entry = found->second;
			same        = SameProgram(*entry.record, identity) && entry.record->push_data_start == push_data_start &&
			       entry.record->specialization == specialization;
			if (entry.plan.has_value()) {
				const auto [first, last] = m_reuse_programs.equal_range(entry.program);
				for (auto it = first; it != last; ++it) {
					if (it->second == key) {
						m_reuse_programs.erase(it);
						break;
					}
				}
			}
			m_reuse_bytes -= entry.bytes + entry.plan_bytes;
			g_reuse.held_bytes.store(m_reuse_bytes, std::memory_order_relaxed);
			taken.emplace(std::move(entry));
			m_reuse.erase(found);
		}
	}
	if (taken.has_value() && same) {
		out.spirv        = std::move(taken->spirv);
		out.info         = std::move(taken->info);
		out.translate_ms = taken->translate_ms;
		out.compile_ms   = taken->compile_ms;
		g_reuse.found.fetch_add(1, std::memory_order_relaxed);
		return true;
	}
	const auto reason = taken.has_value() ? ReuseMiss::Differs : ReuseMissReason(key, queued_dropped);
	g_reuse.misses[static_cast<uint32_t>(reason)].fetch_add(1, std::memory_order_relaxed);
	// Which permutations the GPU thread translates itself every session, and why (the first 256; a few dozen a session).
	static std::atomic_uint logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 256u) {
		static constexpr const char* Names[] = {"unrecorded", "pending", "dropped", "differs"};
		std::fprintf(stderr, "SHADER_REUSE_MISS hash=0x%016" PRIx64 " stage=%u key=0x%016" PRIx64 " push=%u reason=%s\n",
		             identity.shader_hash, static_cast<unsigned>(identity.stage), key, push_data_start,
		             Names[static_cast<uint32_t>(reason)]);
	}
	if (Common::StartupTrace::Enabled()) {
		Common::StartupTrace::Add(Common::StartupTrace::Counter::ShaderReuseMiss, 0);
	}
	return false;
}

// Why a permutation the GPU thread compiles was not in the store (a dropped one is counted by its cause too).
ShaderPrecompiler::ReuseMiss ShaderPrecompiler::ReuseMissReason(uint64_t key, bool queued_dropped) {
	static_assert(static_cast<size_t>(ReuseMiss::Count) == std::extent_v<decltype(ReuseStats::misses)>);
	if (queued_dropped) {
		return ReuseMiss::Pending; // its compute job had not started
	}
	std::lock_guard lock(m_mutex);
	if (const auto found = m_job_index.find(key); found != m_job_index.end() && !m_jobs[found->second].graphics) {
		const auto state = m_jobs[found->second].state;
		if (state == JobState::Queued || state == JobState::Running) return ReuseMiss::Pending;
		g_reuse.dropped_by[2].fetch_add(1, std::memory_order_relaxed);
		return ReuseMiss::Dropped;
	}
	if (const auto found = m_stages.find(key); found != m_stages.end()) {
		const auto& stage = found->second;
		if (stage.state == StageState::Compiling || (stage.state == StageState::Idle && stage.users != 0)) {
			return ReuseMiss::Pending;
		}
		g_reuse.dropped_by[stage.state == StageState::Idle ? 0 : 1].fetch_add(1, std::memory_order_relaxed);
		return ReuseMiss::Dropped;
	}
	return ReuseMiss::Unrecorded;
}

void ShaderPrecompiler::Record(const PrecompileStageRecord& record) {
	const auto      key = record.Key();
	std::lock_guard lock(m_mutex);
	if (record.stage == ShaderType::Compute) {
		if (!m_recorded.insert(key).second || m_journal == nullptr) return;
		WritePayload(m_journal, Serialize(record));
		std::fflush(m_journal);
		return;
	}
	if (m_graphics_journal == nullptr || !m_recorded_graphics.insert(key).second) return;
	const auto words = SerializeStage(record);
	// Guard: a record must reload to the key it was made under, or the next session never finds it (4.11.2026: five
	// pixel inputs of the key were not serialized and those permutations were translated again every session).
	if (PrecompileStageRecord reloaded; !DeserializeStage(words, reloaded) || reloaded.Key() != key) {
		static std::atomic_uint logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 16u) {
			std::fprintf(stderr, "SHADER_PRECOMPILE record does not reload to its key: hash=0x%016llx stage=%u\n",
			             static_cast<unsigned long long>(record.shader_hash), static_cast<unsigned>(record.stage));
		}
	}
	WritePayload(m_graphics_journal, words);
	std::fflush(m_graphics_journal);
}

void ShaderPrecompiler::RecordPipeline(const PrecompileGraphicsRecord& record) {
	const auto      key = record.Key();
	std::lock_guard lock(m_mutex);
	if (m_graphics_journal == nullptr || !m_recorded_graphics.insert(key).second) return;
	WritePayload(m_graphics_journal, SerializePipeline(record));
	std::fflush(m_graphics_journal);
}

PrecompileWarmList::PrecompileWarmList(std::filesystem::path path, std::string cache_signature,
                                       uint64_t loaded_cache_bytes)
    : m_path(std::move(path)), m_signature(std::move(cache_signature)) {
	const auto            load_begin  = std::chrono::steady_clock::now();
	const auto            bytes       = ReadFileBytes(m_path);
	uint64_t              saved_bytes = 0;
	std::vector<uint64_t> entries;
	// KYTY_SHADER_PRECOMPILE_WARM_PRUNE: the list by build.
	const bool                                              by_build = PruneMode() != 0;
	std::vector<std::pair<uint64_t, std::vector<uint64_t>>> sections;
	const char*                                             dropped = nullptr;
	if (bytes.empty()) {
		dropped = "no list yet";
	} else if (by_build ? !ParseWarmListBuilds(bytes, m_signature, saved_bytes, sections)
	                    : !ParseWarmList(bytes, m_signature, saved_bytes, entries)) {
		dropped = "another driver or GPU, or a damaged file";
	} else if (loaded_cache_bytes == 0) {
		dropped = "the driver cache did not load from its file";
	} else if (loaded_cache_bytes < saved_bytes) {
		dropped = "the driver cache is smaller than at the list's save";
	}
	if (dropped == nullptr && !by_build) {
		m_saved.insert(entries.begin(), entries.end());
		std::fprintf(stderr, "SHADER_PRECOMPILE warm list: %zu pipelines in the saved driver cache\n", m_saved.size());
	} else if (dropped == nullptr) {
		// This build's entries; the other builds' as saved (most recent first), the first WarmListOtherBuilds kept.
		const auto revision        = ShaderPrecompiler::Revision();
		size_t     kept_entries    = 0;
		size_t     dropped_builds  = 0;
		size_t     dropped_entries = 0;
		for (auto& [section_revision, section_entries]: sections) {
			if (section_revision == revision) {
				m_saved.insert(section_entries.begin(), section_entries.end());
				continue;
			}
			if (PruneMode() == 2) m_verify_other.insert(section_entries.begin(), section_entries.end());
			if (m_other_builds.size() < WarmListOtherBuilds) {
				kept_entries += section_entries.size();
				m_other_builds.emplace_back(section_revision, std::move(section_entries));
			} else {
				dropped_builds++;
				dropped_entries += section_entries.size();
			}
		}
		std::fprintf(stderr,
		             "SHADER_PRECOMPILE warm list: %zu pipelines in the saved driver cache (this build; %zu other "
		             "builds kept with %zu entries, %zu older builds dropped with %zu entries)\n",
		             m_saved.size(), m_other_builds.size(), kept_entries, dropped_builds, dropped_entries);
	} else {
		std::fprintf(stderr, "SHADER_PRECOMPILE warm list: empty (%s)\n", dropped);
	}
	if (PruneMode() == 2) {
		// The old list of every build (<title>.warm, left in place) is checked too: on a build's first run its entries
		// are all other builds' (a run of this build with the switch off adds its own; then hits are expected).
		auto old_path = m_path;
		old_path.replace_extension(".warm");
		uint64_t              old_bytes = 0;
		std::vector<uint64_t> old_entries;
		if (ParseWarmList(ReadFileBytes(old_path), m_signature, old_bytes, old_entries)) {
			m_verify_other.insert(old_entries.begin(), old_entries.end());
		}
		std::fprintf(stderr, "SHADER_PRECOMPILE_WARM_PRUNE verify: %zu entries of other builds (%zu from %s)\n",
		             m_verify_other.size(), old_entries.size(), old_path.filename().string().c_str());
	}
	// Its own line (scripts take the first "SHADER_PRECOMPILE warm list" line): what loading the list cost.
	std::fprintf(stderr, "SHADER_PRECOMPILE_WARM_LOAD file=%s bytes=%zu ms=%.0f\n", m_path.filename().string().c_str(),
	             bytes.size(), MillisecondsSince(load_begin));
}

uint32_t PrecompileWarmList::PruneMode() {
	static const uint32_t mode = [] {
		const char*    value = std::getenv(WarmPruneSwitchPrefix);
		const uint32_t m     = value != nullptr && (value[0] == '1' || value[0] == '2') && value[1] == '\0'
		                           ? static_cast<uint32_t>(value[0] - '0')
		                           : 0u;
		if (m != 0) {
			std::fprintf(stderr,
			             "SHADER_PRECOMPILE_WARM_PRUNE enabled mode=%u (warm list by build in .warm2: this build and "
			             "%zu before it%s)\n",
			             m, WarmListOtherBuilds, m == 2 ? "; verify: lookups also checked against other builds" : "");
			std::fflush(stderr);
		}
		return m;
	}();
	return mode;
}

const char* PrecompileWarmList::FileExtension() {
	return PruneMode() != 0 ? ".warm2" : ".warm";
}

// The record key's pipeline as this build makes it with these switches (graphics and compute keys apart).
uint64_t PrecompileWarmList::Entry(uint64_t key, bool graphics) {
	const uint64_t words[] = {key, ShaderPrecompiler::Revision(), EnvironmentFingerprint(), graphics ? 1u : 0u};
	return XXH3_64bits(words, sizeof(words));
}

void PrecompileWarmList::Mark(uint64_t key, bool graphics) {
	const auto      entry = Entry(key, graphics);
	std::lock_guard lock(m_mutex);
	if (!m_saved.contains(entry)) m_marked.insert(entry);
}

bool PrecompileWarmList::Contains(uint64_t key, bool graphics) const {
	const auto      entry = Entry(key, graphics);
	std::lock_guard lock(m_mutex);
	if (PruneMode() == 2) {
		// The old list held the other builds' entries too: it answers differently only on a hit among them.
		m_verify_lookups++;
		if (m_verify_other.contains(entry)) m_verify_hits++;
	}
	return m_saved.contains(entry);
}

std::vector<uint64_t> PrecompileWarmList::TakeMarked() {
	std::lock_guard       lock(m_mutex);
	std::vector<uint64_t> entries(m_marked.begin(), m_marked.end());
	m_marked.clear();
	return entries;
}

void PrecompileWarmList::Commit(std::vector<uint64_t> entries, bool written, uint64_t cache_payload_bytes) {
	std::lock_guard lock(m_mutex);
	if (!written) {
		m_marked.insert(entries.begin(), entries.end());
		return;
	}
	m_saved.insert(entries.begin(), entries.end());
	// Rewritten at every save: the payload size it records is the check against a driver cache started again.
	WriteLocked(cache_payload_bytes);
	if (PruneMode() == 2) {
		std::fprintf(stderr,
		             "SHADER_PRECOMPILE_WARM_PRUNE verify lookups=%llu hits_on_other_builds=%llu other_entries=%zu "
		             "(hits must be 0)\n",
		             static_cast<unsigned long long>(m_verify_lookups), static_cast<unsigned long long>(m_verify_hits),
		             m_verify_other.size());
		std::fflush(stderr);
	}
}

// Into a temporary file that replaces the list.
void PrecompileWarmList::WriteLocked(uint64_t cache_payload_bytes) const {
	std::vector<uint8_t> bytes;
	const auto           put = [&](const void* data, size_t size) {
		const auto* begin = static_cast<const uint8_t*>(data);
		bytes.insert(bytes.end(), begin, begin + size);
	};
	const bool     by_build       = PruneMode() != 0; // KYTY_SHADER_PRECOMPILE_WARM_PRUNE
	const auto     version        = by_build ? WarmListBuildsVersion : WarmListVersion;
	const auto     signature_size = static_cast<uint32_t>(m_signature.size());
	const uint64_t count          = m_saved.size();
	put(WarmListMagic, sizeof(WarmListMagic));
	put(&version, sizeof(version));
	put(&signature_size, sizeof(signature_size));
	put(m_signature.data(), m_signature.size());
	put(&cache_payload_bytes, sizeof(cache_payload_bytes));
	if (by_build) {
		// This build's section first, then the other builds kept at load.
		const auto     section_count = static_cast<uint32_t>(1u + m_other_builds.size());
		const uint64_t revision      = ShaderPrecompiler::Revision();
		put(&section_count, sizeof(section_count));
		put(&revision, sizeof(revision));
	}
	put(&count, sizeof(count));
	for (const auto entry: m_saved) {
		put(&entry, sizeof(entry));
	}
	if (by_build) {
		for (const auto& [section_revision, section_entries]: m_other_builds) {
			const uint64_t section_size = section_entries.size();
			put(&section_revision, sizeof(section_revision));
			put(&section_size, sizeof(section_size));
			if (!section_entries.empty()) put(section_entries.data(), section_entries.size() * sizeof(uint64_t));
		}
	}
	const uint64_t checksum = XXH3_64bits(bytes.data(), bytes.size());
	put(&checksum, sizeof(checksum));
	std::error_code ec;
	std::filesystem::create_directories(m_path.parent_path(), ec);
	auto temp_path = m_path;
	temp_path += ".tmp";
	if (std::FILE* f = std::fopen(temp_path.string().c_str(), "wb")) {
		std::fwrite(bytes.data(), 1, bytes.size(), f);
		const bool written = std::fflush(f) == 0 && std::ferror(f) == 0;
		std::fclose(f);
		if (written) {
			std::filesystem::rename(temp_path, m_path, ec);
		}
	}
}

} // namespace Libs::Graphics

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep

namespace Libs::Graphics {
namespace {
void LowerCurrentThreadPriority() {
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
}
void RaiseThreadPriority(std::thread& thread) {
	SetThreadPriority(static_cast<HANDLE>(thread.native_handle()), THREAD_PRIORITY_NORMAL);
}
} // namespace
} // namespace Libs::Graphics
#else
extern char** environ; // NOLINT(readability-redundant-declaration)

namespace Libs::Graphics {
namespace {
void LowerCurrentThreadPriority() {}
void RaiseThreadPriority(std::thread& /*thread*/) {}
} // namespace
} // namespace Libs::Graphics
#endif
