#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include "common/assert.h"
#include "common/envRead.h"
#include "common/liveAb.h"
#include "common/workaroundHits.h"
#include "common/shaderEnv.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/BindlessBindings.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/SrtTape.h"
#include "graphics/shader/shaderBindings.h"

#include <atomic>
#include <algorithm>
#include <span>
#include <chrono>
#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <functional>
#include <limits>
#include <numeric>
#include <tuple>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <set>
#include <string>

namespace Libs::Graphics::ShaderRecompiler::IR {

namespace {
std::atomic_int g_indirect_table_slots_override {-1};
} // namespace

bool IndirectTableSlots() {
	return g_indirect_table_slots_override.load(std::memory_order_relaxed) != 0;
}

void SetIndirectTableSlotsOverride(std::optional<bool> enabled) {
	g_indirect_table_slots_override.store(enabled.has_value() ? (*enabled ? 1 : 0) : -1, std::memory_order_relaxed);
}

namespace {

constexpr uint64_t AddressMask            = 0x0000ffffffffffffull;
constexpr uint64_t MaxIndirectDescriptorProbes = 65536u;

// Per-thread state of MaterializeResourcesAhead: the log the ahead reader appends to and why it refused. A null
// log means no ahead call runs on this thread, which is the GPU thread's normal state.
struct AheadState {
	ReadLog*     log    = nullptr;
	bool         failed = false;
	AheadRefusal reason = AheadRefusal::None;
};
thread_local AheadState t_ahead;

bool SpecializationFail(std::string_view message) {
	// An ahead call that fails is silent: the real call reports its own failures.
	if (t_ahead.log != nullptr) return false;
	std::fprintf(stderr, "shader resource specialization failed: %.*s\n",
	             static_cast<int>(message.size()), message.data());
	return false;
}

// KYTY_DIAG_NULL_FORMATTED_STICKY=1: logged APPROXIMATION. A null formatted buffer keeps the format and swizzle
// this binding last had while live (state kept in ResourcePlan::Scratch across calls).
bool NullFormattedStickyEnabled() {
	static const bool on = [] {
		const char* value = Common::ShaderEnv::Read("KYTY_DIAG_NULL_FORMATTED_STICKY");
		return value != nullptr && value[0] == '1';
	}();
	return on;
}

Decoder::ImageDimension DescriptorDimension(const DescriptorValue&  descriptor,
                                            Decoder::ImageDimension requested) {
	const bool is_array = requested == Decoder::ImageDimension::Dim1DArray ||
	                      requested == Decoder::ImageDimension::Dim2DArray ||
	                      requested == Decoder::ImageDimension::Dim2DMsaaArray;
	switch (static_cast<Prospero::ImageType>((descriptor.dwords[3] >> 28u) & 0xfu)) {
		case Prospero::ImageType::kColor1D: return Decoder::ImageDimension::Dim1D;
		case Prospero::ImageType::kColor1DArray:
			if (is_array) {
				return Decoder::ImageDimension::Dim1DArray;
			}
			return Decoder::ImageDimension::Dim1D;
		case Prospero::ImageType::kColor3D: return Decoder::ImageDimension::Dim3D;
		case Prospero::ImageType::kCube: return Decoder::ImageDimension::Dim2DArray;
		case Prospero::ImageType::kColor2DArray:
			if (is_array) {
				return Decoder::ImageDimension::Dim2DArray;
			}
			return Decoder::ImageDimension::Dim2D;
		case Prospero::ImageType::kColor2DMsaaArray:
			if (is_array) {
				return Decoder::ImageDimension::Dim2DMsaaArray;
			}
			return Decoder::ImageDimension::Dim2DMsaa;
		case Prospero::ImageType::kColor2D: return Decoder::ImageDimension::Dim2D;
		case Prospero::ImageType::kColor2DMsaa: return Decoder::ImageDimension::Dim2DMsaa;
		default: return Decoder::ImageDimension::Unknown;
	}
}

bool NullImageDescriptor(const DescriptorValue& descriptor) {
	return descriptor.dwords[0] == 0 && (descriptor.dwords[1] & 0xffu) == 0;
}

bool ValidImageDescriptor(const DescriptorValue& descriptor, bool r128 = false) {
	const auto& words = descriptor.dwords;
	// Reject texture descriptors with nonzero reserved bits.
	if ((words[1] & 0x20000000u) != 0u || (words[2] & 0x70003000u) != 0u ||
	    (!r128 && ((words[4] & 0xe000e000u) != 0u || (words[5] & 0xf9000000u) != 0u ||
	               (words[6] & 0x00007b00u) != 0u))) {
		return false;
	}
	const auto type   = static_cast<Prospero::ImageType>((descriptor.dwords[3] >> 28u) & 0xfu);
	const auto format = static_cast<Prospero::BufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
	if (type < Prospero::ImageType::kColor1D || format == Prospero::BufferFormat::kInvalid ||
	    format > Prospero::BufferFormat::kBc7Srgb) {
		return false;
	}
	if (r128 && type != Prospero::ImageType::kColor1D && type != Prospero::ImageType::kColor2D &&
	    type != Prospero::ImageType::kColor2DMsaa) {
		return false;
	}
	const bool array = type == Prospero::ImageType::kColor1DArray ||
	                   type == Prospero::ImageType::kColor2DArray ||
	                   type == Prospero::ImageType::kColor2DMsaaArray ||
	                   type == Prospero::ImageType::kCube;
	if (array && ((words[4] >> 16u) & 0x1fffu) > (words[4] & 0x1fffu)) {
		return false;
	}
	if (type == Prospero::ImageType::kColor2DMsaa ||
	    type == Prospero::ImageType::kColor2DMsaaArray) {
		const auto base_level = (descriptor.dwords[3] >> 12u) & 0xfu;
		const auto fragments  = (descriptor.dwords[3] >> 16u) & 0xfu;
		const auto max_mip    = (descriptor.dwords[5] >> 4u) & 0xfu;
		return base_level == 0 && fragments >= 1 && fragments <= 3 &&
		       (r128 || max_mip == fragments);
	}
	return true;
}

uint32_t DescriptorImageSwizzle(const DescriptorValue& descriptor) {
	return descriptor.dwords[3] & 0xfffu;
}

Prospero::BufferFormat ImageConversionFormat(Prospero::BufferFormat format) {
	return Prospero::RemapTextureFormat(format) != format ? format
	                                                      : Prospero::BufferFormat::kInvalid;
}

enum class SamplerClass : uint8_t { Float, Integer, PointInteger, FloatCompare };

// Ported from the peepostarexx/KytyPS5 Yotei branch, f0bed387: one S# that serves both a plain sample and a depth-compare sample (sample_c) gets a host sampler for
// each. The GPU ignores the S#'s compare function for a plain sample; the shared binding was marked depth_compare, so
// NativeSampler kept the compare for both, and a compare sampler on a colour view is undefined in Vulkan (black or
// white areas).

// Every image of an integer format (UINT as well as SINT) samples with point filtering. A bilinear, trilinear or
// anisotropic S# on an R8_UINT view is invalid in Vulkan (no driver, RADV included, exposes
// SAMPLED_IMAGE_FILTER_LINEAR for integer formats; VUID-vkCmdDispatch-magFilter-04553, CS 839cc50bb629b8db: an inline
// bilinear S# 0x92/0x00fff000/0x05500000/0 on an 8_UINT id map whose result is compared with an id). The GCN3 and
// RDNA2 ISA guides document no filtering of UINT/SINT data, so point sampling (a stored texel, aniso off) is the
// defined behaviour. A min/max FILTER_MODE, where RDNA reduces integer footprints (Mesa
// ac_is_reduction_mode_supported, GFX9+), stops the emulator by name (NativeSampler; none in the game's data: 0 of
// 46.8 M samples on the gate and village route, 28.10).

SamplerClass ClassifySamplerFor(Prospero::TextureNumericClass numeric_class, Prospero::BufferFormat conversion_format,
                                bool depth_compare) {
	(void)conversion_format; // (every integer image is point sampled, whatever its conversion format)
	if (numeric_class == Prospero::TextureNumericClass::Sint || numeric_class == Prospero::TextureNumericClass::Uint) {
		return SamplerClass::PointInteger;
	}
	return depth_compare ? SamplerClass::FloatCompare : SamplerClass::Float;
}

SamplerClass ClassifySampler(const ImageResource& image) {
	return ClassifySamplerFor(image.numeric_class, image.conversion_format, image.depth_compare);
}

bool DescriptorIsCube(const DescriptorValue& descriptor) {
	return static_cast<Prospero::ImageType>((descriptor.dwords[3] >> 28u) & 0xfu) ==
	       Prospero::ImageType::kCube;
}

uint32_t ImageMipCount(const ImageResource& image, const DescriptorValue& descriptor) {
	if (image.mip_mode != ImageMipMode::Dynamic || NullImageDescriptor(descriptor)) {
		return 1;
	}
	const auto base = (descriptor.dwords[3] >> 12u) & 0xfu;
	const auto last = (descriptor.dwords[3] >> 16u) & 0xfu;
	return base <= last ? last - base + 1u : 0u;
}

bool DecodeBufferDescriptor(const DescriptorValue& descriptor, ShaderBufferResource& result) {
	if (descriptor.dword_count != std::size(result.fields)) {
		return false;
	}
	std::copy_n(descriptor.dwords.begin(), std::size(result.fields), result.fields);
	return true;
}

struct ReadCapture {
	SrtRuntime                                  source;
	std::vector<std::pair<uint64_t, uint64_t>>& ranges;
	// Ordinary reads that start where the previous captured read ended extend it (they are all overlap
	// tests for the consumers); only for the built-in reader, whose callers see no per-read vector.
	bool                                        merge = false;
};

bool CaptureStrictRead(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<ReadCapture*>(userdata);
	if (!capture.source.read_specialization_memory(capture.source.userdata, address, values)) {
		return false;
	}
	capture.ranges.emplace_back(address, values.size_bytes());
	return true;
}

std::atomic<OrdinaryReadHook>        g_ordinary_read_hook {nullptr};
std::atomic<PlainReadHook>           g_plain_read_hook {nullptr};
std::atomic<OrdinaryReadSessionHook> g_ordinary_read_session_hook {nullptr};
std::atomic<AheadReadHook>           g_ahead_read_hook {nullptr};
std::atomic<FastVerifyHook>          g_fast_verify_hook {nullptr};

bool CaptureOrdinaryRead(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<ReadCapture*>(userdata);
	if (capture.source.read_memory != nullptr) {
		if (!capture.source.read_memory(capture.source.userdata, address, values)) return false;
	} else {
		// Null-pointer snapshot read on an untaken path: zero, never guest memory.
		if (address < 0x10000u) std::fill(values.begin(), values.end(), 0u);
		else ReadOrdinaryMemory(address, values.data(), values.size_bytes());
	}
	// Only reads of real guest memory by the built-in reader merge (the null page is a zero fill).
	AppendSrtRead(capture.ranges, address, values.size_bytes(),
	              capture.merge && capture.source.read_memory == nullptr && address >= 0x10000u);
	return true;
}

// Brackets the ordinary reads of one MaterializeResourcesImpl (see SetOrdinaryReadSessionHook).
class OrdinaryReadSession {
public:
	explicit OrdinaryReadSession(bool enabled)
	    : m_hook(enabled ? g_ordinary_read_session_hook.load(std::memory_order_relaxed) : nullptr) {
		if (m_hook != nullptr) m_hook(true);
	}
	~OrdinaryReadSession() {
		if (m_hook != nullptr) m_hook(false);
	}
	OrdinaryReadSession(const OrdinaryReadSession&)            = delete;
	OrdinaryReadSession& operator=(const OrdinaryReadSession&) = delete;

private:
	OrdinaryReadSessionHook m_hook;
};

const DescriptorSource* Source(const ResourcePlan& program, uint32_t source) {
	if (source >= program.descriptor_sources.size()) {
		return nullptr;
	}
	return &program.descriptor_sources[source];
}

void MarkCleanReads(const std::unordered_set<const Inst*>& planned_reads,
                    const DescriptorSource* source, Value extra = {}) {
	if (source == nullptr && extra.IsEmpty()) {
		return;
	}
	std::vector<Value> pending;
	if (source != nullptr) {
		pending.assign(source->dwords.begin(), source->dwords.begin() + source->dword_count);
	}
	if (!extra.IsEmpty()) pending.push_back(extra);
	std::vector<const Inst*> visited;
	while (!pending.empty()) {
		auto* inst = pending.back().Resolve().TryInstruction();
		pending.pop_back();
		if (inst == nullptr || std::ranges::find(visited, inst) != visited.end()) {
			continue;
		}
		visited.push_back(inst);
		if (planned_reads.contains(inst)) {
			auto flags = inst->Flags<SrtReadFlags>();
			flags.clean = 1u;
			inst->SetFlags(flags);
		}
		for (size_t arg = 0; arg < inst->NumArgs(); arg++) {
			pending.push_back(inst->Arg(arg));
		}
	}
}

bool ReadScalarTable(uint64_t base, uint64_t size, uint64_t dynamic_offset,
                     const SrtRuntime& runtime, std::span<uint32_t> words) {
	const auto offset = dynamic_offset & ~uint64_t {3};
	const auto count = std::min<uint64_t>(words.size(), offset < size ? (size - offset) / 4u : 0u);
	std::ranges::fill(words.subspan(count), 0u);
	if (count == 0u) {
		return true;
	}
	base &= AddressMask & ~uint64_t {3};
	if (offset > AddressMask - base) {
		return false;
	}
	const auto address = base + offset;
	const auto prefix = words.first(count);
	return prefix.size_bytes() - 1u <= AddressMask - address &&
	       runtime.read_specialization_memory != nullptr &&
	       runtime.read_specialization_memory(runtime.userdata, address, prefix);
}

bool NormalizeIndirectStoreBuffer(DescriptorValue& value) {
	ShaderBufferResource descriptor;
	if (!DecodeBufferDescriptor(value, descriptor)) return false;
	if (descriptor.Type() != 0u || descriptor.Format() == Prospero::BufferFormat::kInvalid ||
	    descriptor.Base48() == 0u || descriptor.GetSize() == 0u) {
		value.dwords.fill(0u);
	} else if (descriptor.Stride() != 1u || descriptor.SwizzleEnabled() || descriptor.AddTid() ||
	           descriptor.OutOfBounds() != 3u) {
		return SpecializationFail("indirect buffer stores require linear byte bounds");
	}
	return true;
}

// Diagnostic: names the line of a silent indirect-descriptor materialization failure (first 32).
static bool IndirectFailAt(int line) {
	if (t_ahead.log != nullptr) return false;
	static std::atomic_uint count {0};
	if (count.fetch_add(1) < 32u)
		std::fprintf(stderr, "indirect descriptor materialization returned false at line %d\n", line);
	return false;
}

// KYTY_BINDLESS_STAGES: shader stages allowed to use bindless heaps (V = vertex, P = pixel,
// C = compute; default all), for bisecting a rendering difference.
static bool BindlessStageAllowed(ShaderType stage) {
	static const char* stages = Common::ShaderEnv::Read("KYTY_BINDLESS_STAGES");
	if (stages == nullptr) return true;
	const char letter = stage == ShaderType::Vertex ? 'V' : stage == ShaderType::Pixel ? 'P' : stage == ShaderType::Compute ? 'C' : 'O';
	return std::strchr(stages, letter) != nullptr;
}

// KYTY_BINDLESS_R128=1: images read with 128-bit descriptors (R128) may use bindless heaps too;
// the host resolves their first four words with the rest zeroed.
static bool BindlessImageAllowed(const ImageResource& image) {
	static const bool r128 = [] {
		// Default from Common::EnvDefaults; =0 turns it off.
		const char* value = Common::ShaderEnv::Read("KYTY_BINDLESS_R128");
		return value != nullptr && value[0] == '1';
	}();
	return image.resource_class == ImageResourceClass::Sampled && (!image.r128 || r128);
}

bool IsBoundedDescriptorTable(const ResourcePlan& program,
                              const DescriptorSource::IndirectDescriptor& indirect) {
	const auto* table = Source(program, indirect.table_source);
	// A record-keyed V# (record-keyed V#s) lists its records' ranges instead of scanning the table.
	return indirect.sources.empty() && !indirect.selector && !indirect.all_records && !indirect.record_keyed &&
	       indirect.pointer_offset == UINT32_MAX && indirect.workgroup_axis == UINT32_MAX && table != nullptr &&
	       table->dword_count == 4u;
}

// The bindless tables translate tables of at most 1 MiB (RenderExecutor::PrepareBindlessTable, TableMaxBytes);
// a larger one stays a heap of float views, so no field of it is typed integer.
constexpr uint64_t BindlessTableMaxBytes = uint64_t {1} << 20u;

// The runtime the MaterializeResourcesImpl call was given, without its read capture.
const SrtRuntime& UncapturedRuntime(const SrtRuntime& runtime) {
	if (runtime.read_specialization_memory == &CaptureStrictRead && runtime.userdata != nullptr)
		return static_cast<const ReadCapture*>(runtime.userdata)->source;
	return runtime;
}

// KYTY_PERF_BINDLESS_CLASSES: the numeric class of a bindless resource that serves one table field, whose descriptors
// lie `stride` bytes apart from guest address `first` in `records` records (0: none to look at); key_stride and
// key_offset name the field as the shader does (the registry key). The registry answers every call but a field's
// first. That one, on the GPU thread, reads the field's records once, outside the call's captured reads (the class is
// the registry's, an input of every later materialization like the plan itself, not a dependency of this draw), and
// registers what they hold. An ahead call (MaterializeResourcesAhead) that finds no entry refuses: it writes nothing
// shared, so the GPU thread materializes that draw itself and decides.
Prospero::TextureNumericClass BindlessFieldClass(const ResourcePlan& program, uint32_t resource_index,
                                                 const SrtRuntime& runtime, uint32_t key_stride, uint32_t key_offset,
                                                 uint64_t first, uint32_t stride, uint64_t records) {
	using Class         = Prospero::TextureNumericClass;
	auto&      counters = GetBindlessClassCounters();
	const auto key      = BindlessClassKey(program.shader_hash, resource_index, key_stride, key_offset);
	if (const auto known = FindBindlessClass(key); known != Class::Unsupported) return known;
	// A full registry keeps no decision: the field would be read again at every materialization (and refused ahead).
	if (BindlessClassRegistryFull()) return Class::Float;
	if (t_ahead.log != nullptr) {
		if (!t_ahead.failed) {
			t_ahead.failed = true;
			t_ahead.reason = AheadRefusal::Flag;
		}
		counters.ahead_refused.fetch_add(1, std::memory_order_relaxed);
		return Class::Float;
	}
	const auto&        image      = program.info.images[resource_index];
	const uint32_t     descriptor = image.r128 ? 16u : 32u;
	const auto         binding    = BindlessArrayBinding(image);
	BindlessFieldTally tally;
	bool               read_ok = true;
	if (records != 0u && stride >= descriptor && (stride & 3u) == 0u && (first & 3u) == 0u) {
		const uint64_t bytes  = (records - 1u) * stride + descriptor;
		const auto&    source = UncapturedRuntime(runtime);
		thread_local std::vector<uint32_t> words;
		read_ok = bytes <= BindlessTableMaxBytes && source.read_specialization_memory != nullptr &&
		          first <= AddressMask && bytes - 1u <= AddressMask - first;
		if (read_ok) {
			words.resize(static_cast<size_t>(bytes / 4u));
			read_ok = source.read_specialization_memory(source.userdata, first, std::span<uint32_t>(words));
		}
		if (read_ok) {
			counters.read_bytes.fetch_add(bytes, std::memory_order_relaxed);
			for (uint64_t record = 0; record < records; record++)
				TallyBindlessFieldDescriptor(words.data() + record * (stride / 4u), image.r128, binding, tally);
		} else {
			counters.read_fail.fetch_add(1, std::memory_order_relaxed);
		}
	}
	bool       ambiguous = false;
	const auto decided   = read_ok ? DecideBindlessFieldClass(tally, &ambiguous) : Class::Float;
	counters.decided[decided == Class::Uint ? 1 : decided == Class::Sint ? 2 : 0].fetch_add(1, std::memory_order_relaxed);
	if (decided != Class::Float && tally.float_count != 0u) counters.mixed.fetch_add(1, std::memory_order_relaxed);
	if (ambiguous) counters.ambiguous.fetch_add(1, std::memory_order_relaxed);
	if (!StoreBindlessClass(key, decided)) counters.registry_full.fetch_add(1, std::memory_order_relaxed);
	// BINDLESS_CLASS first: the first decisions of fields that held integer T#s (64) and of the others (16).
	static std::atomic_uint logged_integer {0};
	static std::atomic_uint logged_float {0};
	const bool integer = tally.uint_count != 0u || tally.sint_count != 0u;
	if ((integer ? logged_integer : logged_float).fetch_add(1) < (integer ? 64u : 16u))
		std::fprintf(stderr,
		             "BINDLESS_CLASS first hash=0x%016llx image=%u field(stride:offset)=%u:%u records=%llu uint=%u "
		             "sint=%u float=%u tiny=%u other=%u read=%d -> %s%s\n",
		             static_cast<unsigned long long>(program.shader_hash), resource_index, key_stride, key_offset,
		             static_cast<unsigned long long>(records), tally.uint_count, tally.sint_count, tally.float_count,
		             tally.float_tiny, tally.other, read_ok ? 1 : 0,
		             decided == Class::Uint ? "uint" : decided == Class::Sint ? "sint" : "float",
		             ambiguous ? " (ambiguous: kept float)" : "");
	return decided;
}

// The clean evaluator of one MaterializeResources call as the indirect path sees it: the legacy
// SrtWalker, or (when `tape` is set) the tape's clean walker. Both keep one memo for the whole call.
// `walker` may be null when `tape` is set.
class IndirectEvaluator {
public:
	IndirectEvaluator(SrtWalker* walker, SrtTapeWalker* tape): m_walker(walker), m_tape(tape) {}

	bool EvaluateDescriptor(uint32_t source, DescriptorValue& result) {
		return m_tape != nullptr ? m_tape->EvaluateDescriptor(source, result)
		                         : m_walker->EvaluateDescriptor(source, result);
	}
	// `value` is the member of source's indirect descriptor that `which` names.
	bool Evaluate(uint32_t source, SrtIndirectValue which, Value value, uint32_t& result) {
		return m_tape != nullptr ? m_tape->EvaluateIndirectValue(source, which, result)
		                         : m_walker->Evaluate(value, result);
	}

private:
	SrtWalker*     m_walker;
	SrtTapeWalker* m_tape;
};

// Table slots: a key table's candidates (descriptors[first ...], ordinals 1..n in the table's
// (key, ordinal) pairs at `mapping`) reordered by kind - what BuildResourceSpecialization derives from a descriptor
// (null, dimension, cube, numeric class, conversion, swizzle, mip count) - and each kind padded to a power of two (at
// least 8) with
// copies of its last candidate; the pairs get the new ordinals, so every key names the same descriptor. A new
// candidate of a kind already present then moves no other candidate, and changes the specialization only when its
// kind crosses a power of two.
template <typename Specialization>
void GroupIndirectImageCandidates(const ImageResource& base, std::vector<DescriptorValue>& descriptors,
                                  std::vector<Specialization>& specializations, size_t first,
                                  std::vector<uint32_t>& flattened, size_t mapping, size_t key_count) {
	const size_t n = descriptors.size() - first;
	using Kind     = std::tuple<bool, uint32_t, bool, uint32_t, uint32_t, uint32_t, uint32_t>;
	std::vector<Kind> kinds(n);
	for (size_t i = 0; i < n; ++i) {
		const auto& d      = descriptors[first + i];
		const bool  null   = NullImageDescriptor(d);
		const auto  format = static_cast<Prospero::BufferFormat>((d.dwords[1] >> 20u) & 0x1ffu);
		kinds[i] = null ? Kind {true, 0u, false, 0u, 0u, 0u, 0u}
		                : Kind {false, static_cast<uint32_t>(DescriptorDimension(d, base.dimension)), DescriptorIsCube(d),
		                        static_cast<uint32_t>(Prospero::SampledTextureNumericClass(format)),
		                        static_cast<uint32_t>(ImageConversionFormat(format)), DescriptorImageSwizzle(d),
		                        ImageMipCount(base, d)};
	}
	std::vector<uint32_t> order(n);
	std::iota(order.begin(), order.end(), 0u);
	std::ranges::stable_sort(order, [&](uint32_t a, uint32_t b) { return kinds[a] < kinds[b]; });
	std::vector<DescriptorValue> grouped;
	std::vector<uint32_t>        new_ordinal(n);
	for (size_t i = 0; i < n;) {
		size_t end = i;
		while (end < n && kinds[order[end]] == kinds[order[i]]) ++end;
		for (size_t k = i; k < end; ++k) {
			new_ordinal[order[k]] = static_cast<uint32_t>(grouped.size() + 1u);
			grouped.push_back(descriptors[first + order[k]]);
		}
		// At least 8 slots per kind: a small kind would cross 1, 2, 4 with nearly every new candidate.
		for (size_t k = end - i; k < std::max<size_t>(8u, std::bit_ceil(end - i)); ++k) grouped.push_back(grouped.back());
		i = end;
	}
	for (size_t entry = 0; entry < key_count; ++entry) {
		auto& ordinal = flattened[mapping + 2u + entry * 2u];
		if (ordinal != 0u) ordinal = new_ordinal[ordinal - 1u];
	}
	const auto child = specializations[first]; // every child is the root's copy (MaterializeIndirectDescriptor)
	descriptors.resize(first);
	descriptors.insert(descriptors.end(), grouped.begin(), grouped.end());
	specializations.resize(first);
	specializations.resize(first + grouped.size(), child);
}

template <typename Specialization, typename Normalize>
bool MaterializeIndirectDescriptor(const ResourcePlan&                         program,
                                   const DescriptorSource::IndirectDescriptor& indirect,
                                   uint32_t source_index, uint32_t resource_index, uint32_t dword_count,
                                   const SrtRuntime& runtime, IndirectEvaluator& clean,
                                   ResourceSnapshot&             snapshot,
                                   std::vector<DescriptorValue>& descriptors,
                                   std::vector<Specialization>&  specializations,
                                   uint32_t maximum_resources, Normalize&& normalize) {
	const auto  descriptor_bytes = dword_count * sizeof(uint32_t);
	const auto& sources = indirect.sources;
	const auto* selector = indirect.selector ? &*indirect.selector : nullptr;
	auto& keys = program.ThreadScratch().material_keys;
	const auto children_begin = descriptors.size();
	const auto root_resource = specializations[resource_index];
	// Large tables (thousands of records) need a hashed lookup; a linear search per record is
	// quadratic and dominated GPU-thread time.
	using DescriptorKey = std::array<uint32_t, 9>;
	struct DescriptorKeyHash {
		size_t operator()(const DescriptorKey& key) const {
			uint64_t hash = 0xcbf29ce484222325ull;
			for (const auto word: key) hash = (hash ^ word) * 0x100000001b3ull;
			return static_cast<size_t>(hash);
		}
	};
	// Built on the first candidate that is not the root's own: most calls return before any (the
	// container's construction allocates with some standard libraries).
	std::optional<std::unordered_map<DescriptorKey, uint32_t, DescriptorKeyHash>> child_ordinals;
	const auto intern_candidate = [&](const DescriptorValue& candidate, uint32_t& ordinal) {
		ordinal = 0u;
		if (candidate == descriptors[resource_index]) return true;
		DescriptorKey key {};
		std::copy(candidate.dwords.begin(), candidate.dwords.end(), key.begin());
		key[8] = candidate.dword_count;
		if (!child_ordinals) child_ordinals.emplace();
		if (const auto found = child_ordinals->find(key); found != child_ordinals->end()) {
			ordinal = found->second;
			return true;
		}
		if (descriptors.size() >= maximum_resources) {
			// Logged APPROXIMATION: a GPU-selected sampler table larger than the sampler binding
			// limit maps its overflow records to the root sampler (filtering/addressing only).
			if (dword_count != 4u || maximum_resources != ShaderInfo::MaxSamplers) return false;
			static std::atomic_uint logged {0};
			if (logged.fetch_add(1) < 8u)
				std::fprintf(stderr, "DIAG_SAMPLER_TABLE_OVERFLOW APPROXIMATION candidates>%u use root\n",
				             maximum_resources);
			return true;
		}
		descriptors.push_back(candidate);
		auto child = root_resource;
		child.indirect_root = resource_index;
		specializations.push_back(child);
		ordinal = static_cast<uint32_t>(descriptors.size() - children_begin);
		child_ordinals->emplace(key, ordinal);
		return true;
	};
	// Table slots (IndirectTableSlots): pads a table's candidate list (root + children) to a power of two with copies
	// of its last candidate, so a table that gains a few streamed textures keeps its shader permutation until it
	// crosses the next power of two. Padding is never selected (the table's ordinals name real candidates only).
	const bool slots = IndirectTableSlots();
	const auto pad_candidates = [&]() {
		// Images only: buffer and sampler bindings have small fixed limits.
		if (!slots || maximum_resources != UINT32_MAX || descriptors.size() == children_begin) return;
		const auto total = descriptors.size() - children_begin + 1u;
		const auto target = std::bit_ceil(total);
		for (auto count = total; count < target && descriptors.size() < maximum_resources; ++count) {
			descriptors.push_back(descriptors.back());
			specializations.push_back(specializations.back());
		}
	};
	const auto read_keys = [&](const ShaderBufferResource& material, uint64_t first,
	                           uint64_t step, uint64_t count) {
		keys.resize(count);
		if (step == 4u) {
			return ReadScalarTable(material.Base48(), material.GetSize(), first, runtime, keys);
		}
		for (auto& key: keys) {
			if (!ReadScalarTable(material.Base48(), material.GetSize(), first, runtime, {&key, 1}))
				return IndirectFailAt(__LINE__);
			first += step;
		}
		return true;
	};
	uint64_t table_base = 0;
	uint64_t table_size = UINT64_MAX; // Scalar addresses have no buffer descriptor bounds.
	if (sources.empty()) {
		keys.clear();
		DescriptorValue material_value;
		DescriptorValue table_value;
		if ((selector != nullptr &&
		     !clean.EvaluateDescriptor(selector->source, material_value)) ||
		    !clean.EvaluateDescriptor(indirect.table_source, table_value)) {
			return IndirectFailAt(__LINE__);
		}
		ShaderBufferResource table;
		if (table_value.dword_count == 2u) {
			table_base = (static_cast<uint64_t>(table_value.dwords[1]) << 32u) | table_value.dwords[0];
		} else if (DecodeBufferDescriptor(table_value, table)) {
			table_base = table.Base48();
			table_size = table.GetSize();
		} else {
			return IndirectFailAt(__LINE__);
		}
		if (IsBoundedDescriptorTable(program, indirect)) {
			if (!indirect.table_scalar && table.Format() == Prospero::BufferFormat::kInvalid) {
				descriptors[resource_index] = {.dword_count = dword_count};
				return true;
			}
			if (table_size > UINT32_MAX ||
			    (indirect.table_record_bytes != 0u &&
			     (table.Type() != 0u || table.SwizzleEnabled() || table.AddTid() ||
			      table.OutOfBounds() != 0u || (table_base & 3u) != 0u ||
			      (table.Stride() & 3u) != 0u || table.Stride() < indirect.table_record_bytes)))
				return false;
			// KYTY_BINDLESS: a sampled image heap becomes a bindless heap. The host resolves the
			// textures shaders ask for into the set-1 arrays; no candidate becomes part of the
			// pipeline specialization, so a growing heap compiles no new permutation.
			if constexpr (std::is_same_v<Specialization, ResourceSpecialization::Image>) {
				if (BindlessImagesRequested() && BindlessStageAllowed(program.stage) && dword_count == 8u &&
				    resource_index < program.info.images.size() &&
				    BindlessImageAllowed(program.info.images[resource_index])) {
					const auto mapping_offset = snapshot.flattened_srt.size();
					snapshot.flattened_srt.resize(mapping_offset + 2u, 0u);
					snapshot.bindless_heaps.push_back({.image          = resource_index,
					                                   .base           = table_base,
					                                   .size           = table_size,
					                                   .mapping_offset = static_cast<uint32_t>(mapping_offset),
					                                   .table_base     = table_base,
					                                   .record_bytes   = table.Stride(),
					                                   .table_bytes    = table_size});
					descriptors[resource_index] = {.dword_count = dword_count};
					auto& root = specializations[resource_index];
					root.indirect_root              = resource_index;
					root.indirect_mapping_offset    = static_cast<uint32_t>(mapping_offset);
					root.indirect_search_iterations = 0u;
					root.bindless                   = true;
					// KYTY_PERF_BINDLESS_CLASSES: a resource of one field is typed from that field's records (every
					// record of the V#, as the host translates them: PrepareBindlessTable). Depth-compare samples are
					// float by definition.
					const auto& image = program.info.images[resource_index];
					if (BindlessClassesRequested() && image.bindless_field_split && image.bindless_fields.size() == 1u &&
					    !image.depth_compare) {
						const auto [field_stride, field_offset] = image.bindless_fields.front();
						const uint32_t stride     = field_stride != 0u ? field_stride : table.Stride();
						const uint32_t descriptor = image.r128 ? 16u : 32u;
						const uint64_t end        = table_size <= BindlessTableMaxBytes ? table_size : 0u;
						const uint64_t records    = stride >= 16u && end >= uint64_t {field_offset} + descriptor
						                                ? (end - field_offset - descriptor) / stride + 1u
						                                : 0u;
						root.numeric_class = BindlessFieldClass(program, resource_index, runtime, field_stride,
						                                        field_offset, table_base + field_offset, stride, records);
					}
					return true;
				}
			}
			// Descriptor bytes are refreshed together. The GPU selects the DWORD address;
			// no material selector or unused table field is interpreted on the CPU.
			keys.resize(table_size / 4u);
			if (!ReadScalarTable(table_base, table_size, 0u, runtime, keys)) return false;
			const auto mapping_offset = snapshot.flattened_srt.size();
			descriptors[resource_index] = {.dword_count = dword_count};
			snapshot.flattened_srt.resize(mapping_offset + 2u, 0u);
			for (uint32_t word = 0; word < keys.size(); ++word) {
				DescriptorValue candidate {.dword_count = dword_count};
				std::copy_n(keys.begin() + word, std::min<size_t>(dword_count, keys.size() - word),
				            candidate.dwords.begin());
				if (!normalize(candidate)) return false;
				uint32_t ordinal;
				if (!intern_candidate(candidate, ordinal)) return false;
				if (ordinal == 0u) continue;
				snapshot.flattened_srt.resize(mapping_offset + 2u + word, 0u);
				snapshot.flattened_srt[mapping_offset + 1u + word] = ordinal;
			}
			pad_candidates();
			if constexpr (std::is_same_v<Specialization, ResourceSpecialization::Image>) {
				// KYTY_TRACE_INDIRECT_COUNTS: a bounded image table that enumerates candidates instead
				// of becoming a bindless heap, with the image properties that kept it out.
				static const bool trace_bounded = Common::EnvRead::IndirectCounts();
				if (trace_bounded && dword_count == 8u && resource_index < program.info.images.size()) {
					const auto& image = program.info.images[resource_index];
					static std::mutex            mutex;
					static std::set<std::string> seen;
					const auto line = fmt::format(
					    "INDIRECT_BOUNDED stage={} image={} class={} r128={} table_bytes={} candidates={}",
					    static_cast<int>(program.stage), resource_index, static_cast<int>(image.resource_class),
					    image.r128, table_size, descriptors.size() - children_begin + 1u);
					std::lock_guard lock(mutex);
					if (seen.size() < 200u && seen.insert(line).second) std::fprintf(stderr, "%s\n", line.c_str());
				}
			}
			if (descriptors.size() == children_begin) {
				snapshot.flattened_srt.resize(mapping_offset);
			} else {
				snapshot.flattened_srt[mapping_offset] =
				    static_cast<uint32_t>(snapshot.flattened_srt.size() - mapping_offset - 1u);
				// Table slots: the region takes a power of two of entries (the count stays the
				// real one; the shader reads entries below it only), so the next table's offset stays put.
				if (slots) {
					snapshot.flattened_srt.resize(
					    mapping_offset + 1u + std::bit_ceil(uint64_t {snapshot.flattened_srt[mapping_offset]}), 0u);
				}
				auto& root = specializations[resource_index];
				root.indirect_root = resource_index;
				root.indirect_mapping_offset = static_cast<uint32_t>(mapping_offset);
				root.indirect_search_iterations = 0u;
			}
			return true;
		}
		if (selector == nullptr) {
			uint32_t key_count = 0;
			if (indirect.all_records) {
				// Every record of the table buffer is a candidate.
				if (table_value.dword_count != 4u || indirect.table_stride == 0u) return IndirectFailAt(__LINE__);
				// A two-level record holds an 8-byte pointer instead of the descriptor and must hold all of it. A
				// direct record (table-record S#s, never a two-level one) counts when its first dword
				// lies inside the table: a scalar load reads zero past the V#'s bytes, as ReadScalarTable does.
				const uint64_t first_end = indirect.pointer_offset != UINT32_MAX
				                               ? uint64_t {indirect.table_offset} + 8u
				                               : uint64_t {indirect.table_offset} + indirect.table_immediate + 4u;
				const uint64_t records =
				    table_size >= first_end ? (table_size - first_end) / indirect.table_stride + 1u : 0u;
				if (records > MaxIndirectDescriptorProbes) {
					return SpecializationFail(fmt::format(
					    "GPU-selected descriptor table has {} records (limit {})", records,
					    MaxIndirectDescriptorProbes));
				}
				key_count = static_cast<uint32_t>(records);
			} else if (indirect.workgroup_axis != UINT32_MAX) {
				if (indirect.workgroup_axis >= runtime.workgroup_counts.size()) return IndirectFailAt(__LINE__);
				key_count = runtime.workgroup_counts[indirect.workgroup_axis];
				if (key_count == 0u) return IndirectFailAt(__LINE__);
			} else {
				if (table_value.dword_count != 2u || !clean.Evaluate(source_index, SrtIndirectValue::KeyCount, indirect.key_count, key_count))
					return IndirectFailAt(__LINE__);
				if (std::bit_cast<int32_t>(key_count) <= 0) key_count = 0;
			}
			if (indirect.table_stride == 0u || key_count > MaxIndirectDescriptorProbes ||
			    (key_count != 0u && uint64_t {indirect.table_offset} + indirect.table_immediate +
			         uint64_t {key_count - 1u} * indirect.table_stride + descriptor_bytes >
			             UINT32_MAX + 1ull)) {
				return IndirectFailAt(__LINE__);
			}
			keys.resize(key_count);
			std::iota(keys.begin(), keys.end(), 0u);
			// KYTY_BINDLESS_RECORD_TABLES=1: a sampled image table the shader indexes by record
			// (descriptor at base + offset + key * stride) becomes a bindless heap as well, so the
			// number of distinct textures in it no longer selects a pipeline permutation.
			if constexpr (std::is_same_v<Specialization, ResourceSpecialization::Image>) {
				static const bool record_tables = [] {
					// Default from Common::EnvDefaults; =0 turns it off.
					const char* value = Common::ShaderEnv::Read("KYTY_BINDLESS_RECORD_TABLES");
					return value != nullptr && value[0] == '1';
				}();
				if (record_tables && BindlessImagesRequested() && dword_count == 8u && key_count != 0u &&
				    indirect.pointer_offset == UINT32_MAX &&
				    (indirect.table_stride & 3u) == 0u && (indirect.table_offset & 3u) == 0u &&
				    resource_index < program.info.images.size() &&
				    BindlessImageAllowed(program.info.images[resource_index])) {
					const uint64_t first = uint64_t {indirect.table_offset} + (indirect.table_immediate & ~3u);
					const uint64_t bytes = uint64_t {key_count - 1u} * indirect.table_stride + descriptor_bytes;
					if (table_size != UINT64_MAX && (first > table_size || bytes > table_size - first))
						return IndirectFailAt(__LINE__);
					const auto mapping_offset = snapshot.flattened_srt.size();
					snapshot.flattened_srt.resize(mapping_offset + 2u, 0u);
					snapshot.bindless_heaps.push_back({.image          = resource_index,
					                                   .base           = table_base + first,
					                                   .size           = bytes,
					                                   .mapping_offset = static_cast<uint32_t>(mapping_offset),
					                                   .table_base     = table_base,
					                                   .record_bytes   = indirect.table_stride,
					                                   .table_bytes    = table_size != UINT64_MAX ? table_size : 0u});
					descriptors[resource_index] = {.dword_count = dword_count};
					auto& root = specializations[resource_index];
					root.indirect_root              = resource_index;
					root.indirect_mapping_offset    = static_cast<uint32_t>(mapping_offset);
					root.indirect_search_iterations = 0u;
					root.bindless                   = true;
					root.bindless_stride_dwords     = indirect.table_stride / 4u;
					// KYTY_PERF_BINDLESS_CLASSES: a record table's resource is one field already (its offset is part of
					// the source); typed from the field of its key_count records unless the host would find the table
					// too large to translate (PrepareBindlessTable: the whole records and the field's bytes).
					if (BindlessClassesRequested() && !program.info.images[resource_index].depth_compare) {
						uint64_t whole = uint64_t {key_count} * indirect.table_stride;
						if (table_size != UINT64_MAX) whole = std::min(whole, table_size);
						const bool fits = std::max(whole, first + bytes) <= BindlessTableMaxBytes;
						root.numeric_class =
						    BindlessFieldClass(program, resource_index, runtime, indirect.table_stride,
						                       static_cast<uint32_t>(first), table_base + first, indirect.table_stride,
						                       fits ? key_count : 0u);
					}
					return true;
				}
			}
		} else if (!indirect.selector_first.IsEmpty()) {
			ShaderBufferResource material;
			uint32_t             first = 0, count = 0;
			if (!DecodeBufferDescriptor(material_value, material) ||
			    !clean.Evaluate(source_index, SrtIndirectValue::SelectorFirst, indirect.selector_first, first) ||
			    !clean.Evaluate(source_index, SrtIndirectValue::KeyCount, indirect.key_count, count) ||
			    count > MaxIndirectDescriptorProbes)
				return IndirectFailAt(__LINE__);
			// SMEM uses its instruction's byte stride; V# stride only supplies its bounds.
			const auto step = selector->stride != 0u ? selector->stride : material.Stride();
			if (selector->stride == 0u &&
			    (material.Type() != 0u || material.SwizzleEnabled() || material.AddTid() ||
			     material.OutOfBounds() != 0u || uint64_t {selector->offset} + 4u > step ||
			     uint64_t {first} + count > material.NumRecords()))
				return IndirectFailAt(__LINE__);
			const auto end = count != 0u ? (uint64_t {first} + count - 1u) * step + selector->offset + 4u : 0u;
			if (end > uint64_t {UINT32_MAX} + 1u || end > material.GetSize())
				return IndirectFailAt(__LINE__);
			if (!read_keys(material, uint64_t {first} * step + selector->offset, step, count))
				return false;
		} else if (!indirect.selector_mask.IsEmpty()) {
			uint32_t mask = 0;
			uint32_t count = 0;
			if (material_value.dword_count != 2u || table_value.dword_count != 2u ||
			    !clean.Evaluate(source_index, SrtIndirectValue::SelectorMask, indirect.selector_mask, mask) ||
			    !clean.Evaluate(source_index, SrtIndirectValue::KeyCount, indirect.key_count, count) || count == 0u || count > 32u) {
				return IndirectFailAt(__LINE__);
			}
			if (count < 32u) mask &= (1u << count) - 1u;
			const auto material_base =
			    (static_cast<uint64_t>(material_value.dwords[1]) << 32u) | material_value.dwords[0];
			keys.reserve(std::popcount(mask));
			while (mask != 0u) {
				const auto index = std::countr_zero(mask);
				const auto offset = static_cast<uint64_t>(selector->offset) +
				                    static_cast<uint64_t>(index) * selector->stride;
				if (offset > UINT32_MAX) return false;
				uint32_t key = 0;
				if (!ReadScalarTable(material_base, UINT64_MAX, static_cast<uint32_t>(offset),
				                     runtime, {&key, 1})) return IndirectFailAt(__LINE__);
				keys.push_back(key);
				mask &= mask - 1u;
			}
		} else return false;
		if (selector != nullptr) {
			std::ranges::sort(keys);
			keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
		}
	}

	const auto mapping_offset = snapshot.flattened_srt.size();
	const auto key_count = sources.empty() ? keys.size() : sources.size();
	// Table slots: room for a power of two of (key, ordinal) pairs; the count is the real one, the
	// search reads pairs below it only, and the search steps cover the room (a step with low >= high changes nothing).
	const auto key_room = slots ? std::bit_ceil(std::max<size_t>(key_count, 1u)) : key_count;
	snapshot.flattened_srt.resize(mapping_offset + 1u + key_room * 2u, 0u);
	snapshot.flattened_srt[mapping_offset] = static_cast<uint32_t>(key_count);
	for (uint32_t entry = 0; entry < key_count; ++entry) {
		const auto key = sources.empty() ? keys[entry] : entry;
		DescriptorValue candidate;
		candidate.dword_count = dword_count;
		if (sources.empty()) {
			const uint64_t table_offset =
			    uint64_t {(key * indirect.table_stride + indirect.table_offset) & ~3u} +
			    (indirect.table_immediate & ~3u);
			if (indirect.pointer_offset != UINT32_MAX) {
				// Two-level: the record's pointer, then the descriptor at pointer + pointer_offset.
				std::array<uint32_t, 2> pointer {};
				if (!ReadScalarTable(table_base, table_size, table_offset, runtime, pointer))
					return IndirectFailAt(__LINE__);
				const auto base = (static_cast<uint64_t>(pointer[1] & 0xffffu) << 32u) | pointer[0];
				if (base == 0u || !ReadScalarTable(base, UINT64_MAX, indirect.pointer_offset, runtime,
				                                   std::span(candidate.dwords).first(dword_count))) {
					// A record without a table: the null descriptor (never selected in practice).
					std::fill_n(candidate.dwords.begin(), dword_count, 0u);
				}
			} else if (!ReadScalarTable(table_base, table_size, table_offset, runtime,
			                            std::span(candidate.dwords).first(dword_count))) {
				return IndirectFailAt(__LINE__);
			}
		} else if (!clean.EvaluateDescriptor(sources[entry], candidate)) {
			return IndirectFailAt(__LINE__);
		}
		if (!normalize(candidate)) return IndirectFailAt(__LINE__);
		uint32_t ordinal = 0;
		if (entry == 0) {
			descriptors[resource_index] = candidate;
		} else if (!intern_candidate(candidate, ordinal)) {
			return false;
		}
		snapshot.flattened_srt[mapping_offset + 1u + entry * 2u] = key;
		snapshot.flattened_srt[mapping_offset + 2u + entry * 2u] = ordinal;
	}
	static const bool trace_counts = Common::EnvRead::IndirectCounts();
	if (trace_counts && dword_count == 8u) {
		// Which selection form enumerates image candidates, once per distinct shape.
		const char* kind = !sources.empty()                        ? "sources"
		                   : selector == nullptr && indirect.all_records ? "all_records"
		                   : selector == nullptr && indirect.workgroup_axis != UINT32_MAX ? "workgroup"
		                   : selector == nullptr                     ? "key_count"
		                   : !indirect.selector_first.IsEmpty()      ? "selector_first"
		                                                             : "selector_mask";
		static std::mutex                   mutex;
		static std::set<std::string>        seen;
		const auto line = fmt::format("INDIRECT_COUNTS kind={} keys={} candidates={} table_dwords={} stride={} "
		                              "record_bytes={}",
		                              kind, key_count, descriptors.size() - children_begin + 1u,
		                              Source(program, indirect.table_source) != nullptr
		                                  ? Source(program, indirect.table_source)->dword_count
		                                  : 0u,
		                              indirect.table_stride, indirect.table_record_bytes);
		std::lock_guard lock(mutex);
		if (seen.size() < 400u && seen.insert(line).second) std::fprintf(stderr, "%s\n", line.c_str());
	}
	bool grouped = false;
	if constexpr (std::is_same_v<Specialization, ResourceSpecialization::Image>) {
		// Table slots: candidates grouped by kind, each kind padded (GroupIndirectImageCandidates).
		if (slots && maximum_resources == UINT32_MAX && descriptors.size() > children_begin &&
		    resource_index < program.info.images.size()) {
			GroupIndirectImageCandidates(program.info.images[resource_index], descriptors, specializations, children_begin,
			                             snapshot.flattened_srt, mapping_offset, key_count);
			grouped = true;
		}
	}
	if (!grouped) pad_candidates();
	if (descriptors.size() == children_begin) {
		snapshot.flattened_srt.resize(mapping_offset);
	} else {
		auto& root                      = specializations[resource_index];
		root.indirect_root              = resource_index;
		root.indirect_mapping_offset = static_cast<uint32_t>(mapping_offset);
		root.indirect_search_iterations = std::bit_width(key_room);
	}
	return true;
}

// Record-keyed V#s: a record-keyed source enumerates at most this many records per materialization, and one
// candidate's range covers at most this many bytes (the buffer cache makes every range resident); beyond, refused.
// 65,536 (all-records lists: the proxy-set cull 8457 indexes ~7,000 set records).
constexpr uint32_t MaxIndirectRangeRecords = 65536u;
constexpr uint64_t MaxIndirectRangeBytes   = uint64_t {256} << 20u;
// Bytes of a table V# with the unbounded record count (UINT32_MAX) that every device binds (FindBuffers clamps such a
// read-only buffer to min(256 MiB, maxStorageBufferRange - alignment + 1), and Vulkan guarantees 2^27 for the range).
constexpr uint64_t UnboundedTableBoundBytes = (uint64_t {1} << 27u) - 256u;

// The guest bytes the BDA accesses of `range` can touch through the candidate V# `value`, as [address, address + size)
// widened to whole dwords (size 0: the V# admits no access). Mirrors PrepareIndirectBuffer, CalculateBufferAddress and
// IndirectBufferInBounds (spirvEmitterMemory.cpp): every vector BDA access needs a nonzero data format and record
// count; OOB modes 0 and 1 bound the index by NUM_RECORDS, modes 2 and 3 (unswizzled) do not, so there an access must
// not add an index of its own (ADD_TID adds at most lane 63). Returns why the range cannot be bounded, or nullptr.
const char* IndirectRecordRange(const IndirectRangeResource& range, const DescriptorValue& value, uint64_t& address,
                                uint64_t& size) {
	address = 0;
	size    = 0;
	ShaderBufferResource descriptor;
	if (!DecodeBufferDescriptor(value, descriptor)) return "has an invalid width";
	if (descriptor.NumRecords() == 0u || descriptor.RawFormat() == 0u || range.access_end == 0u) return nullptr;
	const uint64_t stride    = descriptor.Stride();
	uint64_t       max_index = 0;
	if (stride != 0u) {
		if (descriptor.OutOfBounds() <= 1u) {
			max_index = descriptor.NumRecords() - 1u;
		} else if (range.indexed) {
			return "has a bounds mode that does not check its index";
		} else if (descriptor.AddTid()) {
			max_index = 63u;
		}
	}
	// CalculateBufferAddress sums index * stride and the offsets in 32 bits: no access may wrap them.
	uint64_t end   = max_index * stride + range.access_end;
	uint64_t begin = range.access_begin;
	if (stride != 0u && descriptor.SwizzleEnabled()) {
		// A swizzled V# as CalculateBufferAddress lays it out (4-byte elements, groups of 8 << INDEX_STRIDE indices):
		// (index / group * stride + (offset & ~3)) * group + (index % group) * 4 + (offset & 3). The scalar offset
		// is added outside the swizzle, so the range starts at the base (bytes before the first access are harmless).
		const uint64_t group = uint64_t {8} << descriptor.IndexStride();
		end   = ((max_index / group) * stride + ((range.access_end - 1u) & ~uint64_t {3})) * group + group * 4u;
		begin = 0u;
	}
	if (end > uint64_t {UINT32_MAX} + 1u) return "can wrap its 32-bit buffer offset";
	const auto base = descriptor.Base48();
	address         = (base + begin) & ~uint64_t {3};
	size            = ((base + end + 3u) & ~uint64_t {3}) - address;
	if (size > MaxIndirectRangeBytes) return "covers more bytes than the range limit";
	return nullptr;
}

// Record-keyed V#s: the guest ranges of every record-keyed V# source (ShaderInfo::indirect_ranges). The table
// V# and the record count are clean evaluations; the candidate V# of every key below the count is read with the strict
// reader (`runtime` captures it, so the scalar-read check tests it against the draw's writes) and decoded as the GPU
// reads it: the table is bound like any buffer (a typed or empty V# binds null), and a dword past its bytes reads zero.
bool MaterializeIndirectRanges(const ResourcePlan& program, const SrtRuntime& runtime, IndirectEvaluator& clean,
                               std::span<const uint8_t> active, ResourceSnapshot& snapshot) {
	auto& ranges = snapshot.indirect_ranges;
	ranges.clear();
	snapshot.indirect_ranges_complete = false;
	snapshot.indirect_writes_complete = false;
	bool incomplete = false; // a widened source left a candidate unlisted
	bool incomplete_written = false; // ... of a written source (D6 step K2)
	for (const auto& range: program.info.indirect_ranges) {
		const auto* source = Source(program, range.source);
		if (source == nullptr || !source->indirect_descriptor.has_value() || !source->indirect_descriptor->record_keyed)
			return IndirectFailAt(__LINE__);
		if (!active.empty() && !active[range.source]) continue;
		const auto&          indirect = *source->indirect_descriptor;
		if (!indirect.sources.empty()) {
			// Path proofs: direct candidate V#s (the reads a path proof leaves).
			for (const auto candidate_source: indirect.sources) {
				DescriptorValue candidate;
				uint64_t    address = 0, size = 0;
				const char* why     = clean.EvaluateDescriptor(candidate_source, candidate)
				                          ? IndirectRecordRange(range, candidate, address, size)
				                          : "cannot be read";
				// Composed V#s: GPU-thread materializations of composed V#s (PERF
				// workaround_hits rk_composed=<listed>/<bad>).
				if (indirect.composed && t_ahead.log == nullptr)
					Common::WorkaroundHits::Note(why == nullptr ? Common::WorkaroundHits::Hit::RkComposedListings
					                                            : Common::WorkaroundHits::Hit::RkComposedBad);
				if (why != nullptr) {
					// The access would run through this V#: logged, and the call keeps the unknown-address-write path.
					static std::atomic_uint logged {0};
					if (t_ahead.log == nullptr && logged.fetch_add(1) < 32u)
						std::fprintf(stderr,
						             "RECORD_KEYED_PROVEN_BAD hash=0x%016llx source=%u composed=%d why=%s V#=%08x %08x %08x %08x\n",
						             static_cast<unsigned long long>(program.shader_hash), range.source,
						             indirect.composed ? 1 : 0, why, candidate.dwords[0], candidate.dwords[1],
						             candidate.dwords[2], candidate.dwords[3]);
					incomplete = true;
					incomplete_written = incomplete_written || range.written; // D6 step K2
					continue;
				}
				if (size != 0u)
					ranges.push_back({.address = address, .size = size, .written = range.written});
			}
			continue;
		}
		DescriptorValue      table_value;
		ShaderBufferResource table;
		uint32_t             count = 0;
		// Record-keyed address tables: the table is a 64-bit address (scalar loads from memory), not a V#.
		const auto* table_desc    = Source(program, indirect.table_source);
		const bool  address_table = table_desc != nullptr && table_desc->dword_count == 2u;
		// Game counts: the game's count of filled records, kept in CPU memory the shader does not read.
		const bool game_count = indirect.count_pointer != UINT32_MAX;
		if (!clean.EvaluateDescriptor(indirect.table_source, table_value) ||
		    (!address_table && !DecodeBufferDescriptor(table_value, table)) || (address_table && indirect.all_records) ||
		    (game_count && !address_table) ||
		    (!indirect.all_records && !game_count &&
		     !clean.Evaluate(range.source, SrtIndirectValue::KeyCount, indirect.key_count, count)))
			return IndirectFailAt(__LINE__);
		if (game_count) {
			// count = mem32[mem64[table + count_pointer] + count_field] (strict reads, like the records).
			const auto table_address = ((uint64_t {table_value.dwords[1]} << 32u) | table_value.dwords[0]) & AddressMask;
			std::array<uint32_t, 2> pointer {};
			uint32_t                filled = UINT32_MAX, check = UINT32_MAX - 1u;
			const bool read  = ReadScalarTable(table_address, UINT64_MAX, indirect.count_pointer, runtime, pointer);
			const auto block = ((uint64_t {pointer[1]} << 32u) | pointer[0]) & AddressMask;
			if (!read || block == 0u ||
			    !ReadScalarTable(block, UINT64_MAX, indirect.count_field, runtime, {&filled, 1}) ||
			    !ReadScalarTable(block, UINT64_MAX, indirect.count_check, runtime, {&check, 1}) || check != filled ||
			    filled > indirect.count_limit) {
				// Not the game's data layout here: logged, and the call keeps the unknown-address-write path.
				static std::atomic_uint logged {0};
				if (t_ahead.log == nullptr && logged.fetch_add(1) < 32u)
					std::fprintf(stderr,
					             "RECORD_KEYED_GAME_COUNT_BAD hash=0x%016llx source=%u table=0x%llx block=0x%llx count=%u check=%u\n",
					             static_cast<unsigned long long>(program.shader_hash), range.source,
					             static_cast<unsigned long long>(table_address), static_cast<unsigned long long>(block),
					             filled, check);
				if (t_ahead.log == nullptr) Common::WorkaroundHits::Note(Common::WorkaroundHits::Hit::GameCountBad);
				incomplete = true;
				incomplete_written = incomplete_written || range.written; // D6 step K2
				continue;
			}
			count = filled;
			// D6 step K3 (KYTY_TRACE_PERF: PERF workaround_hits game_count, first field).
			if (t_ahead.log == nullptr) Common::WorkaroundHits::Note(Common::WorkaroundHits::Hit::GameCountLists);
		}
		// The loop runs while key < count: no record for a count the comparison rejects at once.
		if (!indirect.all_records && !indirect.unsigned_bound && std::bit_cast<int32_t>(count) <= 0) count = 0;
		const uint64_t record_field = uint64_t {indirect.table_offset} + (indirect.table_immediate & ~3u);
		// An address table is read as far as the last record the count admits (scalar loads check no bounds).
		const auto table_base = address_table ? ((uint64_t {table_value.dwords[1]} << 32u) | table_value.dwords[0]) &
		                                            AddressMask
		                                      : table.Base48();
		const auto table_size = address_table ? record_field + uint64_t {count} * indirect.table_stride + 16u
		                                      : table.GetSize();
		if (indirect.all_records) {
			// All-records lists: every record whose first dword lies in the table's bytes (stride and
			// field are dword multiples, so the GPU's record offset is key * stride + field without wrapping).
			const uint64_t field = uint64_t {indirect.table_offset} + (indirect.table_immediate & ~3u);
			const uint64_t stride = indirect.table_stride;
			const uint64_t records = field < table_size && stride != 0u ? (table_size - field + stride - 1u) / stride : 0u;
			count = static_cast<uint32_t>(std::min<uint64_t>(records, uint64_t {UINT32_MAX}));
		}
		if ((!address_table && table.Type() != 0u) || table_base == 0u || table_size == 0u) count = 0;
		if (count > MaxIndirectRangeRecords) {
			if (indirect.widened) { // a superset too large to list: this call's address writes stay unknown
				incomplete = true;
				incomplete_written = incomplete_written || range.written; // D6 step K2
				continue;
			}
			return SpecializationFail(fmt::format("shader 0x{:016x}: record-keyed V# source {} has {} records (limit {})",
			                                      program.shader_hash, range.source, count, MaxIndirectRangeRecords));
		}
		const auto refuse = [&](uint32_t key, const char* why, const DescriptorValue& candidate) {
			return SpecializationFail(fmt::format(
			    "shader 0x{:016x}: record-keyed V# source {} record {} of {} {} (V# {:08x} {:08x} {:08x} {:08x})",
			    program.shader_hash, range.source, key, count, why, candidate.dwords[0], candidate.dwords[1],
			    candidate.dwords[2], candidate.dwords[3]));
		};
		for (uint32_t key = 0; key < count; ++key) {
			// The GPU's record offset: 32-bit shader arithmetic, then the dword-aligned instruction offset.
			const uint64_t offset = uint64_t {(key * indirect.table_stride + indirect.table_offset) & ~3u} +
			                        (indirect.table_immediate & ~3u);
			DescriptorValue candidate {.dword_count = 4u};
			if (!address_table && table.NumRecords() == UINT32_MAX && offset + 16u > UnboundedTableBoundBytes) {
				if (indirect.widened) {
					incomplete = true;
					incomplete_written = incomplete_written || range.written; // D6 step K2
					continue;
				}
				return refuse(key, "lies past the bytes every device binds of an unbounded table", candidate);
			}
			if (!ReadScalarTable(table_base, table_size, offset, runtime, std::span(candidate.dwords).first(4u)))
				return IndirectFailAt(__LINE__);
			uint64_t address = 0, size = 0;
			if (const char* why = IndirectRecordRange(range, candidate, address, size); why != nullptr) {
				// A superset's record need not hold a usable V# (e.g. float data next to the V#s): listing stays
				// incomplete, and the call keeps the unknown-address-write path.
				if (indirect.widened) {
					incomplete = true;
					incomplete_written = incomplete_written || range.written; // D6 step K2
					continue;
				}
				if (game_count) {
					// The game filled this record, so the GPU may select it: logged (verification counts these), and
					// the call keeps the unknown-address-write path instead of stopping the emulator.
					static std::atomic_uint logged {0};
					if (t_ahead.log == nullptr && logged.fetch_add(1) < 32u)
						std::fprintf(stderr,
						             "RECORD_KEYED_GAME_COUNT_BAD hash=0x%016llx source=%u key=%u why=%s V#=%08x %08x %08x %08x\n",
						             static_cast<unsigned long long>(program.shader_hash), range.source, key, why,
						             candidate.dwords[0], candidate.dwords[1], candidate.dwords[2], candidate.dwords[3]);
					if (t_ahead.log == nullptr) Common::WorkaroundHits::Note(Common::WorkaroundHits::Hit::GameCountBad);
					incomplete = true;
					incomplete_written = incomplete_written || range.written; // D6 step K2
					continue;
				}
				return refuse(key, why, candidate);
			}
			if (size != 0u) ranges.push_back({.address = address, .size = size, .written = range.written});
		}
	}
	// Overlapping and adjacent ranges are one: the renderer obtains and tests each once.
	std::ranges::sort(ranges, [](const IndirectRange& a, const IndirectRange& b) {
		return a.address != b.address ? a.address < b.address : a.size < b.size;
	});
	size_t kept = 0;
	for (size_t index = 0; index < ranges.size(); ++index) {
		const auto range = ranges[index];
		if (kept != 0u && range.address <= ranges[kept - 1u].address + ranges[kept - 1u].size) {
			auto& previous    = ranges[kept - 1u];
			previous.size     = std::max(previous.address + previous.size, range.address + range.size) - previous.address;
			previous.written  = previous.written || range.written;
		} else {
			ranges[kept++] = range;
		}
	}
	ranges.resize(kept);
	snapshot.indirect_ranges_complete = !incomplete;
	snapshot.indirect_writes_complete = !incomplete_written;
	return true;
}

// The KYTY_PERF_BINDLESS_CLASSES registry: open addressing over 64-bit keys (0 = empty), entries never removed (one per
// field resource of every shader seen; Ghost of Yotei has about 2,000 shaders). Readers take no lock (the resolve thread
// of the resolve thread reads it too): a key is published after its value (release / acquire), and a later change of
// the value is a single store. Writers are serialized.
class BindlessClassRegistry {
public:
	static constexpr uint32_t Capacity   = 1u << 14u;
	static constexpr uint32_t MaxEntries = Capacity / 4u * 3u;

	[[nodiscard]] Prospero::TextureNumericClass Find(uint64_t key) const noexcept {
		for (uint32_t probe = 0, index = Slot(key); probe < Capacity; probe++, index = (index + 1u) & (Capacity - 1u)) {
			const auto stored = m_keys[index].load(std::memory_order_acquire);
			if (stored == key)
				return static_cast<Prospero::TextureNumericClass>(m_values[index].load(std::memory_order_acquire));
			if (stored == 0u) break;
		}
		return Prospero::TextureNumericClass::Unsupported;
	}

	bool Store(uint64_t key, Prospero::TextureNumericClass value) {
		std::scoped_lock lock(m_write);
		for (uint32_t probe = 0, index = Slot(key); probe < Capacity; probe++, index = (index + 1u) & (Capacity - 1u)) {
			const auto stored = m_keys[index].load(std::memory_order_relaxed);
			if (stored == key) {
				m_values[index].store(static_cast<uint32_t>(value), std::memory_order_release);
				return true;
			}
			if (stored != 0u) continue;
			if (m_entries >= MaxEntries) {
				m_full.store(true, std::memory_order_release);
				return false;
			}
			m_values[index].store(static_cast<uint32_t>(value), std::memory_order_relaxed);
			m_keys[index].store(key, std::memory_order_release);
			m_entries++;
			return true;
		}
		return false;
	}

	[[nodiscard]] bool Full() const noexcept { return m_full.load(std::memory_order_acquire); }

	[[nodiscard]] std::array<uint32_t, 3> Counts() const {
		std::array<uint32_t, 3> counts {};
		for (uint32_t index = 0; index < Capacity; index++) {
			if (m_keys[index].load(std::memory_order_acquire) == 0u) continue;
			switch (static_cast<Prospero::TextureNumericClass>(m_values[index].load(std::memory_order_acquire))) {
				case Prospero::TextureNumericClass::Uint: counts[1]++; break;
				case Prospero::TextureNumericClass::Sint: counts[2]++; break;
				default: counts[0]++; break;
			}
		}
		return counts;
	}

private:
	[[nodiscard]] static uint32_t Slot(uint64_t key) noexcept {
		return static_cast<uint32_t>(key ^ (key >> 32u)) & (Capacity - 1u);
	}

	std::array<std::atomic<uint64_t>, Capacity> m_keys {};
	std::array<std::atomic<uint32_t>, Capacity> m_values {};
	std::mutex                                  m_write;
	uint32_t                                    m_entries = 0;
	std::atomic<bool>                           m_full {false};
};

BindlessClassRegistry& ClassRegistry() {
	static BindlessClassRegistry registry;
	return registry;
}

} // namespace

// ---- KYTY_PERF_BINDLESS_CLASSES (ResourceMaterialization.h) ----

bool BindlessImageExpected(ShaderType stage, ImageResourceClass resource_class, bool r128) {
	ImageResource image;
	image.resource_class = resource_class;
	image.r128           = r128;
	return BindlessImagesRequested() && BindlessStageAllowed(stage) && BindlessImageAllowed(image);
}

uint32_t BindlessArrayBinding(const ImageResource& image) {
	// As DefineBindlessImages (SPIR-V) and BindlessBindingFor (host).
	if (image.cube) return BindlessImagesCube;
	if (image.dimension == Decoder::ImageDimension::Dim3D) return BindlessImages3D;
	if (image.dimension == Decoder::ImageDimension::Dim2DArray ||
	    image.dimension == Decoder::ImageDimension::Dim2DMsaaArray)
		return BindlessImages2DArray;
	return BindlessImages2D;
}

void TallyBindlessFieldDescriptor(const uint32_t* words, bool r128, uint32_t binding, BindlessFieldTally& tally) {
	DescriptorValue value {.dword_count = 8u};
	std::copy_n(words, r128 ? 4u : 8u, value.dwords.begin()); // R128: words 4..7 read as zero
	ShaderTextureResource texture {};
	std::copy(value.dwords.begin(), value.dwords.end(), texture.fields);
	// The host's test of a T# (LooksLikeImageDescriptor: type 8..15, sane array and mip ranges) and no reserved bits set
	// (ValidImageDescriptor): a field that holds data in some records, or a record past a table's live count, is no T#.
	if (value.dwords[0] == 0u || (value.dwords[3] >> 28u) < 8u || texture.Depth() >= 2048u ||
	    texture.BaseArray5() > texture.Depth() || texture.LastLevel() < texture.BaseLevel() || texture.LastLevel() > 14u ||
	    !ValidImageDescriptor(value, r128)) {
		tally.other++;
		return;
	}
	// The shapes the array holds (BindlessCompatible on the host).
	bool shape = false;
	switch (texture.Type()) {
		case Prospero::ImageType::kColor2D:
		case Prospero::ImageType::kColor2DArray:
			shape = binding == BindlessImages2D || binding == BindlessImages2DArray;
			break;
		case Prospero::ImageType::kCube: shape = binding == BindlessImagesCube; break;
		case Prospero::ImageType::kColor3D: shape = binding == BindlessImages3D; break;
		default: break;
	}
	const auto format        = texture.Format();
	const auto numeric_class = shape ? Prospero::SampledTextureNumericClass(format)
	                                 : Prospero::TextureNumericClass::Unsupported;
	switch (numeric_class) {
		case Prospero::TextureNumericClass::Float:
			tally.float_count++;
			if (texture.Width5() < 4u && texture.Height5() < 4u) tally.float_tiny++;
			return;
		case Prospero::TextureNumericClass::Uint:
		case Prospero::TextureNumericClass::Sint:
			// A format the host converts in the shader (11_11_10_UINT read as 32_UINT) has no plain bindless view.
			if (Prospero::RemapTextureFormat(format) != format) break;
			(numeric_class == Prospero::TextureNumericClass::Uint ? tally.uint_count : tally.sint_count)++;
			return;
		default: break;
	}
	tally.other++;
}

Prospero::TextureNumericClass DecideBindlessFieldClass(const BindlessFieldTally& tally, bool* ambiguous) {
	const bool integer  = tally.uint_count != 0u || tally.sint_count != 0u;
	const bool one      = (tally.uint_count == 0u) != (tally.sint_count == 0u);
	const bool defaults = tally.float_count == tally.float_tiny;
	if (ambiguous != nullptr) *ambiguous = integer && !(one && defaults);
	if (!integer || !one || !defaults) return Prospero::TextureNumericClass::Float;
	return tally.uint_count != 0u ? Prospero::TextureNumericClass::Uint : Prospero::TextureNumericClass::Sint;
}

uint64_t BindlessClassKey(uint64_t shader_hash, uint32_t resource, uint32_t field_stride, uint32_t field_offset) {
	uint64_t hash = shader_hash;
	for (const uint64_t value: {uint64_t {resource} + 1u, uint64_t {field_stride} + (uint64_t {1} << 32u),
	                            uint64_t {field_offset} + (uint64_t {2} << 32u)}) {
		hash ^= value;
		hash *= 0xff51afd7ed558ccdull;
		hash ^= hash >> 33u;
	}
	return hash != 0u ? hash : 1u;
}

Prospero::TextureNumericClass FindBindlessClass(uint64_t key) {
	return ClassRegistry().Find(key);
}

bool StoreBindlessClass(uint64_t key, Prospero::TextureNumericClass numeric_class) {
	return ClassRegistry().Store(key, numeric_class);
}

std::array<uint32_t, 3> BindlessClassRegistryCounts() {
	return ClassRegistry().Counts();
}

bool BindlessClassRegistryFull() {
	return ClassRegistry().Full();
}

BindlessClassCounters& GetBindlessClassCounters() {
	static BindlessClassCounters counters;
	return counters;
}

struct SamplerPlan {
	struct Binding {
		uint32_t     source;
		SamplerClass type;
	};
	std::array<std::array<uint32_t, 4>, ShaderInfo::MaxSamplers> mapping;
	std::array<Binding, ShaderInfo::MaxSamplers>                bindings;
	uint32_t                                                  sampler_count = 0;
};

template <typename T, typename Keep>
void CompactImages(std::vector<T>& images, Keep&& keep) {
	size_t count = 0;
	for (size_t index = 0; index < images.size(); ++index) {
		if (!keep(index)) continue;
		if (count != index) images[count] = std::move(images[index]);
		++count;
	}
	images.resize(count);
}

struct ImageRemap {
	explicit ImageRemap(const ResourceSpecialization& specialization)
	    : indices(specialization.images.size()),
	      source_count(static_cast<uint32_t>(specialization.images.size())) {
		for (uint32_t index = 0; index < source_count; index++) {
			indices[index] = specialization.images[index].fmask ? UINT32_MAX : count++;
		}
	}

	uint32_t operator[](uint32_t index) const {
		EXIT_IF(index >= source_count);
		return indices[index];
	}

	template <typename T>
	void Apply(std::vector<T>& images) const {
		EXIT_IF(images.size() != source_count);
		if (count == source_count) {
			return;
		}
		CompactImages(images, [&](size_t index) { return indices[index] != UINT32_MAX; });
	}

private:
	std::vector<uint32_t> indices;
	uint32_t                                    source_count;
	uint32_t                                    count = 0;
};

// KYTY_PERF_BINDLESS_CLASSES: an integer image samples with an integer sampler (ClassifySampler: point filtering and
// integer border colours), a native variant of the S# its instruction names (BuildSamplerPlan). Whether the shader's
// samplers still fit with these classes: at most ShaderInfo::MaxSamplers native samplers, and a sampler the GPU
// selects from a table serves one class only (ApplyResourceSpecialization stops the emulator otherwise; a plain and a
// compare sample of float images may share one). The same walk as those two, over the specialized image classes.
static bool BindlessIntegerSamplersFit(const ResourcePlan& program, const ResourceSpecialization& specialization) {
	const auto& info = program.info;
	if (info.samplers.size() > ShaderInfo::MaxSamplers) return false;
	std::array<uint8_t, ShaderInfo::MaxSamplers>  usage {};
	std::array<uint32_t, ShaderInfo::MaxSamplers> known;
	known.fill(UINT32_MAX);
	const auto float_class   = static_cast<uint32_t>(SamplerClass::Float);
	const auto compare_class = static_cast<uint32_t>(SamplerClass::FloatCompare);
	for (const auto& pair: info.sampled_pairs) {
		if (pair.image >= info.images.size() || pair.image >= specialization.images.size() ||
		    pair.sampler >= info.samplers.size())
			return false;
		const auto& image = specialization.images[pair.image];
		const auto  type  = static_cast<uint32_t>(ClassifySamplerFor(image.numeric_class, image.conversion_format,
		                                                             info.images[pair.image].depth_compare));
		usage[pair.sampler] |= static_cast<uint8_t>(1u << type);
		for (const auto candidate: info.samplers[pair.sampler].indirect_resources) {
			if (candidate >= ShaderInfo::MaxSamplers) return false;
			usage[candidate] |= usage[pair.sampler];
		}
		if (pair.sampler < specialization.samplers.size() &&
		    specialization.samplers[pair.sampler].indirect_root == pair.sampler) {
			auto&      root_class   = known[pair.sampler];
			const bool compare_pair = (root_class == float_class && type == compare_class) ||
			                          (root_class == compare_class && type == float_class);
			if (root_class != UINT32_MAX && root_class != type && !compare_pair) return false;
			if (!compare_pair) root_class = type;
		}
	}
	// The native samplers: one per S# and class, then the copies of GPU-selected candidates (materialized children).
	uint32_t count = static_cast<uint32_t>(info.samplers.size());
	for (uint32_t index = 0; index < info.samplers.size(); index++)
		count += std::max(std::popcount(static_cast<uint32_t>(usage[index])), 1) - 1u;
	const auto children = specialization.samplers.size() > info.samplers.size()
	                          ? static_cast<uint32_t>(specialization.samplers.size() - info.samplers.size())
	                          : 0u;
	return count + children <= ShaderInfo::MaxSamplers;
}

static bool BuildResourceSpecialization(const ResourcePlan& program, ResourceSnapshot& snapshot,
                                        ResourceSpecialization& specialization) {
	bool bindless_integer = false; // KYTY_PERF_BINDLESS_CLASSES
	for (uint32_t i = 0; i < specialization.images.size(); i++) {
		const auto& descriptor = snapshot.images[i];
		auto&       image      = specialization.images[i];
		const auto  base_index = i < program.info.images.size() ? i : image.indirect_root;
		if (base_index >= program.info.images.size()) {
			return SpecializationFail(fmt::format("image resource {} has an invalid root", i));
		}
		const auto& base = program.info.images[base_index];
		if (base.resource_class == ImageResourceClass::None ||
		    (base.atomic && base.resource_class != ImageResourceClass::Storage)) {
			return SpecializationFail(fmt::format("image resource {} has an invalid class", i));
		}
		image.mip_count = ImageMipCount(base, descriptor);
		if (image.mip_count == 0u) {
			return SpecializationFail(
			    fmt::format("image descriptor {} has an invalid mip range", i));
		}
		if (NullImageDescriptor(descriptor)) {
			// KYTY_PERF_BINDLESS_CLASSES: the class MaterializeIndirectDescriptor gave a bindless field's resource
			// (Uint or Sint only while the switch is on; the plan's class otherwise).
			const auto field_class = image.numeric_class;
			image.numeric_class = base.atomic ? Prospero::TextureNumericClass::Uint
			                                  : Prospero::TextureNumericClass::Float;
			image.dimension     = Decoder::ImageDimension::Dim2D;
			image.cube          = false;
			if (image.bindless) {
				// No one descriptor describes a bindless image: keep the dimension the
				// instruction samples, which picks the set-1 array.
				image.dimension = base.dimension == Decoder::ImageDimension::Unknown
				                      ? Decoder::ImageDimension::Dim2D : base.dimension;
				image.cube      = base.cube;
				if (BindlessClassesRequested() && !base.atomic && !base.depth_compare &&
				    (field_class == Prospero::TextureNumericClass::Uint ||
				     field_class == Prospero::TextureNumericClass::Sint)) {
					image.numeric_class = field_class;
					bindless_integer    = true;
				}
			}
			continue;
		}
		const auto descriptor_dimension = DescriptorDimension(descriptor, base.dimension);
		if (descriptor_dimension == Decoder::ImageDimension::Unknown) {
			return SpecializationFail(fmt::format(
			    "image descriptor {} has unsupported type {}: {:08x},{:08x},{:08x},{:08x},"
			    "{:08x},{:08x},{:08x},{:08x}",
			    i, (descriptor.dwords[3] >> 28u) & 0xfu, descriptor.dwords[0], descriptor.dwords[1],
			    descriptor.dwords[2], descriptor.dwords[3], descriptor.dwords[4],
			    descriptor.dwords[5], descriptor.dwords[6], descriptor.dwords[7]));
		}
		image.dimension = descriptor_dimension;
		image.cube      = DescriptorIsCube(descriptor);
		const auto format =
		    static_cast<Prospero::BufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
		if (base.atomic &&
		    (base.atomic64 ? format != Prospero::BufferFormat::k32_32UInt
		                   : format != Prospero::BufferFormat::k32UInt &&
		                         format != Prospero::BufferFormat::k32Float)) {
			return SpecializationFail(
			    fmt::format("atomic image descriptor {} uses unsupported format {}", i,
			                static_cast<uint32_t>(format)));
		}
		const bool storage      = base.resource_class == ImageResourceClass::Storage;
		image.fmask             = Prospero::IsFmaskTextureFormat(format);
		if (image.fmask) {
			if (storage || base.depth_compare ||
			    image.indirect_root != ImageResource::NoIndirectImage ||
			    std::ranges::any_of(program.info.sampled_pairs,
			                        [&](const auto& pair) { return pair.image == i; })) {
				return SpecializationFail("FMASK requires a direct image load");
			}
		}
		image.conversion_format = ImageConversionFormat(format);
		if (storage || image.conversion_format != Prospero::BufferFormat::kInvalid) {
			image.shader_swizzle = DescriptorImageSwizzle(descriptor);
		}
		// Write-only signed-integer storage stores the same low bits through an unsigned view.
		const bool raw_sint_storage = storage && (format == Prospero::BufferFormat::k32SInt ||
		                                          format == Prospero::BufferFormat::k16SInt ||
		                                          format == Prospero::BufferFormat::k8SInt) &&
		                              base.written && !base.read && !base.atomic;
		// Packed 11_11_10 UNorm (format 30, KYTY_FIX_RT_11_11_10_FLOAT=0) is bound through its R32_UINT backing: the
		// image is typed by the backing's class and the shader decodes the fields (UnpackImageTexel, SamplePackedUnormLz).
		image.numeric_class = Prospero::SampledTextureNumericClass(
		    image.conversion_format == Prospero::BufferFormat::k11_11_10UNorm ? Prospero::RemapTextureFormat(format) : format);
		if (storage) {
			if ((!raw_sint_storage && image.numeric_class == Prospero::TextureNumericClass::Sint) ||
			    image.numeric_class == Prospero::TextureNumericClass::Unsupported) {
				return SpecializationFail(
				    fmt::format("storage image descriptor {} uses unsupported format {}", i,
				                static_cast<uint32_t>(format)));
			}
			if (raw_sint_storage || base.atomic) {
				image.numeric_class = Prospero::TextureNumericClass::Uint;
			}
		} else if (image.numeric_class == Prospero::TextureNumericClass::Unsupported ||
		           (base.depth_compare &&
		            image.numeric_class != Prospero::TextureNumericClass::Float)) {
			return SpecializationFail(
			    fmt::format("sampled image descriptor {} uses unsupported format {}", i,
			                static_cast<uint32_t>(format)));
		}
	}
	for (uint32_t root_index = 0; root_index < specialization.images.size(); root_index++) {
		auto& root = specialization.images[root_index];
		if (root.indirect_root != root_index || root.bindless) {
			continue;
		}
		const auto key_count = root.indirect_mapping_offset < snapshot.flattened_srt.size()
		                           ? snapshot.flattened_srt[root.indirect_mapping_offset]
		                           : 0u;
		if (key_count == 0u ||
		    static_cast<size_t>(root.indirect_mapping_offset) + 1u +
		            static_cast<size_t>(key_count) * (root.indirect_search_iterations == 0u ? 1u : 2u) >
		        snapshot.flattened_srt.size()) {
			return SpecializationFail("indirect image specialization has an invalid key mapping");
		}
		uint32_t exemplar       = ImageResource::NoIndirectImage;
		uint32_t resource_count = 0;
		for (uint32_t resource = 0; resource < specialization.images.size(); resource++) {
			if (specialization.images[resource].indirect_root != root_index) {
				continue;
			}
			resource_count++;
			if (exemplar == ImageResource::NoIndirectImage &&
			    !NullImageDescriptor(snapshot.images[resource])) {
				exemplar = resource;
			}
		}
		if (resource_count < 2u || exemplar == ImageResource::NoIndirectImage) {
			return SpecializationFail("indirect image specialization has no typed candidate");
		}
		const auto& image_class = specialization.images[exemplar];
		const auto is_2d = [](Decoder::ImageDimension dimension) {
			return dimension == Decoder::ImageDimension::Dim2D ||
			       dimension == Decoder::ImageDimension::Dim2DArray;
		};
		for (uint32_t candidate = 0; candidate < specialization.images.size(); candidate++) {
			auto& image = specialization.images[candidate];
			if (image.indirect_root != root_index) {
				continue;
			}
			if (NullImageDescriptor(snapshot.images[candidate])) {
				image.numeric_class     = image_class.numeric_class;
				image.dimension         = image_class.dimension;
				image.mip_count         = image_class.mip_count;
				image.conversion_format = image_class.conversion_format;
				image.shader_swizzle    = image_class.shader_swizzle;
				image.cube              = image_class.cube;
			}
			const bool same_coordinates = image.dimension == image_class.dimension &&
			                              image.cube == image_class.cube;
			// 2D and 2D-array candidates share a table: each keeps its own type (per-candidate access).
			const bool loose_2d = is_2d(image.dimension) && is_2d(image_class.dimension);
			// Indirect image writes: a candidate that differs from the exemplar only in dimension (not cube or
			// MSAA: every access takes its coordinates as the candidate's own type needs them, EmitImageAccess) or in the
			// swizzle of a format without conversion (storage writes apply each candidate's own, StoreTexel; sampled
			// views carry theirs) keeps its own type, as the GPU addresses it by its own T#, instead of a null image.
			const auto plain_dimension = [](Decoder::ImageDimension dimension) {
				return dimension == Decoder::ImageDimension::Dim1D || dimension == Decoder::ImageDimension::Dim1DArray ||
				       dimension == Decoder::ImageDimension::Dim2D || dimension == Decoder::ImageDimension::Dim2DArray ||
				       dimension == Decoder::ImageDimension::Dim3D;
			};
			// Storage tables only: their accesses are ImageRead and ImageWrite (per candidate) or root-only operations;
			// a sampled table's gradients and offsets follow the root's dimension.
			if (program.info.images[root_index].resource_class == ImageResourceClass::Storage && !image.cube &&
			    !image_class.cube &&
			    plain_dimension(image.dimension) && plain_dimension(image_class.dimension) &&
			    image.numeric_class == image_class.numeric_class && image.mip_count == image_class.mip_count &&
			    image.conversion_format == image_class.conversion_format &&
			    image.conversion_format == Prospero::BufferFormat::kInvalid &&
			    (!same_coordinates || image.shader_swizzle != image_class.shader_swizzle)) {
				continue;
			}
			if (image.numeric_class != image_class.numeric_class || (!same_coordinates && !loose_2d) ||
			    image.mip_count != image_class.mip_count ||
			    image.conversion_format != image_class.conversion_format ||
			    image.shader_swizzle != image_class.shader_swizzle) {
				return SpecializationFail(fmt::format("indirect image table at pc 0x{:08x} has incompatible candidates",
				                                      program.info.images[root_index].first_use_pc));
			}
		}
	}
	// KYTY_PERF_BINDLESS_CLASSES: integer field resources whose samplers would not fit keep sampling as float (their
	// integer T#s read the float placeholder, as without the switch).
	if (bindless_integer && !BindlessIntegerSamplersFit(program, specialization)) {
		for (auto& image: specialization.images) {
			if (image.bindless && (image.numeric_class == Prospero::TextureNumericClass::Uint ||
			                       image.numeric_class == Prospero::TextureNumericClass::Sint))
				image.numeric_class = Prospero::TextureNumericClass::Float;
		}
		auto& counters = GetBindlessClassCounters();
		if (counters.sampler_reverts.fetch_add(1, std::memory_order_relaxed) < 8u && t_ahead.log == nullptr)
			std::fprintf(stderr, "BINDLESS_CLASS sampler_revert hash=0x%016llx (integer fields kept float)\n",
			             static_cast<unsigned long long>(program.shader_hash));
	}
	CompactImages(snapshot.images, [&](size_t index) { return !specialization.images[index].fmask; });
	return true;
}

bool BuildSamplerPlan(const ShaderInfo& base, SamplerPlan& plan) {
	if (base.samplers.size() > plan.mapping.size()) {
		return false;
	}
	std::array<uint8_t, ShaderInfo::MaxSamplers> usage {};
	plan.sampler_count = static_cast<uint32_t>(base.samplers.size());
	for (const auto& pair: base.sampled_pairs) {
		if (pair.image >= base.images.size() || pair.sampler >= base.samplers.size()) {
			return false;
		}
		usage[pair.sampler] |= 1u << static_cast<uint32_t>(ClassifySampler(base.images[pair.image]));
		for (const auto candidate: base.samplers[pair.sampler].indirect_resources) {
			if (candidate >= base.samplers.size()) return false;
			usage[candidate] |= usage[pair.sampler];
		}
	}
	for (uint32_t index = 0; index < base.samplers.size(); index++) {
		auto& mapping = plan.mapping[index];
		mapping.fill(UINT32_MAX);
		const auto classes = usage[index] == 0u ? 1u : usage[index];
		bool       first   = true;
		for (uint32_t type = 0; type < mapping.size(); type++) {
			if ((classes & (1u << type)) == 0u) continue;
			const auto target = first ? index : plan.sampler_count++;
			if (target >= ShaderInfo::MaxSamplers) return false;
			mapping[type]         = target;
			plan.bindings[target] = {index, static_cast<SamplerClass>(type)};
			first                = false;
		}
	}
	return true;
}

template <typename Predicate>
static std::vector<ResourceBlock> ResourceControlFlow(const Program& program, const Predicate& predicate) {
	// The renderer checks captured scalar reads against final buffer and image write ranges.
	if (program.has_address_writes) {
		return {};
	}
	std::unordered_map<const Block*, uint32_t> indices;
	for (uint32_t i = 0; i < program.blocks.size(); i++) {
		if (!indices.emplace(program.blocks[i], i).second) {
			return {};
		}
	}
	std::vector<ResourceBlock> blocks(program.blocks.size());
	for (uint32_t i = 0; i < blocks.size(); i++) {
		auto&                 block      = blocks[i];
		const auto&             info       = *program.blocks[i];
		const auto&           terminator = info.terminator;
		const std::array        targets {terminator.true_block, terminator.false_block};
		std::span<Block* const> successors;
		switch (terminator.kind) {
			case CFG::TerminatorKind::Branch: successors = std::span(targets).first(1); break;
			case CFG::TerminatorKind::ConditionalBranch:
				successors      = targets;
				block.condition = info.condition;
				break;
			case CFG::TerminatorKind::IndirectBranch: successors = info.ImmSuccessors(); break;
			case CFG::TerminatorKind::Return: break;
			default: return {};
		}
		for (const auto successor: successors) {
			const auto found = indices.find(successor);
			if (found == indices.end()) {
				return {};
			}
			block.successors.push_back(found->second);
		}
		for (const auto& inst: *program.blocks[i]) {
			const auto op     = inst.GetOpcode();
			if (op == ValueOpcode::ReadConst) {
				block.srt_reads.push_back(inst.Arg(1).Resolve().U32());
				continue;
			}
			const auto buffer = BufferAccessOf(op);
			const auto image  = ImageOpcodeInfoOf(op);
			if (buffer == BufferAccess::None && image.access == ImageAccess::None) {
				continue;
			}
			const auto& memory = program.memory_info.at(inst.Flags<MemoryFlags>().index);
			if (memory.planning_only || memory.kind == ResourceKind::IndirectBuffer) {
				continue;
			}
			if (buffer != BufferAccess::None) {
				block.sources.push_back(program.info.buffers.at(memory.resource).source);
			} else {
				block.sources.push_back(program.info.images.at(memory.resource).source);
				if (image.needs_sampler) {
					block.sources.push_back(program.info.samplers.at(memory.sampler).source);
				}
			}
		}
		std::ranges::sort(block.sources);
		block.sources.erase(std::unique(block.sources.begin(), block.sources.end()),
		                    block.sources.end());
	}
	// A condition the walk does not need: each successor is the join block itself or an inert block (no sources,
	// no flattened reads, no condition, one successor) leading straight to the join. Either outcome visits the
	// join next and the same blocks after it, so evaluating the condition (and reading its memory at record time)
	// changes nothing. Ghost of Yotei: an S_TRAP debug break (ControlNop) behind a test of a flag word that the
	// same dispatch sets with an atomic.
	const auto inert = [&](uint32_t index) {
		const auto& block = blocks[index];
		return block.condition.IsEmpty() && block.sources.empty() && block.srt_reads.empty() &&
		       block.successors.size() == 1u;
	};
	for (uint32_t i = 0; i < blocks.size(); i++) {
		auto& block = blocks[i];
		if (block.condition.IsEmpty() || block.successors.size() != 2u) continue;
		const auto a = block.successors[0];
		const auto b = block.successors[1];
		const bool inert_a = a != i && inert(a);
		const bool inert_b = b != i && inert(b);
		if (a == b || (inert_a && blocks[a].successors[0] == b) || (inert_b && blocks[b].successors[0] == a) ||
		    (inert_a && inert_b && blocks[a].successors[0] == blocks[b].successors[0])) {
			block.condition = Value {};
		}
	}
	for (auto& block: blocks) block.condition = predicate(block.condition);
	if (std::ranges::none_of(
	        blocks, [](const ResourceBlock& block) { return !block.condition.IsEmpty(); })) {
		return {};
	}
	return blocks;
}

// ResolveComputeBufferFill proves complete workgroups and matching requested extents, so every
// dispatch-bound predicate accepted here is true for every invocation covered by the fill.
static bool IsFullDispatchPredicate(Value value, uint32_t depth = 0) {
	value = value.Resolve();
	if (value == Value(true)) return true;
	const auto* inst = value.TryInstruction();
	if (inst == nullptr || depth > 32) return false;
	if (inst->GetOpcode() == ValueOpcode::LogicalAnd) {
		return IsFullDispatchPredicate(inst->Arg(0), depth + 1) &&
		       IsFullDispatchPredicate(inst->Arg(1), depth + 1);
	}
	if (inst->GetOpcode() != ValueOpcode::ULessThan32) return false;
	const auto* global = inst->Arg(0).Resolve().TryInstruction();
	const auto* extent = inst->Arg(1).Resolve().TryInstruction();
	if (global == nullptr || global->GetOpcode() != ValueOpcode::GetBuiltin ||
	    global->Arg(0).Resolve() != Value(static_cast<uint32_t>(StageInputKind::GlobalInvocationId)) ||
	    extent == nullptr || extent->GetOpcode() != ValueOpcode::GetDispatchThreadExtent)
		return false;
	const auto axis = global->Arg(1).Resolve();
	return axis.IsImmediate() && axis.U32() < 3 && extent->Arg(0).Resolve() == axis;
}

// Nonnegative affine coefficients for constant, local and workgroup coordinates. Reject modular
// arithmetic that could wrap; runtime coverage also bounds the largest invocation index.
static std::optional<std::array<uint64_t, 3>> FillIndex(Value value, uint32_t axis, Value guard,
                                                      uint32_t depth = 0) {
	value = ResolveActiveU32(value, guard);
	if (depth > 32 || value.GetType() != Type::U32) {
		return {};
	}
	if (value.IsImmediate()) {
		return std::array<uint64_t, 3> {value.U32(), 0, 0};
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return {};
	}
	const auto op = inst->GetOpcode();
	if (op == ValueOpcode::GetBuiltin && inst->Arg(1) == Value(axis)) {
		if (inst->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId))) {
			return std::array<uint64_t, 3> {0, 1, 0};
		}
		if (inst->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::WorkgroupId))) {
			return std::array<uint64_t, 3> {0, 0, 1};
		}
	}
	if (op != ValueOpcode::IAdd32 && op != ValueOpcode::IMul32 &&
	    op != ValueOpcode::ShiftLeftLogical32) {
		return {};
	}
	auto left  = FillIndex(inst->Arg(0), axis, guard, depth + 1);
	auto right = FillIndex(inst->Arg(1), axis, guard, depth + 1);
	if (!left || !right) {
		return {};
	}
	if (op == ValueOpcode::IMul32 && ((*right)[1] != 0 || (*right)[2] != 0)) {
		std::swap(left, right);
	}
	if (op != ValueOpcode::IAdd32 && ((*right)[1] != 0 || (*right)[2] != 0)) {
		return {};
	}
	if (op == ValueOpcode::ShiftLeftLogical32) {
		if ((*right)[0] >= 32) return {};
		(*right)[0] = uint64_t {1} << (*right)[0];
	}
	for (uint32_t i = 0; i < left->size(); ++i) {
		(*left)[i] =
		    op == ValueOpcode::IAdd32 ? (*left)[i] + (*right)[i] : (*left)[i] * (*right)[0];
		if ((*left)[i] > UINT32_MAX) return {};
	}
	return left;
}

static UniformFillPlan AnalyzeUniformFill(const Program& program) {
	if (program.stage != ShaderType::Compute || program.blocks.empty() || program.info.uses_dma ||
	    !program.info.samplers.empty()) {
		return {};
	}
	std::unordered_set<const Block*> visited;
	const auto*                      block = program.blocks.front();
	const Inst*                      store = nullptr;
	for (;;) {
		if (block == nullptr || !visited.insert(block).second) return {};
		for (const auto& inst: *block) {
			if (AddressOpcodeInfoOf(inst.GetOpcode()).access != AddressAccess::None) return {};
			if (!inst.MayHaveSideEffects()) continue;
			if (store != nullptr || (BufferAccessOf(inst.GetOpcode()) != BufferAccess::Write &&
			                         inst.GetOpcode() != ValueOpcode::ImageWrite))
				return {};
			store = &inst;
		}
		const auto& term = block->terminator;
		if (term.kind == CFG::TerminatorKind::Return) break;
		if (term.kind != CFG::TerminatorKind::Branch) return {};
		block = term.true_block;
	}
	if (store == nullptr || visited.size() != program.blocks.size()) return {};
	for (const auto& buffer: program.info.buffers) {
		if (buffer.read && (!buffer.scalar || buffer.written)) return {};
	}
	const auto& memory = program.memory_info.at(store->Flags<MemoryFlags>().index);
	UniformFillPlan result;
	result.fill.resource = memory.resource;
	Value data;
	Value guard(true);
	if (store->GetOpcode() == ValueOpcode::ImageWrite) {
		if (program.info.images.size() != 1 || memory.dmask != 1 || memory.data_bits != 32 ||
		    memory.image_has_mip || memory.image_sample_flags != 0 || memory.image_r128 ||
		    memory.image_dimension != Decoder::ImageDimension::Dim2DArray ||
		    store->Arg(3).Resolve() != Value(true)) return {};
		const auto& image = program.info.images[memory.resource];
		if (image.read || image.atomic || image.mip_mode != ImageMipMode::None) return {};
		const auto* address = store->Arg(1).ResolveInstruction();
		if (address == nullptr || address->GetOpcode() != ValueOpcode::MakeImageAddress) return {};
		for (uint32_t axis = 0; axis < 3; ++axis) {
			const auto index = FillIndex(address->Arg(axis), axis, guard);
			if (!index || (*index)[0] != 0) return {};
			if (axis < 2) {
				if ((*index)[1] != 1 || (*index)[2] == 0) return {};
			} else if ((*index)[1] != 0 || (*index)[2] != 1) {
				return {};
			}
			result.fill.group_stride[axis] = static_cast<uint32_t>((*index)[2]);
		}
		const auto* values = store->Arg(2).ResolveInstruction();
		if (values == nullptr || values->GetOpcode() != ValueOpcode::CompositeConstructU32x4)
			return {};
		result.fill.kind  = UniformFillKind::Image;
		result.fill.words = 1;
		data = values->Arg(0);
	} else {
		if (!program.info.images.empty()) return {};
		const auto           op = store->GetOpcode();
		constexpr std::array stores {ValueOpcode::StoreBufferU32, ValueOpcode::StoreBufferU32x2,
		                             ValueOpcode::StoreBufferU32x3, ValueOpcode::StoreBufferU32x4};
		const auto           store_op = std::ranges::find(stores, op);
		if (store_op == stores.end() || store->Arg(2).Resolve() != Value(0u) ||
		    store->Arg(3).Resolve() != Value(0u))
			return {};
		guard = store->Arg(5).Resolve();
		if (!IsFullDispatchPredicate(guard)) return {};
		if (!memory.formatted || memory.typed || !memory.idxen || memory.offen || memory.offset != 0 ||
		    memory.data_bits != 32 ||
		    memory.data_dwords != static_cast<uint32_t>(store_op - stores.begin() + 1))
			return {};
		const auto address = FillIndex(store->Arg(1), 0, guard);
		if (!address || (*address)[0] != 0 || (*address)[1] != 1 || (*address)[2] == 0) return {};
		result.fill.kind = UniformFillKind::Buffer;
		result.fill.group_stride[0] = static_cast<uint32_t>((*address)[2]);
		result.fill.words = memory.data_dwords;
		data = store->Arg(4);
	}
	data = ResolveActiveU32(data, guard);
	const auto*          vector = data.TryInstruction();
	constexpr std::array composites {ValueOpcode::CompositeConstructU32x2,
	                                 ValueOpcode::CompositeConstructU32x3,
	                                 ValueOpcode::CompositeConstructU32x4};
	if (result.fill.words > 1 &&
	    (vector == nullptr || vector->GetOpcode() != composites[result.fill.words - 2]))
		return {};
	for (uint32_t i = 0; i < result.fill.words; ++i) {
		const auto word = ResolveActiveU32(result.fill.words == 1 ? data : vector->Arg(i), guard);
		if (word.GetType() != Type::U32 ||
		    !ValidateRuntimeValue(program, word, RuntimeValueType::Integer))
			return {};
		result.values[i] = word;
	}
	return result;
}

std::vector<ResourceBlock> ResourceControlFlowOf(const Program& program, const std::function<Value(Value)>& predicate) {
	return ResourceControlFlow(program, predicate);
}

ResourcePlan ExtractResourcePlan(const Program& program) {
	ResourcePlan plan;
	plan.stage                      = program.stage;
	plan.shader_hash                = program.shader_hash;
	plan.user_data_base             = program.user_data_base;
	plan.user_data_count            = program.user_data_count;
	plan.info                       = program.info;
	plan.memory_info                = program.memory_info;
	plan.source_reads               = program.source_reads;
	plan.srt_plan_complete          = program.srt_plan_complete;
	plan.resource_tracking_complete = program.resource_tracking_complete;

	std::unordered_map<const Inst*, Inst*> cloned;
	std::function<Value(Value)>            Clone = [&](Value value) -> Value {
		value              = value.Resolve();
		const auto* source = value.TryInstruction();
		if (source == nullptr) {
			return value;
		}
		if (source->GetOpcode() == ValueOpcode::ReadConst) {
			const auto slot = source->Arg(1).Resolve();
			EXIT_IF(!slot.IsImmediate() || slot.GetType() != Type::U32 ||
			        slot.U32() >= program.srt_reads.size());
			return Clone(program.srt_reads[slot.U32()].value);
		}
		if (source->GetOpcode() == ValueOpcode::ReadFirstLane &&
		    ValidateRuntimeValue(program, source->Arg(0))) {
			// Uniform values need no new EXEC context; preserve their shared evaluation memo.
			return Clone(source->Arg(0));
		}
		if (source->GetOpcode() == ValueOpcode::Phi) {
			const auto invariant = ResolveInvariantPhi(program, value);
			if (!invariant.IsEmpty() && invariant != value) {
				return Clone(invariant);
			}
		}
		if (const auto found = cloned.find(source); found != cloned.end()) {
			return Value(found->second);
		}
		if (source->GetOpcode() == ValueOpcode::LoadBufferU32) {
			plan.requires_specialization_memory = true;
			plan.capture_specialization_reads   = true;
		}
		auto& target =
		    plan.value_storage.emplace_back(source->GetOpcode(), source->Flags<uint64_t>());
		if (source->GetOpcode() == ValueOpcode::LoadAddressU32 ||
		    source->GetOpcode() == ValueOpcode::ReadConstBuffer ||
		    source->GetOpcode() == ValueOpcode::LoadBufferU32) {
			target.SetFlags(SrtReadFlags {.index = source->Flags<MemoryFlags>().index});
		}
		cloned.emplace(source, &target);
		if (source->GetOpcode() == ValueOpcode::Phi) {
			for (size_t index = 0; index < source->NumArgs(); index++) {
				target.AddPhiOperand(nullptr, Clone(source->Arg(index)));
			}
		} else {
			for (size_t index = 0; index < source->NumArgs(); index++) {
				target.SetArg(index, Clone(source->Arg(index)));
			}
		}
		return Value(&target);
	};

	bool capture_indirect_reads = false;
	plan.descriptor_sources.reserve(program.descriptor_sources.size());
	for (const auto& source: program.descriptor_sources) {
		auto& target          = plan.descriptor_sources.emplace_back();
		target.dword_count    = source.dword_count;
		target.indirect_descriptor = source.indirect_descriptor;
		if (target.indirect_descriptor.has_value()) {
			plan.requires_specialization_memory = true;
			capture_indirect_reads = true;
			target.indirect_descriptor->key_count = Clone(target.indirect_descriptor->key_count);
			target.indirect_descriptor->selector_first =
			    Clone(target.indirect_descriptor->selector_first);
			target.indirect_descriptor->selector_mask =
			    Clone(target.indirect_descriptor->selector_mask);
		}
		for (uint32_t dword = 0; dword < source.dword_count; dword++) {
			target.dwords[dword] = Clone(source.dwords[dword]);
		}
	}
	plan.srt_reads.reserve(program.srt_reads.size());
	for (const auto& read: program.srt_reads) {
		plan.srt_reads.push_back({Clone(read.value), read.flat_offset});
	}
	// A proven uniform factor can decide a branch even when its other lanes are unknown.
	// Keep only that Boolean structure, never the varying shader dependency graph.
	Value unknown;
	bool capture_control_reads = false;
	std::unordered_map<const Inst*, Value> predicates;
	std::function<Value(Value)> ClonePredicate = [&](Value value) -> Value {
		value = value.Resolve();
		if (value.IsEmpty()) return {};
		const auto* inst = value.TryInstruction();
		if (inst == nullptr)
			return ValidateRuntimeValue(program, value, RuntimeValueType::Integer) ? value : Value {};
		// Preserve shared EXEC predicates, including subgraphs with no uniform factor.
		const auto [found, inserted] = predicates.try_emplace(inst);
		if (!inserted) return found->second;
		auto& result = found->second;
		if (ValidateRuntimeValue(program, value, RuntimeValueType::Integer)) {
			capture_control_reads |=
			    !ValidateRuntimeValue(program, value, RuntimeValueType::ImmutableInteger);
			return result = Clone(value);
		}
		const auto op = inst->GetOpcode();
		if (op == ValueOpcode::ConditionRef || op == ValueOpcode::LogicalNot) {
			const auto operand = ClonePredicate(inst->Arg(0));
			if (operand.IsEmpty() || op == ValueOpcode::ConditionRef) return result = operand;
			auto& node = plan.value_storage.emplace_back(op);
			node.SetArg(0, operand);
			return result = Value(&node);
		}
		if (op != ValueOpcode::LogicalAnd && op != ValueOpcode::LogicalOr) return {};
		auto left = ClonePredicate(inst->Arg(0));
		auto right = ClonePredicate(inst->Arg(1));
		if (left.IsEmpty() && right.IsEmpty()) return {};
		if (left.IsEmpty() || right.IsEmpty()) {
			if (unknown.IsEmpty()) unknown = Value(&plan.value_storage.emplace_back(ValueOpcode::UndefU1));
			if (left.IsEmpty()) left = unknown;
			if (right.IsEmpty()) right = unknown;
		}
		auto& node = plan.value_storage.emplace_back(op);
		node.SetArg(0, left);
		node.SetArg(1, right);
		return result = Value(&node);
	};
	plan.control_flow = ResourceControlFlow(program, ClonePredicate);
	plan.uniform_fill = AnalyzeUniformFill(program);
	for (uint32_t i = 0; i < plan.uniform_fill.fill.words; ++i) {
		plan.uniform_fill.values[i] = Clone(plan.uniform_fill.values[i]);
	}
	plan.capture_specialization_reads |=
	    capture_indirect_reads || capture_control_reads || !plan.source_reads.empty();
	if (plan.capture_specialization_reads) {
		// Every former flat alias now targets its retained raw read. Mark those reads
		// on the normalized graph, so discarded Phi/EXEC dependencies stay discarded.
		std::unordered_set<const Inst*> planned_reads;
		for (const auto& read: plan.srt_reads) {
			auto* inst = read.value.ResolveInstruction();
			EXIT_IF(inst == nullptr || (inst->GetOpcode() != ValueOpcode::LoadAddressU32 &&
			                           inst->GetOpcode() != ValueOpcode::ReadConstBuffer));
			planned_reads.insert(inst);
		}
		for (const auto& source: plan.descriptor_sources) {
			if (!source.indirect_descriptor.has_value()) continue;
			const auto& indirect = *source.indirect_descriptor;
			if (indirect.selector)
				MarkCleanReads(planned_reads, Source(plan, indirect.selector->source));
			MarkCleanReads(planned_reads, nullptr, indirect.selector_mask);
			MarkCleanReads(planned_reads, nullptr, indirect.selector_first);
			MarkCleanReads(planned_reads, nullptr, indirect.key_count);
			if (indirect.sources.empty()) {
				MarkCleanReads(planned_reads, Source(plan, indirect.table_source));
			} else {
				for (const auto candidate: indirect.sources) {
					MarkCleanReads(planned_reads, Source(plan, candidate));
				}
			}
		}
		// Clean writable addresses prove that shader writes cannot overlap captured reads.
		for (const auto& buffer: plan.info.buffers) {
			if (buffer.written) MarkCleanReads(planned_reads, Source(plan, buffer.source));
		}
		for (const auto& image: plan.info.images) {
			if (image.written) MarkCleanReads(planned_reads, Source(plan, image.source));
		}
	}
	// Captured reads (indirect tables, a linked call's function code) cannot be proven disjoint from writes whose
	// guest ranges are unknown at record time.
	if (capture_indirect_reads || !plan.source_reads.empty())
		plan.resource_tracking_complete &= !program.has_address_writes;
	if (SrtTapeEnabled()) plan.srt_tape = BuildSrtTape(plan);
	return plan;
}

// Diagnostic: names the line of a silent materialization failure (first 32).
static bool MaterializeFailAt(int line) {
	if (t_ahead.log != nullptr) return false;
	static std::atomic_uint count {0};
	if (count.fetch_add(1) < 32u)
		std::fprintf(stderr, "shader resource materialization returned false at line %d\n", line);
	return false;
}

void SetPlainReadHook(PlainReadHook hook) {
	g_plain_read_hook.store(hook, std::memory_order_relaxed);
}

void ReadPlainMemory(void* data, uint64_t vaddr, uint64_t size) {
	if (const auto hook = g_plain_read_hook.load(std::memory_order_relaxed); hook != nullptr) {
		hook(data, vaddr, size);
	} else {
		std::memcpy(data, reinterpret_cast<const void*>(vaddr), size);
	}
}

void SetOrdinaryReadHook(OrdinaryReadHook hook) {
	g_ordinary_read_hook.store(hook, std::memory_order_relaxed);
}

void SetOrdinaryReadSessionHook(OrdinaryReadSessionHook hook) {
	g_ordinary_read_session_hook.store(hook, std::memory_order_relaxed);
}

void SetAheadReadHook(AheadReadHook hook) {
	g_ahead_read_hook.store(hook, std::memory_order_relaxed);
}

void SetFastVerifyHook(FastVerifyHook hook) {
	g_fast_verify_hook.store(hook, std::memory_order_relaxed);
}

namespace {

// The entries of the current log by address, size: its repeated reads (ReadLog::Clear starts a new epoch).
struct AheadSeen {
	uint32_t epoch = 0;
	uint32_t index = 0;
};
thread_local std::array<AheadSeen, 1024> t_ahead_seen {};
thread_local uint32_t                    t_ahead_epoch = 1;

// The one read of the ahead path (see MaterializeResourcesAhead): through the installed hook, logged. Any
// refusal marks the thread's call as failed and zero-fills the destination; the call goes on with zeros until
// it returns (its result is discarded), so the readers need no way to unwind it. Returns whether the read is
// good. Only runs inside an ahead call (t_ahead.log is set).
bool AheadRead(ReadLog::Kind kind, uint64_t address, void* data, uint64_t size) {
	auto&      state = t_ahead;
	auto&      log   = *state.log;
	const auto fail  = [&](AheadRefusal reason) {
		if (!state.failed) {
			state.failed = true;
			state.reason = reason;
		}
		std::memset(data, 0, size);
		return false;
	};
	if (size == 0u) {
		// The GPU thread's strict reader fails on an empty span without reading; the log keeps that outcome.
		if (kind == ReadLog::Kind::Strict && !state.failed) {
			log.entries.push_back({.address = address, .kind = kind});
		}
		return kind == ReadLog::Kind::Ordinary;
	}
	if (state.failed) {
		// Refused already: no more hook calls or log entries for a result that is thrown away.
		std::memset(data, 0, size);
		return false;
	}
	if (size > ReadLog::MaxCallBytes || log.bytes.size() + size > ReadLog::MaxTotalBytes) {
		return fail(AheadRefusal::Cap);
	}
	const auto hook   = g_ahead_read_hook.load(std::memory_order_relaxed);
	const auto status = hook != nullptr ? hook(address, data, size) : AheadReadStatus::Backing;
	ReadLog::Entry entry {.address = address,
	                      .size    = static_cast<uint32_t>(size),
	                      .first   = static_cast<uint32_t>(log.bytes.size()),
	                      .kind    = kind,
	                      .ok      = status == AheadReadStatus::Ok,
	                      .cls     = t_srt_read_class};
	if (status != AheadReadStatus::Ok) {
		entry.first = 0;
		log.entries.push_back(entry);
		return fail(status == AheadReadStatus::GpuDirty ? AheadRefusal::GpuDirty : AheadRefusal::Backing);
	}
	const auto* source = static_cast<const uint8_t*>(data);
	// The same read with the same bytes is checked by the entry that holds them.
	auto& seen = t_ahead_seen[((address >> 2u) ^ (address >> 13u) ^ size) & (t_ahead_seen.size() - 1u)];
	if (seen.epoch == t_ahead_epoch && seen.index < log.entries.size()) {
		const auto& known = log.entries[seen.index];
		if (known.address == address && known.size == size && known.kind == kind &&
		    std::memcmp(log.bytes.data() + known.first, source, size) == 0) {
			return true;
		}
	}
	seen = {t_ahead_epoch, static_cast<uint32_t>(log.entries.size())};
	log.entries.push_back(entry);
	log.bytes.insert(log.bytes.end(), source, source + size);
	return true;
}

// Restores the thread's scratch slot and ahead state on every exit of MaterializeResourcesAhead.
class AheadScope {
public:
	explicit AheadScope(ReadLog& log): m_previous_slot(t_resource_scratch_slot) {
		t_resource_scratch_slot = t_ahead_scratch_slot;
		t_ahead                 = {.log = &log};
	}
	~AheadScope() {
		t_ahead                 = {};
		t_resource_scratch_slot = m_previous_slot;
	}
	AheadScope(const AheadScope&)            = delete;
	AheadScope& operator=(const AheadScope&) = delete;

private:
	uint32_t m_previous_slot;
};

} // namespace

bool AheadReadMemory(void*, uint64_t address, std::span<uint32_t> values) {
	if (t_ahead.log == nullptr) return false;
	return AheadRead(ReadLog::Kind::Strict, address, values.data(), values.size_bytes());
}

void ReadOrdinaryMemory(uint64_t vaddr, void* data, uint64_t size) {
	// The ahead reader sits below every caller's rules (null page, fused runs, range merging): it only
	// replaces where the bytes come from.
	if (t_ahead.log != nullptr) {
		AheadRead(ReadLog::Kind::Ordinary, vaddr, data, size);
		return;
	}
	if (const auto hook = g_ordinary_read_hook.load(std::memory_order_relaxed); hook == nullptr || !hook(vaddr, data, size)) {
		ReadPlainMemory(data, vaddr, size);
	}
}

// The null share journal: PERF null_share_journal (ReportNullShareJournalPerf).
namespace {
struct NullShareJournalCounters {
	std::atomic<uint64_t> programs {0};      // programs given the journal's last live values
	std::atomic<uint64_t> bindings {0};      // their formatted bindings with a journal value
	std::atomic<uint64_t> journal_multi {0}; // of them: more than one live value in the journal (the later record's taken)
	std::atomic<uint64_t> seeded_shares {0}; // =1: null V#s that took a journal value (no live V# of the session yet)
	std::atomic<uint64_t> checks {0};        // first live V# of the session of a binding with a journal value
	std::atomic<uint64_t> mismatches {0};    // of them: another format or swizzle than the journal's
	std::atomic<uint64_t> live_changes {0};  // live V#s whose format or swizzle differs from the binding's last live one
	std::atomic<uint64_t> null_variants {0}; // permutations compiled with a null formatted buffer's own specialization
	std::atomic<uint64_t> avoidable {0};     // of them: the journal had a live value for that buffer
};
NullShareJournalCounters g_null_share_journal;
} // namespace

// A null formatted V# of binding `i` about to take the binding's last live value (under the shared state's lock, or on
// its only thread): a journal value is taken until the session has a live V# of the binding.
static void NoteNullShareJournalNull(const ResourcePlan::NullFormattedLive& shared, size_t i) {
	if (i >= shared.journal.size() || shared.journal[i][0] == 0u || shared.journal_checked[i] != 0u) return;
	g_null_share_journal.seeded_shares.fetch_add(1, std::memory_order_relaxed);
}

// A live formatted V# of binding `i` about to set the binding's last live value `previous` (same lock): a change of
// format or swizzle is counted (a binding with two live formats has two equivalent targets for its null V#s); the
// session's first one of a seeded binding is checked against the journal's value.
static void NoteNullShareJournalLive(const ResourcePlan& program, ResourcePlan::NullFormattedLive& shared, size_t i,
                                     const std::array<uint32_t, 2>& previous, uint32_t format, uint32_t swizzle) {
	const std::array<uint32_t, 2> value {format, swizzle};
	if (previous[0] != 0u && previous != value && (i >= shared.journal_checked.size() || shared.journal_checked[i] != 0u)) {
		g_null_share_journal.live_changes.fetch_add(1, std::memory_order_relaxed);
	}
	if (i >= shared.journal.size() || shared.journal_checked[i] != 0u) return;
	shared.journal_checked[i] = 1u;
	if (shared.journal[i][0] == 0u) return;
	g_null_share_journal.checks.fetch_add(1, std::memory_order_relaxed);
	if (shared.journal[i] != value && g_null_share_journal.mismatches.fetch_add(1, std::memory_order_relaxed) < 16u) {
		std::fprintf(stderr, "NULL_SHARE_JOURNAL_MISMATCH hash=0x%016llx buffer=%zu journal=%u/0x%x live=%u/0x%x\n",
		             static_cast<unsigned long long>(program.shader_hash), i, shared.journal[i][0], shared.journal[i][1],
		             format, swizzle);
	}
}

void SeedNullFormattedShare(const ResourcePlan& program, std::span<const std::array<uint32_t, 2>> journal,
                            std::span<const uint8_t> journal_multi) {
	if (program.null_formatted_live == nullptr) {
		return;
	}
	auto&            shared = *program.null_formatted_live;
	std::scoped_lock lock(shared.mutex);
	const size_t     buffers = program.info.buffers.size();
	uint64_t         seeded  = 0;
	uint64_t         multi   = 0;
	shared.journal.assign(buffers, {0u, 0u});
	shared.journal_checked.assign(buffers, 0u);
	if (shared.live.size() < buffers) shared.live.resize(buffers, {0u, 0u});
	for (size_t i = 0; i < buffers && i < journal.size(); i++) {
		if (!program.info.buffers[i].formatted || journal[i][0] == 0u) continue;
		shared.journal[i] = journal[i];
		seeded++;
		if (i < journal_multi.size() && journal_multi[i] != 0u) multi++;
		if (shared.live[i][0] == 0u) shared.live[i] = journal[i];
	}
	if (seeded == 0) {
		shared.journal.clear();
		shared.journal_checked.clear();
		return;
	}
	g_null_share_journal.programs.fetch_add(1, std::memory_order_relaxed);
	g_null_share_journal.bindings.fetch_add(seeded, std::memory_order_relaxed);
	g_null_share_journal.journal_multi.fetch_add(multi, std::memory_order_relaxed);
}

void NoteNullShareVariant(const ResourcePlan& program, const ResourceSpecialization& specialization) {
	if (program.null_formatted_live == nullptr) return;
	auto&            shared = *program.null_formatted_live;
	std::scoped_lock lock(shared.mutex);
	bool             null_variant = false;
	bool             avoidable    = false;
	for (size_t i = 0; i < program.info.buffers.size() && i < specialization.buffers.size(); i++) {
		const auto& buffer = specialization.buffers[i];
		if (!program.info.buffers[i].formatted || !buffer.zero_stride_oob ||
		    buffer.descriptor_format != Prospero::BufferFormat::kInvalid) {
			continue;
		}
		null_variant = true;
		avoidable    = avoidable || (i < shared.journal.size() && shared.journal[i][0] != 0u);
	}
	if (null_variant) g_null_share_journal.null_variants.fetch_add(1, std::memory_order_relaxed);
	if (avoidable) g_null_share_journal.avoidable.fetch_add(1, std::memory_order_relaxed);
}

void ReportNullShareJournalPerf() {
	const auto n = [](const std::atomic<uint64_t>& value) {
		return static_cast<unsigned long long>(value.load(std::memory_order_relaxed));
	};
	const auto& c = g_null_share_journal;
	std::fprintf(stderr,
	             "PERF null_share_journal programs=%llu bindings=%llu journal_multi=%llu seeded_shares=%llu "
	             "checks=%llu mismatches=%llu live_changes=%llu null_variants=%llu avoidable=%llu\n",
	             n(c.programs), n(c.bindings), n(c.journal_multi), n(c.seeded_shares), n(c.checks), n(c.mismatches),
	             n(c.live_changes), n(c.null_variants), n(c.avoidable));
}

// `tape` (when set) evaluates the plan's values instead of SrtWalker; `quiet` mutes its diagnostics.
// `raw_reads` keeps every ordinary read as its own captured range (the verifier's legacy side).
static bool MaterializeResourcesImpl(const ResourcePlan& program, const SrtRuntime& runtime,
                                     ResourceSnapshot& snapshot, ResourceSpecialization& specialization,
                                     const SrtTape* tape, bool quiet, bool raw_reads = false) {
	if (!program.resource_tracking_complete ||
	    ((program.requires_specialization_memory || !program.source_reads.empty()) &&
	     runtime.read_specialization_memory == nullptr)) {
		return MaterializeFailAt(__LINE__);
	}
	if (RecordsNullShare()) program.ThreadScratch().null_share_kinds.assign(program.info.buffers.size(), 0u);
	const bool capture_reads = program.capture_specialization_reads || !program.source_reads.empty();
	auto& reads = snapshot.specialization_reads;
	// A linked call's function code is a captured read of every materialization.
	reads = program.source_reads;
	ReadCapture capture {runtime, reads};
	capture.merge = !raw_reads;
	SrtRuntime observed = runtime;
	if (capture_reads) {
		observed.userdata = &capture;
		observed.read_specialization_memory = runtime.read_specialization_memory != nullptr
		                                         ? CaptureStrictRead : nullptr;
		observed.read_memory = CaptureOrdinaryRead;
	}
	// Ordinary reads of this call may rely on facts the call itself cannot change (the hook's page cache).
	// An ahead call never reaches the hook (its reader replaces it), so it needs no session.
	const OrdinaryReadSession read_session(t_ahead.log == nullptr);
	// A tape evaluates everything itself: the legacy walkers (and their evaluation-context acquisition)
	// are needed only when it is absent.
	std::optional<SrtWalker> clean_storage;
	std::optional<SrtWalker> walker_storage;
	SrtWalker*               clean  = nullptr;
	SrtWalker*               walker = nullptr;
	if (tape == nullptr) {
		clean_storage.emplace(program, CleanRuntime(observed));
		walker_storage.emplace(program, observed,
		                       capture_reads || program.requires_specialization_memory ? &*clean_storage
		                                                                              : nullptr);
		clean  = &*clean_storage;
		walker = &*walker_storage;
	}
	std::optional<SrtTapeWalker> tape_clean;
	std::optional<SrtTapeWalker> tape_walker;
	if (tape != nullptr) {
		// Fusing reads replaces the per-DWORD calls of the built-in ordinary reader, so a runtime with its
		// own ordinary reader keeps them.
		const bool fuse = runtime.read_memory == nullptr;
		tape_clean.emplace(*tape, CleanRuntime(observed), 1u, nullptr, quiet);
		tape_walker.emplace(*tape, observed, 0u,
		                    capture_reads || program.requires_specialization_memory ? &*tape_clean : nullptr,
		                    quiet, fuse, capture_reads ? &reads : nullptr);
	}
	// The indirect-descriptor path uses the same clean evaluator (and memo) as the fill and written
	// descriptor reads above and below.
	IndirectEvaluator clean_indirect(clean, tape != nullptr ? &*tape_clean : nullptr);
	if (!(tape != nullptr ? tape_walker->RefreshFlatBuffer(program, snapshot.flattened_srt)
	                      : walker->RefreshFlatBuffer(snapshot.flattened_srt))) {
		return MaterializeFailAt(__LINE__);
	}
	const auto active = std::span<const uint8_t>(program.ThreadScratch().active_sources);
	snapshot.uniform_fill = {};
	snapshot.bindless_heaps.clear();
	const auto& fill = program.uniform_fill;
	const auto words = fill.fill.words;
	std::array<uint32_t, 4> stored {};
	bool uniform_fill = words != 0;
	for (uint32_t i = 0; i < words && uniform_fill; ++i) {
		uniform_fill = (tape != nullptr ? tape_clean->EvaluateFill(i, stored[i])
		                                : clean->Evaluate(fill.values[i], stored[i])) &&
		               stored[i] == stored[0];
	}
	if (uniform_fill) {
		snapshot.uniform_fill = fill.fill;
		snapshot.uniform_fill.value = stored[0];
	}
	const auto evaluate = [&](uint32_t source, DescriptorValue& value, bool written = false) {
		if (source >= program.descriptor_sources.size()) {
			return MaterializeFailAt(__LINE__);
		}
		if (active.empty() || active[source]) {
			if (tape != nullptr) {
				return (capture_reads && written ? *tape_clean : *tape_walker).EvaluateDescriptor(source, value);
			}
			return (capture_reads && written ? *clean : *walker).EvaluateDescriptor(source, value);
		}
		value = {};
		value.dword_count = program.descriptor_sources[source].dword_count;
		return true;
	};
	snapshot.buffers.resize(program.info.buffers.size());
	specialization.buffers.assign(program.info.buffers.size(), {});
	for (uint32_t i = 0; i < snapshot.buffers.size(); ++i) {
		const auto base_index =
		    i < program.info.buffers.size() ? i : specialization.buffers[i].indirect_root;
		if (base_index >= program.info.buffers.size()) return MaterializeFailAt(__LINE__);
		const auto& base = program.info.buffers[base_index];
		if (i < program.info.buffers.size()) {
			const auto* source = Source(program, base.source);
			if (source == nullptr) return MaterializeFailAt(__LINE__);
			if (source->indirect_descriptor.has_value()) {
				snapshot.buffers[i] = {.dword_count = 4u};
				if (active.empty() || active[base.source]) {
					if (!MaterializeIndirectDescriptor(
					        program, *source->indirect_descriptor, base.source, i, 4u, observed, clean_indirect, snapshot,
					        snapshot.buffers, specialization.buffers, ShaderInfo::MaxBuffers,
					        NormalizeIndirectStoreBuffer))
						return MaterializeFailAt(__LINE__);
				}
			} else if (!evaluate(base.source, snapshot.buffers[i], base.written)) {
				return MaterializeFailAt(__LINE__);
			}
		}
		auto&                descriptor_value = snapshot.buffers[i];
		ShaderBufferResource descriptor;
		if (!DecodeBufferDescriptor(descriptor_value, descriptor)) {
			return SpecializationFail(fmt::format("buffer descriptor {} has invalid width", i));
		}
		if (descriptor.Type() != 0) {
			descriptor_value.dwords.fill(0);
			descriptor = {};
		}
		auto       packed_stride = descriptor.PackedStride();
		const auto stride        = packed_stride & 0x3fffu;
		const bool swizzle       = stride != 0u && ((packed_stride >> 14u) & 1u) != 0u;
		if (stride == 0u) {
			packed_stride &= ~((1u << 14u) | (3u << 16u));
		} else if (!swizzle) {
			packed_stride &= ~(3u << 16u);
			// The stride is read from shader data at run time (bit 31 marks it), so buffers that
			// differ only in record size share one pipeline.
			packed_stride = (packed_stride & ~0x3fffu) | (1u << 31u) | 1u;
		}
		auto& buffer         = specialization.buffers[i];
		buffer.packed_stride = packed_stride;
		buffer.descriptor_format =
		    base.formatted ? descriptor.Format() : Prospero::BufferFormat::kInvalid;
		buffer.descriptor_swizzle = base.formatted ? descriptor.DstSelXYZW() : DstSel(4, 5, 6, 7);
		buffer.zero_stride_oob    = descriptor.OutOfBounds() == 0u && stride == 0u;
		if (buffer.zero_stride_oob && !base.formatted) {
			// A null raw buffer reads its (zero) stride at run time too and every access goes out
			// of bounds there, so null and live buffers share one pipeline.
			buffer.packed_stride   = (packed_stride & ~0x3fffu) | (1u << 31u) | 1u;
			buffer.zero_stride_oob = false;
		}
		// The null formatted share: an all-zero V# of a formatted buffer takes the specialization of the
		// binding's last live V#; the emitter returns zeros (the null V#'s DST_SEL 0) when the run-time stride is 0.
		// Other zero-stride V#s keep their own specialization (their result depends on their own format and DST_SEL).
		if (base.formatted && i < program.info.buffers.size() &&
		    program.null_formatted_live != nullptr && RecordsNullShare()) {
			// The resolve thread: only what the GPU thread applies in order (ApplyNullFormattedShare).
			auto& kinds = program.ThreadScratch().null_share_kinds;
			if (kinds.size() < program.info.buffers.size()) kinds.resize(program.info.buffers.size(), 0u);
			const bool all_zero = std::ranges::all_of(std::span(descriptor_value.dwords).first(4),
			                                          [](uint32_t word) { return word == 0u; });
			if (buffer.zero_stride_oob && all_zero) {
				kinds[i] = 1u;
			} else if (!buffer.zero_stride_oob && (buffer.packed_stride >> 31u) != 0u &&
			           buffer.descriptor_format != Prospero::BufferFormat::kInvalid) {
				kinds[i] = 2u;
			}
			continue;
		}
		if (base.formatted && i < program.info.buffers.size() &&
		    program.null_formatted_live != nullptr) {
			auto&            shared = *program.null_formatted_live;
			std::scoped_lock lock(shared.mutex);
			if (shared.live.size() < program.info.buffers.size()) shared.live.resize(program.info.buffers.size(), {0u, 0u});
			const bool all_zero = std::ranges::all_of(std::span(descriptor_value.dwords).first(4),
			                                          [](uint32_t word) { return word == 0u; });
			if (buffer.zero_stride_oob && all_zero && !shared.journal.empty()) {
				NoteNullShareJournalNull(shared, i); // the null share journal
			}
			if (buffer.zero_stride_oob && all_zero && shared.live[i][0] != 0u) {
				buffer.packed_stride      = (packed_stride & ~0x3fffu) | (1u << 31u) | 1u;
				buffer.zero_stride_oob    = false;
				buffer.descriptor_format  = static_cast<Prospero::BufferFormat>(shared.live[i][0]);
				buffer.descriptor_swizzle = shared.live[i][1];
			} else if (!buffer.zero_stride_oob && (buffer.packed_stride >> 31u) != 0u &&
			           buffer.descriptor_format != Prospero::BufferFormat::kInvalid) {
				NoteNullShareJournalLive(program, shared, i, shared.live[i], static_cast<uint32_t>(buffer.descriptor_format),
				                         buffer.descriptor_swizzle);
				shared.live[i] = {static_cast<uint32_t>(buffer.descriptor_format), buffer.descriptor_swizzle};
			}
			continue;
		}
		// KYTY_DIAG_NULL_FORMATTED_STICKY=1: logged APPROXIMATION. A null formatted buffer keeps
		// the format and swizzle this binding last had while live; its zero run-time stride makes
		// every access out of bounds (ONE-swizzled components then read 1 instead of 0).
		if (NullFormattedStickyEnabled() && base.formatted && i < program.info.buffers.size()) {
			auto& sticky = program.ThreadScratch().sticky_buffer_formats;
			if (sticky.size() < program.info.buffers.size())
				sticky.resize(program.info.buffers.size(), {0u, 0u});
			if (buffer.zero_stride_oob && sticky[i][0] != 0u) {
				static std::atomic_uint logged {0};
				if (logged.fetch_add(1) < 8u)
					std::fprintf(stderr, "DIAG_NULL_FORMATTED_STICKY APPROXIMATION buffer=%u format=%u\n", i,
					             sticky[i][0]);
				buffer.packed_stride      = (packed_stride & ~0x3fffu) | (1u << 31u) | 1u;
				buffer.zero_stride_oob    = false;
				buffer.descriptor_format  = static_cast<Prospero::BufferFormat>(sticky[i][0]);
				buffer.descriptor_swizzle = sticky[i][1];
			} else if (!buffer.zero_stride_oob && (buffer.packed_stride >> 31u) != 0u &&
			           buffer.descriptor_format != Prospero::BufferFormat::kInvalid) {
				sticky[i] = {static_cast<uint32_t>(buffer.descriptor_format), buffer.descriptor_swizzle};
			}
		}
	}
	snapshot.images.resize(program.info.images.size());
	specialization.images.resize(program.info.images.size());
	for (uint32_t i = 0; i < program.info.images.size(); ++i) {
		const auto& image = program.info.images[i];
		specialization.images[i] = {
		    .numeric_class = image.numeric_class,
		    .dimension = image.dimension,
		    .mip_count = image.mip_count,
		    .conversion_format = image.conversion_format,
		    .shader_swizzle = image.shader_swizzle,
		    .indirect_root = image.indirect_root,
		    .indirect_mapping_offset = image.indirect_mapping_offset,
		    .indirect_search_iterations = image.indirect_search_iterations,
		    .cube = image.cube,
		};
		const auto* source = Source(program, image.source);
		if (source == nullptr) {
			return MaterializeFailAt(__LINE__);
		}
		if (source->indirect_descriptor.has_value()) {
			snapshot.images[i] = {.dword_count = 8u};
			if (!active.empty() && !active[image.source]) {
				continue;
			}
			const auto& indirect = *source->indirect_descriptor;
			const bool bounded_table = IsBoundedDescriptorTable(program, indirect);
			if (!MaterializeIndirectDescriptor(
			        program, indirect, image.source, i, 8u, observed, clean_indirect, snapshot, snapshot.images,
			        specialization.images, UINT32_MAX, [&](DescriptorValue& value) {
				        // R128 resources ignore descriptor words 4..7; table bytes there belong to the next field.
				        if (image.r128) std::fill(value.dwords.begin() + 4, value.dwords.begin() + 8, 0u);
				        // A broad heap also contains resources for other typed image operations.
				        if (NullImageDescriptor(value) || !ValidImageDescriptor(value, image.r128) ||
				            (bounded_table && DescriptorDimension(value, image.dimension) != image.dimension))
					        value.dwords.fill(0u);
				        return true;
			        })) {
				return MaterializeFailAt(__LINE__);
			}
		} else {
			if (!evaluate(image.source, snapshot.images[i], image.written)) {
				return MaterializeFailAt(__LINE__);
			}
			if (!ValidImageDescriptor(snapshot.images[i], image.r128)) {
				snapshot.images[i].dwords.fill(0);
			}
		}
	}
	snapshot.samplers.resize(program.info.samplers.size());
	specialization.samplers.assign(program.info.samplers.size(), {});
	for (uint32_t i = 0; i < program.info.samplers.size(); ++i) {
		const auto* sampler_source = Source(program, program.info.samplers[i].source);
		const auto* indirect_sampler = sampler_source != nullptr && sampler_source->indirect_descriptor.has_value()
		                                   ? &*sampler_source->indirect_descriptor
		                                   : nullptr;
		// A table-keyed sampler lists no candidates: its children come from the table's records. A finite one
		// (TryMakeFiniteDescriptor) lists its literal candidates, each already a sampler of its own (AddSampler), so its
		// root only reads the first of them and stays out of the table bookkeeping (no indirect_root).
		if (indirect_sampler != nullptr && indirect_sampler->sources.empty()) {
			snapshot.samplers[i] = {.dword_count = 4u};
			if (program.info.samplers[i].gather_lod) {
				return SpecializationFail("explicit-LOD gather with a GPU-selected sampler");
			}
			if (!active.empty() && !active[program.info.samplers[i].source]) {
				continue;
			}
			if (!MaterializeIndirectDescriptor(
			        program, *indirect_sampler, program.info.samplers[i].source, i, 4u, observed,
			        clean_indirect, snapshot,
			        snapshot.samplers, specialization.samplers, ShaderInfo::MaxSamplers,
			        [](DescriptorValue&) { return true; })) {
				return MaterializeFailAt(__LINE__);
			}
			continue;
		}
		if (!evaluate(indirect_sampler != nullptr ? indirect_sampler->sources[0] : program.info.samplers[i].source,
		              snapshot.samplers[i])) {
			return MaterializeFailAt(__LINE__);
		}
		if (program.info.samplers[i].gather_lod) {
			const auto control = snapshot.samplers[i].dwords[2];
			const auto filter = (control >> 26u) & 3u;
			// MipNone always selects the base level. Explicit point gathers currently require
			// encoded-zero primary and secondary bias; linear primary-mip selection is unsupported.
			if (filter > 1u || (filter == 1u && (control & 0xfffffu) != 0u)) {
				return SpecializationFail(
				    "explicit-LOD gather requires mip filtering None or Point with zero LOD biases");
			}
		}
	}
	if (!MaterializeIndirectRanges(program, observed, clean_indirect, active, snapshot)) {
		return MaterializeFailAt(__LINE__);
	}
	snapshot.user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
	return BuildResourceSpecialization(program, snapshot, specialization);
}

// KYTY_FIX_SRT_TAPE_VERIFY=1: materializes with the tape into scratch copies, then with SrtWalker
// into the real outputs, and reports any difference (including the order of captured reads).
static bool MaterializeResourcesVerified(const ResourcePlan& program, const SrtRuntime& runtime,
                                         ResourceSnapshot& snapshot, ResourceSpecialization& specialization,
                                         const SrtTape& tape) {
	ResourceSnapshot       tape_snapshot       = snapshot;
	ResourceSpecialization tape_specialization = specialization;
	auto&                  scratch             = program.ThreadScratch();
	const auto             sticky              = scratch.sticky_buffer_formats;
	const bool tape_ok = MaterializeResourcesImpl(program, runtime, tape_snapshot, tape_specialization, &tape, true);
	const auto tape_active        = scratch.active_sources;
	scratch.sticky_buffer_formats = sticky;
	// The walker side keeps one captured range per ordinary read (the unfused, unmerged reference); both
	// sides are then reduced to the same normal form (adjacent ranges merged in order), which is what the
	// tape's fused and merging captures produce. Merging is associative, so any difference left is a
	// different set, order or extent of reads.
	const bool ok = MaterializeResourcesImpl(program, runtime, snapshot, specialization, nullptr, false, true);
	MergeSrtReads(tape_snapshot.specialization_reads);
	MergeSrtReads(snapshot.specialization_reads);
	const char* what = nullptr;
	if (tape_ok != ok) {
		what = "result";
	} else if (ok) {
		if (tape_snapshot.flattened_srt != snapshot.flattened_srt) what = "flattened_srt";
		else if (tape_snapshot.buffers != snapshot.buffers) what = "buffers";
		else if (tape_snapshot.images != snapshot.images) what = "images";
		else if (tape_snapshot.samplers != snapshot.samplers) what = "samplers";
		else if (tape_snapshot.uniform_fill != snapshot.uniform_fill) what = "uniform_fill";
		else if (!std::ranges::equal(tape_snapshot.bindless_heaps, snapshot.bindless_heaps,
		                             [](const BindlessHeapUse& a, const BindlessHeapUse& b) {
			                             return a.image == b.image && a.base == b.base && a.size == b.size &&
			                                    a.mapping_offset == b.mapping_offset && a.table_base == b.table_base &&
			                                    a.record_bytes == b.record_bytes && a.table_bytes == b.table_bytes;
		                             }))
			what = "bindless_heaps";
		else if (tape_snapshot.indirect_ranges != snapshot.indirect_ranges ||
		         tape_snapshot.indirect_ranges_complete != snapshot.indirect_ranges_complete ||
		         tape_snapshot.indirect_writes_complete != snapshot.indirect_writes_complete)
			what = "indirect_ranges";
		else if (tape_active != scratch.active_sources) what = "active_sources";
		else if (tape_snapshot.specialization_reads != snapshot.specialization_reads) what = "specialization_reads";
		else if (tape_snapshot.user_data != snapshot.user_data) what = "user_data";
		else if (tape_specialization != specialization) what = "specialization";
	}
	static std::atomic_uint checks {0};
	static std::atomic_uint mismatches {0};
	const auto              check_count = checks.fetch_add(1) + 1u;
	if (what != nullptr && mismatches.fetch_add(1) < 16u) {
		std::fprintf(stderr,
		             "SRT_TAPE_MISMATCH hash=0x%016llx what=%s tape_ok=%d walker_ok=%d flat=%zu/%zu reads=%zu/%zu\n",
		             static_cast<unsigned long long>(program.shader_hash), what, tape_ok ? 1 : 0, ok ? 1 : 0,
		             tape_snapshot.flattened_srt.size(), snapshot.flattened_srt.size(),
		             tape_snapshot.specialization_reads.size(), snapshot.specialization_reads.size());
	}
	if (check_count % 4096u == 0u) {
		uint64_t fused_runs = 0, fused_dwords = 0;
		SrtTapeFusionStats(fused_runs, fused_dwords);
		std::fprintf(stderr, "SRT_TAPE_VERIFY checks=%u mismatches=%u fused_runs=%llu fused_dwords=%llu\n",
		             check_count, mismatches.load(), static_cast<unsigned long long>(fused_runs),
		             static_cast<unsigned long long>(fused_dwords));
	}
	return ok;
}

bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          ResourceSnapshot& snapshot, ResourceSpecialization& specialization) {
	const SrtTape* tape =
	    program.srt_tape != nullptr && program.srt_tape->Matches(program) ? program.srt_tape.get() : nullptr;
	if (tape != nullptr && SrtTapeVerifyEnabled()) {
		return MaterializeResourcesVerified(program, runtime, snapshot, specialization, *tape);
	}
	return MaterializeResourcesImpl(program, runtime, snapshot, specialization, tape, false);
}

// Everything MaterializeResourcesImpl writes besides the caller's outputs is thread state or per-slot scratch:
//  * ResourcePlan::Scratch (active sources, walk state, material keys) and the SrtTape memos, generations and
//    Matches() cache are indexed by t_resource_scratch_slot, which AheadScope sets to 1;
//  * the fused-read counters are per slot; the other statics it touches are atomic, mutex-guarded or
//    initialized once (function-local statics of getenv flags);
//  * the plan's values, descriptor sources, info and the tape's nodes are only read;
//  * sticky_buffer_formats is state kept across calls and the reference walker numbers instructions inside
//    the shared Inst objects, so a flag or a plan that needs either is refused.
// ApplyNullFormattedShare's update of `live` (held under its lock).
static void ApplyNullShareTo(const ResourcePlan& program, const std::vector<uint8_t>& kinds,
                             std::vector<std::array<uint32_t, 2>>& live, ResourceSpecialization& specialization,
                             ResourcePlan::NullFormattedLive* journal_note) {
	if (live.size() < program.info.buffers.size()) live.resize(program.info.buffers.size(), {0u, 0u});
	const bool note = journal_note != nullptr;
	for (size_t i = 0; i < kinds.size() && i < specialization.buffers.size() && i < live.size(); i++) {
		auto& buffer = specialization.buffers[i];
		if (note && kinds[i] == 1u) NoteNullShareJournalNull(*journal_note, i);
		if (kinds[i] == 1u && live[i][0] != 0u) {
			buffer.packed_stride      = (buffer.packed_stride & ~0x3fffu) | (1u << 31u) | 1u;
			buffer.zero_stride_oob    = false;
			buffer.descriptor_format  = static_cast<Prospero::BufferFormat>(live[i][0]);
			buffer.descriptor_swizzle = live[i][1];
		} else if (kinds[i] == 2u) {
			if (note) {
				NoteNullShareJournalLive(program, *journal_note, i, live[i], static_cast<uint32_t>(buffer.descriptor_format),
				                         buffer.descriptor_swizzle);
			}
			live[i] = {static_cast<uint32_t>(buffer.descriptor_format), buffer.descriptor_swizzle};
		}
	}
}

void ApplyNullFormattedShare(const ResourcePlan& program, const std::vector<uint8_t>& kinds,
                             ResourceSpecialization& specialization) {
	if (program.null_formatted_live == nullptr || kinds.empty()) return;
	auto&            shared = *program.null_formatted_live;
	std::scoped_lock lock(shared.mutex);
	ApplyNullShareTo(program, kinds, shared.live, specialization, &shared);
}

bool MaterializeResourcesAhead(const ResourcePlan& program, const SrtRuntime& runtime,
                               ResourceSnapshot& snapshot, ResourceSpecialization& specialization,
                               ReadLog& log, AheadRefusal* refusal) {
	const size_t log_entries = log.entries.size();
	const size_t log_bytes   = log.bytes.size();
	const auto   finish      = [&](bool ok, AheadRefusal reason) {
		if (refusal != nullptr) *refusal = ok ? AheadRefusal::None : reason;
		if (!ok) {
			// The operation's other stages keep their reads.
			log.entries.resize(log_entries);
			log.bytes.resize(log_bytes);
		}
		return ok;
	};
	// Not reentrant; and only configurations whose reads and results the ahead path reproduces exactly.
	if (t_ahead.log != nullptr || !SrtTapeEnabled() || SrtTapeVerifyEnabled() ||
	    NullFormattedStickyEnabled() ||
	    (runtime.read_memory != nullptr && runtime.read_memory != &AheadReadMemory)) {
		return finish(false, AheadRefusal::Flag);
	}
	if (program.srt_tape == nullptr) return finish(false, AheadRefusal::NoTape);
	const AheadScope scope(log);
	if (!program.srt_tape->Matches(program)) return finish(false, AheadRefusal::NoTape);
	// Ordinary reads stay the built-in ones (fused runs and range merging apply as in a real call) and are
	// redirected inside ReadOrdinaryMemory; the strict reader is the ahead reader when the plan may use one.
	SrtRuntime ahead  = runtime;
	ahead.read_memory = nullptr;
	ahead.userdata    = nullptr;
	if (runtime.read_specialization_memory != nullptr) ahead.read_specialization_memory = AheadReadMemory;
	const bool ok = MaterializeResourcesImpl(program, ahead, snapshot, specialization,
	                                         program.srt_tape.get(), true);
	if (t_ahead.failed) return finish(false, t_ahead.reason);
	return finish(ok, AheadRefusal::Other);
}

void ReadLog::Clear() {
	entries.clear();
	bytes.clear();
	t_ahead_epoch++;
}

bool ReadGuestAhead(ReadLog& log, uint64_t address, void* data, uint64_t size) {
	if (t_ahead.log != nullptr) return false;
	const size_t log_entries = log.entries.size();
	t_ahead                  = {.log = &log};
	const bool ok            = AheadRead(ReadLog::Kind::Ordinary, address, data, size) && !t_ahead.failed;
	t_ahead                  = {};
	if (!ok) log.entries.resize(log_entries);
	return ok;
}

// One log entry through the GPU thread's readers.
static bool EntryUnchangedViaReaders(const ReadLog& log, const ReadLog::Entry& entry, const SrtRuntime& runtime) {
	std::array<uint32_t, ReadLog::MaxCallBytes / sizeof(uint32_t)> words;
	bool                                                           ok = false;
	if (entry.size <= sizeof(words)) {
		if (entry.kind == ReadLog::Kind::Strict) {
			ok = (entry.size % sizeof(uint32_t)) == 0u && runtime.read_specialization_memory != nullptr &&
			     runtime.read_specialization_memory(runtime.userdata, entry.address,
			                                        std::span<uint32_t>(words.data(), entry.size / sizeof(uint32_t)));
		} else {
			// The ordinary reader has no failure result.
			ReadOrdinaryMemory(entry.address, words.data(), entry.size);
			ok = true;
		}
	}
	return ok == entry.ok &&
	       (!ok || (entry.first <= log.bytes.size() && entry.size <= log.bytes.size() - entry.first &&
	                std::memcmp(words.data(), log.bytes.data() + entry.first, entry.size) == 0));
}

// The check with the kernel's fast verifier in front (see ResourceMaterialization.h). Every entry is verified
// exactly once, in log order: runs the verifier proves are compared in place, the rest one by one through the
// readers (which download GPU-dirty bytes and wait for pending readbacks as a real call does). The result is true
// exactly when reading every entry through the readers would say true for the same memory.
// KYTY_DIAG_READS_ASSUME_UNCHANGED=1 (ceiling test only, map 5.11 "ruch przez miasto": the check was ~16% of the GPU
// thread in the town): every check says "unchanged" without reading - a stale program or resource set may be used (the
// image may break; the game's own work, the draws per frame, stays). Read once.
static bool AssumeReadsUnchanged() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_DIAG_READS_ASSUME_UNCHANGED");
		const bool  set   = value != nullptr && value[0] == '1';
		if (set) std::fprintf(stderr, "DIAG_READS_ASSUME_UNCHANGED ceiling: read checks skipped\n");
		return set;
	}();
	return on;
}

bool ReadsUnchangedFast(const ReadLog& log, const SrtRuntime& runtime, FastAdoptStats* stats) {
	if (t_ahead.log != nullptr) return false;
	if (AssumeReadsUnchanged()) return true;
	t_srt_changed_entry = SIZE_MAX; // KYTY_TRACE_SRT_CLASSES
	FastAdoptStats  local;
	FastAdoptStats& counted = stats != nullptr ? *stats : local;
	++counted.calls;
	// Strict entries are compared with the backing store by the verifier: only valid when the runtime has a strict
	// reader at all (without one a strict entry cannot read as it did).
	const FastVerifyHook hook =
	    runtime.read_specialization_memory != nullptr ? g_fast_verify_hook.load(std::memory_order_relaxed) : nullptr;
	// A real call makes its ordinary reads inside this bracket; the verifier shares its page cache.
	const OrdinaryReadSession session(true);
	const size_t              count = log.entries.size();
	size_t                    index = 0;
	while (index < count) {
		if (hook != nullptr) {
			size_t     stop  = index;
			const auto found = hook(log, index, stop, counted);
			switch (found) {
				case FastVerifyResult::Unchanged: return true;
				case FastVerifyResult::Changed:
					t_srt_changed_entry = stop;
					return false;
				case FastVerifyResult::NeedsReader: break;
			}
			// Entries [index, stop) are proven; entry `stop` is for the readers. Progress is guaranteed.
			index = stop >= index && stop < count ? stop : index;
		}
		++counted.entries_slow;
		t_srt_read_class = log.entries[index].cls; // KYTY_TRACE_SRT_CLASSES (0 when the trace is off)
		const bool same  = EntryUnchangedViaReaders(log, log.entries[index], runtime);
		t_srt_read_class = 0u;
		if (!same) {
			t_srt_changed_entry = index;
			return false;
		}
		++index;
	}
	return true;
}

const char* AheadResultDifference(const ResourceSnapshot& ahead, const ResourceSpecialization& ahead_specialization,
                                  const std::vector<uint8_t>& ahead_active_sources, const ResourceSnapshot& real,
                                  const ResourceSpecialization& real_specialization,
                                  const std::vector<uint8_t>& real_active_sources) {
	// Adjacent captured reads are one range (see MergeSrtReads); the real snapshot stays as it is.
	auto ahead_reads = ahead.specialization_reads;
	auto real_reads  = real.specialization_reads;
	MergeSrtReads(ahead_reads);
	MergeSrtReads(real_reads);
	if (ahead.flattened_srt != real.flattened_srt) return "flattened_srt";
	if (ahead.buffers != real.buffers) return "buffers";
	if (ahead.images != real.images) return "images";
	if (ahead.samplers != real.samplers) return "samplers";
	if (ahead.uniform_fill != real.uniform_fill) return "uniform_fill";
	if (!std::ranges::equal(ahead.bindless_heaps, real.bindless_heaps,
	                        [](const BindlessHeapUse& a, const BindlessHeapUse& b) {
		                        return a.image == b.image && a.base == b.base && a.size == b.size &&
		                               a.mapping_offset == b.mapping_offset && a.table_base == b.table_base &&
		                               a.record_bytes == b.record_bytes && a.table_bytes == b.table_bytes;
	                        })) {
		return "bindless_heaps";
	}
	if (ahead.indirect_ranges != real.indirect_ranges ||
	    ahead.indirect_ranges_complete != real.indirect_ranges_complete ||
	    ahead.indirect_writes_complete != real.indirect_writes_complete) {
		return "indirect_ranges";
	}
	if (ahead_active_sources != real_active_sources) return "active_sources";
	if (ahead_reads != real_reads) return "specialization_reads";
	if (ahead.user_data != real.user_data) return "user_data";
	if (ahead_specialization != real_specialization) return "specialization";
	return nullptr;
}

// Upstream's image heap (imageHeap.cpp, 27dc7c281) classifies its entries with these two helpers. Our
// MaterializeIndirectDescriptor does not map heaps (SESJA-SYNC1: the resource system stays ours), so the renderer
// never creates a heap and nothing calls them on the draw path.
uint32_t ImageHeapKind(const ImageResource& root, DescriptorValue& value) {
	// Mirrors BuildResourceSpecialization for sampled roots: entries it would reject, and
	// entries that need shader-side conversion, read as null instead of failing the draw.
	const auto format = static_cast<Prospero::BufferFormat>((value.dwords[1] >> 20u) & 0x1ffu);
	const auto tile = static_cast<Prospero::TileMode>((value.dwords[3] >> 20u) & 0x1fu);
	const auto numeric_class =
	    Prospero::SampledTextureNumericClass(Prospero::RemapTextureFormat(format));
	if (NullImageDescriptor(value) || !ValidImageDescriptor(value, root.r128) ||
	    DescriptorDimension(value, root.dimension) != root.dimension ||
	    Prospero::IsFmaskTextureFormat(format) || format == Prospero::BufferFormat::k11_11_10UNorm ||
	    ImageConversionFormat(format) != Prospero::BufferFormat::kInvalid ||
	    (root.unfiltered_access_only && format == Prospero::BufferFormat::k16UInt &&
	     tile == Prospero::TileMode::kDepth) ||
	    numeric_class == Prospero::TextureNumericClass::Unsupported ||
	    (root.depth_compare && numeric_class != Prospero::TextureNumericClass::Float)) {
		value.dwords.fill(0u);
		return 0u;
	}
	return 1u + static_cast<uint32_t>(numeric_class) * 2u + (DescriptorIsCube(value) ? 1u : 0u);
}

ImageResource ImageHeapResource(const ImageResource& root, uint32_t kind) {
	ImageResource resource;
	resource.resource_class         = ImageResourceClass::Sampled;
	resource.read                   = true;
	resource.heap                   = true;
	resource.dimension              = root.dimension;
	resource.unfiltered_access_only = root.unfiltered_access_only;
	resource.depth_compare          = root.depth_compare;
	resource.r128                   = root.r128;
	if (kind != 0u) {
		resource.numeric_class = static_cast<Prospero::TextureNumericClass>((kind - 1u) / 2u);
		resource.cube          = ((kind - 1u) & 1u) != 0u;
	}
	return resource;
}

void ApplyResourceSpecialization(Program& program, const ResourceSpecialization& specialization) {
	EXIT_IF(!program.resource_tracking_complete || program.shader_info_complete);
	EXIT_IF(program.info.buffers.size() > specialization.buffers.size() ||
	        program.info.images.size() > specialization.images.size());

	auto& buffers = program.info.buffers;
	const auto original_buffer_count = buffers.size();
	buffers.reserve(specialization.buffers.size());
	for (uint32_t index = 0; index < specialization.buffers.size(); ++index) {
		const auto& source = specialization.buffers[index];
		if (index >= buffers.size()) {
			EXIT_IF(source.indirect_root >= original_buffer_count);
			buffers.push_back(buffers[source.indirect_root]);
		}
		auto& buffer                      = buffers[index];
		buffer.packed_stride              = source.packed_stride;
		buffer.descriptor_format          = source.descriptor_format;
		buffer.descriptor_swizzle         = source.descriptor_swizzle;
		buffer.indirect_root              = source.indirect_root;
		buffer.indirect_mapping_offset    = source.indirect_mapping_offset;
		buffer.indirect_search_iterations = source.indirect_search_iterations;
		buffer.indirect_resources.clear();
	}
	for (uint32_t index = 0; index < buffers.size(); ++index) {
		const auto root = buffers[index].indirect_root;
		if (root != BufferResource::NoIndirectBuffer) {
			EXIT_IF(root >= buffers.size());
			buffers[root].indirect_resources.push_back(index);
		}
	}
	auto& images = program.info.images;
	const auto original_image_count = images.size();
	images.reserve(specialization.images.size());
	for (uint32_t index = 0; index < specialization.images.size(); index++) {
		const auto& source = specialization.images[index];
		if (index >= images.size()) {
			EXIT_IF(source.indirect_root >= original_image_count);
			images.push_back(images[source.indirect_root]);
		}
		auto& image                      = images[index];
		image.numeric_class              = source.numeric_class;
		image.dimension                  = source.dimension;
		image.mip_count                  = source.mip_count;
		image.conversion_format          = source.conversion_format;
		image.shader_swizzle             = source.shader_swizzle;
		image.indirect_root              = source.indirect_root;
		image.indirect_mapping_offset    = source.indirect_mapping_offset;
		image.indirect_search_iterations = source.indirect_search_iterations;
		image.bindless                   = source.bindless;
		image.bindless_stride_dwords     = source.bindless_stride_dwords;
		image.cube                       = source.cube;
		image.indirect_resources.clear();
	}
	for (uint32_t index = 0; index < images.size(); index++) {
		const auto root = images[index].indirect_root;
		if (root != ImageResource::NoIndirectImage) {
			EXIT_IF(root >= images.size());
			images[root].indirect_resources.push_back(index);
		}
	}

	SamplerPlan sampler_plan;
	EXIT_IF(!BuildSamplerPlan(program.info, sampler_plan));
	auto& samplers      = program.info.samplers;
	auto& sampled_pairs = program.info.sampled_pairs;
	const auto original_sampler_count = samplers.size();
	samplers.reserve(sampler_plan.sampler_count);
	for (uint32_t index = 0; index < sampler_plan.sampler_count; index++) {
		const auto& binding = sampler_plan.bindings[index];
		if (index >= original_sampler_count) {
			samplers.push_back(samplers[binding.source]);
		}
		samplers[index].snapshot_index = binding.source;
		samplers[index].force_point_filtering = binding.type == SamplerClass::PointInteger;
		samplers[index].integer_border        = binding.type != SamplerClass::Float &&
		                                        binding.type != SamplerClass::FloatCompare;
	}
	// GPU-selected samplers: one native class per root, so children copy the mapped root.
	std::vector<uint32_t> indirect_class(original_sampler_count, UINT32_MAX);
	for (const auto& pair: sampled_pairs) {
		if (pair.sampler >= original_sampler_count) continue;
		const auto type = static_cast<uint32_t>(ClassifySampler(images[pair.image]));
		auto& known = indirect_class[pair.sampler];
		// A GPU-selected root may serve plain and compare samples: its children
		// copy the first class seen instead of stopping the emulator.
		const auto float_class   = static_cast<uint32_t>(SamplerClass::Float);
		const auto compare_class = static_cast<uint32_t>(SamplerClass::FloatCompare);
		const bool compare_pair  = (known == float_class && type == compare_class) ||
		                          (known == compare_class && type == float_class);
		EXIT_IF(pair.sampler < specialization.samplers.size() &&
		        specialization.samplers[pair.sampler].indirect_root == pair.sampler &&
		        known != UINT32_MAX && known != type && !compare_pair);
		if (!compare_pair) known = type;
	}
	for (uint32_t index = 0; index < sampler_plan.sampler_count; index++) {
		const auto& binding = sampler_plan.bindings[index];
		for (auto& candidate: samplers[index].indirect_resources) {
			candidate = sampler_plan.mapping[candidate][static_cast<uint32_t>(binding.type)];
			EXIT_IF(candidate == UINT32_MAX);
		}
	}
	for (auto& pair: sampled_pairs) {
		const auto type = static_cast<uint32_t>(ClassifySampler(images[pair.image]));
		pair.sampler = sampler_plan.mapping[pair.sampler][type];
		EXIT_IF(pair.sampler == UINT32_MAX);
		samplers[pair.sampler].depth_compare |= images[pair.image].depth_compare;
		for (const auto candidate: samplers[pair.sampler].indirect_resources)
			samplers[candidate].depth_compare |= images[pair.image].depth_compare;
	}
	for (uint32_t root = 0; root < original_sampler_count && root < specialization.samplers.size(); ++root) {
		const auto& source = specialization.samplers[root];
		if (source.indirect_root != root || indirect_class[root] == UINT32_MAX) continue;
		const auto mapped = sampler_plan.mapping[root][indirect_class[root]];
		EXIT_IF(mapped == UINT32_MAX);
		auto& native                     = samplers[mapped];
		native.indirect_root              = mapped;
		native.indirect_mapping_offset    = source.indirect_mapping_offset;
		native.indirect_search_iterations = source.indirect_search_iterations;
		native.indirect_resources.assign(1u, mapped);
		for (uint32_t child = original_sampler_count; child < specialization.samplers.size(); ++child) {
			if (specialization.samplers[child].indirect_root != root) continue;
			auto copy               = samplers[mapped];
			copy.snapshot_index     = child;
			copy.indirect_resources.clear();
			samplers[mapped].indirect_resources.push_back(static_cast<uint32_t>(samplers.size()));
			samplers.push_back(copy);
		}
	}
	EXIT_IF(samplers.size() > ShaderInfo::MaxSamplers);

	auto& memory_info = program.memory_info;
	const ImageRemap image_remap(specialization);
	for (auto* block: program.blocks) {
		for (auto it = block->begin(); it != block->end(); ++it) {
			auto& inst = *it;
			if (inst.GetOpcode() == ValueOpcode::GetImageResource) {
				inst.SetFlags(image_remap[inst.Flags<uint32_t>()]);
				continue;
			}
			if (BufferAccessOf(inst.GetOpcode()) == BufferAccess::Read) {
				const auto& memory = memory_info[inst.Flags<MemoryFlags>().index];
				if (memory.kind == ResourceKind::Buffer &&
				    specialization.buffers[memory.resource].zero_stride_oob) {
					// Bounds mode 0 checks offset >= stride, so zero stride
					// makes every vector read out of bounds regardless of its address.
					const auto count = BufferComponentCount(inst.GetOpcode());
					std::array<Value, 4> values {Value(0u), Value(0u), Value(0u), Value(0u)};
					if (memory.formatted && !memory.typed) {
						const auto& buffer = buffers[memory.resource];
						const auto format = Format::GetFormatInfo(buffer.descriptor_format);
						for (uint32_t component = 0; component < count; component++) {
							if (format.type == Format::ComponentType::Unknown ||
							    GetDstSel(buffer.descriptor_swizzle, component) != 1u) continue;
							const auto one = Format::FormattedConstantBits(
							    format, Format::FormattedSourceKind::One);
							values[component] = Value(&*block->PrependNewInst(
							    it, ValueOpcode::SelectU32,
							    {inst.Arg(inst.NumArgs() - 1), Value(one), Value(0u)}));
						}
					}
					Value result = values[0];
					switch (inst.GetType()) {
						case Type::U8: result = Value(uint8_t {0}); break;
						case Type::U16: result = Value(uint16_t {0}); break;
						case Type::U32x2:
							result = Value(&*block->PrependNewInst(it, ValueOpcode::CompositeConstructU32x2,
							                                      {values[0], values[1]}));
							break;
						case Type::U32x3:
							result = Value(&*block->PrependNewInst(it, ValueOpcode::CompositeConstructU32x3,
							                                      {values[0], values[1], values[2]}));
							break;
						case Type::U32x4:
							result = Value(&*block->PrependNewInst(it, ValueOpcode::CompositeConstructU32x4,
							                                      {values[0], values[1], values[2], values[3]}));
							break;
						default: break;
					}
					inst.ReplaceUsesWith(result);
				}
				continue;
			}
			const auto image_opcode = ImageOpcodeInfoOf(inst.GetOpcode());
			if (image_opcode.access == ImageAccess::None) {
				continue;
			}
			const auto index = inst.Flags<MemoryFlags>().index;
			EXIT_IF(index >= memory_info.size());
			auto& memory = memory_info[index];
			EXIT_IF(memory.resource >= images.size());
			const auto& image = images[memory.resource];
			if (specialization.images[memory.resource].fmask) {
				EXIT_IF(inst.GetOpcode() != ValueOpcode::ImageRead || memory.data_bits != 32u);
				// Vulkan MSAA stores each sample directly; FMASK's four-bit fragment indices
				// therefore map each coverage sample to the same host sample.
				constexpr uint32_t indices[] = {0x76543210u, 0xfedcba98u};
				std::array<Value, 2> fragments;
				for (uint32_t component = 0; component < fragments.size(); component++) {
					const auto selected = block->PrependNewInst(
					    it, ValueOpcode::SelectU32, {inst.Arg(2), Value(indices[component]), Value(0u)});
					fragments[component] = Value(&*selected);
				}
				const auto result = block->PrependNewInst(
				    it, ValueOpcode::CompositeConstructU32x4,
				    {fragments[0], fragments[1], Value(0u), Value(0u)});
				inst.ReplaceUsesWith(Value(&*result));
				continue;
			}
			if (image_opcode.needs_sampler &&
			    memory.sampler < original_sampler_count) {
				const auto type = static_cast<uint32_t>(ClassifySampler(image));
				memory.sampler = sampler_plan.mapping[memory.sampler][type];
				EXIT_IF(memory.sampler == UINT32_MAX);
			}
			// An ImageWrite selects its candidate like a read (the emitter's EmitImageAccess); a bindless root never reaches
			// here with a write (bindless roots are sampled).
			if (image.indirect_root == memory.resource && inst.GetOpcode() == ValueOpcode::ImageWrite && !image.bindless) {
				continue;
			}
			if (image.indirect_root == memory.resource && inst.GetOpcode() != ValueOpcode::ImageSampleRaw &&
			    inst.GetOpcode() != ValueOpcode::ImageGatherRaw && inst.GetOpcode() != ValueOpcode::ImageRead) {
				// Other image operations on a GPU-selected table have no per-candidate form.
				EXIT("image operation %s on a GPU-selected table (shader 0x%016llx)\n",
				     ValueOpcodeName(inst.GetOpcode()).data(), static_cast<unsigned long long>(program.shader_hash));
			}
		}
	}
	for (auto& memory: memory_info) {
		if (memory.kind == ResourceKind::Image && !memory.planning_only) {
			memory.resource = image_remap[memory.resource];
		}
	}
	for (auto& buffer: buffers) {
		if (buffer.image_alias != BufferResource::NoImageAlias) {
			buffer.image_alias = image_remap[buffer.image_alias];
		}
	}
	for (auto& pair: sampled_pairs) {
		pair.image = image_remap[pair.image];
	}
	for (auto& image: images) {
		if (image.indirect_root != ImageResource::NoIndirectImage) {
			image.indirect_root = image_remap[image.indirect_root];
		}
		for (auto& resource: image.indirect_resources) {
			resource = image_remap[resource];
		}
	}
	image_remap.Apply(images);
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
