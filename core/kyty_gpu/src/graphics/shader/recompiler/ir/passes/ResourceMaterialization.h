#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_

#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <array>
#include <atomic>
#include <functional>
#include <optional>
#include <span>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

// Canonical module-affecting resource state. Runtime addresses and descriptor payloads remain in
// ResourceSnapshot and therefore do not create shader permutations.
struct ResourceSpecialization {
	struct Buffer {
		uint32_t               packed_stride                   = 0;
		Prospero::BufferFormat descriptor_format               = Prospero::BufferFormat::kInvalid;
		uint32_t               descriptor_swizzle              = DstSel(4, 5, 6, 7);
		bool                   zero_stride_oob                 = false;
		bool                   scalar_empty                    = false;
		uint32_t               indirect_root                   = BufferResource::NoIndirectBuffer;
		uint32_t               indirect_mapping_offset         = 0;
		uint32_t               indirect_search_iterations      = 0;
		bool                   operator==(const Buffer&) const = default;
	};

	struct Image {
		Prospero::TextureNumericClass numeric_class = Prospero::TextureNumericClass::Unsupported;
		Decoder::ImageDimension       dimension     = Decoder::ImageDimension::Unknown;
		uint32_t                      mip_count     = 1;
		Prospero::BufferFormat        conversion_format          = Prospero::BufferFormat::kInvalid;
		uint32_t                      shader_swizzle             = ShaderImageIdentitySwizzle;
		uint32_t                      indirect_root              = ImageResource::NoIndirectImage;
		uint32_t                      indirect_mapping_offset    = 0;
		uint32_t                      indirect_search_iterations = 0;
		bool                          cube                       = false;
		bool                          fmask                      = false;
		bool                          bindless                   = false;
		uint32_t                      bindless_stride_dwords     = 0;
		bool                          heap                       = false;
		bool                          operator==(const Image&) const = default;
	};

	struct Sampler {
		uint32_t indirect_root              = SamplerResource::NoIndirectSampler;
		uint32_t indirect_mapping_offset    = 0;
		uint32_t indirect_search_iterations = 0;
		bool     operator==(const Sampler&) const = default;
	};

	std::vector<Buffer>  buffers;
	std::vector<Image>   images;
	std::vector<Sampler> samplers;

	bool operator==(const ResourceSpecialization&) const = default;
};

// Extracts the descriptor/SRT value graph before resource specialization. The returned plan owns
// its values and is independent of the translated shader CFG.
ResourcePlan ExtractResourcePlan(const Program& program);

// The control flow ExtractResourcePlan gives the plan (ResourcePlan::control_flow; empty when every slot and descriptor
// is read), each branch condition passed through `predicate` (ExtractResourcePlan's ClonePredicate).
std::vector<ResourceBlock> ResourceControlFlowOf(const Program& program, const std::function<Value(Value)>& predicate);

// Refreshes cached resources and specialization in place. A failed refresh must not be used.
// The ordinary read hook reads guest memory for SRT walks without faulting on GPU-modified pages (byte-precise
// GPU-dirty check); it returns false to fall back to a direct read.
// Table slots: a GPU-selected
// descriptor table takes a power of two of candidates (padding copies, never selected), of flattened-SRT room and of
// search steps, so a table that gains candidates keeps its permutation until it crosses the next power of two.
// The only layout since 26.10 (6cc64dee at the user's new area: 3 -> 1 permutations in 300 s); the unit test sets
// the override to false for the layout it replaced (TestIndirectTableSlotsSelectSame compares both), std::nullopt
// restores the default (on).
bool IndirectTableSlots();
void SetIndirectTableSlotsOverride(std::optional<bool> enabled);

using OrdinaryReadHook = bool (*)(uint64_t vaddr, void* data, uint64_t size);
void SetOrdinaryReadHook(OrdinaryReadHook hook);
// An ordinary SRT-walk read of guest memory: through the hook when one is set, else a direct read.
void ReadOrdinaryMemory(uint64_t vaddr, void* data, uint64_t size);
// The direct read (LibKernel::Memory::ReadGuestPlain: this library does not depend on it).
using PlainReadHook = void (*)(void* data, uint64_t vaddr, uint64_t size);
void SetPlainReadHook(PlainReadHook hook);
void ReadPlainMemory(void* data, uint64_t vaddr, uint64_t size);

// Brackets one MaterializeResources call on the calling thread (called with true on entry and with false on
// every exit). Inside the bracket the ordinary-read hook may keep facts
// that nothing running in the call can change (e.g. "this page is not GPU-modified"). The brackets are
// per thread and may nest. Optional.
using OrdinaryReadSessionHook = void (*)(bool begin);
void SetOrdinaryReadSessionHook(OrdinaryReadSessionHook hook);

bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          ResourceSnapshot& snapshot, ResourceSpecialization& specialization);

// ---- Materialization ahead of the GPU thread ----
//
// MaterializeResourcesAhead computes what MaterializeResources would, but can run on another thread: it writes
// nothing shared (the plan's scratch slot 1, the tape's slot 1 memo, thread-local reader state and the caller's
// outputs only) and reads guest memory through the ahead reader instead of the GPU thread's readers. The reader
// logs every read (ReadLog). The result stands in for a real call when ReadsUnchangedFast(log, ...) holds on the GPU
// thread later and the plan, user data, shader base and workgroup counts are the same: a materialization is a
// function of those and of the bytes it read.

// Why MaterializeResourcesAhead returned false.
enum class AheadRefusal : uint8_t {
	None,
	GpuDirty, // a read touched GPU-dirty bytes (or bytes of a pending eager readback)
	Backing,  // a read could not be served from the backing store (or no ahead read hook is installed)
	Cap,      // a read or the whole log outgrew ReadLog::MaxCallBytes / MaxTotalBytes
	Flag,     // a configuration the ahead path does not model, or a runtime with a reader of its own
	NoTape,   // the plan has no SrtTape (or one of another shape): the reference walker is not thread-safe
	Other,    // the materialization itself failed (MaterializeResources would fail too)
	Count
};

// The guest reads one MaterializeResourcesAhead made, in order, with the bytes they returned.
struct ReadLog {
	// Ordinary: ReadOrdinaryMemory (SRT pointer chasing, fused runs). Strict: the specialization reader
	// (SrtRuntime::read_specialization_memory).
	enum class Kind : uint8_t { Ordinary, Strict };
	struct Entry {
		uint64_t address = 0;
		uint32_t size    = 0; // Bytes.
		uint32_t first   = 0; // Offset of the returned bytes in `bytes` (none when !ok).
		Kind     kind    = Kind::Ordinary;
		bool     ok      = false;
		uint8_t  cls     = 0; // KYTY_TRACE_SRT_CLASSES: t_srt_read_class of the read (in the padding)
	};
	// Exact bytes are kept, so a read larger than MaxCallBytes (or a log larger than MaxTotalBytes) refuses.
	static constexpr uint32_t MaxCallBytes  = 4096;
	static constexpr uint32_t MaxTotalBytes = 1024u << 10u;

	std::vector<Entry>   entries;
	std::vector<uint8_t> bytes;
	// Starts a new log (the reads the stages of one operation repeat are logged once, while they read the same bytes).
	void Clear();

};

// Reads guest memory for the ahead path: the bytes of [vaddr, vaddr + size) as the backing store holds them,
// without faulting or waiting for the GPU. GpuDirty: the range has bytes the GPU wrote (or an eager readback
// is pending on it), so the backing is stale. Backing: the range is not backed. Installed by the kernel
// (it knows the GPU caches); none installed: every ahead read refuses.
enum class AheadReadStatus : uint8_t { Ok, GpuDirty, Backing };
using AheadReadHook = AheadReadStatus (*)(uint64_t vaddr, void* data, uint64_t size);
void SetAheadReadHook(AheadReadHook hook);

// The ahead reader as an SrtMemoryReader (the specialization reader of an ahead runtime; ordinary reads are
// redirected inside ReadOrdinaryMemory). Fails (and refuses the materialization) outside MaterializeResourcesAhead.
bool AheadReadMemory(void* userdata, uint64_t address, std::span<uint32_t> values);

// Materializes into the caller's outputs (scratch copies, not the plan's cached ones). `runtime` is the one a
// MaterializeResources call would get: its read_memory must be null (or AheadReadMemory); its
// read_specialization_memory only says whether the plan may read specialization memory, since the ahead
// reader replaces it. Returns false (and says why in *refusal) when it refused; the outputs are then unusable.
// the null formatted share for a materialization a worker made (see ResourcePlan::Scratch::null_share_kinds):
// applies and updates the plan's shared last live specializations in draw order (GPU thread).
void ApplyNullFormattedShare(const ResourcePlan& program, const std::vector<uint8_t>& kinds,
                             ResourceSpecialization& specialization);
// The null share journal: `journal` is
// the precompile journal's last live {format, swizzle} of each buffer binding of the program ({0, 0}: none); its
// formatted bindings' values are kept (=1 and =2) and, =1, taken as the bindings' last live values. `journal_multi`:
// per binding, 1 when the journal had more than one live value for it (counted).
void SeedNullFormattedShare(const ResourcePlan& program, std::span<const std::array<uint32_t, 2>> journal,
                            std::span<const uint8_t> journal_multi);
// The null share journal: a permutation was compiled with this specialization; one with a null formatted
// buffer's own specialization is counted (and whether the journal had a live value for that buffer).
void NoteNullShareVariant(const ResourcePlan& program, const ResourceSpecialization& specialization);
// KYTY_TRACE_PERF: PERF null_share_journal (with the switch on).
void ReportNullShareJournalPerf();

// A read of the guest memory a resolution depends on besides its materialization (vertex fetch tables), logged as an
// ordinary read. False when the ahead reader refuses it (nothing is logged then).
bool ReadGuestAhead(ReadLog& log, uint64_t address, void* data, uint64_t size);

bool MaterializeResourcesAhead(const ResourcePlan& program, const SrtRuntime& runtime,
                               ResourceSnapshot& snapshot, ResourceSpecialization& specialization,
                               ReadLog& log, AheadRefusal* refusal = nullptr);

// ---- The check of a result ----
//
// ReadsUnchangedFast makes every logged read again through the GPU thread's readers (ReadOrdinaryMemory with its hook, and
// runtime.read_specialization_memory), so GPU-dirty handling and downloads happen as in a real call, and reports whether
// each read succeeds as it did and returns the same bytes. Must run on the GPU thread, outside MaterializeResourcesAhead.
//
// The same question with the same answer, asked more cheaply for the common case. The kernel installs a verifier
// (SetFastVerifyHook) that walks the log and, for every run of logged reads it can PROVE has no GPU state to honour
// (the 4 KiB pages are not GPU-modified and no eager readback is pending: a real read would neither download nor wait
// for anything), compares the guest bytes with the logged ones in place (memcmp, no copy) instead of re-reading
// them one by one through the readers. A run it cannot prove (GPU-modified page, an unusual mapping, no read
// session) is left to the readers, one entry at a time, exactly as ReadsUnchangedFast does. Only valid on the GPU
// thread, with a runtime whose read_specialization_memory is the GPU thread's standard strict reader (what
// PipelineCache::Get passes).

// Why the verifier stopped.
enum class FastVerifyResult : uint8_t {
	Unchanged,   // every entry from `begin` on is proven unchanged
	Changed,     // entry `stop` (or a neighbour merged with it) now reads other bytes than logged
	NeedsReader, // entries [begin, stop) are proven unchanged; entry `stop` must go through the readers
};

// What the fast verifier did (all counted by the GPU thread, plain integers).
struct FastAdoptStats {
	uint64_t calls        = 0; // ReadsUnchangedFast calls
	uint64_t entries_fast = 0; // log entries proven by a memcmp
	uint64_t runs         = 0; // memcmp runs they were compared in (entries_fast - runs entries were merged into a neighbour)
	uint64_t entries_slow = 0; // log entries that went through the readers
};

using FastVerifyHook = FastVerifyResult (*)(const ReadLog& log, size_t begin, size_t& stop, FastAdoptStats& stats);
void SetFastVerifyHook(FastVerifyHook hook);

// ReadsUnchangedFast through the fast verifier (without a hook installed it is ReadsUnchangedFast).
bool ReadsUnchangedFast(const ReadLog& log, const SrtRuntime& runtime, FastAdoptStats* stats = nullptr);

// Names the first field in which a MaterializeResourcesAhead result differs from the result of the real
// MaterializeResources of the same plan (nullptr: identical in every field a consumer reads). The active-source
// vectors are the plan's scratch[1] (ahead) and scratch[0] (real) `active_sources`: a caller that materializes
// the same plan ahead again before it compares keeps a copy of the ahead one next to its result.
const char* AheadResultDifference(const ResourceSnapshot& ahead, const ResourceSpecialization& ahead_specialization,
                                  const std::vector<uint8_t>& ahead_active_sources, const ResourceSnapshot& real,
                                  const ResourceSpecialization& real_specialization,
                                  const std::vector<uint8_t>& real_active_sources);

// Applies an already-derived specialization to native IR before layout and emission.
void ApplyResourceSpecialization(Program& program, const ResourceSpecialization& specialization);

// ---- KYTY_PERF_BINDLESS_CLASSES (ShaderIR.h BindlessClassesMode) ----
//
// The numeric class of a bindless table field comes from its records, the T#s the guest wrote there. The first
// materialization of a field's resource reads the field once (GPU thread, not a captured read) and the host's table
// translation reads every record anyway; both count the T#s with TallyBindlessFieldDescriptor and decide with
// DecideBindlessFieldClass. The decision is kept per (shader, resource, field) in a registry that every later
// materialization reads instead of guest memory (MaterializeResourcesAhead refuses a field the registry does not know
// yet, so the GPU thread decides first). A field typed float is raised to an integer class when a later translation
// finds integer T#s in it; an integer field stays integer (its float T#s, the guest's defaults, read zero).

// A sampled image of a bounded descriptor table that MaterializeIndirectDescriptor makes bindless: KYTY_BINDLESS=1, a
// stage KYTY_BINDLESS_STAGES allows, 128-bit descriptors only with KYTY_BINDLESS_R128=1.
bool BindlessImageExpected(ShaderType stage, ImageResourceClass resource_class, bool r128);

// The set-1 image array (BindlessBindings.h) a bindless image resource samples.
uint32_t BindlessArrayBinding(const ImageResource& image);

// Counts one descriptor slot of a field (BindlessFieldTally, ShaderIR.h). words: the descriptor's eight words (R128:
// words 4..7 are read as zero); binding: the array the field samples.
void TallyBindlessFieldDescriptor(const uint32_t* words, bool r128, uint32_t binding, BindlessFieldTally& tally);
// Uint or Sint when the field holds integer T#s of one class and no float T# but tiny defaults; Float otherwise.
// *ambiguous (optional): it held integer T#s that did not decide it (both integer classes, or real float textures).
Prospero::TextureNumericClass DecideBindlessFieldClass(const BindlessFieldTally& tally, bool* ambiguous = nullptr);

// The registry key of a resource's field (field_stride / field_offset as the shader names the field: for a heap table
// ImageResource::bindless_fields, stride 0 = the V#'s own; for a record table its stride and first byte).
uint64_t BindlessClassKey(uint64_t shader_hash, uint32_t resource, uint32_t field_stride, uint32_t field_offset);
// The registered class; Unsupported when the registry does not know the field. Lock-free (any thread).
Prospero::TextureNumericClass FindBindlessClass(uint64_t key);
// Registers or replaces a field's class (writers are serialized); false when the registry is full.
bool StoreBindlessClass(uint64_t key, Prospero::TextureNumericClass numeric_class);
// Registered fields by class: float, uint, sint.
std::array<uint32_t, 3> BindlessClassRegistryCounts();
// No new field fits any more: unknown fields are float from then on (no first-sight read, no ahead refusal).
bool BindlessClassRegistryFull();

// PERF bindless_classes counters of the recompiler side (any thread; totals since start).
struct BindlessClassCounters {
	std::atomic<uint64_t> split_resources {0}; // field resources ResourceTracking made
	std::atomic<uint64_t> split_refused {0};   // fields left shared: the shader is near the resource limits
	std::atomic<uint64_t> split_retries {0};   // shaders tracked again without field resources (limits exceeded)
	std::atomic<uint64_t> decided[3] {};       // first-sight decisions: float, uint, sint
	std::atomic<uint64_t> learned[2] {};       // fields raised to uint, sint by a table translation
	std::atomic<uint64_t> mixed {0};           // integer fields that also held tiny float defaults
	std::atomic<uint64_t> ambiguous {0};       // fields with integer T#s kept float (strict decision)
	std::atomic<uint64_t> read_bytes {0};      // bytes the first-sight reads read
	std::atomic<uint64_t> read_fail {0};
	std::atomic<uint64_t> ahead_refused {0};   // MaterializeResourcesAhead calls refused for an unknown field
	std::atomic<uint64_t> registry_full {0};
	std::atomic<uint64_t> sampler_reverts {0}; // materializations whose integer fields stayed float (sampler limits)
};
BindlessClassCounters& GetBindlessClassCounters();
// Kind of an image heap entry as root sees it, or 0 (and a zeroed value) when root reads it as
// null. Entries of one kind specialize identically.
uint32_t ImageHeapKind(const ImageResource& root, DescriptorValue& value);
// The sampled resource that root resolves entries of kind as; kind 0 gives the identity of
// root's view of a heap.
ImageResource ImageHeapResource(const ImageResource& root, uint32_t kind);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_ */
