#ifndef KYTY_CX_DISPATCH
#define KYTY_CX_DISPATCH 1
#endif
#ifndef KYTY_CX_INTERCHANGE
#define KYTY_CX_INTERCHANGE 0
#endif
#ifndef KYTY_CX_INDIRECT
#define KYTY_CX_INDIRECT 0
#endif
#ifndef KYTY_CX_SCOUT
#define KYTY_CX_SCOUT 1
#endif
using HashedKey = NativeXprKeyHash;

static uint64_t TableValidity(const Image& image) {
	return image.version != nullptr ? image.version->load(std::memory_order_acquire) : 0;
}

namespace {
std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX>
TableDrawExportMapping(const GraphicContext& graphics, const HW::Context& ctx, uint32_t color_output_mask) {
	std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX> mapping {};
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		const auto& rt = ctx.GetRenderTarget(slot);
		if ((color_output_mask & (1u << slot)) != 0 && rt.base.addr != 0) {
			mapping[slot] =
			    TextureGetRenderTargetFormat(graphics, rt.info.format, rt.info.channel_type, rt.info.channel_order)
			        .export_mapping;
		}
	}
	return mapping;
}

bool TableStridesEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_TABLE_STRIDES");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

bool TableBufferSpecialized(const ShaderRecompiler::IR::BufferResource&                 resource,
                            const ShaderRecompiler::IR::ResourceSpecialization::Buffer& specialized,
                            const std::array<uint32_t, 4>& words4, bool runtime_format) {
	runtime_format = runtime_format && specialized.descriptor_format == Prospero::BufferFormat::kInvalid;
	ShaderBufferResource descriptor;
	std::copy_n(words4.begin(), 4, descriptor.fields);
	if (descriptor.Type() != 0) descriptor = {};
	auto       packed_stride = descriptor.PackedStride();
	const auto stride        = packed_stride & 0x3fffu;
	const bool swizzle       = stride != 0u && ((packed_stride >> 14u) & 1u) != 0u;
	if (stride == 0u) {
		packed_stride &= ~((1u << 14u) | (3u << 16u));
	} else if (!swizzle) {
		packed_stride &= ~(3u << 16u);
	}
	const bool any_stride = TableStridesEnabled() && !swizzle && (specialized.packed_stride & (1u << 14u)) == 0u &&
	                        resource.indirect_root == ShaderRecompiler::IR::BufferResource::NoIndirectBuffer &&
	                        resource.indirect_resources.empty();
	const auto stride_bits = any_stride ? ~0x3fffu : ~0u;
	return (packed_stride & stride_bits) == (specialized.packed_stride & stride_bits) &&
	       (runtime_format ||
	        ((resource.formatted ? descriptor.Format() : Prospero::BufferFormat::kInvalid) == specialized.descriptor_format &&
	         (resource.formatted ? descriptor.DstSelXYZW() : DstSel(4, 5, 6, 7)) == specialized.descriptor_swizzle)) &&
	       (descriptor.OutOfBounds() == 0u && stride == 0u) == specialized.zero_stride_oob &&
	       (resource.scalar && descriptor.GetSize() == 0u) == specialized.scalar_empty;
}

const ShaderRecompiler::IR::ResourceSnapshot g_table_no_resources {};

}

static bool TableReadRun(uint64_t address, uint32_t count, uint32_t* out) {
	const uint64_t bytes = uint64_t {count} * 4u;
	if ((address & 3u) != 0 || !GuestRange {address, bytes}.Valid()) return false;
	if (Libs::LibKernel::Memory::MayFaultOnGpuRead(address, bytes))
		return Libs::LibKernel::Memory::TryReadAroundGpuWrites(address, out, bytes);
	return Libs::LibKernel::Memory::TryReadBacking(address, out, bytes);
}

static thread_local int32_t g_table_offsets_key = -1;
static int32_t TableStartInstance(const HW::VertexShaderInfo&) {
	return g_table_offsets_key;
}
static bool TableMemoOn() {
	return kyty_local_table_memo.load(std::memory_order_relaxed) != 0;
}
static uint32_t TableMemoVerifyEvery() {
	return kyty_local_table_memo_verify.load(std::memory_order_relaxed);
}
static uint32_t TableSerialsMode() {
	return kyty_local_table_serials.load(std::memory_order_relaxed);
}

template <typename T>
struct TableRoleSide {
	std::atomic<T*> block {nullptr};
	TableRoleSide() = default;
	TableRoleSide(TableRoleSide&& other) noexcept: block(other.block.exchange(nullptr)) {}
	TableRoleSide& operator=(TableRoleSide&& other) noexcept {
		delete block.exchange(other.block.exchange(nullptr));
		return *this;
	}
	TableRoleSide(const TableRoleSide&)            = delete;
	TableRoleSide& operator=(const TableRoleSide&) = delete;
	~TableRoleSide() { delete block.load(); }
	[[nodiscard]] T* Peek() const noexcept { return block.load(std::memory_order_acquire); }
	T& Get() {
		return Get([](T&) {});
	}
	template <typename Init>
	T& Get(Init&& init) {
		T* at = block.load(std::memory_order_acquire);
		if (at != nullptr) return *at;
		auto* made = new T;
		init(*made);
		if (block.compare_exchange_strong(at, made, std::memory_order_acq_rel)) return *made;
		delete made;
		return *at;
	}
};

struct RenderExecutor::TableVariant {
	uint64_t                       state          = 0;
	const PipelineCache::Pipeline* pipeline       = nullptr;
	bool                           fetch_embedded = false;
	bool                           refused        = false;
	uint32_t                       colors         = 0;
	bool                           has_depth      = false;
	bool                           has_stencil    = false;
	std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX> mapping {};
};

template <typename T>
struct TableUninitialized: std::allocator<T> {
	using std::allocator<T>::allocator;
	template <typename U>
	struct rebind {
		using other = TableUninitialized<U>;
	};
	template <typename U>
	void construct(U* at) noexcept {
		::new (static_cast<void*>(at)) U;
	}
	template <typename U, typename... Args>
	void construct(U* at, Args&&... args) {
		::new (static_cast<void*>(at)) U(std::forward<Args>(args)...);
	}
};
using TableWords = std::vector<uint32_t, TableUninitialized<uint32_t>>;

struct RenderExecutor::TableImageSet {
	struct Image {
		ImageId                 id;
		uint64_t                serial = 0;
		vk::ImageLayout         layout = vk::ImageLayout::eUndefined;
		ImageSubresourceRange   range;
		TextureCache::ImageDesc desc;
		vk::AccessFlags2        access = vk::AccessFlagBits2::eShaderRead;
		bool                    target  = false;
		vk::AccessFlags2        target_read {};
		bool                    storage = false;
		bool                    whole   = false;
		bool                    proven   = false;
		uint64_t                validity = 0;
		mutable uint64_t        clear_memo = 0;
	};
	TableWords            words;
	vk::DescriptorSet     set  = nullptr;
	vk::DescriptorPool    pool = nullptr;
	std::vector<Image>    images;
	uint64_t              bda_generation = 0;
	uint64_t validated_frame  = UINT64_MAX;
	uint64_t resolution_epoch = 0;
	struct Roles1 {
		std::array<uint64_t, Roles - 1> validated_frame, resolution_epoch {};
		Roles1() { validated_frame.fill(UINT64_MAX); }
	};
	TableRoleSide<Roles1> roles;
	Roles1& Side() {
		return roles.Get([this](Roles1& side) { side.resolution_epoch.fill(RoleZeroEpoch()); });
	}
	[[nodiscard]] uint64_t RoleZeroEpoch() { return std::atomic_ref<uint64_t>(resolution_epoch).load(std::memory_order_relaxed); }
	uint64_t& ValidatedFrame(uint32_t role) { return role == 0 ? validated_frame : Side().validated_frame[role - 1]; }
	uint64_t& ResolutionEpoch(uint32_t role) { return role == 0 ? resolution_epoch : Side().resolution_epoch[role - 1]; }
	std::atomic<uint64_t>       used_frame {0};
	uint64_t                  touched_tick  = 0;
	uint64_t                  touched_frame = 0;
	static constexpr uint64_t TouchTicks    = 8;
	uint64_t                  span_begin = 0, span_end = 0;
	uint64_t                  transit_epoch = UINT64_MAX;
	const NativeXprDrawState* transit_state = nullptr;
	uint64_t                  transit_serials  = UINT64_MAX;
	uint64_t                  transit_capture  = 0;
	std::vector<ImageId>      image_ids;
	bool                      holds_target = false;
	bool                      read_after_write = false;
	bool                  refused = false;
	static constexpr uint64_t RetryFrames = 300;
	std::shared_ptr<const std::atomic<bool>> retry_wait;
	const ShaderRecompiler::IR::CompiledShaderInfo* program  = nullptr;
	const PipelineCache::Pipeline*                  pipeline = nullptr;
};

#if KYTY_CX_INTERCHANGE
static bool TableInterchangeable(const ShaderRecompiler::IR::CompiledShaderInfo& a,
                                 const ShaderRecompiler::IR::CompiledShaderInfo& b) {
	const auto& x = a.bindings;
	const auto& y = b.bindings;
	return a.table_mode && b.table_mode && a.stage == b.stage && a.shader_hash == b.shader_hash &&
	       a.user_data_base == b.user_data_base && a.table_plan == b.table_plan && a.info.buffers == b.info.buffers &&
	       a.info.images.size() == b.info.images.size() && a.info.samplers.size() == b.info.samplers.size() &&
	       x.push_data_start_dword == y.push_data_start_dword &&
	       x.memory_offset_dword == y.memory_offset_dword && x.memory_offset_count == y.memory_offset_count &&
	       x.buffer_word_count == y.buffer_word_count && x.lod_stats_count == y.lod_stats_count &&
	       x.table_block_dwords == y.table_block_dwords && x.user_data_registers == y.user_data_registers;
}
#endif

struct RenderExecutor::TableFlatStage {
	struct Operand {
		uint32_t index = 0;
		uint32_t base  = 2;
	};
	struct Run {
		std::array<Operand, 3> address;
		int32_t                immediate = 0;
		uint32_t               first = 0, count = 0;
		Operand                records;
		bool                   bounded = false;
	};
	std::vector<Run>      runs;
	std::vector<uint32_t> level_ends;
	std::vector<Operand>  image_words;
	std::vector<uint32_t> image_counts;
	std::vector<Operand>  buffer_words;
	std::vector<uint32_t> constants;
	uint32_t              user_words = 0;
};

struct RenderExecutor::TablePair {
	struct Run {
		ShaderRecompiler::IR::TablePlan::Slot read;
		uint32_t                              first = 0, count = 0, level = 0;
	};
	uint64_t                                                    vs = 0, ps = 0;
	int32_t                                                     start_instance = -1;
	bool                                                        refused  = false;
	const char* special = nullptr;
	std::array<const ShaderRecompiler::IR::CompiledShaderInfo*, 2> programs {};
	std::array<const ShaderModule*, 2>                              tables {};
	std::array<const ShaderRecompiler::IR::ResourcePlan*, 2>        plans {};
	uint64_t                                                        static_key = 0;
	bool                                                            has_pixel  = false;
	std::array<uint64_t, 2>                                         code_hashes {};
	bool                                                            pending = false;
	std::array<std::vector<Run>, 2>                             runs;
	struct PackedRun {
		std::array<uint16_t, 3>   address {};
		uint16_t                  records = NotBounded;
		int32_t                   immediate = 0;
		uint16_t                  first = 0, count = 0;
		static constexpr uint16_t NotBounded = 0xffff;
	};
	struct PackedStage {
		std::span<const PackedRun> runs;
		std::span<const uint16_t>  level_ends, image_words, buffer_words;
		std::span<const uint8_t>   image_counts;
		std::span<const uint32_t>  constants;
		uint32_t                   user_words = 0;
		bool                       valid      = false;
	};
	std::array<PackedStage, 2>      packed;
	std::unique_ptr<std::byte[]>    packed_block;
	const std::byte*                packed_at    = nullptr;
	uint32_t                        packed_bytes = 0;
	std::vector<TableVariant>                                   variants;
	uint32_t                                                    variant_hint = 0;
	uint64_t                                                    store_frame = 0, store_state = 0;
	std::unordered_map<uint64_t, std::unique_ptr<TableImageSet>, HashedKey> sets;
	TableImageSet*                                              last_set = nullptr;
	TableImageSet*                                              previous_set = nullptr;
	uint64_t                                                    generation = UINT64_MAX;
	struct Roles1 {
		std::array<TableImageSet*, Roles - 1> last_set {}, previous_set {};
		std::array<uint64_t, Roles - 1>       generation;
		Roles1() { generation.fill(UINT64_MAX); }
	};
	TableRoleSide<Roles1> roles;
	TableImageSet*& LastSet(uint32_t role) { return role == 0 ? last_set : roles.Get().last_set[role - 1]; }
	TableImageSet*& PreviousSet(uint32_t role) { return role == 0 ? previous_set : roles.Get().previous_set[role - 1]; }
	uint64_t&       Generation(uint32_t role) { return role == 0 ? generation : roles.Get().generation[role - 1]; }
	void ClearSets() noexcept {
		last_set = previous_set = nullptr;
		if (auto* side = roles.Peek()) {
			side->last_set.fill(nullptr);
			side->previous_set.fill(nullptr);
		}
	}
	std::unique_ptr<ShaderComputeInputInfo> compute;
#if KYTY_CX_INTERCHANGE
	ShaderProgram                           compute_program;
#endif
	const PipelineCache::Pipeline*          pipeline  = nullptr;
	uint32_t                                sightings = 0;
	std::array<uint32_t, 2> portable {};
	bool                    respecialize = false;
	std::shared_ptr<const std::atomic<bool>> waiting;
	TablePair* next = nullptr;
	uint64_t used_frame = 0;
	uint64_t scout_version = 0;
};

struct RenderExecutor::TableXpr {
	using Pair    = TablePair;
	using Variant = TableVariant;
	std::unordered_map<uint64_t, Pair, HashedKey> pairs;
	std::unordered_map<uint64_t, Pair, HashedKey> dispatches;
	static constexpr uint64_t PairSweepFrames = 600, PairKeepFrames = 3600;
	static constexpr size_t   PairsKept       = 2048;
	uint64_t                  pairs_swept     = 0;
	static constexpr size_t PairCacheEntries = 4096;
	std::array<std::pair<uint64_t, Pair*>, PairCacheEntries> pair_cache {};
	std::array<uint64_t, 3> found_key {};
	Pair*                   found_pair = nullptr;
	Pair*                   previous_dispatch = nullptr;
	std::array<uint64_t, 4> found_dispatch_key {};
	Pair*                   found_dispatch = nullptr;
	uint64_t collected  = 0;
	uint64_t epoch = 0, seen_epoch = 0;
	Pair*    store       = nullptr;
	uint64_t store_state = 0, store_frame = 0;
	uint32_t store_requests = 0;
	TableWords                           words;
	std::vector<uint32_t>                lod_stats;
	std::vector<uint64_t>                addresses;
	std::array<uint64_t, 256> mapped {};
	uint64_t                  mapping_epoch = UINT64_MAX;
	Pair*                     last_pair    = nullptr;
	size_t                    last_variant = 0;
	const NativeXprDrawState* last_state   = nullptr;
	const TableImageSet*      last_set     = nullptr;
	struct Memo {
		const Pair*               pair    = nullptr;
		uint64_t                  key     = 0;
		uint64_t                  epoch   = 0;
		const NativeXprDrawState* state   = nullptr;
		TableImageSet*            set     = nullptr;
		uint32_t                  variant = 0;
	};
	static constexpr size_t     MemoEntries = 4096;
	std::array<Memo, MemoEntries> memo {};
	uint64_t                    memo_epoch  = 1;
	TableImageSet*              memo_set    = nullptr;
	uint64_t                    memo_verify_count = 0;
	bool                        memo_verify       = false;
	[[nodiscard]] Memo& MemoAt(const Pair* pair, uint64_t key) {
		return memo[((reinterpret_cast<uintptr_t>(pair) * 0x9e3779b97f4a7c15ull) ^ (key * 0xc2b2ae3d27d4eb4full)) >> 52u];
	}
	uint64_t                  images_epoch = 0;
	struct HeldBuffer {
		uint64_t begin = 0, end = 0, cpu_epoch = 0, registration = 0;
		Buffer*  buffer = nullptr;
	};
	HeldBuffer                   args;
	std::array<HeldBuffer, 8192> indices;
	struct alignas(64) ResolvedBuffer {
		std::array<uint32_t, 4> words {};
		uint32_t                dwords = 0, special = 0;
		vk::DeviceAddress       device = 0;
		uint64_t                address = 0, size = 0, cpu_epoch = 0, registration = 0;
	};
	static constexpr uint64_t RingCopyMark = uint64_t {1} << 63u;
	static constexpr size_t           ResolvedEntries = 65536;
	std::unique_ptr<ResolvedBuffer[]> resolved = std::make_unique<ResolvedBuffer[]>(ResolvedEntries);
	struct RingCopy {
		uint64_t          address = 0, size = 0, observation = 0, tick = 0, cpu_epoch = 0;
		vk::DeviceAddress device = 0;
	};
	static constexpr size_t           RingCopies = 4096;
	std::unique_ptr<RingCopy[]>       ring_copies = std::make_unique<RingCopy[]>(RingCopies);
	uint32_t                          special_generation = 1;
	static constexpr size_t                           NoImageStartEntries = 1024;
	std::array<std::pair<uint64_t, uint64_t>, NoImageStartEntries> no_image_start {};
	uint64_t                          resolved_start = UINT64_MAX, resolved_mapping = UINT64_MAX;
	std::array<std::vector<std::pair<std::array<uint32_t, 4>, ResolvedBuffer*>>, 2> entries;
	const TablePair*                                                                gathered = nullptr;
	struct WrittenRange {
		std::array<uint32_t, 4> words {};
		uint32_t                special = 0;
		bool                    image_checked = false;
		uint64_t                address = 0, size = 0;
	};
	static constexpr size_t                WrittenEntries = 1024;
	std::array<WrittenRange, WrittenEntries> written;
	std::array<std::vector<uint32_t>, 2> blocks;
	std::array<ShaderRecompiler::IR::ResourceSnapshot, 2> snapshots;
	std::array<ShaderStageRuntime, 2>                     runtimes;
	std::array<PreparedBindings, 2>                       prepared;
	ShaderRecompiler::IR::ResourceSpecialization          specialization;
	std::array<std::vector<ShaderRecompiler::IR::DescriptorValue>, 2> indirect;
#if KYTY_CX_INDIRECT
	ShaderRecompiler::IR::AddressProbeCandidates                      probe;
#endif
};
void RenderExecutor::TableMemoRetired() {
	if (m_table_xpr) ++m_table_xpr->memo_epoch;
}
static void TableMemoChecked(uint32_t differ, uint64_t vs, uint64_t ps, uint64_t key) {
	if (differ == 0) return;
	LiveCounters::Add(LiveCounters::TableMemoMismatch);
	static std::atomic<uint32_t> reports {0};
	if (reports.fetch_add(1, std::memory_order_relaxed) < 8)
		std::printf("TABLE_MEMO_MISMATCH vs=0x%llx ps=0x%llx key=0x%llx variant=%d state=%d set=%d\n",
		            static_cast<unsigned long long>(vs), static_cast<unsigned long long>(ps),
		            static_cast<unsigned long long>(key), (differ & 1u) != 0 ? 1 : 0, (differ & 2u) != 0 ? 1 : 0,
		            (differ & 4u) != 0 ? 1 : 0);
}

#if KYTY_CX_SCOUT
template <typename Pair>
static uint64_t TableScoutHash(const Pair& pair) {
	const std::array<uint64_t, 3> key {pair.vs, pair.ps, static_cast<uint32_t>(pair.start_instance)};
	return XXH3_64bits(key.data(), sizeof(key));
}
#endif

void RenderExecutor::TableReset(TablePair& pair) {
#if KYTY_CX_SCOUT
	if (pair.scout_version != 0) Scout::Retire(TableScoutHash(pair), pair.scout_version);
#endif
	auto& table = *m_table_xpr;
	EXIT_IF(Role() != 0);
	++table.memo_epoch;
	for (auto& [words, set]: pair.sets) NativeXprRetireSet(set->pool, set->set);
	if (table.store == &pair) table.store = nullptr;
	if (table.last_pair == &pair) table.last_pair = nullptr;
	table.last_set = nullptr;
	++table.epoch;
	pair = {.vs = pair.vs, .ps = pair.ps, .start_instance = pair.start_instance, .generation = pair.generation,
	        .roles = std::move(pair.roles), .sightings = pair.sightings, .portable = pair.portable};
	pair.ClearSets();
}

void RenderExecutor::TableForgetStates() {
	auto* table = m_table_xpr.get();
	if (table == nullptr) return;
	++table->memo_epoch;
	table->last_pair  = nullptr;
	table->last_state = nullptr;
	table->last_set   = nullptr;
	++table->epoch;
	for (auto* pairs: {&table->pairs, &table->dispatches})
		for (auto& [key, pair]: *pairs)
			for (auto& [words, set]: pair.sets)
				if (set->transit_state != nullptr) {
					set->transit_state   = nullptr;
					set->transit_epoch   = UINT64_MAX;
					set->transit_serials = UINT64_MAX;
				}
}

void RenderExecutor::TableSweepPairs() {
	auto* table = m_table_xpr.get();
	if (table == nullptr || Role() != 0) return;
	const auto frame = m_context.FrameNumber();
	if (frame - table->pairs_swept < TableXpr::PairSweepFrames) return;
	if (kyty_local_speculations_live.load(std::memory_order_seq_cst) != 0) return;
	table->pairs_swept = frame;
	if (table->pairs.size() + table->dispatches.size() <= TableXpr::PairsKept) return;
	const auto write = CatalogWrite();
	if (kyty_local_speculations_live.load(std::memory_order_seq_cst) != 0) {
		table->pairs_swept = frame - TableXpr::PairSweepFrames;
		return;
	}
	const size_t before = table->pairs.size() + table->dispatches.size();
	for (auto* pairs: {&table->pairs, &table->dispatches})
		std::erase_if(*pairs, [&](auto& item) {
			auto& pair = item.second;
			if (pair.used_frame + TableXpr::PairKeepFrames >= frame || pair.waiting) return false;
#if KYTY_CX_SCOUT
			if (pair.scout_version != 0) Scout::Retire(TableScoutHash(pair), pair.scout_version);
#endif
			for (auto& [words, set]: pair.sets) NativeXprRetireSet(set->pool, set->set);
			return true;
		});
	const size_t after = table->pairs.size() + table->dispatches.size();
	if (after == before) return;
	++table->memo_epoch;
	table->pair_cache         = {};
	table->found_key          = {};
	table->found_pair         = nullptr;
	table->found_dispatch_key = {};
	table->found_dispatch     = nullptr;
	table->previous_dispatch  = nullptr;
	table->store              = nullptr;
	table->gathered           = nullptr;
	table->last_pair          = nullptr;
	table->last_state         = nullptr;
	table->last_set           = nullptr;
	++table->epoch;
	for (auto* pairs: {&table->pairs, &table->dispatches})
		for (auto& [key, pair]: *pairs) pair.next = nullptr;
	if (SlowLog::HitchThreshold() > 0.0)
		std::printf("[tsc %llu] table pairs %zu -> %zu\n", static_cast<unsigned long long>(__rdtsc()), before, after);
}

RenderExecutor::TableXpr& RenderExecutor::OwnTable() {
	if (!m_table_xpr) m_table_xpr = {new TableXpr {}, [](TableXpr* table) { delete table; }};
	return *m_table_xpr;
}

#if 0
void RenderExecutor::DropStaleCatalog() {
	auto* catalog = CatalogTable();
	auto& table   = OwnTable();
	if (Role() == 0 || catalog == nullptr || catalog->epoch == table.seen_epoch) return;
	table.seen_epoch = catalog->epoch;
	table.last_pair  = nullptr;
	table.last_state = nullptr;
	table.last_set   = nullptr;
	++table.memo_epoch;
	table.memo_set = nullptr;
}

void RenderExecutor::ForgetStaleCatalog() {
	if (Role() == 0) return;
	const auto read    = CatalogRead();
	auto*      catalog = CatalogTable();
	if (catalog == nullptr || m_table_xpr == nullptr || catalog->epoch == m_table_xpr->seen_epoch) return;
	auto& table              = *m_table_xpr;
	table.seen_epoch         = catalog->epoch;
	table.pair_cache         = {};
	table.found_key          = {};
	table.found_pair         = nullptr;
	table.found_dispatch_key = {};
	table.found_dispatch     = nullptr;
	table.previous_dispatch  = nullptr;
	table.store              = nullptr;
	table.gathered           = nullptr;
	table.last_pair          = nullptr;
	table.last_state         = nullptr;
	table.last_set           = nullptr;
	table.memo_set           = nullptr;
	++table.memo_epoch;
	if (m_native_xpr) {
		m_native_xpr->emit           = {};
		m_native_xpr->last_state     = nullptr;
		m_native_xpr->last_state_key = 0;
	}
}

#endif

template <typename Pair>
static void TablePrefetchPair(const Pair& pair) {
	for (size_t line = 0; line < sizeof(Pair); line += 64) __builtin_prefetch(reinterpret_cast<const char*>(&pair) + line);
}
template <typename Pair, typename Set>
static void TablePrefetchArrays(const Pair& pair, const Set* set) {
	const auto lines = [](const auto& items) {
		const auto* at    = reinterpret_cast<const char*>(items.data());
		const auto  bytes = std::min<size_t>(items.size() * sizeof(*items.data()), 256);
		for (size_t offset = 0; offset < bytes; offset += 64) __builtin_prefetch(at + offset);
	};
	for (uint32_t offset = 0; offset < std::min<uint32_t>(pair.packed_bytes, 2048); offset += 64)
		__builtin_prefetch(pair.packed_at + offset);
	for (uint32_t st = 0; st < 2 && pair.programs[st] != nullptr; ++st) {
		lines(pair.programs[st]->table_plan.buffers);
		lines(pair.programs[st]->info.buffers);
	}
	if (set != nullptr) {
		__builtin_prefetch(set);
		lines(set->words);
	}
}

template <typename Pair>
static void TablePrefetchNext(Pair*& previous, Pair& pair) {
	if (previous != nullptr && previous != &pair) previous->next = &pair;
	previous = &pair;
	auto* const next = pair.next;
	if (next == nullptr || next == &pair) return;
	using Set = std::remove_pointer_t<std::remove_cvref_t<decltype(next->last_set)>>;
	TablePrefetchArrays(*next, static_cast<const Set*>(nullptr));
	__builtin_prefetch(next->last_set);
	if (auto* const after = next->next; after != nullptr && after != &pair && after != next) TablePrefetchPair(*after);
}

RenderExecutor::TableResult RenderExecutor::TableTry(CommandBuffer& buffer, std::span<const uint64_t> commands,
                                                     uint64_t index_base, uint64_t index_bytes,
                                                     vk::IndexType index_type, const NativeXprDirectDraw* direct) {
	auto& table = OwnTable();
	DropStaleCatalog();
	TableSweepPairs();
	LiveCensus::Scope census(LiveCensus::DrawPhase, 0x7ab1e, 0);
	auto*       catalog = CatalogTable();
	auto&       cache   = *m_native_xpr;
	const auto& sh      = buffer.GetShaders();
	const auto& vs      = sh.GetVs();
	if (vs.es_regs.data_addr == 0 || vs.ls_regs.data_addr != 0 || vs.hs_regs.data_addr != 0)
		return Spec::Refuse("table: stages"), TableResult::Native;
	const auto frame      = m_context.FrameNumber();
	const auto vs_address = vs.es_regs.data_addr, ps_address = sh.GetPs().ps_regs.data_addr;
	const std::array<uint64_t, 3> pair_key {vs_address, ps_address, static_cast<uint32_t>(TableStartInstance(vs))};
	auto* own   = Role() != 0 ? nullptr : &table.pairs;
	auto* known = own != nullptr && table.found_key == pair_key ? table.found_pair : nullptr;
	if (known == nullptr) {
		const auto pair_hash = XXH3_64bits(pair_key.data(), sizeof(pair_key));
		const auto found = catalog == nullptr || Role() == 0 ? decltype(catalog->pairs.end()) {} : catalog->pairs.find(pair_hash);
		if (Role() != 0 && (catalog == nullptr || found == catalog->pairs.end())) return Spec::Refuse("table store"), TableResult::Native;
		const auto entries = (kyty_local_table_mru.load(std::memory_order_relaxed) & 1u) != 0 ? TableXpr::PairCacheEntries : 256;
		auto&      cached  = table.pair_cache[pair_hash & (entries - 1)];
		known        = own != nullptr && cached.first == pair_hash ? cached.second : nullptr;
		if (known == nullptr) {
			auto it = own != nullptr ? own->find(pair_hash) : found;
			if (own != nullptr && it == own->end()) {
				const auto write    = CatalogWrite();
				it                  = own->try_emplace(pair_hash).first;
				Spec::Refuse("pair: created");
				auto& fresh         = it->second;
				fresh.vs            = vs_address;
				fresh.ps            = ps_address;
				fresh.start_instance = TableStartInstance(vs);
			}
			known = &it->second;
			if (own != nullptr) cached = {pair_hash, known};
		}
		if (own != nullptr) {
			table.found_key  = pair_key;
			table.found_pair = known;
		}
	}
	auto& pair = *known;
	TablePrefetchPair(pair);
	if (Role() == 0) pair.used_frame = frame;
	if (pair.vs != vs_address || pair.ps != ps_address || pair.start_instance != TableStartInstance(vs))
		return Spec::Refuse("table: pair key"), TableResult::Native;
	if (const auto generation = ShaderMapGeneration(); generation != pair.Generation(Role())) {
		if (pair.programs[0] != nullptr &&
		    (ShaderHashAt(vs_address) != pair.programs[0]->shader_hash ||
		     (pair.programs[1] != nullptr && ShaderHashAt(ps_address) != pair.programs[1]->shader_hash))) {
			if (Role() != 0) return Spec::Refuse("table: shaders changed"), TableResult::Native;
			const auto write = CatalogWrite();
			pair.portable    = {};
			Spec::Refuse("reset: shader hash changed");
			TableReset(pair);
		}
		pair.Generation(Role()) = generation;
	}
	if (pair.respecialize) {
		if (Role() != 0) return Spec::Refuse("table: shaders changed"), TableResult::Native;
		const auto write = CatalogWrite();
		Spec::Refuse("reset: respecialize");
		TableReset(pair);
	}
	if (pair.refused) {
		m_table_try_refused = true;
		return Spec::Refuse("table: pair refused"), TableResult::Native;
	}
	if (FrameGen::Enabled() && pair.programs[0] != nullptr &&
	    FrameGen::IsGbufferVertexShader(pair.programs[0]->shader_hash) && (Role() != 0 || FrameGen::WantsCamera()))
		return Spec::Refuse("table: frame generation camera"), TableResult::Native;
	const auto request_store = [&] {
		if (Spec::Current() != nullptr) return Spec::Refuse("table store"), TableResult::Native;
		if (pair.waiting) {
			if (!pair.waiting->load(std::memory_order_acquire))
				return Spec::Refuse("deferred: programs compiling"), LiveCounters::Add(LiveCounters::TableStoreDeferred), TableResult::Deferred;
			pair.waiting.reset();
		}
		if (table.store_frame != frame) {
			table.store_frame    = frame;
			table.store_requests = 0;
		}
		if (pair.store_frame == frame && pair.store_state == cache.state_key)
			return Spec::Refuse("deferred: stored this frame"), LiveCounters::Add(LiveCounters::TableStoreDeferred), TableResult::Deferred;
		if (const auto budget = kyty_local_table_store_budget.load(std::memory_order_relaxed);
		    budget != 0 && table.store_requests >= budget)
			return Spec::Refuse("deferred: store budget"), LiveCounters::Add(LiveCounters::TableStoreDeferred), TableResult::Native;
		++table.store_requests;
		Spec::Refuse("store: requested");
		pair.store_frame  = frame;
		pair.store_state  = cache.state_key;
		table.store       = &pair;
		table.store_state = cache.state_key;
		return TableResult::Store;
	};
	const bool memo = Role() == 0 && TableMemoOn();
	if (memo) {
		auto& entry = table.MemoAt(&pair, cache.state_key);
		if (entry.pair == &pair && entry.key == cache.state_key && entry.epoch == table.memo_epoch &&
		    entry.variant < pair.variants.size() && pair.programs[0] != nullptr) {
			auto& memo_variant = pair.variants[entry.variant];
			bool verify = false, differs = false;
			if (const auto every = TableMemoVerifyEvery();
			    every > 0 && memo_variant.state == cache.state_key && memo_variant.pipeline != nullptr &&
			    !memo_variant.refused && ++table.memo_verify_count % static_cast<uint64_t>(every) == 0) {
				verify           = true;
				const bool hinted = (kyty_local_table_mru.load(std::memory_order_relaxed) & 2u) != 0 &&
				                    pair.variant_hint < pair.variants.size() &&
				                    pair.variants[pair.variant_hint].state == cache.state_key;
				const auto full_variant =
				    hinted ? pair.variant_hint
				           : static_cast<uint32_t>(std::ranges::find(pair.variants, cache.state_key, &TableXpr::Variant::state) -
				                                   pair.variants.begin());
				const NativeXprDrawState* full_state = nullptr;
				if (cache.last_state != nullptr && cache.last_state_key == cache.state_key) {
					full_state = cache.last_state;
				} else {
					const auto& states   = Catalog().NativeXprState().states;
					const auto  state_it = states.find(cache.state_key);
					if (state_it != states.end()) full_state = state_it->second.get();
				}
				LiveCounters::Add(LiveCounters::TableMemoVerified);
				const uint32_t differ = (full_variant != entry.variant ? 1u : 0u) | (full_state != entry.state ? 2u : 0u);
				TableMemoChecked(differ, pair.vs, pair.ps, cache.state_key);
				if (differ != 0) {
					differs     = true;
					entry.epoch = 0;
				}
			}
			if (!differs && memo_variant.state == cache.state_key && memo_variant.pipeline != nullptr &&
			    !memo_variant.refused) {
				const auto& memo_state = *entry.state;
				pair.variant_hint      = entry.variant;
				TablePrefetchArrays(pair, pair.LastSet(Role()));
				if (!NativeXprTargetsCurrent(memo_state))
					return LiveCounters::Add(LiveCounters::TableStoreTargets), request_store();
				LiveCounters::Add(LiveCounters::TableMemoHits);
				table.memo_set     = entry.set;
				table.memo_verify  = verify;
				const auto result  = TableDraw(buffer, pair, memo_variant, memo_state, commands, index_base, index_bytes,
				                               index_type, direct);
				table.memo_set     = nullptr;
				table.memo_verify  = false;
				return result;
			}
		}
	}
	const bool hint    = (kyty_local_table_mru.load(std::memory_order_relaxed) & 2u) != 0 &&
	                  pair.variant_hint < pair.variants.size() && pair.variants[pair.variant_hint].state == cache.state_key;
	const auto variant = hint ? pair.variants.begin() + pair.variant_hint
	                          : std::ranges::find(pair.variants, cache.state_key, &TableXpr::Variant::state);
	if (variant != pair.variants.end()) pair.variant_hint = static_cast<uint32_t>(variant - pair.variants.begin());
	if (variant != pair.variants.end() && variant->refused) return Spec::Refuse("table: variant refused"), TableResult::Native;
	if (pair.programs[0] == nullptr || variant == pair.variants.end() || variant->pipeline == nullptr) {
		Spec::Refuse(pair.programs[0] == nullptr ? (pair.pending ? "wait: programs pending" : "wait: no programs")
		             : variant == pair.variants.end() ? "wait: no variant" : "wait: pipeline null");
		return LiveCounters::Add(LiveCounters::TableStoreVariant), request_store();
	}
	if (cache.last_state == nullptr || cache.last_state_key != cache.state_key) {
		const auto& states   = Catalog().NativeXprState().states;
		const auto  state_it = states.find(cache.state_key);
		if (state_it == states.end()) return request_store();
		cache.last_state_key = cache.state_key;
		cache.last_state     = state_it->second.get();
	}
	const auto& draw_state = *cache.last_state;
	TablePrefetchArrays(pair, pair.LastSet(Role()));
	if (!NativeXprTargetsCurrent(draw_state)) return LiveCounters::Add(LiveCounters::TableStoreTargets), request_store();
	const auto result = TableDraw(buffer, pair, *variant, draw_state, commands, index_base, index_bytes, index_type, direct);
	if (memo && result == TableResult::Drawn)
		table.MemoAt(&pair, cache.state_key) = {&pair, cache.state_key, table.memo_epoch, &draw_state,
		                                        const_cast<TableImageSet*>(table.last_set),
		                                        static_cast<uint32_t>(variant - pair.variants.begin())};
	return result;
}

bool RenderExecutor::TableContinue(CommandBuffer& buffer, std::span<const uint64_t> commands, uint64_t index_base,
                                   uint64_t index_bytes, vk::IndexType index_type, const NativeXprDirectDraw* direct) {
	if (kyty_local_table_xpr_mode.load(std::memory_order_relaxed) < 2 || !m_table_xpr || !m_native_xpr) return false;
	auto& table = OwnTable();
	Spec::Refuse("continue: tried");
	if (table.last_pair == nullptr) return Spec::Refuse("continue: no last pair"), false;
	if (buffer.IsInvalid() || index_base == 0 || index_bytes == 0 || commands.size() > 64)
		return Spec::Refuse("continue: draw shape"), false;
	if (table.last_pair->Generation(Role()) != ShaderMapGeneration())
		return Spec::Refuse("continue: shader map generation"), false;
	if (table.last_pair->start_instance != TableStartInstance(buffer.GetShaders().GetVs()))
		return Spec::Refuse(g_table_offsets_key < 0 || table.last_pair->start_instance < 0 ? "continue: direct/indirect switch"
		                                                                                   : "continue: other offsets"), false;
	auto&       variant = table.last_pair->variants[table.last_variant];
	const auto& emit    = m_native_xpr->emit;
	if (emit.generation != m_context.GetCommandScheduler().GetDynamicState().Generation())
		return Spec::Refuse("continue: emit generation"), false;
	if (table.last_state == nullptr || emit.state != table.last_state) return Spec::Refuse("continue: emit state"), false;
	if (emit.pipeline != variant.pipeline) return Spec::Refuse("continue: emit pipeline"), false;
	if (!NativeXprTargetsCurrent(*table.last_state)) return Spec::Refuse("continue: targets"), false;
	if (TableDraw(buffer, *table.last_pair, variant, *table.last_state, commands, index_base, index_bytes, index_type,
	              direct) != TableResult::Drawn)
		return Spec::Refuse("continue: draw refused"), false;
	LiveCounters::Add(LiveCounters::TableContinued, std::max<size_t>(commands.size(), 1));
	return true;
}

static int64_t TableLearnPortable() {
	static auto& value = LiveSwitch::Find("table_learn_portable", 0);
	return Spec::On() ? value.load(std::memory_order_relaxed) : 0;
}

RenderExecutor::TableImageSet* RenderExecutor::TableResolveSet(CommandBuffer& buffer, TablePair& pair,
                                                               const TableVariant*       variant,
                                                               const NativeXprDrawState* draw_state,
                                                               std::span<const uint32_t> words) {
	(void)buffer;
	if (Spec::Current() != nullptr) return Spec::Refuse("table image set"), nullptr;
	auto       set      = std::make_unique<TableImageSet>();
	auto&      table    = OwnTable();
	auto&      textures = m_context.GetTextureCache();
	const auto epoch    = textures.ResolutionEpoch();
	auto&      buffers  = m_context.GetBufferCache();
	const auto stages   = pair.programs[1] != nullptr ? 2u : 1u;
	set->words.assign(words.begin(), words.end());
	set->validated_frame = m_context.FrameNumber();
	const auto publish = [&] {
		const auto write = CatalogWrite();
		auto&      slot  = pair.sets[XXH3_64bits(words.data(), words.size() * sizeof(uint32_t))];
		if (slot) {
			NativeXprRetireSet(slot->pool, slot->set);
			pair.ClearSets();
			++m_table_xpr->epoch;
		}
		slot = std::move(set);
		return slot.get();
	};
	const auto refuse = [&](const char* reason) {
		static std::atomic<uint32_t> reports[2] {};
		if (reports[pair.ps != 0 ? 1 : 0].fetch_add(1, std::memory_order_relaxed) < 32)
			std::printf("Table: VS 0x%llx PS 0x%llx image set refused: %s\n", static_cast<unsigned long long>(pair.vs),
			            static_cast<unsigned long long>(pair.ps), reason);
		set->refused = true;
		return publish();
	};
	size_t at = 0;
	for (uint32_t st = 0; st < stages; ++st) {
		const auto& plan     = pair.programs[st]->table_plan;
		auto&       snapshot = table.snapshots[st];
		snapshot.buffers.resize(plan.buffers.size());
		for (size_t i = 0; i < plan.buffers.size(); ++i) {
			snapshot.buffers[i]             = {};
			snapshot.buffers[i].dword_count = 4;
			std::copy_n(table.entries[st][i].first.begin(), 4, snapshot.buffers[i].dwords.begin());
		}
		const auto take = [&](uint32_t count, ShaderRecompiler::IR::DescriptorValue& value) {
			value             = {};
			value.dword_count = count;
			std::copy_n(words.begin() + static_cast<std::ptrdiff_t>(at), count, value.dwords.begin());
			at += count;
		};
		snapshot.images.resize(plan.images.size());
		for (size_t i = 0; i < plan.images.size(); ++i) take(plan.images[i].count, snapshot.images[i]);
		snapshot.samplers.resize(plan.samplers.size());
		for (size_t i = 0; i < plan.samplers.size(); ++i) take(plan.samplers[i].count, snapshot.samplers[i]);
		snapshot.flattened_srt.clear();
		uint32_t user_words = 0;
		for (const auto reg: pair.programs[st]->info.user_data_registers)
			user_words = std::max<uint32_t>(user_words, reg - pair.programs[st]->user_data_base + 1);
		snapshot.user_data.assign(user_words, 0u);
		snapshot.specialization_reads.clear();
		snapshot.image_tables.clear();
		snapshot.uniform_fill = {};
		auto& specialization = table.specialization;
		if (!m_context.GetPipelineCache().SpecializeDirectResources(*pair.plans[st], snapshot, specialization))
			return refuse("resources that do not specialize");
		if (specialization.images != pair.tables[st]->specialization.images ||
		    specialization.samplers != pair.tables[st]->specialization.samplers)
			return refuse(st == 0 ? "another specialization (VS)" : "another specialization (PS)");
		table.runtimes[st] = {.module = pair.tables[st], .resources = &snapshot};
	}
	struct Unbind {
		RenderExecutor& executor;
		~Unbind() { executor.ResetBindings(); }
	} unbind {*this};
	for (uint32_t st = 0; st < stages; ++st) PrepareBindings(table.runtimes[st], table.prepared[st]);
	for (uint32_t st = 0; st < stages; ++st) RebindImages(table.prepared[st]);
	std::array<std::vector<TableImageSet::Image>, 2> deps;
	for (uint32_t st = 0; st < stages; ++st) {
		deps[st].reserve(table.prepared[st].images.size());
		for (auto& binding: table.prepared[st].images) {
			if (!binding.image_id) continue;
			auto&       image   = textures.GetImage(binding.image_id);
			const bool  storage = binding.desc.type == TextureCache::BindingType::Storage;
			const auto  found   = draw_state == nullptr ? decltype(draw_state->targets.end()) {}
			                                            : std::ranges::find(draw_state->targets, binding.image_id,
			                                                                &NativeXprDrawState::Target::id);
			const auto* target  = draw_state == nullptr || found == draw_state->targets.end() ? nullptr : &*found;
			const auto& view    = binding.desc.view_info;
			const bool target_read = [&] {
				if (target == nullptr || (target->access & vk::AccessFlagBits2::eShaderRead) ||
				    kyty_local_table_target_read.load(std::memory_order_relaxed) == 0)
					return false;
				const auto layout = target->layout;
				if (storage) return !image.info.IsDepth() && layout == vk::ImageLayout::eGeneral;
				if (layout == vk::ImageLayout::eGeneral) return true;
				const bool pixel_feedback = layout == vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT && stages == 2 && st == 1;
				if (!image.info.IsDepth()) return pixel_feedback;
				const auto aspect     = binding.desc.view_info.aspect;
				const bool depth_read = pixel_feedback || layout == vk::ImageLayout::eDepthReadOnlyOptimal ||
				                        layout == vk::ImageLayout::eDepthStencilReadOnlyOptimal ||
				                        layout == vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal;
				const bool stencil_read = layout == vk::ImageLayout::eStencilReadOnlyOptimal ||
				                          layout == vk::ImageLayout::eDepthStencilReadOnlyOptimal ||
				                          layout == vk::ImageLayout::eDepthAttachmentStencilReadOnlyOptimal;
				return (!(aspect & vk::ImageAspectFlagBits::eDepth) || depth_read) &&
				       (!(aspect & vk::ImageAspectFlagBits::eStencil) || stencil_read);
			}();
			if (image.info.data.Empty()) return refuse("an image without data");
			if (target != nullptr && !target_read && (storage || !(target->access & vk::AccessFlagBits2::eShaderRead)))
				return refuse("a target the shader reads");
			if ((storage || (image.binding.force_general && image.binding.shader_write)) && !image.info.IsTiled())
				return refuse("a linear image the shader writes");
			auto& dep = deps[st].emplace_back();
			dep       = {.id     = binding.image_id,
			             .serial = image.serial,
			             .layout = vk::ImageLayout::eUndefined,
			             .range  = {view.base_level, view.level_count, view.base_layer, view.layer_count},
			             .desc   = binding.desc};
			if (target != nullptr) {
				dep.layout      = target->layout;
				dep.target_read = target_read ? vk::AccessFlagBits2::eShaderRead |
				                                    (storage || image.binding.shader_write ? vk::AccessFlagBits2::eShaderWrite
				                                                                           : vk::AccessFlags2 {})
				                              : vk::AccessFlags2 {};
				dep.access      = target->access | dep.target_read;
				dep.target      = true;
			} else if (image.binding.force_general && !image.info.IsDepth()) {
				dep.layout  = vk::ImageLayout::eGeneral;
				dep.access  = vk::AccessFlagBits2::eShaderRead |
				              (image.binding.shader_write ? vk::AccessFlagBits2::eShaderWrite : vk::AccessFlags2 {});
				dep.storage = image.binding.shader_write;
				dep.whole   = true;
			} else if (storage) {
				dep.layout  = vk::ImageLayout::eGeneral;
				dep.access  = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
				dep.storage = true;
			} else {
				dep.layout = image.info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
				                                  : vk::ImageLayout::eShaderReadOnlyOptimal;
			}
			binding.layout = dep.layout;
		}
	}
	using Kind = ShaderRecompiler::IR::DescriptorBindingKind;
	struct Pending {
		vk::WriteDescriptorSet write;
		size_t                 start = 0;
		bool                   image = false;
	};
	std::vector<Pending>                    pending;
	std::array<vk::DescriptorBufferInfo, 4> buffer_infos {};
	size_t                                  buffer_count = 0;
	std::vector<vk::DescriptorImageInfo>    image_infos;
	set->bda_generation = BufferCache::BdaTableGeneration();
	for (uint32_t st = 0; st < stages; ++st) {
		const auto& program  = *pair.programs[st];
		const auto& prepared = table.prepared[st];
		std::array<size_t, ShaderRecompiler::IR::ImageBindingCount> image_starts {};
		for (auto mask = program.bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
			const auto kind_index = std::countr_zero(mask);
			const auto kind       = static_cast<Kind>(kind_index);
			Pending    item;
			item.write.dstBinding      = ShaderRecompiler::IR::NativeBinding(program.stage, kind);
			item.write.descriptorType  = NativeDescriptorType(kind);
			item.write.descriptorCount = program.bindings.descriptor_counts[kind_index];
			if (ShaderRecompiler::IR::ImageBindingResourceClass(kind) != ShaderRecompiler::IR::ImageResourceClass::None) {
				item.image = true;
				item.start = image_infos.size();
				image_starts[ShaderRecompiler::IR::ImageBindingIndex(kind)] = item.start;
				image_infos.resize(image_infos.size() + item.write.descriptorCount);
			} else if (kind == Kind::Samplers) {
				item.image = true;
				item.start = image_infos.size();
				if (prepared.samplers.size() != item.write.descriptorCount) return refuse("sampler count");
				for (const auto sampler: prepared.samplers) {
					if (sampler == nullptr) return refuse("a null sampler");
					image_infos.emplace_back(sampler, nullptr, vk::ImageLayout::eUndefined);
				}
			} else if (kind == Kind::MipStats && buffer_count < buffer_infos.size()) {
				const auto& counters         = m_context.GetLodStats().CounterBuffer();
				item.start                   = buffer_count;
				buffer_infos[buffer_count++] = {counters.Handle(), 0, counters.Size()};
			} else if ((kind == Kind::BdaPagetable || kind == Kind::FaultBuffer) && buffer_count < buffer_infos.size()) {
				const auto* owner = kind == Kind::BdaPagetable ? buffers.GetBdaPageTableBuffer() : buffers.GetFaultBuffer();
				item.start        = buffer_count;
				buffer_infos[buffer_count++] = {owner->Handle(), 0, owner->Size()};
			} else {
				return refuse("a binding table mode does not have");
			}
			pending.push_back(item);
		}
		if (prepared.images.size() != program.info.images.size()) return refuse("image count");
		for (uint32_t i = 0; i < program.info.images.size(); ++i) {
			const auto& resource = program.info.images[i];
			const auto& texture  = prepared.images[i];
			if (texture.image_id && resource.mip_count != (texture.mip_views.empty() ? 1u : texture.mip_views.size()))
				return refuse("image mips");
			const auto kind  = *ShaderRecompiler::IR::DescriptorBindingForImage(resource);
			const auto start = image_starts[ShaderRecompiler::IR::ImageBindingIndex(kind)] + resource.descriptor_index;
			for (uint32_t mip = 0; mip < resource.mip_count; ++mip) {
				if (start + mip >= image_infos.size()) return refuse("image slot");
				image_infos[start + mip] = MakeImageInfo(texture, mip);
			}
		}
		if (st == 0) set->images.reserve(deps[0].size() + deps[1].size());
		for (const auto& dep: deps[st])
			if (dep.layout != vk::ImageLayout::eUndefined) {
				set->images.push_back(dep);
				set->image_ids.push_back(dep.id);
				set->holds_target |= dep.target;
				set->read_after_write |= dep.whole && dep.storage;
				if (dep.desc.info.metadata.kind != ImageMetadataKind::None)
					textures.NoteMetaUse(dep.desc.info.metadata.range.address, dep.id.index);
			}
	}
	set->set = NativeXprAllocateSet((variant != nullptr ? variant->pipeline : pair.pipeline)->descriptor_set_layout, set->pool);
	if (!set->set) return refuse("no descriptor set");
	std::vector<vk::WriteDescriptorSet> writes;
	writes.reserve(pending.size());
	for (auto& item: pending) {
		item.write.dstSet = set->set;
		if (item.image) item.write.pImageInfo = image_infos.data() + item.start;
		else item.write.pBufferInfo = buffer_infos.data() + item.start;
		writes.push_back(item.write);
	}
	if (!writes.empty())
		m_context.GetGraphics().device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0,
		                                                    nullptr);
	set->resolution_epoch = epoch;
	LiveCounters::Add(LiveCounters::TableEvaluations);
	return publish();
}

static bool TableRefreshPending(const Image& image, const TextureCache::ImageDesc& desc) {
	(void)desc;
	return TextureCache::RefreshPending(image);
}

bool RenderExecutor::TableValidateSet(TableImageSet& set) {
	auto&      textures   = m_context.GetTextureCache();
	const auto frame      = m_context.FrameNumber();
	const auto resolution = textures.ResolutionEpoch();
	if (set.bda_generation != BufferCache::BdaTableGeneration()) return false;
	if (Role() != 0) {
		auto& mine = set.ResolutionEpoch(Role());
		mine        = std::max(mine, set.RoleZeroEpoch());
	}
	if (set.ValidatedFrame(Role()) == frame && set.ResolutionEpoch(Role()) == resolution) return true;
	LiveCensus::Scope census(LiveCensus::DrawPhase, 0x7ab1e, 7);
	const auto mode  = kyty_local_table_set_proofs.load(std::memory_order_relaxed);
	const bool own   = Role() == 0 && Spec::Current() == nullptr && mode != 0;
	const bool full  = !own || textures.GcTick() >= set.touched_tick + TableImageSet::TouchTicks ||
	                  textures.CurrentFrame() >= set.touched_frame + TableImageSet::TouchTicks ||
	                  m_context.GetCommandScheduler().Current().IsInvalid();
	bool moved = set.ResolutionEpoch(Role()) != resolution;
	if (moved && own) {
		if (set.span_end == 0) {
			set.span_begin = UINT64_MAX;
			for (const auto& dep: set.images) {
				set.span_begin = std::min(set.span_begin, dep.desc.info.data.address);
				set.span_end   = std::max(set.span_end, dep.desc.info.data.End());
			}
		}
		bool known = false;
		moved      = set.span_end <= set.span_begin ||
		        textures.RegistrationsOverlapSince(set.span_begin, set.span_end - set.span_begin, set.resolution_epoch, known) ||
		        !known;
	}
	if (!full)
		for (const auto& dep: set.images) textures.m_slot_images.prefetch(dep.id);
	for (auto& dep: set.images) {
		auto* image = textures.m_slot_images.try_get(dep.id);
		if (image == nullptr || image->serial != dep.serial) return false;
		const bool held = !full && dep.proven && TableValidity(*image) == dep.validity &&
		                  (!moved || !textures.RegistrationsSince(dep.desc.info.data.address, dep.desc.info.data.size,
		                                                          set.resolution_epoch));
		if (held && mode != 2) continue;
		const auto refuse = [&] {
			if (held) {
				LiveCounters::Add(LiveCounters::TableProofMismatches);
				static std::atomic<uint32_t> reports {0};
				if (reports.fetch_add(1, std::memory_order_relaxed) < 16)
					std::printf("Table: proven image use failed: addr=0x%llx size=0x%llx serial=%llu validity=%llu\n",
					            static_cast<unsigned long long>(dep.desc.info.data.address),
					            static_cast<unsigned long long>(dep.desc.info.data.size),
					            static_cast<unsigned long long>(dep.serial), static_cast<unsigned long long>(dep.validity));
			}
			return false;
		};
		if (TableRefreshPending(*image, dep.desc)) {
			if (held) (void)refuse();
			if (Spec::Current() != nullptr) return Spec::Refuse("texture upload"), false;
			if (!image->registered || image->depth_id || image->binding.needs_rebind) return false;
			(void)textures.FindTexture(dep.id, dep.desc);
			image = textures.m_slot_images.try_get(dep.id);
			if (image == nullptr || image->serial != dep.serial || TableRefreshPending(*image, dep.desc)) return false;
		}
		if (dep.target) {
			if (!image->registered || image->binding.needs_rebind) return refuse();
			(void)textures.GetImage(dep.id);
			if (own) {
				dep.proven   = true;
				dep.validity = TableValidity(*image);
			}
			continue;
		}
		bool sampled = textures.IsSampledImageCurrent(dep.id, dep.desc, true);
		bool current = sampled || textures.IsSampledDepthCurrent(dep.id, dep.desc, set.ResolutionEpoch(Role()));
		const bool texture = dep.desc.type == TextureCache::BindingType::Texture;
		if (!current && own && (texture || dep.desc.type == TextureCache::BindingType::Storage) &&
		    !m_context.GetCommandScheduler().Current().IsInvalid()) {
			auto       desc  = dep.desc;
			const bool taken = !TextureCache::ViewFitsImage(desc) ||
			                   (textures.FindImage(desc) == dep.id && desc.view_info == dep.desc.view_info);
			image            = textures.m_slot_images.try_get(dep.id);
			if (taken && image != nullptr && image->serial == dep.serial && image->registered && !image->depth_id &&
			    !image->binding.needs_rebind) {
				if (texture && (dep.desc.info.IsDepth() || image->info.IsDepth())) {
					current = textures.IsSampledDepthCurrent(dep.id, dep.desc, textures.ResolutionEpoch());
				} else {
					(void)textures.FindTexture(dep.id, dep.desc);
					sampled = textures.IsSampledImageCurrent(dep.id, dep.desc, true);
					current = sampled || (!dep.desc.info.HasMetadata() && !dep.desc.info.HasStencil());
				}
			}
		}
		if (!current) {
			return refuse();
		}
		if (own) {
			dep.proven   = sampled && dep.desc.info.metadata.kind != ImageMetadataKind::Dcc;
			dep.validity = TableValidity(*image);
		}
	}
	set.ValidatedFrame(Role()) = frame;
	if (Role() == 0)
		std::atomic_ref<uint64_t>(set.resolution_epoch).store(resolution, std::memory_order_relaxed);
	else
		set.ResolutionEpoch(Role()) = resolution;
	if (own && full) {
		set.touched_tick  = textures.GcTick();
		set.touched_frame = textures.CurrentFrame();
	}
	return true;
}

static bool TableCodeOperand(const ShaderRecompiler::IR::TablePlan::Operand& operand) {
	using Kind = ShaderRecompiler::IR::TablePlan::Operand::Kind;
	return operand.kind == Kind::ShaderBaseLow || operand.kind == Kind::ShaderBaseHigh;
}
static uint32_t TableCodeWord(const ShaderRecompiler::IR::TablePlan::Operand& operand, uint64_t code) {
	const auto address = code + operand.value;
	return operand.kind == ShaderRecompiler::IR::TablePlan::Operand::Kind::ShaderBaseLow ? static_cast<uint32_t>(address)
	                                                                                    : static_cast<uint32_t>(address >> 32u);
}

bool RenderExecutor::TableEvaluate(TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data) {
	auto&      table     = OwnTable();
	const auto stages    = pair.programs[1] != nullptr ? 2u : 1u;
	if (const auto epoch = m_context.MappedRangesVersion(); epoch != table.mapping_epoch) {
		table.mapped.fill(UINT64_MAX);
		table.mapping_epoch = epoch;
	}
	using Kind  = ShaderRecompiler::IR::TablePlan::Operand::Kind;
	auto& words = table.words;
	words.clear();
	table.lod_stats.clear();
	struct Reader {
		RenderContext&               context;
		const std::atomic<uint64_t>* granules = context.ReadGranules();
		uint64_t                     page     = UINT64_MAX;
		bool                         direct   = false;
		bool                         Read(uint64_t address, uint32_t count, uint32_t* out) {
            const uint64_t bytes = uint64_t {count} * 4u;
            const uint64_t first = address >> 12u;
            if ((address & 3u) == 0 && bytes != 0 && first == (address + bytes - 1u) >> 12u) {
                if (first != page) {
                    page               = first;
                    const auto granule = first >> (PageManager::READ_GRANULE_BITS - 12u);
                    direct             = (address < TRACKER_ADDRESS_SIZE &&
                              ((granules[granule / 64u].load(std::memory_order_acquire) >> (granule % 64u)) & 1u) == 0) ||
                             !context.HasReadWatchers(first << 12u, 4096);
                }
                if (direct) {
                    const auto* from = reinterpret_cast<const volatile uint32_t*>(address);
                    for (uint32_t i = 0; i < count; ++i) out[i] = from[i];
                    return true;
                }
            }
            return TableReadRun(address, count, out);
		}
	} reader {m_context};
	const auto locate = [&](uint64_t low, uint64_t high, uint32_t offset, int32_t immediate, uint32_t count,
	                        uint64_t& out, bool bounded = false, uint32_t records = 0) {
		if (bounded) {
			const auto stride = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
			const auto size   = stride == 0 ? uint64_t {records} : uint64_t {stride} * records;
			if (immediate < 0 || static_cast<uint64_t>(immediate) + count * 4ull > size) return false;
		}
		const auto address = ((low | high << 32u) & 0x0000fffffffffffcull) + (offset & ~3u) +
		                     static_cast<uint64_t>(int64_t {immediate});
		for (auto page = address >> 14u; page <= (address + count * 4ull - 1) >> 14u; ++page) {
			auto& seen = table.mapped[page & 255u];
			if (seen == page) continue;
			if (!Libs::LibKernel::Memory::IsFullyMapped(page << 14u, 16384)) return false;
			seen = page;
		}
		for (auto line = address & ~63ull; line < address + count * 4ull; line += 64)
			__builtin_prefetch(reinterpret_cast<const void*>(line));
		out = address;
		return true;
	};
	const auto lod_stats = [&](const std::array<uint32_t, 8>& image, uint32_t count) {
		table.lod_stats.push_back(count >= 8u ? LodStatsReport::ImageField(image.data()) : LodStatsReport::NoCounterFlag);
	};
	const auto by_operands = [&](uint32_t st) {
		const auto& program = *pair.programs[st];
		const auto& plan    = program.table_plan;
		auto&       slots   = table.blocks[st];
		const auto  value   = [&](const ShaderRecompiler::IR::TablePlan::Operand& operand) -> uint32_t {
			if (operand.kind == Kind::Immediate) return operand.value;
			if (operand.kind == Kind::Slot) return slots[operand.value];
			if (TableCodeOperand(operand)) return TableCodeWord(operand, st == 0 ? pair.vs : pair.ps);
			const auto index = operand.value - program.user_data_base;
			return index < user_data[st].size() ? user_data[st][index] : 0;
		};
		const auto& runs = pair.runs[st];
		for (size_t begin = 0, end = 0; begin < runs.size(); begin = end) {
			for (end = begin; end < runs.size() && runs[end].level == runs[begin].level; ++end) {
				const auto& run = runs[end];
				if (!locate(value(run.read.low), value(run.read.high), value(run.read.offset), run.read.immediate, run.count,
				            table.addresses[end], run.read.bounded, run.read.bounded ? value(run.read.records) : 0))
					return false;
			}
			for (size_t i = begin; i < end; ++i)
				if (!reader.Read(table.addresses[i], runs[i].count, slots.data() + runs[i].first)) return false;
		}
		for (size_t i = 0; i < plan.images.size(); ++i) {
			const auto& descriptor = plan.images[i];
			std::array<uint32_t, 8> image {};
			for (uint32_t w = 0; w < descriptor.count; ++w) {
				image[w] = value(descriptor.words[w]);
				words.push_back(image[w] & ~NativeXprLodStatsMask[w]);
			}
			if (i < program.bindings.mip_stats_count) lod_stats(image, descriptor.count);
		}
		for (const auto& descriptor: plan.samplers)
			for (uint32_t w = 0; w < descriptor.count; ++w) words.push_back(value(descriptor.words[w]));
		return true;
	};
#if KYTY_CX_INDIRECT
	const auto indirect = [&](uint32_t st) {
		const auto& program = *pair.programs[st];
		auto&       out     = table.indirect[st];
		out.clear();
		if (program.table_plan.indirect.empty()) return true;
		thread_local const auto runtime = [] {
			ShaderRecompiler::IR::SrtRuntime value;
			MaterializationReaders(value);
			return value;
		}();
		const auto& slots   = table.blocks[st];
		const auto  operand = [&](const ShaderRecompiler::IR::TablePlan::Operand& value) -> uint32_t {
			if (value.kind == Kind::Immediate) return value.value;
			if (value.kind == Kind::Slot) return slots[value.value];
			if (TableCodeOperand(value)) return TableCodeWord(value, st == 0 ? pair.vs : pair.ps);
			const auto index = value.value - program.user_data_base;
			return index < user_data[st].size() ? user_data[st][index] : 0;
		};
		const auto address = [&](const ShaderRecompiler::IR::TablePlan::Operand& low,
		                         const ShaderRecompiler::IR::TablePlan::Operand& high) {
			return (uint64_t {operand(low)} | uint64_t {operand(high)} << 32u) & 0x0000ffffffffffffull;
		};
		using Plan = ShaderRecompiler::IR::TablePlan;
		for (const auto& entry: program.table_plan.indirect) {
			auto& found = table.probe;
			if (!ShaderRecompiler::IR::EnumerateAddressProbe(entry.probe, program.info.images[entry.root],
			                                                 address(entry.records_low, entry.records_high),
			                                                 address(entry.heap_low, entry.heap_high), runtime, found))
				return false;
			const auto count = static_cast<uint32_t>(found.descriptors.size());
			if (count == 0 || count > Plan::IndirectCapacity || found.keys.size() > Plan::IndirectMaxKeys ||
			    found.candidates.size() != found.keys.size())
				return false;
			std::array<uint32_t, Plan::IndirectCapacity> order {}, position {};
			for (uint32_t j = 0; j < count; ++j) order[j] = j;
			std::sort(order.begin() + 1, order.begin() + count,
			          [&](uint32_t a, uint32_t b) { return found.descriptors[a].dwords < found.descriptors[b].dwords; });
			for (uint32_t j = 0; j < count; ++j) position[order[j]] = j;
			for (uint32_t j = 0; j < count; ++j) out.push_back(found.descriptors[order[j]]);
			std::array<uint32_t, Plan::IndirectMaxKeys> by_key {};
			const auto keys = static_cast<uint32_t>(found.keys.size());
			for (uint32_t k = 0; k < keys; ++k) by_key[k] = k;
			std::sort(by_key.begin(), by_key.begin() + keys,
			          [&](uint32_t a, uint32_t b) { return found.keys[a] < found.keys[b]; });
			auto* mapping = table.blocks[st].data() + entry.mapping;
			mapping[0]    = keys;
			for (uint32_t k = 0; k < keys; ++k) {
				mapping[1 + 2 * k] = found.keys[by_key[k]];
				mapping[2 + 2 * k] = position[found.candidates[by_key[k]]];
			}
			words.push_back(count);
			for (const auto& candidate: out) words.insert(words.end(), candidate.dwords.begin(), candidate.dwords.end());
		}
		return true;
	};
#else
	const auto indirect = [&](uint32_t) { return true; };
#endif
	for (uint32_t st = 0; st < stages; ++st) {
		const auto& program = *pair.programs[st];
		const auto& form    = pair.packed[st];
		auto& slots = table.blocks[st];
		slots.resize(program.table_plan.BlockDwords());
		if (table.addresses.size() < pair.runs[st].size()) table.addresses.resize(pair.runs[st].size());
		if (!form.valid || form.runs.size() != pair.runs[st].size() || user_data[st].size() < form.user_words ||
		    user_data[st].size() > 64) {
			if (!by_operands(st) || !indirect(st)) return false;
			continue;
		}
		const std::array<const uint32_t*, 3> bases {user_data[st].data(), slots.data(), form.constants.data()};
		const auto get = [&](uint16_t operand) { return bases[operand >> 14u][operand & 0x3fffu]; };
		uint32_t   begin = 0;
		for (const auto end: form.level_ends) {
			for (auto i = begin; i < end; ++i) {
				const auto& run     = form.runs[i];
				const bool  bounded = run.records != TablePair::PackedRun::NotBounded;
				if (!locate(get(run.address[0]), get(run.address[1]), get(run.address[2]), run.immediate, run.count,
				            table.addresses[i], bounded, bounded ? get(run.records) : 0))
					return false;
			}
			for (auto i = begin; i < end; ++i)
				if (!reader.Read(table.addresses[i], form.runs[i].count, slots.data() + form.runs[i].first)) return false;
			begin = end;
		}
		const auto first = words.size();
		words.resize(first + form.image_words.size());
		auto*  out = words.data() + first;
		size_t at  = 0;
		for (size_t i = 0; i < form.image_counts.size(); ++i) {
			std::array<uint32_t, 8> image {};
			const uint32_t          count = form.image_counts[i];
			for (uint32_t w = 0; w < count; ++w) {
				image[w]    = get(form.image_words[at + w]);
				out[at + w] = image[w] & ~NativeXprLodStatsMask[w];
			}
			at += count;
			if (i < program.bindings.mip_stats_count) lod_stats(image, count);
		}
		for (; at < form.image_words.size(); ++at) out[at] = get(form.image_words[at]);
		if (!indirect(st)) return false;
	}
	return true;
}

RenderExecutor::TableImageSet* RenderExecutor::TableSet(CommandBuffer& buffer, TablePair& pair,
                                                        const TableVariant*       variant,
                                                        const NativeXprDrawState* draw_state) {
	auto&       table = OwnTable();
	const auto& words = table.words;
	const auto  frame = m_context.FrameNumber();
	auto*       set   = pair.LastSet(Role());
	if (set == nullptr || set->words != words) {
		auto* const last     = set;
		auto* const previous = pair.PreviousSet(Role());
		auto* const memo     = std::exchange(table.memo_set, nullptr);
		const bool  verify   = std::exchange(table.memo_verify, false);
		if (memo != nullptr && memo != last && !memo->refused && memo->words == words) {
			set = memo;
			if (verify) {
				const auto found = pair.sets.find(XXH3_64bits(words.data(), words.size() * sizeof(uint32_t)));
				auto* const full = found != pair.sets.end() && found->second->words == words ? found->second.get() : nullptr;
				if (full != memo) {
					TableMemoChecked(4u, pair.vs, pair.ps, 0);
					set = full;
					if (set == nullptr || (set->refused && frame >= set->validated_frame + TableImageSet::RetryFrames))
						set = TableResolveSet(buffer, pair, variant, draw_state, words);
				}
			}
		} else if ((kyty_local_table_mru.load(std::memory_order_relaxed) & 4u) != 0 && previous != nullptr && previous != last &&
		    !previous->refused && previous->words == words) {
			set = previous;
		} else {
			const auto found = pair.sets.find(XXH3_64bits(words.data(), words.size() * sizeof(uint32_t)));
			set              = found != pair.sets.end() && found->second->words == words ? found->second.get() : nullptr;
			const auto retry_due = [&](const TableImageSet& s) {
				if (s.retry_wait != nullptr && TableLearnPortable() == 2) return s.retry_wait->load(std::memory_order_acquire);
				return frame >= s.validated_frame + TableImageSet::RetryFrames;
			};
			if (set == nullptr || (set->refused && retry_due(*set))) {
				auto* const retried = set;
				set                 = TableResolveSet(buffer, pair, variant, draw_state, words);
				(void)retried;
			}
		}
		if (last != nullptr && pair.LastSet(Role()) == last) pair.PreviousSet(Role()) = last;
		pair.LastSet(Role()) = set;
		if (set == nullptr) return Spec::Refuse("set: none resolved"), nullptr;
	}
	set->used_frame.store(frame, std::memory_order_relaxed);
	if (set->refused) return LiveCounters::Add(LiveCounters::TableRefusedSets), Spec::Refuse("table: refused set"), nullptr;
	if (!TableValidateSet(*set)) {
		Spec::Refuse("set: validation failed");
		if (Spec::Current() != nullptr) return Spec::Refuse("table: set validation"), nullptr;
		const auto write = CatalogWrite();
		if (table.last_set == set) table.last_set = nullptr;
		pair.ClearSets();
		++table.epoch;
		const auto found = pair.sets.find(XXH3_64bits(set->words.data(), set->words.size() * sizeof(uint32_t)));
		NativeXprRetireSet(set->pool, set->set);
		if (found != pair.sets.end() && found->second.get() == set) pair.sets.erase(found);
		return nullptr;
	}
	if (Role() == 0 && frame >= table.collected + 60) {
		const auto write = CatalogWrite();
		table.collected  = frame;
		for (auto* pairs: {&table.pairs, &table.dispatches})
			for (auto& [key, held]: *pairs)
				std::erase_if(held.sets, [&](auto& item) {
					if (item.second->used_frame.load(std::memory_order_relaxed) + 60 >= frame) return false;
					held.ClearSets();
					if (table.last_set == item.second.get()) table.last_set = nullptr;
					++table.epoch;
					NativeXprRetireSet(item.second->pool, item.second->set);
					return true;
				});
	}
	return set;
}

void RenderExecutor::TableGather(const TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data) {
	using Kind        = ShaderRecompiler::IR::TablePlan::Operand::Kind;
	auto&      table  = OwnTable();
	const auto stages = pair.programs[1] != nullptr ? 2u : 1u;
	for (uint32_t st = 0; st < stages; ++st) {
		const auto& program = *pair.programs[st];
		const auto& plan    = program.table_plan;
		const auto& block   = table.blocks[st];
		auto&       entries = table.entries[st];
		entries.resize(plan.buffers.size());
		const auto& form = pair.packed[st];
		const bool  flat = form.valid && form.buffer_words.size() == 4 * plan.buffers.size() &&
		                  user_data[st].size() >= form.user_words && user_data[st].size() <= 64;
		const std::array<const uint32_t*, 3> bases {user_data[st].data(), block.data(), form.constants.data()};
		const bool memo = kyty_local_table_gather_memo.load(std::memory_order_relaxed) != 0;
		uint32_t   memo_hits = 0;
		for (size_t i = 0; i < plan.buffers.size(); ++i) {
			auto& [words4, entry] = entries[i];
			std::array<uint32_t, 4> fresh;
			if (flat) {
				for (uint32_t w = 0; w < 4; ++w) {
					const auto operand = form.buffer_words[4 * i + w];
					fresh[w]           = bases[operand >> 14u][operand & 0x3fffu];
				}
			} else {
				for (uint32_t w = 0; w < 4; ++w) {
					const auto& operand = plan.buffers[i][w];
					fresh[w] = operand.kind == Kind::Immediate ? operand.value
					           : operand.kind == Kind::Slot    ? block[operand.value]
					           : TableCodeOperand(operand)     ? TableCodeWord(operand, st == 0 ? pair.vs : pair.ps)
					           : operand.value - program.user_data_base < user_data[st].size()
					               ? user_data[st][operand.value - program.user_data_base]
					               : 0;
				}
			}
			if (memo && entry != nullptr && words4 == fresh) {
				++memo_hits;
			} else {
				words4 = fresh;
				entry  = &table.resolved[XXH3_64bits(words4.data(), sizeof(words4)) & (TableXpr::ResolvedEntries - 1)];
			}
			__builtin_prefetch(entry);
		}
		if (memo_hits != 0) LiveCounters::Add(LiveCounters::TableGatherMemoHits, memo_hits);
	}
	table.gathered = &pair;
}

bool RenderExecutor::TableBlocks(TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data,
                                 std::array<vk::DeviceAddress, 2>& blocks, bool* writes) {
	using Kind         = ShaderRecompiler::IR::TablePlan::Operand::Kind;
	auto&      table        = OwnTable();
	auto&      buffer_cache = m_context.GetBufferCache();
	auto&      textures     = m_context.GetTextureCache();
	auto&      upload       = buffer_cache.GetTableUploadBuffer();
	const auto stages       = pair.programs[1] != nullptr ? 2u : 1u;
	const auto start_epoch  = textures.StartEpoch();
	const auto image_starts = [&](uint64_t address) {
		auto& known = table.no_image_start[(address * 0x9e3779b97f4a7c15ull) >> 54u];
		if (known.first == address && known.second == start_epoch) return false;
		if (textures.HasImageStartingAt(address)) return true;
		known = {address, start_epoch};
		return false;
	};
	if (table.resolved_start != start_epoch || table.resolved_mapping != table.mapping_epoch) {
		table.resolved_start   = start_epoch;
		table.resolved_mapping = table.mapping_epoch;
		++table.special_generation;
	}
	std::array<std::pair<uint64_t, uint64_t>, 4> region_epochs;
	region_epochs.fill({UINT64_MAX, 0});
	const auto cpu_epoch = [&](uint64_t address, uint64_t size) {
		const auto region = address / TRACKER_REGION_SIZE;
		if (region != (address + size - 1) / TRACKER_REGION_SIZE) return buffer_cache.CpuModificationEpoch(address, size);
		auto& known = region_epochs[region & 3u];
		if (known.first != region) known = {region, buffer_cache.CpuModificationEpoch(address, size)};
		return known.second;
	};
	const auto range_clean = [&](uint64_t address, uint64_t size) {
		for (auto region = address / TRACKER_REGION_SIZE; region <= (address + size - 1) / TRACKER_REGION_SIZE; ++region)
			if (buffer_cache.DeferredProtectionPending(region * TRACKER_REGION_SIZE)) return false;
		return !buffer_cache.IsRegionCpuModified(address, size);
	};
	const vk::DeviceAddress null_device = upload.BufferDeviceAddress();
	const auto word_of = [](const std::array<uint32_t, 4>& words4) -> std::pair<uint32_t, uint32_t> {
		ShaderBufferResource descriptor;
		std::copy_n(words4.begin(), 4, descriptor.fields);
		if (descriptor.Type() != 0) return {0, 0};
		return {words4[3], descriptor.Stride()};
	};
	const auto requested_of = [](const std::array<uint32_t, 4>& words4) -> uint64_t {
		ShaderBufferResource descriptor;
		std::copy_n(words4.begin(), 4, descriptor.fields);
		if (descriptor.Type() != 0 || descriptor.Base48() == 0) return 0;
		return descriptor.Stride() != 0 ? uint64_t {descriptor.Stride()} * descriptor.NumRecords()
		                                : uint64_t {descriptor.NumRecords()};
	};
	const auto range_of = [](const std::array<uint32_t, 4>& words4) -> std::pair<uint64_t, uint64_t> {
		ShaderBufferResource descriptor;
		std::copy_n(words4.begin(), 4, descriptor.fields);
		const auto requested = descriptor.Stride() != 0 ? uint64_t {descriptor.Stride()} * descriptor.NumRecords()
		                                                : uint64_t {descriptor.NumRecords()};
		const auto address   = descriptor.Base48();
		return {address, descriptor.Type() != 0 || address == 0 || requested == 0
		                     ? 0
		                     : Libs::LibKernel::Memory::ClampRangeSizeOrZero(address, requested)};
	};
	const auto specialized = [&](uint32_t st, size_t i, const std::array<uint32_t, 4>& words4) {
		const auto& resources = pair.programs[st]->info.buffers;
		const auto& specified = pair.tables[st]->specialization.buffers;
		return i < specified.size() &&
		       TableBufferSpecialized(resources[i], specified[i], words4,
		                              i < 32 && ((pair.programs[st]->table_plan.runtime_formats >> i) & 1u) != 0);
	};
	if (table.gathered != &pair) TableGather(pair, user_data);
	table.gathered = nullptr;
	for (uint32_t st = 0; st < stages; ++st) {
		const auto& program = *pair.programs[st];
		const auto& plan    = program.table_plan;
		auto&       block   = table.blocks[st];
		auto&       entries = table.entries[st];
		for (size_t n = 0; n < 2 * plan.buffers.size(); ++n) {
			const auto  i         = n < plan.buffers.size() ? n : n - plan.buffers.size();
			const auto& words4    = entries[i].first;
			const auto& resource  = program.info.buffers[i];
			const bool  formatted = resource.formatted;
			if ((resource.written || resource.atomic) != (n < plan.buffers.size())) continue;
			if (!specialized(st, i, words4)) {
				if (Role() == 0 && i < 32 && ((plan.runtime_formats >> i) & 1u) != 0 && ((pair.portable[st] >> i) & 1u) == 0) {
					const auto write = CatalogWrite();
					pair.portable[st] |= 1u << i;
					pair.respecialize = true;
				}
				return Spec::Refuse("blocks: another V# specialization"), false;
			}
			if (resource.written || resource.atomic) {
				if (writes != nullptr && resource.written && (uint64_t {words4[0]} | (words4[1] & 0xffffu)) != 0 &&
				    words4[2] != 0)
					*writes = true;
				auto& proof = table.written[XXH3_64bits(words4.data(), sizeof(words4)) & (TableXpr::WrittenEntries - 1)];
				if (proof.special != table.special_generation || proof.words != words4) {
					const auto [clamped_address, clamped_size] = range_of(words4);
					proof = {.words = words4, .special = table.special_generation, .address = clamped_address, .size = clamped_size};
				}
				const auto address = proof.address, size = proof.size;
				vk::DeviceAddress device = null_device;
				if (size != 0) {
					if (address % 4 != 0) return Spec::Refuse("blocks: written unaligned"), false;
					if (formatted && !proof.image_checked) {
						if (image_starts(address)) return Spec::Refuse("blocks: written image start"), false;
						proof.image_checked = true;
					}
					const auto [owner, offset] = buffer_cache.ObtainBuffer(address, size, true, formatted);
					region_epochs.fill({UINT64_MAX, 0});
					if (owner == nullptr || buffer_cache.GetBufferIfLive(buffer_cache.FindBuffer(address, size)) != owner)
						return Spec::Refuse("blocks: written owner"), false;
					if (formatted) textures.InvalidateMemoryFromGPU(address, size);
					device = owner->BufferDeviceAddress() + offset;
				}
				const auto [word, stride]      = word_of(words4);
				block[plan.BufferDword(i)]     = static_cast<uint32_t>(device);
				block[plan.BufferDword(i) + 1] = static_cast<uint32_t>(device >> 32u);
				block[plan.BufferDword(i) + 2] = static_cast<uint32_t>(std::min<uint64_t>(size / 4, UINT32_MAX));
				block[plan.BufferDword(i) + 3] = word;
				block[plan.BufferDword(i) + 4] = stride;
				continue;
			}
			auto& held    = *entries[i].second;
			bool  current = held.words == words4 && (held.special == 0 || held.special == table.special_generation);
			if (current && (held.registration & TableXpr::RingCopyMark) != 0) {
				if (Role() == 0 && Spec::Current() == nullptr &&
				    held.registration == (TableXpr::RingCopyMark | DrawWindow()) &&
				    cpu_epoch(held.address, held.size) == held.cpu_epoch &&
				    !buffer_cache.IsRegionGpuModified(held.address, held.size)) {
					LiveCounters::Add(LiveCounters::TableRingShared);
					LiveCounters::Add(LiveCounters::TableRingSharedBytes, held.size);
					const auto [word, stride]      = word_of(words4);
					block[plan.BufferDword(i)]     = static_cast<uint32_t>(held.device);
					block[plan.BufferDword(i) + 1] = static_cast<uint32_t>(held.device >> 32u);
					block[plan.BufferDword(i) + 2] = held.dwords;
					block[plan.BufferDword(i) + 3] = word;
					block[plan.BufferDword(i) + 4] = stride;
					continue;
				}
				current = false;
			}
			if (const auto* spec = Spec::Current();
			    current && held.size != 0 && spec != nullptr && spec->JournalIntersects(held.address, held.size))
				current = false;
			if (const auto registration = buffer_cache.RegistrationEpoch();
			    current && held.size != 0 && held.registration != registration) {
				for (auto region = held.address / TRACKER_REGION_SIZE, last = (held.address + held.size - 1) / TRACKER_REGION_SIZE;
				     current && region <= last; ++region)
					current = buffer_cache.RegionRegistrationEpoch(region * TRACKER_REGION_SIZE) <= held.registration;
				if (current) held.registration = registration;
			}
			if (current && held.size != 0) {
				if (const auto epoch = cpu_epoch(held.address, held.size); epoch == 0 || epoch != held.cpu_epoch) {
					LiveCensus::Scope reprove(LiveCensus::DrawPhase, 0x7ab1e, 6);
					current = range_clean(held.address, held.size);
					if (current) held.cpu_epoch = epoch;
				}
			}
			if (!current) {
				LiveCensus::Scope resolve(LiveCensus::DrawPhase, 0x7ab1e, 5);
				held = {};
				TableXpr::ResolvedBuffer fresh {.words = words4, .device = null_device};
				const auto [address, size] = range_of(words4);
				if (formatted || size < requested_of(words4)) fresh.special = table.special_generation;
				if (size != 0) {
					if (address % 4 != 0 || (formatted && image_starts(address)))
						return Spec::Refuse(address % 4 != 0 ? "blocks: read unaligned" : "blocks: read image start"), false;
					auto* const spec        = Spec::Current();
					const bool  cpu_written = buffer_cache.IsRegionCpuDirty(address, size) &&
					                         !buffer_cache.IsRegionGpuModified(address, size);
					if (cpu_written && size <= BufferCache::CACHING_PAGESIZE &&
					    (spec == nullptr || !spec->JournalIntersects(address, size))) {
						auto* const shared = spec == nullptr && Role() == 0
						                         ? &table.ring_copies[((address ^ (size << 40u)) * 0x9e3779b97f4a7c15ull) >> 52u]
						                         : nullptr;
						const auto observation = DrawWindow();
						const auto tick        = m_context.GetCommandScheduler().CurrentTick();
						const auto cpu         = buffer_cache.CpuModificationEpoch(address, size);
						vk::DeviceAddress device = 0;
						if (shared != nullptr && cpu != 0 && shared->address == address && shared->size == size &&
						    shared->observation == observation && shared->tick == tick && shared->cpu_epoch == cpu) {
							device = shared->device;
							LiveCounters::Add(LiveCounters::TableRingShared);
							LiveCounters::Add(LiveCounters::TableRingSharedBytes, size);
						} else if (auto [mapped, offset] = upload.Map(size, 16, false);
						           mapped != nullptr && Libs::LibKernel::Memory::TryReadBacking(address, mapped, size)) {
							upload.Commit();
							if (spec != nullptr) spec->NoteCleanRead(address, size);
							device = upload.BufferDeviceAddress() + offset;
							LiveCounters::Add(LiveCounters::TableRingCopies);
							LiveCounters::Add(LiveCounters::TableRingBytes, size);
							if (shared != nullptr && cpu != 0)
								*shared = {.address     = address,
								           .size        = size,
								           .observation = observation,
								           .tick        = tick,
								           .cpu_epoch   = cpu,
								           .device      = device};
						}
						if (device != 0 && shared != nullptr && cpu != 0)
							held = {.words        = words4,
							        .dwords       = static_cast<uint32_t>(size / 4),
							        .special      = fresh.special,
							        .device       = device,
							        .address      = address,
							        .size         = size,
							        .cpu_epoch    = cpu,
							        .registration = TableXpr::RingCopyMark | observation};
						if (device != 0) {
							const auto [word, stride]      = word_of(words4);
							block[plan.BufferDword(i)]     = static_cast<uint32_t>(device);
							block[plan.BufferDword(i) + 1] = static_cast<uint32_t>(device >> 32u);
							block[plan.BufferDword(i) + 2] = static_cast<uint32_t>(size / 4);
							block[plan.BufferDword(i) + 3] = word;
							block[plan.BufferDword(i) + 4] = stride;
							continue;
						}
					}
					if (cpu_written) buffer_cache.EnsureBufferContents(address, size);
					const auto registration_before = buffer_cache.RegistrationEpoch();
					const auto [owner, offset] = buffer_cache.ObtainBuffer(address, size, false, formatted);
					region_epochs.fill({UINT64_MAX, 0});
					fresh.cpu_epoch = buffer_cache.CpuModificationEpoch(address, size);
					if (owner == nullptr || buffer_cache.GetBufferIfLive(buffer_cache.FindBuffer(address, size)) != owner ||
					    buffer_cache.IsRegionCpuModified(address, size))
						return Spec::Refuse(owner == nullptr ? "blocks: read no owner"
						                    : buffer_cache.IsRegionCpuModified(address, size) ? "blocks: read still CPU-modified"
						                                                                       : "blocks: read owner not live"),
						       false;
					if (Spec::Current() != nullptr && buffer_cache.IsRegionCpuDirty(address, size)) fresh.cpu_epoch = 0;
					fresh.device       = owner->BufferDeviceAddress() + offset;
					fresh.dwords       = static_cast<uint32_t>(std::min<uint64_t>(size / 4, UINT32_MAX));
					fresh.address      = address;
					fresh.size         = size;
					fresh.registration = Spec::Current() != nullptr ? registration_before : buffer_cache.RegistrationEpoch();
				}
				held = fresh;
			}
			if (auto* spec = Spec::Current(); spec != nullptr && held.size != 0) spec->NoteBufferRange(held.address, held.size);
			const auto [word, stride]      = word_of(words4);
			block[plan.BufferDword(i)]     = static_cast<uint32_t>(held.device);
			block[plan.BufferDword(i) + 1] = static_cast<uint32_t>(held.device >> 32u);
			block[plan.BufferDword(i) + 2] = held.dwords;
			block[plan.BufferDword(i) + 3] = word;
			block[plan.BufferDword(i) + 4] = stride;
		}
		if (plan.BlockDwords() != 0) {
			const auto copied = Spec::Current() == nullptr
			                        ? std::optional<uint64_t> {upload.Copy(block.data(), block.size() * sizeof(uint32_t), 16)}
			                        : upload.TryCopy(block.data(), block.size() * sizeof(uint32_t), 16);
			if (!copied) return Spec::Refuse("table: upload ring busy"), false;
			blocks[st] = upload.BufferDeviceAddress() + *copied;
		}
	}
	return true;
}

void RenderExecutor::TablePrepareBda(const TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data) {
	auto&                                    table = OwnTable();
	std::array<const ShaderStageRuntime*, 2> stages {};
	uint32_t                                 count = 0;
	for (uint32_t st = 0; st < 2; ++st) {
		const auto* program = pair.programs[st];
		if (program == nullptr || !program->info.uses_dma) continue;
		auto& snapshot = table.snapshots[st];
		snapshot.user_data.assign(user_data[st].begin(), user_data[st].end());
		snapshot.flattened_srt.assign(table.blocks[st].begin(),
		                              table.blocks[st].begin() + static_cast<std::ptrdiff_t>(program->table_plan.slots.size()));
		table.runtimes[st] = {.module = pair.tables[st], .resources = &snapshot};
		m_context.CacheDmaBases(table.runtimes[st]);
		stages[count++] = &table.runtimes[st];
	}
	if (count == 0) return;
	const bool reach_synchronized = m_context.SynchronizeDmaFootprint(std::span {stages.data(), count});
	const bool writes = (pair.programs[0] != nullptr && pair.programs[0]->has_address_writes) ||
	                    (pair.programs[1] != nullptr && pair.programs[1]->has_address_writes);
	m_context.PrepareBda(writes, !reach_synchronized);
}

vk::AccessFlags2 RenderExecutor::TableTargetRead(const TableImageSet& set, ImageId id) {
	vk::AccessFlags2 read {};
	for (const auto& dep: set.images)
		if (dep.target && dep.id == id) read |= dep.target_read;
	return read;
}

static void TableSerialsUnsafe(const char* what) {
	static std::atomic<uint32_t> count {0};
	LiveCounters::Add(LiveCounters::SerialsUnsafe);
	if (const auto n = count.fetch_add(1, std::memory_order_relaxed); n < 32)
		std::printf("TABLE-SERIALS UNSAFE: %s (%u)\n", what, n + 1);
}

template<typename State>
static uint64_t TableTargetSerials(const State& draw_state) {
	uint64_t sum = ImageSerials::Unattributed();
	for (const auto id: draw_state.target_ids) sum += ImageSerials::Read(id.index);
	return sum;
}
template<typename Set>
static uint64_t TableSetSerials(const Set& set) {
	uint64_t sum = ImageSerials::Unattributed();
	for (const auto id: set.image_ids) sum += ImageSerials::Read(id.index);
	return sum;
}
template<typename Set, typename State>
static bool TableSetApart(const Set& set, const State& draw_state) {
	if (set.holds_target) return false;
	for (const auto id: set.image_ids)
		if (std::ranges::find(draw_state.target_ids, id) != draw_state.target_ids.end()) return false;
	return true;
}

bool RenderExecutor::TableTransit(const TableImageSet& set, CommandRecorder vk_buffer,
                                  const NativeXprDrawState* draw_state) {
	auto& textures = m_context.GetTextureCache();
	for (const auto& dep: set.images) {
		auto* image = textures.m_slot_images.try_get(dep.id);
		if (image == nullptr || image->serial != dep.serial) return false;
		const auto target = draw_state == nullptr
		                        ? decltype(draw_state->targets.end()) {}
		                        : std::ranges::find(draw_state->targets, dep.id, &NativeXprDrawState::Target::id);
		const bool is_target = draw_state != nullptr && target != draw_state->targets.end();
		if (is_target != dep.target ||
		    (is_target && (target->layout != dep.layout || (target->access | dep.target_read) != dep.access)))
			return false;
		if (is_target) continue;
		if ((dep.storage && !image->SafeToDownload()) || TableRefreshPending(*image, dep.desc)) return false;
		if (dep.desc.info.metadata.kind == ImageMetadataKind::Dcc) {
			if (image->info.IsVolume()) return false;
			if (!textures.ColorClearSettled(dep.desc, dep.clear_memo)) return false;
		}
	}
	for (const auto& dep: set.images)
		if (!dep.target)
			textures.m_slot_images[dep.id].Transit(dep.layout, dep.access, dep.whole ? std::nullopt : std::optional {dep.range},
			                                       vk_buffer);
		else if (dep.target_read)
			textures.m_slot_images[dep.id].Transit(dep.layout, dep.access, std::nullopt, vk_buffer);
	return true;
}

#if KYTY_CX_SCOUT
template <typename Pair>
static void TableScoutPublish(Pair& pair) {
	if (!Scout::Running() || pair.programs[0] == nullptr) return;
	auto view    = std::make_unique<Scout::View>();
	view->key    = {pair.vs, pair.ps, static_cast<uint32_t>(pair.start_instance)};
	view->stages = pair.programs[1] != nullptr ? 2u : 1u;
	for (uint32_t st = 0; st < view->stages; ++st) {
		const auto& program = *pair.programs[st];
		const auto& plan    = program.table_plan;
		const auto& form    = pair.packed[st];
		if (!form.valid || form.runs.size() != pair.runs[st].size() || form.user_words > 32 ||
#if KYTY_CX_INDIRECT
		    !plan.indirect.empty() ||
#endif
		    form.buffer_words.size() != 4 * plan.buffers.size())
			return;
		auto& stage = view->stage[st];
		for (const auto& run: form.runs) {
			stage.runs.push_back({run.address, run.records, run.immediate, run.first, run.count});
			stage.slot_words += run.count;
		}
		if (stage.slot_words > 0xffff) return;
		std::vector<uint16_t> order(stage.runs.size());
		for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<uint16_t>(i);
		std::ranges::sort(order, {}, [&](uint16_t i) { return stage.runs[i].first; });
		uint16_t at = 0;
		for (const auto i: order) {
			auto& run = stage.runs[i];
			run.at    = at;
			at        = static_cast<uint16_t>(at + run.count);
			if (!stage.segments.empty() && stage.segments.back().first + stage.segments.back().count == run.first)
				stage.segments.back().count = static_cast<uint16_t>(stage.segments.back().count + run.count);
			else
				stage.segments.push_back({run.first, run.count});
		}
		stage.level_ends.assign(form.level_ends.begin(), form.level_ends.end());
		stage.image_words.assign(form.image_words.begin(), form.image_words.end());
		stage.buffer_words.assign(form.buffer_words.begin(), form.buffer_words.end());
		stage.image_counts.assign(form.image_counts.begin(), form.image_counts.end());
		stage.constants.assign(form.constants.begin(), form.constants.end());
		stage.user_words      = form.user_words;
		stage.block_dwords    = static_cast<uint32_t>(plan.BlockDwords());
		stage.lod_stats_count = static_cast<uint32_t>(program.bindings.mip_stats_count);
		stage.buffers         = static_cast<uint32_t>(plan.buffers.size());
	}
	view->version = pair.scout_version = Scout::NextVersion();
	Scout::Publish(view.release(), TableScoutHash(pair));
}

template <typename Pair>
static const Scout::Ticket* TableScoutTicket(const Pair& pair, std::array<std::span<const uint32_t>, 2> user_data,
                                             uint32_t stages, RenderContext& resources, const Scout::Ticket* ahead) {
	LiveCounters::Add(LiveCounters::ScoutChecks);
	if (pair.scout_version == 0) return LiveCounters::Add(LiveCounters::ScoutLostNoView), nullptr;
	const bool  pipelined = Scout::Mode() == 3;
	const auto* ticket    = pipelined ? ahead : Scout::Current();
	if (ticket == nullptr) {
		LiveCounters::Add(LiveCounters::ScoutLostMissing);
		static_assert(LiveCounters::ScoutMissFenced - LiveCounters::ScoutMissUnwalked == static_cast<uint32_t>(Scout::Reason::Fenced));
		if (!pipelined)
			LiveCounters::Add(static_cast<LiveCounters::Id>(LiveCounters::ScoutMissUnwalked + static_cast<uint32_t>(Scout::CurrentReason())));
		return nullptr;
	}
	if (!pipelined && ticket->packet != Scout::CurrentPacket()) return LiveCounters::Add(LiveCounters::ScoutLostIdentity), nullptr;
	if (ticket->version != pair.scout_version || ticket->stages != stages)
		return LiveCounters::Add(LiveCounters::ScoutLostPair), nullptr;
	for (uint32_t st = 0; st < stages; ++st)
		if (ticket->user_words[st] > user_data[st].size() ||
		    std::memcmp(ticket->user_data[st].data(), user_data[st].data(), ticket->user_words[st] * sizeof(uint32_t)) != 0)
			return LiveCounters::Add(LiveCounters::ScoutLostRegisters), nullptr;
	if (ticket->mapping_epoch != resources.MappingEpoch()) return LiveCounters::Add(LiveCounters::ScoutLostEpoch), nullptr;
	if (bool write_back = false; Scout::WrittenSince(*ticket, write_back)) {
		if (write_back) LiveCounters::Add(LiveCounters::ScoutLostWriteBack);
		return LiveCounters::Add(LiveCounters::ScoutLostWritten), nullptr;
	}
	const auto* pages    = Scout::Pages(*ticket);
	const auto* granules = resources.ReadGranules();
	uint64_t    clear    = UINT64_MAX;
	for (uint32_t i = 0; i < ticket->pages; ++i) {
		if ((pages[2 * i + 1] & Scout::Ticket::PageBacking) != 0) continue;
		const auto page    = Scout::PageAt(pages, i);
		const auto granule = page >> (PageManager::READ_GRANULE_BITS - 12u);
		if (granule == clear) continue;
		if ((page << 12u) < TRACKER_ADDRESS_SIZE &&
		    ((granules[granule / 64u].load(std::memory_order_acquire) >> (granule % 64u)) & 1u) == 0) {
			clear = granule;
			continue;
		}
		if (resources.HasReadWatchers(page << 12u, 4096)) return LiveCounters::Add(LiveCounters::ScoutLostWatched), nullptr;
	}
	const auto* ranges = pages + 2u * ticket->pages;
	for (uint32_t i = 0; i < ticket->backing; ++i) {
		const auto               address = ranges[3 * i] | uint64_t {ranges[3 * i + 1]} << 32u;
		const auto*              held    = ticket->data + (ranges[3 * i + 2] & 0xffffu);
		const uint32_t           count   = ranges[3 * i + 2] >> 16u;
		std::array<uint32_t, 64> fresh;
		for (uint32_t done = 0; done < count; done += static_cast<uint32_t>(fresh.size())) {
			const auto chunk = std::min<uint32_t>(count - done, static_cast<uint32_t>(fresh.size()));
			if (!Libs::LibKernel::Memory::TryReadGpuCleanBackingToHost(address + 4ull * done, fresh.data(), chunk * 4ull))
				return LiveCounters::Add(LiveCounters::ScoutLostDirty), nullptr;
			if (std::memcmp(fresh.data(), held + done, chunk * 4ull) != 0)
				return LiveCounters::Add(LiveCounters::ScoutLostChanged), nullptr;
		}
	}
	LiveCounters::Add(LiveCounters::ScoutUsed);
	if (ticket->backing != 0) LiveCounters::Add(LiveCounters::ScoutUsedBacking);
	return ticket;
}

static inline void TableCopyWords(uint32_t* to, const uint32_t* from, uint32_t count) {
	for (; count >= 8; count -= 8, to += 8, from += 8) __builtin_memcpy_inline(to, from, 32);
	if ((count & 4u) != 0) {
		__builtin_memcpy_inline(to, from, 16);
		to += 4;
		from += 4;
	}
	if ((count & 2u) != 0) {
		__builtin_memcpy_inline(to, from, 8);
		to += 2;
		from += 2;
	}
	if ((count & 1u) != 0) *to = *from;
}

static void TableScoutSlots(uint32_t* slots, const uint32_t* in, const uint32_t* segments, uint32_t count) {
	for (uint32_t i = 0; i < count; ++i) {
		const uint32_t words = segments[i] >> 16u;
		TableCopyWords(slots + (segments[i] & 0xffffu), in, words);
		in += words;
	}
}

template <typename Table, typename Pair>
static void TableScoutApply(Table& table, const Pair& pair, const Scout::Ticket& ticket, uint32_t stages) {
	const uint32_t* in       = ticket.data;
	const uint32_t* segments = ticket.data + ticket.slot_words[0] + ticket.slot_words[1];
	for (uint32_t st = 0; st < stages; ++st) {
		auto& slots = table.blocks[st];
		slots.resize(pair.programs[st]->table_plan.BlockDwords());
		TableScoutSlots(slots.data(), in, segments, ticket.segments[st]);
		in += ticket.slot_words[st];
		segments += ticket.segments[st];
		table.indirect[st].clear();
	}
	in = segments;
	table.words.assign(in, in + ticket.image_words);
	in += ticket.image_words;
	table.lod_stats.assign(in, in + ticket.lod_stats);
	in += ticket.lod_stats;
	for (uint32_t st = 0; st < stages; ++st) {
		auto& entries = table.entries[st];
		entries.resize(ticket.buffers[st]);
		for (auto& [words4, entry]: entries) {
			std::copy_n(in, 4, words4.begin());
			entry = &table.resolved[in[4]];
			__builtin_prefetch(entry);
			in += 5;
		}
	}
	table.gathered = &pair;
}

template <typename Table, typename Pair>
static void TableScoutVerify(const Table& table, const Pair& pair, const Scout::Ticket& ticket, uint32_t stages,
                             bool evaluated) {
	std::array<bool, 5> differ {};
	char                detail[160] = "";
	if (evaluated) {
		const uint32_t* in       = ticket.data;
		const uint32_t* segments = ticket.data + ticket.slot_words[0] + ticket.slot_words[1];
		thread_local std::vector<uint32_t> scratch;
		for (uint32_t st = 0; st < stages; ++st) {
			const auto& runs   = pair.packed[st].runs;
			const bool  before = differ[0];
			for (size_t r = 0; r < runs.size(); ++r) {
				const auto& run  = runs[r];
				const auto* mine = table.blocks[st].data() + run.first;
				uint32_t at = 0;
				for (uint32_t k = 0; k < ticket.segments[st]; ++k) {
					const uint32_t first = segments[k] & 0xffffu, count = segments[k] >> 16u;
					if (run.first >= first && run.first < first + count) {
						at += run.first - first;
						break;
					}
					at += count;
				}
				for (uint32_t w = 0; w < run.count && !differ[0]; ++w) {
					if (mine[w] == in[at + w]) continue;
					differ[0] = true;
					std::snprintf(detail, sizeof(detail), " (stage %u run %zu/%zu slot %u: ticket 0x%08x, render 0x%08x)", st, r,
					              runs.size(), run.first + w, in[at + w], mine[w]);
				}
			}
			scratch.assign(table.blocks[st].begin(), table.blocks[st].end());
			TableScoutSlots(scratch.data(), in, segments, ticket.segments[st]);
			if (differ[0] == before && scratch != table.blocks[st]) {
				differ[4] = true;
				std::snprintf(detail, sizeof(detail), " (stage %u: the apply's block differs)", st);
			}
			LiveCounters::Add(LiveCounters::ScoutApplyRuns, runs.size());
			LiveCounters::Add(LiveCounters::ScoutApplySegments, ticket.segments[st]);
			segments += ticket.segments[st];
			in += ticket.slot_words[st];
		}
		in = segments;
		differ[1] = table.words.size() != ticket.image_words || !std::equal(table.words.begin(), table.words.end(), in);
		in += ticket.image_words;
		differ[2] = table.lod_stats.size() != ticket.lod_stats || !std::equal(table.lod_stats.begin(), table.lod_stats.end(), in);
		in += ticket.lod_stats;
		for (uint32_t st = 0; st < stages; ++st) {
			if (table.entries[st].size() != ticket.buffers[st]) {
				differ[3] = true;
				break;
			}
			for (const auto& [words4, entry]: table.entries[st]) {
				differ[3] = differ[3] || !std::equal(words4.begin(), words4.end(), in) || entry != &table.resolved[in[4]];
				in += 5;
			}
		}
		if (differ == std::array<bool, 5> {}) return;
	}
	LiveCounters::Add(LiveCounters::ScoutMismatches);
	if (!evaluated) LiveCounters::Add(LiveCounters::ScoutMismatchRefused);
	if (differ[0]) LiveCounters::Add(LiveCounters::ScoutMismatchSlots);
	if (differ[1]) LiveCounters::Add(LiveCounters::ScoutMismatchWords);
	if (differ[2]) LiveCounters::Add(LiveCounters::ScoutMismatchLod);
	if (differ[3]) LiveCounters::Add(LiveCounters::ScoutMismatchEntries);
	if (differ[4]) LiveCounters::Add(LiveCounters::ScoutMismatchApply);
	static std::atomic<uint32_t> reports {0};
	if (reports.fetch_add(1, std::memory_order_relaxed) < 64)
		std::printf("Scout: mismatch VS 0x%llx PS 0x%llx submission %llu draw %u:%s%s%s%s%s%s%s\n",
		            static_cast<unsigned long long>(pair.vs), static_cast<unsigned long long>(pair.ps),
		            static_cast<unsigned long long>(ticket.sequence), ticket.ordinal, evaluated ? "" : " refused",
		            differ[0] ? " slots" : "", differ[1] ? " words" : "", differ[2] ? " lod" : "", differ[3] ? " entries" : "",
		            differ[4] ? " apply" : "", detail);
}

#endif

RenderExecutor::TableResult RenderExecutor::TableDraw(CommandBuffer& buffer, TablePair& pair, TableVariant& variant,
                                                      const NativeXprDrawState& draw_state,
                                                      std::span<const uint64_t> commands, uint64_t index_base,
                                                      uint64_t index_bytes, vk::IndexType index_type,
                                                      const NativeXprDirectDraw* direct) {
	auto&       table  = OwnTable();
	Spec::State* const spec_state =
	    ::kyty_local_speculate_mode.load(std::memory_order_relaxed) != 0 ? Spec::Current() : nullptr;
	auto&       cache  = *m_native_xpr;
	const auto& sh     = buffer.GetShaders();
	const auto  stages = pair.programs[1] != nullptr ? 2u : 1u;
	const std::array<std::span<const uint32_t>, 2> user_data {
	    std::span<const uint32_t>(sh.GetVs().gs_user_sgpr.value,
	                              std::min<uint32_t>(sh.GetVs().gs_regs.rsrc2.user_sgpr, HW::UserSgprInfo::SGPRS_MAX)),
	    std::span<const uint32_t>(sh.GetPs().ps_user_sgpr.value,
	                              std::min<uint32_t>(sh.GetPs().ps_regs.rsrc2.user_sgpr, HW::UserSgprInfo::SGPRS_MAX))};
	if (index_base != 0) {
		const auto* held = &table.indices[(index_base * 0x9e3779b97f4a7c15ull) >> 51u];
		__builtin_prefetch(held);
		__builtin_prefetch(reinterpret_cast<const char*>(held + 1) - 1);
	}
	std::optional<LiveCensus::Scope> phase;
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1e, 1);
#if KYTY_CX_SCOUT
	const auto* scout = Scout::Running() && spec_state == nullptr
	                        ? TableScoutTicket(pair, user_data, stages, m_context, static_cast<const Scout::Ticket*>(m_table_ticket))
	                        : nullptr;
	if (scout != nullptr && (Scout::Mode() == 1 || Scout::Mode() == 3)) {
		TableScoutApply(table, pair, *scout, stages);
	} else {
		const bool evaluated = TableEvaluate(pair, user_data);
		if (evaluated) TableGather(pair, user_data);
		if (scout != nullptr) TableScoutVerify(table, pair, *scout, stages, evaluated);
		if (!evaluated)
			return LiveCounters::Add(LiveCounters::TableDrawNative), Spec::Refuse("table: evaluation"), TableResult::Native;
	}
#else
	{
		const bool evaluated = TableEvaluate(pair, user_data);
		if (evaluated) TableGather(pair, user_data);
		if (!evaluated)
			return LiveCounters::Add(LiveCounters::TableDrawNative), Spec::Refuse("table: evaluation"), TableResult::Native;
	}
#endif
	phase.reset();
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1e, 2);
	auto* set = TableSet(buffer, pair, &variant, &draw_state);
	if (spec_state != nullptr && set != nullptr)
		spec_state->NoteSet(reinterpret_cast<uint64_t>(static_cast<VkDescriptorSet>(set->set)));
	if (set == nullptr) return LiveCounters::Add(LiveCounters::TableNoSet), Spec::Refuse("table: no set"), TableResult::Native;

	phase.reset();
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1e, 3);
	auto& buffer_cache = m_context.GetBufferCache();
	const bool     vertex_arguments = direct != nullptr && direct->indirect;
	const uint64_t stride = vertex_arguments ? sizeof(vk::DrawIndirectCommand) : sizeof(vk::DrawIndexedIndirectCommand);
	std::array<std::pair<uint64_t, uint64_t>, 4> region_epochs;
	region_epochs.fill({UINT64_MAX, 0});
	const auto cpu_epoch = [&](uint64_t address, uint64_t size) {
		const auto region = address / TRACKER_REGION_SIZE;
		if (region != (address + size - 1) / TRACKER_REGION_SIZE) return buffer_cache.CpuModificationEpoch(address, size);
		auto& known = region_epochs[region & 3u];
		if (known.first != region) known = {region, buffer_cache.CpuModificationEpoch(address, size)};
		return known.second;
	};
	const auto obtain = [&](TableXpr::HeldBuffer& held, uint64_t address, uint64_t size,
	                        auto&& prepare) -> std::pair<Buffer*, uint64_t> {
		const auto* spec = spec_state;
		if (held.buffer != nullptr && address >= held.begin && address + size <= held.end &&
		    held.registration == buffer_cache.RegistrationEpoch() && held.cpu_epoch != 0 &&
		    cpu_epoch(held.begin, held.end - held.begin) == held.cpu_epoch &&
		    (spec == nullptr || !spec->JournalIntersects(address, size))) {
			if (spec != nullptr) spec_state->NoteBufferRange(address, size);
			return {held.buffer, held.buffer->Offset(address)};
		}
		held = {};
		LiveCensus::Scope miss(LiveCensus::DrawPhase, 0x7ab1e, &held == &table.args ? 10 : 11);
		prepare();
		const auto registration_before = buffer_cache.RegistrationEpoch();
		const auto result = buffer_cache.ObtainBuffer(address, size, false);
		region_epochs.fill({UINT64_MAX, 0});
		const auto owner  = spec_state != nullptr ? buffer_cache.RegisteredBuffer(address, size)
		                                               : buffer_cache.FindBuffer(address, size);
		if (result.first != nullptr && buffer_cache.GetBufferIfLive(owner) == result.first) {
			const auto region = address & ~(TRACKER_REGION_SIZE - 1);
			auto       begin  = std::max(result.first->CpuAddress(), region);
			auto       end    = std::min(result.first->CpuAddress() + result.first->Size(), region + TRACKER_REGION_SIZE);
			if (address + size <= end) {
				const auto epoch = buffer_cache.CpuModificationEpoch(begin, end - begin);
				if (buffer_cache.IsRegionCpuModified(begin, end - begin)) {
					begin = address;
					end   = address + size;
				}
				if (epoch != 0 && !buffer_cache.IsRegionCpuDirty(begin, end - begin))
					held = {begin, end, epoch, spec_state != nullptr ? registration_before : buffer_cache.RegistrationEpoch(),
					        result.first};
			}
		}
		return result;
	};
	std::array<vk::DeviceAddress, 2> blocks {};
	bool                             writes = false;
	TablePrepareBda(pair, user_data);
	const auto registration = buffer_cache.RegistrationEpoch();
	if (!TableBlocks(pair, user_data, blocks, &writes))
		return LiveCounters::Add(LiveCounters::TableDrawNative), Spec::Refuse("table: blocks"), TableResult::Native;
	phase.reset();
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1e, 9);
	std::array<std::pair<Buffer*, uint64_t>, 64> arguments;
	std::array<uint32_t, 64>                     group_sizes;
	uint32_t                                     groups = 0;
	for (size_t first = 0; first < commands.size();) {
		size_t count = 1;
		while (first + count < commands.size() && commands[first + count] == commands[first] + count * stride) ++count;
		const auto [owner, offset] = obtain(table.args, commands[first], count * stride, [] {});
		if (owner == nullptr) return TableResult::Native;
		arguments[groups]   = {owner, offset};
		group_sizes[groups] = static_cast<uint32_t>(count);
		++groups;
		first += count;
	}
	auto&      emit       = cache.emit;
	auto&      dynamic    = m_context.GetCommandScheduler().GetDynamicState();
	const bool indexed    = direct == nullptr || direct->indexed;
	const bool same_index = !indexed || (emit.generation == dynamic.Generation() &&
	                                     emit.index_base == index_base && emit.index_bytes == index_bytes &&
	                                     emit.index_type == index_type);
	std::pair<Buffer*, uint64_t> indices {};
	if (!same_index) {
		indices = obtain(table.indices[(index_base * 0x9e3779b97f4a7c15ull) >> 51u], index_base, index_bytes, [&] {
			if (index_bytes <= BufferCache::CACHING_PAGESIZE && buffer_cache.IsRegionCpuModified(index_base, index_bytes) &&
			    !buffer_cache.IsRegionGpuModified(index_base, index_bytes))
				buffer_cache.EnsureBufferContents(index_base, index_bytes);
		});
		if (indices.first == nullptr) return TableResult::Native;
	}
	if (buffer_cache.RegistrationEpoch() != registration)
		return LiveCounters::Add(LiveCounters::TableRegistered), TableResult::Native;
	if (spec_state != nullptr && spec_state->refused != nullptr) return TableResult::Native;
	phase.reset();
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1e, 4);
	if (buffer.IsInvalid()) return Spec::Refuse("table: buffer"), TableResult::Native;
	auto       vk_buffer    = buffer.Handle();
	const bool same_command = emit.generation == dynamic.Generation();
	auto&      texture_cache = m_context.GetTextureCache();
	const auto images_epoch  = texture_cache.ImagesEpoch();
	const bool own           = spec_state == nullptr;
	const bool read_after_write =
	    set->read_after_write && kyty_local_draw_read_after_write.load(std::memory_order_relaxed) != 0;
	const bool in_place      = same_command && emit.state == &draw_state && table.last_set == set &&
	                      table.images_epoch == images_epoch && own && !read_after_write;
	const auto serials = own ? TableSerialsMode() : 0u;
	(void)serials;
	if (!in_place) {
		const bool epoch_targets = !own || draw_state.emitted_epoch != images_epoch;
		const auto targets_sum   = serials != 0 ? TableTargetSerials(draw_state) : 0;
		const bool sum_targets   = !own || draw_state.emitted_serials != targets_sum;
		const bool due_targets   = serials == 1 ? sum_targets : epoch_targets;
		if (serials != 0 && epoch_targets != sum_targets)
			LiveCounters::Add(sum_targets ? LiveCounters::SerialsExtra : LiveCounters::SerialsSkipped);
		if (due_targets && !NativeXprEmitTargets(draw_state, vk_buffer, set)) {
			if (serials == 2 && !sum_targets) TableSerialsUnsafe("targets refused");
			return LiveCounters::Add(LiveCounters::TableDrawNative), Spec::Refuse("table: targets"), TableResult::Native;
		}
		if (serials == 2 && due_targets && !sum_targets && TableTargetSerials(draw_state) != targets_sum)
			TableSerialsUnsafe("targets moved");
		const bool epoch_transit = !own || set->transit_epoch != images_epoch || set->transit_state != &draw_state;
		const auto set_sum       = serials != 0 ? TableSetSerials(*set) : 0;
		const bool sum_transit   = !own || set->transit_serials != set_sum ||
		                         ((set->transit_state != &draw_state || set->transit_capture != draw_state.capture_id) &&
		                          !TableSetApart(*set, draw_state));
		const bool due_transit   = (serials == 1 ? sum_transit : epoch_transit) || read_after_write;
		if (serials != 0 && epoch_transit != sum_transit)
			LiveCounters::Add(sum_transit ? LiveCounters::SerialsExtra : LiveCounters::SerialsSkipped);
		if (due_transit && !TableTransit(*set, vk_buffer, &draw_state)) {
			if (serials == 2 && !sum_transit && !read_after_write) TableSerialsUnsafe("transit refused");
			return LiveCounters::Add(LiveCounters::TableDrawNative), Spec::Refuse("table: transit"), TableResult::Native;
		}
		if (serials == 2 && due_transit && !sum_transit && !read_after_write && TableSetSerials(*set) != set_sum)
			TableSerialsUnsafe("transit moved");
		if (own) {
			draw_state.emitted_epoch = set->transit_epoch = texture_cache.ImagesEpoch();
			set->transit_state       = &draw_state;
			set->transit_capture     = draw_state.capture_id;
			if (serials != 0) {
				draw_state.emitted_serials = due_targets || due_transit ? TableTargetSerials(draw_state) : targets_sum;
				set->transit_serials       = due_targets || due_transit ? TableSetSerials(*set) : set_sum;
			}
		}
	}
	if (variant.colors != draw_state.rendering.num_color_attachments ||
	    variant.has_depth != draw_state.rendering.depth_stencil_attachment.has_depth ||
	    variant.has_stencil != draw_state.rendering.depth_stencil_attachment.has_stencil) {
		variant.pipeline = nullptr;
		static std::atomic<uint32_t> reports {0};
		if (reports.fetch_add(1, std::memory_order_relaxed) < 16)
			std::printf("Table: VS 0x%llx PS 0x%llx variant made for %u color attachments, its draw state has %u\n",
			            static_cast<unsigned long long>(pair.vs), static_cast<unsigned long long>(pair.ps), variant.colors,
			            draw_state.rendering.num_color_attachments);
		return Spec::Refuse("table: variant colors"), TableResult::Native;
	}
	buffer.BeginRendering(draw_state.rendering);
	const bool query_bias = m_context.GetOcclusionQueries().Counting();
	if (!same_command || emit.state != &draw_state || emit.query_bias != query_bias)
		NativeXprEmitDynamic(cache, vk_buffer, draw_state, same_command, query_bias);
	const auto& pipeline = *variant.pipeline;
	if (!same_command || emit.pipeline != &pipeline)
		vk_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline);
	if (!same_command || emit.pipeline != &pipeline || table.last_set != set)
		vk_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline.pipeline_layout, 0, 1, &set->set, 0,
		                             nullptr);
	ShaderRecompiler::IR::PushData push_data;
	uint32_t                       push_dwords = 0;
	for (uint32_t st = 0; st < stages; ++st) {
		const auto& program  = *pair.programs[st];
		const auto& bindings = program.bindings;
		if (!bindings.UsesPushData()) continue;
		push_dwords = std::max(push_dwords, bindings.push_data_start_dword + bindings.ShaderDataDwords());
		for (size_t i = 0; i < program.info.user_data_registers.size(); ++i) {
			const auto index = program.info.user_data_registers[i] - program.user_data_base;
			push_data.dwords[bindings.push_data_start_dword + i] = index < user_data[st].size() ? user_data[st][index] : 0;
		}
		if (program.stage == ShaderType::Pixel)
			for (size_t i = 0; i < table.lod_stats.size(); ++i)
				push_data.dwords[bindings.push_data_start_dword + bindings.MipStatsDword() + i] = table.lod_stats[i];
		push_data.dwords[bindings.push_data_start_dword + bindings.TableBlockDword()]     = static_cast<uint32_t>(blocks[st]);
		push_data.dwords[bindings.push_data_start_dword + bindings.TableBlockDword() + 1] = static_cast<uint32_t>(blocks[st] >> 32u);
	}
	if (push_dwords != 0)
		vk_buffer.pushConstants(pipeline.pipeline_layout, pipeline.push_stages, 0, push_dwords * sizeof(uint32_t),
		                        push_data.dwords.data());
	if (!same_index) vk_buffer.bindIndexBuffer(indices.first->Handle(), indices.second, index_type);
	if (direct != nullptr && !vertex_arguments) {
		const auto& vertex = *pair.programs[0];
		const auto  word   = [&](int32_t sgpr) {
			const auto index = int64_t {sgpr} - vertex.user_data_base;
			return index >= 0 && index < static_cast<int64_t>(user_data[0].size()) ? user_data[0][index] : 0u;
		};
		const uint32_t index_offset   = buffer.GetUserConfig().GetIndexOffset();
		const uint32_t vertex_offset  = variant.fetch_embedded && index_offset == 0 ? word(vertex.info.vertex_offset_sgpr)
		                                                                            : index_offset;
		const uint32_t first_instance = variant.fetch_embedded ? word(vertex.info.instance_offset_sgpr) : 0u;
		if (indexed)
			vk_buffer.drawIndexed(direct->index_count, direct->instance_count, 0,
			                      static_cast<int32_t>(vertex_offset) + direct->base_vertex, first_instance);
		else
			vk_buffer.draw(direct->index_count, direct->instance_count, vertex_offset, first_instance);
	}
	for (uint32_t group = 0; group < groups; ++group) {
		if (vertex_arguments)
			vk_buffer.drawIndirect(arguments[group].first->Handle(), arguments[group].second, group_sizes[group],
			                       static_cast<uint32_t>(stride));
		else
			vk_buffer.drawIndexedIndirect(arguments[group].first->Handle(), arguments[group].second, group_sizes[group],
			                              static_cast<uint32_t>(stride));
	}
	if (writes) {
		m_context.GetCommandScheduler().EndRendering();
		ShaderWriteBarrier(vk_buffer, vk::PipelineStageFlagBits::eVertexShader | vk::PipelineStageFlagBits::eFragmentShader);
	}
	emit.generation  = dynamic.Generation();
	emit.state       = &draw_state;
	emit.pipeline    = &pipeline;
	emit.query_bias  = query_bias;
	draw_state.Use(m_context.FrameNumber());
	if (indexed) {
		emit.index_base  = index_base;
		emit.index_bytes = index_bytes;
		emit.index_type  = index_type;
	}
	table.last_pair    = &pair;
	table.last_variant = static_cast<size_t>(&variant - pair.variants.data());
	table.last_state   = &draw_state;
	table.last_set     = set;
	table.images_epoch = texture_cache.ImagesEpoch();
	LiveCounters::Add(LiveCounters::TableDraws, std::max<size_t>(commands.size(), 1));
	return TableResult::Drawn;
}

void RenderExecutor::TableRuns(TablePair& pair) {
	using Kind  = ShaderRecompiler::IR::TablePlan::Operand::Kind;
	auto& table = *m_table_xpr;
	std::array<TableFlatStage, 2> flat_forms;
	for (uint32_t st = 0; st < 2; ++st) {
		if (pair.programs[st] == nullptr) continue;
		const auto&           plan = pair.programs[st]->table_plan;
		std::vector<uint32_t> levels(plan.slots.size(), 0);
		const auto            same = [](const ShaderRecompiler::IR::TablePlan::Operand& a,
		                     const ShaderRecompiler::IR::TablePlan::Operand& b) {
			return a.kind == b.kind && a.value == b.value;
		};
		pair.runs[st].clear();
		for (uint32_t first = 0; first < plan.slots.size();) {
			if (plan.cpu[first] == 0) {
				++first;
				continue;
			}
			const auto& slot  = plan.slots[first];
			uint32_t    count = 1;
			while (first + count < plan.slots.size() && plan.cpu[first + count] != 0 &&
			       same(plan.slots[first + count].low, slot.low) && same(plan.slots[first + count].high, slot.high) &&
			       same(plan.slots[first + count].offset, slot.offset) &&
			       plan.slots[first + count].bounded == slot.bounded && same(plan.slots[first + count].records, slot.records) &&
			       plan.slots[first + count].immediate == slot.immediate + static_cast<int32_t>(4 * count))
				++count;
			uint32_t level = 0;
			for (const auto* operand: {&slot.low, &slot.high, &slot.offset, &slot.records})
				if (operand->kind == Kind::Slot) level = std::max(level, levels[operand->value] + 1);
			std::fill_n(levels.begin() + first, count, level);
			pair.runs[st].push_back({slot, first, count, level});
			first += count;
		}
		std::ranges::stable_sort(pair.runs[st], {}, &TablePair::Run::level);
		table.addresses.resize(std::max(table.addresses.size(), pair.runs[st].size()));
		const auto& program = *pair.programs[st];
		auto&       flat    = flat_forms[st];
		flat.constants.push_back(0);
		const auto flat_of = [&](const ShaderRecompiler::IR::TablePlan::Operand& operand) -> TableFlatStage::Operand {
			if (operand.kind == Kind::Slot) return {operand.value, 1};
			if (TableCodeOperand(operand)) {
				flat.constants.push_back(TableCodeWord(operand, st == 0 ? pair.vs : pair.ps));
				return {static_cast<uint32_t>(flat.constants.size() - 1), 2};
			}
			if (operand.kind == Kind::UserData) {
				const auto index = operand.value - program.user_data_base;
				if (index >= 64) return {0, 2};
				flat.user_words = std::max(flat.user_words, index + 1);
				return {index, 0};
			}
			flat.constants.push_back(operand.value);
			return {static_cast<uint32_t>(flat.constants.size() - 1), 2};
		};
		const auto& runs = pair.runs[st];
		for (size_t i = 0; i < runs.size(); ++i) {
			const auto& run = runs[i];
			flat.runs.push_back({{flat_of(run.read.low), flat_of(run.read.high), flat_of(run.read.offset)},
			                     run.read.immediate, run.first, run.count, flat_of(run.read.records), run.read.bounded});
			if (i + 1 == runs.size() || runs[i + 1].level != run.level) flat.level_ends.push_back(static_cast<uint32_t>(i + 1));
		}
		for (const auto& descriptor: plan.images) {
			flat.image_counts.push_back(descriptor.count);
			for (uint32_t w = 0; w < descriptor.count; ++w) flat.image_words.push_back(flat_of(descriptor.words[w]));
		}
		for (const auto& descriptor: plan.samplers)
			for (uint32_t w = 0; w < descriptor.count; ++w) flat.image_words.push_back(flat_of(descriptor.words[w]));
		for (const auto& words: plan.buffers)
			for (const auto& word: words) flat.buffer_words.push_back(flat_of(word));
	}
	TablePack(pair, flat_forms);
}

void RenderExecutor::TablePack(TablePair& pair, const std::array<TableFlatStage, 2>& flat_forms) {
	using Pair = TablePair;
	pair.packed       = {};
	pair.packed_block = nullptr;
	pair.packed_at    = nullptr;
	pair.packed_bytes = 0;
	const auto operand = [](const TableFlatStage::Operand& from, bool& fits) -> uint16_t {
		if (from.base > 2 || from.index >= 0x4000u) fits = false;
		return static_cast<uint16_t>(from.base << 14u | (from.index & 0x3fffu));
	};
	const auto align = [](size_t at, size_t to) { return (at + to - 1) / to * to; };
	std::array<std::array<size_t, 6>, 2> offsets {};
	size_t                               bytes = 0;
	std::array<bool, 2>                  fits {};
	for (uint32_t st = 0; st < 2; ++st) {
		if (pair.programs[st] == nullptr) continue;
		const auto& flat = flat_forms[st];
		fits[st]         = flat.runs.size() < 0x10000 && flat.image_words.size() < 0x10000;
		for (const auto end: flat.level_ends) fits[st] = fits[st] && end < 0x10000;
		for (const auto count: flat.image_counts) fits[st] = fits[st] && count < 0x100;
		for (const auto& run: flat.runs) fits[st] = fits[st] && run.first < 0x10000 && run.count < 0x10000;
		if (!fits[st]) continue;
		bytes          = align(bytes, 64);
		offsets[st][0] = bytes;
		bytes += flat.runs.size() * sizeof(Pair::PackedRun);
		offsets[st][1] = bytes;
		bytes += flat.constants.size() * sizeof(uint32_t);
		offsets[st][2] = bytes;
		bytes += flat.level_ends.size() * sizeof(uint16_t);
		offsets[st][3] = bytes;
		bytes += flat.image_words.size() * sizeof(uint16_t);
		offsets[st][4] = bytes;
		bytes += flat.buffer_words.size() * sizeof(uint16_t);
		offsets[st][5] = bytes;
		bytes += flat.image_counts.size();
	}
	if (bytes == 0) return;
	pair.packed_block = std::make_unique<std::byte[]>(bytes + 64);
	auto* base        = reinterpret_cast<std::byte*>(align(reinterpret_cast<uintptr_t>(pair.packed_block.get()), 64));
	pair.packed_at    = base;
	pair.packed_bytes = static_cast<uint32_t>(bytes);
	for (uint32_t st = 0; st < 2; ++st) {
		if (!fits[st]) continue;
		const auto& flat    = flat_forms[st];
		auto&       packed  = pair.packed[st];
		bool        operand_fits = true;
		auto*       runs    = reinterpret_cast<Pair::PackedRun*>(base + offsets[st][0]);
		for (size_t i = 0; i < flat.runs.size(); ++i) {
			const auto& run = flat.runs[i];
			runs[i]         = {.address   = {operand(run.address[0], operand_fits), operand(run.address[1], operand_fits),
			                                 operand(run.address[2], operand_fits)},
			                   .records   = run.bounded ? operand(run.records, operand_fits) : Pair::PackedRun::NotBounded,
			                   .immediate = run.immediate,
			                   .first     = static_cast<uint16_t>(run.first),
			                   .count     = static_cast<uint16_t>(run.count)};
			if (run.bounded && runs[i].records == Pair::PackedRun::NotBounded) operand_fits = false;
		}
		auto* constants = reinterpret_cast<uint32_t*>(base + offsets[st][1]);
		std::copy(flat.constants.begin(), flat.constants.end(), constants);
		auto* level_ends = reinterpret_cast<uint16_t*>(base + offsets[st][2]);
		for (size_t i = 0; i < flat.level_ends.size(); ++i) level_ends[i] = static_cast<uint16_t>(flat.level_ends[i]);
		auto* image_words = reinterpret_cast<uint16_t*>(base + offsets[st][3]);
		for (size_t i = 0; i < flat.image_words.size(); ++i) image_words[i] = operand(flat.image_words[i], operand_fits);
		auto* buffer_words = reinterpret_cast<uint16_t*>(base + offsets[st][4]);
		for (size_t i = 0; i < flat.buffer_words.size(); ++i) buffer_words[i] = operand(flat.buffer_words[i], operand_fits);
		auto* image_counts = reinterpret_cast<uint8_t*>(base + offsets[st][5]);
		for (size_t i = 0; i < flat.image_counts.size(); ++i) image_counts[i] = static_cast<uint8_t>(flat.image_counts[i]);
		packed = {.runs         = {runs, flat.runs.size()},
		          .level_ends   = {level_ends, flat.level_ends.size()},
		          .image_words  = {image_words, flat.image_words.size()},
		          .buffer_words = {buffer_words, flat.buffer_words.size()},
		          .image_counts = {image_counts, flat.image_counts.size()},
		          .constants    = {constants, flat.constants.size()},
		          .user_words   = flat.user_words,
		          .valid        = operand_fits};
	}
}

void RenderExecutor::TableStore(CommandBuffer& buffer, const DrawRenderState& state, vk::PrimitiveTopology topology,
                                bool primitive_restart_enable, const RenderState& rendering,
                                const DynamicState& dynamic_state, bool storable) {
	auto& table = OwnTable();
	auto* pair  = std::exchange(table.store, nullptr);
	if (pair == nullptr) return;
	Spec::Refuse("store: called");
	const auto& sh = buffer.GetShaders();
	if (sh.GetVs().es_regs.data_addr != pair->vs || sh.GetPs().ps_regs.data_addr != pair->ps)
		return Spec::Refuse("store: other pair"), void();
	const auto refuse = [&](const char* reason) {
		static std::atomic<uint32_t> reports {0};
		if (reports.fetch_add(1, std::memory_order_relaxed) < 64)
			std::printf("Table: VS 0x%llx PS 0x%llx refused: %s\n", static_cast<unsigned long long>(pair->vs),
			            static_cast<unsigned long long>(pair->ps), reason);
		pair->refused = true;
		Spec::Refuse("store: pair refused");
	};
	if (!storable) return refuse("a draw table programs cannot make (vertex buffers, draw data, host indices)");
	if (state.VertexStages().size() != 1 || state.vertex_info[0].stage.module == nullptr ||
	    state.vertex_info[0].logical_stage == ShaderType::Mesh || state.vertex_info[0].user_clip_planes != 0 ||
	    (state.ps_active && state.ps_input_info.stage.module == nullptr))
		return refuse("stages");
	auto&       pipelines = m_context.GetPipelineCache();
	const auto& ctx       = buffer.GetRegisters();
	std::vector<uint32_t> key;
	BuildStageStaticKey(state.vertex_info[0], key);
	if (state.ps_active) {
		std::vector<uint32_t> pixel_key;
		BuildStageStaticKey(state.ps_input_info, pixel_key);
		key.insert(key.end(), pixel_key.begin(), pixel_key.end());
	}
	const auto static_key = XXH3_64bits_withSeed(
	    key.data(), key.size() * sizeof(uint32_t),
	    state.vertex_info[0].stage.module->info.shader_hash * 31u +
	        (state.ps_active ? state.ps_input_info.stage.module->info.shader_hash : 0u));
	if (pair->programs[0] == nullptr) {
		ShaderVertexInputInfo vertex_scratch {};
		ShaderPixelInputInfo  pixel_scratch {};
		const auto vertex_params = PrepareProgram(sh.GetVs(), ctx, buffer.GetUserConfig(), vertex_scratch);
		ShaderParams pixel_params;
		if (state.ps_active)
			pixel_params = PrepareProgram(sh.GetPs(), ctx.GetShaderRegisters(),
			                              TableDrawExportMapping(m_context.GetGraphics(), ctx, state.mrt_mask), pixel_scratch);
		const auto [vertex_plan, vertex_specialization] = pipelines.ProgramPlanOf(*state.vertex_info[0].stage.module);
		const auto [pixel_plan, pixel_specialization] =
		    state.ps_active ? pipelines.ProgramPlanOf(*state.ps_input_info.stage.module)
		                    : std::pair<const ShaderRecompiler::IR::ResourcePlan*,
		                                const ShaderRecompiler::IR::ResourceSpecialization*> {nullptr, nullptr};
		if (vertex_plan == nullptr || vertex_specialization == nullptr ||
		    (state.ps_active && (pixel_plan == nullptr || pixel_specialization == nullptr))) {
			char why[160];
			std::snprintf(why, sizeof(why), "no resource plan (VS plan %p id %llx, PS plan %p id %llx)",
			              static_cast<const void*>(vertex_plan),
			              static_cast<unsigned long long>(state.vertex_info[0].stage.module->id),
			              static_cast<const void*>(pixel_plan),
			              static_cast<unsigned long long>(state.ps_active ? state.ps_input_info.stage.module->id : 0));
			return refuse(why);
		}
		const auto portable = [](const ShaderRecompiler::IR::ResourceSpecialization& specialization, uint32_t mask) {
			auto result = specialization;
			for (size_t i = 0; i < result.buffers.size() && i < 32; ++i) {
				if (((mask >> i) & 1u) == 0) continue;
				result.buffers[i].descriptor_format  = Prospero::BufferFormat::kInvalid;
				result.buffers[i].descriptor_swizzle = DstSel(4, 5, 6, 7);
			}
			return result;
		};
		const auto vertex_portable = portable(*vertex_specialization, pair->portable[0]);
		const auto pixel_portable  = state.ps_active ? portable(*pixel_specialization, pair->portable[1])
		                                             : ShaderRecompiler::IR::ResourceSpecialization {};
		uint32_t cursor = ShaderRecompiler::IR::PushData::GraphicsDrawDwords(false, false, 0);
		PipelineCache::TableProgram pixel {};
		if (state.ps_active)
			pixel = pipelines.GetPixelProgramTable(pixel_params, state.ps_input_info, pixel_portable, cursor);
		const auto vertex = pipelines.GetVertexProgramTable(vertex_params, state.vertex_info[0], vertex_portable, cursor);
		using Status = PipelineCache::TableStatus;
		if (vertex.status == Status::Refused || (state.ps_active && pixel.status == Status::Refused)) {
			const auto* why = vertex.status == Status::Refused ? vertex.refusal : pixel.refusal;
			return refuse(why != nullptr ? why->c_str() : "table mode cannot serve it");
		}
		if (vertex.status == Status::Pending || (state.ps_active && pixel.status == Status::Pending)) {
			pair->pending = true;
			pair->waiting = vertex.status == Status::Pending ? vertex.waiting : pixel.waiting;
			Spec::Refuse("store: programs pending");
			return;
		}
		const auto serves = [](const ShaderModule* module) {
			return module != nullptr && module->info.table_mode && module->info.bindings.UsesPushData() &&
			       !module->info.has_address_writes && !module->info.info.uses_gds;
		};
		if (!serves(vertex.module) || (state.ps_active && !serves(pixel.module))) return refuse("a program table mode cannot draw");
		pair->pending     = false;
		Spec::Refuse("store: programs set");
		pair->tables      = {vertex.module, state.ps_active ? pixel.module : nullptr};
		pair->programs    = {&vertex.module->info, state.ps_active ? &pixel.module->info : nullptr};
		pair->plans       = {vertex_plan, pixel_plan};
		pair->static_key  = static_key;
		pair->has_pixel   = state.ps_active;
		pair->code_hashes = {ShaderHashAt(pair->vs), state.ps_active ? ShaderHashAt(pair->ps) : 0};
		TableRuns(*pair);
#if KYTY_CX_SCOUT
		TableScoutPublish(*pair);
#endif
	}
	auto variant = std::ranges::find(pair->variants, table.store_state, &TableVariant::state);
	if (variant == pair->variants.end()) variant = pair->variants.insert(variant, {.state = table.store_state});
	if (static_key != pair->static_key || state.ps_active != pair->has_pixel) {
		variant->refused = true;
		Spec::Refuse("store: variant refused (static key)");
		return;
	}
	DrawRenderState table_state = state;
	table_state.vertex_info[0].stage = {.module = pair->tables[0], .resources = &g_table_no_resources};
	if (state.ps_active) table_state.ps_input_info.stage = {.module = pair->tables[1], .resources = &g_table_no_resources};
	const auto* pipeline = pipelines.GetGraphicsPipeline(buffer, table_state, *m_draw_targets, NativeXprAsyncPipelines(), true);
	if (pipeline != nullptr && (!pipeline->native_bindings || pipeline->uses_push_descriptors)) {
		variant->refused = true;
		Spec::Refuse("store: variant refused (bindings)");
		return;
	}
	Spec::Refuse(pipeline != nullptr ? "store: pipeline ready" : "store: pipeline null (async)");
	if (pipeline == nullptr) pair->waiting = pipelines.PipelineWaiting();
	variant->pipeline       = pipeline;
	variant->colors         = rendering.num_color_attachments;
	variant->has_depth      = rendering.depth_stencil_attachment.has_depth;
	variant->has_stencil    = rendering.depth_stencil_attachment.has_stencil;
	variant->fetch_embedded = state.vertex_info[0].has_dynamic_vertex_fetch;
	NativeXprCaptureState(state, rendering, table.store_state, dynamic_state);
}

namespace {
void TableLoadSwitches() {
	static const bool loaded = [] {
		const auto load = [](const char* name, volatile std::atomic_uint32_t& value) {
			if (const char* text = std::getenv(name); text != nullptr && text[0] != 0)
				value.store(static_cast<uint32_t>(std::strtoul(text, nullptr, 0)), std::memory_order_relaxed);
		};
		load("KYTY_TABLE_XPR", kyty_local_table_xpr_mode);
		if (::Loader::DemonsSoulsTitle::IsSupportedGame()) kyty_local_table_dispatch_mode.store(1, std::memory_order_relaxed);
		load("KYTY_TABLE_DISPATCH", kyty_local_table_dispatch_mode);
		load("KYTY_TABLE_STORE_BUDGET", kyty_local_table_store_budget);
		load("KYTY_TABLE_SET_PROOFS", kyty_local_table_set_proofs);
		load("KYTY_TABLE_MRU", kyty_local_table_mru);
		load("KYTY_TABLE_TARGET_READ", kyty_local_table_target_read);
		load("KYTY_TABLE_GATHER_MEMO", kyty_local_table_gather_memo);
		load("KYTY_TABLE_MEMO", kyty_local_table_memo);
		load("KYTY_TABLE_MEMO_VERIFY", kyty_local_table_memo_verify);
		load("KYTY_TABLE_SERIALS", kyty_local_table_serials);
		load("KYTY_DRAW_READ_AFTER_WRITE", kyty_local_draw_read_after_write);
		return true;
	}();
	(void)loaded;
}
uint32_t TableXprMode() {
	TableLoadSwitches();
	return kyty_local_table_xpr_mode.load(std::memory_order_relaxed);
}
uint32_t TableDispatchMode() {
	TableLoadSwitches();
	return kyty_local_table_dispatch_mode.load(std::memory_order_relaxed);
}
}

namespace {
uint64_t TableLikelyTag(const HW::Shader& shaders) noexcept {
	return ((shaders.GetVs().es_regs.data_addr * 0x9e3779b97f4a7c15ull) ^ (shaders.GetPs().ps_regs.data_addr * 0xc2b2ae3d27d4eb4full)) | 1ull;
}
}

uint32_t RenderExecutor::TableHint(const HW::Shader& shaders) const noexcept {
	if (TableXprMode() == 0) return 0;
	const auto tag  = TableLikelyTag(shaders);
	const auto slot = (tag >> 40u) & (m_table_drawn.size() - 1);
	if (m_table_drawn[slot].load(std::memory_order_relaxed) != tag) return 0;
	return m_table_mixed_until[slot].load(std::memory_order_relaxed) < m_context.FrameNumber() ? 1u : 3u;
}

uint32_t RenderExecutor::TableDispatchHint(const HW::ComputeShaderInfo& cs) const noexcept {
	if (TableXprMode() == 0 || TableDispatchMode() == 0) return 0;
	const auto tag  = (cs.cs_regs.data_addr * 0x9e3779b97f4a7c15ull) | 1ull;
	const auto slot = (tag >> 40u) & (m_table_dispatched.size() - 1);
	if (m_table_dispatched[slot].load(std::memory_order_relaxed) != tag) return 0;
	return m_table_dispatch_mixed_until[slot].load(std::memory_order_relaxed) < m_context.FrameNumber() ? 1u : 3u;
}
void RenderExecutor::TableNoteDispatched(const HW::ComputeShaderInfo& cs, bool tabled) noexcept {
	static constexpr uint64_t TableMixedDispatchFrames = 120;
	const auto tag  = (cs.cs_regs.data_addr * 0x9e3779b97f4a7c15ull) | 1ull;
	const auto slot = (tag >> 40u) & (m_table_dispatched.size() - 1);
	const bool was  = m_table_dispatched[slot].load(std::memory_order_relaxed) == tag;
	if (tabled && !was) m_table_dispatched[slot].store(tag, std::memory_order_relaxed);
	if (!tabled && was)
		m_table_dispatch_mixed_until[slot].store(m_context.FrameNumber() + TableMixedDispatchFrames, std::memory_order_relaxed);
}

bool RenderExecutor::TableLikely(const HW::Shader& shaders) const noexcept {
	if (TableXprMode() == 0) return false;
	const auto tag  = TableLikelyTag(shaders);
	const auto slot = (tag >> 40u) & (m_table_drawn.size() - 1);
	return m_table_drawn[slot].load(std::memory_order_relaxed) == tag &&
	       m_table_mixed_until[slot].load(std::memory_order_relaxed) < m_context.FrameNumber();
}

bool RenderExecutor::TableXprTry(CommandBuffer& buffer, uint64_t args_addr, uint32_t count, uint32_t stride,
                                 uint64_t index_base, uint64_t index_bytes, vk::IndexType index_type,
                                 const DrawIndexArgs* direct, bool clean, const ShaderIndirectOffsets& offsets,
                                 const uint32_t* run_offsets) {
	bool drawn = false;
	static constexpr uint64_t TableMixedFrames = 120;
	struct DrawnHint {
		std::atomic<uint64_t>& slot;
		std::atomic<uint64_t>& mixed;
		const uint64_t         tag;
		const uint64_t         frame;
		const bool&            drawn;
		~DrawnHint() {
			const bool was = slot.load(std::memory_order_relaxed) == tag;
			if (was && !drawn) mixed.store(frame + TableMixedFrames, std::memory_order_relaxed);
			if (!was && drawn) slot.store(tag, std::memory_order_relaxed);
		}
	};
	const auto      likely_tag  = TableLikelyTag(buffer.GetShaders());
	const auto      likely_slot = (likely_tag >> 40u) & (m_table_drawn.size() - 1);
	const DrawnHint hint {m_table_drawn[likely_slot], m_table_mixed_until[likely_slot], likely_tag, m_context.FrameNumber(), drawn};
	struct DrewDraws {
		std::atomic<bool>& flag;
		const bool&        drawn;
		~DrewDraws() {
			if (drawn && !flag.load(std::memory_order_relaxed)) flag.store(true, std::memory_order_relaxed);
		}
	} const drew_draws {m_table_drew_draws, drawn};
	if (m_table_store) Spec::Refuse("store: lost (the draw never reached TableStore)");
	m_table_store = false;
	g_table_offsets_key = direct != nullptr || (offsets.vertex == ShaderIndirectOffsets::None &&
	                                            offsets.instance == ShaderIndirectOffsets::None)
	                          ? -1
	                          : static_cast<int32_t>(uint32_t {offsets.instance} << 8u | offsets.vertex);
	const auto mode = TableXprMode();
	if (mode == 0 || buffer.IsInvalid()) return false;
	const auto& try_vs = buffer.GetShaders().GetVs();
	if (try_vs.es_regs.data_addr == 0 || try_vs.ls_regs.data_addr != 0 || try_vs.hs_regs.data_addr != 0) return false;
	const auto refused_tag =
	    (likely_tag ^ (uint64_t {static_cast<uint32_t>(g_table_offsets_key)} * 0xff51afd7ed558ccdull)) | 1ull;
	const auto refused_slot       = (refused_tag >> 40u) & (m_table_refused.size() - 1);
	const auto refused_generation = ShaderMapGeneration();
	if (m_table_refused[refused_slot] == refused_tag && m_table_refused_generation[refused_slot] == refused_generation)
		return false;
	NativeXprDirectDraw direct_draw;
	if (direct != nullptr) {
		if (direct->indirect.address != 0 || direct->offset_source != DrawOffsetSource::DrawState ||
		    direct->index_addr == nullptr || direct->index_count == 0 || direct->instance_count == 0 ||
		    direct->first_instance != 0 || direct->index_type_and_size > 1)
			return false;
		index_base  = reinterpret_cast<uint64_t>(direct->index_addr);
		index_bytes = uint64_t {direct->index_count} * (direct->index_type_and_size == 0 ? 2u : 4u);
		index_type  = direct->index_type_and_size == 0 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
		direct_draw = {.index_address  = index_base,
		               .index_count    = direct->index_count,
		               .instance_count = direct->instance_count,
		               .base_vertex    = direct->base_vertex,
		               .indexed        = true,
		               .indirect       = false};
	}
	if (index_base == 0 || index_bytes == 0) return false;
	m_context.GetCommandScheduler().PopPendingOperations();
	Common::LockGuard lock(m_context.GetMutex());
	(void)NativeXprState();
	NativeXprCollect();
	std::array<uint64_t, 64> run;
	if (direct == nullptr) {
		count = std::min<uint32_t>(std::max<uint32_t>(count, 1u), 64u);
		for (uint32_t i = 0; i < count; ++i)
			run[i] = run_offsets != nullptr ? args_addr - run_offsets[0] + run_offsets[5 * i] : args_addr + uint64_t {i} * stride;
	}
	const std::span<const uint64_t> commands = direct == nullptr ? std::span<const uint64_t> {run.data(), count}
	                                                              : std::span<const uint64_t> {};
	const auto* direct_ptr = direct == nullptr ? nullptr : &direct_draw;
	TableResult result     = TableResult::Native;
	Spec::Refuse("xpr: draw runs tried");
	if (clean) Spec::Refuse("xpr: clean draws");
	if (mode >= 2 && clean && TableContinue(buffer, commands, index_base, index_bytes, index_type, direct_ptr)) {
		result = TableResult::Drawn;
	} else {
		const auto& ucfg = buffer.GetUserConfig();
		const auto& ctx  = buffer.GetRegisters();
		if (ucfg.GetPrimType() == Prospero::PrimitiveType::kTriList && !ctx.HasObjectPrimitiveIdOutput() &&
		    ctx.GetColorControl().mode <= 1u && (ucfg.GetPrimitiveResetControl() & 0x1u) == 0) {
			m_native_xpr->state_key = NativeXprStateKeyOf(buffer, &m_state_key_cache, m_context_id, m_user_config_id);
			m_table_try_refused     = false;
			result = TableTry(buffer, commands, index_base, index_bytes, index_type, direct_ptr);
			if (result == TableResult::Native && std::exchange(m_table_try_refused, false)) {
				m_table_refused[refused_slot]            = refused_tag;
				m_table_refused_generation[refused_slot] = refused_generation;
			}
		} else {
			Spec::Refuse("xpr: not a plain triangle list");
		}
	}
	switch (result) {
		case TableResult::Drawn: drawn = true; return true;
		case TableResult::Store: m_table_store = true; break;
		case TableResult::Deferred: break;
		case TableResult::Native: break;
	}
	return false;
}

#if KYTY_CX_DISPATCH

static std::array<uint64_t, 4> TableDispatchWords(const HW::ComputeShaderInfo& cs, bool thread_dimensions) {
	const auto& r = cs.cs_regs;
	return {r.data_addr, uint64_t {r.num_thread_x} | uint64_t {r.num_thread_y} << 32u,
	        uint64_t {r.num_thread_z} | uint64_t {r.lds_size} << 32u,
	        uint64_t {r.wave_size} | uint64_t {r.user_sgpr} << 8u | uint64_t {r.tidig_comp_cnt} << 16u |
	            uint64_t {r.tgid_x_en} << 24u | uint64_t {r.tgid_y_en} << 25u | uint64_t {r.tgid_z_en} << 26u |
	            uint64_t {r.tg_size_en} << 27u | uint64_t {r.scratch_en} << 28u | uint64_t {thread_dimensions} << 29u};
}
static uint64_t TableDispatchKey(const std::array<uint64_t, 4>& words) {
	return XXH3_64bits(words.data(), sizeof(words));
}
static uint64_t TableDispatchKey(const HW::ComputeShaderInfo& cs, bool thread_dimensions) {
	return TableDispatchKey(TableDispatchWords(cs, thread_dimensions));
}

static bool TableDispatchServes(const ShaderRecompiler::IR::CompiledShaderInfo& program, bool thread_dimensions) {
	using Kind = ShaderRecompiler::IR::DescriptorBindingKind;
	return program.table_mode && program.bindings.UsesPushData() && !program.info.uses_gds &&
	       program.bindings.descriptor_counts[static_cast<size_t>(Kind::SharedMemory)] == 0 &&
	       (thread_dimensions || program.bindings.dispatch_thread_dword == ShaderRecompiler::IR::PushData::NoStart);
}

void RenderExecutor::TableDispatchSpecial(const HW::ComputeShaderInfo& cs, bool thread_dimensions, const char* why) {
	if (TableDispatchMode() == 0 || TableXprMode() == 0) return;
	auto&      table = OwnTable();
	const auto key   = TableDispatchKey(cs, thread_dimensions);
	auto       it    = table.dispatches.find(key);
	if (it != table.dispatches.end() && it->second.refused) return;
	if (it == table.dispatches.end()) {
		it            = table.dispatches.try_emplace(key).first;
		it->second.vs = cs.cs_regs.data_addr;
	}
	auto& pair = it->second;
	if (pair.programs[0] != nullptr) {
		const auto write = CatalogWrite();
		TableReset(pair);
	}
	pair.refused = true;
	pair.special = why;
}

void RenderExecutor::TableDispatchLearn(const HW::ComputeShaderInfo& cs, const ShaderModule& module, bool async_compute,
                                        bool thread_dimensions) {
	if (TableDispatchMode() == 0 || TableXprMode() == 0) return;
	auto&      table = OwnTable();
	const auto key   = TableDispatchKey(cs, thread_dimensions);
	auto       it    = table.dispatches.find(key);
	if (it == table.dispatches.end()) {
		it     = table.dispatches.try_emplace(key).first;
		it->second.vs = cs.cs_regs.data_addr;
	}
	auto& pair = it->second;
	++pair.sightings;
	pair.used_frame = m_context.FrameNumber();
	if (pair.refused || pair.programs[0] != nullptr || pair.vs != cs.cs_regs.data_addr)
		return void();
	if (pair.waiting) {
		if (!pair.waiting->load(std::memory_order_acquire)) return;
		pair.waiting.reset();
	}
	const auto& program = module.info;
	auto&       pipelines = m_context.GetPipelineCache();
	const auto [plan, specialization] = pipelines.ProgramPlanOf(module);
	const auto refuse = [&](const char* why) {
		static std::atomic<uint32_t> reports {0};
		if (reports.fetch_add(1, std::memory_order_relaxed) < 64)
			std::printf("Table: CS 0x%llx refused: %s\n", static_cast<unsigned long long>(pair.vs), why);
		pair.refused = true;
		pair.special = why;
	};
	if (plan == nullptr || specialization == nullptr) return refuse("no resource plan");
	if (program.info.uses_gds ||
	    (!thread_dimensions && program.bindings.dispatch_thread_dword != ShaderRecompiler::IR::PushData::NoStart) ||
	    program.bindings.ray_query_dword != ShaderRecompiler::IR::PushData::NoStart ||
	    program.info.uses_image_heap || module.guest_traversal || program.info.bvh ||
	    !program.info.self_sized_buffers.empty() ||
	    plan->uniform_fill.fill.kind != ShaderRecompiler::IR::UniformFillKind::None ||
	    std::ranges::any_of(program.info.buffers, [](const auto& buffer) { return buffer.indirect_write_table; }))
		return refuse("a program table mode does not serve");
	if (RayTracing::TlasSource source; RayTracing::FindTlasSource(program.info, source))
		return refuse("native ray tracing");
	auto compute = std::make_unique<ShaderComputeInputInfo>();
	compute->async_compute              = async_compute;
	compute->dispatch_thread_dimensions = thread_dimensions;
	auto portable = *specialization;
	for (size_t i = 0; i < portable.buffers.size() && i < 32; ++i) {
		if (((pair.portable[0] >> i) & 1u) == 0) continue;
		portable.buffers[i].descriptor_format  = Prospero::BufferFormat::kInvalid;
		portable.buffers[i].descriptor_swizzle = DstSel(4, 5, 6, 7);
	}
	const auto result = pipelines.GetComputeProgramTable(cs, *compute, portable);
	if (result.status == PipelineCache::TableStatus::Pending) return void();
	if (result.status == PipelineCache::TableStatus::Refused || result.module == nullptr ||
	    !TableDispatchServes(result.module->info, thread_dimensions))
		return refuse(result.refusal != nullptr ? result.refusal->c_str() : "table mode cannot serve it");
	compute->stage = {.module = result.module, .resources = &g_table_no_resources};
	auto* const pipeline = pipelines.TryGetComputePipeline(*compute);
	if (pipeline == nullptr) return void();
	if (!pipeline->native_bindings || pipeline->uses_push_descriptors) return refuse("pipeline");
	pair.compute     = std::move(compute);
	pair.tables[0]   = result.module;
	pair.programs[0] = &result.module->info;
	pair.plans[0]    = plan;
	pair.code_hashes[0] = ShaderHashAt(pair.vs);
	pair.pipeline    = pipeline;
	TableRuns(pair);
#if KYTY_CX_SCOUT
	TableScoutPublish(pair);
#endif
}

bool RenderExecutor::TableDispatch(CommandBuffer& buffer, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z,
                                   uint64_t indirect_args, bool thread_dimensions) {
	if (TableDispatchMode() == 0) return false;
	if (indirect_args != 0 && thread_dimensions) return false;
	auto& table = OwnTable();
	if (table.dispatches.empty() || buffer.IsInvalid()) return false;
	TableSweepPairs();
	const auto& sh    = buffer.GetShaders();
	const auto& cs    = sh.GetCs();
	const auto  words = TableDispatchWords(cs, thread_dimensions);
	auto* known = table.found_dispatch_key == words ? table.found_dispatch : nullptr;
	if (known == nullptr) {
		const auto found = table.dispatches.find(TableDispatchKey(words));
		if (found == table.dispatches.end()) return false;
		known                    = &found->second;
		table.found_dispatch_key = words;
		table.found_dispatch     = known;
	}
	auto& pair = *known;
	if (pair.refused || pair.programs[0] == nullptr || pair.pipeline == nullptr || pair.vs != cs.cs_regs.data_addr)
		return false;
	TablePrefetchPair(pair);
	pair.used_frame = m_context.FrameNumber();
	TablePrefetchNext(table.previous_dispatch, pair);
	LiveCensus::Scope census(LiveCensus::DrawPhase, 0x7ab1f, 0);
	NativeXprCollect();
	if (const auto generation = ShaderMapGeneration(); generation != pair.Generation(Role())) {
		if (ShaderHashAt(pair.vs) != pair.programs[0]->shader_hash) {
			const auto write = CatalogWrite();
			pair.portable    = {};
			TableReset(pair);
			return false;
		}
		pair.Generation(Role()) = generation;
	}
	if (pair.respecialize) {
		const auto write = CatalogWrite();
		TableReset(pair);
		return false;
	}
	TablePrefetchArrays(pair, pair.LastSet(Role()));
	const std::array<std::span<const uint32_t>, 2> user_data {
	    std::span<const uint32_t>(cs.cs_user_sgpr.value, std::min<uint32_t>(cs.cs_regs.user_sgpr, HW::UserSgprInfo::SGPRS_MAX)),
	    std::span<const uint32_t> {}};
	std::optional<LiveCensus::Scope> phase;
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1f, 1);
#if KYTY_CX_SCOUT
	const auto* scout = Scout::Mode() == 3 && Scout::Running() && m_table_ticket != nullptr
	                        ? TableScoutTicket(pair, user_data, 1, m_context, static_cast<const Scout::Ticket*>(m_table_ticket))
	                        : nullptr;
	if (scout != nullptr) {
		TableScoutApply(table, pair, *scout, 1);
	} else
#endif
	{
		if (!TableEvaluate(pair, user_data)) return false;
		TableGather(pair, user_data);
	}
	phase.reset();
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1f, 2);
	auto* set = TableSet(buffer, pair, nullptr, nullptr);
	if (set == nullptr) return false;
	phase.reset();
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1f, 3);
	std::array<vk::DeviceAddress, 2> blocks {};
	bool                             storage_writes = false;
	auto&                            cache          = m_context.GetBufferCache();
	TablePrepareBda(pair, user_data);
	const auto registration = cache.RegistrationEpoch();
	if (!TableBlocks(pair, user_data, blocks, &storage_writes)) return false;
	Buffer*  args_owner  = nullptr;
	uint64_t args_offset = 0;
	if (indirect_args != 0) {
		std::tie(args_owner, args_offset) = cache.ObtainBuffer(indirect_args, sizeof(vk::DispatchIndirectCommand), false);
		if (args_owner == nullptr || (args_offset & 3u) != 0) return false;
	}
	if (cache.RegistrationEpoch() != registration) return LiveCounters::Add(LiveCounters::TableRegistered), false;
	phase.reset();
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1f, 4);
	buffer.EndRendering();
	auto       vk_buffer         = buffer.Handle();
	auto&      dispatch_textures = m_context.GetTextureCache();
	const auto images_epoch      = dispatch_textures.ImagesEpoch();
	if ((set->transit_epoch != images_epoch || set->transit_state != nullptr) && !TableTransit(*set, vk_buffer, nullptr))
		return false;
	set->transit_epoch = dispatch_textures.ImagesEpoch();
	set->transit_state = nullptr;
	const auto& pipeline = *pair.pipeline;
	const auto& program  = *pair.programs[0];
	const auto& bindings = program.bindings;
	vk_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline.pipeline_layout, 0, 1, &set->set, 0, nullptr);
	ShaderRecompiler::IR::PushData push_data;
	for (size_t i = 0; i < program.info.user_data_registers.size(); ++i) {
		const auto index = program.info.user_data_registers[i] - program.user_data_base;
		push_data.dwords[bindings.push_data_start_dword + i] = index < user_data[0].size() ? user_data[0][index] : 0;
	}
	if (thread_dimensions && bindings.dispatch_thread_dword != ShaderRecompiler::IR::PushData::NoStart) {
		push_data.dwords[bindings.push_data_start_dword + bindings.dispatch_thread_dword]      = groups_x;
		push_data.dwords[bindings.push_data_start_dword + bindings.dispatch_thread_dword + 1u] = groups_y;
		push_data.dwords[bindings.push_data_start_dword + bindings.dispatch_thread_dword + 2u] = groups_z;
	}
	push_data.dwords[bindings.push_data_start_dword + bindings.TableBlockDword()]     = static_cast<uint32_t>(blocks[0]);
	push_data.dwords[bindings.push_data_start_dword + bindings.TableBlockDword() + 1] = static_cast<uint32_t>(blocks[0] >> 32u);
	vk_buffer.pushConstants(pipeline.pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
	                        (bindings.push_data_start_dword + bindings.ShaderDataDwords()) * sizeof(uint32_t),
	                        push_data.dwords.data());
	const bool chained = std::exchange(m_dispatch_chained, false) && m_deferred_barrier;
	if (storage_writes && !chained) ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
	if (args_owner != nullptr) {
		if (!chained) {
			vk::MemoryBarrier barrier {};
			barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
			barrier.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
			vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllGraphics | vk::PipelineStageFlagBits::eComputeShader |
			                              vk::PipelineStageFlagBits::eTransfer,
			                          vk::PipelineStageFlagBits::eDrawIndirect, {}, 1, &barrier, 0, nullptr, 0, nullptr);
		}
		vk_buffer.dispatchIndirect(args_owner->Handle(), args_offset);
	} else if (thread_dimensions) {
		const auto& r      = cs.cs_regs;
		const auto  groups = [](uint32_t threads, uint32_t size) { return (threads + std::max(size, 1u) - 1u) / std::max(size, 1u); };
		vk_buffer.dispatch(groups(groups_x, r.num_thread_x), groups(groups_y, r.num_thread_y), groups(groups_z, r.num_thread_z));
	} else {
		vk_buffer.dispatch(groups_x, groups_y, groups_z);
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Dispatches);
	m_deferred_barrier       = true;
	m_deferred_barrier_queue = m_guest_queue;
	LiveCounters::Add(LiveCounters::TableDispatches);
	return true;
}
#endif
