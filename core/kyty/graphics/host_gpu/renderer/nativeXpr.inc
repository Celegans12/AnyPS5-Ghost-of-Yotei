namespace {

bool NativeXprSwitch(const char* name) {
	const char* value = std::getenv(name);
	return value == nullptr || value[0] != '0';
}

bool NativeXprAsyncPipelines() {
	static const bool enabled = NativeXprSwitch("KYTY_ASYNC_XPR_PIPELINES");
	return enabled;
}

struct NativeXprKeyHash {
	[[nodiscard]] size_t operator()(uint64_t key) const noexcept { return static_cast<size_t>(key); }
};

constexpr std::array<uint32_t, 8> NativeXprLodStatsMask {0, 0, 0, 0, 0, (0xfffu << 8u) | (1u << 25u), 0xffu, 0};

}

struct RenderExecutor::NativeXprDrawState {
	struct Target {
		ImageId               id;
		uint64_t              serial   = 0;
		vk::ImageLayout       layout   = vk::ImageLayout::eUndefined;
		vk::AccessFlags2      access;
		ImageSubresourceRange range;
		uint64_t              metadata = 0;
		uint32_t              slice    = 0;
		ImageInfo             requested;
		ImageViewInfo         view;
		mutable uint64_t      clear_memo = 0;
	};
	RenderState         rendering;
	std::vector<Target> targets;
	DynamicState        dynamic;
	mutable uint64_t    targets_epoch = 0;
	mutable uint64_t    used_frame    = 0;
	mutable uint64_t    emitted_epoch = UINT64_MAX;
	uint64_t            captures   = 0;
	uint64_t             dynamic_hash = 0;
	void                   Use(uint64_t frame) const { used_frame = frame; }
	[[nodiscard]] uint64_t UsedFrame() const { return used_frame; }
};

struct RenderExecutor::NativeXprCache {
	vk::Device device;
	std::unordered_map<uint64_t, std::unique_ptr<NativeXprDrawState>, NativeXprKeyHash> states;
	const NativeXprDrawState* last_state     = nullptr;
	uint64_t                  last_state_key = 0;
	struct Emit {
		uint64_t                       generation = UINT64_MAX;
		const NativeXprDrawState*      state      = nullptr;
		const PipelineCache::Pipeline* pipeline   = nullptr;
		uint64_t                       index_base = 0, index_bytes = 0;
		vk::IndexType                  index_type = vk::IndexType::eUint16;
		bool                           query_bias = false;
	} emit;
	struct RetiredSet {
		vk::DescriptorPool pool;
		vk::DescriptorSet  set;
		uint64_t           tick = 0;
	};
	std::vector<RetiredSet> retired;
	struct FreeSet {
		vk::DescriptorPool pool;
		vk::DescriptorSet  set;
	};
	static constexpr size_t FreeSetsPerLayout = 2048, FreeSetsMax = 16384;
	std::unordered_map<VkDescriptorSetLayout, std::vector<FreeSet>> free_sets;
	std::unordered_map<VkDescriptorSet, VkDescriptorSetLayout>      set_layouts;
	size_t                                                          free_count = 0;
	std::vector<vk::DescriptorPool>                                 pools;
	uint64_t state_key  = 0;
	uint64_t collected_frame = UINT64_MAX;
	static constexpr uint64_t StateSweepFrames = 600, StateKeepFrames = 3600;
	static constexpr size_t   StatesKept       = 2048;
	uint64_t                  states_swept_frame = UINT64_MAX;
	~NativeXprCache() {
		for (auto pool: pools) device.destroyDescriptorPool(pool);
	}
};

namespace {

bool StateKeyCacheEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_STATE_KEY_CACHE");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

uint64_t NativeXprStateKeyOf(const CommandBuffer& buffer, StateKeyCache* cache = nullptr,
                             uint64_t context_id = 0, uint64_t user_config_id = 0) {
	const auto& ctx = buffer.GetRegisters();
	const auto& cfg = buffer.GetUserConfig();
	const auto& vs  = buffer.GetShaders().GetVs();
	const auto& ps  = buffer.GetShaders().GetPs();
	uint64_t    hash = 0;
	if (cache != nullptr && context_id != 0 && user_config_id != 0 && StateKeyCacheEnabled() &&
	    cache->context_id == context_id && cache->user_config_id == user_config_id) {
		hash = cache->hash;
	} else {
		hash = XXH3_64bits(&ctx, sizeof(ctx));
		hash = XXH3_64bits_withSeed(&cfg, sizeof(cfg), hash);
		if (cache != nullptr) *cache = {context_id, user_config_id, hash};
	}
	hash      = XXH3_64bits_withSeed(&vs.es_regs, sizeof(vs.es_regs), hash);
	hash      = XXH3_64bits_withSeed(&vs.ls_regs, sizeof(vs.ls_regs), hash);
	hash      = XXH3_64bits_withSeed(&vs.hs_regs, sizeof(vs.hs_regs), hash);
	hash      = XXH3_64bits_withSeed(&vs.gs_regs, sizeof(vs.gs_regs), hash);
	hash      = XXH3_64bits_withSeed(&ps.ps_regs, sizeof(ps.ps_regs), hash);
	return hash;
}

}

RenderExecutor::NativeXprCache& RenderExecutor::NativeXprState() {
	if (!m_native_xpr) {
		m_native_xpr = {new NativeXprCache {}, [](NativeXprCache* cache) { delete cache; }};
		m_native_xpr->device  = m_context.GetGraphics().device;
	}
	return *m_native_xpr;
}

vk::DescriptorSet RenderExecutor::NativeXprAllocateSet(vk::DescriptorSetLayout layout,
                                                       vk::DescriptorPool&     pool_out) {
	auto&      cache  = NativeXprState();
	const auto device = cache.device;
	if (const auto found = cache.free_sets.find(static_cast<VkDescriptorSetLayout>(layout));
	    found != cache.free_sets.end() && !found->second.empty()) {
		const auto item = found->second.back();
		found->second.pop_back();
		--cache.free_count;
		pool_out = item.pool;
		return item.set;
	}
	vk::DescriptorSetAllocateInfo allocate {};
	allocate.descriptorSetCount = 1;
	allocate.pSetLayouts        = &layout;
	vk::DescriptorSet set       = nullptr;
	for (size_t i = 0; i < cache.pools.size() && i < 4; ++i) {
		const auto pool         = cache.pools[cache.pools.size() - 1 - i];
		allocate.descriptorPool = pool;
		if (device.allocateDescriptorSets(&allocate, &set) == vk::Result::eSuccess) {
			pool_out = pool;
			cache.set_layouts[static_cast<VkDescriptorSet>(set)] = static_cast<VkDescriptorSetLayout>(layout);
			return set;
		}
	}
	constexpr uint32_t Sets = 1024;
	const std::array<vk::DescriptorPoolSize, 5> sizes {{
	    {vk::DescriptorType::eStorageBuffer, Sets * 16},
	    {vk::DescriptorType::eStorageBufferDynamic, Sets * 4},
	    {vk::DescriptorType::eSampledImage, Sets * 32},
	    {vk::DescriptorType::eSampler, Sets * 8},
	    {vk::DescriptorType::eStorageImage, Sets * 4},
	}};
	vk::DescriptorPoolCreateInfo create {};
	create.flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
	create.maxSets       = Sets;
	create.poolSizeCount = static_cast<uint32_t>(sizes.size());
	create.pPoolSizes    = sizes.data();
	vk::DescriptorPool pool = nullptr;
	EXIT_IF(device.createDescriptorPool(&create, nullptr, &pool) != vk::Result::eSuccess);
	cache.pools.push_back(pool);
	allocate.descriptorPool = pool;
	if (device.allocateDescriptorSets(&allocate, &set) != vk::Result::eSuccess) {
		return nullptr;
	}
	pool_out = pool;
	cache.set_layouts[static_cast<VkDescriptorSet>(set)] = static_cast<VkDescriptorSetLayout>(layout);
	return set;
}

void RenderExecutor::NativeXprRetireSet(vk::DescriptorPool pool, vk::DescriptorSet set) {
	if (!set) return;
	TableMemoRetired();
	NativeXprState().retired.push_back({pool, set, m_context.GetCommandScheduler().CurrentTick()});
}

bool RenderExecutor::NativeXprSweepStates(NativeXprCache& cache, uint64_t frame) {
	if (Role() != 0 || cache.states.size() <= NativeXprCache::StatesKept) return true;
	if (kyty_local_speculations_live.load(std::memory_order_seq_cst) != 0) return false;
	const auto write = CatalogWrite();
	if (kyty_local_speculations_live.load(std::memory_order_seq_cst) != 0) return false;
	const size_t before = cache.states.size();
	std::erase_if(cache.states, [&](const auto& item) {
		return item.second->UsedFrame() + NativeXprCache::StateKeepFrames < frame;
	});
	if (cache.states.size() == before) return true;
	cache.emit           = {};
	cache.last_state     = nullptr;
	cache.last_state_key = 0;
	TableForgetStates();
	return true;
}

void RenderExecutor::NativeXprCollect() {
	if (!m_native_xpr) return;
	auto&      cache = *m_native_xpr;
	const auto frame = m_context.FrameNumber();
	if (frame - cache.states_swept_frame >= NativeXprCache::StateSweepFrames && NativeXprSweepStates(cache, frame))
		cache.states_swept_frame = frame;
	if (cache.collected_frame == frame || cache.retired.empty()) return;
	cache.collected_frame = frame;
	auto& scheduler       = m_context.GetCommandScheduler();
	std::vector<NativeXprCache::RetiredSet> done;
	std::erase_if(cache.retired, [&](const NativeXprCache::RetiredSet& item) {
		if (!scheduler.IsFree(item.tick)) return false;
		const auto layout = cache.set_layouts.find(static_cast<VkDescriptorSet>(item.set));
		if (layout != cache.set_layouts.end() && cache.free_count < NativeXprCache::FreeSetsMax) {
			auto& sets = cache.free_sets[layout->second];
			if (sets.size() < NativeXprCache::FreeSetsPerLayout) {
				sets.push_back({item.pool, item.set});
				++cache.free_count;
				return true;
			}
		}
		if (layout != cache.set_layouts.end()) cache.set_layouts.erase(layout);
		done.push_back(item);
		return true;
	});
	std::ranges::sort(done, {}, [](const auto& item) { return static_cast<VkDescriptorPool>(item.pool); });
	std::vector<vk::DescriptorSet> batch;
	for (size_t first = 0; first < done.size();) {
		size_t last = first;
		batch.clear();
		while (last < done.size() && done[last].pool == done[first].pool) batch.push_back(done[last++].set);
		(void)cache.device.freeDescriptorSets(done[first].pool, static_cast<uint32_t>(batch.size()), batch.data());
		first = last;
	}
}

void RenderExecutor::NativeXprCaptureState(const DrawRenderState& state, const RenderState& rendering,
                                           uint64_t state_key, const DynamicState& dynamic_state) {
	auto&               cache    = NativeXprState();
	auto&               textures = m_context.GetTextureCache();
	NativeXprDrawState  draw_state;
	draw_state.rendering = rendering;
	draw_state.rendering.depth_stencil_attachment.depth_clear   = false;
	draw_state.rendering.depth_stencil_attachment.stencil_clear = false;
	const auto metadata_of = [](const TextureCache::ImageDesc& desc) {
		return desc.info.metadata.kind == ImageMetadataKind::None ? uint64_t {0} : desc.info.metadata.range.address;
	};
	const auto sampled_access = [](const Image& image) {
		return image.binding.is_bound && !image.binding.shader_write
		           ? image.binding.attachment_access | vk::AccessFlagBits2::eShaderRead
		           : image.binding.attachment_access;
	};
	for (uint32_t i = 0; i < m_draw_targets->color_count; ++i) {
		const auto& target = m_draw_targets->colors[i];
		const auto& view   = target.desc.view_info;
		const auto& image  = textures.GetImage(target.image_id);
		draw_state.targets.push_back({target.image_id, image.serial,
		                              rendering.color_attachments[target.target_slot].image_layout,
		                              sampled_access(image),
		                              {view.base_level, view.level_count, view.base_layer, view.layer_count},
		                              metadata_of(target.desc), view.base_layer, target.desc.info, view});
	}
	if (m_draw_targets->depth.image_id) {
		const auto& view  = m_draw_targets->depth.desc.view_info;
		const auto& image = textures.GetImage(m_draw_targets->depth.image_id);
		draw_state.targets.push_back({m_draw_targets->depth.image_id, image.serial,
		                              rendering.depth_stencil_attachment.image_layout,
		                              sampled_access(image),
		                              {view.base_level, view.level_count, view.base_layer, view.layer_count},
		                              metadata_of(m_draw_targets->depth.desc), view.base_layer, m_draw_targets->depth.desc.info,
		                              view});
	}
	draw_state.dynamic       = dynamic_state;
	draw_state.targets_epoch = textures.ImageSetEpoch();
	draw_state.used_frame    = m_context.FrameNumber();
	auto& slot               = cache.states[state_key];
	if (!slot) slot = std::make_unique<NativeXprDrawState>();
	const auto captures = slot->captures;
	*slot               = std::move(draw_state);
	slot->captures   = captures + 1;
	cache.last_state = nullptr;
	if (cache.emit.state == slot.get()) cache.emit.state = nullptr;
}

bool RenderExecutor::NativeXprEmitTargets(const NativeXprDrawState& draw_state, CommandRecorder vk_buffer) {
	auto& textures = m_context.GetTextureCache();
	for (const auto& target: draw_state.targets) {
		auto* image = textures.m_slot_images.try_get(target.id);
		if (image == nullptr || image->serial != target.serial || !image->registered || image->binding.needs_rebind ||
		    !image->SafeToDownload())
			return false;
		if (target.metadata != 0 && target.requested.IsDepth() && textures.IsMetaCleared(target.metadata, target.slice))
			return false;
		if (!target.requested.IsDepth() &&
		    (target.requested.metadata.kind == ImageMetadataKind::Dcc ||
		     target.requested.metadata.kind == ImageMetadataKind::Cmask) &&
		    !textures.ColorClearSettled({target.requested, target.view, TextureCache::BindingType::RenderTarget},
		                                target.clear_memo))
			return false;
	}
	for (const auto& target: draw_state.targets) {
		auto& image = textures.GetImage(target.id);
		image.Transit(target.layout, target.access, target.range, vk_buffer);
	}
	return true;
}

void RenderExecutor::NativeXprEmitDynamic(NativeXprCache&, CommandRecorder vk_buffer,
                                          const NativeXprDrawState& draw_state, bool, bool query_bias) {
	auto& dynamic = m_context.GetCommandScheduler().GetDynamicState();
	dynamic.Assign(draw_state.dynamic);
	if (query_bias) {
		dynamic.BiasTowardViewer(OcclusionCounters::ProxyBiasUnits, OcclusionCounters::ProxyBiasSlope);
	}
	dynamic.Commit(m_context.GetGraphics(), vk_buffer);
}

bool RenderExecutor::NativeXprTargetsCurrent(const NativeXprDrawState& draw_state) {
	auto&      textures = m_context.GetTextureCache();
	const auto current  = textures.ImageSetEpoch();
	if (draw_state.targets_epoch == current) return true;
	for (const auto& target: draw_state.targets) {
		const auto* image = textures.m_slot_images.try_get(target.id);
		if (image == nullptr || image->serial != target.serial || !image->registered) return false;
	}
	bool       moved = false;
	const bool known = textures.ForEachImageSetChangeSince(draw_state.targets_epoch, [&](uint64_t address, uint64_t size) {
		for (const auto& target: draw_state.targets)
			moved = moved || ImageRangeOverlaps(address, size, target.requested.data.address, target.requested.data.size);
	});
	if (!known || moved) {
		for (const auto& target: draw_state.targets)
			if (!textures.IsTargetCurrent(target.id, target.requested)) return false;
	}
	draw_state.targets_epoch = current;
	return true;
}
