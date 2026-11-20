#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#if defined(ANYPS5_KYTY)
#include "graphics/host_gpu/renderer/pipeline/pipelineLibrary.h"
#endif

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#if defined(ANYPS5_KYTY)
#include "common/fallback.h"
#endif
#include "common/file.h"
#include "common/hackNotice.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/execHelpers.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/rayTracing.h"
#include "graphics/host_gpu/renderer/pipeline/blendMapping.h"
#if defined(ANYPS5_KYTY)
#include "graphics/host_gpu/renderer/pipeline/programLock.h"
#endif
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/Tessellation.h"
#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fmt/format.h>
#include <limits>
#include <mutex>
#include <optional>
#include <ranges>
#include <span>
#include <spirv-tools/libspirv.hpp>
#include <string_view>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

constexpr auto DriverCacheSaveInterval = std::chrono::minutes {1};
constexpr auto SlowCreation             = std::chrono::seconds {5};
constexpr auto SlowCreationSaveInterval = std::chrono::seconds {20};
constexpr auto DriverCacheTimerPeriod   = std::chrono::milliseconds {250};
constexpr auto HelperPoll = std::chrono::milliseconds {1};

uint8_t RemapSourceAlphaFactor(uint8_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Alpha);
		case Prospero::BlendFactor::kOneMinusSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
		default: return factor;
	}
}

#if !defined(ANYPS5_KYTY)
vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	EXIT_NOT_IMPLEMENTED(mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default: EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
	}
}
#endif

#if defined(ANYPS5_KYTY)

bool PipelineChainWanted() {
	static const bool wanted = [] {
		const char* value = std::getenv("KYTY_PIPELINE_CHAIN");
		return value == nullptr || std::string_view(value) != "0";
	}();
	return wanted;
}

uint32_t PipelineChainThreads() {
	static const uint32_t threads = [] {
		const char* value = std::getenv("KYTY_PIPELINE_CHAIN_THREADS");
		const auto  count = value != nullptr ? std::strtoul(value, nullptr, 10) : 4ul;
		return static_cast<uint32_t>(std::clamp<unsigned long>(count, 1ul, 16ul));
	}();
	return threads;
}
#endif

std::string DriverCacheSignature(const vk::PhysicalDeviceProperties& properties) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(VK_UUID_SIZE * 2, '0');
	for (size_t i = 0; i < VK_UUID_SIZE; i++) {
		uuid[i * 2]     = hex[properties.pipelineCacheUUID[i] >> 4u];
		uuid[i * 2 + 1] = hex[properties.pipelineCacheUUID[i] & 0xfu];
	}
	return fmt::format("KytyPC2:{:08x}:{:08x}:{:08x}:{}\n",
	                   properties.vendorID, properties.deviceID, properties.driverVersion, uuid);
}

#if defined(ANYPS5_KYTY)
std::string ReadDriverCacheSignature(const std::filesystem::path& path, size_t size) {
	if (!Common::File::IsFileExisting(path)) {
		return {};
	}
	Common::File file(path, Common::File::Mode::Read);
	if (file.IsInvalid()) {
		return {};
	}
	std::string signature(size, '\0');
	uint32_t    read = 0;
	file.Read(signature.data(), static_cast<uint32_t>(size), &read);
	file.Close();
	return read == size ? signature : std::string {};
}

std::filesystem::path DriverCachePath(const std::string& title_id, const std::string& signature) {
	const std::filesystem::path directory = "_PipelineCache";
	const auto                  shared    = directory / (title_id + ".bin");
	const auto own = directory / fmt::format("{}.{:016x}.bin", title_id, XXH3_64bits(signature.data(), signature.size()));
	const bool shared_exists = Common::File::IsFileExisting(shared);
	if (shared_exists && ReadDriverCacheSignature(shared, signature.size()) == signature) {
		return shared;
	}
	if (shared_exists || Common::File::IsFileExisting(own)) {
		return own;
	}
	return shared;
}
#endif

template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

#if defined(ANYPS5_KYTY)
void LogShaderVariantBuild(const char* stage, uint64_t shader_hash, uint32_t user_data_count,
                           const std::vector<uint32_t>& static_state, uint32_t push_data_cursor,
                           const ShaderModule& module) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SHADER_VARIANT_LOG");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	if (!enabled || module.module == nullptr) return;

	ShaderRecompiler::IR::Writer writer;
	const auto fingerprint = [&writer](const auto& fields) {
		writer.bytes.clear();
		ShaderRecompiler::IR::IoAll(writer, fields);
		return XXH3_64bits(writer.bytes.data(), writer.bytes.size());
	};
	const auto& specialization = module.specialization;
	const auto& functions      = module.function_code;
	const auto plain_fields = module.plain != nullptr
	                              ? fmt::format(" plain_module={:016x} plain_spirv_words={}",
	                                            module.plain->id, module.plain->spirv_words)
	                              : std::string {};
	PipelineCacheLog(
	    "[kyty-variant] build stage={} shader={:016x} module={:016x} user_data_count={} "
	    "static={:016x} push_start={} push_cursor={} buffers={:016x} images={:016x} "
	    "samplers={:016x} linked_code={:016x} table={} spirv_words={}{} @{}",
	    stage, shader_hash, module.id, user_data_count, fingerprint(static_state),
	    module.info.bindings.push_data_start_dword, push_data_cursor,
	    fingerprint(specialization.buffers), fingerprint(specialization.images),
	    fingerprint(specialization.samplers), XXH3_64bits(functions.data(), functions.size() * sizeof(uint32_t)),
	    module.info.table_mode, module.spirv_words, plain_fields,
	    std::chrono::steady_clock::now().time_since_epoch().count());
}
#endif

bool ReadAheadStrict(void* userdata, uint64_t address, std::span<uint32_t> values) {
	return !values.empty() && static_cast<GuestReadLog*>(userdata)->Read(address, values.data(),
	                                                                     values.size_bytes(), true);
}

bool ReadAheadShaderState(void* userdata, uint64_t address, void* data, uint64_t size) {
	if (address < 0x40000u && size != 0u) {
		std::memset(data, 0, size);
		return true;
	}
	return static_cast<GuestReadLog*>(userdata)->Read(address, data, size, false);
}

struct AheadWrites {
	static constexpr size_t   Capacity        = 256;
	static constexpr uint64_t BlockShift      = 16;
	static constexpr size_t   FilterBits      = 12;
	static constexpr uint64_t MaxFilterBlocks = 4096;
	std::array<std::pair<uint64_t, uint64_t>, Capacity> ranges {};
	std::array<uint16_t, size_t {1} << FilterBits>     filter {};
	uint32_t                                            wide = 0;
	size_t                                              next = 0;

	static size_t Slot(uint64_t block) {
		return static_cast<size_t>((block * 0x9e3779b97f4a7c15ull) >> (64u - FilterBits));
	}
	void Count(uint64_t begin, uint64_t end, int delta) {
		if (begin >= end) return;
		const auto first = begin >> BlockShift;
		const auto last  = (end - 1) >> BlockShift;
		if (last - first >= MaxFilterBlocks) {
			wide = static_cast<uint32_t>(static_cast<int>(wide) + delta);
			return;
		}
		for (auto block = first; block <= last; block++) {
			auto& count = filter[Slot(block)];
			count       = static_cast<uint16_t>(count + delta);
		}
	}
	[[nodiscard]] bool Has(uint64_t begin, uint64_t end) const {
		return std::ranges::any_of(ranges, [&](const auto& range) { return range.first == begin && range.second == end; });
	}
	void Add(uint64_t begin, uint64_t end) {
		if (Has(begin, end)) return;
		Count(ranges[next].first, ranges[next].second, -1);
		ranges[next] = {begin, end};
		Count(begin, end, 1);
		next = (next + 1) % Capacity;
	}
	[[nodiscard]] bool Contains(uint64_t address) const {
		if (wide == 0 && filter[Slot(address >> BlockShift)] == 0) return false;
		for (const auto& [begin, end]: ranges) {
			if (address >= begin && address < end) return true;
		}
		return false;
	}
};
AheadWrites                               g_ahead_writes;
std::atomic<uint64_t>                     g_ahead_writes_sequence {0};
thread_local std::unique_ptr<AheadWrites> t_ahead_writes_copy;
thread_local uint64_t                     t_ahead_writes_copied = UINT64_MAX;

AheadWrites& ThreadAheadWrites() {
	return t_ahead_writes_copy != nullptr ? *t_ahead_writes_copy : g_ahead_writes;
}

struct AheadGuestWrites {
	struct Write {
		uint64_t              sequence = 0;
		uint64_t              address  = 0;
		std::vector<uint32_t> dwords;

		[[nodiscard]] uint64_t End() const { return address + dwords.size() * sizeof(uint32_t); }
	};
	static constexpr size_t   FilterBits = 12;
	static constexpr uint64_t PageShift  = 12;
	std::deque<Write>                              writes;
	std::array<uint16_t, size_t {1} << FilterBits> filter {};

	static size_t Slot(uint64_t page) {
		return static_cast<size_t>((page * 0x9e3779b97f4a7c15ull) >> (64u - FilterBits));
	}
	void Count(const Write& write, int delta) {
		for (auto page = write.address >> PageShift; page <= (write.End() - 1) >> PageShift; page++) {
			auto& count = filter[Slot(page)];
			count       = static_cast<uint16_t>(count + delta);
		}
	}
	[[nodiscard]] bool MayTouch(uint64_t address, uint64_t size) const {
		if (writes.empty() || size == 0) return false;
		for (auto page = address >> PageShift; page <= (address + size - 1) >> PageShift; page++) {
			if (filter[Slot(page)] != 0) return true;
		}
		return false;
	}
	bool Apply(uint64_t address, void* data, uint64_t size) const {
		if (!MayTouch(address, size)) return false;
		bool applied = false;
		for (const auto& write: writes) {
			const auto begin = std::max(address, write.address);
			const auto end   = std::min(address + size, write.End());
			if (begin >= end) continue;
			std::memcpy(static_cast<uint8_t*>(data) + (begin - address),
			            reinterpret_cast<const uint8_t*>(write.dwords.data()) + (begin - write.address),
			            end - begin);
			applied = true;
		}
		return applied;
	}
};
thread_local AheadGuestWrites t_ahead_guest_writes;

bool AheadPendingWrite(uint64_t address, uint32_t& value) {
	auto& state = t_ahead_guest_writes;
	if (!state.MayTouch(address, sizeof(value))) return false;
	if (!Libs::LibKernel::Memory::TryReadBacking(address, &value, sizeof(value))) value = 0;
	return state.Apply(address, &value, sizeof(value));
}
thread_local bool t_ahead_unread_load = false;

struct GpuProducedBounds {
	static constexpr size_t                     Capacity = 64;
	std::array<std::atomic<uint64_t>, Capacity> addresses {};
	std::atomic<uint32_t>                       count {0};

	[[nodiscard]] bool Contains(uint64_t address) const {
		const auto known = std::min<size_t>(count.load(std::memory_order_acquire), Capacity);
		for (size_t i = 0; i < known; i++) {
			if (addresses[i].load(std::memory_order_relaxed) == address) return true;
		}
		return false;
	}
	void Add(uint64_t address) {
		if (Contains(address)) return;
		const auto index = count.load(std::memory_order_relaxed);
		addresses[index % Capacity].store(address, std::memory_order_relaxed);
		count.store(index + 1, std::memory_order_release);
	}
};
GpuProducedBounds g_gpu_produced_bounds;

void LearnGpuProducedBound(uint64_t address) {
	g_gpu_produced_bounds.Add(address);
}

bool AheadGpuProduced(uint64_t address) {
	if (!ThreadAheadWrites().Contains(address) && !g_gpu_produced_bounds.Contains(address)) return false;
	t_ahead_unread_load = true;
	return true;
}

void NoteAheadWrites(const ShaderRecompiler::IR::ResourcePlan&     plan,
                     const ShaderRecompiler::IR::ResourceSnapshot& resources) {
	for (size_t i = 0; i < plan.info.buffers.size() && i < resources.buffers.size(); i++) {
		const auto& descriptor = resources.buffers[i];
		if (!plan.info.buffers[i].written || descriptor.dword_count != 4u) continue;
		const uint64_t base    = descriptor.dwords[0] | (uint64_t {descriptor.dwords[1] & 0xffffu} << 32u);
		const uint32_t stride  = (descriptor.dwords[1] >> 16u) & 0x3fffu;
		const uint64_t size    = uint64_t {descriptor.dwords[2]} * (stride != 0u ? stride : 1u);
		if (base == 0 || size == 0) continue;
		if (t_ahead_writes_copy != nullptr) {
			t_ahead_writes_copy->Add(base, base + size);
			continue;
		}
		if (g_ahead_writes.Has(base, base + size)) continue;
		const auto sequence = g_ahead_writes_sequence.load(std::memory_order_relaxed);
		g_ahead_writes_sequence.store(sequence + 1, std::memory_order_relaxed);
		std::atomic_thread_fence(std::memory_order_release);
		g_ahead_writes.Add(base, base + size);
		g_ahead_writes_sequence.store(sequence + 2, std::memory_order_release);
	}
}

void RecordAheadLoad(void* userdata, uint64_t address, uint32_t value, uint32_t flat_slot) {
	static_cast<GuestReadLog*>(userdata)->RecordLoad(address, value, flat_slot);
}

void RecordAheadLoads(void* userdata, std::span<const uint64_t> loads) {
	static_cast<GuestReadLog*>(userdata)->RecordLoads(loads);
}

class ProgramLockGuard {
public:
#if defined(ANYPS5_KYTY)
	explicit ProgramLockGuard(std::atomic_flag& flag, bool take = true,
	                          std::source_location where = std::source_location::current())
	    : m_flag(take ? &flag : nullptr) {
		if (m_flag != nullptr) ProgramLock::Acquire(*m_flag, where);
	}
	~ProgramLockGuard() {
		if (m_flag != nullptr) ProgramLock::Release(*m_flag);
	}
#else
	explicit ProgramLockGuard(std::atomic_flag& flag, bool take = true): m_flag(take ? &flag : nullptr) {
		if (m_flag == nullptr) return;
		while (m_flag->test_and_set(std::memory_order_acquire)) {
			while (m_flag->test(std::memory_order_relaxed)) {
				Common::SpinPause();
			}
		}
	}
	~ProgramLockGuard() {
		if (m_flag != nullptr) m_flag->clear(std::memory_order_release);
	}
#endif
	ProgramLockGuard(const ProgramLockGuard&)            = delete;
	ProgramLockGuard& operator=(const ProgramLockGuard&) = delete;

private:
	std::atomic_flag* m_flag;
};

thread_local bool t_ahead_unlocked = false;
#if defined(ANYPS5_KYTY)
thread_local bool t_execute_unlocked = false;
thread_local bool t_ahead_missing    = false;
#endif

bool ReadShaderGuestMemory(void*, uint64_t address, std::span<uint32_t> values) {
#if defined(ANYPS5_KYTY)
	if (ProgramLock::Enabled()) {
		return !values.empty() && ProgramLock::TimedGuestRead(address, values);
	}
#endif
	return !values.empty() &&
	       Libs::LibKernel::Memory::TryReadBufferBacking(address, values.data(), values.size_bytes());
}

void DumpShader(std::span<const uint32_t> words, const std::string& name) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	const auto path = Config::GetShaderLogFolder() / name;
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(words.data(), words.size_bytes());
}

void ValidateShaderSpirv(const char* stage, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv, bool maintenance8_enabled) {
	if (!Config::ShaderValidationEnabled()) {
		return;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	spvtools::ValidatorOptions validator_options;
	validator_options.SetAllowOffsetTextureOperand(maintenance8_enabled);
	if (tools.Validate(spirv.data(), spirv.size(), validator_options)) {
		return;
	}
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF("%s\n", text.c_str());
	EXIT("ShaderRecompiler %s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s", stage,
	     shader_hash, messages.c_str());
}

vk::Format NarrowInputFormat(vk::Format format, uint32_t used_components) {
	switch (format) {
		case vk::Format::eR32G32B32A32Sfloat:
			switch (used_components) {
				case 1: return vk::Format::eR32Sfloat;
				case 2: return vk::Format::eR32G32Sfloat;
				case 3: return vk::Format::eR32G32B32Sfloat;
				default: return format;
			}
		case vk::Format::eR32G32B32Sfloat:
			switch (used_components) {
				case 1: return vk::Format::eR32Sfloat;
				case 2: return vk::Format::eR32G32Sfloat;
				default: return format;
			}
		case vk::Format::eR16G16B16A16Sfloat:
			switch (used_components) {
				case 1: return vk::Format::eR16Sfloat;
				case 2: return vk::Format::eR16G16Sfloat;
				default: return format;
			}
		case vk::Format::eR8G8B8A8Unorm:
			switch (used_components) {
				case 1: return vk::Format::eR8Unorm;
				case 2: return vk::Format::eR8G8Unorm;
				default: return format;
			}
		case vk::Format::eR8G8B8A8Snorm:
			return used_components == 2 ? vk::Format::eR8G8Snorm : format;
		case vk::Format::eR8G8B8A8Uint:
			switch (used_components) {
				case 1: return vk::Format::eR8Uint;
				case 2: return vk::Format::eR8G8Uint;
				default: return format;
			}
		default: return format;
	}
}

vk::Format InputFormat(const ShaderBufferResource& resource, uint32_t used_components) {
	switch (resource.RawFormat()) {
		case 113: return NarrowInputFormat(vk::Format::eR32G32B32A32Sfloat, used_components);
		case 121: return vk::Format::eR16G16Sfloat;
		default: break;
	}
	const auto format = VulkanFormat(resource.Format());
	if (format == vk::Format::eUndefined ||
	    ShaderRecompiler::Format::GetFormatInfo(resource.Format()).component_count == 0) {
		EXIT("unknown vertex format: fmt = %u\n", resource.RawFormat());
	}
	return NarrowInputFormat(format, used_components);
}

std::vector<vk::Format> VertexInputFormats(const ShaderVertexInputInfo&                    input_info,
                                           const ShaderRecompiler::IR::CompiledShaderInfo& program) {
	std::vector<vk::Format> formats(input_info.resources_num);
	for (int index = 0; index < input_info.resources_num; index++) {
		const auto registers_num = input_info.resources_dst[index].registers_num;
		EXIT_IF(registers_num < 1 || registers_num > 4);
		const auto fetched = program.info.vertex_fetch_components[index];
		formats[index] = InputFormat(input_info.resources[index],
		                             fetched > 0 ? fetched : static_cast<uint32_t>(registers_num));
	}
	return formats;
}

void SkipCallExpansionFailure(uint64_t hash, const ShaderRecompiler::ShaderSource& source) {
	Common::HackNotice::Report("dispatch whose call tables do not expand skipped",
	                           "PipelineCache::GetProgram", hash, 0, source.refresh_failure);
}

}

void AppendNativeRayTracingKey(ShaderType stage, std::vector<uint32_t>& key, bool native) {
	if (native && stage == ShaderType::Pixel && RayTracing::NativeTraversals() && Config::RayTracingOracle() != 0) {
		key.push_back(0x52544f00u | Config::RayTracingOracle());
		if (Config::RayTracingStatsEnabled()) {
			key.push_back(0x52545354u);
		}
	}
	if (native && stage == ShaderType::Compute && RayTracing::NativeTraversals()) {
		key.push_back(0x52545041u);
		if (Config::RayTracingStatsEnabled()) {
			key.push_back(0x52545354u);
		}
		if (const auto oracle = Config::RayTracingOracle(); oracle != 0) {
			key.push_back(0x52544f00u | oracle);
		}
	}
}

std::string PipelineCacheTitleId() {
	std::string title_id;
	if ((!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) &&
	    (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) || title_id.empty())) {
		return {};
	}
	if (!std::ranges::all_of(title_id, [](unsigned char c) {
		    return std::isalnum(c) != 0 || c == '-' || c == '_';
	    })) {
		return {};
	}
	return title_id;
}

namespace {
bool LdsLaneSyncEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY76_LDS_SYNC");
		const bool  on    = value == nullptr || std::strcmp(value, "0") != 0;
		Log::WriteToConsoleAndLog(on ? "76 LDS lane sync: on (wave64 halves)\n" : "76 LDS lane sync: off\n");
		return on;
	}();
	return enabled;
}
}

ShaderRecompiler::Spirv::Profile SpirvProfile(const GraphicContext& graphics) {
	return {.maintenance8           = graphics.maintenance8_enabled,
	        .native_float_min_max   = graphics.physical_device_properties.vendorID == 0x1002u,
	        .device_clock_period_ps = graphics.device_clock_period_ps,
	        .robust_buffer_access   = graphics.robust_buffer_access_enabled,
	        .r8_srgb_views          = graphics.r8_srgb_views,
	        .r8g8_srgb_views        = graphics.r8g8_srgb_views,
	        .relaxed_float          = Config::RelaxedShaderFloatEnabled(),
	        .wave_split             = Config::WaveSplitEnabled(),
	        .ray_query              = graphics.ray_query_enabled,
	        .lds_lane_sync          = LdsLaneSyncEnabled()};
}

std::filesystem::path ShaderStorePath() {
	if (KYTY_BUILD != KYTY_BUILD_RELEASE || !Config::ShaderCacheEnabled() ||
	    Config::GraphicsDebugDumpEnabled()) {
		return {};
	}
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return {};
	}
	return std::filesystem::path("_PipelineCache") / (title_id + ".shaders");
}

void GuestReadLog::Append(uint64_t address, const void* data, uint32_t size, bool strict,
                          bool zero_test) {
	m_gpu_written = m_gpu_written || Libs::LibKernel::Memory::MayBeGpuWritten(address, size);
	const auto offset = static_cast<uint32_t>(m_bytes.size());
	m_bytes.insert(m_bytes.end(), static_cast<const uint8_t*>(data),
	               static_cast<const uint8_t*>(data) + size);
	if (!m_entries.empty() && !zero_test) {
		auto& last = m_entries.back();
		if (last.strict == strict && !last.zero_test && last.address + last.size == address &&
		    last.offset + last.size == offset &&
		    (!strict || last.size + size <= StrictCheckBytes)) {
			last.size += size;
			return;
		}
	}
	m_entries.push_back({.address = address, .size = size, .offset = offset, .strict = strict,
	                     .zero_test = zero_test});
}

bool GuestReadLog::Read(uint64_t address, void* data, uint64_t size, bool strict) {
	if (size == 0 || size > UINT32_MAX) {
		m_failed = true;
		return false;
	}
	const auto watched = strict && size >= StrictWatch::MinSize
	                         ? Libs::LibKernel::Memory::WatchStrictRead(address, size)
	                         : 0;
	if (!Libs::LibKernel::Memory::TryReadBacking(address, data, size)) {
		m_failed = true;
		return false;
	}
	const bool ahead = t_ahead_guest_writes.Apply(address, data, size);
	Append(address, data, static_cast<uint32_t>(size), strict);
	if (watched != 0 && !ahead) {
		m_entries.back().watched = watched;
	}
	return true;
}

void GuestReadLog::RecordLoad(uint64_t address, uint32_t value, uint32_t flat_slot) {
	if (flat_slot == ShaderRecompiler::IR::ZeroTestSlot) {
		Append(address, &value, sizeof(value), false, true);
		return;
	}
	if (ShaderRecompiler::IR::IsDescriptorSlot(flat_slot)) {
		const auto source = ShaderRecompiler::IR::DescriptorSlotSource(flat_slot);
		const auto word   = ShaderRecompiler::IR::DescriptorSlotWord(flat_slot);
		const auto kind   = m_plan != nullptr && source < m_plan->descriptor_refresh.size()
		                        ? m_plan->descriptor_refresh[source]
		                        : ShaderRecompiler::IR::DescriptorRefresh::None;
		const auto offset = static_cast<uint32_t>(m_bytes.size());
		Append(address, &value, sizeof(value), false);
		const auto mask = ShaderRecompiler::IR::DescriptorRefreshMask(kind, word);
		if (mask != 0) {
			auto&      entry = m_entries.back();
			const auto index = static_cast<uint32_t>(m_words.size());
			m_words.push_back(
			    {.offset = offset, .stage = m_stage, .source = source, .word = word, .mask = mask});
			if (entry.words_begin == entry.words_end) {
				entry.words_begin = index;
			}
			entry.words_end = index + 1;
		}
		return;
	}
	if (flat_slot != ShaderRecompiler::IR::NoFlatSlot) {
		if (!m_flat_loads.empty()) {
			auto& last = m_flat_loads.back();
			const auto next = last.address + uint64_t {last.count} * sizeof(uint32_t);
			if (last.stage == m_stage && next == address && last.slot + last.count == flat_slot &&
			    (address & 0xfffu) != 0) {
				last.count++;
				return;
			}
		}
		m_flat_loads.push_back({address, m_stage, flat_slot, 1});
		return;
	}
	Append(address, &value, sizeof(value), false);
}

void GuestReadLog::RecordLoads(std::span<const uint64_t> loads) {
	auto bytes_end = m_bytes.size();
	m_bytes.resize(bytes_end + loads.size() / 2u * sizeof(uint32_t));
	const auto append = [&](uint64_t address, uint32_t value, bool zero_test) {
		m_gpu_written =
		    m_gpu_written || Libs::LibKernel::Memory::MayBeGpuWritten(address, sizeof(value));
		const auto offset = static_cast<uint32_t>(bytes_end);
		std::memcpy(m_bytes.data() + bytes_end, &value, sizeof(value));
		bytes_end += sizeof(value);
		if (!m_entries.empty() && !zero_test) {
			auto& last = m_entries.back();
			if (!last.strict && !last.zero_test && last.address + last.size == address &&
			    last.offset + last.size == offset) {
				last.size += sizeof(value);
				return;
			}
		}
		m_entries.push_back({.address = address, .size = sizeof(value), .offset = offset,
		                     .strict = false, .zero_test = zero_test});
	};
	for (size_t i = 0; i + 1 < loads.size(); i += 2) {
		const auto address = loads[i];
		if (address < ShaderRecompiler::IR::SrtGuestMemoryStart) {
			continue;
		}
		const auto value     = static_cast<uint32_t>(loads[i + 1]);
		const auto flat_slot = static_cast<uint32_t>(loads[i + 1] >> 32u);
		if (flat_slot == ShaderRecompiler::IR::ZeroTestSlot) {
			append(address, value, true);
			continue;
		}
		if (ShaderRecompiler::IR::IsDescriptorSlot(flat_slot)) {
			const auto source = ShaderRecompiler::IR::DescriptorSlotSource(flat_slot);
			const auto word   = ShaderRecompiler::IR::DescriptorSlotWord(flat_slot);
			const auto kind   = m_plan != nullptr && source < m_plan->descriptor_refresh.size()
			                        ? m_plan->descriptor_refresh[source]
			                        : ShaderRecompiler::IR::DescriptorRefresh::None;
			const auto offset = static_cast<uint32_t>(bytes_end);
			append(address, value, false);
			const auto mask = ShaderRecompiler::IR::DescriptorRefreshMask(kind, word);
			if (mask != 0) {
				auto&      entry = m_entries.back();
				const auto index = static_cast<uint32_t>(m_words.size());
				m_words.push_back(
				    {.offset = offset, .stage = m_stage, .source = source, .word = word, .mask = mask});
				if (entry.words_begin == entry.words_end) {
					entry.words_begin = index;
				}
				entry.words_end = index + 1;
			}
			continue;
		}
		if (flat_slot != ShaderRecompiler::IR::NoFlatSlot) {
			if (!m_flat_loads.empty()) {
				auto&      last = m_flat_loads.back();
				const auto next = last.address + uint64_t {last.count} * sizeof(uint32_t);
				if (last.stage == m_stage && next == address && last.slot + last.count == flat_slot &&
				    (address & 0xfffu) != 0) {
					last.count++;
					continue;
				}
			}
			m_flat_loads.push_back({address, m_stage, flat_slot, 1});
			continue;
		}
		append(address, value, false);
	}
	m_bytes.resize(bytes_end);
}

#if defined(ANYPS5_KYTY)
static bool FlatWordsReadOnCpu(const ShaderRecompiler::IR::CompiledShaderInfo& program, uint32_t slot, uint32_t count) {
	const auto& info = program.info;
	if (info.ray_query || info.bvh || !info.ray_query_reach.empty() ||
	    program.bindings.ray_query_dword != ShaderRecompiler::IR::PushData::NoStart) {
		return true;
	}
	const auto within = [&](uint32_t word) { return word >= slot && word - slot < count; };
	for (const auto& [low, high]: info.address_bases) {
		if (within(low) || within(high)) return true;
	}
	using ShaderRecompiler::IR::BdaReadWord;
	for (const auto& span: program.bda_read_plan.spans) {
		if ((span.low.kind == BdaReadWord::Kind::Flat && within(span.low.value)) ||
		    (span.high.kind == BdaReadWord::Kind::Flat && within(span.high.value))) {
			return true;
		}
	}
	return false;
}

void GuestReadLog::NoteDeferredLoads(std::span<uint64_t> addresses) {
	std::ranges::sort(addresses);
	for (auto& load: m_flat_loads) {
		if (load.stage != m_stage || load.deferred) continue;
		const auto first = std::ranges::lower_bound(addresses, load.address);
		if (first == addresses.end() || *first - load.address >= uint64_t {load.count} * sizeof(uint32_t)) continue;
		load.deferred = true;
		m_deferred_words += load.count;
	}
}

void GuestReadLog::ReadGuestNow(uint64_t address, void* bytes, uint32_t size) {
	if (!Libs::LibKernel::Memory::MayFaultOnGpuRead(address, size) ||
	    !Libs::LibKernel::Memory::TryReadAroundGpuWrites(address, bytes, size)) {
		std::memcpy(bytes, reinterpret_cast<const void*>(address), size);
	}
}

void GuestReadLog::RefreshFlatLoads(std::span<ShaderRecompiler::IR::ResourceSnapshot> resources, bool gpu_copies_allowed,
                                    const ShaderRecompiler::IR::CompiledShaderInfo* program) const {
	m_deferred_gpu_words = 0;
#else
void GuestReadLog::RefreshFlatLoads(std::span<ShaderRecompiler::IR::ResourceSnapshot> resources,
                                    bool gpu_copies) const {
#endif
	for (auto& snapshot: resources) {
		snapshot.gpu_flat_loads.clear();
	}
	auto& checked = m_checked;
	for (const auto& load: m_flat_loads) {
		if (load.stage >= resources.size() ||
		    uint64_t {load.slot} + load.count > resources[load.stage].flattened_srt.size()) {
			continue;
		}
#if defined(ANYPS5_KYTY)
		const bool gpu_copies = gpu_copies_allowed &&
		                        (!load.deferred || (program != nullptr && !FlatWordsReadOnCpu(*program, load.slot, load.count)));
#endif
		const auto page_index = load.address >> 12u;
		auto&      page       = checked[page_index % checked.size()];
		if (page.page != page_index) {
			page.page    = page_index;
			page.owned   = Libs::LibKernel::Memory::MayFaultOnGpuRead(load.address & ~0xfffull, 4096);
			page.backing = page.owned ? Libs::LibKernel::Memory::FindAroundBackingPage(load.address)
			                          : nullptr;
		}
		auto*      values = resources[load.stage].flattened_srt.data() + load.slot;
		const auto bytes  = load.count * sizeof(uint32_t);
		if (gpu_copies && page.owned && page.backing != nullptr &&
		    Libs::LibKernel::Memory::HasGpuWrittenBytes(load.address, bytes)) {
			resources[load.stage].gpu_flat_loads.push_back({load.address, load.slot, load.count});
#if defined(ANYPS5_KYTY)
			if (load.deferred) m_deferred_gpu_words += load.count;
#endif
			continue;
		}
		const bool around =
		    page.owned &&
		    (page.backing != nullptr ? Libs::LibKernel::Memory::TryReadAroundGpuWritesOnPage(
		                                   load.address, values, bytes, page.backing)
		                             : Libs::LibKernel::Memory::TryReadAroundGpuWrites(
		                                   load.address, values, bytes));
		if (!around) {
			if (gpu_copies && page.owned && page.backing != nullptr) {
				resources[load.stage].gpu_flat_loads.push_back({load.address, load.slot, load.count});
#if defined(ANYPS5_KYTY)
				if (load.deferred) m_deferred_gpu_words += load.count;
#endif
				continue;
			}
			std::memcpy(values, reinterpret_cast<const void*>(load.address), bytes);
		}
	}
}

void GuestReadLog::ReadNow(const Entry& entry, uint8_t* bytes) const {
	if (!Libs::LibKernel::Memory::MayFaultOnGpuRead(entry.address, entry.size) ||
	    !Libs::LibKernel::Memory::TryReadAroundGpuWrites(entry.address, bytes, entry.size)) {
		std::memcpy(bytes, reinterpret_cast<const void*>(entry.address), entry.size);
	}
}

bool GuestReadLog::ZeroTestHolds(const Entry& entry) const {
	uint32_t previous = 0;
	uint32_t current  = 0;
	std::memcpy(&previous, m_bytes.data() + entry.offset, sizeof(previous));
	if (Libs::LibKernel::Memory::MayFaultOnGpuRead(entry.address, sizeof(current))) {
		LearnGpuProducedBound(entry.address);
	}
	ReadNow(entry, reinterpret_cast<uint8_t*>(&current));
	return (current != 0) == (previous != 0);
}

bool GuestReadLog::OnlyRefreshableBitsChanged(const Entry& entry) const {
	static thread_local std::vector<uint8_t> now;
	now.resize(entry.size);
	ReadNow(entry, now.data());
	const auto* recorded = m_bytes.data() + entry.offset;
	auto        word     = entry.words_begin;
	for (uint32_t done = 0; done < entry.size; done += sizeof(uint32_t)) {
		uint32_t current  = 0;
		uint32_t previous = 0;
		const auto bytes  = std::min<uint32_t>(entry.size - done, sizeof(uint32_t));
		std::memcpy(&current, now.data() + done, bytes);
		std::memcpy(&previous, recorded + done, bytes);
		while (word < entry.words_end && m_words[word].offset < entry.offset + done) {
			word++;
		}
		const auto mask = word < entry.words_end && m_words[word].offset == entry.offset + done
		                      ? m_words[word].mask
		                      : 0u;
		if (((current ^ previous) & ~mask) != 0) {
			return false;
		}
	}
	return true;
}

bool GuestReadLog::DwordAligned() const {
	return std::ranges::all_of(m_entries, [](const Entry& entry) {
		return entry.address % 4u == 0 && entry.size % 4u == 0 && entry.offset % 4u == 0;
	});
}

bool GuestReadLog::FirstMismatch(uint32_t& index, uint64_t& address, uint32_t& size, uint32_t& old_value,
                                 uint32_t& new_value) const {
	static thread_local std::vector<uint8_t> now;
	for (uint32_t i = 0; i < m_entries.size(); ++i) {
		const auto& entry = m_entries[i];
		const auto* recorded = m_bytes.data() + entry.offset;
		if (entry.zero_test) {
			if (!ZeroTestHolds(entry)) {
				index = i; address = entry.address; size = entry.size;
				std::memcpy(&old_value, recorded, 4);
				ReadNow(entry, reinterpret_cast<uint8_t*>(&new_value));
				return true;
			}
			continue;
		}
		now.resize(entry.size);
		if (entry.strict) {
			if (!Libs::LibKernel::Memory::TryReadGpuCleanBacking(entry.address, now.data(), entry.size)) now.assign(entry.size, 0xcd);
		} else {
			ReadNow(entry, now.data());
		}
		if (std::memcmp(now.data(), recorded, entry.size) == 0) continue;
		if (entry.words_begin != entry.words_end && OnlyRefreshableBitsChanged(entry)) continue;
		index = i; address = entry.address; size = entry.size;
		for (uint32_t at = 0; at + 4 <= entry.size; at += 4) {
			std::memcpy(&old_value, recorded + at, 4);
			std::memcpy(&new_value, now.data() + at, 4);
			if (old_value != new_value) break;
		}
		return true;
	}
	return false;
}

uint32_t GuestReadLog::WordAt(size_t index) const {
	uint32_t value = 0;
	std::memcpy(&value, m_bytes.data() + index * 4u, 4);
	return value;
}

void GuestReadLog::SetWord(size_t index, uint32_t value) {
	std::memcpy(m_bytes.data() + index * 4u, &value, 4);
}

bool GuestReadLog::SameReads(const GuestReadLog& other) const {
	return m_entries.size() == other.m_entries.size() &&
	       std::ranges::equal(m_entries, other.m_entries, [](const Entry& a, const Entry& b) {
		       return a.address == b.address && a.size == b.size && a.offset == b.offset &&
		              a.strict == b.strict && a.zero_test == b.zero_test;
	       });
}

bool GuestReadLog::CollectChanges(std::vector<std::pair<uint32_t, uint32_t>>& changed) const {
	if (m_failed) {
		return false;
	}
	static thread_local std::vector<uint8_t> current;
	for (const auto& entry: m_entries) {
		if (entry.address % 4u != 0 || entry.size % 4u != 0 || entry.offset % 4u != 0) {
			return false;
		}
		const auto* recorded = m_bytes.data() + entry.offset;
		current.resize(entry.size);
		if (entry.strict) {
			if (!Libs::LibKernel::Memory::TryReadGpuCleanBacking(entry.address, current.data(),
			                                                     entry.size)) {
				return false;
			}
		} else {
			ReadNow(entry, current.data());
		}
		if (std::memcmp(current.data(), recorded, entry.size) == 0) {
			continue;
		}
		for (uint32_t at = 0; at < entry.size; at += 4u) {
			uint32_t now = 0, before = 0;
			std::memcpy(&now, current.data() + at, 4);
			std::memcpy(&before, recorded + at, 4);
			if (now != before) {
				changed.emplace_back((entry.offset + at) / 4u, now);
			}
		}
	}
	return true;
}

void GuestReadLog::RefreshDescriptorWords(
    std::span<ShaderRecompiler::IR::ResourceSnapshot>          resources,
    std::span<const ShaderRecompiler::IR::ResourcePlan* const> plans) const {
	using ShaderRecompiler::IR::DescriptorRefresh;
	static thread_local std::vector<uint8_t> now;
	for (const auto index: m_refresh) {
		const auto& entry = m_entries[index];
		now.resize(entry.size);
		ReadNow(entry, now.data());
		for (uint32_t w = entry.words_begin; w < entry.words_end; w++) {
			const auto& load = m_words[w];
			uint32_t    value    = 0;
			uint32_t    previous = 0;
			std::memcpy(&value, now.data() + (load.offset - entry.offset), sizeof(value));
			std::memcpy(&previous, m_bytes.data() + load.offset, sizeof(previous));
			if (value == previous || load.stage >= resources.size() || load.stage >= plans.size() ||
			    plans[load.stage] == nullptr) {
				continue;
			}
			const auto& plan     = *plans[load.stage];
			auto&       snapshot = resources[load.stage];
			const auto flat_word = uint64_t {load.source} * 8u + load.word;
			if (flat_word < plan.descriptor_word_flat.size() &&
			    plan.descriptor_word_flat[flat_word] < snapshot.flattened_srt.size()) {
				snapshot.flattened_srt[plan.descriptor_word_flat[flat_word]] = value;
			}
			const auto refresh = [&](ShaderRecompiler::IR::DescriptorValue& descriptor) {
				if (load.word >= descriptor.dword_count ||
				    std::ranges::all_of(descriptor.dwords, [](uint32_t dword) { return dword == 0; })) {
					return;
				}
				descriptor.dwords[load.word] =
				    (descriptor.dwords[load.word] & ~load.mask) | (value & load.mask);
			};
			const auto kind = load.source < plan.descriptor_refresh.size()
			                      ? plan.descriptor_refresh[load.source]
			                      : DescriptorRefresh::None;
			if (kind == DescriptorRefresh::Buffer) {
				for (uint32_t k = 0; k < plan.info.buffers.size() && k < snapshot.buffers.size(); ++k) {
					if (plan.info.buffers[k].source == load.source) refresh(snapshot.buffers[k]);
				}
			} else if (kind == DescriptorRefresh::Image) {
				for (uint32_t k = 0; k < plan.info.images.size() && k < snapshot.images.size(); ++k) {
					if (plan.info.images[k].source == load.source) refresh(snapshot.images[k]);
				}
			}
		}
	}
	m_refresh.clear();
}

static bool SameGuestBytes(const void* guest, const uint8_t* recorded, uint32_t size) {
	const auto load = [](const void* address, uint32_t offset, auto value) {
		std::memcpy(&value, static_cast<const uint8_t*>(address) + offset, sizeof(value));
		return value;
	};
	switch (size) {
		case 4: return load(guest, 0, uint32_t {}) == load(recorded, 0, uint32_t {});
		case 8: return load(guest, 0, uint64_t {}) == load(recorded, 0, uint64_t {});
		case 16:
			return ((load(guest, 0, uint64_t {}) ^ load(recorded, 0, uint64_t {})) |
			        (load(guest, 8, uint64_t {}) ^ load(recorded, 8, uint64_t {}))) == 0;
		case 32:
			return ((load(guest, 0, uint64_t {}) ^ load(recorded, 0, uint64_t {})) |
			        (load(guest, 8, uint64_t {}) ^ load(recorded, 8, uint64_t {})) |
			        (load(guest, 16, uint64_t {}) ^ load(recorded, 16, uint64_t {})) |
			        (load(guest, 24, uint64_t {}) ^ load(recorded, 24, uint64_t {}))) == 0;
		default: return std::memcmp(guest, recorded, size) == 0;
	}
}

bool GuestReadLog::StillValid() const {
	m_refresh.clear();
	if (m_failed) {
		return false;
	}
	std::array<uint8_t, StrictCheckBytes> buffer {};
	auto& checked = m_checked;
	checked.fill({});
	const auto refreshable = [&](const Entry& entry) {
		if (entry.words_begin == entry.words_end || !OnlyRefreshableBitsChanged(entry)) {
			return false;
		}
		m_refresh.push_back(static_cast<uint32_t>(&entry - m_entries.data()));
		return true;
	};
	for (const auto& entry: m_entries) {
		const auto* recorded = m_bytes.data() + entry.offset;
		if (entry.zero_test) {
			if (!ZeroTestHolds(entry)) {
				return false;
			}
			continue;
		}
		if (!entry.strict) {
			const auto  first_page   = entry.address >> 12u;
			const auto  last_page    = (entry.address + entry.size - 1) >> 12u;
			bool        gpu_owned    = false;
			const auto* page_backing = static_cast<const uint8_t*>(nullptr);
			if (first_page == last_page) {
				auto& page = checked[first_page % checked.size()];
				if (page.page != first_page) {
					page.page    = first_page;
					page.owned   = Libs::LibKernel::Memory::MayFaultOnGpuRead(entry.address, entry.size);
					page.backing = page.owned
					                   ? Libs::LibKernel::Memory::FindAroundBackingPage(entry.address)
					                   : nullptr;
				}
				gpu_owned    = page.owned;
				page_backing = page.backing;
			} else {
				gpu_owned = Libs::LibKernel::Memory::MayFaultOnGpuRead(entry.address, entry.size);
			}
			bool around = gpu_owned;
			for (uint32_t done = 0; around && done < entry.size; done += sizeof(buffer)) {
				const auto bytes = std::min<uint32_t>(entry.size - done, sizeof(buffer));
				around           = page_backing != nullptr
				                       ? Libs::LibKernel::Memory::TryReadAroundGpuWritesOnPage(
				                             entry.address + done, buffer.data(), bytes, page_backing)
				                       : Libs::LibKernel::Memory::TryReadAroundGpuWrites(
				                             entry.address + done, buffer.data(), bytes);
				if (around && std::memcmp(buffer.data(), recorded + done, bytes) != 0) {
					if (!refreshable(entry)) {
						return false;
					}
					break;
				}
			}
			if (around) {
				continue;
			}
			if (!SameGuestBytes(reinterpret_cast<const void*>(entry.address), recorded,
			                    entry.size) &&
			    !refreshable(entry)) {
				return false;
			}
			continue;
		}
		if (!Libs::LibKernel::Memory::GpuCleanBackingEqualsSince(entry.address, recorded, entry.size, entry.watched,
		                                                        ParallelEqual)) {
			return false;
		}
	}
	return true;
}

#if !defined(ANYPS5_KYTY)
bool GraphicsPipelineKey::operator==(const GraphicsPipelineKey& other) const {
	return std::memcmp(this, &other, sizeof(*this)) == 0;
}
#endif

std::size_t PipelineCache::GraphicsPipelineKeyHash::operator()(const GraphicsPipelineKey& key) const {
	return XXH3_64bits(&key, sizeof(key));
}

std::span<const uint32_t> PipelineCache::Program::FunctionCode() const {
	return call_source
	    ? std::span<const uint32_t>(call_source->linked->code).subspan(call_source->code.size())
	    : std::span<const uint32_t>{};
}

#if defined(ANYPS5_KYTY)
bool PipelineCache::Program::Failed(const ShaderRecompiler::IR::ResourceSpecialization& candidate,
                                    uint32_t push_data_cursor) const {
	return std::ranges::any_of(failed_variants, [&](const FailedVariant& variant) {
		return variant.push_data_cursor == push_data_cursor && variant.specialization == candidate &&
		       std::ranges::equal(variant.function_code, FunctionCode());
	});
}

void PipelineCache::Program::NoteFailed(uint32_t push_data_cursor) {
	const auto code = FunctionCode();
	failed_variants.push_back({.specialization   = specialization,
	                           .push_data_cursor = push_data_cursor,
	                           .function_code    = {code.begin(), code.end()}});
}

ShaderModule* PipelineCache::Program::ShareModule(const ShaderRecompiler::IR::ResourceSpecialization& requested,
                                                  const ShaderRecompiler::IR::ResourceSnapshot&       snapshot,
                                                  uint32_t push_data_cursor, bool calls) {
	if (!ShaderRecompiler::IR::SharesEmptyBuffers()) return nullptr;
	const auto code = FunctionCode();
	for (const auto& candidate: modules) {
		const auto& layout = candidate.info.bindings;
		if (candidate.shared_variant ||
		    layout.push_data_start_dword !=
		        ShaderRecompiler::IR::PushData::StartFor(push_data_cursor, layout.ShaderDataDwords()) ||
		    (calls && !std::ranges::equal(candidate.function_code, code)) ||
		    !ShaderRecompiler::IR::EmptyBuffersServe(resource_plan, snapshot, requested, candidate.specialization)) {
			continue;
		}
		auto& shared          = modules.emplace_back(candidate);
		shared.specialization = requested;
		shared.shared_variant = true;
		static const bool logged = [] {
			const char* value = std::getenv("KYTY_SHADER_VARIANT_LOG");
			return value != nullptr && std::strcmp(value, "1") == 0;
		}();
		if (logged) {
			PipelineCacheLog("[kyty-variant] share stage={} shader={:016x} module={:016x} @{}",
			                 ShaderRecompiler::StageName(resource_plan.stage), resource_plan.shader_hash, shared.id,
			                 std::chrono::steady_clock::now().time_since_epoch().count());
		}
		return &shared;
	}
	return nullptr;
}

void PipelineCache::Program::PromoteEmptyBuffers(const ShaderRecompiler::IR::ResourceSnapshot& snapshot,
                                                 ShaderRecompiler::IR::ResourceSpecialization& specialization,
                                                 uint32_t push_data_cursor) const {
	if (!ShaderRecompiler::IR::SharesEmptyBuffers()) return;
	auto promoted = specialization;
	bool changed  = false;
	for (const auto& module: modules) {
		const auto& candidate = module.specialization.buffers;
		if (module.shared_variant || candidate.size() != promoted.buffers.size()) continue;
		for (uint32_t i = 0; i < candidate.size(); i++) {
			const auto& form = candidate[i];
			if (form.zero_stride_oob || form.scalar_empty || form == specialization.buffers[i] ||
			    promoted.buffers[i] != specialization.buffers[i] ||
			    !ShaderRecompiler::IR::EmptyBufferServes(resource_plan, snapshot, i, specialization.buffers[i],
			                                             form)) {
				continue;
			}
			promoted.buffers[i] = form;
			changed             = true;
		}
	}
	if (changed && !Failed(promoted, push_data_cursor)) {
		specialization = std::move(promoted);
	}
}
#endif

bool PipelineCache::Program::RefreshCallSource(const ShaderRecompiler::IR::SrtRuntime& runtime) {
	const auto revision = call_source->revision;
	ShaderRecompiler::RefreshShaderSource(*call_source, runtime);
	resource_plan.source_reads = call_source->reads;
	return revision != call_source->revision;
}

template <bool HasCalls>
ShaderModule* PipelineCache::Program::FindModule(const ShaderRecompiler::IR::SrtRuntime& runtime,
                                                 uint32_t push_data_cursor, bool& failed) {
	failed = !ShaderRecompiler::IR::MaterializeResources(resource_plan, runtime, resources,
	                                                     specialization);
	if (failed) {
		return nullptr;
	}
#if defined(ANYPS5_KYTY)
	return MatchModule<HasCalls>(push_data_cursor);
}

template <bool HasCalls>
ShaderModule* PipelineCache::Program::MatchModule(uint32_t push_data_cursor) {
#endif
	const auto matches = [&](const ShaderModule& candidate) {
		const auto& layout = candidate.info.bindings;
		if (layout.push_data_start_dword != ShaderRecompiler::IR::PushData::StartFor(
		        push_data_cursor, layout.ShaderDataDwords()) ||
		    candidate.specialization != specialization) {
			return false;
		}
		if constexpr (HasCalls) {
			if (!std::ranges::equal(candidate.function_code, FunctionCode())) return false;
		}
		return true;
	};
	auto& last = last_module[GuestGpu::IsGpuThread() ? 0u : 1u];
	if (last < modules.size() && matches(modules[last])) {
		return &modules[last];
	}
	const auto found = std::ranges::find_if(modules, matches);
	if (found == modules.end()) {
#if defined(ANYPS5_KYTY)
		if (auto* shared = ShareModule(specialization, resources, push_data_cursor, HasCalls)) {
			last = modules.size() - 1;
			return shared;
		}
#endif
		return nullptr;
	}
	last = static_cast<size_t>(found - modules.begin());
	return &*found;
}

ShaderModule PipelineCache::BuildModule(const ProgramKey&                                  key,
                                       const ShaderRecompiler::CompileOptions&            options,
                                       ShaderRecompiler::TranslateResult                  translated,
                                       const ShaderRecompiler::IR::ResourceSpecialization& specialization,
                                       std::span<const uint32_t> function_code,
                                       uint32_t push_data_cursor, std::vector<uint32_t>& spirv,
                                       std::vector<uint32_t>& plain_spirv,
                                       std::string* table_refusal) const {
	Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::ShadersCompiled,
	                                         Common::DebugCounters::Counter::ShaderCompileNs,
	                                         Common::DebugCounters::Peak::ShaderCompileNs);
	const auto profile         = SpirvProfile(m_graphics);
	auto       compile_options = options;
	compile_options.profile    = &profile;
	compile_options.plain_mip_stats_variant = key.stage == ShaderType::Pixel;
	auto result = ShaderRecompiler::CompileProgram(std::move(translated), compile_options,
	                                               specialization, push_data_cursor);
	if (compile_options.table_mode && result.spirv.empty()) {
		if (table_refusal != nullptr) *table_refusal = std::move(result.table_refusal);
		return {};
	}
	auto info = std::move(result.program).TakeCompiledInfo();

	ShaderRecompiler::IR::Writer inputs;
	ShaderRecompiler::IR::IoAll(inputs, key.stage, key.hash, key.user_data_count, key.static_state,
	                            info.bindings.push_data_start_dword, specialization);
	inputs.Raw(function_code.data(), function_code.size_bytes());
	const auto id = XXH3_64bits(inputs.bytes.data(), inputs.bytes.size());

	const auto* stage = ShaderRecompiler::StageName(key.stage);
	DumpShader(result.spirv, fmt::format("{}_{:016x}_{:016x}.spv", stage, key.hash, id));
	ValidateShaderSpirv(stage, key.hash, result.spirv, profile.maintenance8);
	ShaderModule module {
	    .id             = id,
	    .module         = CompileSPV(result.spirv, m_graphics.device),
	    .info           = std::move(info),
	    .specialization = specialization,
	    .function_code  = {function_code.begin(), function_code.end()},
	};
	module.spirv_words = static_cast<uint32_t>(result.spirv.size());
	if (key.stage == ShaderType::Vertex || key.stage == ShaderType::Local) {
		module.vertex_formats = VertexInputFormats(*options.input_info.vertex, module.info);
	} else if (key.stage == ShaderType::Pixel &&
	           options.input_info.pixel->parameter_mode == ShaderPixelParameterMode::Rectangle) {
		module.rect_parameters = RectListParameters(*options.input_info.pixel, module.info);
	} else if (key.stage == ShaderType::Mesh) {
		module.mesh_task = options.input_info.vertex->indirect_draw;
	} else if (key.stage == ShaderType::Compute) {
		module.guest_traversal = RayTracing::NativeTraversals() && !options.native_ray_tracing;
	}
	if (!result.spirv_plain.empty()) {
		ValidateShaderSpirv(stage, key.hash, result.spirv_plain, profile.maintenance8);
		module.plain = MakePlainModule(module, result.spirv_plain, m_graphics.device);
		EXIT_IF(module.plain == nullptr);
	}
#if defined(ANYPS5_KYTY)
	LogShaderVariantBuild(stage, key.hash, key.user_data_count, key.static_state, push_data_cursor, module);
#endif
	spirv       = std::move(result.spirv);
	plain_spirv = std::move(result.spirv_plain);
	return module;
}

std::shared_ptr<const ShaderModule> MakePlainModule(const ShaderModule&       module,
                                                    std::span<const uint32_t> spirv,
                                                    vk::Device                device) {
	auto plain = std::make_shared<ShaderModule>();
	plain->id     = module.id ^ PlainModuleIdXor;
	plain->module = CompileSPV(spirv, device);
	if (plain->module == nullptr) {
		return nullptr;
	}
	plain->info            = module.info;
	plain->specialization  = module.specialization;
	plain->function_code   = module.function_code;
	plain->vertex_formats  = module.vertex_formats;
	plain->rect_parameters = module.rect_parameters;
	plain->spirv_words     = static_cast<uint32_t>(spirv.size());
	return plain;
}

void PipelineCache::AddedModule(const ProgramKey& key, const Program& program,
                                const ShaderModule& module, const std::vector<uint32_t>& spirv,
                                const std::vector<uint32_t>&            plain_spirv,
                                const ShaderRecompiler::CompileOptions& options,
                                std::span<const uint32_t> code, uint32_t push_data_cursor) {
	m_compiled[static_cast<size_t>(key.stage)]++;
#if defined(ANYPS5_KYTY)
	if (ProgramLock::Narrow()) {
		ProgramLock::DeferLine(fmt::format(
		    "Shaders: VS {} | PS {} | CS {} | GS {} | LS {} | HS {} | TES {}\n",
		    m_compiled[static_cast<size_t>(ShaderType::Vertex)], m_compiled[static_cast<size_t>(ShaderType::Pixel)],
		    m_compiled[static_cast<size_t>(ShaderType::Compute)], m_compiled[static_cast<size_t>(ShaderType::Mesh)],
		    m_compiled[static_cast<size_t>(ShaderType::Local)],
		    m_compiled[static_cast<size_t>(ShaderType::TessellationControl)],
		    m_compiled[static_cast<size_t>(ShaderType::TessellationEvaluation)]));
	} else
#endif
	std::printf("Shaders: VS %u | PS %u | CS %u | GS %u | LS %u | HS %u | TES %u\n",
	            m_compiled[static_cast<size_t>(ShaderType::Vertex)],
	            m_compiled[static_cast<size_t>(ShaderType::Pixel)],
	            m_compiled[static_cast<size_t>(ShaderType::Compute)],
	            m_compiled[static_cast<size_t>(ShaderType::Mesh)],
	            m_compiled[static_cast<size_t>(ShaderType::Local)],
	            m_compiled[static_cast<size_t>(ShaderType::TessellationControl)],
	            m_compiled[static_cast<size_t>(ShaderType::TessellationEvaluation)]);
	if (program.call_source == nullptr) {
		StoreModule(key, program, module, spirv, plain_spirv);
		RecordModule(key, options, code, push_data_cursor, module);
	}
}

void PipelineCache::DestroyModule(const ShaderModule& module) const {
	m_graphics.device.destroyShaderModule(module.module, nullptr);
	if (module.plain != nullptr) {
		m_graphics.device.destroyShaderModule(module.plain->module, nullptr);
	}
}

ShaderModule& PipelineCache::CompileModule(Program& program, const ProgramKey& key,
                                           ShaderRecompiler::CompileOptions  options,
                                           ShaderRecompiler::TranslateResult translated,
                                           uint32_t                          push_data_cursor,
                                           std::span<const uint32_t>         code) {
	std::vector<uint32_t> spirv;
	std::vector<uint32_t> plain_spirv;
	auto& module = program.modules.emplace_back(
	    BuildModule(key, options, std::move(translated), program.specialization,
	                program.FunctionCode(), push_data_cursor, spirv, plain_spirv));
	module.plan = &program.resource_plan;
	AddedModule(key, program, module, spirv, plain_spirv, options, code, push_data_cursor);
	return module;
}

template <typename InputInfo>
const ShaderModule* PipelineCache::GetProgram(const ShaderParams& params, InputInfo& input_info,
                                              uint32_t& push_data_cursor) {
	const auto user_data = std::span(params.user_data).first(params.user_data_count);
	ShaderRecompiler::IR::SrtRuntime runtime {
	    .user_data                  = user_data,
	    .shader_base                = params.Base(),
	    .read_specialization_memory = ReadShaderGuestMemory,
	    .image_heap                 = m_image_heap,
#if defined(ANYPS5_KYTY)
	    .image_table_heap           = Config::ImageTableHeapEnabled(),
#endif
	};
	ShaderRecompiler::CompileOptions options {.wave_size = input_info.wave_size};
	if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
		options.stage            = input_info.logical_stage;
		options.input_info.vertex = &input_info;
		options.user_data_base   = 8;
		if (options.stage == ShaderType::Mesh ||
		    options.stage == ShaderType::TessellationControl) {
			options.user_data_base = 0;
		}
		if (options.stage == ShaderType::Mesh) {
			options.wave_size = input_info.mesh.wave_size;
		}
	} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
		options.stage            = ShaderType::Pixel;
		options.input_info.pixel = &input_info;
	} else {
		static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
		options.stage              = ShaderType::Compute;
		options.input_info.compute = &input_info;
		runtime.workgroup_counts   = input_info.workgroup_counts;
		runtime.indirect_args_address = input_info.indirect_args_address;
		runtime.runtime_stride        = RuntimeStride(params.hash);
	}
	m_lookup_key.stage           = options.stage;
	m_lookup_key.hash            = params.hash;
	m_lookup_key.user_data_count = params.user_data_count;
	BuildStageStaticKey(input_info, m_lookup_key.static_state);
	AppendNativeRayTracingKey(options.stage, m_lookup_key.static_state, !m_guest_traversal);
	auto entry = m_program_cache.find(m_lookup_key);
	const auto bind = [&](const ShaderModule& module) {
		input_info.stage = {.module = &module, .resources = &entry->second.resources};
		module.info.bindings.AdvancePushData(push_data_cursor);
		return &module;
	};
	bool       failed       = false;
	const auto skip_failure = [&] {
		Common::HackNotice::Report("draw or dispatch with unresolvable resources skipped",
		                           "PipelineCache::GetProgram", params.hash, 0);
		return nullptr;
	};
	bool source_changed = false;
	if (entry != m_program_cache.end()) {
		auto& cached = entry->second;
		if (cached.skip_dispatch) return nullptr;
		ShaderModule* module = nullptr;
#if defined(ANYPS5_KYTY)
		bool specialized = false;
		if (t_execute_unlocked) {
			const auto& key          = entry->first;
			bool        materialized = true;
			UnlockPrograms();
			if (cached.call_source) {
				source_changed     = cached.RefreshCallSource(runtime);
				runtime.call_words = cached.call_source->call_words;
				materialized       = !source_changed && !cached.call_source->refresh_failed;
			}
			if (materialized) {
				failed = !ShaderRecompiler::IR::MaterializeResources(cached.resource_plan, runtime, cached.resources,
				                                                     cached.specialization);
			}
			LockPrograms();
			entry = m_program_cache.find(key);
			if (cached.call_source && cached.call_source->refresh_failed) {
				SkipCallExpansionFailure(params.hash, *cached.call_source);
				cached.skip_dispatch = true;
				return nullptr;
			}
			if (materialized && !failed) {
				module = cached.call_source ? cached.MatchModule<true>(push_data_cursor)
				                            : cached.MatchModule<false>(push_data_cursor);
				specialized = true;
			}
			if (!failed && module == nullptr) {
				m_lookup_key = key;
			}
		} else
#endif
		if (cached.call_source) {
			source_changed = cached.RefreshCallSource(runtime);
			if (cached.call_source->refresh_failed) {
				SkipCallExpansionFailure(params.hash, *cached.call_source);
				cached.skip_dispatch = true;
				return nullptr;
			}
			runtime.call_words = cached.call_source->call_words;
			if (!source_changed) module = cached.FindModule<true>(runtime, push_data_cursor, failed);
#if defined(ANYPS5_KYTY)
			specialized = !source_changed;
#endif
		} else {
			module = cached.FindModule<false>(runtime, push_data_cursor, failed);
#if defined(ANYPS5_KYTY)
			specialized = true;
#endif
		}
		if (failed) return skip_failure();
		if (module != nullptr) return bind(*module);
#if defined(ANYPS5_KYTY)
		if (specialized && !cached.failed_variants.empty() &&
		    cached.Failed(cached.specialization, push_data_cursor)) {
			return nullptr;
		}
#endif
	}
#if defined(ANYPS5_KYTY)
	struct TranslateScope {
		uint64_t                              hash;
		std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
		~TranslateScope() {
			static const auto threshold = std::chrono::milliseconds([] {
				const char* value = std::getenv("KYTY_PIPELINE_STALL_LOG_MS");
				return value != nullptr ? std::strtoll(value, nullptr, 10) : 100;
			}());
			const auto now = std::chrono::steady_clock::now();
			if (now - started >= threshold) {
				PipelineCacheLog("[kyty-gpl] translate {:.3f} s 0x{:016x} @{}",
				                 std::chrono::duration<double>(now - started).count(), hash,
				                 now.time_since_epoch().count());
			}
		}
	} const translate_scope {params.hash};
#endif
	if (m_queue_only) {
		if (!m_in_flight.contains(m_lookup_key)) {
#if defined(ANYPS5_KYTY)
			std::optional<ShaderRecompiler::IR::ResourceSpecialization> promoted;
			if (entry != m_program_cache.end()) {
				promoted = entry->second.specialization;
				entry->second.PromoteEmptyBuffers(entry->second.resources, *promoted, push_data_cursor);
			}
			QueueCompile(m_lookup_key, params, input_info, push_data_cursor,
			             promoted ? &*promoted : nullptr, m_guest_traversal);
#else
			QueueCompile(m_lookup_key, params, input_info, push_data_cursor,
			             entry != m_program_cache.end() ? &entry->second.specialization : nullptr, m_guest_traversal);
#endif
			m_queued = true;
		}
		return nullptr;
	}
	if (WaitForCompile(m_lookup_key)) {
		return GetProgram(params, input_info, push_data_cursor);
	}

#if defined(ANYPS5_KYTY)
	const auto build = [&]() -> const ShaderModule* {
#endif
	options.shader_hash     = params.hash;
	options.user_data       = user_data;
	options.back_code       = params.back_code;
	options.dump_ir         = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
	options.emit_debug_info = Config::GraphicsDebugDumpEnabled();
	options.native_ray_tracing = RayTracing::NativeTraversals() && !m_guest_traversal;
	options.ray_stats          = options.native_ray_tracing && Config::RayTracingStatsEnabled();
	options.rt_oracle          = options.native_ray_tracing ? Config::RayTracingOracle() : 0u;
	ShaderRecompiler::ShaderSource source;
	if (entry == m_program_cache.end()) {
		DumpShader(params.code, fmt::format("original/{}_{:016x}.bin",
		                                    ShaderRecompiler::StageName(options.stage), params.hash));
		source = ShaderRecompiler::PrepareShaderSource(params.code, options);
		if (source.unsupported_call_pc != UINT32_MAX && !source.expand_calls) {
			PipelineCacheLog("!!! TEMPORARY COMPUTE DISPATCH SKIP !!! shader=0x{:016x} pc=0x{:08x} "
			                 "reason=unsupported GPU-selected callable target; rendering is incomplete",
			                 params.hash, source.unsupported_call_pc);
			m_program_cache.emplace(m_lookup_key, Program {.skip_dispatch = true});
			return nullptr;
		}
	}
	auto* current = entry == m_program_cache.end() ? &source : entry->second.call_source.get();
	ShaderRecompiler::TranslateResult translated;
	if (current != nullptr) {
		if (current->HasCalls()) {
			if (entry == m_program_cache.end()) {
				ShaderRecompiler::RefreshShaderSource(*current, runtime);
				if (current->refresh_failed) {
					SkipCallExpansionFailure(params.hash, *current);
					m_program_cache.emplace(m_lookup_key, Program {.skip_dispatch = true});
					return nullptr;
				}
				runtime.call_words = current->call_words;
			}
			translated = ShaderRecompiler::TranslateProgram(current->linked->decoded, options);
			translated.program.source_reads = current->reads;
		} else {
			translated = ShaderRecompiler::TranslateProgram(current->decoded, options);
		}
	} else {
		translated = ShaderRecompiler::TranslateProgram(params.code, options);
	}
	if (entry == m_program_cache.end()) {
		auto retained = source.HasCalls()
		                    ? std::make_unique<ShaderRecompiler::ShaderSource>(std::move(source))
		                    : nullptr;
		entry = m_program_cache.emplace(m_lookup_key, Program {
		    .call_source   = std::move(retained),
		    .resource_plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program),
		}).first;
		auto& program = entry->second;
		if (!ShaderRecompiler::IR::MaterializeResources(program.resource_plan, runtime,
		                                                program.resources, program.specialization)) {
			return skip_failure();
		}
	} else if (source_changed) {
		entry->second.resource_plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
		if (const auto* module = entry->second.FindModule<true>(runtime, push_data_cursor, failed)) {
			return bind(*module);
		}
		if (failed) return skip_failure();
	}
#if defined(ANYPS5_KYTY)
	entry->second.PromoteEmptyBuffers(entry->second.resources, entry->second.specialization, push_data_cursor);
#endif
	return bind(CompileModule(entry->second, entry->first, options, std::move(translated),
	                          push_data_cursor, params.code));
#if defined(ANYPS5_KYTY)
	};
	const ShaderModule* built = nullptr;
	if (Common::Fallback::Run("draw or dispatch skipped: the shader recompiler stopped",
	                          "PipelineCache::GetProgram", params.hash, 0, [&] { built = build(); })) {
		return built;
	}
	if (const auto found = m_program_cache.find(m_lookup_key); found != m_program_cache.end()) {
		if (found->second.modules.empty()) {
			found->second.skip_dispatch = true;
		} else {
			found->second.NoteFailed(push_data_cursor);
		}
	} else {
		m_program_cache.emplace(m_lookup_key, Program {.skip_dispatch = true});
	}
	return nullptr;
#endif
}

PipelineCache::PipelineCache(GraphicContext&                         graphics,
                             ShaderRecompiler::IR::ImageHeapSource* image_heap,
                             const std::filesystem::path&           shader_store)
    : m_graphics(graphics), m_image_heap(image_heap),
      m_next_driver_cache_save(std::chrono::steady_clock::now() + DriverCacheSaveInterval) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	const bool driver_cache_restored = InitializeDriverCache();
#if defined(ANYPS5_KYTY)
	if (m_graphics.pipeline_library_enabled) {
		m_libraries = std::make_unique<PipelineLibraries>(
		    m_graphics, m_driver_cache,
		    [this](std::chrono::steady_clock::time_point started) { NoteCreatedPipeline(started); });
	}
#endif
	if (Config::PipelineManifestRecord() && !shader_store.empty()) {
		PipelineCacheLog("Shader cache: not used while the pipeline manifest records");
		WarmUp({}, driver_cache_restored);
	} else {
		WarmUp(shader_store, driver_cache_restored);
	}
	StartHelpers();
	OpenManifest();
	for (uint32_t i = 0; i < CompileWorkers; i++) {
		m_workers.emplace_back([this] { CompileWorker(); });
	}
#if defined(ANYPS5_KYTY)
	if (m_libraries != nullptr && PipelineLibrariesAhead() && PipelineChainWanted()) {
		for (uint32_t i = 0; i < PipelineChainThreads(); i++) {
			m_chain_threads.emplace_back([this] { ChainWorker(); });
		}
		PipelineCacheLog("[kyty-gpl] pipeline chains on: {} threads (KYTY_PIPELINE_CHAIN=0 to turn off)",
		                 m_chain_threads.size());
	}
#endif
	if (Config::AsyncPipelinesEnabled()) {
		constexpr uint32_t AsyncThreads = 3;
		for (uint32_t i = 0; i < AsyncThreads; i++) {
			m_async_threads.emplace_back([this] { AsyncPipelineWorker(); });
		}
	}
	m_async_handoff = !m_async_threads.empty() || m_manifest_warm || m_helpers != nullptr;
	StartPrecompile();
}

PipelineCache::~PipelineCache() {
	m_precompile = {};
#if defined(ANYPS5_KYTY)
	StopChains();
#endif
	StopWorkers();
	CloseManifest();
	StopHelpers();
	StopAsyncPipelines();
#if defined(ANYPS5_KYTY)
	if (m_libraries != nullptr) {
		m_libraries->Stop();
		for (const auto& optimized: m_libraries->TakeOptimized()) {
			m_graphics.device.destroyPipeline(optimized.pipeline, nullptr);
		}
	}
	for (const auto& optimized: m_optimized_deferred) {
		m_graphics.device.destroyPipeline(optimized.pipeline, nullptr);
	}
	for (const auto retired: m_retired_pipelines) {
		m_graphics.device.destroyPipeline(retired, nullptr);
	}
#endif
	Save();
	for (const auto cache: m_helper_caches) {
		m_graphics.device.destroyPipelineCache(cache, nullptr);
	}
	m_graphics.device.destroyPipelineCache(m_helper_cache, nullptr);
	auto destroy = [this](const auto& pipelines) {
		for (const auto& pipeline: pipelines | std::views::values) {
			m_graphics.device.destroyPipeline(pipeline.pipeline, nullptr);
			m_graphics.device.destroyPipelineLayout(pipeline.pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout, nullptr);
		}
	};
	destroy(m_graphics_pipelines);
	destroy(m_native_graphics_pipelines);
	destroy(m_compute_pipelines);
	destroy(m_async_ready);
	destroy(m_async_ready_compute);
	for (const auto& native: m_async_native | std::views::values) {
		m_graphics.device.destroyPipeline(native.pipeline.pipeline, nullptr);
		m_graphics.device.destroyPipelineLayout(native.pipeline.pipeline_layout, nullptr);
		m_graphics.device.destroyDescriptorSetLayout(native.pipeline.descriptor_set_layout, nullptr);
	}
	for (const auto& program: m_program_cache | std::views::values) {
		for (const auto& module: program.modules) {
#if defined(ANYPS5_KYTY)
			if (module.shared_variant) continue;
#endif
			DestroyModule(module);
		}
	}
	for (const auto& variants: m_table_programs | std::views::values) {
		for (const auto& variant: variants) {
			if (variant.module) {
				DestroyModule(*variant.module);
			}
		}
	}
#if defined(ANYPS5_KYTY)
	m_libraries.reset();
#endif
	m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
}

bool PipelineCache::InitializeDriverCache() {
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return false;
	}
	if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (non-Release build)");
		return false;
	}
#if defined(ANYPS5_KYTY)
	if (m_graphics.pipeline_cache_persistent) {
		m_driver_cache_path = DriverCachePath(title_id, DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties()));
	}
#else
	m_driver_cache_path     = std::filesystem::path("_PipelineCache") / (title_id + ".bin");
#endif
	const auto path         = Common::PathToString(m_driver_cache_path);
	const bool cache_exists = !m_driver_cache_path.empty() && Common::File::IsFileExisting(m_driver_cache_path);
	if (cache_exists) {
		PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	} else if (!m_driver_cache_path.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: initializing {}", path);
	}
	std::vector<uint8_t> initial_data;
	if (cache_exists) {
		Common::File file(m_driver_cache_path, Common::File::Mode::Read);
		const auto   file_size = file.IsInvalid() ? 0 : file.Size();
		const auto   signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
		if (file_size >= signature.size() + sizeof(uint64_t) &&
		    file_size <= std::numeric_limits<uint32_t>::max()) {
			std::string cached_signature(signature.size(), '\0');
			uint64_t    payload_hash = 0;
			initial_data.resize(file_size - signature.size() - sizeof(payload_hash));
			uint32_t signature_read = 0;
			uint32_t hash_read      = 0;
			uint32_t payload_read   = 0;
			file.Read(cached_signature.data(), static_cast<uint32_t>(cached_signature.size()),
			          &signature_read);
			file.Read(&payload_hash, sizeof(payload_hash), &hash_read);
			file.Read(initial_data.data(), static_cast<uint32_t>(initial_data.size()),
			          &payload_read);
			file.Close();
			if (signature_read != cached_signature.size() || hash_read != sizeof(payload_hash) ||
			    payload_read != initial_data.size() || cached_signature != signature ||
			    XXH3_64bits(initial_data.data(), initial_data.size()) != payload_hash) {
				initial_data.clear();
				PipelineCacheLog(
				    "Vulkan pipeline cache: invalidating {} (driver or data mismatch)",
				    path);
			}
		} else {
			file.Close();
			PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid file size)", path);
		}
	}

	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize = initial_data.size();
	create.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();
	auto result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		create.initialDataSize = 0;
		create.pInitialData    = nullptr;
		result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: disabled ({})", vk::to_string(result));
		m_driver_cache = nullptr;
		return false;
	}
	if (!initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: loaded {:.2f} MB from {}",
		                 static_cast<double>(initial_data.size()) / 1'000'000.0, path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}
	if (m_driver_cache_path.empty()) {
		return false;
	}
	m_driver_cache_posted = std::chrono::steady_clock::now();
	m_driver_cache_timer = std::jthread([this](const std::stop_token& stop) {
		while (!stop.stop_requested()) {
			std::this_thread::sleep_for(DriverCacheTimerPeriod);
			std::lock_guard lock(m_driver_cache_mutex);
			PostDueDriverCache();
		}
	});
	return !initial_data.empty();
}

void PipelineCache::CheckpointDriverCache(bool slow_creation) {
	if (m_driver_cache == nullptr || m_driver_cache_path.empty()) {
		return;
	}
	std::lock_guard lock(m_driver_cache_mutex);
	m_driver_cache_dirty = true;
	if (slow_creation) {
		m_next_driver_cache_save = std::min(m_next_driver_cache_save,
		                                    m_driver_cache_posted + SlowCreationSaveInterval);
	}
	PostDueDriverCache();
}

void PipelineCache::NoteCreatedPipeline(std::chrono::steady_clock::time_point started) {
#if defined(ANYPS5_KYTY)
	m_pipelines_created.fetch_add(1, std::memory_order_relaxed);
#endif
	CheckpointDriverCache(std::chrono::steady_clock::now() - started >= SlowCreation);
}

void PipelineCache::PostDueDriverCache() {
	if (!m_driver_cache_dirty) {
		return;
	}
	if (const auto now = std::chrono::steady_clock::now(); now >= m_next_driver_cache_save) {
		m_driver_cache_dirty     = false;
		m_driver_cache_posted    = now;
		m_next_driver_cache_save = now + DriverCacheSaveInterval;
		m_store.Post([this] { WriteDriverCache(); });
	}
}

void PipelineCache::Save() {
	m_driver_cache_timer = {};
	m_store.Finish();
	bool dirty = false;
	{
		std::lock_guard lock(m_driver_cache_mutex);
		dirty = std::exchange(m_driver_cache_dirty, false);
	}
	if (dirty) {
		WriteDriverCache();
	}
}

void PipelineCache::WriteDriverCache() {
	std::vector<vk::PipelineCache> helper_caches;
	{
		std::scoped_lock lock(m_helper_caches_mutex);
		helper_caches.swap(m_helper_caches);
	}
	const vk::PipelineCacheCreateInfo empty {};
	if (!helper_caches.empty() && m_helper_cache == nullptr &&
	    m_graphics.device.createPipelineCache(&empty, nullptr, &m_helper_cache) != vk::Result::eSuccess) {
		m_helper_cache = nullptr;
	}
	if (m_helper_cache != nullptr && !helper_caches.empty()) {
		(void)m_graphics.device.mergePipelineCaches(m_helper_cache,
		                                            static_cast<uint32_t>(helper_caches.size()),
		                                            helper_caches.data());
	}
	for (const auto cache: helper_caches) {
		m_graphics.device.destroyPipelineCache(cache, nullptr);
	}
	vk::PipelineCache source = m_driver_cache;
	vk::PipelineCache merged = nullptr;
	if (m_helper_cache != nullptr &&
	    m_graphics.device.createPipelineCache(&empty, nullptr, &merged) == vk::Result::eSuccess) {
		const std::array sources {m_driver_cache, m_helper_cache};
		if (m_graphics.device.mergePipelineCaches(merged, static_cast<uint32_t>(sources.size()),
		                                          sources.data()) == vk::Result::eSuccess) {
			source = merged;
		}
	}
	size_t               size = 0;
	vk::Result           result;
	std::vector<uint8_t> payload;
	for (uint32_t attempt = 0; attempt < 3; attempt++) {
		size   = 0;
		result = m_graphics.device.getPipelineCacheData(source, &size, nullptr);
		if (result != vk::Result::eSuccess || size == 0 ||
		    size > std::numeric_limits<uint32_t>::max()) {
			break;
		}
		payload.resize(size);
		result = m_graphics.device.getPipelineCacheData(source, &size, payload.data());
		if (result != vk::Result::eIncomplete) {
			break;
		}
	}
	m_graphics.device.destroyPipelineCache(merged, nullptr);
	if (result != vk::Result::eSuccess || size == 0 ||
	    size > std::numeric_limits<uint32_t>::max()) {
		PipelineCacheLog("Vulkan pipeline cache: save failed ({}, {:.2f} MB)",
		                 vk::to_string(result), static_cast<double>(size) / 1'000'000.0);
		return;
	}
	payload.resize(size);
	auto       prefix       = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	const auto payload_hash = XXH3_64bits(payload.data(), payload.size());
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	if (!Common::File::CreateDirectories(m_driver_cache_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		return;
	}
	auto temp_path = m_driver_cache_path;
	temp_path += ".tmp";
	Common::File file;
	uint32_t     prefix_written  = 0;
	uint32_t     payload_written = 0;
	if (file.Create(temp_path)) {
		file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
		file.Write(payload.data(), static_cast<uint32_t>(payload.size()), &payload_written);
	}
	const bool flushed = !file.IsInvalid() && file.Flush();
	file.Close();
	const bool written = prefix_written == prefix.size() && payload_written == payload.size() && flushed;
	std::error_code error;
	if (written) {
		std::filesystem::rename(temp_path, m_driver_cache_path, error);
	}
	if (!written || error) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}",
		                 Common::PathToString(m_driver_cache_path));
		return;
	}
	PipelineCacheLog("Vulkan pipeline cache: saved {:.2f} MB to {}",
	                 static_cast<double>(payload.size()) / 1'000'000.0,
	                 Common::PathToString(m_driver_cache_path));
}

void PipelineCache::GetGraphicsPrograms(const CommandBuffer& command, DrawRenderState& state) {
	ResolveGraphicsPrograms(command.GetRegisters(), command.GetUserConfig(), command.GetShaders(),
	                        state, nullptr);
}

bool PipelineCache::GetGraphicsProgramsAhead(const HW::Context&    context,
                                             const HW::UserConfig& user_config,
                                             const HW::Shader& shaders, DrawRenderState& state,
                                             ProgramResolution& ahead) {
	return ResolveGraphicsPrograms(context, user_config, shaders, state, &ahead);
}

#if defined(ANYPS5_KYTY)
struct PipelineCache::ChainJob {
	ShaderParams                         pixel_params;
	ShaderPixelInputInfo                 pixel_info;
	std::array<ShaderParams, 3>          vertex_params;
	std::array<ShaderVertexInputInfo, 3> vertex_info;
	uint32_t                             vertex_count     = 1;
	uint32_t                             push_data_cursor = 0;
	PipelineStaticParameters             params;
};
#endif

bool PipelineCache::ResolveGraphicsPrograms(const HW::Context&    context,
                                            const HW::UserConfig& user_config,
                                            const HW::Shader& shaders, DrawRenderState& state,
                                            ProgramResolution* ahead) {
	const auto& sh                = context.GetShaderRegisters();
	const auto& vertex_regs       = shaders.GetVs();
	const auto& pixel_regs        = shaders.GetPs();
	auto&       vertex_info       = state.vertex_info;
	auto&       pixel_info        = state.ps_input_info;
	const bool  object_primitive_id = context.HasObjectPrimitiveIdOutput();
	const auto  write_mask          = context.GetRenderTargetMask() & sh.m_cbShaderMask;
	uint32_t    color_output_mask   = object_primitive_id ? 1u << 7u : 0;
	for (uint32_t index = 0; index < RENDER_COLOR_ATTACHMENTS_MAX; index++) {
		const auto slot = ShaderPixelExportTarget(sh.m_cbShaderMask, index);
		if (slot >= RENDER_COLOR_ATTACHMENTS_MAX) {
			break;
		}
		if (sh.target_output_mode[index] != 0 && render_target_mask_slot(write_mask, slot) != 0) {
			color_output_mask |= 1u << slot;
		}
	}
	const auto& db = sh.db_shader_control;
	const bool pixel_active =
	    pixel_regs.ps_regs.data_addr != 0 &&
	    (color_output_mask != 0 || db.shader_kill_enable || db.shader_z_export_enable ||
	     db.shader_mask_export_enable || db.shader_dual_export_enable ||
	     db.shader_execute_on_noop || sh.AlphaToMask());
	state.ps_active = pixel_active;
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
#if defined(ANYPS5_KYTY)
	if (ahead != nullptr && !t_ahead_unlocked && ProgramLock::Narrow()) {
		SyncAheadWrites(AheadWritesSequence());
	}
#endif
	const ShaderGuestReaderScope reader(ahead != nullptr ? ReadAheadShaderState : nullptr,
	                                    ahead != nullptr ? &ahead->reads : nullptr);
	std::array<ShaderParams, 3> vertex_params;
	if (tess_active) {
#if defined(ANYPS5_KYTY)
		if (!Config::TessellationEnabled()) {
			return false;
		}
#endif
		if ((context.GetShaderStages() & 0x20u) != 0) {
			static std::atomic_bool warned = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(
				    "Warning: dropping tessellation draws with a geometry shader.\n");
			}
			return false;
		}
#if defined(ANYPS5_KYTY)
		if (!Common::Fallback::Run("draw or dispatch skipped: its shader stage could not be prepared", "PipelineCache::ResolveGraphicsPrograms",
		                           0, vertex_regs.ls_regs.data_addr,
		                           [&] { vertex_params = PrepareTessellationPrograms(vertex_regs, context, vertex_info); })) {
			vertex_info[0].stage = {};
			if (ahead != nullptr) ahead->reads.Fail();
			return ahead == nullptr;
		}
#else
		vertex_params = PrepareTessellationPrograms(vertex_regs, context, vertex_info);
#endif
#if defined(ANYPS5_KYTY)
		const auto skip_tessellation = [&] {
			vertex_info[0].stage = {};
			if (ahead != nullptr) ahead->reads.Fail();
			return ahead == nullptr;
		};
		const uint64_t layout_key[] = {vertex_params[0].hash, vertex_params[1].hash,
		                               uint64_t {sh.m_vgtLsHsConfig & 0xfff00u} << 32u |
		                                   sh.m_vgtTfParam |
		                                   uint64_t {vertex_info[0].tess.merged_hull} << 63u};
#else
		const uint64_t layout_key[] = {vertex_params[0].hash, vertex_params[1].hash,
		                               uint64_t {sh.m_vgtLsHsConfig & 0xfff00u} << 32u |
		                                   sh.m_vgtTfParam};
#endif
		const auto layout_hash = XXH3_64bits(layout_key, sizeof(layout_key));
		ShaderTessellationInputInfo tess;
		{
			const ProgramLockGuard lock(m_program_lock);
			if (ahead != nullptr) {
				const auto found = m_tess_layouts.find(layout_hash);
				if (found == m_tess_layouts.end()) {
					ahead->reads.Fail();
					return false;
				}
				tess = found->second;
			} else {
				const auto [layout, inserted] =
				    m_tess_layouts.try_emplace(layout_hash, vertex_info[0].tess);
				if (inserted) {
#if defined(ANYPS5_KYTY)
					if (layout->second.merged_hull != 0) {
						auto merged = layout->second;
						if (!Common::Fallback::Run(
						        "draw or dispatch skipped: its shader stage could not be prepared",
						        "PipelineCache::ResolveGraphicsPrograms", vertex_params[1].hash,
						        vertex_regs.ls_regs.data_addr, [&] {
							        static_cast<void>(ShaderRecompiler::AnalyzeTessellationPrograms(
							            vertex_params[0].code, vertex_params[1].code, merged));
						        })) {
							m_tess_layouts.erase(layout);
							return skip_tessellation();
						}
						layout->second = merged;
					} else
#endif
					if (!ShaderRecompiler::AnalyzeTessellationPrograms(
					        vertex_params[0].code, vertex_params[1].code, layout->second)) {
						EXIT("unsupported tessellation program shape: LS 0x%016" PRIx64
						     " HS 0x%016" PRIx64 "\n",
						     vertex_params[0].hash, vertex_params[1].hash);
					}
					StoreLayout(layout->first, layout->second);
				}
				tess = layout->second;
			}
		}
#if defined(ANYPS5_KYTY)
		if (tess.merged_hull != 0) {
			const auto& device = m_graphics.GetPhysicalDeviceProperties().limits;
			const ShaderRecompiler::TessellationLimits limits {
			    .per_vertex_input_components  = device.maxTessellationControlPerVertexInputComponents,
			    .per_vertex_output_components = device.maxTessellationControlPerVertexOutputComponents,
			    .per_patch_output_components  = device.maxTessellationControlPerPatchOutputComponents,
			    .total_output_components      = device.maxTessellationControlTotalOutputComponents,
			};
			if (!Common::Fallback::Run(
			        "draw or dispatch skipped: its shader stage could not be prepared",
			        "PipelineCache::ResolveGraphicsPrograms", vertex_params[1].hash,
			        vertex_regs.ls_regs.data_addr,
			        [&] { ShaderRecompiler::RequireMergedHullLimits(tess, limits); })) {
				return skip_tessellation();
			}
		}
#endif
		for (auto& stage: vertex_info) {
			stage.tess = tess;
		}
	} else {
#if defined(ANYPS5_KYTY)
		if (!Common::Fallback::Run("draw or dispatch skipped: its shader stage could not be prepared", "PipelineCache::ResolveGraphicsPrograms",
		                           0, vertex_regs.es_regs.data_addr,
		                           [&] { vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]); })) {
			vertex_info[0].stage = {};
			if (ahead != nullptr) ahead->reads.Fail();
			return ahead == nullptr;
		}
#else
		vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
#endif
	}
	if (ahead != nullptr && ahead->reads.Failed()) {
		return false;
	}
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	if (!tess_active && !mesh_active) {
		vertex_info[0].indirect_offsets = state.indirect_offsets;
		vertex_info[0].indirect_draw    = state.indirect_draw;
	}
	if (object_primitive_id) {
		EXIT_NOT_IMPLEMENTED(sh.m_paClObjprimIdCntl != 2u || mesh_active || tess_active ||
		                     !pixel_active || (sh.m_vgtDrawPayloadCntl & 8u) == 0 ||
		                     user_config.GetPrimType() != Prospero::PrimitiveType::kTriList);
		const auto& target = context.GetRenderTarget(7);
		EXIT_NOT_IMPLEMENTED(Prospero::ResolveRenderTargetFormat(target.info.format,
		                                                       target.info.channel_type)
		                         .buffer_format != Prospero::BufferFormat::k32UInt);
	}
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(!m_graphics.mesh_shader_enabled);
		auto& mesh              = vertex_info[0].mesh;
		mesh.host_subgroup_size = m_graphics.subgroup_size;
		const auto& limits      = m_graphics.mesh_shader_properties;
		const auto  logical_threads =
		    mesh.threads_num[0] * mesh.threads_num[1] * mesh.threads_num[2];
		const auto host_threads = ((logical_threads + mesh.wave_size - 1u) / mesh.wave_size) *
		                          std::min(mesh.host_subgroup_size, mesh.wave_size);
		if (host_threads > limits.maxMeshWorkGroupInvocations ||
		    host_threads > limits.maxMeshWorkGroupSize[0] ||
		    mesh.max_vertices > limits.maxMeshOutputVertices ||
		    mesh.max_primitives > limits.maxMeshOutputPrimitives ||
		    mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			static std::atomic_bool warned = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Warning: dropping mesh draws the host does not support (shader 0x{:016x}).\n",
				    vertex_params[0].hash));
			}
			if (ahead != nullptr) {
				ahead->reads.Fail();
			}
			return false;
		}
		if (state.indirect_draw &&
		    (mesh.lds_size_dwords + 2u + mesh.max_vertices +
		     ShaderRecompiler::IR::PushData::MeshTaskPayloadDwordCount) * sizeof(uint32_t) <=
		        limits.maxMeshPayloadAndSharedMemorySize) {
			vertex_info[0].indirect_offsets = state.indirect_offsets;
			vertex_info[0].indirect_draw    = true;
		}
	}
	ShaderParams pixel_params;
	if (pixel_active) {
		std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX>
		    target_export_mapping {};
		for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
			const auto& rt = context.GetRenderTarget(slot);
			if ((color_output_mask & (1u << slot)) != 0 && rt.base.addr != 0) {
				target_export_mapping[slot] =
				    TextureGetRenderTargetFormat(m_graphics, rt.info.format, rt.info.channel_type,
				                                 rt.info.channel_order)
				        .export_mapping;
			}
		}
#if defined(ANYPS5_KYTY)
		if (!Common::Fallback::Run("draw or dispatch skipped: its shader stage could not be prepared", "PipelineCache::ResolveGraphicsPrograms",
		                           0, pixel_regs.ps_regs.data_addr,
		                           [&] { pixel_params = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info); })) {
			vertex_info[0].stage = {};
			if (ahead != nullptr) ahead->reads.Fail();
			return ahead == nullptr;
		}
#else
		pixel_params      = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
#endif
		pixel_info.object_primitive_id = object_primitive_id;
		for (uint32_t export_index = 0; export_index < 8u; export_index++) {
			const auto slot = ShaderPixelExportTarget(pixel_info.target_shader_mask, export_index);
			if (slot >= 8u || pixel_info.target_output_mode[export_index] == 0u) continue;
			const auto& rt = context.GetRenderTarget(slot);
			const auto target_mask = render_target_mask_slot(context.GetRenderTargetMask(), slot);
			if (rt.base.addr == 0 || target_mask == 0u) continue;
			if (rt.info.channel_type == Prospero::ChannelType::kUInt) {
				pixel_info.target_uint_mask |= 1u << export_index;
			} else if (rt.info.channel_type == Prospero::ChannelType::kSInt) {
				pixel_info.target_sint_mask |= 1u << export_index;
			}
			const auto format = Prospero::ResolveRenderTargetFormat(rt.info.format, rt.info.channel_type);
			if (format.buffer_format != Prospero::BufferFormat::k11_11_10UNorm) continue;
			const auto mapping = pixel_info.target_export_mapping[export_index];
			const auto shader_mask = render_target_mask_slot(pixel_info.target_shader_mask, slot);
#if defined(ANYPS5_KYTY)
			const auto& blend_control = context.GetBlendControl(slot);
			const bool  replace       = blend_control.color_srcblend == 1u && blend_control.color_destblend == 0u &&
			                            blend_control.color_comb_fcn == 0u;
			const bool  blend         = blend_control.enable && !rt.info.blend_bypass && !replace;
			const auto  write_mask    = mapping.ApplyMask(target_mask) & mapping.ApplyMask(shader_mask) & 7u;
			if (write_mask == 0u || pixel_info.target_output_mode[export_index] != 9u) {
				EXIT("unsupported packed UNorm render-target export: slot=%u export=%u mode=%u "
				     "target_mask=0x%x shader_mask=0x%x mapping=0x%02x blend=%u src=%u dst=%u fcn=%u\n",
				     slot, export_index, pixel_info.target_output_mode[export_index], target_mask,
				     shader_mask, mapping.packed, static_cast<uint32_t>(blend), blend_control.color_srcblend,
				     blend_control.color_destblend, blend_control.color_comb_fcn);
			}
			if (blend || write_mask != 7u) {
				const auto& depth       = context.GetDepthControl();
				const auto& stencil     = context.GetStencilMask();
				const bool  ds_tests    = depth.z_enable || depth.stencil_enable || depth.depth_bounds_enable;
				const bool  ds_writes   = (depth.z_enable && depth.z_write_enable) ||
				                          (depth.stencil_enable &&
				                           (stencil.stencil_writemask != 0u || stencil.stencil_writemask_bf != 0u));
				const auto  unsupported = [&](uint8_t factor) {
					return factor > static_cast<uint8_t>(Prospero::BlendFactor::kSrcAlphaSaturate);
				};
				const char* refusal = nullptr;
				if (!m_graphics.fragment_shader_pixel_interlock_enabled) {
					refusal = "the device lacks fragmentShaderPixelInterlock";
				} else if (rt.attrib.num_samples != 0u || rt.attrib.num_fragments != 0u) {
					refusal = "the target is multisampled";
				} else if (blend && (unsupported(blend_control.color_srcblend) ||
				                     unsupported(blend_control.color_destblend) ||
				                     blend_control.color_comb_fcn > 4u)) {
					refusal = "the blend uses a constant or second-source factor";
				} else if (pixel_info.ps_depth_export_enable || pixel_info.ps_sample_mask_export_enable ||
				           sh.AlphaToMask()) {
					refusal = "the pixel shader exports depth or coverage";
				} else if (pixel_info.ps_pixel_kill_enable && !pixel_info.ps_early_z && ds_writes) {
					refusal = "the pixel shader kills pixels under late tests of a draw writing depth or stencil";
				}
				if (refusal != nullptr) {
					EXIT("unsupported packed UNorm render-target blend: slot=%u export=%u write_mask=0x%x "
					     "blend=%u src=%u dst=%u fcn=%u: %s\n",
					     slot, export_index, write_mask, static_cast<uint32_t>(blend),
					     blend_control.color_srcblend, blend_control.color_destblend,
					     blend_control.color_comb_fcn, refusal);
				}
				const ShaderPackedBlend packed {
				    .src_factor = static_cast<uint8_t>(blend ? blend_control.color_srcblend : 1u),
				    .dst_factor = static_cast<uint8_t>(blend ? blend_control.color_destblend : 0u),
				    .op         = static_cast<uint8_t>(blend ? blend_control.color_comb_fcn : 0u),
				    .write_mask = static_cast<uint8_t>(write_mask),
				    .blend      = blend,
				    .clamp      = rt.info.blend_clamp};
				pixel_info.target_packed_blend_mask |= 1u << export_index;
				pixel_info.target_packed_blend[export_index] = packed.Encode();
				pixel_info.target_packed_blend_tests |= ds_tests;
			}
#else
			const bool blend = context.GetBlendControl(slot).enable && !rt.info.blend_bypass;
			if (blend || (mapping.ApplyMask(target_mask) & 7u) != 7u ||
			    (mapping.ApplyMask(shader_mask) & 7u) != 7u ||
			    pixel_info.target_output_mode[export_index] != 9u) {
				EXIT("unsupported packed UNorm render-target export: slot=%u export=%u mode=%u "
				     "target_mask=0x%x shader_mask=0x%x mapping=0x%02x blend=%u\n",
				     slot, export_index, pixel_info.target_output_mode[export_index], target_mask,
				     shader_mask, mapping.packed, static_cast<uint32_t>(blend));
			}
#endif
			pixel_info.target_packed_unorm_mask |= 1u << export_index;
			pixel_info.target_round_mode |= static_cast<uint8_t>(rt.info.round_mode) << export_index;
		}
		if (!mesh_active && !tess_active && Prospero::IsRectList(user_config.GetPrimType())) {
			pixel_info.parameter_mode = ShaderPixelParameterMode::Rectangle;
		} else if (context.GetModeControl().provoking_vtx_last) {
			pixel_info.parameter_mode = ShaderPixelParameterMode::LastVertex;
		} else {
			pixel_info.parameter_mode = ShaderPixelParameterMode::FirstVertex;
		}
		const auto& blend = context.GetBlendControl(0);
		pixel_info.dual_source_blending =
		    blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		    (BlendFactorIsDualSource(blend.color_srcblend) ||
		     BlendFactorIsDualSource(blend.color_destblend) ||
		     (blend.separate_alpha_blend && (BlendFactorIsDualSource(blend.alpha_srcblend) ||
		                                     BlendFactorIsDualSource(blend.alpha_destblend))));
		if (pixel_info.dual_source_blending) {
			pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
			pixel_info.target_export_mapping[1] = pixel_info.target_export_mapping[0];
			pixel_info.target_uint_mask &= ~2u;
			pixel_info.target_sint_mask &= ~2u;
		} else if (blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		           pixel_info.target_output_mode[0] != 0 && pixel_info.target_output_mode[0] != 7 &&
		           std::all_of(std::begin(pixel_info.target_output_mode) + 1,
		                       std::end(pixel_info.target_output_mode),
		                       [](uint8_t mode) { return mode == 0; })) {
			switch (ClassifyBlendMapping(blend, pixel_info.target_export_mapping[0])) {
				case BlendMappingSupport::SourceAlpha:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlpha;
					break;
				case BlendMappingSupport::SourceAlphaOne:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaOne;
					break;
				case BlendMappingSupport::SourceAlphaZero:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaZero;
					break;
				default: break;
			}
			if (pixel_info.alpha_blend_source != ShaderAlphaBlendSource::None) {
				pixel_info.dual_source_blending     = true;
				pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
				pixel_info.target_export_mapping[1] = {};
			}
		}
#if defined(ANYPS5_KYTY)
		if ((pixel_info.target_packed_blend_mask & 1u) != 0u &&
		    pixel_info.alpha_blend_source != ShaderAlphaBlendSource::None) {
			pixel_info.alpha_blend_source       = ShaderAlphaBlendSource::None;
			pixel_info.dual_source_blending     = false;
			pixel_info.target_output_mode[1]    = 0;
			pixel_info.target_export_mapping[1] = {};
		}
#endif
	}
	auto&       last_stage   = vertex_info[tess_active ? 2u : 0u];
	const auto& clip_control = context.GetClipControl();
	if (clip_control.clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		const auto& limits   = m_graphics.GetPhysicalDeviceProperties().limits;
		auto&       clip     = last_stage.clip_space;
		clip.scale[0]        = viewport.xscale;
		clip.scale[1]        = viewport.yscale;
		clip.offset[0]       = viewport.xoffset;
		clip.offset[1]       = viewport.yoffset;
		clip.half_extent[0] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
		clip.half_extent[1] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		clip.enabled = true;
	} else if (clip_control.user_clip_planes != 0) {
		EXIT_NOT_IMPLEMENTED((last_stage.pa_cl_vs_out_cntl & 0xffffu) != 0 || tess_active ||
		                     Prospero::IsRectList(user_config.GetPrimType()));
		last_stage.user_clip_planes = clip_control.user_clip_planes;
	}
	uint32_t push_data_cursor = ShaderRecompiler::IR::PushData::GraphicsDrawDwords(
	    mesh_active, object_primitive_id, last_stage.user_clip_planes);
	const ProgramLockGuard lock(m_program_lock, ahead == nullptr || !t_ahead_unlocked);
	if (ahead != nullptr) {
#if defined(ANYPS5_KYTY)
		const uint32_t first_cursor = push_data_cursor;
		t_ahead_missing             = false;
#endif
		const ShaderModule* pixel_module = nullptr;
		if (pixel_active) {
			pixel_module = GetAhead(pixel_params, pixel_info, push_data_cursor, *ahead);
		}
		for (uint32_t i = 0; i < (tess_active ? 3u : 1u) && !ahead->reads.Failed(); i++) {
			GetAhead(vertex_params[i], vertex_info[i], push_data_cursor, *ahead);
		}
		if (ahead->reads.Failed()) {
#if defined(ANYPS5_KYTY)
			if (t_ahead_missing && pixel_active && !m_chain_threads.empty() && PreparesGraphicsLibraries()) {
				auto job          = std::make_unique<ChainJob>();
				job->pixel_params = pixel_params;
				job->pixel_info   = pixel_info;
				job->vertex_count = tess_active ? 3u : 1u;
				for (uint32_t i = 0; i < job->vertex_count; i++) {
					job->vertex_params[i] = vertex_params[i];
					job->vertex_info[i]   = vertex_info[i];
				}
				job->push_data_cursor = first_cursor;
				job->params           = LibraryStaticParameters(context, user_config, tess_active,
				                                                vertex_info[0].tess, true,
				                                                pixel_info.ps_sample_shading);
				QueueChain(std::move(job));
			}
#endif
			return false;
		}
		if (pixel_module != nullptr) {
			for (const auto& output: pixel_module->info.info.outputs) {
				if (output.kind == ShaderRecompiler::IR::StageOutputKind::Mrt ||
				    output.kind == ShaderRecompiler::IR::StageOutputKind::ObjectPrimitiveId) {
					state.mrt_mask |= 1u << output.location;
				}
			}
			state.mrt_mask &= color_output_mask;
		}
		return true;
	}
	if (pixel_active) {
		const auto* pixel_module = GetProgram(pixel_params, pixel_info, push_data_cursor);
		if (pixel_module == nullptr) {
			vertex_info[0].stage = {};
			return true;
		}
		for (const auto& output: pixel_module->info.info.outputs) {
			if (output.kind == ShaderRecompiler::IR::StageOutputKind::Mrt ||
			    output.kind == ShaderRecompiler::IR::StageOutputKind::ObjectPrimitiveId) {
				state.mrt_mask |= 1u << output.location;
			}
		}
		state.mrt_mask &= color_output_mask;
	}
	for (uint32_t i = 0; i < (tess_active ? 3u : 1u); i++) {
		GetProgram(vertex_params[i], vertex_info[i], push_data_cursor);
	}
	return true;
}

template <typename InputInfo>
const ShaderModule* PipelineCache::GetAhead(const ShaderParams& params, InputInfo& input_info,
                                            uint32_t& push_data_cursor, ProgramResolution& ahead) {
	thread_local ProgramKey key;
	if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
		key.stage = input_info.logical_stage;
	} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
		key.stage = ShaderType::Pixel;
	} else {
		static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
		key.stage = ShaderType::Compute;
	}
	key.hash            = params.hash;
	key.user_data_count = params.user_data_count;
	BuildStageStaticKey(input_info, key.static_state);
	AppendNativeRayTracingKey(key.stage, key.static_state);
	const bool unlocked = t_ahead_unlocked;
	Program*   program  = nullptr;
	{
		const ProgramLockGuard lock(m_program_lock, unlocked);
		const auto             entry = m_program_cache.find(key);
		if (entry == m_program_cache.end() && ahead.stages != ProgramResolution::MaxStages) {
			QueueCompile(key, params, input_info, push_data_cursor, nullptr);
#if defined(ANYPS5_KYTY)
			t_ahead_missing = true;
#endif
		}
		if (entry == m_program_cache.end() || ahead.stages == ProgramResolution::MaxStages ||
		    entry->second.skip_dispatch || entry->second.call_source ||
		    std::ranges::any_of(entry->second.modules, [](const ShaderModule& module) {
			    return module.info.info.uses_image_heap;
		    })) {
			ahead.reads.Fail();
			return nullptr;
		}
		program = &entry->second;
	}
	const auto stage_index    = ahead.stages++;
	auto&      resources      = ahead.resources[stage_index];
	auto&      specialization = ahead.specializations[stage_index];
	ahead.plans[stage_index] = &program->resource_plan;
	ahead.reads.SetStage(stage_index, &program->resource_plan);
	const auto pending_write = t_ahead_guest_writes.writes.empty() ? nullptr : AheadPendingWrite;
	ShaderRecompiler::IR::SrtRuntime runtime {
	    .user_data                  = std::span(params.user_data).first(params.user_data_count),
	    .shader_base                = params.Base(),
	    .userdata                   = &ahead.reads,
	    .read_specialization_memory = ReadAheadStrict,
	    .record_load                = RecordAheadLoad,
	    .record_loads               = RecordAheadLoads,
	    .gpu_produced               = AheadGpuProduced,
	    .pending_write              = pending_write,
#if defined(ANYPS5_KYTY)
	    .image_table_heap           = Config::ImageTableHeapEnabled(),
#endif
	};
	if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
		runtime.workgroup_counts      = input_info.workgroup_counts;
		runtime.indirect_args_address = input_info.indirect_args_address;
		runtime.runtime_stride        = RuntimeStride(params.hash);
	}
	t_ahead_unread_load = false;
#if defined(ANYPS5_KYTY)
	const bool defers = ShaderRecompiler::IR::SrtFlatDeferMode() != ShaderRecompiler::IR::SrtFlatDefer::Off;
	runtime.defer_flat_loads = defers;
	if (defers) ShaderRecompiler::IR::SrtDeferredLoadsOfThread().clear();
#endif
	if (!ShaderRecompiler::IR::MaterializeResources(program->resource_plan, runtime, resources,
	                                                specialization)) {
		ahead.reads.Fail();
		return nullptr;
	}
#if defined(ANYPS5_KYTY)
	if (defers && !ShaderRecompiler::IR::SrtDeferredLoadsOfThread().empty()) {
		ahead.reads.NoteDeferredLoads(ShaderRecompiler::IR::SrtDeferredLoadsOfThread());
	}
#endif
	NoteAheadWrites(program->resource_plan, resources);
	const auto matches = [&](const ShaderModule& candidate) {
		const auto& layout = candidate.info.bindings;
		return layout.push_data_start_dword ==
		           ShaderRecompiler::IR::PushData::StartFor(push_data_cursor,
		                                                    layout.ShaderDataDwords()) &&
		       candidate.specialization == specialization;
	};
	const ProgramLockGuard lock(m_program_lock, unlocked);
	auto&                  modules = program->modules;
	auto&                  last    = program->last_module[1];
	auto                   module  = last < modules.size() && matches(modules[last])
	                    ? modules.begin() + static_cast<std::ptrdiff_t>(last)
	                    : std::ranges::find_if(modules, matches);
#if defined(ANYPS5_KYTY)
	if (module == modules.end() && program->ShareModule(specialization, resources, push_data_cursor, false)) {
		module = modules.end() - 1;
	}
	if (module != modules.end() && module->info.info.uses_image_heap) {
		ahead.reads.Fail();
		return nullptr;
	}
#endif
	if (module == modules.end()) {
#if defined(ANYPS5_KYTY)
		if (!program->failed_variants.empty() && program->Failed(specialization, push_data_cursor)) {
			ahead.reads.Fail();
			return nullptr;
		}
#endif
		if (!ahead.reads.MayHaveReadGpuWrites() && !t_ahead_unread_load) {
#if defined(ANYPS5_KYTY)
			auto promoted = specialization;
			program->PromoteEmptyBuffers(resources, promoted, push_data_cursor);
			QueueCompile(key, params, input_info, push_data_cursor, &promoted);
			t_ahead_missing = true;
#else
			QueueCompile(key, params, input_info, push_data_cursor, &specialization);
#endif
		}
		ahead.reads.Fail();
		return nullptr;
	}
	last = static_cast<size_t>(module - modules.begin());
	input_info.stage = {.module = &*module, .resources = &resources};
	module->info.bindings.AdvancePushData(push_data_cursor);
	return &*module;
}

struct PipelineCache::CompileJob {
	ProgramKey                                                  key;
	ShaderParams                                                params;
	std::unique_ptr<ShaderVertexInputInfo>                      vertex;
	std::unique_ptr<ShaderPixelInputInfo>                       pixel;
	std::unique_ptr<ShaderComputeInputInfo>                     compute;
	uint32_t                                                    push_data_cursor = 0;
	std::optional<ShaderRecompiler::IR::ResourceSpecialization> specialization;
	TableVariant*                                               table = nullptr;
	bool                                                        guest_traversal = false;
};

#if defined(ANYPS5_KYTY)
void PipelineCache::LockPrograms(std::source_location where) {
	ProgramLock::Acquire(m_program_lock, where);
}

void PipelineCache::UnlockPrograms() {
	ProgramLock::Release(m_program_lock);
}
#else
void PipelineCache::LockPrograms() {
	while (m_program_lock.test_and_set(std::memory_order_acquire)) {
		while (m_program_lock.test(std::memory_order_relaxed)) {
			Common::SpinPause();
		}
	}
}

void PipelineCache::UnlockPrograms() {
	m_program_lock.clear(std::memory_order_release);
}
#endif

template <typename InputInfo>
void PipelineCache::QueueCompile(const ProgramKey& key, const ShaderParams& params,
                                 const InputInfo&                                    input_info,
                                 uint32_t                                            push_data_cursor,
                                 const ShaderRecompiler::IR::ResourceSpecialization* specialization,
                                 bool                                                guest_traversal) {
	if (m_in_flight.contains(key)) {
		return;
	}
	auto job              = std::make_unique<CompileJob>();
	job->key              = key;
	job->params           = params;
	job->push_data_cursor = push_data_cursor;
	job->guest_traversal  = guest_traversal;
	if (specialization != nullptr) {
		job->specialization = *specialization;
	}
	if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
		job->vertex = std::make_unique<ShaderVertexInputInfo>(input_info);
	} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
		job->pixel = std::make_unique<ShaderPixelInputInfo>(input_info);
	} else {
		job->compute = std::make_unique<ShaderComputeInputInfo>(input_info);
	}
	m_in_flight[key]++;
	{
		std::lock_guard lock(m_job_mutex);
		m_jobs.push_back(std::move(job));
	}
	m_job_available.notify_one();
}

bool PipelineCache::WaitForCompile(const ProgramKey& key) {
	if (!m_in_flight.contains(key)) {
		return false;
	}
	const auto copy = key;
	do {
		UnlockPrograms();
		GuestGpu::RunUrgentCommandsWhileWaiting();
		std::this_thread::yield();
		LockPrograms();
	} while (m_in_flight.contains(copy));
	return true;
}

template <typename InputInfo>
void PipelineCache::RunCompileJob(CompileJob& job, InputInfo& input_info) {
	std::optional<ShaderModule>                       module;
	std::optional<ShaderRecompiler::IR::ResourcePlan> plan;
	std::vector<uint32_t>                             spirv;
	std::vector<uint32_t>                             plain_spirv;
	ShaderRecompiler::CompileOptions                  options {.wave_size = input_info.wave_size};
#if defined(ANYPS5_KYTY)
	const auto compile = [&] {
#else
	[&] {
#endif
		const auto user_data = std::span(job.params.user_data).first(job.params.user_data_count);
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			options.stage             = input_info.logical_stage;
			options.input_info.vertex = &input_info;
			options.user_data_base    = 8;
			if (options.stage == ShaderType::Mesh ||
			    options.stage == ShaderType::TessellationControl) {
				options.user_data_base = 0;
			}
			if (options.stage == ShaderType::Mesh) {
				options.wave_size = input_info.mesh.wave_size;
			}
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			options.stage            = ShaderType::Pixel;
			options.input_info.pixel = &input_info;
		} else {
			options.stage              = ShaderType::Compute;
			options.input_info.compute = &input_info;
		}
		options.shader_hash     = job.params.hash;
		options.user_data       = user_data;
		options.back_code       = job.params.back_code;
		options.dump_ir         = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
		options.emit_debug_info = Config::GraphicsDebugDumpEnabled();
		options.native_ray_tracing = RayTracing::NativeTraversals() && !job.guest_traversal;
		options.ray_stats          = options.native_ray_tracing && Config::RayTracingStatsEnabled();
		options.rt_oracle          = options.native_ray_tracing ? Config::RayTracingOracle() : 0u;
		auto source = ShaderRecompiler::PrepareShaderSource(job.params.code, options);
		if (!source.calls.empty() || source.unsupported_call_pc != UINT32_MAX) {
			return;
		}
		auto translated = ShaderRecompiler::TranslateProgram(source.decoded, options);
		plan            = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
		auto specialization = job.specialization;
		if (!specialization) {
			GuestReadLog                           reads;
			ShaderRecompiler::IR::ResourceSnapshot resources;
			specialization.emplace();
			ShaderRecompiler::IR::SrtRuntime runtime {
			    .user_data                  = user_data,
			    .shader_base                = job.params.Base(),
			    .userdata                   = &reads,
			    .read_specialization_memory = ReadAheadStrict,
			    .record_load                = RecordAheadLoad,
			    .record_loads               = RecordAheadLoads,
#if defined(ANYPS5_KYTY)
			    .image_table_heap           = Config::ImageTableHeapEnabled(),
#endif
			};
			if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
				runtime.workgroup_counts      = input_info.workgroup_counts;
				runtime.indirect_args_address = input_info.indirect_args_address;
				runtime.runtime_stride        = RuntimeStride(job.params.hash);
			}
			if (!ShaderRecompiler::IR::MaterializeResources(*plan, runtime, resources,
			                                                *specialization)) {
				specialization.reset();
			}
		}
		if (specialization) {
			module = BuildModule(job.key, options, std::move(translated), *specialization, {},
			                     job.push_data_cursor, spirv, plain_spirv);
		}
#if defined(ANYPS5_KYTY)
	};
	if (!Common::Fallback::Run("background shader compile dropped: the shader recompiler stopped",
	                           "PipelineCache::RunCompileJob", job.params.hash, 0, compile)) {
		module.reset();
	}
#else
	}();
#endif
	const ShaderModule* ahead = nullptr;
	LockPrograms();
	if (module) {
		auto entry = m_program_cache.find(job.key);
		if (entry == m_program_cache.end()) {
			entry = m_program_cache.try_emplace(job.key, Program {.resource_plan = std::move(*plan)})
			            .first;
		}
		auto&       modules = entry->second.modules;
		const auto& layout  = module->info.bindings;
		const bool  known   = std::ranges::any_of(modules, [&](const ShaderModule& other) {
			return other.specialization == module->specialization &&
			       other.info.bindings.push_data_start_dword == layout.push_data_start_dword;
		});
		if (known) {
			DestroyModule(*module);
		} else {
			auto& added = modules.emplace_back(std::move(*module));
			added.plan  = &entry->second.resource_plan;
			AddedModule(job.key, entry->second, added, spirv, plain_spirv, options, job.params.code,
			            job.push_data_cursor);
			if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
				ahead = &added;
			}
		}
	} else if (job.guest_traversal && plan) {
		m_program_cache.try_emplace(job.key, Program {.resource_plan = std::move(*plan)});
	}
	if (const auto job_count = m_in_flight.find(job.key); --job_count->second == 0) {
		m_in_flight.erase(job_count);
	}
	UnlockPrograms();
	if (ahead != nullptr) {
		CreateComputePipelineAhead(*ahead, spirv);
	}
}

void PipelineCache::CompileWorker() {
	KYTY_PROFILER_THREAD("Thread_ShaderCompile");
	for (;;) {
		std::unique_ptr<CompileJob> job;
		{
			std::unique_lock lock(m_job_mutex);
			m_job_available.wait(lock, [&] { return m_stopping || !m_jobs.empty(); });
			if (m_stopping) {
				return;
			}
			job = std::move(m_jobs.front());
			m_jobs.pop_front();
		}
		if (job->table != nullptr) {
			if (job->vertex != nullptr) {
				RunTableJob(*job, *job->vertex);
			} else if (job->pixel != nullptr) {
				RunTableJob(*job, *job->pixel);
			} else {
				RunTableJob(*job, *job->compute);
			}
		} else if (job->vertex != nullptr) {
			RunCompileJob(*job, *job->vertex);
		} else if (job->pixel != nullptr) {
			RunCompileJob(*job, *job->pixel);
		} else {
			RunCompileJob(*job, *job->compute);
		}
	}
}

void PipelineCache::StopWorkers() {
	{
		std::lock_guard lock(m_job_mutex);
		m_stopping = true;
	}
	m_job_available.notify_all();
	m_workers.clear();
}

#if defined(ANYPS5_KYTY)
void PipelineCache::QueueChain(std::unique_ptr<ChainJob>&& job) {
	thread_local std::vector<uint32_t>       words;
	thread_local std::array<uint64_t, 4096> seen {};
	uint64_t hash = XXH3_64bits_withSeed(&job->params, sizeof(job->params), job->push_data_cursor);
	BuildStageStaticKey(job->pixel_info, words);
	hash = XXH3_64bits_withSeed(words.data(), words.size() * sizeof(uint32_t), hash ^ job->pixel_params.hash);
	for (uint32_t i = 0; i < job->vertex_count; i++) {
		BuildStageStaticKey(job->vertex_info[i], words);
		hash = XXH3_64bits_withSeed(words.data(), words.size() * sizeof(uint32_t), hash ^ job->vertex_params[i].hash);
	}
	hash |= 1u;
	auto& entry = seen[hash & (seen.size() - 1u)];
	if (entry == hash) {
		return;
	}
	entry = hash;
	{
		std::lock_guard lock(m_chain_mutex);
		if (m_chain_stopping || m_chain_jobs.size() >= 256) {
			m_chain_dropped.fetch_add(1, std::memory_order_relaxed);
			return;
		}
		m_chain_jobs.push_back(std::move(job));
	}
	m_chain_queued.fetch_add(1, std::memory_order_relaxed);
	m_chain_available.notify_one();
}

void PipelineCache::StopChains() {
	{
		std::lock_guard lock(m_chain_mutex);
		m_chain_stopping = true;
		m_chain_jobs.clear();
	}
	m_chain_available.notify_all();
	m_chain_threads.clear();
}

void PipelineCache::ChainWorker() {
	KYTY_PROFILER_THREAD("Thread_PipelineChain");
	PlaceBackgroundCompileThread(false);
	for (;;) {
		std::unique_ptr<ChainJob> job;
		{
			std::unique_lock lock(m_chain_mutex);
			m_chain_available.wait(lock, [&] { return m_chain_stopping || !m_chain_jobs.empty(); });
			if (m_chain_stopping) {
				return;
			}
			job = std::move(m_chain_jobs.front());
			m_chain_jobs.pop_front();
		}
		const auto started = std::chrono::steady_clock::now();
		RunChain(*job);
		m_chain_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                               std::chrono::steady_clock::now() - started)
		                                               .count()),
		                     std::memory_order_relaxed);
	}
}

template <typename InputInfo>
const ShaderModule* PipelineCache::ChainModule(const ShaderParams& params, InputInfo& input_info,
                                               uint32_t& push_data_cursor,
                                               ShaderRecompiler::IR::ResourceSnapshot& resources) {
	ProgramKey key;
	if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
		key.stage = input_info.logical_stage;
	} else {
		static_assert(std::is_same_v<InputInfo, ShaderPixelInputInfo>);
		key.stage = ShaderType::Pixel;
	}
	key.hash            = params.hash;
	key.user_data_count = params.user_data_count;
	BuildStageStaticKey(input_info, key.static_state);
	AppendNativeRayTracingKey(key.stage, key.static_state);
	ShaderRecompiler::IR::ResourceSpecialization specialization;
	bool                                         materialized = false;
	for (uint32_t attempt = 0; attempt < 3; attempt++) {
		LockPrograms();
		while (m_in_flight.contains(key)) {
			UnlockPrograms();
			{
				std::lock_guard lock(m_chain_mutex);
				if (m_chain_stopping) {
					return nullptr;
				}
			}
			std::this_thread::sleep_for(std::chrono::microseconds(200));
			LockPrograms();
		}
		const auto entry   = m_program_cache.find(key);
		Program*   program = entry != m_program_cache.end() ? &entry->second : nullptr;
		if (program != nullptr &&
		    (program->skip_dispatch || program->call_source ||
		     std::ranges::any_of(program->modules,
		                         [](const ShaderModule& module) { return module.info.info.uses_image_heap; }))) {
			UnlockPrograms();
			return nullptr;
		}
		if (program != nullptr && !materialized) {
			UnlockPrograms();
			GuestReadLog                     reads;
			ShaderRecompiler::IR::SrtRuntime runtime {
			    .user_data                  = std::span(params.user_data).first(params.user_data_count),
			    .shader_base                = params.Base(),
			    .userdata                   = &reads,
			    .read_specialization_memory = ReadAheadStrict,
			    .record_load                = RecordAheadLoad,
			    .record_loads               = RecordAheadLoads,
#if defined(ANYPS5_KYTY)
			    .image_table_heap           = Config::ImageTableHeapEnabled(),
#endif
			};
			if (!ShaderRecompiler::IR::MaterializeResources(program->resource_plan, runtime, resources,
			                                                specialization)) {
				return nullptr;
			}
			materialized = true;
			LockPrograms();
		}
#if defined(ANYPS5_KYTY)
		if (program != nullptr && materialized && program->Failed(specialization, push_data_cursor)) {
			UnlockPrograms();
			return nullptr;
		}
#endif
		if (program != nullptr) {
			auto module = std::ranges::find_if(program->modules, [&](const ShaderModule& candidate) {
				const auto& layout = candidate.info.bindings;
				return layout.push_data_start_dword ==
				           ShaderRecompiler::IR::PushData::StartFor(push_data_cursor, layout.ShaderDataDwords()) &&
				       candidate.specialization == specialization;
			});
#if defined(ANYPS5_KYTY)
			if (module == program->modules.end() && materialized &&
			    program->ShareModule(specialization, resources, push_data_cursor, false)) {
				module = program->modules.end() - 1;
			}
			if (module != program->modules.end() && module->info.info.uses_image_heap) {
				UnlockPrograms();
				return nullptr;
			}
#endif
			if (module != program->modules.end()) {
				UnlockPrograms();
				module->info.bindings.AdvancePushData(push_data_cursor);
				return &*module;
			}
		}
		if (m_in_flight.contains(key)) {
			UnlockPrograms();
			continue;
		}
		m_in_flight[key]++;
#if defined(ANYPS5_KYTY)
		if (program != nullptr && materialized) {
			program->PromoteEmptyBuffers(resources, specialization, push_data_cursor);
		}
#endif
		UnlockPrograms();
		CompileJob job {.key = key, .params = params, .push_data_cursor = push_data_cursor};
		if (materialized) {
			job.specialization = specialization;
		}
		RunCompileJob(job, input_info);
		m_chain_compiled.fetch_add(1, std::memory_order_relaxed);
		if (!materialized) {
			LockPrograms();
			const bool added = m_program_cache.contains(key);
			UnlockPrograms();
			if (!added) {
				return nullptr;
			}
		}
	}
	return nullptr;
}

void PipelineCache::RunChain(ChainJob& job) {
	uint32_t                               cursor = job.push_data_cursor;
	ShaderRecompiler::IR::ResourceSnapshot pixel_resources;
	ShaderRecompiler::IR::ResourceSnapshot vertex_resources;
	std::array<const ShaderModule*, 4>     stages {};
	const auto* pixel = ChainModule(job.pixel_params, job.pixel_info, cursor, pixel_resources);
	if (pixel == nullptr) {
		m_chain_failed.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	for (uint32_t i = 0; i < job.vertex_count; i++) {
		stages[i] = ChainModule(job.vertex_params[i], job.vertex_info[i], cursor, vertex_resources);
		if (stages[i] == nullptr) {
			m_chain_failed.fetch_add(1, std::memory_order_relaxed);
			return;
		}
	}
	stages[3] = RecordsNoMipStats({.module = pixel, .resources = &pixel_resources}) ? pixel->plain.get() : pixel;
	GraphicsPipelineKey key {};
	key.static_params = job.params;
	for (uint32_t i = 0; i < stages.size(); i++) {
		key.stage_ids[i] = stages[i] != nullptr ? stages[i]->id : 0;
	}
	m_libraries->PrepareNow(key, stages, false);
	m_chain_prepared.fetch_add(1, std::memory_order_relaxed);
}
#endif

void PipelineCache::QueueAsyncPipeline(const AsyncPipelineJob& job) {
	{
		std::lock_guard lock(m_async_mutex);
		if (job.compute) {
			m_async_jobs.push_front(job);
		} else {
			m_async_jobs.push_back(job);
		}
	}
	m_async_available.notify_one();
}

template <typename Matches>
bool PipelineCache::TakeQueuedAsyncPipeline(Matches&& matches) {
	for (auto* jobs: {&m_async_jobs, &m_async_sibling_jobs}) {
		const auto job = std::ranges::find_if(*jobs, matches);
		if (job != jobs->end()) {
			jobs->erase(job);
			return true;
		}
	}
	return false;
}

void PipelineCache::QueueSiblingPipeline(const GraphicsPipelineKey&                key,
                                         const std::array<const ShaderModule*, 4>& stages,
                                         const ShaderModule*                       pixel) {
	if (m_async_threads.empty() || pixel == nullptr || pixel->plain == nullptr || stages[3] == nullptr) {
		return;
	}
	AsyncPipelineJob job {.key = key, .stages = stages};
	job.stages[3]        = stages[3] == pixel ? pixel->plain.get() : pixel;
	job.key.stage_ids[3] = job.stages[3]->id;
	if (m_graphics_pipelines.contains(job.key) || m_async_ready.contains(job.key) ||
	    m_async_failed_graphics.contains(job.key) ||
	    (m_manifest_warm && m_claimed_graphics.contains(job.key)) ||
	    !m_async_pending_graphics.insert(job.key).second) {
		return;
	}
	ClaimPipeline(job.key);
	m_async_sibling_jobs.push_back(job);
	m_async_available.notify_one();
}

void PipelineCache::StopAsyncPipelines() {
	{
		std::lock_guard lock(m_async_mutex);
		m_async_stopping = true;
		m_async_jobs.clear();
		m_async_sibling_jobs.clear();
	}
	m_async_available.notify_all();
	m_async_threads.clear();
}

void PipelineCache::AsyncPipelineWorker() {
	KYTY_PROFILER_THREAD("Thread_PipelineAsync");
	for (;;) {
		AsyncPipelineJob job;
		{
			std::unique_lock lock(m_async_mutex);
			m_async_available.wait(lock, [&] {
				return m_async_stopping || !m_async_jobs.empty() || !m_async_sibling_jobs.empty();
			});
			if (m_async_stopping) {
				return;
			}
			auto& jobs = !m_async_jobs.empty() ? m_async_jobs : m_async_sibling_jobs;
			job        = jobs.front();
			jobs.pop_front();
		}
		Pipeline   pipeline;
		pipeline.native_bindings = job.native;
		const auto started = std::chrono::steady_clock::now();
		if (job.compute) {
			CreatePipelineInternal(m_graphics, pipeline, *job.module, m_driver_cache);
		} else {
#if defined(ANYPS5_KYTY)
			if (!CreateGraphicsPipelineFast(pipeline, job.key, job.stages))
#endif
				CreatePipelineInternal(m_graphics, pipeline, job.key, job.stages, m_driver_cache);
		}
		const bool created = pipeline.pipeline != nullptr && pipeline.pipeline_layout != nullptr;
		if (created) {
			NoteCreatedPipeline(started);
		}
		std::lock_guard lock(m_async_mutex);
		if (job.native) {
			auto& native = m_async_native.at(job.key);
			if (created) {
				native.pipeline = pipeline;
			}
			native.done->store(true, std::memory_order_release);
		} else if (job.compute) {
			if (created) {
				m_async_ready_compute.try_emplace(job.module->id, pipeline);
			}
			m_async_pending_compute.erase(job.module->id);
		} else {
			if (created) {
				m_async_ready.try_emplace(job.key, pipeline);
			} else {
				m_async_failed_graphics.insert(job.key);
			}
			m_async_pending_graphics.erase(job.key);
		}
	}
}

void PipelineCache::ClaimPipeline(const GraphicsPipelineKey& key) {
	if (m_manifest_warm) {
		m_claimed_graphics.insert(key);
	}
}

void PipelineCache::ClaimComputePipeline(uint64_t module_id) {
	if (m_manifest_warm) {
		m_claimed_compute.insert(module_id);
	}
}

#if defined(ANYPS5_KYTY)
void PipelineCache::RequestComputePipelineAhead(const ShaderModule& module) {
	if (m_libraries == nullptr) {
		return;
	}
	std::atomic_ref<bool> requested(module.pipeline_requested);
	if (requested.load(std::memory_order_relaxed) || requested.exchange(true, std::memory_order_relaxed)) {
		return;
	}
	CreateComputePipelineAhead(module, {});
}
#endif

void PipelineCache::CreateComputePipelineAhead(const ShaderModule&          module,
                                               const std::vector<uint32_t>& spirv) {
#if defined(ANYPS5_KYTY)
	std::atomic_ref<bool>(module.pipeline_requested).store(true, std::memory_order_relaxed);
#endif
	if (m_helpers == nullptr && m_async_threads.empty()) {
		return;
	}
	{
		std::lock_guard lock(m_async_mutex);
		if (m_async_ready_compute.contains(module.id) ||
		    (m_manifest_warm && m_claimed_compute.contains(module.id)) ||
		    !m_async_pending_compute.insert(module.id).second) {
			return;
		}
		ClaimComputePipeline(module.id);
	}
	if (CreateComputePipelineElsewhere(module, spirv, true)) {
		return;
	}
	if (m_async_threads.empty()) {
		std::lock_guard lock(m_async_mutex);
		m_async_pending_compute.erase(module.id);
		m_claimed_compute.erase(module.id);
		return;
	}
	QueueAsyncPipeline({.compute = true, .module = &module});
}

void PipelineCache::StartHelpers() {
	if (m_driver_cache == nullptr) {
		return;
	}
	m_helpers = PipelineHelpers::Start(m_graphics, PipelineHelperProcesses(m_graphics),
	                                   [this](PipelineHelpers::Result&& result) {
		                                   OnHelperResult(std::move(result));
	                                   });
}

void PipelineCache::StopHelpers() {
	if (m_helpers == nullptr) {
		return;
	}
	m_helpers->LogProgress();
	PipelineCacheLog("Pipeline helpers: {} compute pipelines found in the driver cache, {} created "
	                 "from a helper's cache data, {} compiled here after a helper failed, {} taken "
	                 "back by a dispatch",
	                 m_helper_found.load(), m_helper_created.load(), m_helper_compiled_here.load(),
	                 m_helper_taken_back.load());
	m_helpers.reset();
}

bool PipelineCache::CreateComputePipelineElsewhere(const ShaderModule&          module,
                                                   const std::vector<uint32_t>& spirv, bool urgent) {
	if (m_helpers == nullptr || spirv.empty()) {
		return false;
	}
	const auto started = std::chrono::steady_clock::now();
	Pipeline   pipeline;
	if (FindComputePipeline(m_graphics, pipeline, module, m_driver_cache)) {
		m_helper_found++;
		PublishComputePipeline(module, pipeline, started);
		return true;
	}
	{
		std::lock_guard lock(m_async_mutex);
		m_helper_jobs.emplace(module.id, &module);
	}
	m_helpers->Queue(module.id, EncodeComputeModule(module, spirv), urgent,
	                 UseRank(module.info.shader_hash));
	return true;
}

void PipelineCache::OnHelperResult(PipelineHelpers::Result&& result) {
	const ShaderModule* module = nullptr;
	{
		std::lock_guard lock(m_async_mutex);
		const auto      job = m_helper_jobs.find(result.ticket);
		if (job == m_helper_jobs.end()) {
			return;
		}
		module = job->second;
		m_helper_jobs.erase(job);
	}
	const auto started = std::chrono::steady_clock::now();
	Pipeline   pipeline;
	bool       found = false;
	if (result.ok && !result.cache_data.empty()) {
		vk::PipelineCacheCreateInfo create {};
		create.initialDataSize  = result.cache_data.size();
		create.pInitialData     = result.cache_data.data();
		vk::PipelineCache cache = nullptr;
		if (m_graphics.device.createPipelineCache(&create, nullptr, &cache) == vk::Result::eSuccess) {
			found = FindComputePipeline(m_graphics, pipeline, *module, cache);
			if (found) {
				std::lock_guard lock(m_helper_caches_mutex);
				m_helper_caches.push_back(cache);
			} else {
				m_graphics.device.destroyPipelineCache(cache, nullptr);
			}
		}
	}
	if (found) {
		m_helper_created++;
	} else {
		m_helper_compiled_here++;
		Common::SoftExitScope soft_exit;
		try {
			CreatePipelineInternal(m_graphics, pipeline, *module, m_driver_cache);
		} catch (const std::exception& error) {
			PipelineCacheLog("Pipeline helpers: cs {:016x}: {}", module->info.shader_hash, error.what());
			pipeline = {};
		}
	}
	PublishComputePipeline(*module, pipeline, started);
}

void PipelineCache::PublishComputePipeline(const ShaderModule& module, const Pipeline& pipeline,
                                           std::chrono::steady_clock::time_point started) {
	const bool created = pipeline.pipeline != nullptr && pipeline.pipeline_layout != nullptr;
	if (created) {
		NoteCreatedPipeline(started);
	}
	std::lock_guard lock(m_async_mutex);
	if (created) {
		m_async_ready_compute.try_emplace(module.id, pipeline);
	} else {
		m_claimed_compute.erase(module.id);
	}
	m_async_pending_compute.erase(module.id);
}

std::pair<const ShaderRecompiler::IR::ResourcePlan*, const ShaderRecompiler::IR::ResourceSpecialization*>
PipelineCache::ProgramPlanOf(const ShaderModule& module) const {
	if (module.plan == nullptr) {
		return {nullptr, nullptr};
	}
	return {module.plan, &module.specialization};
}

bool PipelineCache::SpecializeDirectResources(const ShaderRecompiler::IR::ResourcePlan&     plan,
                                              ShaderRecompiler::IR::ResourceSnapshot&       snapshot,
                                              ShaderRecompiler::IR::ResourceSpecialization& specialization) {
	const ProgramLockGuard lock(m_program_lock);
	return ShaderRecompiler::IR::SpecializeDirectResources(plan, snapshot, specialization);
}

namespace {

constexpr uint32_t TableKeyMarker = 0x54424c45u;

}

template <typename InputInfo>
PipelineCache::TableProgram PipelineCache::GetTableProgram(
    const ShaderParams& params, const InputInfo& input_info,
    const ShaderRecompiler::IR::ResourceSpecialization& specialization, uint32_t push_data_cursor) {
	ProgramKey key;
	if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
		key.stage = input_info.logical_stage;
	} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
		key.stage = ShaderType::Pixel;
	} else {
		static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
		key.stage = ShaderType::Compute;
	}
	key.hash            = params.hash;
	key.user_data_count = params.user_data_count;
	BuildStageStaticKey(input_info, key.static_state);
	key.static_state.push_back(TableKeyMarker);
	const ProgramLockGuard lock(m_program_lock);
	auto&                  variants = m_table_programs[key];
	for (auto& variant: variants) {
		if (variant.push_data_cursor != push_data_cursor || !(variant.specialization == specialization)) {
			continue;
		}
		if (!variant.done) {
			return {.status = TableStatus::Pending, .waiting = variant.waiting};
		}
		if (!variant.module) {
			return {.status = TableStatus::Refused, .refusal = &variant.refusal};
		}
		return {.module = &*variant.module, .status = TableStatus::Ready};
	}
	auto& variant            = variants.emplace_back();
	variant.specialization   = specialization;
	variant.push_data_cursor = push_data_cursor;
	auto job                 = std::make_unique<CompileJob>();
	job->key                 = key;
	job->params              = params;
	job->push_data_cursor    = push_data_cursor;
	job->specialization      = specialization;
	job->table               = &variant;
	if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
		job->vertex = std::make_unique<ShaderVertexInputInfo>(input_info);
	} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
		job->pixel = std::make_unique<ShaderPixelInputInfo>(input_info);
	} else {
		job->compute = std::make_unique<ShaderComputeInputInfo>(input_info);
	}
	{
		std::lock_guard job_lock(m_job_mutex);
		m_jobs.push_back(std::move(job));
	}
	m_job_available.notify_one();
	return {.status = TableStatus::Pending, .waiting = variant.waiting};
}

template <typename InputInfo>
void PipelineCache::RunTableJob(CompileJob& job, InputInfo& input_info) {
	std::optional<ShaderModule> module;
	std::string                 refusal;
	std::vector<uint32_t>       spirv;
	std::vector<uint32_t>       plain_spirv;
	[&] {
		ShaderRecompiler::CompileOptions options {.wave_size = input_info.wave_size};
		const auto user_data = std::span(job.params.user_data).first(job.params.user_data_count);
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			options.stage             = input_info.logical_stage;
			options.input_info.vertex = &input_info;
			options.user_data_base    = 8;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			options.stage            = ShaderType::Pixel;
			options.input_info.pixel = &input_info;
		} else {
			options.stage              = ShaderType::Compute;
			options.input_info.compute = &input_info;
		}
		if (options.stage == ShaderType::Mesh || options.stage == ShaderType::TessellationControl ||
		    options.stage == ShaderType::Local || options.stage == ShaderType::TessellationEvaluation) {
			refusal = "mesh and tessellation stages are not served";
			return;
		}
		options.shader_hash        = job.params.hash;
		options.user_data          = user_data;
		options.back_code          = job.params.back_code;
		options.dump_ir            = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
		options.emit_debug_info    = Config::GraphicsDebugDumpEnabled();
		options.native_ray_tracing = false;
		options.table_mode         = true;
		auto source = ShaderRecompiler::PrepareShaderSource(job.params.code, options);
		if (!source.calls.empty() || source.unsupported_call_pc != UINT32_MAX) {
			refusal = "GPU-selected calls";
			return;
		}
		auto translated = ShaderRecompiler::TranslateProgram(source.decoded, options);
		auto built      = BuildModule(job.key, options, std::move(translated), *job.specialization, {},
		                              job.push_data_cursor, spirv, plain_spirv, &refusal);
		if (built.module != nullptr) {
			module = std::move(built);
		}
	}();
	LockPrograms();
	job.table->module  = std::move(module);
	job.table->refusal = job.table->module ? std::string {} : std::move(refusal);
	job.table->done    = true;
	job.table->waiting->store(true, std::memory_order_release);
	UnlockPrograms();
}

PipelineCache::TableProgram PipelineCache::GetComputeProgramTable(
    const HW::ComputeShaderInfo& regs, ShaderComputeInputInfo& input_info,
    const ShaderRecompiler::IR::ResourceSpecialization& specialization) {
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto params             = PrepareProgram(regs, input_info);
	input_info.lds_storage = input_info.lds_size_dwords * 4u >
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	const uint32_t push_data_cursor =
	    input_info.dispatch_indirect_threads ? ShaderRecompiler::IR::PushData::NoStart : 0u;
	return GetTableProgram(params, input_info, specialization, push_data_cursor);
}

PipelineCache::TableProgram PipelineCache::GetVertexProgramTable(
    const ShaderParams& params, const ShaderVertexInputInfo& input_info,
    const ShaderRecompiler::IR::ResourceSpecialization& specialization, uint32_t& push_data_cursor) {
	const auto result = GetTableProgram(params, input_info, specialization, push_data_cursor);
	if (result.module != nullptr) {
		result.module->info.bindings.AdvancePushData(push_data_cursor);
	}
	return result;
}

PipelineCache::TableProgram PipelineCache::GetPixelProgramTable(
    const ShaderParams& params, const ShaderPixelInputInfo& input_info,
    const ShaderRecompiler::IR::ResourceSpecialization& specialization, uint32_t& push_data_cursor) {
	const auto result = GetTableProgram(params, input_info, specialization, push_data_cursor);
	if (result.module != nullptr) {
		result.module->info.bindings.AdvancePushData(push_data_cursor);
	}
	return result;
}

const ShaderModule* PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                                     ShaderComputeInputInfo&      input_info,
                                                     bool                         guest_traversal) {
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
#if defined(ANYPS5_KYTY)
	ShaderParams params;
	if (!Common::Fallback::Run("draw or dispatch skipped: its shader stage could not be prepared", "PipelineCache::GetComputeProgram", 0, regs.cs_regs.data_addr,
	                           [&] { params = PrepareProgram(regs, input_info); })) {
		return nullptr;
	}
#else
	const auto        params      = PrepareProgram(regs, input_info);
#endif
	input_info.lds_storage = input_info.lds_size_dwords * 4u >
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	uint32_t push_data_cursor =
	    input_info.dispatch_indirect_threads ? ShaderRecompiler::IR::PushData::NoStart : 0u;
	const ProgramLockGuard lock(m_program_lock);
	m_guest_traversal             = guest_traversal;
	const auto*            module = GetProgram(params, input_info, push_data_cursor);
	m_guest_traversal             = false;
	return module;
}

const ShaderModule* PipelineCache::GetGuestComputeProgramQueued(const HW::ComputeShaderInfo& regs,
                                                                ShaderComputeInputInfo& input_info, bool& queued) {
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
#if defined(ANYPS5_KYTY)
	ShaderParams params;
	if (!Common::Fallback::Run("draw or dispatch skipped: its shader stage could not be prepared",
	                           "PipelineCache::GetGuestComputeProgramQueued", 0, regs.cs_regs.data_addr,
	                           [&] { params = PrepareProgram(regs, input_info); })) {
		return nullptr;
	}
#else
	const auto params = PrepareProgram(regs, input_info);
#endif
	input_info.lds_storage = input_info.lds_size_dwords * 4u >
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	uint32_t push_data_cursor =
	    input_info.dispatch_indirect_threads ? ShaderRecompiler::IR::PushData::NoStart : 0u;
	const ProgramLockGuard lock(m_program_lock);
	m_guest_traversal  = true;
	m_queue_only       = true;
	m_queued           = false;
	const auto* module = GetProgram(params, input_info, push_data_cursor);
	m_guest_traversal  = false;
	m_queue_only       = false;
	queued             = m_queued;
	return module;
}

void PipelineCache::NoteAheadGuestWrite(uint64_t sequence, uint64_t address,
                                        std::span<const uint32_t> data, bool one_address) {
	if (address == 0 || data.empty()) {
		return;
	}
	auto& state    = t_ahead_guest_writes;
	auto& write    = state.writes.emplace_back();
	write.sequence = sequence;
	write.address  = address;
	if (one_address) {
		write.dwords.assign(1, data.back());
	} else {
		write.dwords.assign(data.begin(), data.end());
	}
	state.Count(write, 1);
}

bool PipelineCache::AheadGuestWritesPending() {
	return !t_ahead_guest_writes.writes.empty();
}

void PipelineCache::BeginAheadThread(uint32_t slot) {
	ShaderRecompiler::IR::UsePlanScratchSlot(slot);
	t_ahead_unlocked = true;
}

#if defined(ANYPS5_KYTY)
bool PipelineCache::NarrowProgramLock() {
	return ProgramLock::Narrow();
}

void PipelineCache::BeginExecuteThread() {
	static std::atomic_flag claimed;
	if (!ProgramLock::Narrow() || claimed.test_and_set(std::memory_order_relaxed)) {
		return;
	}
	ShaderRecompiler::IR::UsePlanScratchSlot(2);
	t_execute_unlocked = true;
}
#endif

uint64_t PipelineCache::AheadWritesSequence() {
	return g_ahead_writes_sequence.load(std::memory_order_relaxed);
}

void PipelineCache::SyncAheadWrites(uint64_t sequence) {
	if (t_ahead_writes_copy == nullptr) {
		t_ahead_writes_copy = std::make_unique<AheadWrites>();
	}
	if (t_ahead_writes_copied == sequence) {
		return;
	}
	for (;;) {
		const auto before = g_ahead_writes_sequence.load(std::memory_order_acquire);
		if ((before & 1u) != 0) {
			Common::SpinPause();
			continue;
		}
		std::memcpy(static_cast<void*>(t_ahead_writes_copy.get()), &g_ahead_writes, sizeof(AheadWrites));
		std::atomic_thread_fence(std::memory_order_acquire);
		if (g_ahead_writes_sequence.load(std::memory_order_relaxed) == before) {
			t_ahead_writes_copied = before;
			return;
		}
	}
}

void PipelineCache::RetireAheadGuestWrites(uint64_t executed) {
	auto& state = t_ahead_guest_writes;
	while (!state.writes.empty() && state.writes.front().sequence <= executed) {
		state.Count(state.writes.front(), -1);
		state.writes.pop_front();
	}
}

const ShaderModule* PipelineCache::GetComputeProgramAhead(const HW::ComputeShaderInfo& regs,
                                                          ShaderComputeInputInfo&      input_info,
                                                          ProgramResolution&           ahead) {
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
#if defined(ANYPS5_KYTY)
	ShaderParams params;
	if (!Common::Fallback::Run("draw or dispatch skipped: its shader stage could not be prepared", "PipelineCache::GetComputeProgramAhead", 0, regs.cs_regs.data_addr,
	                           [&] { params = PrepareProgram(regs, input_info); })) {
		ahead.reads.Fail();
		return nullptr;
	}
#else
	const auto params             = PrepareProgram(regs, input_info);
#endif
	input_info.lds_storage = input_info.lds_size_dwords * 4u >
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	uint32_t push_data_cursor =
	    input_info.dispatch_indirect_threads ? ShaderRecompiler::IR::PushData::NoStart : 0u;
#if defined(ANYPS5_KYTY)
	if (!t_ahead_unlocked && ProgramLock::Narrow()) {
		SyncAheadWrites(AheadWritesSequence());
	}
#endif
#if defined(ANYPS5_KYTY)
	const ShaderModule* module = nullptr;
	{
		const ProgramLockGuard lock(m_program_lock, !t_ahead_unlocked);
		module = GetAhead(params, input_info, push_data_cursor, ahead);
	}
	if (module != nullptr) {
		RequestComputePipelineAhead(*module);
	}
	return module;
#else
	const ProgramLockGuard lock(m_program_lock, !t_ahead_unlocked);
	return GetAhead(params, input_info, push_data_cursor, ahead);
#endif
}

constexpr uint64_t PipelineSyncWords = 12000;
constexpr uint64_t PipelineSyncUiVsWords = 8000;
constexpr std::chrono::milliseconds PipelineWait {20};
constexpr uint32_t SmallTargetSync = 1024;

bool SmallColorTargetDraw(const HW::Context& context) {
	constexpr uint32_t limit = SmallTargetSync;
	const auto mask  = context.GetRenderTargetMask();
	bool       found = false;
	for (uint32_t slot = 0; slot < 8; slot++) {
		const auto& target = context.GetRenderTarget(slot);
		if (((mask >> (slot * 4u)) & 0x0fu) == 0 || target.base.addr == 0) {
			continue;
		}
		if (target.attrib2.width + 1u > limit || target.attrib2.height + 1u > limit) {
			return false;
		}
		found = true;
	}
	return found;
}

#if defined(ANYPS5_KYTY)
static const char* AsyncDrawName(PipelineCache::AsyncDraw async) {
	using AsyncDraw = PipelineCache::AsyncDraw;
	switch (async) {
		case AsyncDraw::Off: return "off";
		case AsyncDraw::WritesMemory: return "writes-memory";
		case AsyncDraw::LinearTarget: return "linear-target";
		case AsyncDraw::Occlusion: return "occlusion";
		case AsyncDraw::SmallTarget: return "small-target";
		case AsyncDraw::Eligible: return "eligible";
	}
	return "?";
}

constexpr std::array<const char*, 5> SkipClassHacks = {
    "draw skipped while its pipeline compiles",
    "draw counted by an occlusion query skipped while its pipeline compiles (KYTY_ASYNC_PIPELINES=2)",
    "draw into a small colour target skipped while its pipeline compiles (KYTY_ASYNC_PIPELINES=2)",
    "draw with a small pipeline skipped while it compiles (KYTY_ASYNC_PIPELINES=2)",
    "draw without depth and with small vertex programs skipped while its pipeline compiles "
    "(KYTY_ASYNC_PIPELINES=2)"};
#endif

#if defined(ANYPS5_KYTY)
void PipelineCache::NoteFirstWait(std::chrono::steady_clock::time_point started, bool ready) {
	const auto waited = std::chrono::steady_clock::now() - started;
	m_async_first_waits.fetch_add(1, std::memory_order_relaxed);
	m_async_first_wait_ns.fetch_add(
	    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(waited).count()),
	    std::memory_order_relaxed);
	if (ready) {
		m_async_first_ready.fetch_add(1, std::memory_order_relaxed);
	}
}
#endif

#if defined(ANYPS5_KYTY)
PipelineCache::AsyncDraw PipelineCache::HeldDraw(const CommandBuffer& command, const DrawRenderState& state,
                                                 const DrawTargets& targets) const {
	if (!Config::AsyncPipelinesEnabled()) {
		return AsyncDraw::Off;
	}
	const auto writes_memory = [](const ShaderStageRuntime& stage) {
		if (stage.module == nullptr) {
			return true;
		}
		const auto& program = stage.module->info;
		return program.has_address_writes ||
		       std::ranges::any_of(program.info.buffers, [](const auto& buffer) { return buffer.written; }) ||
		       std::ranges::any_of(program.info.images, [](const auto& image) { return image.written; });
	};
	for (const auto& stage: state.VertexStages()) {
		if (writes_memory(stage.stage)) {
			return AsyncDraw::WritesMemory;
		}
	}
	if (state.ps_active && writes_memory(state.ps_input_info.stage)) {
		return AsyncDraw::WritesMemory;
	}
	for (uint32_t i = 0; i < targets.color_count; i++) {
		if (!targets.colors[i].desc.info.IsTiled()) {
			return AsyncDraw::LinearTarget;
		}
	}
	if (static_cast<bool>(targets.depth.image_id) && !targets.depth.desc.info.IsTiled()) {
		return AsyncDraw::LinearTarget;
	}
	if (m_render != nullptr && m_render->GetOcclusionQueries().Counting()) {
		return AsyncDraw::Occlusion;
	}
	return SmallColorTargetDraw(command.GetRegisters()) ? AsyncDraw::SmallTarget : AsyncDraw::Off;
}

bool PipelineCache::LiveTargets(const DrawTargets& targets) const {
	if (m_render == nullptr) {
		return false;
	}
	const auto frame = m_render->FlipNumber();
	const auto drawn = [&](ImageId id) {
		const auto& stamps = m_render->GetTextureCache().GetImage(id).target_frames;
		return (frame >= 1u && stamps[(frame - 1u) & 3u] == frame) ||
		       (frame >= 2u && stamps[(frame - 2u) & 3u] == frame - 1u);
	};
	for (uint32_t i = 0; i < targets.color_count; i++) {
		if (!drawn(targets.colors[i].image_id)) {
			return false;
		}
	}
	return !static_cast<bool>(targets.depth.image_id) || drawn(targets.depth.image_id);
}
#endif

PipelineCache::Pipeline* PipelineCache::GetGraphicsPipeline(const CommandBuffer&   command,
                                                            const DrawRenderState& state,
                                                            const DrawTargets&     targets,
                                                            bool                   allow_async,
                                                            bool                   native_bindings) {
	const auto* colors        = targets.colors;
	const auto& depth         = targets.depth;
	const auto  vertex_info   = state.VertexStages();
	const auto& ps_input_info = state.ps_input_info;
	const auto& vs_input_info = vertex_info.front();
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);
#if defined(ANYPS5_KYTY)
	if (m_libraries != nullptr &&
	    (m_libraries->OptimizedReady() != m_optimized_seen ||
	     (!m_optimized_deferred.empty() &&
	      m_graphics_pipelines.size() + m_native_graphics_pipelines.size() != m_optimized_scanned))) {
		ApplyOptimizedPipelines();
	}
#endif

	const bool ps_active    = state.ps_active;
	const bool tessellation = vertex_info.size() == 3;
	std::array<const ShaderModule*, 4> stages {};
	for (uint32_t i = 0; i < vertex_info.size(); i++) {
		stages[i] = vertex_info[i].stage.module;
	}
	if (ps_active) {
		stages[3] = ps_input_info.stage.module;
		if (state.ps_plain && stages[3] != nullptr && stages[3]->plain != nullptr) {
			stages[3] = stages[3]->plain.get();
		}
	}
	const bool mesh        = stages[0]->info.stage == ShaderType::Mesh;
	const auto color_count = targets.color_count;

	auto&             ctx = command.GetRegisters();
#if defined(ANYPS5_KYTY)
	const auto packed_blend_slots = ps_active ? ShaderPixelPackedBlendSlots(ps_input_info) : 0u;
#endif

	const HW::ModeControl& mc = ctx.GetModeControl();

	GraphicsPipelineKey key {};
	for (uint32_t i = 0; i < stages.size(); i++) {
		if (stages[i] != nullptr) {
			key.stage_ids[i] = stages[i]->id;
		}
	}
	auto&    static_params      = key.static_params;
	auto&    rendering          = key.rendering;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		const auto slot = colors[i].target_slot;
		EXIT_IF(slot >= RENDER_COLOR_ATTACHMENTS_MAX);
		rendering.color_count = std::max(rendering.color_count, slot + 1);
		EXIT_IF(!colors[i].image_id || colors[i].desc.view_info.format == vk::Format::eUndefined);
		static_params.color_mask[slot] = colors[i].export_mapping.ApplyMask(
		    render_target_mask_slot(ctx.GetRenderTargetMask(), colors[i].target_slot));
		if (colors[i].desc.info.guest_format == Prospero::BufferFormat::k11_11_10UNorm) {
			static_params.color_mask[slot] = 1u;
		}
		rendering.color_formats[slot] = colors[i].desc.view_info.format;
		if (attachment_samples == 0) {
			attachment_samples = colors[i].desc.info.samples;
		} else if (attachment_samples != colors[i].desc.info.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, colors[i].desc.info.samples);
		}
		const auto& rt                        = ctx.GetRenderTarget(colors[i].target_slot);
		const auto& bc                        = ctx.GetBlendControl(colors[i].target_slot);
		auto alpha_source = ShaderAlphaBlendSource::None;
		if (slot == 0 && ps_active) {
			alpha_source = ps_input_info.alpha_blend_source;
		}
		static_params.blend_enable[slot] = bc.enable && !rt.info.blend_bypass;
#if defined(ANYPS5_KYTY)
		if (colors[i].desc.info.guest_format == Prospero::BufferFormat::k11_11_10UNorm) {
			static_params.blend_enable[slot] = false;
		}
		if ((packed_blend_slots & (1u << slot)) != 0u) {
			static_params.color_mask[slot] = 0u;
			rendering.color_formats[slot]  = vk::Format::eUndefined;
		}
#endif
		if (static_params.blend_enable[slot] && alpha_source == ShaderAlphaBlendSource::None &&
		    ClassifyBlendMapping(bc, colors[i].export_mapping) != BlendMappingSupport::Direct) {
			static_params.blend_enable[slot] = false;
			static std::atomic_bool warned = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Warning: blending disabled for unsupported color mapping "
				    "(slot={} mapping=0x{:02x} color={}/{} alpha={}/{} separate={}).\n",
				    slot, colors[i].export_mapping.packed, bc.color_srcblend, bc.color_destblend,
				    bc.alpha_srcblend, bc.alpha_destblend, bc.separate_alpha_blend ? 1 : 0));
			}
		}
		if (static_params.blend_enable[slot]) {
			auto blend = bc;
			switch (alpha_source) {
				case ShaderAlphaBlendSource::SourceAlpha:
					blend.color_srcblend  = RemapSourceAlphaFactor(blend.color_srcblend);
					blend.color_destblend = RemapSourceAlphaFactor(blend.color_destblend);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::SourceAlphaOne:
				case ShaderAlphaBlendSource::SourceAlphaZero:
					blend.color_srcblend = static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Color);
					blend.color_destblend =
					    static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::None: break;
			}
			static_params.color_srcblend[slot]       = blend.color_srcblend;
			static_params.color_comb_fcn[slot]       = blend.color_comb_fcn;
			static_params.color_destblend[slot]      = blend.color_destblend;
			static_params.separate_alpha_blend[slot] = blend.separate_alpha_blend;
			if (blend.separate_alpha_blend) {
				static_params.alpha_srcblend[slot]  = blend.alpha_srcblend;
				static_params.alpha_comb_fcn[slot]  = blend.alpha_comb_fcn;
				static_params.alpha_destblend[slot] = blend.alpha_destblend;
			}
		}
	}
	const bool with_depth =
	    depth.desc.view_info.format != vk::Format::eUndefined && static_cast<bool>(depth.image_id);
	if (with_depth) {
		const auto aspects       = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		rendering.depth_format   = aspects & vk::ImageAspectFlagBits::eDepth
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		rendering.stencil_format = aspects & vk::ImageAspectFlagBits::eStencil
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		if (attachment_samples == 0) {
			attachment_samples = depth.desc.info.samples;
		} else if (attachment_samples != depth.desc.info.samples) {
			EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
			     depth.desc.info.samples);
		}
	}
	if (color_count == 0 && !with_depth) {
		attachment_samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
		EXIT_IF(!static_cast<bool>(
		    m_graphics.GetPhysicalDeviceProperties().limits.framebufferNoAttachmentsSampleCounts &
		    vulkan_sample_count(attachment_samples)));
	}
	EXIT_IF(attachment_samples == 0 ||
	        vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {});

	const auto& clip_control               = ctx.GetClipControl();
	static_params.negative_one_to_one      = !clip_control.dx_clip_space;
	static_params.depth_clip_enable        = clip_control.IsZClipEnabled();
	static_params.topology                 = state.topology;
	static_params.primitive_restart_enable = state.primitive_restart_enable;
	static_params.samples                  = attachment_samples;
	static_params.sample_shading_enable =
	    ps_active && attachment_samples > 1 && ps_input_info.ps_sample_shading;
	static_params.alpha_to_coverage_enable = ps_active && ctx.GetShaderRegisters().AlphaToMask();
	const bool rect_list = Prospero::IsRectList(command.GetUserConfig().GetPrimType());
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	static_params.polygon_mode =
	    ResolvePolygonMode(mc, !rect_list && mc.cull_front, !rect_list && mc.cull_back);
	if (tessellation) {
		static_params.patch_control_points = vs_input_info.tess.input_control_points;
#if defined(ANYPS5_KYTY)
		static_params.merged_hull = vs_input_info.tess.merged_hull;
#endif
	} else if (!mesh && static_params.topology == vk::PrimitiveTopology::ePatchList) {
		static_params.patch_control_points = 3;
	}

	if (!mesh) {
		EXIT_IF(vs_input_info.buffers_num < 0 ||
		        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		        vs_input_info.resources_num < 0 ||
		        vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX);
		key.vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
		key.vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
		for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
			const auto& buffer = vs_input_info.buffers[binding];
			key.vertex_input.bindings[binding] = {.stride   = buffer.stride,
			                                      .instance = buffer.fetch_index != 0};
		}
		for (int attribute = 0; attribute < vs_input_info.resources_num; attribute++) {
			const auto binding = vs_input_info.resources_dst[attribute].buffer_index;
			EXIT_IF(binding < 0 || binding >= vs_input_info.buffers_num);
			key.vertex_input.attributes[attribute] = {
			    .offset = static_cast<uint32_t>(vs_input_info.resources[attribute].Base48() -
			                                    vs_input_info.buffers[binding].addr),
			    .binding = static_cast<uint8_t>(binding),
			};
		}
	}

	if (native_bindings) {
		if (auto iter = m_native_graphics_pipelines.find(key); iter != m_native_graphics_pipelines.end()) {
			return &iter->second;
		}
		if (allow_async && !m_async_threads.empty()) {
			std::unique_lock lock(m_async_mutex);
			auto             found = m_async_native.find(key);
			if (found == m_async_native.end()) {
				found = m_async_native.try_emplace(key).first;
				m_async_jobs.push_back({.native = true, .key = key, .stages = stages});
				m_pipeline_waiting = found->second.done;
				lock.unlock();
				m_async_available.notify_one();
				return nullptr;
			}
			if (!found->second.done->load(std::memory_order_acquire)) {
				m_pipeline_waiting = found->second.done;
				return nullptr;
			}
			const auto created = found->second.pipeline;
			m_async_native.erase(found);
			lock.unlock();
			if (created.pipeline != nullptr && created.pipeline_layout != nullptr) {
				return &m_native_graphics_pipelines.try_emplace(key, created).first->second;
			}
		}
		auto& cached           = m_native_graphics_pipelines[key];
		cached.native_bindings = true;
#if defined(ANYPS5_KYTY)
		if (!CreateGraphicsPipelineFast(cached, key, stages))
#endif
			CreatePipelineInternal(m_graphics, cached, key, stages, m_driver_cache);
		EXIT_NOT_IMPLEMENTED(cached.pipeline == nullptr || cached.pipeline_layout == nullptr);
		return &cached;
	}
	if (m_last_graphics_pipeline != nullptr && key == m_last_graphics_key) {
		return m_last_graphics_pipeline;
	}
	if (const auto found = m_graphics_pipelines.find(key); found != m_graphics_pipelines.end()) {
		m_last_graphics_key      = key;
		m_last_graphics_pipeline = &found->second;
		if (!found->second.use_noted) {
			NoteUse(found->second, stages);
		}
		return &found->second;
	}
#if defined(ANYPS5_KYTY)
	const AsyncDraw held      = allow_async ? AsyncDraw::Eligible : HeldDraw(command, state, targets);
	const bool      skip_more = Config::AsyncPipelinesSkipMore() &&
	                       (allow_async || held == AsyncDraw::Occlusion || held == AsyncDraw::SmallTarget) &&
	                       LiveTargets(targets);
	allow_async = allow_async || skip_more;
	uint32_t    skip_class = held == AsyncDraw::Occlusion ? 1u : held == AsyncDraw::SmallTarget ? 2u : 0u;
	const char* sync_why   = AsyncDrawName(held);
#endif
	if (allow_async) {
		uint64_t vertex_words = 0;
		for (uint32_t i = 0; i < 3u; i++) {
			if (stages[i] != nullptr) {
				vertex_words += stages[i]->spirv_words;
			}
		}
		const uint64_t spirv_words =
		    vertex_words + (stages[3] != nullptr ? stages[3]->spirv_words : 0u);
		const bool layer_draw = !depth.depth_test_enable && !depth.depth_write_enable &&
		                        vertex_words <= PipelineSyncUiVsWords;
		const bool small_program = spirv_words <= PipelineSyncWords;
#if defined(ANYPS5_KYTY)
		if (small_program || layer_draw) {
			sync_why = small_program ? "small-program" : "layer";
			if (skip_class == 0) {
				skip_class = small_program ? 3u : 4u;
			}
		}
		allow_async = (!small_program && !layer_draw) || skip_more;
#else
		allow_async = !small_program && !layer_draw;
#endif
	}
	std::optional<StallScope> stall;
#if defined(ANYPS5_KYTY)
	const uint64_t stall_hash = stages[3] != nullptr ? stages[3]->info.shader_hash : stages[0]->info.shader_hash;
#endif
	if (m_async_handoff) {
#if defined(ANYPS5_KYTY)
		if (allow_async && m_async_threads.empty()) {
			sync_why = "no-threads";
		}
#endif
		allow_async = allow_async && !m_async_threads.empty();
		std::unique_lock lock(m_async_mutex);
		std::optional<std::chrono::steady_clock::time_point> wait_until;
#if defined(ANYPS5_KYTY)
		std::chrono::steady_clock::time_point wait_from;
#endif
		for (;;) {
			if (auto node = m_async_ready.extract(key); !node.empty()) {
				lock.unlock();
#if defined(ANYPS5_KYTY)
				if (wait_until) {
					NoteFirstWait(wait_from, true);
				}
#endif
				auto& ready              = m_graphics_pipelines.insert(std::move(node)).position->second;
				m_last_graphics_key      = key;
				m_last_graphics_pipeline = &ready;
				StorePipeline(key);
				NoteUse(ready, stages);
				return &ready;
			}
			if (!m_async_pending_graphics.contains(key)) {
				if (!allow_async || m_async_failed_graphics.contains(key)) {
#if defined(ANYPS5_KYTY)
					if (allow_async) {
						sync_why = "failed";
					}
#endif
					ClaimPipeline(key);
					QueueSiblingPipeline(key, stages, ps_active ? ps_input_info.stage.module : nullptr);
					break;
				}
#if defined(ANYPS5_KYTY)
				const bool skip_now = skip_more;
				if (skip_now) {
					lock.unlock();
					Pipeline   created;
					const auto probe_started = std::chrono::steady_clock::now();
					bool       made          = false;
					{
						const StallScope probe(*this, "graphics-probe", stall_hash, sync_why);
						made = CreateGraphicsPipelineNow(created, key, stages);
					}
					m_async_probes.fetch_add(1, std::memory_order_relaxed);
					m_async_probe_ns.fetch_add(
					    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
					                              std::chrono::steady_clock::now() - probe_started)
					                              .count()),
					    std::memory_order_relaxed);
					lock.lock();
					if (made) {
						ClaimPipeline(key);
						QueueSiblingPipeline(key, stages, ps_active ? ps_input_info.stage.module : nullptr);
						lock.unlock();
						m_async_created_now.fetch_add(1, std::memory_order_relaxed);
						auto& ready              = m_graphics_pipelines.try_emplace(key, created).first->second;
						m_last_graphics_key      = key;
						m_last_graphics_pipeline = &ready;
						StorePipeline(key);
						NoteCreatedPipeline(probe_started);
						NoteUse(ready, stages);
						return &ready;
					}
				}
#endif
				m_async_pending_graphics.insert(key);
				ClaimPipeline(key);
				QueueSiblingPipeline(key, stages, ps_active ? ps_input_info.stage.module : nullptr);
				lock.unlock();
				QueueAsyncPipeline({.key = key, .stages = stages});
#if defined(ANYPS5_KYTY)
				wait_from  = std::chrono::steady_clock::now();
				wait_until = skip_now ? wait_from : wait_from + PipelineWait;
#else
				wait_until = std::chrono::steady_clock::now() + PipelineWait;
#endif
				lock.lock();
				continue;
			}
			if (allow_async) {
				if (const auto sibling = std::ranges::find(m_async_sibling_jobs, key, &AsyncPipelineJob::key);
				    sibling != m_async_sibling_jobs.end()) {
					m_async_jobs.push_back(*sibling);
					m_async_sibling_jobs.erase(sibling);
					m_async_available.notify_one();
				}
				if (!wait_until || std::chrono::steady_clock::now() >= *wait_until) {
#if defined(ANYPS5_KYTY)
					lock.unlock();
					if (wait_until) {
						NoteFirstWait(wait_from, false);
					}
					m_async_skipped[skip_class].fetch_add(1, std::memory_order_relaxed);
					if (m_render != nullptr && m_render->GetOcclusionQueries().Counting()) {
						m_render->GetOcclusionQueries().CountSkippedDraw();
						Common::HackNotice::Report("occlusion query of a skipped draw reads visible",
						                           "PipelineCache::GetGraphicsPipeline", stall_hash, skip_class);
					}
					if (skip_class != 0) {
						Common::HackNotice::Report(SkipClassHacks[skip_class], "PipelineCache::GetGraphicsPipeline",
						                           stall_hash, skip_class);
					}
#endif
					return nullptr;
				}
				lock.unlock();
				std::this_thread::yield();
				lock.lock();
				continue;
			}
			if (TakeQueuedAsyncPipeline([&](const AsyncPipelineJob& job) {
				    return !job.compute && !job.native && job.key == key;
			    })) {
				m_async_pending_graphics.erase(key);
				ClaimPipeline(key);
				break;
			}
			if (!stall) {
#if defined(ANYPS5_KYTY)
				stall.emplace(*this, "graphics-wait", stall_hash, sync_why);
#else
				stall.emplace(*this);
#endif
			}
			lock.unlock();
			std::this_thread::yield();
			lock.lock();
		}
	}
	auto& pipeline = m_graphics_pipelines[key];

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(vs_input_info);
		if (ps_active) {
			ShaderDbgDumpInputInfo(ps_input_info);
		}
	}

	const auto started = std::chrono::steady_clock::now();
	{
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::PipelinesCreated,
		                                         Common::DebugCounters::Counter::PipelineCreateNs,
		                                         Common::DebugCounters::Peak::PipelineCreateNs);
		if (!stall) {
#if defined(ANYPS5_KYTY)
			stall.emplace(*this, "graphics-sync", stall_hash, sync_why);
#else
			stall.emplace(*this);
#endif
		}
#if defined(ANYPS5_KYTY)
		if (!CreateGraphicsPipelineFast(pipeline, key, stages))
#endif
			CreatePipelineInternal(m_graphics, pipeline, key, stages, m_driver_cache);
	}
	StorePipeline(key);
	NoteCreatedPipeline(started);
	NoteUse(pipeline, stages);
	m_last_graphics_key      = key;
	m_last_graphics_pipeline = &pipeline;

	return &pipeline;
}

#if defined(ANYPS5_KYTY)
bool PipelineCache::CreateGraphicsPipelineNow(Pipeline& pipeline, const GraphicsPipelineKey& key,
                                              const std::array<const ShaderModule*, 4>& stages) {
	const bool cached = m_graphics.pipeline_cache_lookup_enabled && m_driver_cache != nullptr;
	const bool linked = m_libraries != nullptr && PipelineLibraries::Eligible(stages);
	if (!cached && !linked) {
		return false;
	}
	CreateGraphicsPipelineLayout(m_graphics, pipeline, key, stages);
	if (cached && CreateGraphicsPipelineWhole(m_graphics, pipeline, key, stages, m_driver_cache,
	                                          vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequired,
	                                          pipeline.pipeline) == vk::Result::eSuccess &&
	    pipeline.pipeline) {
		return true;
	}
	if (linked && m_libraries->Link(pipeline, key, stages, true)) {
		m_libraries->QueueOptimized(pipeline, key, stages);
		return true;
	}
	m_graphics.device.destroyPipeline(pipeline.pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(pipeline.pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout, nullptr);
	pipeline = {};
	return false;
}

bool PipelineCache::CreateGraphicsPipelineFast(Pipeline& pipeline, const GraphicsPipelineKey& key,
                                               const std::array<const ShaderModule*, 4>& stages) {
	auto& route = PipelineRouteNote();
	if (m_libraries == nullptr) {
		route = "whole";
		return false;
	}
	if (!PipelineLibraries::Eligible(stages)) {
		route = "whole-no-pixel";
		return false;
	}
	CreateGraphicsPipelineLayout(m_graphics, pipeline, key, stages);
	if (m_graphics.pipeline_cache_lookup_enabled && m_driver_cache != nullptr) {
		vk::Pipeline cached = nullptr;
		if (CreateGraphicsPipelineWhole(m_graphics, pipeline, key, stages, m_driver_cache,
		                                vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequired,
		                                cached) == vk::Result::eSuccess &&
		    cached) {
			pipeline.pipeline = cached;
			route             = "driver-cache";
			return true;
		}
	}
	if (m_libraries->Link(pipeline, key, stages)) {
		m_libraries->QueueOptimized(pipeline, key, stages);
		return true;
	}
	route = "whole-failed-link";
	const auto result = CreateGraphicsPipelineWhole(m_graphics, pipeline, key, stages, m_driver_cache, {},
	                                                pipeline.pipeline);
	if (result != vk::Result::eSuccess) {
		EXIT("vkCreateGraphicsPipelines failed: %s, vs 0x%016" PRIx64 ", ps 0x%016" PRIx64 "\n",
		     vk::to_string(result).c_str(), stages[0]->info.shader_hash, stages[3]->info.shader_hash);
	}
	return true;
}

void PipelineCache::ApplyOptimizedPipelines() {
	if (const auto ready = m_libraries->OptimizedReady(); ready != m_optimized_seen) {
		m_optimized_seen = ready;
		for (auto& optimized: m_libraries->TakeOptimized()) {
			m_optimized_deferred.push_back(optimized);
		}
	}
	std::erase_if(m_optimized_deferred, [&](const OptimizedGraphicsPipeline& optimized) {
		auto&      pipelines = optimized.native ? m_native_graphics_pipelines : m_graphics_pipelines;
		const auto found     = pipelines.find(optimized.key);
		if (found == pipelines.end()) {
			return false;
		}
		m_retired_pipelines.push_back(found->second.pipeline);
		found->second.pipeline = optimized.pipeline;
		return true;
	});
	m_optimized_scanned = m_graphics_pipelines.size() + m_native_graphics_pipelines.size();
}

bool PipelineCache::PreparesGraphicsLibraries() const {
	return m_libraries != nullptr && m_libraries->Active() && PipelineLibrariesAhead();
}

void PipelineCache::PrepareGraphicsLibraries(const HW::Context& context, const HW::UserConfig& user_config,
                                             const DrawRenderState& state) {
	const auto                         vertex_info = state.VertexStages();
	std::array<const ShaderModule*, 4> stages {};
	for (uint32_t i = 0; i < vertex_info.size(); i++) {
		stages[i] = vertex_info[i].stage.module;
	}
	if (state.ps_active) {
		stages[3] = state.ps_input_info.stage.module;
	}
	struct Last {
		std::array<const ShaderModule*, 4> stages {};
		HW::ModeControl                    mode;
		HW::ClipControl                    clip;
		Prospero::PrimitiveType            primitive {};
		uint32_t                           target_mask = 0;
		uint32_t                           target0     = 0;
	};
	thread_local Last last;
	const auto&       mode        = context.GetModeControl();
	const auto&       clip        = context.GetClipControl();
	const auto        primitive   = user_config.GetPrimType();
	const auto        target_mask = context.GetRenderTargetMask();
	const auto        target0     = context.GetRenderTarget(0).attrib.num_fragments;
	if (last.stages == stages && std::memcmp(&last.mode, &mode, sizeof(mode)) == 0 &&
	    std::memcmp(&last.clip, &clip, sizeof(clip)) == 0 && last.primitive == primitive &&
	    last.target_mask == target_mask && last.target0 == target0) {
		return;
	}
	last.stages      = stages;
	last.mode        = mode;
	last.clip        = clip;
	last.primitive   = primitive;
	last.target_mask = target_mask;
	last.target0     = target0;
	if (stages[3] != nullptr && stages[3]->plain != nullptr && RecordsNoMipStats(state.ps_input_info.stage)) {
		stages[3] = stages[3]->plain.get();
	}
	if (!PipelineLibraries::Eligible(stages)) {
		return;
	}
	GraphicsPipelineKey key {};
	for (uint32_t i = 0; i < stages.size(); i++) {
		key.stage_ids[i] = stages[i] != nullptr ? stages[i]->id : 0;
	}
	auto& params = key.static_params;
	params       = LibraryStaticParameters(context, user_config, vertex_info.size() == 3,
	                                       vertex_info[0].tess, state.ps_active,
	                                       state.ps_input_info.ps_sample_shading);
	struct Seen {
		std::array<uint64_t, 4>  stage_ids;
		PipelineStaticParameters params;
	};
	Seen seen {};
	seen.stage_ids = key.stage_ids;
	seen.params    = params;
	thread_local std::array<uint64_t, 4096> prepared {};
	const auto hash  = XXH3_64bits(&seen, sizeof(seen)) | 1u;
	auto&      entry = prepared[hash & (prepared.size() - 1u)];
	if (entry == hash) {
		return;
	}
	entry = hash;
	m_libraries->Prepare(key, stages, false);
}
#endif

PipelineCache::Pipeline* PipelineCache::TryGetComputePipeline(const ShaderComputeInputInfo& input_info) {
	const auto& module = *input_info.stage.module;
	if (const auto found = m_compute_pipelines.find(module.id); found != m_compute_pipelines.end()) {
		return &found->second;
	}
	if (m_async_threads.empty()) {
		return &GetComputePipeline(input_info);
	}
	{
		std::unique_lock lock(m_async_mutex);
		if (!m_async_ready_compute.contains(module.id)) {
			if (m_async_pending_compute.contains(module.id)) {
				return nullptr;
			}
			if (m_async_asked_compute.insert(module.id).second) {
				m_async_pending_compute.insert(module.id);
				ClaimComputePipeline(module.id);
				lock.unlock();
				QueueAsyncPipeline({.compute = true, .module = &module});
				return nullptr;
			}
		}
	}
	return &GetComputePipeline(input_info);
}

PipelineCache::Pipeline& PipelineCache::GetComputePipeline(const ShaderComputeInputInfo& input_info) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);

	const auto& module = *input_info.stage.module;

	const ShaderModule* const stages[] = {&module};
	if (const auto found = m_compute_pipelines.find(module.id); found != m_compute_pipelines.end()) {
		if (!found->second.use_noted) {
			NoteUse(found->second, stages);
		}
		return found->second;
	}
#if defined(ANYPS5_KYTY)
	std::atomic_ref<bool>(module.pipeline_requested).store(true, std::memory_order_relaxed);
#endif
	std::optional<StallScope> stall;
	if (m_async_handoff) {
		std::unique_lock lock(m_async_mutex);
		for (;;) {
			if (auto node = m_async_ready_compute.extract(module.id); !node.empty()) {
				lock.unlock();
				auto& ready = m_compute_pipelines.insert(std::move(node)).position->second;
				NoteUse(ready, stages);
				return ready;
			}
			if (!m_async_pending_compute.contains(module.id)) {
				ClaimComputePipeline(module.id);
				break;
			}
			if (TakeQueuedAsyncPipeline([&](const AsyncPipelineJob& job) {
				    return job.compute && job.module->id == module.id;
			    })) {
				m_async_pending_compute.erase(module.id);
				ClaimComputePipeline(module.id);
				break;
			}
			bool in_helper = false;
			if (const auto job = m_helper_jobs.find(module.id); job != m_helper_jobs.end()) {
				if (m_helpers->Withdraw(module.id)) {
					m_helper_jobs.erase(job);
					m_async_pending_compute.erase(module.id);
					ClaimComputePipeline(module.id);
					m_helper_taken_back++;
					break;
				}
				in_helper = true;
			}
			if (!stall) {
#if defined(ANYPS5_KYTY)
				stall.emplace(*this, "compute-wait", module);
#else
				stall.emplace(*this);
#endif
			}
			lock.unlock();
			if (in_helper) {
				std::this_thread::sleep_for(HelperPoll);
			} else {
				std::this_thread::yield();
			}
			lock.lock();
		}
	}
	auto& pipeline = m_compute_pipelines[module.id];

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(input_info);
	}

	const auto started = std::chrono::steady_clock::now();
	{
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::PipelinesCreated,
		                                         Common::DebugCounters::Counter::PipelineCreateNs,
		                                         Common::DebugCounters::Peak::PipelineCreateNs);
		if (!stall) {
#if defined(ANYPS5_KYTY)
			stall.emplace(*this, "compute-sync", module);
#else
			stall.emplace(*this);
#endif
		}
		CreatePipelineInternal(m_graphics, pipeline, module, m_driver_cache);
	}
	NoteCreatedPipeline(started);
	NoteUse(pipeline, stages);

	return pipeline;
}
}
