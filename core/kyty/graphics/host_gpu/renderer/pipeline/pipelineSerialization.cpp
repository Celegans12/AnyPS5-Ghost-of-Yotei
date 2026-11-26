#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineStoreFormat.h"

#include "common/parallelFor.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"

#include <algorithm>
#include <cstring>
#include <xxhash.h>
#if defined(ANYPS5_KYTY)
#include "kytyShaderStoreStamp.h"

#include <cstdlib>
#endif
#include <zstd.h>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

namespace Libs::Graphics {

using ShaderRecompiler::IR::IoAll;
using ShaderRecompiler::IR::Reader;
using ShaderRecompiler::IR::Writer;

using namespace ShaderStoreFormat;

#if defined(ANYPS5_KYTY)
uint64_t ShaderStoreBuildStamp() {
	static const uint64_t stamp = [] {
		if (const char* value = std::getenv("KYTY_SHADER_CACHE_STAMP"); value != nullptr) {
			char*      end    = nullptr;
			const auto parsed = std::strtoull(value, &end, 0);
			if (end != value && *end == '\0') {
				return static_cast<uint64_t>(parsed);
			}
		}
		return static_cast<uint64_t>(KYTY_SHADER_STORE_STAMP);
	}();
	return stamp;
}
#else
uint64_t ShaderStoreBuildStamp() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	std::array<wchar_t, 32768> name {};
	GetModuleFileNameW(nullptr, name.data(), static_cast<DWORD>(name.size()));
	const std::filesystem::path path = name.data();
#elif defined(__APPLE__)
	std::array<char, 4096> name {};
	auto size = static_cast<uint32_t>(name.size());
	_NSGetExecutablePath(name.data(), &size);
	const std::filesystem::path path = name.data();
#else
	const std::filesystem::path path = "/proc/self/exe";
#endif
	std::error_code error;
	const auto      time = std::filesystem::last_write_time(path, error).time_since_epoch().count();
	return error ? 0 : static_cast<uint64_t>(time);
}
#endif

std::vector<uint8_t> StoreHeader(const GraphicContext& graphics, uint64_t stamp) {
	if (stamp == 0) {
		return {};
	}
	Writer header;
	const auto profile = SpirvProfile(graphics);
	const auto limit = graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	IoStoreHeader(header, stamp, profile, limit);
	return std::move(header.bytes);
}

std::string StoreMismatch(std::span<const uint8_t> bytes, std::span<const uint8_t> header, uint64_t current_stamp) {
	uint64_t stamp = 0;
	Reader   reader {bytes};
	IoAll(reader, stamp);
	if (reader.ok && stamp != current_stamp) {
		return fmt::format("it was written by another build (stamp {:016x}, this build {:016x})", stamp,
		                   current_stamp);
	}
	if (bytes.size() < header.size()) {
		return fmt::format("its header is truncated ({} of {} bytes)", bytes.size(), header.size());
	}
	return "its header was written for another device, other shader settings or another store layout";
}

void ShaderStoreWriter::Open(const std::filesystem::path& path, std::span<const uint8_t> header,
                             uint64_t kept) {
	Common::File::CreateDirectories(path.parent_path());
	if (kept == 0) {
		if (m_file.Create(path)) {
			m_file.Write(header.data(), static_cast<uint32_t>(header.size()));
		}
	} else if (m_file.Open(path, Common::File::Mode::ReadWrite)) {
		m_file.Truncate(kept);
		m_file.Seek(kept);
	}
}

void ShaderStoreWriter::Append(std::vector<uint8_t>&& record) {
	Post([this, record = std::move(record)] {
		std::vector<uint8_t> bytes(sizeof(Frame) + ZSTD_compressBound(record.size()));
		const auto size = ZSTD_compress(bytes.data() + sizeof(Frame), bytes.size() - sizeof(Frame),
		                                record.data(), record.size(), 3);
		EXIT_IF(ZSTD_isError(size));
		const Frame frame {.compressed_size = static_cast<uint32_t>(size),
		                   .raw_size        = static_cast<uint32_t>(record.size()),
		                   .checksum        = XXH3_64bits(bytes.data() + sizeof(Frame), size)};
		std::memcpy(bytes.data(), &frame, sizeof(frame));
		m_file.Write(bytes.data(), static_cast<uint32_t>(sizeof(Frame) + size));
	});
}

struct PipelineCache::StoreRecord {
	Record                             type = Record::Module;
	ProgramKey                         key;
	ShaderRecompiler::IR::ResourcePlan plan;
	bool                               plan_loaded = false;
	ShaderModule                       module;
	GraphicsPipelineKey                pipeline {};
	uint64_t                           layout_key = 0;
	ShaderTessellationInputInfo        layout;
};

size_t PipelineCache::RestoreStoreRecords(std::span<const uint8_t> bytes, std::span<const uint8_t> header) {
	AssumeProgramsConfined();
	auto       records  = bytes;
	const bool reusable =
	    records.size() >= header.size() && std::ranges::equal(records.first(header.size()), header);
	if (reusable) {
		records = records.subspan(header.size());
		std::vector<std::span<const uint8_t>> frames;
		for (auto rest = records; rest.size() >= sizeof(Frame);) {
			Frame frame;
			std::memcpy(&frame, rest.data(), sizeof(frame));
			if (rest.size() - sizeof(frame) < frame.compressed_size) {
				break;
			}
			frames.push_back(rest.first(sizeof(frame) + frame.compressed_size));
			rest = rest.subspan(frames.back().size());
		}
		std::vector<StoreRecord> loaded(frames.size());
		std::vector<uint8_t>     read(frames.size());
		Common::ParallelFor(frames.size(), [&](size_t i) { read[i] = ReadRecord(frames[i], loaded[i]) ? 1 : 0; });
		size_t kept = 0;
		while (kept < frames.size() && read[kept] != 0 && AddRecord(loaded[kept])) {
			records = records.subspan(frames[kept].size());
			kept++;
		}
		for (size_t i = kept + 1; i < frames.size(); i++) {
			if (read[i] != 0 && loaded[i].type == Record::Module) {
				DestroyModule(loaded[i].module);
			}
		}
	}
	return reusable ? bytes.size() - records.size() : 0;
}

bool PipelineCache::ReadRecord(std::span<const uint8_t> frame_bytes, StoreRecord& record) const {
	Frame frame;
	std::memcpy(&frame, frame_bytes.data(), sizeof(frame));
	const auto compressed = frame_bytes.subspan(sizeof(frame), frame.compressed_size);
	std::vector<uint8_t> raw;
	if (!DecodeShaderStoreFrame(frame, compressed, raw)) {
		return false;
	}
	Reader     reader {raw};
	const auto complete = [&] { return reader.ok && reader.bytes.empty(); };
	IoAll(reader, record.type);
	switch (record.type) {
		case Record::Module: {
			std::vector<uint8_t>  plan;
			auto&                 module = record.module;
			std::vector<uint32_t> spirv;
			std::vector<uint32_t> plain_spirv;
			IoStoreModule(reader, record.key, module, plan, spirv, plain_spirv);
			if (!complete()) {
				return false;
			}
			record.plan_loaded = ShaderRecompiler::IR::LoadResourcePlan(plan, record.plan);
			module.module      = CompileSPV(spirv, m_graphics.device);
			module.spirv_words = static_cast<uint32_t>(spirv.size());
			if (!plain_spirv.empty()) {
				module.plain = MakePlainModule(module, plain_spirv, m_graphics.device);
				if (module.plain == nullptr) {
					m_graphics.device.destroyShaderModule(module.module, nullptr);
					return false;
				}
			}
			return true;
		}
		case Record::GraphicsPipeline: {
			reader.Raw(&record.pipeline, sizeof(record.pipeline));
			return complete();
		}
		case Record::TessellationLayout: {
			IoAll(reader, record.layout_key, record.layout);
			return complete();
		}
		default: return false;
	}
}

bool PipelineCache::AddRecord(StoreRecord& record) {
	switch (record.type) {
		case Record::Module: {
			const auto [program, inserted] = m_program_cache.try_emplace(std::move(record.key));
			if (inserted) {
				if (!record.plan_loaded) {
					m_program_cache.erase(program);
					DestroyModule(record.module);
					return false;
				}
				program->second.resource_plan = std::move(record.plan);
			}
			program->second.modules.push_back(std::move(record.module));
			program->second.modules.back().plan = &program->second.resource_plan;
			break;
		}
		case Record::GraphicsPipeline: {
			m_graphics_pipelines.try_emplace(record.pipeline);
			break;
		}
		case Record::TessellationLayout: {
			m_tess_layouts.try_emplace(record.layout_key, record.layout);
			break;
		}
	}
	return true;
}

void PipelineCache::StoreModule(const ProgramKey& key, const Program& program,
                                const ShaderModule& module, const std::vector<uint32_t>& spirv,
                                const std::vector<uint32_t>& plain_spirv) {
	if (!m_store) {
		return;
	}
	std::vector<uint8_t> record;
	if (EncodeModuleRecord(key.stage, key.hash, key.user_data_count, key.static_state,
	                       program.resource_plan, module, spirv, plain_spirv, record)) {
		m_store.Append(std::move(record));
	}
}

void PipelineCache::StorePipeline(const GraphicsPipelineKey& key) {
	if (!m_store) {
		return;
	}
	Writer record;
	IoAll(record, Record::GraphicsPipeline);
	record.Raw(&key, sizeof(key));
	m_store.Append(std::move(record.bytes));
}

void PipelineCache::StoreLayout(uint64_t key, const ShaderTessellationInputInfo& layout) {
	RecordLayout(key, layout);
	if (!m_store) {
		return;
	}
	Writer record;
	IoAll(record, Record::TessellationLayout, key, layout);
	m_store.Append(std::move(record.bytes));
}

#if defined(ANYPS5_KYTY)
std::vector<uint8_t> ComputeShaderStoreHeader(const GraphicContext& graphics) {
	return StoreHeader(graphics, ShaderStoreBuildStamp());
}

#endif

}
