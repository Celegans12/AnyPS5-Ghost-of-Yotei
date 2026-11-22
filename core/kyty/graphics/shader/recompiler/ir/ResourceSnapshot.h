#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCESNAPSHOT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCESNAPSHOT_H_

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

struct DescriptorValue {
	std::array<uint32_t, 8> dwords      = {};
	uint32_t                dword_count = 0;

	bool operator==(const DescriptorValue& other) const {
		return dword_count == other.dword_count && dwords == other.dwords;
	}
};

enum class UniformFillKind { None, Buffer, Image, Replicate };

struct UniformFill {
	UniformFillKind          kind         = UniformFillKind::None;
	uint32_t                 resource     = 0;
	std::array<uint32_t, 3> group_stride {};
	uint32_t                 words        = 0;
	uint32_t                 value        = 0;
	uint32_t                 bounded_elements = 0;
	uint32_t                 source           = 0;
	uint32_t                 pattern_elements = 0;

	bool operator==(const UniformFill&) const = default;
};

struct ImageTableSnapshot {
	uint64_t                     base         = 0;
	uint64_t                     size         = 0;
	uint32_t                     offset       = 0;
	uint32_t                     entry_mask   = 0;
	std::vector<uint32_t>        raw;
	std::vector<DescriptorValue> slots;
	std::vector<uint32_t>        mapping;
	uint64_t                     slots_id = 0;
};

struct GpuFlatLoad {
	uint64_t address = 0;
	uint32_t slot    = 0;
	uint32_t count   = 0;
};

struct GpuFormedView {
	uint32_t resource = 0;
	uint32_t element  = 0;
	uint64_t address  = 0;
};

struct ResourceSnapshot {
	std::vector<DescriptorValue> buffers;
	std::vector<DescriptorValue> images;
	std::vector<DescriptorValue> samplers;
	std::vector<uint32_t>        flattened_srt;
	std::vector<ImageTableSnapshot> image_tables;
	std::vector<uint32_t>        user_data;
	std::vector<std::pair<uint64_t, uint64_t>> specialization_reads;
	UniformFill                 uniform_fill;
	mutable std::vector<GpuFlatLoad> gpu_flat_loads;
	std::vector<GpuFormedView>       gpu_formed;
};

}

#endif
