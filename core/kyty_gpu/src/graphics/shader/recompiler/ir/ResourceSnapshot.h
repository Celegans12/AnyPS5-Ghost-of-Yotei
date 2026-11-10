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

enum class UniformFillKind { None, Buffer, Image };

struct UniformFill {
	UniformFillKind          kind         = UniformFillKind::None;
	uint32_t                 resource     = 0;
	std::array<uint32_t, 3> group_stride {};
	uint32_t                 words        = 0;
	uint32_t                 value        = 0;
	uint32_t                 bounded_elements = 0;

	bool operator==(const UniformFill&) const = default;
};

// A bindless image heap one image of the shader samples: the host gives it a region of the
// translation table and patches flattened_srt[mapping_offset] (region) and [+1] (entries).
struct BindlessHeapUse {
	uint32_t image          = 0;
	uint64_t base           = 0;
	uint64_t size           = 0;
	uint32_t mapping_offset = 0;
	// The bindless tables: the table the heap lies in, as the draw's V# or record pointer gives it: its first
	// byte (a record table's heap starts at the field it reads, base - table_base), its record stride in bytes (the
	// V#'s for a heap table, the shader's for a record table; 0 when there is none) and its bytes (the V#'s size; 0
	// for a record pointer, which has no bound).
	uint64_t table_base     = 0;
	uint32_t record_bytes   = 0;
	uint64_t table_bytes    = 0;
};

// Record-keyed V#s: guest bytes the shader can touch through BDA via a record-keyed V#
// (ShaderInfo::indirect_ranges) in this draw or dispatch, from the candidate V#s read at record time.
struct IndirectRange {
	uint64_t address = 0;
	uint64_t size    = 0;
	bool     written = false;

	bool operator==(const IndirectRange& other) const = default;
};

struct ResourceSnapshot {
	std::vector<DescriptorValue> buffers;
	std::vector<DescriptorValue> images;
	std::vector<DescriptorValue> samplers;
	std::vector<uint32_t>        flattened_srt;
	std::vector<uint32_t>        user_data;
	std::vector<std::pair<uint64_t, uint64_t>> specialization_reads;
	UniformFill                 uniform_fill;
	std::vector<BindlessHeapUse> bindless_heaps;
	std::vector<IndirectRange>   indirect_ranges;
	// Set by the materialization that listed every range of every record-keyed source; the renderer treats a
	// program with record-keyed sources and an incomplete list like one that writes through unknown addresses.
	bool                         indirect_ranges_complete = false;
	// D6 step K2: every written record-keyed source was listed in full (an incomplete list of read-only sources
	// leaves no store unlisted). A call without it stops (KYTY_RECORD_KEYED_VERIFY counts it instead).
	bool                         indirect_writes_complete = false;
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif // EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCESNAPSHOT_H_
