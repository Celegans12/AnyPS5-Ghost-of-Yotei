#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_INDIRECTWRITETABLES_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_INDIRECTWRITETABLES_H_

#include "common/abi.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/shader/shaderBindings.h"

#include <cstdint>
#include <span>
#include <vector>

namespace Libs::Graphics {

class RenderContext;

class IndirectWriteTables {
public:
	explicit IndirectWriteTables(RenderContext& context): m_context(context) {}
	KYTY_CLASS_NO_COPY(IndirectWriteTables);

	bool Prepare(uint64_t address, uint64_t size);
	void NotePendingWrites() noexcept;
	[[nodiscard]] bool TakePendingWrites() noexcept {
		const bool pending = m_pending_writes;
		m_pending_writes   = false;
		return pending;
	}
	void NoteWrites(std::span<const uint64_t> pages);

	[[nodiscard]] static GuestRange TargetRange(const ShaderBufferResource& target);
	[[nodiscard]] static std::vector<GuestRange> CoalesceTargets(std::vector<GuestRange> ranges);

private:
	static constexpr size_t MaxTables = 8;

	struct Table {
		uint64_t                          address = 0;
		uint64_t                          size    = 0;
		std::vector<ShaderBufferResource> entries;
		std::vector<GuestRange> unmapped;
		uint64_t                mapped_version = 0;
		uint64_t                target_count   = 0;
		uint64_t                last_use       = 0;
	};

	void Scan(Table& table, uint64_t first, uint64_t last, bool changed_only,
	          std::vector<GuestRange>& targets);
	void Cover(Table& table, std::vector<GuestRange>& targets);

	RenderContext&     m_context;
	std::vector<Table> m_tables;
	RangeSet m_targets;
	uint64_t m_use_count      = 0;
	bool     m_pending_writes = false;
};

}

#endif
