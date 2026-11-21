#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <span>

namespace Libs::Graphics::ShaderRecompiler::IR {

using SrtMemoryReader = bool (*)(void* userdata, uint64_t address, std::span<uint32_t> values);
using SrtLoadRecorder = void (*)(void* userdata, uint64_t address, uint32_t value,
                                 uint32_t flat_slot);
using SrtLoadsRecorder = void (*)(void* userdata, std::span<const uint64_t> loads);
inline constexpr uint64_t SrtGuestMemoryStart = 0x40000;
inline constexpr uint32_t NoFlatSlot = UINT32_MAX;
inline constexpr uint32_t DescriptorSlotFlag = 0x80000000u;
inline constexpr uint32_t ZeroTestSlot = UINT32_MAX - 1u;
inline constexpr uint32_t UnreadFlatSlotFlag = 0x40000000u;
constexpr uint32_t DescriptorSlot(uint32_t source, uint32_t word) {
	return DescriptorSlotFlag | (source << 3u) | word;
}
constexpr bool IsDescriptorSlot(uint32_t slot) {
	return slot != NoFlatSlot && slot != ZeroTestSlot && (slot & DescriptorSlotFlag) != 0;
}
constexpr uint32_t DescriptorSlotSource(uint32_t slot) {
	return (slot & ~DescriptorSlotFlag) >> 3u;
}
constexpr uint32_t DescriptorSlotWord(uint32_t slot) {
	return slot & 7u;
}

struct ImageHeapRequest {
	uint64_t             base  = 0;
	uint64_t             size  = 0;
	uint64_t             step  = 0;
	uint64_t             first = 0;
	uint64_t             count = 0;
	const ImageResource* root  = nullptr;
	bool                 table = false;
};

struct ImageHeapView {
	uint64_t                         slots      = 0;
	uint32_t                         slot_count = 0;
	std::span<const DescriptorValue> kinds;
};

struct ImageHeapList {
	const void*                      owner    = nullptr;
	uint32_t                         resource = 0;
	const ImageResource*             root     = nullptr;
	std::span<const DescriptorValue> values;
};

struct ImageHeapPrediction {
	bool                         acquired   = false;
	uint64_t                     slots      = 0;
	uint32_t                     slot_count = 0;
	std::vector<DescriptorValue> kinds;
};

class ImageHeapSource {
public:
	virtual bool Acquire(const ImageHeapRequest& request, ImageHeapView& view) = 0;
	virtual bool AcquireList(const ImageHeapList& , ImageHeapView& ) {
		return false;
	}
	virtual bool Predict(const ImageHeapRequest& , ImageHeapPrediction& ) {
		return false;
	}
	virtual bool PredictList(const ImageHeapList& , ImageHeapPrediction& ) {
		return false;
	}

protected:
	~ImageHeapSource() = default;
};

struct SrtRuntime {
	std::span<const uint32_t> user_data;
	uint64_t                  shader_base                = 0;
	void*                     userdata                   = nullptr;
	SrtMemoryReader           read_specialization_memory = nullptr;
	std::span<const uint32_t> workgroup_counts;
	uint64_t                  indirect_args_address = 0;
	ImageHeapSource*          image_heap            = nullptr;
	SrtLoadRecorder           record_load           = nullptr;
	SrtLoadsRecorder          record_loads          = nullptr;
	bool (*gpu_produced)(uint64_t address) = nullptr;
	bool (*pending_write)(uint64_t address, uint32_t& value) = nullptr;
	std::span<const uint32_t> call_words;
	bool runtime_stride = false;
	bool image_table_heap = false;
#if defined(ANYPS5_KYTY)
	bool defer_flat_loads = false;
	bool note_around_loads = false;
#endif
};

class SrtReadCapture {
public:
	SrtReadCapture(SrtRuntime source, std::vector<std::pair<uint64_t, uint64_t>>& ranges):
	    m_source(source), m_ranges(ranges) {}
	SrtRuntime ObservedRuntime();

private:
	static bool Read(void* userdata, uint64_t address, std::span<uint32_t> values);

	SrtRuntime                                  m_source;
	std::vector<std::pair<uint64_t, uint64_t>>& m_ranges;
};

enum class RuntimeValueType { Any, Integer, ImmutableInteger };

bool ValidateRuntimeValue(const ResourcePlan& program, Value value,
                          RuntimeValueType type = RuntimeValueType::Any);
bool ValidateWaveUniformValue(const ResourcePlan& program, Value value);
void CompileSrtWalker(ResourcePlan& plan, std::span<const Value> values = {});
bool WalkSrt(const ResourcePlan& plan, const SrtRuntime& runtime, std::vector<uint32_t>& flat,
             std::vector<std::pair<uint64_t, uint64_t>>* reads);
bool WalkedValue(const ResourcePlan& plan, Value value, uint32_t& result);
bool WalkedDescriptor(const ResourcePlan& plan, uint32_t source, DescriptorValue& result);
bool WalkedActive(const ResourcePlan& plan, uint32_t source);
bool WalkedImageActive(const ResourcePlan& plan, uint32_t image);
enum class SrtWalkerPage : uint8_t { Direct, Backing, PerLoad };
struct SrtWalkerGuestMemory {
	SrtWalkerPage (*page)(uint64_t address, bool ahead, const uint8_t*& backing) = nullptr;
	bool (*read)(uint64_t address, uint32_t& value, bool ahead, const uint8_t* backing) = nullptr;
#if defined(ANYPS5_KYTY)
	const uint8_t* (*around_page)(uint64_t address, bool& clean) = nullptr;
	bool (*read_on_page)(uint64_t address, uint32_t& value, const uint8_t* backing) = nullptr;
#endif
};
void SetSrtWalkerGuestMemory(const SrtWalkerGuestMemory& memory);
#if defined(ANYPS5_KYTY)
[[nodiscard]] std::vector<uint64_t>& SrtDeferredLoadsOfThread();
[[nodiscard]] std::vector<uint64_t>& SrtAroundLoadsOfThread();
enum class SrtShortWalkers : uint8_t { Off, On, Verify };
struct SrtWalkerChecks {
	uint64_t walks      = 0;
	uint64_t unstable   = 0;
	uint64_t mismatches = 0;
};
[[nodiscard]] SrtShortWalkers SrtShortWalkersMode();
[[nodiscard]] SrtWalkerChecks SrtWalkerChecksSoFar();
#endif
uint32_t SrtWalkerLoadLength(uint64_t rip);
uint32_t SrtWalkerAheadLoadLength(uint64_t rip, uint64_t address);

template <size_t Size>
std::optional<std::array<uint64_t, Size>> AffineIndex(
    Value value, const auto& resolve, const auto& leaf, uint32_t depth = 0) {
	value = resolve(value);
	if (depth > 32u || value.GetType() != Type::U32) return {};
	std::array<uint64_t, Size> result {};
	if (value.IsImmediate()) {
		result[0] = value.U32();
		return result;
	}
	if (const auto index = leaf(value)) {
		result[*index] = 1u;
		return result;
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) return {};
	const auto op = inst->GetOpcode();
	if (op != ValueOpcode::IAdd32 && op != ValueOpcode::IMul32 &&
	    op != ValueOpcode::ShiftLeftLogical32) return {};
	auto left = AffineIndex<Size>(inst->Arg(0), resolve, leaf, depth + 1u);
	auto right = AffineIndex<Size>(inst->Arg(1), resolve, leaf, depth + 1u);
	if (!left || !right) return {};
	const auto constant = [](const auto& coefficients) {
		return std::ranges::all_of(std::span(coefficients).subspan(1u),
		                          [](uint64_t coefficient) { return coefficient == 0u; });
	};
	if (op == ValueOpcode::IMul32 && !constant(*right)) std::swap(left, right);
	if (op != ValueOpcode::IAdd32 && !constant(*right)) return {};
	if (op == ValueOpcode::ShiftLeftLogical32) {
		if ((*right)[0] >= 32u) return {};
		(*right)[0] = uint64_t {1} << (*right)[0];
	}
	for (size_t index = 0; index < Size; ++index) {
		(*left)[index] = op == ValueOpcode::IAdd32
		    ? (*left)[index] + (*right)[index] : (*left)[index] * (*right)[0];
		if ((*left)[index] > UINT32_MAX) return {};
	}
	return left;
}

}

#endif
