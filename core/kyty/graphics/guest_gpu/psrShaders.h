#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_PSRSHADERS_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_PSRSHADERS_H_

#include <cstdint>
#include <span>

namespace Libs::Graphics::Psr {

enum class Shader : uint8_t {
	None,

	MortonComputeBoundingBoxForInstances,
	MortonInitializeIndirectArgsTop,
	SinglePassScanInitializeDescriptors,
	SinglePassScanUint32,
	MortonComputeValidInstanceIndices,
	MortonSetupIndirectArgsTop,
	MortonInitOneInstanceRootNode,
	MortonComputeCodeForInstances,
	BitonicSortPassKeyValueIndirect,
	MortonBuildQbvhHierarchyForInstances,
	MortonBuildInstanceNode,
	MortonRefitQbvhHierarchyForInstances,
	MortonRotateTreeFirstRunTop,
	MortonRotateTreeTop,
	CollapseToQbvhSimpleComputeCompactedNodesIndirect,
	SinglePassScanSetupIndirectArgsVec2u,
	SinglePassScanInitializeDescriptorsVec2uIndirect,
	SinglePassScanVec2uIndirect,
	CollapseToQbvhSimpleTop,

	QbvhFanRefitCopyDispatchArgs,
	QbvhFanRefitAsync,
	QbvhFanRefitRecoverAabbs,
	QbvhRefitBottomLevel,
	QbvhFanRefitAsyncShaderMesh,

	MortonComputeBoundingBoxForVertices,
	MortonComputeCodeForMesh,
	CollapseToQbvhSimpleBottom,

	MortonCaptureMemory,
	DecompressHeaderBase,
	DecompressComputeFanNode,

	Other,
	Trinity,

	Count
};

enum class Generation : uint8_t {
	Unknown,
	Base460,
	Base252,
	Base515,
};

[[nodiscard]] Generation LibraryGeneration();

[[nodiscard]] Shader IdentifyShader(uint64_t code_address);

[[nodiscard]] uint64_t CodeFingerprint(std::span<const uint32_t> binary);

struct CodeMatch {
	Generation generation = Generation::Unknown;
	Shader     shader     = Shader::None;
};
[[nodiscard]] CodeMatch IdentifyCode(std::span<const uint32_t> binary);

[[nodiscard]] bool IsTopLevelBuildShader(Shader shader);

[[nodiscard]] bool ReplacementEnabled();
void               StopReplacement(const char* reason, const char* shader_name);

}

#endif
