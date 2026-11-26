# kyty

## add Functional

- [IMAGE_GATHER on a 3D image](../../../core/kyty/graphics/shader/recompiler/backend/spirv/spirvEmitterImage.cpp) is refused because Vulkan does not permit `OpImageGather` with a 3D image. The console operation is not measured, so there is no validated alternative lowering. Closes when: its semantics are measured and a valid lowering is tested.
