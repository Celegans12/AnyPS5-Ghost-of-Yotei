# kyty

## add Functional

- [IMAGE_GET_LOD](../../../core/kyty/graphics/shader/recompiler/frontend/translate/Memory.cpp) is refused outside pixel and compute shaders because its current lowering emits `OpImageQueryLod`, which is invalid in vertex, local and tessellation execution models. Mesh also lacks the derivative-group execution mode that `OpImageQueryLod` requires. Closes when these stages have a validated alternative lowering or, for mesh, a valid derivative mode and coverage.
- [EXP](../../../core/kyty/graphics/shader/recompiler/frontend/translate/Attribute.cpp) with enabled components and a non-null target is refused in compute shaders. A parameter export otherwise creates an `Output` storage variable, which Vulkan forbids in `GLCompute`; other live export targets lack a compute output contract. Null and disabled exports retain their no-output behavior. Closes when a valid compute-stage meaning is established and its lowering passes SPIR-V validation.
