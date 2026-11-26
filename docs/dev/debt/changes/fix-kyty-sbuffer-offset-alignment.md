# kyty

## add Functional

- [Scalar buffer load offsets](../../../core/kyty/graphics/shader/recompiler/backend/spirv/spirvEmitterMemory.cpp) have not been measured on PS5 hardware when the low bits of `SOFFSET` and the instruction immediate carry into the next dword. The implementation combines the byte offsets before alignment as [RDNA2 ISA section 7.2.1](https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture) specifies. Synthetic replay covers a carry, no carry and a crossed bound; the console result remains unverified. Closes when a PS5 hardware oracle measures those cases, including a multi-dword load at the bound.
