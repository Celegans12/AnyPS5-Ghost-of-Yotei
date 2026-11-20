# kyty

## add Functional

- [Merged hull whole-quad masks](../../../core/kyty/graphics/shader/recompiler/Tessellation.cpp) still stop when another lane's bit is read or when the compiler cannot prove that helper-only values are unobserved. The Kyty pass does not prove reconverging helper control flow or changes to memory access. The pruning rule follows `S_WQM_B64` in the [AMD RDNA 2 ISA](https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture) and the own-bit proof in [AnyPS5](https://github.com/boykopovar/AnyPS5/commit/c5291de0d19d4c71a4f01b95dc2c0112b441f0b3). The fused hull lowering has not been measured on PS5 hardware.
