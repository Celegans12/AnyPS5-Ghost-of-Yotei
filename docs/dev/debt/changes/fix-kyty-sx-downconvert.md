# kyty

## add Functional

- [SX downconvert](../../../core/kyty/graphics/host_gpu/renderer/pipeline/pipelineCache.cpp) - Active nonzero manual conversions throw during normal color draws. Ahead preparation declines them; metadata, resolves and disabled-color paths do not consume them. Selectors and formats 0 through 11 are stored; manual `NO_CONVERSION` uses the existing SPI/Vulkan lowering. Automatic mode retains that lowering without reproducing hardware mapping. RB+ rounding/blending and `SX_BLEND_OPT_EPSILON`/`SX_BLEND_OPT_CONTROL` remain unimplemented. `DISABLE_DUAL_QUAD` alone does not establish conversion equivalence. Closes when: manual conversions and automatic mapping are validated against console output, including blend/rounding boundaries. Labels: gpu, tested, windows.
