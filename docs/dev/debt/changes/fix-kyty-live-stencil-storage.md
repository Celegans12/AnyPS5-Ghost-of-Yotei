# kyty

## add Functional

- [Live stencil storage](../../../core/kyty/graphics/host_gpu/renderer/cache/textureCache.cpp) (Kyty) - storage bindings require a matching R8_UINT view of the held, single-sample, uncompressed stencil plane. Partial and bindless views, inactive planes, and concurrent use as a depth/stencil attachment throw. Direct CPU readback of the stencil association remains unsupported.
