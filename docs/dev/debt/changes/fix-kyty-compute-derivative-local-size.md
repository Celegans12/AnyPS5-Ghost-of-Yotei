# kyty

## add Functional

- [Compute quad derivatives](../../../core/kyty/graphics/shader/recompiler/backend/spirv/spirvEmitterModule.cpp) retain `DerivativeGroupQuadsKHR` on compatible 2D layouts and use `DerivativeGroupLinearKHR` for other complete four-lane groups when the device enables it. Incomplete guest groups and unavailable device features are refused. Whether Linear better matches console lane order for every compatible 2D layout remains unverified. Closes when: lane-order behavior is measured across layouts and a valid fallback for unsupported devices is tested.
