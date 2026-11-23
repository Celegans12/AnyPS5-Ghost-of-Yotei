# kyty

## add Functional

- [GPU shutdown](../../../core/kyty/graphics/host_gpu/renderer/renderContext.cpp) does not coordinate admission of new guest memory faults with stopping the GPU worker. A fault that needs a new readback can still attempt to use the worker after it stops. Closes when guest fault admission is ordered with GPU shutdown and a watched guest page can be accessed while shutdown runs.
