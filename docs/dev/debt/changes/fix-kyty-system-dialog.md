# libSceAgcDriver

## add Functional

- [System dialogs and the system keyboard with the kyty backend](../../../core/libs/prx/libSceAgcDriver/KytyShim/src/VideoOut.cpp) (libSceAgcDriver, kyty backend) - the system dialog and the system keyboard are laid out as the default backend lays them out (`KeyboardOverlay.cpp`) and drawn over each presentation in Kyty's system overlay step, one cleared rectangle per layout rectangle. There is no blending: a palette color with alpha below 1 is drawn opaque, as if over black (only the panel, at 0.94). They are drawn only into an 8-bit SDR swapchain; an HDR10 swapchain throws while one is shown. As in the default backend, they appear only with the title's flips. Not checked on a console.
