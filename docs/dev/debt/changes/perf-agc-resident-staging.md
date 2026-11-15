# libSceAgcDriver

## add Opt-in draw paths

- [Resident staging](../../../core/libs/prx/libSceAgcDriver/Graphics/src/GuestBufferMemory.cpp) (libSceAgcDriver) - with `APS5_RESIDENT_STAGING`, a use of a staged range whose shadow is still current copies nothing in, or only the pages the CPU wrote before the use was recorded, and still copies the whole written range back: the window in which a CPU store to bytes the shader leaves alone is rolled back (see the device-local staging entry) opens when the use is recorded instead of when the GPU runs its copy-in. The copy-back is still recorded before the next label store, whole: no CPU read is trapped on Windows, so nothing proves a staged range has no CPU reader, and draws and dispatches that read the range still read the import. Not measured on a title; the only check is `APS5_VERIFY_RESIDENT_STAGING`.
