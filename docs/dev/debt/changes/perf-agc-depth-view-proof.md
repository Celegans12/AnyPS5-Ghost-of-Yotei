# libSceAgcDriver

## add Opt-in draw paths

- [Depth view proofs](../../../core/libs/prx/libSceAgcDriver/Graphics/src/DepthSurface.cpp) - a single-sample depth view skips its depth plane, stencil and HTILE synchronisations with guest memory, its memory notes and the stencil write-watch query while the write versions of those ranges have not moved since one of them found nothing to do, so a CPU store to a plane is seen at the next collect epoch, as the synchronisations see it. Everything else the view does runs at every draw. The HTILE proof also needs no pending GPU write over the HTILE, the same HTILE read stamp and the depth plane proof no pending clear and an unchanged allocation registry. Multisampled and partially resident targets, a view that creates its surface and ranges outside the write watch keep the full path. `APS5_NO_DEPTH_VIEW_PROOFS=1` restores it.
