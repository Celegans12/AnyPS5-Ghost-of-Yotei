# libSceAgcDriver

## add Functional

- [Overlay output bus with the kyty backend](../../../core/libs/prx/libSceAgcDriver/KytyShim/src/VideoOut.cpp) (libSceAgcDriver, libSceVideoOut, kyty backend) - the overlay port's displayed buffer is copied into a second Kyty presenter frame at every main presentation and shown on Kyty's overlay layer (bus 1); a presentation without one clears that layer. An overlay buffer the kyty presenter refuses (tiling, pitch, compression, format) still stops the presentation. The blend is premultiplied alpha, as the default backend infers from one title (see Overlay output bus in libSceVideoOut); Kyty's own video out reads the per-buffer `ALPHA_PREMULTIPLIED` option instead, which libSceVideoOut refuses. Kyty draws the overlay into the main plane's rectangle, where the default backend fits each plane to the window. HDR overlays and overlays over an HDR10 main plane go through Kyty's overlay shader, untested. Not checked on a console.
