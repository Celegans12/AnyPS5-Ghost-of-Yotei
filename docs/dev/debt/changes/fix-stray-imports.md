# libSceLibcInternal

## change 283 entries observed in a 3.20-labelled image

- [Additional callable exports](../../../core/libs/prx/libSceLibcInternal/InventoryExports.cpp) - 282 entries observed in a 3.20-labelled image and a later 13.60-labelled image; 276 match the public 3.20 catalogue and 6 do not. Apart from the `showmanyc`, empty facet, runtime hook and system_error constructor exports below, these remain import-resolution declarations: signatures and behavior are unverified, parameterless `int` declarations are placeholders, and their bodies throw through `NotImplemented_nid_no_patch`. All declarations use literal NID aliases; the table records candidate names where known. No other runtime functionality is implemented; data exports need separate analysis. Identified candidate names match their NIDs in the [public symbol database](https://github.com/zecoxao/sce_symbols/blob/2883963a0a514ba6e77407a08ef4730a65e88254/aerolib.csv) or [supplemental catalog](https://github.com/claimore22/ps5rs/blob/2304be5131221bae0dac24b9767cabffda33f747/data/nids.csv).

## remove `_ZSt22_Random_device_entropyv`

# libc

## add Unknown function info

- [std::_Random_device_entropy](../../../core/libs/prx/libc/src/RuntimeSupport.cpp) (libSceLibcInternal, implemented in libc, NID `mDIHE-aaRaI`) returns 0.0, as the `libc.prx` bundled with PPSA02453 and PPSA21402 does (`vxorps xmm0, xmm0, xmm0; ret`, read in the binaries, not run on a console). Stray (PPSA02101), whose dump ships no `sce_module`, imports it from `libc.prx`
