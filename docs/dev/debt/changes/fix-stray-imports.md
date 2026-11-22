# libSceLibcInternal

## remove `_ZSt22_Random_device_entropyv`

# libc

## add Unknown function info

- [std::_Random_device_entropy](../../../core/libs/prx/libc/src/RuntimeSupport.cpp) (libSceLibcInternal, implemented in libc, NID `mDIHE-aaRaI`) returns 0.0, as the `libc.prx` bundled with PPSA02453 and PPSA21402 does (`vxorps xmm0, xmm0, xmm0; ret`, read in the binaries, not run on a console). Stray (PPSA02101), whose dump ships no `sce_module`, imports it from `libc.prx`

# libSceNpCppWebApi

## change `TransactionBase<...>::setResponseInformationOption` (5 instantiations each

- [sce::Np::CppWebApi::Common](../../../core/libs/prx/libSceNpCppWebApi/Export.cpp) (libSceNpCppWebApi) - the `finish` overrides of `DownStreamTransaction<...>` (2 instantiations) and `Vector<IntrusivePtr<Leaderboards::V1::User>>::pushBack` return success without doing work; a failed request never produces a transaction to finish. `Transaction<...>::setOnFinishedCallback` (5 instantiations, the `UserProfile::V1`, `ProfanityFilter::V2` and `CommunicationRestrictionStatus::V3` responses) and `TransactionBase<...>::setResponseInformationOption` (the same 5) return success without storing the callback or the option: `start` fails, so neither is ever used. The constructors and destructors of the same five `Transaction<...>` instantiations succeed without keeping state

## change The InGameCatalog Container transaction's `start` runs the `LibContext` checks

- [sce::Np::CppWebApi::Common::TransactionBase and Transaction](../../../core/libs/prx/libSceNpCppWebApi/Export.cpp) (libSceNpCppWebApi) - not read in a module for the four `UserProfile::V1` transactions: their `abort` and `getId` return 0x80553508 and 0, as the `TransactionBase` code read in the PPSA21564 module does for six and five other instantiations (#11235). The InGameCatalog Container transaction's `start` runs the `LibContext` checks (0x80553503, 0x80553502), then returns `SCE_NP_WEBAPI_ERROR_NOT_SIGNED_IN`, as `DataApi::getMultiDataStatusesByUser` does directly; the TitleCloudStorage transaction's `start` throws - what `libSceNpWebApi2` and the service answer past those checks is not measured. `getResponse` returns 0x80553508 while the words at +0 and +8 are null ([that module](https://github.com/boykopovar/AnyPS5/pull/10832#issuecomment-6416121806)); the Container one throws once both are set, the response likewise unmeasured.
