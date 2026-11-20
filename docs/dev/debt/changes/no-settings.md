# libSceNpEntitlementAccess

## add Functional

- [Add-ons reported as installed by default](../../../core/libs/prx/libSceNpEntitlementAccess/Export.cpp) (libSceNpEntitlementAccess) - with `ANYPS5_GPU_BACKEND=kyty`, no `ANYPS5_ENTITLEMENTS` and no `anyps5-entitlements.ini` beside the executable, `sceNpEntitlementAccessGetAddcontEntitlementInfo` reports every non-empty label a title asks for as installed (package type 3, download status 4) although no add-on content exists to mount, and the list getter still returns no add-on because the labels a title does not ask for are unknown. What a console answers for an add-on the account does not own is not measured.
