#include "prx/libSceAgcDriver/KytyShim/host/DllCreate.hpp"
#include "prx/libSceAgcDriver/KytyShim/include/KytyDll.h"

#if defined(_WIN32)
#include <windows.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

void Require(bool value, const char* what) {
    if (value) return;
    std::fprintf(stderr, "kyty_shim_dll_abi: %s\n", what);
    std::abort();
}

KytyDllApi g_table {};
KytyDllStatus g_status = KYTY_DLL_OK;

KytyDllStatus FakeCreate(const KytyDllHost* host, const KytyDllApi** api) {
    if (g_status != KYTY_DLL_OK) return g_status;
    if (!KytyDllHostMatches(host)) return KYTY_DLL_ABI_MISMATCH;
    *api = &g_table;
    return KYTY_DLL_OK;
}

KytyDllHost MatchingHost() {
    KytyDllHost host {};
    host.abi_version = KYTY_DLL_ABI_VERSION;
    host.size        = sizeof(KytyDllHost);
    return host;
}

std::string Refusal(KytyDllCreateFunction create, const KytyDllHost& host) {
    try {
        KytyShim::CreateDll(create, host);
    } catch (const std::runtime_error& error) {
        return error.what();
    }
    return {};
}

bool Contains(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

void CheckSeam() {
    KytyDllHost host = MatchingHost();
    Require(KytyDllHostMatches(&host), "a host table of this KytyDll.h was refused");
    host.abi_version = KYTY_DLL_ABI_VERSION + 1;
    Require(!KytyDllHostMatches(&host), "a host table of another ABI version was accepted");
    host = MatchingHost();
    host.size = sizeof(KytyDllHost) - sizeof(void*);
    Require(!KytyDllHostMatches(&host), "a host table of another size was accepted");
    Require(!KytyDllHostMatches(nullptr), "a missing host table was accepted");

    g_table.abi_version = KYTY_DLL_ABI_VERSION;
    g_table.size        = sizeof(KytyDllApi);
    Require(KytyDllApiMatches(&g_table), "an API table of this KytyDll.h was refused");
    Require(KytyShim::CreateDll(&FakeCreate, MatchingHost()) == &g_table, "the loader refused a matching kyty_gpu.prx");

    g_table.abi_version = KYTY_DLL_ABI_VERSION + 1;
    const std::string api_version = Refusal(&FakeCreate, MatchingHost());
    Require(Contains(api_version, "kyty_gpu.prx was built for another KytyDll.h: its API table has ABI version " + std::to_string(KYTY_DLL_ABI_VERSION + 1)),
            "the loader took an API table of another ABI version or did not name it");

    g_table.abi_version = KYTY_DLL_ABI_VERSION;
    g_table.size        = sizeof(KytyDllApi) + 8;
    Require(Contains(Refusal(&FakeCreate, MatchingHost()), "kyty_gpu.prx was built for another KytyDll.h"), "the loader took an API table of another size");
    g_table.size = sizeof(KytyDllApi);

    KytyDllHost old_host = MatchingHost();
    old_host.abi_version = KYTY_DLL_ABI_VERSION - 1;
    const std::string refused = Refusal(&FakeCreate, old_host);
    Require(Contains(refused, "kyty_gpu.prx was built for another KytyDll.h: it refused libSceAgcDriver's ABI version " + std::to_string(KYTY_DLL_ABI_VERSION)),
            "the loader did not report a refused host table");

    g_status = KYTY_DLL_ERROR;
    Require(Contains(Refusal(&FakeCreate, MatchingHost()), "kyty_gpu.prx did not start (status 1)"), "the loader did not report a failed start");
    g_status = KYTY_DLL_OK;
}

#if defined(_WIN32)
void CheckDll(const char* path) {
    const HMODULE module = LoadLibraryExA(path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    Require(module != nullptr, "kyty_gpu.prx could not be loaded");
    const auto create = reinterpret_cast<KytyDllCreateFunction>(reinterpret_cast<void*>(GetProcAddress(module, KYTY_DLL_CREATE_SYMBOL)));
    Require(create != nullptr, "kyty_gpu.prx does not export KytyDllCreate");

    KytyDllHost host = MatchingHost();
    host.abi_version = KYTY_DLL_ABI_VERSION + 1;
    const KytyDllApi* api = nullptr;
    Require(create(&host, &api) == KYTY_DLL_ABI_MISMATCH && api == nullptr, "kyty_gpu.prx accepted a host table of another ABI version");
    host = MatchingHost();
    host.size = sizeof(KytyDllHost) + 8;
    Require(create(&host, &api) == KYTY_DLL_ABI_MISMATCH && api == nullptr, "kyty_gpu.prx accepted a host table of another size");

    host = MatchingHost();
    host.abi_version = KYTY_DLL_ABI_VERSION - 1;
    Require(Contains(Refusal(create, host), "it refused libSceAgcDriver's ABI version"), "the loader did not report kyty_gpu.prx refusing the host table");
}
#endif

}

int main(int argc, char** argv) {
    CheckSeam();
#if defined(_WIN32)
    if (argc > 1) CheckDll(argv[1]);
#else
    (void)argc;
    (void)argv;
#endif
    std::puts("kyty_shim_dll_abi: ok");
    return 0;
}
