#include "prx/libSceAgcDriver/KytyShim/dll/Trampoline.hpp"
#include "prx/libSceAgcDriver/KytyShim/host/DllCreate.hpp"
#include "prx/libSceAgcDriver/KytyShim/include/KytyDll.h"

#if defined(_WIN32)
#include <windows.h>
#endif

#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

std::FILE* KytyShim::TrampolineLog() noexcept {
    return stderr;
}

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

struct FilterTag {
    static constexpr const char* kName = "the test filter";
};

struct PreparerTag {
    static constexpr const char* kName = "the test preparer";
};

struct ShutdownTag {
    static constexpr const char* kName = "a test shutdown callback";
};

using Filter   = KytyShim::Trampoline<FilterTag, bool(std::uintptr_t, std::size_t)>;
using Preparer = KytyShim::Trampoline<PreparerTag, bool(std::uintptr_t, std::size_t)>;
using Shutdown = KytyShim::TrampolinePool<ShutdownTag, 2, void()>;

static_assert(noexcept(Filter::Call(0, 0)));
static_assert(noexcept(Filter::Bind(nullptr)));
static_assert(noexcept(KytyShim::Trampoline<ShutdownTag, void()>::Call()));

std::uintptr_t g_address = 0;
std::size_t g_bytes      = 0;
int g_first              = 0;
int g_second             = 0;

bool Even(std::uintptr_t address, std::size_t bytes) {
    g_address = address;
    g_bytes   = bytes;
    return address % 2 == 0;
}

bool Always(std::uintptr_t, std::size_t) {
    return true;
}

bool Throws(std::uintptr_t, std::size_t) {
    throw std::runtime_error("the filter failed");
}

void First() {
    g_first++;
}

void Second() {
    g_second++;
}

void CheckTrampolines() {
    Require(Filter::Bind(nullptr) == nullptr, "a cleared callback did not stay null across the trampoline");
    const auto filter = Filter::Bind(&Even);
    Require(filter != nullptr && filter != &Even, "a callback crossed without its trampoline");
    Require(filter(0x1000, 64) && g_address == 0x1000 && g_bytes == 64, "the trampoline did not forward its arguments and result");
    Require(!filter(0x1001, 8) && g_address == 0x1001 && g_bytes == 8, "the trampoline did not forward a false result");
    Require(Filter::Bind(nullptr) == nullptr && !filter(0x1003, 8), "clearing a callback unbound a trampoline that may still run");
    Require(Filter::Bind(&Always) == filter && filter(0x1001, 8), "a callback set again did not replace the first");
    const auto preparer = Preparer::Bind(&Even);
    Require(preparer != filter && !preparer(0x2001, 4) && g_address == 0x2001, "two callbacks share one trampoline");

    const auto first  = Shutdown::Bind(&First);
    const auto second = Shutdown::Bind(&Second);
    Require(first != nullptr && second != nullptr && first != second, "two shutdown callbacks share one trampoline");
    second();
    first();
    first();
    Require(g_first == 2 && g_second == 1, "the shutdown trampolines did not run their own callbacks");
    Require(Shutdown::Bind(nullptr) == nullptr, "a missing shutdown callback got a trampoline");
    bool refused = false;
    try {
        Shutdown::Bind(&First);
    } catch (const std::runtime_error& error) {
        refused = Contains(error.what(), "more than 2 registrations of a test shutdown callback");
    }
    Require(refused, "a shutdown callback beyond the trampolines was accepted");
}

int EscapeThroughTrampoline() {
    std::signal(SIGABRT, [](int) { std::_Exit(3); });
    const bool result = Filter::Bind(&Throws)(0, 0);
    std::fprintf(stderr, "kyty_shim_dll_abi: the exception was swallowed (%d)\n", result ? 1 : 0);
    return 1;
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
    if (argc > 1 && std::strcmp(argv[1], "escape") == 0) return EscapeThroughTrampoline();
    CheckSeam();
    CheckTrampolines();
#if defined(_WIN32)
    if (argc > 1) CheckDll(argv[1]);
#else
    (void)argc;
    (void)argv;
#endif
    std::puts("kyty_shim_dll_abi: ok");
    return 0;
}
