#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_REDZONEGATE_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_REDZONEGATE_HPP

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <mutex>

namespace RedZoneGate {

inline constexpr std::size_t MarkOffset = 0x28;
inline constexpr unsigned char Mark[8] = {'A', 'P', 'S', '5', 'R', 'Z', '0', '1'};
inline constexpr std::size_t HeaderBytes = 0x80;
inline constexpr wchar_t GuestSuffix[] = L".guest.prx";

enum class Image { Host, Framed, Unframed };

inline Image Classify(const unsigned char* header, std::size_t bytes, bool guestName) {
    if (header == nullptr || bytes < HeaderBytes || header[0] != 'M' || header[1] != 'Z') return guestName ? Image::Unframed : Image::Host;
    std::uint32_t peOffset = 0;
    std::memcpy(&peOffset, header + 0x3c, sizeof(peOffset));
    bool relinked = peOffset == HeaderBytes;
    for (std::size_t at = 0x40; relinked && at < HeaderBytes; ++at) relinked = header[at] == 0;
    if (!relinked && !guestName) return Image::Host;
    return std::memcmp(header + MarkOffset, Mark, sizeof(Mark)) == 0 ? Image::Framed : Image::Unframed;
}

inline bool GuestName(const wchar_t* path, std::size_t length) {
    constexpr std::size_t suffixLength = std::size(GuestSuffix) - 1;
    if (path == nullptr || length < suffixLength) return false;
    for (std::size_t at = 0; at < suffixLength; ++at) {
        wchar_t letter = path[length - suffixLength + at];
        if (letter >= L'A' && letter <= L'Z') letter = static_cast<wchar_t>(letter - L'A' + L'a');
        if (letter != GuestSuffix[at]) return false;
    }
    return true;
}

struct Gate {
    std::atomic<bool> unframed{false};
    std::atomic<bool> told{false};
    std::atomic<bool> watching{false};
    std::atomic<int> named{0};
    bool overridden = false;
    char name[96] = {};
};

inline Gate& State() {
    static Gate gate;
    return gate;
}

inline void Note(const wchar_t* path, std::size_t length, const unsigned char* header, std::size_t bytes) {
    if (Classify(header, bytes, GuestName(path, length)) != Image::Unframed) return;
    auto& gate = State();
    int unnamed = 0;
    if (gate.named.compare_exchange_strong(unnamed, 1, std::memory_order_acq_rel)) {
        std::size_t begin = 0;
        for (std::size_t at = 0; path != nullptr && at < length; ++at) if (path[at] == L'\\' || path[at] == L'/') begin = at + 1;
        std::size_t count = 0;
        for (std::size_t at = begin; path != nullptr && at < length && count + 1 < sizeof(gate.name); ++at) gate.name[count++] = path[at] >= 0x20 && path[at] < 0x7f ? static_cast<char>(path[at]) : '?';
        gate.name[count] = '\0';
        gate.named.store(2, std::memory_order_release);
    }
    gate.unframed.store(true, std::memory_order_release);
}

struct LoaderString {
    USHORT Length;
    USHORT MaximumLength;
    PWSTR Buffer;
};

struct LoaderNotice {
    ULONG Flags;
    const LoaderString* FullDllName;
    const LoaderString* BaseDllName;
    PVOID DllBase;
    ULONG SizeOfImage;
};

using LoaderCallback = VOID (CALLBACK*)(ULONG reason, const LoaderNotice* notice, PVOID context);
using LoaderRegister = LONG (NTAPI*)(ULONG flags, LoaderCallback callback, PVOID context, PVOID* cookie);

inline VOID CALLBACK Loaded(ULONG reason, const LoaderNotice* notice, PVOID) {
    if (reason != 1 || notice == nullptr || notice->FullDllName == nullptr || notice->DllBase == nullptr || notice->SizeOfImage < HeaderBytes) return;
    Note(notice->FullDllName->Buffer, notice->FullDllName->Length / sizeof(wchar_t), static_cast<const unsigned char*>(notice->DllBase), HeaderBytes);
}

inline void Scan() {
    HMODULE modules[2048];
    DWORD needed = 0;
    const auto process = GetCurrentProcess();
    if (!K32EnumProcessModules(process, modules, sizeof(modules), &needed)) return;
    const auto count = std::min<DWORD>(needed / sizeof(HMODULE), static_cast<DWORD>(std::size(modules)));
    for (DWORD index = 0; index < count; ++index) {
        wchar_t path[MAX_PATH];
        const auto length = GetModuleFileNameW(modules[index], path, MAX_PATH);
        MODULEINFO info{};
        if (!K32GetModuleInformation(process, modules[index], &info, sizeof(info)) || info.lpBaseOfDll == nullptr || info.SizeOfImage < HeaderBytes) continue;
        Note(path, length, static_cast<const unsigned char*>(info.lpBaseOfDll), HeaderBytes);
    }
}

inline void Watch() {
    static std::once_flag once;
    std::call_once(once, [] {
        auto& gate = State();
        const char* value = std::getenv("APS5_ALLOW_UNFRAMED_FAULTS");
        gate.overridden = value != nullptr && value[0] != '\0' && !(value[0] == '0' && value[1] == '\0');
        if (const auto ntdll = GetModuleHandleW(L"ntdll.dll")) {
            if (const auto entry = GetProcAddress(ntdll, "LdrRegisterDllNotification")) {
                const auto subscribe = reinterpret_cast<LoaderRegister>(reinterpret_cast<void*>(entry));
                PVOID cookie = nullptr;
                gate.watching.store(subscribe(0, Loaded, nullptr, &cookie) == 0, std::memory_order_release);
            }
        }
        Scan();
    });
}

inline bool Open() {
    Watch();
    auto& gate = State();
    if (!gate.watching.load(std::memory_order_acquire)) Scan();
    if (!gate.unframed.load(std::memory_order_acquire)) return true;
    if (!gate.told.exchange(true, std::memory_order_acq_rel)) {
        const char* name = gate.named.load(std::memory_order_acquire) == 2 ? gate.name : "an image";
        if (gate.overridden) std::fprintf(stderr, "[red-zone] %s is guest code without the red-zone framing mark; APS5_ALLOW_UNFRAMED_FAULTS keeps page tracking and read traps on, and a fault on a guest thread can overwrite the values a leaf function keeps below RSP\n", name);
        else std::fprintf(stderr, "[red-zone] %s is guest code without the red-zone framing mark: no page tracking block is armed and no read trap is set, since a fault on a guest thread would overwrite the values a leaf function keeps below RSP. Convert the title again with --frame-red-zone, or set APS5_ALLOW_UNFRAMED_FAULTS=1 to take the faults anyway\n", name);
        std::fflush(stderr);
    }
    return gate.overridden;
}

}
#endif

#endif
