#ifdef _WIN32

#include <windows.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <malloc.h>

extern "C" IMAGE_DOS_HEADER __ImageBase;

namespace {

struct EmutlsObject {
    std::size_t size;
    std::size_t align;
    std::uintptr_t index;
    void* initial;
};

struct Slots {
    std::size_t count;
    void* values[1];
};

using Fallback = void* (*)(EmutlsObject*);

constexpr std::uintptr_t Owned = std::uintptr_t{1} << 62u;
constexpr std::size_t TebTlsSlots = 0x1480;

DWORD tlsSlot = TLS_OUT_OF_INDEXES;
std::uintptr_t imageBegin = 0;
std::uintptr_t imageEnd = 0;
Fallback fallback = nullptr;
std::atomic<bool> ready{false};
bool fast = true;
std::uintptr_t indices = 0;
SRWLOCK lock = SRWLOCK_INIT;

void* ReadSlot() {
    if (tlsSlot < 64) return reinterpret_cast<void* const*>(reinterpret_cast<const char*>(NtCurrentTeb()) + TebTlsSlots)[tlsSlot];
    const DWORD error = GetLastError();
    void* value = TlsGetValue(tlsSlot);
    SetLastError(error);
    return value;
}

void Prepare() {
    AcquireSRWLockExclusive(&lock);
    if (!ready.load(std::memory_order_relaxed)) {
        tlsSlot = TlsAlloc();
        const auto* dos = &__ImageBase;
        const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(reinterpret_cast<const char*>(dos) + dos->e_lfanew);
        imageBegin = reinterpret_cast<std::uintptr_t>(dos);
        imageEnd = imageBegin + nt->OptionalHeader.SizeOfImage;
        if (HMODULE gcc = GetModuleHandleW(L"libgcc_s_seh-1.dll")) fallback = reinterpret_cast<Fallback>(GetProcAddress(gcc, "__emutls_get_address"));
        const char* setting = std::getenv("KYTY_FAST_TLS");
        fast = fallback == nullptr || setting == nullptr || setting[0] != '0' || setting[1] != '\0';
        ready.store(true, std::memory_order_release);
    }
    ReleaseSRWLockExclusive(&lock);
}

std::uintptr_t IndexOf(EmutlsObject* object) {
    std::uintptr_t index = __atomic_load_n(&object->index, __ATOMIC_ACQUIRE);
    if (index != 0) return index & ~Owned;
    AcquireSRWLockExclusive(&lock);
    index = object->index;
    if (index == 0) {
        index = ++indices | Owned;
        __atomic_store_n(&object->index, index, __ATOMIC_RELEASE);
    }
    ReleaseSRWLockExclusive(&lock);
    return index & ~Owned;
}

void* Allocate(const EmutlsObject* object) {
    const std::size_t align = object->align > sizeof(void*) ? object->align : sizeof(void*);
    void* value = _aligned_malloc(object->size, align);
    if (value == nullptr) std::abort();
    if (object->initial != nullptr) std::memcpy(value, object->initial, object->size);
    else std::memset(value, 0, object->size);
    return value;
}

__attribute__((noinline)) void* Slow(EmutlsObject* object) {
    const DWORD error = GetLastError();
    if (!ready.load(std::memory_order_acquire)) Prepare();
    const auto address = reinterpret_cast<std::uintptr_t>(object);
    if ((!fast || address < imageBegin || address >= imageEnd || tlsSlot == TLS_OUT_OF_INDEXES) && fallback != nullptr) {
        SetLastError(error);
        return fallback(object);
    }
    const std::uintptr_t index = IndexOf(object);
    auto* slots = static_cast<Slots*>(ReadSlot());
    if (slots == nullptr || index > slots->count) {
        std::size_t count = slots == nullptr ? 32 : slots->count * 2;
        while (count < index) count *= 2;
        auto* grown = static_cast<Slots*>(std::calloc(1, sizeof(Slots) + (count - 1) * sizeof(void*)));
        if (grown == nullptr) std::abort();
        grown->count = count;
        if (slots != nullptr) {
            std::memcpy(grown->values, slots->values, slots->count * sizeof(void*));
            std::free(slots);
        }
        slots = grown;
        TlsSetValue(tlsSlot, slots);
    }
    void*& value = slots->values[index - 1];
    if (value == nullptr) value = Allocate(object);
    SetLastError(error);
    return value;
}

void NTAPI ReleaseThread(PVOID, DWORD reason, PVOID) {
    if (reason != DLL_THREAD_DETACH || !ready.load(std::memory_order_acquire) || tlsSlot == TLS_OUT_OF_INDEXES) return;
    const DWORD error = GetLastError();
    auto* slots = static_cast<Slots*>(ReadSlot());
    if (slots != nullptr) {
        TlsSetValue(tlsSlot, nullptr);
        for (std::size_t index = 0; index < slots->count; ++index) _aligned_free(slots->values[index]);
        std::free(slots);
    }
    SetLastError(error);
}

[[gnu::section(".CRT$XLY"), gnu::used]] PIMAGE_TLS_CALLBACK releaseThreadCallback = ReleaseThread;

}

extern "C" void* __emutls_get_address(EmutlsObject* object) {
    const std::uintptr_t tagged = __atomic_load_n(&object->index, __ATOMIC_ACQUIRE);
    if ((tagged & Owned) != 0 && tlsSlot < 64) {
        const auto* slots = static_cast<const Slots*>(reinterpret_cast<void* const*>(reinterpret_cast<const char*>(NtCurrentTeb()) + TebTlsSlots)[tlsSlot]);
        const std::uintptr_t index = tagged & ~Owned;
        if (slots != nullptr && index <= slots->count) {
            if (void* value = slots->values[index - 1]) return value;
        }
    }
    return Slow(object);
}

#endif
