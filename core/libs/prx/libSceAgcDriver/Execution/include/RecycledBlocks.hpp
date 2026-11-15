#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_RECYCLEDBLOCKS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_RECYCLEDBLOCKS_HPP

#include "prx/libc/include/HostThreadLocal.hpp"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <vector>

namespace AgcDriver {

struct RecycledBlockCounts {
    std::uint64_t reused = 0, fresh = 0, kept = 0, released = 0;
};

template <typename TTag>
class RecycledBlockStore {
public:
    static constexpr std::size_t Limit = 64;

    static void* Take(std::size_t bytes) {
        auto& store = HostThreadLocal<Store, TTag>();
        if (bytes == store.bytes && !store.blocks.empty()) {
            void* block = store.blocks.back();
            store.blocks.pop_back();
            reused.fetch_add(1, std::memory_order_relaxed);
            return block;
        }
        fresh.fetch_add(1, std::memory_order_relaxed);
        return ::operator new(bytes);
    }

    static void Give(void* block, std::size_t bytes) noexcept {
        try {
            auto& store = HostThreadLocal<Store, TTag>();
            if (store.bytes == 0) store.bytes = bytes;
            if (bytes == store.bytes && store.blocks.size() < Limit) {
                store.blocks.push_back(block);
                kept.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        } catch (...) {
        }
        released.fetch_add(1, std::memory_order_relaxed);
        ::operator delete(block);
    }

    static RecycledBlockCounts Counted() {
        return {reused.load(std::memory_order_relaxed), fresh.load(std::memory_order_relaxed), kept.load(std::memory_order_relaxed), released.load(std::memory_order_relaxed)};
    }

private:
    struct Store {
        std::size_t bytes = 0;
        std::vector<void*> blocks;
        Store() = default;
        Store(const Store&) = delete;
        Store& operator=(const Store&) = delete;
        ~Store() {
            for (void* block : blocks) ::operator delete(block);
        }
    };

    static inline std::atomic<std::uint64_t> reused{0}, fresh{0}, kept{0}, released{0};
};

template <typename TValue, typename TTag>
struct RecycledBlocks {
    using value_type = TValue;

    template <typename TOther>
    struct rebind {
        using other = RecycledBlocks<TOther, TTag>;
    };

    RecycledBlocks() noexcept = default;

    template <typename TOther>
    RecycledBlocks(const RecycledBlocks<TOther, TTag>&) noexcept {}

    TValue* allocate(std::size_t count) {
        if (alignof(TValue) > __STDCPP_DEFAULT_NEW_ALIGNMENT__) return static_cast<TValue*>(::operator new(count * sizeof(TValue), std::align_val_t{alignof(TValue)}));
        if (count != 1) return static_cast<TValue*>(::operator new(count * sizeof(TValue)));
        return static_cast<TValue*>(RecycledBlockStore<TTag>::Take(sizeof(TValue)));
    }

    void deallocate(TValue* block, std::size_t count) noexcept {
        if (alignof(TValue) > __STDCPP_DEFAULT_NEW_ALIGNMENT__) {
            ::operator delete(block, std::align_val_t{alignof(TValue)});
            return;
        }
        if (count != 1) {
            ::operator delete(block);
            return;
        }
        RecycledBlockStore<TTag>::Give(block, sizeof(TValue));
    }

    template <typename TOther>
    bool operator==(const RecycledBlocks<TOther, TTag>&) const noexcept {
        return true;
    }
};

}

#endif
