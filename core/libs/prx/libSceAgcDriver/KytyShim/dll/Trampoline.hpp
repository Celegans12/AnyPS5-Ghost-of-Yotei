#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_DLL_TRAMPOLINE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_DLL_TRAMPOLINE_HPP

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

namespace KytyShim {

std::FILE* TrampolineLog() noexcept;

[[noreturn]] inline void TrampolineEscaped(const char* callback, const char* what) noexcept {
    std::FILE* log = TrampolineLog();
    std::fprintf(log, "kyty_gpu: %s let an exception escape towards AnyPS5: %s\n", callback, what);
    std::fflush(log);
    std::abort();
}

template<class Tag, class Signature>
class Trampoline;

template<class Tag, class Result, class... Arguments>
class Trampoline<Tag, Result(Arguments...)> {
public:
    using Function = Result (*)(Arguments...);

    static Function Bind(Function target) noexcept {
        if (target == nullptr) return nullptr;
        s_target.store(target, std::memory_order_release);
        return &Call;
    }

    static Result Call(Arguments... arguments) noexcept {
        try {
            return s_target.load(std::memory_order_acquire)(arguments...);
        } catch (const std::exception& error) {
            TrampolineEscaped(Tag::kName, error.what());
        } catch (...) {
            TrampolineEscaped(Tag::kName, "unknown exception");
        }
    }

private:
    static inline std::atomic<Function> s_target{nullptr};
};

template<class Tag, std::size_t Count, class Signature>
class TrampolinePool;

template<class Tag, std::size_t Count, class Result, class... Arguments>
class TrampolinePool<Tag, Count, Result(Arguments...)> {
public:
    using Function = Result (*)(Arguments...);

    static Function Bind(Function target) {
        if (target == nullptr) return nullptr;
        const auto index = s_used.fetch_add(1, std::memory_order_relaxed);
        if (index >= Count) throw std::runtime_error(std::string("kyty_gpu: more than ") + std::to_string(Count) + " registrations of " + Tag::kName);
        return Binders(std::make_index_sequence<Count>{})[index](target);
    }

private:
    template<std::size_t Index>
    struct Slot {
        static constexpr const char* kName = Tag::kName;
    };

    using Binder = Function (*)(Function) noexcept;

    template<std::size_t... Index>
    static std::array<Binder, Count> Binders(std::index_sequence<Index...>) noexcept {
        return {&Trampoline<Slot<Index>, Result(Arguments...)>::Bind...};
    }

    static inline std::atomic<std::size_t> s_used{0};
};

}

#endif
