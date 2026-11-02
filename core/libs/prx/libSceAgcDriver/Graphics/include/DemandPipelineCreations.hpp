#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DEMANDPIPELINECREATIONS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DEMANDPIPELINECREATIONS_HPP

#include <atomic>
#include <chrono>
#include <cstdint>

namespace AgcDriver::Graphics {

class DemandPipelineCreations {
public:
    using IdleListener = void (*)();

    class Scope {
    public:
        Scope() : counted(!SpeculativeFlag()) {
            if (counted) ++Counter();
        }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        ~Scope() {
            if (counted && --Counter() == 0) {
                IdleSince().store(NowMs());
                if (const auto listener = Listener().load()) listener();
            }
        }

    private:
        bool counted;
    };

    class SpeculativeThread {
    public:
        SpeculativeThread() : previous(SpeculativeFlag()) { SpeculativeFlag() = true; }
        SpeculativeThread(const SpeculativeThread&) = delete;
        SpeculativeThread& operator=(const SpeculativeThread&) = delete;
        ~SpeculativeThread() { SpeculativeFlag() = previous; }

    private:
        bool previous;
    };

    static std::uint32_t Active() { return Counter().load(); }
    static bool OnSpeculativeThread() { return SpeculativeFlag(); }
    static void SetIdleListener(IdleListener listener) { Listener().store(listener); }
    static std::chrono::milliseconds IdleFor() { return std::chrono::milliseconds(Active() != 0 ? 0 : NowMs() - IdleSince().load()); }

private:
    static std::atomic<std::uint32_t>& Counter() {
        static std::atomic<std::uint32_t> counter{0};
        return counter;
    }
    static std::int64_t NowMs() { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
    static std::atomic<std::int64_t>& IdleSince() {
        static std::atomic<std::int64_t> since{0};
        return since;
    }
    static std::atomic<IdleListener>& Listener() {
        static std::atomic<IdleListener> listener{nullptr};
        return listener;
    }
    static bool& SpeculativeFlag() {
        thread_local bool flag = false;
        return flag;
    }
};

}

#endif
