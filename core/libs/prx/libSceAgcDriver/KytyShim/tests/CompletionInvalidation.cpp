#include "graphics/host_gpu/regionManager.h"
#include "graphics/host_gpu/renderer/cache/completionInvalidation.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Libs::Graphics::CompletionInvalidation;
using Libs::Graphics::InvalidateWithoutWaiting;
using Libs::Graphics::TrackingSpinLock;
using namespace std::chrono_literals;

constexpr auto Patience = 5s;
constexpr auto Stuck    = 300ms;

int g_failures = 0;

void Check(bool condition, const char* what) {
	if (condition) return;
	std::fprintf(stderr, "kyty_shim_completion_invalidation: FAILED %s\n", what);
	g_failures++;
}

class PriorityOperations {
public:
	PriorityOperations(): m_thread([this] { Run(); }) {}
	~PriorityOperations() {
		{
			std::lock_guard lock(m_mutex);
			m_stop = true;
		}
		m_changed.notify_all();
		m_thread.join();
	}

	void Defer(uint64_t tick, std::function<void()> operation) {
		{
			std::lock_guard lock(m_mutex);
			m_queued.emplace_back(tick, std::move(operation));
		}
		m_changed.notify_all();
	}

	[[nodiscard]] bool WaitFor(uint64_t tick, std::chrono::milliseconds limit) {
		std::unique_lock lock(m_mutex);
		return m_changed.wait_for(lock, limit, [&] {
			const bool active = m_active != 0 && m_active <= tick;
			const bool queued = !m_queued.empty() && m_queued.front().first <= tick;
			return !active && !queued;
		});
	}

private:
	void Run() {
		for (;;) {
			std::pair<uint64_t, std::function<void()>> operation;
			{
				std::unique_lock lock(m_mutex);
				m_changed.wait(lock, [&] { return m_stop || !m_queued.empty(); });
				if (m_queued.empty()) return;
				operation = std::move(m_queued.front());
				m_queued.pop_front();
				m_active = operation.first;
			}
			operation.second();
			{
				std::lock_guard lock(m_mutex);
				m_active = 0;
			}
			m_changed.notify_all();
		}
	}

	std::mutex                                             m_mutex;
	std::condition_variable                                m_changed;
	std::deque<std::pair<uint64_t, std::function<void()>>> m_queued;
	uint64_t                                               m_active = 0;
	bool                                                   m_stop   = false;
	std::thread                                            m_thread;
};

class GpuCommands {
public:
	void Post(std::function<void()> command) {
		{
			std::lock_guard lock(m_mutex);
			m_commands.push_back(std::move(command));
		}
		m_changed.notify_all();
	}

	void PostAndWait(std::function<void()> command) {
		std::atomic<bool> done {false};
		Post([&, command = std::move(command)] {
			command();
			done.store(true, std::memory_order_release);
			done.notify_all();
		});
		done.wait(false, std::memory_order_acquire);
	}

	size_t RunAll() {
		std::vector<std::function<void()>> commands;
		{
			std::lock_guard lock(m_mutex);
			commands.swap(m_commands);
		}
		for (auto& command: commands) command();
		return commands.size();
	}

	[[nodiscard]] bool WaitPosted(std::chrono::milliseconds limit) {
		std::unique_lock lock(m_mutex);
		return m_changed.wait_for(lock, limit, [&] { return !m_commands.empty(); });
	}

private:
	std::mutex                         m_mutex;
	std::condition_variable            m_changed;
	std::vector<std::function<void()>> m_commands;
};

struct Cache {
	TrackingSpinLock  lock;
	std::atomic<bool> armed {false};
	std::atomic<int>  invalidations {0};
	std::atomic<int>  flushes {0};
	std::atomic<int>  in_place {0};
	std::atomic<int>  deferred {0};

	void InvalidateLocked() { invalidations.fetch_add(1, std::memory_order_relaxed); }

	void FlushAndInvalidateOnGpuThread() {
		std::scoped_lock guard(lock);
		if (armed.exchange(false)) flushes.fetch_add(1, std::memory_order_relaxed);
		InvalidateLocked();
	}

	void FromCompletion(GpuCommands& gpu) {
		const auto path = InvalidateWithoutWaiting(
		    lock, [&] { return armed.load(); }, [&] { InvalidateLocked(); },
		    [&] { gpu.Post([this] { FlushAndInvalidateOnGpuThread(); }); });
		(path == CompletionInvalidation::InPlace ? in_place : deferred).fetch_add(1);
	}

	void FromCompletionBlocking(GpuCommands& gpu) {
		if (armed.load()) {
			gpu.PostAndWait([this] {
				std::scoped_lock guard(lock);
				if (armed.exchange(false)) flushes.fetch_add(1, std::memory_order_relaxed);
			});
		}
		std::scoped_lock guard(lock);
		InvalidateLocked();
	}
};

void GarbageCollectorHoldsLockWhileWaiting() {
	Cache              cache;
	GpuCommands        gpu;
	PriorityOperations priority;
	cache.lock.lock();
	priority.Defer(1, [&] { cache.FromCompletion(gpu); });
	Check(priority.WaitFor(1, Patience), "gc order: the completion did not finish while the cache lock was held");
	cache.lock.unlock();
	Check(cache.deferred.load() == 1 && cache.in_place.load() == 0, "gc order: the completion was not deferred");
	Check(cache.invalidations.load() == 0, "gc order: the completion invalidated without the lock");
	Check(gpu.RunAll() == 1, "gc order: no deferred invalidation for the gpu thread");
	Check(cache.invalidations.load() == 1, "gc order: the deferred invalidation did not run");
}

void GarbageCollectorOrderDeadlocksWhenBlocking() {
	Cache              cache;
	GpuCommands        gpu;
	PriorityOperations priority;
	cache.lock.lock();
	priority.Defer(1, [&] { cache.FromCompletionBlocking(gpu); });
	Check(!priority.WaitFor(1, Stuck), "gc order: the blocking completion should be stuck on the cache lock");
	cache.lock.unlock();
	Check(priority.WaitFor(1, Patience), "gc order: the blocking completion did not finish once the lock was free");
	Check(cache.invalidations.load() == 1, "gc order: the blocking completion did not invalidate");
}

void LazyReadbackDoesNotWaitForGpuThread() {
	Cache              cache;
	GpuCommands        gpu;
	PriorityOperations priority;
	cache.armed = true;
	priority.Defer(1, [&] { cache.FromCompletion(gpu); });
	Check(priority.WaitFor(1, Patience), "readback: the completion waited for the gpu thread");
	Check(cache.deferred.load() == 1, "readback: an armed readback was invalidated in place");
	Check(cache.lock.TryLockSpinning(), "readback: the cache lock was kept after deferring");
	cache.lock.unlock();
	Check(gpu.RunAll() == 1, "readback: no deferred flush for the gpu thread");
	Check(cache.flushes.load() == 1 && cache.invalidations.load() == 1,
	      "readback: the gpu thread did not flush before invalidating");
}

void LazyReadbackOrderDeadlocksWhenBlocking() {
	Cache              cache;
	GpuCommands        gpu;
	PriorityOperations priority;
	cache.armed = true;
	priority.Defer(1, [&] { cache.FromCompletionBlocking(gpu); });
	Check(gpu.WaitPosted(Patience), "readback: the blocking completion posted nothing");
	Check(!priority.WaitFor(1, Stuck), "readback: the blocking completion should wait for the gpu thread");
	Check(gpu.RunAll() == 1, "readback: the gpu thread did not find the posted flush");
	Check(priority.WaitFor(1, Patience), "readback: the blocking completion did not finish after the flush");
	Check(cache.flushes.load() == 1 && cache.invalidations.load() == 1, "readback: blocking flush or invalidation missing");
}

void UncontendedCompletionInvalidatesInPlace() {
	Cache              cache;
	GpuCommands        gpu;
	PriorityOperations priority;
	priority.Defer(1, [&] { cache.FromCompletion(gpu); });
	Check(priority.WaitFor(1, Patience), "in place: the completion did not finish");
	Check(cache.in_place.load() == 1 && cache.deferred.load() == 0, "in place: a free lock deferred the invalidation");
	Check(cache.invalidations.load() == 1, "in place: no invalidation");
	Check(gpu.RunAll() == 0, "in place: something was posted to the gpu thread");
}

void MixedTraffic() {
	constexpr uint64_t Operations = 4000;
	Cache              cache;
	GpuCommands        gpu;
	std::atomic<bool>  done {false};
	{
		PriorityOperations priority;
		std::thread        guest([&] {
			while (!done.load(std::memory_order_acquire)) {
				{
					std::scoped_lock guard(cache.lock);
					cache.armed.store(!cache.armed.load());
				}
				std::this_thread::yield();
			}
		});
		for (uint64_t tick = 1; tick <= Operations; tick++) {
			priority.Defer(tick, [&] { cache.FromCompletion(gpu); });
			if (tick % 8 == 0) {
				cache.lock.lock();
				const bool finished = priority.WaitFor(tick, Patience);
				cache.lock.unlock();
				if (!finished) {
					Check(false, "traffic: a completion waited behind the cache lock");
					break;
				}
			}
			if (tick % 64 == 0) gpu.RunAll();
		}
		Check(priority.WaitFor(Operations, Patience), "traffic: the completions did not drain");
		done.store(true, std::memory_order_release);
		guest.join();
	}
	gpu.RunAll();
	Check(cache.in_place.load() + cache.deferred.load() == static_cast<int>(Operations), "traffic: a completion was lost");
	Check(cache.invalidations.load() == static_cast<int>(Operations), "traffic: an invalidation was lost");
}

}

int main() {
	GarbageCollectorOrderDeadlocksWhenBlocking();
	GarbageCollectorHoldsLockWhileWaiting();
	LazyReadbackOrderDeadlocksWhenBlocking();
	LazyReadbackDoesNotWaitForGpuThread();
	UncontendedCompletionInvalidatesInPlace();
	MixedTraffic();
	if (g_failures != 0) {
		std::fprintf(stderr, "kyty_shim_completion_invalidation: %d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("kyty_shim_completion_invalidation: ok\n");
	return 0;
}
