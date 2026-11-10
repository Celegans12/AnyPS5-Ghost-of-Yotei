#ifndef KYTY_COMMON_ONCEVALUE_H_
#define KYTY_COMMON_ONCEVALUE_H_

#include <atomic>
#include <cstdint>
#include <limits>
#include <mutex>
#include <new>
#include <type_traits>

// Hot-path switch gates without thread-safe-static guards (after jetsku 657ce747, Common::Once).
//
// A function-local `static const T x = init();` is a thread-safe static: under the MSVC ABI (clang-cl
// too) every call reads the thread-local initialization epoch and compares it with the static's guard,
// and the gate cannot be inlined. jetsku measured 3-5% of the command processor's time in such gates.
// The slots below are constant-initialized and trivially destructible, so a `static constinit` slot
// has no guard at all; after the first call a read is one atomic load and a compare.
//
//     static constinit Common::OnceValue<bool> enabled_once;
//     const bool enabled = enabled_once.Get([] { ...the initializer of the replaced static... });
//
//     static constinit Common::OnceObject<std::string> list_once;
//     const std::string& list = list_once.Get([] { ...the initializer of the replaced static... });
//
// The semantics are those of the replaced static: `init` runs exactly once (std::call_once), on the
// first call, on the calling thread; concurrent first callers wait until it has finished; an exception
// leaves the slot unset, so a later call runs `init` again. Messages printed by `init` appear once.
// The value is copy-initialized from `init()`, as `static const T x = init();` is.

#if defined(_MSC_VER)
#define KYTY_ONCE_NOINLINE __declspec(noinline)
#else
#define KYTY_ONCE_NOINLINE __attribute__((noinline))
#endif

namespace Common {

// bool, integer and enum gates.
template <typename T>
class OnceValue {
	static_assert(std::is_integral_v<T> || std::is_enum_v<T>, "OnceValue holds bool, integer and enum values; use OnceObject");
	static_assert(sizeof(T) <= sizeof(int64_t));

public:
	constexpr OnceValue() noexcept = default;
	OnceValue(const OnceValue&)            = delete;
	OnceValue& operator=(const OnceValue&) = delete;

	// Acquire (a plain load on x86): whatever `init` did before storing the value is visible to a thread
	// that reads it here, as it is after a thread-safe static's guard check.
	template <typename Init>
	[[nodiscard]] T Get(Init&& init) {
		const auto stored = m_value.load(std::memory_order_acquire);
		if (stored != Unset) [[likely]] {
			return FromStorage(stored);
		}
		return GetSlow(init);
	}

private:
	// A value that is stored as Unset (only a 64-bit INT64_MIN / 0x8000000000000000) stays correct: such
	// a gate just takes the slow path, where call_once returns at once, on every call.
	static constexpr int64_t Unset = std::numeric_limits<int64_t>::min();

	static int64_t ToStorage(T value) {
		if constexpr (std::is_enum_v<T>) {
			return static_cast<int64_t>(static_cast<std::underlying_type_t<T>>(value));
		} else {
			return static_cast<int64_t>(value);
		}
	}
	static T FromStorage(int64_t stored) {
		if constexpr (std::is_enum_v<T>) {
			return static_cast<T>(static_cast<std::underlying_type_t<T>>(stored));
		} else if constexpr (std::is_same_v<T, bool>) {
			return stored != 0;
		} else {
			return static_cast<T>(stored);
		}
	}

	template <typename Init>
	KYTY_ONCE_NOINLINE T GetSlow(Init& init) {
		std::call_once(m_flag, [&] {
			const T value = init();
			m_value.store(ToStorage(value), std::memory_order_release);
		});
		// call_once's completion happens-before every return from it.
		return FromStorage(m_value.load(std::memory_order_acquire));
	}

	std::atomic<int64_t> m_value {Unset};
	std::once_flag       m_flag;
};

// Any other type (strings, lists, pairs, parsed specs). The value lives inside the slot and is never
// destroyed (the slot must stay trivially destructible to need no guard); the replaced static was
// destroyed at exit, after the last use of the value.
template <typename T>
class OnceObject {
public:
	constexpr OnceObject() noexcept = default;
	OnceObject(const OnceObject&)            = delete;
	OnceObject& operator=(const OnceObject&) = delete;

	template <typename Init>
	[[nodiscard]] const T& Get(Init&& init) {
		const T* object = m_object.load(std::memory_order_acquire);
		if (object != nullptr) [[likely]] {
			return *object;
		}
		return GetSlow(init);
	}

private:
	template <typename Init>
	KYTY_ONCE_NOINLINE const T& GetSlow(Init& init) {
		std::call_once(m_flag, [&] {
			// `return init();` copy-initializes the value, as `static const T x = init();` does.
			const T* object = ::new (static_cast<void*>(m_storage)) T([&]() -> T { return init(); }());
			m_object.store(object, std::memory_order_release);
		});
		return *m_object.load(std::memory_order_acquire);
	}

	std::atomic<const T*> m_object {nullptr};
	std::once_flag        m_flag;
	alignas(T) unsigned char m_storage[sizeof(T)] {};
};

// No guard for the slots: constant initialization (checked by `constinit` at each use) and nothing
// to register for destruction at exit.
static_assert(std::is_trivially_destructible_v<OnceValue<bool>>);
static_assert(std::is_trivially_destructible_v<OnceObject<int>>);

} // namespace Common

#endif // KYTY_COMMON_ONCEVALUE_H_
