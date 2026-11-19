#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_HOTWRITEWATCH_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_HOTWRITEWATCH_H_

#include <cstdint>

namespace Libs::Graphics {

class HotWriteWatch final {
public:
	using Written = void (*)(void* context, uint64_t begin, uint64_t end);

	HotWriteWatch() noexcept;
	~HotWriteWatch();
	HotWriteWatch(const HotWriteWatch&)            = delete;
	HotWriteWatch& operator=(const HotWriteWatch&) = delete;

	static HotWriteWatch& Get() noexcept;

	[[nodiscard]] bool Available() const noexcept { return m_uffd >= 0 && m_pagemap >= 0; }
	[[nodiscard]] bool Watch(uint64_t begin, uint64_t end) const noexcept;
	void               Unwatch(uint64_t begin, uint64_t end) const noexcept;
	[[nodiscard]] bool Collect(uint64_t begin, uint64_t end, Written written, void* context) const noexcept;

private:
	int m_uffd    = -1;
	int m_pagemap = -1;
};

}

#endif
