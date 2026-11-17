#ifndef EMULATOR_INCLUDE_EMULATOR_KERNEL_GUESTPAUSE_H_
#define EMULATOR_INCLUDE_EMULATOR_KERNEL_GUESTPAUSE_H_

#include "common/common.h"

#include <cstdint>
#include <memory>

namespace Libs::LibKernel {

class GuestPause {
public:
	GuestPause();
	~GuestPause();
	KYTY_CLASS_NO_COPY(GuestPause);

	[[nodiscard]] static bool Supported() noexcept;
	[[nodiscard]] static bool GuestStarted();

	void Poll();
	[[nodiscard]] uint32_t Suspended() const noexcept;

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

}

#endif
