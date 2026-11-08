// The handful of Kyty kernel, loader and libs services that Kyty's graphics calls, answered by AnyPS5 through the host
// table (kyty_gpu.dll). AnyPS5 keeps the guest's event queues, threads, clocks and param.sfo; nothing of Kyty's kernel,
// loader or HLE libraries is in the DLL.

#include "aps5/aps5Host.h"
#include "common/assert.h"
#include "graphics/guest_gpu/pm4.h"
#include "kernel/eventQueue.h"
#include "kernel/pthread.h"
#include "libs/agc.h"
#include "loader/systemContent.h"

#include <cstring>
#include <string>

namespace Libs::LibKernel {

// Kyty asks whether the calling thread is a guest thread (the image read watch serves only guest reads); AnyPS5 answers.
// The returned pointer is only compared with null.
Pthread PthreadSelfOrNull() {
	static PthreadPrivate* const guest_marker = reinterpret_cast<PthreadPrivate*>(uintptr_t {1});
	return Aps5::IsGuestThread() ? guest_marker : nullptr;
}

uint64_t KYTY_SYSV_ABI KernelReadTsc() {
	return Aps5::ReadTsc();
}

uint64_t KYTY_SYSV_ABI KernelGetTscFrequency() {
	return Aps5::TscFrequency();
}

namespace EventQueue {

// AnyPS5 registers the guest's GPU interrupt events itself (sceAgcDriverAddEqEvent stays on its side) and Kyty raises
// interrupts through the host (RenderContext::TriggerInterrupt), so Kyty's own event-queue calls are never reached.
int KYTY_SYSV_ABI KernelAddEvent(KernelEqueue /*eq*/, const KernelEqueueEvent& /*event*/) {
	Aps5::Fatal("Kyty's KernelAddEvent is not used under AnyPS5 (AnyPS5 owns the guest's event queues)");
}

int KYTY_SYSV_ABI KernelDeleteEvent(KernelEqueue /*eq*/, uintptr_t /*ident*/, int16_t /*filter*/) {
	Aps5::Fatal("Kyty's KernelDeleteEvent is not used under AnyPS5 (AnyPS5 owns the guest's event queues)");
}

int KYTY_SYSV_ABI KernelTriggerEvent(KernelEqueue /*eq*/, uintptr_t /*ident*/, int16_t /*filter*/, void* /*trigger_data*/) {
	Aps5::Fatal("Kyty's KernelTriggerEvent is not used under AnyPS5 (interrupts go through the host table)");
}

} // namespace EventQueue

} // namespace Libs::LibKernel

namespace Loader {

// The param.sfo strings the graphics reads (cache file names): the title id and the content version AnyPS5 knows.
bool SystemContentParamSfoGetString(const char* name, std::string* value) {
	if (name == nullptr || value == nullptr) {
		return false;
	}
	const auto& host   = Aps5::Host();
	const char* answer = nullptr;
	if (std::strcmp(name, "TITLE_ID") == 0) {
		answer = host.app_title_id(host.context);
	} else if (std::strcmp(name, "APP_VER") == 0) {
		answer = host.app_content_version(host.context);
	}
	if (answer == nullptr || answer[0] == '\0') {
		return false;
	}
	*value = answer;
	return true;
}

} // namespace Loader

namespace Libs::Graphics {

// Command-buffer dumps (--dump-command-buffers) belong to Kyty's HLE AGC library, which is not in the DLL.
void GraphicsDbgDumpDcb(const char* /*type*/, uint32_t /*num_dw*/, const uint32_t* /*cmd_buffer*/) {}

namespace Gen5 {

// The values of src/libs/agc.cpp l.2478-2485.
constexpr uint32_t AGC_INTERNAL_DATA_REGISTER        = 0x342u;
constexpr uint32_t AGC_DRAW_INDIRECT_MULTI_BEGIN_TAG = 0xc6000008u;
constexpr uint32_t AGC_DRAW_INDIRECT_MULTI_END_TAG   = 0xc6000000u;
constexpr uint32_t AGC_WAIT_USER_DATA_BEGIN_32_TAG   = 0xc8010000u;
constexpr uint32_t AGC_WAIT_USER_DATA_BEGIN_64_TAG   = 0xc8020000u;
constexpr uint32_t AGC_WAIT_USER_DATA_END_TAG        = 0xc8000000u;
constexpr uint32_t AGC_WAIT_USER_DATA_BEGIN_DW       = 4u;
constexpr uint32_t AGC_INTERNAL_DATA_PACKET_SIZE_DW  = 3u;

// Kyty's src/libs/agc.cpp (v0.1.5 + v4c's patch 3): the AGC internal-data wrappers, custom code 1 (Kyty's builder) or
// 0 (AnyPS5's, header bit 0, which the command processor clears).
bool AgcIsInternalDataPacket(uint32_t cmd_id, const uint32_t* payload) {
	if (payload == nullptr || ((cmd_id >> 8u) & 0xffu) != Pm4::IT_SET_UCONFIG_REG ||
	    (KYTY_PM4_R(cmd_id) != 1u && KYTY_PM4_R(cmd_id) != 0u) || payload[0] != AGC_INTERNAL_DATA_REGISTER) {
		return false;
	}
	const auto size_dw = KYTY_PM4_LEN(cmd_id);
	if (size_dw == AGC_WAIT_USER_DATA_BEGIN_DW) {
		const auto tag = payload[1] & 0xffff0000u;
		return tag == AGC_WAIT_USER_DATA_BEGIN_32_TAG || tag == AGC_WAIT_USER_DATA_BEGIN_64_TAG;
	}
	return size_dw == AGC_INTERNAL_DATA_PACKET_SIZE_DW &&
	       (payload[1] == AGC_WAIT_USER_DATA_END_TAG || payload[1] == AGC_DRAW_INDIRECT_MULTI_BEGIN_TAG ||
	        payload[1] == AGC_DRAW_INDIRECT_MULTI_END_TAG);
}

} // namespace Gen5

} // namespace Libs::Graphics
