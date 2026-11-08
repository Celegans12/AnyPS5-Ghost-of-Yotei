#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_BACKEND_INCLUDE_KYTYADAPTER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_BACKEND_INCLUDE_KYTYADAPTER_HPP

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include "SceTypes.hpp"
#include "SceShaders.hpp"
#include "prx/libSceAgcDriver/Execution/include/DisplayBuffer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Presentation.hpp"
#include "prx/libSceAgcDriver/Execution/include/VideoOutput.hpp"

namespace AgcDriver::Kyty {

bool Active();
void Submit(const Packet* packet, std::uint32_t queue);
void SubmitFlip(const FlipInfo& info);
void SuspendPoint();
void WaitIdle();
void Shutdown();
void RegisterShader(const Shader* shader);
void RegisterVideoOutput(std::uint32_t handle, const std::shared_ptr<IVideoOutput>& output);
void UnregisterVideoOutput(std::uint32_t handle, const std::shared_ptr<IVideoOutput>& output);
void AttachWindow(const PresentationWindow& window);
void Present(const PresentationWindow& window, const DisplayBuffer* buffer, bool opaque, void (*gpuReady)(void*), void* context);
void ReleaseWindow(void* window);
void ReportFailure(std::exception_ptr error);
void* StaticTable();
void CheckGuestMemory(const void* pointer, std::size_t bytes, std::size_t alignment, bool writable);
void StoreGuestLabel(std::uint64_t* pointer, std::uint64_t value);
int AddEqEvent(KernelEqueue eq, int id, void* udata);
int DeleteEqEvent(KernelEqueue eq, int id);
std::uint32_t EqContextId(const KernelEvent* event);
[[noreturn]] void Unsupported(const char* function);

}

#endif
