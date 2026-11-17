#include "prx/libSceAgcDriver/KytyShim/src/EntryApi.hpp"
#include "prx/libSceAgcDriver/KytyShim/src/EntryPure.hpp"
#include "prx/libc/include/Shutdown.hpp"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "libs/agc.h"

#include <atomic>
#include <cinttypes>
#include <exception>
#include <span>
#include <type_traits>

namespace KytyShim::Present {

void WaitRendering(std::uint32_t handle, std::uint32_t index);

}

namespace KytyShim::Entry {

namespace {

void WaitForRendering(const RenderingWaitTarget& target) {
    Renderer().GetCommandScheduler().Flush();
    try {
        Present::WaitRendering(target.handle, target.index);
    } catch (const ProcessShutdown&) {
    } catch (const std::exception& error) {
        EXIT("AnyPS5 rendering wait: video output handle %" PRIu32 " buffer %" PRIu32 ": %s\n", target.handle, target.index, error.what());
    }
}

}

}

namespace Libs::Graphics {

uint32_t CpOpAnyPs5Flip(CommandProcessor& cp, uint32_t cmd_id, const uint32_t* buffer, uint32_t dw, uint32_t num_dw) {
    const auto* packet = buffer - 1;
    const auto flip = cmd_id == AgcDriver::FlipPacketHeader && dw >= AgcDriver::FlipPacketWords
        ? KytyShim::Entry::DecodeFlipPacket(std::span<const uint32_t>(packet, AgcDriver::FlipPacketWords))
        : std::nullopt;
    if (!flip) {
        EXIT("AnyPS5 flip packet at 0x%05" PRIx32 ": header 0x%08" PRIx32 " with %" PRIu32 " dwords left, expected 0x%08" PRIx32 " and %" PRIu32 "\n",
             num_dw - dw, cmd_id, dw, AgcDriver::FlipPacketHeader, AgcDriver::FlipPacketWords);
    }
    cp.WriteBackImages();
    cp.Flip(static_cast<int>(flip->handle), flip->index, static_cast<int>(flip->mode), flip->argument);
    if (KytyShim::Entry::OwnsFlipPacket(packet)) {
        cp.EmitHostCommand([packet] { KytyShim::Entry::ReleaseFlipPacket(packet); });
    }
    return AgcDriver::FlipPacketWords - 1u;
}

uint32_t CpOpAnyPs5RenderingWait(CommandProcessor& cp, uint32_t cmd_id, const uint32_t* buffer, uint32_t dw, uint32_t num_dw) {
    const auto target = cmd_id == AgcDriver::RenderingWaitPacketHeader && dw >= AgcDriver::RenderingWaitPacketWords
        ? KytyShim::Entry::DecodeRenderingWaitPacket(std::span<const uint32_t>(buffer - 1, AgcDriver::RenderingWaitPacketWords))
        : std::nullopt;
    if (!target) {
        EXIT("AnyPS5 rendering wait packet at 0x%05" PRIx32 ": header 0x%08" PRIx32 " with %" PRIu32 " dwords left, expected 0x%08" PRIx32 " and %" PRIu32 "\n",
             num_dw - dw, cmd_id, dw, AgcDriver::RenderingWaitPacketHeader, AgcDriver::RenderingWaitPacketWords);
    }
    if (target->mode != 0) EXIT("AnyPS5 rendering wait packet at 0x%05" PRIx32 ": mode %" PRIu32 " is not implemented\n", num_dw - dw, target->mode);
    cp.EmitHostCommand([wait = *target] { KytyShim::Entry::WaitForRendering(wait); });
    return AgcDriver::RenderingWaitPacketWords - 1u;
}

static_assert(std::is_same_v<decltype(&CpOpAnyPs5Flip), cp_op_parser_func_t>);
static_assert(std::is_same_v<decltype(&CpOpAnyPs5RenderingWait), cp_op_parser_func_t>);
static_assert(KytyShim::Entry::FlipCustomCode == KYTY_PM4_R(AgcDriver::FlipPacketHeader));
static_assert(KytyShim::Entry::RenderingWaitCustomCode == KYTY_PM4_R(AgcDriver::RenderingWaitPacketHeader));
static_assert(((AgcDriver::FlipPacketHeader >> 8u) & 0xffu) == Pm4::IT_NOP);
static_assert(((AgcDriver::RenderingWaitPacketHeader >> 8u) & 0xffu) == Pm4::IT_NOP);
static_assert(KYTY_PM4_LEN(AgcDriver::FlipPacketHeader) == AgcDriver::FlipPacketWords);
static_assert(KYTY_PM4_LEN(AgcDriver::RenderingWaitPacketHeader) == AgcDriver::RenderingWaitPacketWords);

void GraphicsDbgDumpDcb(const char* type, uint32_t num_dw, const uint32_t* cmd_buffer) {
    EXIT_IF(type == nullptr);

    static std::atomic_int id = 0;

    if (Config::CommandBufferDumpEnabled() && num_dw > 0 && cmd_buffer != nullptr) {
        Common::File f;
        const int frame = KytyShim::Entry::GpuStarted() ? KytyShim::Entry::Gpu().GetFrameNum() : 0;
        auto file_name = Config::GetCommandBufferDumpFolder() / fmt::format("{:04d}_{:04d}_buffer_{}.log", frame, id++, type);
        Common::File::CreateDirectories(file_name.parent_path());
        f.Create(file_name);
        if (f.IsInvalid()) {
            auto file_name_text = Common::PathToString(file_name);
            LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", file_name_text.c_str());
            return;
        }
        Pm4::DumpPm4PacketStream(&f, cmd_buffer, 0, num_dw);
        f.Close();
    }
}

namespace Gen5 {

namespace {

constexpr uint32_t AGC_DRAW_INDIRECT_MULTI_BEGIN_TAG  = 0xc6000008u;
constexpr uint32_t AGC_DRAW_INDIRECT_MULTI_END_TAG    = 0xc6000000u;
constexpr uint32_t AGC_WAIT_USER_DATA_BEGIN_32_TAG    = 0xc8010000u;
constexpr uint32_t AGC_WAIT_USER_DATA_BEGIN_64_TAG    = 0xc8020000u;
constexpr uint32_t AGC_WAIT_USER_DATA_END_TAG         = 0xc8000000u;
constexpr uint32_t AGC_MEM_SEMAPHORE_SIGNAL_BEGIN_TAG = 0xc9010000u;
constexpr uint32_t AGC_MEM_SEMAPHORE_WAIT_BEGIN_TAG   = 0xc9020000u;
constexpr uint32_t AGC_MEM_SEMAPHORE_END_TAG          = 0xc9000000u;
constexpr uint32_t AGC_WAIT_USER_DATA_BEGIN_DW        = 4u;
constexpr uint32_t AGC_INTERNAL_DATA_PACKET_SIZE_DW   = 3u;

}

bool AgcIsInternalDataPacket(uint32_t cmd_id, const uint32_t* payload) {
    if (payload == nullptr || ((cmd_id >> 8u) & 0xffu) != Pm4::IT_SET_UCONFIG_REG ||
        KYTY_PM4_R(cmd_id) != 1u || payload[0] != Pm4::SQ_THREAD_TRACE_USERDATA_2) {
        return false;
    }

    const auto size_dw = KYTY_PM4_LEN(cmd_id);
    if (size_dw == AGC_WAIT_USER_DATA_BEGIN_DW) {
        const auto tag = payload[1] & 0xffff0000u;
        return tag == AGC_WAIT_USER_DATA_BEGIN_32_TAG || tag == AGC_WAIT_USER_DATA_BEGIN_64_TAG ||
               tag == AGC_MEM_SEMAPHORE_SIGNAL_BEGIN_TAG || tag == AGC_MEM_SEMAPHORE_WAIT_BEGIN_TAG;
    }

    return size_dw == AGC_INTERNAL_DATA_PACKET_SIZE_DW &&
           (payload[1] == AGC_WAIT_USER_DATA_END_TAG || payload[1] == AGC_MEM_SEMAPHORE_END_TAG ||
            payload[1] == AGC_DRAW_INDIRECT_MULTI_BEGIN_TAG ||
            payload[1] == AGC_DRAW_INDIRECT_MULTI_END_TAG);
}

}

}
