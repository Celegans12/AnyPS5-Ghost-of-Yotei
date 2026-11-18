#include "prx/libSceAgcDriver/Execution/include/UpscaleMode.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

namespace {

void Require(bool value, const char* what) {
    if (value) return;
    std::fprintf(stderr, "kyty_shim_presentation_upscale: %s\n", what);
    std::abort();
}

template<typename Function>
bool Refused(Function&& function) {
    try {
        function();
        return false;
    } catch (const std::exception&) {
        return true;
    }
}

struct Extent3D {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 1;
};

struct Extent2D {
    uint32_t width = 0;
    uint32_t height = 0;
};

struct Offset2D {
    int32_t x = 0;
    int32_t y = 0;
};

struct Rect2D {
    Offset2D offset;
    Extent2D extent;
};

Rect2D FitAspect(const Extent3D& source, const Extent2D& target) {
    const uint64_t src_w = std::max(source.width, 1u);
    const uint64_t src_h = std::max(source.height, 1u);
    uint32_t w = target.width;
    uint32_t h = target.height;
    if (src_w * h > src_h * w) {
        h = std::max<uint32_t>(1, static_cast<uint32_t>(src_h * w / src_w));
    } else {
        w = std::max<uint32_t>(1, static_cast<uint32_t>(src_w * h / src_h));
    }
    return {{static_cast<int32_t>((target.width - w) / 2),
             static_cast<int32_t>((target.height - h) / 2)},
            {w, h}};
}

bool DecidesUpscale(bool fsr, const Extent3D& source, const Extent2D& target) {
    const auto rect = FitAspect(source, target);
    return fsr && (rect.extent.width > source.width || rect.extent.height > source.height);
}

void TestUpscaleModeSelection() {
    using AgcDriver::SelectUpscaleMode;
    const auto unset = SelectUpscaleMode(nullptr, nullptr);
    Require(!unset.fsr && unset.sharpness == 0.2f, "unset ANYPS5_UPSCALE does not default to disabled FSR with 0.2 sharpness");
    Require(!SelectUpscaleMode("", "").fsr, "empty ANYPS5_UPSCALE is not treated as unset");
    const auto fsr = SelectUpscaleMode("fsr", "1.5");
    Require(fsr.fsr && fsr.sharpness == 1.5f, "ANYPS5_UPSCALE=fsr with ANYPS5_FSR_SHARPNESS=1.5 failed selection");
    Require(SelectUpscaleMode("fsr", "0").sharpness == 0.0f, "ANYPS5_FSR_SHARPNESS=0 rejected");
    Require(SelectUpscaleMode("fsr", "2.0").sharpness == 2.0f, "ANYPS5_FSR_SHARPNESS=2.0 rejected");
    Require(SelectUpscaleMode("fsr", nullptr).sharpness == 0.2f, "ANYPS5_FSR_SHARPNESS unset with FSR did not default to 0.2");
    Require(SelectUpscaleMode("fsr", "").sharpness == 0.2f, "ANYPS5_FSR_SHARPNESS empty with FSR did not default to 0.2");

    for (const char* value : {"FSR", "fsr1", "bilinear", " fsr", "fsr ", "dlss", "1"}) {
        Require(Refused([value] { static_cast<void>(SelectUpscaleMode(value, nullptr)); }), (std::string("ANYPS5_UPSCALE accepted ") + value).c_str());
    }
    for (const char* value : {"-0.1", "2.01", "abc", "0.5x", "nan", "inf", " 1", "1 "}) {
        Require(Refused([value] { static_cast<void>(SelectUpscaleMode("fsr", value)); }), (std::string("ANYPS5_FSR_SHARPNESS accepted ") + value).c_str());
    }
}

void TestFitAspectAndUpscaleDecisions() {
    const Extent3D display_1080p {1920, 1080, 1};
    const Extent3D display_720p {1280, 720, 1};

    Require(DecidesUpscale(true, display_1080p, {2560, 1440}), "1080p to 1440p 16:9 failed to decide upscale");
    Require(DecidesUpscale(true, display_1080p, {3840, 2160}), "1080p to 4K failed to decide upscale");
    Require(DecidesUpscale(true, display_720p, {1920, 1080}), "720p to 1080p failed to decide upscale");

    Require(!DecidesUpscale(false, display_1080p, {2560, 1440}), "FSR disabled decided upscale on 1440p");
    Require(!DecidesUpscale(true, display_1080p, {1920, 1080}), "1:1 presentation decided upscale");
    Require(!DecidesUpscale(true, display_1080p, {1280, 720}), "downscale decided upscale");

    const auto ultrawide_fit = FitAspect(display_1080p, {2560, 1080});
    Require(ultrawide_fit.extent.width == 1920 && ultrawide_fit.extent.height == 1080, "ultrawide fit extent incorrect");
    Require(ultrawide_fit.offset.x == 320 && ultrawide_fit.offset.y == 0, "ultrawide fit letterbox offset incorrect");
    Require(!DecidesUpscale(true, display_1080p, {2560, 1080}), "ultrawide pillarbox at 1:1 height decided upscale");

    const auto aspect_16_10 = FitAspect(display_1080p, {1920, 1200});
    Require(aspect_16_10.extent.width == 1920 && aspect_16_10.extent.height == 1080, "16:10 fit extent incorrect");
    Require(aspect_16_10.offset.x == 0 && aspect_16_10.offset.y == 60, "16:10 letterbox offset incorrect");
    Require(!DecidesUpscale(true, display_1080p, {1920, 1200}), "16:10 letterbox at 1:1 width decided upscale");
}

void TestDynamicResolutionScalingTransitions() {
    const Extent2D target_4k {3840, 2160};
    const Extent2D target_1080p {1920, 1080};

    const Extent3D drs_frames[] = {
        {1920, 1080, 1},
        {1600, 900, 1},
        {1440, 810, 1},
        {1280, 720, 1},
        {1920, 1080, 1},
        {2560, 1440, 1},
    };

    for (const auto& frame : drs_frames) {
        Require(DecidesUpscale(true, frame, target_4k), "DRS frame failed to decide upscale to 4K");
    }

    Require(!DecidesUpscale(true, drs_frames[0], target_1080p), "1080p to 1080p DRS frame decided upscale");
    Require(DecidesUpscale(true, drs_frames[1], target_1080p), "900p to 1080p DRS frame failed to decide upscale");
    Require(DecidesUpscale(true, drs_frames[2], target_1080p), "810p to 1080p DRS frame failed to decide upscale");
    Require(DecidesUpscale(true, drs_frames[3], target_1080p), "720p to 1080p DRS frame failed to decide upscale");
    Require(!DecidesUpscale(true, drs_frames[5], target_1080p), "1440p downscaled to 1080p DRS frame decided upscale");
}

void TestSlotRotationInvariants() {
    constexpr uint32_t image_count = 3;
    uint32_t frame_index = 0;
    uint64_t ticks[image_count] = {0, 0, 0};

    for (uint32_t turn = 1; turn <= 10; ++turn) {
        const uint32_t slot = frame_index;
        Require(ticks[slot] < turn, "slot reuse collided with incomplete submission tick");
        ticks[slot] = turn;
        frame_index = (frame_index + 1u) % image_count;
    }
}

}

int main() {
    TestUpscaleModeSelection();
    TestFitAspectAndUpscaleDecisions();
    TestDynamicResolutionScalingTransitions();
    TestSlotRotationInvariants();
    std::puts("kyty_shim_presentation_upscale: ok");
    return 0;
}
