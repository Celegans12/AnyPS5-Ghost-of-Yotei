#include "graphics/host_gpu/renderer/rayTracing.h"

#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct RayTracingTestAccess {
    static bool CanGather(bool replaced, uint64_t built_sequence, uint32_t dynamic_until) {
        RayTracing::Slot slot;
        slot.built           = true;
        slot.replaced        = replaced;
        slot.built_refit_seq = built_sequence;
        slot.refit_seq       = 9;
        slot.dynamic_until   = dynamic_until;
        return slot.CanGather(slot.refit_seq);
    }
};

}

namespace {

using Libs::Graphics::RayTracing;

int failures = 0;

void Expect(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAILED: %s\n", what);
    ++failures;
}

RayTracing::HleRefitGeometry Geometry(uint64_t blas, uint32_t index) {
    RayTracing::HleRefitGeometry geometry;
    geometry.blas     = blas;
    geometry.geometry = index;
    return geometry;
}

void DropsOutsideGuest() {
    constexpr uint64_t Low      = 0x100000000ull;
    constexpr uint64_t Edge     = (uint64_t {1} << 40u) - 0x200u;
    constexpr uint64_t Gap      = uint64_t {1} << 41u;
    constexpr uint64_t Extended = 0x80000001000ull;
    constexpr uint64_t Small    = 0x200000000ull;
    std::vector<uint64_t> blases {Low, 0, Gap, Edge, Extended, Small};
    std::vector<uint64_t> read;
    const auto size_of = [&](uint64_t blas) -> uint32_t {
        read.push_back(blas);
        if (blas == Edge) return 0x1000u;
        if (blas == Small) return 128u;
        return 0x800u;
    };
    std::vector<RayTracing::HleRefitGeometry> geometries {Geometry(Low, 0), Geometry(Gap, 0), Geometry(Edge, 1),
                                                          Geometry(Extended, 2), Geometry(Gap, 3)};
    std::vector<uint32_t> sizes;
    const auto dropped = RayTracing::DropHleRefitOutsideGuest(blases, sizes, geometries, size_of);
    Expect(blases == std::vector<uint64_t> {Low, 0, Extended, Small},
           "BLASes outside guest memory leave the refit, the others keep their order");
    Expect(sizes == std::vector<uint32_t> {0x800u, 0u, 0x800u, 128u},
           "each kept BLAS keeps the size read from its header, a null BLAS has none");
    Expect(dropped.size() == 2 && dropped[0] == std::pair<uint64_t, uint32_t> {Gap, 0u} &&
               dropped[1] == std::pair<uint64_t, uint32_t> {Edge, 0x1000u},
           "a BLAS whose header is outside guest memory is dropped unread, one whose size runs out of it with that size");
    Expect(read == std::vector<uint64_t> {Low, Edge, Extended, Small},
           "the size is read only from headers inside guest memory, never for a null BLAS");
    Expect(geometries.size() == 2 && geometries[0].blas == Low && geometries[1].blas == Extended,
           "geometries of dropped BLASes leave the refit");
}

void KeepsEverythingInside() {
    std::vector<uint64_t>                     blases {0x100000000ull, 0x100001000ull};
    std::vector<uint32_t>                     sizes {7u};
    std::vector<RayTracing::HleRefitGeometry> geometries {Geometry(0x100000000ull, 0)};
    const auto dropped = RayTracing::DropHleRefitOutsideGuest(blases, sizes, geometries, [](uint64_t) { return 512u; });
    Expect(dropped.empty() && blases.size() == 2 && sizes == std::vector<uint32_t> {512u, 512u} && geometries.size() == 1,
           "nothing is dropped inside guest memory, and the sizes are those of this call only");
}

}

extern "C" int RunKytyRtRefitGuestTest() {
    DropsOutsideGuest();
    KeepsEverythingInside();
    using Libs::Graphics::RayTracingTestAccess;
    Expect(RayTracingTestAccess::CanGather(false, 8, 0), "a consecutive refit gathers its triangle entries");
    Expect(!RayTracingTestAccess::CanGather(true, 8, 0), "a replacement during a consecutive refit needs a walk");
    Expect(RayTracingTestAccess::CanGather(false, 9, 1), "a dynamic mesh can gather again in the same refit");
    Expect(!RayTracingTestAccess::CanGather(true, 9, 1), "a replacement of a dynamic mesh needs a walk");
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("kyty shim ray tracing refit guard: all checks passed\n");
    return 0;
}
