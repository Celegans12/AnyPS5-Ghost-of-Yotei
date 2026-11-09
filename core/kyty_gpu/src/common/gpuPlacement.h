#ifndef KYTY_COMMON_GPUPLACEMENT_H_
#define KYTY_COMMON_GPUPLACEMENT_H_

#include <array>
#include <cstdint>

// Where the GPU stage threads, the BAR copy helpers and the guest's job workers run. Default: the start placement of
// Config::ConfigureGpuStageThread (stages pinned, their physical cores reserved), and nothing here runs.
//
// KYTY_PLACEMENT_LAYOUT_<n>=<field>+<field>+... (n = 1..4; '/' separates fields too) defines layout n. KYTY_PERF_PLACEMENT=<n> runs layout n from
// the first frame; 0 (the default) is the start placement. KYTY_LIVE_AB may list KYTY_PERF_PLACEMENT=<n> once for each
// layout to compare them in one run (its phase "off" is the start placement). Fields, each at most once; a field left
// out keeps the start placement's choice:
//   stages:<e>,<r>,<v>,<c>   the logical processors of the execution, recording, resolve and command processor threads
//   default:reserve|all|<cpus>
//                            the process default CPU sets (for every thread that chooses no processors): reserve = all
//                            logical processors outside the stages' physical cores (the start placement), all = none set
//   barcopy:<cpus>           the BAR copy helper threads, one logical processor each, in turn
//   jobs:<cpus>              the guest threads whose name starts with KYTY_PLACEMENT_JOB_PREFIX (default "JobWorker")
//   prio:0|1                 whether the execution and command processor threads run at the raised priority
// <cpus>: logical processors of group 0 and ranges, "0,1,3-5". Only where and at what priority threads run changes;
// nothing they compute does, so a layout leaves every output as it is and can change at any frame boundary.
namespace Common::GpuPlacement {

// Config::ConfigureGpuStageThread, once, before the first stage thread pins itself: the start placement.
void SetStart(const std::array<int, 4>& stage_cpus, bool reserved, bool raised_priority);
// Config::ConfigureGpuStageThread, on each stage thread after it pinned itself (stage: Config::GpuStageThread).
void RegisterStage(uint32_t stage);
// A BAR copy helper thread, on itself, once.
void RegisterBarCopyThread();
// The GPU thread, at each frame boundary (after Common::LiveAb::OnFrameBoundary): runs the layout KYTY_PERF_PLACEMENT
// selects, live or from the start.
void OnFrameBoundary();

} // namespace Common::GpuPlacement

#endif // KYTY_COMMON_GPUPLACEMENT_H_
