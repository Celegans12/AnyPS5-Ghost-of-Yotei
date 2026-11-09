#ifndef KYTY_APS5_PGOWRITER_H_
#define KYTY_APS5_PGOWRITER_H_

namespace Aps5 {

// KYTY_PGO=GEN builds only (CMake option; a no-op otherwise): starts the thread that writes the profile counters every
// KYTY_PGO_WRITE_SECONDS. Called once, after the APS5_KYTY_* settings are in the environment.
void StartPgoWriter();

} // namespace Aps5

#endif // KYTY_APS5_PGOWRITER_H_
