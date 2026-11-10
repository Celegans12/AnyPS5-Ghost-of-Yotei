#include "common/profiler.h"

#include "common/diagFastPath.h"
#include "common/emulatorConfig.h"
#include "common/threads.h"

#include <common/TracyProtocol.hpp>
#include <common/TracyVersion.hpp>
#include <cstdio>
#include <tracy/Tracy.hpp>

namespace Profiler {

void SetThreadName(const char* name) {
	Common::Thread::SetCurrentName(name);
	if (tracy::ProfilerAvailable() && name != nullptr) {
		tracy::SetThreadName(name);
	}
}

void Initialize() {
	if (Config::ProfilerEnabled() && !tracy::ProfilerAvailable()) {
		tracy::StartupProfiler();
		TracySetProgramName("KytyPS5");
		::printf("Tracy profiler enabled: client %d.%d.%d, protocol %u, "
		         "broadcast %u, connect to 127.0.0.1:8086\n",
		         tracy::Version::Major, tracy::Version::Minor, tracy::Version::Patch,
		         tracy::ProtocolVersion, tracy::BroadcastVersion);
	}
	// BlockActive() (profiler.h) answers from this bit instead of tracy::ProfilerAvailable().
	Common::DiagFastPath::NoteProfiler(tracy::ProfilerAvailable());
}

void Shutdown() {
	if (tracy::ProfilerAvailable()) {
		// Cleared first: a zone that begins after this point is not recorded (the zones already open end as before).
		Common::DiagFastPath::NoteProfiler(false);
		tracy::ShutdownProfiler();
	}
}

} // namespace Profiler
