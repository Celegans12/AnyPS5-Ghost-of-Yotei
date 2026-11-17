#include "common/profiler.h"

#include "common/emulatorConfig.h"

#include <common/TracyProtocol.hpp>
#include <common/TracyVersion.hpp>
#include <cstdio>
#include <tracy/Tracy.hpp>
#if defined(ANYPS5_KYTY)
#include <cstring>
#include <pthread.h>
#endif

namespace Profiler {

void SetThreadName(const char* name) {
#if defined(ANYPS5_KYTY)
	if (name != nullptr) {
		const char* tail = std::strncmp(name, "Thread_", 7) == 0 ? name + 7 : name;
		char host_name[16] = "kyty";
		std::strncat(host_name, tail, sizeof(host_name) - 1 - std::strlen(host_name));
		pthread_setname_np(pthread_self(), host_name);
	}
#endif
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
}

void Shutdown() {
	if (tracy::ProfilerAvailable()) {
		tracy::ShutdownProfiler();
	}
}

}
