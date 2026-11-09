#include "aps5/pgoWriter.h"

#if defined(KYTY_PGO_GENERATE)

#include "aps5/aps5Host.h"

#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep

// clang's profile runtime (libclang_rt.profile), linked into instrumented builds.
extern "C" {
int  __llvm_profile_write_file(void);
void __llvm_profile_set_filename(const char* name);
}

namespace Aps5 {

namespace {

std::string ProfileDir() {
	const char* dir = std::getenv("KYTY_PGO_DIR");
	return dir != nullptr && dir[0] != '\0' ? std::string(dir) : std::string(KYTY_PGO_DEFAULT_DIR);
}

} // namespace

// KYTY_PGO=GEN: the profile runtime writes the counters only at a clean exit, and AnyPS5's runs end by killing the
// process. So the counters are written every KYTY_PGO_WRITE_SECONDS (default 60; 0: only at exit) into KYTY_PGO_DIR
// (default: the configure-time directory) as kyty_gpu-<pid>-<seconds>s.profraw. Each file holds every count since the
// process started, so the last complete file of a run is the run's profile; a kill during a write leaves only that
// file incomplete (llvm-profdata refuses it, and the one before it is used).
void StartPgoWriter() {
	const char*    seconds_text = std::getenv("KYTY_PGO_WRITE_SECONDS");
	const uint32_t seconds =
	    seconds_text != nullptr && seconds_text[0] != '\0' ? static_cast<uint32_t>(std::strtoul(seconds_text, nullptr, 10)) : 60u;
	const auto        dir      = ProfileDir();
	const auto        pid      = std::to_string(GetCurrentProcessId());
	static std::string exit_name;
	exit_name = dir + "/kyty_gpu-" + pid + "-exit.profraw";
	__llvm_profile_set_filename(exit_name.c_str());
	Log("PGO: instrumented build; counters written to %s/kyty_gpu-%s-<seconds>s.profraw every %u s (0: only at exit)",
	    dir.c_str(), pid.c_str(), seconds);
	if (seconds == 0) return;
	std::thread([dir, pid, seconds] {
		const auto start = std::chrono::steady_clock::now();
		for (uint64_t n = 1;; n++) {
			std::this_thread::sleep_until(start + std::chrono::seconds(seconds * n));
			const auto name = dir + "/kyty_gpu-" + pid + "-" + std::to_string(seconds * n) + "s.profraw";
			__llvm_profile_set_filename(name.c_str());
			const auto t0 = std::chrono::steady_clock::now();
			const int  rc = __llvm_profile_write_file();
			const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
			__llvm_profile_set_filename(exit_name.c_str());
			Log("PGO: counters written to %s (rc %d, %lld ms)", name.c_str(), rc, static_cast<long long>(ms));
		}
	}).detach();
}

} // namespace Aps5

#else

namespace Aps5 {

void StartPgoWriter() {}

} // namespace Aps5

#endif
