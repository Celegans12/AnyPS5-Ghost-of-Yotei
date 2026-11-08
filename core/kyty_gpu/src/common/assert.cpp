#include "common/assert.h"

#include "common/logging/log.h"
#include "common/subsystems.h"
#include "kytyGitVersion.h"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <string>

#if defined(_WIN32)
#include <windows.h> // IWYU pragma: keep
#endif

namespace Common {

static std::string BuildFatalReport(const char* title, std::string_view text, const char* file,
                                    int line) {
	return fmt::format("--- Build ---\n{}\n{}\n{} in {}:{}\n", KYTY_BUILD_LABEL, title, text, file,
	                   line);
}

// The call stack of a fatal error, as offsets inside the executable (resolve them with the linker map,
// e.g. build_upstream/kyty_emulator_clang_lld_link.map, as scripts/resolve-perf.py does), to stderr: a fatal
// check names only the line that failed, not who called it.
static void PrintFatalFrames() {
#if defined(_WIN32)
	void*       frames[48] {};
	const auto  count  = RtlCaptureStackBackTrace(1, 48, frames, nullptr);
	HMODULE     module = nullptr;
	// The module holding this code: the executable for Kyty, kyty_gpu.dll inside AnyPS5.
	if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
	                       reinterpret_cast<LPCWSTR>(&PrintFatalFrames), &module) == 0) {
		module = GetModuleHandleW(nullptr);
	}
	const auto  base  = reinterpret_cast<uint64_t>(module);
	std::string text  = fmt::format("FATAL_FRAMES exe_base=0x{:x}", base);
	for (USHORT i = 0; i < count; i++) {
		const auto address = reinterpret_cast<uint64_t>(frames[i]);
		text += address >= base && address < base + 0x10000000ull ? fmt::format(" +0x{:x}", address - base)
		                                                           : fmt::format(" 0x{:x}", address);
	}
	std::fprintf(stderr, "%s\n", text.c_str());
	std::fflush(stderr);
#endif
}

static FatalHook                g_fatal_hook = nullptr;
static thread_local std::string g_fatal_report;

void SetFatalHook(FatalHook hook) {
	g_fatal_hook = hook;
}

static int DbgReport(const char* title, std::string_view text, const char* file, int line) {
	PrintFatalFrames();
	g_fatal_report = BuildFatalReport(title, text, file, line);
	Log::WriteFatal(g_fatal_report);
	Subsystems::EmergencyShutdownActive();
	return 1;
}

int DbgExitIfHandler(const char* expr, const char* file, int line) {
	return DbgReport("--- Fatal Error ---", fmt::format("Error: condition ({}) is true", expr),
	                 file, line);
}

int DbgNotImplementedHandler(const char* expr, const char* file, int line) {
	return DbgReport("--- Fatal Error ---", fmt::format("Not implemented ({})", expr), file, line);
}

int DbgExitHandler(const char* file, int line, std::string_view text) {
	PrintFatalFrames();
	g_fatal_report = BuildFatalReport("--- Error ---", text, file, line);
	Log::WriteFatal(g_fatal_report);
	return 1;
}

int DbgExitHandler(const char* file, int line, fmt::text_style style, std::string_view text) {
	PrintFatalFrames();
	g_fatal_report = BuildFatalReport("--- Error ---", text, file, line);
	Log::WriteFatal(style, g_fatal_report);
	return 1;
}

// The k70 builds (27.10) ended 2 of 9 game runs with 0xC0000409 and no message: abort() (an uncaught exception reaches
// it through std::terminate), an invalid CRT parameter or a pure virtual call fail fast without these reporters.
static void ReportAbort(const char* kind) {
	std::fprintf(stderr, "FATAL_ABORT %s (thread %lu)\n", kind,
#if defined(_WIN32)
	             static_cast<unsigned long>(GetCurrentThreadId())
#else
	             0ul
#endif
	);
	PrintFatalFrames();
	std::fflush(nullptr);
}

void InstallAbortReporters() {
	std::signal(SIGABRT, [](int) {
		ReportAbort("abort (SIGABRT; std::terminate on an uncaught exception ends here)");
		std::_Exit(3);
	});
#if defined(_WIN32)
	_set_invalid_parameter_handler([](const wchar_t*, const wchar_t*, const wchar_t*, unsigned int, uintptr_t) {
		ReportAbort("invalid CRT parameter");
		std::_Exit(3);
	});
	_set_purecall_handler([] {
		ReportAbort("pure virtual call");
		std::_Exit(3);
	});
#endif
}

void DbgExit(int status) {
	Subsystems::EmergencyShutdownActive();
	std::fflush(nullptr);
	if (g_fatal_hook != nullptr) {
		g_fatal_hook(g_fatal_report.c_str(), status);
	}
	std::_Exit(status);
}

} // namespace Common
