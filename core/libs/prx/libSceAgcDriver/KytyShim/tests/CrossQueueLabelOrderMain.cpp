#include <string_view>

extern "C" int RunKytyCrossQueueLabelOrderTest(bool buffer_write_coverage);
extern "C" int RunKytyRendererShutdownTest();

int main(int argc, char** argv) {
	if (argc == 2 && std::string_view(argv[1]) == "renderer-shutdown") {
		return RunKytyRendererShutdownTest();
	}
	return RunKytyCrossQueueLabelOrderTest(argc == 2 && std::string_view(argv[1]) == "buffer-write-coverage");
}
