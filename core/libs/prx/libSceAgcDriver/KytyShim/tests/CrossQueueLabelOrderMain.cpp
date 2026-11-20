#include <string_view>

extern "C" int RunKytyCrossQueueLabelOrderTest(bool buffer_write_coverage);

int main(int argc, char** argv) {
	return RunKytyCrossQueueLabelOrderTest(argc == 2 && std::string_view(argv[1]) == "buffer-write-coverage");
}
