#include <string_view>

extern "C" int RunKytyPm4ContextRegistersTest();
extern "C" int RunKytyPm4SxDownconvertTest();

int main(int argc, char** argv) {
	if (argc == 2 && std::string_view(argv[1]) == "sx") return RunKytyPm4SxDownconvertTest();
	if (argc != 1) return 1;
	return RunKytyPm4ContextRegistersTest();
}
