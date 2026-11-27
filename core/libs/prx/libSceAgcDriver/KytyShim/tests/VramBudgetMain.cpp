#include <string_view>

extern "C" int RunKytyVramBudgetTest(bool guest_eviction);

int main(int argc, char** argv) {
	return RunKytyVramBudgetTest(argc == 2 && std::string_view(argv[1]) == "guest-eviction");
}
