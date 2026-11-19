#include "graphics/host_gpu/hotWriteWatch.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {

using Libs::Graphics::HotWriteWatch;

constexpr uint64_t Page  = 4096;
constexpr uint64_t Pages = 8;

void Require(bool condition, const char* what) {
	if (!condition) {
		std::fprintf(stderr, "kyty_shim_hot_write_watch: FAILED: %s\n", what);
		std::exit(1);
	}
}

struct Views {
	int            file    = -1;
	volatile char* guest   = nullptr;
	volatile char* backing = nullptr;

	Views() {
		file = static_cast<int>(syscall(SYS_memfd_create, "hot write watch test", 0));
		Require(file >= 0 && ftruncate(file, Pages * Page) == 0, "memfd");
		guest   = static_cast<volatile char*>(mmap(nullptr, Pages * Page, PROT_READ | PROT_WRITE, MAP_SHARED, file, 0));
		backing = static_cast<volatile char*>(mmap(nullptr, Pages * Page, PROT_READ | PROT_WRITE, MAP_SHARED, file, 0));
		Require(guest != MAP_FAILED && backing != MAP_FAILED, "two views");
	}

	[[nodiscard]] uint64_t At(uint64_t page) const { return reinterpret_cast<uint64_t>(guest) + page * Page; }
};

std::set<uint64_t> Collect(const HotWriteWatch& watch, uint64_t begin, uint64_t end, bool* ok) {
	std::set<uint64_t> pages;
	*ok = watch.Collect(
	    begin, end,
	    [](void* context, uint64_t from, uint64_t to) {
		    for (auto page = from; page < to; page += Page) static_cast<std::set<uint64_t>*>(context)->insert(page);
	    },
	    &pages);
	return pages;
}

std::set<uint64_t> CollectOk(const HotWriteWatch& watch, uint64_t begin, uint64_t end) {
	bool ok    = false;
	auto pages = Collect(watch, begin, end, &ok);
	Require(ok, "collection of watched pages succeeds");
	return pages;
}

}

int main() {
	HotWriteWatch watch;
	if (!watch.Available()) {
		std::printf("kyty_shim_hot_write_watch: skipped, asynchronous userfaultfd write protection is unavailable\n");
		return 77;
	}
	Views views;
	const auto first = views.At(1);
	const auto end   = views.At(4);

	views.guest[1 * Page] = 1;
	Require(watch.Watch(first, end), "watch three pages, one populated and two not");
	Require(CollectOk(watch, first, end).empty(), "no write right after the watch");

	views.guest[1 * Page + 5] = 2;
	Require(CollectOk(watch, first, end) == std::set<uint64_t> {views.At(1)}, "a guest write is collected");
	Require(CollectOk(watch, first, end).empty(), "a collected write is protected again");

	views.guest[3 * Page] = 3;
	Require(CollectOk(watch, first, end) == std::set<uint64_t> {views.At(3)}, "a write to a never populated page is collected");

	views.backing[2 * Page] = 4;
	views.guest[0]          = 5;
	views.guest[5 * Page]   = 6;
	Require(CollectOk(watch, first, end).empty(), "backing view and unwatched pages are not collected");
	Require(views.guest[2 * Page] == 4, "the guest view sees the backing write");

	Require(mprotect(const_cast<char*>(views.guest + 2 * Page), Page, PROT_READ) == 0, "read only");
	Require(mprotect(const_cast<char*>(views.guest + 2 * Page), Page, PROT_READ | PROT_WRITE) == 0, "writable again");
	Require(CollectOk(watch, first, end).empty(), "mprotect alone is no write");
	views.guest[2 * Page] = 7;
	Require(CollectOk(watch, first, end) == std::set<uint64_t> {views.At(2)}, "the watch survives mprotect");

	bool ok = true;
	(void)Collect(watch, views.At(0), end, &ok);
	Require(!ok, "a range with an unwatched page fails instead of hiding writes");

	views.guest[0]          = 10;
	views.guest[1 * Page]   = 11;
	std::set<uint64_t> loose;
	Require(watch.Collect(
	            views.At(0), views.At(6),
	            [](void* context, uint64_t from, uint64_t to) {
		            for (auto page = from; page < to; page += Page) static_cast<std::set<uint64_t>*>(context)->insert(page);
	            },
	            &loose, false),
	        "a loose collection walks over unwatched pages");
	Require(loose == std::set<uint64_t> {views.At(1)}, "a loose collection reports the watched written page only");
	Require(CollectOk(watch, first, end).empty(), "a loose collection protects again what it reports");

	watch.Unwatch(views.At(2), views.At(3));
	views.guest[2 * Page] = 8;
	(void)Collect(watch, first, end, &ok);
	Require(!ok, "an unwatched page fails the collection");
	Require(CollectOk(watch, first, views.At(2)).empty() && CollectOk(watch, views.At(3), end).empty(), "the rest stays watched");

	void* replaced = mmap(const_cast<char*>(views.guest + 3 * Page), Page, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, views.file,
	                      3 * Page);
	Require(replaced == views.guest + 3 * Page, "map the page again");
	views.guest[3 * Page] = 9;
	(void)Collect(watch, views.At(3), end, &ok);
	Require(!ok, "a mapping that replaced a watched page fails the collection");

	Require(watch.Watch(views.At(6), views.At(7)), "watch another page");
	for (int round = 0; round < 1000; round++) {
		views.guest[6 * Page + (round % 64)] = static_cast<char>(round);
		Require(CollectOk(watch, views.At(6), views.At(7)).size() == 1, "every round is collected once");
	}

	std::printf("kyty_shim_hot_write_watch: ok\n");
	return 0;
}
