#include "graphics/host_gpu/hotWriteWatch.h"

#if defined(__linux__)
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <linux/fs.h>
#include <linux/userfaultfd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef UFFD_USER_MODE_ONLY
#define UFFD_USER_MODE_ONLY 1
#endif
#ifndef UFFD_FEATURE_WP_HUGETLBFS_SHMEM
#define UFFD_FEATURE_WP_HUGETLBFS_SHMEM (1 << 12)
#endif
#ifndef UFFD_FEATURE_WP_UNPOPULATED
#define UFFD_FEATURE_WP_UNPOPULATED (1 << 13)
#endif
#ifndef UFFD_FEATURE_WP_ASYNC
#define UFFD_FEATURE_WP_ASYNC (1 << 15)
#endif
#ifndef PAGEMAP_SCAN
struct page_region {
	__u64 start;
	__u64 end;
	__u64 categories;
};
struct pm_scan_arg {
	__u64 size;
	__u64 flags;
	__u64 start;
	__u64 end;
	__u64 walk_end;
	__u64 vec;
	__u64 vec_len;
	__u64 max_pages;
	__u64 category_inverted;
	__u64 category_mask;
	__u64 category_anyof_mask;
	__u64 return_mask;
};
#define PAGE_IS_WRITTEN       (1 << 1)
#define PM_SCAN_WP_MATCHING   (1 << 0)
#define PM_SCAN_CHECK_WPASYNC (1 << 1)
#define PAGEMAP_SCAN          _IOWR('f', 16, struct pm_scan_arg)
#endif
#endif

namespace Libs::Graphics {

#if defined(__linux__)
namespace {

constexpr uint64_t PageBytes = 4096;

bool Protect(int uffd, uint64_t begin, uint64_t end, bool on) {
	uffdio_writeprotect protection {};
	protection.range.start = begin;
	protection.range.len   = end - begin;
	protection.mode        = on ? UFFDIO_WRITEPROTECT_MODE_WP : 0;
	return ioctl(uffd, UFFDIO_WRITEPROTECT, &protection) == 0;
}

bool Register(int uffd, uint64_t begin, uint64_t end) {
	uffdio_register registration {};
	registration.range.start = begin;
	registration.range.len   = end - begin;
	registration.mode        = UFFDIO_REGISTER_MODE_WP;
	return ioctl(uffd, UFFDIO_REGISTER, &registration) == 0;
}

void Unregister(int uffd, uint64_t begin, uint64_t end) {
	uffdio_range range {};
	range.start = begin;
	range.len   = end - begin;
	(void)ioctl(uffd, UFFDIO_UNREGISTER, &range);
}

long Scan(int pagemap, uint64_t begin, uint64_t end, std::array<page_region, 64>& regions, uint64_t* walk_end) {
	pm_scan_arg scan {};
	scan.size          = sizeof(scan);
	scan.flags         = PM_SCAN_WP_MATCHING | PM_SCAN_CHECK_WPASYNC;
	scan.start         = begin;
	scan.end           = end;
	scan.vec           = reinterpret_cast<uint64_t>(regions.data());
	scan.vec_len       = regions.size();
	scan.category_mask = PAGE_IS_WRITTEN;
	scan.return_mask   = PAGE_IS_WRITTEN;
	const auto count   = ioctl(pagemap, PAGEMAP_SCAN, &scan);
	*walk_end          = scan.walk_end;
	return count;
}

bool Probe(int uffd, int pagemap) {
	const int file = static_cast<int>(syscall(SYS_memfd_create, "hot write watch probe", MFD_CLOEXEC));
	if (file < 0) return false;
	constexpr uint64_t Bytes = 4 * PageBytes;
	bool               works = ftruncate(file, Bytes) == 0;
	void*              view  = works ? mmap(nullptr, Bytes, PROT_READ | PROT_WRITE, MAP_SHARED, file, 0) : MAP_FAILED;
	close(file);
	if (view == MAP_FAILED) return false;
	const auto begin = reinterpret_cast<uint64_t>(view);
	auto*      bytes = static_cast<volatile char*>(view);
	bytes[0]         = 1;
	works            = Register(uffd, begin, begin + Bytes) && Protect(uffd, begin, begin + Bytes, true);
	std::array<page_region, 64> regions {};
	const auto written = [&]() -> long {
		uint64_t   walk_end = 0;
		const auto count    = Scan(pagemap, begin, begin + Bytes, regions, &walk_end);
		if (count < 0) return -1;
		long pages = 0;
		for (long i = 0; i < count; i++) pages += static_cast<long>((regions[i].end - regions[i].start) / PageBytes);
		return pages;
	};
	works = works && written() == 0;
	if (works) {
		bytes[0]             = 2;
		bytes[3 * PageBytes] = 1;
		works                = written() == 2 && written() == 0;
	}
	munmap(view, Bytes);
	return works;
}

}

HotWriteWatch::HotWriteWatch() noexcept {
	m_uffd = static_cast<int>(syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK));
	if (m_uffd < 0 && errno == EPERM) {
		m_uffd = static_cast<int>(syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK | UFFD_USER_MODE_ONLY));
	}
	const char* failed = nullptr;
	if (m_uffd < 0) {
		failed = "userfaultfd";
	} else {
		uffdio_api api {};
		api.api      = UFFD_API;
		api.features = UFFD_FEATURE_WP_ASYNC | UFFD_FEATURE_WP_UNPOPULATED | UFFD_FEATURE_WP_HUGETLBFS_SHMEM;
		if (ioctl(m_uffd, UFFDIO_API, &api) != 0 || (api.features & UFFD_FEATURE_WP_ASYNC) == 0 ||
		    (api.features & UFFD_FEATURE_WP_HUGETLBFS_SHMEM) == 0 || (api.features & UFFD_FEATURE_WP_UNPOPULATED) == 0) {
			failed = "asynchronous userfaultfd write protection";
		} else if (m_pagemap = open("/proc/self/pagemap", O_RDONLY | O_CLOEXEC); m_pagemap < 0) {
			failed = "/proc/self/pagemap";
		} else if (!Probe(m_uffd, m_pagemap)) {
			failed = "PAGEMAP_SCAN on shared memory";
		}
	}
	if (failed != nullptr) {
		const int error = errno;
		if (m_pagemap >= 0) close(m_pagemap);
		if (m_uffd >= 0) close(m_uffd);
		m_pagemap = -1;
		m_uffd    = -1;
		std::fprintf(stderr, "[kyty-hot-pages] unavailable: %s failed (%s); every watched page keeps its fault\n", failed,
		             std::strerror(error));
	}
}

HotWriteWatch::~HotWriteWatch() {
	if (m_pagemap >= 0) close(m_pagemap);
	if (m_uffd >= 0) close(m_uffd);
}

bool HotWriteWatch::Watch(uint64_t begin, uint64_t end) const noexcept {
	if (!Available() || end <= begin) return false;
	if (!Register(m_uffd, begin, end)) return false;
	if (!Protect(m_uffd, begin, end, true)) {
		Unregister(m_uffd, begin, end);
		return false;
	}
	return true;
}

void HotWriteWatch::Unwatch(uint64_t begin, uint64_t end) const noexcept {
	if (!Available() || end <= begin) return;
	(void)Protect(m_uffd, begin, end, false);
	Unregister(m_uffd, begin, end);
}

bool HotWriteWatch::Collect(uint64_t begin, uint64_t end, Written written, void* context) const noexcept {
	if (!Available()) return false;
	std::array<page_region, 64> regions {};
	for (auto cursor = begin; cursor < end;) {
		uint64_t   walk_end = 0;
		const auto count    = Scan(m_pagemap, cursor, end, regions, &walk_end);
		if (count < 0) return false;
		for (long i = 0; i < count; i++) written(context, regions[i].start, regions[i].end);
		if (walk_end <= cursor || walk_end >= end) break;
		cursor = walk_end;
	}
	return true;
}
#else
HotWriteWatch::HotWriteWatch() noexcept = default;

HotWriteWatch::~HotWriteWatch() = default;

bool HotWriteWatch::Watch(uint64_t, uint64_t) const noexcept {
	return false;
}

void HotWriteWatch::Unwatch(uint64_t, uint64_t) const noexcept {}

bool HotWriteWatch::Collect(uint64_t, uint64_t, Written, void*) const noexcept {
	return false;
}
#endif

HotWriteWatch& HotWriteWatch::Get() noexcept {
	static auto* watch = new HotWriteWatch;
	return *watch;
}

}
