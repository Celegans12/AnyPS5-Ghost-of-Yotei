#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_WINDOWSMAPPINGS_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_WINDOWSMAPPINGS_HPP

#ifdef _WIN32
#include <windows.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <map>
#include <set>
#include <memory>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <string>
#include <system_error>

namespace GuestArena {

class WindowsMappings {
public:
    enum class Collection { All, Armed, Fresh };

    static WindowsMappings& Get() {
        static auto* mappings = new WindowsMappings;
        return *mappings;
    }

#ifdef APS5_COUNT_MAPPING_QUERIES
    static std::uint64_t MemoryQueries() {
        return queries.load(std::memory_order_relaxed);
    }
#endif

    void* Reserve(void* address, std::size_t bytes) {
        return allocate(GetCurrentProcess(), address, bytes, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0);
    }

    std::vector<std::pair<std::uintptr_t, std::size_t>> Commit(void* address, std::size_t bytes, DWORD protection, std::size_t granule, bool watched) {
        if (granule == 0) throw std::invalid_argument("guest memory commit granule is zero");
        std::vector<std::pair<std::uintptr_t, std::size_t>> created;
        std::lock_guard lock(mutex);
        collectableRanges.clear();
        disown(reinterpret_cast<std::uintptr_t>(address), reinterpret_cast<std::uintptr_t>(address) + bytes);
        const DWORD flags = MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER | (watched ? MEM_WRITE_WATCH : 0);
        const auto cut = [](std::uintptr_t at, std::size_t size, const char* operation) {
            if (!VirtualFree(reinterpret_cast<void*>(at), size, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) fail(operation);
        };
        const auto replace = [&](std::uintptr_t at, std::size_t size) {
            if (!allocate(GetCurrentProcess(), reinterpret_cast<void*>(at), size, flags, protection, nullptr, 0)) fail("replace guest placeholder with private memory");
            privateAllocations.emplace(at, at + size);
            if (watched) markZero(at, at + size);
            if (!created.empty() && created.back().first + created.back().second == at) created.back().second += size;
            else created.emplace_back(at, size);
        };
        const auto end = reinterpret_cast<std::uintptr_t>(address) + bytes;
        for (auto cursor = reinterpret_cast<std::uintptr_t>(address); cursor < end;) {
            const auto memory = query(cursor);
            const auto regionEnd = reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize;
            const auto stop = std::min(end, regionEnd);
            if (memory.State == MEM_RESERVE && std::min(end, cursor + granule) <= regionEnd) {
                privateAllocations.erase(privateAllocations.lower_bound(cursor), privateAllocations.lower_bound(stop));
                forgetClean(cursor, stop);
                const auto holder = reinterpret_cast<std::uintptr_t>(memory.AllocationBase);
                if (holder != cursor) cut(holder, cursor - holder, "split guest placeholder prefix");
                if (stop != regionEnd) cut(cursor, stop - cursor, "split guest placeholder suffix");
                while (cursor < stop && std::min(end, cursor + granule) <= stop) {
                    const auto size = std::min(end, cursor + granule) - cursor;
                    if (cursor + size != stop) cut(cursor, size, "split guest placeholder suffix");
                    replace(cursor, size);
                    cursor += size;
                }
                continue;
            }
            if (memory.State == MEM_RESERVE) {
                const auto limit = std::min(end, cursor + granule);
                auto placeholderEnd = stop;
                while (placeholderEnd < limit) {
                    const auto next = query(placeholderEnd);
                    if (next.State != MEM_RESERVE) break;
                    placeholderEnd = reinterpret_cast<std::uintptr_t>(next.BaseAddress) + next.RegionSize;
                }
                const auto size = std::min(limit, placeholderEnd) - cursor;
                reset(cursor, size);
                replace(cursor, size);
                cursor += size;
            } else {
                if (memory.State != MEM_COMMIT) throw std::runtime_error("guest memory is not committed");
                for (auto mapped = views.lower_bound(cursor & ~(pageBytes - 1)); mapped != views.end() && mapped->first < stop; ++mapped) {
                    mapped->second.protection = protection;
                    mapped->second.armed = 0;
                    invalidate(mapped->first, mapped->second);
                }
                openArmed(cursor, stop);
                DWORD previous;
                if (!VirtualProtect(reinterpret_cast<void*>(cursor), stop - cursor, protection, &previous)) fail("protect guest memory");
                cursor = stop;
            }
        }
        return created;
    }

    void Reset(void* address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        reset(reinterpret_cast<std::uintptr_t>(address), bytes);
    }

    void MapZero(void* address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        if (zeroSection == nullptr && (zeroSection = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READONLY, 0, static_cast<DWORD>(pageBytes), nullptr)) == nullptr) fail("create the read-only zero guest page");
        const auto start = reinterpret_cast<std::uintptr_t>(address);
        reset(start, bytes);
        for (auto cursor = start; cursor < start + bytes; cursor += pageBytes) {
            split(cursor, pageBytes);
            if (!map(zeroSection, GetCurrentProcess(), reinterpret_cast<void*>(cursor), 0, pageBytes, MEM_REPLACE_PLACEHOLDER, PAGE_READONLY, nullptr, 0)) fail("map the read-only zero guest page");
        }
    }

    void Map(void* address, std::size_t bytes, HANDLE section, std::uint64_t offset, DWORD protection, bool watched = false, bool resident = false, DWORD viewProtection = PAGE_EXECUTE_READWRITE, std::size_t sectionBytes = SIZE_MAX, std::vector<std::pair<std::uintptr_t, std::size_t>>* createdPrivate = nullptr) {
        if (sectionBytes < bytes && (resident || sectionBytes == 0 || bytes - sectionBytes >= pageBytes)) throw std::runtime_error("shared guest mapping extends a whole page past its section");
        std::lock_guard lock(mutex);
        collectableRanges.clear();
        auto cursor = reinterpret_cast<std::uintptr_t>(address);
        if (createdPrivate != nullptr) createdPrivate->reserve(createdPrivate->size() + bytes / pageBytes);
        reset(cursor, bytes);
        if (resident) watchResident = watched;
        HANDLE duplicate = nullptr;
        if (!DuplicateHandle(GetCurrentProcess(), section, GetCurrentProcess(), &duplicate, 0, FALSE, DUPLICATE_SAME_ACCESS)) fail("keep shared guest section");
        const auto owned = std::make_shared<Section>(duplicate);
        const auto key = reinterpret_cast<std::uintptr_t>(section);
        const auto slotOf = [&](std::size_t done) -> const std::shared_ptr<Physical>& {
            auto& slot = physical[std::make_pair(key, offset + done)];
            if (!slot) slot = std::make_shared<Physical>();
            return slot;
        };
        const auto pageOf = [](Physical& slot) {
            auto shared = slot.page.lock();
            if (!shared) {
                shared = std::make_shared<SharedPage>();
                shared->comparedWhenShared = slot.comparedWhenShared;
                slot.page = shared;
            }
            return shared;
        };
        const auto single = [&](std::size_t done) {
            if (!resident || residentRun(key, offset + done) != residentRuns.end()) return false;
            const auto found = physical.find(std::make_pair(key, offset + done));
            if (found == physical.end()) return true;
            const auto shared = found->second->page.lock();
            return !found->second->stored && (!shared || shared->aliases.empty());
        };
        const bool perPage = !resident && sectionBytes == SIZE_MAX;
        if (perPage) commitSection(section, offset, bytes);
        for (std::size_t done = 0; done < bytes;) {
            const auto base = cursor + done;
            if (single(done)) {
                auto runBytes = pageBytes;
                while (done + runBytes < bytes && (base + runBytes) % chunkBytes != 0 && single(done + runBytes)) runBytes += pageBytes;
                split(base, runBytes);
                commitResident(base, runBytes);
                if (createdPrivate != nullptr) {
                    if (!createdPrivate->empty() && createdPrivate->back().first + createdPrivate->back().second == base) createdPrivate->back().second += runBytes;
                    else createdPrivate->emplace_back(base, runBytes);
                }
                const auto run = residentRuns.emplace(base, Run{base + runBytes, {}, static_cast<std::uint32_t>(runBytes >> 12), key, offset + done}).first;
                run->second.watched = watchResident;
                residentIndex.emplace(std::make_pair(key, offset + done), run);
                for (std::size_t at = 0; at < runBytes; at += pageBytes) {
                    bool compared = false;
                    if (const auto found = physical.find(std::make_pair(key, offset + done + at)); found != physical.end()) {
                        const auto& slot = *found->second;
                        if (slot.saved) {
                            std::memcpy(reinterpret_cast<void*>(base + at), slot.saved->bytes + slot.savedOffset, pageBytes);
                            forgetZero(base + at, base + at + pageBytes);
                        }
                        compared = slot.comparedWhenShared;
                        physical.erase(found);
                    }
                    views.emplace(base + at, View{nullptr, protection, {}, 0, compared, 0, owned, offset + done + at, 0, 0, true});
                }
                forgetClean(base, base + runBytes);
                remember(freshRanges, base, base + runBytes);
                DWORD previous;
                if (protection != PAGE_READWRITE && !VirtualProtect(reinterpret_cast<void*>(base), runBytes, protection, &previous)) fail("protect resident guest pages");
                done += runBytes;
                continue;
            }
            if (sectionBytes != SIZE_MAX) {
                if (const auto run = residentRun(key, offset + done); run != residentRuns.end()) demote(run->first);
                const auto slot = slotOf(done);
                const auto shared = pageOf(*slot);
                for (const auto alias : shared->aliases) views.at(alias).aliasDemoted = false;
                const auto viewBytes = std::min(pageBytes, sectionBytes - done);
                split(base, pageBytes);
                mapShared(base, section, offset + done, protection, viewProtection, viewBytes);
                slot->stored = true;
                shared->aliases.push_back(base);
                const auto created = views.emplace(base, View{shared, protection, {}, 0, false, 0, owned, offset + done, 0, 0, false, slot, hostPages(viewBytes)}).first;
                if (shared->comparedWhenShared) created->second.unreported = allSubpages;
                remember(freshRanges, base, base + pageBytes);
                invalidate(*shared);
                done += pageBytes;
                continue;
            }
            std::vector<std::pair<std::shared_ptr<Physical>, std::shared_ptr<SharedPage>>> pages;
            pages.reserve((bytes - done) / pageBytes);
            auto runBytes = std::size_t{0};
            do {
                if (const auto run = residentRun(key, offset + done + runBytes); run != residentRuns.end()) demote(run->first);
                const auto slot = slotOf(done + runBytes);
                const auto shared = pageOf(*slot);
                for (const auto alias : shared->aliases) views.at(alias).aliasDemoted = false;
                pages.emplace_back(slot, shared);
                runBytes += pageBytes;
            } while (done + runBytes < bytes && !single(done + runBytes));
            if (!perPage) commitSection(section, offset + done, runBytes);
            split(base, runBytes);
            mapRun(base, runBytes, section, offset + done);
            DWORD previous;
            if (!VirtualProtect(reinterpret_cast<void*>(base), runBytes, protection, &previous)) fail("protect shared guest pages");
            collectableRanges.clear();
            for (std::size_t at = 0; at < runBytes; at += pageBytes) {
                auto& [slot, shared] = pages[at / pageBytes];
                slot->stored = true;
                shared->aliases.push_back(base + at);
                invalidate(*shared);
                const auto created = views.emplace(base + at, View{std::move(shared), protection, {}, 0, false, 0, owned, offset + done + at, 0, 0, false, std::move(slot)}).first;
                if (created->second.page->comparedWhenShared) created->second.unreported = allSubpages;
            }
            remember(freshRanges, base, base + runBytes);
            done += runBytes;
        }
    }

    void SetProtection(std::uintptr_t address, std::size_t bytes, DWORD protection) {
        std::lock_guard lock(mutex);
        disown(address, address + bytes);
        collectableRanges.clear();
        // The caller applies the guest protection after this returns, outside the lock: a block armed
        // in between would keep a stale record of being writable, so mprotected memory is never armed.
        const auto first = address & ~std::uintptr_t{4095};
        const auto last = (address + bytes + 4095) & ~std::uintptr_t{4095};
        openArmed(first, last);
        remember(unarmable, first, last);
        for (auto it = views.lower_bound(address); it != views.end() && it->first < address + bytes; ++it) {
            it->second.protection = protection;
            it->second.armed = 0;
            invalidate(it->first, it->second);
        }
    }

    void Pin(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        disown(address, address + bytes);
        ownedPins.emplace(address, address + bytes);
        openArmed(address & ~std::uintptr_t{4095}, (address + bytes + 4095) & ~std::uintptr_t{4095});
        for (auto it = views.lower_bound(address & ~(pageBytes - 1)); it != views.end() && it->first < address + bytes; ++it) {
            if (it->second.resident) {
                ++it->second.pins;
                invalidate(it->first, it->second);
                continue;
            }
            auto& page = *it->second.page;
            ++page.pins;
            for (const auto alias : page.aliases) {
                auto& view = views.at(alias);
                if (view.armed == 0) continue;
                protectSubpages(alias, view.armed, view.protection, "pin shared guest page writable", view.viewBytes);
                view.armed = 0;
            }
            invalidate(page);
        }
    }

    void Unpin(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        disown(address, address + bytes);
        eraseOwnedRange(ownedPins, address, address + bytes);
        retryImports(address, address + bytes);
        for (auto it = views.lower_bound(address & ~(pageBytes - 1)); it != views.end() && it->first < address + bytes; ++it) {
            auto& pins = it->second.resident ? it->second.pins : it->second.page->pins;
            if (pins != 0) --pins;
            invalidate(it->first, it->second);
        }
    }

    void CompareWhenShared(std::uintptr_t address, std::size_t bytes, bool residentOnly = false) {
        std::lock_guard lock(mutex);
        collectableRanges.clear();
        for (auto it = views.lower_bound(address & ~(pageBytes - 1)); it != views.end() && it->first < address + bytes; ++it) {
            if (residentOnly && !it->second.resident && !it->second.aliasDemoted) continue;
            if (it->second.resident) {
                it->second.comparedWhenShared = true;
                continue;
            }
            auto& page = *it->second.page;
            const bool marked = page.comparedWhenShared;
            page.comparedWhenShared = true;
            it->second.slot->comparedWhenShared = true;
            for (const auto alias : page.aliases) {
                auto& view = views.at(alias);
                if (view.resident) continue;
                if (!marked) view.unreported = allSubpages;
                if (view.armed == 0) continue;
                protectSubpages(alias, view.armed, view.protection, "stop arming a compared shared page", view.viewBytes);
                view.armed = 0;
            }
        }
    }

    void ComparedPages(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count, bool unreported) {
        std::lock_guard lock(mutex);
        const auto capacity = *count;
        *count = 0;
        const auto end = address + bytes;
        for (auto it = views.lower_bound(address & ~(pageBytes - 1)); it != views.end() && it->first < end; ++it) {
            auto& view = it->second;
            if (view.resident || !view.page->comparedWhenShared) continue;
            const auto listed = static_cast<std::uint8_t>(subpagesOf(it->first, address, end) & (unreported ? view.unreported : allSubpages));
            for (std::size_t sub = 0; sub < subpages; ++sub) {
                const auto bit = static_cast<std::uint8_t>(1u << sub);
                if ((listed & bit) == 0) continue;
                if (*count == capacity) return;
                pages[(*count)++] = reinterpret_cast<void*>(it->first + sub * subpageBytes);
                if (unreported) view.unreported &= static_cast<std::uint8_t>(~bit);
            }
        }
    }

    bool HandleWrite(std::uintptr_t address, std::uintptr_t instruction = 0) {
        if (const auto owner = submitter.load(std::memory_order_acquire); owner != 0 && protectTrace())
            std::fprintf(stderr, "[protect] write fault at 0x%llx on thread %lu during the submission of thread %lu\n", static_cast<unsigned long long>(address), GetCurrentThreadId(), owner);
        std::lock_guard lock(mutex);
        if (armedAt(address)) {
            // Section A: a CPU-clean block becomes CPU-dirty. The OS write watch records the store
            // once it resumes, and the next collect walks the block again.
            const auto block = address & ~(armBlockBytes - 1);
            forgetClean(block, block + armBlockBytes);
            ++armFaults;
            if (groupProbe() || groupWalkProbe()) {
                const auto next = probeGroups.upper_bound(address);
                if (next != probeGroups.begin() && std::prev(next)->second > address) {
                    const auto [start, end] = *std::prev(next);
                    ++groupFaultBlocks[{start, end}][block];
                } else {
                    ++unmappedGroupFaults;
                }
            }
            if (walkProbe() && (armFaults & 63) == 1) {
                ++faultBlocks[block];
                ++faultSites[instruction];
            }
            if (overlapsStore(block, block + armBlockBytes)) remember(faultedBlocks, block, block + armBlockBytes);
            auto& strikes = armStrikes[block];
            strikes = static_cast<std::uint8_t>(std::min(255, strikes + armFaultStrikes));
            return true;
        }
        const auto base = address & ~(pageBytes - 1);
        const auto found = views.find(base);
        // A store that faulted while its block was being armed (the arming thread held the lock and
        // then left the block writable): retry it.
        if (found == views.end()) return protectTracking() && privateAllocationEnd(address) != 0 && writable(query(address).Protect);
        if (!writable(found->second.protection)) return false;
        auto& view = found->second;
        if (view.resident) {
            const auto memory = query(address);
            return memory.State == MEM_COMMIT && writable(memory.Protect & 0xffu);
        }
        const auto written = subpagesOf(base, address, address + 1);
        invalidate(*view.page, written);
        protectSubpages(base, written, view.protection, "resume shared memory write", view.viewBytes);
        view.armed &= static_cast<std::uint8_t>(~written);
        return true;
    }

    bool HandleRead(std::uintptr_t address) {
        std::lock_guard lock(mutex);
        if (!views.contains(address & ~(pageBytes - 1))) return false;
        const auto memory = query(address);
        return memory.State == MEM_COMMIT && (memory.Protect & (PAGE_NOACCESS | PAGE_GUARD)) == 0;
    }

    bool ResidentSpan(std::uintptr_t address, std::size_t bytes, std::uintptr_t* start, std::uintptr_t* end) {
        std::lock_guard lock(mutex);
        auto run = residentRuns.upper_bound(address);
        if (run != residentRuns.begin() && std::prev(run)->second.end > address) --run;
        if (run == residentRuns.end() || run->first >= address + bytes) return false;
        *start = run->first;
        for (; run != residentRuns.end() && run->first < address + bytes; ++run) *end = run->second.end;
        return true;
    }

    bool ForgetSection(HANDLE section) {
        std::lock_guard lock(mutex);
        const auto key = reinterpret_cast<std::uintptr_t>(section);
        const auto first = physical.lower_bound(std::make_pair(key, std::uint64_t{0}));
        const auto last = physical.lower_bound(std::make_pair(key + 1, std::uint64_t{0}));
        bool blank = std::none_of(first, last, [](const auto& entry) { return entry.second->stored || !entry.second->page.expired(); });
        physical.erase(first, last);
        for (auto run = residentIndex.lower_bound(std::make_pair(key, std::uint64_t{0})); run != residentIndex.end() && run->first.first == key; run = residentIndex.erase(run)) blank = false;
        return blank;
    }

    bool BeginHostWrite(std::uintptr_t address, std::size_t bytes, bool store = true) {
        std::lock_guard lock(mutex);
        const auto first = views.lower_bound(address & ~(pageBytes - 1));
        const auto end = address + bytes;
        for (auto it = first; it != views.end() && it->first < end; ++it) {
            if (!writable(it->second.protection)) return false;
        }
        disown(address, end);
        forgetZero(address, end);
        ownedHostAccess.emplace(address, end);
        // Kernel and other non-faulting writes (ReadFile, WriteProcessMemory, imports) need the
        // pages writable for real; the open bracket keeps the range from being armed again.
        openArmed(address & ~std::uintptr_t{4095}, (end + 4095) & ~std::uintptr_t{4095});
        for (auto it = first; it != views.end() && it->first < end; ++it) {
            auto& view = it->second;
            if (store) ++view.hostWrites;
            else ++view.hostImports;
            const auto written = subpagesOf(it->first, address, end);
            if (store) invalidate(it->first, view, written);
            else if (view.resident) {
                forgetClean(it->first, it->first + pageBytes);
                markImported(it->first);
            }
            const auto opened = static_cast<std::uint8_t>((store || view.resident) ? view.armed & written : 0);
            if (opened == 0) continue;
            protectSubpages(it->first, opened, view.protection, "open shared memory to a host write", view.viewBytes);
            view.armed &= static_cast<std::uint8_t>(~opened);
        }
        return true;
    }

    void EndHostWrite(std::uintptr_t address, std::size_t bytes, bool store = true) {
        std::lock_guard lock(mutex);
        const auto end = address + bytes;
        disown(address, end);
        eraseOwnedRange(ownedHostAccess, address, end);
        retryImports(address, end);
        for (auto it = views.lower_bound(address & ~(pageBytes - 1)); it != views.end() && it->first < end; ++it) {
            if (store) --it->second.hostWrites;
            else --it->second.hostImports;
            if (store) invalidate(it->first, it->second, subpagesOf(it->first, address, end));
            else if (it->second.resident) forgetClean(it->first, it->first + pageBytes);
        }
    }

    static std::vector<bool> TouchedPages(std::uintptr_t start, std::size_t pages, void* const* written, std::size_t count) {
        std::vector<bool> touched(pages);
        for (std::size_t index = 0; index < count; ++index) {
            const auto page = (reinterpret_cast<std::uintptr_t>(written[index]) - start) / pageBytes;
            if (page < pages) touched[page] = true;
        }
        return touched;
    }

    void markImported(std::uintptr_t page) {
        const auto next = residentRuns.upper_bound(page);
        if (next == residentRuns.begin()) return;
        const auto run = std::prev(next);
        if (page < run->second.end) run->second.imported = true;
    }

    void* MapAlias(std::uintptr_t address, std::size_t bytes, bool* boundsRefused = nullptr) {
        std::lock_guard lock(mutex);
        if (boundsRefused != nullptr) *boundsRefused = false;
        const auto refuse = [&](const char* reason) {
            char text[192];
            std::snprintf(text, sizeof(text), "read-write alias of shared guest memory 0x%llx+0x%llx: %s", static_cast<unsigned long long>(address), static_cast<unsigned long long>(bytes), reason);
            return std::runtime_error(text);
        };
        if (address % pageBytes != 0 || bytes % pageBytes != 0 || bytes == 0) throw refuse("the range is not made of whole shared pages");
        for (std::size_t done = 0; done < bytes; done += pageBytes) {
            const auto found = views.find(address + done);
            if (found == views.end() || !found->second.resident) continue;
            const auto run = std::prev(residentRuns.upper_bound(address + done));
            if (run->first < address || run->second.end > address + bytes) {
                if (boundsRefused != nullptr) *boundsRefused = true;
                throw refuse("a resident run of direct memory extends outside the range");
            }
            for (auto page = views.find(run->first); page != views.end() && page->first < run->second.end; ++page) {
                if (page->second.hostWrites != 0) return nullptr;
            }
        }
        for (std::size_t done = 0; done < bytes; done += pageBytes) {
            const auto found = views.find(address + done);
            if (found != views.end() && found->second.resident) demote(address + done, true);
        }
        auto view = views.find(address);
        if (view == views.end()) throw refuse("the range does not start at a shared view");
        const auto section = view->second.section;
        const auto offset = view->second.offset;
        SYSTEM_INFO system{};
        GetSystemInfo(&system);
        bool contiguous = true;
        for (std::size_t done = 0; done < bytes; done += pageBytes) {
            const auto page = views.find(address + done);
            if (page == views.end()) throw refuse("the range is not a run of shared views");
            if (page->second.viewBytes != pageBytes) throw refuse("a page extends past the end of its file");
            if (page->second.offset != offset + done || (page->second.section != section && !sameSection(page->second.section->handle, section->handle))) contiguous = false;
        }
        if (!contiguous) return mapAliasRuns(address, bytes);
        const auto lead = offset % system.dwAllocationGranularity;
        void* alias = map(section->handle, GetCurrentProcess(), nullptr, offset - lead, lead + bytes, 0, PAGE_READWRITE, nullptr, 0);
        if (alias == nullptr) {
            char text[160];
            std::snprintf(text, sizeof(text), "MapViewOfFile3 of a read-write alias of shared guest memory 0x%llx+0x%llx", static_cast<unsigned long long>(address), static_cast<unsigned long long>(bytes));
            throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), text);
        }
        return static_cast<char*>(alias) + lead;
    }

    void UnmapAlias(void* alias) {
        if (alias == nullptr) return;
        {
            std::lock_guard lock(mutex);
            if (const auto found = splitAliases.find(reinterpret_cast<std::uintptr_t>(alias)); found != splitAliases.end()) {
                const auto error = releaseAlias(found->first, found->second);
                splitAliases.erase(found);
                if (error != ERROR_SUCCESS) throw std::system_error(static_cast<int>(error), std::system_category(), "unmap a run of a shared guest alias");
                return;
            }
        }
        SYSTEM_INFO system{};
        GetSystemInfo(&system);
        const auto base = reinterpret_cast<std::uintptr_t>(alias) & ~(static_cast<std::uintptr_t>(system.dwAllocationGranularity) - 1);
        if (!unmap(GetCurrentProcess(), reinterpret_cast<void*>(base), 0)) fail("unmap shared guest alias");
    }

    bool Protection(std::uintptr_t address, std::uint32_t* protection) {
        std::lock_guard lock(mutex);
        if (armedAt(address)) {
            // Armed for write tracking only: to the guest and the driver it is writable.
            *protection = PAGE_READWRITE;
            return true;
        }
        const auto found = views.find(address & ~(pageBytes - 1));
        if (found == views.end()) return false;
        *protection = found->second.protection;
        return true;
    }

    // Section A (event-driven write tracking): private and resident guest memory that a resetting
    // collect found consumed is armed read-only (CPU-clean) and skipped by later collects without a
    // system call; a guest store faults once, opens its 64 KiB block (CPU-dirty) and the next collect
    // walks it. Needs the libc exception handler, so tests without it keep the walk-only model.
    // APS5_NO_PROTECT_TRACKING=1 turns it off.
    // Precondition (red-zone safety): a fault on a guest thread makes Windows write the exception
    // frame below RSP, so arming is allowed only when the executable carries the relinker's mark for
    // a conversion framed with no leaf skipped in any module (FramedMark at FramedMarkOffset of its
    // DOS header). Programs without guest code (tests, probes) pass unframed, or set
    // APS5_PROTECT_TRACKING=1.
    static constexpr std::size_t FramedMarkOffset = 0x28;
    static constexpr char FramedMark[8] = {'A', 'P', 'S', '5', 'R', 'Z', '0', '1'};

    static void EnableWriteFaults(bool unframed = false) {
        static const bool forced = std::getenv("APS5_PROTECT_TRACKING") != nullptr;
        if (!unframed && !forced) {
            const auto* image = reinterpret_cast<const char*>(GetModuleHandleW(nullptr));
            if (image == nullptr || std::memcmp(image + FramedMarkOffset, FramedMark, sizeof(FramedMark)) != 0) return;
        }
        writeFaultsHandled().store(true, std::memory_order_release);
    }

    static bool Arming() {
        return protectTracking();
    }

    // Memory a thread may run on as its stack (the guest main stack): never armed, as an exception
    // cannot be delivered on a read-only stack and the process would end without a report.
    void NeverArm(std::uintptr_t address, std::size_t bytes) {
        if (bytes == 0) return;
        std::lock_guard lock(mutex);
        const auto first = address & ~std::uintptr_t{4095};
        const auto last = (address + bytes + 4095) & ~std::uintptr_t{4095};
        openArmed(first, last);
        remember(unarmable, first, last);
    }

    // Section A at a submission. Measured on the RTX 5080 (probe-patches\import_protect_probe.cpp,
    // 2026-11-13): the driver drops the GPU writes of a submission into a host import whose pages are
    // partly read-only at vkQueueSubmit (about 1 in 12, no error), while an import that is uniformly
    // read-only or read-write at the call is unaffected, also when its protection changes later
    // (in flight or still queued). So every import is made uniform here, and the lock is held until
    // EndSubmit, so no fault or host write opens a block during the call: the blocks opened since the
    // last submission are re-armed (the submission consumes the dirty list), and an import holding
    // memory that cannot be armed is opened as a whole.
    // Only an import the submission stores through must be uniform (a read of a mixed import loses
    // nothing: 0 of 60, and a store through its own uniform import over the same pages loses
    // nothing: 0 of 120 fills, 0 of 120 shader stores, 2026-11-14). So the driver stores through
    // small store imports of their own (RegisterImport with store set) and names the guest ranges
    // each submission stores into: `stores` went through store imports, and only the groups of
    // store imports holding them are made uniform (armed as a whole, or open as a whole). `regular`
    // went through ordinary imports: every armed page of the import group holding them is opened
    // for this submission. With all set the stores are unknown and every armed page inside any
    // import is opened. An ordinary import is otherwise armed per block by the collects
    // (armWalked) and may stay mixed.
    void BeginSubmit(const std::pair<std::uint64_t, std::uint64_t>* stores, std::size_t count, const std::pair<std::uint64_t, std::uint64_t>* regular, std::size_t regularCount, bool all) {
        mutex.lock();
        ++submits;
        if (anyPendingGpuWrites) countGpuWrites();
        while (!hotRetries.empty() && hotRetries.begin()->first <= submits) {
            const auto [from, to] = hotRetries.begin()->second;
            hotRetries.erase(hotRetries.begin());
            if (!overlapsStore(from, to)) continue;
            forget(openImports, from, to);
            if (walkProbe()) for (auto& cause : openCauses) forget(cause, from, to);
            remember(unevenImports, from, to);
        }
        if (!protectTracking()) {
            submitter.store(GetCurrentThreadId(), std::memory_order_release);
            return;
        }
        try {
            if (all) {
                ++submitsAll;
                std::vector<std::pair<std::uintptr_t, std::uintptr_t>> armed;
                for (const auto& [start, end] : armedRanges) {
                    if (overlapsImport(start, end)) armed.emplace_back(start, end);
                }
                for (const auto& [start, end] : armed) {
                    regularOpened += end - start;
                    openArmed(start, end);
                }
            } else {
                if (count != 0 || regularCount != 0) ++submitsStoring;
                for (std::size_t i = 0; i < regularCount; ++i) {
                    auto from = static_cast<std::uintptr_t>(regular[i].first), to = static_cast<std::uintptr_t>(regular[i].second);
                    if (from >= to || !importGroup(importExtents, longestImport, from, to)) continue;
                    ++regularStores;
                    if (!overlaps(armedRanges, from, to)) continue;
                    ++regularOpens;
                    auto it = armedRanges.lower_bound(from);
                    if (it != armedRanges.begin() && std::prev(it)->second > from) --it;
                    for (; it != armedRanges.end() && it->first < to; ++it) regularOpened += std::min(to, it->second) - std::max(from, it->first);
                    openArmed(from, to);
                }
                for (std::size_t i = 0; i < count; ++i) {
                    auto from = static_cast<std::uintptr_t>(stores[i].first), to = static_cast<std::uintptr_t>(stores[i].second);
                    if (from >= to || !importGroup(storeExtents, longestStore, from, to)) continue;
                    if (!overlaps(unevenImports, from, to)) continue;
                    ++storeUniforms;
                    uniform(from, to);
                    forget(unevenImports, from, to);
                }
            }
        } catch (...) {
            mutex.unlock();
            throw;
        }
        submitter.store(GetCurrentThreadId(), std::memory_order_release);
    }

    // Grows [from, to) to the group of the given imports overlapping it (the imports overlapping it,
    // transitively). False when none overlaps it.
    static bool importGroup(const std::multimap<std::uintptr_t, std::uintptr_t>& extents, std::uintptr_t longest, std::uintptr_t& from, std::uintptr_t& to) {
        bool any = false;
        for (bool grew = true; grew;) {
            grew = false;
            for (auto it = extents.lower_bound(from > longest ? from - longest : 0); it != extents.end() && it->first < to; ++it) {
                if (it->second <= from) continue;
                any = true;
                if (it->first < from) from = it->first, grew = true;
                if (it->second > to) to = it->second, grew = true;
            }
        }
        return any;
    }

    void EndSubmit() {
        submitter.store(0, std::memory_order_release);
        mutex.unlock();
    }

    // Diagnostic only (APS5_WALK_PROBE with APS5_WALK_PROBE_GROUP_WALKS): the driver notes a guest
    // range the GPU will write (label stores, storage write-backs, reach notes). The notes are
    // charged to the next submission of any kind, per import group (printGpuWrites).
    // kind: 0 own label store, 1 note on an in-flight batch, 2 storage target created, 3 write-back
    // windows, 4 storage layers marked GPU-pending, 5 write-back layers, 6 a shader's bound storage
    // writes, 7 reach (ranges an address-chasing shader could store into), 8 other command stores
    // (fills, copies, metadata). Kinds 2, 4 and 7 are not counted as stores.
    void GpuWrites(std::uintptr_t address, std::size_t bytes, unsigned kind) {
        if (!groupWalkProbe() || bytes == 0 || kind >= GpuWriteKinds) return;
        std::lock_guard lock(mutex);
        ++gpuWriteNotes;
        anyPendingGpuWrites = true;
        remember(pendingGpuWrites[kind], address, address + bytes);
    }

    // A host import of the guest's own pages (no alias) was made: its extent is kept until
    // ForgetImport (the import can outlive the guest range), because an import must be uniformly
    // protected at each submission that stores through it (BeginSubmit). With store set it is a
    // store import: the driver stores through it and names those stores. With used false the GPU
    // has not used the import yet: its pages stay open until ImportUsed, as a first use over
    // read-only pages loses the import's later writes where another import covers the same pages.
    void RegisterImport(std::uintptr_t address, std::size_t bytes, bool used, bool store) {
        if (bytes == 0) return;
        std::lock_guard lock(mutex);
        const auto from = address & ~std::uintptr_t{4095};
        const auto to = (address + bytes + 4095) & ~std::uintptr_t{4095};
        importExtents.emplace(from, to);
        longestImport = std::max(longestImport, to - from);
        if (store) {
            storeExtents.emplace(from, to);
            longestStore = std::max(longestStore, to - from);
            remember(unevenImports, from, to);
        }
        if (!used) unusedImports.emplace(from, to);
        openArmed(from, to);
    }

    // Changes whenever armed (CPU-clean) memory was opened; unchanged means no store landed in any
    // memory that was armed at the earlier reading.
    std::uint64_t ArmedOpens() const {
        return armedOpens.load(std::memory_order_acquire);
    }

    void ImportUsed(std::uintptr_t address, std::size_t bytes) {
        if (bytes == 0) return;
        std::lock_guard lock(mutex);
        const auto from = address & ~std::uintptr_t{4095};
        const auto to = (address + bytes + 4095) & ~std::uintptr_t{4095};
        eraseOwnedRange(unusedImports, from, to);
        retryImports(from, to);
    }

    // The import registered with the same range was destroyed.
    void ForgetImport(std::uintptr_t address, std::size_t bytes, bool store) {
        if (bytes == 0) return;
        std::lock_guard lock(mutex);
        const auto from = address & ~std::uintptr_t{4095};
        const auto to = (address + bytes + 4095) & ~std::uintptr_t{4095};
        eraseOwnedRange(importExtents, from, to);
        if (store) eraseOwnedRange(storeExtents, from, to);
        if (groupProbe() || groupWalkProbe()) forget(probeGroups, from, to);
        eraseOwnedRange(unusedImports, from, to);
        retryImports(from, to);
        if (!overlapsStore(from, to)) {
            forget(openImports, from, to);
            if (walkProbe()) for (auto& cause : openCauses) forget(cause, from, to);
            forget(unevenImports, from, to);
            forget(faultedBlocks, from, to);
            groupHeat.erase(from);
        }
    }

    // Whether no CPU write can have landed in [address, address + bytes) since its last resetting
    // collect: the whole range is clean (armed private or resident memory, or protection-tracked
    // shared views). A collect of such a range reports nothing, so callers may skip it.
    bool Clean(std::uintptr_t address, std::size_t bytes) {
        if (bytes == 0) return false;
        std::lock_guard lock(mutex);
        // Armed and clean only: shared views the write-watch model keeps clean are not walkable,
        // and their collect must still answer as the walk does.
        const auto from = address & ~std::uintptr_t{4095}, to = (address + bytes + 4095) & ~std::uintptr_t{4095};
        return covered(armedRanges, from, to) && covered(cleanRanges, from, to);
    }

    struct ProtectCounts {
        std::uint64_t walks = 0;
        std::uint64_t armedRuns = 0;
        std::uint64_t faults = 0;
        std::uint64_t hotSkips = 0;
        std::uint64_t raced = 0;
        std::uint64_t armedBytes = 0;
        std::uint64_t armedPieces = 0;
        std::uint64_t submits = 0;
        std::uint64_t submitArms = 0;
        std::uint64_t importOpens = 0;
    };

    ProtectCounts ProtectCounted() {
        std::lock_guard lock(mutex);
        ProtectCounts counts{collectWalks, armRuns, armFaults, armHotSkips, armRaced, 0, armedRanges.size(), submits, submitArms, importOpens};
        for (const auto& [start, end] : armedRanges) counts.armedBytes += end - start;
        return counts;
    }

#ifdef APS5_COUNT_MAPPING_QUERIES
    void CollectWork(std::uint64_t* systemCalls, std::uint64_t* pieces) {
        std::lock_guard lock(mutex);
        *systemCalls = collectSystemCalls;
        *pieces = collectPieces;
    }
#endif

    static constexpr std::size_t ownedBlockBytes = 0x10000;
#ifdef GUEST_OWNED_LIMIT_TEST
    static constexpr std::size_t ownedEntryLimit = 8;
#else
    static constexpr std::size_t ownedEntryLimit = 65536;
#endif
    static constexpr unsigned OwnedPrivate = 1;

    bool ArmOwned(std::uintptr_t address, std::size_t bytes, std::uint64_t* words, std::size_t count, unsigned flags = 0) {
        std::lock_guard lock(mutex);
        if (words == nullptr || (flags & ~OwnedPrivate) != 0 || !ownedRange(address, bytes, count)) return false;
        if (nextOwned == 0 || count > UINT64_MAX - nextOwned + 1) return false;
        const auto first = address & ~(ownedBlockBytes - 1);
        const auto end = (address + bytes - 1) | (ownedBlockBytes - 1);
        if (count > ownedEntryLimit) return false;
        std::size_t alreadyPending = 0;
        for (auto it = ownedPending.lower_bound(first); it != ownedPending.end() && it->first <= end; ++it) ++alreadyPending;
        if (count - alreadyPending > ownedEntryLimit - ownedPending.size()) return false;
        for (auto block = first; block < end; block += ownedBlockBytes) {
            if (!ownable(block, flags)) return false;
        }
        disown(first, end + 1);
        while (ownedWords.size() > ownedEntryLimit - count) ownedWords.erase(ownedWords.begin());
        for (auto block = first; block < end; block += ownedBlockBytes) {
            auto& pending = ownedPending[block];
            auto run = residentRuns.upper_bound(block);
            if (run != residentRuns.begin() && std::prev(run)->second.end > block) --run;
            for (; run != residentRuns.end() && run->first < block + ownedBlockBytes; ++run) run->second.reset = true;
            std::array<void*, ownedBlockBytes / 4096> written{};
            ULONG_PTR reported = written.size();
            DWORD granularity = 0;
            if (watchWrites(WRITE_WATCH_FLAG_RESET, reinterpret_cast<void*>(block), ownedBlockBytes, written.data(), &reported, &granularity) != 0) return false;
            for (ULONG_PTR index = 0; index < reported; ++index) pending |= std::uint16_t{1} << ((reinterpret_cast<std::uintptr_t>(written[index]) - block) / 4096);
            if (pending == 0) ownedPending.erase(block);
        }
        for (auto block = first; block < end; block += ownedBlockBytes) {
            ownedWords[block] = nextOwned++;
        }
        for (auto block = first; block < end; block += ownedBlockBytes) words[(block - first) / ownedBlockBytes] = ownedWords.at(block);
        return true;
    }

    bool OwnedUnchanged(std::uintptr_t address, std::size_t bytes, const std::uint64_t* words, std::size_t count) {
        std::lock_guard lock(mutex);
        if (words == nullptr || !ownedRange(address, bytes, count)) return false;
        const auto first = address & ~(ownedBlockBytes - 1);
        const auto end = (address + bytes - 1) | (ownedBlockBytes - 1);
        for (auto block = first; block < end; block += ownedBlockBytes) {
            const auto found = ownedWords.find(block);
            if (found == ownedWords.end() || found->second != words[(block - first) / ownedBlockBytes]) return false;
            std::array<void*, ownedBlockBytes / 4096> written{};
            ULONG_PTR reported = written.size();
            DWORD granularity = 0;
            if (watchWrites(0, reinterpret_cast<void*>(block), ownedBlockBytes, written.data(), &reported, &granularity) != 0 || reported != 0) {
                ownedWords.erase(found);
                return false;
            }
        }
        return true;
    }

    void ForgetKnownZero(std::uintptr_t address, std::size_t bytes) {
        if (bytes == 0 || bytes > UINTPTR_MAX - address) return;
        std::lock_guard lock(mutex);
        forgetZero(address, address + bytes);
    }

    void KnownZero(std::uintptr_t address, std::size_t bytes, void (*zero)(void* context, std::uintptr_t begin, std::uintptr_t end), void* context) {
        if (bytes == 0 || bytes > UINTPTR_MAX - address) return;
        const auto end = address + bytes;
        std::uintptr_t reportStart = 0;
        std::uintptr_t reportEnd = 0;
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> found;
        const auto report = [&](std::uintptr_t from, std::uintptr_t to) {
            from = std::max(from, address);
            to = std::min(to, end);
            if (from >= to) return;
            if (!found.empty() && found.back().second == from) found.back().second = to;
            else found.emplace_back(from, to);
        };
        const auto deliver = [&] {
            for (const auto& [from, to] : found) {
                if (reportEnd == from) {
                    reportEnd = to;
                    continue;
                }
                if (reportEnd > reportStart) zero(context, reportStart, reportEnd);
                reportStart = from;
                reportEnd = to;
            }
            found.clear();
        };
        for (auto cursor = address & ~std::uintptr_t{4095}; cursor < end;) {
            const auto chunk = cursor & ~(chunkBytes - 1);
            const auto chunkEnd = std::min(end, chunk + chunkBytes);
            std::unique_lock lock(mutex);
            if (zeroPages.empty()) break;
            if (zeroPages.find(chunk) == zeroPages.end()) {
                cursor = chunkEnd;
                continue;
            }
            while (cursor < chunkEnd) {
                const auto limit = zeroSpanEnd(cursor);
                const auto stop = limit == 0 ? std::min(chunkEnd, (cursor & ~(pageBytes - 1)) + pageBytes) : std::min(chunkEnd, limit);
                if (limit == 0) forgetZero(cursor, stop);
                else harvestZero(cursor, stop);
                if (const auto bits = zeroPages.find(chunk); bits != zeroPages.end()) {
                    for (auto page = cursor; page < stop; page += 4096) {
                        const auto bit = (page - chunk) >> 12;
                        if ((bits->second[bit >> 6] >> (bit & 63) & 1) != 0) report(page, page + 4096);
                    }
                }
                cursor = stop;
            }
            lock.unlock();
            deliver();
        }
        deliver();
        if (reportEnd > reportStart) zero(context, reportStart, reportEnd);
    }

    bool Collect(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count, bool clear, Collection collection = Collection::All) {
        std::lock_guard lock(mutex);
        const auto capacity = *count;
        *count = 0;
        const auto end = address + bytes;
        if (clear) disown(address, end);
        if (collection == Collection::Fresh) {
            const auto fresh = freshRanges.upper_bound(address);
            if ((fresh == freshRanges.begin() || std::prev(fresh)->second <= address) && (fresh == freshRanges.end() || fresh->first >= end)) return true;
            for (auto it = views.lower_bound(address & ~(pageBytes - 1)); it != views.end() && it->first < end; ++it) {
                if (it->second.resident) {
                    if (!watchResident) return false;
                    continue;
                }
                if (it->second.page->comparedWhenShared) return false;
                if (it->second.protection != PAGE_NOACCESS) continue;
                for (auto at = std::max(address, it->first); at < std::min(end, it->first + pageBytes); at += 4096) {
                    if ((it->second.collected & (1u << ((at - it->first) / 4096))) == 0) return false;
                }
            }
        } else if (clear && !collectable(address, end)) return false;
        auto position = views.lower_bound(address & ~(pageBytes - 1));
        std::uintptr_t cleanStart = 0;
        std::uintptr_t cleanEnd = 0;
        const auto flushClean = [&] {
            if (cleanEnd > cleanStart) rememberClean(cleanStart, cleanEnd);
            cleanStart = cleanEnd = 0;
        };
        for (auto cursor = address; cursor < end;) {
            const auto nextClean = cleanRanges.upper_bound(cursor);
            if (nextClean != cleanRanges.begin()) {
                const auto clean = std::prev(nextClean);
                if (cursor < clean->second) {
                    cursor = std::min(end, clean->second);
                    continue;
                }
            }
            const auto base = cursor & ~(pageBytes - 1);
            const auto found = viewAt(base, position);
            if (found != views.end() && found->second.resident) {
                flushClean();
                if (!watchResident) return false;
                const auto available = capacity - *count;
                if (available == 0) return true;
                const auto run = std::prev(residentRuns.upper_bound(cursor));
                const auto stop = std::min({end, run->second.end, (cursor & ~std::uintptr_t{4095}) + available * 4096, nextClean == cleanRanges.end() ? end : nextClean->first});
                // A resetting collect of everything settles each page it walks (a page never
                // collected before is reported and marked collected), whatever the rest of the
                // run still holds: the walked range can be armed.
                const bool settled = collection == Collection::All;
                collectResident(run->first, run->second, cursor, stop, pages, count, clear, collection);
                if (walkProbe() && collection != Collection::Fresh) ++residentWalkWhy[!clear ? 0 : collection != Collection::All ? 1 : !settled ? 2 : 3];
                if (clear && settled && protectTracking()) armWalked(cursor, stop);
                if (*count == capacity) return true;
                cursor = stop;
            } else if (found != views.end()) {
                auto& view = found->second;
                auto stop = std::min(std::min(end, base + pageBytes), nextClean == cleanRanges.end() ? end : nextClean->first);
                if (collection != Collection::All) {
                    const auto selected = [&](std::uintptr_t at) {
                        const bool collected = (view.collected & (1u << ((at - base) / 4096))) != 0;
                        return collection == Collection::Armed ? collected : !collected;
                    };
                    while (cursor < stop && !selected(cursor)) cursor = std::min(stop, cursor + 4096);
                    if (cursor == stop) continue;
                    for (auto at = cursor + 4096; at < stop; at += 4096) {
                        if (!selected(at)) {
                            stop = at;
                            break;
                        }
                    }
                }
                if (view.protection == PAGE_NOACCESS) {
                    flushClean();
                    return false;
                }
                const bool pinned = view.page->pins != 0;
                std::uint8_t settled = 0;
                auto settledEnd = cursor;
                bool full = false;
                for (std::size_t sub = 0; sub < subpages; ++sub) {
                    const auto from = std::max(cursor, base + sub * subpageBytes);
                    const auto to = std::min(stop, base + (sub + 1) * subpageBytes);
                    if (from >= to) continue;
                    if (collection == Collection::Fresh || pinned || view.seen[sub] != view.page->generation[sub]) {
                        if (*count == capacity) {
                            full = true;
                            break;
                        }
                        pages[(*count)++] = reinterpret_cast<void*>(from);
                    }
                    settled |= static_cast<std::uint8_t>(1u << sub);
                    settledEnd = to;
                }
                if (clear) {
                    for (std::size_t sub = 0; sub < subpages; ++sub) {
                        const auto bit = static_cast<std::uint8_t>(1u << sub);
                        if ((settled & bit) == 0 || (view.collected & bit) != 0) continue;
                        view.collected |= bit;
                        forget(freshRanges, base + sub * subpageBytes, base + (sub + 1) * subpageBytes);
                    }
                }
                if (clear && !pinned && settled != 0) {
                    std::uint8_t untracked = 0;
                    for (const auto alias : view.page->aliases) {
                        auto& other = views.at(alias);
                        if (other.resident) continue;
                        const auto arming = static_cast<std::uint8_t>(settled & ~other.armed);
                        if (!writable(other.protection) || arming == 0) continue;
                        if (other.hostWrites != 0) {
                            untracked |= arming;
                            continue;
                        }
                        const DWORD protection = other.protection == PAGE_EXECUTE_READWRITE ? PAGE_EXECUTE_READ : PAGE_READONLY;
                        protectSubpages(alias, arming, protection, "arm shared memory write tracking", other.viewBytes);
                        other.armed |= arming;
                    }
                    const auto tracked = static_cast<std::uint8_t>(settled & ~untracked);
                    for (std::size_t sub = 0; sub < subpages; ++sub) {
                        if ((tracked & (1u << sub)) == 0) continue;
                        const auto from = std::max(cursor, base + sub * subpageBytes);
                        const auto to = std::min(settledEnd, base + (sub + 1) * subpageBytes);
                        if (from == base + sub * subpageBytes && to == base + (sub + 1) * subpageBytes) view.seen[sub] = view.page->generation[sub];
                        if (from != cleanEnd) {
                            flushClean();
                            cleanStart = from;
                        }
                        cleanEnd = to;
                    }
                }
                if (full) {
                    flushClean();
                    return true;
                }
                cursor = stop;
            } else {
                flushClean();
                if (collection == Collection::Fresh) {
                    const auto next = views.lower_bound(cursor);
                    cursor = next == views.end() ? end : std::min(end, next->first);
                    continue;
                }
                const auto allocationEnd = privateAllocationEnd(cursor);
                if (allocationEnd == 0) return false;
                if (capacity == *count) return true;
                const auto stop = ownedPending.empty() ? std::min(end, allocationEnd) : std::min({end, allocationEnd, (cursor & ~std::uintptr_t{4095}) + (capacity - *count) * 4096});
                const auto touched = (((stop + 4095) & ~static_cast<std::uintptr_t>(4095)) - (cursor & ~static_cast<std::uintptr_t>(4095))) / 4096;
                ULONG_PTR available = std::min<ULONG_PTR>(capacity - *count, touched);
                const auto requested = available;
                DWORD granularity = 0;
#ifdef APS5_COUNT_MAPPING_QUERIES
                ++collectSystemCalls;
                collectPieces += (stop + pageBytes - 1) / pageBytes - cursor / pageBytes;
#endif
                countWalk(cursor, stop, false);
                const bool committed = !verifyDirect() || committedPrivate(cursor, stop);
                if (collectOwnedWrites(clear ? WRITE_WATCH_FLAG_RESET : 0, reinterpret_cast<void*>(cursor), stop - cursor, pages + *count, &available, &granularity) != 0) {
                    const auto memory = query(cursor);
                    if (memory.State != MEM_COMMIT || memory.Type != MEM_PRIVATE) return false;
                    fail("collect private guest writes");
                }
                countHotResult(cursor, stop, available);
                if (!committed) staleProof(cursor, stop);
                *count += available;
                // Arm only after a walk that reported every written page (a full buffer may have
                // left some of the range unreset).
                if (clear && collection == Collection::All && (available < requested || requested == touched) && protectTracking()) armWalked(cursor, stop);
                if (*count == capacity) return true;
                cursor = stop;
            }
        }
        flushClean();
        return true;
    }

private:
    using OwnedRanges = std::multimap<std::uintptr_t, std::uintptr_t>;

    static bool ownedRange(std::uintptr_t address, std::size_t bytes, std::size_t count) {
        if (address == 0 || bytes == 0 || address >= (std::uintptr_t{1} << 47) || bytes > (std::uintptr_t{1} << 47) - address) return false;
        return count == ((address + bytes - 1) / ownedBlockBytes - address / ownedBlockBytes + 1);
    }

    static bool overlapsOwnedRange(const OwnedRanges& ranges, std::uintptr_t first, std::uintptr_t end) {
        for (auto it = ranges.begin(); it != ranges.end() && it->first < end; ++it) {
            if (it->second > first) return true;
        }
        return false;
    }

    static void eraseOwnedRange(OwnedRanges& ranges, std::uintptr_t first, std::uintptr_t end) {
        const auto [begin, stop] = ranges.equal_range(first);
        for (auto it = begin; it != stop; ++it) {
            if (it->second == end) {
                ranges.erase(it);
                return;
            }
        }
    }

    void disown(std::uintptr_t first, std::uintptr_t end) {
        if (end <= first) return;
        ownedWords.erase(ownedWords.lower_bound(first & ~(ownedBlockBytes - 1)), ownedWords.lower_bound(end));
    }

    bool ownable(std::uintptr_t block, unsigned flags) {
        const auto end = block + ownedBlockBytes;
        if (overlapsOwnedRange(ownedPins, block, end) || overlapsOwnedRange(ownedHostAccess, block, end)) return false;
        for (auto cursor = block; cursor < end;) {
            const auto memory = query(cursor);
            const auto protect = memory.Protect == PAGE_READONLY && armedAt(cursor) ? PAGE_READWRITE : memory.Protect;
            if (memory.State != MEM_COMMIT || memory.Type != MEM_PRIVATE || (protect != PAGE_READWRITE && protect != PAGE_EXECUTE_READWRITE)) return false;
            const auto found = views.find(cursor & ~(pageBytes - 1));
            if (found != views.end()) {
                const auto& view = found->second;
                if (!view.resident || view.pins != 0 || view.hostWrites != 0 || view.hostImports != 0 || !writable(view.protection)) return false;
                const auto next = residentRuns.upper_bound(cursor);
                if (next == residentRuns.begin() || cursor >= std::prev(next)->second.end || !std::prev(next)->second.watched) return false;
            } else if ((flags & OwnedPrivate) == 0 || privateAllocationEnd(cursor) == 0) return false;
            cursor = std::min({end, (cursor & ~(pageBytes - 1)) + pageBytes, reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize});
        }
        std::array<void*, ownedBlockBytes / 4096> written{};
        ULONG_PTR reported = written.size();
        DWORD granularity = 0;
        return watchWrites(0, reinterpret_cast<void*>(block), ownedBlockBytes, written.data(), &reported, &granularity) == 0;
    }

    void forgetOwnedPending(std::uintptr_t first, std::uintptr_t end) {
        for (auto it = ownedPending.lower_bound(first & ~(ownedBlockBytes - 1)); it != ownedPending.end() && it->first < end;) {
            for (unsigned bit = 0; bit < ownedBlockBytes / 4096; ++bit) {
                const auto page = it->first + bit * 4096;
                if (page < end && page + 4096 > first) it->second &= ~(std::uint16_t{1} << bit);
            }
            if (it->second == 0) it = ownedPending.erase(it);
            else ++it;
        }
    }

    void markZero(std::uintptr_t start, std::uintptr_t end) {
        start = (start + 4095) & ~std::uintptr_t{4095};
        end &= ~std::uintptr_t{4095};
        while (start < end) {
            const auto chunk = start & ~(chunkBytes - 1);
            const auto stop = std::min(end, chunk + chunkBytes);
            auto& bits = zeroPages[chunk];
            for (auto bit = (start - chunk) >> 12; bit < (stop - chunk) >> 12; ++bit) bits[bit >> 6] |= std::uint64_t{1} << (bit & 63);
            start = stop;
        }
    }

    static void clearZeroBits(auto& bits, std::uintptr_t chunk, std::uintptr_t from, std::uintptr_t to) {
        for (auto bit = (from - chunk) >> 12; bit < (to - chunk) >> 12; ++bit) bits[bit >> 6] &= ~(std::uint64_t{1} << (bit & 63));
    }

    static bool noZeroBits(const auto& bits) {
        return std::all_of(bits.begin(), bits.end(), [](std::uint64_t word) { return word == 0; });
    }

    void forgetZero(std::uintptr_t start, std::uintptr_t end) {
        if (zeroPages.empty() || end <= start) return;
        start &= ~std::uintptr_t{4095};
        end = (end + 4095) & ~std::uintptr_t{4095};
        if ((end - start) / chunkBytes > zeroPages.size()) {
            for (auto it = zeroPages.begin(); it != zeroPages.end();) {
                const auto chunk = it->first;
                if (chunk + chunkBytes <= start || chunk >= end) {
                    ++it;
                    continue;
                }
                clearZeroBits(it->second, chunk, std::max(start, chunk), std::min(end, chunk + chunkBytes));
                if (noZeroBits(it->second)) it = zeroPages.erase(it);
                else ++it;
            }
            return;
        }
        while (start < end) {
            const auto chunk = start & ~(chunkBytes - 1);
            const auto stop = std::min(end, chunk + chunkBytes);
            if (const auto found = zeroPages.find(chunk); found != zeroPages.end()) {
                clearZeroBits(found->second, chunk, start, stop);
                if (noZeroBits(found->second)) zeroPages.erase(found);
            }
            start = stop;
        }
    }

    bool anyZero(std::uintptr_t start, std::uintptr_t end) const {
        if (zeroPages.empty()) return false;
        start &= ~std::uintptr_t{4095};
        end = (end + 4095) & ~std::uintptr_t{4095};
        while (start < end) {
            const auto chunk = start & ~(chunkBytes - 1);
            const auto stop = std::min(end, chunk + chunkBytes);
            if (const auto found = zeroPages.find(chunk); found != zeroPages.end()) {
                for (auto bit = (start - chunk) >> 12; bit < (stop - chunk) >> 12; ++bit) {
                    if ((found->second[bit >> 6] >> (bit & 63) & 1) != 0) return true;
                }
            }
            start = stop;
        }
        return false;
    }

    bool stillZero(std::uintptr_t page) const {
        const auto found = zeroPages.find(page & ~(chunkBytes - 1));
        if (found == zeroPages.end()) return false;
        const auto bit = (page & (chunkBytes - 1)) >> 12;
        if ((found->second[bit >> 6] >> (bit & 63) & 1) == 0) return false;
        thread_local std::array<std::uint64_t, 512> bytes{};
        SIZE_T read = 0;
        if (!ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(page), bytes.data(), sizeof(bytes), &read) || read != sizeof(bytes)) return false;
        return std::all_of(bytes.begin(), bytes.end(), [](std::uint64_t word) { return word == 0; });
    }

    UINT watchWrites(DWORD flags, void* address, std::size_t bytes, void** pages, ULONG_PTR* count, DWORD* granularity, bool dropZero = false) {
        const auto capacity = *count;
        const auto result = GetWriteWatch(flags, address, bytes, pages, count, granularity);
        if (result != 0 || zeroPages.empty()) return result;
        const auto first = reinterpret_cast<std::uintptr_t>(address) & ~std::uintptr_t{4095};
        const auto last = (reinterpret_cast<std::uintptr_t>(address) + bytes + 4095) & ~std::uintptr_t{4095};
        const bool complete = dropZero && (*count < capacity || capacity >= (last - first) / 4096);
        ULONG_PTR kept = 0;
        for (ULONG_PTR index = 0; index < *count; ++index) {
            const auto page = reinterpret_cast<std::uintptr_t>(pages[index]) & ~std::uintptr_t{4095};
            if (complete && stillZero(page)) continue;
            forgetZero(page, page + 4096);
            pages[kept++] = pages[index];
        }
        *count = kept;
        return result;
    }

    void harvestZero(std::uintptr_t start, std::uintptr_t end) {
        start &= ~std::uintptr_t{4095};
        end = (end + 4095) & ~std::uintptr_t{4095};
        if (!anyZero(start, end)) return;
        std::vector<void*> written((end - start) / 4096);
        ULONG_PTR reported = written.size();
        DWORD granularity = 0;
        if (watchWrites(0, reinterpret_cast<void*>(start), end - start, written.data(), &reported, &granularity, true) != 0 || reported == written.size()) forgetZero(start, end);
    }

    bool resetWrites(std::uintptr_t start, std::uintptr_t end) {
        if (!anyZero(start, end)) return ResetWriteWatch(reinterpret_cast<void*>(start), end - start) == 0;
        std::vector<void*> written(((end + 4095) & ~std::uintptr_t{4095}) / 4096 - (start & ~std::uintptr_t{4095}) / 4096);
        ULONG_PTR reported = written.size();
        DWORD granularity = 0;
        if (watchWrites(WRITE_WATCH_FLAG_RESET, reinterpret_cast<void*>(start), end - start, written.data(), &reported, &granularity, true) != 0) {
            forgetZero(start, end);
            return false;
        }
        return true;
    }

    std::uintptr_t zeroSpanEnd(std::uintptr_t at) {
        const auto run = residentRuns.upper_bound(at);
        if (run != residentRuns.begin() && std::prev(run)->second.end > at) {
            const auto& resident = std::prev(run)->second;
            return resident.watched && !resident.imported ? resident.end : 0;
        }
        if (views.find(at & ~(pageBytes - 1)) != views.end()) return 0;
        return privateAllocationEnd(at);
    }

    UINT collectOwnedWrites(DWORD flags, void* address, std::size_t bytes, void** pages, ULONG_PTR* count, DWORD* granularity) {
        const auto capacity = *count;
        const auto result = watchWrites(flags, address, bytes, pages, count, granularity);
        if (result != 0 || ownedPending.empty()) return result;
        std::sort(pages, pages + *count);
        const auto reported = *count;
        const auto first = reinterpret_cast<std::uintptr_t>(address);
        const auto end = first + bytes;
        for (auto it = ownedPending.lower_bound(first & ~(ownedBlockBytes - 1)); it != ownedPending.end() && it->first < end; ++it) {
            for (unsigned bit = 0; bit < ownedBlockBytes / 4096; ++bit) {
                const auto page = it->first + bit * 4096;
                if ((it->second & (std::uint16_t{1} << bit)) == 0 || page >= end || page + 4096 <= first) continue;
                auto* pointer = reinterpret_cast<void*>(page);
                if (!std::binary_search(pages, pages + reported, pointer)) {
                    if (*count == capacity) throw std::runtime_error("owned write collection exceeds its page buffer");
                    pages[(*count)++] = pointer;
                }
            }
        }
        std::sort(pages, pages + *count);
        if ((flags & WRITE_WATCH_FLAG_RESET) != 0) forgetOwnedPending(first, end);
        return result;
    }

    bool collectable(std::uintptr_t address, std::uintptr_t end) {
        const auto proven = collectableRanges.upper_bound(address);
        if (proven != collectableRanges.begin() && std::prev(proven)->second >= end) return true;
        auto position = views.lower_bound(address & ~(pageBytes - 1));
        for (auto cursor = address; cursor < end;) {
            const auto base = cursor & ~(pageBytes - 1);
            if (const auto found = viewAt(base, position); found != views.end()) {
                if (found->second.protection == PAGE_NOACCESS || (found->second.resident ? !watchResident : found->second.page->comparedWhenShared)) return false;
                cursor = std::min(end, base + pageBytes);
                continue;
            }
            const auto allocationEnd = privateAllocationEnd(cursor);
            if (allocationEnd == 0) return false;
            cursor = std::min(end, allocationEnd);
        }
        remember(collectableRanges, address, end);
        return true;
    }

    static bool verifyDirect() {
        static const bool verify = std::getenv("APS5_VERIFY_DIRECT_COLLECT") != nullptr;
        return verify;
    }

    static bool committedPrivate(std::uintptr_t address, std::uintptr_t end) {
        for (auto cursor = address; cursor < end;) {
            const auto memory = query(cursor);
            if (memory.State != MEM_COMMIT || memory.Type != MEM_PRIVATE) return false;
            cursor = reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize;
        }
        return true;
    }

    [[noreturn]] static void staleProof(std::uintptr_t address, std::uintptr_t end) {
        char text[192];
        std::snprintf(text, sizeof(text), "a private collect of guest writes succeeded over 0x%llx+0x%llx, which is no longer committed private memory (APS5_VERIFY_DIRECT_COLLECT)", static_cast<unsigned long long>(address), static_cast<unsigned long long>(end - address));
        throw std::runtime_error(text);
    }

    // End of the guest private allocation or resident run holding address, or 0.
    std::uintptr_t guestAllocationEnd(std::uintptr_t address) const {
        if (const auto end = privateAllocationEnd(address); end != 0) return end;
        const auto next = residentRuns.upper_bound(address);
        if (next == residentRuns.begin()) return 0;
        const auto run = std::prev(next);
        return address < run->second.end ? run->second.end : 0;
    }

    std::uintptr_t privateAllocationEnd(std::uintptr_t address) const {
        const auto next = privateAllocations.upper_bound(address);
        if (next == privateAllocations.begin()) return 0;
        const auto allocation = std::prev(next);
        return address < allocation->second ? allocation->second : 0;
    }

    static constexpr std::size_t pageBytes = 0x4000;
    static constexpr std::size_t subpageBytes = 0x1000;
    static_assert(subpageBytes == 4096, "Collect reports one page per subpage");
    static constexpr std::size_t subpages = pageBytes / subpageBytes;
    static constexpr std::uint8_t allSubpages = (1u << subpages) - 1;
    static constexpr std::size_t chunkBytes = 0x200000;
    struct SharedPage {
        std::array<std::uint64_t, subpages> generation = [] {
            std::array<std::uint64_t, subpages> initial{};
            initial.fill(1);
            return initial;
        }();
        std::vector<std::uintptr_t> aliases;
        std::uint32_t pins = 0;
        bool comparedWhenShared = false;
    };
    struct Section {
        HANDLE handle;
        explicit Section(HANDLE handle) : handle(handle) {}
        Section(const Section&) = delete;
        Section& operator=(const Section&) = delete;
        ~Section() { CloseHandle(handle); }
    };
    struct Saved {
        unsigned char* bytes = nullptr;
        std::size_t length;
        std::array<std::uint64_t, chunkBytes / pageBytes / 64> committed{};
        explicit Saved(std::size_t length) : length(length) {
            bytes = static_cast<unsigned char*>(VirtualAlloc(nullptr, length, MEM_RESERVE, PAGE_READWRITE));
            if (bytes == nullptr) fail("reserve unmapped direct memory");
        }
        Saved(const Saved&) = delete;
        Saved& operator=(const Saved&) = delete;
        ~Saved() { VirtualFree(bytes, 0, MEM_RELEASE); }

        void Commit(std::size_t offset) {
            const auto page = offset / pageBytes;
            const auto bit = std::uint64_t{1} << (page & 63);
            if ((committed[page >> 6] & bit) != 0) return;
            if (!VirtualAlloc(bytes + offset, pageBytes, MEM_COMMIT, PAGE_READWRITE)) fail("keep unmapped direct memory");
            committed[page >> 6] |= bit;
        }
    };
    struct Physical {
        std::weak_ptr<SharedPage> page;
        bool stored = false;
        bool comparedWhenShared = false;
        std::shared_ptr<Saved> saved;
        std::size_t savedOffset = 0;
    };
    struct View {
        std::shared_ptr<SharedPage> page;
        DWORD protection;
        std::array<std::uint64_t, subpages> seen;
        std::uint8_t armed;
        bool comparedWhenShared = false;
        std::uint32_t pins = 0;
        std::shared_ptr<Section> section;
        std::uint64_t offset;
        std::uint32_t hostWrites;
        std::uint8_t collected = 0;
        bool resident = false;
        std::shared_ptr<Physical> slot;
        std::size_t viewBytes = pageBytes;
        std::uint32_t hostImports = 0;
        bool aliasDemoted = false;
        std::uint8_t unreported = 0;
    };
    struct Run {
        std::uintptr_t end;
        std::array<std::uint64_t, 8> collected;
        std::uint32_t fresh;
        std::uintptr_t key;
        std::uint64_t offset;
        bool watched = false;
        bool reset = false;
        bool imported = false;
    };
    struct Window {
        void* view = nullptr;
        unsigned char* bytes = nullptr;
    };
    using Ranges = std::map<std::uintptr_t, std::uintptr_t>;

    static bool runCollected(const Run& run, std::size_t bit) {
        return (run.collected[bit >> 6] >> (bit & 63) & 1) != 0;
    }

    void collectResident(std::uintptr_t start, Run& run, std::uintptr_t from, std::uintptr_t stop, void** pages, std::size_t* count, bool clear, Collection collection) {
        if (clear) run.reset = true;
        countWalk(from, stop, collection == Collection::Fresh);
        const auto first = from & ~std::uintptr_t{4095};
        if (run.fresh == 0 && collection != Collection::Fresh) {
            ULONG_PTR reported = (stop - first + 4095) / 4096;
            DWORD granularity = 0;
            if (collectOwnedWrites(clear ? WRITE_WATCH_FLAG_RESET : 0, reinterpret_cast<void*>(first), stop - first, pages + *count, &reported, &granularity) != 0) fail("collect resident guest writes");
            for (ULONG_PTR index = 0; index < reported; ++index) pages[*count + index] = reinterpret_cast<void*>(std::max(from, reinterpret_cast<std::uintptr_t>(pages[*count + index])));
            countHotResult(from, stop, reported);
            *count += reported;
            return;
        }
        std::vector<std::uintptr_t> written;
        if (collection != Collection::Fresh) {
            std::vector<void*> found((stop - first + 4095) / 4096);
            ULONG_PTR reported = found.size();
            DWORD granularity = 0;
            if (collectOwnedWrites(clear ? WRITE_WATCH_FLAG_RESET : 0, reinterpret_cast<void*>(first), stop - first, found.data(), &reported, &granularity) != 0) fail("collect resident guest writes");
            for (ULONG_PTR index = 0; index < reported; ++index) written.push_back(reinterpret_cast<std::uintptr_t>(found[index]));
        }
        if (collection != Collection::Fresh) countHotResult(from, stop, written.size());
        auto next = written.begin();
        std::uintptr_t resetFrom = 0;
        const auto flushReset = [&](std::uintptr_t to) {
            if (resetFrom != 0) forgetOwnedPending(resetFrom, to);
            if (resetFrom != 0 && !resetWrites(resetFrom, to)) fail("reset resident guest writes");
            resetFrom = 0;
        };
        for (auto at = first; at < stop; at += 4096) {
            const auto bit = (at - start) >> 12;
            const bool fresh = !runCollected(run, bit);
            bool wrote = false;
            while (next != written.end() && *next < at) ++next;
            if (next != written.end() && *next == at) wrote = true;
            const bool report = collection == Collection::Fresh ? fresh : collection == Collection::Armed ? !fresh && wrote : fresh || wrote;
            if (report) pages[(*count)++] = reinterpret_cast<void*>(std::max(at, from));
            const bool settle = clear && fresh && collection != Collection::Armed;
            if (settle) {
                run.collected[bit >> 6] |= 1ull << (bit & 63);
                --run.fresh;
                forget(freshRanges, at, at + 4096);
                if (collection == Collection::Fresh && resetFrom == 0) resetFrom = at;
            } else if (collection == Collection::Fresh) {
                flushReset(at);
            }
        }
        if (collection == Collection::Fresh) flushReset(stop);
    }

    Window open(HANDLE section, std::uint64_t offset, std::size_t bytes) {
        const auto lead = offset % granularity;
        Window window;
        window.view = map(section, GetCurrentProcess(), nullptr, offset - lead, lead + bytes, 0, PAGE_READWRITE, nullptr, 0);
        if (window.view == nullptr) fail("open a window on direct memory");
        window.bytes = static_cast<unsigned char*>(window.view) + lead;
        return window;
    }

    void close(Window& window) {
        if (window.view != nullptr && !unmap(GetCurrentProcess(), window.view, 0)) fail("close a window on direct memory");
        window = {};
    }

    void commitResident(std::uintptr_t base, std::size_t bytes) {
        const DWORD flags = MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER | (watchResident ? MEM_WRITE_WATCH : 0);
        if (!allocate(GetCurrentProcess(), reinterpret_cast<void*>(base), bytes, flags, PAGE_READWRITE, nullptr, 0)) fail("replace guest placeholder with resident direct memory");
        if (watchResident) markZero(base, base + bytes);
    }

    static std::size_t hostPages(std::size_t bytes) {
        return (bytes + subpageBytes - 1) & ~(subpageBytes - 1);
    }

    void commitSection(HANDLE section, std::uint64_t offset, std::size_t bytes) {
        auto window = open(section, offset, bytes);
        const auto memory = query(reinterpret_cast<std::uintptr_t>(window.bytes));
        if ((memory.State != MEM_COMMIT || memory.RegionSize < bytes) && !VirtualAlloc(window.bytes, bytes, MEM_COMMIT, PAGE_READWRITE)) {
            const auto error = GetLastError();
            close(window);
            throw std::system_error(static_cast<int>(error), std::system_category(), "commit direct memory pages");
        }
        close(window);
    }

    void mapShared(std::uintptr_t base, HANDLE section, std::uint64_t offset, DWORD protection, DWORD viewProtection = PAGE_EXECUTE_READWRITE, std::size_t sectionBytes = pageBytes, bool committed = false) {
        void* page = reinterpret_cast<void*>(base);
        const auto viewBytes = hostPages(sectionBytes);
        if (viewBytes != pageBytes) split(base, viewBytes);
        if (!map(section, GetCurrentProcess(), page, offset, sectionBytes, MEM_REPLACE_PLACEHOLDER, viewProtection, nullptr, 0)) fail("map shared guest page");
        if (!committed && query(base).State == MEM_RESERVE && !VirtualAlloc(page, viewBytes, MEM_COMMIT, PAGE_EXECUTE_READWRITE)) fail("commit shared guest page");
        DWORD previous;
        if (!VirtualProtect(page, viewBytes, protection, &previous)) fail("protect shared guest page");
        if (viewBytes == pageBytes) return;
        void* zeros = reinterpret_cast<void*>(base + viewBytes);
        if (!allocate(GetCurrentProcess(), zeros, pageBytes - viewBytes, MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, nullptr, 0)) fail("replace guest placeholder past the end of a file with zero pages");
        if (!VirtualProtect(zeros, pageBytes - viewBytes, protection, &previous)) fail("protect guest pages past the end of a file");
    }

    void mapRun(std::uintptr_t base, std::size_t bytes, HANDLE section, std::uint64_t offset) {
        if (!map(section, GetCurrentProcess(), reinterpret_cast<void*>(base), offset, bytes, MEM_REPLACE_PLACEHOLDER, PAGE_EXECUTE_READWRITE, nullptr, 0)) fail("map a run of shared guest pages");
    }

    void breakView(std::uintptr_t at) {
        const auto memory = query(at);
        if (memory.State == MEM_FREE || memory.Type != MEM_MAPPED) return;
        const auto start = reinterpret_cast<std::uintptr_t>(memory.AllocationBase);
        if (start == at) return;
        const auto first = views.find(start);
        if (first == views.end()) return;
        struct Part {
            std::uintptr_t base;
            std::uintptr_t end;
            DWORD protection;
            bool committed;
        };
        std::vector<Part> parts;
        auto stop = start;
        for (;;) {
            const auto part = query(stop);
            if (part.AllocationBase != memory.AllocationBase) break;
            const auto partEnd = reinterpret_cast<std::uintptr_t>(part.BaseAddress) + part.RegionSize;
            parts.push_back({stop, partEnd, part.Protect, part.State == MEM_COMMIT});
            stop = partEnd;
        }
        const auto section = first->second.section;
        const auto offset = first->second.offset;
        std::vector<unsigned char> saved;
        for (auto it = views.lower_bound(start); it != views.end() && it->first < stop; ++it) {
            for (std::size_t sub = 0; sub < subpages; ++sub) {
                const auto* bytes = reinterpret_cast<const unsigned char*>(it->first + sub * subpageBytes);
                if ((it->second.armed & (1u << sub)) != 0) saved.insert(saved.end(), bytes, bytes + subpageBytes);
            }
        }
        if (!unmap(GetCurrentProcess(), reinterpret_cast<void*>(start), MEM_PRESERVE_PLACEHOLDER)) fail("unmap a run of shared guest pages to split it");
        split(start, at - start);
        mapRun(start, at - start, section->handle, offset);
        mapRun(at, stop - at, section->handle, offset + (at - start));
        for (const auto& part : parts) {
            if (!part.committed) continue;
            for (const auto [from, to] : {std::pair{part.base, std::min(part.end, at)}, std::pair{std::max(part.base, at), part.end}}) {
                DWORD previous;
                if (from < to && !VirtualProtect(reinterpret_cast<void*>(from), to - from, part.protection, &previous)) fail("protect a split run of shared guest pages");
            }
        }
        const auto* before = saved.data();
        for (auto it = views.lower_bound(start); it != views.end() && it->first < stop; ++it) {
            std::uint8_t written = 0;
            for (std::size_t sub = 0; sub < subpages; ++sub) {
                if ((it->second.armed & (1u << sub)) == 0) continue;
                if (std::memcmp(reinterpret_cast<const void*>(it->first + sub * subpageBytes), before, subpageBytes) != 0) written |= static_cast<std::uint8_t>(1u << sub);
                before += subpageBytes;
            }
            if (written != 0) invalidate(*it->second.page, written);
        }
    }

    static bool blank(std::uintptr_t page) {
        const auto* words = reinterpret_cast<const std::uint64_t*>(page);
        for (std::size_t i = 0; i < pageBytes / sizeof(std::uint64_t); ++i) {
            if (words[i] != 0) return false;
        }
        return true;
    }

    std::map<std::uintptr_t, Run>::iterator residentRun(std::uintptr_t key, std::uint64_t offset) {
        const auto next = residentIndex.upper_bound(std::make_pair(key, offset));
        if (next == residentIndex.begin()) return residentRuns.end();
        const auto& [first, run] = *std::prev(next);
        return first.first == key && offset - first.second < run->second.end - run->first ? run : residentRuns.end();
    }

    std::map<std::pair<std::uintptr_t, std::uint64_t>, std::map<std::uintptr_t, Run>::iterator>::iterator indexEntry(std::map<std::uintptr_t, Run>::iterator run) {
        const auto found = residentIndex.find(std::make_pair(run->second.key, run->second.offset));
        return found != residentIndex.end() && found->second == run ? found : residentIndex.end();
    }

    void demote(std::uintptr_t address, bool alias = false) {
        const auto run = std::prev(residentRuns.upper_bound(address));
        const auto start = run->first;
        const auto stop = run->second.end;
        disown(start, stop);
        forgetOwnedPending(start, stop);
        forgetZero(start, stop);
        const auto entry = indexEntry(run);
        const bool indexed = entry != residentIndex.end();
        std::vector<std::pair<std::shared_ptr<SharedPage>, std::shared_ptr<Physical>>> created;
        created.reserve((stop - start) / pageBytes);
        std::map<std::pair<std::uintptr_t, std::uint64_t>, std::shared_ptr<Physical>> staged;
        for (auto page = start; page < stop; page += pageBytes) {
            const auto& view = views.at(page);
            auto shared = std::make_shared<SharedPage>();
            shared->aliases.push_back(page);
            shared->pins = view.pins;
            shared->comparedWhenShared = view.comparedWhenShared;
            auto slot = std::make_shared<Physical>();
            slot->page = shared;
            slot->comparedWhenShared = view.comparedWhenShared;
            if (indexed) staged.emplace(std::make_pair(run->second.key, view.offset), slot);
            created.emplace_back(std::move(shared), std::move(slot));
        }
        DWORD previous;
        if (!VirtualProtect(reinterpret_cast<void*>(start), stop - start, PAGE_READONLY, &previous)) fail("fence resident guest pages");
        const auto section = views.at(start).section;
        auto window = open(section->handle, views.at(start).offset, stop - start);
        if (query(reinterpret_cast<std::uintptr_t>(window.bytes)).State == MEM_RESERVE && !VirtualAlloc(window.bytes, stop - start, MEM_COMMIT, PAGE_READWRITE)) fail("commit direct memory pages");
        std::memcpy(window.bytes, reinterpret_cast<const void*>(start), stop - start);
        close(window);
        if (!VirtualFree(reinterpret_cast<void*>(start), stop - start, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) fail("release resident guest pages");
        physical.merge(staged);
        for (auto& [key, slot] : staged) physical.find(key)->second = std::move(slot);
        for (auto page = start; page < stop; page += pageBytes) {
            auto& view = views.at(page);
            split(page, pageBytes);
            mapShared(page, view.section->handle, view.offset, view.protection);
            auto& [shared, slot] = created[(page - start) / pageBytes];
            view.page = std::move(shared);
            view.slot = std::move(slot);
            view.slot->stored = true;
            view.resident = false;
            view.aliasDemoted = alias;
            view.unreported = view.page->comparedWhenShared ? allSubpages : std::uint8_t{0};
            view.seen.fill(0);
            view.armed = 0;
            view.collected = 0;
            for (std::size_t sub = 0; sub < subpages; ++sub) {
                if (runCollected(run->second, ((page - start) >> 12) + sub)) view.collected |= static_cast<std::uint8_t>(1u << sub);
            }
            invalidate(*view.page);
        }
        if (indexed) residentIndex.erase(entry);
        residentRuns.erase(run);
    }

    void breakRun(std::uintptr_t at) {
        const auto next = residentRuns.upper_bound(at);
        if (next == residentRuns.begin()) return;
        const auto run = std::prev(next);
        if (at > run->first && at < run->second.end) demote(at);
    }

    void evict(std::map<std::uintptr_t, Run>::iterator run) {
        const auto start = run->first;
        const auto stop = run->second.end;
        disown(start, stop);
        forgetOwnedPending(start, stop);
        forgetZero(start, stop);
        DWORD previous;
        if (!VirtualProtect(reinterpret_cast<void*>(start), stop - start, PAGE_READONLY, &previous)) fail("open resident guest pages");
        const auto entry = indexEntry(run);
        const bool indexed = entry != residentIndex.end();
        std::vector<void*> dirty;
        std::vector<bool> touched;
        bool filtered = false;
        if (indexed && run->second.watched && !run->second.reset && !run->second.imported) {
            dirty.resize((stop - start) / subpageBytes);
            ULONG_PTR reported = dirty.size();
            DWORD granularity = 0;
            if (watchWrites(0, reinterpret_cast<void*>(start), stop - start, dirty.data(), &reported, &granularity) == 0) {
                touched = TouchedPages(start, (stop - start) / pageBytes, dirty.data(), reported);
                filtered = true;
            }
        }
        std::shared_ptr<Saved> saved;
        for (auto page = start; page < stop; page += pageBytes) {
            const auto found = views.find(page);
            const auto& view = found->second;
            const bool written = indexed && (!filtered || touched[(page - start) / pageBytes]) && !blank(page);
            if (written || (indexed && view.comparedWhenShared)) {
                auto& slot = physical[std::make_pair(run->second.key, view.offset)];
                if (!slot) slot = std::make_shared<Physical>();
                slot->comparedWhenShared = view.comparedWhenShared;
                if (written) {
                    if (!saved) saved = savedMemory(stop - start);
                    saved->Commit(page - start);
                    std::memcpy(saved->bytes + (page - start), reinterpret_cast<const void*>(page), pageBytes);
                    slot->saved = saved;
                    slot->savedOffset = page - start;
                }
            }
            views.erase(found);
        }
        if (indexed) residentIndex.erase(entry);
        forget(freshRanges, start, stop);
        residentRuns.erase(run);
    }

    std::shared_ptr<Saved> savedMemory(std::size_t bytes) {
        for (const auto& saved : savedPool) {
            if (saved.use_count() == 1 && saved->length == bytes) return saved;
        }
        constexpr std::size_t limit = 32u * 1024u * 1024u;
        const auto reserved = (bytes + granularity - 1) / granularity * granularity;
        for (auto it = savedPool.begin(); reserved <= limit && savedPoolBytes > limit - reserved && it != savedPool.end();) {
            if (it->use_count() != 1) {
                ++it;
                continue;
            }
            savedPoolBytes -= ((*it)->length + granularity - 1) / granularity * granularity;
            it = savedPool.erase(it);
        }
        auto saved = std::make_shared<Saved>(bytes);
        if (reserved <= limit && savedPoolBytes <= limit - reserved) {
            savedPool.push_back(saved);
            savedPoolBytes += reserved;
        }
        return saved;
    }

    static void forget(Ranges& ranges, std::uintptr_t start, std::uintptr_t end) {
        auto it = ranges.lower_bound(start);
        if (it != ranges.begin() && std::prev(it)->second > start) --it;
        while (it != ranges.end() && it->first < end) {
            const auto first = it->first;
            const auto last = it->second;
            it = ranges.erase(it);
            if (first < start) ranges.emplace(first, start);
            if (last > end) it = ranges.emplace(end, last).first;
        }
    }

    static void remember(Ranges& ranges, std::uintptr_t start, std::uintptr_t end) {
        auto it = ranges.lower_bound(start);
        if (it != ranges.begin() && std::prev(it)->second >= start) --it;
        while (it != ranges.end() && it->first <= end) {
            start = std::min(start, it->first);
            end = std::max(end, it->second);
            it = ranges.erase(it);
        }
        ranges.emplace(start, end);
    }

    // Every armed range is also clean, so whatever ends a range's clean state (a fault, a host
    // write, a pin, a protection change, a release) opens its armed part as well.
    void forgetClean(std::uintptr_t start, std::uintptr_t end) {
        forget(cleanRanges, start, end);
        openArmed(start, end);
    }

    static std::atomic<bool>& writeFaultsHandled() {
        static std::atomic<bool> handled{false};
        return handled;
    }

    static bool protectTracking() {
        static const bool off = std::getenv("APS5_NO_PROTECT_TRACKING") != nullptr;
        return !off && writeFaultsHandled().load(std::memory_order_acquire);
    }

    static bool overlaps(const Ranges& ranges, std::uintptr_t start, std::uintptr_t end) {
        if (ranges.empty() || end <= start) return false;
        auto it = ranges.lower_bound(start);
        if (it != ranges.begin() && std::prev(it)->second > start) return true;
        return it != ranges.end() && it->first < end;
    }

    static bool covered(const Ranges& ranges, std::uintptr_t start, std::uintptr_t end) {
        const auto next = ranges.upper_bound(start);
        return next != ranges.begin() && std::prev(next)->second >= end;
    }

    bool armedAt(std::uintptr_t address) const {
        if (armedRanges.empty()) return false;
        const auto next = armedRanges.upper_bound(address);
        return next != armedRanges.begin() && std::prev(next)->second > address;
    }

    // Whether [start, end) may be armed: not pinned, not inside an open host write or import, never
    // mprotected by the guest, no owned-proof pages pending, and only resident views without pins,
    // host writes or imports.
    static bool walkProbe() {
        static const bool enabled = std::getenv("APS5_WALK_PROBE") != nullptr;
        return enabled;
    }

    static bool groupProbe() {
        static const bool enabled = walkProbe() && std::getenv("APS5_WALK_PROBE_GROUPS") != nullptr;
        return enabled;
    }

    static bool groupWalkProbe() {
        static const bool enabled = walkProbe() && std::getenv("APS5_WALK_PROBE_GROUP_WALKS") != nullptr;
        return enabled;
    }

    void countGroupWalk(std::uintptr_t from, std::uintptr_t to) {
        auto group = probeGroups.lower_bound(from);
        if (group != probeGroups.begin() && std::prev(group)->second > from) --group;
        for (; group != probeGroups.end() && group->first < to; ++group) {
            const auto start = std::max(from, group->first);
            const auto end = std::min(to, group->second);
            if (start >= end || !covered(openCauses[0], start, end)) continue;
            auto& [walks, bytes] = groupWalks[*group];
            ++walks;
            bytes += end - start;
        }
    }

    void printGroupWalks() {
        using Row = std::pair<std::pair<std::uint64_t, std::uint64_t>, std::pair<std::uintptr_t, std::uintptr_t>>;
        std::vector<Row> rows;
        std::uint64_t walks = 0, bytes = 0;
        for (const auto& [group, counts] : groupWalks) {
            rows.emplace_back(counts, group);
            walks += counts.first;
            bytes += counts.second;
        }
        std::fprintf(stderr, "[group-walk] total touches %llu bytes %llu groups %zu\n",
            static_cast<unsigned long long>(walks), static_cast<unsigned long long>(bytes), rows.size());
        const auto report = [&](const char* order, auto compare) {
            const auto n = std::min<std::size_t>(rows.size(), 12);
            std::partial_sort(rows.begin(), rows.begin() + n, rows.end(), compare);
            for (std::size_t i = 0; i < n; ++i) {
                const auto [counts, group] = rows[i];
                std::fprintf(stderr, "[group-walk] %s %llx-%llx touches %llu bytes %llu\n", order,
                    static_cast<unsigned long long>(group.first), static_cast<unsigned long long>(group.second),
                    static_cast<unsigned long long>(counts.first), static_cast<unsigned long long>(counts.second));
            }
        };
        report("by-touches", std::greater<>{});
        report("by-bytes", [](const Row& left, const Row& right) { return left.first.second > right.first.second; });
    }

    struct GpuWriteGroup {
        std::uint64_t submits = 0;
        std::uint64_t whileHot = 0;
        std::uint64_t bytes = 0;
        std::uint64_t overflow = 0;
        std::uint64_t kinds[9]{};
        // Written ranges rounded to arm blocks, with how many submissions' notes named each.
        std::map<std::pair<std::uintptr_t, std::uintptr_t>, std::uint64_t> ranges;
    };

    void countGpuWrites() {
        ++gpuWriteSubmits;
        anyPendingGpuWrites = false;
        std::set<std::pair<std::uintptr_t, std::uintptr_t>> touched;
        for (unsigned kind = 0; kind < GpuWriteKinds; ++kind) countGpuWriteKind(kind, touched);
    }

    void countGpuWriteKind(unsigned kind, std::set<std::pair<std::uintptr_t, std::uintptr_t>>& touched) {
        std::set<std::pair<std::uintptr_t, std::uintptr_t>> touchedKind;
        const bool stores = kind != 2 && kind != 4 && kind != 7;
        for (const auto& [start, end] : pendingGpuWrites[kind]) {
            auto from = start, to = end;
            bool any = false;
            for (bool grew = true; grew;) {
                grew = false;
                for (auto it = importExtents.lower_bound(from > longestImport ? from - longestImport : 0); it != importExtents.end() && it->first < to; ++it) {
                    if (it->second <= from) continue;
                    any = true;
                    if (it->first < from) from = it->first, grew = true;
                    if (it->second > to) to = it->second, grew = true;
                }
            }
            if (!any) {
                ++gpuWritesOutside;
                gpuWriteOutsideBytes += end - start;
                continue;
            }
            auto& group = gpuWriteGroups[{from, to}];
            if (touchedKind.emplace(from, to).second) ++group.kinds[kind];
            if (!stores) continue;
            if (touched.emplace(from, to).second) {
                ++group.submits;
                if (overlaps(openCauses[0], from, to)) ++group.whileHot;
            }
            group.bytes += end - start;
            const std::pair<std::uintptr_t, std::uintptr_t> key{start & ~(armBlockBytes - 1), (end + armBlockBytes - 1) & ~(armBlockBytes - 1)};
            if (group.ranges.size() < 4096 || group.ranges.contains(key)) ++group.ranges[key];
            else ++group.overflow;
        }
        pendingGpuWrites[kind].clear();
    }

    void printGpuWrites() {
        std::fprintf(stderr, "[gpu-write] submissions %llu: stores unknown %llu, stores named %llu, store groups made uniform %llu, stores through ordinary imports %llu (%llu opened armed pages, %llu MiB opened with unknown stores), store imports %zu of %zu imports\n",
            static_cast<unsigned long long>(submits), static_cast<unsigned long long>(submitsAll), static_cast<unsigned long long>(submitsStoring), static_cast<unsigned long long>(storeUniforms),
            static_cast<unsigned long long>(regularStores), static_cast<unsigned long long>(regularOpens), static_cast<unsigned long long>(regularOpened >> 20u), storeExtents.size(), importExtents.size());
        std::fprintf(stderr, "[gpu-write] notes %llu; %llu of %llu submissions carried notes; %llu ranges (%llu bytes) outside imports; groups %zu\n",
            static_cast<unsigned long long>(gpuWriteNotes), static_cast<unsigned long long>(gpuWriteSubmits), static_cast<unsigned long long>(submits),
            static_cast<unsigned long long>(gpuWritesOutside), static_cast<unsigned long long>(gpuWriteOutsideBytes), gpuWriteGroups.size());
        using Group = std::pair<std::uintptr_t, std::uintptr_t>;
        std::vector<std::pair<std::uint64_t, Group>> top;
        for (const auto& [group, counts] : gpuWriteGroups) top.emplace_back(counts.submits + counts.kinds[4], group);
        const auto n = std::min<std::size_t>(top.size(), 10);
        std::partial_sort(top.begin(), top.begin() + n, top.end(), std::greater<>{});
        for (std::size_t i = 0; i < n; ++i) {
            const auto [start, end] = top[i].second;
            const auto& counts = gpuWriteGroups.at(top[i].second);
            const auto base = start & ~(armBlockBytes - 1);
            const auto capacity = static_cast<std::size_t>((end - 1) / armBlockBytes - start / armBlockBytes + 1);
            std::vector<bool> written(capacity);
            for (const auto& [range, count] : counts.ranges) {
                for (auto block = std::max(range.first, base); block < range.second && (block - base) / armBlockBytes < capacity; block += armBlockBytes) written[(block - base) / armBlockBytes] = true;
            }
            std::size_t writtenBlocks = 0;
            for (const bool bit : written) writtenBlocks += bit;
            std::size_t faulted = 0, faultedWritten = 0;
            std::uint64_t faults = 0, faultsWritten = 0;
            for (const auto& [faultGroup, blocks] : groupFaultBlocks) {
                if (faultGroup.first >= end || faultGroup.second <= start) continue;
                for (const auto& [block, count] : blocks) {
                    if (block < base || (block - base) / armBlockBytes >= capacity) continue;
                    ++faulted;
                    faults += count;
                    if (written[(block - base) / armBlockBytes]) ++faultedWritten, faultsWritten += count;
                }
            }
            std::fprintf(stderr, "[gpu-write] group %llx-%llx submissions %llu (%llu while hot-open) bytes %llu ranges %zu (+%llu uncounted) blocks written %zu/%zu; faulted blocks %zu, %zu of them written; faults %llu, %llu in written blocks; submissions by kind %llu/%llu/%llu/%llu/%llu/%llu/%llu/%llu/%llu\n",
                static_cast<unsigned long long>(start), static_cast<unsigned long long>(end), static_cast<unsigned long long>(counts.submits),
                static_cast<unsigned long long>(counts.whileHot), static_cast<unsigned long long>(counts.bytes), counts.ranges.size(),
                static_cast<unsigned long long>(counts.overflow), writtenBlocks, capacity, faulted, faultedWritten,
                static_cast<unsigned long long>(faults), static_cast<unsigned long long>(faultsWritten),
                static_cast<unsigned long long>(counts.kinds[0]), static_cast<unsigned long long>(counts.kinds[1]), static_cast<unsigned long long>(counts.kinds[2]),
                static_cast<unsigned long long>(counts.kinds[3]), static_cast<unsigned long long>(counts.kinds[4]), static_cast<unsigned long long>(counts.kinds[5]),
                static_cast<unsigned long long>(counts.kinds[6]), static_cast<unsigned long long>(counts.kinds[7]), static_cast<unsigned long long>(counts.kinds[8]));
            std::vector<std::pair<std::uint64_t, Group>> ranges;
            for (const auto& [range, count] : counts.ranges) ranges.emplace_back(count, range);
            const auto m = std::min<std::size_t>(ranges.size(), 6);
            std::partial_sort(ranges.begin(), ranges.begin() + m, ranges.end(), std::greater<>{});
            for (std::size_t j = 0; j < m; ++j)
                std::fprintf(stderr, "[gpu-write]   range block %llu + %llu blocks: %llu\n",
                    static_cast<unsigned long long>((ranges[j].second.first - base) / armBlockBytes),
                    static_cast<unsigned long long>((ranges[j].second.second - ranges[j].second.first) / armBlockBytes),
                    static_cast<unsigned long long>(ranges[j].first));
        }
    }

    // Submission counts (BeginSubmit): stores unknown, stores named, store groups made uniform,
    // named stores through ordinary imports, those that opened armed pages, and the bytes opened
    // for them and for unknown stores.
    std::uint64_t submitsAll = 0;
    std::uint64_t submitsStoring = 0;
    std::uint64_t storeUniforms = 0;
    std::uint64_t regularStores = 0;
    std::uint64_t regularOpens = 0;
    std::uint64_t regularOpened = 0;
    static constexpr unsigned GpuWriteKinds = 9;
    Ranges pendingGpuWrites[GpuWriteKinds];
    bool anyPendingGpuWrites = false;
    std::map<std::pair<std::uintptr_t, std::uintptr_t>, GpuWriteGroup> gpuWriteGroups;
    std::uint64_t gpuWriteNotes = 0;
    std::uint64_t gpuWriteSubmits = 0;
    std::uint64_t gpuWritesOutside = 0;
    std::uint64_t gpuWriteOutsideBytes = 0;

    void countHotResult(std::uintptr_t from, std::uintptr_t to, std::size_t pages) {
        if (!groupWalkProbe() || !covered(openCauses[0], from, to)) return;
        ++hotResults;
        if (pages == 0) ++hotEmptyResults;
        hotResultPages += pages;
        if ((hotResults & 0xffff) == 0)
            std::fprintf(stderr, "[hot-result] walks %llu empty %llu pages %llu\n",
                static_cast<unsigned long long>(hotResults), static_cast<unsigned long long>(hotEmptyResults),
                static_cast<unsigned long long>(hotResultPages));
    }

    std::uint64_t hotResults = 0;
    std::uint64_t hotEmptyResults = 0;
    std::uint64_t hotResultPages = 0;

    void printGroupFaults() {
        using Group = std::pair<std::uintptr_t, std::uintptr_t>;
        std::vector<std::pair<std::uint64_t, Group>> top;
        std::uint64_t total = 0;
        for (const auto& [group, blocks] : groupFaultBlocks) {
            std::uint64_t faults = 0;
            for (const auto& [block, count] : blocks) faults += count;
            top.emplace_back(faults, group);
            total += faults;
        }
        const auto n = std::min<std::size_t>(top.size(), 8);
        std::partial_sort(top.begin(), top.begin() + n, top.end(), std::greater<>{});
        std::fprintf(stderr, "[group-probe] mapped %llu unmapped %llu groups %zu\n",
            static_cast<unsigned long long>(total), static_cast<unsigned long long>(unmappedGroupFaults), top.size());
        for (std::size_t i = 0; i < n; ++i) {
            const auto [faults, group] = top[i];
            const auto& blocks = groupFaultBlocks.at(group);
            std::vector<std::uint64_t> counts;
            for (const auto& [block, count] : blocks) counts.push_back(count);
            std::sort(counts.begin(), counts.end(), std::greater<>{});
            std::uint64_t topEight = 0;
            for (std::size_t j = 0; j < std::min<std::size_t>(8, counts.size()); ++j) topEight += counts[j];
            const auto [start, end] = group;
            const auto capacity = (end - 1) / armBlockBytes - start / armBlockBytes + 1;
            std::fprintf(stderr, "[group-probe] group %llx-%llx faults %llu blocks %zu/%zu top8 %llu peak %llu\n",
                static_cast<unsigned long long>(start), static_cast<unsigned long long>(end), static_cast<unsigned long long>(faults),
                blocks.size(), static_cast<std::size_t>(capacity), static_cast<unsigned long long>(topEight), static_cast<unsigned long long>(counts.front()));
        }
    }

    void countWalk(std::uintptr_t from, std::uintptr_t to, bool fresh) {
        ++collectWalks;
        if (!walkProbe()) return;
        unsigned reason = 0;
        if (fresh) reason = 8;
        else if (overlaps(openImports, from, to)) {
            unsigned matches = 0;
            for (unsigned i = 0; i < openCauses.size(); ++i) {
                if (overlaps(openCauses[i], from, to)) {
                    ++matches;
                    reason = i + 1;
                }
            }
            if (matches != 1) reason = 4;
        } else if (overlaps(unarmable, from, to) || overlapsOwnedRange(ownedPins, from, to) || overlapsOwnedRange(ownedHostAccess, from, to)) reason = 3;
        else if (covered(armedRanges, from & ~std::uintptr_t{4095}, (to + 4095) & ~std::uintptr_t{4095})) reason = 5;
        else if (overlaps(everArmed, from, to)) reason = 6;
        else reason = 7;
        ++walkCauses[reason];
        if (reason == 7) {
            // Why the walked range was never armed: it holds no whole page, a shared view covers
            // it, its memory is not private read-write, its block is hot, or none of these.
            const auto first = (from + 4095) & ~std::uintptr_t{4095}, last = to & ~std::uintptr_t{4095};
            unsigned why = 4;
            if (first >= last) why = 0;
            else if (const auto view = views.lower_bound(first & ~(pageBytes - 1)); view != views.end() && view->first < last) {
                // A view record covers it: not resident, pinned, inside a host write or import
                // bracket, with owned-proof pages pending, hot, not private read-write, or none.
                const auto& found = view->second;
                const auto memory = query(first);
                const auto strikes = armStrikes.find(first & ~(armBlockBytes - 1));
                const auto pending = ownedPending.lower_bound(first & ~(ownedBlockBytes - 1));
                why = !found.resident ? 1 : found.pins != 0 ? 5 : found.hostWrites != 0 ? 6 : found.hostImports != 0 ? 7 : pending != ownedPending.end() && pending->first < last ? 8
                    : strikes != armStrikes.end() && strikes->second >= armHotLimit ? 9 : memory.State != MEM_COMMIT || memory.Type != MEM_PRIVATE || memory.Protect != PAGE_READWRITE ? 10 : 11;
            } else if (const auto memory = query(first); memory.State != MEM_COMMIT || memory.Type != MEM_PRIVATE || memory.Protect != PAGE_READWRITE) why = 2;
            else if (const auto strikes = armStrikes.find(first & ~(armBlockBytes - 1)); strikes != armStrikes.end() && strikes->second >= armHotLimit) why = 3;
            ++neverArmedWhy[why];
            neverArmedBytes[why] += to - from;
        }
        if (groupWalkProbe() && !fresh && overlaps(openCauses[0], from, to)) countGroupWalk(from, to);
        if ((collectWalks & 0xffff) != 0) return;
        std::fprintf(stderr, "[walk-probe] walks %llu unknown %llu hot %llu unused %llu refused %llu mixed-open %llu armed %llu previously-armed %llu never-armed %llu fresh %llu\n",
            static_cast<unsigned long long>(collectWalks), static_cast<unsigned long long>(walkCauses[0]), static_cast<unsigned long long>(walkCauses[1]),
            static_cast<unsigned long long>(walkCauses[2]), static_cast<unsigned long long>(walkCauses[3]), static_cast<unsigned long long>(walkCauses[4]),
            static_cast<unsigned long long>(walkCauses[5]), static_cast<unsigned long long>(walkCauses[6]), static_cast<unsigned long long>(walkCauses[7]),
            static_cast<unsigned long long>(walkCauses[8]));
        std::fprintf(stderr, "[walk-probe] never-armed walks by cause (walks/MiB): no whole page %llu/%llu, shared view %llu/%llu, not private read-write %llu/%llu, hot block %llu/%llu, other %llu/%llu\n",
            static_cast<unsigned long long>(neverArmedWhy[0]), static_cast<unsigned long long>(neverArmedBytes[0] >> 20u), static_cast<unsigned long long>(neverArmedWhy[1]), static_cast<unsigned long long>(neverArmedBytes[1] >> 20u),
            static_cast<unsigned long long>(neverArmedWhy[2]), static_cast<unsigned long long>(neverArmedBytes[2] >> 20u), static_cast<unsigned long long>(neverArmedWhy[3]), static_cast<unsigned long long>(neverArmedBytes[3] >> 20u),
            static_cast<unsigned long long>(neverArmedWhy[4]), static_cast<unsigned long long>(neverArmedBytes[4] >> 20u));
        std::fprintf(stderr, "[walk-probe] never-armed walks of resident views (walks): pinned %llu, in a host write %llu, in a host import %llu, owned pages pending %llu, hot block %llu, not private read-write %llu, none of these %llu\n",
            static_cast<unsigned long long>(neverArmedWhy[5]), static_cast<unsigned long long>(neverArmedWhy[6]), static_cast<unsigned long long>(neverArmedWhy[7]), static_cast<unsigned long long>(neverArmedWhy[8]),
            static_cast<unsigned long long>(neverArmedWhy[9]), static_cast<unsigned long long>(neverArmedWhy[10]), static_cast<unsigned long long>(neverArmedWhy[11]));
        std::fprintf(stderr, "[walk-probe] resident walks: not resetting %llu, armed-only collection %llu, run not settled %llu, offered for arming %llu; arm pieces: already armed %llu, open store group %llu, hot %llu, pinned %llu, host access or unused import %llu, unarmable %llu, owned pending %llu, view %llu, arm failed %llu, armed %llu\n",
            static_cast<unsigned long long>(residentWalkWhy[0]), static_cast<unsigned long long>(residentWalkWhy[1]), static_cast<unsigned long long>(residentWalkWhy[2]), static_cast<unsigned long long>(residentWalkWhy[3]),
            static_cast<unsigned long long>(armPieceWhy[0]), static_cast<unsigned long long>(armPieceWhy[1]), static_cast<unsigned long long>(armPieceWhy[2]), static_cast<unsigned long long>(armPieceWhy[3]),
            static_cast<unsigned long long>(armPieceWhy[4]), static_cast<unsigned long long>(armPieceWhy[5]), static_cast<unsigned long long>(armPieceWhy[6]), static_cast<unsigned long long>(armPieceWhy[7]),
            static_cast<unsigned long long>(armPieceWhy[8]), static_cast<unsigned long long>(armPieceWhy[9]));
        const auto printTop = [](const char* name, const auto& histogram) {
            std::vector<std::pair<std::uint64_t, std::uintptr_t>> top;
            std::uint64_t total = 0;
            for (const auto& [address, count] : histogram) {
                top.emplace_back(count, address);
                total += count;
            }
            const auto n = std::min<std::size_t>(top.size(), 8);
            std::partial_sort(top.begin(), top.begin() + n, top.end(), std::greater<>{});
            std::fprintf(stderr, "[walk-probe] %s unique %zu samples %llu", name, top.size(), static_cast<unsigned long long>(total));
            for (std::size_t i = 0; i < n; ++i) {
                const auto [count, address] = top[i];
                char image[MAX_PATH]{};
                MEMORY_BASIC_INFORMATION memory{};
                if (name[0] == 'i' && VirtualQuery(reinterpret_cast<void*>(address), &memory, sizeof(memory)) != 0)
                    GetModuleFileNameA(static_cast<HMODULE>(memory.AllocationBase), image, MAX_PATH);
                std::fprintf(stderr, " 0x%llx=%llu[%s+0x%llx]", static_cast<unsigned long long>(address), static_cast<unsigned long long>(count),
                    image, static_cast<unsigned long long>(memory.AllocationBase ? address - reinterpret_cast<std::uintptr_t>(memory.AllocationBase) : 0));
            }
            std::fprintf(stderr, "\n");
        };
        printTop("blocks", faultBlocks);
        printTop("instructions", faultSites);
        if (groupProbe()) printGroupFaults();
        if (groupWalkProbe()) {
            printGroupWalks();
            printGroupFaults();
            printGpuWrites();
        }
    }

    bool armable(std::uintptr_t start, std::uintptr_t end) {
        if (overlapsOwnedRange(ownedPins, start, end)) return refuse(0, start, end);
        if (overlapsOwnedRange(ownedHostAccess, start, end) || overlapsOwnedRange(unusedImports, start, end)) return refuse(1, start, end);
        if (overlaps(unarmable, start, end)) return refuse(2, start, end);
        if (const auto pending = ownedPending.lower_bound(start & ~(ownedBlockBytes - 1)); pending != ownedPending.end() && pending->first < end) return refuse(3, start, end);
        for (auto view = views.lower_bound(start & ~(pageBytes - 1)); view != views.end() && view->first < end; ++view) {
            const auto& found = view->second;
            if (!found.resident || found.pins != 0 || found.hostWrites != 0 || found.hostImports != 0) return refuse(4, start, end);
        }
        return true;
    }

    // Probe: bytes refused per reason (pins, open host access, unarmable, owned pending, views),
    // printed with APS5_PROTECT_PROBE=1 every 2^16 refusals.
    static bool protectProbe() {
        static const bool probe = std::getenv("APS5_PROTECT_PROBE") != nullptr;
        return probe;
    }

    bool refuse(int reason, std::uintptr_t start, std::uintptr_t end) {
        groupRefusal = reason + 1;
        if (!protectProbe()) return false;
        refusedBytes[reason] += end - start;
        if ((++refusals & 0xffff) == 0) {
            std::uint64_t unarmableBytes = 0;
            for (const auto& [from, to] : unarmable) unarmableBytes += to - from;
            std::fprintf(stderr, "[protect] refused MiB: pins %.0f host %.0f unarmable %.0f pending %.0f views %.0f; unarmable ranges %zu (%.0f MiB)\n", refusedBytes[0] / 1048576.0, refusedBytes[1] / 1048576.0, refusedBytes[2] / 1048576.0, refusedBytes[3] / 1048576.0, refusedBytes[4] / 1048576.0, unarmable.size(), unarmableBytes / 1048576.0);
        }
        return false;
    }
    std::array<std::uint64_t, 5> refusedBytes{};
    std::uint64_t refusals = 0;

    // A range a resetting collect just walked under the lock: its armed part had no store since (a
    // store faults first), so it becomes clean. Unarmed, armable memory outside imports is armed; a
    // block that faulted recently (strikes at or above armHotLimit) stays CPU-dirty, and each attempt
    // wears a strike off. Memory inside imports is armed the same way (a store import left uneven
    // is made uniform at its next storing submission), except inside a store group opened as a
    // whole, which stays open until it is retried.
    void armWalked(std::uintptr_t from, std::uintptr_t to) {
        from = (from + 4095) & ~std::uintptr_t{4095};
        to &= ~std::uintptr_t{4095};
        if (from >= to) return;
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> runs;
        auto armed = armedRanges.upper_bound(from);
        if (armed != armedRanges.begin() && std::prev(armed)->second > from) --armed;
        for (auto cursor = from; cursor < to;) {
            if (armed != armedRanges.end() && armed->first <= cursor) {
                const auto armedEnd = std::min(to, armed->second);
                if (cursor < armedEnd) remember(cleanRanges, cursor, armedEnd);
                ++armPieceWhy[0];
                cursor = std::max(cursor, armed->second);
                ++armed;
                continue;
            }
            const auto gapEnd = armed == armedRanges.end() ? to : std::min(to, armed->first);
            const bool openGroups = overlaps(openImports, cursor, gapEnd);
            while (cursor < gapEnd) {
                const auto block = cursor & ~(armBlockBytes - 1);
                const auto pieceEnd = std::min(gapEnd, block + armBlockBytes);
                if (openGroups && overlaps(openImports, cursor, pieceEnd)) {
                    ++armPieceWhy[1];
                    cursor = pieceEnd;
                    continue;
                }
                bool eligible = true;
                if (const auto strikes = armStrikes.find(block); strikes != armStrikes.end()) {
                    eligible = strikes->second < armHotLimit;
                    if (--strikes->second == 0) armStrikes.erase(strikes);
                    if (!eligible) ++armHotSkips, ++armPieceWhy[2];
                }
                if (eligible) {
                    if (armable(cursor, pieceEnd)) {
                        if (!runs.empty() && runs.back().second == cursor) runs.back().second = pieceEnd;
                        else runs.emplace_back(cursor, pieceEnd);
                    } else ++armPieceWhy[2 + std::clamp(groupRefusal, 1, 5)];
                }
                cursor = pieceEnd;
            }
        }
        for (const auto& [start, end] : runs) ++armPieceWhy[armRun(start, end) ? 9 : 8];
    }

    // Whether [start, end) holds committed private read-write memory only (what armRun arms).
    bool armablePrivate(std::uintptr_t start, std::uintptr_t end) {
        for (auto at = start; at < end;) {
            const auto memory = query(at);
            if (memory.State != MEM_COMMIT || memory.Type != MEM_PRIVATE || memory.Protect != PAGE_READWRITE) return false;
            at = reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize;
        }
        return true;
    }

    // Arms [start, end) read-only. With walked set, a resetting collect has just walked it and it
    // becomes clean unless a store landed after the walk's reset (it is then armed but not clean,
    // and the next walk consumes the store). Without, it is armed and not clean: its stores since the
    // last walk are still in the write watch.
    bool armRun(std::uintptr_t start, std::uintptr_t end, bool walked = true) {
        if (!armablePrivate(start, end)) return false;
        // Recorded before the protection changes: a store faulting meanwhile waits for the lock and
        // then finds the block armed and resumes.
        remember(armedRanges, start, end);
        DWORD previous;
        if (!VirtualProtect(reinterpret_cast<void*>(start), end - start, PAGE_READONLY, &previous)) {
            groupError = GetLastError();
            forget(armedRanges, start, end);
            return false;
        }
        noteUneven(start, end);
        ++armRuns;
        if (walkProbe()) remember(everArmed, start, end);
        if (protectTrace()) std::fprintf(stderr, "[protect] arm 0x%llx-0x%llx%s\n", static_cast<unsigned long long>(start), static_cast<unsigned long long>(end), walked ? "" : " (submission)");
        if (!walked) return true;
        void* written[1]{};
        ULONG_PTR reported = 1;
        DWORD granularity = 0;
        if (GetWriteWatch(0, reinterpret_cast<void*>(start), end - start, written, &reported, &granularity) != 0 || reported != 0) {
            ++armRaced;
            return true;
        }
        remember(cleanRanges, start, end);
        return true;
    }

    bool overlapsImport(std::uintptr_t start, std::uintptr_t end) const {
        for (auto it = importExtents.lower_bound(start > longestImport ? start - longestImport : 0); it != importExtents.end() && it->first < end; ++it) {
            if (it->second > start) return true;
        }
        return false;
    }

    bool overlapsStore(std::uintptr_t start, std::uintptr_t end) const {
        for (auto it = storeExtents.lower_bound(start > longestStore ? start - longestStore : 0); it != storeExtents.end() && it->first < end; ++it) {
            if (it->second > start) return true;
        }
        return false;
    }

    void noteUneven(std::uintptr_t start, std::uintptr_t end) {
        if (overlapsStore(start, end)) remember(unevenImports, start, end);
    }

    // A cause that kept an import group open (a pin, an open host access) ended in [start, end):
    // the next submission tries to arm the group again.
    void retryImports(std::uintptr_t start, std::uintptr_t end) {
        if (!overlaps(openImports, start, end)) return;
        auto it = openImports.lower_bound(start);
        if (it != openImports.begin() && std::prev(it)->second > start) --it;
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> groups;
        for (; it != openImports.end() && it->first < end; ++it) groups.emplace_back(*it);
        for (const auto& [from, to] : groups) {
            forget(openImports, from, to);
            if (walkProbe()) for (auto& cause : openCauses) forget(cause, from, to);
            remember(unevenImports, from, to);
        }
    }

    // Makes the group of store imports overlapping [start, end) (those overlapping it,
    // transitively) uniform: all armed, or, when some committed page in it cannot be armed, all open.
    void uniform(std::uintptr_t start, std::uintptr_t end) {
        auto from = start, to = end;
        if (!importGroup(storeExtents, longestStore, from, to)) return;
        if (groupProbe() || groupWalkProbe()) {
            forget(probeGroups, from, to);
            probeGroups.insert_or_assign(from, to);
        }
        groupRefusal = 0;
        const bool wasOpen = overlaps(openImports, from, to);
        // A group whose blocks fault again soon after each arming costs more armed than walked: it
        // stays CPU-dirty as a whole (uniform read-write) and is retried later, each time later.
        const bool faulted = overlaps(faultedBlocks, from, to);
        if (faulted) forget(faultedBlocks, from, to);
        auto heat = groupHeat.find(from);
        if (faulted && !wasOpen) {
            if (heat == groupHeat.end()) heat = groupHeat.emplace(from, GroupHeat{}).first;
            auto& entry = heat->second;
            if (entry.armed != 0 && submits - entry.armed < hotSubmits) entry.strikes = static_cast<std::uint8_t>(std::min(255, entry.strikes + 1));
            else if (entry.strikes != 0) --entry.strikes;
            if (entry.strikes >= hotStrikes && !groupProbe()) {
                const auto wait = (hotSubmits * 2) << std::min<unsigned>(entry.strikes - hotStrikes, 6);
                hotRetries.emplace(submits + wait, std::pair{from, to});
                ++hotGroups;
                if (groupWalkProbe() && to - from >= 64ull * 1024 * 1024) {
                    const auto found = groupFaultBlocks.find({from, to});
                    std::fprintf(stderr, "[group-trigger] %llx-%llx strikes %u blocks %zu\n",
                        static_cast<unsigned long long>(from), static_cast<unsigned long long>(to), static_cast<unsigned>(entry.strikes),
                        found == groupFaultBlocks.end() ? 0 : found->second.size());
                    if (found != groupFaultBlocks.end()) {
                        for (const auto& [block, count] : found->second)
                            std::fprintf(stderr, "[group-trigger-block] %llx %llu\n", static_cast<unsigned long long>(block), static_cast<unsigned long long>(count));
                    }
                    for (auto it = importExtents.lower_bound(from > longestImport ? from - longestImport : 0);
                         it != importExtents.end() && it->first < to; ++it) {
                        if (it->second > from)
                            std::fprintf(stderr, "[group-trigger-import] %llx-%llx\n", static_cast<unsigned long long>(it->first), static_cast<unsigned long long>(it->second));
                    }
                }
                if (protectProbe() && groupReports++ < 400) std::fprintf(stderr, "[protect] import group 0x%llx+%.1f MiB stays open: hot (strikes %u, retried in %llu submissions)\n", static_cast<unsigned long long>(from), (to - from) / 1048576.0, static_cast<unsigned>(entry.strikes), static_cast<unsigned long long>(wait));
                remember(openImports, from, to);
                if (walkProbe()) remember(openCauses[0], from, to);
                if (overlaps(armedRanges, from, to)) openArmed(from, to);
                return;
            }
        }
        if (!wasOpen && armGroup(from, to)) {
            if (heat != groupHeat.end()) heat->second.armed = submits;
            return;
        }
        remember(openImports, from, to);
        if (walkProbe() && !wasOpen) {
            const bool unused = groupRefusal == 2 && overlapsOwnedRange(unusedImports, from, to);
            const bool host = overlapsOwnedRange(ownedHostAccess, from, to);
            remember(openCauses[unused && !host ? 1 : 2], from, to);
        }
        if (protectProbe() && !wasOpen && groupReports++ < 400) {
            // Reasons: 1 pin, 2 open host access or unused import, 3 unarmable, 4 owned pending,
            // 5 shared view, 6 not a guest allocation, 7 not private read-write, 8 protect failed.
            std::uint64_t openBytes = 0;
            for (const auto& [start, stop] : openImports) openBytes += stop - start;
            const auto memory = query(groupRefusalAt);
            std::fprintf(stderr, "[protect] import group 0x%llx+%.1f MiB stays open: reason %d at 0x%llx (state 0x%lx type 0x%lx protect 0x%lx); open groups hold %.0f MiB, %zu imports, %zu unused, %zu host accesses, %zu pins, error %lu\n", static_cast<unsigned long long>(from), (to - from) / 1048576.0, groupRefusal, static_cast<unsigned long long>(groupRefusalAt), memory.State, memory.Type, memory.Protect, openBytes / 1048576.0, importExtents.size(), unusedImports.size(), ownedHostAccess.size(), ownedPins.size(), groupError);
        }
        if (!overlaps(armedRanges, from, to)) return;
        ++importOpens;
        if (protectTrace()) std::fprintf(stderr, "[protect] open import group 0x%llx-0x%llx\n", static_cast<unsigned long long>(from), static_cast<unsigned long long>(to));
        openArmed(from, to);
    }

    // Arms every unarmed committed page of [from, to), or nothing when one of them cannot be armed.
    // Pages not committed are left (they hold no imported memory).
    bool armGroup(std::uintptr_t from, std::uintptr_t to) {
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> runs;
        std::uintptr_t runAllocation = 0;
        auto armed = armedRanges.upper_bound(from);
        if (armed != armedRanges.begin() && std::prev(armed)->second > from) --armed;
        for (auto cursor = from; cursor < to;) {
            if (armed != armedRanges.end() && armed->first <= cursor) {
                cursor = std::max(cursor, armed->second);
                ++armed;
                continue;
            }
            const auto gapEnd = armed == armedRanges.end() ? to : std::min(to, armed->first);
            while (cursor < gapEnd) {
                const auto memory = query(cursor);
                auto regionEnd = std::min(gapEnd, reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize);
                if (memory.State == MEM_COMMIT) {
                    // Only guest allocations are armed: an import of host memory (driver pools,
                    // test buffers) has no fault handling for its writers, so its group stays open.
                    const auto guestEnd = guestAllocationEnd(cursor);
                    groupRefusalAt = cursor;
                    if (guestEnd == 0) return groupRefusal = 6, false;
                    regionEnd = std::min(regionEnd, guestEnd);
                    if (memory.Type != MEM_PRIVATE || memory.Protect != PAGE_READWRITE) return groupRefusal = 7, false;
                    if (!armable(cursor, regionEnd)) return false;
                    // A protection change cannot span two allocations: runs join only inside one.
                    const auto allocation = reinterpret_cast<std::uintptr_t>(memory.AllocationBase);
                    if (!runs.empty() && runs.back().second == cursor && runAllocation == allocation) runs.back().second = regionEnd;
                    else runs.emplace_back(cursor, regionEnd);
                    runAllocation = allocation;
                }
                cursor = regionEnd;
            }
        }
        for (const auto& [start, end] : runs) {
            if (!armRun(start, end, false)) return groupRefusalAt = start, groupRefusal = 8, false;
            submitArms += end - start;
        }
        return true;
    }

    static bool protectTrace() {
        static const bool trace = std::getenv("APS5_PROTECT_TRACE") != nullptr;
        return trace;
    }

    void openArmed(std::uintptr_t start, std::uintptr_t end) {
        if (!overlaps(armedRanges, start, end)) return;
        armedOpens.fetch_add(1, std::memory_order_release);
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> opened;
        auto it = armedRanges.lower_bound(start);
        if (it != armedRanges.begin() && std::prev(it)->second > start) --it;
        for (; it != armedRanges.end() && it->first < end; ++it) opened.emplace_back(std::max(start, it->first), std::min(end, it->second));
        forget(armedRanges, start, end);
        for (const auto& [from, to] : opened) {
            forget(cleanRanges, from, to);
            noteUneven(from, to);
            for (auto at = from; at < to;) {
                const auto memory = query(at);
                const auto regionEnd = std::min(to, reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize);
                DWORD previous;
                if (memory.State == MEM_COMMIT && memory.Protect == PAGE_READONLY && !VirtualProtect(reinterpret_cast<void*>(at), regionEnd - at, PAGE_READWRITE, &previous)) fail("open armed guest memory");
                at = regionEnd;
            }
        }
    }

    void rememberClean(std::uintptr_t start, std::uintptr_t end) {
        remember(cleanRanges, start, end);
    }

    void invalidate(SharedPage& page, std::uint8_t mask = allSubpages) {
        for (std::size_t sub = 0; sub < subpages; ++sub) {
            if ((mask & (1u << sub)) == 0) continue;
            ++page.generation[sub];
            for (const auto alias : page.aliases) forgetClean(alias + sub * subpageBytes, alias + (sub + 1) * subpageBytes);
        }
    }

    void invalidate(std::uintptr_t base, const View& view, std::uint8_t mask = allSubpages) {
        if (view.page) {
            invalidate(*view.page, mask);
            return;
        }
        for (std::size_t sub = 0; sub < subpages; ++sub) {
            if ((mask & (1u << sub)) != 0) forgetClean(base + sub * subpageBytes, base + (sub + 1) * subpageBytes);
        }
    }

    static std::uint8_t subpagesOf(std::uintptr_t base, std::uintptr_t address, std::uintptr_t end) {
        const auto from = std::max(address, base);
        const auto to = std::min(end, base + pageBytes);
        if (to <= from) return 0;
        const auto first = (from - base) / subpageBytes;
        const auto last = (to - 1 - base) / subpageBytes;
        return static_cast<std::uint8_t>(((1u << (last + 1)) - 1) & ~((1u << first) - 1));
    }

    static void protectSubpages(std::uintptr_t base, std::uint8_t mask, DWORD protection, const char* operation, std::size_t viewBytes = pageBytes) {
        for (std::size_t sub = 0; sub < subpages;) {
            if ((mask & (1u << sub)) == 0) {
                ++sub;
                continue;
            }
            auto stop = sub + 1;
            while (stop < subpages && (mask & (1u << stop)) != 0 && stop * subpageBytes != viewBytes) ++stop;
            DWORD previous;
            if (!VirtualProtect(reinterpret_cast<void*>(base + sub * subpageBytes), (stop - sub) * subpageBytes, protection, &previous)) fail(operation);
            sub = stop;
        }
    }

    bool sameSection(HANDLE first, HANDLE second) const {
        return compare != nullptr && compare(first, second);
    }

    static bool writable(DWORD protection) {
        return protection == PAGE_READWRITE || protection == PAGE_EXECUTE_READWRITE;
    }
    using AllocateFunction = PVOID (WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER*, ULONG);
    using MapFunction = PVOID (WINAPI*)(HANDLE, HANDLE, PVOID, ULONG64, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER*, ULONG);
    using UnmapFunction = BOOL (WINAPI*)(HANDLE, PVOID, ULONG);
    using CompareFunction = BOOL (WINAPI*)(HANDLE, HANDLE);

    WindowsMappings() {
        const auto module = GetModuleHandleW(L"KernelBase.dll");
        if (!module) fail("load Windows memory API");
        allocate = reinterpret_cast<AllocateFunction>(GetProcAddress(module, "VirtualAlloc2"));
        map = reinterpret_cast<MapFunction>(GetProcAddress(module, "MapViewOfFile3"));
        unmap = reinterpret_cast<UnmapFunction>(GetProcAddress(module, "UnmapViewOfFile2"));
        if (!allocate || !map || !unmap) throw std::runtime_error("Windows placeholder memory APIs are required");
        compare = reinterpret_cast<CompareFunction>(GetProcAddress(module, "CompareObjectHandles"));
        SYSTEM_INFO system{};
        GetSystemInfo(&system);
        granularity = system.dwAllocationGranularity;
    }

    [[noreturn]] static void fail(const char* operation) {
        throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), operation);
    }

    static MEMORY_BASIC_INFORMATION query(std::uintptr_t address) {
        MEMORY_BASIC_INFORMATION memory{};
#ifdef APS5_COUNT_MAPPING_QUERIES
        queries.fetch_add(1, std::memory_order_relaxed);
#endif
        if (VirtualQuery(reinterpret_cast<void*>(address), &memory, sizeof(memory)) != sizeof(memory)) fail("query guest memory");
        return memory;
    }

    static std::uintptr_t split(std::uintptr_t address, std::size_t bytes) {
        auto memory = query(address);
        memory = query(reinterpret_cast<std::uintptr_t>(memory.AllocationBase));
        if (memory.State != MEM_RESERVE) throw std::runtime_error("guest mapping requires a placeholder");
        const auto base = reinterpret_cast<std::uintptr_t>(memory.BaseAddress);
        if (address != base) {
            if (!VirtualFree(reinterpret_cast<void*>(base), address - base, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) fail("split guest placeholder prefix");
            memory = query(address);
        }
        if (memory.RegionSize < bytes) throw std::runtime_error("guest placeholder is too small");
        if (memory.RegionSize != bytes && !VirtualFree(reinterpret_cast<void*>(address), bytes, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) fail("split guest placeholder suffix");
        return address + memory.RegionSize;
    }

    void reset(std::uintptr_t address, std::size_t bytes) {
        disown(address, address + bytes);
        forgetOwnedPending(address, address + bytes);
        forgetZero(address, address + bytes);
        collectableRanges.clear();
        const auto end = address + bytes;
        privateAllocations.erase(privateAllocations.lower_bound(address), privateAllocations.lower_bound(end));
        forgetClean(address, end);
        forget(unarmable, address, end);
        if (walkProbe()) forget(everArmed, address, end);
        breakRun(address);
        breakRun(end);
        breakView(address);
        breakView(end);
        if (const auto run = residentRuns.find(address); run != residentRuns.end() && run->second.end == end) {
            const auto memory = query(address);
            if (reinterpret_cast<std::uintptr_t>(memory.AllocationBase) != address) throw std::runtime_error("cannot release part of a host allocation");
            if (memory.Type != MEM_PRIVATE) throw std::runtime_error("resident guest mapping is not private memory");
            auto allocationEnd = address;
            do {
                const auto part = query(allocationEnd);
                if (part.AllocationBase != memory.AllocationBase) break;
                allocationEnd = reinterpret_cast<std::uintptr_t>(part.BaseAddress) + part.RegionSize;
            } while (allocationEnd < end);
            if (allocationEnd != end || query(end).AllocationBase == memory.AllocationBase) throw std::runtime_error("guest release truncates a host allocation");
            evict(run);
            if (!VirtualFree(reinterpret_cast<void*>(address), bytes, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) fail("release private guest memory");
            return;
        }
        for (auto cursor = address; cursor < end;) {
            const auto memory = query(cursor);
            if (memory.State == MEM_RESERVE) {
                cursor = std::min(end, reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize);
                continue;
            }
            if (reinterpret_cast<std::uintptr_t>(memory.AllocationBase) != cursor) throw std::runtime_error("cannot release part of a host allocation");
            auto allocationEnd = cursor;
            do {
                const auto part = query(allocationEnd);
                if (part.AllocationBase != memory.AllocationBase) break;
                allocationEnd = reinterpret_cast<std::uintptr_t>(part.BaseAddress) + part.RegionSize;
            } while (allocationEnd < end);
            if (allocationEnd > end || query(allocationEnd).AllocationBase == memory.AllocationBase) throw std::runtime_error("guest release truncates a host allocation");
            if (memory.Type == MEM_MAPPED) {
                if (!unmap(GetCurrentProcess(), reinterpret_cast<void*>(cursor), MEM_PRESERVE_PLACEHOLDER)) fail("unmap shared guest page");
                for (auto found = views.lower_bound(cursor); found != views.end() && found->first < allocationEnd;) {
                    if (found->second.page != nullptr) std::erase(found->second.page->aliases, found->first);
                    found = views.erase(found);
                }
                forget(freshRanges, cursor, allocationEnd);
            } else if (memory.Type == MEM_PRIVATE) {
                if (const auto run = residentRuns.find(cursor); run != residentRuns.end()) evict(run);
                if (!VirtualFree(reinterpret_cast<void*>(cursor), allocationEnd - cursor, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) fail("release private guest memory");
            } else {
                throw std::runtime_error("unsupported guest mapping type");
            }
            cursor = allocationEnd;
        }
        const auto last = query(reinterpret_cast<std::uintptr_t>(query(end - 1).AllocationBase));
        const auto lastBase = reinterpret_cast<std::uintptr_t>(last.BaseAddress);
        if (lastBase + last.RegionSize > end) split(lastBase, end - lastBase);
        const auto first = query(address);
        split(address, std::min(bytes, reinterpret_cast<std::uintptr_t>(first.BaseAddress) + first.RegionSize - address));
        if (query(address).RegionSize != bytes && !VirtualFree(reinterpret_cast<void*>(address), bytes, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS)) fail("coalesce guest placeholders");
    }

    void* mapAliasRuns(std::uintptr_t address, std::size_t bytes) {
        auto* reserved = allocate(GetCurrentProcess(), nullptr, bytes, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0);
        if (reserved == nullptr) fail("reserve a read-write alias of shared guest memory");
        const auto base = reinterpret_cast<std::uintptr_t>(reserved);
        try {
            for (std::size_t done = 0; done < bytes;) {
                const auto& first = views.at(address + done);
                auto run = pageBytes;
                while (done + run < bytes) {
                    const auto& next = views.at(address + done + run);
                    if (next.offset != first.offset + run || (next.section != first.section && !sameSection(next.section->handle, first.section->handle))) break;
                    run += pageBytes;
                }
                split(base + done, run);
                if (!map(first.section->handle, GetCurrentProcess(), reinterpret_cast<void*>(base + done), first.offset, run, MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, nullptr, 0)) fail("map a run of a read-write alias of shared guest memory");
                done += run;
            }
        } catch (...) {
            releaseAlias(base, bytes);
            throw;
        }
        splitAliases.emplace(base, bytes);
        return reserved;
    }

    DWORD releaseAlias(std::uintptr_t base, std::size_t bytes) {
        DWORD error = ERROR_SUCCESS;
        for (auto cursor = base; cursor < base + bytes;) {
            const auto memory = query(cursor);
            const auto released = memory.State == MEM_RESERVE ? VirtualFree(memory.BaseAddress, 0, MEM_RELEASE) : unmap(GetCurrentProcess(), memory.BaseAddress, 0);
            if (!released && error == ERROR_SUCCESS) error = GetLastError();
            cursor = reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize;
        }
        return error;
    }

    std::map<std::uintptr_t, std::uint64_t> ownedWords;
    std::map<std::uintptr_t, std::uint16_t> ownedPending;
    OwnedRanges ownedPins;
    OwnedRanges ownedHostAccess;
    // Registered imports the GPU has not used yet (RegisterImport).
    OwnedRanges unusedImports;
    // Probe (APS5_PROTECT_PROBE): why the last import group could not be armed, and where.
    int groupRefusal = 0;
    std::uintptr_t groupRefusalAt = 0;
    std::uint64_t groupReports = 0;
    DWORD groupError = 0;
    std::uint64_t nextOwned = 1;
    Ranges cleanRanges;
    static constexpr std::uintptr_t armBlockBytes = 0x10000;
    static constexpr std::uint8_t armHotLimit = 8;
    static constexpr int armFaultStrikes = 16;
    Ranges armedRanges;
    Ranges unarmable;
    // Times armed memory was opened (a fault, a host write, a pin, an import): read without the
    // lock by ArmedOpens.
    std::atomic<std::uint64_t> armedOpens{0};
    std::multimap<std::uintptr_t, std::uintptr_t> importExtents;
    std::uintptr_t longestImport = 0;
    // The store imports among them (RegisterImport with store set).
    std::multimap<std::uintptr_t, std::uintptr_t> storeExtents;
    std::uintptr_t longestStore = 0;
    // Ranges whose armed state changed inside a store import since it was last made uniform
    // (BeginSubmit), and store groups found unarmable or hot (opened; retried once a cause ends).
    Ranges unevenImports;
    Ranges openImports;
    std::array<Ranges, 3> openCauses;
    Ranges everArmed;
    Ranges probeGroups;
    std::map<std::pair<std::uintptr_t, std::uintptr_t>, std::pair<std::uint64_t, std::uint64_t>> groupWalks;
    std::map<std::pair<std::uintptr_t, std::uintptr_t>, std::map<std::uintptr_t, std::uint64_t>> groupFaultBlocks;
    std::uint64_t unmappedGroupFaults = 0;
    std::array<std::uint64_t, 9> walkCauses{};
    std::array<std::uint64_t, 12> neverArmedWhy{};
    std::array<std::uint64_t, 12> neverArmedBytes{};
    // Probe: resident walks by what kept them from arming, and armWalked pieces by outcome.
    std::array<std::uint64_t, 4> residentWalkWhy{};
    std::array<std::uint64_t, 10> armPieceWhy{};
    std::map<std::uintptr_t, std::uint64_t> faultBlocks;
    std::map<std::uintptr_t, std::uint64_t> faultSites;
    std::atomic<DWORD> submitter{0};
    std::uint64_t submits = 0;
    std::uint64_t submitArms = 0;
    std::uint64_t importOpens = 0;
    // Import groups (by start) that faulted after an arming: when they were last armed and how
    // often they faulted again within hotSubmits submissions; blocks inside imports that faulted
    // since their group was last made uniform; hot groups by the submission that retries them.
    struct GroupHeat {
        std::uint64_t armed = 0;
        std::uint8_t strikes = 0;
    };
    static constexpr std::uint64_t hotSubmits = 256;
    static constexpr std::uint8_t hotStrikes = 3;
    std::unordered_map<std::uintptr_t, GroupHeat> groupHeat;
    Ranges faultedBlocks;
    std::multimap<std::uint64_t, std::pair<std::uintptr_t, std::uintptr_t>> hotRetries;
    std::uint64_t hotGroups = 0;
    std::unordered_map<std::uintptr_t, std::uint8_t> armStrikes;
    std::uint64_t collectWalks = 0;
    std::uint64_t armRuns = 0;
    std::uint64_t armFaults = 0;
    std::uint64_t armHotSkips = 0;
    std::uint64_t armRaced = 0;
    Ranges collectableRanges;
    Ranges privateAllocations;
#ifdef APS5_COUNT_MAPPING_QUERIES
    std::uint64_t collectSystemCalls = 0;
    std::uint64_t collectPieces = 0;
#endif
    Ranges freshRanges;
    std::unordered_map<std::uintptr_t, std::array<std::uint64_t, chunkBytes / 4096 / 64>> zeroPages;
    std::map<std::uintptr_t, View> views;

    using Views = std::map<std::uintptr_t, View>;

    Views::iterator viewAt(std::uintptr_t base, Views::iterator& position) {
        if (position != views.end() && position->first < base) {
            ++position;
            if (position != views.end() && position->first < base) position = views.lower_bound(base);
        }
        return position != views.end() && position->first == base ? position : views.end();
    }
    std::map<std::pair<std::uintptr_t, std::uint64_t>, std::shared_ptr<Physical>> physical;
    std::map<std::uintptr_t, Run> residentRuns;
    std::map<std::pair<std::uintptr_t, std::uint64_t>, std::map<std::uintptr_t, Run>::iterator> residentIndex;
    std::vector<std::shared_ptr<Saved>> savedPool;
    std::size_t savedPoolBytes = 0;
    std::map<std::uintptr_t, std::size_t> splitAliases;
    std::size_t granularity = 0x10000;
    bool watchResident = false;
#ifdef APS5_COUNT_MAPPING_QUERIES
    static inline std::atomic<std::uint64_t> queries{0};
#endif
    std::mutex mutex;
    HANDLE zeroSection = nullptr;
    AllocateFunction allocate = nullptr;
    MapFunction map = nullptr;
    UnmapFunction unmap = nullptr;
    CompareFunction compare = nullptr;
};

}
#endif

#endif
