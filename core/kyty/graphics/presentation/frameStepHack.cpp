#include "graphics/presentation/frameStepHack.h"

#include "common/alignment.h"
#include "common/hackNotice.h"
#include "common/logging/log.h"
#include "common/timer.h"
#include "common/virtualMemory.h"
#include "kernel/memory.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fmt/format.h>
#include <functional>
#include <initializer_list>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace Libs::VideoOut::FrameStepHack {

namespace {

struct Title {
	std::string_view id;
	const char*      name;
	bool             pacing;
};

constexpr std::array<Title, 5> Titles {{
    {"PPSA21559", "ASTRO BOT", true},
    {"PPSA21564", "ASTRO BOT", true},
    {"PPSA21567", "ASTRO BOT", true},
    {"PPSA21568", "ASTRO BOT", true},
    {"PPSA01325", "ASTRO's PLAYROOM", false},
}};

constexpr float OwnFrameConstant = 1.0f / 60.0f;

constexpr double MaxStep        = 1.0 / 20.0;
constexpr double MaxStepUnfixed = 1.0 / 30.0;
constexpr uint32_t RescueHoldFrames = 3;
constexpr double AverageFrames = 8.0;
constexpr double FullRateSlack = 1.08;

struct Sites {
	uint64_t rate           = 0;
	uint64_t step           = 0;
	uint64_t period         = 0;
	uint64_t frame_constant = 0;
	uint64_t pacing_site    = 0;
	uint64_t pacing_frame   = 0;
};

struct State {
	bool        active       = false;
	const char* name         = nullptr;
	uint64_t    base         = 0;
	Sites       sites;
	double      own_rate     = 0.0;
	float       own_step     = 0.0f;
	uint64_t    own_period   = 0;
	double      written_rate = 0.0;
	uint64_t    last_counter = 0;
	double      frequency    = 0.0;
	double      average      = 0.0;
	bool        scaled       = false;
	double      max_step     = MaxStepUnfixed;
	uint64_t    cave         = 0;
	uint32_t    null_queries = 0;
	uint32_t    overflows    = 0;
	uint32_t    rescues      = 0;
	uint32_t    rescue_hold  = 0;
};

State& GetState() {
	static State state;
	return state;
}

template <typename T>
T Read(uint64_t address) {
	return std::atomic_ref<T>(*reinterpret_cast<T*>(address)).load(std::memory_order_relaxed);
}

template <typename T>
void Write(uint64_t address, T value) {
	std::atomic_ref<T>(*reinterpret_cast<T*>(address)).store(value, std::memory_order_relaxed);
}

std::string ParamString(const char* name) {
	std::string value;
	return Loader::SystemContentParamSfoGetString(name, &value) ? value : std::string("?");
}

class Image {
public:
	Image(uint64_t base, uint64_t size)
	    : m_data(reinterpret_cast<const uint8_t*>(base)), m_size(size) {}

	[[nodiscard]] bool Has(uint64_t offset, uint64_t size) const {
		return offset <= m_size && size <= m_size - offset;
	}
	[[nodiscard]] bool Is(uint64_t offset, std::span<const uint8_t> bytes) const {
		return Has(offset, bytes.size()) &&
		       std::memcmp(m_data + offset, bytes.data(), bytes.size()) == 0;
	}
	template <typename T>
	[[nodiscard]] T Get(uint64_t offset) const {
		T value {};
		std::memcpy(&value, m_data + offset, sizeof(T));
		return value;
	}
	[[nodiscard]] uint64_t Target(uint64_t disp, uint64_t end) const {
		return end + static_cast<uint64_t>(static_cast<int64_t>(Get<int32_t>(disp)));
	}
	[[nodiscard]] bool IsDouble(uint64_t offset, double value) const {
		return Has(offset, sizeof(double)) && Get<double>(offset) == value;
	}
	[[nodiscard]] bool IsFloat(uint64_t offset, float value) const {
		return Has(offset, sizeof(float)) && Get<float>(offset) == value;
	}
	[[nodiscard]] bool Contains(uint64_t begin, uint64_t end,
	                            std::span<const uint8_t> bytes) const {
		bool found = false;
		ForEach(begin, end + bytes.size() - 1, bytes,
		        [&](uint64_t p) { found = found || p < end; });
		return found;
	}
	template <typename F>
	void ForEach(uint64_t begin, uint64_t end, std::span<const uint8_t> bytes, F&& f) const {
		end = std::min(end, m_size);
		if (begin >= end || end - begin < bytes.size()) {
			return;
		}
		const auto* first = m_data + begin;
		const auto* last  = m_data + end;
		const std::boyer_moore_horspool_searcher searcher(bytes.begin(), bytes.end());
		for (auto it = std::search(first, last, searcher); it != last;
		     it      = std::search(it + 1, last, searcher)) {
			f(static_cast<uint64_t>(it - m_data));
		}
	}

private:
	const uint8_t* m_data;
	uint64_t       m_size;
};

constexpr std::array<uint8_t, 4> VmovsdLoad {0xc5, 0xfb, 0x10, 0x05};
constexpr std::array<uint8_t, 4> VdivsdLoad {0xc5, 0xfb, 0x5e, 0x05};
constexpr std::array<uint8_t, 4> VmulsdLoad {0xc5, 0xfb, 0x59, 0x05};
constexpr std::array<uint8_t, 4> VmovssLoad {0xc5, 0xfa, 0x10, 0x05};
constexpr std::array<uint8_t, 3> MovRaxLoad {0x48, 0x8b, 0x05};
constexpr std::array<uint8_t, 4> Vcvtsd2ss {0xc5, 0xfb, 0x5a, 0xc0};
constexpr std::array<uint8_t, 1> Call {0xe8};
constexpr std::array<uint8_t, 8> Int3x8 {0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc};
constexpr uint8_t                Ret = 0xc3;
constexpr uint8_t                Int3 = 0xcc;

std::string_view FindSites(const Image& image, std::span<const CodeRange> code, bool pacing,
                           Sites* sites) {
	uint64_t getter  = 0;
	uint32_t getters = 0;
	for (const auto& range: code) {
		image.ForEach(range.begin, range.end, VmovsdLoad, [&](uint64_t p) {
			if (!image.Has(p, 17) || !image.Is(p + 8, VdivsdLoad) ||
			    image.Get<uint8_t>(p + 16) != Ret || !image.IsDouble(image.Target(p + 4, p + 8), 1.0)) {
				return;
			}
			const auto rate          = image.Target(p + 12, p + 16);
			bool       period_getter = false;
			for (uint64_t q = p >= 32 ? p - 32 : 0; q < p && !period_getter; q++) {
				period_getter = image.Is(q, MovRaxLoad) && image.Get<uint8_t>(q + 7) == Ret &&
				                image.Target(q + 3, q + 7) == rate + 16;
			}
			bool step_getter = false;
			for (uint64_t q = p + 17; q < p + 64 && image.Has(q, 9) && !step_getter; q++) {
				step_getter = image.Is(q, VmovssLoad) && image.Get<uint8_t>(q + 8) == Ret &&
				              image.Target(q + 4, q + 8) == rate + 8;
			}
			if (period_getter && step_getter && image.Has(rate, 24)) {
				getter      = p;
				sites->rate = rate;
				getters++;
			}
		});
	}
	if (getters != 1) {
		return getters == 0 ? "frame-rate getters not found" : "frame-rate getters found twice";
	}
	sites->step   = sites->rate + 8;
	sites->period = sites->rate + 16;

	uint32_t pacing_sites = 0;
	bool     update       = false;
	for (const auto& range: code) {
		image.ForEach(range.begin, range.end, Call, [&](uint64_t p) {
			if (!image.Has(p, 21) || image.Target(p + 1, p + 5) != getter) {
				return;
			}
			if (image.Is(p + 5, VmulsdLoad) && image.IsDouble(image.Target(p + 9, p + 13), 1000.0)) {
				sites->pacing_site = p;
				pacing_sites++;
			}
			for (uint64_t q = p + 5; q + Vcvtsd2ss.size() <= p + 21 && !update; q++) {
				update = image.Is(q, Vcvtsd2ss);
			}
		});
	}
	if (!pacing && pacing_sites != 0) {
		return "unexpected frame pacing found";
	}
	if (pacing && pacing_sites != 1) {
		return pacing_sites == 0 ? "frame pacing not found" : "frame pacing found twice";
	}
	if (!update) {
		return "main update not found";
	}

	if (pacing) {
		const auto range = std::find_if(code.begin(), code.end(), [&](const CodeRange& r) {
			return sites->pacing_site >= r.begin && sites->pacing_site < r.end;
		});
		const uint64_t start =
		    sites->pacing_site >= 8 ? (sites->pacing_site - 8) & ~uint64_t {7} : 0;
		for (uint64_t a = start; a >= range->begin && a + 0x10000 > sites->pacing_site && a >= 8;
		     a -= 8) {
			if (image.Is(a, Int3x8)) {
				sites->pacing_frame = a;
				break;
			}
		}
		if (sites->pacing_frame == 0) {
			return "padding for the pacing frame time not found";
		}
	}

	std::vector<uint64_t> constants;
	image.ForEach(0, UINT64_MAX, std::bit_cast<std::array<uint8_t, 4>>(OwnFrameConstant),
	              [&](uint64_t a) {
		              if (a % 4 == 0 && (a < sites->rate || a >= sites->rate + 24)) {
			              constants.push_back(a);
		              }
	              });
	if (constants.empty()) {
		return "1/60 constant not found";
	}
	std::vector<uint32_t> reads(constants.size());
	for (const auto& r: code) {
		for (uint64_t p = r.begin; p + 4 <= r.end && image.Has(p, 4); p++) {
			const auto target = image.Target(p, p + 4);
			if (target < constants.front() || target > constants.back()) {
				continue;
			}
			const auto it = std::lower_bound(constants.begin(), constants.end(), target);
			if (it != constants.end() && *it == target) {
				reads[static_cast<size_t>(it - constants.begin())]++;
			}
		}
	}
	const auto most   = std::max_element(reads.begin(), reads.end());
	uint32_t   second = 0;
	for (auto it = reads.begin(); it != reads.end(); ++it) {
		second = it != most ? std::max(second, *it) : second;
	}
	if (*most < 32 || *most < 8 * second) {
		return "1/60 constant of the fixed-step updates not found";
	}
	sites->frame_constant = constants[static_cast<size_t>(most - reads.begin())];
	return {};
}

class Pattern {
public:
	explicit Pattern(std::string_view text) {
		const auto nibble = [](char c) {
			return c <= '9' ? c - '0' : (c | 0x20) - 'a' + 10;
		};
		for (size_t i = 0; i + 1 < text.size(); i++) {
			if (text[i] == ' ') {
				continue;
			}
			m_bytes.push_back(text[i] == '?' ? -1 : nibble(text[i]) * 16 + nibble(text[i + 1]));
			i++;
		}
		while (m_prefix.size() < m_bytes.size() && m_bytes[m_prefix.size()] >= 0) {
			m_prefix.push_back(static_cast<uint8_t>(m_bytes[m_prefix.size()]));
		}
	}

	template <typename F>
	void ForEach(const Image& image, std::span<const CodeRange> code, F&& f) const {
		for (const auto& range: code) {
			image.ForEach(range.begin, range.end, m_prefix, [&](uint64_t p) {
				if (!image.Has(p, m_bytes.size())) {
					return;
				}
				for (size_t i = m_prefix.size(); i < m_bytes.size(); i++) {
					if (m_bytes[i] >= 0 && image.Get<uint8_t>(p + i) != m_bytes[i]) {
						return;
					}
				}
				f(p);
			});
		}
	}

private:
	std::vector<int>     m_bytes;
	std::vector<uint8_t> m_prefix;
};

struct PhysicsSites {
	uint64_t null_query        = 0;
	uint64_t null_query_target = 0;
	std::array<uint64_t, 3> stack_pushes {};
	std::array<uint64_t, 3> substeps {};
	std::array<uint64_t, 3> solvers {};
	uint64_t cloth = 0;
	uint64_t rescue        = 0;
	uint64_t rescue_target = 0;
};

std::string_view FindPhysicsSites(const Image& image, std::span<const CodeRange> code,
                                  PhysicsSites* sites) {
	uint32_t found = 0;
	Pattern("c4 e3 41 40 ff 7f c5 c8 c6 f6 ff c5 f8 2e f7 0f 82 ?? ?? ?? ??")
	    .ForEach(image, code, [&](uint64_t p) {
		    sites->null_query        = p + 11;
		    sites->null_query_target = image.Target(p + 17, p + 21);
		    found++;
	    });
	if (found != 1) {
		return found == 0 ? "QueryPoint sphere test not found"
		                  : "QueryPoint sphere test found twice";
	}

	constexpr std::array<uint8_t, 8>  advance {0x48, 0xc1, 0xe0, 0x02, 0x48, 0x01, 0x45, 0xc0};
	constexpr std::array<uint8_t, 14> pop {0x48, 0x8b, 0x4d, 0xc0, 0x48, 0x8d, 0x85,
	                                       0xc0, 0xfb, 0xff, 0xff, 0x48, 0x39, 0xc1};
	found = 0;
	Pattern("c5 fa 7f 00 44 89 e8").ForEach(image, code, [&](uint64_t p) {
		if (image.Contains(p + 7, p + 0x20, advance) &&
		    image.Contains(p >= 0x1000 ? p - 0x1000 : 0, p, pop)) {
			if (found < sites->stack_pushes.size()) {
				sites->stack_pushes[found] = p;
			}
			found++;
		}
	});
	if (found != sites->stack_pushes.size()) {
		return "tree traversal pushes not found three times";
	}

	constexpr std::array<std::array<uint8_t, 3>, 5> movs_rdi {{
	    {0x48, 0x89, 0xdf},
	    {0x4c, 0x89, 0xe7},
	    {0x4c, 0x89, 0xef},
	    {0x4c, 0x89, 0xf7},
	    {0x4c, 0x89, 0xff},
	}};
	constexpr std::array<uint8_t, 4> vmovss_rbp {0xc5, 0xfa, 0x10, 0x85};
	constexpr std::array<uint8_t, 4> vmovss_store_rbp {0xc5, 0xfa, 0x11, 0x85};
	constexpr uint64_t               unit = 16;
	found                                 = 0;
	for (const auto& range: code) {
		image.ForEach(range.begin, range.end, vmovss_rbp, [&](uint64_t p) {
			if (p < range.begin + 0x35 || !image.Has(p, 9 * unit) ||
			    image.Get<uint8_t>(p - 5) != Call[0]) {
				return;
			}
			const auto call   = p - 5;
			const auto solver = image.Target(call + 1, call + 5);
			const auto disp   = image.Get<uint32_t>(p + 4);
			const auto mov    = std::find_if(movs_rdi.begin(), movs_rdi.end(), [&](const auto& m) {
                return image.Is(p + 8, m);
            });
			if (mov == movs_rdi.end()) {
				return;
			}
			for (uint64_t u = p; u < p + 9 * unit; u += unit) {
				if (!image.Is(u, vmovss_rbp) || image.Get<uint32_t>(u + 4) != disp ||
				    !image.Is(u + 8, *mov) || image.Get<uint8_t>(u + 11) != Call[0] ||
				    image.Target(u + 12, u + 16) != solver) {
					return;
				}
			}
			std::array<uint8_t, 8> store {};
			std::memcpy(store.data(), vmovss_store_rbp.data(), vmovss_store_rbp.size());
			std::memcpy(store.data() + 4, &disp, sizeof(disp));
			bool by_ten = false;
			for (uint64_t q = call - 0x30; q + 8 <= call && !by_ten; q++) {
				by_ten = image.Is(q, std::array<uint8_t, 4> {0xc5, 0xfa, 0x5e, 0x05}) &&
				         image.IsFloat(image.Target(q + 4, q + 8), 10.0f);
			}
			if (!by_ten || !image.Contains(call - 0x30, call, store)) {
				return;
			}
			if (found < sites->substeps.size()) {
				sites->substeps[found] = call;
				sites->solvers[found]  = solver;
			}
			found++;
		});
	}
	if (found != sites->substeps.size()) {
		return "ten-step solvers not found three times";
	}

	found = 0;
	Pattern("55 48 89 e5 41 57 41 56 41 55 41 54 53 48 81 ec 08 08 00 00 4c 8b 35 ?? ?? ?? ?? "
	        "48 89 fb 49 8b 06 48 89 45 d0 80 bf 95 01 00 00 00 0f 84 ?? ?? ?? ?? "
	        "c5 fa 10 83 88 01 00 00")
	    .ForEach(image, code, [&](uint64_t p) {
		    sites->cloth = p;
		    found++;
	    });
	if (found != 1) {
		return found == 0 ? "cloth update not found" : "cloth update found twice";
	}

	found = 0;
	Pattern("80 7b 08 00 74 15 c5 fa 10 45 c0 48 89 df e8 ?? ?? ?? ?? 48 89 df e8 ?? ?? ?? ?? "
	        "49 8d 86 78 07 00 00")
	    .ForEach(image, code, [&](uint64_t p) {
		    sites->rescue        = p + 14;
		    sites->rescue_target = image.Target(p + 15, p + 19);
		    found++;
	    });
	if (found != 1) {
		sites->rescue = 0;
	}
	return {};
}

class Emitter {
public:
	explicit Emitter(uint64_t address): m_address(address) {}

	[[nodiscard]] uint64_t                 Here() const { return m_address + m_code.size(); }
	[[nodiscard]] bool                     Ok() const { return m_ok; }
	[[nodiscard]] std::span<const uint8_t> Code() const { return m_code; }

	void Bytes(std::initializer_list<uint8_t> bytes) { m_code.insert(m_code.end(), bytes); }
	void Rel32(std::initializer_list<uint8_t> opcode, uint64_t target) {
		Bytes(opcode);
		const auto rel = static_cast<int64_t>(target) - static_cast<int64_t>(Here() + 4);
		m_ok           = m_ok && rel >= INT32_MIN && rel <= INT32_MAX;
		const auto value = static_cast<int32_t>(rel);
		const auto bytes = std::bit_cast<std::array<uint8_t, 4>>(value);
		m_code.insert(m_code.end(), bytes.begin(), bytes.end());
	}
	[[nodiscard]] size_t Forward(uint8_t opcode) {
		Bytes({opcode, 0});
		return m_code.size() - 1;
	}
	void Bind(size_t fixup) {
		const auto rel = m_code.size() - (fixup + 1);
		m_ok           = m_ok && rel <= 127;
		m_code[fixup]  = static_cast<uint8_t>(rel);
	}
	void Back(uint8_t opcode, uint64_t target) {
		const auto rel = static_cast<int64_t>(target) - static_cast<int64_t>(Here() + 2);
		m_ok           = m_ok && rel >= -128 && rel < 0;
		Bytes({opcode, static_cast<uint8_t>(static_cast<int8_t>(rel))});
	}
	void FillTo(size_t size) {
		m_ok = m_ok && m_code.size() <= size;
		m_code.resize(std::max(m_code.size(), size), Int3);
	}

private:
	uint64_t             m_address;
	std::vector<uint8_t> m_code;
	bool                 m_ok = true;
};

struct CaveData {
	uint32_t null_queries = 0;
	uint32_t overflows    = 0;
	uint32_t rescues      = 0;
	uint32_t padding      = 0;
	float    per_second   = 60.0f;
	float    per_tenth    = 600.0f;
	float    round_up     = 0.95f;
};
constexpr uint64_t CaveCode = 0x40;
constexpr uint64_t CaveSize = 0x4000;

void EmitStepCount(Emitter* e, uint64_t cave, uint64_t scale) {
	e->Rel32({0xc5, 0xfa, 0x59, 0x0d}, scale);
	e->Rel32({0xc5, 0xf2, 0x58, 0x0d}, cave + offsetof(CaveData, round_up));
	e->Bytes({0xc5, 0xfa, 0x2c, 0xc1});
	e->Bytes({0x83, 0xf8, 0x01, 0x7d, 0x05, 0xb8, 0x01, 0x00, 0x00, 0x00});
	e->Bytes({0x83, 0xf8, 0x04, 0x7e, 0x05, 0xb8, 0x04, 0x00, 0x00, 0x00});
}

void EmitStepLoop(Emitter* e, uint64_t solver, uint8_t calls_per_step) {
	e->Bytes({0x53, 0x41, 0x54, 0x41, 0x55});
	e->Bytes({0x48, 0x83, 0xec, 0x10});
	e->Bytes({0x48, 0x89, 0xfb});
	e->Bytes({0xc5, 0xf2, 0x2a, 0xc8});
	e->Bytes({0xc5, 0xfa, 0x5e, 0xc1});
	e->Bytes({0xc5, 0xfa, 0x11, 0x04, 0x24});
	e->Bytes({0x6b, 0xc0, calls_per_step});
	e->Bytes({0x41, 0x89, 0xc4});
	const auto loop = e->Here();
	e->Bytes({0x48, 0x89, 0xdf});
	e->Bytes({0xc5, 0xfa, 0x10, 0x04, 0x24});
	e->Rel32({0xe8}, solver);
	e->Bytes({0x41, 0xff, 0xcc});
	e->Back(0x75, loop);
	e->Bytes({0x48, 0x83, 0xc4, 0x10});
	e->Bytes({0x41, 0x5d, 0x41, 0x5c, 0x5b, Ret});
}

bool WriteCode(uint64_t address, std::span<const uint8_t> code) {
	std::memcpy(reinterpret_cast<void*>(address), code.data(), code.size());
	return Common::VirtualMemory::FlushInstructionCache(address, code.size());
}

uint64_t PatchPhysics(uint64_t base, uint64_t mapped_size, const PhysicsSites& sites) {
	const auto cave = Libs::LibKernel::Memory::AllocateRuntimeMemory(
	    Common::AlignUp(base + mapped_size, CaveSize), CaveSize,
	    Common::VirtualMemory::Mode::ExecuteReadWrite, "frame_step_physics_fixes", false);
	if (cave == 0) {
		return 0;
	}
	const auto null_queries = cave + offsetof(CaveData, null_queries);
	const auto overflows    = cave + offsetof(CaveData, overflows);
	const auto rescues      = cave + offsetof(CaveData, rescues);

	Emitter                                   e(cave + CaveCode);
	std::vector<std::pair<uint64_t, Emitter>> site_code;

	{
		const auto site  = base + sites.null_query;
		const auto entry = e.Here();
		e.Bytes({0xc5, 0xf8, 0x2e, 0xf7});
		e.Rel32({0x0f, 0x83}, site + 10);
		e.Bytes({0x4d, 0x85, 0xff});
		e.Rel32({0x0f, 0x85}, base + sites.null_query_target);
		e.Rel32({0xf0, 0xff, 0x05}, null_queries);
		e.Rel32({0xe9}, site + 10);
		Emitter s(site);
		s.Rel32({0xe9}, entry);
		s.FillTo(10);
		site_code.emplace_back(site, std::move(s));
	}

	for (const auto offset: sites.stack_pushes) {
		const auto site  = base + offset;
		const auto entry = e.Here();
		e.Bytes({0x48, 0x8d, 0x40, 0x50});
		e.Bytes({0x48, 0x39, 0xe8});
		e.Bytes({0x48, 0x8d, 0x40, 0xb0});
		const auto full = e.Forward(0x77);
		e.Bytes({0xc5, 0xfa, 0x7f, 0x00});
		e.Bytes({0x44, 0x89, 0xe8});
		e.Rel32({0xe9}, site + 7);
		e.Bind(full);
		e.Bytes({0x48, 0x8d, 0x85, 0xc0, 0xfb, 0xff, 0xff});
		e.Bytes({0x48, 0x89, 0x45, 0xc0});
		e.Bytes({0x31, 0xc0});
		e.Rel32({0xf0, 0xff, 0x05}, overflows);
		e.Rel32({0xe9}, site + 7);
		Emitter s(site);
		s.Rel32({0xe9}, entry);
		s.FillTo(7);
		site_code.emplace_back(site, std::move(s));
	}

	for (size_t i = 0; i < sites.substeps.size(); i++) {
		const auto site  = base + sites.substeps[i];
		const auto entry = e.Here();
		EmitStepCount(&e, cave, cave + offsetof(CaveData, per_tenth));
		EmitStepLoop(&e, base + sites.solvers[i], 10);
		Emitter s(site);
		s.Rel32({0xe8}, entry);
		s.Rel32({0xe9}, site + 10 * 5 + 9 * 11);
		s.FillTo(10 * 5 + 9 * 11);
		site_code.emplace_back(site, std::move(s));
	}

	{
		const auto site     = base + sites.cloth;
		const auto original = e.Here();
		e.Bytes({0x55, 0x48, 0x89, 0xe5, 0x41, 0x57});
		e.Rel32({0xe9}, site + 6);
		const auto entry = e.Here();
		EmitStepCount(&e, cave, cave + offsetof(CaveData, per_second));
		e.Bytes({0x83, 0xf8, 0x01});
		e.Back(0x7e, original);
		EmitStepLoop(&e, original, 1);
		Emitter s(site);
		s.Rel32({0xe9}, entry);
		s.FillTo(6);
		site_code.emplace_back(site, std::move(s));
	}

	if (sites.rescue != 0) {
		const auto site  = base + sites.rescue;
		const auto entry = e.Here();
		e.Rel32({0xf0, 0xff, 0x05}, rescues);
		e.Rel32({0xe9}, base + sites.rescue_target);
		Emitter s(site);
		s.Rel32({0xe8}, entry);
		site_code.emplace_back(site, std::move(s));
	}

	bool ok = e.Ok() && CaveCode + e.Code().size() <= CaveSize;
	for (const auto& [site, s]: site_code) {
		ok = ok && s.Ok();
	}
	if (!ok) {
		Libs::LibKernel::Memory::FreeGuestMemory(cave, CaveSize);
		return 0;
	}
	const CaveData data;
	std::memcpy(reinterpret_cast<void*>(cave), &data, sizeof(data));
	if (!WriteCode(cave + CaveCode, e.Code())) {
		Libs::LibKernel::Memory::FreeGuestMemory(cave, CaveSize);
		return 0;
	}
	for (const auto& [site, s]: site_code) {
		WriteCode(site, s.Code());
	}
	return cave;
}

bool IsOwnFrameRate(double rate, float step, uint64_t period) {
	return rate >= 20.0 && rate <= 240.0 &&
	       std::fabs(static_cast<double>(step) * rate - 1.0) < 1e-4 &&
	       period == static_cast<uint64_t>(1000000.0 / rate);
}

void WriteFrame(State* state, double frame) {
	const double rate  = 1.0 / frame;
	const auto&  sites = state->sites;
	Write<double>(state->base + sites.rate, rate);
	Write<float>(state->base + sites.step, static_cast<float>(1.0 / rate));
	Write<uint64_t>(state->base + sites.period, static_cast<uint64_t>(1000000.0 / rate));
	Write<float>(state->base + sites.frame_constant,
	             static_cast<float>(OwnFrameConstant * frame * state->own_rate));
	state->written_rate = rate;
}

void WriteOwnFrame(State* state) {
	const auto& sites = state->sites;
	Write<double>(state->base + sites.rate, state->own_rate);
	Write<float>(state->base + sites.step, state->own_step);
	Write<uint64_t>(state->base + sites.period, state->own_period);
	Write<float>(state->base + sites.frame_constant, OwnFrameConstant);
	state->written_rate = state->own_rate;
}

}

void PatchMainProgram(uint64_t base_vaddr, uint64_t size, uint64_t mapped_size,
                      std::span<const CodeRange> code, bool cheat_file) noexcept {
	const auto id    = ParamString("TITLE_ID");
	const auto title = std::find_if(Titles.begin(), Titles.end(),
	                                [&](const Title& t) { return t.id == id; });
	if (title == Titles.end()) {
		return;
	}
	const auto version = ParamString("APP_VER");
	Sites      sites;
	const auto failure = base_vaddr == 0
	                         ? std::string_view("no executable")
	                         : FindSites(Image(base_vaddr, size), code, title->pacing, &sites);
	if (!failure.empty()) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "[HACK] {} frame time: not applied to {} {} ({}{})\n", title->name, id, version,
		    failure,
		    cheat_file ? "; a cheat file is in use, and its frame-time options change this code"
		               : ""));
		return;
	}

	if (sites.pacing_site != 0) {
		constexpr std::array<uint8_t, 5> nop5 {0x0f, 0x1f, 0x44, 0x00, 0x00};
		const auto                       site = base_vaddr + sites.pacing_site;
		const int32_t disp = static_cast<int32_t>(static_cast<int64_t>(sites.pacing_frame) -
		                                          static_cast<int64_t>(sites.pacing_site + 8));
		std::array<uint8_t, 13> pacing {};
		std::memcpy(pacing.data(), VmovsdLoad.data(), VmovsdLoad.size());
		std::memcpy(pacing.data() + 4, &disp, sizeof(disp));
		std::memcpy(pacing.data() + 8, nop5.data(), nop5.size());
		std::array<uint8_t, 13> original {};
		std::memcpy(original.data(), reinterpret_cast<const void*>(site), original.size());
		Write<double>(base_vaddr + sites.pacing_frame, (1.0 / 60.0) * 1000.0);
		std::memcpy(reinterpret_cast<void*>(site), pacing.data(), pacing.size());
		if (!Common::VirtualMemory::FlushInstructionCache(site, pacing.size())) {
			std::memcpy(reinterpret_cast<void*>(site), original.data(), original.size());
			Log::WriteToConsoleAndLog(
			    fmt::format("[HACK] {} frame time: not applied to {} {} (code write failed)\n",
			                title->name, id, version));
			return;
		}
	}

	auto& state     = GetState();
	state.base      = base_vaddr;
	state.sites     = sites;
	state.frequency = 1e9;
	state.name      = title->name;

	PhysicsSites physics;
	auto         physics_failure = FindPhysicsSites(Image(base_vaddr, size), code, &physics);
	if (physics_failure.empty()) {
		state.cave = PatchPhysics(base_vaddr, mapped_size, physics);
		if (state.cave == 0) {
			physics_failure = "code page not written";
		}
	}
	if (physics_failure.empty()) {
		state.max_step = MaxStep;
		Log::WriteToConsoleAndLog(fmt::format(
		    "[HACK] {} physics fixes for stretched frames ({} {}: QueryPoint {:#x}, traversals "
		    "{:#x} {:#x} {:#x}, solvers {:#x} {:#x} {:#x}, cloth {:#x}, bot rescue {:#x}, code at "
		    "{:#x})\n",
		    title->name, id, version, physics.null_query, physics.stack_pushes[0],
		    physics.stack_pushes[1], physics.stack_pushes[2], physics.substeps[0],
		    physics.substeps[1], physics.substeps[2], physics.cloth, physics.rescue, state.cave));
	} else {
		state.max_step = MaxStepUnfixed;
		Log::WriteToConsoleAndLog(
		    fmt::format("[HACK] {} physics fixes for stretched frames: not applied to {} {} ({})\n",
		                title->name, id, version, physics_failure));
	}

	state.active = true;
	Log::WriteToConsoleAndLog(fmt::format(
	    "[HACK] {} frame time ({} {}: rate {:#x}, constant {:#x}, pacing {:#x}): below the game's "
	    "frame rate its frame time follows the presented frames (at most {:.0f} ms), so that it "
	    "does not run in slow motion\n",
	    title->name, id, version, sites.rate, sites.frame_constant, sites.pacing_site,
	    state.max_step * 1000.0));
}

void OnPresentedFrame() noexcept {
	auto& state = GetState();
	if (!state.active) {
		return;
	}
	const auto counter = Common::SteadyNs();
	const auto last    = std::exchange(state.last_counter, counter);

	const auto& sites = state.sites;
	const auto  rate  = Read<double>(state.base + sites.rate);
	if (rate != state.written_rate) {
		const auto step   = Read<float>(state.base + sites.step);
		const auto period = Read<uint64_t>(state.base + sites.period);
		if (!IsOwnFrameRate(rate, step, period)) {
			return;
		}
		if (rate != state.own_rate) {
			Log::WriteToConsoleAndLog(fmt::format("[HACK] {} frame time: the game runs at {:.2f} fps\n",
			                                      state.name, rate));
		}
		state.own_rate     = rate;
		state.own_step     = step;
		state.own_period   = period;
		state.written_rate = rate;
		state.average      = 0.0;
		state.scaled       = false;
		Write<float>(state.base + sites.frame_constant, OwnFrameConstant);
		if (sites.pacing_frame != 0) {
			Write<double>(state.base + sites.pacing_frame, (1.0 / rate) * 1000.0);
		}
	}
	if (last == 0 || state.own_rate == 0.0) {
		return;
	}

	if (state.cave != 0) {
		const auto null_queries = Read<uint32_t>(state.cave + offsetof(CaveData, null_queries));
		if (null_queries != state.null_queries) {
			state.null_queries = null_queries;
			Common::HackNotice::Report(
			    "Physics query without a body skipped", "FrameStepHack::QueryPoint", 0, 0,
			    fmt::format("{}: {} Havok world queries of a contact without a body (a character "
			                "at a NaN position) skipped",
			                state.name, null_queries));
		}
		const auto overflows = Read<uint32_t>(state.cave + offsetof(CaveData, overflows));
		if (overflows != state.overflows) {
			state.overflows = overflows;
			Common::HackNotice::Report(
			    "Physics tree traversal ended by a full stack", "FrameStepHack::Traversal", 0, 0,
			    fmt::format("{}: {} Havok tree traversals ended at 256 pending nodes", state.name,
			                overflows));
		}
		const auto rescues = Read<uint32_t>(state.cave + offsetof(CaveData, rescues));
		if (rescues != state.rescues) {
			state.rescues     = rescues;
			state.rescue_hold = RescueHoldFrames;
			Common::HackNotice::Report(
			    "Own frame time during a bot rescue", "FrameStepHack::Rescue", 0, 0,
			    fmt::format("{}: the bot-rescue update runs at the game's own frame time",
			                state.name));
		}
	}
	const bool rescue = state.rescue_hold != 0;
	if (rescue) {
		state.rescue_hold--;
	}

	const double own_frame = 1.0 / state.own_rate;
	const double frame =
	    std::min(static_cast<double>(counter - last) / state.frequency, state.max_step);
	state.average =
	    state.average == 0.0 ? frame : state.average + (frame - state.average) / AverageFrames;
	if (rescue || state.average <= own_frame * FullRateSlack) {
		if (state.scaled) {
			WriteOwnFrame(&state);
			state.scaled = false;
		}
		return;
	}

	WriteFrame(&state, state.average);
	state.scaled = true;
	Common::HackNotice::Report("Frame time follows the presented frames",
	                           "FrameStepHack::OnPresentedFrame", 0, 0,
	                           fmt::format("{}: frame {:.1f} ms instead of {:.1f} ms", state.name,
	                                       state.average * 1000.0, own_frame * 1000.0));
}

}
