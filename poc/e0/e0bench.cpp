#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#define CHECK(x)                                                                                                       \
	do {                                                                                                               \
		VkResult r_ = (x);                                                                                             \
		if (r_ != VK_SUCCESS) {                                                                                        \
			fprintf(stderr, "%s:%d: %s = %d\n", __FILE__, __LINE__, #x, int(r_));                                      \
			exit(2);                                                                                                   \
		}                                                                                                              \
	} while (0)

static constexpr uint32_t KindCount       = 61;
static constexpr uint32_t KindBuffers     = 0;
static constexpr uint32_t KindSamplers    = 51;
static constexpr uint32_t KindGds         = 52;
static constexpr uint32_t KindPagetable   = 53;
static constexpr uint32_t KindFault       = 54;
static constexpr uint32_t KindFlat        = 55;
static constexpr uint32_t KindShaderData  = 56;
static constexpr uint32_t KindShared      = 57;
static constexpr uint32_t KindUniform     = 58;
static constexpr uint32_t KindMipStats    = 59;
static constexpr uint64_t LowerSize       = 1ull << 40;
static constexpr uint64_t ExtendedBase    = 0x080000000000ull;
static constexpr uint64_t ExtendedSize    = 512ull << 30;
static constexpr uint64_t PageBits        = 14;
static constexpr uint64_t NumPages        = (LowerSize + ExtendedSize) >> PageBits;
static constexpr uint64_t PagetableBytes  = (NumPages + 1) * 8;
static constexpr uint64_t FaultBytes      = 3 * (NumPages / 8) + 256 * 16 * 4;

static VkInstance                       g_inst;
static VkPhysicalDevice                 g_phys;
static VkDevice                         g_dev;
static VkQueue                          g_queue;
static uint32_t                         g_family = 0;
static VkPhysicalDeviceMemoryProperties g_memprops;
static VkPhysicalDeviceProperties       g_props;
static VkCommandPool                    g_pool;
static VkFence                          g_fence;

struct Buf {
	VkBuffer       buffer = VK_NULL_HANDLE;
	VkDeviceMemory memory = VK_NULL_HANDLE;
	uint64_t       size   = 0;
	void*          map    = nullptr;
	uint64_t       bda    = 0;
};

static uint32_t MemType(uint32_t bits, VkMemoryPropertyFlags want) {
	for (uint32_t i = 0; i < g_memprops.memoryTypeCount; i++) {
		if ((bits & (1u << i)) && (g_memprops.memoryTypes[i].propertyFlags & want) == want) return i;
	}
	fprintf(stderr, "no memory type\n");
	exit(2);
}

static Buf MakeBuf(uint64_t size, VkBufferUsageFlags usage, bool host) {
	Buf b;
	b.size = std::max<uint64_t>(size, 16);
	VkBufferCreateInfo bi {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	bi.size  = b.size;
	bi.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	CHECK(vkCreateBuffer(g_dev, &bi, nullptr, &b.buffer));
	VkMemoryRequirements mr;
	vkGetBufferMemoryRequirements(g_dev, b.buffer, &mr);
	VkMemoryAllocateFlagsInfo fi {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
	fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
	VkMemoryAllocateInfo ai {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &fi};
	ai.allocationSize  = mr.size;
	ai.memoryTypeIndex = MemType(mr.memoryTypeBits, host ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
	                                                     : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	CHECK(vkAllocateMemory(g_dev, &ai, nullptr, &b.memory));
	CHECK(vkBindBufferMemory(g_dev, b.buffer, b.memory, 0));
	if (host) CHECK(vkMapMemory(g_dev, b.memory, 0, VK_WHOLE_SIZE, 0, &b.map));
	VkBufferDeviceAddressInfo di {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
	di.buffer = b.buffer;
	b.bda     = vkGetBufferDeviceAddress(g_dev, &di);
	return b;
}

static VkCommandBuffer NewCmd() {
	VkCommandBufferAllocateInfo cai {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
	cai.commandPool        = g_pool;
	cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cai.commandBufferCount = 1;
	VkCommandBuffer cb;
	CHECK(vkAllocateCommandBuffers(g_dev, &cai, &cb));
	return cb;
}

static void Begin(VkCommandBuffer c) {
	CHECK(vkResetCommandBuffer(c, 0));
	VkCommandBufferBeginInfo bi {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	CHECK(vkBeginCommandBuffer(c, &bi));
}

static void Submit(VkCommandBuffer c) {
	CHECK(vkEndCommandBuffer(c));
	VkSubmitInfo si {VK_STRUCTURE_TYPE_SUBMIT_INFO};
	si.commandBufferCount = 1;
	si.pCommandBuffers    = &c;
	CHECK(vkQueueSubmit(g_queue, 1, &si, g_fence));
	CHECK(vkWaitForFences(g_dev, 1, &g_fence, VK_TRUE, UINT64_MAX));
	CHECK(vkResetFences(g_dev, 1, &g_fence));
}

struct Image {
	VkImage        image  = VK_NULL_HANDLE;
	VkDeviceMemory memory = VK_NULL_HANDLE;
	VkImageView    view   = VK_NULL_HANDLE;
	VkFormat       format = VK_FORMAT_UNDEFINED;
	uint32_t       w = 1, h = 1, d = 1, layers = 1;
	uint32_t       texel = 4;
	bool           depth = false;
};

struct ImageType {
	uint32_t sampled_kind = 0;
	uint32_t dim          = 1;
	uint32_t depth        = 0;
	uint32_t arrayed      = 0;
	uint32_t ms           = 0;
	uint32_t sampled      = 1;
	uint32_t format       = 0;
};

struct Binding {
	uint32_t         set = 0, binding = 0, count = 1;
	VkDescriptorType type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	ImageType        image;
	bool             is_image = false;
};

struct Interface {
	uint32_t location = 0, components = 4, kind = 0;
	bool     output = false;
};

static std::vector<uint32_t> LoadSpv(const std::string& path) {
	std::ifstream f(path, std::ios::binary);
	std::vector<char> bytes {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
	std::vector<uint32_t> w(bytes.size() / 4);
	memcpy(w.data(), bytes.data(), w.size() * 4);
	return w;
}

static std::vector<Binding> Reflect(const std::vector<uint32_t>& w, uint32_t& model, std::vector<Interface>* interface = nullptr) {
	const uint32_t bound = w[3];
	std::vector<uint32_t> set(bound), bnd(bound), ptr_sc(bound), ptr_ty(bound), var_ty(bound), op_of(bound), arr_el(bound),
	    arr_len(bound), cval(bound), block(bound), bufblock(bound), int_sign(bound), int_width(bound), samp_img(bound);
	std::vector<uint8_t> has_set(bound), has_bnd(bound), is_var(bound), has_loc(bound);
	std::vector<uint32_t> loc(bound), vec_el(bound), vec_n(bound), var_sc(bound);
	std::vector<ImageType> img(bound);
	model = UINT32_MAX;
	for (size_t i = 5; i < w.size();) {
		const uint32_t wc = w[i] >> 16, op = w[i] & 0xffff;
		if (wc == 0) break;
		const uint32_t* a = &w[i + 1];
		if (op == 15 && model == UINT32_MAX) model = a[0];
		else if (op == 71 && a[0] < bound) {
			if (a[1] == 34) { set[a[0]] = a[2]; has_set[a[0]] = 1; }
			else if (a[1] == 33) { bnd[a[0]] = a[2]; has_bnd[a[0]] = 1; }
			else if (a[1] == 30) { loc[a[0]] = a[2]; has_loc[a[0]] = 1; }
			else if (a[1] == 2) block[a[0]] = 1;
			else if (a[1] == 3) bufblock[a[0]] = 1;
		} else if (op == 43 && wc >= 4 && a[1] < bound) cval[a[1]] = a[2];
		else if (op == 21 && a[0] < bound) { op_of[a[0]] = 21; int_width[a[0]] = a[1]; int_sign[a[0]] = a[2]; }
		else if (op == 22 && a[0] < bound) op_of[a[0]] = 22;
		else if (op == 23 && a[0] < bound) { op_of[a[0]] = 23; vec_el[a[0]] = a[1]; vec_n[a[0]] = a[2]; }
		else if (op == 32 && a[0] < bound) { op_of[a[0]] = 32; ptr_sc[a[0]] = a[1]; ptr_ty[a[0]] = a[2]; }
		else if (op == 28 && a[0] < bound) { op_of[a[0]] = 28; arr_el[a[0]] = a[1]; arr_len[a[0]] = a[2] < bound ? cval[a[2]] : 1; }
		else if (op == 29 && a[0] < bound) { op_of[a[0]] = 29; arr_el[a[0]] = a[1]; arr_len[a[0]] = 0; }
		else if (op == 25 && a[0] < bound) {
			op_of[a[0]] = 25;
			auto& t = img[a[0]];
			t.sampled_kind = a[1] < bound && op_of[a[1]] == 21 ? (int_sign[a[1]] ? 2u : 1u) : 0u;
			t.dim = a[2]; t.depth = a[3]; t.arrayed = a[4]; t.ms = a[5]; t.sampled = a[6]; t.format = a[7];
		} else if (op == 27 && a[0] < bound) { op_of[a[0]] = 27; samp_img[a[0]] = a[1]; }
		else if ((op == 26 || op == 5341 || op == 30) && a[0] < bound) op_of[a[0]] = op;
		else if (op == 59 && a[1] < bound) { is_var[a[1]] = 1; var_ty[a[1]] = a[0]; var_sc[a[1]] = a[2]; }
		i += wc;
	}
	std::vector<Binding> out;
	for (uint32_t id = 1; id < bound && interface != nullptr; id++) {
		if (!is_var[id] || !has_loc[id] || (var_sc[id] != 1 && var_sc[id] != 3)) continue;
		const uint32_t p = var_ty[id];
		if (op_of[p] != 32) continue;
		uint32_t t = ptr_ty[p];
		Interface x;
		x.location = loc[id];
		x.output   = var_sc[id] == 3;
		x.components = 1;
		if (op_of[t] == 23) { x.components = vec_n[t]; t = vec_el[t]; }
		x.kind = op_of[t] == 21 ? (int_sign[t] ? 2u : 1u) : 0u;
		interface->push_back(x);
	}
	for (uint32_t id = 1; id < bound; id++) {
		if (!is_var[id] || !has_set[id] || !has_bnd[id]) continue;
		const uint32_t p = var_ty[id];
		if (op_of[p] != 32) continue;
		const uint32_t sc = ptr_sc[p];
		uint32_t t = ptr_ty[p], count = 1;
		bool runtime = false;
		while (op_of[t] == 28 || op_of[t] == 29) {
			if (op_of[t] == 29) runtime = true;
			count *= arr_len[t] ? arr_len[t] : 1;
			t = arr_el[t];
		}
		Binding b;
		b.set = set[id]; b.binding = bnd[id]; b.count = runtime ? 0 : count;
		if (sc == 12) b.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		else if (sc == 2) b.type = bufblock[t] ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		else if (sc == 0) {
			if (op_of[t] == 26) b.type = VK_DESCRIPTOR_TYPE_SAMPLER;
			else if (op_of[t] == 27) { b.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; b.image = img[samp_img[t]]; b.is_image = true; }
			else if (op_of[t] == 5341) b.type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
			else if (op_of[t] == 25) {
				b.image = img[t]; b.is_image = true;
				if (img[t].dim == 5) b.type = img[t].sampled == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
				else b.type = img[t].sampled == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
			} else continue;
		} else continue;
		auto found = std::find_if(out.begin(), out.end(), [&](const Binding& o) { return o.set == b.set && o.binding == b.binding; });
		if (found == out.end()) out.push_back(b);
		else if (b.count > found->count) found->count = b.count;
	}
	return out;
}

struct RecipeBuf {
	uint32_t index = 0;
	uint64_t address = 0, size = 0;
	int      uniform = -1;
	int      written = 0;
	uint32_t stride  = 0;
};

struct RecipeImg {
	uint32_t binding = 0, index = 0, count = 1, type = 0, w = 1, h = 1, d = 1;
	int      written = 0, storage = 0, cube = 0;
	uint32_t table = 0;
};

struct Form {
	std::string            name, spv;
	int                    push = 0;
	uint32_t               start = 0;
	std::vector<uint32_t>  sdata;
	uint32_t               memoff = 0, memsize = 0, stride = 0, tblock = 0, stride_count = 0;
	std::vector<uint32_t>  counts;
	int                    dma = 0;
	std::vector<uint32_t>  flat;
	std::vector<RecipeBuf> bufs;
	struct Root { uint32_t k, uniform; uint64_t address; };
	std::vector<Root> roots;
	std::vector<RecipeImg> imgs;
	bool                   gpu = false;
};

struct Region {
	uint64_t    start = 0, size = 0;
	std::string file;
	uint64_t    pristine_offset = 0;
};

struct Recipe {
	std::string         name, stage = "cs";
	uint64_t            hash = 0;
	uint32_t            local[3] {1, 1, 1}, groups[3] {1, 1, 1};
	std::vector<Region> regions;
	std::vector<Form>   forms;
	std::vector<std::pair<uint64_t, uint64_t>> written;
};

static bool ParseRecipe(const std::string& path, const std::string& mem_root, Recipe& r) {
	std::ifstream f(path);
	if (!f) return false;
	const auto dir = path.substr(0, path.find_last_of('/') + 1);
	std::string line;
	Form* form = nullptr;
	while (std::getline(f, line)) {
		std::istringstream in(line);
		std::string key;
		in >> key;
		if (key == "name") in >> r.name;
		else if (key == "hash") in >> std::hex >> r.hash;
		else if (key == "stage") in >> r.stage;
		else if (key == "local") in >> r.local[0] >> r.local[1] >> r.local[2];
		else if (key == "groups") in >> r.groups[0] >> r.groups[1] >> r.groups[2];
		else if (key == "mem") {
			Region g;
			in >> std::hex >> g.start >> g.size >> g.file;
			if (!mem_root.empty() && g.file != "-") g.file = mem_root + "/" + g.file.substr(g.file.find_last_of('/') + 1);
			r.regions.push_back(g);
		} else if (key == "form") {
			r.forms.emplace_back();
			form = &r.forms.back();
			in >> form->name >> form->spv;
			form->spv = dir + form->spv;
		} else if (key == "sdata" && form) {
			size_t n = 0;
			in >> form->push >> form->start >> n;
			form->sdata.resize(n);
			for (auto& v: form->sdata) in >> std::hex >> v;
		} else if (key == "layout" && form) {
			in >> form->memoff >> form->memsize >> form->stride >> form->tblock >> form->stride_count;
		} else if (key == "counts" && form) {
			uint32_t v;
			while (in >> v) form->counts.push_back(v);
		} else if (key == "flags" && form) {
			std::string k;
			int v;
			while (in >> k >> v) {
				if (k == "dma") form->dma = v;
				if (k == "gpu") form->gpu = v != 0;
			}
		} else if (key == "flat" && form) {
			size_t n = 0;
			in >> n;
			form->flat.resize(n);
			for (auto& v: form->flat) in >> std::hex >> v;
		} else if (key == "buf" && form) {
			RecipeBuf b;
			in >> b.index >> std::hex >> b.address >> b.size >> std::dec >> b.uniform >> b.written >> b.stride;
			form->bufs.push_back(b);
		} else if (key == "root" && form) {
			Form::Root root {};
			in >> root.k >> root.uniform >> std::hex >> root.address;
			form->roots.push_back(root);
		} else if (key == "img" && form) {
			RecipeImg m;
			in >> m.binding >> m.index >> m.count >> m.type >> m.w >> m.h >> m.d >> m.written >> m.storage >> m.cube >> m.table;
			form->imgs.push_back(m);
		} else if (key == "written") {
			uint64_t a, s;
			in >> std::hex >> a >> s;
			r.written.emplace_back(a, s);
		}
	}
	return !r.forms.empty();
}

static VkFormat StorageFormat(uint32_t f) {
	switch (f) {
		case 1: return VK_FORMAT_R32G32B32A32_SFLOAT;
		case 2: return VK_FORMAT_R16G16B16A16_SFLOAT;
		case 3: return VK_FORMAT_R32_SFLOAT;
		case 4: return VK_FORMAT_R8G8B8A8_UNORM;
		case 5: return VK_FORMAT_R8G8B8A8_SNORM;
		case 6: return VK_FORMAT_R32G32_SFLOAT;
		case 7: return VK_FORMAT_R16G16_SFLOAT;
		case 8: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
		case 9: return VK_FORMAT_R16_SFLOAT;
		case 10: return VK_FORMAT_R16G16B16A16_UNORM;
		case 11: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
		case 12: return VK_FORMAT_R16G16_UNORM;
		case 13: return VK_FORMAT_R8G8_UNORM;
		case 14: return VK_FORMAT_R16_UNORM;
		case 15: return VK_FORMAT_R8_UNORM;
		case 21: return VK_FORMAT_R32G32B32A32_SINT;
		case 22: return VK_FORMAT_R16G16B16A16_SINT;
		case 23: return VK_FORMAT_R8G8B8A8_SINT;
		case 24: return VK_FORMAT_R32_SINT;
		case 30: return VK_FORMAT_R32G32B32A32_UINT;
		case 31: return VK_FORMAT_R16G16B16A16_UINT;
		case 32: return VK_FORMAT_R8G8B8A8_UINT;
		case 33: return VK_FORMAT_R32_UINT;
		case 34: return VK_FORMAT_A2B10G10R10_UINT_PACK32;
		case 35: return VK_FORMAT_R32G32_UINT;
		case 36: return VK_FORMAT_R16G16_UINT;
		case 37: return VK_FORMAT_R8G8_UINT;
		case 38: return VK_FORMAT_R16_UINT;
		case 39: return VK_FORMAT_R8_UINT;
		case 40: return VK_FORMAT_R64_UINT;
		case 41: return VK_FORMAT_R64_SINT;
		default: return VK_FORMAT_UNDEFINED;
	}
}

static uint32_t TexelBytes(VkFormat f) {
	switch (f) {
		case VK_FORMAT_R32G32B32A32_SFLOAT: case VK_FORMAT_R32G32B32A32_SINT: case VK_FORMAT_R32G32B32A32_UINT: return 16;
		case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R16G16B16A16_UNORM: case VK_FORMAT_R16G16B16A16_SINT:
		case VK_FORMAT_R16G16B16A16_UINT: case VK_FORMAT_R32G32_SFLOAT: case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R64_UINT:
		case VK_FORMAT_R64_SINT: return 8;
		case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_R16G16_UNORM: case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R8G8B8A8_UNORM:
		case VK_FORMAT_R8G8B8A8_SNORM: case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_R32_SFLOAT:
		case VK_FORMAT_R32_SINT: case VK_FORMAT_R32_UINT: case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
		case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2B10G10R10_UINT_PACK32: case VK_FORMAT_D32_SFLOAT: return 4;
		case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R16_UNORM: case VK_FORMAT_R16_UINT: case VK_FORMAT_R8G8_UNORM:
		case VK_FORMAT_R8G8_UINT: return 2;
		default: return 1;
	}
}

static uint32_t g_seed = 0x12345u;
static uint32_t Rnd() {
	g_seed = g_seed * 1664525u + 1013904223u;
	return g_seed;
}

static void FillTexels(uint8_t* out, uint64_t bytes, VkFormat f) {
	const bool f16 = f == VK_FORMAT_R16G16B16A16_SFLOAT || f == VK_FORMAT_R16G16_SFLOAT || f == VK_FORMAT_R16_SFLOAT;
	const bool f32 = f == VK_FORMAT_R32G32B32A32_SFLOAT || f == VK_FORMAT_R32G32_SFLOAT || f == VK_FORMAT_R32_SFLOAT;
	if (f16) {
		for (uint64_t i = 0; i + 2 <= bytes; i += 2) {
			const uint16_t v = uint16_t(0x3000u + (Rnd() >> 22));
			memcpy(out + i, &v, 2);
		}
	} else if (f32) {
		for (uint64_t i = 0; i + 4 <= bytes; i += 4) {
			const float v = float(Rnd() >> 8) / 16777216.0f;
			memcpy(out + i, &v, 4);
		}
	} else if (f == VK_FORMAT_R32G32B32A32_UINT || f == VK_FORMAT_R32_UINT || f == VK_FORMAT_R32G32_UINT ||
	           f == VK_FORMAT_R32G32B32A32_SINT || f == VK_FORMAT_R32_SINT) {
		for (uint64_t i = 0; i + 4 <= bytes; i += 4) {
			const uint32_t v = Rnd() >> 26;
			memcpy(out + i, &v, 4);
		}
	} else {
		for (uint64_t i = 0; i < bytes; i++) out[i] = uint8_t(Rnd() >> 24);
	}
}

static Image MakeImage(const ImageType& type, uint32_t w, uint32_t h, uint32_t d, VkCommandBuffer cb, Buf& staging) {
	Image im;
	const bool cube = type.dim == 3;
	const bool is3d = type.dim == 2;
	im.depth = type.depth == 1 && type.sampled != 2;
	im.format = type.sampled == 2 ? StorageFormat(type.format) : VK_FORMAT_UNDEFINED;
	if (im.format == VK_FORMAT_UNDEFINED) {
		im.format = im.depth ? VK_FORMAT_D32_SFLOAT
		            : type.sampled_kind == 0 ? VK_FORMAT_R16G16B16A16_SFLOAT
		            : type.sampled_kind == 1 ? VK_FORMAT_R32G32B32A32_UINT : VK_FORMAT_R32G32B32A32_SINT;
	}
	im.texel  = TexelBytes(im.format);
	im.w      = std::clamp(w, 1u, 4096u);
	im.h      = type.dim == 0 ? 1u : std::clamp(h, 1u, 4096u);
	if (cube) im.h = im.w = std::min(im.w, 1024u);
	im.d      = is3d ? std::clamp(d, 1u, 256u) : 1u;
	im.layers = cube ? 6u * (type.arrayed ? std::clamp(d, 1u, 4u) : 1u) : (type.arrayed ? std::clamp(d, 1u, 64u) : 1u);
	if (is3d) { im.w = std::min(im.w, 256u); im.h = std::min(im.h, 256u); }
	VkImageCreateInfo ci {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
	ci.flags       = cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
	ci.imageType   = is3d ? VK_IMAGE_TYPE_3D : type.dim == 0 ? VK_IMAGE_TYPE_1D : VK_IMAGE_TYPE_2D;
	ci.format      = im.format;
	ci.extent      = {im.w, im.h, im.d};
	ci.mipLevels   = 1;
	ci.arrayLayers = im.layers;
	ci.samples     = type.ms ? VK_SAMPLE_COUNT_4_BIT : VK_SAMPLE_COUNT_1_BIT;
	ci.tiling      = VK_IMAGE_TILING_OPTIMAL;
	ci.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
	                 (type.sampled == 2 ? VK_IMAGE_USAGE_STORAGE_BIT : 0) |
	                 (type.ms ? (im.depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) : 0);
	CHECK(vkCreateImage(g_dev, &ci, nullptr, &im.image));
	VkMemoryRequirements mr;
	vkGetImageMemoryRequirements(g_dev, im.image, &mr);
	VkMemoryAllocateInfo ai {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	ai.allocationSize  = mr.size;
	ai.memoryTypeIndex = MemType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	CHECK(vkAllocateMemory(g_dev, &ai, nullptr, &im.memory));
	CHECK(vkBindImageMemory(g_dev, im.image, im.memory, 0));
	VkImageViewCreateInfo vi {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
	vi.image    = im.image;
	vi.viewType = is3d ? VK_IMAGE_VIEW_TYPE_3D
	              : cube ? (type.arrayed ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE)
	              : type.dim == 0 ? (type.arrayed ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D)
	              : (type.arrayed ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D);
	vi.format   = im.format;
	vi.subresourceRange = {VkImageAspectFlags(im.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 1, 0, im.layers};
	CHECK(vkCreateImageView(g_dev, &vi, nullptr, &im.view));
	const uint64_t bytes = uint64_t(im.w) * im.h * im.d * im.layers * im.texel;
	Begin(cb);
	VkImageMemoryBarrier bar {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
	bar.srcAccessMask = 0;
	bar.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	bar.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	bar.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	bar.srcQueueFamilyIndex = bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	bar.image = im.image;
	bar.subresourceRange = vi.subresourceRange;
	vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &bar);
	if (type.ms) {
	} else if (im.depth) {
		VkClearDepthStencilValue clear {0.5f, 0};
		vkCmdClearDepthStencilImage(cb, im.image, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &vi.subresourceRange);
	} else if (bytes <= staging.size) {
		FillTexels(static_cast<uint8_t*>(staging.map), bytes, im.format);
		VkBufferImageCopy region {};
		region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, im.layers};
		region.imageExtent      = {im.w, im.h, im.d};
		vkCmdCopyBufferToImage(cb, staging.buffer, im.image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
	} else {
		VkClearColorValue clear {};
		clear.float32[0] = 0.25f; clear.float32[1] = 0.5f; clear.float32[2] = 0.75f; clear.float32[3] = 1.0f;
		vkCmdClearColorImage(cb, im.image, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &vi.subresourceRange);
	}
	Submit(cb);
	return im;
}

struct Built {
	Form*                 form = nullptr;
	VkShaderModule        module = VK_NULL_HANDLE;
	VkPipelineLayout      layout = VK_NULL_HANDLE;
	VkPipeline            pipeline = VK_NULL_HANDLE;
	std::vector<VkDescriptorSet> sets;
	std::vector<uint32_t> push;
	std::vector<Image>    images;
	std::vector<std::pair<Image*, std::string>> outputs;
	std::vector<Image>    targets;
	std::vector<std::vector<uint8_t>> check;
	std::vector<double>   samples;
	double                compile_ms = 0;
};

static std::vector<uint32_t> MakeVertexShader(const std::vector<Interface>& interface) {
	std::vector<uint32_t> w {0x07230203, 0x00010300, 0, 0, 0};
	uint32_t next = 1;
	const auto id = [&]() { return next++; };
	const auto op = [&](uint32_t code, std::initializer_list<uint32_t> args) {
		w.push_back(uint32_t((args.size() + 1) << 16) | code);
		for (auto a: args) w.push_back(a);
	};
	const uint32_t t_void = id(), t_fn = id(), t_f32 = id(), t_i32 = id(), t_u32 = id(), t_v4 = id(), main_fn = id(), vi = id(), pos = id();
	const uint32_t p_in_i32 = id(), p_out_v4 = id();
	struct Out { uint32_t var, type, ptr, kind, n; };
	std::vector<Out> outs;
	std::vector<Interface> inputs;
	for (auto& x: interface) {
		if (x.output) continue;
		if (std::find_if(inputs.begin(), inputs.end(), [&](const Interface& o) { return o.location == x.location; }) != inputs.end()) continue;
		inputs.push_back(x);
	}
	for (auto& x: inputs) outs.push_back({id(), id(), id(), x.kind, x.components});
	op(17, {1});
	{
		std::vector<uint32_t> name {0x4c534c47, 0x6474732e, 0x3035342e, 0};
		w.push_back(uint32_t((name.size() + 2) << 16) | 11);
		const uint32_t glsl = id();
		w.push_back(glsl);
		for (auto n: name) w.push_back(n);
	}
	op(14, {0, 1});
	{
		std::vector<uint32_t> head {0, main_fn, 0x6e69616d, 0, vi, pos};
		for (auto& o: outs) head.push_back(o.var);
		w.push_back(uint32_t((head.size() + 1) << 16) | 15);
		for (auto h: head) w.push_back(h);
	}
	op(71, {vi, 11, 42});
	op(71, {pos, 11, 0});
	for (size_t k = 0; k < outs.size(); k++) {
		op(71, {outs[k].var, 30, inputs[k].location});
		if (outs[k].kind != 0) op(71, {outs[k].var, 14});
	}
	op(19, {t_void});
	op(33, {t_fn, t_void});
	op(22, {t_f32, 32});
	op(21, {t_i32, 32, 1});
	op(21, {t_u32, 32, 0});
	op(23, {t_v4, t_f32, 4});
	op(32, {p_in_i32, 1, t_i32});
	op(32, {p_out_v4, 3, t_v4});
	const uint32_t c_i1 = id(), c_i2 = id(), c_f2 = id(), c_fm1 = id(), c_fh = id(), c_f1 = id(), c_u7 = id(), c_i7 = id();
	op(43, {t_i32, c_i1, 1});
	op(43, {t_i32, c_i2, 2});
	op(43, {t_f32, c_f2, 0x40000000});
	op(43, {t_f32, c_fm1, 0xbf800000});
	op(43, {t_f32, c_fh, 0x3f000000});
	op(43, {t_f32, c_f1, 0x3f800000});
	op(43, {t_u32, c_u7, 7});
	op(43, {t_i32, c_i7, 7});
	for (auto& o: outs) {
		const uint32_t scalar = o.kind == 0 ? t_f32 : o.kind == 1 ? t_u32 : t_i32;
		if (o.n == 1) o.type = scalar;
		else op(23, {o.type, scalar, o.n});
		op(32, {o.ptr, 3, o.type});
	}
	op(59, {p_in_i32, vi, 1});
	op(59, {p_out_v4, pos, 3});
	for (auto& o: outs) op(59, {o.ptr, o.var, 3});
	op(54, {t_void, main_fn, 0, t_fn});
	op(248, {id()});
	const uint32_t index = id(), sh = id(), a = id(), b = id(), fa = id(), fb = id(), x = id(), y = id(), xx = id(), yy = id(), p = id();
	op(61, {t_i32, index, vi});
	op(196, {t_i32, sh, index, c_i1});
	op(199, {t_i32, a, sh, c_i2});
	op(199, {t_i32, b, index, c_i2});
	op(111, {t_f32, fa, a});
	op(111, {t_f32, fb, b});
	op(133, {t_f32, x, fa, c_f2});
	op(133, {t_f32, y, fb, c_f2});
	op(129, {t_f32, xx, x, c_fm1});
	op(129, {t_f32, yy, y, c_fm1});
	op(80, {t_v4, p, xx, yy, c_fh, c_f1});
	op(62, {pos, p});
	const uint32_t u = id(), v = id();
	op(133, {t_f32, u, fa, c_fh});
	op(133, {t_f32, v, fb, c_fh});
	for (auto& o: outs) {
		std::vector<uint32_t> parts;
		for (uint32_t c = 0; c < o.n; c++) {
			if (o.kind == 0) parts.push_back(c == 0 ? u : c == 1 ? v : c == 2 ? c_fh : c_f1);
			else parts.push_back(o.kind == 1 ? c_u7 : c_i7);
		}
		uint32_t value = parts[0];
		if (o.n > 1) {
			value = id();
			w.push_back(uint32_t((parts.size() + 3) << 16) | 80);
			w.push_back(o.type);
			w.push_back(value);
			for (auto q: parts) w.push_back(q);
		}
		op(62, {o.var, value});
	}
	op(253, {});
	op(56, {});
	w[3] = next;
	return w;
}

static void PrintStats(const char* name, const char* form, VkPipeline pipeline) {
	auto gprops = (PFN_vkGetPipelineExecutablePropertiesKHR)vkGetDeviceProcAddr(g_dev, "vkGetPipelineExecutablePropertiesKHR");
	auto gstats = (PFN_vkGetPipelineExecutableStatisticsKHR)vkGetDeviceProcAddr(g_dev, "vkGetPipelineExecutableStatisticsKHR");
	if (gprops == nullptr || gstats == nullptr) return;
	VkPipelineInfoKHR pi {VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR, nullptr, pipeline};
	uint32_t nx = 0;
	gprops(g_dev, &pi, &nx, nullptr);
	std::vector<VkPipelineExecutablePropertiesKHR> xp(nx, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
	gprops(g_dev, &pi, &nx, xp.data());
	for (uint32_t x = 0; x < nx; x++) {
		VkPipelineExecutableInfoKHR xi {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR, nullptr, pipeline, x};
		uint32_t ns = 0;
		gstats(g_dev, &xi, &ns, nullptr);
		std::vector<VkPipelineExecutableStatisticKHR> st(ns, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
		gstats(g_dev, &xi, &ns, st.data());
		printf("[e0-stats] %s form %s exe=%s stages=%x subgroup=%u", name, form, xp[x].name, uint32_t(xp[x].stages), xp[x].subgroupSize);
		for (uint32_t i = 0; i < ns; i++) {
			std::string label = st[i].name;
			std::replace(label.begin(), label.end(), ' ', '_');
			switch (st[i].format) {
				case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR: printf(" %s=%u", label.c_str(), st[i].value.b32); break;
				case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: printf(" %s=%lld", label.c_str(), (long long)st[i].value.i64); break;
				case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: printf(" %s=%llu", label.c_str(), (unsigned long long)st[i].value.u64); break;
				default: printf(" %s=%g", label.c_str(), st[i].value.f64); break;
			}
		}
		printf("\n");
	}
}

static uint64_t PageIndex(uint64_t address) {
	return (address < LowerSize ? address : address - ExtendedBase + LowerSize) >> PageBits;
}

int main(int argc, char** argv) {
	std::string recipe_path, mem_root, want_gpu, only_forms;
	uint32_t    batch = 16, rounds = 30, barrier = 1, check = 2, warm = 3, width = 1920, height = 1080;
	bool        stats_only = false;
	double      min_us     = 2000.0;
	for (int i = 1; i < argc; i++) {
		const std::string a = argv[i];
		const auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
		if (a == "--recipe") recipe_path = next();
		else if (a == "--mem-root") mem_root = next();
		else if (a == "--gpu") want_gpu = next();
		else if (a == "--forms") only_forms = next();
		else if (a == "--batch") batch = uint32_t(atoi(next().c_str()));
		else if (a == "--rounds") rounds = uint32_t(atoi(next().c_str()));
		else if (a == "--barrier") barrier = uint32_t(atoi(next().c_str()));
		else if (a == "--check") check = uint32_t(atoi(next().c_str()));
		else if (a == "--warm") warm = uint32_t(atoi(next().c_str()));
		else if (a == "--stats-only") stats_only = true;
		else if (a == "--min-us") min_us = atof(next().c_str());
		else if (a == "--size") { const auto v = next(); sscanf(v.c_str(), "%ux%u", &width, &height); }
	}
	Recipe recipe;
	if (!ParseRecipe(recipe_path, mem_root, recipe)) {
		fprintf(stderr, "usage: e0bench --recipe <recipe.txt> [--mem-root dir] [--gpu name] [--forms a,b] [--batch N] [--rounds R] "
		                "[--barrier 0|1] [--check K]\n");
		return 64;
	}
	if (recipe.stage != "cs" && recipe.stage != "ps") {
		printf("[e0] %s stage %s is not played by this bench\n", recipe.name.c_str(), recipe.stage.c_str());
		return 3;
	}
	const bool pixel = recipe.stage == "ps";
	const VkShaderStageFlags stage_flags = pixel ? VK_SHADER_STAGE_FRAGMENT_BIT : VK_SHADER_STAGE_COMPUTE_BIT;
	const VkPipelineStageFlags shader_stage = pixel ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	const VkPipelineBindPoint bind_point = pixel ? VK_PIPELINE_BIND_POINT_GRAPHICS : VK_PIPELINE_BIND_POINT_COMPUTE;
	VkApplicationInfo app {VK_STRUCTURE_TYPE_APPLICATION_INFO};
	app.apiVersion = VK_API_VERSION_1_3;
	VkInstanceCreateInfo ici {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	ici.pApplicationInfo = &app;
	CHECK(vkCreateInstance(&ici, nullptr, &g_inst));
	uint32_t n = 0;
	vkEnumeratePhysicalDevices(g_inst, &n, nullptr);
	std::vector<VkPhysicalDevice> devs(n);
	vkEnumeratePhysicalDevices(g_inst, &n, devs.data());
	g_phys = VK_NULL_HANDLE;
	for (auto d: devs) {
		vkGetPhysicalDeviceProperties(d, &g_props);
		if (want_gpu.empty() ? g_props.vendorID == 0x10de : std::string(g_props.deviceName).find(want_gpu) != std::string::npos) {
			g_phys = d;
			break;
		}
	}
	if (g_phys == VK_NULL_HANDLE && !devs.empty() && want_gpu.empty()) g_phys = devs[0];
	if (g_phys == VK_NULL_HANDLE) {
		fprintf(stderr, "no device\n");
		return 2;
	}
	vkGetPhysicalDeviceProperties(g_phys, &g_props);
	vkGetPhysicalDeviceMemoryProperties(g_phys, &g_memprops);
	const char* wanted[] = {"VK_KHR_pipeline_executable_properties", "VK_KHR_ray_query", "VK_KHR_acceleration_structure", "VK_KHR_deferred_host_operations",
	                        "VK_EXT_shader_image_atomic_int64", "VK_KHR_compute_shader_derivatives",
	                        "VK_KHR_workgroup_memory_explicit_layout", "VK_EXT_robustness2", "VK_KHR_shader_clock",
	                        "VK_KHR_maintenance8", "VK_EXT_shader_atomic_float", "VK_EXT_shader_atomic_float2",
	                        "VK_KHR_shader_maximal_reconvergence", "VK_KHR_shader_subgroup_uniform_control_flow",
	                        "VK_KHR_shader_relaxed_extended_instruction", "VK_KHR_shader_float_controls2",
	                        "VK_KHR_fragment_shader_barycentric", "VK_EXT_fragment_shader_interlock"};
	uint32_t ne = 0;
	vkEnumerateDeviceExtensionProperties(g_phys, nullptr, &ne, nullptr);
	std::vector<VkExtensionProperties> ep(ne);
	vkEnumerateDeviceExtensionProperties(g_phys, nullptr, &ne, ep.data());
	std::vector<const char*> en;
	bool has_rq = false, has_as = false;
	for (const char* want: wanted) {
		for (const auto& e: ep) {
			if (!strcmp(e.extensionName, want)) {
				en.push_back(want);
				if (!strcmp(want, "VK_KHR_ray_query")) has_rq = true;
				if (!strcmp(want, "VK_KHR_acceleration_structure")) has_as = true;
			}
		}
	}
	VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pe {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
	VkPhysicalDeviceRayQueryFeaturesKHR rq {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR, &pe};
	VkPhysicalDeviceAccelerationStructureFeaturesKHR as {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR, &rq};
	VkPhysicalDeviceVulkan13Features f13 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, has_rq && has_as ? (void*)&as : (void*)&pe};
	VkPhysicalDeviceVulkan12Features f12 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
	VkPhysicalDeviceVulkan11Features f11 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
	VkPhysicalDeviceFeatures2 f2 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11};
	vkGetPhysicalDeviceFeatures2(g_phys, &f2);
	f12.bufferDeviceAddressCaptureReplay = VK_FALSE;
	f12.bufferDeviceAddressMultiDevice   = VK_FALSE;
	as.accelerationStructureCaptureReplay = VK_FALSE;
	uint32_t qn = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &qn, nullptr);
	std::vector<VkQueueFamilyProperties> qf(qn);
	vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &qn, qf.data());
	for (uint32_t i = 0; i < qn; i++) {
		if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && (qf[i].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) &&
		    (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
			g_family = i;
			break;
		}
	}
	float prio = 1.0f;
	VkDeviceQueueCreateInfo qci {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
	qci.queueFamilyIndex = g_family;
	qci.queueCount       = 1;
	qci.pQueuePriorities = &prio;
	VkDeviceCreateInfo dci {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2};
	dci.queueCreateInfoCount    = 1;
	dci.pQueueCreateInfos       = &qci;
	dci.enabledExtensionCount   = uint32_t(en.size());
	dci.ppEnabledExtensionNames = en.data();
	CHECK(vkCreateDevice(g_phys, &dci, nullptr, &g_dev));
	vkGetDeviceQueue(g_dev, g_family, 0, &g_queue);
	VkCommandPoolCreateInfo cpi {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
	cpi.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	cpi.queueFamilyIndex = g_family;
	CHECK(vkCreateCommandPool(g_dev, &cpi, nullptr, &g_pool));
	VkFenceCreateInfo fci {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
	CHECK(vkCreateFence(g_dev, &fci, nullptr, &g_fence));
	VkCommandBuffer cb = NewCmd();
	const double ts_period = g_props.limits.timestampPeriod;
	const uint64_t storage_align = g_props.limits.minStorageBufferOffsetAlignment;
	const uint64_t uniform_align = std::max<uint64_t>(storage_align, g_props.limits.minUniformBufferOffsetAlignment);
	printf("[e0] gpu \"%s\" driver %u.%u ssbo_align %" PRIu64 " ubo_align %" PRIu64 " sparse %d residency %d\n", g_props.deviceName,
	       g_props.driverVersion >> 22, (g_props.driverVersion >> 14) & 0xff, storage_align,
	       uint64_t(g_props.limits.minUniformBufferOffsetAlignment), f2.features.sparseBinding, f2.features.sparseResidencyBuffer);

	uint64_t lo = UINT64_MAX, hi = 0, total = 0;
	for (auto& g: recipe.regions) {
		lo = std::min(lo, g.start);
		hi = std::max(hi, g.start + g.size);
	}
	const uint64_t Block = 0x10000;
	const uint64_t guest_base = lo == UINT64_MAX ? 0 : lo & ~(Block - 1);
	const uint64_t mirror_size = lo == UINT64_MAX ? Block : ((hi + Block - 1) & ~(Block - 1)) - guest_base;
	const VkBufferUsageFlags mirror_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
	                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
	                                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	VkBufferCreateInfo mbi {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	mbi.flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT;
	mbi.size  = mirror_size;
	mbi.usage = mirror_usage;
	VkBuffer mirror;
	CHECK(vkCreateBuffer(g_dev, &mbi, nullptr, &mirror));
	VkMemoryRequirements mmr;
	vkGetBufferMemoryRequirements(g_dev, mirror, &mmr);
	const uint64_t sparse_align = std::max<uint64_t>(mmr.alignment, Block);
	std::vector<std::pair<uint64_t, uint64_t>> spans;
	for (auto& g: recipe.regions) {
		const uint64_t a = (g.start - guest_base) & ~(sparse_align - 1);
		const uint64_t b = std::min((g.start + g.size - guest_base + sparse_align - 1) & ~(sparse_align - 1), mirror_size);
		spans.emplace_back(a, b);
	}
	std::sort(spans.begin(), spans.end());
	std::vector<std::pair<uint64_t, uint64_t>> merged;
	for (auto& s: spans) {
		if (!merged.empty() && s.first <= merged.back().second) merged.back().second = std::max(merged.back().second, s.second);
		else merged.push_back(s);
	}
	uint64_t bound_bytes = 0;
	for (auto& s: merged) bound_bytes += s.second - s.first;
	VkMemoryAllocateFlagsInfo mfi {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
	mfi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
	VkMemoryAllocateInfo mai {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &mfi};
	mai.allocationSize  = std::max<uint64_t>(bound_bytes, sparse_align);
	mai.memoryTypeIndex = MemType(mmr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	VkDeviceMemory mirror_mem;
	CHECK(vkAllocateMemory(g_dev, &mai, nullptr, &mirror_mem));
	{
		std::vector<VkSparseMemoryBind> binds;
		uint64_t at = 0;
		for (auto& s: merged) {
			VkSparseMemoryBind b {};
			b.resourceOffset = s.first;
			b.size           = s.second - s.first;
			b.memory         = mirror_mem;
			b.memoryOffset   = at;
			at += b.size;
			binds.push_back(b);
		}
		VkSparseBufferMemoryBindInfo bbi {mirror, uint32_t(binds.size()), binds.data()};
		VkBindSparseInfo bsi {VK_STRUCTURE_TYPE_BIND_SPARSE_INFO};
		bsi.bufferBindCount = binds.empty() ? 0 : 1;
		bsi.pBufferBinds    = &bbi;
		CHECK(vkQueueBindSparse(g_queue, 1, &bsi, g_fence));
		CHECK(vkWaitForFences(g_dev, 1, &g_fence, VK_TRUE, UINT64_MAX));
		CHECK(vkResetFences(g_dev, 1, &g_fence));
	}
	VkBufferDeviceAddressInfo mdi {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
	mdi.buffer = mirror;
	const uint64_t mirror_bda  = vkGetBufferDeviceAddress(g_dev, &mdi);
	const uint64_t mirror_zero = mirror_bda - guest_base;
	for (auto& g: recipe.regions) {
		if (g.file == "-") continue;
		g.pristine_offset = total;
		total += g.size;
	}
	const VkBufferUsageFlags xfer = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	Buf staging  = MakeBuf(std::max<uint64_t>(total, 96ull << 20), xfer, true);
	Buf pristine = MakeBuf(std::max<uint64_t>(total, 16), xfer, false);
	for (auto& g: recipe.regions) {
		if (g.file == "-") continue;
		std::ifstream f(g.file, std::ios::binary);
		if (!f) {
			fprintf(stderr, "cannot read %s\n", g.file.c_str());
			return 2;
		}
		memset(static_cast<char*>(staging.map) + g.pristine_offset, 0, g.size);
		f.read(static_cast<char*>(staging.map) + g.pristine_offset, std::streamsize(g.size));
	}
	if (total != 0) {
		Begin(cb);
		VkBufferCopy c {0, 0, total};
		vkCmdCopyBuffer(cb, staging.buffer, pristine.buffer, 1, &c);
		Submit(cb);
	}
	const auto Reset = [&](VkCommandBuffer c) {
		for (auto& g: recipe.regions) {
			if (g.file == "-") vkCmdFillBuffer(c, mirror, g.start - guest_base, g.size, 0);
		}
		for (auto& g: recipe.regions) {
			if (g.file == "-") continue;
			VkBufferCopy r {g.pristine_offset, g.start - guest_base, g.size};
			vkCmdCopyBuffer(c, pristine.buffer, mirror, 1, &r);
		}
		VkMemoryBarrier mb {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
		mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_UNIFORM_READ_BIT;
		vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TRANSFER_BIT, shader_stage, 0, 1, &mb, 0, nullptr, 0, nullptr);
	};
	printf("[e0] %s mirror guest_base %" PRIx64 " size %" PRIx64 " bound %" PRIu64 " KiB in %zu spans, %zu regions\n",
	       recipe.name.c_str(), guest_base, mirror_size, bound_bytes >> 10, merged.size(), recipe.regions.size());

	Buf dummy   = MakeBuf(1 << 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | xfer, false);
	Buf scratch = MakeBuf(8 << 20, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | xfer, false);
	Buf pagetable, faults;
	bool any_dma = false;
	for (auto& f: recipe.forms) any_dma |= f.dma != 0;
	{
		Begin(cb);
		vkCmdFillBuffer(cb, dummy.buffer, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(cb, scratch.buffer, 0, VK_WHOLE_SIZE, 0);
		Submit(cb);
	}
	if (any_dma) {
		pagetable = MakeBuf(PagetableBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | xfer, false);
		faults    = MakeBuf(FaultBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | xfer, false);
		Begin(cb);
		vkCmdFillBuffer(cb, pagetable.buffer, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(cb, faults.buffer, 0, VK_WHOLE_SIZE, 0);
		Submit(cb);
		for (auto& s: merged) {
			const uint64_t first = guest_base + s.first, pages = (s.second - s.first) >> PageBits;
			for (uint64_t done = 0; done < pages;) {
				const uint64_t chunk = std::min<uint64_t>(pages - done, staging.size / 8);
				auto* out = static_cast<uint64_t*>(staging.map);
				for (uint64_t p = 0; p < chunk; p++) out[p] = mirror_zero + first + ((done + p) << PageBits);
				Begin(cb);
				VkBufferCopy c {0, PageIndex(first + (done << PageBits)) * 8, chunk * 8};
				vkCmdCopyBuffer(cb, staging.buffer, pagetable.buffer, 1, &c);
				Submit(cb);
				done += chunk;
			}
		}
	}
	VkSamplerCreateInfo sci {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
	sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
	sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
	sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	sci.maxLod = 0.0f;
	VkSampler sampler, shadow_sampler;
	CHECK(vkCreateSampler(g_dev, &sci, nullptr, &sampler));
	sci.compareEnable = VK_TRUE;
	sci.compareOp     = VK_COMPARE_OP_LESS_OR_EQUAL;
	CHECK(vkCreateSampler(g_dev, &sci, nullptr, &shadow_sampler));

	std::vector<Built> built;
	std::vector<Buf>   keep;
	for (auto& form: recipe.forms) {
		if (!only_forms.empty() && ("," + only_forms + ",").find("," + form.name + ",") == std::string::npos) continue;
		Built b;
		b.form = &form;
		g_seed = 0x12345u;
		const auto spv = LoadSpv(form.spv);
		if (spv.size() < 5) {
			fprintf(stderr, "cannot read %s\n", form.spv.c_str());
			return 2;
		}
		uint32_t model = 0;
		std::vector<Interface> interface;
		auto bindings = Reflect(spv, model, &interface);
		uint32_t maxset = 0;
		for (auto& x: bindings) maxset = std::max(maxset, x.set);
		bool unsupported = false;
		for (auto& x: bindings) {
			if (x.count == 0 || x.type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR || x.set > 0) {
				if (!stats_only) {
					printf("[e0] %s %s unsupported binding set %u binding %u type %d count %u\n", recipe.name.c_str(),
					       form.name.c_str(), x.set, x.binding, int(x.type), x.count);
				}
				unsupported = true;
				if (x.count == 0) x.count = 65536;
			}
		}
		if (unsupported && !stats_only) continue;
		std::vector<VkDescriptorSetLayout> layouts(maxset + 1);
		std::map<VkDescriptorType, uint32_t> pool_counts;
		for (uint32_t index = 0; index <= maxset; index++) {
			std::vector<VkDescriptorSetLayoutBinding> lb;
			for (auto& x: bindings) {
				if (x.set != index) continue;
				lb.push_back({x.binding, x.type, x.count, stage_flags, nullptr});
				pool_counts[x.type] += x.count;
			}
			VkDescriptorSetLayoutCreateInfo lci {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
			lci.bindingCount = uint32_t(lb.size());
			lci.pBindings    = lb.data();
			CHECK(vkCreateDescriptorSetLayout(g_dev, &lci, nullptr, &layouts[index]));
		}
		VkDescriptorSetLayout sl = layouts[0];
		VkPushConstantRange pcr {stage_flags, 0, 128};
		VkPipelineLayoutCreateInfo plci {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
		plci.setLayoutCount         = uint32_t(layouts.size());
		plci.pSetLayouts            = layouts.data();
		plci.pushConstantRangeCount = 1;
		plci.pPushConstantRanges    = &pcr;
		CHECK(vkCreatePipelineLayout(g_dev, &plci, nullptr, &b.layout));
		VkShaderModuleCreateInfo smci {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
		smci.codeSize = spv.size() * 4;
		smci.pCode    = spv.data();
		CHECK(vkCreateShaderModule(g_dev, &smci, nullptr, &b.module));
		std::chrono::steady_clock::time_point t0, t1;
		VkResult created = VK_SUCCESS;
		if (!pixel) {
			VkComputePipelineCreateInfo cci {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
			cci.flags  = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
			cci.stage  = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, b.module, "main", nullptr};
			cci.layout = b.layout;
			t0 = std::chrono::steady_clock::now();
			created = vkCreateComputePipelines(g_dev, VK_NULL_HANDLE, 1, &cci, nullptr, &b.pipeline);
			t1 = std::chrono::steady_clock::now();
		} else {
			const auto vs = MakeVertexShader(interface);
			VkShaderModuleCreateInfo vmci {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
			vmci.codeSize = vs.size() * 4;
			vmci.pCode    = vs.data();
			VkShaderModule vsm;
			CHECK(vkCreateShaderModule(g_dev, &vmci, nullptr, &vsm));
			std::vector<Interface> outputs;
			for (auto& x: interface) if (x.output) outputs.push_back(x);
			std::sort(outputs.begin(), outputs.end(), [](const Interface& l, const Interface& r) { return l.location < r.location; });
			uint32_t attachments = 0;
			for (auto& x: outputs) attachments = std::max(attachments, x.location + 1);
			std::vector<VkFormat> formats(attachments, VK_FORMAT_R16G16B16A16_SFLOAT);
			for (auto& x: outputs) {
				formats[x.location] = x.kind == 0 ? VK_FORMAT_R16G16B16A16_SFLOAT : x.kind == 1 ? VK_FORMAT_R32G32B32A32_UINT : VK_FORMAT_R32G32B32A32_SINT;
			}
			for (uint32_t i = 0; i < attachments && !stats_only; i++) {
				ImageType type;
				type.sampled_kind = formats[i] == VK_FORMAT_R16G16B16A16_SFLOAT ? 0 : formats[i] == VK_FORMAT_R32G32B32A32_UINT ? 1 : 2;
				Image im;
				im.format = formats[i];
				im.texel  = TexelBytes(im.format);
				im.w = width; im.h = height;
				VkImageCreateInfo ci {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
				ci.imageType = VK_IMAGE_TYPE_2D;
				ci.format = im.format;
				ci.extent = {width, height, 1};
				ci.mipLevels = 1; ci.arrayLayers = 1;
				ci.samples = VK_SAMPLE_COUNT_1_BIT;
				ci.tiling = VK_IMAGE_TILING_OPTIMAL;
				ci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
				CHECK(vkCreateImage(g_dev, &ci, nullptr, &im.image));
				VkMemoryRequirements mr;
				vkGetImageMemoryRequirements(g_dev, im.image, &mr);
				VkMemoryAllocateInfo ai {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
				ai.allocationSize = mr.size;
				ai.memoryTypeIndex = MemType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
				CHECK(vkAllocateMemory(g_dev, &ai, nullptr, &im.memory));
				CHECK(vkBindImageMemory(g_dev, im.image, im.memory, 0));
				VkImageViewCreateInfo vi {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
				vi.image = im.image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = im.format;
				vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
				CHECK(vkCreateImageView(g_dev, &vi, nullptr, &im.view));
				Begin(cb);
				VkImageMemoryBarrier bar {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
				bar.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
				bar.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
				bar.newLayout = VK_IMAGE_LAYOUT_GENERAL;
				bar.srcQueueFamilyIndex = bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				bar.image = im.image;
				bar.subresourceRange = vi.subresourceRange;
				vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &bar);
				Submit(cb);
				b.targets.push_back(im);
			}
			VkPipelineShaderStageCreateInfo stages[2] {
			    {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vsm, "main", nullptr},
			    {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, b.module, "main", nullptr}};
			VkPipelineVertexInputStateCreateInfo vin {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
			VkPipelineInputAssemblyStateCreateInfo ia {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
			ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
			VkViewport viewport {0, 0, float(width), float(height), 0, 1};
			VkRect2D scissor {{0, 0}, {width, height}};
			VkPipelineViewportStateCreateInfo vp {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
			vp.viewportCount = 1; vp.pViewports = &viewport; vp.scissorCount = 1; vp.pScissors = &scissor;
			VkPipelineRasterizationStateCreateInfo rs {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
			rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1.0f;
			VkPipelineMultisampleStateCreateInfo ms {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
			ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
			VkPipelineDepthStencilStateCreateInfo ds {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
			std::vector<VkPipelineColorBlendAttachmentState> blends(attachments);
			for (auto& x: blends) x.colorWriteMask = 0xf;
			VkPipelineColorBlendStateCreateInfo cbs {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
			cbs.attachmentCount = attachments; cbs.pAttachments = blends.data();
			VkPipelineRenderingCreateInfo rci {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
			rci.colorAttachmentCount = attachments; rci.pColorAttachmentFormats = formats.data();
			VkGraphicsPipelineCreateInfo gci {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rci};
			gci.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
			gci.stageCount = 2; gci.pStages = stages;
			gci.pVertexInputState = &vin; gci.pInputAssemblyState = &ia; gci.pViewportState = &vp; gci.pRasterizationState = &rs;
			gci.pMultisampleState = &ms; gci.pDepthStencilState = &ds; gci.pColorBlendState = &cbs;
			gci.layout = b.layout;
			t0 = std::chrono::steady_clock::now();
			created = vkCreateGraphicsPipelines(g_dev, VK_NULL_HANDLE, 1, &gci, nullptr, &b.pipeline);
			t1 = std::chrono::steady_clock::now();
		}
		b.compile_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
		if (created != VK_SUCCESS) {
			printf("[e0] %s %s pipeline creation failed %d\n", recipe.name.c_str(), form.name.c_str(), int(created));
			continue;
		}
		PrintStats(recipe.name.c_str(), form.name.c_str(), b.pipeline);
		if (stats_only) {
			printf("[e0-compile] %s form %s compile_ms %.0f\n", recipe.name.c_str(), form.name.c_str(), b.compile_ms);
			continue;
		}
		std::vector<VkDescriptorPoolSize> sizes;
		for (auto& [type, count]: pool_counts) sizes.push_back({type, count});
		VkDescriptorPoolCreateInfo dpi {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
		dpi.maxSets       = 1;
		dpi.poolSizeCount = uint32_t(sizes.size());
		dpi.pPoolSizes    = sizes.data();
		VkDescriptorPool dpool;
		if (!sizes.empty()) CHECK(vkCreateDescriptorPool(g_dev, &dpi, nullptr, &dpool));
		VkDescriptorSet set = VK_NULL_HANDLE;
		if (!sizes.empty()) {
			VkDescriptorSetAllocateInfo dai {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
			dai.descriptorPool     = dpool;
			dai.descriptorSetCount = 1;
			dai.pSetLayouts        = &sl;
			CHECK(vkAllocateDescriptorSets(g_dev, &dai, &set));
		}
		b.sets.push_back(set);
		auto data = form.sdata;
		std::vector<VkDescriptorBufferInfo> buffer_views(256, {dummy.buffer, 0, VK_WHOLE_SIZE});
		std::vector<VkDescriptorBufferInfo> uniform_views(64, {dummy.buffer, 0, 65536});
		for (auto& rb: form.bufs) {
			if (rb.index >= buffer_views.size() || rb.address == 0 || rb.size == 0) continue;
			const uint64_t align   = rb.uniform >= 0 ? uniform_align : storage_align;
			const uint64_t offset  = rb.address - guest_base;
			const uint64_t aligned = offset & ~(align - 1);
			const uint64_t adjust  = offset - aligned;
			buffer_views[rb.index] = {mirror, aligned, rb.size + adjust};
			if (form.memoff + rb.index / 4 < data.size()) data[form.memoff + rb.index / 4] |= uint32_t(adjust) << ((rb.index % 4) * 8);
			if (form.memsize + rb.index < data.size()) data[form.memsize + rb.index] = uint32_t(rb.size + adjust);
			if (form.stride_count != 0 && form.stride + rb.index < data.size()) data[form.stride + rb.index] = rb.stride;
			if (rb.uniform >= 0 && size_t(rb.uniform) < uniform_views.size()) {
				uniform_views[rb.uniform] = {mirror, aligned, std::min<uint64_t>(rb.size + adjust, 65536)};
			}
		}
		if (form.gpu) {
			if (form.tblock + 1 < data.size()) {
				data[form.tblock]     = uint32_t(mirror_zero);
				data[form.tblock + 1] = uint32_t(mirror_zero >> 32);
			}
			for (auto& root: form.roots) {
				const uint64_t offset  = root.address - guest_base;
				const uint64_t aligned = offset & ~(uniform_align - 1);
				if (root.address < guest_base || aligned >= mirror_size) {
					printf("[e0] %s %s root %u at %" PRIx64 " is outside the mirror\n", recipe.name.c_str(), form.name.c_str(), root.k,
					       root.address);
					continue;
				}
				if (root.uniform < uniform_views.size()) {
					uniform_views[root.uniform] = {mirror, aligned, std::min<uint64_t>(65536, mirror_size - aligned)};
				}
				if (form.tblock + 4 + root.k < data.size()) data[form.tblock + 4 + root.k] = uint32_t((offset - aligned) / 4);
			}
		}
		b.push.assign(32, 0);
		if (form.push) {
			for (size_t i = 0; i < data.size() && form.start + i < 32; i++) b.push[form.start + i] = data[i];
		}
		std::vector<VkWriteDescriptorSet> writes;
		std::vector<std::vector<VkDescriptorBufferInfo>> buffer_infos;
		std::vector<std::vector<VkDescriptorImageInfo>> image_infos;
		buffer_infos.reserve(bindings.size());
		image_infos.reserve(bindings.size());
		b.images.reserve(4096);
		for (auto& x: bindings) {
			const uint32_t kind = x.binding % KindCount;
			VkWriteDescriptorSet wds {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
			wds.dstSet          = set;
			wds.dstBinding      = x.binding;
			wds.descriptorCount = x.count;
			wds.descriptorType  = x.type;
			if (x.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER || x.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
				auto& infos = buffer_infos.emplace_back(x.count, VkDescriptorBufferInfo {dummy.buffer, 0, VK_WHOLE_SIZE});
				if (kind == KindBuffers) {
					for (uint32_t i = 0; i < x.count && i < buffer_views.size(); i++) infos[i] = buffer_views[i];
				} else if (kind == KindUniform) {
					for (uint32_t i = 0; i < x.count && i < uniform_views.size(); i++) infos[i] = uniform_views[i];
				} else if (kind == KindFlat || kind == KindShaderData) {
					const auto& words = kind == KindFlat ? form.flat : data;
					Buf up = MakeBuf(std::max<uint64_t>(words.size() * 4, 65536),
					                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | xfer, false);
					memset(staging.map, 0, up.size);
					memcpy(staging.map, words.data(), words.size() * 4);
					Begin(cb);
					VkBufferCopy c {0, 0, up.size};
					vkCmdCopyBuffer(cb, staging.buffer, up.buffer, 1, &c);
					Submit(cb);
					infos[0] = {up.buffer, 0, x.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ? 65536 : VK_WHOLE_SIZE};
					keep.push_back(up);
				} else if (kind == KindPagetable) {
					infos[0] = {pagetable.buffer, 0, VK_WHOLE_SIZE};
				} else if (kind == KindFault) {
					infos[0] = {faults.buffer, 0, VK_WHOLE_SIZE};
				} else {
					for (auto& info: infos) info = {scratch.buffer, 0, VK_WHOLE_SIZE};
				}
				if (x.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
					for (auto& info: infos) if (info.range == VK_WHOLE_SIZE) info.range = 65536;
				}
				wds.pBufferInfo = infos.data();
			} else if (x.type == VK_DESCRIPTOR_TYPE_SAMPLER) {
				auto& infos = image_infos.emplace_back(x.count, VkDescriptorImageInfo {sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL});
				wds.pImageInfo = infos.data();
			} else if (x.is_image && x.image.dim != 5) {
				auto& infos = image_infos.emplace_back(x.count, VkDescriptorImageInfo {});
				Image* fallback = nullptr;
				for (uint32_t i = 0; i < x.count; i++) {
					const RecipeImg* hint = nullptr;
					for (auto& m: form.imgs) {
						if (m.binding == x.binding && i >= m.index && i < m.index + std::max(m.count, 1u)) hint = &m;
					}
					Image* use = nullptr;
					if (hint != nullptr) {
						b.images.push_back(MakeImage(x.image, hint->w, hint->h, hint->d, cb, staging));
						use = &b.images.back();
						if (hint->written) b.outputs.emplace_back(use, "img" + std::to_string(x.binding) + "." + std::to_string(i));
					} else {
						if (fallback == nullptr) {
							b.images.push_back(MakeImage(x.image, 16, 16, 16, cb, staging));
							fallback = &b.images.back();
						}
						use = fallback;
					}
					infos[i] = {x.image.depth == 1 ? shadow_sampler : sampler, use->view, VK_IMAGE_LAYOUT_GENERAL};
				}
				wds.pImageInfo = infos.data();
			} else {
				printf("[e0] %s %s binding %u of type %d is not handled\n", recipe.name.c_str(), form.name.c_str(), x.binding, int(x.type));
				unsupported = true;
			}
			writes.push_back(wds);
		}
		if (unsupported) continue;
		if (!writes.empty()) vkUpdateDescriptorSets(g_dev, uint32_t(writes.size()), writes.data(), 0, nullptr);
		built.push_back(std::move(b));
	}
	if (built.empty()) {
		if (!stats_only) printf("[e0] %s nothing to play\n", recipe.name.c_str());
		return stats_only ? 0 : 3;
	}
	VkQueryPoolCreateInfo qpi {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
	qpi.queryType  = VK_QUERY_TYPE_TIMESTAMP;
	qpi.queryCount = 2;
	VkQueryPool qpool;
	CHECK(vkCreateQueryPool(g_dev, &qpi, nullptr, &qpool));
	const auto Dispatches = [&](VkCommandBuffer c, Built& b, uint32_t count, uint32_t reps = 1) {
		vkCmdBindPipeline(c, bind_point, b.pipeline);
		if (b.sets[0] != VK_NULL_HANDLE) {
			vkCmdBindDescriptorSets(c, bind_point, b.layout, 0, 1, b.sets.data(), 0, nullptr);
		}
		vkCmdPushConstants(c, b.layout, stage_flags, 0, 128, b.push.data());
		if (pixel) {
			std::vector<VkRenderingAttachmentInfo> colors(b.targets.size(), {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO});
			for (size_t i = 0; i < colors.size(); i++) {
				colors[i].imageView   = b.targets[i].view;
				colors[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
				colors[i].loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
				colors[i].storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
			}
			VkRenderingInfo ri {VK_STRUCTURE_TYPE_RENDERING_INFO};
			ri.renderArea           = {{0, 0}, {width, height}};
			ri.layerCount           = 1;
			ri.colorAttachmentCount = uint32_t(colors.size());
			ri.pColorAttachments    = colors.data();
			for (uint32_t r = 0; r < reps; r++) {
				vkCmdBeginRendering(c, &ri);
				for (uint32_t i = 0; i < count; i++) vkCmdDraw(c, 3, 1, 0, 0);
				vkCmdEndRendering(c);
			}
			return;
		}
		for (uint32_t r = 0; r < reps; r++) {
			for (uint32_t i = 0; i < count; i++) {
				vkCmdDispatch(c, recipe.groups[0], recipe.groups[1], recipe.groups[2]);
				if ((barrier != 0 && i + 1 < count) || (i + 1 == count && r + 1 < reps)) {
					VkMemoryBarrier mb {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
					mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
					mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_UNIFORM_READ_BIT;
					vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
					                     nullptr, 0, nullptr);
				}
			}
		}
	};
	uint64_t written_total = 0;
	for (auto& [a, s]: recipe.written) written_total += s;
	Buf readback = MakeBuf(std::max<uint64_t>(written_total, 256ull << 20), xfer, true);
	std::vector<std::vector<uint8_t>> again;
	for (size_t pass = 0; pass <= built.size(); pass++) {
		if (check == 0) break;
		const bool repeat = pass == built.size();
		auto& b = built[repeat ? 0 : pass];
		auto& sink = repeat ? again : b.check;
		Begin(cb);
		Reset(cb);
		Dispatches(cb, b, check);
		VkMemoryBarrier mb {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
		mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		vkCmdPipelineBarrier(cb, pixel ? VkPipelineStageFlags(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)
		                               : VkPipelineStageFlags(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT),
		                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
		uint64_t at = 0;
		for (auto& [a, s]: recipe.written) {
			VkBufferCopy c {a - guest_base, at, s};
			vkCmdCopyBuffer(cb, mirror, readback.buffer, 1, &c);
			at += s;
		}
		Submit(cb);
		sink.emplace_back(static_cast<uint8_t*>(readback.map), static_cast<uint8_t*>(readback.map) + written_total);
		if (!repeat) for (auto& target: b.targets) b.outputs.emplace_back(&target, "target");
		for (auto& [image, label]: b.outputs) {
			const uint64_t bytes = uint64_t(image->w) * image->h * image->d * image->layers * image->texel;
			if (bytes > readback.size || image->depth) continue;
			Begin(cb);
			VkBufferImageCopy region {};
			region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, image->layers};
			region.imageExtent      = {image->w, image->h, image->d};
			vkCmdCopyImageToBuffer(cb, image->image, VK_IMAGE_LAYOUT_GENERAL, readback.buffer, 1, &region);
			Submit(cb);
			sink.emplace_back(static_cast<uint8_t*>(readback.map), static_cast<uint8_t*>(readback.map) + bytes);
		}
	}
	if (check != 0) {
		const auto& ref = built[0];
		uint64_t ref_bytes = 0;
		for (auto& part: ref.check) ref_bytes += part.size();
		for (size_t k = 1; k <= built.size(); k++) {
			const bool  repeat = k == built.size();
			const auto& other  = repeat ? again : built[k].check;
			const char* label  = repeat ? "a-again" : built[k].form->name.c_str();
			uint64_t diff = 0, first = UINT64_MAX;
			size_t first_part = 0;
			for (size_t p = 0; p < ref.check.size() && p < other.size(); p++) {
				const auto& x = ref.check[p];
				const auto& y = other[p];
				for (size_t i = 0; i + 4 <= x.size() && i + 4 <= y.size(); i += 4) {
					if (memcmp(&x[i], &y[i], 4) != 0) {
						if (first == UINT64_MAX) { first = i; first_part = p; }
						diff++;
					}
				}
			}
			printf("[e0-check] %s %s vs %s: %s, %" PRIu64 " dwords differ of %" PRIu64 " (%zu outputs)", recipe.name.c_str(),
			       label, ref.form->name.c_str(), diff == 0 ? "identical" : "DIFFERENT", diff, ref_bytes / 4, ref.check.size());
			if (first != UINT64_MAX) {
				uint32_t xa, xb;
				memcpy(&xa, &ref.check[first_part][first], 4);
				memcpy(&xb, &other[first_part][first], 4);
				printf(" first at output %zu +%" PRIx64 ": %08x vs %08x", first_part, first, xa, xb);
			}
			printf("\n");
		}
		uint64_t hash = 1469598103934665603ull;
		for (auto& part: ref.check) for (auto byte: part) hash = (hash ^ byte) * 1099511628211ull;
		printf("[e0-check] %s reference %s written bytes %" PRIu64 " fnv %016" PRIx64 "\n", recipe.name.c_str(), ref.form->name.c_str(),
		       ref_bytes, hash);
	}
	const std::vector<uint32_t> modes = barrier == 2 ? std::vector<uint32_t> {1, 0} : std::vector<uint32_t> {barrier};
	const auto Sample = [&](Built& b, uint32_t reps) {
		Begin(cb);
		Reset(cb);
		Submit(cb);
		Begin(cb);
		vkCmdResetQueryPool(cb, qpool, 0, 2);
		Dispatches(cb, b, 1);
		if (!pixel) {
			VkMemoryBarrier mb {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
			mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_UNIFORM_READ_BIT;
			vkCmdPipelineBarrier(cb, shader_stage, shader_stage, 0, 1, &mb, 0, nullptr, 0, nullptr);
		}
		vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 0);
		Dispatches(cb, b, batch, reps);
		vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 1);
		Submit(cb);
		uint64_t ts[2] {};
		CHECK(vkGetQueryPoolResults(g_dev, qpool, 0, 2, sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
		return double(ts[1] - ts[0]) * ts_period / 1e3;
	};
	for (const auto mode: modes) {
		barrier = mode;
		for (auto& item: built) item.samples.clear();
		double once = 0;
		for (uint32_t k = 0; k < 3; k++) once = Sample(built[0], 1);
		const uint32_t reps = uint32_t(std::clamp(once > 0 ? std::ceil(min_us / once) : 1.0, 1.0, 4000.0));
		for (uint32_t round = 0; round < rounds + warm; round++) {
			for (size_t k = 0; k < built.size(); k++) {
				auto& b = built[(k + round) % built.size()];
				const double us = Sample(b, reps);
				if (round >= warm) b.samples.push_back(us / (double(batch) * reps));
			}
		}
		double ref_median = 0;
		for (size_t k = 0; k < built.size(); k++) {
			auto v = built[k].samples;
			std::sort(v.begin(), v.end());
			const double med = v.empty() ? 0 : v[v.size() / 2], p10 = v.empty() ? 0 : v[v.size() / 10], p90 = v.empty() ? 0 : v[v.size() * 9 / 10];
			if (k == 0) ref_median = med;
			printf("[e0-time] %s form %s batch %u barrier %u rounds %zu us_per_dispatch %.3f p10 %.3f p90 %.3f ratio %.4f compile_ms %.0f reps %u\n",
			       recipe.name.c_str(), built[k].form->name.c_str(), batch, barrier, v.size(), med, p10, p90,
			       ref_median > 0 ? med / ref_median : 0.0, built[k].compile_ms, reps);
		}
	}
	return 0;
}
