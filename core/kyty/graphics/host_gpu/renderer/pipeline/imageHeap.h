#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGEHEAP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGEHEAP_H_

#include "common/common.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/pipeline/imageHeapPredictions.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class Buffer;
class CommandBuffer;
class PageManager;
class RenderContext;
class RenderExecutor;
struct ImageHeapTestAccess;

class ImageHeap final: public ShaderRecompiler::IR::ImageHeapSource {
public:
	ImageHeap(RenderContext& context, PageManager& page_manager);
	~ImageHeap();
	KYTY_CLASS_NO_COPY(ImageHeap);

	[[nodiscard]] static vk::DescriptorSetLayout CreateLayout(vk::Device device);

	void BeginDraw() { ++m_draw; }
	bool Acquire(const ShaderRecompiler::IR::ImageHeapRequest& request,
	             ShaderRecompiler::IR::ImageHeapView&          result) override;
	bool AcquireList(const ShaderRecompiler::IR::ImageHeapList& list,
	                 ShaderRecompiler::IR::ImageHeapView&       result) override;
	bool Predict(const ShaderRecompiler::IR::ImageHeapRequest& request,
	             ShaderRecompiler::IR::ImageHeapPrediction&    prediction) override;
	bool PredictList(const ShaderRecompiler::IR::ImageHeapList& list,
	                 ShaderRecompiler::IR::ImageHeapPrediction& prediction) override;
	void Rebind(RenderExecutor& executor);
	void Commit(CommandBuffer& buffer, vk::PipelineBindPoint bind_point, vk::PipelineLayout layout);
	bool InvalidateMemory(uint64_t address, uint64_t size);
	void UnmapMemory(uint64_t address, uint64_t size);
	void RunGarbageCollector();

private:
	using DescriptorValue = ShaderRecompiler::IR::DescriptorValue;

	struct Key {
		DescriptorValue value;
		uint32_t        profile = 0;

		bool operator==(const Key&) const = default;
	};

	struct KeyHash {
		size_t operator()(const Key& key) const noexcept;
	};

	struct ListKey {
		const void* owner    = nullptr;
		uint32_t    resource = 0;
		bool        operator==(const ListKey&) const = default;
	};
	struct ListKeyHash {
		size_t operator()(const ListKey& key) const noexcept {
			return std::hash<const void*> {}(key.owner) ^ (size_t {key.resource} * 0x9e3779b97f4a7c15ull);
		}
	};

	bool AcquireNow(const ShaderRecompiler::IR::ImageHeapRequest& request,
	                ShaderRecompiler::IR::ImageHeapView&          result);
	bool AcquireListNow(const ShaderRecompiler::IR::ImageHeapList& list,
	                    ShaderRecompiler::IR::ImageHeapView&       result);

	enum class State : uint8_t { Live, Stale, Buried };

	struct Element {
		Key      key;
		ImageId  image;
		uint64_t serial = 0;
		uint32_t kind   = 0;
		uint32_t refs   = 0;
		State    state  = State::Live;
	};

	struct Memory;

	struct View {
		Memory&                      memory;
		uint64_t                     step    = 0;
		uint64_t                     first   = 0;
		uint64_t                     count   = 0;
		uint64_t                     limit   = 0;
		uint32_t                     profile = 0;
		std::vector<uint32_t>        slots;
		std::vector<uint32_t>        kind_keys;
		std::vector<DescriptorValue> kinds;
		std::unique_ptr<Buffer>      buffer;
		size_t                       dirty_begin   = 0;
		size_t                       dirty_end     = 0;
		size_t                       changed_begin = 0;
		size_t                       changed_end   = 0;
		bool                         overflowed    = false;
		bool                         released      = false;
	};

	struct Memory {
		uint64_t                           base = 0;
		uint64_t                           size = 0;
		std::vector<uint32_t>              words;
		std::vector<uint8_t>               dirty_pages;
		std::unordered_map<uint32_t, uint64_t> dead;
		uint64_t                           draw = 0;
		uint32_t                           idle = 0;
		std::vector<std::unique_ptr<View>> views;
	};

	struct List {
		View*                   view = nullptr;
		std::vector<uint32_t>   slots;
		std::unique_ptr<Buffer> buffer;
		size_t                  dirty_begin = 0;
		size_t                  dirty_end   = 0;
	};

	[[nodiscard]] bool Refresh(Memory& memory);
	void               MarkDirty(Memory& memory, size_t begin, size_t end);
	template <bool track>
	void     Watch(const Memory& memory, size_t begin, size_t end);
	void     Unwatch(const Memory& memory);
	void     Drop(Memory& memory);
	void     UpdateProbes(View& view, uint64_t begin, uint64_t end);
	void     UpdateProbe(View& view, uint64_t probe);
	void     Assign(View& view, size_t slot, uint32_t entry);
	void     Assign(List& list, size_t slot, uint32_t entry);
	void     Upload(CommandBuffer& buffer, const std::vector<uint32_t>& slots, Buffer& device,
	                size_t& dirty_begin, size_t& dirty_end);
	uint32_t Intern(View& view, uint32_t kind, const Key& key);
	uint32_t Profile(const ShaderRecompiler::IR::ImageResource& root, bool table);
	void     Release(uint32_t entry);
	void     Retire();
	void     Mark(uint32_t index, State state);
	template <typename Fn>
	bool     ForEachChanged(Fn&& fn);
	void     Repoint();
	void     TakeChanges();
	ImageId  CurrentImage(const Image::HeapChange& change);
	uint32_t Allocate(bool may_fail);
	void     Resize(Memory& memory, uint64_t base, uint64_t size);
	void     WriteDescriptors();

	RenderContext&                       m_context;
	PageManager&                         m_page_manager;
	std::mutex                           m_mutex;
	ImageHeapPredictions                 m_predictions;
	std::vector<std::unique_ptr<Memory>> m_memories;
	std::vector<std::unique_ptr<Memory>>           m_list_memories;
	std::unordered_map<ListKey, List, ListKeyHash> m_lists;
	std::vector<View*>                   m_active;
	std::vector<List*>                   m_active_lists;
	uint64_t                             m_draw         = 1;
	uint64_t                             m_suspend_draw = 0;
	std::vector<ShaderRecompiler::IR::ImageResource>  m_profiles;
	std::vector<bool>                                 m_table_profiles;
	std::vector<Element>                              m_elements;
	std::unordered_map<Key, uint32_t, KeyHash>        m_index;
	std::vector<uint32_t>                             m_free;
	std::vector<uint32_t>                             m_pending;
	std::vector<uint32_t>                             m_released;
	std::vector<uint32_t>                             m_held;
	std::vector<uint32_t>                             m_dcc;
	std::vector<std::pair<uint64_t, ImageId>>         m_away;
	uint64_t                                          m_held_serial = 0;
	std::deque<std::pair<uint64_t, uint32_t>>         m_retired;
	std::vector<std::pair<size_t, size_t>>            m_runs;
	std::vector<uint32_t>                             m_scratch;
	std::vector<Image::HeapChange>                    m_changes;
	std::vector<std::pair<uint32_t, vk::DescriptorImageInfo>> m_writes;
	vk::DescriptorSetLayout                           m_layout = nullptr;
	vk::DescriptorPool                                m_pool   = nullptr;
	vk::DescriptorSet                                 m_set    = nullptr;

	friend struct ImageHeapTestAccess;
};

}

#endif
