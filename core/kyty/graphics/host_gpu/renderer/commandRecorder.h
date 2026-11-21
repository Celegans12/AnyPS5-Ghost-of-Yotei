#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_

#include "common/assert.h"
#include "common/common.h"
#include "common/spscQueue.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstddef>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class CommandChunkQueue;

class CommandStream {
public:
	struct Chunk {
		explicit Chunk(size_t capacity);
		~Chunk();
		KYTY_CLASS_NO_COPY(Chunk);

		std::byte* data     = nullptr;
		size_t     capacity = 0;
		size_t     used     = 0;
		size_t     arrays   = 0;
	};

	enum class RecordKind : uint32_t { Command, Submit };

	struct Entry {
		using Fn = void (*)(vk::CommandBuffer command, const void* payload);
		Fn         fn   = nullptr;
		uint32_t   size = 0;
		RecordKind kind = RecordKind::Command;

		[[nodiscard]] const void* Payload() const noexcept {
			return reinterpret_cast<const std::byte*>(this) + sizeof(Entry);
		}
	};
	static_assert(sizeof(Entry) == 16);

	static constexpr size_t ChunkCapacity = 64u * 1024u;
	static constexpr size_t Alignment     = 16;

	CommandStream() = default;
	explicit CommandStream(CommandChunkQueue* output): m_output(output) {}
	~CommandStream() = default;
	KYTY_CLASS_NO_COPY(CommandStream);

	void Reserve(size_t bytes) {
		if (m_current == nullptr || m_current->arrays - m_current->used < bytes) {
			NextChunk(bytes);
		}
	}

	template <typename T>
	[[nodiscard]] T* Copy(const T* source, size_t count) {
		static_assert(std::is_trivially_copyable_v<T>);
		if (count == 0 || source == nullptr) {
			return nullptr;
		}
		auto* target = static_cast<T*>(AllocateArray(sizeof(T) * count));
		std::memcpy(static_cast<void*>(target), source, sizeof(T) * count);
		return target;
	}

	template <typename F>
	void Record(F&& function) {
		using Callable = std::decay_t<F>;
		static_assert(std::is_trivially_copyable_v<Callable> &&
		              std::is_trivially_destructible_v<Callable>);
		static_assert(alignof(Callable) <= Alignment);
		constexpr auto size = RoundUp(sizeof(Entry) + sizeof(Callable));
		auto*          record = static_cast<Entry*>(Allocate(size));
		record->fn            = [](vk::CommandBuffer command, const void* payload) {
			(*static_cast<const Callable*>(payload))(command);
		};
		record->size = static_cast<uint32_t>(size);
		record->kind = RecordKind::Command;
		::new (const_cast<void*>(record->Payload())) Callable(std::forward<F>(function));
		m_commands++;
	}

	template <typename T>
	[[nodiscard]] T* RecordSpecial(RecordKind kind) {
		static_assert(std::is_trivially_copyable_v<T> && alignof(T) <= Alignment);
		constexpr auto size = RoundUp(sizeof(Entry) + sizeof(T));
		auto*          record = static_cast<Entry*>(Allocate(size));
		record->fn            = nullptr;
		record->size          = static_cast<uint32_t>(size);
		record->kind          = kind;
		return ::new (const_cast<void*>(record->Payload())) T {};
	}

	void Publish();
	[[nodiscard]] std::vector<std::unique_ptr<Chunk>> TakeChunks();
	void Recycle(std::unique_ptr<Chunk> chunk);

	[[nodiscard]] uint64_t CommandCount() const noexcept { return m_commands; }

	template <typename Acquire, typename Special>
	static void Replay(const Chunk& chunk, Acquire&& acquire, Special&& special) {
		size_t offset = 0;
		while (offset < chunk.used) {
			const auto* record = reinterpret_cast<const Entry*>(chunk.data + offset);
			if (record->kind == RecordKind::Command) {
				record->fn(acquire(), record->Payload());
			} else {
				special(*record);
			}
			offset += record->size;
		}
	}

	static constexpr size_t RoundUp(size_t size) {
		return (size + Alignment - 1) & ~(Alignment - 1);
	}

private:
	void* Allocate(size_t bytes) {
		bytes = RoundUp(bytes);
		Reserve(bytes);
		auto* result = m_current->data + m_current->used;
		m_current->used += bytes;
		return result;
	}
	void* AllocateArray(size_t bytes) {
		bytes = RoundUp(bytes);
		Reserve(bytes);
		m_current->arrays -= bytes;
		return m_current->data + m_current->arrays;
	}
	void NextChunk(size_t bytes);

	CommandChunkQueue*                  m_output = nullptr;
	std::vector<std::unique_ptr<Chunk>> m_chunks;
	Chunk*                              m_current = nullptr;
	std::vector<std::unique_ptr<Chunk>> m_free;
	uint64_t                            m_commands = 0;
};

class CommandChunkQueue {
public:
	using ChunkPtr = std::unique_ptr<CommandStream::Chunk>;

	CommandChunkQueue(): m_chunks(MaxQueued), m_recycled(MaxRecycled) {}
	~CommandChunkQueue() = default;
	KYTY_CLASS_NO_COPY(CommandChunkQueue);

	void Push(ChunkPtr chunk) { m_chunks.Push(std::move(chunk)); }
	[[nodiscard]] ChunkPtr TakeRecycled() {
		auto chunk = m_recycled.TryPop();
		return chunk ? std::move(*chunk) : nullptr;
	}

	[[nodiscard]] ChunkPtr Pop() {
		auto chunk = m_chunks.Pop();
		return chunk ? std::move(*chunk) : nullptr;
	}
	void Recycle(ChunkPtr chunk);

	void Stop() { m_chunks.Stop(); }

private:
	static constexpr size_t MaxQueued   = 256;
	static constexpr size_t MaxRecycled = 256;

	Common::SpscQueue<ChunkPtr> m_chunks;
	Common::SpscQueue<ChunkPtr> m_recycled;
};

class CommandRecorder {
public:
	CommandRecorder() = default;
	explicit CommandRecorder(vk::CommandBuffer direct) noexcept: m_direct(direct) {}
	explicit CommandRecorder(CommandStream* stream) noexcept: m_stream(stream) {}

	[[nodiscard]] bool operator==(std::nullptr_t) const noexcept {
		return m_stream == nullptr && m_direct == nullptr;
	}
	[[nodiscard]] vk::CommandBuffer Direct() const {
		EXIT_IF(m_stream != nullptr || m_direct == nullptr);
		return m_direct;
	}

	void pipelineBarrier(vk::PipelineStageFlags src_stages, vk::PipelineStageFlags dst_stages,
	                     vk::DependencyFlags dependency, uint32_t memory_count,
	                     const vk::MemoryBarrier* memory, uint32_t buffer_count,
	                     const vk::BufferMemoryBarrier* buffers, uint32_t image_count,
	                     const vk::ImageMemoryBarrier* images) const;
	void pipelineBarrier2(const vk::DependencyInfo& dependency) const;
	void pipelineBarrier2(const vk::DependencyInfo* dependency) const {
		pipelineBarrier2(*dependency);
	}

	void bindPipeline(vk::PipelineBindPoint bind_point, vk::Pipeline pipeline) const {
		if (m_stream == nullptr) {
			m_direct.bindPipeline(bind_point, pipeline);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.bindPipeline(bind_point, pipeline); });
	}
	void bindDescriptorSets(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
	                        uint32_t first_set, uint32_t set_count,
	                        const vk::DescriptorSet* sets, uint32_t dynamic_count,
	                        const uint32_t* dynamic_offsets) const;
	void pushDescriptorSetKHR(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
	                          uint32_t set, uint32_t write_count,
	                          const vk::WriteDescriptorSet* writes) const;
	void pushDescriptorSetKHR(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
	                          uint32_t set, vk::ArrayProxy<const vk::WriteDescriptorSet> const& writes) const {
		pushDescriptorSetKHR(bind_point, layout, set, writes.size(), writes.data());
	}
	void pushConstants(vk::PipelineLayout layout, vk::ShaderStageFlags stages, uint32_t offset,
	                   uint32_t size, const void* values) const;
	void pushDescriptorSetWithTemplateKHR(vk::DescriptorUpdateTemplate update_template,
	                                      vk::PipelineLayout layout, uint32_t set, const void* data,
	                                      size_t size) const;

	void beginRendering(const vk::RenderingInfo& rendering) const;
	void beginRendering(const vk::RenderingInfo* rendering) const { beginRendering(*rendering); }
	void endRendering() const {
		if (m_stream == nullptr) {
			m_direct.endRendering();
			return;
		}
		m_stream->Record([](vk::CommandBuffer c) { c.endRendering(); });
	}

	void beginQuery(vk::QueryPool pool, uint32_t query, vk::QueryControlFlags flags) const {
		if (m_stream == nullptr) {
			m_direct.beginQuery(pool, query, flags);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.beginQuery(pool, query, flags); });
	}
	void endQuery(vk::QueryPool pool, uint32_t query) const {
		if (m_stream == nullptr) {
			m_direct.endQuery(pool, query);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.endQuery(pool, query); });
	}

	void copyQueryPoolResults(vk::QueryPool pool, uint32_t first, uint32_t count, vk::Buffer buffer,
	                          vk::DeviceSize offset, vk::DeviceSize stride,
	                          vk::QueryResultFlags flags) const {
		if (m_stream == nullptr) {
			m_direct.copyQueryPoolResults(pool, first, count, buffer, offset, stride, flags);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) {
			c.copyQueryPoolResults(pool, first, count, buffer, offset, stride, flags);
		});
	}

	void dispatch(uint32_t x, uint32_t y, uint32_t z) const {
		if (m_stream == nullptr) {
			m_direct.dispatch(x, y, z);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.dispatch(x, y, z); });
	}
	void clearAttachments(uint32_t attachment_count, const vk::ClearAttachment* attachments,
	                      uint32_t rect_count, const vk::ClearRect* rects) const {
		if (m_stream == nullptr) {
			m_direct.clearAttachments(attachment_count, attachments, rect_count, rects);
			return;
		}
		m_stream->Reserve(CommandStream::RoundUp(sizeof(vk::ClearAttachment) * attachment_count) +
		                  CommandStream::RoundUp(sizeof(vk::ClearRect) * rect_count) + 128);
		const auto* attachments_copy = m_stream->Copy(attachments, attachment_count);
		const auto* rects_copy       = m_stream->Copy(rects, rect_count);
		m_stream->Record([=](vk::CommandBuffer c) {
			c.clearAttachments(attachment_count, attachments_copy, rect_count, rects_copy);
		});
	}
	void beginConditionalRenderingEXT(const vk::ConditionalRenderingBeginInfoEXT* info) const {
		if (m_stream == nullptr) {
			m_direct.beginConditionalRenderingEXT(info);
			return;
		}
		const auto copy = *info;
		m_stream->Record([=](vk::CommandBuffer c) { c.beginConditionalRenderingEXT(&copy); });
	}
	void endConditionalRenderingEXT() const {
		if (m_stream == nullptr) {
			m_direct.endConditionalRenderingEXT();
			return;
		}
		m_stream->Record([](vk::CommandBuffer c) { c.endConditionalRenderingEXT(); });
	}
	void dispatchBase(uint32_t base_x, uint32_t base_y, uint32_t base_z, uint32_t x, uint32_t y,
	                  uint32_t z) const {
		if (m_stream == nullptr) {
			m_direct.dispatchBase(base_x, base_y, base_z, x, y, z);
			return;
		}
		m_stream->Record(
		    [=](vk::CommandBuffer c) { c.dispatchBase(base_x, base_y, base_z, x, y, z); });
	}
	void dispatchIndirect(vk::Buffer buffer, vk::DeviceSize offset) const {
		if (m_stream == nullptr) {
			m_direct.dispatchIndirect(buffer, offset);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.dispatchIndirect(buffer, offset); });
	}
	void draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex,
	          uint32_t first_instance) const {
		if (m_stream == nullptr) {
			m_direct.draw(vertex_count, instance_count, first_vertex, first_instance);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) {
			c.draw(vertex_count, instance_count, first_vertex, first_instance);
		});
	}
	void drawIndexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index,
	                 int32_t vertex_offset, uint32_t first_instance) const {
		if (m_stream == nullptr) {
			m_direct.drawIndexed(index_count, instance_count, first_index, vertex_offset,
			                     first_instance);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) {
			c.drawIndexed(index_count, instance_count, first_index, vertex_offset, first_instance);
		});
	}
	void drawIndirect(vk::Buffer buffer, vk::DeviceSize offset, uint32_t draw_count,
	                  uint32_t stride) const {
		if (m_stream == nullptr) {
			m_direct.drawIndirect(buffer, offset, draw_count, stride);
			return;
		}
		m_stream->Record(
		    [=](vk::CommandBuffer c) { c.drawIndirect(buffer, offset, draw_count, stride); });
	}
	void drawIndexedIndirect(vk::Buffer buffer, vk::DeviceSize offset, uint32_t draw_count,
	                         uint32_t stride) const {
		if (m_stream == nullptr) {
			m_direct.drawIndexedIndirect(buffer, offset, draw_count, stride);
			return;
		}
		m_stream->Record(
		    [=](vk::CommandBuffer c) { c.drawIndexedIndirect(buffer, offset, draw_count, stride); });
	}
	void drawIndirectCount(vk::Buffer buffer, vk::DeviceSize offset, vk::Buffer count_buffer,
	                       vk::DeviceSize count_offset, uint32_t max_draw_count,
	                       uint32_t stride) const {
		if (m_stream == nullptr) {
			m_direct.drawIndirectCount(buffer, offset, count_buffer, count_offset, max_draw_count,
			                           stride);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) {
			c.drawIndirectCount(buffer, offset, count_buffer, count_offset, max_draw_count, stride);
		});
	}
	void drawIndexedIndirectCount(vk::Buffer buffer, vk::DeviceSize offset,
	                              vk::Buffer count_buffer, vk::DeviceSize count_offset,
	                              uint32_t max_draw_count, uint32_t stride) const {
		if (m_stream == nullptr) {
			m_direct.drawIndexedIndirectCount(buffer, offset, count_buffer, count_offset,
			                                  max_draw_count, stride);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) {
			c.drawIndexedIndirectCount(buffer, offset, count_buffer, count_offset, max_draw_count,
			                           stride);
		});
	}
	void drawMeshTasksEXT(uint32_t x, uint32_t y, uint32_t z) const {
		if (m_stream == nullptr) {
			m_direct.drawMeshTasksEXT(x, y, z);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.drawMeshTasksEXT(x, y, z); });
	}

	void bindVertexBuffers2(uint32_t first, uint32_t count, const vk::Buffer* buffers,
	                        const vk::DeviceSize* offsets, const vk::DeviceSize* sizes,
	                        const vk::DeviceSize* strides) const;
	void bindIndexBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::IndexType type) const {
		if (m_stream == nullptr) {
			m_direct.bindIndexBuffer(buffer, offset, type);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.bindIndexBuffer(buffer, offset, type); });
	}

	void copyBuffer(vk::Buffer source, vk::Buffer destination, uint32_t count,
	                const vk::BufferCopy* regions) const;
	void copyImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	               vk::ImageLayout destination_layout, uint32_t count,
	               const vk::ImageCopy* regions) const;
	void copyImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	               vk::ImageLayout destination_layout,
	               vk::ArrayProxy<const vk::ImageCopy> const& regions) const {
		copyImage(source, source_layout, destination, destination_layout, regions.size(),
		          regions.data());
	}
	void resolveImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	                  vk::ImageLayout destination_layout, uint32_t count,
	                  const vk::ImageResolve* regions) const;
	void resolveImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	                  vk::ImageLayout destination_layout,
	                  vk::ArrayProxy<const vk::ImageResolve> const& regions) const {
		resolveImage(source, source_layout, destination, destination_layout, regions.size(),
		             regions.data());
	}
	void blitImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	               vk::ImageLayout destination_layout, uint32_t count,
	               const vk::ImageBlit* regions, vk::Filter filter) const;
	void copyImageToBuffer(vk::Image source, vk::ImageLayout source_layout,
	                       vk::Buffer destination, uint32_t count,
	                       const vk::BufferImageCopy* regions) const;
	void copyImageToBuffer(vk::Image source, vk::ImageLayout source_layout,
	                       vk::Buffer                                      destination,
	                       vk::ArrayProxy<const vk::BufferImageCopy> const& regions) const {
		copyImageToBuffer(source, source_layout, destination, regions.size(), regions.data());
	}
	void copyBufferToImage(vk::Buffer source, vk::Image destination,
	                       vk::ImageLayout destination_layout, uint32_t count,
	                       const vk::BufferImageCopy* regions) const;
	void copyBufferToImage(vk::Buffer source, vk::Image destination,
	                       vk::ImageLayout                                  destination_layout,
	                       vk::ArrayProxy<const vk::BufferImageCopy> const& regions) const {
		copyBufferToImage(source, destination, destination_layout, regions.size(), regions.data());
	}
	template <typename F>
	void Custom(F&& function) const {
		if (m_stream == nullptr) {
			function(m_direct);
			return;
		}
		m_stream->Record(std::forward<F>(function));
	}
	template <typename T>
	[[nodiscard]] const T* CopyArray(const T* source, size_t count) const {
		return m_stream == nullptr ? source : m_stream->Copy(source, count);
	}
	void fillBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size,
	                uint32_t value) const {
		if (m_stream == nullptr) {
			m_direct.fillBuffer(buffer, offset, size, value);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.fillBuffer(buffer, offset, size, value); });
	}
	void clearColorImage(vk::Image image, vk::ImageLayout layout, const vk::ClearColorValue* color,
	                     uint32_t range_count, const vk::ImageSubresourceRange* ranges) const;
	void clearDepthStencilImage(vk::Image image, vk::ImageLayout layout,
	                            const vk::ClearDepthStencilValue* value, uint32_t range_count,
	                            const vk::ImageSubresourceRange* ranges) const;
	void clearColorImage(vk::Image image, vk::ImageLayout layout, const vk::ClearColorValue& color,
	                     const vk::ImageSubresourceRange& range) const {
		clearColorImage(image, layout, &color, 1, &range);
	}
	void clearDepthStencilImage(vk::Image image, vk::ImageLayout layout,
	                            const vk::ClearDepthStencilValue& value,
	                            const vk::ImageSubresourceRange& range) const {
		clearDepthStencilImage(image, layout, &value, 1, &range);
	}

	void setViewport(uint32_t first, uint32_t count, const vk::Viewport* viewports) const;
	void setScissor(uint32_t first, uint32_t count, const vk::Rect2D* scissors) const;
	void setViewportWithCount(uint32_t count, const vk::Viewport* viewports) const;
	void setScissorWithCount(uint32_t count, const vk::Rect2D* scissors) const;
	void setLineWidth(float width) const {
		if (m_stream == nullptr) {
			m_direct.setLineWidth(width);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.setLineWidth(width); });
	}
	void setBlendConstants(const float constants[4]) const {
		if (m_stream == nullptr) {
			m_direct.setBlendConstants(constants);
			return;
		}
		struct Values {
			float v[4];
		};
		Values values {};
		std::memcpy(values.v, constants, sizeof(values.v));
		m_stream->Record([=](vk::CommandBuffer c) { c.setBlendConstants(values.v); });
	}
	void setDepthTestEnable(vk::Bool32 enable) const {
		if (m_stream == nullptr) {
			m_direct.setDepthTestEnable(enable);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.setDepthTestEnable(enable); });
	}
	void setDepthWriteEnable(vk::Bool32 enable) const {
		if (m_stream == nullptr) {
			m_direct.setDepthWriteEnable(enable);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.setDepthWriteEnable(enable); });
	}
	void setDepthCompareOp(vk::CompareOp op) const {
		if (m_stream == nullptr) {
			m_direct.setDepthCompareOp(op);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.setDepthCompareOp(op); });
	}
	void setDepthBiasEnable(vk::Bool32 enable) const {
		if (m_stream == nullptr) {
			m_direct.setDepthBiasEnable(enable);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.setDepthBiasEnable(enable); });
	}
	void setDepthBias(float constant, float clamp, float slope) const {
		if (m_stream == nullptr) {
			m_direct.setDepthBias(constant, clamp, slope);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.setDepthBias(constant, clamp, slope); });
	}
	void setCullMode(vk::CullModeFlags mode) const {
		if (m_stream == nullptr) {
			m_direct.setCullMode(mode);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.setCullMode(mode); });
	}
	void setFrontFace(vk::FrontFace face) const {
		if (m_stream == nullptr) {
			m_direct.setFrontFace(face);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.setFrontFace(face); });
	}
	void setDepthBoundsTestEnable(vk::Bool32 enable) const {
		if (m_stream == nullptr) {
			m_direct.setDepthBoundsTestEnable(enable);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.setDepthBoundsTestEnable(enable); });
	}
	void setDepthBounds(float min_bounds, float max_bounds) const {
		if (m_stream == nullptr) {
			m_direct.setDepthBounds(min_bounds, max_bounds);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.setDepthBounds(min_bounds, max_bounds); });
	}
	void setStencilTestEnable(vk::Bool32 enable) const {
		if (m_stream == nullptr) {
			m_direct.setStencilTestEnable(enable);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.setStencilTestEnable(enable); });
	}
	void setStencilOp(vk::StencilFaceFlags faces, vk::StencilOp fail, vk::StencilOp pass,
	                  vk::StencilOp depth_fail, vk::CompareOp compare) const {
		if (m_stream == nullptr) {
			m_direct.setStencilOp(faces, fail, pass, depth_fail, compare);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) {
			c.setStencilOp(faces, fail, pass, depth_fail, compare);
		});
	}
	void setStencilCompareMask(vk::StencilFaceFlags faces, uint32_t mask) const {
		if (m_stream == nullptr) {
			m_direct.setStencilCompareMask(faces, mask);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.setStencilCompareMask(faces, mask); });
	}
	void setStencilWriteMask(vk::StencilFaceFlags faces, uint32_t mask) const {
		if (m_stream == nullptr) {
			m_direct.setStencilWriteMask(faces, mask);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.setStencilWriteMask(faces, mask); });
	}
	void setStencilReference(vk::StencilFaceFlags faces, uint32_t reference) const {
		if (m_stream == nullptr) {
			m_direct.setStencilReference(faces, reference);
			return;
		}
		m_stream->Record([=](vk::CommandBuffer c) { c.setStencilReference(faces, reference); });
	}
	void setAttachmentFeedbackLoopEnableEXT(vk::ImageAspectFlags aspects) const {
		if (m_stream == nullptr) {
			m_direct.setAttachmentFeedbackLoopEnableEXT(aspects);
			return;
		}
		m_stream->Record(
		    [=](vk::CommandBuffer c) { c.setAttachmentFeedbackLoopEnableEXT(aspects); });
	}

private:
	vk::CommandBuffer m_direct = nullptr;
	CommandStream*    m_stream = nullptr;
};

}

#endif
