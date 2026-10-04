#include "application.hpp"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <imgui.h>

#include "math3d.hpp"
#include "octahedron.hpp"

namespace application {

namespace {

// Данные, которые уходят в uniform-буфер (std140: mat4 + vec4 = 80 байт)
struct ObjectUniforms {
	float mvp[4][4];
	float color[4];
};

// Параметры одного объекта, которыми управляет интерфейс (доп. задания 2, 3, 4)
struct SceneObject {
	// Параметры из интерфейса
	float position[3];
	float rotation[3]; // градусы, вокруг X, Y, Z
	float scale[3];
	float color[3];    // умножается на цвет вершин
	bool animate;      // двигать ли объект по траектории
	float phase;       // сдвиг по фазе траектории (чтобы объекты не двигались синхронно)

	// Ресурсы Vulkan: у каждого объекта свой uniform-буфер и свой набор дескрипторов
	// (доп. задание 6)
	VkBuffer uniform_buffer;
	VmaAllocation uniform_allocation;
	ObjectUniforms* uniform_memory;
	VkDescriptorSet descriptor_set;
};

constexpr int OBJECT_COUNT = 2;

// Значения по умолчанию (используются при запуске и по кнопке Reset)
SceneObject makeObject(float x, float s, float phase) {
	SceneObject o = {};
	o.position[0] = x;
	o.scale[0] = o.scale[1] = o.scale[2] = s;
	o.color[0] = o.color[1] = o.color[2] = 1.0f;
	o.animate = true;
	o.phase = phase;
	return o;
}

SceneObject objects[OBJECT_COUNT];

// Общие настройки сцены
bool use_perspective = true; // доп. задание 1
float fov_degrees = 60.0f;
float camera_distance = 7.0f;

// Настройки анимации (доп. задание 3)
bool animation_paused = false;
float animation_speed = 1.0f;
float trajectory_radius = 1.5f;
float animation_time = 0.0f;

// Общие объекты Vulkan
VkShaderModule vertex_shader;
VkShaderModule fragment_shader;

VkDescriptorSetLayout descriptor_set_layout;
VkDescriptorPool descriptor_pool;
VkPipelineLayout pipeline_layout;
VkPipeline pipeline;

VkBuffer vertex_buffer;
VmaAllocation vertex_buffer_allocation;
VkBuffer index_buffer;
VmaAllocation index_buffer_allocation;

// Вспомогательные функции
constexpr float DEG_TO_RAD = 3.14159265358979f / 180.0f;

// Ищет файл шейдера в нескольких папках (зависит от того, откуда запущена программа)
// и создаёт из него шейдерный модуль. name -- например "octahedron.vert.spv".
VkShaderModule loadShaderModule(const char name[]) {
	const char* directories[] = {
		"shaders/",
		"./shaders/",
		"../shaders/",
		"build-debug/shaders/",
		"../build-debug/shaders/",
	};

	std::ifstream file;
	for (const char* directory : directories) {
		const std::string path = std::string(directory) + name;
		file.open(path, std::ios::binary | std::ios::ate);
		if (file.is_open()) {
			std::cout << "Loaded shader: " << path << '\n';
			break;
		}
	}

	if (!file.is_open()) {
		std::cerr << "Failed to find shader file: " << name
		          << " (run the program from the project folder or build-debug)\n";
		return nullptr;
	}

	const size_t size = static_cast<size_t>(file.tellg());
	std::vector<uint32_t> buffer(size / sizeof(uint32_t));

	file.seekg(0);
	file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(size));
	file.close();

	VkShaderModuleCreateInfo info{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = buffer.data(),
	};

	VkShaderModule result;
	if (vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &result) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module: " << name << '\n';
		return nullptr;
	}

	return result;
}

// Создаёт буфер в памяти, доступной с CPU, и сразу копирует туда данные (если data != nullptr).
// Возвращает указатель на отображённую память.
void* createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, const void* data,
                   VkBuffer& buffer, VmaAllocation& allocation) {
	auto& context = graphics::internal::context;

	VkBufferCreateInfo buffer_info{
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	VmaAllocationCreateInfo allocation_info{
		.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
		         VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	VmaAllocationInfo result_info;
	if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info,
	                    &buffer, &allocation, &result_info) != VK_SUCCESS) {
		std::cerr << "Failed to create and allocate buffer\n";
		return nullptr;
	}

	if (data != nullptr) {
		std::memcpy(result_info.pMappedData, data, static_cast<size_t>(size));
	}

	return result_info.pMappedData;
}

} // namespace

// initialize: создаём все объекты Vulkan
bool initialize() {
	auto& context = graphics::internal::context;

	// Начальные параметры объектов
	objects[0] = makeObject(-2.5f, 1.0f, 0.0f);
	objects[1] = makeObject(+2.5f, 0.7f, 3.14159265f);
	objects[1].color[0] = 1.0f;
	objects[1].color[1] = 0.8f;
	objects[1].color[2] = 0.6f;

	// 1. Шейдеры
	vertex_shader = loadShaderModule("octahedron.vert.spv");
	if (!vertex_shader) {
		std::cerr << "Failed to load vertex shader\n";
		return false;
	}

	fragment_shader = loadShaderModule("octahedron.frag.spv");
	if (!fragment_shader) {
		std::cerr << "Failed to load fragment shader\n";
		return false;
	}

	// 2. Вершинный и индексный буферы (общие для обоих объектов)
	if (!createBuffer(sizeof(octahedron_vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
	                  octahedron_vertices, vertex_buffer, vertex_buffer_allocation)) {
		return false;
	}

	if (!createBuffer(sizeof(octahedron_indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
	                  octahedron_indices, index_buffer, index_buffer_allocation)) {
		return false;
	}

	// 3. Uniform-буфер для каждого объекта (размер выровнен до 16 байт)
	for (SceneObject& object : objects) {
		void* memory = createBuffer((sizeof(ObjectUniforms) + 0xf) & ~0xf,
		                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, nullptr,
		                            object.uniform_buffer, object.uniform_allocation);
		if (!memory) {
			return false;
		}
		object.uniform_memory = static_cast<ObjectUniforms*>(memory);
	}

	// 4. Описание набора дескрипторов: один uniform-буфер в binding = 0
	{
		const VkDescriptorSetLayoutBinding bindings[] = {
			{
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
			},
		};

		const VkDescriptorSetLayoutCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = sizeof(bindings) / sizeof(bindings[0]),
			.pBindings = bindings,
		};

		if (vkCreateDescriptorSetLayout(context.device, &info, nullptr,
		                                &descriptor_set_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create descriptor set layout\n";
			return false;
		}
	}

	// 5. Пул дескрипторов: на каждый объект по одному набору и по одному uniform-буферу
	{
		const VkDescriptorPoolSize pool_sizes[] = {
			{
				.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = OBJECT_COUNT,
			},
		};

		const VkDescriptorPoolCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.maxSets = OBJECT_COUNT,
			.poolSizeCount = sizeof(pool_sizes) / sizeof(pool_sizes[0]),
			.pPoolSizes = pool_sizes,
		};

		if (vkCreateDescriptorPool(context.device, &info, nullptr, &descriptor_pool) != VK_SUCCESS) {
			std::cerr << "Failed to create descriptor pool\n";
			return false;
		}
	}

	// 6. Выделяем наборы дескрипторов и привязываем к ним буферы объектов
	for (SceneObject& object : objects) {
		const VkDescriptorSetAllocateInfo allocate_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = descriptor_pool,
			.descriptorSetCount = 1,
			.pSetLayouts = &descriptor_set_layout,
		};

		if (vkAllocateDescriptorSets(context.device, &allocate_info,
		                             &object.descriptor_set) != VK_SUCCESS) {
			std::cerr << "Failed to allocate descriptor set\n";
			return false;
		}

		const VkDescriptorBufferInfo buffer_descriptor{
			.buffer = object.uniform_buffer,
			.offset = 0,
			.range = sizeof(ObjectUniforms),
		};

		const VkWriteDescriptorSet write{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = object.descriptor_set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &buffer_descriptor,
		};

		vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
	}

	// 7. Layout конвейера (знает про набор дескрипторов)
	{
		const VkPipelineLayoutCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &descriptor_set_layout,
		};

		if (vkCreatePipelineLayout(context.device, &info, nullptr, &pipeline_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create pipeline layout\n";
			return false;
		}
	}

	// 8. Графический конвейер
	{
		const VkPipelineShaderStageCreateInfo stage_infos[2] = {
			{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_VERTEX_BIT,
				.module = vertex_shader,
				.pName = "main",
			},
			{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
				.module = fragment_shader,
				.pName = "main",
			},
		};

		// Откуда брать вершины: один буфер, шаг -- одна структура Vertex
		const VkVertexInputBindingDescription vertex_bindings[] = {
			{
				.binding = 0,
				.stride = sizeof(Vertex),
				.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
			},
		};

		// Атрибуты: location 0 -- позиция, location 1 -- цвет
		const VkVertexInputAttributeDescription vertex_attributes[] = {
			{
				.location = 0,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, position),
			},
			{
				.location = 1,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, color),
			},
		};

		const VkPipelineVertexInputStateCreateInfo input_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = sizeof(vertex_bindings) / sizeof(vertex_bindings[0]),
			.pVertexBindingDescriptions = vertex_bindings,
			.vertexAttributeDescriptionCount = sizeof(vertex_attributes) / sizeof(vertex_attributes[0]),
			.pVertexAttributeDescriptions = vertex_attributes,
		};

		const VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		};

		const VkPipelineViewportStateCreateInfo viewport_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
			.viewportCount = 1,
			.scissorCount = 1,
		};

		// Лицевые грани у нас идут против часовой стрелки (см. octahedron.hpp)
		const VkPipelineRasterizationStateCreateInfo raster_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_BACK_BIT,
			.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
			.lineWidth = 1.0f,
		};

		const VkPipelineMultisampleStateCreateInfo sample_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
		};

		const VkPipelineDepthStencilStateCreateInfo depth_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = true,
			.depthWriteEnable = true,
			.depthCompareOp = VK_COMPARE_OP_LESS,
		};

		const VkPipelineColorBlendAttachmentState attachment_info{
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
			                  VK_COLOR_COMPONENT_G_BIT |
			                  VK_COLOR_COMPONENT_B_BIT |
			                  VK_COLOR_COMPONENT_A_BIT,
		};

		const VkPipelineColorBlendStateCreateInfo blend_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
			.attachmentCount = 1,
			.pAttachments = &attachment_info,
		};

		const VkDynamicState dynamic_states[] = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR,
		};

		const VkPipelineDynamicStateCreateInfo dynamic_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
			.dynamicStateCount = sizeof(dynamic_states) / sizeof(dynamic_states[0]),
			.pDynamicStates = dynamic_states,
		};

		const VkGraphicsPipelineCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = 2,
			.pStages = stage_infos,
			.pVertexInputState = &input_state_info,
			.pInputAssemblyState = &assembly_state_info,
			.pViewportState = &viewport_info,
			.pRasterizationState = &raster_info,
			.pMultisampleState = &sample_info,
			.pDepthStencilState = &depth_info,
			.pColorBlendState = &blend_info,
			.pDynamicState = &dynamic_state_info,
			.layout = pipeline_layout,
			.renderPass = context.render_pass,
		};

		if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline) != VK_SUCCESS) {
			std::cerr << "Failed to create graphics pipeline\n";
			return false;
		}
	}

	return true;
}

// shutdown: уничтожаем всё в обратном порядке
void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	vkDestroyPipeline(context.device, pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);

	// Наборы дескрипторов освобождаются вместе с пулом
	vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, descriptor_set_layout, nullptr);

	for (SceneObject& object : objects) {
		vmaDestroyBuffer(context.allocator, object.uniform_buffer, object.uniform_allocation);
	}

	vmaDestroyBuffer(context.allocator, index_buffer, index_buffer_allocation);
	vmaDestroyBuffer(context.allocator, vertex_buffer, vertex_buffer_allocation);

	vkDestroyShaderModule(context.device, fragment_shader, nullptr);
	vkDestroyShaderModule(context.device, vertex_shader, nullptr);
}

// update: интерфейс ImGui + пересчёт матриц
void update([[maybe_unused]] double time) {
	auto& context = graphics::internal::context;

	// Интерфейс
	ImGui::Begin("Octahedron");

	if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
		// Доп. задание 1: переключение проекции
		if (ImGui::RadioButton("Perspective", use_perspective)) {
			use_perspective = true;
		}
		ImGui::SameLine();
		if (ImGui::RadioButton("Orthographic", !use_perspective)) {
			use_perspective = false;
		}

		if (use_perspective) {
			ImGui::SliderFloat("FOV (deg)", &fov_degrees, 20.0f, 120.0f);
		}
		ImGui::SliderFloat("Camera distance", &camera_distance, 3.0f, 20.0f);
	}

	if (ImGui::CollapsingHeader("Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
		// Доп. задание 3: пауза, скорость, параметры траектории
		ImGui::Checkbox("Pause", &animation_paused);
		ImGui::SliderFloat("Speed", &animation_speed, 0.0f, 5.0f);
		ImGui::SliderFloat("Trajectory radius", &trajectory_radius, 0.0f, 3.0f);
	}

	for (int i = 0; i < OBJECT_COUNT; ++i) {
		SceneObject& object = objects[i];

		ImGui::PushID(i);
		const char* title = (i == 0) ? "Object 1" : "Object 2";
		if (ImGui::CollapsingHeader(title, ImGuiTreeNodeFlags_DefaultOpen)) {
			// Доп. задание 2: позиция, поворот, растяжение
			ImGui::DragFloat3("Position", object.position, 0.05f);
			ImGui::SliderFloat3("Rotation (deg)", object.rotation, -180.0f, 180.0f);
			ImGui::DragFloat3("Scale", object.scale, 0.02f, 0.1f, 5.0f);
			// Доп. задание 4: цвет
			ImGui::ColorEdit3("Color", object.color);
			ImGui::Checkbox("Animate", &object.animate);
		}
		ImGui::PopID();
	}

	ImGui::End();

	// Время анимации (копим сами, чтобы работали пауза и смена скорости) ---
	if (!animation_paused) {
		animation_time += ImGui::GetIO().DeltaTime * animation_speed;
	}

	// Матрицы
	const float aspect = float(context.swapchain_extent.width) /
	                     float(context.swapchain_extent.height);

	Mat4 projection;
	if (use_perspective) {
		projection = mat4Perspective(fov_degrees * DEG_TO_RAD, aspect, 0.1f, 100.0f);
	} else {
		// Видимая высота такая же, как у перспективной проекции на расстоянии камеры
		const float visible_height = 2.0f * camera_distance * std::tan(fov_degrees * DEG_TO_RAD * 0.5f);
		projection = mat4Ortho(visible_height, aspect, 0.1f, 100.0f);
	}

	// Камера стоит в (0, 0, camera_distance) и смотрит на начало координат
	const Mat4 view = mat4Translate(0.0f, 0.0f, -camera_distance);
	const Mat4 view_projection = mat4Mul(projection, view);

	for (SceneObject& object : objects) {
		float x = object.position[0];
		float y = object.position[1];
		float z = object.position[2];
		float rx = object.rotation[0] * DEG_TO_RAD;
		float ry = object.rotation[1] * DEG_TO_RAD;
		float rz = object.rotation[2] * DEG_TO_RAD;

		if (object.animate) {
			// Сложная траектория: фигура-«восьмёрка» (кривая Лиссажу) в пространстве
			// плюс вращение вокруг своих осей.
			const float t = animation_time + object.phase;
			x += trajectory_radius * std::cos(t);
			y += trajectory_radius * 0.5f * std::sin(2.0f * t);
			z += trajectory_radius * std::sin(t);

			rx += t;
			ry += t * 1.5f;
		}

		// Модельная матрица: масштаб -> поворот -> перенос
		const Mat4 rotation = mat4Mul(mat4RotateZ(rz), mat4Mul(mat4RotateY(ry), mat4RotateX(rx)));
		const Mat4 scale = mat4Scale(object.scale[0], object.scale[1], object.scale[2]);
		const Mat4 model = mat4Mul(mat4Translate(x, y, z), mat4Mul(rotation, scale));

		const Mat4 mvp = mat4Mul(view_projection, model);

		ObjectUniforms uniforms = {};
		std::memcpy(uniforms.mvp, mvp.m, sizeof(uniforms.mvp));
		uniforms.color[0] = object.color[0];
		uniforms.color[1] = object.color[1];
		uniforms.color[2] = object.color[2];
		uniforms.color[3] = 1.0f;

		std::memcpy(object.uniform_memory, &uniforms, sizeof(uniforms));
	}
}

// render: записываем команды для GPU
void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;

	vkResetCommandBuffer(fd.command_buffer, 0);

	const VkCommandBufferBeginInfo command_buffer_begin{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};

	vkBeginCommandBuffer(fd.command_buffer, &command_buffer_begin);

	const VkClearValue clear_values[] = {
		{ .color = { .float32 = { 0.1f, 0.1f, 0.1f, 1.0f } } },
		{ .depthStencil = { 1.0f, 0 } },
	};

	const VkRenderPassBeginInfo render_pass_begin{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .extent = context.swapchain_extent },
		.clearValueCount = sizeof(clear_values) / sizeof(clear_values[0]),
		.pClearValues = clear_values,
	};

	vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

	const VkViewport viewport{
		.x = 0.0f,
		.y = 0.0f,
		.width = float(context.swapchain_extent.width),
		.height = float(context.swapchain_extent.height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};

	const VkRect2D scissor{ .extent = context.swapchain_extent };

	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

	const VkDeviceSize vertex_buffer_offset = 0;
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer, &vertex_buffer_offset);
	vkCmdBindIndexBuffer(fd.command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT32);

	// Один и тот же меш рисуем дважды, меняя только набор дескрипторов
	// (у каждого объекта свой uniform-буфер -> своя матрица и цвет).
	for (const SceneObject& object : objects) {
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
		                        pipeline_layout, 0, 1, &object.descriptor_set, 0, nullptr);

		vkCmdDrawIndexed(fd.command_buffer, octahedron_index_count, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(fd.command_buffer);

	vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application