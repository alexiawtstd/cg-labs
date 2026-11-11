#include "application.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <imgui.h>

#include "geometry.hpp"
#include "math3d.hpp"
#include "texture.hpp"   // лаб. 3: загрузка текстур, сэмплер
#include "vk_utils.hpp"  // лаб. 3: буферы в видеопамяти, одноразовые команды

namespace application {

namespace {

constexpr float PI = 3.14159265358979f;
constexpr float DEG_TO_RAD = PI / 180.0f;

// ===========================================================================
// структуры, которые копируются в буферы gpu.
// раскладка полей должна совпадать с объявлениями в шейдерах
// ===========================================================================

// set = 0, binding = 0: uniform-буфер сцены (std140, 128 байт).
// по std140 vec3 занимает 12 байт, но выравнивается на 16, поэтому после vec3
// может стоять одиночный float/uint - он займёт оставшиеся 4 байта
struct SceneUniforms {
	float view_projection[4][4];         // 0
	float camera_position[3];            // 64
	float _pad0;
	float directional_light_direction[3];// 80
	float _pad1;
	float directional_light_color[3];    // 96
	uint32_t point_light_count;          // 108
	float ambient_light[3];              // 112
	uint32_t spot_light_count;           // 124
};
static_assert(sizeof(SceneUniforms) == 128, "scene uniforms must match std140 layout");

// set = 1, binding = 0: uniform-буфер модели (std140, 176 байт).
// лаб. 3: буфер ДИНАМИЧЕСКИЙ - в нём лежит по одной такой структуре на каждый объект,
// а нужная выбирается смещением (dynamic offset) в vkCmdBindDescriptorSets
struct ModelUniforms {
	float model[4][4];                   // 0
	float normal_matrix[4][4];           // 64
	float albedo[3];                     // 128
	float shininess;                     // 140
	float specular[3];                   // 144
	float use_vertex_color;              // 156
	float uv_scale[2];                   // 160  лаб. 3: во сколько раз повторить текстуру
	float emissive_strength;             // 168  лаб. 3: яркость самосвечения
	float _pad0;                         // 172
};
static_assert(sizeof(ModelUniforms) == 176, "model uniforms must match std140 layout");

// элемент storage-буфера точечных источников (std430, 32 байта)
struct PointLightGPU {
	float position[3];
	float intensity;
	float color[3];
	float _pad;
};
static_assert(sizeof(PointLightGPU) == 32, "point light must match std430 layout");

// элемент storage-буфера прожекторов (std430, 48 байт)
struct SpotLightGPU {
	float position[3];
	float intensity;
	float color[3];
	float cos_inner;
	float direction[3];
	float cos_outer;
};
static_assert(sizeof(SpotLightGPU) == 48, "spot light must match std430 layout");

// push-константы для визуализации источников
struct GizmoPushConstants {
	float model[4][4];
	float color[4];
};

// лаб. 3: push-константы фрагментного шейдера освещаемых объектов (8 байт).
// меняются от объекта к объекту без всяких буферов
struct LitPushConstants {
	uint32_t flags;   // бит 0 - включить нетривиальное сэмплирование текстуры
	float time;       // время анимации в секундах
};
constexpr uint32_t FLAG_ANIMATED_SAMPLING = 1u;

// ===========================================================================
// буферы и меши
// ===========================================================================

// буфер в памяти, доступной с cpu (постоянно отображён в адресное пространство)
struct HostBuffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = nullptr;
	void* memory = nullptr;
	VkDeviceSize size = 0;
};

// лаб. 3: вершины и индексы лежат в видеопамяти (VRAM), а не в памяти, доступной процессору.
// загружаются один раз через промежуточный staging-буфер (см. vk_utils.cpp)
struct Mesh {
	vkutil::Buffer vertices;
	vkutil::Buffer indices;
	uint32_t index_count = 0;
};

// ===========================================================================
// камера
// ===========================================================================

struct Camera {
	float position[3];
	float yaw;           // градусы
	float pitch;         // градусы
	float fov;           // градусы
	float near_z;
	float far_z;
	bool perspective;    // из лабы 1: перспективная или ортографическая проекция
	float ortho_height;  // видимая высота для ортографической проекции
	float move_speed;    // единиц в секунду
	float mouse_sensitivity;
};

Camera camera;

// ===========================================================================
// источники света
// ===========================================================================

struct AmbientLight {
	float color[3];
	float intensity;
};

struct DirectionalLight {
	bool enabled;
	float yaw;           // градусы, куда светит
	float pitch;         // градусы
	float color[3];
	float intensity;
};

struct PointLight {
	bool enabled;
	float position[3];
	float color[3];
	float intensity;
	bool orbit;          // вращаться вокруг вертикальной оси сцены
	float orbit_speed;   // радиан в секунду
};

struct SpotLight {
	bool enabled;
	float position[3];
	float yaw;           // градусы, направление прожектора
	float pitch;         // градусы
	float color[3];
	float intensity;
	float inner_angle;   // градусы, от оси до края яркой области
	float outer_angle;   // градусы, от оси до края освещаемой области
	bool follow_camera;  // "фонарик": прикреплён к камере
};

AmbientLight ambient;
DirectionalLight directional;
std::vector<PointLight> point_lights;
std::vector<SpotLight> spot_lights;

// визуализация источников (доп. задание 4)
bool show_light_gizmos = true;
float gizmo_size = 0.12f;
float spot_cone_length = 1.5f;

// анимация
bool animation_paused = false;
float animation_speed = 1.0f;
float animation_time = 0.0f;

// ===========================================================================
// объекты сцены
// ===========================================================================

// лаб. 3: материал в смысле текстур - набор из трёх изображений.
// albedo - основной цвет, specular - маска блеска (белое блестит), emissive - самосвечение.
// если specular или emissive нет (""), вместо них подставляется чёрная текстура 1x1
struct TextureMaterialDescription {
	const char* name;
	const char* albedo;
	const char* specular;
	const char* emissive;
};

constexpr TextureMaterialDescription texture_material_descriptions[] = {
	{ "Lava rock",     "lava_albedo.png",    "lava_specular.png",    "lava_emissive.png" },
	{ "Bricks",        "bricks_albedo.png",  "bricks_specular.png",  "" },
	{ "Checker tiles", "checker_albedo.png", "checker_specular.png", "" },
	{ "Wood planks",   "wood_albedo.png",    "wood_specular.png",    "" },
};
constexpr int TEXTURE_MATERIAL_COUNT = int(sizeof(texture_material_descriptions) / sizeof(texture_material_descriptions[0]));

struct TextureMaterial {
	Texture albedo;
	Texture specular;
	Texture emissive;
	VkDescriptorSet set = VK_NULL_HANDLE;   // set = 2: три combined image sampler
};

// шейдинг-материал из лабы 2 (параметры блинна-фонга)
struct Material {
	float albedo[3];     // теперь это оттенок, умножаемый на цвет из текстуры
	float specular[3];
	float shininess;
	bool use_vertex_color;
};

struct SceneObject {
	std::string name;
	Mesh* mesh;

	// преобразование (из лабы 1)
	float position[3];
	float rotation[3];   // градусы
	float scale[3];

	Material material;

	// анимация: вращение вокруг оси y и покачивание вверх-вниз
	bool animate;
	float spin_speed;
	float bob_height;
	float phase;

	// лаб. 3: текстуры и их параметры
	int texture_material;         // индекс в texture_material_descriptions
	float uv_scale[2];            // повторение текстуры по u и v
	float emissive_strength;      // множитель текстуры самосвечения
	bool animated_sampling;       // нетривиальное (анимированное) чтение текстуры

	// данные, посчитанные в update и копируемые на gpu в render
	ModelUniforms uniforms;
};

std::vector<SceneObject> objects;

// ===========================================================================
// объекты vulkan
// ===========================================================================

VkShaderModule lit_vertex_shader;
VkShaderModule lit_fragment_shader;
VkShaderModule gizmo_vertex_shader;
VkShaderModule gizmo_fragment_shader;

// два вида наборов дескрипторов (доп. задание 1)
VkDescriptorSetLayout scene_set_layout; // set = 0: uniform сцены + 2 storage-буфера
VkDescriptorSetLayout model_set_layout; // set = 1: ДИНАМИЧЕСКИЙ uniform модели (лаб. 3)
VkDescriptorSetLayout material_set_layout; // set = 2: три текстуры материала (лаб. 3)
VkDescriptorPool descriptor_pool;
VkDescriptorSet scene_descriptor_set;
VkDescriptorSet model_descriptor_set;   // один набор на ВСЕ объекты (динамическое смещение выбирает структуру)

VkPipelineLayout lit_pipeline_layout;
VkPipelineLayout gizmo_pipeline_layout;
VkPipeline lit_pipeline;
VkPipeline gizmo_pipeline;   // треугольники (сферы источников)
VkPipeline line_pipeline;    // линии (конусы прожекторов)

HostBuffer scene_uniform_buffer;
HostBuffer model_buffer;       // лаб. 3: массив ModelUniforms всех объектов (с выравниванием)
VkDeviceSize model_stride = 0; // шаг массива: sizeof(ModelUniforms), выровненный по пределу видеокарты
HostBuffer point_light_buffer;
HostBuffer spot_light_buffer;
HostBuffer line_vertex_buffer;

Mesh octahedron_mesh;
Mesh floor_mesh;
Mesh quad_mesh;      // лаб. 3: квадрат для стен
Mesh sphere_mesh;

// лаб. 3: текстуры
VkSampler texture_sampler = VK_NULL_HANDLE;   // правила чтения текстур, общий для всех
Texture black_texture;                        // чёрный пиксель - заглушка для отсутствующих specular/emissive
TextureMaterial texture_materials[TEXTURE_MATERIAL_COUNT];

// данные кадра, подготовленные в update и загружаемые на gpu в render
SceneUniforms scene_uniforms;
std::vector<PointLightGPU> point_light_data;
std::vector<SpotLightGPU> spot_light_data;
std::vector<GizmoPushConstants> gizmo_draws;
std::vector<Vertex> line_vertices;

constexpr VkBufferUsageFlags STORAGE_USAGE = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

// ===========================================================================
// вспомогательные функции
// ===========================================================================

// ищет файл шейдера в нескольких папках и создаёт из него шейдерный модуль
VkShaderModule loadShaderModule(const char name[]) {
	const char* directories[] = {
		SHADER_DIR,      // лаб. 3: абсолютный путь, его подставляет CMake
		"shaders/",
		"./shaders/",
		"../shaders/",
		"../../shaders/",
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

// создаёт буфер в памяти, доступной с cpu, и копирует туда данные (если data != nullptr)
bool createHostBuffer(VkDeviceSize size, VkBufferUsageFlags usage, const void* data, HostBuffer& out) {
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
	                    &out.buffer, &out.allocation, &result_info) != VK_SUCCESS) {
		std::cerr << "Failed to create and allocate buffer\n";
		return false;
	}

	out.memory = result_info.pMappedData;
	out.size = size;

	if (data != nullptr) {
		std::memcpy(out.memory, data, static_cast<size_t>(size));
	}

	return true;
}

void destroyHostBuffer(HostBuffer& buffer) {
	if (buffer.buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(graphics::internal::context.allocator, buffer.buffer, buffer.allocation);
	}
	buffer = {};
}

// увеличивает буфер, если в него не помещается needed байт.
// размер растёт степенями двойки, чтобы не пересоздавать буфер при каждом добавлении.
// возвращает true, если буфер был пересоздан (тогда нужно обновить дескриптор)
bool ensureCapacity(HostBuffer& buffer, VkDeviceSize needed, VkBufferUsageFlags usage) {
	if (buffer.buffer != VK_NULL_HANDLE && buffer.size >= needed) {
		return false;
	}

	VkDeviceSize new_size = 256;
	while (new_size < needed) {
		new_size *= 2;
	}

	// вызывается только из render, когда gpu уже закончил предыдущий кадр
	// (prepare() ждёт fence), поэтому старый буфер можно безопасно удалить
	destroyHostBuffer(buffer);
	createHostBuffer(new_size, usage, nullptr, buffer);
	return true;
}

// лаб. 3: меш загружается в видеопамять через staging-буфер
bool createMesh(const MeshData& data, Mesh& mesh) {
	if (!vkutil::createDeviceBuffer(data.vertices.data(), data.vertices.size() * sizeof(Vertex),
	                                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, mesh.vertices)) {
		return false;
	}
	if (!vkutil::createDeviceBuffer(data.indices.data(), data.indices.size() * sizeof(uint32_t),
	                                VK_BUFFER_USAGE_INDEX_BUFFER_BIT, mesh.indices)) {
		return false;
	}
	mesh.index_count = uint32_t(data.indices.size());
	return true;
}

void destroyMesh(Mesh& mesh) {
	vkutil::destroyBuffer(mesh.vertices);
	vkutil::destroyBuffer(mesh.indices);
	mesh.index_count = 0;
}

// записывает буфер в дескриптор набора
void writeBufferDescriptor(VkDescriptorSet set, uint32_t binding, VkDescriptorType type,
                           VkBuffer buffer, VkDeviceSize range) {
	const VkDescriptorBufferInfo buffer_info{
		.buffer = buffer,
		.offset = 0,
		.range = range,
	};

	const VkWriteDescriptorSet write{
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = set,
		.dstBinding = binding,
		.descriptorCount = 1,
		.descriptorType = type,
		.pBufferInfo = &buffer_info,
	};

	vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
}

// создаёт графический конвейер. у всех конвейеров один формат вершин (Vertex)
VkPipeline createPipeline(VkShaderModule vertex_shader, VkShaderModule fragment_shader,
                          VkPipelineLayout layout, VkPrimitiveTopology topology,
                          VkCullModeFlags cull_mode) {
	auto& context = graphics::internal::context;

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

	const VkVertexInputBindingDescription vertex_bindings[] = {
		{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		},
	};

	// атрибуты: location 0 - позиция, location 1 - нормаль, location 2 - цвет,
	// location 3 - текстурные координаты (лаб. 3)
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
			.offset = offsetof(Vertex, normal),
		},
		{
			.location = 2,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, color),
		},
		{
			.location = 3,
			.binding = 0,
			.format = VK_FORMAT_R32G32_SFLOAT,
			.offset = offsetof(Vertex, uv),
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
		.topology = topology,
	};

	const VkPipelineViewportStateCreateInfo viewport_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	// лицевые грани идут против часовой стрелки (см. лабу 1)
	const VkPipelineRasterizationStateCreateInfo raster_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = cull_mode,
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
		.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
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
		.layout = layout,
		.renderPass = context.render_pass,
	};

	VkPipeline pipeline = VK_NULL_HANDLE;
	if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline) != VK_SUCCESS) {
		std::cerr << "Failed to create graphics pipeline\n";
		return VK_NULL_HANDLE;
	}
	return pipeline;
}

void copyMat4(const Mat4& m, float out[4][4]) {
	std::memcpy(out, m.m, sizeof(m.m));
}

void copyColor(const float in[3], float scale, float out[3]) {
	out[0] = in[0] * scale;
	out[1] = in[1] * scale;
	out[2] = in[2] * scale;
}

// ===========================================================================
// начальное состояние сцены
// ===========================================================================

SceneObject makeObject(const char* name, Mesh* mesh, Vec3 position, float scale,
                       Vec3 albedo, Vec3 specular, float shininess, int texture_material) {
	SceneObject o = {};
	o.name = name;
	o.mesh = mesh;
	storeVec3(position, o.position);
	o.scale[0] = o.scale[1] = o.scale[2] = scale;
	storeVec3(albedo, o.material.albedo);
	storeVec3(specular, o.material.specular);
	o.material.shininess = shininess;
	o.material.use_vertex_color = false;
	o.animate = true;
	o.spin_speed = 0.6f;
	o.bob_height = 0.25f;
	o.texture_material = texture_material;
	o.uv_scale[0] = o.uv_scale[1] = 1.0f;
	o.emissive_strength = 0.0f;
	o.animated_sampling = false;
	return o;
}

PointLight makePointLight(Vec3 position, Vec3 color, float intensity) {
	PointLight l = {};
	l.enabled = true;
	storeVec3(position, l.position);
	storeVec3(color, l.color);
	l.intensity = intensity;
	l.orbit = true;
	l.orbit_speed = 0.7f;
	return l;
}

SpotLight makeSpotLight(Vec3 position, float yaw, float pitch, Vec3 color, float intensity) {
	SpotLight l = {};
	l.enabled = true;
	storeVec3(position, l.position);
	l.yaw = yaw;
	l.pitch = pitch;
	storeVec3(color, l.color);
	l.intensity = intensity;
	l.inner_angle = 15.0f;
	l.outer_angle = 25.0f;
	l.follow_camera = false;
	return l;
}

void resetCamera() {
	camera = {};
	camera.position[0] = 0.0f;
	camera.position[1] = 3.5f;
	camera.position[2] = 9.0f;
	camera.yaw = 0.0f;
	camera.pitch = -15.0f;
	camera.fov = 60.0f;
	camera.near_z = 0.1f;
	camera.far_z = 100.0f;
	camera.perspective = true;
	camera.ortho_height = 8.0f;
	camera.move_speed = 4.0f;
	camera.mouse_sensitivity = 0.2f;
}

void resetLights() {
	ambient = { { 0.55f, 0.65f, 1.0f }, 0.08f };

	directional = {};
	directional.enabled = true;
	directional.yaw = 35.0f;
	directional.pitch = -50.0f;
	directional.color[0] = 1.0f;
	directional.color[1] = 0.95f;
	directional.color[2] = 0.85f;
	directional.intensity = 0.25f;

	point_lights.clear();
	point_lights.push_back(makePointLight({ 3.0f, 1.2f, 2.0f }, { 1.0f, 0.25f, 0.1f }, 8.0f));
	point_lights.push_back(makePointLight({ -3.0f, 2.5f, -2.0f }, { 0.1f, 0.6f, 1.0f }, 10.0f));
	point_lights[1].orbit_speed = -0.5f;

	spot_lights.clear();
	// светит сверху-спереди на центральный октаэдр
	spot_lights.push_back(makeSpotLight({ 0.0f, 7.0f, 3.5f }, 0.0f, -63.0f, { 1.0f, 0.9f, 0.7f }, 45.0f));
	// фиолетовый прожектор сбоку
	spot_lights.push_back(makeSpotLight({ 6.0f, 4.0f, -1.0f }, -90.0f, -35.0f, { 0.8f, 0.3f, 1.0f }, 35.0f));
	spot_lights[1].inner_angle = 10.0f;
	spot_lights[1].outer_angle = 30.0f;
}

void resetObjects() {
	// материалы по номерам: 0 Lava rock, 1 Bricks, 2 Checker tiles, 3 Wood planks

	// комната: пол и стены (лаб. 3, творческое задание). не двигаются.
	// uv_scale повторяет текстуру, чтобы кирпичи и плитка не растягивались на 24 метра
	SceneObject floor = makeObject("Floor", &floor_mesh, { 0.0f, 0.0f, 0.0f }, 1.0f,
	                               { 0.9f, 0.9f, 0.9f }, { 0.5f, 0.5f, 0.5f }, 32.0f, 2);
	floor.animate = false;
	floor.uv_scale[0] = floor.uv_scale[1] = 6.0f;

	// стены - один и тот же квадрат 1 x 1, растянутый и повёрнутый матрицей модели
	SceneObject back_wall = makeObject("Back wall", &quad_mesh, { 0.0f, 4.0f, -12.0f }, 1.0f,
	                                   { 0.8f, 0.8f, 0.8f }, { 0.3f, 0.3f, 0.3f }, 16.0f, 1);
	back_wall.scale[0] = 24.0f;
	back_wall.scale[1] = 8.0f;
	back_wall.animate = false;
	back_wall.uv_scale[0] = 6.0f;
	back_wall.uv_scale[1] = 2.0f;

	SceneObject left_wall = makeObject("Left wall", &quad_mesh, { -12.0f, 4.0f, 0.0f }, 1.0f,
	                                   { 0.8f, 0.8f, 0.8f }, { 0.3f, 0.3f, 0.3f }, 16.0f, 3);
	left_wall.rotation[1] = 90.0f;   // нормаль +z -> +x (смотрит в комнату)
	left_wall.scale[0] = 24.0f;
	left_wall.scale[1] = 8.0f;
	left_wall.animate = false;
	left_wall.uv_scale[0] = 6.0f;
	left_wall.uv_scale[1] = 2.0f;

	SceneObject right_wall = makeObject("Right wall", &quad_mesh, { 12.0f, 4.0f, 0.0f }, 1.0f,
	                                    { 0.8f, 0.8f, 0.8f }, { 0.3f, 0.3f, 0.3f }, 16.0f, 3);
	right_wall.rotation[1] = -90.0f; // нормаль +z -> -x
	right_wall.scale[0] = 24.0f;
	right_wall.scale[1] = 8.0f;
	right_wall.animate = false;
	right_wall.uv_scale[0] = 6.0f;
	right_wall.uv_scale[1] = 2.0f;

	// центральный октаэдр: лава + самосвечение + нетривиальное сэмплирование
	SceneObject ramiel = makeObject("Octahedron (center)", &octahedron_mesh, { 0.0f, 2.2f, 0.0f }, 1.4f,
	                                { 1.0f, 1.0f, 1.0f }, { 1.0f, 1.0f, 1.0f }, 96.0f, 0);
	ramiel.spin_speed = 0.4f;
	ramiel.emissive_strength = 1.0f;
	ramiel.animated_sampling = true;

	SceneObject left = makeObject("Octahedron (left)", &octahedron_mesh, { -3.5f, 1.3f, 1.0f }, 0.8f,
	                              { 1.0f, 1.0f, 1.0f }, { 0.6f, 0.6f, 0.6f }, 32.0f, 1);
	left.phase = 2.0f;
	left.spin_speed = -0.9f;

	SceneObject right = makeObject("Octahedron (right)", &octahedron_mesh, { 3.5f, 1.3f, 1.0f }, 0.8f,
	                               { 1.0f, 0.95f, 0.85f }, { 1.0f, 0.9f, 0.6f }, 160.0f, 3);
	right.phase = 4.0f;
	right.spin_speed = 1.1f;

	// у объектов больше нет собственных ресурсов vulkan (один общий динамический буфер),
	// поэтому сброс - это просто присваивание
	objects = { floor, back_wall, left_wall, right_wall, ramiel, left, right };
}

// ===========================================================================
// управление камерой с клавиатуры и мыши
// ===========================================================================

void updateCameraInput(float dt) {
	ImGuiIO& io = ImGui::GetIO();

	// поворот: зажать правую кнопку мыши и двигать
	if (!io.WantCaptureMouse && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
		camera.yaw += io.MouseDelta.x * camera.mouse_sensitivity;
		camera.pitch -= io.MouseDelta.y * camera.mouse_sensitivity;
	}

	// колесо мыши меняет угол обзора
	if (!io.WantCaptureMouse && io.MouseWheel != 0.0f) {
		camera.fov -= io.MouseWheel * 2.0f;
	}

	// ограничиваем углы, чтобы камера не переворачивалась
	camera.pitch = std::clamp(camera.pitch, -89.0f, 89.0f);
	camera.fov = std::clamp(camera.fov, 10.0f, 120.0f);
	if (camera.yaw > 180.0f) camera.yaw -= 360.0f;
	if (camera.yaw < -180.0f) camera.yaw += 360.0f;

	if (io.WantCaptureKeyboard) {
		return;
	}

	// перемещение: wasd - вперёд/влево/назад/вправо, e/q - вверх/вниз, shift - быстрее
	const Vec3 forward = directionFromYawPitch(camera.yaw * DEG_TO_RAD, camera.pitch * DEG_TO_RAD);
	const Vec3 right = normalize(cross(forward, { 0.0f, 1.0f, 0.0f }));
	const Vec3 up = { 0.0f, 1.0f, 0.0f };

	Vec3 move = { 0.0f, 0.0f, 0.0f };
	if (ImGui::IsKeyDown(ImGuiKey_W)) move = move + forward;
	if (ImGui::IsKeyDown(ImGuiKey_S)) move = move - forward;
	if (ImGui::IsKeyDown(ImGuiKey_D)) move = move + right;
	if (ImGui::IsKeyDown(ImGuiKey_A)) move = move - right;
	if (ImGui::IsKeyDown(ImGuiKey_E)) move = move + up;
	if (ImGui::IsKeyDown(ImGuiKey_Q)) move = move - up;

	float speed = camera.move_speed;
	if (ImGui::IsKeyDown(ImGuiKey_LeftShift)) {
		speed *= 3.0f;
	}

	if (length(move) > 0.0f) {
		const Vec3 position = toVec3(camera.position) + normalize(move) * (speed * dt);
		storeVec3(position, camera.position);
	}
}

// ===========================================================================
// интерфейс
// ===========================================================================

void drawSceneWindow() {
	ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(340, 520), ImGuiCond_FirstUseEver);
	ImGui::Begin("Camera & Objects");

	ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);
	ImGui::TextDisabled("RMB + mouse: look, WASD/QE: move, Shift: faster, wheel: FOV");

	if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::DragFloat3("Position", camera.position, 0.05f);
		ImGui::SliderFloat("Yaw (deg)", &camera.yaw, -180.0f, 180.0f);
		ImGui::SliderFloat("Pitch (deg)", &camera.pitch, -89.0f, 89.0f);

		// из лабы 1: переключение проекции
		if (ImGui::RadioButton("Perspective", camera.perspective)) {
			camera.perspective = true;
		}
		ImGui::SameLine();
		if (ImGui::RadioButton("Orthographic", !camera.perspective)) {
			camera.perspective = false;
		}

		if (camera.perspective) {
			ImGui::SliderFloat("FOV (deg)", &camera.fov, 10.0f, 120.0f);
		} else {
			ImGui::SliderFloat("Ortho height", &camera.ortho_height, 1.0f, 40.0f);
		}
		ImGui::DragFloatRange2("Near / far", &camera.near_z, &camera.far_z, 0.05f, 0.01f, 500.0f);
		camera.near_z = std::max(camera.near_z, 0.01f);
		camera.far_z = std::max(camera.far_z, camera.near_z + 0.1f);

		ImGui::SliderFloat("Move speed", &camera.move_speed, 0.5f, 20.0f);
		ImGui::SliderFloat("Mouse sensitivity", &camera.mouse_sensitivity, 0.05f, 1.0f);
		if (ImGui::Button("Reset camera")) {
			resetCamera();
		}
	}

	if (ImGui::CollapsingHeader("Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Checkbox("Pause", &animation_paused);
		ImGui::SliderFloat("Speed", &animation_speed, 0.0f, 5.0f);
	}

	if (ImGui::CollapsingHeader("Objects", ImGuiTreeNodeFlags_DefaultOpen)) {
		for (size_t i = 0; i < objects.size(); ++i) {
			SceneObject& object = objects[i];

			ImGui::PushID(int(i));
			if (ImGui::TreeNode(object.name.c_str())) {
				ImGui::SeparatorText("Transform");
				ImGui::DragFloat3("Position", object.position, 0.05f);
				ImGui::SliderFloat3("Rotation (deg)", object.rotation, -180.0f, 180.0f);
				ImGui::DragFloat3("Scale", object.scale, 0.02f, 0.05f, 10.0f);

				ImGui::SeparatorText("Material (Blinn-Phong)");
				ImGui::ColorEdit3("Tint (x texture)", object.material.albedo);
				ImGui::ColorEdit3("Specular color", object.material.specular);
				ImGui::SliderFloat("Shininess", &object.material.shininess, 1.0f, 512.0f, "%.1f",
				                   ImGuiSliderFlags_Logarithmic);
				ImGui::Checkbox("Multiply by vertex colors", &object.material.use_vertex_color);

				// лаб. 3: выбор текстур
				ImGui::SeparatorText("Textures");
				if (ImGui::BeginCombo("Material", texture_material_descriptions[object.texture_material].name)) {
					for (int m = 0; m < TEXTURE_MATERIAL_COUNT; ++m) {
						const bool selected = (object.texture_material == m);
						if (ImGui::Selectable(texture_material_descriptions[m].name, selected)) {
							object.texture_material = m;
						}
						if (selected) {
							ImGui::SetItemDefaultFocus();
						}
					}
					ImGui::EndCombo();
				}
				ImGui::DragFloat2("UV scale", object.uv_scale, 0.02f, 0.1f, 40.0f);
				ImGui::SliderFloat("Emissive strength", &object.emissive_strength, 0.0f, 4.0f);
				ImGui::Checkbox("Animated sampling", &object.animated_sampling);

				ImGui::SeparatorText("Animation");
				ImGui::Checkbox("Animate", &object.animate);
				ImGui::SliderFloat("Spin speed", &object.spin_speed, -3.0f, 3.0f);
				ImGui::SliderFloat("Bob height", &object.bob_height, 0.0f, 1.5f);

				ImGui::TreePop();
			}
			ImGui::PopID();
		}

		if (ImGui::Button("Reset objects")) {
			resetObjects();
		}
	}

	ImGui::End();
}

// общий редактор направления по двум углам
void directionEditor(float* yaw, float* pitch) {
	ImGui::SliderFloat("Yaw (deg)", yaw, -180.0f, 180.0f);
	ImGui::SliderFloat("Pitch (deg)", pitch, -89.9f, 89.9f);
	const Vec3 d = directionFromYawPitch(*yaw * DEG_TO_RAD, *pitch * DEG_TO_RAD);
	ImGui::TextDisabled("direction = (%.2f, %.2f, %.2f)", d.x, d.y, d.z);
}

void drawLightsWindow() {
	const float width = 350.0f;
	ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x - width - 10.0f, 10.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(width, 520), ImGuiCond_FirstUseEver);
	ImGui::Begin("Lights");

	if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen)) {
		// доп. задание 4: показ/скрытие визуализации источников
		ImGui::Checkbox("Show light gizmos", &show_light_gizmos);
		ImGui::SliderFloat("Gizmo size", &gizmo_size, 0.02f, 0.5f);
		ImGui::SliderFloat("Spot cone length", &spot_cone_length, 0.2f, 6.0f);
	}

	if (ImGui::CollapsingHeader("Ambient light", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::PushID("ambient");
		ImGui::ColorEdit3("Color", ambient.color);
		ImGui::SliderFloat("Intensity", &ambient.intensity, 0.0f, 1.0f);
		ImGui::PopID();
	}

	if (ImGui::CollapsingHeader("Directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::PushID("directional");
		ImGui::Checkbox("Enabled", &directional.enabled);
		directionEditor(&directional.yaw, &directional.pitch);
		ImGui::ColorEdit3("Color", directional.color);
		ImGui::SliderFloat("Intensity", &directional.intensity, 0.0f, 5.0f);
		ImGui::PopID();
	}

	// доп. задание 2: массив точечных источников с добавлением и удалением
	if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::PushID("point");
		if (ImGui::Button("Add point light")) {
			PointLight light = makePointLight({ 0.0f, 2.0f, 2.0f }, { 1.0f, 1.0f, 1.0f }, 6.0f);
			light.orbit = false;
			point_lights.push_back(light);
		}
		ImGui::SameLine();
		ImGui::Text("count: %d", int(point_lights.size()));

		int remove_index = -1;
		for (size_t i = 0; i < point_lights.size(); ++i) {
			PointLight& light = point_lights[i];

			ImGui::PushID(int(i));
			if (ImGui::TreeNode("light", "Point light #%d", int(i) + 1)) {
				ImGui::Checkbox("Enabled", &light.enabled);
				ImGui::SameLine();
				if (ImGui::SmallButton("Remove")) {
					remove_index = int(i);
				}
				ImGui::DragFloat3("Position", light.position, 0.05f);
				ImGui::ColorEdit3("Color", light.color);
				ImGui::DragFloat("Intensity", &light.intensity, 0.1f, 0.0f, 500.0f);
				ImGui::Checkbox("Orbit around scene", &light.orbit);
				if (light.orbit) {
					ImGui::SliderFloat("Orbit speed", &light.orbit_speed, -3.0f, 3.0f);
				}
				ImGui::TreePop();
			}
			ImGui::PopID();
		}

		if (remove_index >= 0) {
			point_lights.erase(point_lights.begin() + remove_index);
		}
		ImGui::PopID();
	}

	// доп. задание 3: массив прожекторов с добавлением и удалением
	if (ImGui::CollapsingHeader("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::PushID("spot");
		if (ImGui::Button("Add spot light")) {
			// новый прожектор ставим в позицию камеры и направляем туда же, куда смотрит камера
			SpotLight light = makeSpotLight(toVec3(camera.position), camera.yaw, camera.pitch,
			                                { 1.0f, 1.0f, 1.0f }, 30.0f);
			spot_lights.push_back(light);
		}
		ImGui::SameLine();
		ImGui::Text("count: %d", int(spot_lights.size()));

		int remove_index = -1;
		for (size_t i = 0; i < spot_lights.size(); ++i) {
			SpotLight& light = spot_lights[i];

			ImGui::PushID(int(i));
			if (ImGui::TreeNode("light", "Spot light #%d", int(i) + 1)) {
				ImGui::Checkbox("Enabled", &light.enabled);
				ImGui::SameLine();
				if (ImGui::SmallButton("Remove")) {
					remove_index = int(i);
				}
				ImGui::Checkbox("Attach to camera (flashlight)", &light.follow_camera);
				if (!light.follow_camera) {
					ImGui::DragFloat3("Position", light.position, 0.05f);
					directionEditor(&light.yaw, &light.pitch);
				}
				ImGui::ColorEdit3("Color", light.color);
				ImGui::DragFloat("Intensity", &light.intensity, 0.1f, 0.0f, 1000.0f);

				// внешний угол не может быть меньше внутреннего
				ImGui::SliderFloat("Inner angle (deg)", &light.inner_angle, 0.0f, 89.0f);
				ImGui::SliderFloat("Outer angle (deg)", &light.outer_angle, 0.0f, 89.0f);
				light.outer_angle = std::max(light.outer_angle, light.inner_angle);

				ImGui::TreePop();
			}
			ImGui::PopID();
		}

		if (remove_index >= 0) {
			spot_lights.erase(spot_lights.begin() + remove_index);
		}
		ImGui::PopID();
	}

	if (ImGui::Button("Reset lights")) {
		resetLights();
	}

	ImGui::End();
}

// ===========================================================================
// подготовка данных кадра
// ===========================================================================

// модельная матрица для визуализации: перенос + равномерный масштаб
GizmoPushConstants makeGizmo(Vec3 position, float size, const float color[3]) {
	GizmoPushConstants g = {};
	copyMat4(mat4Mul(mat4Translate(position), mat4Scale(size, size, size)), g.model);
	g.color[0] = color[0];
	g.color[1] = color[1];
	g.color[2] = color[2];
	g.color[3] = 1.0f;
	return g;
}

void addLine(Vec3 a, Vec3 b, Vec3 color) {
	line_vertices.push_back(makeVertex(a, color));
	line_vertices.push_back(makeVertex(b, color));
}

// конус прожектора белыми линиями: окружность внешнего угла, образующие,
// окружность внутреннего угла (серым) и ось направления
void addSpotCone(Vec3 position, Vec3 direction, float inner_deg, float outer_deg, float cone_length) {
	const Vec3 white = { 1.0f, 1.0f, 1.0f };
	const Vec3 grey = { 0.55f, 0.55f, 0.55f };

	// базис, перпендикулярный направлению
	const Vec3 helper = std::fabs(direction.y) < 0.99f ? Vec3{ 0.0f, 1.0f, 0.0f } : Vec3{ 1.0f, 0.0f, 0.0f };
	const Vec3 side = normalize(cross(direction, helper));
	const Vec3 up = cross(side, direction);

	const Vec3 center = position + direction * cone_length;
	const float outer_radius = cone_length * std::tan(std::min(outer_deg, 89.0f) * DEG_TO_RAD);
	const float inner_radius = cone_length * std::tan(std::min(inner_deg, 89.0f) * DEG_TO_RAD);

	constexpr int SEGMENTS = 32;
	for (int i = 0; i < SEGMENTS; ++i) {
		const float a0 = 2.0f * PI * float(i) / float(SEGMENTS);
		const float a1 = 2.0f * PI * float(i + 1) / float(SEGMENTS);
		const Vec3 d0 = side * std::cos(a0) + up * std::sin(a0);
		const Vec3 d1 = side * std::cos(a1) + up * std::sin(a1);

		addLine(center + d0 * outer_radius, center + d1 * outer_radius, white);
		if (inner_radius > 0.0f) {
			addLine(center + d0 * inner_radius, center + d1 * inner_radius, grey);
		}

		// 8 образующих конуса
		if (i % (SEGMENTS / 8) == 0) {
			addLine(position, center + d0 * outer_radius, white);
		}
	}

	// ось прожектора
	addLine(position, position + direction * (cone_length * 1.25f), white);
}

void prepareFrameData(float dt) {
	auto& context = graphics::internal::context;

	// --- камера ---
	const float aspect = float(context.swapchain_extent.width) /
	                     float(std::max(context.swapchain_extent.height, 1u));

	Mat4 projection;
	if (camera.perspective) {
		projection = mat4Perspective(camera.fov * DEG_TO_RAD, aspect, camera.near_z, camera.far_z);
	} else {
		projection = mat4Ortho(camera.ortho_height, aspect, camera.near_z, camera.far_z);
	}

	const Vec3 camera_position = toVec3(camera.position);
	const float camera_yaw = camera.yaw * DEG_TO_RAD;
	const float camera_pitch = camera.pitch * DEG_TO_RAD;
	const Vec3 camera_forward = directionFromYawPitch(camera_yaw, camera_pitch);
	const Vec3 camera_right = normalize(cross(camera_forward, { 0.0f, 1.0f, 0.0f }));

	const Mat4 view = mat4View(camera_position, camera_yaw, camera_pitch);
	const Mat4 view_projection = mat4Mul(projection, view);

	// --- анимация точечных источников: поворот вокруг оси y ---
	if (!animation_paused) {
		for (PointLight& light : point_lights) {
			if (!light.orbit) {
				continue;
			}
			const float angle = light.orbit_speed * dt * animation_speed;
			const float c = std::cos(angle), s = std::sin(angle);
			const float x = light.position[0], z = light.position[2];
			light.position[0] = c * x - s * z;
			light.position[2] = s * x + c * z;
		}
	}

	// --- uniform-буфер сцены ---
	scene_uniforms = {};
	copyMat4(view_projection, scene_uniforms.view_projection);
	storeVec3(camera_position, scene_uniforms.camera_position);

	const Vec3 sun_direction = directionFromYawPitch(directional.yaw * DEG_TO_RAD, directional.pitch * DEG_TO_RAD);
	storeVec3(sun_direction, scene_uniforms.directional_light_direction);
	copyColor(directional.color, directional.enabled ? directional.intensity : 0.0f,
	          scene_uniforms.directional_light_color);
	copyColor(ambient.color, ambient.intensity, scene_uniforms.ambient_light);

	// --- массивы источников для storage-буферов (выключенные не отправляем) ---
	gizmo_draws.clear();
	line_vertices.clear();

	point_light_data.clear();
	for (const PointLight& light : point_lights) {
		if (!light.enabled) {
			continue;
		}

		PointLightGPU gpu = {};
		std::memcpy(gpu.position, light.position, sizeof(gpu.position));
		std::memcpy(gpu.color, light.color, sizeof(gpu.color));
		gpu.intensity = light.intensity;
		point_light_data.push_back(gpu);

		if (show_light_gizmos) {
			gizmo_draws.push_back(makeGizmo(toVec3(light.position), gizmo_size, light.color));
		}
	}

	spot_light_data.clear();
	for (SpotLight& light : spot_lights) {
		if (light.follow_camera) {
			// "фонарик": чуть правее и ниже камеры, светит туда же, куда она смотрит
			const Vec3 offset = camera_right * 0.25f + Vec3{ 0.0f, -0.2f, 0.0f };
			storeVec3(camera_position + offset, light.position);
			light.yaw = camera.yaw;
			light.pitch = camera.pitch;
		}

		if (!light.enabled) {
			continue;
		}

		const Vec3 direction = directionFromYawPitch(light.yaw * DEG_TO_RAD, light.pitch * DEG_TO_RAD);

		SpotLightGPU gpu = {};
		std::memcpy(gpu.position, light.position, sizeof(gpu.position));
		std::memcpy(gpu.color, light.color, sizeof(gpu.color));
		storeVec3(direction, gpu.direction);
		gpu.intensity = light.intensity;
		gpu.cos_inner = std::cos(light.inner_angle * DEG_TO_RAD);
		gpu.cos_outer = std::cos(light.outer_angle * DEG_TO_RAD);
		spot_light_data.push_back(gpu);

		// фонарик, прикреплённый к камере, не рисуем - он был бы прямо перед глазами
		if (show_light_gizmos && !light.follow_camera) {
			gizmo_draws.push_back(makeGizmo(toVec3(light.position), gizmo_size, light.color));
			addSpotCone(toVec3(light.position), direction, light.inner_angle, light.outer_angle,
			            spot_cone_length);
		}
	}

	scene_uniforms.point_light_count = uint32_t(point_light_data.size());
	scene_uniforms.spot_light_count = uint32_t(spot_light_data.size());

	// --- uniform-буферы моделей ---
	for (SceneObject& object : objects) {
		float x = object.position[0];
		float y = object.position[1];
		float z = object.position[2];
		float rx = object.rotation[0] * DEG_TO_RAD;
		float ry = object.rotation[1] * DEG_TO_RAD;
		float rz = object.rotation[2] * DEG_TO_RAD;

		if (object.animate) {
			const float t = animation_time + object.phase;
			y += object.bob_height * std::sin(t * 1.3f);
			ry += animation_time * object.spin_speed;
		}

		// модельная матрица: масштаб - поворот - перенос
		const Mat4 rotation = mat4Mul(mat4RotateZ(rz), mat4Mul(mat4RotateY(ry), mat4RotateX(rx)));
		const Mat4 scale = mat4Scale(object.scale[0], object.scale[1], object.scale[2]);
		const Mat4 model = mat4Mul(mat4Translate(x, y, z), mat4Mul(rotation, scale));

		ModelUniforms& u = object.uniforms;
		u = {};
		copyMat4(model, u.model);
		copyMat4(mat4NormalMatrix(model), u.normal_matrix);
		std::memcpy(u.albedo, object.material.albedo, sizeof(u.albedo));
		std::memcpy(u.specular, object.material.specular, sizeof(u.specular));
		u.shininess = std::max(object.material.shininess, 1.0f);
		u.use_vertex_color = object.material.use_vertex_color ? 1.0f : 0.0f;
		u.uv_scale[0] = object.uv_scale[0];
		u.uv_scale[1] = object.uv_scale[1];
		u.emissive_strength = object.emissive_strength;
	}
}

// ===========================================================================
// лаб. 3: загрузка текстур
// ===========================================================================

// ищет папку textures/ в нескольких местах (как и шейдеры) и возвращает путь с / на конце
std::string findTextureDirectory() {
	const char* directories[] = {
		TEXTURE_DIR,       // абсолютный путь, его подставляет CMake
		"textures/",
		"./textures/",
		"../textures/",
		"../../textures/",
	};

	for (const char* directory : directories) {
		std::ifstream probe(std::string(directory) + texture_material_descriptions[0].albedo, std::ios::binary);
		if (probe.is_open()) {
			return directory;
		}
	}
	return TEXTURE_DIR;
}

bool createTextures() {
	if (!createSampler(texture_sampler)) {
		return false;
	}

	// чёрный пиксель 1x1: если у материала нет specular или emissive, подставляем его
	if (!createSolidTexture(0, 0, 0, 255, false, black_texture)) {
		return false;
	}

	const std::string directory = findTextureDirectory();
	for (int i = 0; i < TEXTURE_MATERIAL_COUNT; ++i) {
		const TextureMaterialDescription& description = texture_material_descriptions[i];
		TextureMaterial& material = texture_materials[i];

		// albedo и emissive - цвета (формат sRGB), specular - просто числа (формат UNORM)
		if (!loadTextureFromFile(directory + description.albedo, true, material.albedo)) {
			return false;
		}
		if (description.specular[0] != '\0' &&
		    !loadTextureFromFile(directory + description.specular, false, material.specular)) {
			return false;
		}
		if (description.emissive[0] != '\0' &&
		    !loadTextureFromFile(directory + description.emissive, true, material.emissive)) {
			return false;
		}
	}
	return true;
}

} // namespace

// ===========================================================================
// initialize: создаём все объекты vulkan
// ===========================================================================

bool initialize() {
	auto& context = graphics::internal::context;

	resetCamera();
	resetLights();

	// лаб. 3: пул команд для одноразовых операций (копирование в видеопамять)
	if (!vkutil::initialize()) {
		return false;
	}

	// шейдеры
	lit_vertex_shader = loadShaderModule("lit.vert.spv");
	lit_fragment_shader = loadShaderModule("lit.frag.spv");
	gizmo_vertex_shader = loadShaderModule("gizmo.vert.spv");
	gizmo_fragment_shader = loadShaderModule("gizmo.frag.spv");
	if (!lit_vertex_shader || !lit_fragment_shader || !gizmo_vertex_shader || !gizmo_fragment_shader) {
		std::cerr << "Failed to load shaders\n";
		return false;
	}

	// меши (нормали генерируются процедурно в geometry.hpp)
	if (!createMesh(makeOctahedron(), octahedron_mesh) ||
	    !createMesh(makeCheckerFloor(24.0f, 16), floor_mesh) ||
	    !createMesh(makeQuad(), quad_mesh) ||
	    !createMesh(makeSphere(10, 16), sphere_mesh)) {
		return false;
	}

	resetObjects();

	// буферы сцены: uniform + два storage-буфера (их размер потом растёт по мере надобности)
	if (!createHostBuffer(sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, nullptr,
	                      scene_uniform_buffer)) {
		return false;
	}
	ensureCapacity(point_light_buffer, sizeof(PointLightGPU), STORAGE_USAGE);
	ensureCapacity(spot_light_buffer, sizeof(SpotLightGPU), STORAGE_USAGE);
	ensureCapacity(line_vertex_buffer, sizeof(Vertex) * 2, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

	// лаб. 3: ОДИН динамический uniform-буфер на все объекты.
	// шаг между структурами обязан быть кратен minUniformBufferOffsetAlignment видеокарты
	// (у nvidia, amd и intel значение разное, поэтому в код его зашивать нельзя)
	{
		VkPhysicalDeviceProperties properties;
		vkGetPhysicalDeviceProperties(context.physical_device, &properties);
		const VkDeviceSize alignment = properties.limits.minUniformBufferOffsetAlignment;
		model_stride = (sizeof(ModelUniforms) + alignment - 1) / alignment * alignment;

		if (!createHostBuffer(model_stride * objects.size(), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, nullptr,
		                      model_buffer)) {
			return false;
		}
	}

	// лаб. 3: сэмплер, чёрная заглушка и текстуры всех материалов
	if (!createTextures()) {
		return false;
	}

	// доп. задание 1: два layout'а наборов дескрипторов
	// set = 0 (сцена): binding 0 - uniform сцены, binding 1 - точечные, binding 2 - прожекторы
	{
		const VkDescriptorSetLayoutBinding bindings[] = {
			{
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
			},
			{
				.binding = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
			},
			{
				.binding = 2,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
			},
		};

		const VkDescriptorSetLayoutCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = sizeof(bindings) / sizeof(bindings[0]),
			.pBindings = bindings,
		};

		if (vkCreateDescriptorSetLayout(context.device, &info, nullptr, &scene_set_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create scene descriptor set layout\n";
			return false;
		}
	}

	// set = 1 (модель): binding 0 - uniform модели (матрица + материал).
	// лаб. 3: тип DYNAMIC - смещение внутри буфера задаётся при привязке набора
	{
		const VkDescriptorSetLayoutBinding binding{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		};

		const VkDescriptorSetLayoutCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = 1,
			.pBindings = &binding,
		};

		if (vkCreateDescriptorSetLayout(context.device, &info, nullptr, &model_set_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create model descriptor set layout\n";
			return false;
		}
	}

	// лаб. 3: set = 2 (материал): три текстуры, каждая - пара "изображение + сэмплер"
	{
		const VkDescriptorSetLayoutBinding bindings[] = {
			{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			  .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },   // albedo
			{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			  .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },   // specular
			{ .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			  .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },   // emissive
		};

		const VkDescriptorSetLayoutCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = sizeof(bindings) / sizeof(bindings[0]),
			.pBindings = bindings,
		};

		if (vkCreateDescriptorSetLayout(context.device, &info, nullptr, &material_set_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create material descriptor set layout\n";
			return false;
		}
	}

	// пул: набор сцены, один (динамический) набор модели и по набору на каждый материал
	{
		const VkDescriptorPoolSize pool_sizes[] = {
			{ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1 },
			{ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = 1 },
			{ .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 },
			{ .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 3 * TEXTURE_MATERIAL_COUNT },
		};

		const VkDescriptorPoolCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.maxSets = 2 + TEXTURE_MATERIAL_COUNT,
			.poolSizeCount = sizeof(pool_sizes) / sizeof(pool_sizes[0]),
			.pPoolSizes = pool_sizes,
		};

		if (vkCreateDescriptorPool(context.device, &info, nullptr, &descriptor_pool) != VK_SUCCESS) {
			std::cerr << "Failed to create descriptor pool\n";
			return false;
		}
	}

	// набор дескрипторов сцены
	{
		const VkDescriptorSetAllocateInfo allocate_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = descriptor_pool,
			.descriptorSetCount = 1,
			.pSetLayouts = &scene_set_layout,
		};

		if (vkAllocateDescriptorSets(context.device, &allocate_info, &scene_descriptor_set) != VK_SUCCESS) {
			std::cerr << "Failed to allocate scene descriptor set\n";
			return false;
		}

		writeBufferDescriptor(scene_descriptor_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		                      scene_uniform_buffer.buffer, sizeof(SceneUniforms));
		writeBufferDescriptor(scene_descriptor_set, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		                      point_light_buffer.buffer, VK_WHOLE_SIZE);
		writeBufferDescriptor(scene_descriptor_set, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		                      spot_light_buffer.buffer, VK_WHOLE_SIZE);
	}

	// набор модели: один на все объекты. range - размер ОДНОЙ структуры,
	// к нужной структуре видеокарта перейдёт по смещению из vkCmdBindDescriptorSets
	{
		const VkDescriptorSetAllocateInfo allocate_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = descriptor_pool,
			.descriptorSetCount = 1,
			.pSetLayouts = &model_set_layout,
		};

		if (vkAllocateDescriptorSets(context.device, &allocate_info, &model_descriptor_set) != VK_SUCCESS) {
			std::cerr << "Failed to allocate model descriptor set\n";
			return false;
		}

		writeBufferDescriptor(model_descriptor_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
		                      model_buffer.buffer, sizeof(ModelUniforms));
	}

	// лаб. 3: по набору дескрипторов на каждый материал (у каждого материала свои текстуры)
	for (TextureMaterial& material : texture_materials) {
		const VkDescriptorSetAllocateInfo allocate_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = descriptor_pool,
			.descriptorSetCount = 1,
			.pSetLayouts = &material_set_layout,
		};

		if (vkAllocateDescriptorSets(context.device, &allocate_info, &material.set) != VK_SUCCESS) {
			std::cerr << "Failed to allocate material descriptor set\n";
			return false;
		}

		// нет specular или emissive - подставляем чёрную заглушку: блеск и свечение будут нулевыми
		const VkImageView views[3] = {
			material.albedo.view,
			material.specular.view != VK_NULL_HANDLE ? material.specular.view : black_texture.view,
			material.emissive.view != VK_NULL_HANDLE ? material.emissive.view : black_texture.view,
		};

		for (uint32_t binding = 0; binding < 3; ++binding) {
			const VkDescriptorImageInfo image_info{
				.sampler = texture_sampler,
				.imageView = views[binding],
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,   // в этом состоянии лежит текстура
			};

			const VkWriteDescriptorSet write{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = material.set,
				.dstBinding = binding,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &image_info,
			};
			vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
		}
	}

	// layout конвейера освещения: set 0 - сцена, set 1 - модель, set 2 - материал (текстуры)
	// + push-константы фрагментного шейдера (лаб. 3)
	{
		const VkDescriptorSetLayout set_layouts[] = { scene_set_layout, model_set_layout, material_set_layout };

		const VkPushConstantRange push_range{
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
			.offset = 0,
			.size = sizeof(LitPushConstants),
		};

		const VkPipelineLayoutCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 3,
			.pSetLayouts = set_layouts,
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &push_range,
		};

		if (vkCreatePipelineLayout(context.device, &info, nullptr, &lit_pipeline_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create lit pipeline layout\n";
			return false;
		}
	}

	// layout конвейера визуализации: set 0 - сцена (нужна только view_projection) + push-константы
	{
		const VkPushConstantRange push_range{
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
			.offset = 0,
			.size = sizeof(GizmoPushConstants),
		};

		const VkPipelineLayoutCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &scene_set_layout,
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &push_range,
		};

		if (vkCreatePipelineLayout(context.device, &info, nullptr, &gizmo_pipeline_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create gizmo pipeline layout\n";
			return false;
		}
	}

	// конвейеры
	lit_pipeline = createPipeline(lit_vertex_shader, lit_fragment_shader, lit_pipeline_layout,
	                              VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_CULL_MODE_BACK_BIT);
	gizmo_pipeline = createPipeline(gizmo_vertex_shader, gizmo_fragment_shader, gizmo_pipeline_layout,
	                                VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_CULL_MODE_NONE);
	line_pipeline = createPipeline(gizmo_vertex_shader, gizmo_fragment_shader, gizmo_pipeline_layout,
	                               VK_PRIMITIVE_TOPOLOGY_LINE_LIST, VK_CULL_MODE_NONE);
	if (!lit_pipeline || !gizmo_pipeline || !line_pipeline) {
		return false;
	}

	return true;
}

// ===========================================================================
// shutdown: уничтожаем всё в обратном порядке
// ===========================================================================

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	vkDestroyPipeline(context.device, line_pipeline, nullptr);
	vkDestroyPipeline(context.device, gizmo_pipeline, nullptr);
	vkDestroyPipeline(context.device, lit_pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, gizmo_pipeline_layout, nullptr);
	vkDestroyPipelineLayout(context.device, lit_pipeline_layout, nullptr);

	// наборы дескрипторов освобождаются вместе с пулом
	vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, material_set_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, model_set_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, scene_set_layout, nullptr);

	// лаб. 3: текстуры и сэмплер
	for (TextureMaterial& material : texture_materials) {
		destroyTexture(material.albedo);
		destroyTexture(material.specular);
		destroyTexture(material.emissive);
	}
	destroyTexture(black_texture);
	destroySampler(texture_sampler);

	destroyHostBuffer(model_buffer);

	destroyHostBuffer(line_vertex_buffer);
	destroyHostBuffer(spot_light_buffer);
	destroyHostBuffer(point_light_buffer);
	destroyHostBuffer(scene_uniform_buffer);

	destroyMesh(sphere_mesh);
	destroyMesh(quad_mesh);
	destroyMesh(floor_mesh);
	destroyMesh(octahedron_mesh);

	vkDestroyShaderModule(context.device, gizmo_fragment_shader, nullptr);
	vkDestroyShaderModule(context.device, gizmo_vertex_shader, nullptr);
	vkDestroyShaderModule(context.device, lit_fragment_shader, nullptr);
	vkDestroyShaderModule(context.device, lit_vertex_shader, nullptr);

	vkutil::shutdown();
}

// ===========================================================================
// update: интерфейс, управление камерой, подготовка данных кадра
// ===========================================================================

void update([[maybe_unused]] double time) {
	const float dt = std::min(ImGui::GetIO().DeltaTime, 0.1f);

	drawSceneWindow();
	drawLightsWindow();

	updateCameraInput(dt);

	// время анимации копим сами, чтобы работали пауза и смена скорости
	if (!animation_paused) {
		animation_time += dt * animation_speed;
	}

	prepareFrameData(dt);
}

// ===========================================================================
// render: загружаем данные на gpu и записываем команды
// ===========================================================================

void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;

	// к этому моменту prepare() уже дождался окончания предыдущего кадра,
	// поэтому буферы можно спокойно перезаписывать и пересоздавать

	// storage-буферы источников: при необходимости увеличиваем и обновляем дескрипторы
	const VkDeviceSize point_bytes = std::max<size_t>(point_light_data.size(), 1) * sizeof(PointLightGPU);
	if (ensureCapacity(point_light_buffer, point_bytes, STORAGE_USAGE)) {
		writeBufferDescriptor(scene_descriptor_set, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		                      point_light_buffer.buffer, VK_WHOLE_SIZE);
	}
	const VkDeviceSize spot_bytes = std::max<size_t>(spot_light_data.size(), 1) * sizeof(SpotLightGPU);
	if (ensureCapacity(spot_light_buffer, spot_bytes, STORAGE_USAGE)) {
		writeBufferDescriptor(scene_descriptor_set, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		                      spot_light_buffer.buffer, VK_WHOLE_SIZE);
	}
	ensureCapacity(line_vertex_buffer, std::max<size_t>(line_vertices.size(), 2) * sizeof(Vertex),
	               VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

	std::memcpy(scene_uniform_buffer.memory, &scene_uniforms, sizeof(scene_uniforms));
	if (!point_light_data.empty()) {
		std::memcpy(point_light_buffer.memory, point_light_data.data(),
		            point_light_data.size() * sizeof(PointLightGPU));
	}
	if (!spot_light_data.empty()) {
		std::memcpy(spot_light_buffer.memory, spot_light_data.data(),
		            spot_light_data.size() * sizeof(SpotLightGPU));
	}
	if (!line_vertices.empty()) {
		std::memcpy(line_vertex_buffer.memory, line_vertices.data(), line_vertices.size() * sizeof(Vertex));
	}
	// структура i-го объекта лежит в model_buffer со смещением i * model_stride
	for (size_t i = 0; i < objects.size(); ++i) {
		std::memcpy(static_cast<char*>(model_buffer.memory) + i * model_stride,
		            &objects[i].uniforms, sizeof(ModelUniforms));
	}

	// запись команд
	vkResetCommandBuffer(fd.command_buffer, 0);

	const VkCommandBufferBeginInfo command_buffer_begin{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};

	vkBeginCommandBuffer(fd.command_buffer, &command_buffer_begin);

	const VkClearValue clear_values[] = {
		{ .color = { .float32 = { 0.02f, 0.02f, 0.03f, 1.0f } } },
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

	// 1) освещаемые объекты
	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, lit_pipeline);

	// набор сцены (set = 0) привязываем один раз на все объекты
	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
	                        lit_pipeline_layout, 0, 1, &scene_descriptor_set, 0, nullptr);

	for (size_t i = 0; i < objects.size(); ++i) {
		const SceneObject& object = objects[i];

		// лаб. 3: push-константы - флаг нетривиального сэмплирования и время
		const LitPushConstants push = { object.animated_sampling ? FLAG_ANIMATED_SAMPLING : 0u, animation_time };
		vkCmdPushConstants(fd.command_buffer, lit_pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT,
		                   0, sizeof(push), &push);

		// привязываем сразу два набора, начиная с номера 1:
		//   set 1 - динамический буфер, dynamic_offset выбирает i-ю структуру ModelUniforms;
		//   set 2 - текстуры материала этого объекта
		const VkDescriptorSet sets[] = { model_descriptor_set, texture_materials[object.texture_material].set };
		const uint32_t dynamic_offset = uint32_t(model_stride * i);
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
		                        lit_pipeline_layout, 1, 2, sets, 1, &dynamic_offset);

		const VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &object.mesh->vertices.buffer, &offset);
		vkCmdBindIndexBuffer(fd.command_buffer, object.mesh->indices.buffer, 0, VK_INDEX_TYPE_UINT32);
		vkCmdDrawIndexed(fd.command_buffer, object.mesh->index_count, 1, 0, 0, 0);
	}

	// 2) визуализация источников (доп. задание 4): без освещения, постоянный цвет
	if (show_light_gizmos) {
		vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, gizmo_pipeline);
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
		                        gizmo_pipeline_layout, 0, 1, &scene_descriptor_set, 0, nullptr);

		// сферы источников
		const VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &sphere_mesh.vertices.buffer, &offset);
		vkCmdBindIndexBuffer(fd.command_buffer, sphere_mesh.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
		for (const GizmoPushConstants& gizmo : gizmo_draws) {
			vkCmdPushConstants(fd.command_buffer, gizmo_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
			                   0, sizeof(gizmo), &gizmo);
			vkCmdDrawIndexed(fd.command_buffer, sphere_mesh.index_count, 1, 0, 0, 0);
		}

		// конусы прожекторов линиями
		if (!line_vertices.empty()) {
			vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, line_pipeline);

			const float white[3] = { 1.0f, 1.0f, 1.0f };
			GizmoPushConstants identity = makeGizmo({ 0.0f, 0.0f, 0.0f }, 1.0f, white);
			vkCmdPushConstants(fd.command_buffer, gizmo_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
			                   0, sizeof(identity), &identity);

			vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &line_vertex_buffer.buffer, &offset);
			vkCmdDraw(fd.command_buffer, uint32_t(line_vertices.size()), 1, 0, 0);
		}
	}

	vkCmdEndRenderPass(fd.command_buffer);

	vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application
