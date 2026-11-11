#version 450 core

// вершинный шейдер для объектов, освещаемых по модели блинна-фонга

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_color;
layout(location = 3) in vec2 in_uv;        // лаб. 3: текстурные координаты

layout(location = 0) out vec3 out_world_position;
layout(location = 1) out vec3 out_world_normal;
layout(location = 2) out vec3 out_color;
layout(location = 3) out vec2 out_uv;

// set = 0: общие данные сцены (одинаковы для всех объектов)
layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec3 camera_position;
    vec3 directional_light_direction; // куда светит направленный источник
    vec3 directional_light_color;     // цвет * интенсивность
    uint point_light_count;
    vec3 ambient_light;               // цвет * интенсивность рассеянного света
    uint spot_light_count;
} scene;

// set = 1: данные конкретной модели
// лаб. 3: буфер динамический (UNIFORM_BUFFER_DYNAMIC), в шейдере он выглядит как обычный uniform
layout(std140, set = 1, binding = 0) uniform ModelUniforms {
    mat4 model;
    mat4 normal_matrix;   // обратная транспонированная к model (блок 3x3)
    vec3 albedo;          // диффузный цвет материала
    float shininess;      // коэффициент блеска
    vec3 specular;        // цвет блика
    float use_vertex_color;
    vec2 uv_scale;        // лаб. 3: сколько раз повторить текстуру
    float emissive_strength;
} object;

void main() {
    vec4 world_position = object.model * vec4(in_position, 1.0);

    out_world_position = world_position.xyz;
    out_world_normal = mat3(object.normal_matrix) * in_normal;
    out_color = in_color;
    out_uv = in_uv * object.uv_scale; // повторение текстуры (сэмплер в режиме REPEAT)

    gl_Position = scene.view_projection * world_position;
}
