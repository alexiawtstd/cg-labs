#version 450 core

// вершинный шейдер для визуализации источников света (сферы и линии конусов).
// освещение не считается: цвет постоянный

layout(location = 0) in vec3 in_position;
layout(location = 2) in vec3 in_color;

layout(location = 0) out vec3 out_color;

layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec3 camera_position;
    vec3 directional_light_direction;
    vec3 directional_light_color;
    uint point_light_count;
    vec3 ambient_light;
    uint spot_light_count;
} scene;

// матрица модели и цвет передаются через push-константы, чтобы не заводить
// отдельный набор дескрипторов на каждую маленькую сферу
layout(push_constant) uniform GizmoPush {
    mat4 model;
    vec4 color;
} gizmo;

void main() {
    gl_Position = scene.view_projection * gizmo.model * vec4(in_position, 1.0);
    out_color = in_color * gizmo.color.rgb;
}
