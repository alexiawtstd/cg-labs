#version 450 core

layout(location = 0) in vec3 in_color;

layout(location = 0) out vec4 out_color;

layout(std140, set = 0, binding = 0) uniform ObjectUniforms {
    mat4 mvp;
    vec4 color;
} object;

void main() {
    // Цвет вершин (интерполированный) умножаем на цвет из интерфейса
    out_color = vec4(in_color, 1.0) * object.color;
}
