#pragma once

#include <cstdint>

// Вершина: позиция + цвет (формат как в лекции 3)
struct Vertex {
	float position[3];
	float color[3];
};

// Правильный октаэдр: 6 вершин на осях, 8 треугольных граней.
// Цвет вершины считается из её позиции в локальных координатах:
// color = position * 0.5 + 0.5  (переводим диапазон [-1, 1] в [0, 1]).
// Отсюда доп. задание 5 (процедурные цвета вершин).
#define OCT_VERTEX(x, y, z) { { x, y, z }, { (x) * 0.5f + 0.5f, (y) * 0.5f + 0.5f, (z) * 0.5f + 0.5f } }

const Vertex octahedron_vertices[] = {
	OCT_VERTEX(+1.0f,  0.0f,  0.0f), // 0: +X
	OCT_VERTEX(-1.0f,  0.0f,  0.0f), // 1: -X
	OCT_VERTEX( 0.0f, +1.0f,  0.0f), // 2: +Y
	OCT_VERTEX( 0.0f, -1.0f,  0.0f), // 3: -Y
	OCT_VERTEX( 0.0f,  0.0f, +1.0f), // 4: +Z
	OCT_VERTEX( 0.0f,  0.0f, -1.0f), // 5: -Z
};

// 8 граней * 3 индекса = 24.
// Каждая грань обходится против часовой стрелки, если смотреть снаружи.
// В проекции мы переворачиваем ось Y под Vulkan, поэтому на экране лицевые
// грани тоже идут против часовой -> в конвейере VK_FRONT_FACE_COUNTER_CLOCKWISE.
const uint32_t octahedron_indices[] = {
	0, 2, 4, // (+X, +Y, +Z)
	1, 4, 2, // (-X, +Y, +Z)
	0, 4, 3, // (+X, -Y, +Z)
	0, 5, 2, // (+X, +Y, -Z)
	1, 3, 4, // (-X, -Y, +Z)
	1, 2, 5, // (-X, +Y, -Z)
	0, 3, 5, // (+X, -Y, -Z)
	1, 5, 3, // (-X, -Y, -Z)
};

const uint32_t octahedron_index_count = sizeof(octahedron_indices) / sizeof(octahedron_indices[0]);
