#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "math3d.hpp"

// вершина: позиция + нормаль + цвет (один формат для всех мешей сцены)
struct Vertex {
	float position[3];
	float normal[3];
	float color[3];
	float uv[2];       // лаб. 3: текстурные координаты
};

// меш на стороне cpu: вершины + индексы треугольников
struct MeshData {
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
};

inline Vertex makeVertex(Vec3 position, Vec3 color) {
	Vertex v = {};
	storeVec3(position, v.position);
	storeVec3(color, v.color);
	return v;
}

// процедурная генерация нормалей для плоского (flat) затенения:
// для каждого треугольника считаем нормаль грани n = normalize((b - a) x (c - a))
// и записываем её во все три его вершины.
// важно: вершины у разных граней не должны быть общими, иначе нормали перезапишутся
inline void generateFlatNormals(MeshData& mesh) {
	for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
		Vertex& va = mesh.vertices[mesh.indices[i + 0]];
		Vertex& vb = mesh.vertices[mesh.indices[i + 1]];
		Vertex& vc = mesh.vertices[mesh.indices[i + 2]];

		const Vec3 a = toVec3(va.position);
		const Vec3 b = toVec3(vb.position);
		const Vec3 c = toVec3(vc.position);

		// грани обходятся против часовой стрелки, если смотреть снаружи,
		// поэтому векторное произведение смотрит наружу
		const Vec3 n = normalize(cross(b - a, c - a));

		storeVec3(n, va.normal);
		storeVec3(n, vb.normal);
		storeVec3(n, vc.normal);
	}
}

// ---------------------------------------------------------------------------
// правильный октаэдр (вариант 2): 6 вершин на осях, 8 треугольных граней
// ---------------------------------------------------------------------------

inline MeshData makeOctahedron() {
	// исходные данные из лабораторной работы 1
	const Vec3 corners[6] = {
		{ +1.0f,  0.0f,  0.0f }, // 0: +x
		{ -1.0f,  0.0f,  0.0f }, // 1: -x
		{  0.0f, +1.0f,  0.0f }, // 2: +y
		{  0.0f, -1.0f,  0.0f }, // 3: -y
		{  0.0f,  0.0f, +1.0f }, // 4: +z
		{  0.0f,  0.0f, -1.0f }, // 5: -z
	};

	// каждая грань обходится против часовой стрелки, если смотреть снаружи
	const uint32_t faces[24] = {
		0, 2, 4, // (+x, +y, +z)
		1, 4, 2, // (-x, +y, +z)
		0, 4, 3, // (+x, -y, +z)
		0, 5, 2, // (+x, +y, -z)
		1, 3, 4, // (-x, -y, +z)
		1, 2, 5, // (-x, +y, -z)
		0, 3, 5, // (+x, -y, -z)
		1, 5, 3, // (-x, -y, -z)
	};

	// у октаэдра острые рёбра, поэтому каждой грани нужны свои 3 вершины
	// со своей нормалью: 8 граней * 3 = 24 вершины
	MeshData mesh;
	for (uint32_t i = 0; i < 24; ++i) {
		const Vec3 p = corners[faces[i]];
		// процедурный цвет вершины из лабы 1: позиция [-1, 1] -> цвет [0, 1]
		const Vec3 color = p * 0.5f + Vec3{ 0.5f, 0.5f, 0.5f };
		mesh.vertices.push_back(makeVertex(p, color));
		mesh.indices.push_back(i);
	}

	generateFlatNormals(mesh);

	// лаб. 3: текстурные координаты. каждая грань - треугольник, на который накладывается
	// треугольный кусок текстуры: основание внизу, вершина вверху (одинаково для всех 8 граней)
	const float triangle_uv[3][2] = { { 0.0f, 1.0f }, { 1.0f, 1.0f }, { 0.5f, 0.0f } };
	for (uint32_t i = 0; i < 24; ++i) {
		mesh.vertices[i].uv[0] = triangle_uv[i % 3][0];
		mesh.vertices[i].uv[1] = triangle_uv[i % 3][1];
	}

	return mesh;
}

// ---------------------------------------------------------------------------
// пол: квадрат size x size в плоскости y = 0, разбитый на клетки "шахматкой"
// ---------------------------------------------------------------------------

inline MeshData makeCheckerFloor(float size, int tiles) {
	MeshData mesh;
	const float tile = size / float(tiles);
	const float start = -size * 0.5f;

	for (int i = 0; i < tiles; ++i) {
		for (int j = 0; j < tiles; ++j) {
			const float x0 = start + tile * float(i);
			const float z0 = start + tile * float(j);
			const float x1 = x0 + tile;
			const float z1 = z0 + tile;

			const float shade = ((i + j) % 2 == 0) ? 0.75f : 0.3f;
			const Vec3 color = { shade, shade, shade };

			const uint32_t base = uint32_t(mesh.vertices.size());
			mesh.vertices.push_back(makeVertex({ x0, 0.0f, z0 }, color));
			mesh.vertices.push_back(makeVertex({ x0, 0.0f, z1 }, color));
			mesh.vertices.push_back(makeVertex({ x1, 0.0f, z1 }, color));
			mesh.vertices.push_back(makeVertex({ x1, 0.0f, z0 }, color));

			// лаб. 3: текстурные координаты пола - его позиция (x, z), нормированная на размер:
			// по всему полу получается [0, 1], а повторение задаёт uv_scale объекта
			const float us[4] = { (x0 - start) / size, (x0 - start) / size, (x1 - start) / size, (x1 - start) / size };
			const float vs[4] = { (z0 - start) / size, (z1 - start) / size, (z1 - start) / size, (z0 - start) / size };
			for (int k = 0; k < 4; ++k) {
				mesh.vertices[base + uint32_t(k)].uv[0] = us[k];
				mesh.vertices[base + uint32_t(k)].uv[1] = vs[k];
			}

			// два треугольника против часовой стрелки, если смотреть сверху
			const uint32_t quad[6] = { 0, 1, 2, 0, 2, 3 };
			for (uint32_t k : quad) {
				mesh.indices.push_back(base + k);
			}
		}
	}

	// нормали тоже считаются процедурно (получится (0, 1, 0))
	generateFlatNormals(mesh);
	return mesh;
}

// ---------------------------------------------------------------------------
// лаб. 3: квадрат 1 x 1 в плоскости xy, нормаль +z. из него сделаны стены:
// объект поворачивается и растягивается матрицей модели
// ---------------------------------------------------------------------------

inline MeshData makeQuad() {
	MeshData mesh;
	const Vec3 white = { 1.0f, 1.0f, 1.0f };
	const float corners[4][2] = { { -0.5f, -0.5f }, { 0.5f, -0.5f }, { 0.5f, 0.5f }, { -0.5f, 0.5f } };
	const float uvs[4][2] = { { 0.0f, 1.0f }, { 1.0f, 1.0f }, { 1.0f, 0.0f }, { 0.0f, 0.0f } };

	for (int i = 0; i < 4; ++i) {
		Vertex v = makeVertex({ corners[i][0], corners[i][1], 0.0f }, white);
		v.uv[0] = uvs[i][0];
		v.uv[1] = uvs[i][1];
		mesh.vertices.push_back(v);
	}

	// два треугольника против часовой стрелки, если смотреть с +z
	const uint32_t quad[6] = { 0, 1, 2, 0, 2, 3 };
	for (uint32_t k : quad) {
		mesh.indices.push_back(k);
	}

	generateFlatNormals(mesh);
	return mesh;
}

// ---------------------------------------------------------------------------
// сфера единичного радиуса (для визуализации источников света)
// ---------------------------------------------------------------------------

inline MeshData makeSphere(int stacks, int slices) {
	constexpr float PI = 3.14159265358979f;
	MeshData mesh;

	for (int i = 0; i <= stacks; ++i) {
		const float theta = PI * float(i) / float(stacks); // от северного полюса к южному
		for (int j = 0; j <= slices; ++j) {
			const float phi = 2.0f * PI * float(j) / float(slices);
			const Vec3 p = {
				std::sin(theta) * std::cos(phi),
				std::cos(theta),
				std::sin(theta) * std::sin(phi),
			};
			Vertex v = makeVertex(p, { 1.0f, 1.0f, 1.0f });
			storeVec3(p, v.normal); // у единичной сферы нормаль совпадает с позицией
			v.uv[0] = float(j) / float(slices);
			v.uv[1] = float(i) / float(stacks);
			mesh.vertices.push_back(v);
		}
	}

	const uint32_t row = uint32_t(slices + 1);
	for (int i = 0; i < stacks; ++i) {
		for (int j = 0; j < slices; ++j) {
			const uint32_t a = uint32_t(i) * row + uint32_t(j);
			const uint32_t b = a + row;
			mesh.indices.insert(mesh.indices.end(), { a, a + 1, b, a + 1, b + 1, b });
		}
	}

	return mesh;
}
