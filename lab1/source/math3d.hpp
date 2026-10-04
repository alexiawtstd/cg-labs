#pragma once

#include <cmath>

// Простая матрица 4x4 для лабораторной работы.
// Хранение column-major: m[столбец][строка] -- так же, как mat4 в GLSL,
// поэтому массив можно напрямую копировать в uniform-буфер.
// Система координат: правая, камера смотрит вдоль -Z.
// Проекции учитывают особенности Vulkan: глубина в [0, 1] и ось Y экрана
// направлена вниз (поэтому в проекциях стоит минус в элементе [1][1]).
struct Mat4 {
	float m[4][4];
};

inline Mat4 mat4Identity() {
	Mat4 r = {};
	r.m[0][0] = r.m[1][1] = r.m[2][2] = r.m[3][3] = 1.0f;
	return r;
}

// r = a * b (сначала применяется b, потом a)
inline Mat4 mat4Mul(const Mat4& a, const Mat4& b) {
	Mat4 r = {};
	for (int c = 0; c < 4; ++c)
		for (int row = 0; row < 4; ++row)
			for (int k = 0; k < 4; ++k)
				r.m[c][row] += a.m[k][row] * b.m[c][k];
	return r;
}

inline Mat4 mat4Translate(float x, float y, float z) {
	Mat4 r = mat4Identity();
	r.m[3][0] = x;
	r.m[3][1] = y;
	r.m[3][2] = z;
	return r;
}

inline Mat4 mat4Scale(float x, float y, float z) {
	Mat4 r = mat4Identity();
	r.m[0][0] = x;
	r.m[1][1] = y;
	r.m[2][2] = z;
	return r;
}

// Углы в радианах
inline Mat4 mat4RotateX(float a) {
	Mat4 r = mat4Identity();
	float c = std::cos(a), s = std::sin(a);
	r.m[1][1] = c;  r.m[2][1] = -s;
	r.m[1][2] = s;  r.m[2][2] = c;
	return r;
}

inline Mat4 mat4RotateY(float a) {
	Mat4 r = mat4Identity();
	float c = std::cos(a), s = std::sin(a);
	r.m[0][0] = c;  r.m[2][0] = s;
	r.m[0][2] = -s; r.m[2][2] = c;
	return r;
}

inline Mat4 mat4RotateZ(float a) {
	Mat4 r = mat4Identity();
	float c = std::cos(a), s = std::sin(a);
	r.m[0][0] = c;  r.m[1][0] = -s;
	r.m[0][1] = s;  r.m[1][1] = c;
	return r;
}

// Перспективная проекция. fov_y в радианах, aspect = ширина / высота.
inline Mat4 mat4Perspective(float fov_y, float aspect, float near_z, float far_z) {
	Mat4 r = {};
	float f = 1.0f / std::tan(fov_y * 0.5f);
	r.m[0][0] = f / aspect;
	r.m[1][1] = -f;                                  // Vulkan: Y вниз
	r.m[2][2] = far_z / (near_z - far_z);
	r.m[2][3] = -1.0f;                               // w_clip = -z_view
	r.m[3][2] = near_z * far_z / (near_z - far_z);
	return r;
}

// Ортографическая проекция. height -- видимая высота сцены в мировых единицах.
inline Mat4 mat4Ortho(float height, float aspect, float near_z, float far_z) {
	Mat4 r = {};
	float h = height * 0.5f;
	r.m[0][0] = 1.0f / (h * aspect);
	r.m[1][1] = -1.0f / h;                           // Vulkan: Y вниз
	r.m[2][2] = 1.0f / (near_z - far_z);
	r.m[3][2] = near_z / (near_z - far_z);
	r.m[3][3] = 1.0f;
	return r;
}
