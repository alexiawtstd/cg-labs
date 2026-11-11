#pragma once

#include <cmath>

// простая математика для лабораторной работы.
// матрица хранится column-major: m[столбец][строка] - так же, как mat4 в glsl,
// поэтому массив можно напрямую копировать в uniform-буфер.
// система координат: правая, камера смотрит вдоль -z.
// проекции учитывают особенности vulkan: глубина в [0, 1] и ось y экрана
// направлена вниз (поэтому в проекциях стоит минус в элементе [1][1]).

// ---------------------------------------------------------------------------
// трёхмерный вектор
// ---------------------------------------------------------------------------

struct Vec3 {
	float x, y, z;
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
inline Vec3 operator-(Vec3 a) { return { -a.x, -a.y, -a.z }; }
inline Vec3 operator*(Vec3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
inline Vec3 operator*(float s, Vec3 a) { return a * s; }

inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

inline Vec3 cross(Vec3 a, Vec3 b) {
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x,
	};
}

inline float length(Vec3 a) { return std::sqrt(dot(a, a)); }

inline Vec3 normalize(Vec3 a) {
	const float len = length(a);
	return len > 1e-6f ? a * (1.0f / len) : Vec3{ 0.0f, 0.0f, 0.0f };
}

inline Vec3 toVec3(const float v[3]) { return { v[0], v[1], v[2] }; }

inline void storeVec3(Vec3 a, float out[3]) {
	out[0] = a.x;
	out[1] = a.y;
	out[2] = a.z;
}

// направление по двум углам (в радианах):
// yaw - поворот вокруг оси y (0 - смотрим вдоль -z, положительный - вправо, к +x),
// pitch - наклон вверх/вниз (положительный - вверх)
inline Vec3 directionFromYawPitch(float yaw, float pitch) {
	return {
		std::sin(yaw) * std::cos(pitch),
		std::sin(pitch),
		-std::cos(yaw) * std::cos(pitch),
	};
}

// ---------------------------------------------------------------------------
// матрица 4x4
// ---------------------------------------------------------------------------

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

inline Mat4 mat4Translate(Vec3 v) { return mat4Translate(v.x, v.y, v.z); }

inline Mat4 mat4Scale(float x, float y, float z) {
	Mat4 r = mat4Identity();
	r.m[0][0] = x;
	r.m[1][1] = y;
	r.m[2][2] = z;
	return r;
}

// углы в радианах
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

// перспективная проекция. fov_y в радианах, aspect = ширина / высота.
inline Mat4 mat4Perspective(float fov_y, float aspect, float near_z, float far_z) {
	Mat4 r = {};
	float f = 1.0f / std::tan(fov_y * 0.5f);
	r.m[0][0] = f / aspect;
	r.m[1][1] = -f;                                  // vulkan: y вниз
	r.m[2][2] = far_z / (near_z - far_z);
	r.m[2][3] = -1.0f;                               // w_clip = -z_view
	r.m[3][2] = near_z * far_z / (near_z - far_z);
	return r;
}

// ортографическая проекция. height - видимая высота сцены в мировых единицах.
inline Mat4 mat4Ortho(float height, float aspect, float near_z, float far_z) {
	Mat4 r = {};
	float h = height * 0.5f;
	r.m[0][0] = 1.0f / (h * aspect);
	r.m[1][1] = -1.0f / h;                           // vulkan: y вниз
	r.m[2][2] = 1.0f / (near_z - far_z);
	r.m[3][2] = near_z / (near_z - far_z);
	r.m[3][3] = 1.0f;
	return r;
}

// матрица вида камеры = обратная матрица "модели" камеры (лекция 4).
// модель камеры: M = T(position) * Ry(-yaw) * Rx(pitch),
// значит V = M^-1 = Rx(-pitch) * Ry(yaw) * T(-position)
// (обратная к композиции - произведение обратных в обратном порядке).
inline Mat4 mat4View(Vec3 position, float yaw, float pitch) {
	return mat4Mul(mat4RotateX(-pitch),
	               mat4Mul(mat4RotateY(yaw),
	                       mat4Translate(-position.x, -position.y, -position.z)));
}

// матрица нормалей: обратная транспонированная к левому верхнему блоку 3x3
// матрицы модели. нужна, чтобы нормали оставались перпендикулярными
// поверхности при неравномерном масштабировании.
// результат кладём в mat4 (так его удобнее передавать в uniform-буфер по std140).
inline Mat4 mat4NormalMatrix(const Mat4& model) {
	// элементы блока 3x3 по строкам: a b c / d e f / g h i
	const float a = model.m[0][0], b = model.m[1][0], c = model.m[2][0];
	const float d = model.m[0][1], e = model.m[1][1], f = model.m[2][1];
	const float g = model.m[0][2], h = model.m[1][2], i = model.m[2][2];

	// алгебраические дополнения
	const float c00 = e * i - f * h, c01 = -(d * i - f * g), c02 = d * h - e * g;
	const float c10 = -(b * i - c * h), c11 = a * i - c * g, c12 = -(a * h - b * g);
	const float c20 = b * f - c * e, c21 = -(a * f - c * d), c22 = a * e - b * d;

	float det = a * c00 + b * c01 + c * c02;
	if (std::fabs(det) < 1e-8f) {
		det = 1e-8f;
	}
	const float inv = 1.0f / det;

	// (A^-1)^T = (матрица дополнений) / det
	Mat4 r = mat4Identity();
	r.m[0][0] = c00 * inv; r.m[1][0] = c01 * inv; r.m[2][0] = c02 * inv;
	r.m[0][1] = c10 * inv; r.m[1][1] = c11 * inv; r.m[2][1] = c12 * inv;
	r.m[0][2] = c20 * inv; r.m[1][2] = c21 * inv; r.m[2][2] = c22 * inv;
	return r;
}
