#version 450 core

// фрагментный шейдер: модель освещения блинна-фонга с текстурами (лаб. 3)
//   albedo   = tint * цвет_из_текстуры
//   specular = specular_color * маска_блеска_из_текстуры
//   color = ambient * albedo
//     + sum_по_источникам [ albedo + specular * (n . h)^shininess ] * (n . l) * E
//     + самосвечение (emissive), которое прибавляется уже ПОСЛЕ освещения
// где E - освещённость от источника (цвет * интенсивность, с затуханием)

layout(location = 0) in vec3 in_world_position;
layout(location = 1) in vec3 in_world_normal;
layout(location = 2) in vec3 in_color;
layout(location = 3) in vec2 in_uv;

layout(location = 0) out vec4 out_color;

layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec3 camera_position;
    vec3 directional_light_direction;
    vec3 directional_light_color;
    uint point_light_count;
    vec3 ambient_light;
    uint spot_light_count;
} scene;

layout(std140, set = 1, binding = 0) uniform ModelUniforms {
    mat4 model;
    mat4 normal_matrix;
    vec3 albedo;
    float shininess;
    vec3 specular;
    float use_vertex_color;
    vec2 uv_scale;        // уже учтён в вершинном шейдере, здесь не нужен
    float emissive_strength;
} object;

// лаб. 3: set = 2 - материал. три текстуры, каждая - пара "изображение + сэмплер"
layout(set = 2, binding = 0) uniform sampler2D albedo_map;    // основной цвет
layout(set = 2, binding = 1) uniform sampler2D specular_map;  // где поверхность блестит
layout(set = 2, binding = 2) uniform sampler2D emissive_map;  // где поверхность светится сама

// лаб. 3: push-константы - быстрые данные, меняющиеся от объекта к объекту
layout(push_constant) uniform PushConstants {
    uint flags;   // бит 0 - включить нетривиальное сэмплирование
    float time;   // время анимации в секундах
} pc;

const uint FLAG_ANIMATED_SAMPLING = 1u;

// точечный источник (std430: 32 байта)
struct PointLight {
    vec3 position;
    float intensity;
    vec3 color;
    float _pad;
};

// прожекторный источник (std430: 48 байт)
struct SpotLight {
    vec3 position;
    float intensity;
    vec3 color;
    float cos_inner;   // косинус внутреннего угла (полная яркость внутри)
    vec3 direction;    // куда светит прожектор (нормализован)
    float cos_outer;   // косинус внешнего угла (за ним темно)
};

// storage-буферы: массивы без фиксированного размера в шейдере
layout(std430, set = 0, binding = 1) readonly buffer PointLights {
    PointLight point_lights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight spot_lights[];
};

// вклад одного источника по блинну-фонгу.
// n - нормаль, v - направление на камеру, l - направление на источник,
// radiance - освещённость от источника (уже с затуханием)
// specular_color - цвет блика уже с учётом маски из текстуры
vec3 blinnPhong(vec3 n, vec3 v, vec3 l, vec3 radiance, vec3 albedo, vec3 specular_color) {
    float n_dot_l = max(dot(n, l), 0.0);
    vec3 h = normalize(l + v); // половинчатый вектор
    float spec = pow(max(dot(n, h), 0.0), object.shininess);
    return (albedo + specular_color * spec) * n_dot_l * radiance;
}

// лаб. 3: нетривиальное сэмплирование. одно изображение читается ДВАЖДЫ с разными
// координатами, результаты смешиваются, а цвет пульсирует по времени:
//   слой A: текстура плывёт вдоль u;
//   слой B: другой масштаб (x1.7), поворот на 35 градусов и сдвиг вдоль v;
//   доля слоя B медленно колеблется между 0 и 1 (mix);
//   каждый цветовой канал пульсирует со своим сдвигом фазы (120 градусов).
vec3 sampleAlbedoAnimated(vec2 uv) {
    float t = pc.time;

    vec2 uv_a = uv + vec2(0.08 * t, 0.0);

    float angle = radians(35.0);
    mat2 rotation = mat2(cos(angle), sin(angle), -sin(angle), cos(angle));
    vec2 uv_b = rotation * (uv * 1.7) - vec2(0.0, 0.05 * t);

    vec3 layer_a = texture(albedo_map, uv_a).rgb;
    vec3 layer_b = texture(albedo_map, uv_b).rgb;

    vec3 blended = mix(layer_a, layer_b, 0.5 + 0.5 * sin(0.8 * t));
    vec3 pulse = 0.8 + 0.4 * vec3(sin(t), sin(t + 2.094), sin(t + 4.189));
    return blended * pulse;
}

void main() {
    vec3 n = normalize(in_world_normal);
    vec3 v = normalize(scene.camera_position - in_world_position);

    // цвет из текстуры. условие зависит только от push-константы (одинаково для всех
    // пикселей), поэтому texture() внутри ветки безопасна: производные считаются корректно
    vec3 texture_color;
    if ((pc.flags & FLAG_ANIMATED_SAMPLING) != 0u) {
        texture_color = sampleAlbedoAnimated(in_uv);
    } else {
        texture_color = texture(albedo_map, in_uv).rgb;
    }

    // альбедо = текстура * оттенок материала, по желанию * процедурный цвет вершины (лаба 1)
    vec3 albedo = texture_color * object.albedo * mix(vec3(1.0), in_color, object.use_vertex_color);

    // specular-текстура - маска блеска: белое блестит, чёрное матовое
    vec3 specular_color = object.specular * texture(specular_map, in_uv).r;

    // рассеянный свет
    vec3 color = scene.ambient_light * albedo;

    // направленный источник: лучи параллельны, затухания нет
    {
        vec3 l = normalize(-scene.directional_light_direction);
        color += blinnPhong(n, v, l, scene.directional_light_color, albedo, specular_color);
    }

    // точечные источники: затухание по закону обратных квадратов
    for (uint i = 0u; i < scene.point_light_count; ++i) {
        PointLight light = point_lights[i];

        vec3 to_light = light.position - in_world_position;
        float distance2 = max(dot(to_light, to_light), 1e-4);
        vec3 l = to_light * inversesqrt(distance2);

        vec3 radiance = light.color * light.intensity / distance2;
        color += blinnPhong(n, v, l, radiance, albedo, specular_color);
    }

    // прожекторные источники: обратные квадраты + мягкие края конуса
    for (uint i = 0u; i < scene.spot_light_count; ++i) {
        SpotLight light = spot_lights[i];

        vec3 to_light = light.position - in_world_position;
        float distance2 = max(dot(to_light, to_light), 1e-4);
        vec3 l = to_light * inversesqrt(distance2);

        // косинус угла между осью прожектора и направлением на точку
        float cos_theta = dot(-l, light.direction);

        // внутри внутреннего конуса 1, за внешним 0, между ними плавный переход.
        // если углы равны, знаменатель почти 0 и край получается жёстким
        float edge = clamp((cos_theta - light.cos_outer) /
                           max(light.cos_inner - light.cos_outer, 1e-4), 0.0, 1.0);
        edge = edge * edge * (3.0 - 2.0 * edge); // smoothstep для более мягкого края

        vec3 radiance = light.color * light.intensity * edge / distance2;
        color += blinnPhong(n, v, l, radiance, albedo, specular_color);
    }

    // свечение прибавляется ПОСЛЕ освещения: оно не зависит от источников света,
    // поэтому светящиеся места остаются яркими даже в темноте
    color += texture(emissive_map, in_uv).rgb * object.emissive_strength;

    out_color = vec4(color, 1.0);
}
