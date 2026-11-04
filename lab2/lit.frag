#version 450 core

// фрагментный шейдер: модель освещения блинна-фонга
//   color = ambient * albedo
//     + sum_по_источникам [ albedo + specular * (n . h)^shininess ] * (n . l) * E
// где E - освещённость от источника (цвет * интенсивность, с затуханием)

layout(location = 0) in vec3 in_world_position;
layout(location = 1) in vec3 in_world_normal;
layout(location = 2) in vec3 in_color;

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
} object;

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
vec3 blinnPhong(vec3 n, vec3 v, vec3 l, vec3 radiance, vec3 albedo) {
    float n_dot_l = max(dot(n, l), 0.0);
    vec3 h = normalize(l + v); // половинчатый вектор
    float spec = pow(max(dot(n, h), 0.0), object.shininess);
    return (albedo + object.specular * spec) * n_dot_l * radiance;
}

void main() {
    vec3 n = normalize(in_world_normal);
    vec3 v = normalize(scene.camera_position - in_world_position);

    // альбедо материала, по желанию умноженное на процедурный цвет вершины
    vec3 albedo = object.albedo * mix(vec3(1.0), in_color, object.use_vertex_color);

    // рассеянный свет
    vec3 color = scene.ambient_light * albedo;

    // направленный источник: лучи параллельны, затухания нет
    {
        vec3 l = normalize(-scene.directional_light_direction);
        color += blinnPhong(n, v, l, scene.directional_light_color, albedo);
    }

    // точечные источники: затухание по закону обратных квадратов
    for (uint i = 0u; i < scene.point_light_count; ++i) {
        PointLight light = point_lights[i];

        vec3 to_light = light.position - in_world_position;
        float distance2 = max(dot(to_light, to_light), 1e-4);
        vec3 l = to_light * inversesqrt(distance2);

        vec3 radiance = light.color * light.intensity / distance2;
        color += blinnPhong(n, v, l, radiance, albedo);
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
        color += blinnPhong(n, v, l, radiance, albedo);
    }

    out_color = vec4(color, 1.0);
}
