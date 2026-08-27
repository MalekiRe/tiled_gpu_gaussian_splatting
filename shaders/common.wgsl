struct Camera {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    near: f32,
    far: f32,
    focal: vec2<f32>,
    viewport: vec2<f32>,
    cam_pos: vec3<f32>,
};

struct Object {
    model: mat4x4<f32>,
    color: vec4<f32>,
};

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> object: Object;

fn basic_vertex(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = object.model * vec4<f32>(input.position, 1.0);
    out.clip_position = camera.view_proj * world_pos;
    out.world_normal = normalize((object.model * vec4<f32>(input.normal, 0.0)).xyz);
    out.color = input.color * object.color;
    return out;
}

fn simple_lighting(normal: vec3<f32>, base_color: vec4<f32>) -> vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.8));
    let ambient = 0.3;
    let diffuse = max(dot(normalize(normal), light_dir), 0.0) * 0.7;
    let lit = base_color.rgb * (ambient + diffuse);
    return vec4<f32>(lit, base_color.a);
}
