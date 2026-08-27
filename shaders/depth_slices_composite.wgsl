@group(0) @binding(0) var slice0_tex: texture_2d<f32>;
@group(0) @binding(1) var slice1_tex: texture_2d<f32>;
@group(0) @binding(2) var slice2_tex: texture_2d<f32>;
@group(0) @binding(3) var slice3_tex: texture_2d<f32>;

struct CompositeOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> CompositeOutput {
    var out: CompositeOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index & 2u) * 2 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

fn composite_slice(accum: vec4<f32>) -> vec4<f32> {
    let alpha = 1.0 - exp(-accum.a);
    let color = accum.rgb / max(accum.a, 1e-5);
    return vec4<f32>(color * alpha, alpha);
}

@fragment
fn fs_main(in: CompositeOutput) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(in.position.xy);
    let layers = array<vec4<f32>, 4>(
        composite_slice(textureLoad(slice0_tex, pixel, 0)),
        composite_slice(textureLoad(slice1_tex, pixel, 0)),
        composite_slice(textureLoad(slice2_tex, pixel, 0)),
        composite_slice(textureLoad(slice3_tex, pixel, 0)),
    );
    var color = vec3<f32>(0.0);
    var transmittance = 1.0;
    for (var i = 0u; i < 4u; i++) {
        color += transmittance * layers[i].rgb;
        transmittance *= 1.0 - layers[i].a;
    }
    return vec4<f32>(color, 1.0 - transmittance);
}
