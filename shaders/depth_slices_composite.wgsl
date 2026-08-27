@group(0) @binding(0) var slice0_tex: texture_2d<f32>;
@group(0) @binding(1) var slice1_tex: texture_2d<f32>;
@group(0) @binding(2) var slice2_tex: texture_2d<f32>;
@group(0) @binding(3) var slice3_tex: texture_2d<f32>;
@group(0) @binding(4) var front_color_filtered: texture_2d<f32>;

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

fn composite_pixel(pixel: vec2<i32>) -> vec4<f32> {
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

@fragment
fn fs_main(in: CompositeOutput) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(in.position.xy);
    let center = composite_pixel(pixel);
    if (center.a <= 1e-5) {
        return center;
    }
    let center_color = center.rgb / center.a;
    let guide = textureLoad(front_color_filtered, pixel, 0);
    let guide_confidence = smoothstep(1.0, 9.0, guide.a)
        * smoothstep(0.50, 0.95, center.a);
    let guided_color = mix(center_color, guide.rgb, 0.05 * guide_confidence);
    return vec4f(guided_color * center.a, center.a);
}
