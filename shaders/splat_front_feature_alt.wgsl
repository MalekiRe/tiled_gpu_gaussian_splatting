// splat_common.wgsl is prepended.
// Mode 7's second independent stochastic front-surface realization.

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> SplatVsOut {
    return splat_vertex(vertex_index, instance_index);
}

fn hash_u32(value: u32) -> u32 {
    var x = value;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

fn stochastic_sample(pixel: vec2<u32>, splat_index: u32) -> f32 {
    let phase = hash_u32(splat_index);
    let x = (pixel.x + phase) & 15u;
    let y = (pixel.y + (phase >> 4u)) & 15u;
    let diagonal = x ^ y;
    var threshold = 0u;
    for (var bit = 0u; bit < 4u; bit++) {
        threshold |= ((diagonal >> bit) & 1u) << (7u - 2u * bit);
        threshold |= ((y >> bit) & 1u) << (6u - 2u * bit);
    }
    let sample = (f32(threshold) + 0.5) / 256.0;
    return 1.0 - sample;
}

fn encode_octahedral(normal: vec3<f32>) -> vec2<f32> {
    var projected = normal.xy / (abs(normal.x) + abs(normal.y) + abs(normal.z));
    if (normal.z < 0.0) {
        projected = (1.0 - abs(projected.yx)) * sign(projected.xy);
    }
    return projected * 0.5 + 0.5;
}

struct FrontFeatureOutput {
    @location(0) feature: vec4<f32>,
    @location(1) color: vec4<f32>,
};

@fragment
fn fs_main(in: SplatVsOut) -> FrontFeatureOutput {
    let alpha = splat_alpha(in);
    if (alpha < 0.0) {
        discard;
    }
    let pixel = vec2<u32>(in.clip_position.xy);
    if (stochastic_sample(pixel, in.splat_index) > alpha) {
        discard;
    }
    let linear_z = 1.0 / in.clip_position.w;
    let normalized_z = clamp(
        (linear_z - camera.depth_min) / camera.depth_range,
        0.0,
        1.0,
    );
    let rgb3 = round(clamp(in.color, vec3<f32>(0.0), vec3<f32>(1.0)) * 7.0);
    let packed_rgb = rgb3.r + 8.0 * rgb3.g + 64.0 * rgb3.b;
    var out: FrontFeatureOutput;
    out.feature = vec4<f32>(
        encode_octahedral(in.normal),
        normalized_z,
        1.0 + packed_rgb / 512.0,
    );
    out.color = vec4f(clamp(in.color, vec3f(0.0), vec3f(1.0)), 1.0);
    return out;
}
