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
    let x = (pixel.x + phase) & 7u;
    let y = (pixel.y + (phase >> 3u)) & 7u;
    let bayer = array<u32, 64>(
         0u, 32u,  8u, 40u,  2u, 34u, 10u, 42u,
        48u, 16u, 56u, 24u, 50u, 18u, 58u, 26u,
        12u, 44u,  4u, 36u, 14u, 46u,  6u, 38u,
        60u, 28u, 52u, 20u, 62u, 30u, 54u, 22u,
         3u, 35u, 11u, 43u,  1u, 33u,  9u, 41u,
        51u, 19u, 59u, 27u, 49u, 17u, 57u, 25u,
        15u, 47u,  7u, 39u, 13u, 45u,  5u, 37u,
        63u, 31u, 55u, 23u, 61u, 29u, 53u, 21u,
    );
    let sample = (f32(bayer[y * 8u + x]) + 0.5) / 64.0;
    return 1.0 - sample;
}

fn encode_octahedral(normal: vec3<f32>) -> vec2<f32> {
    var projected = normal.xy / (abs(normal.x) + abs(normal.y) + abs(normal.z));
    if (normal.z < 0.0) {
        projected = (1.0 - abs(projected.yx)) * sign(projected.xy);
    }
    return projected * 0.5 + 0.5;
}

@fragment
fn fs_main(in: SplatVsOut) -> @location(0) vec4<f32> {
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
    return vec4<f32>(encode_octahedral(in.normal), normalized_z, 1.0 + packed_rgb / 512.0);
}
