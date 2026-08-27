// splat_common.wgsl is prepended.
// Mode 6 prepass: stochastic alpha testing plus normal depth testing picks one plausible
// front splat at each pixel without sorting or fragment atomics.

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
    let x = (pixel.x + phase) & 3u;
    let y = (pixel.y + (phase >> 2u)) & 3u;
    let bayer = array<u32, 16>(
         0u,  8u,  2u, 10u,
        12u,  4u, 14u,  6u,
         3u, 11u,  1u,  9u,
        15u,  7u, 13u,  5u,
    );
    return (f32(bayer[y * 4u + x]) + 0.5) / 16.0;
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

    // Stable high-opacity core. The companion pass handles the Gaussian fringe with
    // ordered stochastic coverage, so this pass contains no screen-space noise.
    if (alpha < 0.15) {
        discard;
    }

    let linear_z = 1.0 / in.clip_position.w;
    let normalized_z = clamp(
        (linear_z - camera.depth_min) / camera.depth_range,
        0.0,
        1.0,
    );
    let luminance = clamp(dot(in.color, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
    return vec4<f32>(encode_octahedral(in.normal), normalized_z, 1.0 + luminance);
}
