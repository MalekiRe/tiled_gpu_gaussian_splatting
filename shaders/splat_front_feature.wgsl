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
    let seed = pixel.x * 0x9e3779b9u ^ pixel.y * 0x85ebca6bu ^ splat_index;
    return f32(hash_u32(seed)) * (1.0 / 4294967296.0);
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
        (linear_z - camera.near) / (camera.far - camera.near),
        0.0,
        1.0,
    );
    return vec4<f32>(in.normal, normalized_z);
}
