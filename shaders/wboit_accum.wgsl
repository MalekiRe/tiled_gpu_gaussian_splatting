// common.wgsl is prepended

struct WboitOutput {
    @location(0) accum: vec4<f32>,
    @location(1) revealage: f32,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    return basic_vertex(input);
}

@fragment
fn fs_main(in: VertexOutput) -> WboitOutput {
    let lit = simple_lighting(in.world_normal, in.color);
    let alpha = lit.a;

    // Linearize depth: clip_position.w = 1/w_clip, and w_clip = eye-space distance
    let linear_z = 1.0 / in.clip_position.w;

    // Normalize to [0, 1] using camera near/far planes
    let d = clamp((linear_z - camera.near) / (camera.far - camera.near), 0.0, 1.0);

    // Exponential weight spanning the usable f16 accumulation range (~7.8 orders of magnitude)
    // d=0 (near) → 2^13 = 8192, d=1 (far) → 2^-13 ≈ 1.2e-4
    // Steeper than McGuire: steps evenly through f16 exponent bits for max layer discrimination
    let w = alpha * clamp(exp2(13.0 - 26.0 * d), 1e-4, 8192.0);

    var out: WboitOutput;
    out.accum = vec4<f32>(lit.rgb * alpha * w, alpha * w);
    out.revealage = alpha;
    return out;
}
