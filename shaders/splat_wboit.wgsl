// splat_common.wgsl is prepended.
// Mode 2: naive WBOIT accumulation. Same weight curve as the mesh path, so the two are
// directly comparable; splats just feed it far more overlapping fragments.

struct WboitOutput {
    @location(0) accum: vec4<f32>,
    @location(1) revealage: f32,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> SplatVsOut {
    return splat_vertex(vertex_index, instance_index);
}

@fragment
fn fs_main(in: SplatVsOut) -> WboitOutput {
    let alpha = splat_alpha(in);
    if (alpha < 0.0) {
        discard;
    }

    // clip_position.w is 1/w_clip in the fragment stage, and w_clip is the eye-space depth.
    let linear_z = 1.0 / in.clip_position.w;
    let d = clamp((linear_z - camera.near) / (camera.far - camera.near), 0.0, 1.0);

    // d=0 (near) -> 2^13, d=1 (far) -> 2^-13: steps evenly through the f16 exponent range.
    let w = alpha * clamp(exp2(13.0 - 26.0 * d), 1e-4, 8192.0);

    var out: WboitOutput;
    out.accum = vec4<f32>(in.color * alpha * w, alpha * w);
    out.revealage = alpha;
    return out;
}
