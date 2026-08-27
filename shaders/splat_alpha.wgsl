// splat_common.wgsl is prepended.
// Mode 1: ordinary back-to-front alpha blending. Correctness here depends entirely on the
// CPU depth sort feeding splat_order.

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> SplatVsOut {
    return splat_vertex(vertex_index, instance_index);
}

@fragment
fn fs_main(in: SplatVsOut) -> @location(0) vec4<f32> {
    let alpha = splat_alpha(in);
    if (alpha < 0.0) {
        discard;
    }
    return vec4<f32>(in.color, alpha);
}
