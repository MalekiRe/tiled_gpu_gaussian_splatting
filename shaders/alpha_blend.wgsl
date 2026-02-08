// common.wgsl is prepended

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    return basic_vertex(input);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return simple_lighting(in.world_normal, in.color);
}
