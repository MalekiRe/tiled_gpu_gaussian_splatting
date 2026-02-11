@group(0) @binding(0) var accum_tex: texture_2d<f32>;
@group(0) @binding(1) var revealage_tex: texture_2d<f32>;

@group(1) @binding(0) var<uniform> use_revealage: u32;

struct CompositeOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> CompositeOutput {
    var out: CompositeOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index & 2u) * 2 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs_main(in: CompositeOutput) -> @location(0) vec4<f32> {
    let coords = vec2<i32>(in.position.xy);

    let accum = textureLoad(accum_tex, coords, 0);
    let avg_color = accum.rgb / max(accum.a, 1e-5);

    var alpha: f32;
    if (use_revealage != 0u) {
        let revealage = textureLoad(revealage_tex, coords, 0).r;
        alpha = 1.0 - revealage;
    } else {
        alpha = 1.0 - exp(-accum.a);
    }

    return vec4<f32>(avg_color * alpha, alpha);
}
