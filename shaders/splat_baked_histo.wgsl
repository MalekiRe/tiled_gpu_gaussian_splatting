// splat_common.wgsl is prepended.
// Mode 5: sample the spatially baked directional CDF without recording a live histogram.
// This deliberately avoids fragment-stage storage atomics for mobile/tiled GPUs.

const TILE_SIZE: u32 = 32u;

struct HistoParams {
    tile_count_x: u32,
    tile_count_y: u32,
    num_bins: u32,
    tile_size: u32,
};

@group(2) @binding(1) var cdf_texture: texture_3d<f32>;
@group(2) @binding(2) var cdf_sampler: sampler;
@group(2) @binding(3) var<uniform> histo_params: HistoParams;
@group(2) @binding(4) var prev_revealage_tex: texture_2d<f32>;

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

    let linear_z = 1.0 / in.clip_position.w;
    let normalized_z = clamp(
        (linear_z - camera.near) / (camera.far - camera.near),
        0.0,
        1.0,
    );

    let u = in.clip_position.x / f32(histo_params.tile_count_x * TILE_SIZE);
    let v = in.clip_position.y / f32(histo_params.tile_count_y * TILE_SIZE);
    let equalized_z = textureSampleLevel(
        cdf_texture,
        cdf_sampler,
        vec3f(u, v, normalized_z),
        0.0,
    ).r;

    let prev_r = textureLoad(prev_revealage_tex, vec2<i32>(in.clip_position.xy), 0).r;
    let wt = pow(max(prev_r, 1e-4), equalized_z);

    var out: WboitOutput;
    out.accum = vec4<f32>(in.color * alpha * wt, alpha * wt);
    out.revealage = alpha;
    return out;
}
