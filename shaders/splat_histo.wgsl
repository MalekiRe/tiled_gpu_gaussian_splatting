// splat_common.wgsl is prepended.
// Mode 3: histogram-equalized WBOIT. Identical machinery to the mesh path -- record this
// fragment's optical depth into the tile's depth histogram, and weight it by the
// transmittance implied by the previous frame's CDF.

const TILE_SIZE: u32 = 32u;
const OD_SCALE: f32 = 4096.0;

struct HistoParams {
    tile_count_x: u32,
    tile_count_y: u32,
    num_bins: u32,
    tile_size: u32,
};

@group(2) @binding(0) var<storage, read_write> histogram: array<atomic<u32>>;
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
        (linear_z - camera.depth_min) / camera.depth_range,
        0.0,
        1.0,
    );

    let nb = histo_params.num_bins;
    let bin = min(u32(normalized_z * f32(nb)), nb - 1u);

    let tile_x = u32(in.clip_position.x) / TILE_SIZE;
    let tile_y = u32(in.clip_position.y) / TILE_SIZE;
    let tile_idx = tile_y * histo_params.tile_count_x + tile_x;

    let optical_depth = -log(max(1.0 - alpha, 1e-6));
    let quantized_od = u32(clamp(optical_depth * OD_SCALE, 0.0, 65535.0));
    atomicAdd(&histogram[tile_idx * nb + bin], quantized_od);

    // Trilinear sampling of the CDF volume gives spatial and depth interpolation for free.
    let u = in.clip_position.x / f32(histo_params.tile_count_x * TILE_SIZE);
    let v = in.clip_position.y / f32(histo_params.tile_count_y * TILE_SIZE);
    let cdf_z = clamp(normalized_z + 0.5 / f32(nb), 0.0, 1.0);
    let equalized_z = textureSampleLevel(cdf_texture, cdf_sampler, vec3f(u, v, cdf_z), 0.0).r;

    let prev_tau = textureLoad(prev_revealage_tex, vec2<i32>(in.clip_position.xy), 0).r;
    let wt = exp(-prev_tau * equalized_z);

    var out: WboitOutput;
    out.accum = vec4<f32>(in.color * alpha * wt, alpha * wt);
    out.revealage = optical_depth;
    return out;
}
