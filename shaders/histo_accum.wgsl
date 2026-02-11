// common.wgsl is prepended

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
fn vs_main(input: VertexInput) -> VertexOutput {
    return basic_vertex(input);
}

@fragment
fn fs_main(in: VertexOutput) -> WboitOutput {
    let lit = simple_lighting(in.world_normal, in.color);
    let alpha = lit.a;

    // Linearize depth
    let linear_z = 1.0 / in.clip_position.w;
    let normalized_z = clamp(
        (linear_z - camera.near) / (camera.far - camera.near),
        0.0,
        1.0,
    );

    let nb = histo_params.num_bins;
    let bin = min(u32(normalized_z * f32(nb)), nb - 1u);

    // Tiled atomic: only fragments in this tile compete
    let tile_x = u32(in.clip_position.x) / TILE_SIZE;
    let tile_y = u32(in.clip_position.y) / TILE_SIZE;
    let tile_idx = tile_y * histo_params.tile_count_x + tile_x;

    let optical_depth = -log(max(1.0 - alpha, 1e-6));
    let quantized_od = u32(clamp(optical_depth * OD_SCALE, 0.0, 65535.0));
    atomicAdd(&histogram[tile_idx * nb + bin], quantized_od);

    // Sample CDF from 3D texture — hardware trilinear gives free spatial + depth interpolation
    let u = in.clip_position.x / f32(histo_params.tile_count_x * TILE_SIZE);
    let v = in.clip_position.y / f32(histo_params.tile_count_y * TILE_SIZE);
    let w = normalized_z;
    let equalized_z = textureSampleLevel(cdf_texture, cdf_sampler, vec3f(u, v, w), 0.0).r;

    // Transmittance weight from previous frame's per-pixel revealage
    let prev_R = textureLoad(prev_revealage_tex, vec2<i32>(in.clip_position.xy), 0).r;
    let wt = pow(max(prev_R, 1e-4), equalized_z);

    var out: WboitOutput;
    out.accum = vec4<f32>(lit.rgb * alpha * wt, alpha * wt);
    out.revealage = alpha;
    return out;
}
