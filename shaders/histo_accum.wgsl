// common.wgsl is prepended

struct HistoParams {
    tile_count_x: u32,
    tile_count_y: u32,
    num_bins: u32,
    depth_range: f32,
};

@group(2) @binding(0) var<storage, read_write> histogram: array<atomic<u32>>;
@group(2) @binding(1) var<storage, read> cdf: array<f32>;
@group(2) @binding(2) var<uniform> histo_params: HistoParams;

struct WboitOutput {
    @location(0) accum: vec4<f32>,
    @location(1) revealage: f32,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    return basic_vertex(input);
}
fn trilinear_cdf_sample(
    normalized_z: f32,
    clip_pos_xy: vec2<f32>,
    nb: u32,
    tcx: u32,
    tcy: u32,
) -> f32 {
    // Z: continuous bin coordinate, centered so bin centers are at integers
    let fbin = clamp(normalized_z * f32(nb) - 0.5, 0.0, f32(nb - 1u));
    let bin_lo = u32(floor(fbin));
    let bin_hi = min(bin_lo + 1u, nb - 1u);
    let tz = fbin - f32(bin_lo);

    // XY: continuous tile coordinates centered on tile centers (center of tile i = pixel i*16+8)
    let cx = clamp((clip_pos_xy.x - 8.0) / 16.0, 0.0, f32(tcx - 1u));
    let cy = clamp((clip_pos_xy.y - 8.0) / 16.0, 0.0, f32(tcy - 1u));
    let tx0 = u32(floor(cx));
    let tx1 = min(tx0 + 1u, tcx - 1u);
    let ty0 = u32(floor(cy));
    let ty1 = min(ty0 + 1u, tcy - 1u);
    let fx = cx - f32(tx0);
    let fy = cy - f32(ty0);

    // 8 CDF samples: 2x2 neighboring tiles × 2 adjacent depth bins
    let i00 = (ty0 * tcx + tx0) * nb;
    let i10 = (ty0 * tcx + tx1) * nb;
    let i01 = (ty1 * tcx + tx0) * nb;
    let i11 = (ty1 * tcx + tx1) * nb;

    // Bilinear XY interpolation at bin_lo
    let c_lo = mix(
        mix(cdf[i00 + bin_lo], cdf[i10 + bin_lo], fx),
        mix(cdf[i01 + bin_lo], cdf[i11 + bin_lo], fx),
        fy
    );
    // Bilinear XY interpolation at bin_hi
    let c_hi = mix(
        mix(cdf[i00 + bin_hi], cdf[i10 + bin_hi], fx),
        mix(cdf[i01 + bin_hi], cdf[i11 + bin_hi], fx),
        fy
    );

    // Linear Z interpolation between bins
    return mix(c_lo, c_hi, tz);
}

@fragment
fn fs_main(in: VertexOutput) -> WboitOutput {
    let lit = simple_lighting(in.world_normal, in.color);
    let alpha = lit.a;

    // Linearize depth: 1/clip_position.w recovers eye-space distance
    let linear_z = 1.0 / in.clip_position.w;

    // Normalize to [0, 1] using camera near/far planes
    let normalized_z = clamp(
        (linear_z - camera.near) / (camera.far - camera.near),
        0.0,
        1.0,
    );
    let bin = min(
        u32(normalized_z * f32(histo_params.num_bins)),
        histo_params.num_bins - 1u,
    );

    // Tile index (GLOBAL_HISTO const is prepended at pipeline creation)
    var tile_index = 0u;
    if (!GLOBAL_HISTO) {
        let tile_x = u32(in.clip_position.x) / 16u;
        let tile_y = u32(in.clip_position.y) / 16u;
        tile_index = tile_y * histo_params.tile_count_x + tile_x;
    }

    // Record this fragment in the histogram
    let histo_idx = tile_index * histo_params.num_bins + bin;
    atomicAdd(&histogram[histo_idx], 1u);

    // --- CDF lookup (with and without trilinear interpolation) ---
    let nb = histo_params.num_bins;
    let tcx = histo_params.tile_count_x;
    let tcy = histo_params.tile_count_y;

    // Trilinear CDF interpolation across XY (tiles) and Z (depth bins)
    // let equalized_z = trilinear_cdf_sample(
    //     normalized_z,
    //     in.clip_position.xy,
    //     nb,
    //     tcx,
    //     tcy,
    // );

    // Piecewise-linear CDF interpolation for smooth equalized depth.
    // Without interpolation, equalized_z is a step function that jumps
    // at bin boundaries, creating visible contour-line artifacts.
    // Interpolating between cdf[bin-1] (left edge) and cdf[bin] (right edge)
    // using the fractional position within the bin makes equalized_z continuous.
    let fbin = clamp(normalized_z * f32(nb), 0.0, f32(nb));
    let bin_lo = min(u32(floor(fbin)), nb - 1u);
    let t = fbin - f32(bin_lo);
    let cdf_hi = cdf[tile_index * nb + bin_lo];
    let cdf_lo = select(cdf[tile_index * nb + bin_lo - 1u], 0.0, bin_lo == 0u);
    let equalized_z = mix(cdf_lo, cdf_hi, t);

    // Exponential weight spanning the usable f16 accumulation range
    // equalized_z=0 (near) → 2^13 = 8192, equalized_z=1 (far) → 2^-13 ≈ 1.2e-4
    let w = alpha * clamp(exp2(13.0 - 26.0 * equalized_z), 1e-4, 8192.0);

    var out: WboitOutput;
    out.accum = vec4<f32>(lit.rgb * alpha * w, alpha * w);
    out.revealage = alpha;
    return out;
}
