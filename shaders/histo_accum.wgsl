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
@group(2) @binding(3) var prev_revealage_tex: texture_2d<f32>;

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

    // Always use tile_index = 0 for global histogram
    let tile_index = 0u;

    // Record this fragment in the histogram
    let histo_idx = tile_index * histo_params.num_bins + bin;
    atomicAdd(&histogram[histo_idx], 1u);

    // --- CDF lookup with piecewise-linear interpolation ---
    let nb = histo_params.num_bins;

    // Piecewise-linear CDF interpolation for smooth equalized depth.
    // Without interpolation, equalized_z is a step function that jumps
    // at bin boundaries, creating visible contour-line artifacts.
    // Interpolating between cdf[bin-1] (left edge) and cdf[bin] (right edge)
    // using the fractional position within the bin makes equalized_z continuous.
    let fbin = clamp(normalized_z * f32(nb), 0.0, f32(nb));
    let bin_lo = min(u32(floor(fbin)), nb - 1u);
    let t = fbin - f32(bin_lo);
    let cdf_hi = cdf[bin_lo];
    let cdf_lo = select(cdf[bin_lo - 1u], 0.0, bin_lo == 0u);
    let equalized_z = mix(cdf_lo, cdf_hi, t);

    // Transmittance-based weight derived from compositing equation.
    // True weight for layer i is T_i = (1-a)^i = R^(i/N) where R = total revealage.
    // With histogram equalization, equalized_z ≈ i/N, giving w = R^equalized_z.
    // Exact under uniform-alpha assumption. R is per-pixel from previous frame.
    let prev_R = textureLoad(prev_revealage_tex, vec2<i32>(in.clip_position.xy), 0).r;
    let w = pow(max(prev_R, 1e-4), equalized_z);

    var out: WboitOutput;
    out.accum = vec4<f32>(lit.rgb * alpha * w, alpha * w);
    out.revealage = alpha;
    return out;
}
