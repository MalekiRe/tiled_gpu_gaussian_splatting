// common.wgsl is prepended

struct HistoParams {
    num_bins: u32,
    depth_range: f32,
};

// Optical depth scale factor for quantizing into u32 atomics.
// -ln(1-alpha) ranges ~0.01 to ~4.6 for alpha in [0.01, 0.99].
// Scale of 4096 gives u32 values ~41 to ~18841 per fragment.
const OD_SCALE: f32 = 4096.0;

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

    // Record this fragment's optical depth in the histogram (quantized).
    // Unlike counting fragments, this weights bins by actual occlusion contribution,
    // so high-alpha fragments shift more of the CDF budget than low-alpha ones.
    let optical_depth = -log(max(1.0 - alpha, 1e-6));
    let quantized_od = u32(clamp(optical_depth * OD_SCALE, 0.0, 65535.0));
    atomicAdd(&histogram[bin], quantized_od);

    // --- Piecewise-linear CDF interpolation ---
    // The CDF maps depth bins to [0,1] based on cumulative optical depth fraction.
    // equalized_z represents "what fraction of total scene optical depth is in front."
    let nb = histo_params.num_bins;

    let fbin = clamp(normalized_z * f32(nb), 0.0, f32(nb));
    let bin_lo = min(u32(floor(fbin)), nb - 1u);
    let t = fbin - f32(bin_lo);
    let cdf_hi = cdf[bin_lo];
    let cdf_lo = select(cdf[bin_lo - 1u], 0.0, bin_lo == 0u);
    let equalized_z = mix(cdf_lo, cdf_hi, t);

    // Transmittance weight using per-pixel revealage from previous frame.
    // R = Π(1-α_i) = exp(-τ_total_pixel), so R^equalized_z = exp(-equalized_z * τ_total_pixel).
    // When equalized_z comes from an optical depth CDF, it equals τ_before / τ_total_global.
    // Under the assumption that per-pixel OD distribution ≈ global OD distribution,
    // this gives T ≈ exp(-τ_before_pixel), which is exact transmittance.
    let prev_R = textureLoad(prev_revealage_tex, vec2<i32>(in.clip_position.xy), 0).r;
    let w = pow(max(prev_R, 1e-4), equalized_z);

    var out: WboitOutput;
    out.accum = vec4<f32>(lit.rgb * alpha * w, alpha * w);
    out.revealage = alpha;
    return out;
}
