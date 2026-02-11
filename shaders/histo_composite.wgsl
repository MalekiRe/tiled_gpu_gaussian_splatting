struct HistoParams {
    num_bins: u32,
    depth_range: f32,
};

// Must match histo_accum.wgsl
const OD_SCALE: f32 = 4096.0;

@group(0) @binding(0) var accum_tex: texture_2d<f32>;
@group(0) @binding(1) var revealage_tex: texture_2d<f32>;

@group(1) @binding(0) var<storage, read_write> histogram: array<atomic<u32>>;
@group(1) @binding(1) var<storage, read_write> cdf: array<f32>;
@group(1) @binding(2) var<uniform> histo_params: HistoParams;

@group(2) @binding(0) var<uniform> use_revealage: u32;

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

    // 1. WBOIT composite
    let accum = textureLoad(accum_tex, coords, 0);
    let avg_color = accum.rgb / max(accum.a, 1e-5);

    // Compute alpha: use revealage if enabled, otherwise use exponential approximation
    var alpha: f32;
    if (use_revealage != 0u) {
        let revealage = textureLoad(revealage_tex, coords, 0).r;
        alpha = 1.0 - revealage;
    } else {
        alpha = 1.0 - exp(-accum.a);
    }

    // Only pixel (0,0) builds the global CDF from optical depth histogram
    let should_build_cdf = (u32(in.position.x) == 0u && u32(in.position.y) == 0u);

    if (should_build_cdf) {
        let nb = histo_params.num_bins;

        // Prefix-sum of optical depth (dequantized from u32 atomics).
        // Unlike a count histogram, this weights each bin by the actual occlusion
        // contributed by its fragments, so high-alpha clusters get proportionally
        // more of the CDF range.
        var prefix_sum: f32 = 0.0;
        for (var b = 0u; b < nb; b = b + 1u) {
            let raw_od = f32(atomicLoad(&histogram[b])) / OD_SCALE;
            prefix_sum += raw_od;
            cdf[b] = prefix_sum;
        }

        let total_od = prefix_sum;

        // Normalize and clear histogram for next frame
        for (var b = 0u; b < nb; b = b + 1u) {
            if total_od > 0.0 {
                cdf[b] = cdf[b] / total_od;
            } else {
                cdf[b] = f32(b + 1u) / f32(nb);
            }
            atomicStore(&histogram[b], 0u);
        }
    }

    return vec4<f32>(avg_color * alpha, alpha);
}
