struct HistoParams {
    tile_count_x: u32,
    tile_count_y: u32,
    num_bins: u32,
    tile_size: u32,
};

struct DirectionalPriorParams {
    mix_factor: f32,
    enabled: u32,
    _padding: vec2<u32>,
};

const OD_SCALE: f32 = 4096.0;

@group(0) @binding(0) var<storage, read_write> histogram: array<atomic<u32>>;
@group(0) @binding(1) var cdf_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var<uniform> histo_params: HistoParams;
@group(0) @binding(3) var<storage, read> directional_prior: array<f32>;
@group(0) @binding(4) var<uniform> prior_params: DirectionalPriorParams;

// Shared memory for Hillis-Steele prefix sum (64 bins)
var<workgroup> buf_a: array<f32, 64>;
var<workgroup> buf_b: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tile_x = wg.x;
    let tile_y = wg.y;
    let bin = lid.x;
    let nb = histo_params.num_bins;
    let tile_idx = tile_y * histo_params.tile_count_x + tile_x;

    // Load and dequantize this tile's live histogram bin.
    var val: f32 = 0.0;
    if (bin < nb) {
        val = f32(atomicLoad(&histogram[tile_idx * nb + bin])) / OD_SCALE;
    }
    buf_a[bin] = val;
    workgroupBarrier();

    // Find this tile's total optical depth. The baked histogram is normalized, so scaling
    // it by the live total makes mix_factor a true distribution blend rather than an
    // arbitrary scene-size-dependent amplitude.
    if (bin < 32u) { buf_a[bin] += buf_a[bin + 32u]; }
    workgroupBarrier();
    if (bin < 16u) { buf_a[bin] += buf_a[bin + 16u]; }
    workgroupBarrier();
    if (bin < 8u) { buf_a[bin] += buf_a[bin + 8u]; }
    workgroupBarrier();
    if (bin < 4u) { buf_a[bin] += buf_a[bin + 4u]; }
    workgroupBarrier();
    if (bin < 2u) { buf_a[bin] += buf_a[bin + 2u]; }
    workgroupBarrier();
    if (bin < 1u) { buf_a[bin] += buf_a[bin + 1u]; }
    workgroupBarrier();

    let live_total = buf_a[0];
    var mix_factor = 0.0;
    if (prior_params.enabled != 0u) {
        mix_factor = clamp(prior_params.mix_factor, 0.0, 1.0);
    }
    let prior_scale = max(live_total, 1.0);
    let bin_od = val * (1.0 - mix_factor)
        + directional_prior[bin] * prior_scale * mix_factor;
    buf_a[bin] = bin_od;
    workgroupBarrier();

    // Hillis-Steele inclusive prefix sum (6 steps for 64 bins)
    // Step 1: stride=1
    if (bin >= 1u) { buf_b[bin] = buf_a[bin] + buf_a[bin - 1u]; } else { buf_b[bin] = buf_a[bin]; }
    workgroupBarrier();
    // Step 2: stride=2
    if (bin >= 2u) { buf_a[bin] = buf_b[bin] + buf_b[bin - 2u]; } else { buf_a[bin] = buf_b[bin]; }
    workgroupBarrier();
    // Step 3: stride=4
    if (bin >= 4u) { buf_b[bin] = buf_a[bin] + buf_a[bin - 4u]; } else { buf_b[bin] = buf_a[bin]; }
    workgroupBarrier();
    // Step 4: stride=8
    if (bin >= 8u) { buf_a[bin] = buf_b[bin] + buf_b[bin - 8u]; } else { buf_a[bin] = buf_b[bin]; }
    workgroupBarrier();
    // Step 5: stride=16
    if (bin >= 16u) { buf_b[bin] = buf_a[bin] + buf_a[bin - 16u]; } else { buf_b[bin] = buf_a[bin]; }
    workgroupBarrier();
    // Step 6: stride=32
    if (bin >= 32u) { buf_a[bin] = buf_b[bin] + buf_b[bin - 32u]; } else { buf_a[bin] = buf_b[bin]; }
    workgroupBarrier();

    // Store the optical depth strictly in front of this bin. Texture filtering then
    // interpolates continuously between true bin edges rather than counting the current
    // fragment's entire bin against itself.
    if (bin < nb) {
        let total_od = buf_a[nb - 1u];
        var cdf_val: f32;
        if (total_od > 0.0) {
            cdf_val = (buf_a[bin] - bin_od) / total_od;
        } else {
            // Linear fallback when no fragments hit this tile
            cdf_val = f32(bin) / f32(nb);
        }

        textureStore(cdf_out, vec3i(i32(tile_x), i32(tile_y), i32(bin)), vec4f(cdf_val, 0.0, 0.0, 0.0));

        // Clear histogram for next frame
        atomicStore(&histogram[tile_idx * nb + bin], 0u);
    }
}
