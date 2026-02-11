struct HistoParams {
    tile_count_x: u32,
    tile_count_y: u32,
    num_bins: u32,
    tile_size: u32,
};

const OD_SCALE: f32 = 4096.0;

@group(0) @binding(0) var<storage, read_write> histogram: array<atomic<u32>>;
@group(0) @binding(1) var cdf_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var<uniform> histo_params: HistoParams;

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

    // Load and dequantize this tile's histogram bin
    var val: f32 = 0.0;
    if (bin < nb) {
        val = f32(atomicLoad(&histogram[tile_idx * nb + bin])) / OD_SCALE;
    }
    buf_a[bin] = val;
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

    // buf_a now has inclusive prefix sum
    if (bin < nb) {
        let total_od = buf_a[nb - 1u];
        var cdf_val: f32;
        if (total_od > 0.0) {
            cdf_val = buf_a[bin] / total_od;
        } else {
            // Linear fallback when no fragments hit this tile
            cdf_val = f32(bin + 1u) / f32(nb);
        }

        textureStore(cdf_out, vec3i(i32(tile_x), i32(tile_y), i32(bin)), vec4f(cdf_val, 0.0, 0.0, 0.0));

        // Clear histogram for next frame
        atomicStore(&histogram[tile_idx * nb + bin], 0u);
    }
}
