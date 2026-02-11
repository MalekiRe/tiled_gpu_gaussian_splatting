struct HistoParams {
    num_bins: u32,
    depth_range: f32,
};

const OD_SCALE: f32 = 4096.0;

@group(0) @binding(0) var<storage, read_write> histogram: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> cdf: array<f32>;
@group(0) @binding(2) var<uniform> histo_params: HistoParams;

// Inclusive prefix sum via Hillis-Steele (simpler for 256 elements, log2(256)=8 steps)
var<workgroup> buf_a: array<f32, 256>;
var<workgroup> buf_b: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let id = lid.x;
    let nb = histo_params.num_bins;

    // Load and dequantize
    var val: f32 = 0.0;
    if (id < nb) {
        val = f32(atomicLoad(&histogram[id])) / OD_SCALE;
    }
    buf_a[id] = val;
    workgroupBarrier();

    // Hillis-Steele inclusive scan (ping-pong between buf_a and buf_b)
    // Step 1: stride=1
    if (id >= 1u) { buf_b[id] = buf_a[id] + buf_a[id - 1u]; } else { buf_b[id] = buf_a[id]; }
    workgroupBarrier();
    // Step 2: stride=2
    if (id >= 2u) { buf_a[id] = buf_b[id] + buf_b[id - 2u]; } else { buf_a[id] = buf_b[id]; }
    workgroupBarrier();
    // Step 3: stride=4
    if (id >= 4u) { buf_b[id] = buf_a[id] + buf_a[id - 4u]; } else { buf_b[id] = buf_a[id]; }
    workgroupBarrier();
    // Step 4: stride=8
    if (id >= 8u) { buf_a[id] = buf_b[id] + buf_b[id - 8u]; } else { buf_a[id] = buf_b[id]; }
    workgroupBarrier();
    // Step 5: stride=16
    if (id >= 16u) { buf_b[id] = buf_a[id] + buf_a[id - 16u]; } else { buf_b[id] = buf_a[id]; }
    workgroupBarrier();
    // Step 6: stride=32
    if (id >= 32u) { buf_a[id] = buf_b[id] + buf_b[id - 32u]; } else { buf_a[id] = buf_b[id]; }
    workgroupBarrier();
    // Step 7: stride=64
    if (id >= 64u) { buf_b[id] = buf_a[id] + buf_a[id - 64u]; } else { buf_b[id] = buf_a[id]; }
    workgroupBarrier();
    // Step 8: stride=128
    if (id >= 128u) { buf_a[id] = buf_b[id] + buf_b[id - 128u]; } else { buf_a[id] = buf_b[id]; }
    workgroupBarrier();

    // buf_a now has inclusive prefix sum
    if (id < nb) {
        let inclusive = buf_a[id];
        let total_od = buf_a[nb - 1u];

        if (total_od > 0.0) {
            cdf[id] = inclusive / total_od;
        } else {
            cdf[id] = f32(id + 1u) / f32(nb);
        }

        if (id == 0u) {
            cdf[nb] = total_od;
        }

        // Clear histogram for next frame
        atomicStore(&histogram[id], 0u);
    }
}
