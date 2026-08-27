struct HistoParams {
    tile_count_x: u32,
    tile_count_y: u32,
    num_bins: u32,
    tile_size: u32,
};

const PRIOR_WIDTH: u32 = 8u;
const PRIOR_HEIGHT: u32 = 8u;
const DEPTH_BINS: u32 = 64u;

@group(0) @binding(1) var cdf_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var<uniform> histo_params: HistoParams;
@group(0) @binding(3) var<storage, read> spatial_prior: array<f32>;

var<workgroup> buf_a: array<f32, 64>;
var<workgroup> buf_b: array<f32, 64>;

fn prior_value(x: u32, y: u32, bin: u32) -> f32 {
    return spatial_prior[(y * PRIOR_WIDTH + x) * DEPTH_BINS + bin];
}

fn prior_chroma(x: u32, y: u32) -> vec2<f32> {
    let histogram_len = PRIOR_WIDTH * PRIOR_HEIGHT * DEPTH_BINS;
    let base = histogram_len + (y * PRIOR_WIDTH + x) * 2u;
    return vec2f(spatial_prior[base], spatial_prior[base + 1u]);
}

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tile_x = wg.x;
    let tile_y = wg.y;
    let bin = lid.x;

    // Map every live 32px tile into the baked 8x8 grid and bilinearly interpolate it.
    let sx = clamp(
        (f32(tile_x) + 0.5) / f32(histo_params.tile_count_x) * f32(PRIOR_WIDTH) - 0.5,
        0.0,
        f32(PRIOR_WIDTH - 1u),
    );
    let sy = clamp(
        (f32(tile_y) + 0.5) / f32(histo_params.tile_count_y) * f32(PRIOR_HEIGHT) - 0.5,
        0.0,
        f32(PRIOR_HEIGHT - 1u),
    );
    let x0 = u32(floor(sx));
    let y0 = u32(floor(sy));
    let x1 = min(x0 + 1u, PRIOR_WIDTH - 1u);
    let y1 = min(y0 + 1u, PRIOR_HEIGHT - 1u);
    let fx = fract(sx);
    let fy = fract(sy);

    let top = mix(prior_value(x0, y0, bin), prior_value(x1, y0, bin), fx);
    let bottom = mix(prior_value(x0, y1, bin), prior_value(x1, y1, bin), fx);
    let bin_value = mix(top, bottom, fy);
    let chroma_top = mix(prior_chroma(x0, y0), prior_chroma(x1, y0), fx);
    let chroma_bottom = mix(prior_chroma(x0, y1), prior_chroma(x1, y1), fx);
    let chroma = mix(chroma_top, chroma_bottom, fy);
    buf_a[bin] = bin_value;
    workgroupBarrier();

    if (bin >= 1u) { buf_b[bin] = buf_a[bin] + buf_a[bin - 1u]; } else { buf_b[bin] = buf_a[bin]; }
    workgroupBarrier();
    if (bin >= 2u) { buf_a[bin] = buf_b[bin] + buf_b[bin - 2u]; } else { buf_a[bin] = buf_b[bin]; }
    workgroupBarrier();
    if (bin >= 4u) { buf_b[bin] = buf_a[bin] + buf_a[bin - 4u]; } else { buf_b[bin] = buf_a[bin]; }
    workgroupBarrier();
    if (bin >= 8u) { buf_a[bin] = buf_b[bin] + buf_b[bin - 8u]; } else { buf_a[bin] = buf_b[bin]; }
    workgroupBarrier();
    if (bin >= 16u) { buf_b[bin] = buf_a[bin] + buf_a[bin - 16u]; } else { buf_b[bin] = buf_a[bin]; }
    workgroupBarrier();
    if (bin >= 32u) { buf_a[bin] = buf_b[bin] + buf_b[bin - 32u]; } else { buf_a[bin] = buf_b[bin]; }
    workgroupBarrier();

    let total = buf_a[DEPTH_BINS - 1u];
    var cdf = f32(bin) / f32(DEPTH_BINS);
    if (total > 0.0) {
        cdf = (buf_a[bin] - bin_value) / total;
    }
    textureStore(
        cdf_out,
        vec3i(i32(tile_x), i32(tile_y), i32(bin)),
        vec4f(
            cdf,
            select(0.0, bin_value / total, total > 0.0),
            chroma * 0.5 + vec2f(0.5),
        ),
    );
}
