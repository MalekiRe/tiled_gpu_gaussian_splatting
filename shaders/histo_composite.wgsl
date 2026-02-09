struct HistoParams {
    tile_count_x: u32,
    tile_count_y: u32,
    num_bins: u32,
    depth_range: f32,
};

@group(0) @binding(0) var accum_tex: texture_2d<f32>;
@group(0) @binding(1) var revealage_tex: texture_2d<f32>;

@group(1) @binding(0) var<storage, read_write> histogram: array<atomic<u32>>;
@group(1) @binding(1) var<storage, read_write> cdf: array<f32>;
@group(1) @binding(2) var<uniform> histo_params: HistoParams;

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

    // 1. WBOIT composite (same as naive)
    let accum = textureLoad(accum_tex, coords, 0);
    let revealage = textureLoad(revealage_tex, coords, 0).r;
    let avg_color = accum.rgb / max(accum.a, 1e-5);
    let alpha = 1.0 - revealage;

    // Only pixel (0,0) builds the global CDF
    let should_build_cdf = (u32(in.position.x) == 0u && u32(in.position.y) == 0u);

    if (should_build_cdf) {
	    // 2. Build CDF from histogram, then clear histogram
	    // Always use tile_index = 0 for global histogram
	    let tile_index = 0u;
	    let base = 0u;  // tile_index * num_bins = 0

	    // Compute total (atomicLoad is non-destructive, all fragments in tile read same values)
	    var total = 0u;
	    for (var b = 0u; b < histo_params.num_bins; b = b + 1u) {
	        total = total + atomicLoad(&histogram[base + b]);
	    }

	    // Build CDF using atomicLoad (not atomicExchange — all fragments compute identical CDF)
	    var running = 0u;
	    for (var b = 0u; b < histo_params.num_bins; b = b + 1u) {
	        running = running + atomicLoad(&histogram[base + b]);
	        if total > 0u {
	            cdf[base + b] = f32(running) / f32(total);
	        } else {
	            cdf[base + b] = f32(b) / f32(histo_params.num_bins);
	        }
	    }

	    // Clear histogram for next frame (all fragments write 0 — benign race)
	    for (var b = 0u; b < histo_params.num_bins; b = b + 1u) {
	        atomicStore(&histogram[base + b], 0u);
	    }
    }

    return vec4<f32>(avg_color * alpha, alpha);
}
