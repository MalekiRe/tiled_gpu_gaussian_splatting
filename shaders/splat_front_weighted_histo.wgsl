// splat_common.wgsl is prepended.
// Mode 6: mode 5's atomics-free CDF plus a realtime stochastic front-depth/normal prior.

const TILE_SIZE: u32 = 32u;

struct HistoParams {
    tile_count_x: u32,
    tile_count_y: u32,
    num_bins: u32,
    tile_size: u32,
};

@group(2) @binding(1) var cdf_texture: texture_3d<f32>;
@group(2) @binding(2) var cdf_sampler: sampler;
@group(2) @binding(3) var<uniform> histo_params: HistoParams;
@group(2) @binding(4) var prev_revealage_tex: texture_2d<f32>;
@group(2) @binding(5) var front_feature_tex: texture_2d<f32>;

struct WboitOutput {
    @location(0) accum: vec4<f32>,
    @location(1) revealage: f32,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> SplatVsOut {
    return splat_vertex(vertex_index, instance_index);
}

@fragment
fn fs_main(in: SplatVsOut) -> WboitOutput {
    let alpha = splat_alpha(in);
    if (alpha < 0.0) {
        discard;
    }

    let linear_z = 1.0 / in.clip_position.w;
    let normalized_z = clamp(
        (linear_z - camera.near) / (camera.far - camera.near),
        0.0,
        1.0,
    );

    let pixel = vec2<i32>(in.clip_position.xy);
    let front = textureLoad(front_feature_tex, pixel, 0);
    let confidence = clamp(length(front.xyz), 0.0, 1.0);
    let depth_delta = normalized_z - front.w;

    // Only change fragments meaningfully behind the stochastic front. Depth thresholds
    // scale with the object rather than the camera's absolute clip range.
    let radius_z = max(params.scene_radius / (camera.far - camera.near), 1e-5);
    let thickness = 0.035 * radius_z;
    let softness = max(0.20 * radius_z, 1e-5);
    let behind = smoothstep(0.0, thickness * 2.0, max(depth_delta, 0.0));
    let excess_depth = max(depth_delta - thickness, 0.0);
    let depth_gate = exp(-pow(excess_depth / softness, 2.0));

    let front_normal = front.xyz / max(length(front.xyz), 1e-5);
    let similarity = clamp(dot(in.normal, front_normal), -1.0, 1.0);
    let direction_gate = exp(-1.5 * (1.0 - similarity));
    // Keep a residual contribution so a noisy heuristic cannot erase geometry outright.
    let back_gate = max(0.10, depth_gate * direction_gate);
    let gate = mix(1.0, back_gate, behind * confidence);
    let effective_alpha = alpha * gate;
    if (effective_alpha < 1.0 / 255.0) {
        discard;
    }

    let u = in.clip_position.x / f32(histo_params.tile_count_x * TILE_SIZE);
    let v = in.clip_position.y / f32(histo_params.tile_count_y * TILE_SIZE);
    let equalized_z = textureSampleLevel(
        cdf_texture,
        cdf_sampler,
        vec3f(u, v, normalized_z),
        0.0,
    ).r;

    let prev_r = textureLoad(prev_revealage_tex, pixel, 0).r;
    let wt = pow(max(prev_r, 1e-4), equalized_z);

    var out: WboitOutput;
    out.accum = vec4<f32>(in.color * effective_alpha * wt, effective_alpha * wt);
    out.revealage = effective_alpha;
    return out;
}
