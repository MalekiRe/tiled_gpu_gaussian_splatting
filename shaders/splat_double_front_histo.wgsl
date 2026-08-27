// splat_common.wgsl is prepended.
// Mode 7: mode 6 with two independent stochastic front estimates. Agreement raises
// confidence; disagreement conservatively preserves deeper contributions.

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
@group(2) @binding(5) var front_feature_a: texture_2d<f32>;
@group(2) @binding(6) var front_feature_b: texture_2d<f32>;

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
    let a = textureLoad(front_feature_a, pixel, 0);
    let b = textureLoad(front_feature_b, pixel, 0);
    let valid_a = clamp(length(a.xyz), 0.0, 1.0);
    let valid_b = clamp(length(b.xyz), 0.0, 1.0);
    let both_valid = valid_a * valid_b;
    let either_valid = max(valid_a, valid_b);

    let radius_z = max(params.scene_radius / (camera.far - camera.near), 1e-5);
    let depth_difference = abs(a.w - b.w);
    let depth_agreement = exp(-pow(depth_difference / max(0.12 * radius_z, 1e-5), 2.0));
    let normal_a = a.xyz / max(length(a.xyz), 1e-5);
    let normal_b = b.xyz / max(length(b.xyz), 1e-5);
    let normal_agreement = clamp(0.5 + 0.5 * dot(normal_a, normal_b), 0.0, 1.0);
    let agreement = depth_agreement * normal_agreement;

    let combined_normal = normalize(a.xyz + b.xyz + vec3<f32>(1e-8, 0.0, 0.0));
    let single_normal = select(normal_b, normal_a, valid_a >= valid_b);
    let front_normal = select(single_normal, combined_normal, both_valid > 0.0);
    let average_depth = 0.5 * (a.w + b.w);
    let single_depth = select(b.w, a.w, valid_a >= valid_b);
    let front_depth = select(single_depth, average_depth, both_valid > 0.0);
    let confidence = select(0.25 * either_valid, both_valid * agreement, both_valid > 0.0);

    let depth_delta = normalized_z - front_depth;
    let thickness = 0.035 * radius_z;
    let softness = max(0.20 * radius_z, 1e-5);
    let behind = smoothstep(0.0, thickness * 2.0, max(depth_delta, 0.0));
    let excess_depth = max(depth_delta - thickness, 0.0);
    let depth_gate = exp(-pow(excess_depth / softness, 2.0));
    let similarity = clamp(dot(in.normal, front_normal), -1.0, 1.0);
    let direction_gate = exp(-1.5 * (1.0 - similarity));
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
