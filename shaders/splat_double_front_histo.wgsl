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

fn decode_octahedral(encoded: vec2<f32>) -> vec3<f32> {
    let projected = encoded * 2.0 - 1.0;
    var normal = vec3<f32>(projected, 1.0 - abs(projected.x) - abs(projected.y));
    let correction = clamp(-normal.z, 0.0, 1.0);
    normal.x += select(-correction, correction, normal.x < 0.0);
    normal.y += select(-correction, correction, normal.y < 0.0);
    return normalize(normal);
}

fn decode_rgb3(encoded: f32) -> vec3<f32> {
    let code = round(max(encoded - 1.0, 0.0) * 512.0);
    return vec3<f32>(code % 8.0, floor(code / 8.0) % 8.0, floor(code / 64.0)) / 7.0;
}

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
        (linear_z - camera.depth_min) / camera.depth_range,
        0.0,
        1.0,
    );
    let pixel = vec2<i32>(in.clip_position.xy);
    let a = textureLoad(front_feature_a, pixel, 0);
    let b = textureLoad(front_feature_b, pixel, 0);
    let valid_a = select(0.0, 1.0, a.w >= 1.0);
    let valid_b = select(0.0, 1.0, b.w >= 1.0);
    let both_valid = valid_a * valid_b;
    let either_valid = max(valid_a, valid_b);

    let radius_z = max(params.scene_radius / camera.depth_range, 1e-5);
    let depth_difference = abs(a.z - b.z);
    let depth_agreement = exp(-pow(depth_difference / max(0.12 * radius_z, 1e-5), 2.0));
    let normal_a = decode_octahedral(a.xy);
    let normal_b = decode_octahedral(b.xy);
    let normal_agreement = clamp(0.5 + 0.5 * dot(normal_a, normal_b), 0.0, 1.0);
    let agreement = depth_agreement * normal_agreement;

    // When the two stochastic realizations hit different layers, averaging invents a
    // surface between them. Keep the nearer actual sample instead; disagreement already
    // lowers confidence below, so this remains conservative at noisy edges.
    // Prefer the deterministic core whenever it exists; the stochastic fringe is a
    // coverage fallback, not an equally noisy vote that can displace a stable surface.
    let choose_a = valid_a > 0.0;
    let front_normal = select(normal_b, normal_a, choose_a);
    let front_depth = select(b.z, a.z, choose_a);
    let fallback_luminance = dot(decode_rgb3(b.w), vec3<f32>(0.2126, 0.7152, 0.0722));
    let front_luminance = select(fallback_luminance, a.w - 1.0, choose_a);
    let confidence = select(
        0.5 * either_valid,
        both_valid * (0.5 + 0.5 * agreement),
        both_valid > 0.0,
    );

    let depth_delta = normalized_z - front_depth;
    let thickness = 0.015 * radius_z;
    let softness = max(0.05 * radius_z, 1e-5);
    let behind = smoothstep(0.0, thickness * 2.0, max(depth_delta, 0.0));
    let excess_depth = max(depth_delta - thickness, 0.0);
    let depth_gate = exp(-pow(excess_depth / softness, 2.0));
    let similarity = clamp(dot(in.normal, front_normal), -1.0, 1.0);
    let direction_gate = exp(-128.0 * (1.0 - similarity));
    let fragment_luminance = clamp(dot(in.color, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
    let luminance_gate = exp(-2.0 * abs(fragment_luminance - front_luminance));
    let back_gate = max(0.0, depth_gate * direction_gate * luminance_gate);
    let gate = mix(1.0, back_gate, behind * confidence);
    let effective_alpha = alpha * gate;

    let u = in.clip_position.x / f32(histo_params.tile_count_x * TILE_SIZE);
    let v = in.clip_position.y / f32(histo_params.tile_count_y * TILE_SIZE);
    let equalized_z = textureSampleLevel(
        cdf_texture,
        cdf_sampler,
        vec3f(u, v, clamp(normalized_z + 0.5 / f32(histo_params.num_bins), 0.0, 1.0)),
        0.0,
    ).r;
    let prev_tau = textureLoad(prev_revealage_tex, pixel, 0).r;
    let wt = exp(-prev_tau * equalized_z);

    var out: WboitOutput;
    out.accum = vec4<f32>(in.color * effective_alpha * wt, effective_alpha * wt);
    // The front feature is a color-ordering heuristic, not geometry. Preserve the exact
    // order-independent opacity product so suppressing a questionable back contribution
    // cannot punch holes through the object or shrink its silhouette.
    out.revealage = -log(max(1.0 - alpha, 1e-6));
    return out;
}
