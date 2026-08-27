// splat_common.wgsl is prepended. Four fixed eye-depth slabs accumulate optical depth
// independently; no sorting, atomics, stochastic sampling, or temporal history.

struct SliceOutput {
    @location(0) slice0: vec4<f32>,
    @location(1) slice1: vec4<f32>,
    @location(2) slice2: vec4<f32>,
    @location(3) slice3: vec4<f32>,
};

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
@group(2) @binding(5) var front_feature: texture_2d<f32>;
@group(2) @binding(6) var front_feature_fallback: texture_2d<f32>;

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
fn fs_main(in: SplatVsOut) -> SliceOutput {
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
    let u = in.clip_position.x / f32(histo_params.tile_count_x * TILE_SIZE);
    let v = in.clip_position.y / f32(histo_params.tile_count_y * TILE_SIZE);
    let cdf_sample = textureSampleLevel(
        cdf_texture,
        cdf_sampler,
        vec3f(u, v, clamp(normalized_z + 0.5 / f32(histo_params.num_bins), 0.0, 1.0)),
        0.0,
    );
    var optical_quantile = cdf_sample.r;
    let pixel = vec2<i32>(in.clip_position.xy);
    let primary_feature = textureLoad(front_feature, pixel, 0);
    let fallback_feature = textureLoad(front_feature_fallback, pixel, 0);
    let primary_valid = primary_feature.w >= 1.0;
    let fallback_valid = fallback_feature.w >= 1.0;
    let fallback_color = decode_rgb3(fallback_feature.w);
    let fallback_luminance = dot(fallback_color, vec3<f32>(0.2126, 0.7152, 0.0722));
    let feature = select(fallback_feature, primary_feature, primary_valid);
    if (feature.w >= 1.0) {
        let radius_z = max(params.scene_radius / camera.depth_range, 1e-5);
        var observation_confidence = select(0.5, 1.0, primary_valid);
        var front_depth = feature.z;
        var front_luminance = select(fallback_luminance, primary_feature.w - 1.0, primary_valid);
        var front_normal = decode_octahedral(feature.xy);
        if (primary_valid && fallback_valid) {
            let observation_depth_delta = abs(primary_feature.z - fallback_feature.z);
            let depth_agreement = exp(-pow(observation_depth_delta / (0.12 * radius_z), 2.0));
            let primary_normal = decode_octahedral(primary_feature.xy);
            let fallback_normal = decode_octahedral(fallback_feature.xy);
            let observation_normal_agreement = clamp(
                0.5 + 0.5 * dot(primary_normal, fallback_normal),
                0.0,
                1.0,
            );
            let observation_luminance_agreement = exp(
                -2.0 * abs((primary_feature.w - 1.0) - fallback_luminance),
            );
            let observation_agreement = depth_agreement
                * observation_normal_agreement
                * observation_luminance_agreement;
            observation_confidence = 0.75 + 0.25 * observation_agreement;
            let consensus_blend = 0.25 * observation_agreement;
            front_depth = mix(primary_feature.z, fallback_feature.z, consensus_blend);
            front_luminance = mix(
                primary_feature.w - 1.0,
                fallback_luminance,
                consensus_blend,
            );
            front_normal = normalize(mix(primary_normal, fallback_normal, consensus_blend));
        }
        let depth_delta = normalized_z - front_depth;
        let behind = smoothstep(0.0, 0.03 * radius_z, max(depth_delta, 0.0));
        let depth_gate = exp(-pow(max(depth_delta, 0.0) / (0.08 * radius_z), 2.0));
        let normal_similarity = clamp(dot(in.normal, front_normal), -1.0, 1.0);
        let normal_gate = exp(-32.0 * (1.0 - normal_similarity));
        let fragment_luminance = clamp(dot(in.color, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
        let luminance_gate = exp(-2.0 * abs(fragment_luminance - front_luminance));
        let raw_color_gate = exp(-4.0 * dot(in.color - fallback_color, in.color - fallback_color));
        let color_gate = select(
            1.0,
            mix(1.0, raw_color_gate, 0.5 * observation_confidence),
            fallback_valid,
        );
        let appearance_agreement = normal_gate * luminance_gate * color_gate;
        let front_band = 1.0 - smoothstep(
            0.0,
            0.04 * radius_z,
            abs(depth_delta),
        );
        // Anchor the stable Gaussian body, but let its faint support retain the
        // continuous tent basis so slice changes cannot turn into sparkling edges.
        let core_confidence = smoothstep(0.04, 0.18, alpha);
        let raw_front_anchor = front_band * appearance_agreement * core_confidence
            * observation_confidence;
        let front_anchor = raw_front_anchor * raw_front_anchor * raw_front_anchor;
        optical_quantile *= 1.0 - front_anchor;
        let disagreement = behind * (1.0 - depth_gate * appearance_agreement)
            * observation_confidence;
        optical_quantile = mix(optical_quantile, 1.0, disagreement);
    }
    // A tent basis over four ordered quantile representatives avoids hard layer
    // transitions without adding another pass or attachment.
    // Front-loaded representatives: an error near the eye modulates every layer
    // behind it, so spend more of the fixed four-layer budget there.
    let slice_position = clamp(optical_quantile * 3.0, 0.0, 3.0);
    // Keep the visually solid Gaussian core in one ordered representative. Only
    // the translucent fringe uses the tent basis. This avoids manufacturing two
    // semi-transparent copies of a surface-defining sample.
    let assignment_gradient = fwidth(slice_position);
    let spatial_stability = 1.0 - smoothstep(0.10, 0.60, assignment_gradient);
    let depth_certainty = 1.0 - smoothstep(0.04, 0.20, cdf_sample.g);
    let hard_assignment = smoothstep(0.10, 0.24, alpha)
        * spatial_stability
        * depth_certainty;
    let assigned_position = mix(slice_position, round(slice_position), hard_assignment);
    let lower_slice = u32(floor(assigned_position));
    let upper_slice = min(lower_slice + 1u, 3u);
    let upper_weight = smoothstep(0.0, 1.0, fract(assigned_position));
    let optical_depth = -log(max(1.0 - alpha, 1e-5));
    let contribution = vec4<f32>(in.color * optical_depth, optical_depth);

    var out: SliceOutput;
    out.slice0 = vec4<f32>(0.0);
    out.slice1 = vec4<f32>(0.0);
    out.slice2 = vec4<f32>(0.0);
    out.slice3 = vec4<f32>(0.0);
    switch lower_slice {
        case 0u: { out.slice0 += contribution * (1.0 - upper_weight); }
        case 1u: { out.slice1 += contribution * (1.0 - upper_weight); }
        case 2u: { out.slice2 += contribution * (1.0 - upper_weight); }
        default: { out.slice3 += contribution * (1.0 - upper_weight); }
    }
    switch upper_slice {
        case 0u: { out.slice0 += contribution * upper_weight; }
        case 1u: { out.slice1 += contribution * upper_weight; }
        case 2u: { out.slice2 += contribution * upper_weight; }
        default: { out.slice3 += contribution * upper_weight; }
    }
    return out;
}
