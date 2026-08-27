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
    var optical_quantile = textureSampleLevel(
        cdf_texture,
        cdf_sampler,
        vec3f(u, v, clamp(normalized_z + 0.5 / f32(histo_params.num_bins), 0.0, 1.0)),
        0.0,
    ).r;
    let pixel = vec2<i32>(in.clip_position.xy);
    let primary_feature = textureLoad(front_feature, pixel, 0);
    let fallback_feature = textureLoad(front_feature_fallback, pixel, 0);
    let feature = select(fallback_feature, primary_feature, primary_feature.w >= 1.0);
    if (feature.w >= 1.0) {
        let radius_z = max(params.scene_radius / camera.depth_range, 1e-5);
        let depth_delta = normalized_z - feature.z;
        let behind = smoothstep(0.0, 0.03 * radius_z, max(depth_delta, 0.0));
        let depth_gate = exp(-pow(max(depth_delta, 0.0) / (0.08 * radius_z), 2.0));
        let normal_similarity = clamp(dot(in.normal, decode_octahedral(feature.xy)), -1.0, 1.0);
        let normal_gate = exp(-32.0 * (1.0 - normal_similarity));
        let fragment_luminance = clamp(dot(in.color, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
        let luminance_gate = exp(-2.0 * abs(fragment_luminance - (feature.w - 1.0)));
        let appearance_agreement = normal_gate * luminance_gate;
        let front_band = 1.0 - smoothstep(
            0.0,
            0.04 * radius_z,
            abs(depth_delta),
        );
        // Anchor the stable Gaussian body, but let its faint support retain the
        // continuous tent basis so slice changes cannot turn into sparkling edges.
        let core_confidence = smoothstep(0.04, 0.18, alpha);
        let raw_front_anchor = front_band * appearance_agreement * core_confidence;
        let front_anchor = raw_front_anchor * raw_front_anchor * raw_front_anchor;
        optical_quantile *= 1.0 - front_anchor;
        let disagreement = behind * (1.0 - depth_gate * appearance_agreement);
        optical_quantile = mix(optical_quantile, 1.0, disagreement);
    }
    // A tent basis over four ordered quantile representatives avoids hard layer
    // transitions without adding another pass or attachment.
    // Front-loaded representatives: an error near the eye modulates every layer
    // behind it, so spend more of the fixed four-layer budget there.
    let slice_position = clamp(pow(optical_quantile, 0.65) * 3.0, 0.0, 3.0);
    // Keep the visually solid Gaussian core in one ordered representative. Only
    // the translucent fringe uses the tent basis. This avoids manufacturing two
    // semi-transparent copies of a surface-defining sample.
    let hard_assignment = smoothstep(0.10, 0.24, alpha);
    let assigned_position = mix(slice_position, round(slice_position), hard_assignment);
    let lower_slice = u32(floor(assigned_position));
    let upper_slice = min(lower_slice + 1u, 3u);
    let upper_weight = fract(assigned_position);
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
