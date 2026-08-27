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

fn decode_primary_payload(encoded: f32) -> vec2<f32> {
    let code = round(max(encoded - 1.0, 0.0) * 1024.0);
    let sigma_code = code % 16.0;
    let sigma_ratio = select(
        0.0,
        exp2(-10.0 + (sigma_code - 1.0) * (9.0 / 14.0)),
        sigma_code > 0.0,
    );
    return vec2<f32>(floor(code / 16.0) / 63.0, sigma_ratio);
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
    let cdf_z = normalized_z + 0.5 / f32(histo_params.num_bins);
    let cdf_center = textureSampleLevel(
        cdf_texture,
        cdf_sampler,
        vec3f(u, v, clamp(cdf_z, 0.0, 1.0)),
        0.0,
    );
    // Use the Gaussian's eye-depth extent to estimate local PDF curvature. A
    // mild unsharp correction sharpens peak confidence without perturbing the
    // monotone CDF quantile itself.
    let cdf_extent = 1.7320508 * in.depth_sigma;
    let cdf_front = textureSampleLevel(
        cdf_texture,
        cdf_sampler,
        vec3f(u, v, clamp(cdf_z - cdf_extent, 0.0, 1.0)),
        0.0,
    );
    let cdf_back = textureSampleLevel(
        cdf_texture,
        cdf_sampler,
        vec3f(u, v, clamp(cdf_z + cdf_extent, 0.0, 1.0)),
        0.0,
    );
    let cdf_far_front = textureSampleLevel(
        cdf_texture,
        cdf_sampler,
        vec3f(u, v, clamp(cdf_z - 2.0 * cdf_extent, 0.0, 1.0)),
        0.0,
    );
    let cdf_far_back = textureSampleLevel(
        cdf_texture,
        cdf_sampler,
        vec3f(u, v, clamp(cdf_z + 2.0 * cdf_extent, 0.0, 1.0)),
        0.0,
    );
    let pdf_curvature = (
        -cdf_far_front.g + 16.0 * cdf_front.g - 30.0 * cdf_center.g
            + 16.0 * cdf_back.g - cdf_far_back.g
    ) / 12.0;
    let cdf_sample = vec4<f32>(
        cdf_center.r,
        clamp(cdf_center.g - pdf_curvature, 0.0, 1.0),
        cdf_center.ba,
    );
    var optical_quantile = cdf_sample.r;
    let pixel = vec2<i32>(in.clip_position.xy);
    let primary_feature = textureLoad(front_feature, pixel, 0);
    let fallback_feature = textureLoad(front_feature_fallback, pixel, 0);
    let primary_valid = primary_feature.w >= 1.0;
    let fallback_valid = fallback_feature.w >= 1.0;
    let primary_payload = decode_primary_payload(primary_feature.w);
    let fallback_color = decode_rgb3(fallback_feature.w);
    let fallback_luminance = dot(fallback_color, vec3<f32>(0.2126, 0.7152, 0.0722));
    let feature = select(fallback_feature, primary_feature, primary_valid);
    if (feature.w >= 1.0) {
        let radius_z = max(params.scene_radius / camera.depth_range, 1e-5);
        var observation_confidence = select(0.5, 1.0, primary_valid);
        var coherent_layer_evidence = 0.0;
        var front_depth = feature.z;
        var front_luminance = select(fallback_luminance, primary_payload.x, primary_valid);
        var front_color = fallback_color;
        if (primary_valid && !fallback_valid) {
            // Recombine the primary's precise luminance with the coarse baked
            // Co/Cg tile moment carried in the CDF's otherwise-unused channels.
            let baked_chroma = cdf_center.ba * 2.0 - vec2f(1.0);
            front_color = clamp(
                vec3f(
                    front_luminance + 0.4298 * baked_chroma.x - 0.7152 * baked_chroma.y,
                    front_luminance - 0.0702 * baked_chroma.x + 0.2848 * baked_chroma.y,
                    front_luminance - 0.5702 * baked_chroma.x - 0.7152 * baked_chroma.y,
                ),
                vec3f(0.0),
                vec3f(1.0),
            );
        }
        let front_depth_sigma = select(0.0, primary_payload.y * radius_z, primary_valid);
        var front_normal = decode_octahedral(feature.xy);
        if (primary_valid && fallback_valid) {
            let observation_depth_delta = abs(primary_feature.z - fallback_feature.z);
            let depth_agreement = exp(-pow(observation_depth_delta / (0.12 * radius_z), 2.0));
            let primary_normal = decode_octahedral(primary_feature.xy);
            let fallback_normal = decode_octahedral(fallback_feature.xy);
            let raw_observation_normal_agreement = clamp(
                0.5 + 0.5 * dot(primary_normal, fallback_normal),
                0.0,
                1.0,
            );
            // Covariance-derived normal signs are only a heuristic, so even
            // perfectly matching stochastic observations must not dominate.
            let observation_normal_agreement = min(raw_observation_normal_agreement, 0.64);
            let observation_luminance_agreement = exp(
                -2.0 * abs(primary_payload.x - fallback_luminance),
            );
            coherent_layer_evidence = (1.0 - depth_agreement)
                * raw_observation_normal_agreement
                * observation_luminance_agreement;
            let observation_agreement = depth_agreement
                * observation_normal_agreement
                * observation_luminance_agreement;
            observation_confidence = 0.75 + 0.25 * observation_agreement;
            let consensus_blend = 0.25 * observation_agreement;
            front_depth = mix(primary_feature.z, fallback_feature.z, consensus_blend);
            front_luminance = mix(
                primary_payload.x,
                fallback_luminance,
                consensus_blend,
            );
            // The fallback's packed RGB is deliberately noisy.  Average only
            // color from strongly coplanar neighbors; depth, normal, and all
            // confidence decisions continue to use the unbiased center sample.
            var reconstruction_color = 4.0 * fallback_color;
            var reconstruction_weight = 4.0;
            let fallback_size = vec2<i32>(textureDimensions(front_feature_fallback));
            let neighbor_offsets = array<vec2<i32>, 4>(
                vec2<i32>(-1, 0), vec2<i32>(1, 0),
                vec2<i32>(0, -1), vec2<i32>(0, 1),
            );
            for (var neighbor_index = 0u; neighbor_index < 4u; neighbor_index++) {
                let neighbor_pixel = clamp(
                    pixel + neighbor_offsets[neighbor_index],
                    vec2<i32>(0),
                    fallback_size - vec2<i32>(1),
                );
                let neighbor = textureLoad(front_feature_fallback, neighbor_pixel, 0);
                if (neighbor.w >= 1.0) {
                    let neighbor_depth_gate = exp(-pow(
                        abs(neighbor.z - fallback_feature.z) / (0.02 * radius_z),
                        2.0,
                    ));
                    let neighbor_normal_gate = smoothstep(
                        0.75,
                        0.95,
                        dot(fallback_normal, decode_octahedral(neighbor.xy)),
                    );
                    let neighbor_weight = neighbor_depth_gate * neighbor_normal_gate;
                    reconstruction_color += neighbor_weight * decode_rgb3(neighbor.w);
                    reconstruction_weight += neighbor_weight;
                }
            }
            reconstruction_color /= reconstruction_weight;
            let reconstruction_luminance = dot(
                reconstruction_color,
                vec3<f32>(0.2126, 0.7152, 0.0722),
            );
            let missing_luminance = max(front_luminance - reconstruction_luminance, 0.0);
            let additive_front_color = clamp(
                reconstruction_color + vec3f(missing_luminance),
                vec3f(0.0),
                vec3f(1.0),
            );
            let multiplicative_scale = clamp(
                front_luminance / max(reconstruction_luminance, 1.0 / 63.0),
                1.0,
                4.0,
            );
            let multiplicative_front_color = clamp(
                reconstruction_color * multiplicative_scale,
                vec3f(0.0),
                vec3f(1.0),
            );
            let packed_min = min(
                reconstruction_color.r,
                min(reconstruction_color.g, reconstruction_color.b),
            );
            let packed_max = max(
                reconstruction_color.r,
                max(reconstruction_color.g, reconstruction_color.b),
            );
            let additive_weight = mix(0.95, 0.65, packed_max - packed_min);
            front_color = mix(
                multiplicative_front_color,
                additive_front_color,
                additive_weight,
            );
            front_normal = normalize(mix(primary_normal, fallback_normal, consensus_blend));
        }
        let depth_delta = normalized_z - front_depth;
        let combined_depth_sigma = sqrt(
            in.depth_sigma * in.depth_sigma + front_depth_sigma * front_depth_sigma,
        );
        let local_thickness = max(0.03 * radius_z, 2.0 * combined_depth_sigma);
        let local_softness = max(0.08 * radius_z, 4.0 * combined_depth_sigma);
        let view_normal = (camera.view * vec4<f32>(in.normal, 0.0)).xyz;
        let back_facing = smoothstep(0.0, 0.0625, -view_normal.z);
        let core_confidence = smoothstep(0.04, 0.18, alpha);
        // Solid, corroborated samples can tolerate a much narrower back-layer
        // boundary. Preserve a wider band for faint/noisy support, where an
        // overconfident split turns directly into pinholes.
        let peak_support = smoothstep(0.04, 0.20, cdf_sample.g);
        let orientation_strength = back_facing * max(
            0.0,
            mix(0.05, 0.50, core_confidence * observation_confidence)
                - 2.0 * core_confidence * coherent_layer_evidence,
        ) * mix(1.0, 1.25, core_confidence * peak_support);
        let oriented_thickness = mix(
            local_thickness,
            max(0.015 * radius_z, combined_depth_sigma),
            orientation_strength,
        );
        let behind = smoothstep(0.0, oriented_thickness, max(depth_delta, 0.0));
        let depth_gate = exp(-pow(max(depth_delta, 0.0) / local_softness, 2.0));
        let normal_similarity = clamp(dot(in.normal, front_normal), -1.0, 1.0);
        let normal_gate = exp(-32.0 * (1.0 - normal_similarity));
        let fragment_luminance = clamp(dot(in.color, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
        let luminance_gate = exp(-2.0 * abs(fragment_luminance - front_luminance));
        let raw_color_gate = exp(
            -4.0 * dot(in.color - front_color, in.color - front_color),
        );
        let color_gate_strength = select(
            0.75,
            0.5 * observation_confidence,
            fallback_valid,
        );
        let color_gate = mix(1.0, raw_color_gate, color_gate_strength);
        let appearance_agreement = normal_gate * luminance_gate * color_gate;
        let front_band = 1.0 - smoothstep(
            0.0,
            0.04 * radius_z,
            abs(depth_delta),
        );
        // Anchor the stable Gaussian body, but let its faint support retain the
        // continuous tent basis so slice changes cannot turn into sparkling edges.
        let raw_front_anchor = front_band * appearance_agreement * core_confidence
            * observation_confidence;
        let front_anchor = raw_front_anchor * raw_front_anchor * raw_front_anchor;
        optical_quantile *= 1.0 - front_anchor;
        let surface_disagreement = max(
            1.0 - depth_gate * appearance_agreement,
            back_facing,
        );
        let disagreement = behind * surface_disagreement
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
    let assignment_fraction = fract(assigned_position);
    let cubic_weight = smoothstep(0.0, 1.0, assignment_fraction);
    let basis_confidence = spatial_stability * depth_certainty;
    let upper_weight = mix(assignment_fraction, cubic_weight, basis_confidence);
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
