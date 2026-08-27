@group(0) @binding(0) var fallback_feature: texture_2d<f32>;
@group(0) @binding(1) var filtered_color_out: texture_storage_2d<rgba16float, write>;

fn decode_rgb3(encoded: f32) -> vec3<f32> {
    let code = round(max(encoded - 1.0, 0.0) * 512.0);
    return vec3f(code % 8.0, floor(code / 8.0) % 8.0, floor(code / 64.0)) / 7.0;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2i(textureDimensions(fallback_feature));
    let pixel = vec2i(id.xy);
    if (any(pixel >= size)) {
        return;
    }

    let center = textureLoad(fallback_feature, pixel, 0);
    if (center.w < 1.0) {
        textureStore(filtered_color_out, pixel, vec4f(0.0));
        return;
    }

    var color_sum = vec3f(0.0);
    var color_min = vec3f(1.0);
    var color_max = vec3f(0.0);
    var sample_count = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let sample_pixel = clamp(pixel + vec2i(x, y), vec2i(0), size - vec2i(1));
            let feature = textureLoad(fallback_feature, sample_pixel, 0);
            if (feature.w >= 1.0) {
                let color = decode_rgb3(feature.w);
                color_sum += color;
                color_min = min(color_min, color);
                color_max = max(color_max, color);
                sample_count += 1.0;
            }
        }
    }

    let raw_mean = color_sum / sample_count;
    var filtered_color = raw_mean;
    if (sample_count > 2.0) {
        let trimmed_mean = (color_sum - color_min - color_max) / (sample_count - 2.0);
        filtered_color = clamp(
            mix(raw_mean, trimmed_mean, 1.55),
            vec3f(0.0),
            vec3f(1.0),
        );
    }
    textureStore(filtered_color_out, pixel, vec4f(filtered_color, sample_count));
}
