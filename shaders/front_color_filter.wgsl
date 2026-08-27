@group(0) @binding(0) var fallback_feature: texture_2d<f32>;
@group(0) @binding(1) var filtered_color_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var fallback_color: texture_2d<f32>;

const WORKGROUP_WIDTH: u32 = 8u;
const TILE_WIDTH: u32 = WORKGROUP_WIDTH + 2u;
var<workgroup> feature_tile: array<vec4<f32>, 100>;
var<workgroup> color_tile: array<vec4<f32>, 100>;

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let size = vec2i(textureDimensions(fallback_feature));
    let tile_origin = vec2i(workgroup_id.xy * WORKGROUP_WIDTH) - vec2i(1);
    for (var tile_index = local_index; tile_index < TILE_WIDTH * TILE_WIDTH; tile_index += 64u) {
        let tile_position = vec2i(vec2u(tile_index % TILE_WIDTH, tile_index / TILE_WIDTH));
        let sample_pixel = clamp(tile_origin + tile_position, vec2i(0), size - vec2i(1));
        feature_tile[tile_index] = textureLoad(fallback_feature, sample_pixel, 0);
        color_tile[tile_index] = textureLoad(fallback_color, sample_pixel, 0);
    }
    workgroupBarrier();

    let pixel = vec2i(id.xy);
    if (any(pixel >= size)) {
        return;
    }

    let center_index = (local_id.y + 1u) * TILE_WIDTH + local_id.x + 1u;

    var color_sum = vec3f(0.0);
    var color_min = vec3f(1.0);
    var color_max = vec3f(0.0);
    var sample_count = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let sample_index = u32(i32(center_index) + y * i32(TILE_WIDTH) + x);
            let feature = feature_tile[sample_index];
            if (feature.w >= 1.0) {
                let color = color_tile[sample_index].rgb;
                color_sum += color;
                color_min = min(color_min, color);
                color_max = max(color_max, color);
                sample_count += 1.0;
            }
        }
    }

    if (sample_count < 1.0) {
        textureStore(filtered_color_out, pixel, vec4f(0.0));
        return;
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
