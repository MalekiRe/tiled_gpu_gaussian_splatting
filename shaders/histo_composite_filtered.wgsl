@group(0) @binding(0) var accum_tex: texture_2d<f32>;
@group(0) @binding(1) var revealage_tex: texture_2d<f32>;

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

fn composite_at(coords: vec2<i32>) -> vec4<f32> {
    let accum = textureLoad(accum_tex, coords, 0);
    let alpha = 1.0 - exp(-textureLoad(revealage_tex, coords, 0).r);
    return vec4<f32>(accum.rgb / max(accum.a, 1e-5) * alpha, alpha);
}

@fragment
fn fs_main(in: CompositeOutput) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(in.position.xy);
    let size = vec2<i32>(textureDimensions(accum_tex));
    let center = composite_at(pixel);
    let offsets = array<vec2<i32>, 4>(
        vec2<i32>(-1, 0),
        vec2<i32>(1, 0),
        vec2<i32>(0, -1),
        vec2<i32>(0, 1),
    );

    var sum = vec3<f32>(0.0);
    var weight_sum = 0.0;
    var samples = array<vec3<f32>, 4>();
    for (var i = 0u; i < 4u; i++) {
        let coords = clamp(pixel + offsets[i], vec2<i32>(0), size - 1);
        let neighbor = composite_at(coords);
        let weight = 1.0 - smoothstep(0.015, 0.10, abs(neighbor.a - center.a));
        samples[i] = neighbor.rgb;
        sum += neighbor.rgb * weight;
        weight_sum += weight;
    }
    let neighbor_mean = sum / max(weight_sum, 1e-5);
    var dispersion = 0.0;
    for (var i = 0u; i < 4u; i++) {
        dispersion += length(samples[i] - neighbor_mean);
    }
    dispersion *= 0.25;

    // Correct only isolated color outliers. Alpha remains the exact revealage product,
    // and alpha-discontinuous neighbors receive no weight, preserving silhouettes.
    let deviation = length(center.rgb - neighbor_mean);
    let isolated = smoothstep(dispersion + 0.02, dispersion + 0.12, deviation);
    let strength = 0.45 * isolated * clamp(weight_sum * 0.25, 0.0, 1.0);
    return vec4<f32>(mix(center.rgb, neighbor_mean, strength), center.a);
}
