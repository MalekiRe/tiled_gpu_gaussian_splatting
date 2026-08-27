// Shared 3D Gaussian Splatting front-end: EWA projection in the vertex shader, Gaussian
// evaluation in the fragment shader. Each of the three transparency modes prepends this
// file and supplies its own fragment output.
//
// Follows Zwicker et al.'s EWA splatting as used by Kerbl et al. 2023: the 3D covariance is
// pushed through the affine approximation of the projection (Jacobian J and the view
// rotation W) to get a 2D screen-space covariance, whose inverse -- the conic -- gives the
// per-pixel falloff.

struct Camera {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    near: f32,
    far: f32,
    focal: vec2<f32>,
    viewport: vec2<f32>,
    depth_min: f32,
    depth_range: f32,
    cam_pos: vec3<f32>,
};

struct Splat {
    pos_opacity: vec4<f32>,
    cov_a: vec4<f32>,   // xx, xy, xz, heuristic normal x
    cov_b: vec4<f32>,   // yy, yz, zz, heuristic normal y
    color: vec4<f32>,   // rgb = DC band, w = heuristic normal z
};

struct SplatParams {
    count: u32,
    sh_degree: u32,
    splat_scale: f32,
    scene_radius: f32,
};

@group(0) @binding(0) var<uniform> camera: Camera;

@group(1) @binding(0) var<storage, read> splats: array<Splat>;
@group(1) @binding(1) var<storage, read> sh_data: array<f32>;
@group(1) @binding(2) var<storage, read> splat_order: array<u32>;
@group(1) @binding(3) var<uniform> params: SplatParams;

struct SplatVsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    // Inverse 2D covariance (a, b, c of [[a, b], [b, c]]) plus the splat's opacity.
    @location(1) conic_opacity: vec4<f32>,
    // Offset from the Gaussian centre, in the same pixel space the conic lives in.
    @location(2) delta: vec2<f32>,
    @location(3) @interpolate(flat) normal: vec3<f32>,
    @location(4) @interpolate(flat) splat_index: u32,
};

const SH_C1: f32 = 0.4886025119029199;
const SH_C2_0: f32 = 1.0925484305920792;
const SH_C2_1: f32 = -1.0925484305920792;
const SH_C2_2: f32 = 0.31539156525252005;
const SH_C2_3: f32 = -1.0925484305920792;
const SH_C2_4: f32 = 0.5462742152960396;
const SH_C3_0: f32 = -0.5900435899266435;
const SH_C3_1: f32 = 2.890611442640554;
const SH_C3_2: f32 = -0.4570457994644658;
const SH_C3_3: f32 = 0.3731763325901154;
const SH_C3_4: f32 = -0.4570457994644658;
const SH_C3_5: f32 = 1.445305721320277;
const SH_C3_6: f32 = -0.5900435899266435;

// Coefficient k of channel c for splat i. Channel-major, 15 coefficients per channel.
fn sh_coeff(i: u32, c: u32, k: u32) -> f32 {
    return sh_data[i * 45u + c * 15u + k];
}

fn sh_band(i: u32, k: u32) -> vec3<f32> {
    return vec3<f32>(sh_coeff(i, 0u, k), sh_coeff(i, 1u, k), sh_coeff(i, 2u, k));
}

// `dc` is the already-evaluated DC band (0.5 + C0 * f_dc); this adds the view-dependent rest.
fn eval_sh(i: u32, dir: vec3<f32>, dc: vec3<f32>) -> vec3<f32> {
    var c = dc;
    if (params.sh_degree == 0u) {
        return c;
    }

    let x = dir.x;
    let y = dir.y;
    let z = dir.z;

    c += SH_C1 * (-y * sh_band(i, 0u) + z * sh_band(i, 1u) - x * sh_band(i, 2u));

    if (params.sh_degree >= 2u) {
        let xx = x * x; let yy = y * y; let zz = z * z;
        let xy = x * y; let yz = y * z; let xz = x * z;
        c += SH_C2_0 * xy * sh_band(i, 3u)
           + SH_C2_1 * yz * sh_band(i, 4u)
           + SH_C2_2 * (2.0 * zz - xx - yy) * sh_band(i, 5u)
           + SH_C2_3 * xz * sh_band(i, 6u)
           + SH_C2_4 * (xx - yy) * sh_band(i, 7u);

        if (params.sh_degree >= 3u) {
            c += SH_C3_0 * y * (3.0 * xx - yy) * sh_band(i, 8u)
               + SH_C3_1 * xy * z * sh_band(i, 9u)
               + SH_C3_2 * y * (4.0 * zz - xx - yy) * sh_band(i, 10u)
               + SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_band(i, 11u)
               + SH_C3_4 * x * (4.0 * zz - xx - yy) * sh_band(i, 12u)
               + SH_C3_5 * z * (xx - yy) * sh_band(i, 13u)
               + SH_C3_6 * x * (xx - 3.0 * yy) * sh_band(i, 14u);
        }
    }

    return max(c, vec3<f32>(0.0));
}

fn culled() -> SplatVsOut {
    var out: SplatVsOut;
    // Behind w, so the clipper throws the whole primitive away.
    out.clip_position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    out.color = vec3<f32>(0.0);
    out.conic_opacity = vec4<f32>(0.0);
    out.delta = vec2<f32>(0.0);
    out.normal = vec3<f32>(0.0);
    out.splat_index = 0u;
    return out;
}

// vertex_index 0..3 of a triangle strip -> the four corners of the splat's bounding quad.
fn corner_of(vertex_index: u32) -> vec2<f32> {
    return vec2<f32>(
        f32(vertex_index & 1u) * 2.0 - 1.0,
        f32((vertex_index >> 1u) & 1u) * 2.0 - 1.0,
    );
}

fn splat_vertex(vertex_index: u32, instance_index: u32) -> SplatVsOut {
    let idx = splat_order[instance_index];
    let s = splats[idx];
    let p_world = s.pos_opacity.xyz;

    let p_view = (camera.view * vec4<f32>(p_world, 1.0)).xyz;
    // View space is right-handed looking down -Z, so anything at or behind the near plane
    // has z > -near and must go.
    if (p_view.z > -camera.near) {
        return culled();
    }

    let scale2 = params.splat_scale * params.splat_scale;
    let cov3d = mat3x3<f32>(
        vec3<f32>(s.cov_a.x, s.cov_a.y, s.cov_a.z),
        vec3<f32>(s.cov_a.y, s.cov_b.x, s.cov_b.y),
        vec3<f32>(s.cov_a.z, s.cov_b.y, s.cov_b.z),
    ) * scale2;

    // Clamp the sample point towards the frustum before differentiating: the affine
    // approximation blows up on splats far outside the field of view.
    let tan_fov = 0.5 * camera.viewport / camera.focal;
    let limit = 1.3 * tan_fov * (-p_view.z);
    let t = vec3<f32>(
        clamp(p_view.x, -limit.x, limit.x),
        clamp(p_view.y, -limit.y, limit.y),
        p_view.z,
    );

    // Jacobian of (x, y, z) -> pixel (u, v), with u = -focal.x * x / z (y-up pixel space).
    let inv_z = 1.0 / t.z;
    let j = mat3x2<f32>(
        vec2<f32>(-camera.focal.x * inv_z, 0.0),
        vec2<f32>(0.0, -camera.focal.y * inv_z),
        vec2<f32>(camera.focal.x * t.x * inv_z * inv_z, camera.focal.y * t.y * inv_z * inv_z),
    );

    // W is the view rotation; glam/WGSL matrices are column-major, so the upper-left 3x3
    // of camera.view is exactly it.
    let w = mat3x3<f32>(
        camera.view[0].xyz,
        camera.view[1].xyz,
        camera.view[2].xyz,
    );

    let cov_view = w * cov3d * transpose(w);
    let cov2d_full = j * cov_view * transpose(j);

    // Low-pass filter: guarantees every splat covers at least ~one pixel, which is what
    // keeps distant Gaussians from flickering.
    let a = cov2d_full[0].x + 0.3;
    let b = cov2d_full[1].x;
    let c = cov2d_full[1].y + 0.3;

    let det = a * c - b * b;
    if (det <= 0.0) {
        return culled();
    }
    let inv_det = 1.0 / det;

    // Eigen-decomposition of the 2x2 covariance gives an oriented quad, which wastes far
    // fewer fragments than an axis-aligned one when splats are elongated.
    let mid = 0.5 * (a + c);
    let disc = sqrt(max(mid * mid - det, 0.01));
    let l1 = mid + disc;
    let l2 = max(mid - disc, 0.01);

    var major_dir = vec2<f32>(1.0, 0.0);
    if (abs(b) > 1e-9) {
        major_dir = normalize(vec2<f32>(b, l1 - a));
    } else if (c > a) {
        major_dir = vec2<f32>(0.0, 1.0);
    }
    let minor_dir = vec2<f32>(-major_dir.y, major_dir.x);

    // 3 sigma captures ~99% of the Gaussian's mass; beyond it the fragment shader's
    // alpha cutoff would discard everything anyway.
    let axis1 = major_dir * (3.0 * sqrt(l1));
    let axis2 = minor_dir * (3.0 * sqrt(l2));

    let corner = corner_of(vertex_index);
    let offset_px = corner.x * axis1 + corner.y * axis2;

    let clip_center = camera.view_proj * vec4<f32>(p_world, 1.0);
    let ndc_offset = offset_px * 2.0 / camera.viewport;

    var out: SplatVsOut;
    out.clip_position = vec4<f32>(
        clip_center.xy + ndc_offset * clip_center.w,
        clip_center.z,
        clip_center.w,
    );
    // Undo the world-space Y/Z flip applied at load time: SH coefficients are stored in
    // the reconstruction's original frame.
    let dir = normalize(p_world - camera.cam_pos) * vec3<f32>(1.0, -1.0, -1.0);
    out.color = eval_sh(idx, dir, s.color.rgb);
    out.conic_opacity = vec4<f32>(c * inv_det, -b * inv_det, a * inv_det, s.pos_opacity.w);
    out.delta = offset_px;
    out.normal = normalize(vec3<f32>(s.cov_a.w, s.cov_b.w, s.color.w));
    out.splat_index = idx;
    return out;
}

/// Gaussian alpha at this fragment, or a negative value when the fragment should be dropped.
fn splat_alpha(in: SplatVsOut) -> f32 {
    let conic = in.conic_opacity.xyz;
    let d = in.delta;
    let power = -0.5 * (conic.x * d.x * d.x + conic.z * d.y * d.y) - conic.y * d.x * d.y;
    if (power > 0.0) {
        return -1.0;
    }
    let alpha = min(0.99, in.conic_opacity.w * exp(power));
    if (alpha < 1.0 / 255.0) {
        return -1.0;
    }
    return alpha;
}
