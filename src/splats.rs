//! CPU-side Gaussian splat scene: packing for the GPU, an importance ordering used by
//! the render cap, and a background depth sorter for the alpha-blended mode.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};

use crate::ply::SplatData;

pub const DIRECTIONAL_VIEW_COUNT: usize = 64;
pub const DIRECTIONAL_DEPTH_BINS: usize = 64;
pub const SPATIAL_PRIOR_WIDTH: usize = 8;
pub const SPATIAL_PRIOR_HEIGHT: usize = 8;
pub const SPATIAL_PRIOR_DEPTH_BINS: usize = 32;
pub const HQ_DIRECTIONAL_VIEW_COUNT: usize = 512;
pub const HQ_SPATIAL_PRIOR_WIDTH: usize = 20;
pub const HQ_SPATIAL_PRIOR_HEIGHT: usize = 12;

/// A compact view-dependent optical-depth prior. Each row is baked from one evenly
/// distributed camera direction and uses scene-relative depth in [-radius, radius].
#[derive(Clone)]
pub struct DirectionalHistogramPrior {
    directions: [[f32; 3]; DIRECTIONAL_VIEW_COUNT],
    histograms: [[f32; DIRECTIONAL_DEPTH_BINS]; DIRECTIONAL_VIEW_COUNT],
    radius: f32,
}

/// A mobile-oriented directional prior that retains coarse screen-space information.
/// Values are laid out as [view][tile_y][tile_x][depth].
#[derive(Clone)]
pub struct SpatialDirectionalHistogramPrior {
    directions: [[f32; 3]; DIRECTIONAL_VIEW_COUNT],
    histograms: Vec<f32>,
    radius: f32,
}

/// Higher quality spatial bake used by mode 6. Unlike mode 5, this rasterizes each
/// projected Gaussian ellipse across every coarse screen cell it overlaps.
#[derive(Clone)]
pub struct HighQualitySpatialDirectionalPrior {
    directions: Vec<[f32; 3]>,
    histograms: Vec<f32>,
    radius: f32,
}

/// One splat as the shaders see it: 64 bytes, four `vec4`s.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SplatGpu {
    /// `xyz` = world position, `w` = opacity.
    pub pos_opacity: [f32; 4],
    /// 3D covariance upper triangle, part one: `xx, xy, xz`; `w` is normal X.
    pub cov_a: [f32; 4],
    /// 3D covariance upper triangle, part two: `yy, yz, zz`; `w` is normal Y.
    pub cov_b: [f32; 4],
    /// `rgb` = DC colour; `w` is normal Z.
    pub color: [f32; 4],
}

pub struct SplatScene {
    pub gpu: Vec<SplatGpu>,
    /// World-space positions, shared with the sorter thread.
    pub positions: Arc<Vec<[f32; 3]>>,
    /// Higher SH bands, 45 floats per splat, channel-major. Empty when the file had none.
    pub sh: Vec<f32>,
    pub sh_degree: u32,
    pub center: glam::Vec3,
    pub radius: f32,
    pub directional_prior: DirectionalHistogramPrior,
    pub spatial_directional_prior: SpatialDirectionalHistogramPrior,
    pub high_quality_spatial_prior: HighQualitySpatialDirectionalPrior,
}

/// 3DGS scenes come out of COLMAP with +Y down and +Z into the screen; flip both so the
/// orbit camera's +Y-up convention shows the scene the right way round. The shaders undo
/// this when evaluating view-dependent SH, whose coefficients live in the original frame.
const FLIP: glam::Vec3 = glam::Vec3::new(1.0, -1.0, -1.0);

impl SplatScene {
    pub fn from_ply(data: SplatData) -> Self {
        let n = data.len();
        let mut gpu = Vec::with_capacity(n);

        for i in 0..n {
            let s = glam::Vec3::from(data.scale[i]);
            let [qw, qx, qy, qz] = data.rot[i];
            let rot = glam::Mat3::from_quat(glam::Quat::from_xyzw(qx, qy, qz, qw));

            // M = flip * R * S, and cov = M M^T.
            let m = glam::Mat3::from_diagonal(FLIP)
                * rot
                * glam::Mat3::from_diagonal(s);
            let cov = m * m.transpose();

            // A Gaussian's thinnest covariance axis is the best normal-like signal the
            // reconstruction gives us. Its sign is ambiguous and is fixed after finding
            // the scene centre below.
            let normal_axis = if s.x <= s.y && s.x <= s.z {
                glam::Vec3::X
            } else if s.y <= s.z {
                glam::Vec3::Y
            } else {
                glam::Vec3::Z
            };
            let normal = (glam::Mat3::from_diagonal(FLIP) * rot * normal_axis).normalize();

            let p = glam::Vec3::from(data.pos[i]) * FLIP;

            gpu.push(SplatGpu {
                pos_opacity: [p.x, p.y, p.z, data.opacity[i]],
                cov_a: [cov.x_axis.x, cov.x_axis.y, cov.x_axis.z, normal.x],
                cov_b: [cov.y_axis.y, cov.y_axis.z, cov.z_axis.z, normal.y],
                color: [
                    data.color[i][0],
                    data.color[i][1],
                    data.color[i][2],
                    normal.z,
                ],
            });
        }

        // Order splats by visual importance so the render cap can simply draw a prefix.
        let mut order: Vec<u32> = (0..n as u32).collect();
        let importance: Vec<f32> = (0..n)
            .map(|i| {
                let s = data.scale[i];
                data.opacity[i] * (s[0] * s[1] * s[2]).abs().cbrt()
            })
            .collect();
        order.sort_unstable_by(|&a, &b| {
            importance[b as usize]
                .partial_cmp(&importance[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut gpu = permute(&gpu, &order);
        let sh = if data.sh_degree > 0 {
            let mut out = vec![0.0f32; n * 45];
            for (dst, &src) in order.iter().enumerate() {
                let (d, s) = (dst * 45, src as usize * 45);
                out[d..d + 45].copy_from_slice(&data.sh[s..s + 45]);
            }
            out
        } else {
            Vec::new()
        };

        let positions: Vec<[f32; 3]> = gpu
            .iter()
            .map(|s| [s.pos_opacity[0], s.pos_opacity[1], s.pos_opacity[2]])
            .collect();
        let (center, radius) = robust_bounds(&positions);
        // Covariance eigenvectors have no inherent sign. Point them away from the robust
        // scene centre as a procedural, object-agnostic front/back heuristic.
        for splat in &mut gpu {
            let p = glam::Vec3::from_array([
                splat.pos_opacity[0],
                splat.pos_opacity[1],
                splat.pos_opacity[2],
            ]);
            let mut normal = glam::Vec3::new(splat.cov_a[3], splat.cov_b[3], splat.color[3]);
            if normal.dot(p - center) < 0.0 {
                normal = -normal;
            }
            splat.cov_a[3] = normal.x;
            splat.cov_b[3] = normal.y;
            splat.color[3] = normal.z;
        }
        let directional_prior = DirectionalHistogramPrior::bake(&gpu, center, radius);
        let spatial_directional_prior =
            SpatialDirectionalHistogramPrior::bake(&gpu, center, radius);
        let high_quality_spatial_prior =
            HighQualitySpatialDirectionalPrior::bake(&gpu, center, radius);

        Self {
            gpu,
            positions: Arc::new(positions),
            sh,
            sh_degree: data.sh_degree,
            center,
            radius,
            directional_prior,
            spatial_directional_prior,
            high_quality_spatial_prior,
        }
    }

    pub fn len(&self) -> usize {
        self.gpu.len()
    }
}

fn fibonacci_directions() -> [[f32; 3]; DIRECTIONAL_VIEW_COUNT] {
    let mut directions = [[0.0; 3]; DIRECTIONAL_VIEW_COUNT];
    let golden_angle = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
    for (view, direction) in directions.iter_mut().enumerate() {
        let y = 1.0 - 2.0 * (view as f32 + 0.5) / DIRECTIONAL_VIEW_COUNT as f32;
        let r = (1.0 - y * y).sqrt();
        let phi = golden_angle * view as f32;
        *direction = [r * phi.cos(), y, r * phi.sin()];
    }
    directions
}

fn fibonacci_direction_vec(count: usize) -> Vec<[f32; 3]> {
    let golden_angle = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
    (0..count)
        .map(|view| {
            let y = 1.0 - 2.0 * (view as f32 + 0.5) / count as f32;
            let r = (1.0 - y * y).sqrt();
            let phi = golden_angle * view as f32;
            [r * phi.cos(), y, r * phi.sin()]
        })
        .collect()
}

fn nearest_direction_weights(
    directions: &[[f32; 3]; DIRECTIONAL_VIEW_COUNT],
    forward: glam::Vec3,
) -> ([(f32, usize); 3], [f32; 3]) {
    let mut nearest = [(f32::NEG_INFINITY, 0usize); 3];
    for (i, direction) in directions.iter().enumerate() {
        let dot = forward.dot(glam::Vec3::from_array(*direction));
        if dot > nearest[0].0 {
            nearest[2] = nearest[1];
            nearest[1] = nearest[0];
            nearest[0] = (dot, i);
        } else if dot > nearest[1].0 {
            nearest[2] = nearest[1];
            nearest[1] = (dot, i);
        } else if dot > nearest[2].0 {
            nearest[2] = (dot, i);
        }
    }

    let mut weights = [0.0; 3];
    for (weight, &(dot, _)) in weights.iter_mut().zip(&nearest) {
        *weight = 1.0 / (1e-3 + 1.0 - dot.clamp(-1.0, 1.0));
    }
    let weight_sum: f32 = weights.iter().sum();
    for weight in &mut weights {
        *weight /= weight_sum;
    }
    (nearest, weights)
}

fn nearest_direction_weights_dynamic(
    directions: &[[f32; 3]],
    forward: glam::Vec3,
) -> ([(f32, usize); 3], [f32; 3]) {
    let mut nearest = [(f32::NEG_INFINITY, 0usize); 3];
    for (i, direction) in directions.iter().enumerate() {
        let dot = forward.dot(glam::Vec3::from_array(*direction));
        if dot > nearest[0].0 {
            nearest[2] = nearest[1];
            nearest[1] = nearest[0];
            nearest[0] = (dot, i);
        } else if dot > nearest[1].0 {
            nearest[2] = nearest[1];
            nearest[1] = (dot, i);
        } else if dot > nearest[2].0 {
            nearest[2] = (dot, i);
        }
    }

    let mut weights = [0.0; 3];
    for (weight, &(dot, _)) in weights.iter_mut().zip(&nearest) {
        *weight = 1.0 / (1e-3 + 1.0 - dot.clamp(-1.0, 1.0));
    }
    let sum: f32 = weights.iter().sum();
    for weight in &mut weights {
        *weight /= sum;
    }
    (nearest, weights)
}

impl SpatialDirectionalHistogramPrior {
    fn index(view: usize, x: usize, y: usize, depth: usize) -> usize {
        (((view * SPATIAL_PRIOR_HEIGHT + y) * SPATIAL_PRIOR_WIDTH + x)
            * SPATIAL_PRIOR_DEPTH_BINS)
            + depth
    }

    fn bake(splats: &[SplatGpu], center: glam::Vec3, radius: f32) -> Self {
        let directions = fibonacci_directions();
        let mut histograms = vec![
            0.0;
            DIRECTIONAL_VIEW_COUNT
                * SPATIAL_PRIOR_WIDTH
                * SPATIAL_PRIOR_HEIGHT
                * SPATIAL_PRIOR_DEPTH_BINS
        ];
        let tan_half_fov_y = (45.0_f32.to_radians() * 0.5).tan();
        let aspect = 16.0 / 9.0;
        let tan_half_fov_x = tan_half_fov_y * aspect;
        // The bounding sphere is just inside the vertical field of view.
        let camera_distance = radius / (45.0_f32.to_radians() * 0.5).sin();

        for (view, direction) in directions.iter().enumerate() {
            let forward = glam::Vec3::from_array(*direction);
            let helper = if forward.y.abs() < 0.95 {
                glam::Vec3::Y
            } else {
                glam::Vec3::X
            };
            let screen_x = forward.cross(helper).normalize();
            let screen_y = screen_x.cross(forward).normalize();

            for splat in splats {
                let p = glam::Vec3::from_slice(&splat.pos_opacity[..3]);
                let relative = p - center;
                let depth = camera_distance + relative.dot(forward);
                if depth <= 1e-5 {
                    continue;
                }

                let ndc_x = relative.dot(screen_x) / (depth * tan_half_fov_x);
                let ndc_y = relative.dot(screen_y) / (depth * tan_half_fov_y);
                if !(-1.1..=1.1).contains(&ndc_x) || !(-1.1..=1.1).contains(&ndc_y) {
                    continue;
                }

                let covariance = glam::Mat3::from_cols(
                    glam::Vec3::new(splat.cov_a[0], splat.cov_a[1], splat.cov_a[2]),
                    glam::Vec3::new(splat.cov_a[1], splat.cov_b[0], splat.cov_b[1]),
                    glam::Vec3::new(splat.cov_a[2], splat.cov_b[1], splat.cov_b[2]),
                );
                let cxx = screen_x.dot(covariance * screen_x);
                let cxy = screen_x.dot(covariance * screen_y);
                let cyy = screen_y.dot(covariance * screen_y);
                let projected_area = (cxx * cyy - cxy * cxy).max(0.0).sqrt()
                    / (depth * depth).max(1e-8);
                let opacity = splat.pos_opacity[3].clamp(0.0, 1.0 - 1e-6);
                let weight = -(1.0 - opacity).ln() * projected_area.max(1e-10);

                let tile_x = ((ndc_x * 0.5 + 0.5) * SPATIAL_PRIOR_WIDTH as f32 - 0.5)
                    .clamp(0.0, (SPATIAL_PRIOR_WIDTH - 1) as f32);
                // Screen-space Y grows down in the render targets.
                let tile_y = ((-ndc_y * 0.5 + 0.5) * SPATIAL_PRIOR_HEIGHT as f32 - 0.5)
                    .clamp(0.0, (SPATIAL_PRIOR_HEIGHT - 1) as f32);
                let x0 = tile_x.floor() as usize;
                let y0 = tile_y.floor() as usize;
                let x1 = (x0 + 1).min(SPATIAL_PRIOR_WIDTH - 1);
                let y1 = (y0 + 1).min(SPATIAL_PRIOR_HEIGHT - 1);
                let fx = tile_x - x0 as f32;
                let fy = tile_y - y0 as f32;
                let relative_depth =
                    (0.5 + relative.dot(forward) / (2.0 * radius)).clamp(0.0, 1.0);
                let depth_bin = ((relative_depth * SPATIAL_PRIOR_DEPTH_BINS as f32) as usize)
                    .min(SPATIAL_PRIOR_DEPTH_BINS - 1);

                for (x, y, spatial_weight) in [
                    (x0, y0, (1.0 - fx) * (1.0 - fy)),
                    (x1, y0, fx * (1.0 - fy)),
                    (x0, y1, (1.0 - fx) * fy),
                    (x1, y1, fx * fy),
                ] {
                    histograms[Self::index(view, x, y, depth_bin)] +=
                        weight * spatial_weight;
                }
            }

            for y in 0..SPATIAL_PRIOR_HEIGHT {
                for x in 0..SPATIAL_PRIOR_WIDTH {
                    let base = Self::index(view, x, y, 0);
                    let sum: f32 = histograms[base..base + SPATIAL_PRIOR_DEPTH_BINS]
                        .iter()
                        .sum();
                    if sum > 0.0 {
                        for value in &mut histograms[base..base + SPATIAL_PRIOR_DEPTH_BINS] {
                            *value /= sum;
                        }
                    } else {
                        histograms[base..base + SPATIAL_PRIOR_DEPTH_BINS]
                            .fill(1.0 / SPATIAL_PRIOR_DEPTH_BINS as f32);
                    }
                }
            }
        }

        Self {
            directions,
            histograms,
            radius,
        }
    }

    /// Returns one 8x8x64 histogram volume in the live camera's normalized depth space.
    pub fn sample_for_camera(
        &self,
        forward: glam::Vec3,
        distance: f32,
        near: f32,
        far: f32,
    ) -> Vec<f32> {
        let (nearest, weights) = nearest_direction_weights(&self.directions, forward);
        let mut output = vec![
            0.0;
            SPATIAL_PRIOR_WIDTH * SPATIAL_PRIOR_HEIGHT * DIRECTIONAL_DEPTH_BINS
        ];
        let depth_range = (far - near).max(1e-6);

        for y in 0..SPATIAL_PRIOR_HEIGHT {
            for x in 0..SPATIAL_PRIOR_WIDTH {
                let out_base = (y * SPATIAL_PRIOR_WIDTH + x) * DIRECTIONAL_DEPTH_BINS;
                for source_bin in 0..SPATIAL_PRIOR_DEPTH_BINS {
                    let mut value = 0.0;
                    for (neighbor, weight) in nearest.iter().zip(weights) {
                        value += weight
                            * self.histograms
                                [Self::index(neighbor.1, x, y, source_bin)];
                    }
                    let scene_t = (source_bin as f32 + 0.5) / SPATIAL_PRIOR_DEPTH_BINS as f32;
                    let offset = (scene_t * 2.0 - 1.0) * self.radius;
                    let camera_t = ((distance + offset - near) / depth_range).clamp(0.0, 1.0);
                    let target = camera_t * (DIRECTIONAL_DEPTH_BINS - 1) as f32;
                    let lo = target.floor() as usize;
                    let hi = (lo + 1).min(DIRECTIONAL_DEPTH_BINS - 1);
                    let fraction = target - lo as f32;
                    output[out_base + lo] += value * (1.0 - fraction);
                    output[out_base + hi] += value * fraction;
                }
            }
        }
        output
    }
}

impl HighQualitySpatialDirectionalPrior {
    fn index(view: usize, x: usize, y: usize, depth: usize) -> usize {
        (((view * HQ_SPATIAL_PRIOR_HEIGHT + y) * HQ_SPATIAL_PRIOR_WIDTH + x)
            * SPATIAL_PRIOR_DEPTH_BINS)
            + depth
    }

    fn bake(splats: &[SplatGpu], center: glam::Vec3, radius: f32) -> Self {
        let directions = fibonacci_direction_vec(HQ_DIRECTIONAL_VIEW_COUNT);
        let mut histograms = vec![
            0.0;
            HQ_DIRECTIONAL_VIEW_COUNT
                * HQ_SPATIAL_PRIOR_WIDTH
                * HQ_SPATIAL_PRIOR_HEIGHT
                * SPATIAL_PRIOR_DEPTH_BINS
        ];
        let tan_half_fov_y = (45.0_f32.to_radians() * 0.5).tan();
        let tan_half_fov_x = tan_half_fov_y * (16.0 / 9.0);
        let camera_distance = radius / (45.0_f32.to_radians() * 0.5).sin();

        for (view, direction) in directions.iter().enumerate() {
            let forward = glam::Vec3::from_array(*direction);
            let helper = if forward.y.abs() < 0.9999 {
                glam::Vec3::Y
            } else {
                glam::Vec3::X
            };
            let screen_x = forward.cross(helper).normalize();
            let screen_y = screen_x.cross(forward).normalize();

            for splat in splats {
                let p = glam::Vec3::from_slice(&splat.pos_opacity[..3]);
                let relative = p - center;
                let view_x = relative.dot(screen_x);
                let view_y = relative.dot(screen_y);
                let depth = camera_distance + relative.dot(forward);
                if depth <= 1e-5 {
                    continue;
                }

                let ndc_x = view_x / (depth * tan_half_fov_x);
                let ndc_y = view_y / (depth * tan_half_fov_y);
                if !(-1.25..=1.25).contains(&ndc_x) || !(-1.25..=1.25).contains(&ndc_y) {
                    continue;
                }

                let covariance = glam::Mat3::from_cols(
                    glam::Vec3::new(splat.cov_a[0], splat.cov_a[1], splat.cov_a[2]),
                    glam::Vec3::new(splat.cov_a[1], splat.cov_b[0], splat.cov_b[1]),
                    glam::Vec3::new(splat.cov_a[2], splat.cov_b[1], splat.cov_b[2]),
                );

                // Perspective Jacobian in world space, including the depth derivative.
                let grad_x = screen_x / (depth * tan_half_fov_x)
                    - forward * (view_x / (depth * depth * tan_half_fov_x));
                let grad_y = screen_y / (depth * tan_half_fov_y)
                    - forward * (view_y / (depth * depth * tan_half_fov_y));
                let ndc_cxx = grad_x.dot(covariance * grad_x).max(0.0);
                let ndc_cxy = grad_x.dot(covariance * grad_y);
                let ndc_cyy = grad_y.dot(covariance * grad_y).max(0.0);
                let ndc_det = (ndc_cxx * ndc_cyy - ndc_cxy * ndc_cxy).max(0.0);

                let sx = HQ_SPATIAL_PRIOR_WIDTH as f32 * 0.5;
                let sy = -(HQ_SPATIAL_PRIOR_HEIGHT as f32) * 0.5;
                let center_x = (ndc_x * 0.5 + 0.5) * HQ_SPATIAL_PRIOR_WIDTH as f32 - 0.5;
                let center_y = (-ndc_y * 0.5 + 0.5) * HQ_SPATIAL_PRIOR_HEIGHT as f32 - 0.5;

                // A half-cell reconstruction footprint prevents sub-cell Gaussians from
                // disappearing while preserving the actual ellipse for larger splats.
                let cxx = ndc_cxx * sx * sx + 0.25;
                let cxy = ndc_cxy * sx * sy;
                let cyy = ndc_cyy * sy * sy + 0.25;
                let det = cxx * cyy - cxy * cxy;
                if det <= 1e-10 {
                    continue;
                }
                let trace_half = 0.5 * (cxx + cyy);
                let lambda_max = trace_half
                    + (trace_half * trace_half - det).max(0.0).sqrt();
                let extent = 3.0 * lambda_max.sqrt();
                let min_x = (center_x - extent)
                    .floor()
                    .clamp(0.0, (HQ_SPATIAL_PRIOR_WIDTH - 1) as f32)
                    as usize;
                let max_x = (center_x + extent)
                    .ceil()
                    .clamp(0.0, (HQ_SPATIAL_PRIOR_WIDTH - 1) as f32)
                    as usize;
                let min_y = (center_y - extent)
                    .floor()
                    .clamp(0.0, (HQ_SPATIAL_PRIOR_HEIGHT - 1) as f32)
                    as usize;
                let max_y = (center_y + extent)
                    .ceil()
                    .clamp(0.0, (HQ_SPATIAL_PRIOR_HEIGHT - 1) as f32)
                    as usize;

                let inv_xx = cyy / det;
                let inv_xy = -cxy / det;
                let inv_yy = cxx / det;
                let mut kernel_sum = 0.0;
                for y in min_y..=max_y {
                    for x in min_x..=max_x {
                        let dx = x as f32 - center_x;
                        let dy = y as f32 - center_y;
                        let power = -0.5
                            * (inv_xx * dx * dx + 2.0 * inv_xy * dx * dy + inv_yy * dy * dy);
                        if power >= -4.5 {
                            kernel_sum += power.exp();
                        }
                    }
                }
                if kernel_sum <= 0.0 {
                    continue;
                }

                let opacity = splat.pos_opacity[3].clamp(0.0, 1.0 - 1e-6);
                let optical_depth = -(1.0 - opacity).ln();
                let total_weight = optical_depth * ndc_det.sqrt().max(1e-10);
                let relative_depth =
                    (0.5 + relative.dot(forward) / (2.0 * radius)).clamp(0.0, 1.0);
                let depth_bin = ((relative_depth * SPATIAL_PRIOR_DEPTH_BINS as f32) as usize)
                    .min(SPATIAL_PRIOR_DEPTH_BINS - 1);

                for y in min_y..=max_y {
                    for x in min_x..=max_x {
                        let dx = x as f32 - center_x;
                        let dy = y as f32 - center_y;
                        let power = -0.5
                            * (inv_xx * dx * dx + 2.0 * inv_xy * dx * dy + inv_yy * dy * dy);
                        if power >= -4.5 {
                            histograms[Self::index(view, x, y, depth_bin)] +=
                                total_weight * power.exp() / kernel_sum;
                        }
                    }
                }
            }

            for y in 0..HQ_SPATIAL_PRIOR_HEIGHT {
                for x in 0..HQ_SPATIAL_PRIOR_WIDTH {
                    let base = Self::index(view, x, y, 0);
                    let values = &mut histograms[base..base + SPATIAL_PRIOR_DEPTH_BINS];
                    let sum: f32 = values.iter().sum();
                    if sum > 0.0 {
                        for value in values {
                            *value /= sum;
                        }
                    } else {
                        values.fill(1.0 / SPATIAL_PRIOR_DEPTH_BINS as f32);
                    }
                }
            }
        }

        Self {
            directions,
            histograms,
            radius,
        }
    }

    pub fn sample_for_camera(
        &self,
        forward: glam::Vec3,
        distance: f32,
        near: f32,
        far: f32,
    ) -> Vec<f32> {
        let (nearest, weights) = nearest_direction_weights_dynamic(&self.directions, forward);
        let mut output = vec![
            0.0;
            HQ_SPATIAL_PRIOR_WIDTH * HQ_SPATIAL_PRIOR_HEIGHT * DIRECTIONAL_DEPTH_BINS
        ];
        let depth_range = (far - near).max(1e-6);
        let baked_distance = self.radius / (45.0_f32.to_radians() * 0.5).sin();

        for y in 0..HQ_SPATIAL_PRIOR_HEIGHT {
            for x in 0..HQ_SPATIAL_PRIOR_WIDTH {
                let out_base = (y * HQ_SPATIAL_PRIOR_WIDTH + x) * DIRECTIONAL_DEPTH_BINS;
                for source_bin in 0..SPATIAL_PRIOR_DEPTH_BINS {
                    let scene_t = (source_bin as f32 + 0.5) / SPATIAL_PRIOR_DEPTH_BINS as f32;
                    let offset = (scene_t * 2.0 - 1.0) * self.radius;
                    let projection_scale = (distance + offset)
                        / (baked_distance + offset).max(1e-5);
                    let source_x = (((x as f32 + 0.5)
                        - HQ_SPATIAL_PRIOR_WIDTH as f32 * 0.5)
                        * projection_scale
                        + HQ_SPATIAL_PRIOR_WIDTH as f32 * 0.5
                        - 0.5)
                        .clamp(0.0, (HQ_SPATIAL_PRIOR_WIDTH - 1) as f32);
                    let source_y = (((y as f32 + 0.5)
                        - HQ_SPATIAL_PRIOR_HEIGHT as f32 * 0.5)
                        * projection_scale
                        + HQ_SPATIAL_PRIOR_HEIGHT as f32 * 0.5
                        - 0.5)
                        .clamp(0.0, (HQ_SPATIAL_PRIOR_HEIGHT - 1) as f32);
                    let x0 = source_x.floor() as usize;
                    let y0 = source_y.floor() as usize;
                    let x1 = (x0 + 1).min(HQ_SPATIAL_PRIOR_WIDTH - 1);
                    let y1 = (y0 + 1).min(HQ_SPATIAL_PRIOR_HEIGHT - 1);
                    let fx = source_x - x0 as f32;
                    let fy = source_y - y0 as f32;
                    let mut value = 0.0;
                    for (neighbor, weight) in nearest.iter().zip(weights) {
                        let top = self.histograms[Self::index(neighbor.1, x0, y0, source_bin)]
                            * (1.0 - fx)
                            + self.histograms[Self::index(neighbor.1, x1, y0, source_bin)] * fx;
                        let bottom = self.histograms[Self::index(neighbor.1, x0, y1, source_bin)]
                            * (1.0 - fx)
                            + self.histograms[Self::index(neighbor.1, x1, y1, source_bin)] * fx;
                        value += weight * (top * (1.0 - fy) + bottom * fy);
                    }
                    let camera_t = ((distance + offset - near) / depth_range).clamp(0.0, 1.0);
                    let target = camera_t * (DIRECTIONAL_DEPTH_BINS - 1) as f32;
                    let lo = target.floor() as usize;
                    let hi = (lo + 1).min(DIRECTIONAL_DEPTH_BINS - 1);
                    let fraction = target - lo as f32;
                    output[out_base + lo] += value * (1.0 - fraction);
                    output[out_base + hi] += value * fraction;
                }
            }
        }
        output
    }
}

impl DirectionalHistogramPrior {
    fn bake(splats: &[SplatGpu], center: glam::Vec3, radius: f32) -> Self {
        let directions = fibonacci_directions();
        let mut histograms = [[0.0; DIRECTIONAL_DEPTH_BINS]; DIRECTIONAL_VIEW_COUNT];

        for view in 0..DIRECTIONAL_VIEW_COUNT {
            let forward = glam::Vec3::from_array(directions[view]);

            let helper = if forward.y.abs() < 0.9 {
                glam::Vec3::Y
            } else {
                glam::Vec3::X
            };
            let screen_x = forward.cross(helper).normalize();
            let screen_y = screen_x.cross(forward).normalize();

            for splat in splats {
                let p = glam::Vec3::from_slice(&splat.pos_opacity[..3]);
                let normalized_depth =
                    (0.5 + (p - center).dot(forward) / (2.0 * radius)).clamp(0.0, 1.0);
                let bin = ((normalized_depth * DIRECTIONAL_DEPTH_BINS as f32) as usize)
                    .min(DIRECTIONAL_DEPTH_BINS - 1);

                let covariance = glam::Mat3::from_cols(
                    glam::Vec3::new(splat.cov_a[0], splat.cov_a[1], splat.cov_a[2]),
                    glam::Vec3::new(splat.cov_a[1], splat.cov_b[0], splat.cov_b[1]),
                    glam::Vec3::new(splat.cov_a[2], splat.cov_b[1], splat.cov_b[2]),
                );
                let cxx = screen_x.dot(covariance * screen_x);
                let cxy = screen_x.dot(covariance * screen_y);
                let cyy = screen_y.dot(covariance * screen_y);
                let projected_area = (cxx * cyy - cxy * cxy).max(0.0).sqrt();
                let opacity = splat.pos_opacity[3].clamp(0.0, 1.0 - 1e-6);
                let optical_depth = -(1.0 - opacity).ln();
                histograms[view][bin] += optical_depth * projected_area.max(1e-8);
            }

            let sum: f32 = histograms[view].iter().sum();
            if sum > 0.0 {
                for value in &mut histograms[view] {
                    *value /= sum;
                }
            } else {
                histograms[view].fill(1.0 / DIRECTIONAL_DEPTH_BINS as f32);
            }
        }

        Self {
            directions,
            histograms,
            radius,
        }
    }

    /// Blend the three closest baked directions, then map the scene-relative bins onto
    /// the live camera's near/far depth coordinate so zooming does not invalidate the bake.
    pub fn sample_for_camera(
        &self,
        forward: glam::Vec3,
        distance: f32,
        near: f32,
        far: f32,
    ) -> [f32; DIRECTIONAL_DEPTH_BINS] {
        let (nearest, weights) = nearest_direction_weights(&self.directions, forward);

        let mut relative = [0.0; DIRECTIONAL_DEPTH_BINS];
        for (neighbor, weight) in nearest.iter().zip(weights) {
            for (dst, src) in relative.iter_mut().zip(&self.histograms[neighbor.1]) {
                *dst += weight * src;
            }
        }

        let mut camera_bins = [0.0; DIRECTIONAL_DEPTH_BINS];
        let depth_range = (far - near).max(1e-6);
        for (i, &value) in relative.iter().enumerate() {
            let scene_t = (i as f32 + 0.5) / DIRECTIONAL_DEPTH_BINS as f32;
            let offset = (scene_t * 2.0 - 1.0) * self.radius;
            let camera_t = ((distance + offset - near) / depth_range).clamp(0.0, 1.0);
            let x = camera_t * (DIRECTIONAL_DEPTH_BINS - 1) as f32;
            let lo = x.floor() as usize;
            let hi = (lo + 1).min(DIRECTIONAL_DEPTH_BINS - 1);
            let fraction = x - lo as f32;
            camera_bins[lo] += value * (1.0 - fraction);
            camera_bins[hi] += value * fraction;
        }
        camera_bins
    }
}

fn permute<T: Copy>(src: &[T], order: &[u32]) -> Vec<T> {
    order.iter().map(|&i| src[i as usize]).collect()
}

/// Centre and radius that ignore the outlier haze most 3DGS reconstructions carry,
/// by clipping each axis to its 2nd..98th percentile.
fn robust_bounds(positions: &[[f32; 3]]) -> (glam::Vec3, f32) {
    if positions.is_empty() {
        return (glam::Vec3::ZERO, 1.0);
    }

    let mut lo = glam::Vec3::ZERO;
    let mut hi = glam::Vec3::ZERO;
    let mut axis: Vec<f32> = Vec::with_capacity(positions.len());
    for a in 0..3 {
        axis.clear();
        axis.extend(positions.iter().map(|p| p[a]));
        axis.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        let k = axis.len() / 50;
        lo[a] = axis[k];
        hi[a] = axis[axis.len() - 1 - k];
    }

    let center = (lo + hi) * 0.5;
    let radius = ((hi - lo) * 0.5).length().max(1e-3);
    (center, radius)
}

// ---------------------------------------------------------------------------
// Depth sorting
// ---------------------------------------------------------------------------

struct SortRequest {
    forward: glam::Vec3,
    count: usize,
}

/// Back-to-front depth sort, run off-thread.
///
/// The key is view-space depth quantized to 16 bits, which turns the sort into a single
/// counting-sort pass: O(n) with no comparisons, ~5-10 ms for a million splats. The render
/// thread never blocks on it — it keeps drawing the previous frame's order until a fresh
/// one shows up, which at orbit speeds is visually indistinguishable.
pub struct Sorter {
    tx: Sender<SortRequest>,
    rx: Receiver<Vec<u32>>,
    /// Whether a request is in flight; keeps the queue from growing without bound.
    pending: bool,
    alive: bool,
}

impl Sorter {
    pub fn new(positions: Arc<Vec<[f32; 3]>>) -> Self {
        let (tx, req_rx) = channel::<SortRequest>();
        let (res_tx, rx) = channel::<Vec<u32>>();

        std::thread::Builder::new()
            .name("splat-sort".into())
            .spawn(move || {
                let mut keys: Vec<u16> = Vec::new();
                let mut counts: Vec<u32> = vec![0; 65536];
                let mut out: Vec<u32> = Vec::new();

                while let Ok(mut req) = req_rx.recv() {
                    // Only the newest camera matters; drop anything that queued up behind it.
                    while let Ok(newer) = req_rx.try_recv() {
                        req = newer;
                    }
                    let started = std::time::Instant::now();
                    counting_sort(&positions, &req, &mut keys, &mut counts, &mut out);
                    log::debug!(
                        "sorted {} splats in {:.2} ms",
                        out.len(),
                        started.elapsed().as_secs_f32() * 1000.0
                    );
                    if res_tx.send(out.clone()).is_err() {
                        break;
                    }
                }
            })
            .expect("failed to spawn splat sort thread");

        Self {
            tx,
            rx,
            pending: false,
            alive: true,
        }
    }

    /// Ask for a new ordering unless one is already being computed.
    pub fn request(&mut self, forward: glam::Vec3, count: usize) {
        if self.pending || !self.alive || count == 0 {
            return;
        }
        if self.tx.send(SortRequest { forward, count }).is_ok() {
            self.pending = true;
        } else {
            self.alive = false;
        }
    }

    /// Pick up a finished ordering, if there is one.
    pub fn poll(&mut self) -> Option<Vec<u32>> {
        match self.rx.try_recv() {
            Ok(order) => {
                self.pending = false;
                Some(order)
            }
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => {
                self.alive = false;
                self.pending = false;
                None
            }
        }
    }
}

/// Deterministic full-precision back-to-front order for benchmark reference images.
/// The interactive sorter uses a faster quantized counting sort; benchmarking pays the
/// one-time `O(n log n)` cost so sort quantization is not folded into the measured error.
pub fn exact_back_to_front_order(
    positions: &[[f32; 3]],
    forward: glam::Vec3,
    count: usize,
) -> Vec<u32> {
    let mut order: Vec<u32> = (0..count.min(positions.len()) as u32).collect();
    order.sort_unstable_by(|&a, &b| {
        let a = glam::Vec3::from_array(positions[a as usize]).dot(forward);
        let b = glam::Vec3::from_array(positions[b as usize]).dot(forward);
        b.partial_cmp(&a).unwrap_or(std::cmp::Ordering::Equal)
    });
    order
}

fn counting_sort(
    positions: &[[f32; 3]],
    req: &SortRequest,
    keys: &mut Vec<u16>,
    counts: &mut [u32],
    out: &mut Vec<u32>,
) {
    let n = req.count.min(positions.len());
    let f = req.forward;

    keys.clear();
    keys.reserve(n);
    out.clear();
    out.resize(n, 0);

    let mut min_d = f32::INFINITY;
    let mut max_d = f32::NEG_INFINITY;
    for p in &positions[..n] {
        let d = p[0] * f.x + p[1] * f.y + p[2] * f.z;
        min_d = min_d.min(d);
        max_d = max_d.max(d);
    }
    let span = max_d - min_d;
    let scale = if span > 1e-9 { 65535.0 / span } else { 0.0 };

    counts.fill(0);
    for p in &positions[..n] {
        let d = p[0] * f.x + p[1] * f.y + p[2] * f.z;
        let k = ((d - min_d) * scale) as u16;
        keys.push(k);
        counts[k as usize] += 1;
    }

    // Prefix sum from the far end down, so the largest depth lands first: back to front.
    let mut running = 0u32;
    for b in (0..65536).rev() {
        let c = counts[b];
        counts[b] = running;
        running += c;
    }

    for (i, &k) in keys.iter().enumerate() {
        let slot = &mut counts[k as usize];
        out[*slot as usize] = i as u32;
        *slot += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counting_sort_orders_back_to_front() {
        // Points along +Z at increasing distance from an eye looking down +Z.
        let positions: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 9.0],
            [0.0, 0.0, 3.0],
        ];
        let req = SortRequest {
            forward: glam::Vec3::Z,
            count: positions.len(),
        };
        let (mut keys, mut counts, mut out) = (Vec::new(), vec![0u32; 65536], Vec::new());
        counting_sort(&positions, &req, &mut keys, &mut counts, &mut out);

        // Farthest first.
        assert_eq!(out, vec![2, 0, 3, 1]);

        // Depths must be non-increasing along the draw order.
        let depths: Vec<f32> = out.iter().map(|&i| positions[i as usize][2]).collect();
        assert!(depths.windows(2).all(|w| w[0] >= w[1]));
    }

    #[test]
    fn counting_sort_respects_the_render_cap() {
        let positions: Vec<[f32; 3]> = (0..10).map(|i| [0.0, 0.0, i as f32]).collect();
        let req = SortRequest {
            forward: glam::Vec3::Z,
            count: 4,
        };
        let (mut keys, mut counts, mut out) = (Vec::new(), vec![0u32; 65536], Vec::new());
        counting_sort(&positions, &req, &mut keys, &mut counts, &mut out);

        // Only the first `count` splats participate, still ordered far to near.
        assert_eq!(out, vec![3, 2, 1, 0]);
    }

    #[test]
    fn degenerate_scenes_do_not_panic() {
        let positions = vec![[1.0, 2.0, 3.0]; 5];
        let req = SortRequest {
            forward: glam::Vec3::Z,
            count: 5,
        };
        let (mut keys, mut counts, mut out) = (Vec::new(), vec![0u32; 65536], Vec::new());
        counting_sort(&positions, &req, &mut keys, &mut counts, &mut out);
        assert_eq!(out.len(), 5);
        let mut seen = out.clone();
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 2, 3, 4]);
    }
}
