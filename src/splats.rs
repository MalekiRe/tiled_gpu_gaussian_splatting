//! CPU-side Gaussian splat scene: packing for the GPU, an importance ordering used by
//! the render cap, and a background depth sorter for the alpha-blended mode.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};

use crate::ply::SplatData;

/// One splat as the shaders see it: 64 bytes, four `vec4`s.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SplatGpu {
    /// `xyz` = world position, `w` = opacity.
    pub pos_opacity: [f32; 4],
    /// 3D covariance upper triangle, part one: `xx, xy, xz`.
    pub cov_a: [f32; 4],
    /// 3D covariance upper triangle, part two: `yy, yz, zz`.
    pub cov_b: [f32; 4],
    /// `rgb` = DC colour.
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

            let p = glam::Vec3::from(data.pos[i]) * FLIP;

            gpu.push(SplatGpu {
                pos_opacity: [p.x, p.y, p.z, data.opacity[i]],
                cov_a: [cov.x_axis.x, cov.x_axis.y, cov.x_axis.z, 0.0],
                cov_b: [cov.y_axis.y, cov.y_axis.z, cov.z_axis.z, 0.0],
                color: [
                    data.color[i][0],
                    data.color[i][1],
                    data.color[i][2],
                    0.0,
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

        let gpu = permute(&gpu, &order);
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

        Self {
            gpu,
            positions: Arc::new(positions),
            sh,
            sh_degree: data.sh_degree,
            center,
            radius,
        }
    }

    pub fn len(&self) -> usize {
        self.gpu.len()
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
