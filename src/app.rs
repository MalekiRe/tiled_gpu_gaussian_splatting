use crate::camera::Camera;
use crate::renderer::{RenderMode, Renderer};
use crate::scene::Scene;
use crate::splats::{SplatScene, Sorter, exact_back_to_front_order};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

pub struct App {
    window: Option<std::sync::Arc<Window>>,
    renderer: Option<Renderer>,
    camera: Camera,
    scene: Scene,
    last_frame: std::time::Instant,
    scene_uploaded: bool,

    /// Loaded from the command line; `None` means the built-in quad scene.
    splats: Option<SplatScene>,
    sorter: Option<Sorter>,
    splats_uploaded: bool,
    /// Index into `CAP_FRACTIONS`.
    cap: usize,
    benchmark: Option<BenchmarkConfig>,
}

#[derive(Clone, Copy)]
pub struct BenchmarkConfig {
    pub views: usize,
    pub seed: u64,
}

/// Render-cap steps cycled with `C`, as a fraction of the loaded scene.
const CAP_FRACTIONS: [f32; 4] = [1.0, 0.5, 0.25, 0.1];

impl App {
    pub fn new(splats: Option<SplatScene>, benchmark: Option<BenchmarkConfig>) -> Self {
        let sorter = if benchmark.is_none() {
            splats
                .as_ref()
                .map(|s| Sorter::new(std::sync::Arc::clone(&s.positions)))
        } else {
            None
        };
        Self {
            window: None,
            renderer: None,
            camera: Camera::new(16.0 / 9.0),
            scene: Scene::new(),
            last_frame: std::time::Instant::now(),
            scene_uploaded: false,
            splats,
            sorter,
            splats_uploaded: false,
            cap: 0,
            benchmark,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let title = if self.splats.is_some() {
            "3DGS WBOIT Demo - 1/2/3/4/5/6/7/8 modes, C cap, [ ] size, R reset"
        } else {
            "WBOIT Demo - Press 1/2/3/4/5/6/7/8 to switch modes, M meshes"
        };
        let benchmark_size = if self.benchmark.is_some() {
            winit::dpi::LogicalSize::new(640, 360)
        } else {
            winit::dpi::LogicalSize::new(1280, 720)
        };
        let attrs = Window::default_attributes()
            .with_title(title)
            .with_inner_size(benchmark_size)
            .with_visible(self.benchmark.is_none())
            .with_transparent(true);

        let window = std::sync::Arc::new(event_loop.create_window(attrs).unwrap());
        let size = window.inner_size();
        self.camera.aspect = size.width as f32 / size.height as f32;
        self.camera.viewport = (size.width as f32, size.height as f32);
        if let Some(splats) = &self.splats {
            self.camera.fit_to(splats.center, splats.radius);
        }

        let renderer = Renderer::new(window.clone());
        self.renderer = Some(renderer);
        self.window = Some(window);

        if let Some(renderer) = &self.renderer {
            println!("Mode: {}", renderer.mode.name());
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                if let Some(renderer) = &mut self.renderer {
                    self.camera.aspect = new_size.width as f32 / new_size.height.max(1) as f32;
                    self.camera.viewport =
                        (new_size.width as f32, new_size.height.max(1) as f32);
                    renderer.resize(new_size.width, new_size.height);
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key,
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                if let Some(renderer) = &mut self.renderer {
                    match &logical_key {
                        Key::Character(c) => match c.as_str() {
                            "1" => {
                                renderer.mode = RenderMode::AlphaBlend;
                                println!("Mode: {}", renderer.mode.name());
                            }
                            "2" => {
                                renderer.mode = RenderMode::NaiveWboit;
                                println!("Mode: {}", renderer.mode.name());
                            }
                            "3" => {
                                renderer.mode = RenderMode::HistogramWboit;
                                println!("Mode: {}", renderer.mode.name());
                            }
                            "4" => {
                                renderer.mode = RenderMode::DirectionalHistogramWboit;
                                println!("Mode: {}", renderer.mode.name());
                            }
                            "5" => {
                                renderer.mode = RenderMode::SpatialBakedHistogramWboit;
                                println!("Mode: {}", renderer.mode.name());
                            }
                            "6" => {
                                renderer.mode = RenderMode::HighQualitySpatialBakedWboit;
                                println!("Mode: {}", renderer.mode.name());
                            }
                            "7" => {
                                renderer.mode = RenderMode::DoubleSampleFrontWboit;
                                println!("Mode: {}", renderer.mode.name());
                            }
                            "8" => {
                                renderer.mode = RenderMode::DepthSlicedOit;
                                println!("Mode: {}", renderer.mode.name());
                            }
                            "r" | "R" => {
                                self.camera.reset();
                                println!("Camera reset");
                            }
                            "m" | "M" => {
                                self.scene.show_meshes = !self.scene.show_meshes;
                                println!(
                                    "Meshes: {}",
                                    if self.scene.show_meshes { "ON" } else { "OFF" }
                                );
                            }
                            "c" | "C" if renderer.has_splats() => {
                                self.cap = (self.cap + 1) % CAP_FRACTIONS.len();
                                if let Some((drawn, total)) =
                                    renderer.set_splat_fraction(CAP_FRACTIONS[self.cap])
                                {
                                    println!(
                                        "Render cap: {:.0}% ({drawn} / {total} splats)",
                                        CAP_FRACTIONS[self.cap] * 100.0
                                    );
                                }
                            }
                            "[" | "]" => {
                                let factor = if c.as_str() == "[" { 1.0 / 1.15 } else { 1.15 };
                                if let Some(scale) = renderer.adjust_splat_scale(factor) {
                                    println!("Splat size: {scale:.2}x");
                                }
                            }
                            "o" | "O" => {
                                self.scene.force_opaque = !self.scene.force_opaque;
                                println!(
                                    "Opaque: {}",
                                    if self.scene.force_opaque { "ON" } else { "OFF" }
                                );
                            }
                            _ => {}
                        },
                        Key::Named(NamedKey::Escape) => {
                            event_loop.exit();
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == winit::event::MouseButton::Left {
                    self.camera.dragging = state == ElementState::Pressed;
                    if !self.camera.dragging {
                        self.camera.last_mouse = None;
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.camera.on_mouse_move(position.x, position.y);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 50.0,
                };
                self.camera.on_scroll(scroll);
            }
            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
                let dt = (now - self.last_frame).as_secs_f32();
                self.last_frame = now;

                self.scene.update(dt);

                if let Some(renderer) = &mut self.renderer {
                    if !self.scene_uploaded {
                        renderer.upload_scene(&self.scene);
                        self.scene_uploaded = true;
                    }
                    if let Some(splats) = &self.splats
                        && !self.splats_uploaded
                    {
                        renderer.upload_splats(splats);
                        self.splats_uploaded = true;
                    }

                    if let Some(config) = self.benchmark.take() {
                        let splats = self
                            .splats
                            .as_ref()
                            .expect("benchmark mode requires a splat PLY");
                        run_benchmark(renderer, &mut self.camera, &self.scene, splats, config);
                        event_loop.exit();
                        return;
                    }

                    // Keep the depth order fresh for mode 1. The sort runs off-thread, so
                    // this never blocks; we just pick up whatever it has finished.
                    if let Some(sorter) = &mut self.sorter {
                        if let Some(order) = sorter.poll() {
                            renderer.upload_splat_order(&order);
                        }
                        if renderer.mode == RenderMode::AlphaBlend {
                            sorter.request(self.camera.forward(), renderer.splat_draw_count());
                        }
                    }

                    renderer.render(&self.camera, &self.scene);
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

const BENCHMARK_MODES: [RenderMode; 1] = [RenderMode::DepthSlicedOit];

fn run_benchmark(
    renderer: &mut Renderer,
    camera: &mut Camera,
    scene: &Scene,
    splats: &SplatScene,
    config: BenchmarkConfig,
) {
    #[derive(Default)]
    struct Samples {
        foreground: Vec<f64>,
        full_frame: Vec<f64>,
        sparkle: Vec<f64>,
    }

    let views = config.views.max(1);
    let mut results: Vec<(RenderMode, Samples)> = BENCHMARK_MODES
        .iter()
        .copied()
        .map(|mode| (mode, Samples::default()))
        .collect();
    let mut rng = SplitMix64(config.seed);

    println!(
        "Benchmark: {views} deterministic stochastic views, seed {}, {}x{}",
        config.seed, camera.viewport.0 as u32, camera.viewport.1 as u32
    );
    println!("Reference: mode 1 with full-precision per-view back-to-front sorting");
    println!("Candidate: mode 8 only");

    for view_index in 0..views {
        // Uniform direction on the sphere and a modest randomized orbit radius. At the
        // minimum distance the robust scene sphere remains inside the 45-degree frustum.
        camera.yaw = rng.unit_f32() * std::f32::consts::TAU;
        camera.pitch = (2.0 * rng.unit_f32() - 1.0).asin();
        camera.distance = splats.radius * (2.8 + 0.7 * rng.unit_f32());

        let order = exact_back_to_front_order(
            &splats.positions,
            camera.forward(),
            renderer.splat_draw_count(),
        );
        renderer.upload_splat_order(&order);
        renderer.mode = RenderMode::AlphaBlend;
        let reference = renderer.capture_linear_rgba(camera, scene);

        for (mode, samples) in &mut results {
            renderer.mode = *mode;
            if matches!(
                mode,
                RenderMode::SpatialBakedHistogramWboit
                    | RenderMode::HighQualitySpatialBakedWboit
                    | RenderMode::DoubleSampleFrontWboit
            ) {
                // These modes consume one-frame-old revealage. Prime it at this exact pose
                // before scoring so view-to-view ordering does not contaminate the loss.
                let _ = renderer.capture_linear_rgba(camera, scene);
            }
            let candidate = renderer.capture_linear_rgba(camera, scene);
            let (foreground, full_frame, sparkle) = image_losses(
                &reference,
                &candidate,
                camera.viewport.0 as usize,
                camera.viewport.1 as usize,
            );
            samples.foreground.push(foreground);
            samples.full_frame.push(full_frame);
            samples.sparkle.push(sparkle);
        }
        println!("  view {:>3}/{views} complete", view_index + 1);
    }

    println!("\nLinear premultiplied-RGBA MSE (lower is better)");
    println!(
        "{:<5} {:<39} {:>12} {:>12} {:>12} {:>10}",
        "Mode", "Name", "Mean", "Variance", "Worst", "PSNR"
    );
    for (mode, samples) in &results {
        let mean = mean(&samples.foreground);
        let variance = variance(&samples.foreground, mean);
        let worst = samples.foreground.iter().copied().fold(0.0, f64::max);
        let psnr = if mean > 0.0 {
            -10.0 * mean.log10()
        } else {
            f64::INFINITY
        };
        println!(
            "{:<5} {:<39} {:>12.6e} {:>12.6e} {:>12.6e} {:>9.2} dB",
            *mode as u32,
            mode.name(),
            mean,
            variance,
            worst,
            psnr
        );
    }
    println!("Foreground is the union of reference/candidate pixels with alpha > 1/255.");
    println!("Full-frame mean MSE (useful for catching silhouette spill):");
    for (mode, samples) in &results {
        println!("  mode {}: {:.6e}", *mode as u32, mean(&samples.full_frame));
    }
    println!("High-frequency residual energy (sparkle/grain; lower is better):");
    for (mode, samples) in &results {
        let sparkle_mean = mean(&samples.sparkle);
        println!(
            "  mode {}: mean {:.6e}, variance {:.6e}",
            *mode as u32,
            sparkle_mean,
            variance(&samples.sparkle, sparkle_mean),
        );
    }
}

fn image_losses(
    reference: &[[f32; 4]],
    candidate: &[[f32; 4]],
    width: usize,
    height: usize,
) -> (f64, f64, f64) {
    assert_eq!(reference.len(), candidate.len());
    assert_eq!(reference.len(), width * height);
    let mut foreground_sum = 0.0f64;
    let mut foreground_values = 0usize;
    let mut full_sum = 0.0f64;
    for (reference, candidate) in reference.iter().zip(candidate) {
        let mut pixel_sum = 0.0f64;
        for channel in 0..4 {
            let difference = (reference[channel] - candidate[channel]) as f64;
            pixel_sum += difference * difference;
        }
        full_sum += pixel_sum;
        if reference[3] > 1.0 / 255.0 || candidate[3] > 1.0 / 255.0 {
            foreground_sum += pixel_sum;
            foreground_values += 4;
        }
    }
    let error_signal = |index: usize| {
        let r = reference[index];
        let c = candidate[index];
        0.2126 * (r[0] - c[0]) as f64
            + 0.7152 * (r[1] - c[1]) as f64
            + 0.0722 * (r[2] - c[2]) as f64
            + 0.5 * (r[3] - c[3]) as f64
    };
    let mut sparkle_sum = 0.0;
    let mut sparkle_edges = 0usize;
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            let error = error_signal(index);
            if x + 1 < width {
                let difference = error - error_signal(index + 1);
                sparkle_sum += difference * difference;
                sparkle_edges += 1;
            }
            if y + 1 < height {
                let difference = error - error_signal(index + width);
                sparkle_sum += difference * difference;
                sparkle_edges += 1;
            }
        }
    }

    (
        foreground_sum / foreground_values.max(1) as f64,
        full_sum / (reference.len().max(1) * 4) as f64,
        sparkle_sum / sparkle_edges.max(1) as f64,
    )
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

fn variance(values: &[f64], mean: f64) -> f64 {
    values
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / values.len().max(1) as f64
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn unit_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^= value >> 31;
        ((value >> 40) as f32) / ((1u32 << 24) as f32)
    }
}
