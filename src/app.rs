use crate::camera::Camera;
use crate::renderer::{RenderMode, Renderer};
use crate::scene::Scene;
use crate::splats::{SplatScene, Sorter};
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
}

/// Render-cap steps cycled with `C`, as a fraction of the loaded scene.
const CAP_FRACTIONS: [f32; 4] = [1.0, 0.5, 0.25, 0.1];

impl App {
    pub fn new(splats: Option<SplatScene>) -> Self {
        let sorter = splats
            .as_ref()
            .map(|s| Sorter::new(std::sync::Arc::clone(&s.positions)));
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
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let title = if self.splats.is_some() {
            "3DGS WBOIT Demo - 1/2/3/4/5/6/7 modes, A revealage, C cap, [ ] size, R reset"
        } else {
            "WBOIT Demo - Press 1/2/3/4/5/6/7 to switch modes, A revealage, M meshes"
        };
        let attrs = Window::default_attributes()
            .with_title(title)
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
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
                            "a" | "A" => {
                                renderer.use_revealage = !renderer.use_revealage;
                                println!(
                                    "Revealage: {} (alpha computed via {})",
                                    if renderer.use_revealage { "ON" } else { "OFF" },
                                    if renderer.use_revealage {
                                        "revealage buffer"
                                    } else {
                                        "exp(-accum.a) approximation"
                                    }
                                );
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
