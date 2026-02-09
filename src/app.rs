use crate::camera::Camera;
use crate::renderer::{RenderMode, Renderer};
use crate::scene::Scene;
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
}

impl App {
    pub fn new() -> Self {
        Self {
            window: None,
            renderer: None,
            camera: Camera::new(16.0 / 9.0),
            scene: Scene::new(),
            last_frame: std::time::Instant::now(),
            scene_uploaded: false,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = Window::default_attributes()
            .with_title("WBOIT Demo - Press 1/2/3 to switch modes, A to toggle revealage, M to toggle meshes")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
            .with_transparent(true);

        let window = std::sync::Arc::new(event_loop.create_window(attrs).unwrap());
        let size = window.inner_size();
        self.camera.aspect = size.width as f32 / size.height as f32;

        let renderer = Renderer::new(window.clone());
        self.renderer = Some(renderer);
        self.window = Some(window);

        println!("Mode: Alpha Blend");
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                if let Some(renderer) = &mut self.renderer {
                    self.camera.aspect = new_size.width as f32 / new_size.height.max(1) as f32;
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
