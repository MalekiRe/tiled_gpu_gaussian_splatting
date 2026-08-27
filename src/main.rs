mod app;
mod camera;
mod mesh;
mod pipeline;
mod ply;
mod renderer;
mod scene;
mod splats;
mod vertex;

use std::path::PathBuf;

fn main() {
    env_logger::init();

    // With a PLY argument the demo renders that Gaussian splat scene through all three
    // transparency modes; without one it falls back to the built-in quad/mesh scene.
    let path: Option<PathBuf> = std::env::args().nth(1).map(PathBuf::from);
    let splats = match path {
        Some(path) => {
            let started = std::time::Instant::now();
            let data = match ply::load(&path) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("Failed to load {}: {e}", path.display());
                    std::process::exit(1);
                }
            };
            println!(
                "Loaded {} splats from {} ({} SH bands) in {:.2}s",
                data.len(),
                path.display(),
                data.sh_degree,
                started.elapsed().as_secs_f32(),
            );
            let scene = splats::SplatScene::from_ply(data);
            println!(
                "Scene centre {:?}, radius {:.2}",
                scene.center.to_array(),
                scene.radius
            );
            Some(scene)
        }
        None => None,
    };

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = app::App::new(splats);
    event_loop.run_app(&mut app).unwrap();
}
