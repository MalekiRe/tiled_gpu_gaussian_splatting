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

    let mut path = None;
    let mut benchmark = None;
    let mut seed = 1u64;
    let mut args = std::env::args().skip(1).peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--benchmark" => {
                let views = args
                    .next_if(|next| !next.starts_with('-'))
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(16);
                benchmark = Some(views);
            }
            "--seed" => {
                seed = args
                    .next()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("--seed requires an unsigned integer");
                        std::process::exit(2);
                    });
            }
            _ if arg.starts_with('-') => {
                eprintln!("Unknown option: {arg}");
                std::process::exit(2);
            }
            _ if path.is_none() => path = Some(PathBuf::from(arg)),
            _ => {
                eprintln!("Only one PLY path may be supplied");
                std::process::exit(2);
            }
        }
    }
    let benchmark = benchmark.map(|views| app::BenchmarkConfig { views, seed });
    if benchmark.is_some() && path.is_none() {
        eprintln!("--benchmark requires a splat PLY path");
        std::process::exit(2);
    }

    // With a PLY argument the demo renders that Gaussian splat scene through all three
    // transparency modes; mode 4 adds a baked directional prior to mode 3. Without a PLY
    // it falls back to the built-in quad/mesh scene.
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
    let mut app = app::App::new(splats, benchmark);
    event_loop.run_app(&mut app).unwrap();
}
