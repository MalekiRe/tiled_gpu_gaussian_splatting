use crate::camera::Camera;
use crate::pipeline::alpha_blend::AlphaBlendPipeline;
use crate::pipeline::histogram_wboit::HistogramWboitPipeline;
use crate::pipeline::naive_wboit::NaiveWboitPipeline;
use crate::scene::Scene;
use crate::vertex::{CameraUniform, HistogramParams, ObjectUniform};

const NUM_DEPTH_BINS: u32 = 64;
const TILE_SIZE: u32 = 16;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RenderMode {
    AlphaBlend = 1,
    NaiveWboit = 2,
    HistogramWboit = 3,
    GlobalHistogramWboit = 4,
}

impl RenderMode {
    pub fn name(&self) -> &'static str {
        match self {
            RenderMode::AlphaBlend => "Alpha Blend",
            RenderMode::NaiveWboit => "Naive WBOIT",
            RenderMode::HistogramWboit => "Histogram-Equalized WBOIT (per-tile)",
            RenderMode::GlobalHistogramWboit => "Histogram-Equalized WBOIT (global)",
        }
    }
}

struct GpuMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    object_buffer: wgpu::Buffer,
    object_bind_group: wgpu::BindGroup,
}

pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub mode: RenderMode,

    // Shared resources
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    depth_texture_view: wgpu::TextureView,
    gpu_meshes: Vec<GpuMesh>,

    // Pipelines
    alpha_blend: AlphaBlendPipeline,
    naive_wboit: NaiveWboitPipeline,
    histogram_wboit: HistogramWboitPipeline,

    // WBOIT textures
    accum_texture_view: wgpu::TextureView,
    revealage_texture_view: wgpu::TextureView,
    wboit_composite_bind_group: wgpu::BindGroup,

    // Histogram WBOIT resources
    histogram_buffer: wgpu::Buffer,
    cdf_buffer: wgpu::Buffer,
    histo_params_buffer: wgpu::Buffer,
    histo_accum_bind_group: wgpu::BindGroup,
    histo_composite_tex_bind_group: wgpu::BindGroup,
    histo_composite_buf_bind_group: wgpu::BindGroup,
    histo_params: HistogramParams,

    // Bind group layouts (needed for recreation)
    #[allow(dead_code)]
    camera_bgl: wgpu::BindGroupLayout,
    object_bgl: wgpu::BindGroupLayout,
}

impl Renderer {
    pub fn new(window: std::sync::Arc<winit::window::Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance.create_surface(window).unwrap();

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("Failed to find adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .expect("Failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: if surface_caps
                .alpha_modes
                .contains(&wgpu::CompositeAlphaMode::PreMultiplied)
            {
                wgpu::CompositeAlphaMode::PreMultiplied
            } else if surface_caps
                .alpha_modes
                .contains(&wgpu::CompositeAlphaMode::PostMultiplied)
            {
                wgpu::CompositeAlphaMode::PostMultiplied
            } else {
                surface_caps.alpha_modes[0]
            },
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        // Bind group layouts
        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let object_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("object bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Camera buffer
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera bind group"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // Pipelines
        let alpha_blend =
            AlphaBlendPipeline::new(&device, surface_format, &camera_bgl, &object_bgl);
        let naive_wboit =
            NaiveWboitPipeline::new(&device, surface_format, &camera_bgl, &object_bgl);
        let histogram_wboit =
            HistogramWboitPipeline::new(&device, surface_format, &camera_bgl, &object_bgl);

        // Depth texture
        let depth_texture_view =
            create_depth_texture(&device, surface_config.width, surface_config.height);

        // WBOIT textures
        let (accum_texture_view, revealage_texture_view) =
            create_wboit_textures(&device, surface_config.width, surface_config.height);

        let wboit_composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wboit composite bg"),
            layout: &naive_wboit.composite_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&accum_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&revealage_texture_view),
                },
            ],
        });

        // Histogram resources
        let histo_params = HistogramParams {
            tile_count_x: surface_config.width.div_ceil(TILE_SIZE),
            tile_count_y: surface_config.height.div_ceil(TILE_SIZE),
            num_bins: NUM_DEPTH_BINS,
            depth_range: 50.0,
        };

        let total_bins = histo_params.tile_count_x * histo_params.tile_count_y * NUM_DEPTH_BINS;
        let histogram_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("histogram buffer"),
            size: (total_bins * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Initialize CDF with linear fallback
        let cdf_init: Vec<f32> = (0..total_bins)
            .map(|i| {
                let bin = i % NUM_DEPTH_BINS;
                bin as f32 / NUM_DEPTH_BINS as f32
            })
            .collect();
        let cdf_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cdf buffer"),
            size: (total_bins * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&cdf_buffer, 0, bytemuck::cast_slice(&cdf_init));

        let histo_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("histo params buffer"),
            size: std::mem::size_of::<HistogramParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&histo_params_buffer, 0, bytemuck::bytes_of(&histo_params));

        let histo_accum_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("histo accum bg"),
            layout: &histogram_wboit.histo_accum_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: histogram_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cdf_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: histo_params_buffer.as_entire_binding(),
                },
            ],
        });

        let histo_composite_tex_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("histo composite tex bg"),
            layout: &histogram_wboit.histo_composite_tex_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&accum_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&revealage_texture_view),
                },
            ],
        });

        let histo_composite_buf_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("histo composite buf bg"),
            layout: &histogram_wboit.histo_composite_buf_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: histogram_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cdf_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: histo_params_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            device,
            queue,
            surface,
            surface_config,
            mode: RenderMode::AlphaBlend,
            camera_buffer,
            camera_bind_group,
            depth_texture_view,
            gpu_meshes: Vec::new(),
            alpha_blend,
            naive_wboit,
            histogram_wboit,
            accum_texture_view,
            revealage_texture_view,
            wboit_composite_bind_group,
            histogram_buffer,
            cdf_buffer,
            histo_params_buffer,
            histo_accum_bind_group,
            histo_composite_tex_bind_group,
            histo_composite_buf_bind_group,
            histo_params,
            camera_bgl,
            object_bgl,
        }
    }

    pub fn upload_scene(&mut self, scene: &Scene) {
        self.gpu_meshes.clear();
        for obj in &scene.objects {
            let vertex_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vertex buffer"),
                size: (obj.mesh.vertices.len() * std::mem::size_of::<crate::vertex::Vertex>())
                    as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue
                .write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&obj.mesh.vertices));

            let index_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("index buffer"),
                size: (obj.mesh.indices.len() * 2) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue
                .write_buffer(&index_buffer, 0, bytemuck::cast_slice(&obj.mesh.indices));

            let object_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("object buffer"),
                size: std::mem::size_of::<ObjectUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let object_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("object bg"),
                layout: &self.object_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: object_buffer.as_entire_binding(),
                }],
            });

            self.gpu_meshes.push(GpuMesh {
                vertex_buffer,
                index_buffer,
                num_indices: obj.mesh.indices.len() as u32,
                object_buffer,
                object_bind_group,
            });
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);

        self.depth_texture_view = create_depth_texture(&self.device, width, height);

        let (accum, revealage) = create_wboit_textures(&self.device, width, height);
        self.accum_texture_view = accum;
        self.revealage_texture_view = revealage;

        // Recreate WBOIT composite bind group
        self.wboit_composite_bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wboit composite bg"),
                layout: &self.naive_wboit.composite_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.accum_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.revealage_texture_view),
                    },
                ],
            });

        // Recreate histogram resources
        self.histo_params = HistogramParams {
            tile_count_x: width.div_ceil(TILE_SIZE),
            tile_count_y: height.div_ceil(TILE_SIZE),
            num_bins: NUM_DEPTH_BINS,
            depth_range: 50.0,
        };

        let total_bins =
            self.histo_params.tile_count_x * self.histo_params.tile_count_y * NUM_DEPTH_BINS;

        self.histogram_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("histogram buffer"),
            size: (total_bins * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cdf_init: Vec<f32> = (0..total_bins)
            .map(|i| {
                let bin = i % NUM_DEPTH_BINS;
                bin as f32 / NUM_DEPTH_BINS as f32
            })
            .collect();
        self.cdf_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cdf buffer"),
            size: (total_bins * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&self.cdf_buffer, 0, bytemuck::cast_slice(&cdf_init));

        self.queue.write_buffer(
            &self.histo_params_buffer,
            0,
            bytemuck::bytes_of(&self.histo_params),
        );

        // Recreate bind groups
        self.histo_accum_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("histo accum bg"),
            layout: &self.histogram_wboit.histo_accum_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.histogram_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.cdf_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.histo_params_buffer.as_entire_binding(),
                },
            ],
        });

        self.histo_composite_tex_bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("histo composite tex bg"),
                layout: &self.histogram_wboit.histo_composite_tex_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.accum_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.revealage_texture_view),
                    },
                ],
            });

        self.histo_composite_buf_bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("histo composite buf bg"),
                layout: &self.histogram_wboit.histo_composite_buf_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.histogram_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.cdf_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.histo_params_buffer.as_entire_binding(),
                    },
                ],
            });
    }

    pub fn render(&mut self, camera: &Camera, scene: &Scene) {
        // Update camera
        let cam_uniform = camera.uniform();
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&cam_uniform));

        // Update object transforms
        let mut visible: Vec<usize> = scene
            .objects
            .iter()
            .enumerate()
            .filter(|(_, o)| {
                if o.is_extra_mesh {
                    scene.show_meshes
                } else {
                    true
                }
            })
            .map(|(i, _)| i)
            .collect();

        // Sort back-to-front for alpha blend mode
        if self.mode == RenderMode::AlphaBlend {
            let eye = camera.eye();
            visible.sort_by(|&a, &b| {
                let pos_a = scene.objects[a].transform.col(3).truncate();
                let pos_b = scene.objects[b].transform.col(3).truncate();
                let dist_a = (pos_a - eye).length_squared();
                let dist_b = (pos_b - eye).length_squared();
                dist_b
                    .partial_cmp(&dist_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        for &idx in &visible {
            let mut uniform = scene.objects[idx].uniform();
            if scene.force_opaque {
                uniform.color[3] = 1.0 / scene.objects[idx].original_alpha;
            }
            self.queue.write_buffer(
                &self.gpu_meshes[idx].object_buffer,
                0,
                bytemuck::bytes_of(&uniform),
            );
        }

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.surface_config);
                return;
            }
            Err(e) => {
                log::error!("Surface error: {:?}", e);
                return;
            }
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render encoder"),
            });

        match self.mode {
            RenderMode::AlphaBlend => {
                self.render_alpha_blend(&mut encoder, &view, &visible);
            }
            RenderMode::NaiveWboit => {
                self.render_naive_wboit(&mut encoder, &view, &visible);
            }
            RenderMode::HistogramWboit | RenderMode::GlobalHistogramWboit => {
                self.render_histogram_wboit(&mut encoder, &view, &visible);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }

    fn render_alpha_blend(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        visible: &[usize],
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("alpha blend pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_texture_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });

        pass.set_pipeline(&self.alpha_blend.pipeline);
        pass.set_bind_group(0, &self.camera_bind_group, &[]);

        for &idx in visible {
            let mesh = &self.gpu_meshes[idx];
            pass.set_bind_group(1, &mesh.object_bind_group, &[]);
            pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..mesh.num_indices, 0, 0..1);
        }
    }

    fn render_naive_wboit(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        visible: &[usize],
    ) {
        // Pass 1: Accumulation
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("wboit accum pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.accum_texture_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.revealage_texture_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 1.0,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_pipeline(&self.naive_wboit.accum_pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);

            for &idx in visible {
                let mesh = &self.gpu_meshes[idx];
                pass.set_bind_group(1, &mesh.object_bind_group, &[]);
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                pass.draw_indexed(0..mesh.num_indices, 0, 0..1);
            }
        }

        // Pass 2: Composite
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("wboit composite pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            pass.set_pipeline(&self.naive_wboit.composite_pipeline);
            pass.set_bind_group(0, &self.wboit_composite_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    fn render_histogram_wboit(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        visible: &[usize],
    ) {
        let is_global = self.mode == RenderMode::GlobalHistogramWboit;
        let accum_pipeline = if is_global {
            &self.histogram_wboit.global_accum_pipeline
        } else {
            &self.histogram_wboit.accum_pipeline
        };
        let composite_pipeline = if is_global {
            &self.histogram_wboit.global_composite_pipeline
        } else {
            &self.histogram_wboit.composite_pipeline
        };

        // Pass 1: Accumulation + histogram recording
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("histo accum pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.accum_texture_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.revealage_texture_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 1.0,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_pipeline(accum_pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_bind_group(2, &self.histo_accum_bind_group, &[]);

            for &idx in visible {
                let mesh = &self.gpu_meshes[idx];
                pass.set_bind_group(1, &mesh.object_bind_group, &[]);
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                pass.draw_indexed(0..mesh.num_indices, 0, 0..1);
            }
        }

        // Pass 2: Composite + CDF build
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("histo composite pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            pass.set_pipeline(composite_pipeline);
            pass.set_bind_group(0, &self.histo_composite_tex_bind_group, &[]);
            pass.set_bind_group(1, &self.histo_composite_buf_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

fn create_wboit_textures(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::TextureView, wgpu::TextureView) {
    let accum = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("accum texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let revealage = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("revealage texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    (
        accum.create_view(&wgpu::TextureViewDescriptor::default()),
        revealage.create_view(&wgpu::TextureViewDescriptor::default()),
    )
}
