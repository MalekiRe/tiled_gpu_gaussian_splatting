use crate::camera::Camera;
use crate::pipeline::alpha_blend::AlphaBlendPipeline;
use crate::pipeline::histogram_wboit::HistogramWboitPipeline;
use crate::pipeline::naive_wboit::NaiveWboitPipeline;
use crate::pipeline::splat::SplatPipelines;
use crate::scene::Scene;
use crate::splats::{
    DIRECTIONAL_DEPTH_BINS, DirectionalHistogramPrior, HQ_SPATIAL_PRIOR_HEIGHT,
    HQ_SPATIAL_PRIOR_WIDTH, HighQualitySpatialDirectionalPrior,
    SpatialDirectionalHistogramPrior, SplatScene,
};
use crate::vertex::{
    CameraUniform, DirectionalPriorParams, HistogramParams, ObjectUniform, SplatParams,
};

const NUM_DEPTH_BINS: u32 = 64;
const TILE_SIZE: u32 = 32;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RenderMode {
    AlphaBlend = 1,
    NaiveWboit = 2,
    HistogramWboit = 3,
    DirectionalHistogramWboit = 4,
    SpatialBakedHistogramWboit = 5,
    HighQualitySpatialBakedWboit = 6,
    DoubleSampleFrontWboit = 7,
}

impl RenderMode {
    pub fn name(&self) -> &'static str {
        match self {
            RenderMode::AlphaBlend => "Alpha Blend",
            RenderMode::NaiveWboit => "Naive WBOIT",
            RenderMode::HistogramWboit => "Histogram-Equalized WBOIT (tiled)",
            RenderMode::DirectionalHistogramWboit => {
                "Directional-Prior Histogram WBOIT (64 baked views)"
            }
            RenderMode::SpatialBakedHistogramWboit => {
                "Spatial Baked Histogram WBOIT (atomics-free)"
            }
            RenderMode::HighQualitySpatialBakedWboit => {
                "HQ Spatial Bake + Realtime Front Feature"
            }
            RenderMode::DoubleSampleFrontWboit => {
                "HQ Front Feature + Two-Sample Confidence"
            }
        }
    }
}

/// GPU-resident splat scene plus the knobs the UI can turn.
struct SplatGpuState {
    /// Held only to keep the allocation alive alongside the bind group.
    _sh_buffer: wgpu::Buffer,
    order_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    total: u32,
    draw_count: u32,
    sh_degree: u32,
    splat_scale: f32,
    scene_radius: f32,
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
    pub use_revealage: bool,

    // Shared resources
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    depth_texture_view: wgpu::TextureView,
    gpu_meshes: Vec<GpuMesh>,

    // Pipelines
    alpha_blend: AlphaBlendPipeline,
    naive_wboit: NaiveWboitPipeline,
    histogram_wboit: HistogramWboitPipeline,
    splat_pipelines: SplatPipelines,

    /// Present only when a PLY was supplied on the command line; when it is, splats
    /// replace the quad/mesh scene entirely.
    splats: Option<SplatGpuState>,

    // WBOIT textures (double-buffered revealage for transmittance feedback)
    accum_texture_view: wgpu::TextureView,
    revealage_views: [wgpu::TextureView; 2],
    front_feature_view: wgpu::TextureView,
    front_feature_depth_view: wgpu::TextureView,
    front_feature_alt_view: wgpu::TextureView,
    front_feature_alt_depth_view: wgpu::TextureView,
    frame_index: usize,

    // Double-buffered bind groups indexed by frame_index:
    // [i] renders to revealage_views[i], reads prev from revealage_views[1-i]
    wboit_composite_bind_groups: [wgpu::BindGroup; 2],
    histo_accum_bind_groups: [wgpu::BindGroup; 2],
    histo_composite_tex_bind_groups: [wgpu::BindGroup; 2],

    // Revealage flag uniform
    revealage_flag_buffer: wgpu::Buffer,
    naive_revealage_bind_group: wgpu::BindGroup,

    // Histogram WBOIT resources (tiled)
    histogram_buffer: wgpu::Buffer,
    cdf_texture_view: wgpu::TextureView,
    cdf_sampler: wgpu::Sampler,
    histo_params_buffer: wgpu::Buffer,
    cdf_build_bind_group: wgpu::BindGroup,
    histo_composite_flag_bind_group: wgpu::BindGroup,
    histo_params: HistogramParams,
    directional_prior: Option<DirectionalHistogramPrior>,
    spatial_directional_prior: Option<SpatialDirectionalHistogramPrior>,
    high_quality_spatial_prior: Option<HighQualitySpatialDirectionalPrior>,
    directional_prior_buffer: wgpu::Buffer,
    directional_prior_params_buffer: wgpu::Buffer,

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

        // Big splat scenes need storage buffers well past the conservative defaults.
        let adapter_limits = adapter.limits();
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size;
        limits.max_buffer_size = adapter_limits.max_buffer_size;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("device"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
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
        let splat_pipelines = SplatPipelines::new(
            &device,
            surface_format,
            &camera_bgl,
            &histogram_wboit.histo_accum_bgl,
        );

        // Revealage flag buffer (u32: 0 = use exp approximation, 1 = use revealage)
        let revealage_flag_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("revealage flag buffer"),
            size: 4,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Depth texture
        let depth_texture_view =
            create_depth_texture(&device, surface_config.width, surface_config.height);

        // WBOIT textures (double-buffered revealage)
        let (accum_texture_view, revealage_views) =
            create_wboit_textures(&device, surface_config.width, surface_config.height);
        let (front_feature_view, front_feature_depth_view) = create_front_feature_textures(
            &device,
            surface_config.width,
            surface_config.height,
        );
        let (front_feature_alt_view, front_feature_alt_depth_view) =
            create_front_feature_textures(
                &device,
                surface_config.width,
                surface_config.height,
            );

        let wboit_composite_bind_groups = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wboit composite bg"),
                layout: &naive_wboit.composite_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&accum_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&revealage_views[i]),
                    },
                ],
            })
        });

        let naive_revealage_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("naive revealage flag bg"),
            layout: &naive_wboit.flag_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: revealage_flag_buffer.as_entire_binding(),
            }],
        });

        // Tiled histogram resources
        let tiles_x = (surface_config.width + TILE_SIZE - 1) / TILE_SIZE;
        let tiles_y = (surface_config.height + TILE_SIZE - 1) / TILE_SIZE;

        let histo_params = HistogramParams {
            tile_count_x: tiles_x,
            tile_count_y: tiles_y,
            num_bins: NUM_DEPTH_BINS,
            tile_size: TILE_SIZE,
        };

        let histogram_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("histogram buffer"),
            size: (tiles_x as u64) * (tiles_y as u64) * (NUM_DEPTH_BINS as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let directional_prior_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("directional histogram prior buffer"),
            size: (HQ_SPATIAL_PRIOR_WIDTH
                * HQ_SPATIAL_PRIOR_HEIGHT
                * DIRECTIONAL_DEPTH_BINS
                * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_prior = [1.0 / DIRECTIONAL_DEPTH_BINS as f32; DIRECTIONAL_DEPTH_BINS];
        queue.write_buffer(
            &directional_prior_buffer,
            0,
            bytemuck::cast_slice(&uniform_prior),
        );

        let directional_prior_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("directional histogram prior params"),
            size: std::mem::size_of::<DirectionalPriorParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &directional_prior_params_buffer,
            0,
            bytemuck::bytes_of(&DirectionalPriorParams {
                mix_factor: 0.0,
                enabled: 0,
                _padding: [0; 2],
            }),
        );

        let (cdf_texture, cdf_texture_view) =
            create_cdf_texture(&device, tiles_x, tiles_y, NUM_DEPTH_BINS);
        let _ = cdf_texture; // view keeps texture alive via Arc internally

        let cdf_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("cdf sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let histo_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("histo params buffer"),
            size: std::mem::size_of::<HistogramParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&histo_params_buffer, 0, bytemuck::bytes_of(&histo_params));

        // histo_accum_bind_groups[i]: used when frame_index=i, reads prev revealage from [1-i]
        let histo_accum_bind_groups = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("histo accum bg"),
                layout: &histogram_wboit.histo_accum_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: histogram_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&cdf_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&cdf_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: histo_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&revealage_views[1 - i]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&front_feature_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(&front_feature_alt_view),
                    },
                ],
            })
        });

        // histo_composite_tex_bind_groups[i]: reads current frame's accum + revealage[i]
        let histo_composite_tex_bind_groups = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("histo composite tex bg"),
                layout: &histogram_wboit.histo_composite_tex_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&accum_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&revealage_views[i]),
                    },
                ],
            })
        });

        let cdf_build_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cdf build bg"),
            layout: &histogram_wboit.cdf_build_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: histogram_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&cdf_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: histo_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: directional_prior_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: directional_prior_params_buffer.as_entire_binding(),
                },
            ],
        });

        let histo_composite_flag_bind_group =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("histo composite flag bg"),
                layout: &histogram_wboit.flag_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: revealage_flag_buffer.as_entire_binding(),
                }],
            });

        Self {
            device,
            queue,
            surface,
            surface_config,
            mode: RenderMode::AlphaBlend,
            use_revealage: true,
            camera_buffer,
            camera_bind_group,
            depth_texture_view,
            gpu_meshes: Vec::new(),
            alpha_blend,
            naive_wboit,
            histogram_wboit,
            splat_pipelines,
            splats: None,
            accum_texture_view,
            revealage_views,
            front_feature_view,
            front_feature_depth_view,
            front_feature_alt_view,
            front_feature_alt_depth_view,
            frame_index: 0,
            wboit_composite_bind_groups,
            histo_accum_bind_groups,
            histo_composite_tex_bind_groups,
            revealage_flag_buffer,
            naive_revealage_bind_group,
            histogram_buffer,
            cdf_texture_view,
            cdf_sampler,
            histo_params_buffer,
            cdf_build_bind_group,
            histo_composite_flag_bind_group,
            histo_params,
            directional_prior: None,
            spatial_directional_prior: None,
            high_quality_spatial_prior: None,
            directional_prior_buffer,
            directional_prior_params_buffer,
            camera_bgl,
            object_bgl,
        }
    }

    /// Move a parsed splat scene onto the GPU. From here on, `render` draws splats.
    pub fn upload_splats(&mut self, scene: &SplatScene) {
        let total = scene.len() as u32;
        self.directional_prior = Some(scene.directional_prior.clone());
        self.spatial_directional_prior = Some(scene.spatial_directional_prior.clone());
        self.high_quality_spatial_prior = Some(scene.high_quality_spatial_prior.clone());

        let splat_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splat buffer"),
            size: (scene.gpu.len() * std::mem::size_of::<crate::splats::SplatGpu>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&splat_buffer, 0, bytemuck::cast_slice(&scene.gpu));

        // The binding must exist even for files without higher SH bands; a single dummy
        // float keeps the layout uniform across both cases.
        let sh_src: &[f32] = if scene.sh.is_empty() {
            &[0.0]
        } else {
            &scene.sh
        };
        let sh_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splat sh buffer"),
            size: (sh_src.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&sh_buffer, 0, bytemuck::cast_slice(sh_src));

        // Identity order until the sort thread reports in.
        let identity: Vec<u32> = (0..total).collect();
        let order_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splat order buffer"),
            size: (identity.len().max(1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&order_buffer, 0, bytemuck::cast_slice(&identity));

        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splat params buffer"),
            size: std::mem::size_of::<SplatParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("splat bg"),
            layout: &self.splat_pipelines.splat_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: splat_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sh_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: order_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        self.splats = Some(SplatGpuState {
            _sh_buffer: sh_buffer,
            order_buffer,
            params_buffer,
            bind_group,
            total,
            draw_count: total,
            sh_degree: scene.sh_degree,
            splat_scale: 1.0,
            scene_radius: scene.radius,
        });
    }

    pub fn has_splats(&self) -> bool {
        self.splats.is_some()
    }

    /// How many splats are drawn this frame; also what the sorter is asked to order.
    pub fn splat_draw_count(&self) -> usize {
        self.splats.as_ref().map_or(0, |s| s.draw_count as usize)
    }

    pub fn upload_splat_order(&mut self, order: &[u32]) {
        if let Some(sp) = &self.splats
            && !order.is_empty()
        {
            self.queue
                .write_buffer(&sp.order_buffer, 0, bytemuck::cast_slice(order));
        }
    }

    /// Set the render cap as a fraction of the scene. Splats are stored most-important
    /// first, so a prefix is the best subset of that size.
    pub fn set_splat_fraction(&mut self, fraction: f32) -> Option<(u32, u32)> {
        let sp = self.splats.as_mut()?;
        sp.draw_count = ((sp.total as f32 * fraction) as u32).clamp(1, sp.total);
        Some((sp.draw_count, sp.total))
    }

    pub fn adjust_splat_scale(&mut self, factor: f32) -> Option<f32> {
        let sp = self.splats.as_mut()?;
        sp.splat_scale = (sp.splat_scale * factor).clamp(0.05, 8.0);
        Some(sp.splat_scale)
    }

    /// Issue the draws for whichever scene is loaded.
    fn draw_scene(&self, pass: &mut wgpu::RenderPass<'_>, visible: &[usize], mode: RenderMode) {
        pass.set_bind_group(0, &self.camera_bind_group, &[]);

        if let Some(sp) = &self.splats {
            let pipeline = match mode {
                RenderMode::AlphaBlend => &self.splat_pipelines.alpha_pipeline,
                RenderMode::NaiveWboit => &self.splat_pipelines.wboit_pipeline,
                RenderMode::HistogramWboit | RenderMode::DirectionalHistogramWboit => {
                    &self.splat_pipelines.histo_pipeline
                }
                RenderMode::SpatialBakedHistogramWboit => {
                    &self.splat_pipelines.baked_histo_pipeline
                }
                RenderMode::HighQualitySpatialBakedWboit => {
                    &self.splat_pipelines.front_weighted_histo_pipeline
                }
                RenderMode::DoubleSampleFrontWboit => {
                    &self.splat_pipelines.double_front_histo_pipeline
                }
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(1, &sp.bind_group, &[]);
            if matches!(
                mode,
                RenderMode::HistogramWboit | RenderMode::DirectionalHistogramWboit
                    | RenderMode::SpatialBakedHistogramWboit
                    | RenderMode::HighQualitySpatialBakedWboit
                    | RenderMode::DoubleSampleFrontWboit
            ) {
                pass.set_bind_group(2, &self.histo_accum_bind_groups[self.frame_index], &[]);
            }
            // One instanced quad per splat, expanded to the projected 3-sigma extent.
            pass.draw(0..4, 0..sp.draw_count);
            return;
        }

        let pipeline = match mode {
            RenderMode::AlphaBlend => &self.alpha_blend.pipeline,
            RenderMode::NaiveWboit => &self.naive_wboit.accum_pipeline,
            RenderMode::HistogramWboit | RenderMode::DirectionalHistogramWboit => {
                &self.histogram_wboit.accum_pipeline
            }
            RenderMode::SpatialBakedHistogramWboit => &self.histogram_wboit.accum_pipeline,
            RenderMode::HighQualitySpatialBakedWboit => &self.histogram_wboit.accum_pipeline,
            RenderMode::DoubleSampleFrontWboit => &self.histogram_wboit.accum_pipeline,
        };
        pass.set_pipeline(pipeline);
        if matches!(
            mode,
            RenderMode::HistogramWboit | RenderMode::DirectionalHistogramWboit
                | RenderMode::SpatialBakedHistogramWboit
                | RenderMode::HighQualitySpatialBakedWboit
                | RenderMode::DoubleSampleFrontWboit
        ) {
            pass.set_bind_group(2, &self.histo_accum_bind_groups[self.frame_index], &[]);
        }

        for &idx in visible {
            let mesh = &self.gpu_meshes[idx];
            pass.set_bind_group(1, &mesh.object_bind_group, &[]);
            pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..mesh.num_indices, 0, 0..1);
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

        let (accum_view, revealage_views) = create_wboit_textures(&self.device, width, height);
        self.accum_texture_view = accum_view;
        self.revealage_views = revealage_views;
        let (front_feature_view, front_feature_depth_view) =
            create_front_feature_textures(&self.device, width, height);
        self.front_feature_view = front_feature_view;
        self.front_feature_depth_view = front_feature_depth_view;
        let (front_feature_alt_view, front_feature_alt_depth_view) =
            create_front_feature_textures(&self.device, width, height);
        self.front_feature_alt_view = front_feature_alt_view;
        self.front_feature_alt_depth_view = front_feature_alt_depth_view;

        // Recreate double-buffered bind groups
        self.wboit_composite_bind_groups = std::array::from_fn(|i| {
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
                        resource: wgpu::BindingResource::TextureView(&self.revealage_views[i]),
                    },
                ],
            })
        });

        // Recreate tiled histogram resources
        let tiles_x = (width + TILE_SIZE - 1) / TILE_SIZE;
        let tiles_y = (height + TILE_SIZE - 1) / TILE_SIZE;

        self.histo_params = HistogramParams {
            tile_count_x: tiles_x,
            tile_count_y: tiles_y,
            num_bins: NUM_DEPTH_BINS,
            tile_size: TILE_SIZE,
        };

        self.histogram_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("histogram buffer"),
            size: (tiles_x as u64) * (tiles_y as u64) * (NUM_DEPTH_BINS as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (cdf_texture, cdf_texture_view) =
            create_cdf_texture(&self.device, tiles_x, tiles_y, NUM_DEPTH_BINS);
        let _ = cdf_texture;
        self.cdf_texture_view = cdf_texture_view;

        self.queue.write_buffer(
            &self.histo_params_buffer,
            0,
            bytemuck::bytes_of(&self.histo_params),
        );

        self.histo_accum_bind_groups = std::array::from_fn(|i| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("histo accum bg"),
                layout: &self.histogram_wboit.histo_accum_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.histogram_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.cdf_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.cdf_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.histo_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&self.revealage_views[1 - i]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&self.front_feature_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(
                            &self.front_feature_alt_view,
                        ),
                    },
                ],
            })
        });

        self.histo_composite_tex_bind_groups = std::array::from_fn(|i| {
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
                        resource: wgpu::BindingResource::TextureView(&self.revealage_views[i]),
                    },
                ],
            })
        });

        self.cdf_build_bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cdf build bg"),
                layout: &self.histogram_wboit.cdf_build_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.histogram_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.cdf_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.histo_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.directional_prior_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.directional_prior_params_buffer.as_entire_binding(),
                    },
                ],
            });
    }

    pub fn render(&mut self, camera: &Camera, scene: &Scene) {
        // Update camera
        let cam_uniform = camera.uniform();
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&cam_uniform));

        let use_directional_prior =
            self.mode == RenderMode::DirectionalHistogramWboit && self.directional_prior.is_some();
        if use_directional_prior {
            let prior = self.directional_prior.as_ref().unwrap().sample_for_camera(
                camera.forward(),
                camera.distance,
                camera.near,
                camera.far,
            );
            self.queue.write_buffer(
                &self.directional_prior_buffer,
                0,
                bytemuck::cast_slice(&prior),
            );
        }
        let use_spatial_prior = self.mode == RenderMode::SpatialBakedHistogramWboit
            && self.spatial_directional_prior.is_some();
        if use_spatial_prior {
            let prior = self
                .spatial_directional_prior
                .as_ref()
                .unwrap()
                .sample_for_camera(
                    camera.forward(),
                    camera.distance,
                    camera.near,
                    camera.far,
                );
            self.queue.write_buffer(
                &self.directional_prior_buffer,
                0,
                bytemuck::cast_slice(&prior),
            );
        }
        let use_high_quality_spatial_prior = matches!(
            self.mode,
            RenderMode::HighQualitySpatialBakedWboit | RenderMode::DoubleSampleFrontWboit
        ) && self.high_quality_spatial_prior.is_some();
        if use_high_quality_spatial_prior {
            let prior = self
                .high_quality_spatial_prior
                .as_ref()
                .unwrap()
                .sample_for_camera(
                    camera.forward(),
                    camera.distance,
                    camera.near,
                    camera.far,
                );
            self.queue.write_buffer(
                &self.directional_prior_buffer,
                0,
                bytemuck::cast_slice(&prior),
            );
        }
        self.queue.write_buffer(
            &self.directional_prior_params_buffer,
            0,
            bytemuck::bytes_of(&DirectionalPriorParams {
                mix_factor: if use_directional_prior { 0.75 } else { 0.0 },
                enabled: u32::from(use_directional_prior),
                _padding: [0; 2],
            }),
        );

        // Update revealage flag
        let flag: u32 = if self.use_revealage { 1 } else { 0 };
        self.queue
            .write_buffer(&self.revealage_flag_buffer, 0, bytemuck::bytes_of(&flag));

        if let Some(sp) = &self.splats {
            let params = SplatParams {
                count: sp.draw_count,
                sh_degree: sp.sh_degree,
                splat_scale: sp.splat_scale,
                scene_radius: sp.scene_radius,
            };
            self.queue
                .write_buffer(&sp.params_buffer, 0, bytemuck::bytes_of(&params));
        }

        // Update object transforms (mesh scene only)
        let mut visible: Vec<usize> = if self.splats.is_some() {
            Vec::new()
        } else {
            scene
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
            .collect()
        };

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
            RenderMode::HistogramWboit => {
                self.render_histogram_wboit(&mut encoder, &view, &visible, 0);
            }
            RenderMode::DirectionalHistogramWboit => {
                self.render_histogram_wboit(&mut encoder, &view, &visible, 0);
            }
            RenderMode::SpatialBakedHistogramWboit => {
                self.render_histogram_wboit(
                    &mut encoder,
                    &view,
                    &visible,
                    if self.splats.is_some() { 1 } else { 0 },
                );
            }
            RenderMode::HighQualitySpatialBakedWboit => {
                self.render_histogram_wboit(
                    &mut encoder,
                    &view,
                    &visible,
                    if self.splats.is_some() { 2 } else { 0 },
                );
            }
            RenderMode::DoubleSampleFrontWboit => {
                self.render_histogram_wboit(
                    &mut encoder,
                    &view,
                    &visible,
                    if self.splats.is_some() { 3 } else { 0 },
                );
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        // Flip double buffer index
        self.frame_index = 1 - self.frame_index;
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

        self.draw_scene(&mut pass, visible, RenderMode::AlphaBlend);
    }

    fn render_naive_wboit(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        visible: &[usize],
    ) {
        let fi = self.frame_index;

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
                        view: &self.revealage_views[fi],
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

            self.draw_scene(&mut pass, visible, RenderMode::NaiveWboit);
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
            pass.set_bind_group(0, &self.wboit_composite_bind_groups[fi], &[]);
            pass.set_bind_group(1, &self.naive_revealage_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    fn render_histogram_wboit(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        visible: &[usize],
        spatial_baked: u32,
    ) {
        let fi = self.frame_index;

        // Mode 6 first builds a per-pixel stochastic front surface. Ordinary depth
        // testing chooses the closest alpha-surviving splat, so this stays atomics-free
        // and does not depend on draw order.
        if spatial_baked >= 2
            && let Some(sp) = &self.splats
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("splat front feature pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.front_feature_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.front_feature_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&self.splat_pipelines.front_feature_pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_bind_group(1, &sp.bind_group, &[]);
            pass.draw(0..4, 0..sp.draw_count);
        }

        if spatial_baked == 3
            && let Some(sp) = &self.splats
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("splat alternate front feature pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.front_feature_alt_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.front_feature_alt_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&self.splat_pipelines.front_feature_alt_pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_bind_group(1, &sp.bind_group, &[]);
            pass.draw(0..4, 0..sp.draw_count);
        }

        // Mode 5's CDF depends only on the tiny baked volume, so build it before drawing.
        // The following fragment pass can consume it immediately and performs no atomics.
        if spatial_baked != 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("spatial baked cdf build pass"),
                ..Default::default()
            });
            pass.set_pipeline(if spatial_baked >= 2 {
                &self
                    .histogram_wboit
                    .high_quality_spatial_cdf_build_pipeline
            } else {
                &self.histogram_wboit.spatial_cdf_build_pipeline
            });
            pass.set_bind_group(0, &self.cdf_build_bind_group, &[]);
            pass.dispatch_workgroups(
                self.histo_params.tile_count_x,
                self.histo_params.tile_count_y,
                1,
            );
        }

        // Accumulation. Modes 3/4 also record a live histogram; mode 5 only samples.
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
                        view: &self.revealage_views[fi],
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

            self.draw_scene(
                &mut pass,
                visible,
                if spatial_baked == 3 {
                    RenderMode::DoubleSampleFrontWboit
                } else if spatial_baked == 2 {
                    RenderMode::HighQualitySpatialBakedWboit
                } else if spatial_baked == 1 {
                    RenderMode::SpatialBakedHistogramWboit
                } else {
                    self.mode
                },
            );
        }

        // Modes 3/4 build next frame's CDF from the histogram recorded above.
        if spatial_baked == 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cdf build pass"),
                ..Default::default()
            });
            pass.set_pipeline(&self.histogram_wboit.cdf_build_pipeline);
            pass.set_bind_group(0, &self.cdf_build_bind_group, &[]);
            pass.dispatch_workgroups(
                self.histo_params.tile_count_x,
                self.histo_params.tile_count_y,
                1,
            );
        }

        // Pass 3: Composite
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

            pass.set_pipeline(&self.histogram_wboit.composite_pipeline);
            pass.set_bind_group(0, &self.histo_composite_tex_bind_groups[fi], &[]);
            pass.set_bind_group(1, &self.histo_composite_flag_bind_group, &[]);
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

fn create_front_feature_textures(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::TextureView, wgpu::TextureView) {
    let feature = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("splat front feature texture"),
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
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("splat front feature depth"),
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
    (
        feature.create_view(&wgpu::TextureViewDescriptor::default()),
        depth.create_view(&wgpu::TextureViewDescriptor::default()),
    )
}

fn create_wboit_textures(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::TextureView, [wgpu::TextureView; 2]) {
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

    // Double-buffered revealage: both are render targets and texture inputs
    let revealage_views = std::array::from_fn(|i| {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("revealage texture {i}")),
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
        tex.create_view(&wgpu::TextureViewDescriptor::default())
    });

    (
        accum.create_view(&wgpu::TextureViewDescriptor::default()),
        revealage_views,
    )
}

fn create_cdf_texture(
    device: &wgpu::Device,
    tiles_x: u32,
    tiles_y: u32,
    num_bins: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("cdf 3d texture"),
        size: wgpu::Extent3d {
            width: tiles_x,
            height: tiles_y,
            depth_or_array_layers: num_bins,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}
