use crate::vertex::Vertex;

pub struct HistogramWboitPipeline {
    pub accum_pipeline: wgpu::RenderPipeline,
    pub composite_pipeline: wgpu::RenderPipeline,
    pub cdf_build_pipeline: wgpu::ComputePipeline,
    pub spatial_cdf_build_pipeline: wgpu::ComputePipeline,
    pub histo_accum_bgl: wgpu::BindGroupLayout,
    pub histo_composite_tex_bgl: wgpu::BindGroupLayout,
    pub cdf_build_bgl: wgpu::BindGroupLayout,
    pub flag_bgl: wgpu::BindGroupLayout,
}

impl HistogramWboitPipeline {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera_bgl: &wgpu::BindGroupLayout,
        object_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        // Bind group layout for histogram accum (group 2)
        // b0: histogram storage (rw), b1: CDF 3D texture, b2: CDF sampler,
        // b3: params uniform, b4: prev revealage texture
        let histo_accum_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("histo accum bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let histo_composite_tex_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("histo composite tex bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        // CDF build compute: live histogram, CDF output, tiled params, directional prior,
        // and controls for mixing the prior into the live distribution.
        let cdf_build_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cdf build bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let flag_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("histo flag bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let accum_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("histo_accum pipeline layout"),
            bind_group_layouts: &[camera_bgl, object_bgl, &histo_accum_bgl],
            immediate_size: 0,
        });

        let composite_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("histo composite pipeline layout"),
            bind_group_layouts: &[&histo_composite_tex_bgl, &flag_bgl],
            immediate_size: 0,
        });

        let cdf_build_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cdf build pipeline layout"),
            bind_group_layouts: &[&cdf_build_bgl],
            immediate_size: 0,
        });

        // Shader sources
        let common_wgsl = include_str!("../../shaders/common.wgsl");
        let accum_wgsl = include_str!("../../shaders/histo_accum.wgsl");
        let composite_wgsl = include_str!("../../shaders/histo_composite.wgsl");
        let cdf_build_wgsl = include_str!("../../shaders/histo_cdf_build.wgsl");
        let spatial_cdf_build_wgsl = include_str!("../../shaders/spatial_cdf_build.wgsl");

        let (accum_pipeline, composite_pipeline) = create_pipeline_pair(
            device,
            surface_format,
            &accum_layout,
            &composite_layout,
            &format!("{}\n{}", common_wgsl, accum_wgsl),
            composite_wgsl,
            "histo",
        );

        let cdf_build_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cdf_build shader"),
            source: wgpu::ShaderSource::Wgsl(cdf_build_wgsl.into()),
        });
        let spatial_cdf_build_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("spatial cdf build shader"),
                source: wgpu::ShaderSource::Wgsl(spatial_cdf_build_wgsl.into()),
            });

        let cdf_build_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cdf_build pipeline"),
                layout: Some(&cdf_build_layout),
                module: &cdf_build_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let spatial_cdf_build_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("spatial cdf build pipeline"),
                layout: Some(&cdf_build_layout),
                module: &spatial_cdf_build_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            accum_pipeline,
            composite_pipeline,
            cdf_build_pipeline,
            spatial_cdf_build_pipeline,
            histo_accum_bgl,
            histo_composite_tex_bgl,
            cdf_build_bgl,
            flag_bgl,
        }
    }
}

fn create_pipeline_pair(
    device: &wgpu::Device,
    surface_format: wgpu::TextureFormat,
    accum_layout: &wgpu::PipelineLayout,
    composite_layout: &wgpu::PipelineLayout,
    accum_source: &str,
    composite_source: &str,
    label_prefix: &str,
) -> (wgpu::RenderPipeline, wgpu::RenderPipeline) {
    let accum_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&format!("{label_prefix}_accum shader")),
        source: wgpu::ShaderSource::Wgsl(accum_source.into()),
    });

    let accum_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(&format!("{label_prefix}_accum pipeline")),
        layout: Some(accum_layout),
        vertex: wgpu::VertexState {
            module: &accum_shader,
            entry_point: Some("vs_main"),
            buffers: &[Vertex::layout()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &accum_shader,
            entry_point: Some("fs_main"),
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R8Unorm,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::Zero,
                            dst_factor: wgpu::BlendFactor::OneMinusSrc,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::Zero,
                            dst_factor: wgpu::BlendFactor::OneMinusSrc,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });

    let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&format!("{label_prefix}_composite shader")),
        source: wgpu::ShaderSource::Wgsl(composite_source.into()),
    });

    let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(&format!("{label_prefix}_composite pipeline")),
        layout: Some(composite_layout),
        vertex: wgpu::VertexState {
            module: &composite_shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &composite_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });

    (accum_pipeline, composite_pipeline)
}
