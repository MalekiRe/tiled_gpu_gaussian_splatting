//! Render pipelines for the Gaussian splat front-end. One per transparency mode; they all
//! share the same vertex stage and differ only in their fragment outputs and blend state.
//!
//! Splats carry their own ordering (or deliberately ignore it), so depth testing is off in
//! every mode -- but the pipelines still declare a depth-stencil state, because they draw
//! into the same render passes as the mesh path.

pub struct SplatPipelines {
    pub alpha_pipeline: wgpu::RenderPipeline,
    pub wboit_pipeline: wgpu::RenderPipeline,
    pub histo_pipeline: wgpu::RenderPipeline,
    pub baked_histo_pipeline: wgpu::RenderPipeline,
    pub splat_bgl: wgpu::BindGroupLayout,
}

fn depth_state() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth32Float,
        depth_write_enabled: false,
        depth_compare: wgpu::CompareFunction::Always,
        stencil: Default::default(),
        bias: Default::default(),
    }
}

fn shader(device: &wgpu::Device, label: &str, body: &str) -> wgpu::ShaderModule {
    let common = include_str!("../../shaders/splat_common.wgsl");
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(format!("{}\n{}", common, body).into()),
    })
}

impl SplatPipelines {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera_bgl: &wgpu::BindGroupLayout,
        histo_accum_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        let storage = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let splat_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("splat bgl"),
            entries: &[
                storage(0), // splats
                storage(1), // spherical harmonics
                storage(2), // depth-sorted draw order
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let primitive = wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            cull_mode: None,
            ..Default::default()
        };

        // Mode 1: sorted back-to-front alpha blending.
        let alpha_shader = shader(
            device,
            "splat alpha shader",
            include_str!("../../shaders/splat_alpha.wgsl"),
        );
        let alpha_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("splat alpha layout"),
            bind_group_layouts: &[camera_bgl, &splat_bgl],
            immediate_size: 0,
        });
        let alpha_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("splat alpha pipeline"),
            layout: Some(&alpha_layout),
            vertex: wgpu::VertexState {
                module: &alpha_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &alpha_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive,
            depth_stencil: Some(depth_state()),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        // Modes 2 and 3 share the MRT accumulation targets: additive accum, multiplicative
        // revealage.
        let mrt_targets = [
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
                write_mask: wgpu::ColorWrites::RED,
            }),
        ];

        let wboit_shader = shader(
            device,
            "splat wboit shader",
            include_str!("../../shaders/splat_wboit.wgsl"),
        );
        let wboit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("splat wboit layout"),
            bind_group_layouts: &[camera_bgl, &splat_bgl],
            immediate_size: 0,
        });
        let wboit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("splat wboit pipeline"),
            layout: Some(&wboit_layout),
            vertex: wgpu::VertexState {
                module: &wboit_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &wboit_shader,
                entry_point: Some("fs_main"),
                targets: &mrt_targets,
                compilation_options: Default::default(),
            }),
            primitive,
            depth_stencil: Some(depth_state()),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let histo_shader = shader(
            device,
            "splat histo shader",
            include_str!("../../shaders/splat_histo.wgsl"),
        );
        let histo_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("splat histo layout"),
            bind_group_layouts: &[camera_bgl, &splat_bgl, histo_accum_bgl],
            immediate_size: 0,
        });
        let histo_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("splat histo pipeline"),
            layout: Some(&histo_layout),
            vertex: wgpu::VertexState {
                module: &histo_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &histo_shader,
                entry_point: Some("fs_main"),
                targets: &mrt_targets,
                compilation_options: Default::default(),
            }),
            primitive,
            depth_stencil: Some(depth_state()),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let baked_histo_shader = shader(
            device,
            "splat baked histo shader",
            include_str!("../../shaders/splat_baked_histo.wgsl"),
        );
        let baked_histo_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("splat baked histo pipeline"),
                layout: Some(&histo_layout),
                vertex: wgpu::VertexState {
                    module: &baked_histo_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &baked_histo_shader,
                    entry_point: Some("fs_main"),
                    targets: &mrt_targets,
                    compilation_options: Default::default(),
                }),
                primitive,
                depth_stencil: Some(depth_state()),
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            });

        Self {
            alpha_pipeline,
            wboit_pipeline,
            histo_pipeline,
            baked_histo_pipeline,
            splat_bgl,
        }
    }
}
