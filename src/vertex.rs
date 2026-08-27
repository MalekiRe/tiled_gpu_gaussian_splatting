#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
}

impl Vertex {
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // normal
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // color
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    /// View matrix on its own; the splat shaders need it to build the 3D->2D covariance
    /// Jacobian, which is only defined in view space.
    pub view: [[f32; 4]; 4],
    pub near: f32,
    pub far: f32,
    /// Focal length in pixels, used to project the covariance to screen space.
    pub focal: [f32; 2],
    pub viewport: [f32; 2],
    pub _padding0: [f32; 2],
    /// World-space eye position, for view-dependent SH evaluation.
    pub cam_pos: [f32; 3],
    pub _padding1: f32,
}

/// Per-draw constants for the splat pipelines.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SplatParams {
    /// Number of splats actually drawn this frame (the render cap).
    pub count: u32,
    pub sh_degree: u32,
    /// Global multiplier on splat size; 1.0 is the reconstruction's own scale.
    pub splat_scale: f32,
    /// Robust scene radius, used to express mode 6 depth tolerances in object units.
    pub scene_radius: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ObjectUniform {
    pub model: [[f32; 4]; 4],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HistogramParams {
    pub tile_count_x: u32,
    pub tile_count_y: u32,
    pub num_bins: u32,
    pub tile_size: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DirectionalPriorParams {
    pub mix_factor: f32,
    pub enabled: u32,
    pub _padding: [u32; 2],
}
