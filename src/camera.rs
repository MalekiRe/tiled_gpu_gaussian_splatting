use crate::vertex::CameraUniform;

pub struct Camera {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub target: glam::Vec3,
    pub aspect: f32,
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
    // drag state
    pub dragging: bool,
    pub last_mouse: Option<(f64, f64)>,
}

impl Camera {
    pub fn new(aspect: f32) -> Self {
        Self {
            yaw: 0.5,
            pitch: 0.3,
            distance: 8.0,
            target: glam::Vec3::ZERO,
            aspect,
            fov_y: 45.0_f32.to_radians(),
            near: 2.0,
            far: 15.0,
            dragging: false,
            last_mouse: None,
        }
    }

    pub fn eye(&self) -> glam::Vec3 {
        let x = self.pitch.cos() * self.yaw.sin();
        let y = self.pitch.sin();
        let z = self.pitch.cos() * self.yaw.cos();
        self.target + self.distance * glam::Vec3::new(x, y, z)
    }

    pub fn view_matrix(&self) -> glam::Mat4 {
        glam::Mat4::look_at_rh(self.eye(), self.target, glam::Vec3::Y)
    }

    pub fn proj_matrix(&self) -> glam::Mat4 {
        glam::Mat4::perspective_rh(self.fov_y, self.aspect, self.near, self.far)
    }

    pub fn uniform(&self) -> CameraUniform {
        let vp = self.proj_matrix() * self.view_matrix();
        CameraUniform {
            view_proj: vp.to_cols_array_2d(),
            near: self.near,
            far: self.far,
            _padding: [0.0; 2],
        }
    }

    pub fn on_mouse_move(&mut self, x: f64, y: f64) {
        if self.dragging
            && let Some((lx, ly)) = self.last_mouse
        {
            let dx = (x - lx) as f32;
            let dy = (y - ly) as f32;
            self.yaw -= dx * 0.005;
            self.pitch += dy * 0.005;
            self.pitch = self.pitch.clamp(-1.4, 1.4);
        }
        self.last_mouse = Some((x, y));
    }

    pub fn on_scroll(&mut self, delta: f32) {
        self.distance -= delta * 0.5;
        self.distance = self.distance.clamp(1.0, 50.0);
    }

    pub fn reset(&mut self) {
        self.yaw = 0.5;
        self.pitch = 0.3;
        self.distance = 8.0;
        self.target = glam::Vec3::ZERO;
        self.dragging = false;
        self.last_mouse = None;
    }
}
