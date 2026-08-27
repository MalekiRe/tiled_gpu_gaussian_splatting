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
    /// Viewport in pixels; the splat shaders need it to size projected Gaussians.
    pub viewport: (f32, f32),
    pub min_distance: f32,
    pub max_distance: f32,
    /// Where `reset()` returns to.
    home: (f32, f32, f32, glam::Vec3),
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
            far: 100.0,
            viewport: (1280.0, 720.0),
            min_distance: 1.0,
            max_distance: 50.0,
            home: (0.5, 0.3, 8.0, glam::Vec3::ZERO),
            dragging: false,
            last_mouse: None,
        }
    }

    /// Frame an arbitrary scene: pull the orbit target onto its centre and back off far
    /// enough to see all of it, then rescale the near/far planes and zoom limits to match.
    pub fn fit_to(&mut self, center: glam::Vec3, radius: f32) {
        let radius = radius.max(1e-3);
        self.target = center;
        self.distance = radius * 2.5;
        self.near = (radius * 0.01).max(1e-3);
        self.far = radius * 50.0;
        self.min_distance = radius * 0.05;
        self.max_distance = radius * 20.0;
        self.pitch = 0.15;
        self.yaw = 0.0;
        self.home = (self.yaw, self.pitch, self.distance, self.target);
    }

    /// Unit vector from the eye towards the orbit target.
    pub fn forward(&self) -> glam::Vec3 {
        (self.target - self.eye()).normalize_or_zero()
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
        let view = self.view_matrix();
        let vp = self.proj_matrix() * view;
        let (w, h) = self.viewport;
        // Pixels per unit at unit depth. Both axes share it: width / aspect == height.
        let fy = h / (2.0 * (self.fov_y * 0.5).tan());
        CameraUniform {
            view_proj: vp.to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            near: self.near,
            far: self.far,
            focal: [fy, fy],
            viewport: [w, h],
            _padding0: [0.0; 2],
            cam_pos: self.eye().to_array(),
            _padding1: 0.0,
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
        // Proportional zoom, so the step stays sensible at any scene scale.
        self.distance *= (-delta * 0.12).exp();
        self.distance = self.distance.clamp(self.min_distance, self.max_distance);
    }

    pub fn reset(&mut self) {
        let (yaw, pitch, distance, target) = self.home;
        self.yaw = yaw;
        self.pitch = pitch;
        self.distance = distance;
        self.target = target;
        self.dragging = false;
        self.last_mouse = None;
    }
}
