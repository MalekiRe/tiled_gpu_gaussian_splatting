use crate::mesh::{self, Mesh};
use crate::vertex::ObjectUniform;

pub struct SceneObject {
    pub mesh: Mesh,
    pub transform: glam::Mat4,
    pub color: [f32; 4],
    pub is_extra_mesh: bool, // toggled by M key
    pub original_alpha: f32, // vertex alpha for opaque toggle
}

impl SceneObject {
    pub fn uniform(&self) -> ObjectUniform {
        ObjectUniform {
            model: self.transform.to_cols_array_2d(),
            color: self.color,
        }
    }
}

pub struct Scene {
    pub objects: Vec<SceneObject>,
    pub show_meshes: bool,
    pub force_opaque: bool,
    pub time: f32,
}

impl Scene {
    pub fn new() -> Self {
        let mut objects = Vec::new();

        // Overlapping semi-transparent quads at various depths and angles
        let quad_configs: [(glam::Vec3, glam::Quat, [f32; 4]); 6] = [
            (
                glam::Vec3::new(0.0, 0.0, 0.0),
                glam::Quat::IDENTITY,
                [1.0, 0.2, 0.2, 0.4],
            ),
            (
                glam::Vec3::new(0.3, 0.2, 0.5),
                glam::Quat::from_rotation_y(0.4),
                [0.2, 1.0, 0.2, 0.5],
            ),
            (
                glam::Vec3::new(-0.2, -0.1, 1.0),
                glam::Quat::from_rotation_y(-0.3),
                [0.2, 0.2, 1.0, 0.45],
            ),
            (
                glam::Vec3::new(0.1, 0.3, 1.5),
                glam::Quat::from_rotation_x(0.2),
                [1.0, 1.0, 0.2, 0.35],
            ),
            (
                glam::Vec3::new(-0.3, -0.2, 0.3),
                glam::Quat::from_rotation_z(0.3) * glam::Quat::from_rotation_y(0.5),
                [1.0, 0.2, 1.0, 0.55],
            ),
            (
                glam::Vec3::new(0.0, 0.1, 0.7),
                glam::Quat::from_rotation_x(-0.2) * glam::Quat::from_rotation_y(0.8),
                [0.2, 1.0, 1.0, 0.5],
            ),
        ];

        for (pos, rot, color) in &quad_configs {
            let transform = glam::Mat4::from_rotation_translation(*rot, *pos)
                * glam::Mat4::from_scale(glam::Vec3::splat(1.5));
            objects.push(SceneObject {
                mesh: mesh::quad(*color),
                transform,
                color: [1.0, 1.0, 1.0, 1.0],
                is_extra_mesh: false,
                original_alpha: color[3],
            });
        }

        // Transparent cube
        objects.push(SceneObject {
            mesh: mesh::cube([0.8, 0.5, 0.2, 0.3]),
            transform: glam::Mat4::from_translation(glam::Vec3::new(2.5, 0.0, 0.0)),
            color: [1.0, 1.0, 1.0, 1.0],
            is_extra_mesh: true,
            original_alpha: 0.3,
        });

        // Transparent sphere
        objects.push(SceneObject {
            mesh: mesh::uv_sphere(24, 16, [0.3, 0.6, 0.9, 0.35]),
            transform: glam::Mat4::from_translation(glam::Vec3::new(-2.5, 0.0, 0.0))
                * glam::Mat4::from_scale(glam::Vec3::splat(1.2)),
            color: [1.0, 1.0, 1.0, 1.0],
            is_extra_mesh: true,
            original_alpha: 0.35,
        });

        Scene {
            objects,
            show_meshes: true,
            force_opaque: false,
            time: 0.0,
        }
    }

    pub fn update(&mut self, dt: f32) {
        self.time += dt;
        // Auto-rotate extra meshes
        let n = self.objects.len();
        // Cube
        if n >= 2 {
            let cube_idx = n - 2;
            self.objects[cube_idx].transform =
                glam::Mat4::from_translation(glam::Vec3::new(2.5, 0.0, 0.0))
                    * glam::Mat4::from_rotation_y(self.time * 0.5)
                    * glam::Mat4::from_rotation_x(self.time * 0.3);
        }
        // Sphere
        if n >= 1 {
            let sphere_idx = n - 1;
            self.objects[sphere_idx].transform =
                glam::Mat4::from_translation(glam::Vec3::new(-2.5, 0.0, 0.0))
                    * glam::Mat4::from_rotation_y(self.time * 0.4)
                    * glam::Mat4::from_scale(glam::Vec3::splat(1.2));
        }
    }

}
