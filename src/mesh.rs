use crate::vertex::Vertex;

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u16>,
}

pub fn quad(color: [f32; 4]) -> Mesh {
    let n = [0.0, 0.0, 1.0];
    let vertices = vec![
        Vertex { position: [-1.0, -1.0, 0.0], normal: n, color },
        Vertex { position: [ 1.0, -1.0, 0.0], normal: n, color },
        Vertex { position: [ 1.0,  1.0, 0.0], normal: n, color },
        Vertex { position: [-1.0,  1.0, 0.0], normal: n, color },
    ];
    let indices = vec![0, 1, 2, 0, 2, 3];
    Mesh { vertices, indices }
}

pub fn cube(color: [f32; 4]) -> Mesh {
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        // +Z
        ([0.0, 0.0, 1.0], [[-1.0,-1.0, 1.0],[ 1.0,-1.0, 1.0],[ 1.0, 1.0, 1.0],[-1.0, 1.0, 1.0]]),
        // -Z
        ([0.0, 0.0,-1.0], [[ 1.0,-1.0,-1.0],[-1.0,-1.0,-1.0],[-1.0, 1.0,-1.0],[ 1.0, 1.0,-1.0]]),
        // +X
        ([1.0, 0.0, 0.0], [[ 1.0,-1.0, 1.0],[ 1.0,-1.0,-1.0],[ 1.0, 1.0,-1.0],[ 1.0, 1.0, 1.0]]),
        // -X
        ([-1.0, 0.0, 0.0], [[-1.0,-1.0,-1.0],[-1.0,-1.0, 1.0],[-1.0, 1.0, 1.0],[-1.0, 1.0,-1.0]]),
        // +Y
        ([0.0, 1.0, 0.0], [[-1.0, 1.0, 1.0],[ 1.0, 1.0, 1.0],[ 1.0, 1.0,-1.0],[-1.0, 1.0,-1.0]]),
        // -Y
        ([0.0,-1.0, 0.0], [[-1.0,-1.0,-1.0],[ 1.0,-1.0,-1.0],[ 1.0,-1.0, 1.0],[-1.0,-1.0, 1.0]]),
    ];

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for (normal, positions) in &faces {
        let base = vertices.len() as u16;
        for &pos in positions {
            vertices.push(Vertex { position: pos, normal: *normal, color });
        }
        indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
    }

    Mesh { vertices, indices }
}

pub fn uv_sphere(slices: u32, stacks: u32, color: [f32; 4]) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for i in 0..=stacks {
        let v = i as f32 / stacks as f32;
        let phi = v * std::f32::consts::PI;
        for j in 0..=slices {
            let u = j as f32 / slices as f32;
            let theta = u * 2.0 * std::f32::consts::PI;

            let x = theta.sin() * phi.sin();
            let y = phi.cos();
            let z = theta.cos() * phi.sin();

            vertices.push(Vertex {
                position: [x, y, z],
                normal: [x, y, z],
                color,
            });
        }
    }

    for i in 0..stacks {
        for j in 0..slices {
            let a = i * (slices + 1) + j;
            let b = a + slices + 1;
            indices.push(a as u16);
            indices.push(b as u16);
            indices.push((a + 1) as u16);
            indices.push((a + 1) as u16);
            indices.push(b as u16);
            indices.push((b + 1) as u16);
        }
    }

    Mesh { vertices, indices }
}
