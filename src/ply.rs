//! Minimal binary PLY reader for 3D Gaussian Splatting scenes.
//!
//! Handles the two variants found in the wild:
//!   * the original INRIA export (`x`,`y`,`z`,`scale_*`,`rot_*`,`opacity`,`f_dc_*`,`f_rest_*`)
//!   * the PlayCanvas / SuperSplat compressed export (`element chunk` bounds + 4 packed u32 per
//!     vertex).  Bit layouts follow the PlayCanvas engine decoder: position and scale are
//!     11/10/11 unorm lerped between the chunk bounds, rotation is 2-10-10-10 largest-three,
//!     colour is 8888 (rgb lerped between chunk colour bounds, a used directly as opacity).

use std::path::Path;

pub const SH_C0: f32 = 0.28209479177387814;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Scalar {
    F32,
    F64,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
}

impl Scalar {
    fn from_name(name: &str) -> Option<Scalar> {
        Some(match name {
            "float" | "float32" => Scalar::F32,
            "double" | "float64" => Scalar::F64,
            "uchar" | "uint8" => Scalar::U8,
            "char" | "int8" => Scalar::I8,
            "ushort" | "uint16" => Scalar::U16,
            "short" | "int16" => Scalar::I16,
            "uint" | "uint32" => Scalar::U32,
            "int" | "int32" => Scalar::I32,
            _ => return None,
        })
    }

    fn size(self) -> usize {
        match self {
            Scalar::U8 | Scalar::I8 => 1,
            Scalar::U16 | Scalar::I16 => 2,
            Scalar::F32 | Scalar::U32 | Scalar::I32 => 4,
            Scalar::F64 => 8,
        }
    }

    fn read_f32(self, b: &[u8]) -> f32 {
        match self {
            Scalar::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            Scalar::F64 => f64::from_le_bytes([
                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
            ]) as f32,
            Scalar::U8 => b[0] as f32,
            Scalar::I8 => (b[0] as i8) as f32,
            Scalar::U16 => u16::from_le_bytes([b[0], b[1]]) as f32,
            Scalar::I16 => i16::from_le_bytes([b[0], b[1]]) as f32,
            Scalar::U32 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
            Scalar::I32 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
        }
    }

    fn read_u32(self, b: &[u8]) -> u32 {
        match self {
            Scalar::U32 | Scalar::I32 | Scalar::F32 => {
                u32::from_le_bytes([b[0], b[1], b[2], b[3]])
            }
            Scalar::U16 | Scalar::I16 => u16::from_le_bytes([b[0], b[1]]) as u32,
            Scalar::U8 | Scalar::I8 => b[0] as u32,
            Scalar::F64 => 0,
        }
    }
}

struct Prop {
    name: String,
    ty: Scalar,
    offset: usize,
}

struct Element {
    name: String,
    count: usize,
    props: Vec<Prop>,
    stride: usize,
    start: usize,
}

impl Element {
    fn prop(&self, name: &str) -> Option<&Prop> {
        self.props.iter().find(|p| p.name == name)
    }

    fn has(&self, name: &str) -> bool {
        self.prop(name).is_some()
    }

    /// Byte slice of row `i`.
    fn row<'a>(&self, data: &'a [u8], i: usize) -> &'a [u8] {
        let off = self.start + i * self.stride;
        &data[off..off + self.stride]
    }
}

/// One Gaussian, fully decoded and in the source file's coordinate frame.
pub struct SplatData {
    pub pos: Vec<[f32; 3]>,
    /// Linear (already exponentiated) per-axis standard deviations.
    pub scale: Vec<[f32; 3]>,
    /// Normalized rotation quaternion, `(w, x, y, z)`.
    pub rot: Vec<[f32; 4]>,
    /// Base colour, i.e. the DC band already evaluated: `0.5 + SH_C0 * f_dc`.
    pub color: Vec<[f32; 3]>,
    /// Opacity in `[0, 1]` (sigmoid already applied).
    pub opacity: Vec<f32>,
    /// Higher SH bands, channel-major (`R` coeffs 0..14, then `G`, then `B`),
    /// `45 * len()` entries, or empty when the file carries no `f_rest`.
    pub sh: Vec<f32>,
    pub sh_degree: u32,
}

impl SplatData {
    pub fn len(&self) -> usize {
        self.pos.len()
    }
}

fn header_end(data: &[u8]) -> Option<usize> {
    let needle = b"end_header";
    let limit = data.len().min(1 << 20);
    let pos = data[..limit]
        .windows(needle.len())
        .position(|w| w == needle)?;
    let mut i = pos + needle.len();
    while i < data.len() && data[i] != b'\n' {
        i += 1;
    }
    Some(i + 1)
}

fn parse_header(text: &str) -> Result<Vec<Element>, String> {
    let mut elements: Vec<Element> = Vec::new();
    let mut format_ok = false;

    for line in text.lines() {
        let line = line.trim();
        let mut it = line.split_whitespace();
        match it.next() {
            Some("format") => {
                let f = it.next().unwrap_or("");
                if f != "binary_little_endian" {
                    return Err(format!(
                        "unsupported PLY format `{f}` (only binary_little_endian is supported)"
                    ));
                }
                format_ok = true;
            }
            Some("element") => {
                let name = it.next().unwrap_or("").to_string();
                let count: usize = it
                    .next()
                    .unwrap_or("0")
                    .parse()
                    .map_err(|_| "bad element count".to_string())?;
                elements.push(Element {
                    name,
                    count,
                    props: Vec::new(),
                    stride: 0,
                    start: 0,
                });
            }
            Some("property") => {
                let ty = it.next().unwrap_or("");
                if ty == "list" {
                    return Err("list properties are not supported".into());
                }
                let ty = Scalar::from_name(ty)
                    .ok_or_else(|| format!("unknown property type `{ty}`"))?;
                let name = it.next().unwrap_or("").to_string();
                let el = elements
                    .last_mut()
                    .ok_or_else(|| "property before element".to_string())?;
                let offset = el.stride;
                el.stride += ty.size();
                el.props.push(Prop { name, ty, offset });
            }
            _ => {}
        }
    }

    if !format_ok {
        return Err("missing `format` line".into());
    }

    let mut cursor = 0usize;
    for el in &mut elements {
        el.start = cursor;
        cursor += el.stride * el.count;
    }
    Ok(elements)
}

pub fn load(path: &Path) -> Result<SplatData, String> {
    let raw = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let body_at = header_end(&raw).ok_or("no `end_header` found")?;
    let header = String::from_utf8_lossy(&raw[..body_at]).to_string();
    let elements = parse_header(&header)?;
    let body = &raw[body_at..];

    let vertex = elements
        .iter()
        .find(|e| e.name == "vertex")
        .ok_or("no `vertex` element")?;

    let needed = vertex.start + vertex.stride * vertex.count;
    if body.len() < needed {
        return Err(format!(
            "truncated file: body has {} bytes, header describes {needed}",
            body.len()
        ));
    }

    if vertex.has("packed_position") {
        load_compressed(body, &elements, vertex)
    } else {
        load_uncompressed(body, vertex)
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn load_uncompressed(body: &[u8], vertex: &Element) -> Result<SplatData, String> {
    let n = vertex.count;

    let fetch = |name: &str| -> Result<&Prop, String> {
        vertex
            .prop(name)
            .ok_or_else(|| format!("missing required property `{name}`"))
    };

    let px = fetch("x")?;
    let py = fetch("y")?;
    let pz = fetch("z")?;
    let s0 = fetch("scale_0")?;
    let s1 = fetch("scale_1")?;
    let s2 = fetch("scale_2")?;
    let r0 = fetch("rot_0")?;
    let r1 = fetch("rot_1")?;
    let r2 = fetch("rot_2")?;
    let r3 = fetch("rot_3")?;
    let op = fetch("opacity")?;
    let d0 = fetch("f_dc_0")?;
    let d1 = fetch("f_dc_1")?;
    let d2 = fetch("f_dc_2")?;

    // Count contiguous f_rest_* properties; 45 of them means SH degree 3.
    let mut rest: Vec<&Prop> = Vec::new();
    while let Some(p) = vertex.prop(&format!("f_rest_{}", rest.len())) {
        rest.push(p);
    }
    let sh_degree: u32 = match rest.len() {
        45 => 3,
        24 => 2,
        9 => 1,
        _ => 0,
    };
    let coeffs_per_channel = match sh_degree {
        3 => 15,
        2 => 8,
        1 => 3,
        _ => 0,
    };

    let mut out = SplatData {
        pos: Vec::with_capacity(n),
        scale: Vec::with_capacity(n),
        rot: Vec::with_capacity(n),
        color: Vec::with_capacity(n),
        opacity: Vec::with_capacity(n),
        sh: if sh_degree > 0 {
            vec![0.0; n * 45]
        } else {
            Vec::new()
        },
        sh_degree,
    };

    let get = |row: &[u8], p: &Prop| p.ty.read_f32(&row[p.offset..]);

    for i in 0..n {
        let row = vertex.row(body, i);

        out.pos.push([get(row, px), get(row, py), get(row, pz)]);
        out.scale.push([
            get(row, s0).exp(),
            get(row, s1).exp(),
            get(row, s2).exp(),
        ]);

        let q = [get(row, r0), get(row, r1), get(row, r2), get(row, r3)];
        let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        let inv = if len > 1e-12 { 1.0 / len } else { 0.0 };
        out.rot.push(if inv == 0.0 {
            [1.0, 0.0, 0.0, 0.0]
        } else {
            [q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv]
        });

        out.color.push([
            0.5 + SH_C0 * get(row, d0),
            0.5 + SH_C0 * get(row, d1),
            0.5 + SH_C0 * get(row, d2),
        ]);
        out.opacity.push(sigmoid(get(row, op)));

        if sh_degree > 0 {
            // f_rest is channel-major: 15 coeffs of R, then G, then B (padded to 15 each).
            let base = i * 45;
            for c in 0..3usize {
                for k in 0..coeffs_per_channel {
                    let src = c * coeffs_per_channel + k;
                    out.sh[base + c * 15 + k] = get(row, rest[src]);
                }
            }
        }
    }

    Ok(out)
}

fn unpack_unorm(value: u32, bits: u32) -> f32 {
    let t = (1u32 << bits) - 1;
    (value & t) as f32 / t as f32
}

fn unpack_111011(value: u32) -> [f32; 3] {
    [
        unpack_unorm(value >> 21, 11),
        unpack_unorm(value >> 11, 10),
        unpack_unorm(value, 11),
    ]
}

/// 2-10-10-10 "largest three" quaternion, returned as `(w, x, y, z)`.
fn unpack_rot(value: u32) -> [f32; 4] {
    const NORM: f32 = std::f32::consts::SQRT_2;
    let a = (unpack_unorm(value >> 20, 10) - 0.5) * NORM;
    let b = (unpack_unorm(value >> 10, 10) - 0.5) * NORM;
    let c = (unpack_unorm(value, 10) - 0.5) * NORM;
    let m = (1.0 - (a * a + b * b + c * c)).max(0.0).sqrt();
    // PlayCanvas stores (x, y, z, w); reorder to (w, x, y, z).
    match value >> 30 {
        0 => [m, a, b, c],
        1 => [a, m, b, c],
        2 => [a, b, m, c],
        _ => [a, b, c, m],
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

fn load_compressed(
    body: &[u8],
    elements: &[Element],
    vertex: &Element,
) -> Result<SplatData, String> {
    let chunk_el = elements
        .iter()
        .find(|e| e.name == "chunk")
        .ok_or("compressed PLY without a `chunk` element")?;

    // Chunk bounds, 18 floats per chunk. Colour bounds are optional (older exports
    // omit them, in which case the packed colour is used directly).
    let names = [
        "min_x", "min_y", "min_z", "max_x", "max_y", "max_z", "min_scale_x", "min_scale_y",
        "min_scale_z", "max_scale_x", "max_scale_y", "max_scale_z", "min_r", "min_g", "min_b",
        "max_r", "max_g", "max_b",
    ];
    let has_color_bounds = chunk_el.has("min_r");
    let mut chunks = vec![0.0f32; chunk_el.count * 18];
    for i in 0..chunk_el.count {
        let row = chunk_el.row(body, i);
        for (j, name) in names.iter().enumerate() {
            chunks[i * 18 + j] = match chunk_el.prop(name) {
                Some(p) => p.ty.read_f32(&row[p.offset..]),
                // Identity bounds so the lerp becomes a pass-through.
                None => if j >= 15 { 1.0 } else { 0.0 },
            };
        }
    }

    let pp = vertex.prop("packed_position").unwrap();
    let pr = vertex
        .prop("packed_rotation")
        .ok_or("missing `packed_rotation`")?;
    let ps = vertex.prop("packed_scale").ok_or("missing `packed_scale`")?;
    let pc = vertex.prop("packed_color").ok_or("missing `packed_color`")?;

    let n = vertex.count;
    let mut out = SplatData {
        pos: Vec::with_capacity(n),
        scale: Vec::with_capacity(n),
        rot: Vec::with_capacity(n),
        color: Vec::with_capacity(n),
        opacity: Vec::with_capacity(n),
        sh: Vec::new(),
        sh_degree: 0,
    };

    for i in 0..n {
        let row = vertex.row(body, i);
        let ci = (i / 256).min(chunk_el.count.saturating_sub(1)) * 18;

        let p = unpack_111011(pp.ty.read_u32(&row[pp.offset..]));
        out.pos.push([
            lerp(chunks[ci], chunks[ci + 3], p[0]),
            lerp(chunks[ci + 1], chunks[ci + 4], p[1]),
            lerp(chunks[ci + 2], chunks[ci + 5], p[2]),
        ]);

        out.rot.push(unpack_rot(pr.ty.read_u32(&row[pr.offset..])));

        // Scales are stored in log space, like the uncompressed format.
        let s = unpack_111011(ps.ty.read_u32(&row[ps.offset..]));
        out.scale.push([
            lerp(chunks[ci + 6], chunks[ci + 9], s[0]).exp(),
            lerp(chunks[ci + 7], chunks[ci + 10], s[1]).exp(),
            lerp(chunks[ci + 8], chunks[ci + 11], s[2]).exp(),
        ]);

        let cv = pc.ty.read_u32(&row[pc.offset..]);
        let c = [
            unpack_unorm(cv >> 24, 8),
            unpack_unorm(cv >> 16, 8),
            unpack_unorm(cv >> 8, 8),
        ];
        // Compressed colour is already the evaluated DC band, not a raw f_dc coefficient.
        out.color.push(if has_color_bounds {
            [
                lerp(chunks[ci + 12], chunks[ci + 15], c[0]),
                lerp(chunks[ci + 13], chunks[ci + 16], c[1]),
                lerp(chunks[ci + 14], chunks[ci + 17], c[2]),
            ]
        } else {
            c
        });
        // ...and the packed alpha is already sigmoid-applied opacity.
        out.opacity.push(unpack_unorm(cv, 8));
    }

    // Optional quantized higher SH bands, one row per splat.
    if let Some(sh_el) = elements.iter().find(|e| e.name == "sh") {
        let mut rest: Vec<&Prop> = Vec::new();
        while let Some(p) = sh_el.prop(&format!("f_rest_{}", rest.len())) {
            rest.push(p);
        }
        let per_channel = rest.len() / 3;
        let degree = match per_channel {
            15 => 3,
            8 => 2,
            3 => 1,
            _ => 0,
        };
        if degree > 0 && sh_el.count >= n {
            out.sh = vec![0.0; n * 45];
            for i in 0..n {
                let row = sh_el.row(body, i);
                for c in 0..3usize {
                    for k in 0..per_channel {
                        let p = rest[c * per_channel + k];
                        let q = p.ty.read_f32(&row[p.offset..]);
                        // PlayCanvas quantization: byte -> [-4, 4].
                        out.sh[i * 45 + c * 15 + k] = q * (8.0 / 255.0) - 4.0;
                    }
                }
            }
            out.sh_degree = degree;
        }
    }

    Ok(out)
}
