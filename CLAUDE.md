# Histogram-Equalized WBOIT Demo

## Context

Comparing three transparency rendering techniques side-by-side in a wgpu demo. The demo
renders either a built-in quad/mesh scene or a **3D Gaussian Splatting** scene loaded from a
PLY file (see *3DGS Mode* below), through the same three techniques:
1. **Regular alpha blending** (baseline with ordering artifacts)
2. **Naive WBOIT** (McGuire & Bavoil 2013 - static depth-based weight function)
3. **Histogram-equalized WBOIT** (novel technique - uses a global depth histogram from the previous frame to build a CDF, which remaps depth values so WBOIT weights spread across the full f32 exponential range)

The goal is to show that adaptive weight redistribution via histogram equalization significantly improves WBOIT quality when fragments cluster at similar depths.

## Project Setup

```
cargo new wboit-demo
```
Path: `/run/media/nova/MEDIA/2d/wboit-demo/`

### Cargo.toml dependencies
> Implemented in `Cargo.toml`

```toml
[package]
name = "wboit-demo"
version = "0.1.0"
edition = "2024"

[dependencies]
wgpu = "28"
winit = "0.30"
pollster = "0.4"
bytemuck = { version = "1", features = ["derive"] }
glam = "0.29"
env_logger = "0.11"
log = "0.4"
```

## File Structure

All files below are implemented and `cargo build` compiles clean (zero warnings).

```
wboit-demo/
├── .gitignore                  # DONE - Standard Rust gitignore
├── Cargo.toml                  # DONE - Project manifest with dependencies
├── Cargo.lock                  # DONE - Dependency lock file
├── CLAUDE.md                   # DONE - This file (project documentation)
├── splats/                     # Test 3DGS PLY files (not tracked)
├── src/
│   ├── main.rs                 # DONE - env_logger init, optional PLY argument, winit event loop
│   ├── app.rs                  # DONE - ApplicationHandler, keyboard/mouse/scroll input, redraw loop
│   ├── renderer.rs             # DONE - GPU state, 3 render paths, resize, buffer management
│   ├── camera.rs               # DONE - Orbit camera (spherical coords, mouse drag, scroll zoom)
│   ├── scene.rs                # DONE - 6 quads + cube + sphere, auto-rotation, mesh toggle
│   ├── mesh.rs                 # DONE - Procedural quad, cube, UV-sphere generators
│   ├── ply.rs                  # DONE - Binary PLY reader: INRIA + SuperSplat compressed
│   ├── splats.rs               # DONE - Splat GPU packing, importance order, async depth sorter
│   ├── vertex.rs               # DONE - Vertex, CameraUniform, ObjectUniform, HistogramParams, SplatParams
│   └── pipeline/
│       ├── mod.rs              # DONE - Re-exports
│       ├── alpha_blend.rs      # DONE - Mode 1 pipeline (SrcAlpha/OneMinusSrcAlpha blend)
│       ├── naive_wboit.rs      # DONE - Mode 2 accum pipeline (MRT) + composite pipeline
│       ├── histogram_wboit.rs  # DONE - Mode 3 accum + compute CDF + composite pipelines
│       └── splat.rs            # DONE - 3DGS variants of all three accumulation pipelines
└── shaders/
    ├── common.wgsl             # DONE - Camera/Object/VertexInput/VertexOutput structs, basic_vertex, simple_lighting
    ├── alpha_blend.wgsl        # DONE - vs_main/fs_main calling basic_vertex + simple_lighting
    ├── wboit_accum.wgsl        # DONE - McGuire&Bavoil weight function, MRT output (accum + revealage)
    ├── wboit_composite.wgsl    # DONE - Fullscreen triangle, textureLoad accum/revealage, alpha composite
    ├── histo_accum.wgsl        # DONE - atomicAdd to global histogram, CDF lookup, transmittance weight
    ├── histo_cdf_build.wgsl    # DONE - Compute shader: parallel prefix sum (Hillis-Steele), CDF normalize, histogram clear
    ├── histo_composite.wgsl    # DONE - WBOIT composite only (textures + revealage flag)
    ├── splat_common.wgsl       # DONE - EWA projection, conic evaluation, SH degree-3 eval
    ├── splat_alpha.wgsl        # DONE - Mode 1 fragment output for splats
    ├── splat_wboit.wgsl        # DONE - Mode 2 fragment output for splats
    └── splat_histo.wgsl        # DONE - Mode 3 fragment output for splats
```

## Data Types

### Vertex (40 bytes)
> Implemented in `src/vertex.rs:4-9` with `layout()` method at `:11-35`

```rust
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],  // location 0
    normal: [f32; 3],    // location 1
    color: [f32; 4],     // location 2
}
```

### CameraUniform (64 bytes)
> Implemented in `src/vertex.rs:38-41`

```rust
#[repr(C)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}
```

### ObjectUniform (80 bytes)
> Implemented in `src/vertex.rs:44-48`

```rust
#[repr(C)]
struct ObjectUniform {
    model: [[f32; 4]; 4],
    color: [f32; 4],  // RGBA override/tint
}
```

### HistogramParams (16 bytes)
> Implemented in `src/vertex.rs:51-56`

```rust
#[repr(C)]
struct HistogramParams {
    tile_count_x: u32,
    tile_count_y: u32,
    num_bins: u32,
    depth_range: f32,  // max linear depth for binning
}
```

## Render Architecture

### Mode 1: Alpha Blend
> Pipeline: `src/pipeline/alpha_blend.rs` (single `RenderPipeline`)
> Shader: `shaders/alpha_blend.wgsl` (prepended with `shaders/common.wgsl`)
> Render path: `src/renderer.rs` `render_alpha_blend()`

Single pass:
- Color target: swapchain (clear to dark gray)
- Depth: depth texture (Depth32Float, clear)
- Blend: `SrcAlpha / OneMinusSrcAlpha` (standard)
- Draw all transparent objects unsorted

### Mode 2: Naive WBOIT
> Pipelines: `src/pipeline/naive_wboit.rs` (`accum_pipeline` + `composite_pipeline` + `composite_bgl`)
> Shaders: `shaders/wboit_accum.wgsl` (prepended with common), `shaders/wboit_composite.wgsl` (standalone)
> Render path: `src/renderer.rs` `render_naive_wboit()`

**Pass 1 - Accumulation (MRT):**
- Color 0: `Rgba16Float` accum texture (clear to 0,0,0,0)
  - Blend: `One + One` (additive)
- Color 1: `R8Unorm` revealage texture (clear to 1,0,0,0)
  - Blend: `Zero + OneMinusSrc` (multiplicative)
- Depth: depth texture (write disabled, depth test only)
- Fragment outputs: `(premul_color * w, alpha * w)` to color0, `alpha` to color1
- Weight function: `w = alpha * max(1e-2, min(3e3, 10.0 / (1e-5 + pow(z/5, 2) + pow(z/200, 4))))`

**Pass 2 - Composite (fullscreen triangle):**
- Color target: swapchain (clear to background)
- Blend: `SrcAlpha / OneMinusSrcAlpha`
- Sample accum and revealage textures
- Output: `color = accum.rgb / max(accum.a, 1e-5)`, `alpha = 1 - revealage`

### Mode 3: Histogram-Equalized WBOIT (global, 3 passes)
> Pipelines: `src/pipeline/histogram_wboit.rs` (`accum_pipeline` + `cdf_build_pipeline` (compute) + `composite_pipeline`)
> Shaders: `shaders/histo_accum.wgsl` (prepended with common), `shaders/histo_cdf_build.wgsl` (compute), `shaders/histo_composite.wgsl` (standalone)
> Render path: `src/renderer.rs` `render_histogram_wboit()`
> Buffer setup: `src/renderer.rs` `new()` (histogram_buffer, cdf_buffer, histo_params_buffer) and `resize()`

**Pass 1 - Accumulation + Optical Depth Histogram Recording:**
- Same MRT setup as Mode 2 (accum `Rgba16Float` + revealage `R8Unorm`)
- Additional bind groups: histogram buffer (`read_write`, atomic), CDF buffer (`read`), params uniform
- Fragment shader does:
  1. Compute linear depth `z` from clip-space (using `clip_position.w`)
  2. Compute bin index: `bin = clamp(u32(normalized_z * f32(num_bins)), 0, num_bins - 1)`
  3. Compute optical depth: `od = -ln(1 - alpha)`, quantize: `u32(od * OD_SCALE)` where `OD_SCALE = 4096`
  4. `atomicAdd(&histogram[bin], quantized_od)` - accumulates optical depth per bin (not fragment count)
  5. Read CDF from **previous frame** with piecewise-linear interpolation between `cdf[bin-1]` and `cdf[bin]`
  6. Reconstruct absolute cumulative optical depth: `tau = interpolated_cdf * cdf[num_bins]`
  7. Weight: `w = exp(-tau)` — exact transmittance before this layer (per-bin resolution)
  8. Output to MRT: `(premul_color * w, alpha * w)` to accum, `alpha` to revealage

**Pass 2 - CDF Build (compute shader):**
- Dispatch: `(1, 1, 1)` workgroups, `@workgroup_size(256, 1, 1)`
- Each thread handles one histogram bin
- Parallel inclusive prefix sum via Hillis-Steele (8 steps for 256 bins) using workgroup shared memory
- Normalize by total optical depth, write CDF. Store total OD in `cdf[num_bins]`. If total == 0, write linear fallback.
- Clear histogram to 0 via `atomicStore` for next frame.

**Pass 3 - Composite (fullscreen triangle):**
- Color target: swapchain (clear to background)
- Bind groups: accum texture, revealage texture, flag uniform
- Same composite math as Mode 2: read accum+revealage, output blended color

### Histogram Buffer Layout
> Created in `src/renderer.rs` `new()` and recreated in `resize()`

- Storage: `array<atomic<u32>>` of length `NUM_DEPTH_BINS` (256)
- `NUM_DEPTH_BINS = 256` (const in `src/renderer.rs:8`)
- Each bin accumulates **quantized optical depth** (`-ln(1-alpha) * OD_SCALE`), not fragment count

### CDF Buffer Layout
> Created in `src/renderer.rs` `new()`, initialized with linear fallback, recreated in `resize()`

- Storage: `array<f32>` of length `NUM_DEPTH_BINS + 1` (257)
- Entries 0-255: normalized cumulative optical depth; entry 256: total absolute optical depth
- `cdf[b]` = cumulative optical depth through bins 0..b, normalized to [0, 1]
- `cdf[num_bins]` = total absolute optical depth
- Written during compute pass (pass 2), read during accumulation pass (pass 1) of the *next* frame

## Key Technical Notes

- **Atomics in fragment shaders**: `atomicAdd` on `var<storage, read_write>` IS allowed in fragment shaders per WGSL spec. The underlying Vulkan feature `fragmentStoresAndAtomics` is widely supported on desktop.
- **CDF build via compute shader**: A single workgroup of 256 threads performs a parallel Hillis-Steele inclusive prefix sum, normalizes, writes CDF, and clears the histogram. Dispatched as `(1,1,1)` between the accum and composite render passes.
- **First frame**: CDF buffer initialized to linear fallback values `(1/256, 2/256, ..., 256/256)` with total optical depth = `257/256` on creation (`src/renderer.rs` `new()`). Histogram starts zeroed.
- **Temporal lag**: 1-frame delay is acceptable. CDF adapts smoothly as camera moves.
- **R8Unorm for revealage**: Well-tested, universally blendable. Multiplicative blend `(Zero, OneMinusSrc)` implements product of `(1 - alpha)`.
- **Rgba16Float for accum**: Supports blending without the `float32-blendable` feature.
- **Shader loading**: `format!("{}\n{}", common_wgsl, specific_wgsl)` concatenation since WGSL has no `#include`. Used in all three pipeline files via `include_str!`.
- **CDF buffer as non-atomic storage**: The CDF buffer is written by compute shader (pass 2) and read by fragment shader (pass 1 of next frame). Since these are in different passes of different frames, there's no synchronization issue. Use `var<storage, read_write>` with plain `f32` (not atomic).
- **wgpu 28 API**: `PipelineLayoutDescriptor` uses `immediate_size` (not `push_constant_ranges`), `RenderPipelineDescriptor` uses `multiview_mask: None` (not `multiview: None`), `RenderPassColorAttachment` requires `depth_slice: None`.

## 3DGS Mode
> Implemented in `src/ply.rs`, `src/splats.rs`, `src/pipeline/splat.rs`, `shaders/splat_*.wgsl`

```
cargo run --release -- splats/rem_v3_clear.ply
```

With a PLY argument the splat scene replaces the quad/mesh scene entirely; all three
transparency modes render it. Without an argument, nothing changes from the mesh demo.

### PLY parsing
> `src/ply.rs`

Two on-disk variants are handled, both `binary_little_endian`:

- **INRIA / original 3DGS**: `x,y,z`, `scale_0..2` (log space, exponentiated on load),
  `rot_0..3` (`w,x,y,z`, normalized), `opacity` (logit, sigmoid applied on load),
  `f_dc_0..2`, and optionally `f_rest_0..44` (channel-major: 15 coeffs of R, then G, then B).
- **PlayCanvas / SuperSplat compressed**: an `element chunk` of per-256-splat bounds plus
  four packed `u32` per vertex. Bit layouts match the PlayCanvas engine decoder:
  - `packed_position` / `packed_scale`: 11/10/11 unorm, lerped between the chunk's
    min/max bounds. Scale is still log space.
  - `packed_rotation`: 2-10-10-10 "largest three" — the top 2 bits name the omitted
    (largest) component, the rest are unorm mapped to `[-1/sqrt(2), 1/sqrt(2)]`.
  - `packed_color`: 8888. RGB lerps between the chunk's colour bounds and is *already* the
    evaluated DC band (not a raw `f_dc` coefficient); A is *already* sigmoid-applied opacity.
  - An optional `element sh` carries quantized higher bands (`byte * 8/255 - 4`).

The loader reports a clear error on a truncated file rather than producing garbage.

### Coordinate frame
3DGS reconstructions come out of COLMAP with +Y down and +Z into the screen. Positions and
covariances are flipped by `diag(1, -1, -1)` at load time so the existing orbit camera's
+Y-up convention is correct. The shader undoes that flip on the view direction before
evaluating SH, whose coefficients live in the original frame.

### GPU layout
- `SplatGpu`, 64 bytes: `vec4(pos.xyz, opacity)`, `vec4(cov_xx, cov_xy, cov_xz, _)`,
  `vec4(cov_yy, cov_yz, cov_zz, _)`, `vec4(color.rgb, _)`. The 3D covariance is precomputed
  on the CPU as `M M^T` where `M = flip * R(quat) * diag(scale)`.
- SH buffer: `array<f32>`, 45 floats per splat, channel-major. A one-float dummy is bound
  when the file has no higher bands, so the bind group layout is the same either way.
- Order buffer: `array<u32>`, instance index -> splat index.

Splats are stored **sorted by importance** (`opacity * cbrt(sx*sy*sz)`) so the render cap
can simply draw a prefix and get the best subset of that size.

### Rendering
> `shaders/splat_common.wgsl`

One instanced 4-vertex triangle strip per splat. The vertex shader does EWA splatting
(Zwicker et al., as used by Kerbl et al. 2023):

1. Transform the centre to view space; cull anything at or behind the near plane.
2. Build the projection Jacobian `J` (with the sample point clamped to 1.3x the frustum
   extent, so the affine approximation stays sane off-screen) and the view rotation `W`.
3. `cov2d = J W cov3d W^T J^T`, plus a `+0.3` low-pass on the diagonal so every splat
   covers about a pixel.
4. Eigen-decompose the 2x2 result to get an oriented 3-sigma quad — far fewer wasted
   fragments than an axis-aligned bound on elongated splats.
5. Evaluate SH for the view direction and emit the conic (inverse `cov2d`).

The fragment shader evaluates `alpha = opacity * exp(-0.5 * d^T conic d)`, discards below
`1/255`, and hands the result to whichever mode's output is compiled in. All three splat
pipelines disable depth testing (`depth_write: false`, `compare: Always`) but still declare
a depth-stencil state, because they draw into the same render passes as the mesh path.

### Sorting (mode 1 only)
> `src/splats.rs`

Modes 2 and 3 are order-independent by construction and draw in whatever order the buffer
holds. Mode 1 needs a real back-to-front sort every frame, which is done as a **16-bit
counting sort on a background thread**: view-space depth is quantized to `u16`, so the sort
is a single O(n) counting pass with no comparisons. Measured at ~6 ms for 921k splats.

The render thread never blocks on it — it keeps drawing the previous frame's order until a
new one arrives, and only one request is ever in flight (the worker drops stale requests and
sorts the newest camera). At orbit speeds the one-frame lag is not visible.

## Scene
> Implemented in `src/scene.rs` (scene setup + auto-rotation) and `src/mesh.rs` (geometry generators)

- **6 overlapping semi-transparent quads** at various depths and angles, each a different color with alpha 0.35-0.55 (`src/scene.rs` `new()`)
- **Toggleable meshes**: A transparent cube (`mesh::cube`) and UV-sphere (`mesh::uv_sphere`), slowly auto-rotating (`src/scene.rs` `update()`)
- **Camera**: Orbit camera using spherical coordinates (`src/camera.rs`)
  - `glam::Mat4::perspective_rh()` for projection
  - `glam::Mat4::look_at_rh()` for view
  - Spherical coords: `eye = target + distance * vec3(cos(pitch)*sin(yaw), sin(pitch), cos(pitch)*cos(yaw))`
- **Background**: Dark gray (0.1, 0.1, 0.1) clear color (set in each render pass in `src/renderer.rs`)

## Controls
> Implemented in `src/app.rs` `window_event()` handler

- `1` / `2` / `3`: Switch rendering mode (sets `renderer.mode`)
- `M`: Toggle mesh visibility (toggles `scene.show_meshes`)
- `A`: Toggle revealage vs. the `exp(-accum.a)` approximation
- `C`: (3DGS only) Cycle the render cap: 100% / 50% / 25% / 10% of the scene
- `[` / `]`: (3DGS only) Shrink / grow splats, for dialling overdraw up and down
- `R`: Reset the camera to its framing of the loaded scene
- `Escape`: Exit
- Mouse drag (left button): Orbit camera (`camera.on_mouse_move`)
- Scroll wheel: Zoom in/out (`camera.on_scroll`)
- Print current mode name to console on switch

## Implementation Order

1. **Scaffolding** - DONE: `Cargo.toml`, `src/main.rs` (event loop + env_logger), `src/app.rs` (ApplicationHandler)
2. **Core types** - DONE: `src/vertex.rs` (all 4 types + vertex buffer layout), `src/camera.rs` (orbit cam with drag/scroll)
3. **Renderer init** - DONE: `src/renderer.rs` (device/surface/queue, depth texture, bind group layouts, all buffer creation)
4. **Mesh generation** - DONE: `src/mesh.rs` (quad, cube, uv_sphere returning `Mesh { vertices, indices }`)
5. **Scene** - DONE: `src/scene.rs` (6 quads + cube + sphere, auto-rotation in `update()`, visibility filtering in `render()`)
6. **Mode 1** - DONE: `src/pipeline/alpha_blend.rs` + `shaders/alpha_blend.wgsl` + `shaders/common.wgsl`
7. **Mode 2** - DONE: `src/pipeline/naive_wboit.rs` + `shaders/wboit_accum.wgsl` + `shaders/wboit_composite.wgsl`
8. **Mode 3** - DONE: `src/pipeline/histogram_wboit.rs` + `shaders/histo_accum.wgsl` + `shaders/histo_composite.wgsl`
9. **Polish** - DONE: Input handling in `app.rs`, resize in `renderer.rs` `resize()`, mode switching, console output

## Verification

1. `cargo build` compiles clean - VERIFIED (zero warnings)
2. `cargo run` opens a window with transparent geometry
3. Press `1` - alpha blending with visible ordering artifacts
4. Press `2` - WBOIT (order-independent but static weights)
5. Press `3` - histogram-equalized WBOIT (global, better layer separation at clustered depths)
6. Press `M` - toggle meshes on/off
7. Mouse drag to orbit, verify all modes render correctly from different angles
8. Resize window - no crashes, all textures/buffers recreated
