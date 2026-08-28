use std::borrow::Cow;

use ez_gfx_ffi::EzGfxDrawIndexedCommand;

use crate::Example;

pub(crate) struct TextureData {
    pub bytes: Cow<'static, [u8]>,
    pub source_format: u8,
    pub width: u32,
    pub height: u32,
}

pub(crate) struct SceneData {
    pub positions: Vec<[f32; 4]>,
    pub colors: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
    pub commands: Vec<EzGfxDrawIndexedCommand>,
    pub texture: Option<TextureData>,
    pub compute: bool,
}

pub(crate) fn scene_data(example: Example) -> Result<SceneData, String> {
    match example {
        Example::Triangle => Ok(simple(
            vec![
                [-0.65, -0.55, 0.0, 1.0],
                [0.65, -0.55, 0.0, 1.0],
                [0.0, 0.65, 0.0, 1.0],
            ],
            vec![[1.0, 0.05, 0.02, 1.0]; 3],
            vec![0, 1, 2],
            None,
            false,
        )),
        Example::TexturedCube => {
            let (positions, indices) = cube();
            let colors = vec![[1.0; 4]; positions.len()];
            Ok(simple(
                positions,
                colors,
                indices,
                Some(TextureData {
                    bytes: Cow::Borrowed(include_bytes!("../assets/cube.png")),
                    source_format: 4,
                    width: 0,
                    height: 0,
                }),
                false,
            ))
        }
        Example::ComputeStructured => Ok(simple(
            vec![
                [-0.75, -0.5, 0.0, 1.0],
                [-0.05, -0.5, 0.0, 1.0],
                [-0.4, 0.5, 0.0, 1.0],
                [0.05, -0.5, 0.0, 1.0],
                [0.75, -0.5, 0.0, 1.0],
                [0.4, 0.5, 0.0, 1.0],
            ],
            vec![[0.0, 0.2, 0.8, 1.0]; 6],
            vec![0, 1, 2, 3, 4, 5],
            None,
            true,
        )),
        Example::ImGui => imgui(),
        Example::Helmet => model(include_bytes!("../assets/helmet.glb"), false),
        Example::SponzaKtx2 => model(include_bytes!("../assets/sponza.glb"), true),
    }
}

// Contiguous procedural geometry uses exactly one indirect command.
fn simple(
    positions: Vec<[f32; 4]>,
    colors: Vec<[f32; 4]>,
    indices: Vec<u32>,
    texture: Option<TextureData>,
    compute: bool,
) -> SceneData {
    let command = EzGfxDrawIndexedCommand {
        index_count: indices.len() as u32,
        instance_count: 1,
        first_index: 0,
        vertex_offset: 0,
        first_instance: 0,
    };
    SceneData {
        positions,
        colors,
        indices,
        commands: vec![command],
        texture,
        compute,
    }
}

// Face-local vertices preserve the original cube's UV corner convention.
fn cube() -> (Vec<[f32; 4]>, Vec<u32>) {
    let positions = vec![
        [-0.55, -0.55, 0.2, 1.0],
        [0.55, -0.55, 0.2, 1.0],
        [0.55, 0.55, 0.2, 1.0],
        [-0.55, 0.55, 0.2, 1.0],
        [0.55, -0.55, -0.2, 1.0],
        [-0.55, -0.55, -0.2, 1.0],
        [-0.55, 0.55, -0.2, 1.0],
        [0.55, 0.55, -0.2, 1.0],
        [-0.55, -0.55, -0.2, 1.0],
        [-0.55, -0.55, 0.2, 1.0],
        [-0.55, 0.55, 0.2, 1.0],
        [-0.55, 0.55, -0.2, 1.0],
        [0.55, -0.55, 0.2, 1.0],
        [0.55, -0.55, -0.2, 1.0],
        [0.55, 0.55, -0.2, 1.0],
        [0.55, 0.55, 0.2, 1.0],
        [-0.55, 0.55, 0.2, 1.0],
        [0.55, 0.55, 0.2, 1.0],
        [0.55, 0.55, -0.2, 1.0],
        [-0.55, 0.55, -0.2, 1.0],
        [-0.55, -0.55, -0.2, 1.0],
        [0.55, -0.55, -0.2, 1.0],
        [0.55, -0.55, 0.2, 1.0],
        [-0.55, -0.55, 0.2, 1.0],
    ];
    let mut indices = Vec::with_capacity(36);
    for face in 0..6_u32 {
        let base = face * 4;
        indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
    }
    (positions, indices)
}

// Actual ImGui draw lists supply the migrated UI vertex and index streams; empty output is rejected.
fn imgui() -> Result<SceneData, String> {
    let mut context = imgui::Context::create();
    context.io_mut().display_size = [640.0, 480.0];
    context.io_mut().delta_time = 1.0 / 60.0;
    let atlas = context.fonts().build_rgba32_texture();
    let texture = TextureData {
        bytes: Cow::Owned(atlas.data.to_vec()),
        source_format: 1,
        width: atlas.width,
        height: atlas.height,
    };
    let ui = context.frame();
    ui.window("ez_gfx_api2 ImGui")
        .position([80.0, 70.0], imgui::Condition::Always)
        .size([360.0, 220.0], imgui::Condition::Always)
        .build(|| {
            ui.text("Rust migration");
            ui.separator();
            ui.text("ImGui-generated indexed geometry");
        });
    let draw = context.render();
    let mut positions = Vec::new();
    let mut colors = Vec::new();
    let mut indices = Vec::new();
    let mut commands = Vec::new();
    for list in draw.draw_lists() {
        let vertex_base = positions.len() as u32;
        let index_base = indices.len() as u32;
        for vertex in list.vtx_buffer() {
            positions.push([
                vertex.pos[0] / 320.0 - 1.0,
                1.0 - vertex.pos[1] / 240.0,
                0.0,
                1.0,
            ]);
            colors.push([
                vertex.col[0] as f32 / 255.0,
                vertex.col[1] as f32 / 255.0,
                vertex.col[2] as f32 / 255.0,
                vertex.col[3] as f32 / 255.0,
            ]);
        }
        indices.extend(
            list.idx_buffer()
                .iter()
                .map(|index| vertex_base + u32::from(*index)),
        );
        for command in list.commands() {
            if let imgui::DrawCmd::Elements { count, cmd_params } = command {
                commands.push(EzGfxDrawIndexedCommand {
                    index_count: count as u32,
                    instance_count: 1,
                    first_index: index_base + cmd_params.idx_offset as u32,
                    vertex_offset: 0,
                    first_instance: 0,
                });
            }
        }
    }
    if commands.is_empty() {
        return Err("ImGui produced no draw commands".to_owned());
    }
    Ok(SceneData {
        positions,
        colors,
        indices,
        commands,
        texture: Some(texture),
        compute: false,
    })
}

// Embedded GLB buffer views are flattened while preserving one indirect draw per indexed primitive.
fn model(bytes: &[u8], ktx2: bool) -> Result<SceneData, String> {
    let gltf = gltf::Gltf::from_slice(bytes).map_err(|error| format!("decode GLB: {error}"))?;
    let blob = gltf
        .blob
        .as_deref()
        .ok_or_else(|| "GLB has no embedded binary buffer".to_owned())?;
    let mut positions = Vec::new();
    let mut colors = Vec::new();
    let mut indices = Vec::new();
    let mut commands = Vec::new();
    for mesh in gltf.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| match buffer.source() {
                gltf::buffer::Source::Bin => Some(blob),
                _ => None,
            });
            let Some(source_positions) = reader.read_positions() else {
                continue;
            };
            let vertex_base = positions.len() as u32;
            let first_index = indices.len() as u32;
            let normals = reader
                .read_normals()
                .map(Iterator::collect::<Vec<_>>)
                .unwrap_or_default();
            for (index, position) in source_positions.enumerate() {
                positions.push([position[0], position[1], position[2], 1.0]);
                let normal = normals.get(index).copied().unwrap_or([0.0, 0.0, 1.0]);
                colors.push([
                    normal[0] * 0.5 + 0.5,
                    normal[1] * 0.5 + 0.5,
                    normal[2] * 0.5 + 0.5,
                    1.0,
                ]);
            }
            let local = reader
                .read_indices()
                .map(|values| values.into_u32().collect::<Vec<_>>())
                .unwrap_or_else(|| (0..positions.len() as u32 - vertex_base).collect());
            indices.extend(local.iter().map(|index| vertex_base + *index));
            commands.push(EzGfxDrawIndexedCommand {
                index_count: local.len() as u32,
                instance_count: 1,
                first_index,
                vertex_offset: 0,
                first_instance: 0,
            });
        }
    }
    normalize(&mut positions)?;
    let texture = embedded_texture(&gltf, blob, ktx2)?;
    Ok(SceneData {
        positions,
        colors,
        indices,
        commands,
        texture,
        compute: true,
    })
}

// Only embedded image buffer views are accepted; URI I/O and unsupported encodings fail closed.
fn embedded_texture(
    gltf: &gltf::Gltf,
    blob: &[u8],
    require_ktx2: bool,
) -> Result<Option<TextureData>, String> {
    let mut fallback = None;
    for image in gltf.images() {
        let gltf::image::Source::View { view, mime_type } = image.source() else {
            continue;
        };
        let source_format = match mime_type {
            "image/png" => 4,
            "image/jpeg" => 3,
            "image/ktx2" => 6,
            _ => continue,
        };
        let end = view
            .offset()
            .checked_add(view.length())
            .ok_or_else(|| "embedded image range overflow".to_owned())?;
        let bytes = blob
            .get(view.offset()..end)
            .ok_or_else(|| "embedded image exceeds GLB buffer".to_owned())?;
        let texture = TextureData {
            bytes: Cow::Owned(bytes.to_vec()),
            source_format,
            width: 0,
            height: 0,
        };
        if source_format == 6 {
            return Ok(Some(texture));
        }
        if fallback.is_none() {
            fallback = Some(texture);
        }
    }
    if require_ktx2 {
        Err("GLB contains no embedded KTX2 image".to_owned())
    } else {
        Ok(fallback)
    }
}

// Non-finite or degenerate bounds fail; valid models are centered and uniformly scaled into clip space.
fn normalize(positions: &mut [[f32; 4]]) -> Result<(), String> {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for position in positions.iter() {
        if !position[..3].iter().all(|value| value.is_finite()) {
            return Err("model contains non-finite positions".to_owned());
        }
        for axis in 0..3 {
            min[axis] = min[axis].min(position[axis]);
            max[axis] = max[axis].max(position[axis]);
        }
    }
    let extent = (0..3)
        .map(|axis| max[axis] - min[axis])
        .fold(0.0_f32, f32::max);
    if positions.is_empty() || !extent.is_finite() || extent <= 0.0 {
        return Err("model has degenerate bounds".to_owned());
    }
    let center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];
    for position in positions {
        for axis in 0..3 {
            position[axis] = (position[axis] - center[axis]) / extent * 1.6;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every migrated source maps to complete indexed geometry; model paths retain multiple primitives.
    #[test]
    fn every_scene_mapping_produces_complete_geometry() {
        for example in [
            Example::Triangle,
            Example::TexturedCube,
            Example::ComputeStructured,
            Example::ImGui,
        ] {
            let scene = scene_data(example).unwrap();
            assert!(!scene.positions.is_empty(), "{example:?}");
            assert_eq!(scene.positions.len(), scene.colors.len(), "{example:?}");
            assert!(!scene.indices.is_empty(), "{example:?}");
            assert!(!scene.commands.is_empty(), "{example:?}");
            assert!(scene.commands.iter().all(|command| command.index_count > 0));
        }
        for example in [Example::Helmet, Example::SponzaKtx2] {
            let scene = std::thread::Builder::new()
                .stack_size(16 * 1024 * 1024)
                .spawn(move || scene_data(example))
                .unwrap()
                .join()
                .unwrap()
                .unwrap();
            assert_eq!(scene.positions.len(), scene.colors.len(), "{example:?}");
            assert!(!scene.commands.is_empty(), "{example:?}");
        }
    }

    // Only the mapped compute/model examples request compute work; texture formats remain source-specific.
    #[test]
    fn scene_features_match_original_examples() {
        assert!(!scene_data(Example::Triangle).unwrap().compute);
        assert!(scene_data(Example::ComputeStructured).unwrap().compute);
        let helmet = std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| scene_data(Example::Helmet))
            .unwrap()
            .join()
            .unwrap()
            .unwrap();
        let sponza = std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| scene_data(Example::SponzaKtx2))
            .unwrap()
            .join()
            .unwrap()
            .unwrap();
        assert!(helmet.compute);
        assert_eq!(
            scene_data(Example::TexturedCube)
                .unwrap()
                .texture
                .unwrap()
                .source_format,
            4
        );
        assert_eq!(sponza.texture.unwrap().source_format, 6);
    }
}
