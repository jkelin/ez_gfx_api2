use super::math::{from_gltf, row_major};

use glam::{Mat4, Vec3};

#[derive(Clone, Debug)]
pub struct PrimitiveData {
    pub first_index: u32,
    pub index_count: u32,
    pub vertex_offset: u32,
    pub normal_offset: u32,
    pub uv_offset: u32,
    pub transform: Mat4,
    pub image: Option<usize>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BasicPrimitive {
    pub first_index: u32,
    pub index_count: u32,
    pub vertex_offset: u32,
    pub normal_offset: u32,
    pub transform: Mat4,
}

#[derive(Clone, Debug)]
pub struct ImageData {
    pub bytes: Vec<u8>,
    pub mime_type: &'static str,
}

#[derive(Clone, Debug)]
pub struct MeshData {
    pub positions: Vec<[f32; 4]>,
    pub normals: Vec<[f32; 4]>,
    pub uvs: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
    pub primitives: Vec<PrimitiveData>,
    pub images: Vec<ImageData>,
}

pub fn basic_primitives(
    mesh: &MeshData,
    uploaded_first_index: u32,
) -> crate::shared::Result<Vec<BasicPrimitive>> {
    mesh.primitives
        .iter()
        .map(|primitive| {
            Ok(BasicPrimitive {
                first_index: primitive
                    .first_index
                    .checked_add(uploaded_first_index)
                    .ok_or_else(|| {
                        crate::shared::Error::message(format!("primitive index offset exceeds u32"))
                    })?,
                index_count: primitive.index_count,
                vertex_offset: primitive.vertex_offset,
                normal_offset: primitive.normal_offset,
                transform: row_major(primitive.transform),
            })
        })
        .collect()
}

pub fn load_geometry_glb(bytes: &[u8]) -> crate::shared::Result<MeshData> {
    load_glb(bytes, false)
}

pub fn load_textured_glb(bytes: &[u8]) -> crate::shared::Result<MeshData> {
    load_glb(bytes, true)
}

fn load_glb(bytes: &[u8], textured: bool) -> crate::shared::Result<MeshData> {
    let gltf = gltf::Gltf::from_slice(bytes)?;
    let blob = gltf.blob.as_deref().ok_or_else(|| {
        crate::shared::Error::message(format!("GLB has no embedded binary buffer"))
    })?;
    let (images, image_map) = if textured {
        embedded_images(&gltf, blob)?
    } else {
        (Vec::new(), Vec::new())
    };
    let mut result = MeshData {
        positions: Vec::new(),
        normals: Vec::new(),
        uvs: Vec::new(),
        indices: Vec::new(),
        primitives: Vec::new(),
        images,
    };
    let mut roots = gltf
        .default_scene()
        .into_iter()
        .flat_map(|scene| scene.nodes())
        .collect::<Vec<_>>();
    if roots.is_empty() {
        roots.extend(gltf.scenes().flat_map(|scene| scene.nodes()));
    }
    for node in roots {
        append_node(
            node,
            Mat4::IDENTITY,
            blob,
            textured,
            &image_map,
            &mut result,
        )?;
    }
    if result.primitives.is_empty() || result.positions.is_empty() || result.indices.is_empty() {
        return Err(crate::shared::Error::message(format!(
            "GLB contains no indexed mesh primitives"
        )));
    }
    normalize_scene(&mut result, 3.0)?;
    Ok(result)
}

fn normalize_scene(mesh: &mut MeshData, target_extent: f32) -> crate::shared::Result<()> {
    let mut minimum = Vec3::splat(f32::INFINITY);
    let mut maximum = Vec3::splat(f32::NEG_INFINITY);

    for (index, primitive) in mesh.primitives.iter().enumerate() {
        let end = mesh
            .primitives
            .get(index + 1)
            .map_or(mesh.positions.len(), |next| next.vertex_offset as usize);
        for position in &mesh.positions[primitive.vertex_offset as usize..end] {
            let world = primitive.transform.transform_point3(Vec3::new(
                position[0],
                position[1],
                position[2],
            ));
            minimum = minimum.min(world);
            maximum = maximum.max(world);
        }
    }

    let extent = maximum - minimum;
    let largest = extent.max_element();
    // Empty, degenerate, or non-finite bounds cannot produce a stable camera-space scene.
    if !target_extent.is_finite()
        || target_extent <= 0.0
        || !largest.is_finite()
        || largest <= f32::EPSILON
    {
        return Err(crate::shared::Error::message(format!(
            "GLB mesh bounds cannot be normalized"
        )));
    }

    let scale = target_extent / largest;
    let center = (minimum + maximum) * 0.5;
    let normalization = Mat4::from_cols_array_2d(&[
        [scale, 0.0, 0.0, 0.0],
        [0.0, scale, 0.0, 0.0],
        [0.0, 0.0, scale, 0.0],
        [-center.x * scale, -center.y * scale, -center.z * scale, 1.0],
    ]);
    for primitive in &mut mesh.primitives {
        primitive.transform = normalization * primitive.transform;
    }
    Ok(())
}

fn append_node(
    node: gltf::Node<'_>,
    parent: Mat4,
    blob: &[u8],
    textured: bool,
    image_map: &[Option<usize>],
    result: &mut MeshData,
) -> crate::shared::Result<()> {
    let transform = parent * from_gltf(node.transform().matrix());
    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            append_primitive(primitive, transform, blob, textured, image_map, result)?;
        }
    }
    for child in node.children() {
        append_node(child, transform, blob, textured, image_map, result)?;
    }
    Ok(())
}

fn append_primitive(
    primitive: gltf::Primitive<'_>,
    transform: Mat4,
    blob: &[u8],
    textured: bool,
    image_map: &[Option<usize>],
    result: &mut MeshData,
) -> crate::shared::Result<()> {
    let reader = primitive.reader(|buffer| match buffer.source() {
        gltf::buffer::Source::Bin => Some(blob),
        gltf::buffer::Source::Uri(_) => None,
    });
    let positions = reader
        .read_positions()
        .ok_or_else(|| crate::shared::Error::message(format!("mesh primitive has no positions")))?
        .collect::<Vec<_>>();
    if positions.is_empty() {
        return Err(crate::shared::Error::message(format!(
            "mesh primitive has no vertices"
        )));
    }
    let normals = reader
        .read_normals()
        .map(Iterator::collect::<Vec<_>>)
        .unwrap_or_default();
    let uvs = if textured {
        reader
            .read_tex_coords(0)
            .map(|values| values.into_f32().collect::<Vec<_>>())
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    let local_indices = reader
        .read_indices()
        .map(|values| values.into_u32().collect::<Vec<_>>())
        .unwrap_or_else(|| (0..positions.len() as u32).collect());
    if local_indices.is_empty()
        || local_indices
            .iter()
            .any(|index| *index as usize >= positions.len())
    {
        return Err(crate::shared::Error::message(format!(
            "mesh primitive contains invalid indices"
        )));
    }
    let vertex_offset = u32::try_from(result.positions.len())?;
    let normal_offset = u32::try_from(result.normals.len())?;
    let uv_offset = if textured {
        u32::try_from(result.uvs.len())?
    } else {
        0
    };
    let first_index = u32::try_from(result.indices.len())?;
    result.positions.extend(
        positions
            .iter()
            .map(|value| [value[0], value[1], value[2], 1.0]),
    );
    result.normals.extend((0..positions.len()).map(|index| {
        let value = normals.get(index).copied().unwrap_or([0.0, 1.0, 0.0]);
        [value[0], value[1], value[2], 0.0]
    }));
    if textured {
        result.uvs.extend((0..positions.len()).map(|index| {
            let value = uvs.get(index).copied().unwrap_or([0.0, 0.0]);
            [value[0], value[1], 0.0, 0.0]
        }));
    }
    result
        .indices
        .extend(local_indices.iter().map(|index| index + vertex_offset));
    let image = if textured {
        let texture = primitive
            .material()
            .pbr_metallic_roughness()
            .base_color_texture()
            .map(|view| view.texture());
        let image_index = texture.and_then(|texture| {
            texture.source().map(|image| image.index()).or_else(|| {
                texture
                    .extension_value("KHR_texture_basisu")
                    .and_then(|extension| extension.get("source"))
                    .and_then(|value| value.as_u64())
                    .and_then(|index| usize::try_from(index).ok())
            })
        });
        image_index.and_then(|index| image_map.get(index).copied().flatten())
    } else {
        None
    };
    result.primitives.push(PrimitiveData {
        first_index,
        index_count: local_indices.len() as u32,
        vertex_offset,
        normal_offset,
        uv_offset,
        transform,
        image,
    });
    Ok(())
}

fn embedded_images(
    gltf: &gltf::Gltf,
    blob: &[u8],
) -> crate::shared::Result<(Vec<ImageData>, Vec<Option<usize>>)> {
    let mut images = Vec::new();
    let mut map = vec![None; gltf.images().len()];
    for image in gltf.images() {
        let gltf::image::Source::View { view, mime_type } = image.source() else {
            return Err(crate::shared::Error::message(format!(
                "external GLB image URIs are unsupported"
            )));
        };
        let mime_type = match mime_type {
            "image/ktx2" => "image/ktx2",
            "image/png" => "image/png",
            "image/jpeg" => "image/jpeg",
            _ => {
                return Err(crate::shared::Error::message(format!(
                    "unsupported embedded image format `{mime_type}`"
                )));
            }
        };
        let end = view
            .offset()
            .checked_add(view.length())
            .ok_or_else(|| crate::shared::Error::message(format!("image range overflow")))?;
        let bytes = blob.get(view.offset()..end).ok_or_else(|| {
            crate::shared::Error::message(format!("embedded image exceeds GLB buffer"))
        })?;
        map[image.index()] = Some(images.len());
        images.push(ImageData {
            bytes: bytes.to_vec(),
            mime_type,
        });
    }
    Ok((images, map))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn assert_pod<T: bytemuck::Pod>() {}

    #[test]
    fn basic_primitives_apply_uploaded_index_base() {
        assert_pod::<BasicPrimitive>();
        let mut mesh = MeshData {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
            primitives: vec![PrimitiveData {
                first_index: 7,
                index_count: 9,
                vertex_offset: 11,
                normal_offset: 13,
                uv_offset: 17,
                transform: Mat4::IDENTITY,
                image: None,
            }],
            images: Vec::new(),
        };

        assert_eq!(
            basic_primitives(&mesh, 5).unwrap(),
            vec![BasicPrimitive {
                first_index: 12,
                index_count: 9,
                vertex_offset: 11,
                normal_offset: 13,
                transform: Mat4::IDENTITY,
            }]
        );

        mesh.primitives[0].first_index = u32::MAX;
        assert!(basic_primitives(&mesh, 1).is_err());
    }

    #[test]
    fn geometry_loading_omits_texture_payloads() {
        for bytes in [
            include_bytes!("assets/sponza.glb").as_slice(),
            include_bytes!("../05_helmet/helmet.glb").as_slice(),
        ] {
            let mesh = load_geometry_glb(bytes).unwrap();
            assert!(!mesh.primitives.is_empty());
            assert_eq!(mesh.positions.len(), mesh.normals.len());
            assert!(mesh.uvs.is_empty());
            assert!(mesh.images.is_empty());
            assert!(
                mesh.primitives
                    .iter()
                    .all(|primitive| primitive.image.is_none())
            );
        }
    }

    #[test]
    fn textured_loading_retains_sponza_payloads() {
        let mesh = load_textured_glb(include_bytes!("assets/sponza.glb")).unwrap();
        assert_eq!(mesh.positions.len(), mesh.uvs.len());
        assert!(!mesh.images.is_empty());
        assert!(
            mesh.primitives
                .iter()
                .any(|primitive| primitive.image.is_some())
        );
    }
}
