use super::math::{Mat4, from_gltf, identity, mul, transform_point};

#[derive(Clone, Debug)]
pub struct PrimitiveData {
    pub first_index: u32,
    pub index_count: u32,
    pub vertex_offset: u32,
    pub normal_offset: u32,
    pub transform: Mat4,
}

#[derive(Clone, Debug)]
pub struct MeshData {
    pub positions: Vec<[f32; 4]>,
    pub normals: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
    pub primitives: Vec<PrimitiveData>,
}

pub fn load_glb(bytes: &[u8]) -> Result<MeshData, String> {
    let gltf = gltf::Gltf::from_slice(bytes).map_err(|error| format!("decode GLB: {error}"))?;
    let blob = gltf
        .blob
        .as_deref()
        .ok_or_else(|| "GLB has no embedded binary buffer".to_owned())?;
    let mut result = MeshData {
        positions: Vec::new(),
        normals: Vec::new(),
        indices: Vec::new(),
        primitives: Vec::new(),
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
        append_node(node, identity(), blob, &mut result)?;
    }
    if result.primitives.is_empty() || result.positions.is_empty() || result.indices.is_empty() {
        return Err("GLB contains no indexed mesh primitives".to_owned());
    }
    normalize_scene(&mut result, 3.0)?;
    Ok(result)
}

fn normalize_scene(mesh: &mut MeshData, target_extent: f32) -> Result<(), String> {
    let mut minimum = [f32::INFINITY; 3];
    let mut maximum = [f32::NEG_INFINITY; 3];

    for (index, primitive) in mesh.primitives.iter().enumerate() {
        let end = mesh
            .primitives
            .get(index + 1)
            .map_or(mesh.positions.len(), |next| next.vertex_offset as usize);
        for position in &mesh.positions[primitive.vertex_offset as usize..end] {
            let world =
                transform_point(primitive.transform, [position[0], position[1], position[2]]);
            for axis in 0..3 {
                minimum[axis] = minimum[axis].min(world[axis]);
                maximum[axis] = maximum[axis].max(world[axis]);
            }
        }
    }

    let extent = [
        maximum[0] - minimum[0],
        maximum[1] - minimum[1],
        maximum[2] - minimum[2],
    ];
    let largest = extent.into_iter().fold(0.0_f32, f32::max);
    // Empty, degenerate, or non-finite bounds cannot produce a stable camera-space scene.
    if !target_extent.is_finite()
        || target_extent <= 0.0
        || !largest.is_finite()
        || largest <= f32::EPSILON
    {
        return Err("GLB mesh bounds cannot be normalized".to_owned());
    }

    let scale = target_extent / largest;
    let center = [
        (minimum[0] + maximum[0]) * 0.5,
        (minimum[1] + maximum[1]) * 0.5,
        (minimum[2] + maximum[2]) * 0.5,
    ];
    let normalization = [
        scale,
        0.0,
        0.0,
        -center[0] * scale,
        0.0,
        scale,
        0.0,
        -center[1] * scale,
        0.0,
        0.0,
        scale,
        -center[2] * scale,
        0.0,
        0.0,
        0.0,
        1.0,
    ];
    for primitive in &mut mesh.primitives {
        primitive.transform = mul(normalization, primitive.transform);
    }
    Ok(())
}

fn append_node(
    node: gltf::Node<'_>,
    parent: Mat4,
    blob: &[u8],
    result: &mut MeshData,
) -> Result<(), String> {
    let transform = mul(parent, from_gltf(node.transform().matrix()));
    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            append_primitive(primitive, transform, blob, result)?;
        }
    }
    for child in node.children() {
        append_node(child, transform, blob, result)?;
    }
    Ok(())
}

fn append_primitive(
    primitive: gltf::Primitive<'_>,
    transform: Mat4,
    blob: &[u8],
    result: &mut MeshData,
) -> Result<(), String> {
    let reader = primitive.reader(|buffer| match buffer.source() {
        gltf::buffer::Source::Bin => Some(blob),
        gltf::buffer::Source::Uri(_) => None,
    });
    let positions = reader
        .read_positions()
        .ok_or_else(|| "mesh primitive has no positions".to_owned())?
        .collect::<Vec<_>>();
    if positions.is_empty() {
        return Err("mesh primitive has no vertices".to_owned());
    }
    let normals = reader
        .read_normals()
        .map(Iterator::collect::<Vec<_>>)
        .unwrap_or_default();
    let local_indices = reader.read_indices().map_or_else(
        || (0..positions.len() as u32).collect(),
        |values| values.into_u32().collect::<Vec<_>>(),
    );
    if local_indices.is_empty()
        || local_indices
            .iter()
            .any(|index| *index as usize >= positions.len())
    {
        return Err("mesh primitive contains invalid indices".to_owned());
    }
    let vertex_offset = u32::try_from(result.positions.len())
        .map_err(|_| "vertex offset exceeds ABI".to_owned())?;
    let normal_offset =
        u32::try_from(result.normals.len()).map_err(|_| "normal offset exceeds ABI".to_owned())?;
    let first_index =
        u32::try_from(result.indices.len()).map_err(|_| "index offset exceeds ABI".to_owned())?;
    result.positions.extend(
        positions
            .iter()
            .map(|value| [value[0], value[1], value[2], 1.0]),
    );
    result.normals.extend((0..positions.len()).map(|index| {
        let value = normals.get(index).copied().unwrap_or([0.0, 1.0, 0.0]);
        [value[0], value[1], value[2], 0.0]
    }));
    result.indices.extend(local_indices.iter().copied());
    result.primitives.push(PrimitiveData {
        first_index,
        index_count: local_indices.len() as u32,
        vertex_offset,
        normal_offset,
        transform,
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_asset_preserves_mesh_geometry() {
        let mesh = load_glb(include_bytes!("../helmet.glb")).unwrap();
        assert!(!mesh.primitives.is_empty());
        assert_eq!(mesh.positions.len(), mesh.normals.len());
        assert!(
            mesh.primitives
                .iter()
                .all(|primitive| primitive.index_count > 0)
        );
    }
}
