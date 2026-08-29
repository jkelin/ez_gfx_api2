pub type Mat4 = [f32; 16];

pub const fn identity() -> Mat4 {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

#[cfg(test)]
pub fn translation(value: [f32; 3]) -> Mat4 {
    let mut matrix = identity();
    matrix[3] = value[0];
    matrix[7] = value[1];
    matrix[11] = value[2];
    matrix
}

pub fn mul(left: Mat4, right: Mat4) -> Mat4 {
    let mut result = [0.0; 16];
    for row in 0..4 {
        for column in 0..4 {
            result[row * 4 + column] = (0..4)
                .map(|index| left[row * 4 + index] * right[index * 4 + column])
                .sum();
        }
    }
    result
}

pub fn transform_point(matrix: Mat4, point: [f32; 3]) -> [f32; 3] {
    [
        matrix[0] * point[0] + matrix[1] * point[1] + matrix[2] * point[2] + matrix[3],
        matrix[4] * point[0] + matrix[5] * point[1] + matrix[6] * point[2] + matrix[7],
        matrix[8] * point[0] + matrix[9] * point[1] + matrix[10] * point[2] + matrix[11],
    ]
}

pub fn from_gltf(matrix: [[f32; 4]; 4]) -> Mat4 {
    let mut result = [0.0; 16];
    for row in 0..4 {
        for column in 0..4 {
            result[row * 4 + column] = matrix[column][row];
        }
    }
    result
}

pub fn perspective(fovy: f32, aspect: f32, near: f32, far: f32) -> Result<Mat4, String> {
    // Projection boundaries must remain finite and ordered; invalid resize state is rejected by the runner.
    if ![fovy, aspect, near, far]
        .iter()
        .all(|value| value.is_finite())
        || fovy <= 0.0
        || fovy >= core::f32::consts::PI
        || aspect <= 0.0
        || near <= 0.0
        || far <= near
    {
        return Err("invalid perspective parameters".to_owned());
    }
    let focal = 1.0 / (fovy * 0.5).tan();
    // Metal's viewport convention already maps clip-space Y to the top-left drawable origin.
    let vertical = if cfg!(target_vendor = "apple") {
        focal
    } else {
        -focal
    };
    Ok([
        focal / aspect,
        0.0,
        0.0,
        0.0,
        0.0,
        vertical,
        0.0,
        0.0,
        0.0,
        0.0,
        far / (near - far),
        (far * near) / (near - far),
        0.0,
        0.0,
        -1.0,
        0.0,
    ])
}

pub fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> Result<Mat4, String> {
    let forward = normalize(sub(target, eye))?;
    let side = normalize(cross(forward, up))?;
    let camera_up = cross(side, forward);
    Ok([
        side[0],
        side[1],
        side[2],
        -dot(side, eye),
        camera_up[0],
        camera_up[1],
        camera_up[2],
        -dot(camera_up, eye),
        -forward[0],
        -forward[1],
        -forward[2],
        dot(forward, eye),
        0.0,
        0.0,
        0.0,
        1.0,
    ])
}

fn sub(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn dot(left: [f32; 3], right: [f32; 3]) -> f32 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn normalize(value: [f32; 3]) -> Result<[f32; 3], String> {
    let length = dot(value, value).sqrt();
    if !length.is_finite() || length <= f32::EPSILON {
        return Err("cannot normalize a degenerate vector".to_owned());
    }
    Ok([value[0] / length, value[1] / length, value[2] / length])
}

#[derive(Clone, Copy, Debug)]
pub struct OrbitCamera {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    cursor: Option<[f64; 2]>,
    dragging: bool,
}

impl OrbitCamera {
    pub const fn new(yaw: f32, pitch: f32, distance: f32) -> Self {
        Self {
            yaw,
            pitch,
            distance,
            cursor: None,
            dragging: false,
        }
    }

    pub fn cursor(&mut self, position: [f64; 2]) {
        if let Some(previous) = self.cursor
            && self.dragging
        {
            self.rotate(position[0] - previous[0], position[1] - previous[1]);
        }
        self.cursor = Some(position);
    }

    pub fn set_dragging(&mut self, dragging: bool) {
        self.dragging = dragging;
    }

    pub fn rotate(&mut self, delta_x: f64, delta_y: f64) {
        let sensitivity = 0.18_f32.to_radians();
        self.yaw -= delta_x as f32 * sensitivity;
        self.pitch = (self.pitch + delta_y as f32 * sensitivity)
            .clamp((-80.0_f32).to_radians(), 80.0_f32.to_radians());
    }

    pub fn zoom(&mut self, lines: f32) {
        self.distance = (self.distance * 0.85_f32.powf(lines)).clamp(0.1, 100.0);
    }

    pub fn view(&self, target: [f32; 3]) -> Result<Mat4, String> {
        let cos_pitch = self.pitch.cos();
        let eye = [
            target[0] + self.distance * self.yaw.sin() * cos_pitch,
            target[1] + self.distance * self.pitch.sin(),
            target[2] + self.distance * self.yaw.cos() * cos_pitch,
        ];
        look_at(eye, target, [0.0, 1.0, 0.0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_product_preserves_identity_and_translation() {
        let translation = translation([2.0, -3.0, 4.0]);
        assert_eq!(mul(identity(), translation), translation);
        assert_eq!(mul(translation, identity()), translation);
        assert_eq!(
            transform_point(translation, [1.0, 2.0, 3.0]),
            [3.0, -1.0, 7.0]
        );
    }

    #[test]
    fn perspective_and_look_at_reject_invalid_boundaries() {
        assert!(perspective(0.0, 1.0, 0.1, 100.0).is_err());
        assert!(perspective(1.0, 0.0, 0.1, 100.0).is_err());
        assert!(perspective(1.0, 1.0, 1.0, 1.0).is_err());
        assert!(look_at([0.0; 3], [0.0; 3], [0.0, 1.0, 0.0]).is_err());
    }

    #[test]
    fn orbit_defaults_match_original_examples() {
        let mut camera = OrbitCamera::new(35.0_f32.to_radians(), 22.0_f32.to_radians(), 5.0);
        camera.rotate(100.0, -100.0);
        assert!(camera.pitch <= 80.0_f32.to_radians());
        camera.zoom(1000.0);
        assert!(camera.distance >= 0.1);
        assert!(camera.view([0.0; 3]).is_ok());
    }
}
