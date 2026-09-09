use super::input::SceneInput;
use glam::{DVec2, Mat3, Mat4, Vec3};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClipY {
    Vulkan,
    Dx12,
    Metal,
}

pub fn perspective(
    fovy: f32,
    aspect: f32,
    near: f32,
    far: f32,
    clip_y: ClipY,
) -> crate::shared::Result<Mat4> {
    // Zero, reversed, non-finite, and half-turn projection bounds would produce infinities or inverted depth.
    if ![fovy, aspect, near, far].into_iter().all(f32::is_finite)
        || fovy <= 0.0
        || fovy >= core::f32::consts::PI
        || aspect <= 0.0
        || near <= 0.0
        || far <= near
    {
        return Err(crate::shared::Error::message(format!(
            "invalid perspective parameters"
        )));
    }

    let focal = 1.0 / (fovy * 0.5).tan();
    let vertical = if matches!(clip_y, ClipY::Vulkan | ClipY::Dx12) {
        -focal
    } else {
        focal
    };
    let depth = far / (near - far);
    let depth_offset = (far * near) / (near - far);

    Ok(Mat4::from_cols_array_2d(&[
        [focal / aspect, 0.0, 0.0, 0.0],
        [0.0, vertical, 0.0, 0.0],
        [0.0, 0.0, depth, -1.0],
        [0.0, 0.0, depth_offset, 0.0],
    ]))
}

pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> crate::shared::Result<Mat4> {
    let forward = target - eye;
    // Coincident points, non-finite coordinates, and a parallel up vector do not define a camera basis.
    if !eye.is_finite()
        || !target.is_finite()
        || !up.is_finite()
        || forward.length_squared() <= f32::EPSILON * f32::EPSILON
        || forward.cross(up).length_squared() <= f32::EPSILON * f32::EPSILON
    {
        return Err(crate::shared::Error::message(format!(
            "cannot construct a degenerate look-at matrix"
        )));
    }
    let forward = forward / forward.length();
    let side = forward.cross(up);
    let side = side / side.length();
    let camera_up = side.cross(forward);

    Ok(Mat4::from_cols_array_2d(&[
        [side.x, camera_up.x, -forward.x, 0.0],
        [side.y, camera_up.y, -forward.y, 0.0],
        [side.z, camera_up.z, -forward.z, 0.0],
        [-side.dot(eye), -camera_up.dot(eye), forward.dot(eye), 1.0],
    ]))
}

pub fn normal_transform(matrix: Mat4) -> crate::shared::Result<Mat3> {
    let linear = Mat3::from_mat4(matrix);
    let determinant = linear.determinant();
    // Singular and non-finite transforms have no inverse-transpose normal transform.
    if !matrix.is_finite() || !determinant.is_finite() || determinant.abs() <= f32::EPSILON {
        return Err(crate::shared::Error::message(format!(
            "cannot transform normals with a singular matrix"
        )));
    }
    Ok(linear.inverse().transpose())
}

pub fn from_gltf(matrix: [[f32; 4]; 4]) -> Mat4 {
    // glTF and glam both expose column arrays, including translation in the fourth column.
    Mat4::from_cols_array_2d(&matrix)
}

pub fn row_major(matrix: Mat4) -> Mat4 {
    // A transposed glam matrix has row-major mathematical values in its contiguous column-major bytes.
    matrix.transpose()
}

#[derive(Clone, Copy, Debug)]
pub struct OrbitCamera {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    fovy: f32,
    near: f32,
    far: f32,
    clip_y: ClipY,
    cursor: Option<DVec2>,
    dragging: bool,
}

impl OrbitCamera {
    pub const fn new(yaw: f32, pitch: f32, distance: f32) -> Self {
        // Inputs are retained exactly so callers can choose their initial orbit without hidden normalization.
        // The default frustum matches the 60-degree, 0.1-to-100 setup previously duplicated in examples,
        // and the default clip matches the non-Apple host backend; examples override both via builders.
        Self {
            yaw,
            pitch,
            distance,
            fovy: 60.0_f32.to_radians(),
            near: 0.1,
            far: 100.0,
            clip_y: ClipY::Vulkan,
            cursor: None,
            dragging: false,
        }
    }

    /// Overrides the default projection frustum; returns the updated camera for chaining.
    pub const fn with_frustum(mut self, fovy: f32, near: f32, far: f32) -> Self {
        self.fovy = fovy;
        self.near = near;
        self.far = far;
        self
    }

    /// Overrides the default clip-space convention; returns the updated camera for chaining.
    pub const fn with_clip_y(mut self, clip_y: ClipY) -> Self {
        self.clip_y = clip_y;
        self
    }

    /// Applies one window-frame event to the orbit state.
    ///
    /// Returns whether the event was consumed; cursor, primary-button, and scroll
    /// events are consumed, everything else (keys, text) is left for the caller.
    pub fn handle_window_event(&mut self, event: SceneInput) -> bool {
        match event {
            SceneInput::CursorMoved { x, y } => {
                self.cursor(DVec2::new(x, y));
                true
            }
            SceneInput::PrimaryButton(dragging) => {
                self.set_dragging(dragging);
                true
            }
            SceneInput::ScrollLines(lines) => {
                self.zoom(lines);
                true
            }
            _ => false,
        }
    }

    /// Applies a whole window frame's events; returns how many were consumed.
    ///
    /// Unconsumed events remain the caller's responsibility.
    pub fn handle_window_events(&mut self, events: &[SceneInput]) -> usize {
        events
            .iter()
            .copied()
            .filter(|event| self.handle_window_event(*event))
            .count()
    }

    /// Builds the aspect-correct projection for the current swapchain size.
    ///
    /// Degenerate (zero-sized) frames propagate through [`perspective`] as errors
    /// instead of uploading NaNs.
    pub fn projection(&self, size: [u32; 2]) -> crate::shared::Result<Mat4> {
        perspective(
            self.fovy,
            size[0] as f32 / size[1] as f32,
            self.near,
            self.far,
            self.clip_y,
        )
    }

    pub fn cursor(&mut self, position: DVec2) {
        // The first cursor sample establishes a baseline; movement rotates only while dragging.
        if let Some(previous) = self.cursor
            && self.dragging
        {
            let delta = position - previous;
            self.rotate(delta.x, delta.y);
        }
        self.cursor = Some(position);
    }

    pub fn set_dragging(&mut self, dragging: bool) {
        // Releasing preserves the last cursor position so the next press has no discontinuity.
        self.dragging = dragging;
    }

    pub fn rotate(&mut self, delta_x: f64, delta_y: f64) {
        const MAX_PITCH: f32 = 80.0_f32.to_radians();
        let sensitivity = 0.18_f32.to_radians();
        // Pitch stays away from the look-at pole; finite input from winit preserves an unconstrained yaw.
        self.yaw -= delta_x as f32 * sensitivity;
        self.pitch = (self.pitch + delta_y as f32 * sensitivity).clamp(-MAX_PITCH, MAX_PITCH);
    }

    pub fn zoom(&mut self, lines: f32) {
        // Extreme wheel deltas saturate at useful distances instead of reaching zero or infinity.
        self.distance = (self.distance * 0.85_f32.powf(lines)).clamp(0.1, 100.0);
    }

    pub fn view(&self, target: Vec3) -> crate::shared::Result<Mat4> {
        let cos_pitch = self.pitch.cos();
        let eye = target
            + Vec3::new(
                self.distance * self.yaw.sin() * cos_pitch,
                self.distance * self.pitch.sin(),
                self.distance * self.yaw.cos() * cos_pitch,
            );
        // Invalid camera state propagates through look_at rather than uploading NaNs.
        look_at(eye, target, Vec3::Y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::bytes_of;
    use glam::{Mat3, Mat4, Quat, Vec3};

    fn assert_approx(left: f32, right: f32) {
        assert!((left - right).abs() <= 1.0e-5, "{left} != {right}");
    }

    fn assert_vec3_approx(left: Vec3, right: Vec3) {
        assert!(
            (left - right).abs().max_element() <= 1.0e-5,
            "{left:?} != {right:?}"
        );
    }

    #[test]
    fn perspective_preserves_backend_clip_y_and_zero_to_one_depth() {
        for (clip_y, expected_y) in [
            (ClipY::Vulkan, -1.0),
            (ClipY::Dx12, -1.0),
            (ClipY::Metal, 1.0),
        ] {
            let projection = perspective(90.0_f32.to_radians(), 1.0, 0.25, 100.0, clip_y).unwrap();
            assert_eq!(projection.y_axis.y.signum(), expected_y);

            let near = projection * Vec3::new(0.0, 0.0, -0.25).extend(1.0);
            let far = projection * Vec3::new(0.0, 0.0, -100.0).extend(1.0);
            assert_approx(near.z / near.w, 0.0);
            assert_approx(far.z / far.w, 1.0);
        }
    }

    #[test]
    fn perspective_and_look_at_reject_degenerate_inputs() {
        for arguments in [
            [0.0, 1.0, 0.1, 100.0],
            [f32::NAN, 1.0, 0.1, 100.0],
            [1.0, 0.0, 0.1, 100.0],
            [1.0, 1.0, 0.0, 100.0],
            [1.0, 1.0, 1.0, 1.0],
        ] {
            assert!(
                perspective(
                    arguments[0],
                    arguments[1],
                    arguments[2],
                    arguments[3],
                    ClipY::Vulkan,
                )
                .is_err()
            );
        }
        assert!(look_at(Vec3::ZERO, Vec3::ZERO, Vec3::Y).is_err());
        assert!(look_at(Vec3::ZERO, -Vec3::Z, -Vec3::Z).is_err());
    }

    #[test]
    fn look_at_and_orbit_camera_preserve_view_contract() {
        let view = look_at(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y).unwrap();
        assert_vec3_approx(view.transform_point3(Vec3::new(0.0, 0.0, 5.0)), Vec3::ZERO);
        assert_vec3_approx(view.transform_point3(Vec3::ZERO), Vec3::new(0.0, 0.0, -5.0));

        let mut camera = OrbitCamera::new(0.0, 0.0, 5.0);
        assert_eq!(camera.view(Vec3::ZERO).unwrap(), view);
        camera.rotate(0.0, 100_000.0);
        assert_approx(camera.pitch, 80.0_f32.to_radians());
        camera.zoom(f32::INFINITY);
        assert_approx(camera.distance, 0.1);
        camera.zoom(f32::NEG_INFINITY);
        assert_approx(camera.distance, 100.0);
    }

    #[test]
    fn glam_transforms_points_and_normals_without_translation_leakage() {
        let transform = Mat4::from_scale_rotation_translation(
            Vec3::new(2.0, 3.0, 4.0),
            Quat::from_rotation_z(90.0_f32.to_radians()),
            Vec3::new(5.0, 6.0, 7.0),
        );
        assert_vec3_approx(
            transform.transform_point3(Vec3::new(1.0, 0.0, 0.0)),
            Vec3::new(5.0, 8.0, 7.0),
        );

        let transformed = normal_transform(transform).unwrap() * Vec3::X;
        assert_vec3_approx(transformed.normalize(), Vec3::Y);
        assert!(normal_transform(Mat4::from_scale(Vec3::new(1.0, 0.0, 1.0))).is_err());
        assert_eq!(Mat3::from_mat4(Mat4::IDENTITY), Mat3::IDENTITY);
    }

    #[test]
    fn gltf_columns_convert_to_the_same_mathematical_transform() {
        let gltf = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [5.0, 6.0, 7.0, 1.0],
        ];
        assert_vec3_approx(
            from_gltf(gltf).transform_point3(Vec3::ONE),
            Vec3::new(7.0, 9.0, 11.0),
        );
    }

    #[test]
    fn glam_operations_preserve_composition_and_point_transform_contracts() {
        let left = Mat4::from_scale_rotation_translation(
            Vec3::new(2.0, 3.0, 4.0),
            Quat::from_rotation_x(0.37),
            Vec3::new(5.0, 6.0, 7.0),
        );
        let right = Mat4::from_scale_rotation_translation(
            Vec3::new(0.5, 0.25, 2.0),
            Quat::from_rotation_y(-0.91),
            Vec3::new(-2.0, 1.0, 3.0),
        );
        let point = Vec3::new(1.25, -2.5, 0.75);

        let composed = left * right;
        assert_vec3_approx(
            composed.transform_point3(point),
            left.transform_point3(right.transform_point3(point)),
        );
    }

    #[test]
    fn row_major_upload_bytes_match_slang_matrix_rows() {
        let matrix = Mat4::from_cols_array(&[
            1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0, 4.0, 8.0, 12.0, 16.0,
        ]);
        let uploaded = row_major(matrix);
        assert_eq!(
            bytes_of(&uploaded),
            bytemuck::cast_slice::<f32, u8>(&[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ])
        );
    }

    #[test]
    fn camera_consumes_orbit_events_and_leaves_keys_to_caller() {
        use super::super::input::{SceneInput, SceneKey};

        let mut camera = OrbitCamera::new(0.0, 0.0, 5.0);
        assert!(camera.handle_window_event(SceneInput::PrimaryButton(true)));
        let yaw_before = camera.yaw;
        assert!(camera.handle_window_event(SceneInput::CursorMoved { x: 10.0, y: 0.0 }));
        assert!(camera.handle_window_event(SceneInput::CursorMoved { x: 20.0, y: 0.0 }));
        assert_ne!(camera.yaw, yaw_before);
        let distance_before = camera.distance;
        assert!(camera.handle_window_event(SceneInput::ScrollLines(1.0)));
        assert!(camera.distance < distance_before);

        assert!(!camera.handle_window_event(SceneInput::Character('x')));
        assert!(!camera.handle_window_event(SceneInput::Key {
            key: SceneKey::Escape,
            pressed: true,
        }));

        let events = [
            SceneInput::ScrollLines(1.0),
            SceneInput::Key {
                key: SceneKey::Escape,
                pressed: true,
            },
        ];
        assert_eq!(camera.handle_window_events(&events), 1);
    }

    #[test]
    fn camera_projection_tracks_frame_size_and_rejects_degenerate_frames() {
        let camera = OrbitCamera::new(0.0, 0.0, 5.0).with_clip_y(ClipY::Vulkan);
        let square = camera.projection([480, 480]).unwrap();
        let wide = camera.projection([960, 480]).unwrap();
        assert_approx(square.x_axis.x, 2.0 * wide.x_axis.x);
        assert!(camera.projection([640, 0]).is_err());
        assert!(camera.projection([0, 480]).is_err());

        let metal = OrbitCamera::new(0.0, 0.0, 5.0).with_clip_y(ClipY::Metal);
        assert_eq!(square.y_axis.y.signum(), -1.0);
        assert_eq!(metal.projection([480, 480]).unwrap().y_axis.y.signum(), 1.0);
    }
}
