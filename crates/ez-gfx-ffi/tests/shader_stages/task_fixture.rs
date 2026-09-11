#[cfg(any(not(test), not(target_vendor = "apple")))]
struct TaskFixture {
    native: TestContext,
    captures: Box<Captures>,
    compute_red: u64,
    compute_green: u64,
    task: u64,
    mesh: u64,
    task_fixed: u64,
    mesh_fixed: u64,
    fragment: u64,
    target: u64,
}

#[cfg(any(not(test), not(target_vendor = "apple")))]
impl TaskFixture {
    fn new(native: TestContext) -> Self {
        let mut captures = Box::<Captures>::default();
        assert_eq!(
            // SAFETY: captures outlives registration and is cleared in Drop.
            unsafe {
                ez_gfx_context_register_callback(
                    native.context,
                    Some(collect_readback),
                    (&raw mut *captures).cast(),
                )
            },
            EzGfxResult::Ok
        );
        let compute_red = load_stage!(
            ez_gfx_compute_shader_load,
            native.context,
            task_artifact(),
            b"compute_red"
        );
        let compute_green = load_stage!(
            ez_gfx_compute_shader_load,
            native.context,
            task_artifact(),
            b"compute_green"
        );
        let task = load_stage!(
            ez_gfx_task_shader_load,
            native.context,
            task_artifact(),
            b"taskmain"
        );
        let mesh = load_stage!(
            ez_gfx_mesh_shader_load,
            native.context,
            task_artifact(),
            b"meshmain"
        );
        let task_fixed = load_stage!(
            ez_gfx_task_shader_load,
            native.context,
            task_artifact(),
            b"taskmain_fixed"
        );
        let mesh_fixed = load_stage!(
            ez_gfx_mesh_shader_load,
            native.context,
            task_artifact(),
            b"meshmain_fixed"
        );
        let fragment = load_stage!(
            ez_gfx_fragment_shader_load,
            native.context,
            task_artifact(),
            b"fragmentmain_target"
        );
        Self {
            native,
            captures,
            compute_red,
            compute_green,
            task,
            mesh,
            task_fixed,
            mesh_fixed,
            fragment,
            target: 0,
        }
    }

    fn render_task(&mut self, compute: &[u8]) -> (EzGfxResult, Option<Vec<u8>>) {
        let (task, mesh) = (self.task, self.mesh);
        self.render_task_stages(compute, task, mesh, [1, 1, 1])
    }

    fn render_task_groups(&mut self, groups: [u32; 3]) -> (EzGfxResult, Option<Vec<u8>>) {
        let mut frame = 0;
        assert_ne!(self.target, 0, "offscreen target must be initialized");
        assert_eq!(
            // SAFETY: frame output storage is writable and aligned.
            unsafe {
                ez_gfx_render_target_frame_begin(self.native.context, self.target, &raw mut frame)
            },
            EzGfxResult::Ok
        );
        // SAFETY: the stage descriptor and state remain readable through the call.
        // Zero dimensions fail at the interface before binding or backend delegation.
        let status = unsafe {
            ez_gfx_frame_execute_mesh(
                self.native.context,
                frame,
                &EzGfxMeshShaders {
                    task_shader: self.task,
                    mesh_shader: self.mesh,
                    fragment_shader: self.fragment,
                },
                groups[0],
                groups[1],
                groups[2],
                &raw const OPAQUE_STATE,
            )
        };
        assert_eq!(
            status,
            EzGfxResult::InvalidArgument,
            "rejected execute status"
        );
        let readbacks = self.captures.0.len();
        assert_eq!(
            ez_gfx_frame_end(self.native.context, frame),
            EzGfxResult::NotReady,
            "rejected execute must leave no submittable frame"
        );
        assert_eq!(
            self.captures.0.len(),
            readbacks,
            "failed frame must not read back"
        );
        (status, None)
    }

    fn render_task_stages(
        &mut self,
        compute: &[u8],
        task: u64,
        mesh: u64,
        groups: [u32; 3],
    ) -> (EzGfxResult, Option<Vec<u8>>) {
        self.render_task_stages_inner(compute, task, mesh, groups)
    }

    fn render_task_stages_inner(
        &mut self,
        compute: &[u8],
        task: u64,
        mesh: u64,
        groups: [u32; 3],
    ) -> (EzGfxResult, Option<Vec<u8>>) {
        let shader = if compute == b"compute_red".as_slice() {
            self.compute_red
        } else {
            self.compute_green
        };
        let mut frame = 0;
        assert_ne!(self.target, 0, "offscreen target must be initialized");
        assert_eq!(
            // SAFETY: frame output storage is writable and aligned.
            unsafe {
                ez_gfx_render_target_frame_begin(self.native.context, self.target, &raw mut frame)
            },
            EzGfxResult::Ok
        );
        // The draws buffer is written by compute and read by task with no
        // CPU synchronization between the two records.
        let mut draws = 0;
        assert_eq!(
            // SAFETY: label and output storage remain live through the call.
            unsafe {
                ez_gfx_buffer_acquire(
                    self.native.context,
                    32,
                    1,
                    b"draws".as_ptr(),
                    b"draws".len(),
                    &raw mut draws,
                )
            },
            EzGfxResult::Ok
        );
        let binding = EzGfxBinding {
            name: b"draws".as_ptr(),
            name_length: b"draws".len(),
            buffer: draws,
            counter_buffer: 0,
            render_target: 0,
        };
        assert_eq!(
            // SAFETY: the binding stays readable through the call.
            unsafe { ez_gfx_frame_bind(self.native.context, frame, &raw const binding) },
            EzGfxResult::Ok
        );
        assert_eq!(
            ez_gfx_frame_execute_compute(self.native.context, frame, shader, 1, 1, 1),
            EzGfxResult::Ok
        );
        // SAFETY: the stage descriptor and state stay readable through the call.
        let status = unsafe {
            ez_gfx_frame_execute_mesh(
                self.native.context,
                frame,
                &EzGfxMeshShaders {
                    task_shader: task,
                    mesh_shader: mesh,
                    fragment_shader: self.fragment,
                },
                groups[0],
                groups[1],
                groups[2],
                &raw const OPAQUE_STATE,
            )
        };
        assert_eq!(status, EzGfxResult::Ok, "successful execute status");
        let mut request = 0;
        assert_eq!(
            // SAFETY: request output storage is writable and aligned.
            unsafe {
                ez_gfx_frame_enqueue_render_target_readback(
                    self.native.context,
                    frame,
                    self.target,
                    &raw mut request,
                )
            },
            EzGfxResult::Ok
        );
        let status = ez_gfx_frame_end(self.native.context, frame);
        let pixels =
            (status == EzGfxResult::Ok).then(|| take_readback(&mut self.captures, request));
        assert_eq!(
            ez_gfx_frame_end(self.native.context, frame),
            EzGfxResult::InvalidContext,
            "submitted frames retire exactly once"
        );
        (status, pixels)
    }

    fn render_task_after_owner_destroy(&mut self) -> (EzGfxResult, Option<Vec<u8>>) {
        // Fresh owners prove the recorded frame retains its own references:
        // destroying compute, task, mesh, and fragment shaders after
        // recording must still submit correctly.
        let compute = load_stage!(
            ez_gfx_compute_shader_load,
            self.native.context,
            task_artifact(),
            b"compute_red"
        );
        let task = load_stage!(
            ez_gfx_task_shader_load,
            self.native.context,
            task_artifact(),
            b"taskmain"
        );
        let mesh = load_stage!(
            ez_gfx_mesh_shader_load,
            self.native.context,
            task_artifact(),
            b"meshmain"
        );
        let fragment = load_stage!(
            ez_gfx_fragment_shader_load,
            self.native.context,
            task_artifact(),
            b"fragmentmain_target"
        );
        let mut frame = 0;
        assert_ne!(self.target, 0, "offscreen target must be initialized");
        assert_eq!(
            // SAFETY: frame output storage is writable and aligned.
            unsafe {
                ez_gfx_render_target_frame_begin(self.native.context, self.target, &raw mut frame)
            },
            EzGfxResult::Ok
        );
        let mut draws = 0;
        assert_eq!(
            // SAFETY: label and output storage remain live through the call.
            unsafe {
                ez_gfx_buffer_acquire(
                    self.native.context,
                    32,
                    1,
                    b"draws".as_ptr(),
                    b"draws".len(),
                    &raw mut draws,
                )
            },
            EzGfxResult::Ok
        );
        let binding = EzGfxBinding {
            name: b"draws".as_ptr(),
            name_length: b"draws".len(),
            buffer: draws,
            counter_buffer: 0,
            render_target: 0,
        };
        assert_eq!(
            // SAFETY: the binding stays readable through the call.
            unsafe { ez_gfx_frame_bind(self.native.context, frame, &raw const binding) },
            EzGfxResult::Ok
        );
        assert_eq!(
            ez_gfx_frame_execute_compute(self.native.context, frame, compute, 1, 1, 1),
            EzGfxResult::Ok
        );
        let shaders = EzGfxMeshShaders {
            task_shader: task,
            mesh_shader: mesh,
            fragment_shader: fragment,
        };
        // SAFETY: the stage descriptor and state stay readable through the call.
        let status = unsafe {
            ez_gfx_frame_execute_mesh(
                self.native.context,
                frame,
                &raw const shaders,
                1,
                1,
                1,
                &raw const OPAQUE_STATE,
            )
        };
        ez_gfx_compute_shader_destroy(self.native.context, compute);
        ez_gfx_task_shader_destroy(self.native.context, task);
        ez_gfx_mesh_shader_destroy(self.native.context, mesh);
        ez_gfx_fragment_shader_destroy(self.native.context, fragment);
        assert_eq!(status, EzGfxResult::Ok);
        let mut request = 0;
        assert_eq!(
            // SAFETY: request output storage is writable and aligned.
            unsafe {
                ez_gfx_frame_enqueue_render_target_readback(
                    self.native.context,
                    frame,
                    self.target,
                    &raw mut request,
                )
            },
            EzGfxResult::Ok
        );
        let status = ez_gfx_frame_end(self.native.context, frame);
        let pixels =
            (status == EzGfxResult::Ok).then(|| take_readback(&mut self.captures, request));
        (status, pixels)
    }
}

#[cfg(any(not(test), not(target_vendor = "apple")))]
impl Drop for TaskFixture {
    fn drop(&mut self) {
        // SAFETY: clearing a live registration retains no user-data pointer.
        let _ = unsafe {
            ez_gfx_context_register_callback(self.native.context, None, core::ptr::null_mut())
        };
        ez_gfx_compute_shader_destroy(self.native.context, self.compute_red);
        ez_gfx_compute_shader_destroy(self.native.context, self.compute_green);
        ez_gfx_task_shader_destroy(self.native.context, self.task);
        ez_gfx_mesh_shader_destroy(self.native.context, self.mesh);
        ez_gfx_task_shader_destroy(self.native.context, self.task_fixed);
        ez_gfx_mesh_shader_destroy(self.native.context, self.mesh_fixed);
        ez_gfx_fragment_shader_destroy(self.native.context, self.fragment);
        if self.target != 0 {
            ez_gfx_render_target_destroy(self.native.context, self.target);
        }
    }
}
