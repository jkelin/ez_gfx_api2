//! Hidden main-thread Metal presentation runner.
//!
//! `AppKit` surfaces can only be created on the main thread, while libtest and
//! nextest workers run on spawned threads. Each `metal_present` scenario
//! therefore runs here, one scenario per process, started by
//! `tests/metal_present.rs` through `CARGO_BIN_EXE_metal_present`. The runner
//! stays headless: its `NSView` is never attached to a window, shown, or
//! activated. Usage: `metal_present <artifact> <output> <cache 0|1>`; the
//! presented bytes are written to `<output>` when caching is enabled.
#[cfg(target_vendor = "apple")]
#[path = "../../tests/metal_present/collect.rs"]
mod collect;

#[cfg(target_vendor = "apple")]
mod apple {
    use super::collect;
    use core::mem::size_of;
    use ez_gfx_ffi::{
        EzGfxBackendContextDesc, EzGfxBinding, EzGfxDrawIndexedCommand, EzGfxDynamicState,
        EzGfxResult, EzGfxWindowSurfaceDesc, ez_gfx_compute_shader_destroy,
        ez_gfx_compute_shader_load, ez_gfx_context_create_backend, ez_gfx_context_destroy,
        ez_gfx_context_init_device, ez_gfx_context_register_callback, ez_gfx_context_wait_idle,
        ez_gfx_counter_buffer_acquire, ez_gfx_counter_buffer_write_draws,
        ez_gfx_fragment_shader_destroy, ez_gfx_fragment_shader_load, ez_gfx_frame_begin,
        ez_gfx_frame_bind, ez_gfx_frame_end, ez_gfx_frame_execute_compute,
        ez_gfx_frame_execute_graphics, ez_gfx_index_allocation_create,
        ez_gfx_index_allocation_get_range, ez_gfx_surface_create_window, ez_gfx_surface_destroy,
        ez_gfx_value_buffer_acquire, ez_gfx_vertex_shader_destroy, ez_gfx_vertex_shader_load,
    };
    use objc2::{MainThreadMarker, rc::Retained};
    use objc2_app_kit::NSView;
    use objc2_core_foundation::CGSize;
    use objc2_quartz_core::CAMetalLayer;

    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 64;

    fn bind_params(context: u64, frame: u64, values: &[f32; 8]) {
        let name = b"params";
        let mut buffer = 0;
        assert_eq!(
            // SAFETY: Value, name, and output ranges remain live for the call.
            unsafe {
                ez_gfx_value_buffer_acquire(
                    context,
                    values.as_ptr().cast(),
                    u32::try_from(core::mem::size_of_val(values)).unwrap(),
                    name.as_ptr(),
                    name.len(),
                    &raw mut buffer,
                )
            },
            EzGfxResult::Ok
        );
        let binding = EzGfxBinding {
            name: name.as_ptr(),
            name_length: name.len(),
            buffer,
            counter_buffer: 0,
            render_target: 0,
        };
        assert_eq!(
            // SAFETY: The binding and name remain readable through the call.
            unsafe { ez_gfx_frame_bind(context, frame, &raw const binding) },
            EzGfxResult::Ok
        );
    }

    fn submit_render_nodes(
        context: u64,
        frame: u64,
        compute_shader: u64,
        vertex_shader: u64,
        fragment_shader: u64,
        indirect: u64,
    ) {
        let left = [-0.45_f32, 0.0, 0.2, 0.0, 1.0, 0.0, 0.0, 0.5];
        let right = [0.45_f32, 0.0, 0.2, 0.0, 0.0, 1.0, 0.0, 1.0];
        let occluded = [-0.45_f32, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0, 1.0];
        let alpha_blend = EzGfxDynamicState {
            cull_mode: 0,
            front_face: 0,
            primitive_type: 0,
            blend_mode: 1,
        };
        bind_params(context, frame, &left);
        assert_eq!(
            // SAFETY: The state remains readable for the call.
            unsafe {
                ez_gfx_frame_execute_graphics(
                    context,
                    frame,
                    vertex_shader,
                    fragment_shader,
                    indirect,
                    &raw const alpha_blend,
                )
            },
            EzGfxResult::Ok
        );
        assert_eq!(
            ez_gfx_frame_execute_compute(context, frame, compute_shader, 1, 1, 1),
            EzGfxResult::Ok
        );
        bind_params(context, frame, &right);
        assert_eq!(
            // SAFETY: Null state selects the default dynamic state.
            unsafe {
                ez_gfx_frame_execute_graphics(
                    context,
                    frame,
                    vertex_shader,
                    fragment_shader,
                    indirect,
                    core::ptr::null(),
                )
            },
            EzGfxResult::Ok
        );
        bind_params(context, frame, &occluded);
        assert_eq!(
            // SAFETY: Null state selects the default dynamic state.
            unsafe {
                ez_gfx_frame_execute_graphics(
                    context,
                    frame,
                    vertex_shader,
                    fragment_shader,
                    indirect,
                    core::ptr::null(),
                )
            },
            EzGfxResult::Ok
        );
        assert_eq!(ez_gfx_frame_end(context, frame), EzGfxResult::Ok);
        assert_eq!(ez_gfx_context_wait_idle(context), EzGfxResult::Ok);
    }

    // The allocated view and its pre-existing layer outlive surface destruction.
    fn alloc_view() -> Retained<NSView> {
        let main_thread =
            MainThreadMarker::new().expect("metal_present must run on AppKit's main thread");
        let view = NSView::new(main_thread);
        let extent = CGSize {
            width: f64::from(WIDTH),
            height: f64::from(HEIGHT),
        };
        let layer = CAMetalLayer::new();
        layer.setContentsScale(1.0);
        layer.setDrawableSize(extent);
        view.setFrameSize(extent);
        view.setWantsLayer(true);
        view.setLayer(Some(&layer));
        view
    }
    fn teardown(view: Retained<NSView>, context: u64, surface: u64) {
        let surface_status = ez_gfx_surface_destroy(context, surface);
        let context_status = ez_gfx_context_destroy(context);
        match (surface_status, context_status) {
            (EzGfxResult::Ok, EzGfxResult::Ok) => {
                // Both teardown calls drained, so `view` may drop on return.
            }
            (surface, context) => {
                // Either teardown left borrowed natives unproven: retain the host process-long.
                core::mem::forget(view);
                panic!("metal_present teardown failed: surface={surface:?} context={context:?}");
            }
        }
    }

    fn render(artifact: &[u8], cache_presented_snapshots: bool) -> Vec<u8> {
        // The windowless view is never shown or activated.
        let view = alloc_view();

        let context_desc = EzGfxBackendContextDesc {
            enable_debug: 0,
            enable_validation: 0,
            backend: 3,
            texture_decode_workers: 0,
            adapter_count: 0,
            adapter: core::ptr::null(),
        };
        let surface_desc = EzGfxWindowSurfaceDesc {
            system: 4,
            cache_presented_snapshots: u8::from(cache_presented_snapshots),
            reserved: [0; 6],
            handle_a: Retained::as_ptr(&view) as usize as u64,
            handle_b: 0,
        };
        let mut context = 0;
        let mut surface = 0;
        assert_eq!(
            {
                // SAFETY: Non-null arguments use live runner-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
                unsafe { ez_gfx_context_create_backend(&raw const context_desc, &raw mut context) }
            },
            EzGfxResult::Ok
        );
        assert_eq!(
            {
                // SAFETY: Non-null arguments use live runner-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
                unsafe {
                    ez_gfx_surface_create_window(context, &raw const surface_desc, &raw mut surface)
                }
            },
            EzGfxResult::Ok
        );
        assert_eq!(
            ez_gfx_context_init_device(context, surface),
            EzGfxResult::Ok
        );
        let mut collected = collect::Collected::default();
        assert_eq!(
            // SAFETY: `collected` remains alive until registration is explicitly cleared.
            unsafe {
                ez_gfx_context_register_callback(
                    context,
                    Some(collect::collect_event),
                    (&raw mut collected).cast(),
                )
            },
            EzGfxResult::Ok
        );

        let mut compute_shader = 0;
        let mut vertex_shader = 0;
        let mut fragment_shader = 0;
        for (entry, output, load) in [
            (
                b"computemain".as_slice(),
                &raw mut compute_shader,
                ez_gfx_compute_shader_load as unsafe extern "C" fn(_, _, _, _, _, _) -> _,
            ),
            (
                b"vertexmain".as_slice(),
                &raw mut vertex_shader,
                ez_gfx_vertex_shader_load as unsafe extern "C" fn(_, _, _, _, _, _) -> _,
            ),
            (
                b"fragmentmain".as_slice(),
                &raw mut fragment_shader,
                ez_gfx_fragment_shader_load as unsafe extern "C" fn(_, _, _, _, _, _) -> _,
            ),
        ] {
            assert_eq!(
                // SAFETY: Artifact, exact entry name, and output storage remain live.
                unsafe {
                    load(
                        context,
                        artifact.as_ptr(),
                        artifact.len(),
                        entry.as_ptr(),
                        entry.len(),
                        output,
                    )
                },
                EzGfxResult::Ok
            );
        }
        let label = b"metal-present";

        let indices = [0_u32, 1, 2];
        let mut index_allocation = 0;
        assert_eq!(
            {
                // SAFETY: Non-null arguments use live runner-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
                unsafe {
                    ez_gfx_index_allocation_create(
                        context,
                        indices.as_ptr().cast(),
                        u32::try_from(indices.len()).unwrap(),
                        &raw mut index_allocation,
                    )
                }
            },
            EzGfxResult::Ok
        );
        let mut first_index = 0;
        let mut index_count = 0;
        assert_eq!(
            {
                // SAFETY: Non-null arguments use live runner-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
                unsafe {
                    ez_gfx_index_allocation_get_range(
                        context,
                        index_allocation,
                        &raw mut first_index,
                        &raw mut index_count,
                    )
                }
            },
            EzGfxResult::Ok
        );
        assert_eq!(index_count, 3);
        let mut frame = 0;
        assert_eq!(
            // SAFETY: frame output storage is live and aligned.
            unsafe { ez_gfx_frame_begin(context, surface, &raw mut frame) },
            EzGfxResult::Ok
        );
        let mut indirect = 0;
        assert_eq!(
            {
                // SAFETY: Non-null arguments use live runner-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
                unsafe {
                    ez_gfx_counter_buffer_acquire(
                        context,
                        u32::try_from(size_of::<EzGfxDrawIndexedCommand>()).unwrap(),
                        1,
                        label.as_ptr(),
                        label.len(),
                        &raw mut indirect,
                    )
                }
            },
            EzGfxResult::Ok
        );
        let command = EzGfxDrawIndexedCommand {
            index_count: 3,
            instance_count: 1,
            first_index,
            vertex_offset: 0,
            first_instance: 0,
        };
        assert_eq!(
            {
                // SAFETY: Non-null arguments use live runner-owned storage with the export contract's required size, alignment, and access; nulls intentionally exercise checked rejection.
                unsafe {
                    ez_gfx_counter_buffer_write_draws(context, indirect, 0, &raw const command, 1)
                }
            },
            EzGfxResult::Ok
        );

        submit_render_nodes(
            context,
            frame,
            compute_shader,
            vertex_shader,
            fragment_shader,
            indirect,
        );

        let bytes = if cache_presented_snapshots {
            let bytes = collected
                .readback
                .take()
                .expect("presented readback event delivered");
            assert_eq!(bytes.len(), WIDTH as usize * HEIGHT as usize * 4);
            bytes
        } else {
            assert!(collected.readback.is_none());
            Vec::new()
        };
        assert_eq!(
            // SAFETY: clearing a live registration retains no user-data pointer.
            unsafe { ez_gfx_context_register_callback(context, None, core::ptr::null_mut()) },
            EzGfxResult::Ok
        );

        ez_gfx_compute_shader_destroy(context, compute_shader);
        ez_gfx_vertex_shader_destroy(context, vertex_shader);
        ez_gfx_fragment_shader_destroy(context, fragment_shader);
        teardown(view, context, surface);
        bytes
    }

    pub(super) fn main() {
        let mut args = std::env::args_os();
        let _ = args.next();
        let (Some(artifact), Some(output), Some(cache)) = (args.next(), args.next(), args.next())
        else {
            eprintln!("usage: metal_present <artifact> <output> <cache 0|1>");
            std::process::exit(2);
        };
        if args.next().is_some() {
            eprintln!("usage: metal_present <artifact> <output> <cache 0|1>");
            std::process::exit(2);
        }
        let caching = match cache.to_str() {
            Some("1") => true,
            Some("0") => false,
            _ => {
                eprintln!("metal_present: cache must be 0 or 1");
                std::process::exit(2);
            }
        };
        let artifact = std::fs::read(&artifact).expect("read shader artifact");
        let bytes = render(&artifact, caching);
        // The launcher asserts the payload; an empty uncached file still proves clean teardown.
        std::fs::write(&output, &bytes).expect("write presented bytes");
    }
}

fn main() {
    #[cfg(target_vendor = "apple")]
    apple::main();
    #[cfg(not(target_vendor = "apple"))]
    {
        eprintln!("metal_present: Apple Metal host required");
        std::process::exit(2);
    }
}
