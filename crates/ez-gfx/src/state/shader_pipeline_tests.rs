#[cfg(windows)]
fn mesh_stage_test_artifact() -> Vec<u8> {
    use ez_gfx_artifact::{
        Artifact, Provenance, Stage, Target, TargetCompatibility, TargetVariant,
    };

    let stages = [
        (Stage::Task, "taskmain"),
        (Stage::Mesh, "meshmain"),
        (Stage::Fragment, "fragmentmain"),
        (Stage::Vertex, "vertexmain"),
    ];
    let variants = stages
        .into_iter()
        .map(|(stage, entry)| {
            TargetVariant::new(
                Target::Dxil,
                stage,
                entry,
                "ez-gfx-v1",
                TargetCompatibility::portable(Target::Dxil).unwrap(),
                vec![stage as u8 + 1],
            )
            .unwrap()
        })
        .collect();
    let metadata = br#"{"semantic_abi":1,"reflections":[
        {"target":"Dxil","entry":"taskmain","stage":"Task","reflection":{"parameters":[],"workgroup_size":[1,1,1]}},
        {"target":"Dxil","entry":"meshmain","stage":"Mesh","reflection":{"parameters":[],"workgroup_size":[1,1,1]}},
        {"target":"Dxil","entry":"fragmentmain","stage":"Fragment","reflection":{"parameters":[]}},
        {"target":"Dxil","entry":"vertexmain","stage":"Vertex","reflection":{"parameters":[]}}
    ]}"#;
    Artifact::new(
        metadata.to_vec(),
        Provenance::new("test", "1", vec![], "host"),
        variants,
    )
    .unwrap()
    .encode()
    .unwrap()
}

#[cfg(windows)]
#[test]
fn mesh_stage_validation_checks_owner_generation_and_exact_stage() {
    use ez_gfx_artifact::Stage;

    let context = initialized_shader_context();
    with_context_mut(context, |state| {
        state.shader_capabilities_override = Some(ShaderCapabilities {
            task: true,
            mesh: true,
        });
        Ok(())
    })
    .unwrap();
    let artifact = mesh_stage_test_artifact();
    let task = load_shader(context, &artifact, Stage::Task, "taskmain").unwrap();
    let mesh = load_shader(context, &artifact, Stage::Mesh, "meshmain").unwrap();
    let fragment = load_shader(context, &artifact, Stage::Fragment, "fragmentmain").unwrap();
    let vertex = load_shader(context, &artifact, Stage::Vertex, "vertexmain").unwrap();
    let stages = ez_gfx_hal::MeshStages {
        task: Some(task),
        mesh,
        fragment,
    };

    assert_eq!(validate_mesh_stages(context, stages), Ok(()));
    assert_eq!(
        validate_mesh_stages(
            context,
            ez_gfx_hal::MeshStages {
                task: Some(vertex),
                mesh,
                fragment,
            }
        ),
        Err(Error::InvalidContext)
    );

    let foreign = initialized_shader_context();
    assert_eq!(
        validate_mesh_stages(foreign, stages),
        Err(Error::Lifecycle(LifecycleError::WrongOwner))
    );
    assert_eq!(destroy_context(foreign), Ok(()));

    destroy_shader(context, task);
    assert_eq!(
        validate_mesh_stages(context, stages),
        Err(Error::Lifecycle(LifecycleError::StaleHandle))
    );
    for shader in [mesh, fragment, vertex] {
        destroy_shader(context, shader);
    }
    destroy_shader(context, mesh);
    assert_eq!(
        with_context_mut(context, |state| Ok(state.native_shader_destroys)).unwrap(),
        4
    );
    assert_eq!(destroy_context(context), Ok(()));
}

#[cfg(windows)]
fn mesh_test_render_target(context: ContextHandle, name: &str) -> RenderTargetHandle {
    use ez_gfx_runtime::target::{ClearValue, TargetDeclaration, TargetUsage};

    let declaration = TargetDeclaration::new(
        name,
        TargetUsage::Color,
        1.0,
        1,
        vec![Format::Rgba8Unorm],
        ClearValue::Color([0.0, 0.0, 0.0, 1.0]),
        true,
    )
    .unwrap();
    create_render_target(context, &declaration, 16, 16).unwrap()
}

#[cfg(windows)]
fn load_mesh_test_stages(
    context: ContextHandle,
    artifact: &[u8],
) -> ez_gfx_hal::MeshStages<ShaderHandle> {
    ez_gfx_hal::MeshStages {
        task: Some(
            load_shader(context, artifact, ez_gfx_artifact::Stage::Task, "taskmain").unwrap(),
        ),
        mesh: load_shader(context, artifact, ez_gfx_artifact::Stage::Mesh, "meshmain").unwrap(),
        fragment: load_shader(
            context,
            artifact,
            ez_gfx_artifact::Stage::Fragment,
            "fragmentmain",
        )
        .unwrap(),
    }
}

#[cfg(windows)]
#[test]
fn retained_mesh_stage_natives_destroy_once_after_submit_and_abort() {
    fn exercise(abort: bool) {
        let context = initialized_shader_context();
        inject_shader_capabilities(
            context,
            ShaderCapabilities {
                task: true,
                mesh: true,
            },
        )
        .unwrap();
        with_context_mut(context, |state| {
            state.raw_native_frame_test_probe.enabled = true;
            Ok(())
        })
        .unwrap();
        let artifact = mesh_stage_test_artifact();
        let stages = load_mesh_test_stages(context, &artifact);
        let target = mesh_test_render_target(context, if abort { "abort" } else { "submit" });
        begin_render_target(context, target).unwrap();
        execute_mesh(
            context,
            stages,
            [1, 1, 1],
            &[],
            ez_gfx_hal::MeshPipelineState {
                cull: ez_gfx_hal::CullMode::None,
                front_face: ez_gfx_hal::FrontFace::CounterClockwise,
                blend: ez_gfx_hal::BlendMode::None,
            },
        )
        .unwrap();

        for shader in [stages.task.unwrap(), stages.mesh, stages.fragment] {
            destroy_shader(context, shader);
        }
        assert_eq!(
            with_context_mut(context, |state| Ok(state.native_shader_destroys)).unwrap(),
            0
        );

        if abort {
            frame_abort(context).unwrap();
        } else {
            frame_submit(context).unwrap();
        }
        let probe =
            with_context_mut(context, |state| Ok(state.raw_native_frame_test_probe)).unwrap();
        assert_eq!(probe.submits, usize::from(!abort));
        assert_eq!(probe.mesh_executions, usize::from(!abort));
        assert_eq!(probe.graphics_executions, 0);
        assert_eq!(
            with_context_mut(context, |state| Ok(state.native_shader_destroys)).unwrap(),
            3
        );
        with_context_mut(context, |state| {
            super::shader::release_frame_shaders(state);
            Ok(())
        })
        .unwrap();
        assert_eq!(
            with_context_mut(context, |state| Ok(state.native_shader_destroys)).unwrap(),
            3
        );
        destroy_context(context).unwrap();
    }

    exercise(false);
    exercise(true);
}

#[cfg(windows)]
#[test]
fn raw_submit_probe_rejects_missing_retained_mesh_shader() {
    let context = initialized_shader_context();
    inject_shader_capabilities(
        context,
        ShaderCapabilities {
            task: true,
            mesh: true,
        },
    )
    .unwrap();
    with_context_mut(context, |state| {
        state.raw_native_frame_test_probe.enabled = true;
        Ok(())
    })
    .unwrap();
    let artifact = mesh_stage_test_artifact();
    let stages = load_mesh_test_stages(context, &artifact);
    let target = mesh_test_render_target(context, "missing-retained-mesh");
    begin_render_target(context, target).unwrap();
    execute_mesh(
        context,
        stages,
        [1, 1, 1],
        &[],
        ez_gfx_hal::MeshPipelineState {
            cull: ez_gfx_hal::CullMode::None,
            front_face: ez_gfx_hal::FrontFace::CounterClockwise,
            blend: ez_gfx_hal::BlendMode::None,
        },
    )
    .unwrap();
    let removed =
        with_context_mut(context, |state| Ok(state.shaders.remove(&stages.mesh))).unwrap();

    assert_eq!(frame_submit(context), Err(Error::NativeFailure));
    let probe = with_context_mut(context, |state| Ok(state.raw_native_frame_test_probe)).unwrap();
    assert_eq!(probe.submits, 0);

    with_context_mut(context, |state| {
        state
            .shaders
            .insert(stages.mesh, removed.ok_or(Error::NativeFailure)?);
        Ok(())
    })
    .unwrap();
    for shader in [stages.task.unwrap(), stages.mesh, stages.fragment] {
        destroy_shader(context, shader);
    }
    destroy_context(context).unwrap();
}

#[cfg(windows)]
#[test]
fn mesh_only_profile_submits_mesh_then_graphics_after_task_rejection() {
    let context = initialized_shader_context();
    inject_shader_capabilities(
        context,
        ShaderCapabilities {
            task: false,
            mesh: true,
        },
    )
    .unwrap();
    with_context_mut(context, |state| {
        state.raw_native_frame_test_probe.enabled = true;
        Ok(())
    })
    .unwrap();
    let artifact = mesh_stage_test_artifact();
    assert_eq!(
        load_shader(context, &artifact, ez_gfx_artifact::Stage::Task, "taskmain"),
        Err(Error::Unsupported)
    );
    assert_eq!(native_shader_allocation_attempts(context).unwrap(), 0);

    let mesh = load_shader(context, &artifact, ez_gfx_artifact::Stage::Mesh, "meshmain").unwrap();
    let fragment = load_shader(
        context,
        &artifact,
        ez_gfx_artifact::Stage::Fragment,
        "fragmentmain",
    )
    .unwrap();
    let vertex = load_shader(
        context,
        &artifact,
        ez_gfx_artifact::Stage::Vertex,
        "vertexmain",
    )
    .unwrap();
    assert_eq!(native_shader_allocation_attempts(context).unwrap(), 3);

    let target = mesh_test_render_target(context, "mesh-only");
    begin_render_target(context, target).unwrap();
    execute_mesh(
        context,
        ez_gfx_hal::MeshStages {
            task: None,
            mesh,
            fragment,
        },
        [1, 1, 1],
        &[],
        ez_gfx_hal::MeshPipelineState {
            cull: ez_gfx_hal::CullMode::None,
            front_face: ez_gfx_hal::FrontFace::CounterClockwise,
            blend: ez_gfx_hal::BlendMode::None,
        },
    )
    .unwrap();
    frame_submit(context).unwrap();
    let probe = with_context_mut(context, |state| Ok(state.raw_native_frame_test_probe)).unwrap();
    assert_eq!(
        (
            probe.submits,
            probe.mesh_executions,
            probe.graphics_executions
        ),
        (1, 1, 0)
    );

    assert_eq!(
        execute_mesh(
            context,
            ez_gfx_hal::MeshStages {
                task: Some(vertex),
                mesh,
                fragment,
            },
            [1, 1, 1],
            &[],
            ez_gfx_hal::MeshPipelineState {
                cull: ez_gfx_hal::CullMode::None,
                front_face: ez_gfx_hal::FrontFace::CounterClockwise,
                blend: ez_gfx_hal::BlendMode::None,
            },
        ),
        Err(Error::Unsupported)
    );
    assert_eq!(native_shader_allocation_attempts(context).unwrap(), 3);

    begin_render_target(context, target).unwrap();
    create_index_heap(context, 64).unwrap();
    let counter = acquire_counter(context, 1).unwrap();
    execute_graphics(
        context,
        vertex,
        fragment,
        counter,
        &[],
        DynamicPipelineState::from_abi(0, 0, 0, 0).unwrap(),
    )
    .unwrap();
    frame_submit(context).unwrap();
    let probe = with_context_mut(context, |state| Ok(state.raw_native_frame_test_probe)).unwrap();
    assert_eq!(
        (
            probe.submits,
            probe.mesh_executions,
            probe.graphics_executions
        ),
        (2, 1, 1)
    );

    for shader in [mesh, fragment, vertex] {
        destroy_shader(context, shader);
    }
    destroy_context(context).unwrap();
}
#[cfg(not(target_vendor = "apple"))]
#[test]
fn raw_mesh_prevalidation_failure_aborts_prior_recording() {
    let context = create_context(vulkan_options().unwrap()).unwrap();
    let surface =
        create_surface_headless(context, HeadlessSurfaceOptions::new(1, 1, 0).unwrap()).unwrap();
    init_device(context, surface).unwrap();
    frame_begin(context).unwrap();
    with_context_mut(context, |state| {
        state
            .frame
            .record_node(
                NodeDesc::new("prior-valid-node", QueueKind::Compute),
                ExecutableNode::Present { surface },
            )
            .map_err(|error| map_frame(&error))?;
        Ok(())
    })
    .unwrap();

    assert!(
        execute_mesh(
            context,
            ez_gfx_hal::MeshStages {
                task: None,
                mesh: shader(29),
                fragment: shader(30),
            },
            [0, 1, 1],
            &[],
            ez_gfx_hal::MeshPipelineState {
                cull: ez_gfx_hal::CullMode::None,
                front_face: ez_gfx_hal::FrontFace::CounterClockwise,
                blend: ez_gfx_hal::BlendMode::None,
            },
        )
        .is_err()
    );
    assert_eq!(frame_submit(context), Err(Error::NotReady));
    assert_eq!(destroy_context(context), Ok(()));
}

#[test]
fn pipeline_layout_keys_ignore_reflection_order() {
    let first = ez_gfx_hal::ShaderBufferLayout::new(0, 3, 1, false).unwrap();
    let second = ez_gfx_hal::ShaderBufferLayout::new(0, 1, 2, true).unwrap();

    assert_eq!(
        pipeline_layout_key(&[first, second]),
        pipeline_layout_key(&[second, first])
    );
}
#[test]
fn mesh_dispatch_errors_preserve_argument_and_capability_classes() {
    assert_eq!(
        super::frame::map_mesh_dispatch(ez_gfx_hal::MeshDispatchError::InvalidGroups),
        Error::InvalidArgument
    );
    assert_eq!(
        super::frame::map_mesh_dispatch(ez_gfx_hal::MeshDispatchError::InvalidWorkgroup),
        Error::InvalidArgument
    );
    assert_eq!(
        super::frame::map_mesh_dispatch(ez_gfx_hal::MeshDispatchError::UnsupportedWorkgroup),
        Error::Unsupported
    );
}

#[test]
fn graphics_pipeline_keys_include_state_attachment_and_texture_interface() {
    let state = DynamicPipelineState::from_abi(2, 0, 0, 0).unwrap();
    let key = PipelineKey::Graphics {
        backend: Backend::Vulkan,
        vertex_shader: shader(7),
        vertex_digest: [1; 32],
        vertex_entry: "vertexmain".to_owned(),
        fragment_shader: shader(8),
        fragment_digest: [2; 32],
        fragment_entry: "fragmentmain".to_owned(),
        texture_heap: None,
        layouts: Vec::new(),
        state,
        depth_required: false,
        color_format: 44,
        depth_format: 0,
        sample_count: 1,
    };
    let mut changed_state = key.clone();
    let PipelineKey::Graphics { state, .. } = &mut changed_state else {
        unreachable!()
    };
    state.blend = ez_gfx_hal::BlendMode::Alpha;
    let mut changed_format = key.clone();
    let PipelineKey::Graphics { color_format, .. } = &mut changed_format else {
        unreachable!()
    };
    *color_format = 50;
    let mut changed_heap = key.clone();
    let PipelineKey::Graphics { texture_heap, .. } = &mut changed_heap else {
        unreachable!()
    };
    *texture_heap = Some(ez_gfx_hal::ShaderTextureHeapLayout::new(0, 4, 16, 2, 0, 1).unwrap());

    assert_eq!(
        HashSet::from([key, changed_state, changed_format, changed_heap]).len(),
        4
    );
}

#[test]
fn mesh_pipeline_keys_distinguish_per_stage_physical_layouts() {
    let identities = |metadata: &[u8]| {
        let reflections = ez_gfx_runtime::binding::validate_stage_reflections(
            metadata,
            Backend::Vulkan,
            &[
                (ez_gfx_artifact::Stage::Task, "taskmain"),
                (ez_gfx_artifact::Stage::Mesh, "meshmain"),
                (ez_gfx_artifact::Stage::Fragment, "fragmentmain"),
            ],
        )
        .unwrap();
        ez_gfx_hal::MeshStages {
            task: Some(reflections[0].physical_layout_identity()),
            mesh: reflections[1].physical_layout_identity(),
            fragment: reflections[2].physical_layout_identity(),
        }
    };
    let task_first = identities(
        br#"{"reflections":[
            {"target":"Spirv","entry":"taskmain","stage":"Task","reflection":{"parameters":[{"semantic_name":"first","api_kind":"buffer","binding_index":1,"binding_space":0}],"texture_heap":{"binding_index":4,"binding_space":0,"capacity":16,"argument_stride":2,"texture_argument_offset":0,"sampler_argument_offset":1},"workgroup_size":[1,1,1]}},
            {"target":"Spirv","entry":"meshmain","stage":"Mesh","reflection":{"parameters":[{"semantic_name":"second","api_kind":"buffer","binding_index":2,"binding_space":0}],"workgroup_size":[1,1,1]}},
            {"target":"Spirv","entry":"fragmentmain","stage":"Fragment","reflection":{"parameters":[]}}
        ]}"#,
    );
    let mesh_first = identities(
        br#"{"reflections":[
            {"target":"Spirv","entry":"taskmain","stage":"Task","reflection":{"parameters":[{"semantic_name":"second","api_kind":"buffer","binding_index":2,"binding_space":0}],"workgroup_size":[1,1,1]}},
            {"target":"Spirv","entry":"meshmain","stage":"Mesh","reflection":{"parameters":[{"semantic_name":"first","api_kind":"buffer","binding_index":1,"binding_space":0}],"texture_heap":{"binding_index":4,"binding_space":0,"capacity":16,"argument_stride":2,"texture_argument_offset":0,"sampler_argument_offset":1},"workgroup_size":[1,1,1]}},
            {"target":"Spirv","entry":"fragmentmain","stage":"Fragment","reflection":{"parameters":[]}}
        ]}"#,
    );
    let prepare = |stage_layouts| {
        let mut slot = None;
        PipelineKey::prepare_mesh_slot(
            &mut slot,
            MeshPipelineKeyDesc {
                backend: Backend::Vulkan,
                task_shader: Some(shader(9)),
                task_digest: Some([4; 32]),
                task_entry: Some("taskmain"),
                mesh_shader: shader(10),
                mesh_digest: [5; 32],
                mesh_entry: "meshmain",
                fragment_shader: shader(11),
                fragment_digest: [6; 32],
                fragment_entry: "fragmentmain",
                stage_layouts,
                state: ez_gfx_hal::MeshPipelineState {
                    cull: ez_gfx_hal::CullMode::Back,
                    front_face: ez_gfx_hal::FrontFace::CounterClockwise,
                    blend: ez_gfx_hal::BlendMode::None,
                },
                depth_required: false,
                color_format: 44,
                depth_format: 0,
                sample_count: 1,
            },
        );
        slot.unwrap()
    };

    // Merged bindings and heap are equal; only their execution-stage ownership differs.
    let task_first = prepare(&task_first);
    let mesh_first = prepare(&mesh_first);
    let mut absent_task = task_first.clone();
    let PipelineKey::Mesh { stage_layouts, .. } = &mut absent_task else {
        unreachable!()
    };
    stage_layouts[0] = None;

    assert_eq!(
        HashSet::from([task_first, mesh_first, absent_task]).len(),
        3
    );
}

#[test]
fn pipeline_keys_track_every_owning_shader_identity() {
    let compute = PipelineKey::Compute {
        backend: Backend::Vulkan,
        shader: shader(6),
        shader_digest: [3; 32],
        entry: "computemain".to_owned(),
        layouts: Vec::new(),
    };
    let graphics = PipelineKey::Graphics {
        backend: Backend::Vulkan,
        vertex_shader: shader(7),
        vertex_digest: [1; 32],
        vertex_entry: "vertexmain".to_owned(),
        fragment_shader: shader(8),
        fragment_digest: [2; 32],
        fragment_entry: "fragmentmain".to_owned(),
        texture_heap: None,
        layouts: Vec::new(),
        state: DynamicPipelineState::from_abi(2, 0, 0, 0).unwrap(),
        depth_required: false,
        color_format: 44,
        depth_format: 0,
        sample_count: 1,
    };
    let mesh = PipelineKey::Mesh {
        backend: Backend::Vulkan,
        task_shader: Some(shader(9)),
        task_digest: Some([4; 32]),
        task_entry: "taskmain".to_owned(),
        has_task_entry: true,
        mesh_shader: shader(10),
        mesh_digest: [5; 32],
        mesh_entry: "meshmain".to_owned(),
        fragment_shader: shader(11),
        fragment_digest: [6; 32],
        fragment_entry: "fragmentmain".to_owned(),
        stage_layouts: [None; 3],
        state: ez_gfx_hal::MeshPipelineState {
            cull: ez_gfx_hal::CullMode::Back,
            front_face: ez_gfx_hal::FrontFace::CounterClockwise,
            blend: ez_gfx_hal::BlendMode::None,
        },
        depth_required: false,
        color_format: 44,
        depth_format: 0,
        sample_count: 1,
    };

    assert!(compute.involves_shader(shader(6)));
    assert!(graphics.involves_shader(shader(7)));
    assert!(graphics.involves_shader(shader(8)));
    assert!(!graphics.involves_shader(shader(9)));
    assert!(mesh.involves_shader(shader(9)));
    assert!(mesh.involves_shader(shader(10)));
    assert!(mesh.involves_shader(shader(11)));
    assert!(!mesh.involves_shader(shader(12)));
    assert!(!compute.is_render());
    assert!(graphics.is_render());
    assert!(mesh.is_render());

    let mut switched_task = mesh.clone();
    let PipelineKey::Mesh { task_shader, .. } = &mut switched_task else {
        unreachable!()
    };
    *task_shader = Some(shader(12));
    let mut switched_mesh = mesh.clone();
    let PipelineKey::Mesh { mesh_shader, .. } = &mut switched_mesh else {
        unreachable!()
    };
    *mesh_shader = shader(12);
    let mut switched_fragment = mesh.clone();
    let PipelineKey::Mesh {
        fragment_shader, ..
    } = &mut switched_fragment
    else {
        unreachable!()
    };
    *fragment_shader = shader(12);
    let mut switched_state = mesh.clone();
    let PipelineKey::Mesh { state, .. } = &mut switched_state else {
        unreachable!()
    };
    state.blend = ez_gfx_hal::BlendMode::Alpha;
    let mut switched_format = mesh.clone();
    let PipelineKey::Mesh { color_format, .. } = &mut switched_format else {
        unreachable!()
    };
    *color_format = 50;
    assert_eq!(
        HashSet::from([
            mesh.clone(),
            switched_task,
            switched_mesh,
            switched_fragment,
            switched_state,
            switched_format,
        ])
        .len(),
        6
    );

    let PipelineKey::Mesh {
        task_entry,
        mesh_entry,
        fragment_entry,
        ..
    } = &mesh
    else {
        unreachable!()
    };
    assert_eq!(
        mesh.retained_bytes(),
        task_entry.capacity() + mesh_entry.capacity() + fragment_entry.capacity()
    );
}
