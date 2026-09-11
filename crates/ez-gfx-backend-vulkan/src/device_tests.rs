use super::*;
use std::collections::BTreeSet;

fn family(flags: vk::QueueFlags, count: u32) -> vk::QueueFamilyProperties {
    vk::QueueFamilyProperties {
        queue_flags: flags,
        queue_count: count,
        ..Default::default()
    }
}

#[test]
fn draw_admission_requires_indirect_count_and_core_draw_features() {
    let supported = vk::PhysicalDeviceVulkan12Features {
        draw_indirect_count: vk::TRUE,
        ..Default::default()
    };
    assert_eq!(draw_feature_rejection(true, true, &supported), None);
    assert_eq!(
        draw_feature_rejection(false, true, &supported),
        Some("vertex_pipeline_stores_and_atomics")
    );
    assert_eq!(
        draw_feature_rejection(true, false, &supported),
        Some("multi_draw_indirect")
    );

    let missing_count = vk::PhysicalDeviceVulkan12Features::default();
    assert_eq!(
        draw_feature_rejection(true, true, &missing_count),
        Some("draw_indirect_count")
    );
}

#[test]
fn mesh_shader_capabilities_require_the_extension_and_mesh_for_task() {
    let both = vk::PhysicalDeviceMeshShaderFeaturesEXT {
        task_shader: vk::TRUE,
        mesh_shader: vk::TRUE,
        ..Default::default()
    };
    assert_eq!(
        normalized_mesh_shader_capabilities(false, &both),
        ShaderCapabilities::default()
    );
    assert_eq!(
        normalized_mesh_shader_capabilities(true, &both),
        ShaderCapabilities {
            task: true,
            mesh: true,
        }
    );

    let task_only = vk::PhysicalDeviceMeshShaderFeaturesEXT {
        task_shader: vk::TRUE,
        mesh_shader: vk::FALSE,
        ..Default::default()
    };
    assert_eq!(
        normalized_mesh_shader_capabilities(true, &task_only),
        ShaderCapabilities::default()
    );
}

#[test]
fn transfer_family_prefers_non_graphics_hardware_queue() {
    let families = [
        family(vk::QueueFlags::GRAPHICS | vk::QueueFlags::TRANSFER, 2),
        family(vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER, 1),
    ];

    assert_eq!(select_transfer_family(&families, 0), 1);
}

#[test]
fn transfer_family_falls_back_when_specialized_queue_is_unavailable() {
    let families = [
        family(vk::QueueFlags::GRAPHICS | vk::QueueFlags::TRANSFER, 1),
        family(vk::QueueFlags::TRANSFER, 0),
    ];

    assert_eq!(select_transfer_family(&families, 0), 0);
}

#[test]
fn enumeration_reports_unique_named_adapters() {
    // No surface is created, shown, or activated by this test.
    let adapters = NativeContext::enumerate_adapters().expect("Vulkan enumerates adapters");
    assert!(!adapters.is_empty());
    let mut identities = BTreeSet::new();
    for adapter in &adapters {
        assert_ne!(adapter.stable_id(), [0; 16]);
        assert!(!adapter.name().is_empty());
        assert!(!adapter.driver().is_empty());
        assert!(identities.insert(adapter.stable_id()));
    }
}

#[test]
fn explicit_selection_rejects_unknown_identity() {
    // No surface is created, shown, or activated by this test.
    let mut context = NativeContext::create(false, false).expect("Vulkan instance");
    assert_eq!(
        context.init_device_for_adapter(None, [0xA5; 16], false),
        Err(HalError::InvalidArgument)
    );
}

#[test]
fn explicit_selection_admits_enumerated_adapter() {
    // No surface is created, shown, or activated by this test.
    let adapters = NativeContext::enumerate_adapters().expect("Vulkan enumerates adapters");
    let wanted = adapters.first().expect("at least one adapter").stable_id();
    let mut context = NativeContext::create(false, false).expect("Vulkan instance");
    let admitted = context
        .init_device_for_adapter(None, wanted, true)
        .expect("enumerated adapter initializes");
    assert_eq!(admitted.stable_id(), wanted);
    let shader_stages = admitted.capabilities().shader_stages;
    assert!(!shader_stages.task || shader_stages.mesh);
    assert_eq!(context.mesh_shader_loader.is_some(), shader_stages.mesh);
    assert_eq!(context.mesh_shader_limits.is_some(), shader_stages.mesh);
}

#[cfg(test)]
mod mesh_dispatch_tests {
    use crate::{
        MeshShaderLimits, NativeContext, NativeMeshPipelineDesc, NativeShader, check_mesh_groups,
    };
    use ash::vk;
    use ez_gfx_hal::{BlendMode, CullMode, FrontFace, HalError, MeshPipelineState};

    fn limits() -> MeshShaderLimits {
        MeshShaderLimits {
            task_supported: true,
            task_work_group_invocations: 128,
            mesh_work_group_invocations: 256,
            mesh_output_vertices: 256,
            mesh_output_primitives: 128,
            task_group_count: [65_535, 65_535, 65_535],
            task_group_total_count: 1 << 22,
            mesh_group_count: [65_535, 65_535, 65_535],
            mesh_group_total_count: 1 << 22,
        }
    }

    fn raster() -> MeshPipelineState {
        MeshPipelineState {
            cull: CullMode::None,
            front_face: FrontFace::CounterClockwise,
            blend: BlendMode::None,
        }
    }
    fn describe<'a>(
        empty: &'a NativeShader,
        task: Option<(&'a NativeShader, usize)>,
    ) -> NativeMeshPipelineDesc<'a> {
        NativeMeshPipelineDesc {
            task,
            mesh: (empty, 0),
            fragment: (empty, 0),
            state: raster(),
            color_format: None,
            layouts: &[],
            depth_required: false,
            task_workgroup_size: task.map(|_| [1, 1, 1]),
            mesh_workgroup_size: [32, 1, 1],
        }
    }

    #[test]
    fn mesh_groups_accept_valid_task_and_mesh_dispatches() {
        let limits = limits();
        assert_eq!(
            check_mesh_groups(&limits, true, [4, 2, 1], [128, 2, 1], Some([32, 1, 1])),
            Ok(())
        );
        assert_eq!(
            check_mesh_groups(&limits, false, [4, 2, 1], [128, 2, 1], None),
            Ok(())
        );
    }

    #[test]
    fn mesh_groups_reject_zero_per_dimension_overflow_and_total_overflow() {
        let limits = limits();
        // Zero dimensions fail even though every ceiling is nonzero.
        assert_eq!(
            check_mesh_groups(&limits, false, [0, 1, 1], [8, 1, 1], None),
            Err(HalError::InvalidArgument)
        );
        // One dimension above its ceiling fails.
        assert_eq!(
            check_mesh_groups(&limits, false, [65_536, 1, 1], [8, 1, 1], None),
            Err(HalError::InvalidArgument)
        );
        // In-dimension counts can still exceed the native total grid count.
        assert_eq!(
            check_mesh_groups(&limits, false, [4096, 4096, 1], [8, 1, 1], None),
            Err(HalError::InvalidArgument)
        );
    }

    #[test]
    fn mesh_groups_reject_inconsistent_task_selection() {
        let limits = limits();
        // A task size without a task stage, or a missing one with it, describes
        // no executable dispatch.
        assert_eq!(
            check_mesh_groups(&limits, false, [1, 1, 1], [8, 1, 1], Some([8, 1, 1])),
            Err(HalError::InvalidArgument)
        );
        assert_eq!(
            check_mesh_groups(&limits, true, [1, 1, 1], [8, 1, 1], None),
            Err(HalError::InvalidArgument)
        );
    }

    #[test]
    fn mesh_pipeline_rejects_unsupported_before_allocation() {
        // No surface is created, shown, or activated by this test.
        let adapters = NativeContext::enumerate_adapters().expect("Vulkan enumerates");
        let wanted = adapters.first().expect("at least one adapter").stable_id();
        let mut context = NativeContext::create(false, false).expect("Vulkan instance");
        let admitted = context
            .init_device_for_adapter(None, wanted, true)
            .expect("enumerated adapter initializes");
        let stages = admitted.capabilities().shader_stages;
        // The public limits getter follows the same gates without allocation.
        assert_eq!(context.mesh_dispatch_limits(false).is_ok(), stages.mesh);
        assert_eq!(context.mesh_dispatch_limits(true).is_ok(), stages.task);
        // Empty shaders carry no modules, so a mesh-capable device fails on the
        // product index while an incapable one fails on the earlier support gate.
        let empty = NativeShader {
            modules: Vec::new(),
        };
        assert_eq!(
            context
                .create_mesh_pipeline(describe(&empty, None))
                .map(|_| ()),
            if stages.mesh {
                Err(HalError::InvalidArgument)
            } else {
                Err(HalError::Unsupported)
            }
        );
        // A task stage on a mesh-only device fails on task support, never on indices.
        assert_eq!(
            context
                .create_mesh_pipeline(describe(&empty, Some((&empty, 0))))
                .map(|_| ()),
            if stages.task {
                Err(HalError::InvalidArgument)
            } else {
                Err(HalError::Unsupported)
            }
        );
        // An inconsistent stage/size selection is invalid on any device.
        let mismatched = NativeMeshPipelineDesc {
            task: None,
            task_workgroup_size: Some([1, 1, 1]),
            ..describe(&empty, None)
        };
        assert_eq!(
            context.create_mesh_pipeline(mismatched).map(|_| ()),
            Err(HalError::InvalidArgument)
        );
        // A null module passes index checks without reaching native creation, so
        // a well-formed but oversized workgroup is unsupported on any device,
        // while a zero workgroup is malformed only where mesh is supported.
        let present = NativeShader {
            modules: vec![vk::ShaderModule::null()],
        };
        let oversized = NativeMeshPipelineDesc {
            mesh: (&present, 0),
            fragment: (&present, 0),
            mesh_workgroup_size: [1_000_000, 1, 1],
            ..describe(&present, None)
        };
        assert_eq!(
            context.create_mesh_pipeline(oversized).map(|_| ()),
            Err(HalError::Unsupported)
        );
        let empty_workgroup = NativeMeshPipelineDesc {
            mesh_workgroup_size: [0, 1, 1],
            ..describe(&present, None)
        };
        assert_eq!(
            context.create_mesh_pipeline(empty_workgroup).map(|_| ()),
            if stages.mesh {
                Err(HalError::InvalidArgument)
            } else {
                Err(HalError::Unsupported)
            }
        );
    }
}
