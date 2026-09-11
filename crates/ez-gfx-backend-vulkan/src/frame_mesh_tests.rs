#[cfg(test)]
mod mesh_plan_tests {
    use super::super::{
        HalError, NativeComputeDispatch, NativeFrameAction, NativeMeshDraw, NativePipeline,
        NativePipelineKind,
    };
    use super::plan::{validate_frame_plan, validate_mesh_plan};
    use ash::vk;

    fn pipeline(kind: NativePipelineKind) -> NativePipeline {
        // Null handles never record; validation reads only the retained kind.
        NativePipeline {
            pipeline: vk::Pipeline::null(),
            layout: vk::PipelineLayout::null(),
            public_descriptor_layout: vk::DescriptorSetLayout::null(),
            buffer_writable: Vec::new(),
            buffer_bindings: Vec::new(),
            kind,
        }
    }

    fn draw(pipeline: &NativePipeline, has_task: bool) -> NativeMeshDraw<'_> {
        NativeMeshDraw {
            width: 64,
            height: 64,
            pipeline,
            groups: [2, 1, 1],
            has_task,
            mesh_workgroup_size: [8, 1, 1],
            task_workgroup_size: has_task.then_some([4, 1, 1]),
            bindings: &[],
        }
    }

    #[test]
    fn mesh_plan_accepts_matching_task_selection() {
        let taskless = pipeline(NativePipelineKind::Mesh { task_stage: false });
        let tasked = pipeline(NativePipelineKind::Mesh { task_stage: true });
        assert_eq!(
            validate_mesh_plan(&draw(&taskless, false), (128, 128), true),
            Ok(())
        );
        assert_eq!(
            validate_mesh_plan(&draw(&tasked, true), (128, 128), true),
            Ok(())
        );
    }

    #[test]
    fn mesh_plan_rejects_pipeline_task_mismatch() {
        // A dispatch flag disagreeing with the pipeline's retained task stage
        // fails even though each side is separately well-formed.
        let taskless = pipeline(NativePipelineKind::Mesh { task_stage: false });
        let tasked = pipeline(NativePipelineKind::Mesh { task_stage: true });
        assert_eq!(
            validate_mesh_plan(&draw(&taskless, true), (128, 128), true),
            Err(HalError::InvalidArgument)
        );
        assert_eq!(
            validate_mesh_plan(&draw(&tasked, false), (128, 128), true),
            Err(HalError::InvalidArgument)
        );
    }

    #[test]
    fn mesh_plan_rejects_compute_and_indexed_graphics_pipelines() {
        for kind in [NativePipelineKind::Compute, NativePipelineKind::Graphics] {
            let wrong = pipeline(kind);
            assert_eq!(
                validate_mesh_plan(&draw(&wrong, false), (128, 128), true),
                Err(HalError::InvalidArgument)
            );
        }
    }

    #[test]
    fn compute_plan_rejects_mesh_pipeline() {
        let mesh = pipeline(NativePipelineKind::Mesh { task_stage: false });
        let actions = [NativeFrameAction::Compute(NativeComputeDispatch {
            pipeline: &mesh,
            groups: [1, 1, 1],
            bindings: &[],
        })];

        assert_eq!(
            validate_frame_plan(&actions, (64, 64), false, false).map(|_| ()),
            Err(HalError::InvalidArgument)
        );
    }
}
