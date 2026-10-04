use super::*;

#[test]
fn render_target_msaa_clear_resolves_into_sampled_image() {
    use ez_gfx_hal::{
        AttachmentLoadOp, AttachmentStoreOp, ExecutionBarrier, ExecutionPass, ExecutionRange,
        ImageSubresources, QueueKind, ResourceAccess, ResourceState, ShaderStage,
    };
    use ez_gfx_runtime::target::Format;
    let mut context = context();
    // The multisample count follows the probed ceiling so the test stays
    // meaningful on adapters below 4-sample support; single-sample-only
    // adapters skip the resolve path they cannot exercise.
    let formats = context.probe_target_formats().unwrap();
    let msaa_samples = [4_u8, 2].into_iter().find(|msaa_samples| {
        ez_gfx_runtime::target::TargetDeclaration::new(
            "msaa",
            ez_gfx_runtime::target::TargetUsage::Color,
            1.0,
            *msaa_samples,
            vec![Format::Rgba8Unorm],
            ez_gfx_runtime::target::ClearValue::None,
            true,
        )
        .is_ok_and(|declaration| formats.resolve(&declaration).is_ok())
    });
    let Some(msaa_samples) = msaa_samples else {
        eprintln!("skipping MSAA resolve: adapter admits single-sample only");
        return;
    };
    let target = context
        .create_render_target(
            Format::Rgba8Unorm,
            Some(Format::Depth32Float),
            64,
            64,
            17,
            msaa_samples,
        )
        .unwrap();
    assert!(target.msaa.is_some());
    assert_eq!(
        target.depth.as_ref().map(|depth| depth.samples),
        super::super::msaa::sample_count_flags(msaa_samples)
    );
    let attach = ResourceState::new(
        QueueKind::Graphics,
        ShaderStage::AllGraphics,
        ResourceAccess::ColorAttachmentWrite,
    )
    .unwrap();
    let depth_attach = ResourceState::new(
        QueueKind::Graphics,
        ShaderStage::Fragment,
        ResourceAccess::DepthStencilWrite,
    )
    .unwrap();
    let sampled_state = ResourceState::new(
        QueueKind::Graphics,
        ShaderStage::Fragment,
        ResourceAccess::SampledRead,
    )
    .unwrap();
    let range = ExecutionRange::Image(ImageSubresources::new(0, 1, 0, 1).unwrap());
    let pass = ExecutionPass {
        nodes: vec![],
        colors: vec![0],
        depth: Some(1),
        area: [0, 0, 64, 64],
        samples: msaa_samples,
        load: AttachmentLoadOp::Clear,
        store: AttachmentStoreOp::Store,
    };
    let actions = vec![
        NativeFrameAction::Barrier {
            barrier: ExecutionBarrier {
                node: 0,
                resource: 0,
                range,
                before: None,
                after: attach,
            },
            resource: NativeFrameResource::RenderTarget(&target),
        },
        NativeFrameAction::Barrier {
            barrier: ExecutionBarrier {
                node: 0,
                resource: 1,
                range,
                before: None,
                after: depth_attach,
            },
            resource: NativeFrameResource::RenderTargetDepth(&target),
        },
        NativeFrameAction::BeginPass {
            pass: &pass,
            colors: [PassAttachment {
                resource: NativeFrameResource::RenderTarget(&target),
                clear: [0.0, 1.0, 0.0, 1.0],
            }]
            .into(),
        },
        NativeFrameAction::EndPass,
        NativeFrameAction::Barrier {
            barrier: ExecutionBarrier {
                node: 0,
                resource: 0,
                range,
                before: Some(attach),
                after: sampled_state,
            },
            resource: NativeFrameResource::RenderTarget(&target),
        },
    ];
    context.execute_frame(None, &actions, false).unwrap();
    drop(actions);
    // The pass clears multisampled storage; the resolve writes the exact clear
    // color into the sampled image that readback copies.
    let bytes = context.readback_texture_rgba8(&target, 64, 64).unwrap();
    assert_eq!(bytes.len(), 64 * 64 * 4);
    for pixel in bytes.as_chunks::<4>().0 {
        assert_eq!(*pixel, [0, 255, 0, 255]);
    }
    // A 4-sample pass against a single-sample target (and vice versa) is rejected.
    let single = context
        .create_render_target(Format::Rgba8Unorm, None, 64, 64, 19, 1)
        .unwrap();
    let mismatch = NativeFrameAction::BeginPass {
        pass: &pass,
        colors: [PassAttachment {
            resource: NativeFrameResource::RenderTarget(&single),
            clear: [0.0, 0.0, 0.0, 1.0],
        }]
        .into(),
    };
    assert!(context.execute_frame(None, &[mismatch], false).is_err());
    context.destroy_texture(target).unwrap();
    context.destroy_texture(single).unwrap();
    context.wait_idle().unwrap();
}
