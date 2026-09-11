
#[allow(
    clippy::too_many_arguments,
    reason = "each independently retained lowering vector is accounted and discarded together"
)]
fn account_frame_lowering_scratch(
    frame: &mut ez_gfx_runtime::frame::FrameRecorder,
    pipeline_keys: &mut Vec<Option<PipelineKey>>,
    action_indices: &mut Vec<usize>,
    bindings: &mut Vec<FrameBufferBindingRecord>,
    ranges: &mut Vec<core::ops::Range<usize>>,
    #[cfg(target_vendor = "apple")]
    texture_heaps: &mut Vec<Option<ez_gfx_hal::ShaderTextureHeapLayout>>,
    #[cfg(target_vendor = "apple")] workgroup_sizes: &mut Vec<Option<[u32; 3]>>,
) -> Result<()> {
    let bytes = pipeline_keys
        .capacity()
        .saturating_mul(core::mem::size_of::<Option<PipelineKey>>())
        .saturating_add(
            pipeline_keys.iter().flatten().fold(0_usize, |total, key| {
                total.saturating_add(key.retained_bytes())
            }),
        )
        .saturating_add(
            action_indices
                .capacity()
                .saturating_mul(core::mem::size_of::<usize>()),
        )
        .saturating_add(
            bindings
                .capacity()
                .saturating_mul(core::mem::size_of::<FrameBufferBindingRecord>()),
        )
        .saturating_add(
            ranges
                .capacity()
                .saturating_mul(core::mem::size_of::<core::ops::Range<usize>>()),
        );
    #[cfg(target_vendor = "apple")]
    let bytes = bytes
        .saturating_add(
            texture_heaps
                .capacity()
                .saturating_mul(core::mem::size_of::<
                    Option<ez_gfx_hal::ShaderTextureHeapLayout>,
                >()),
        )
        .saturating_add(
            workgroup_sizes
                .capacity()
                .saturating_mul(core::mem::size_of::<Option<[u32; 3]>>()),
        );
    if let Err(error) = frame.set_lowering_scratch_bytes(bytes) {
        // Drop the complete lowering workspace together: retaining only some
        // vectors would make accounting dependent on the rejected frame.
        *pipeline_keys = Vec::new();
        *action_indices = Vec::new();
        *bindings = Vec::new();
        *ranges = Vec::new();
        #[cfg(target_vendor = "apple")]
        {
            *texture_heaps = Vec::new();
            *workgroup_sizes = Vec::new();
        }
        frame
            .set_lowering_scratch_bytes(0)
            .map_err(|reset| map_frame(&reset))?;
        return Err(map_frame(&error));
    }
    Ok(())
}
