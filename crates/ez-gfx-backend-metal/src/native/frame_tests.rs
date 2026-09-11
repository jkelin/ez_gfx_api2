#[cfg(test)]
mod tests {
    use super::{NativeContext, buffer_range_fits, draw_ranges_fit, metal_size};
    use ez_gfx_hal::{BufferRange, COUNTER_BUFFER_ELEMENT_OFFSET};

    #[test]
    fn buffer_barrier_range_must_fit_allocation() {
        assert!(buffer_range_fits(64, BufferRange::new(16, 48).unwrap()));
        assert!(!buffer_range_fits(63, BufferRange::new(16, 48).unwrap()));
        assert!(!buffer_range_fits(
            u64::MAX,
            BufferRange {
                offset: u64::MAX - 3,
                size: 4,
            }
        ));
    }

    #[test]
    fn indexed_indirect_logical_ranges_include_aligned_element_offset() {
        let required = COUNTER_BUFFER_ELEMENT_OFFSET + 40;
        assert!(draw_ranges_fit(64, 64, required, required, 2));
        assert!(!draw_ranges_fit(64, 0, required, required, 2));
        assert!(!draw_ranges_fit(64, 65, required, required, 2));
        assert!(!draw_ranges_fit(64, 64, required, required - 1, 2));
        assert!(!draw_ranges_fit(64, 64, required - 1, required, 2));
    }

    #[test]
    fn reflected_threadgroup_dimensions_reach_metal_dispatch_shape() {
        let size = metal_size([8, 2, 1]);
        assert_eq!((size.width, size.height, size.depth), (8, 2, 1));
    }

    #[test]
    fn absent_task_stage_uses_only_the_inapplicable_object_size_default() {
        assert_eq!(super::encode::object_threadgroup_size(None), [1, 1, 1]);
        assert_eq!(
            super::encode::object_threadgroup_size(Some([32, 2, 1])),
            [32, 2, 1]
        );
    }

    #[test]
    fn mesh_buffer_bindings_reach_only_reflected_stage_intervals() {
        let layouts = [ez_gfx_hal::ShaderBufferLayout::new(0, 4, 2, false).unwrap()];

        assert!(!super::encode::buffer_stage_uses_binding(&layouts, 3));
        assert!(super::encode::buffer_stage_uses_binding(&layouts, 4));
        assert!(super::encode::buffer_stage_uses_binding(&layouts, 5));
        assert!(!super::encode::buffer_stage_uses_binding(&layouts, 6));
        assert!(!super::encode::buffer_stage_uses_binding(
            &layouts,
            usize::MAX
        ));
    }

    #[test]
    fn demoted_view_extent_follows_published_coarse_tail() {
        use NativeContext as Context;
        // A fully resident view exposes the stored base extent.
        assert_eq!(Context::published_view_extent(64, 64, 3, 3), Some((64, 64)));
        // Demotion drops fine levels: level one of a 64-wide chain is 16 wide.
        assert_eq!(Context::published_view_extent(64, 64, 3, 1), Some((16, 16)));
        // Odd edges clamp at one texel rather than shifting to zero.
        assert_eq!(Context::published_view_extent(7, 3, 3, 1), Some((1, 1)));
        // An unpublished or over-published view has no readable extent.
        assert_eq!(Context::published_view_extent(64, 64, 3, 0), None);
        assert_eq!(Context::published_view_extent(64, 64, 3, 4), None);
    }
}
