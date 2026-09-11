fn validate_surface_request(
    surface: Option<FrameSurface<'_>>,
) -> Result<ResolvedFrameSurface<'_>, HalError> {
    match surface {
        Some((surface, extent, mode)) if extent.0 != 0 && extent.1 != 0 => {
            surface.set_presentation_mode(mode)?;
            Ok((Some(surface), extent))
        }
        Some(_) => Err(HalError::InvalidArgument),
        None => Ok((None, (0, 0))),
    }
}

fn buffer_range_fits(allocation_size: u64, range: ez_gfx_hal::BufferRange) -> bool {
    range
        .offset
        .checked_add(range.size)
        .is_some_and(|end| end <= allocation_size)
}

fn draw_ranges_fit(
    index_physical_size: u64,
    index_logical_size: u64,
    indirect_physical_size: u64,
    indirect_logical_size: u64,
    draw_count: u32,
) -> bool {
    let required_indirect = u64::from(draw_count)
        .checked_mul(20)
        .and_then(|size| size.checked_add(COUNTER_BUFFER_ELEMENT_OFFSET));
    index_logical_size != 0
        && index_logical_size <= index_physical_size
        && indirect_logical_size <= indirect_physical_size
        && required_indirect.is_some_and(|required| required <= indirect_logical_size)
}

fn bindings_fit(bindings: &dyn super::NativeBufferBindingSource) -> Result<bool, HalError> {
    let mut fits = true;
    bindings.visit(&mut |_, binding| {
        fits &= (binding.offset as u64) < binding.allocation.allocation.size();
        Ok(())
    })?;
    Ok(fits)
}

fn metal_size(size: [u32; 3]) -> MTLSize {
    MTLSize {
        width: size[0] as usize,
        height: size[1] as usize,
        depth: size[2] as usize,
    }
}
