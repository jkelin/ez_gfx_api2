use super::{EZ_GFX_MAX_BOUNDARY_BYTES, EzGfxHandleParts, EzGfxResult, catch_status};
use ez_gfx::SemanticId;
use ez_gfx_core::handle::{HandleParts, PackedHandle};

#[unsafe(no_mangle)]
/// Decodes a packed handle into its context and optional child slot generations.
///
/// # Safety
///
/// A non-null `out_parts` must address one writable, aligned handle description for this call.
pub unsafe extern "C" fn ez_gfx_handle_inspect(
    handle: u64,
    out_parts: *mut EzGfxHandleParts,
) -> EzGfxResult {
    catch_status(|| {
        if out_parts.is_null() {
            return EzGfxResult::InvalidArgument;
        }
        let Ok(parts) = PackedHandle::from_raw(handle).and_then(PackedHandle::parts) else {
            return EzGfxResult::InvalidContext;
        };
        let abi = match parts {
            HandleParts::Context(context) => EzGfxHandleParts {
                context_slot: context.slot(),
                context_generation: context.generation(),
                child_slot: 0,
                child_generation: 0,
                is_context: 1,
                _padding: [0; 3],
            },
            HandleParts::Child { owner, child } => EzGfxHandleParts {
                context_slot: owner.slot(),
                context_generation: owner.generation(),
                child_slot: child.slot(),
                child_generation: child.generation(),
                is_context: 0,
                _padding: [0; 3],
            },
        };
        // SAFETY: `out_parts` is non-null, and the caller keeps writable, properly aligned storage for one `EzGfxHandleParts` alive through this write.
        unsafe { out_parts.write(abi) };
        EzGfxResult::Ok
    })
}

#[unsafe(no_mangle)]
/// Computes the fixed-size semantic identifier for a UTF-8 name.
///
/// # Safety
///
/// Non-null `name` must be readable for `length` bytes, and non-null `out_id` writable for 16 bytes, for this call.
pub unsafe extern "C" fn ez_gfx_semantic_id(
    name: *const u8,
    length: usize,
    out_id: *mut u8,
) -> EzGfxResult {
    catch_status(|| {
        if out_id.is_null()
            || name.is_null()
            || length == 0
            || length > EZ_GFX_MAX_BOUNDARY_BYTES
            || length > isize::MAX as usize
        {
            return EzGfxResult::InvalidArgument;
        }
        // SAFETY: `length` is checked nonzero and within both the byte and `isize` limits; the caller keeps `name` readable for `length` `u8` values (alignment 1) through UTF-8 validation.
        let bytes = unsafe { core::slice::from_raw_parts(name, length) };
        let Ok(name) = core::str::from_utf8(bytes) else {
            return EzGfxResult::InvalidArgument;
        };
        let id = match SemanticId::from_name(name) {
            Ok(id) => id.bytes(),
            Err(_) => return EzGfxResult::InvalidArgument,
        };
        // SAFETY: `out_id` is non-null, and the caller provides an alignment-1 writable range of `id.len()` bytes that is disjoint from the live local `id` storage through the copy.
        unsafe { core::ptr::copy_nonoverlapping(id.as_ptr(), out_id, id.len()) };
        EzGfxResult::Ok
    })
}
