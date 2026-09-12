//! Custom texture-decoder C ABI contracts.

use core::{
    ffi::c_void,
    mem::{align_of, offset_of, size_of},
    sync::atomic::{AtomicUsize, Ordering},
};

use ez_gfx::{TextureDecoder, TextureSource};
use ez_gfx_ffi::{
    EzGfxDecodedTexture, EzGfxDecodedTextureMip, EzGfxResult, ez_gfx_texture_decoder_register,
    ez_gfx_texture_decoder_unregister,
};

unsafe extern "C" fn decode(
    data: *const u8,
    data_size: usize,
    _compression: u8,
    out: *mut EzGfxDecodedTexture,
    _user_data: *mut c_void,
) -> EzGfxResult {
    // SAFETY: The ABI provides a readable input; allocations transfer to the paired release.
    let bytes = unsafe { core::slice::from_raw_parts(data, data_size) }
        .to_vec()
        .into_boxed_slice();
    let data = Box::into_raw(bytes).cast::<u8>();
    let mip = Box::new(EzGfxDecodedTextureMip {
        width: 1,
        height: 1,
        data,
        data_size,
    });
    // SAFETY: `out` is writable under the callback contract.
    unsafe {
        out.write(EzGfxDecodedTexture {
            format: 0,
            mip_count: 1,
            mips: Box::into_raw(mip),
        });
    }
    EzGfxResult::Ok
}

unsafe extern "C" fn release(texture: *const EzGfxDecodedTexture, _user_data: *mut c_void) {
    // SAFETY: `decode` allocated one mip and byte slice; release runs exactly once.
    unsafe {
        let texture = texture.read();
        let mip = Box::from_raw(texture.mips.cast_mut());
        drop(Box::from_raw(core::ptr::slice_from_raw_parts_mut(
            mip.data.cast_mut(),
            mip.data_size,
        )));
    }
}

#[test]
fn layouts_and_export_signatures_are_stable() {
    assert_eq!(
        (
            size_of::<EzGfxDecodedTextureMip>(),
            align_of::<EzGfxDecodedTextureMip>()
        ),
        (24, 8)
    );
    assert_eq!(
        [
            offset_of!(EzGfxDecodedTextureMip, width),
            offset_of!(EzGfxDecodedTextureMip, height),
            offset_of!(EzGfxDecodedTextureMip, data),
            offset_of!(EzGfxDecodedTextureMip, data_size),
        ],
        [0, 4, 8, 16]
    );
    assert_eq!(
        (
            size_of::<EzGfxDecodedTexture>(),
            align_of::<EzGfxDecodedTexture>()
        ),
        (16, 8)
    );
    assert_eq!(
        [
            offset_of!(EzGfxDecodedTexture, format),
            offset_of!(EzGfxDecodedTexture, mip_count),
            offset_of!(EzGfxDecodedTexture, mips),
        ],
        [0, 4, 8]
    );
    let _: unsafe extern "C" fn(
        u8,
        ez_gfx_ffi::EzGfxTextureDecoderCallback,
        ez_gfx_ffi::EzGfxTextureDecoderReleaseCallback,
        *mut c_void,
    ) -> EzGfxResult = ez_gfx_texture_decoder_register;
    let _: extern "C" fn(u8) -> EzGfxResult = ez_gfx_texture_decoder_unregister;
}

#[test]
fn registration_copies_and_releases_decoder_output() {
    const SOURCE: u8 = 202;
    let _ = ez_gfx_texture_decoder_unregister(SOURCE);
    assert_eq!(
        // SAFETY: Function pointers and null user data remain valid for this test.
        unsafe {
            ez_gfx_texture_decoder_register(
                SOURCE,
                Some(decode),
                Some(release),
                core::ptr::null_mut(),
            )
        },
        EzGfxResult::Ok
    );
    let decoded = TextureDecoder::decode(TextureSource::Custom(SOURCE), &[1, 2, 3, 4]).unwrap();
    assert_eq!(decoded.mips[0].bytes, [1, 2, 3, 4]);
    assert_eq!(ez_gfx_texture_decoder_unregister(SOURCE), EzGfxResult::Ok);
}

unsafe extern "C" fn invalid_decode(
    _data: *const u8,
    _data_size: usize,
    _compression: u8,
    out: *mut EzGfxDecodedTexture,
    _user_data: *mut c_void,
) -> EzGfxResult {
    // SAFETY: `out` is writable under the callback contract.
    unsafe {
        out.write(EzGfxDecodedTexture {
            format: 1,
            mip_count: 0,
            mips: core::ptr::null(),
        });
    }
    EzGfxResult::Ok
}

unsafe extern "C" fn count_release(_texture: *const EzGfxDecodedTexture, user_data: *mut c_void) {
    // SAFETY: This test retains the pointed-to atomic through decode and unregister.
    unsafe { &*user_data.cast::<AtomicUsize>() }.fetch_add(1, Ordering::Relaxed);
}

#[test]
fn registration_rejects_invalid_boundaries_and_releases_invalid_success_output() {
    const SOURCE: u8 = 204;

    assert_eq!(
        // SAFETY: Null callbacks intentionally exercise boundary validation.
        unsafe {
            ez_gfx_texture_decoder_register(203, None, Some(count_release), core::ptr::null_mut())
        },
        EzGfxResult::InvalidArgument
    );
    assert_eq!(
        // SAFETY: Reserved source intentionally exercises boundary validation.
        unsafe {
            ez_gfx_texture_decoder_register(
                6,
                Some(invalid_decode),
                Some(count_release),
                core::ptr::null_mut(),
            )
        },
        EzGfxResult::InvalidArgument
    );

    let releases = AtomicUsize::new(0);
    assert_eq!(
        // SAFETY: The atomic remains live until synchronous decode completes and unregister returns.
        unsafe {
            ez_gfx_texture_decoder_register(
                SOURCE,
                Some(invalid_decode),
                Some(count_release),
                (&raw const releases).cast_mut().cast(),
            )
        },
        EzGfxResult::Ok
    );
    assert_eq!(
        TextureDecoder::decode(TextureSource::Custom(SOURCE), &[1]),
        Err(ez_gfx::TextureError::InvalidData)
    );
    assert_eq!(releases.load(Ordering::Relaxed), 1);
    assert_eq!(ez_gfx_texture_decoder_unregister(SOURCE), EzGfxResult::Ok);
}
