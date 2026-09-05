//! Hidden native compressed sampling, edge-update, and submitted-frame retirement regression.
#![cfg(windows)]

mod common;

use std::{
    sync::{Arc, LazyLock},
    time::{Duration, Instant},
};

use common::TestContext;
use ez_gfx::*;
use ez_gfx_compiler::{Target, compile_shader};

#[test]
fn vulkan_bc_pixels_survive_region_updates_and_unload() {
    const CHILD: &str = "EZ_GFX_TEXTURE_PIXELS_VALIDATION_CHILD";
    if std::env::var_os(CHILD).is_some() {
        exercise_backend(1);
    } else {
        assert_validation_clean(CHILD);
    }
}

#[test]
fn dx12_bc_pixels_survive_region_updates_and_unload() {
    exercise_backend(2);
}

fn assert_validation_clean(child_marker: &str) {
    use std::{
        io::{Read, Write},
        os::windows::process::CommandExt,
        process::{Command, Stdio},
    };

    // The validation layer writes directly to native stdout/stderr, outside Rust's test capture.
    // A hidden child makes those messages test failures even when the rendered pixels look right.
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "vulkan_bc_pixels_survive_region_updates_and_unload",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(child_marker, "1")
        .creation_flags(0x0800_0000) // CREATE_NO_WINDOW: never create or activate a console.
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn hidden Vulkan validation regression");
    let mut stdout = child.stdout.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();
    // Drain both pipes concurrently: a flood of validation errors must not deadlock the child.
    let output = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        stdout
            .read_to_end(&mut bytes)
            .expect("read Vulkan child stdout");
        bytes
    });
    let errors = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        stderr
            .read_to_end(&mut bytes)
            .expect("read Vulkan child stderr");
        bytes
    });
    let deadline = Instant::now() + Duration::from_secs(120);
    let (status, timed_out) = loop {
        if let Some(status) = child.try_wait().expect("poll Vulkan validation child") {
            break (status, false);
        }
        if Instant::now() >= deadline {
            child
                .kill()
                .expect("terminate stalled Vulkan validation child");
            break (child.wait().expect("reap Vulkan validation child"), true);
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let output = output.join().expect("join Vulkan stdout reader");
    let errors = errors.join().expect("join Vulkan stderr reader");
    std::io::stdout()
        .write_all(&output)
        .expect("re-emit Vulkan stdout");
    std::io::stderr()
        .write_all(&errors)
        .expect("re-emit Vulkan stderr");
    assert!(!timed_out, "Vulkan validation child exceeded 120 seconds");
    assert!(status.success(), "Vulkan validation child failed: {status}");
    for bytes in [&output, &errors] {
        let text = String::from_utf8_lossy(bytes);
        assert!(
            !text.contains("VUID-") && !text.contains("Validation Error"),
            "Vulkan validation reported an error; see native output above"
        );
    }
}

fn exercise_backend(backend: u8) {
    // One compilation avoids concurrent compiler artifact writes by the backend tests.
    static ARTIFACT: LazyLock<Vec<u8>> = LazyLock::new(|| {
        compile_shader(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../examples/02_textured_cube/02_textured_cube.slang"),
            &[Target::Spirv, Target::Dxil, Target::Metal],
            true,
        )
        .expect("compile cube shader artifact")
    });
    // A missing Vulkan validation layer is a failure, never a silent unvalidated run.
    let native = if backend == 1 {
        TestContext::create_with_validation(backend, true)
    } else {
        TestContext::create(backend)
    };
    let context = ContextHandle::from_raw(native.context).unwrap();
    let surface = SurfaceHandle::from_raw(native.surface).unwrap();
    assert_eq!(surface_extent(context, surface).unwrap(), (64, 64));
    let quad = Quad::create(context, surface, &ARTIFACT);

    for (width, height) in [(8, 4), (7, 3)] {
        for (format, destination) in [
            (TextureFormat::Bc1Unorm, TextureDestination::Bc1Unorm),
            (TextureFormat::Bc3Unorm, TextureDestination::Bc3Unorm),
            (TextureFormat::Bc7Unorm, TextureDestination::Bc7Unorm),
            (TextureFormat::Bc1Srgb, TextureDestination::Bc1Srgb),
            (TextureFormat::Bc3Srgb, TextureDestination::Bc3Srgb),
            (TextureFormat::Bc7Srgb, TextureDestination::Bc7Srgb),
            (
                TextureFormat::Astc4x4Unorm,
                TextureDestination::Astc4x4Unorm,
            ),
            (TextureFormat::Astc4x4Srgb, TextureDestination::Astc4x4Srgb),
            (TextureFormat::Rgba8Unorm, TextureDestination::Rgba8Unorm),
        ] {
            exercise_case(backend, context, &quad, format, destination, width, height);
        }
    }
    assert_eq!(wait_idle(context), EzGfxResult::Ok);
    release_indirect(context, quad.indirect);
    release_structured(context, quad.positions);
    destroy_shader(context, quad.shader);
    destroy_index_heap(context);
    drop(native);
}

struct PixelData {
    initial: Vec<u8>,
    red: Vec<u8>,
    green: Vec<u8>,
    updated: Vec<u8>,
}

fn exercise_case(
    backend: u8,
    context: ContextHandle,
    quad: &Quad,
    format: TextureFormat,
    destination: TextureDestination,
    width: u32,
    height: u32,
) {
    // Each concurrently running backend owns a distinct decoder ID; registration is RAII.
    let decoder = Decoder::register(128 + backend, format, width, height);
    let config = TextureConfig {
        width,
        height,
        mip_count: 1,
        destination,
        sampler: TextureSamplerDesc {
            min_filter: SamplerFilter::Nearest,
            mag_filter: SamplerFilter::Nearest,
            max_anisotropy: 1.0,
            address_u: SamplerAddressMode::Clamp,
            address_v: SamplerAddressMode::Clamp,
            address_w: SamplerAddressMode::Clamp,
        },
    };
    let initial = adjacent_blocks(format, 0, width, height);
    let mut green = solid_block(format, 1);
    let mut red = solid_block(format, 0);
    if format == TextureFormat::Rgba8Unorm {
        green.truncate(height as usize * 16);
        red.truncate(height as usize * 16);
    }
    let updated = adjacent_blocks(format, 1, width, height);
    let pixels = PixelData {
        initial,
        red,
        green,
        updated,
    };
    let texture = match load_texture(context, decoder.source(), &pixels.initial, false, &config) {
        Ok(texture) => texture,
        Err(status)
            if matches!(
                format,
                TextureFormat::Astc4x4Unorm | TextureFormat::Astc4x4Srgb
            ) =>
        {
            // Explicit ASTC admission consults native compression support; never fall back.
            assert_eq!(status, EzGfxResult::Unsupported);
            return;
        }
        Err(status) => panic!("{format:?} admission failed: {status:?}"),
    };
    let status = await_texture_status(context, texture);
    if backend == 2
        && width == 7
        && matches!(
            format,
            TextureFormat::Bc1Unorm
                | TextureFormat::Bc1Srgb
                | TextureFormat::Bc3Unorm
                | TextureFormat::Bc3Srgb
                | TextureFormat::Bc7Unorm
                | TextureFormat::Bc7Srgb
        )
    {
        // DX12 rejects odd BC bases, but must support the same extent at a valid mip.
        assert_eq!(status, EzGfxResult::Unsupported, "{format:?} odd base");
        assert_eq!(
            poll_texture_load(context, texture),
            EzGfxResult::Unsupported
        );
        unload_texture(context, texture);
        assert_abi_odd_base_unsupported(context, &decoder, format, &pixels.initial);
        drop(decoder);
        exercise_dx12_odd_mip(context, quad, format, config);
        return;
    }
    if matches!(
        format,
        TextureFormat::Astc4x4Unorm | TextureFormat::Astc4x4Srgb
    ) && status == EzGfxResult::Unsupported
    {
        // Custom decoders reveal their format asynchronously; rejection can follow admission.
        unload_texture(context, texture);
        assert_stale(context, texture, &pixels.green);
        return;
    }
    assert_eq!(status, EzGfxResult::Ok, "{format:?} upload failed");
    quad.draw(texture, true);
    let before = frame_readback(context).unwrap();
    assert_halves(&before, format, 0, width);

    if backend == 1 && format == TextureFormat::Bc1Unorm && width == 8 {
        // Vulkan retains the submitted slot until owner-thread frame reclamation.
        // This forces the real descriptor-publication gate even after copies complete.
        quad.draw(texture, false);
        let pending =
            load_texture(context, decoder.source(), &pixels.initial, false, &config).unwrap();
        let deadline = Instant::now() + Duration::from_millis(100);
        while Instant::now() < deadline {
            assert_eq!(poll_texture_load(context, pending), EzGfxResult::NotReady);
            assert_eq!(
                texture_binding(context, pending),
                Err(EzGfxResult::NotReady)
            );
            if let Ok((resident, total)) = texture_residency(context, pending) {
                assert_eq!((resident, total), (0, 1));
            }
            std::thread::yield_now();
        }
        assert_eq!(wait_idle(context), EzGfxResult::Ok);
        await_texture(context, pending);
        assert_eq!(texture_residency(context, pending), Ok((1, 1)));
        quad.draw(pending, true);
        assert_halves(&frame_readback(context).unwrap(), format, 0, width);
        unload_texture(context, pending);
    }
    let after = exercise_updates(context, quad, texture, format, &config, &pixels, &before);
    exercise_retirement(context, quad, texture, &decoder, &config, &pixels, &after);
}

fn exercise_updates(
    context: ContextHandle,
    quad: &Quad,
    texture: TextureHandle,
    format: TextureFormat,
    config: &TextureConfig,
    pixels: &PixelData,
    before: &[u8],
) -> Vec<u8> {
    let (width, height) = (config.width, config.height);
    // Capture synchronizes readback. Submit a fresh uncaptured sampling draw so the update
    // must preserve a submitted read rather than relying on the earlier capture's wait.
    quad.draw(texture, false);
    assert_eq!(
        update_texture_region(
            context,
            texture,
            TextureRegion {
                mip_level: 0,
                x: 0,
                y: 0,
                width: 4,
                height,
                bytes: &pixels.green,
            }
        ),
        EzGfxResult::Ok
    );
    await_texture(context, texture);
    quad.draw(texture, true);
    let after = frame_readback(context).unwrap();
    assert_halves(&after, format, 1, width);
    let split = (256 + width / 2) / width;
    for y in 0..64 {
        let right = (y * 64 + split as usize) * 4..(y + 1) * 64 * 4;
        assert_eq!(
            before[right.clone()],
            after[right],
            "untouched blue block changed"
        );
    }

    // Queue an update and immediately record/submit a sampling draw. Its GPU dependency,
    // not a CPU poll or idle, must make the changed pixels visible.
    let binding = texture_binding(context, texture).unwrap();
    assert_eq!(
        update_texture_region(
            context,
            texture,
            TextureRegion {
                mip_level: 0,
                x: 0,
                y: 0,
                width: 4,
                height,
                bytes: &pixels.red,
            }
        ),
        EzGfxResult::Ok
    );
    quad.record(binding, true);
    assert_eq!(finish_render(context), EzGfxResult::Ok);
    assert_eq!(frame_readback(context).unwrap(), before);

    // A dependency captured while recording becomes stale if an update is queued before
    // submission. Submission must refresh it rather than sample the prior red contents.
    quad.record(binding, true);
    assert_eq!(
        update_texture_region(
            context,
            texture,
            TextureRegion {
                mip_level: 0,
                x: 0,
                y: 0,
                width: 4,
                height,
                bytes: &pixels.green,
            }
        ),
        EzGfxResult::Ok
    );
    assert_eq!(finish_render(context), EzGfxResult::Ok);
    assert_eq!(frame_readback(context).unwrap(), after);

    // The final block may contain fewer than four columns/rows, but its payload stays whole.
    // Updating that clipped edge must leave every pixel sampled from the left block intact.
    if width == 7 && format != TextureFormat::Rgba8Unorm {
        assert_eq!(
            update_texture_region(
                context,
                texture,
                TextureRegion {
                    mip_level: 0,
                    x: 4,
                    y: 0,
                    width: 3,
                    height,
                    bytes: &pixels.red,
                }
            ),
            EzGfxResult::Ok
        );
        quad.record(binding, true);
        assert_eq!(finish_render(context), EzGfxResult::Ok);
        let edge = frame_readback(context).unwrap();
        for (index, pixel) in edge.chunks_exact(4).enumerate() {
            let expected = if index % 64 < split as usize {
                &after[index * 4..index * 4 + 4]
            } else {
                &before[..4]
            };
            assert_eq!(pixel, expected, "{format:?} edge pixel {index}");
        }
    }
    after
}

fn exercise_retirement(
    context: ContextHandle,
    quad: &Quad,
    texture: TextureHandle,
    decoder: &Decoder,
    config: &TextureConfig,
    pixels: &PixelData,
    after: &[u8],
) {
    let original_binding = texture_binding(context, texture).unwrap();
    quad.draw(texture, false);
    unload_texture(context, texture);
    assert_stale(context, texture, &pixels.green);
    let mut retired_bindings = std::collections::HashSet::from([original_binding]);
    let mut reused = false;
    for _ in 0..32 {
        let replacement =
            load_texture(context, decoder.source(), &pixels.updated, false, config).unwrap();
        // Unload has already run against submitted work. Draining afterward lets the next
        // upload publish descriptors without depending on frame-slot reclamation timing.
        assert_eq!(wait_idle(context), EzGfxResult::Ok);
        await_texture(context, replacement);
        reused |= !retired_bindings.insert(texture_binding(context, replacement).unwrap());
        quad.draw(replacement, false);
        // No idle or capture between submission and unload: retirement must retain native
        // storage and its descriptor until that submitted frame has stopped sampling it.
        unload_texture(context, replacement);
        assert_stale(context, replacement, &pixels.green);
    }
    assert!(
        reused,
        "no retired descriptor binding was reclaimed in 32 cycles"
    );

    let admitted = load_texture(context, decoder.source(), &pixels.updated, false, config).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        // Residency succeeds once native admission exists; do not await full upload first.
        match texture_residency(context, admitted) {
            Ok(_) => break,
            Err(EzGfxResult::NotReady) => {
                assert!(Instant::now() < deadline, "native admission timed out");
                std::thread::yield_now();
            }
            other => panic!("native admission failed: {other:?}"),
        }
    }
    unload_texture(context, admitted);
    assert_stale(context, admitted, &pixels.green);

    let final_texture =
        load_texture(context, decoder.source(), &pixels.updated, false, config).unwrap();
    assert_eq!(wait_idle(context), EzGfxResult::Ok);
    await_texture(context, final_texture);
    quad.draw(final_texture, true);
    assert_eq!(
        frame_readback(context).unwrap(),
        after,
        "retirement changed final pixels"
    );
    unload_texture(context, final_texture);
    assert_eq!(wait_idle(context), EzGfxResult::Ok);
}

fn assert_abi_odd_base_unsupported(
    context: ContextHandle,
    decoder: &Decoder,
    format: TextureFormat,
    bytes: &[u8],
) {
    use ez_gfx_ffi::{
        EzGfxTextureDesc, ez_gfx_texture_load, ez_gfx_texture_poll, ez_gfx_texture_unload,
    };

    // Call the exported C ABI, including descriptor validation and asynchronous status mapping.
    let destination_format = match format {
        TextureFormat::Bc1Unorm => 3,
        TextureFormat::Bc1Srgb => 4,
        TextureFormat::Bc3Unorm => 5,
        TextureFormat::Bc3Srgb => 6,
        TextureFormat::Bc7Unorm => 7,
        TextureFormat::Bc7Srgb => 8,
        _ => panic!("odd-base admission requires BC"),
    };
    let desc = EzGfxTextureDesc {
        source_format: decoder.0,
        destination_format,
        width: 7,
        height: 3,
        mip_count: 1,
        generate_mips: 0,
        min_filter: 0,
        mag_filter: 0,
        max_anisotropy: 1.0,
        address_mode_u: 1,
        address_mode_v: 1,
        address_mode_w: 1,
        debug_label: core::ptr::null(),
        debug_label_length: 0,
    };
    let mut texture = 0;
    // SAFETY: Input/descriptor remain readable and the output handle writable through the call.
    assert_eq!(
        unsafe {
            ez_gfx_texture_load(
                bytes.as_ptr(),
                bytes.len(),
                &raw const desc,
                &raw mut texture,
                context.into_raw(),
            )
        },
        EzGfxResult::Ok
    );
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let status = ez_gfx_texture_poll(texture, context.into_raw());
        if status != EzGfxResult::NotReady {
            assert_eq!(status, EzGfxResult::Unsupported, "{format:?} ABI odd base");
            break;
        }
        assert!(
            Instant::now() < deadline,
            "ABI odd-base admission timed out"
        );
        std::thread::yield_now();
    }
    assert_eq!(
        ez_gfx_texture_poll(texture, context.into_raw()),
        EzGfxResult::Unsupported
    );
    ez_gfx_texture_unload(texture, context.into_raw());
}

fn exercise_dx12_odd_mip(
    context: ContextHandle,
    quad: &Quad,
    format: TextureFormat,
    mut config: TextureConfig,
) {
    // Only the coarse 7x3 mip is exposed; no padded extent or UV compensation is involved.
    let decoder = Decoder::register_chain(130, format);
    config.width = 28;
    config.height = 12;
    config.mip_count = 3;
    let red = solid_block(format, 0);
    let green = solid_block(format, 1);
    let mut bytes = green.repeat(21 + 8);
    bytes.extend(adjacent_blocks(format, 0, 7, 3));
    let texture = load_texture(context, decoder.source(), &bytes, false, &config).unwrap();
    await_texture(context, texture);
    assert_eq!(set_texture_residency(context, texture, 1), EzGfxResult::Ok);
    assert_eq!(texture_residency(context, texture), Ok((1, 3)));
    quad.draw(texture, true);
    let before = frame_readback(context).unwrap();
    assert_halves(&before, format, 0, 7);
    let binding = texture_binding(context, texture).unwrap();

    // Preserve a submitted read, then sample the queued left-edge copy without a CPU wait.
    quad.draw(texture, false);
    assert_eq!(
        update_texture_region(
            context,
            texture,
            TextureRegion {
                mip_level: 2,
                x: 0,
                y: 0,
                width: 4,
                height: 3,
                bytes: &green,
            }
        ),
        EzGfxResult::Ok
    );
    quad.record(binding, true);
    assert_eq!(finish_render(context), EzGfxResult::Ok);
    let after = frame_readback(context).unwrap();
    assert_halves(&after, format, 1, 7);

    // The final 3x3 region still consumes one complete compressed block. Queue after recording
    // to require submission-time dependency refresh as well as preservation of untouched texels.
    quad.record(binding, true);
    assert_eq!(
        update_texture_region(
            context,
            texture,
            TextureRegion {
                mip_level: 2,
                x: 4,
                y: 0,
                width: 3,
                height: 3,
                bytes: &red,
            }
        ),
        EzGfxResult::Ok
    );
    assert_eq!(finish_render(context), EzGfxResult::Ok);
    let edge = frame_readback(context).unwrap();
    for (index, pixel) in edge.chunks_exact(4).enumerate() {
        let left = (index % 64 * 2 + 1) * 7 < 512;
        if !left {
            assert_eq!(
                &before[index * 4..index * 4 + 4],
                &after[index * 4..index * 4 + 4]
            );
        }
        let expected = if left {
            &after[index * 4..index * 4 + 4]
        } else {
            &before[..4]
        };
        assert_eq!(pixel, expected, "{format:?} mip2 edge pixel {index}");
    }
    quad.draw(texture, false);
    unload_texture(context, texture);
    assert_stale(context, texture, &green);
    assert_eq!(wait_idle(context), EzGfxResult::Ok);
}

fn await_texture(context: ContextHandle, texture: TextureHandle) {
    assert_eq!(await_texture_status(context, texture), EzGfxResult::Ok);
}

fn await_texture_status(context: ContextHandle, texture: TextureHandle) -> EzGfxResult {
    // Polling drives worker admission and completion; a bounded deadline also catches deadlocks.
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match poll_texture_load(context, texture) {
            EzGfxResult::Ok => return EzGfxResult::Ok,
            EzGfxResult::NotReady => {
                assert!(Instant::now() < deadline, "sample-ready handoff timed out");
                std::thread::yield_now();
            }
            status => return status,
        }
    }
}

fn assert_stale(context: ContextHandle, texture: TextureHandle, bytes: &[u8]) {
    // Stale generations must fail even when their descriptor slot is recycled later.
    assert_eq!(
        texture_binding(context, texture),
        Err(EzGfxResult::InvalidContext)
    );
    assert_eq!(
        poll_texture_load(context, texture),
        EzGfxResult::InvalidContext
    );
    assert_eq!(
        update_texture_region(
            context,
            texture,
            TextureRegion {
                mip_level: 0,
                x: 0,
                y: 0,
                width: 4,
                height: 4,
                bytes,
            }
        ),
        EzGfxResult::InvalidContext
    );
}

fn assert_halves(pixels: &[u8], format: TextureFormat, left_channel: usize, width: u32) {
    assert_eq!(pixels.len(), 64 * 64 * 4);
    // Mode 6 shares endpoint p-bits: low channels are 1/255 linear, yielding 13 in sRGB.
    // Nearest sampling and identical rows avoid interpolation and Y-orientation ambiguity.
    let low = match format {
        TextureFormat::Bc7Unorm => 13,
        TextureFormat::Bc7Srgb => 1,
        _ => 0,
    };
    for (index, pixel) in pixels.chunks_exact(4).enumerate() {
        let channel = if (index % 64 * 2 + 1) * (width as usize) < 512 {
            left_channel
        } else {
            2
        };
        let mut expected = [low, low, low, 255];
        // The sRGB target re-encodes sampled linear RGB. Midtones distinguish sRGB from UNORM.
        expected[channel] = match format {
            TextureFormat::Bc1Srgb | TextureFormat::Bc3Srgb => [132, 130, 132][channel],
            TextureFormat::Bc7Srgb => 129,
            TextureFormat::Astc4x4Srgb => 128,
            _ => 255,
        };
        assert_eq!(
            pixel,
            expected,
            "{format:?} pixel ({}, {})",
            index % 64,
            index / 64
        );
    }
}

struct Decoder(u8);

impl Decoder {
    fn register(id: u8, format: TextureFormat, width: u32, height: u32) -> Self {
        // These fixtures contain exactly two block columns and one (possibly clipped) block row.
        assert!((5..=8).contains(&width) && (1..=4).contains(&height));
        register_texture_decoder(
            id,
            Arc::new(move |bytes, _| {
                // Reject malformed custom payloads instead of padding missing compressed blocks.
                let expected = match format {
                    TextureFormat::Bc1Unorm | TextureFormat::Bc1Srgb => 16,
                    TextureFormat::Rgba8Unorm => width as usize * height as usize * 4,
                    _ => 32,
                };
                if bytes.len() != expected {
                    return Err(TextureError::InvalidData);
                }
                Ok(DecodedTexture {
                    width,
                    height,
                    mip_count: 1,
                    format,
                    mips: vec![DecodedMip {
                        width,
                        height,
                        bytes: bytes.to_vec(),
                    }],
                })
            }),
        )
        .unwrap();
        Self(id)
    }

    fn register_chain(id: u8, format: TextureFormat) -> Self {
        // Three real BC mips use ceil-divided block counts, including both clipped mip2 edges.
        let block_bytes = solid_block(format, 0).len();
        register_texture_decoder(
            id,
            Arc::new(move |bytes, _| {
                if bytes.len() != (21 + 8 + 2) * block_bytes {
                    return Err(TextureError::InvalidData);
                }
                let mut offset = 0;
                let mips = [(28, 12, 21), (14, 6, 8), (7, 3, 2)]
                    .into_iter()
                    .map(|(width, height, blocks)| {
                        let end = offset + blocks * block_bytes;
                        let mip = DecodedMip {
                            width,
                            height,
                            bytes: bytes[offset..end].to_vec(),
                        };
                        offset = end;
                        mip
                    })
                    .collect();
                Ok(DecodedTexture {
                    width: 28,
                    height: 12,
                    mip_count: 3,
                    format,
                    mips,
                })
            }),
        )
        .unwrap();
        Self(id)
    }

    fn source(&self) -> TextureSource {
        TextureSource::Custom(self.0)
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // All loads copy/retain their callback at admission; unregister cannot invalidate work.
        unregister_texture_decoder(self.0).unwrap();
    }
}

fn adjacent_blocks(format: TextureFormat, left_channel: usize, width: u32, height: u32) -> Vec<u8> {
    // Width four would erase the untouched neighbor; larger dimensions need additional blocks.
    assert!((5..=8).contains(&width) && (1..=4).contains(&height));
    let left = solid_block(format, left_channel);
    let right = solid_block(format, 2);
    // Compressed blocks are contiguous; RGBA regions must instead be interleaved per texel row.
    let row_bytes = if format == TextureFormat::Rgba8Unorm {
        16
    } else {
        left.len()
    };
    let mut bytes = Vec::with_capacity(left.len() + right.len());
    for (row, (left_row, right_row)) in left
        .chunks_exact(row_bytes)
        .zip(right.chunks_exact(row_bytes))
        .enumerate()
    {
        if format == TextureFormat::Rgba8Unorm && row >= height as usize {
            break;
        }
        bytes.extend_from_slice(left_row);
        let right_len = if format == TextureFormat::Rgba8Unorm {
            (width as usize - 4) * 4
        } else {
            right_row.len()
        };
        bytes.extend_from_slice(&right_row[..right_len]);
    }
    bytes
}

fn solid_block(format: TextureFormat, channel: usize) -> Vec<u8> {
    assert!(channel < 3, "solid block requires an RGB channel");
    match format {
        TextureFormat::Rgba8Unorm => {
            let mut pixel = [0, 0, 0, 255];
            pixel[channel] = 255;
            pixel.repeat(16)
        }
        TextureFormat::Bc1Unorm
        | TextureFormat::Bc3Unorm
        | TextureFormat::Bc1Srgb
        | TextureFormat::Bc3Srgb => {
            let endpoints = if matches!(format, TextureFormat::Bc1Srgb | TextureFormat::Bc3Srgb) {
                [0x8000_u16, 0x0400, 0x0010]
            } else {
                [0xf800_u16, 0x07e0, 0x001f]
            };
            let endpoint = endpoints[channel].to_le_bytes();
            // Index zero selects endpoint zero even for BC1's three-color endpoint ordering.
            let color = [endpoint[0], endpoint[1], 0, 0, 0, 0, 0, 0];
            if matches!(format, TextureFormat::Bc1Unorm | TextureFormat::Bc1Srgb) {
                color.to_vec()
            } else {
                let mut block = vec![255, 255, 0, 0, 0, 0, 0, 0];
                block.extend(color);
                block
            }
        }
        TextureFormat::Bc7Unorm | TextureFormat::Bc7Srgb => {
            // Mode 6: unary mode bit, component-major paired 7-bit endpoints, two p-bits,
            // then zero interpolation indices (the anchor index has only three bits).
            let mut bits = 1_u128 << 6;
            let mut offset = 7;
            for component in 0..4 {
                let endpoint = if component == channel && format == TextureFormat::Bc7Srgb {
                    64_u128
                } else if component == channel || component == 3 {
                    127_u128
                } else {
                    0
                };
                for _ in 0..2 {
                    bits |= endpoint << offset;
                    offset += 7;
                }
            }
            bits |= 3_u128 << offset;
            bits.to_le_bytes().to_vec()
        }
        TextureFormat::Astc4x4Unorm | TextureFormat::Astc4x4Srgb => {
            // LDR void-extent block: reserved bits and all four 13-bit extents are ones.
            // RGBA endpoints are UNORM16, not float16; 0xffff extents cover the whole block.
            let mut block = vec![0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff];
            for component in 0..4 {
                let value = if component == 3 {
                    u16::MAX
                } else if component == channel {
                    if format == TextureFormat::Astc4x4Srgb {
                        128 * 257
                    } else {
                        u16::MAX
                    }
                } else {
                    0
                };
                block.extend_from_slice(&value.to_le_bytes());
            }
            block
        }
        TextureFormat::Rgba8Srgb => panic!("unsupported regression format: {format:?}"),
    }
}

struct Quad {
    context: ContextHandle,
    surface: SurfaceHandle,
    shader: ShaderHandle,
    positions: StructuredBufferHandle,
    indirect: IndirectBufferHandle,
}

impl Quad {
    fn create(context: ContextHandle, surface: SurfaceHandle, artifact: &[u8]) -> Self {
        // The cube shader derives UVs from vertex_index & 3; one face covers the entire target.
        let vertices = [
            [-1.0_f32, -1.0, 0.5, 1.0],
            [1.0, -1.0, 0.5, 1.0],
            [1.0, 1.0, 0.5, 1.0],
            [-1.0, 1.0, 0.5, 1.0],
        ];
        let positions_bytes: Vec<u8> = vertices
            .into_iter()
            .flatten()
            .flat_map(f32::to_ne_bytes)
            .collect();
        let index_bytes: Vec<u8> = [0_u32, 1, 2, 2, 3, 0]
            .into_iter()
            .flat_map(u32::to_ne_bytes)
            .collect();
        assert_eq!(create_index_heap(context, 24), EzGfxResult::Ok);
        let first_index = upload_indices(context, 6, &index_bytes).unwrap();
        let positions = acquire_structured(context, 64).unwrap();
        assert_eq!(
            write_structured(context, positions, &positions_bytes),
            EzGfxResult::Ok
        );
        let indirect = acquire_indirect(context, 1).unwrap();
        assert_eq!(
            write_indirect(
                context,
                indirect,
                0,
                DrawIndexedCommand {
                    index_count: 6,
                    instance_count: 1,
                    first_index,
                    vertex_offset: 0,
                    first_instance: 0,
                }
            ),
            EzGfxResult::Ok
        );
        assert_eq!(set_indirect_count(context, indirect, 1), EzGfxResult::Ok);
        Self {
            context,
            surface,
            shader: load_shader(context, artifact).unwrap(),
            positions,
            indirect,
        }
    }

    fn draw(&self, texture: TextureHandle, capture: bool) {
        self.record(texture_binding(self.context, texture).unwrap(), capture);
        assert_eq!(finish_render(self.context), EzGfxResult::Ok);
    }

    fn record(&self, texture_id: u32, capture: bool) {
        // Use a previously resolved stable binding to exercise queued updates without CPU polling.
        let mut push = [0_u8; 80];
        for diagonal in [0, 5, 10, 15] {
            push[diagonal * 4..diagonal * 4 + 4].copy_from_slice(&1.0_f32.to_ne_bytes());
        }
        push[64..68].copy_from_slice(&texture_id.to_ne_bytes());
        assert_eq!(
            set_snapshot_cache(self.context, self.surface, capture),
            EzGfxResult::Ok
        );
        assert_eq!(begin_render(self.context, self.surface), EzGfxResult::Ok);
        assert_eq!(
            render_add_graphics(
                self.context,
                self.shader,
                self.indirect,
                &[PublicBinding {
                    name: "positions".to_owned(),
                    resource: ResourceIdentity::Structured(self.positions)
                }],
                // Disable culling: Vulkan and DX12 may use opposite framebuffer Y conventions.
                DynamicPipelineState::from_abi(0, 0, 0, 0).unwrap(),
                &push,
            ),
            EzGfxResult::Ok
        );
    }
}
