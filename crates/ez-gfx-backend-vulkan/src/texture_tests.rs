use super::*;
use crate::{
    NativeBufferBinding, NativeComputeDispatch, NativeFrameAction, NativePipeline, NativeShader,
    SurfacePlatform,
};
use ez_gfx_compiler::{Target, compile_shader};
use ez_gfx_core::{Backend, capability::SemanticProfile};
use ez_gfx_runtime::shader::RuntimeShader;
use std::{
    path::PathBuf,
    sync::{Arc, atomic::AtomicBool},
    time::{Duration, Instant},
};

const GPU_TIMEOUT: u64 = 10_000_000_000;

struct QueueGate {
    device: ash::Device,
    fence: vk::Fence,
    submitted: bool,
    semaphore: vk::Semaphore,
    released: bool,
}

impl QueueGate {
    fn new(device: &ash::Device, queue: vk::Queue) -> Self {
        let mut kind =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        // SAFETY: this device enables timeline semaphores; all submit storage spans the call.
        let semaphore = unsafe {
            device.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut kind),
                None,
            )
        }
        .unwrap();
        // SAFETY: the unsignaled fence is private to this gate submission.
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }.unwrap();
        let mut gate = Self {
            device: device.clone(),
            fence,
            submitted: false,
            semaphore,
            released: false,
        };
        let waits = [semaphore];
        let values = [1];
        let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
        let mut timeline =
            vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&values);
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&waits)
            .wait_dst_stage_mask(&stages)
            .push_next(&mut timeline);
        // SAFETY: the worker is flushed before insertion; no concurrent submission uses this queue.
        unsafe { device.queue_submit(queue, &[submit], fence) }.unwrap();
        gate.submitted = true;
        gate
    }

    fn release(&mut self) {
        // Releasing twice is invalid for a timeline; unwinding after explicit release is harmless.
        if !self.released {
            // SAFETY: this gate owns a live timeline whose current value is zero.
            unsafe {
                self.device.signal_semaphore(
                    &vk::SemaphoreSignalInfo::default()
                        .semaphore(self.semaphore)
                        .value(1),
                )
            }
            .unwrap();
            self.released = true;
        }
    }
}

impl Drop for QueueGate {
    fn drop(&mut self) {
        // The guard is declared after the context, so failure releases the GPU before context shutdown.
        if !self.released {
            // SAFETY: the retained device and semaphore outlive the queued wait.
            let _ = unsafe {
                self.device.signal_semaphore(
                    &vk::SemaphoreSignalInfo::default()
                        .semaphore(self.semaphore)
                        .value(1),
                )
            };
        }
        // SAFETY: fence completion retires this semaphore's sole queued wait. Unlike queue-idle,
        // a fence wait does not require external queue synchronization with the transfer worker.
        unsafe {
            if self.submitted {
                let _ = self.device.wait_for_fences(&[self.fence], true, u64::MAX);
            }
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_semaphore(self.semaphore, None);
        }
    }
}

struct ShaderSource(PathBuf);
impl Drop for ShaderSource {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn context() -> NativeContext {
    // No surface is created, shown, or activated by these tests.
    let mut context = NativeContext::create(false, false, SurfacePlatform::Win32).unwrap();
    context.init_device(None).unwrap();
    context
}

fn texture_queue(context: &NativeContext) -> vk::Queue {
    let family = context.transfer_queue_family.unwrap();
    assert_ne!(
        Some(family),
        context.graphics_queue_family,
        "deterministic proof requires a dedicated transfer family"
    );
    // SAFETY: this physical device belongs to the live instance; device creation uses min(count, 2).
    let families = unsafe {
        context
            .instance
            .get_physical_device_queue_family_properties(context.physical_device.unwrap())
    };
    let index = u32::from(families[family as usize].queue_count > 1);
    // SAFETY: device initialization requested this exact family/index for its texture worker.
    unsafe {
        context
            .device
            .as_ref()
            .unwrap()
            .get_device_queue(family, index)
    }
}

fn empty_texture(context: &mut NativeContext, binding: u32) -> NativeTexture {
    // Both levels start undefined; only explicitly uploaded subresources may be published.
    let device = context.device.as_ref().unwrap().clone();
    let (image, allocation) = context
        .create_texture_image(&device, 2, 2, 2, TextureFormat::Rgba8Unorm)
        .unwrap();
    // SAFETY: the image contains mip 1, and neither object is exposed until transfer completion.
    let view = unsafe {
        device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(1)
                        .level_count(1)
                        .layer_count(1),
                ),
            None,
        )
    }
    .unwrap();
    // SAFETY: nearest filtering needs no optional anisotropy feature.
    let sampler = unsafe {
        device.create_sampler(
            &vk::SamplerCreateInfo::default()
                .min_filter(vk::Filter::NEAREST)
                .mag_filter(vk::Filter::NEAREST)
                .max_lod(2.0),
            None,
        )
    }
    .unwrap();
    NativeTexture {
        image,
        view,
        allocation,
        sampler,
        format: TextureFormat::Rgba8Unorm,
        width: 2,
        height: 2,
        mip_count: 2,
        resident_mips: 0,
        mip_completions: vec![0; 2],
        cancellation: Arc::new(AtomicBool::new(false)),
        binding,
    }
}

fn upload(
    context: &mut NativeContext,
    textures: &mut [&mut NativeTexture],
    mip: u32,
    colors: &[[u8; 4]],
) -> NativeAllocation {
    assert_eq!(textures.len(), colors.len());
    assert!(mip < 2);
    let width = 2 >> mip;
    let bytes_per_image = u64::from(width * width * 4);
    let size = bytes_per_image * textures.len() as u64;
    let mut staging = context
        .allocate(AllocationRequest::new(size, 4, MemoryClass::Upload, true, None).unwrap())
        .unwrap();
    for (bytes, color) in context.mapped_slice_mut(&mut staging).unwrap()
        [..usize::try_from(size).unwrap()]
        .chunks_exact_mut(usize::try_from(bytes_per_image).unwrap())
        .zip(colors)
    {
        for pixel in bytes.chunks_exact_mut(4) {
            pixel.copy_from_slice(color);
        }
    }
    context.flush(&mut staging, 0, size).unwrap();
    let mut jobs = Vec::new();
    for (index, texture) in textures.iter_mut().enumerate() {
        let value = context.next_texture_value;
        context.next_texture_value += 1;
        texture.mip_completions[mip as usize] = value;
        jobs.push(VulkanTransferJob {
            value,
            bytes: bytes_per_image,
            cancelled: Some(texture.cancellation.clone()),
            copy: VulkanTransferCopy::Texture {
                source: staging.buffer,
                destination: texture.image,
                initialized: false,
                stream_stage: 1 - mip,
                region: vk::BufferImageCopy::default()
                    .buffer_offset(index as u64 * bytes_per_image)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(mip)
                            .layer_count(1),
                    )
                    .image_extent(vk::Extent3D {
                        width,
                        height: width,
                        depth: 1,
                    }),
            },
        });
    }
    // Each call writes a previously undefined mip, never the exposed coarse mip.
    context
        .texture_worker
        .as_ref()
        .unwrap()
        .submit_batch(jobs)
        .unwrap();
    staging
}

fn wait_texture(context: &NativeContext, value: u64) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while context.completed_texture_transfer_value().unwrap() < value {
        assert!(Instant::now() < deadline, "texture transfer timed out");
        std::thread::yield_now();
    }
    context.texture_worker.as_ref().unwrap().flush().unwrap();
}

fn sampler_pipeline(context: &NativeContext, name: &str) -> (NativeShader, NativePipeline) {
    use std::io::Write;
    // Distinct test names isolate simultaneous shader compilations; create_new rejects stale files.
    let source = ShaderSource(
        std::env::temp_dir().join(format!("ezgfx-vulkan-{name}-{}.slang", std::process::id())),
    );
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&source.0)
        .unwrap();
    file.write_all(br#"[__AttributeUsage(_AttributeTargets.Var)]
struct StructuredBufferAttribute { string name; };
[__AttributeUsage(_AttributeTargets.Var)]
struct BindlessTextureHeapAttribute { int capacity; };
struct TextureEntry { Texture2D<float4> texture; SamplerState sampler; };
struct TextureHeap { TextureEntry entries[1024]; };
[StructuredBuffer("values")] RWStructuredBuffer<uint> values;
[BindlessTextureHeap(1024)] ParameterBlock<TextureHeap> texture_heap;
[shader("compute")] [numthreads(1, 1, 1)]
void computemain(uint3 id : SV_DispatchThreadID) {
    float4 color = texture_heap.entries[id.x].texture.SampleLevel(texture_heap.entries[id.x].sampler, float2(0.5, 0.5), 0);
    values[id.x] = uint(round(color.r * 255)) | (uint(round(color.g * 255)) << 8) | (uint(round(color.b * 255)) << 16) | (uint(round(color.a * 255)) << 24);
}
"#).unwrap();
    drop(file);
    let artifact = compile_shader(&source.0, &[Target::Spirv], false).unwrap();
    let runtime = RuntimeShader::load(&artifact, Backend::Vulkan, SemanticProfile::V1).unwrap();
    let (index, _, entry) = runtime.compute_product().unwrap();
    let requirements = runtime.bindings(ez_gfx_artifact::Stage::Compute).unwrap();
    let layouts = requirements
        .requirements()
        .iter()
        .map(|binding| {
            crate::ShaderBufferLayout::new(
                binding.space,
                binding.binding,
                binding.descriptor_count,
                binding.writable,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let products = runtime
        .products()
        .map(|(_, bytes)| bytes)
        .collect::<Vec<_>>();
    let shader = context.create_shader(&products).unwrap();
    let pipeline = context
        .create_compute_pipeline(&shader, index, entry, &layouts)
        .unwrap();
    (shader, pipeline)
}

fn dispatch(
    context: &mut NativeContext,
    pipeline: &NativePipeline,
    output: &NativeAllocation,
    count: u32,
) -> usize {
    let slot = context.frame_cursor;
    context
        .execute_frame(
            None,
            &[NativeFrameAction::Compute(NativeComputeDispatch {
                pipeline,
                groups: [count, 1, 1],
                push_constants: &[],
                bindings: &[NativeBufferBinding {
                    allocation: output,
                    offset: 0,
                    range: u64::from(count) * 4,
                    writable: true,
                }],
            })],
            false,
        )
        .unwrap();
    slot
}

fn frame_pixels(
    context: &mut NativeContext,
    slot: usize,
    output: &mut NativeAllocation,
    expected: &[[u8; 4]],
) {
    let device = context.device.as_ref().unwrap();
    // Only the graphics frame is waited: waiting device-idle would deadlock the intentionally gated copy.
    // SAFETY: this fence belongs to the submitted frame and output remains alive until completion.
    unsafe { device.wait_for_fences(&[context.frame_slots[slot].fence], true, GPU_TIMEOUT) }
        .unwrap();
    context.frame_slots[slot].in_flight = false;
    context.complete_frame_slot(slot).unwrap();
    context
        .invalidate(output, 0, expected.len() as u64 * 4)
        .unwrap();
    assert_eq!(
        &context.mapped_slice(output).unwrap()[..expected.len() * 4],
        expected.as_flattened()
    );
}

#[test]
fn coarse_sampling_frame_completes_while_fine_copy_is_gpu_blocked() {
    let mut context = context();
    let queue = texture_queue(&context);
    let (shader, pipeline) = sampler_pipeline(&context, "coarse");
    let mut texture = empty_texture(&mut context, 0);
    let coarse = upload(&mut context, &mut [&mut texture], 1, &[[255, 0, 0, 255]]);
    wait_texture(&context, texture.mip_completions[1]);
    context.publish_texture_mips(&mut texture, 1).unwrap();
    let mut output = context
        .allocate(AllocationRequest::new(4, 4, MemoryClass::Readback, true, None).unwrap())
        .unwrap();
    let mut gate = QueueGate::new(context.device.as_ref().unwrap(), queue);
    let submitted = crate::transfer::submission_observation::Observer::new(
        context.device.as_ref().unwrap(),
        queue,
    );
    let fine = upload(&mut context, &mut [&mut texture], 0, &[[0, 255, 0, 255]]);
    let required = texture.last_transfer_value();
    let deadline = Instant::now() + Duration::from_secs(10);
    while submitted.value() < required {
        assert!(
            Instant::now() < deadline,
            "fine native copy submission timed out"
        );
        std::thread::yield_now();
    }
    assert!(context.completed_texture_transfer_value().unwrap() < required);
    assert!(context.publish_texture_mips(&mut texture, 2).is_err());
    let slot = dispatch(&mut context, &pipeline, &output, 1);
    frame_pixels(&mut context, slot, &mut output, &[[255, 0, 0, 255]]);
    assert!(context.completed_texture_transfer_value().unwrap() < required);
    gate.release();
    wait_texture(&context, required);
    context.publish_texture_mips(&mut texture, 2).unwrap();
    let slot = dispatch(&mut context, &pipeline, &output, 1);
    frame_pixels(&mut context, slot, &mut output, &[[0, 255, 0, 255]]);
    drop(gate);
    context.destroy_texture(texture).unwrap();
    context.destroy_pipeline(pipeline);
    context.destroy_shader(shader);
    for allocation in [coarse, fine, output] {
        context.free(allocation).unwrap();
    }
    context.wait_idle().unwrap();
}

#[test]
fn cross_texture_bundle_uses_one_native_transfer_submission() {
    let mut context = context();
    texture_queue(&context);
    let (shader, pipeline) = sampler_pipeline(&context, "batch");
    let mut first = empty_texture(&mut context, 0);
    let mut second = empty_texture(&mut context, 1);
    let device = context.device.as_ref().unwrap().clone();
    let ownership = context.texture_ownership_timeline.unwrap();
    // SAFETY: this native timeline is owned by the context and advanced once per transfer batch.
    let before = unsafe { device.get_semaphore_counter_value(ownership) }.unwrap();
    let colors = [[23, 47, 89, 255], [191, 113, 59, 255]];
    let staging = upload(&mut context, &mut [&mut first, &mut second], 1, &colors);
    wait_texture(&context, second.last_transfer_value());
    // SAFETY: completion includes the matching graphics acquire, so the native copy signal is visible.
    let after = unsafe { device.get_semaphore_counter_value(ownership) }.unwrap();
    assert_eq!(
        after - before,
        1,
        "two images must share one native copy submission"
    );
    context.publish_texture_mips(&mut first, 1).unwrap();
    context.publish_texture_mips(&mut second, 1).unwrap();
    let mut output = context
        .allocate(AllocationRequest::new(8, 4, MemoryClass::Readback, true, None).unwrap())
        .unwrap();
    let slot = dispatch(&mut context, &pipeline, &output, 2);
    frame_pixels(&mut context, slot, &mut output, &colors);
    context.destroy_texture(first).unwrap();
    context.destroy_texture(second).unwrap();
    context.destroy_pipeline(pipeline);
    context.destroy_shader(shader);
    context.free(staging).unwrap();
    context.free(output).unwrap();
    context.wait_idle().unwrap();
}

#[test]
fn unsignaled_graphics_fence_retains_texture_and_frame_descriptors() {
    let mut context = context();
    let (shader, pipeline) = sampler_pipeline(&context, "retirement");
    let mut texture = empty_texture(&mut context, 0);
    let staging = upload(&mut context, &mut [&mut texture], 1, &[[61, 127, 193, 255]]);
    wait_texture(&context, texture.last_transfer_value());
    context.publish_texture_mips(&mut texture, 1).unwrap();
    let completion =
        CompletionToken::new(QueueKind::TextureTransfer, texture.last_transfer_value()).unwrap();
    let mut output = context
        .allocate(AllocationRequest::new(4, 4, MemoryClass::Readback, true, None).unwrap())
        .unwrap();
    let mut gate = QueueGate::new(
        context.device.as_ref().unwrap(),
        context.graphics_queue.unwrap(),
    );
    let slot = dispatch(&mut context, &pipeline, &output, 1);
    let image = texture.image;
    let view = texture.view;
    let descriptor_pool = context.frame_slots[slot].descriptor_pool;
    context.destroy_texture(texture).unwrap();
    // SAFETY: fence status is queried without modifying the gated submission.
    assert!(
        !unsafe {
            context
                .device
                .as_ref()
                .unwrap()
                .get_fence_status(context.frame_slots[slot].fence)
        }
        .unwrap()
    );
    assert!(!context.texture_retirement_ready(completion).unwrap());
    assert!(!context.texture_descriptor_update_ready());
    assert!(context.deferred.iter().any(|item| item.pending_slots & (1 << slot) != 0 && matches!(&item.resource, crate::DeferredResource::Texture(texture) if texture.image == image && texture.view == view)));
    assert_eq!(context.frame_slots[slot].descriptor_pool, descriptor_pool);
    gate.release();
    frame_pixels(&mut context, slot, &mut output, &[[61, 127, 193, 255]]);
    assert!(context.texture_retirement_ready(completion).unwrap());
    assert!(context.texture_descriptor_update_ready());
    assert!(context.deferred.is_empty());
    drop(gate);
    context.destroy_pipeline(pipeline);
    context.destroy_shader(shader);
    context.free(staging).unwrap();
    context.free(output).unwrap();
    context.wait_idle().unwrap();
}

fn nearest_sampler() -> TextureSamplerDesc {
    TextureSamplerDesc {
        min_filter: crate::SamplerFilter::Nearest,
        mag_filter: crate::SamplerFilter::Nearest,
        max_anisotropy: 1.0,
        address_u: crate::SamplerAddressMode::Clamp,
        address_v: crate::SamplerAddressMode::Clamp,
        address_w: crate::SamplerAddressMode::Clamp,
    }
}

#[test]
fn full_texture_queue_rejects_without_poisoning_idle_or_next_frame() {
    let mut context = context();
    let queue = texture_queue(&context);
    let (shader, pipeline) = sampler_pipeline(&context, "full-texture");
    let mut output = context
        .allocate(AllocationRequest::new(4, 4, MemoryClass::Readback, true, None).unwrap())
        .unwrap();
    let colors = [[17, 83, 149, 255]];
    let mips = [ImageMip {
        width: 1,
        height: 1,
        bytes: &colors[0],
    }];
    let mut gate = QueueGate::new(context.device.as_ref().unwrap(), queue);
    let submitted = crate::transfer::submission_observation::Observer::new(
        context.device.as_ref().unwrap(),
        queue,
    );
    let (first, tokens) = context
        .create_texture(TextureFormat::Rgba8Unorm, &mips, 0, nearest_sampler())
        .unwrap();
    let mut textures = vec![first];
    let deadline = Instant::now() + Duration::from_secs(10);
    while submitted.value() < tokens[0].value {
        assert!(
            Instant::now() < deadline,
            "initial native copy submission timed out"
        );
        std::thread::yield_now();
    }

    // The real owner is now blocked on transfer ownership; fill its bounded inbox, not a mock.
    let mut rejected = false;
    for _ in 0..65 {
        let before = context.next_texture_value;
        match context.create_texture(TextureFormat::Rgba8Unorm, &mips, 0, nearest_sampler()) {
            Ok((texture, _)) => textures.push(texture),
            Err(error) => {
                assert_eq!(error, AllocationError::OutOfMemory);
                assert_eq!(context.next_texture_value, before);
                rejected = true;
                break;
            }
        }
    }
    assert!(rejected, "bounded texture inbox must reject admission");
    gate.release();
    context.wait_idle().unwrap();
    context.publish_texture_mips(&mut textures[0], 1).unwrap();
    let slot = dispatch(&mut context, &pipeline, &output, 1);
    frame_pixels(&mut context, slot, &mut output, &colors);
    drop(gate);
    for texture in textures {
        context.destroy_texture(texture).unwrap();
    }
    context.destroy_pipeline(pipeline);
    context.destroy_shader(shader);
    context.free(output).unwrap();
    context.wait_idle().unwrap();
}

#[test]
fn full_buffer_queue_rejects_without_poisoning_idle() {
    use crate::BufferTransfer;
    let mut context = context();
    // Keep COPY independent of graphics, so test cleanup never needs a surface or graphics wait.
    texture_queue(&context);
    let queue = context.transfer_queue.unwrap();
    let mut source = context
        .allocate(AllocationRequest::new(4, 4, MemoryClass::Upload, true, None).unwrap())
        .unwrap();
    let mut destination = context
        .allocate(AllocationRequest::new(4, 4, MemoryClass::Readback, true, None).unwrap())
        .unwrap();
    let bytes = [13, 71, 137, 251];
    context.mapped_slice_mut(&mut source).unwrap()[..4].copy_from_slice(&bytes);
    context.flush(&mut source, 0, 4).unwrap();
    let mut gate = QueueGate::new(context.device.as_ref().unwrap(), queue);
    // Four distinct submissions fill the native command slots; the fifth owner callback then
    // waits for its first still-unsignaled slot while subsequent requests fill the real inbox.
    for _ in 0..4 {
        let token = context.copy_buffer(&source, &destination, 0, 0, 4).unwrap();
        context
            .transfer_worker
            .as_ref()
            .unwrap()
            .flush_through(token.value)
            .unwrap();
    }
    let mut rejected = false;
    // At most 64 jobs enter the blocked fifth batch and 64 remain in the channel.
    for _ in 0..129 {
        let before = context.next_transfer_value;
        match context.copy_buffer(&source, &destination, 0, 0, 4) {
            Ok(_) => {}
            Err(error) => {
                assert_eq!(error, AllocationError::OutOfMemory);
                assert_eq!(context.next_transfer_value, before);
                rejected = true;
                break;
            }
        }
    }
    assert!(rejected, "bounded buffer inbox must reject admission");
    gate.release();
    context.wait_idle().unwrap();
    context.invalidate(&mut destination, 0, 4).unwrap();
    assert_eq!(&context.mapped_slice(&destination).unwrap()[..4], &bytes);
    drop(gate);
    context.free(source).unwrap();
    context.free(destination).unwrap();
    context.wait_idle().unwrap();
}

#[test]
fn partial_copy_submission_failure_drains_before_future_coarse_frame() {
    let mut context = context();
    let queue = texture_queue(&context);
    let (shader, pipeline) = sampler_pipeline(&context, "partial-failure");
    let mut coarse_texture = empty_texture(&mut context, 0);
    let colors = [[41, 103, 167, 255]];
    let coarse_staging = upload(&mut context, &mut [&mut coarse_texture], 1, &colors);
    wait_texture(&context, coarse_texture.last_transfer_value());
    context
        .publish_texture_mips(&mut coarse_texture, 1)
        .unwrap();
    let mut output = context
        .allocate(AllocationRequest::new(4, 4, MemoryClass::Readback, true, None).unwrap())
        .unwrap();
    let mut failed_texture = empty_texture(&mut context, 1);
    let mut gate = QueueGate::new(context.device.as_ref().unwrap(), queue);
    let submitted = crate::transfer::submission_observation::Observer::new(
        context.device.as_ref().unwrap(),
        queue,
    );
    submitted.fail_next_submission();
    let failed_staging = upload(
        &mut context,
        &mut [&mut failed_texture],
        1,
        &[[229, 157, 79, 255]],
    );
    let required = failed_texture.last_transfer_value();
    let deadline = Instant::now() + Duration::from_secs(10);
    while submitted.value() < required {
        assert!(
            Instant::now() < deadline,
            "injected partial copy was not submitted"
        );
        std::thread::yield_now();
    }
    // The native copy exists, but its application completion cannot signal: acquire was never
    // submitted. Cleanup must drain the actual queues rather than wait forever for that token.
    let device = context.device.as_ref().unwrap().clone();
    let timeline = context.texture_timeline.unwrap();
    // SAFETY: the live context retains the completion timeline during this status query.
    assert!(unsafe { device.get_semaphore_counter_value(timeline) }.unwrap() < required);
    gate.release();
    assert_eq!(context.wait_idle(), Err(crate::HalError::NativeFailure));
    assert!(
        context.is_drained(),
        "typed callback failure must still establish native release safety"
    );
    // SAFETY: queue-idle drain must have retired the successful COPY submission's native signal.
    assert_eq!(
        unsafe { device.get_semaphore_counter_value(context.texture_ownership_timeline.unwrap()) }
            .unwrap(),
        2
    );
    // SAFETY: no acquire was submitted, so cleanup must not fabricate application completion.
    assert!(unsafe { device.get_semaphore_counter_value(timeline) }.unwrap() < required);
    let slot = dispatch(&mut context, &pipeline, &output, 1);
    frame_pixels(&mut context, slot, &mut output, &colors);
    drop(gate);
    context.destroy_texture(failed_texture).unwrap();
    context.destroy_texture(coarse_texture).unwrap();
    context.destroy_pipeline(pipeline);
    context.destroy_shader(shader);
    for allocation in [coarse_staging, failed_staging, output] {
        context.free(allocation).unwrap();
    }
    assert_eq!(context.wait_idle(), Err(crate::HalError::NativeFailure));
}
