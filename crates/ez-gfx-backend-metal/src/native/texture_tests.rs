use super::*;
use ez_gfx_compiler::{Target, compile_shader};
use ez_gfx_runtime::shader::RuntimeShader;
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

const RED: [u8; 4] = [255, 0, 0, 255];
const GREEN: [u8; 4] = [0, 255, 0, 255];
const BLUE: [u8; 4] = [0, 0, 255, 255];
const LIMIT: Duration = Duration::from_secs(5);
const SAMPLER: TextureSamplerDesc = TextureSamplerDesc {
    min_filter: SamplerFilter::Nearest,
    mag_filter: SamplerFilter::Nearest,
    max_anisotropy: 1.0,
    address_u: SamplerAddressMode::Clamp,
    address_v: SamplerAddressMode::Clamp,
    address_w: SamplerAddressMode::Clamp,
};

struct QueueGate {
    release: Option<mpsc::Sender<()>>,
    watchdog: Option<thread::JoinHandle<()>>,
    expired: Arc<AtomicBool>,
}

impl QueueGate {
    fn new(context: &NativeContext, queue: &ProtocolObject<dyn MTLCommandQueue>) -> Self {
        // A separate queue releases the event even when the tested owner blocks on the CPU.
        let event = context.device.newEvent().unwrap();
        let release_queue = context.device.newCommandQueue().unwrap();
        let blocked = queue.commandBuffer().unwrap();
        blocked.encodeWaitForEvent_value(&event, 1);
        let (sent, received) = mpsc::channel();
        let expired = Arc::new(AtomicBool::new(false));
        let timeout = expired.clone();
        let watchdog = thread::spawn(move || {
            if matches!(
                received.recv_timeout(LIMIT * 2),
                Err(mpsc::RecvTimeoutError::Timeout)
            ) {
                timeout.store(true, Ordering::Release);
            }
            let signal = release_queue.commandBuffer().unwrap();
            signal.encodeSignalEvent_value(&event, 1);
            signal.commit();
        });
        // The watchdog exists before committing any wait, including panic paths.
        blocked.commit();
        Self {
            release: Some(sent),
            watchdog: Some(watchdog),
            expired,
        }
    }

    fn assert_closed(&self) {
        assert!(
            !self.expired.load(Ordering::Acquire),
            "GPU gate watchdog expired"
        );
    }
}

impl Drop for QueueGate {
    fn drop(&mut self) {
        // Never assert while unwinding: disconnection also tells the watchdog to release.
        if let Some(release) = self.release.take() {
            let _ = release.send(());
        }
        if let Some(watchdog) = self.watchdog.take() {
            let _ = watchdog.join();
        }
    }
}

fn until(mut ready: impl FnMut() -> bool) {
    let deadline = Instant::now() + LIMIT;
    while !ready() {
        assert!(
            Instant::now() < deadline,
            "bounded Metal completion expired"
        );
        thread::sleep(Duration::from_millis(1));
    }
}

fn wait_command(command: &ProtocolObject<dyn MTLCommandBuffer>) {
    until(|| {
        matches!(
            command.status(),
            MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
        )
    });
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    assert!(command.error().is_none());
}

fn wait_texture(context: &NativeContext, value: u64) {
    until(|| context.completed_texture_transfer_value().unwrap() >= value);
}

struct SamplingPipeline {
    shader: NativeShader,
    pipeline: NativePipeline,
    heap: ShaderTextureHeapLayout,
    binding: usize,
    threads: [u32; 3],
}

fn sampling_pipeline(context: &NativeContext) -> SamplingPipeline {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("native_texture.slang");
    std::fs::write(
        &source,
        r#"[__AttributeUsage(_AttributeTargets.Var)]
struct StructuredBufferAttribute { string name; };
[__AttributeUsage(_AttributeTargets.Var)]
struct BindlessTextureHeapAttribute { int capacity; };
struct TextureEntry { Texture2D<float4> texture; SamplerState sampler; };
struct TextureHeap { TextureEntry entries[1024]; };
[StructuredBuffer("values")]
RWStructuredBuffer<uint> values;
[BindlessTextureHeap(1024)]
ParameterBlock<TextureHeap> texture_heap;
[shader("compute")]
[numthreads(1, 1, 1)]
void computemain(uint3 id : SV_DispatchThreadID) {
    float4 pixel = texture_heap.entries[0].texture.SampleLevel(
        texture_heap.entries[0].sampler, float2(0.5, 0.5), 0);
    values[0] = uint(round(pixel.r * 255));
    values[1] = uint(round(pixel.g * 255));
    values[2] = uint(round(pixel.b * 255));
    values[3] = uint(round(pixel.a * 255));
}
"#,
    )
    .unwrap();
    let artifact = compile_shader(&source, &[Target::Metal], false).unwrap();
    let runtime =
        RuntimeShader::load(&artifact, ez_gfx_core::Backend::Metal, SemanticProfile::V1).unwrap();
    let (product, _, entry) = runtime.compute_product().unwrap();
    let layout = runtime
        .pipeline_layout(ez_gfx_artifact::Stage::Compute)
        .unwrap();
    let reflected = layout.texture_heap().unwrap();
    let heap = ShaderTextureHeapLayout::new(
        reflected.space,
        reflected.binding,
        reflected.capacity,
        reflected.argument_stride,
        reflected.texture_argument_offset,
        reflected.sampler_argument_offset,
    )
    .unwrap();
    let products = runtime
        .products()
        .map(|(_, bytes)| bytes)
        .collect::<Vec<_>>();
    let shader = context.create_shader(&products).unwrap();
    let pipeline = context
        .create_compute_pipeline(&shader, product, entry, Some(heap))
        .unwrap();
    SamplingPipeline {
        shader,
        pipeline,
        heap,
        binding: runtime
            .bindings(ez_gfx_artifact::Stage::Compute)
            .unwrap()
            .requirements()[0]
            .binding as usize,
        threads: runtime.compute_workgroup_size().unwrap(),
    }
}

struct Sample {
    output: NativeAllocation,
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

fn enqueue_sample(
    context: &mut NativeContext,
    texture: &NativeTexture,
    pipeline: &SamplingPipeline,
    wait: Option<CompletionToken>,
) -> Sample {
    let mut output = context
        .allocate(AllocationRequest::new(16, 4, MemoryClass::Readback, true, None).unwrap())
        .unwrap();
    context.mapped_slice_mut(&mut output).unwrap().fill(0);
    context.flush(&mut output, 0, 16).unwrap();
    let binding = NativeBufferBinding {
        allocation: &output,
        offset: 0,
        index: pipeline.binding,
    };
    let textures = [texture];
    let dispatch = NativeFrameAction::Compute(NativeComputeDispatch {
        pipeline: &pipeline.pipeline,
        groups: [1, 1, 1],
        threads_per_group: pipeline.threads,
        push_constants: &[],
        bindings: std::slice::from_ref(&binding),
        texture_heap: Some(pipeline.heap),
        textures: &textures,
    });
    // Explicit waits model the frame-plan adapter's visible-mip dependencies.
    if let Some(wait) = wait {
        context
            .execute_frame(None, &[NativeFrameAction::Wait(wait), dispatch], false)
            .unwrap();
    } else {
        context.execute_frame(None, &[dispatch], false).unwrap();
    }
    let command = context.queue.commandBuffer().unwrap();
    command.commit();
    Sample { output, command }
}

fn sampled_pixel(context: &mut NativeContext, mut sample: Sample) -> [u8; 4] {
    // A graphics FIFO marker avoids wait_idle, which would also drain gated hidden uploads.
    wait_command(&sample.command);
    context.invalidate(&mut sample.output, 0, 16).unwrap();
    let bytes = context.mapped_slice(&sample.output).unwrap();
    let pixel = core::array::from_fn(|channel| {
        u8::try_from(u32::from_ne_bytes(
            bytes[channel * 4..channel * 4 + 4].try_into().unwrap(),
        ))
        .unwrap()
    });
    context.free(sample.output).unwrap();
    pixel
}

fn capture_uploads(
    context: &mut NativeContext,
) -> mpsc::Receiver<Vec<transfer::TextureTransferJob>> {
    context.wait_idle().unwrap();
    let (sent, received) = mpsc::channel();
    context.texture_worker = Some(
        ez_gfx_hal::TransferWorker::new_ordered_with_shutdown(
            64,
            ez_gfx_hal::DEFAULT_STAGING_POLICY,
            |job: &transfer::TextureTransferJob| job.bytes,
            |job| job.stage,
            |job| job.value,
            move |jobs| {
                sent.send(jobs).unwrap();
                Ok(())
            },
            || Ok(()),
        )
        .unwrap(),
    );
    received
}

fn install_worker(context: &mut NativeContext) {
    // Only captured or fully shut-down workers are replaced; no GPU ownership is discarded.
    context.texture_worker = Some(
        transfer::start_texture_worker(
            context.transfer_queue.clone(),
            context.texture_graphics_event.clone(),
            context.texture_completion_event.clone(),
        )
        .unwrap(),
    );
}

fn ready_texture(
    context: &mut NativeContext,
    mips: &[ImageMip<'_>],
) -> (NativeTexture, CompletionToken) {
    let (mut texture, completions) = context
        .create_texture(TextureFormat::Rgba8Unorm, mips, 0, SAMPLER)
        .unwrap();
    let completion = *completions.last().unwrap();
    wait_texture(context, completion.value);
    context.publish_texture_mips(&mut texture, 1).unwrap();
    (texture, completion)
}

fn finish(context: &mut NativeContext, texture: NativeTexture, pipeline: SamplingPipeline) {
    context.destroy_texture(texture).unwrap();
    context.destroy_pipeline(pipeline.pipeline);
    context.destroy_shader(pipeline.shader);
    context.wait_idle().unwrap();
}

#[test]
fn coarse_compute_frame_completes_while_fine_upload_is_gpu_gated() {
    let mut context = NativeContext::create_default().unwrap();
    let pipeline = sampling_pipeline(&context);
    let received = capture_uploads(&mut context);
    let fine = GREEN.repeat(4);
    let (mut texture, completions) = context
        .create_texture(
            TextureFormat::Rgba8Unorm,
            &[
                ImageMip {
                    width: 2,
                    height: 2,
                    bytes: &fine,
                },
                ImageMip {
                    width: 1,
                    height: 1,
                    bytes: &RED,
                },
            ],
            0,
            SAMPLER,
        )
        .unwrap();
    context.texture_worker.as_ref().unwrap().flush().unwrap();
    let coarse = received.recv_timeout(LIMIT).unwrap();
    let fine = received.recv_timeout(LIMIT).unwrap();
    install_worker(&mut context);
    context
        .texture_worker
        .as_ref()
        .unwrap()
        .submit_batch(coarse)
        .unwrap();
    wait_texture(&context, completions[0].value);
    context.publish_texture_mips(&mut texture, 1).unwrap();

    let gate = QueueGate::new(&context, &context.transfer_queue);
    let submission = fine[0].submission.clone();
    context
        .texture_worker
        .as_ref()
        .unwrap()
        .submit_batch(fine)
        .unwrap();
    until(|| submission.submitted());
    assert!(!submission.completed().unwrap());
    assert!(context.publish_texture_mips(&mut texture, 2).is_err());
    let sample = enqueue_sample(&mut context, &texture, &pipeline, Some(completions[0]));
    assert_eq!(sampled_pixel(&mut context, sample), RED);
    assert!(context.completed_texture_transfer_value().unwrap() < completions[1].value);
    let completed = context.completed_texture_transfer_value().unwrap();
    assert!(context.texture_staging.take(1, completed).is_none());
    gate.assert_closed();
    drop(gate);

    wait_texture(&context, completions[1].value);
    context.publish_texture_mips(&mut texture, 2).unwrap();
    let sample = enqueue_sample(&mut context, &texture, &pipeline, Some(completions[1]));
    assert_eq!(sampled_pixel(&mut context, sample), GREEN);
    finish(&mut context, texture, pipeline);
}

#[test]
fn hidden_mip_update_preserves_coarse_pixels_until_publication() {
    let mut context = NativeContext::create_default().unwrap();
    let pipeline = sampling_pipeline(&context);
    let fine = GREEN.repeat(4);
    let (mut texture, _) = ready_texture(
        &mut context,
        &[
            ImageMip {
                width: 2,
                height: 2,
                bytes: &fine,
            },
            ImageMip {
                width: 1,
                height: 1,
                bytes: &RED,
            },
        ],
    );
    // Gate the concrete queue owned by this real texture worker, not the buffer worker.
    install_worker(&mut context);
    let gate = QueueGate::new(&context, &context.transfer_queue);
    let replacement = BLUE.repeat(4);
    let update = context
        .update_texture_region(
            &mut texture,
            &TextureRegion {
                mip_level: 0,
                x: 0,
                y: 0,
                width: 2,
                height: 2,
                bytes: &replacement,
            },
        )
        .unwrap();
    let submission = context
        .pending_texture_transfers
        .last()
        .unwrap()
        .submission
        .clone();
    until(|| submission.submitted());
    let sample = enqueue_sample(&mut context, &texture, &pipeline, None);
    assert_eq!(sampled_pixel(&mut context, sample), RED);
    assert!(!submission.completed().unwrap());
    assert!(context.publish_texture_mips(&mut texture, 2).is_err());
    gate.assert_closed();
    drop(gate);

    wait_texture(&context, update.value);
    let sample = enqueue_sample(&mut context, &texture, &pipeline, None);
    assert_eq!(sampled_pixel(&mut context, sample), RED);
    context.publish_texture_mips(&mut texture, 2).unwrap();
    let sample = enqueue_sample(&mut context, &texture, &pipeline, Some(update));
    assert_eq!(sampled_pixel(&mut context, sample), BLUE);
    finish(&mut context, texture, pipeline);
}

#[test]
fn visible_mip_update_waits_for_submitted_reader_and_changes_future_pixels() {
    let mut context = NativeContext::create_default().unwrap();
    let pipeline = sampling_pipeline(&context);
    let (mut texture, _) = ready_texture(
        &mut context,
        &[ImageMip {
            width: 1,
            height: 1,
            bytes: &RED,
        }],
    );
    let gate = QueueGate::new(&context, &context.queue);
    let old = enqueue_sample(&mut context, &texture, &pipeline, None);
    let update = context
        .update_texture_region(
            &mut texture,
            &TextureRegion {
                mip_level: 0,
                x: 0,
                y: 0,
                width: 1,
                height: 1,
                bytes: &GREEN,
            },
        )
        .unwrap();
    let submission = context
        .pending_texture_transfers
        .last()
        .unwrap()
        .submission
        .clone();
    until(|| submission.submitted());
    assert!(!submission.completed().unwrap());
    assert_ne!(old.command.status(), MTLCommandBufferStatus::Completed);
    gate.assert_closed();
    drop(gate);

    assert_eq!(sampled_pixel(&mut context, old), RED);
    let new = enqueue_sample(&mut context, &texture, &pipeline, Some(update));
    assert_eq!(sampled_pixel(&mut context, new), GREEN);
    wait_texture(&context, update.value);
    finish(&mut context, texture, pipeline);
}

#[test]
fn submitted_texture_retirement_preserves_pixels_until_graphics_gate_releases() {
    let mut context = NativeContext::create_default().unwrap();
    let pipeline = sampling_pipeline(&context);
    let (texture, completion) = ready_texture(
        &mut context,
        &[ImageMip {
            width: 1,
            height: 1,
            bytes: &RED,
        }],
    );
    let gate = QueueGate::new(&context, &context.queue);
    let sample = enqueue_sample(&mut context, &texture, &pipeline, None);
    assert!(!context.texture_retirement_ready(completion).unwrap());
    assert!(!context.texture_descriptor_update_ready());
    context.destroy_texture(texture).unwrap();
    assert!(context.deferred.iter().any(|entry| matches!(&entry.resource, DeferredResource::Texture(texture) if texture.binding == 0)));
    assert_ne!(sample.command.status(), MTLCommandBufferStatus::Completed);
    gate.assert_closed();
    drop(gate);

    assert_eq!(sampled_pixel(&mut context, sample), RED);
    context.wait_idle().unwrap();
    assert!(context.texture_retirement_ready(completion).unwrap());
    assert!(context.texture_descriptor_update_ready());
    assert!(
        !context
            .deferred
            .iter()
            .any(|entry| matches!(&entry.resource, DeferredResource::Texture(_)))
    );
    let (replacement, _) = ready_texture(
        &mut context,
        &[ImageMip {
            width: 1,
            height: 1,
            bytes: &GREEN,
        }],
    );
    let sample = enqueue_sample(&mut context, &replacement, &pipeline, None);
    assert_eq!(sampled_pixel(&mut context, sample), GREEN);
    finish(&mut context, replacement, pipeline);
}

#[test]
fn rejected_update_does_not_publish_a_wait_or_poison_future_frames() {
    let mut context = NativeContext::create_default().unwrap();
    let pipeline = sampling_pipeline(&context);
    let (mut texture, _) = ready_texture(
        &mut context,
        &[ImageMip {
            width: 1,
            height: 1,
            bytes: &RED,
        }],
    );
    context.wait_idle().unwrap();
    context.texture_worker.as_mut().unwrap().shutdown();
    let before = texture.mip_transfer_values().to_vec();
    assert!(
        context
            .update_texture_region(
                &mut texture,
                &TextureRegion {
                    mip_level: 0,
                    x: 0,
                    y: 0,
                    width: 1,
                    height: 1,
                    bytes: &GREEN,
                }
            )
            .is_err()
    );
    assert_eq!(texture.mip_transfer_values(), before);
    // Restore admission only after the rejected owner has completed shutdown.
    install_worker(&mut context);
    let sample = enqueue_sample(&mut context, &texture, &pipeline, None);
    assert_eq!(sampled_pixel(&mut context, sample), RED);
    let update = context
        .update_texture_region(
            &mut texture,
            &TextureRegion {
                mip_level: 0,
                x: 0,
                y: 0,
                width: 1,
                height: 1,
                bytes: &GREEN,
            },
        )
        .unwrap();
    let sample = enqueue_sample(&mut context, &texture, &pipeline, Some(update));
    assert_eq!(sampled_pixel(&mut context, sample), GREEN);
    finish(&mut context, texture, pipeline);
}

#[test]
fn partial_buffer_submission_failure_drains_real_copy_and_preserves_sampled_frames() {
    let mut context = NativeContext::create_default().unwrap();
    let pipeline = sampling_pipeline(&context);
    let (texture, _) = ready_texture(
        &mut context,
        &[ImageMip {
            width: 1,
            height: 1,
            bytes: &RED,
        }],
    );
    context.wait_idle().unwrap();
    let mut source = context
        .allocate(AllocationRequest::new(4, 4, MemoryClass::Upload, true, None).unwrap())
        .unwrap();
    let mut destination = context
        .allocate(AllocationRequest::new(4, 4, MemoryClass::Readback, true, None).unwrap())
        .unwrap();
    context.mapped_slice_mut(&mut source).unwrap()[..4].copy_from_slice(&GREEN);
    context.flush(&mut source, 0, 4).unwrap();
    let gate = QueueGate::new(&context, &context.transfer_queue);
    let token = context.copy_buffer(&source, &destination, 0, 0, 4).unwrap();
    // An invalid unordered token fails natively after the genuine ordered copy has committed.
    context
        .transfer_worker
        .as_ref()
        .unwrap()
        .submit(transfer::MetalTransferJob {
            value: 0,
            bytes: 1,
            command: transfer::TransferCommand::new(
                context.transfer_queue.commandBuffer().unwrap(),
            ),
        })
        .unwrap();
    until(|| context.transfer_worker.as_ref().unwrap().failed());
    assert!(matches!(
        context.execute_frame(None, &[NativeFrameAction::Wait(token)], false),
        Err(HalError::NativeFailure)
    ));
    assert_ne!(
        context.pending_transfers.last().unwrap().command.status(),
        MTLCommandBufferStatus::Completed
    );
    gate.assert_closed();
    drop(gate);
    assert!(matches!(context.wait_idle(), Err(HalError::NativeFailure)));
    assert!(context.is_drained());
    context.invalidate(&mut destination, 0, 4).unwrap();
    assert_eq!(&context.mapped_slice(&destination).unwrap()[..4], &GREEN);
    let sample = enqueue_sample(&mut context, &texture, &pipeline, None);
    assert_eq!(sampled_pixel(&mut context, sample), RED);
    context.free(source).unwrap();
    context.free(destination).unwrap();
    context.destroy_texture(texture).unwrap();
    context.destroy_pipeline(pipeline.pipeline);
    context.destroy_shader(pipeline.shader);
    assert!(matches!(context.wait_idle(), Err(HalError::NativeFailure)));
    assert!(context.is_drained());
}

#[test]
fn accepted_buffer_wait_orders_a_gpu_gated_copy_before_sampled_frame_completion() {
    let mut context = NativeContext::create_default().unwrap();
    let pipeline = sampling_pipeline(&context);
    let (texture, _) = ready_texture(
        &mut context,
        &[ImageMip {
            width: 1,
            height: 1,
            bytes: &RED,
        }],
    );
    context.wait_idle().unwrap();
    let mut source = context
        .allocate(AllocationRequest::new(4, 4, MemoryClass::Upload, true, None).unwrap())
        .unwrap();
    let mut destination = context
        .allocate(AllocationRequest::new(4, 4, MemoryClass::Readback, true, None).unwrap())
        .unwrap();
    context.mapped_slice_mut(&mut source).unwrap()[..4].copy_from_slice(&GREEN);
    context.mapped_slice_mut(&mut destination).unwrap()[..4].fill(0);
    context.flush(&mut source, 0, 4).unwrap();
    context.flush(&mut destination, 0, 4).unwrap();
    let gate = QueueGate::new(&context, &context.transfer_queue);
    let token = context.copy_buffer(&source, &destination, 0, 0, 4).unwrap();
    assert!(context.completed_transfer_value().unwrap() < token.value);
    // Release only after the frame accepts this incomplete transfer dependency.
    context.buffer_wait_observer = gate.release.clone();
    let sample = enqueue_sample(&mut context, &texture, &pipeline, Some(token));
    assert_eq!(sampled_pixel(&mut context, sample), RED);
    gate.assert_closed();
    assert!(context.completed_transfer_value().unwrap() >= token.value);
    context.invalidate(&mut destination, 0, 4).unwrap();
    assert_eq!(&context.mapped_slice(&destination).unwrap()[..4], &GREEN);
    drop(gate);
    context.free(source).unwrap();
    context.free(destination).unwrap();
    finish(&mut context, texture, pipeline);
}
