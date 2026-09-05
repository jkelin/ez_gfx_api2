use super::*;
use ez_gfx_compiler::{Target, compile_shader};
use ez_gfx_core::{Backend, capability::SemanticProfile};
use ez_gfx_runtime::shader::RuntimeShader;
use std::{
    sync::mpsc,
    time::{Duration, Instant},
};

const RED: [u8; 4] = [255, 0, 0, 255];
const GREEN: [u8; 4] = [0, 255, 0, 255];
const SAMPLER: TextureSamplerDesc = TextureSamplerDesc {
    min_filter: SamplerFilter::Nearest,
    mag_filter: SamplerFilter::Nearest,
    max_anisotropy: 1.0,
    address_u: SamplerAddressMode::Clamp,
    address_v: SamplerAddressMode::Clamp,
    address_w: SamplerAddressMode::Clamp,
};

struct QueueGate(ID3D12Fence);
impl QueueGate {
    fn new(context: &NativeContext, queue: &ID3D12CommandQueue) -> Self {
        // Install the guard before enqueueing the wait, including the failure path.
        // SAFETY: the queue and fence belong to the live device.
        let gate = Self(unsafe { context.device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.unwrap());
        // SAFETY: the guard retains the fence and releases it on unwind.
        unsafe { queue.Wait(&gate.0, 1) }.unwrap();
        gate
    }
}
impl Drop for QueueGate {
    fn drop(&mut self) {
        // SAFETY: CPU signaling releases every queued wait before context destruction.
        let _ = unsafe { self.0.Signal(1) };
    }
}

type CopyObservers = Vec<(ID3D12CommandQueue, mpsc::Sender<()>, bool)>;
static COPY_OBSERVERS: parking_lot::Mutex<CopyObservers> = parking_lot::Mutex::new(Vec::new());

pub(super) fn copy_submitted(queue: &ID3D12CommandQueue) -> windows::core::Result<()> {
    // Queue identity isolates failures; only the selected native COPY submission fails.
    for (observed, sent, fail) in COPY_OBSERVERS.lock().iter() {
        if observed == queue {
            let _ = sent.send(());
            if *fail {
                return Err(windows::core::Error::from_hresult(
                    windows::Win32::Foundation::E_FAIL,
                ));
            }
        }
    }
    Ok(())
}

struct CopyObserver(ID3D12CommandQueue, mpsc::Receiver<()>);
impl CopyObserver {
    fn new(queue: &ID3D12CommandQueue) -> Self {
        let (sent, received) = mpsc::channel();
        COPY_OBSERVERS.lock().push((queue.clone(), sent, false));
        Self(queue.clone(), received)
    }

    fn failing(queue: &ID3D12CommandQueue) -> Self {
        // Install before admission, so no native submission can outrun fault registration.
        let observer = Self::new(queue);
        COPY_OBSERVERS
            .lock()
            .iter_mut()
            .find(|(observed, _, _)| observed == queue)
            .unwrap()
            .2 = true;
        observer
    }
}
impl Drop for CopyObserver {
    fn drop(&mut self) {
        // Unregister even when the bounded native-submission wait fails.
        COPY_OBSERVERS
            .lock()
            .retain(|(queue, _, _)| queue != &self.0);
    }
}

fn wait_fence(fence: &ID3D12Fence, value: u64) {
    // A timeout panics through QueueGate rather than leaving shutdown behind an unsignaled wait.
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        // SAFETY: the borrowed COM reference retains the fence.
        let completed = unsafe { fence.GetCompletedValue() };
        assert_ne!(completed, u64::MAX, "device removed");
        if completed >= value {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "GPU fence {value} did not complete"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn copy_queue(context: &NativeContext) -> ID3D12CommandQueue {
    // SAFETY: the initialized COPY queue descriptor is used only during creation.
    unsafe {
        context
            .device
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_COPY,
                ..Default::default()
            })
    }
    .unwrap()
}

fn capture_uploads(context: &mut NativeContext) -> mpsc::Receiver<Vec<transfer::Dx12TransferJob>> {
    // Only idle workers are replaced; captured jobs retain the real resources and staging bytes.
    context.wait_idle().unwrap();
    let (sent, received) = mpsc::channel();
    context.texture_worker = Some(
        TransferWorker::new_grouped_with_shutdown(
            64,
            ez_gfx_hal::DEFAULT_STAGING_POLICY,
            transfer::job_bytes,
            transfer::job_group,
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

fn install_worker(context: &mut NativeContext, queue: &ID3D12CommandQueue) {
    // Capture admission must drain before its worker is replaced.
    context.texture_worker.as_ref().unwrap().flush().unwrap();
    context.texture_worker = Some(
        transfer::start_worker(
            &context.device,
            queue.clone(),
            context.queue.clone(),
            context.texture_fence.clone(),
        )
        .unwrap(),
    );
}

struct ShaderSource(std::path::PathBuf);
impl Drop for ShaderSource {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn sampling_pipeline(context: &NativeContext) -> NativePipeline {
    // Parallel tests receive different paths; the guard also removes files after compilation errors.
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let source = ShaderSource(std::env::temp_dir().join(format!(
        "ez-gfx-dx12-texture-{}-{}.slang",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
    )));
    std::fs::write(
        &source.0,
        r#"
[__AttributeUsage(_AttributeTargets.Var)]
struct StructuredBufferAttribute { string name; };
[__AttributeUsage(_AttributeTargets.Var)]
struct BindlessTextureHeapAttribute { int capacity; };
struct TextureEntry { Texture2D<float4> texture; SamplerState sampler; };
struct TextureHeap { TextureEntry entries[1024]; };
[StructuredBuffer("values")] RWStructuredBuffer<uint> values;
[BindlessTextureHeap(1024)] ParameterBlock<TextureHeap> texture_heap;
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
    let artifact = compile_shader(&source.0, &[Target::Dxil], false).unwrap();
    let runtime = RuntimeShader::load(&artifact, Backend::Dx12, SemanticProfile::V1).unwrap();
    let (product, _, _) = runtime.compute_product().unwrap();
    let products = runtime
        .products()
        .map(|(_, bytes)| bytes)
        .collect::<Vec<_>>();
    let bindings = runtime.bindings(ez_gfx_artifact::Stage::Compute).unwrap();
    let layouts = bindings
        .requirements()
        .iter()
        .map(|binding| {
            ShaderBufferLayout::new(
                binding.space,
                binding.binding,
                binding.descriptor_count,
                binding.writable,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let shader = context.create_shader(&products).unwrap();
    let pipeline = context
        .create_compute_pipeline(&shader, product, &layouts)
        .unwrap();
    context.destroy_shader(shader);
    pipeline
}

fn command_list(context: &NativeContext) -> (ID3D12CommandAllocator, ID3D12GraphicsCommandList) {
    // SAFETY: the new list uses a fresh allocator from the same device.
    unsafe {
        let allocator = context
            .device
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .unwrap();
        let list = context
            .device
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
            .unwrap();
        (allocator, list)
    }
}

fn transition(
    list: &ID3D12GraphicsCommandList,
    resource: &ID3D12Resource,
    mip: u32,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) {
    // Transition only resident mips: the fine mip can remain COPY_DEST on another queue.
    let mut barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: core::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: core::mem::ManuallyDrop::new(Some(resource.clone())),
                Subresource: mip,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    };
    // SAFETY: the initialized union selects Transition; recording copies it synchronously.
    unsafe {
        list.ResourceBarrier(core::slice::from_ref(&barrier));
        core::mem::ManuallyDrop::drop(&mut (*barrier.Anonymous.Transition).pResource);
    }
}

struct SampleReadback {
    output: NativeAllocation,
    readback: NativeAllocation,
    commands: [(ID3D12CommandAllocator, ID3D12GraphicsCommandList); 2],
    completion: u64,
}

fn enqueue_sample(
    context: &mut NativeContext,
    texture: &NativeTexture,
    pipeline: &NativePipeline,
) -> SampleReadback {
    // Readback heaps cannot be UAVs on DX12: dispatch writes device memory, then DIRECT copies it.
    let output = context
        .allocate(AllocationRequest::new(16, 4, MemoryClass::Device, false, None).unwrap())
        .unwrap();
    let readback = context
        .allocate(AllocationRequest::new(16, 4, MemoryClass::Readback, true, None).unwrap())
        .unwrap();
    let before = command_list(context);
    for mip in texture.mip_count - texture.resident_mips..texture.mip_count {
        transition(
            &before.1,
            &texture.resource,
            mip,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        );
    }
    transition(
        &before.1,
        &output.resource,
        0,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    );
    // SAFETY: closed lists and their allocators remain retained by SampleReadback until completion.
    unsafe {
        before.1.Close().unwrap();
        context
            .queue
            .ExecuteCommandLists(&[Some(before.1.cast().unwrap())]);
    }
    let binding = NativeBufferBinding {
        allocation: &output,
        offset: 0,
        writable: true,
    };
    context
        .execute_frame(
            None,
            &[NativeFrameAction::Compute(NativeComputeDispatch {
                pipeline,
                groups: [1, 1, 1],
                push_constants: &[],
                bindings: std::slice::from_ref(&binding),
            })],
            false,
        )
        .unwrap();
    let after = command_list(context);
    for mip in texture.mip_count - texture.resident_mips..texture.mip_count {
        transition(
            &after.1,
            &texture.resource,
            mip,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
    }
    transition(
        &after.1,
        &output.resource,
        0,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    );
    let completion = context.next_fence;
    context.next_fence += 1;
    // SAFETY: both buffers contain 16 bytes and remain alive until the following fence retires.
    unsafe {
        after
            .1
            .CopyBufferRegion(&readback.resource, 0, &output.resource, 0, 16);
        after.1.Close().unwrap();
        context
            .queue
            .ExecuteCommandLists(&[Some(after.1.cast().unwrap())]);
        context.queue.Signal(&context.fence, completion).unwrap();
    }
    SampleReadback {
        output,
        readback,
        commands: [before, after],
        completion,
    }
}

fn sampled_pixel(context: &mut NativeContext, mut sample: SampleReadback) -> [u8; 4] {
    // Never wait_idle here: fine-copy completion must not be a prerequisite for this readback.
    wait_fence(&context.fence, sample.completion);
    context.invalidate(&mut sample.readback, 0, 16).unwrap();
    let mut pixel = [0; 4];
    for (channel, bytes) in pixel
        .iter_mut()
        .zip(context.mapped_slice(&sample.readback).unwrap()[..16].chunks_exact(4))
    {
        *channel = u8::try_from(u32::from_ne_bytes(bytes.try_into().unwrap())).unwrap();
    }
    context.free(sample.output).unwrap();
    context.free(sample.readback).unwrap();
    drop(sample.commands);
    pixel
}

#[test]
fn coarse_compute_frame_completes_while_fine_copy_is_gpu_blocked() {
    let mut context = NativeContext::create_default(false).unwrap();
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
    let coarse_jobs = received.recv_timeout(Duration::from_secs(5)).unwrap();
    let fine_jobs = received.recv_timeout(Duration::from_secs(5)).unwrap();
    let queue = copy_queue(&context);
    install_worker(&mut context, &queue);
    context
        .texture_worker
        .as_ref()
        .unwrap()
        .submit_batch(coarse_jobs)
        .unwrap();
    wait_fence(&context.texture_fence, completions[0].value);
    context.publish_texture_mips(&mut texture, 1).unwrap();

    // Declare after context and uploads: panic releases COPY before context joins its worker.
    let gate = QueueGate::new(&context, &queue);
    let submitted = CopyObserver::new(&queue);
    context
        .texture_worker
        .as_ref()
        .unwrap()
        .submit_batch(fine_jobs)
        .unwrap();
    submitted.1.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(context.completed_texture_transfer_value().unwrap() < completions[1].value);
    assert!(context.publish_texture_mips(&mut texture, 2).is_err());
    let sample = enqueue_sample(&mut context, &texture, &pipeline);
    assert_eq!(sampled_pixel(&mut context, sample), RED);
    assert!(context.completed_texture_transfer_value().unwrap() < completions[1].value);
    // The staging backing fine work must not become reusable while its fence is unsignaled.
    let completed = context.completed_texture_transfer_value().unwrap();
    assert!(context.texture_staging.take(1, completed).is_none());

    drop(gate);
    wait_fence(&context.texture_fence, completions[1].value);
    context.publish_texture_mips(&mut texture, 2).unwrap();
    let sample = enqueue_sample(&mut context, &texture, &pipeline);
    assert_eq!(sampled_pixel(&mut context, sample), GREEN);
    context.destroy_texture(texture).unwrap();
    context.destroy_pipeline(pipeline);
    context.wait_idle().unwrap();
}

#[test]
fn distinct_textures_share_one_native_copy_batch_and_preserve_pixels() {
    let mut context = NativeContext::create_default(false).unwrap();
    let received = capture_uploads(&mut context);
    let (mut red, _) = context
        .create_texture(
            TextureFormat::Rgba8Unorm,
            &[ImageMip {
                width: 1,
                height: 1,
                bytes: &RED,
            }],
            0,
            SAMPLER,
        )
        .unwrap();
    let (mut green, _) = context
        .create_texture(
            TextureFormat::Rgba8Unorm,
            &[ImageMip {
                width: 1,
                height: 1,
                bytes: &GREEN,
            }],
            1,
            SAMPLER,
        )
        .unwrap();
    context.texture_worker.as_ref().unwrap().flush().unwrap();
    let mut jobs = Vec::new();
    while jobs.len() < 2 {
        jobs.extend(received.recv_timeout(Duration::from_secs(5)).unwrap());
    }
    let queue = copy_queue(&context);
    install_worker(&mut context, &queue);
    assert_eq!(
        transfer::submit_test_batch(
            &context.device,
            &queue,
            &context.queue,
            &context.texture_fence,
            &jobs
        ),
        1,
        "one native COPY signal for both textures"
    );
    context.publish_texture_mips(&mut red, 1).unwrap();
    context.publish_texture_mips(&mut green, 1).unwrap();
    assert_eq!(context.readback_texture_rgba8(&red, 1, 1).unwrap(), RED);
    assert_eq!(context.readback_texture_rgba8(&green, 1, 1).unwrap(), GREEN);
    context.destroy_texture(red).unwrap();
    context.destroy_texture(green).unwrap();
    context.wait_idle().unwrap();
}

#[test]
fn unsignaled_graphics_fence_retains_texture_until_descriptor_reuse_is_safe() {
    let mut context = NativeContext::create_default(false).unwrap();
    let pipeline = sampling_pipeline(&context);
    let (mut texture, completion) = context
        .create_texture(
            TextureFormat::Rgba8Unorm,
            &[ImageMip {
                width: 1,
                height: 1,
                bytes: &RED,
            }],
            0,
            SAMPLER,
        )
        .unwrap();
    wait_fence(&context.texture_fence, completion[0].value);
    context.publish_texture_mips(&mut texture, 1).unwrap();
    let gate = QueueGate::new(&context, &context.queue);
    let sample = enqueue_sample(&mut context, &texture, &pipeline);
    assert!(!context.texture_retirement_ready(completion[0]).unwrap());
    assert!(!context.texture_descriptor_update_ready().unwrap());
    context.destroy_texture(texture).unwrap();
    context.reclaim_deferred().unwrap();
    assert!(
        context
            .deferred
            .iter()
            .any(|entry| matches!(&entry.resource,
        DeferredResource::Texture(texture) if texture.binding == 0))
    );
    drop(gate);
    assert_eq!(sampled_pixel(&mut context, sample), RED);
    context.wait_idle().unwrap();
    assert!(context.texture_retirement_ready(completion[0]).unwrap());
    assert!(context.texture_descriptor_update_ready().unwrap());
    assert!(
        !context
            .deferred
            .iter()
            .any(|entry| matches!(entry.resource, DeferredResource::Texture(_)))
    );

    let (mut replacement, completion) = context
        .create_texture(
            TextureFormat::Rgba8Unorm,
            &[ImageMip {
                width: 1,
                height: 1,
                bytes: &GREEN,
            }],
            0,
            SAMPLER,
        )
        .unwrap();
    wait_fence(&context.texture_fence, completion[0].value);
    context.publish_texture_mips(&mut replacement, 1).unwrap();
    let sample = enqueue_sample(&mut context, &replacement, &pipeline);
    assert_eq!(sampled_pixel(&mut context, sample), GREEN);
    context.destroy_texture(replacement).unwrap();
    context.destroy_pipeline(pipeline);
    context.wait_idle().unwrap();
}

struct AdmissionGate(Option<mpsc::Sender<()>>);
impl Drop for AdmissionGate {
    fn drop(&mut self) {
        // Release a blocked callback before the context drops or replaces its worker.
        if let Some(release) = self.0.take() {
            let _ = release.send(());
        }
    }
}

fn full_worker(
    source: &NativeAllocation,
    destination: &NativeAllocation,
) -> (TransferWorker<transfer::Dx12TransferJob>, AdmissionGate) {
    let (release, blocked) = mpsc::channel();
    let (entered, reached) = mpsc::channel();
    let mut first = true;
    let worker = TransferWorker::new(
        1,
        ez_gfx_hal::DEFAULT_STAGING_POLICY,
        transfer::job_bytes,
        move |_| {
            // These synthetic occupancy jobs never touch a native queue or context timeline.
            if first {
                first = false;
                entered.send(()).unwrap();
                blocked.recv().unwrap();
            }
            Ok(())
        },
    )
    .unwrap();
    let gate = AdmissionGate(Some(release));
    let job = || transfer::Dx12TransferJob {
        value: 0,
        bytes: 4,
        cancelled: None,
        copy: transfer::Dx12TransferCopy::Buffer {
            source: source.resource.clone(),
            destination: destination.resource.clone(),
            source_offset: 0,
            destination_offset: 0,
            size: 4,
        },
    };
    worker.submit(job()).unwrap();
    reached.recv_timeout(Duration::from_secs(5)).unwrap();
    worker.submit(job()).unwrap();
    (worker, gate)
}

fn bounded_idle(
    mut context: NativeContext,
    gate: Option<QueueGate>,
) -> (NativeContext, Result<(), HalError>) {
    let (sent, received) = mpsc::channel();
    let (started, entered) = mpsc::channel();
    let waiting = std::thread::spawn(move || {
        started.send(()).unwrap();
        let result = context.wait_idle();
        // On timeout the detached thread retains its context; never forge completion to join it.
        let _ = sent.send((context, result));
    });
    let entered = entered.recv_timeout(Duration::from_secs(5)).is_ok();
    let early = gate
        .as_ref()
        .and_then(|_| received.recv_timeout(Duration::from_millis(50)).ok());
    // Release the actual GPU gate before any assertion or waiting-thread cleanup.
    drop(gate);
    assert!(entered, "idle thread did not start");
    assert!(
        early.is_none(),
        "idle returned before gated native work drained"
    );
    let result = received
        .recv_timeout(Duration::from_secs(15))
        .expect("idle waited for a completion that rejected/failed work cannot signal");
    waiting.join().unwrap();
    result
}

#[test]
fn full_texture_and_buffer_admission_does_not_strand_idle() {
    let mut context = NativeContext::create_default(false).unwrap();
    let pipeline = sampling_pipeline(&context);
    let (mut texture, completion) = context
        .create_texture(
            TextureFormat::Rgba8Unorm,
            &[ImageMip {
                width: 1,
                height: 1,
                bytes: &RED,
            }],
            0,
            SAMPLER,
        )
        .unwrap();
    wait_fence(&context.texture_fence, completion[0].value);
    context.publish_texture_mips(&mut texture, 1).unwrap();
    context.wait_idle().unwrap();
    let source = context
        .allocate(AllocationRequest::new(4, 4, MemoryClass::Upload, true, None).unwrap())
        .unwrap();
    let destination = context
        .allocate(AllocationRequest::new(4, 4, MemoryClass::Readback, true, None).unwrap())
        .unwrap();
    let (texture_worker, texture_gate) = full_worker(&source, &destination);
    let (buffer_worker, buffer_gate) = full_worker(&source, &destination);
    let original_texture = context.texture_worker.replace(texture_worker);
    let original_buffer = context.transfer_worker.replace(buffer_worker);

    assert!(matches!(
        context.create_texture(
            TextureFormat::Rgba8Unorm,
            &[ImageMip {
                width: 1,
                height: 1,
                bytes: &GREEN
            }],
            1,
            SAMPLER
        ),
        Err(AllocationError::OutOfMemory)
    ));
    assert!(matches!(
        context.update_texture_region(
            &mut texture,
            &TextureRegion {
                mip_level: 0,
                x: 0,
                y: 0,
                width: 1,
                height: 1,
                bytes: &GREEN,
            }
        ),
        Err(AllocationError::OutOfMemory)
    ));
    assert!(matches!(
        context.copy_buffer(&source, &destination, 0, 0, 4),
        Err(AllocationError::OutOfMemory)
    ));

    drop(texture_gate);
    drop(buffer_gate);
    context.texture_worker = original_texture;
    context.transfer_worker = original_buffer;
    let (mut context, idle) = bounded_idle(context, None);
    idle.unwrap();
    let sample = enqueue_sample(&mut context, &texture, &pipeline);
    assert_eq!(sampled_pixel(&mut context, sample), RED);
    context.free(source).unwrap();
    context.free(destination).unwrap();
    context.destroy_texture(texture).unwrap();
    context.destroy_pipeline(pipeline);
    context.wait_idle().unwrap();
}

#[test]
fn partial_native_copy_failure_drains_before_idle_returns_and_preserves_future_frames() {
    let mut context = NativeContext::create_default(false).unwrap();
    let pipeline = sampling_pipeline(&context);
    let (mut texture, completion) = context
        .create_texture(
            TextureFormat::Rgba8Unorm,
            &[ImageMip {
                width: 1,
                height: 1,
                bytes: &RED,
            }],
            0,
            SAMPLER,
        )
        .unwrap();
    wait_fence(&context.texture_fence, completion[0].value);
    context.publish_texture_mips(&mut texture, 1).unwrap();
    context.wait_idle().unwrap();
    let mut source = context
        .allocate(AllocationRequest::new(4, 4, MemoryClass::Upload, true, None).unwrap())
        .unwrap();
    let mut destination = context
        .allocate(AllocationRequest::new(4, 4, MemoryClass::Readback, true, None).unwrap())
        .unwrap();
    context.mapped_slice_mut(&mut source).unwrap()[..4].copy_from_slice(&GREEN);
    context.flush(&mut source, 0, 4).unwrap();
    let queue = copy_queue(&context);
    context.transfer_worker = Some(
        transfer::start_worker(
            &context.device,
            queue.clone(),
            context.queue.clone(),
            context.transfer_fence.clone(),
        )
        .unwrap(),
    );
    let gate = QueueGate::new(&context, &queue);
    let failure = CopyObserver::failing(&queue);
    let token = context.copy_buffer(&source, &destination, 0, 0, 4).unwrap();
    failure.1.recv_timeout(Duration::from_secs(5)).unwrap();
    let (mut context, idle) = bounded_idle(context, Some(gate));
    assert!(matches!(idle, Err(HalError::NativeFailure)));
    assert!(
        context.is_drained(),
        "failed submission must finish native cleanup before readback"
    );
    // Drain safety is distinct from success: never manufacture application completion.
    // SAFETY: context owns this fence throughout the counter read.
    assert!(unsafe { context.transfer_fence.GetCompletedValue() } < token.value);
    context.invalidate(&mut destination, 0, 4).unwrap();
    assert_eq!(&context.mapped_slice(&destination).unwrap()[..4], &GREEN);
    let sample = enqueue_sample(&mut context, &texture, &pipeline);
    assert_eq!(sampled_pixel(&mut context, sample), RED);
    context.free(source).unwrap();
    context.free(destination).unwrap();
    context.destroy_texture(texture).unwrap();
    context.destroy_pipeline(pipeline);
    assert!(matches!(context.wait_idle(), Err(HalError::NativeFailure)));
}
