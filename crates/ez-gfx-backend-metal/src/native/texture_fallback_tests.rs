use super::*;

#[test]
fn fallback_binding_stays_magenta_until_real_publication() {
    let mut context = NativeContext::create_default().unwrap();
    let pipeline = sampling_pipeline(&context);
    let (mut fallback, fallback_completions) = context
        .create_texture(
            TextureFormat::Rgba8Unorm,
            &[ImageMip {
                width: 1,
                height: 1,
                bytes: &MAGENTA,
            }],
            0,
            SAMPLER,
        )
        .unwrap();
    wait_texture(&context, fallback_completions[0].value);
    context.publish_texture_mips(&mut fallback, 1).unwrap();
    context.publish_texture_fallback(&fallback, 0).unwrap();

    let (mut real, real_completions) = context
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
    assert_eq!(fallback.binding, real.binding);
    let sampled = fallback.fallback_sampled(0);
    let sample = enqueue_sampled(&mut context, sampled, &pipeline, None);
    assert_eq!(sampled_pixel(&mut context, sample), MAGENTA);

    wait_texture(&context, real_completions[0].value);
    context.publish_texture_mips(&mut real, 1).unwrap();
    let sample = enqueue_sample(&mut context, &real, &pipeline, None);
    assert_eq!(sampled_pixel(&mut context, sample), GREEN);

    context.destroy_texture(real).unwrap();
    let sample = enqueue_sampled(&mut context, fallback.fallback_sampled(0), &pipeline, None);
    assert_eq!(sampled_pixel(&mut context, sample), MAGENTA);
    context.destroy_texture(fallback).unwrap();
    context.destroy_pipeline(pipeline.pipeline);
    context.destroy_shader(pipeline.shader);
    context.wait_idle().unwrap();
}
