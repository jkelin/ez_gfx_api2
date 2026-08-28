// Any compiler or runtime failure exits nonzero; no backend fallback is attempted.
fn main() {
    if let Err(error) = ez_gfx_examples::run(ez_gfx_examples::Example::ComputeStructured) {
        eprintln!("compute structured-buffer example failed: {error}");
        std::process::exit(1);
    }
}
