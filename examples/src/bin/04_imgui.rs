// Any API failure exits nonzero so automation cannot mistake partial setup for success.
fn main() {
    if let Err(error) = ez_gfx_examples::run(ez_gfx_examples::Example::ImGui) {
        eprintln!("ImGui stream example failed: {error}");
        std::process::exit(1);
    }
}
