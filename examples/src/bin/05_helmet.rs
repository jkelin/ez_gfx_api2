// Any API failure exits nonzero so automation cannot mistake partial setup for success.
fn main() {
    if let Err(error) = ez_gfx_examples::run(ez_gfx_examples::Example::Helmet) {
        eprintln!("helmet geometry example failed: {error}");
        std::process::exit(1);
    }
}
