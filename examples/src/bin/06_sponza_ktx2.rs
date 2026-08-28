// Decode or upload failures exit nonzero; malformed KTX2 data is never ignored.
fn main() {
    if let Err(error) = ez_gfx_examples::run(ez_gfx_examples::Example::SponzaKtx2) {
        eprintln!("Sponza KTX2 example failed: {error}");
        std::process::exit(1);
    }
}
