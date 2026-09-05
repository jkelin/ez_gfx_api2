//! Builds the thin standalone Basis bridge against the shared native transcoder.

fn main() {
    println!("cargo:rerun-if-changed=src/basis_wrapper.cpp");
    println!("cargo:rerun-if-changed=vendor/transcoder");

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file("src/basis_wrapper.cpp")
        .include("vendor/transcoder")
        .define("BASISD_SUPPORT_KTX2", "0")
        .warnings(false);
    if build.get_compiler().is_like_msvc() {
        build.flag("/std:c++17");
    } else {
        build.flag("-std=c++17").flag("-fno-strict-aliasing");
    }
    build.compile("ez_gfx_basis_wrapper");
}
