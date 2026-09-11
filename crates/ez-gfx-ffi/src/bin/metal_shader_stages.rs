//! Hidden main-thread Metal shader-stage integration runner.
//!
//! Libtest runs tests on worker threads, but `AppKit` surfaces must be created and
//! used on the process main thread. This helper never creates or shows a window.

#[cfg(target_vendor = "apple")]
#[path = "../../tests/shader_stages.rs"]
mod shader_stages;

fn main() {
    #[cfg(target_vendor = "apple")]
    {
        let mut args = std::env::args_os().skip(1);
        let scenario = args.next().expect("missing shader-stage scenario");
        let artifact = args.next().expect("missing shader artifact");
        assert!(args.next().is_none(), "unexpected shader-stage argument");
        let scenario = scenario
            .to_str()
            .expect("shader-stage scenario must be UTF-8");
        let artifact = std::fs::read(artifact).expect("read shader-stage artifact");
        shader_stages::run_metal_scenario(scenario, artifact);
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        eprintln!("metal_shader_stages: Apple Metal host required");
        std::process::exit(2);
    }
}
