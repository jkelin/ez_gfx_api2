use super::*;

pub(super) fn assert_validation_clean(child_marker: &str) {
    #[cfg(windows)]
    use std::os::windows::process::CommandExt;
    use std::{
        io::{Read, Write},
        process::{Command, Stdio},
    };

    // The validation layer writes directly to native stdout/stderr, outside Rust's test capture.
    // A child process makes those messages test failures even when the rendered pixels look right.
    let mut command = Command::new(std::env::current_exe().unwrap());
    command.args([
        "--exact",
        "vulkan_bc_pixels_survive_region_updates_and_unload",
        "--nocapture",
        "--test-threads=1",
    ]);
    command.env(child_marker, "1");
    // CREATE_NO_WINDOW: never create or activate a console.
    #[cfg(windows)]
    command.creation_flags(0x0800_0000);
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn hidden Vulkan validation regression");
    let mut stdout = child.stdout.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();
    // Drain both pipes concurrently: a flood of validation errors must not deadlock the child.
    let output = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        stdout
            .read_to_end(&mut bytes)
            .expect("read Vulkan child stdout");
        bytes
    });
    let errors = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        stderr
            .read_to_end(&mut bytes)
            .expect("read Vulkan child stderr");
        bytes
    });
    let deadline = Instant::now() + Duration::from_secs(120);
    let (status, timed_out) = loop {
        if let Some(status) = child.try_wait().expect("poll Vulkan validation child") {
            break (status, false);
        }
        if Instant::now() >= deadline {
            child
                .kill()
                .expect("terminate stalled Vulkan validation child");
            break (child.wait().expect("reap Vulkan validation child"), true);
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let output = output.join().expect("join Vulkan stdout reader");
    let errors = errors.join().expect("join Vulkan stderr reader");
    std::io::stdout()
        .write_all(&output)
        .expect("re-emit Vulkan stdout");
    std::io::stderr()
        .write_all(&errors)
        .expect("re-emit Vulkan stderr");
    assert!(!timed_out, "Vulkan validation child exceeded 120 seconds");
    assert!(status.success(), "Vulkan validation child failed: {status}");
    for bytes in [&output, &errors] {
        let text = String::from_utf8_lossy(bytes);
        assert!(
            !text.contains("VUID-") && !text.contains("Validation EzGfxResult"),
            "Vulkan validation reported an error; see native output above"
        );
    }
}
