use std::{
    env,
    ffi::OsStr,
    fs::{self, File},
    io,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use flate2::{Compression, write::GzEncoder};
use sha2::{Digest, Sha256};
use walkdir::WalkDir;
use zip::{ZipWriter, write::SimpleFileOptions};

fn main() {
    if let Err(error) = dispatch() {
        eprintln!("xtask failed: {error}");
        std::process::exit(1);
    }
}

// Unknown tasks fail instead of silently selecting packaging defaults.
fn dispatch() -> Result<(), String> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("package") => package(PackageArgs::parse(args.collect())?),
        Some(task) => Err(format!("unknown task: {task}")),
        None => Err("task is required".to_owned()),
    }
}

struct PackageArgs {
    target: String,
    version: String,
    output: PathBuf,
}

impl PackageArgs {
    // Positional arguments are bounded to target, version, and output; extras are rejected.
    fn parse(args: Vec<String>) -> Result<Self, String> {
        if args.len() > 3 {
            return Err("usage: xtask package [target] [version] [output]".to_owned());
        }
        Ok(Self {
            target: args.first().cloned().map(Ok).unwrap_or_else(host_target)?,
            version: args.get(1).cloned().unwrap_or_else(|| "0.1.0".to_owned()),
            output: args
                .get(2)
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("dist")),
        })
    }
}

// A malformed rustc version response is rejected rather than producing an incorrectly named package.
fn host_target() -> Result<String, String> {
    let output = Command::new("rustc")
        .arg("-vV")
        .output()
        .map_err(|error| format!("run rustc: {error}"))?;
    if !output.status.success() {
        return Err("rustc -vV failed".to_owned());
    }
    String::from_utf8(output.stdout)
        .map_err(|_| "rustc output is not UTF-8".to_owned())?
        .lines()
        .find_map(|line| line.strip_prefix("host: ").map(str::to_owned))
        .ok_or_else(|| "rustc host target is missing".to_owned())
}

// Packaging is fail-fast: missing products, compiler libraries, or dependency isolation abort before archives are emitted.
fn package(args: PackageArgs) -> Result<(), String> {
    run(
        "cargo",
        &[
            "build",
            "--release",
            "--target",
            &args.target,
            "-p",
            "ez-gfx-ffi",
        ],
    )?;
    run(
        "cargo",
        &[
            "build",
            "--release",
            "--target",
            &args.target,
            "-p",
            "ez-gfx-compiler",
            "--bin",
            "ez-gfx-compile",
        ],
    )?;
    verify_runtime_tree(&args.target)?;

    let runtime_name = format!("ez-gfx-runtime-{}-{}", args.target, args.version);
    let compiler_name = format!("ez-gfx-compiler-{}-{}", args.target, args.version);
    let runtime = args.output.join(&runtime_name);
    let compiler = args.output.join(&compiler_name);
    remove_dir_if_present(&runtime)?;
    remove_dir_if_present(&compiler)?;
    fs::create_dir_all(&runtime).map_err(io_error("create runtime package"))?;
    fs::create_dir_all(&compiler).map_err(io_error("create compiler package"))?;

    copy_required("include/ez_gfx_api.h", runtime.join("ez_gfx_api.h"))?;
    copy_required("README.md", runtime.join("ROOT-README.md"))?;
    copy_required(
        "crates/ez-gfx-runtime/README.md",
        runtime.join("RUNTIME-README.md"),
    )?;
    copy_required("licenses/NOTICE.txt", runtime.join("NOTICE.txt"))?;
    copy_required("README.md", compiler.join("ROOT-README.md"))?;
    copy_required(
        "crates/ez-gfx-compiler/README.md",
        compiler.join("COMPILER-README.md"),
    )?;
    copy_required("licenses/NOTICE.txt", compiler.join("NOTICE.txt"))?;

    let release = PathBuf::from("target").join(&args.target).join("release");
    let (runtime_files, compiler_binary, native_names, archive) = platform_products(&args.target)?;
    for file in runtime_files {
        copy_required(release.join(file), runtime.join(file))?;
    }
    copy_required(
        release.join(compiler_binary),
        compiler.join(compiler_binary),
    )?;
    for name in native_names {
        let source = find_native_library(name)?;
        copy_required(source, compiler.join(name))?;
    }

    write_manifest(&runtime, &args.target, &args.version)?;
    write_manifest(&compiler, &args.target, &args.version)?;
    match archive {
        Archive::Zip => {
            write_zip(&runtime, &args.output.join(format!("{runtime_name}.zip")))?;
            write_zip(&compiler, &args.output.join(format!("{compiler_name}.zip")))?;
        }
        Archive::TarGz => {
            write_tar_gz(
                &runtime,
                &args.output.join(format!("{runtime_name}.tar.gz")),
            )?;
            write_tar_gz(
                &compiler,
                &args.output.join(format!("{compiler_name}.tar.gz")),
            )?;
        }
    }
    Ok(())
}

enum Archive {
    Zip,
    TarGz,
}

type PlatformProducts = (
    &'static [&'static str],
    &'static str,
    &'static [&'static str],
    Archive,
);

// Target families have explicit required products; unsupported triples cannot produce partial packages.
fn platform_products(target: &str) -> Result<PlatformProducts, String> {
    if target.contains("windows") {
        Ok((
            &["ez_gfx_ffi.dll", "ez_gfx_ffi.lib", "ez_gfx_ffi.dll.lib"],
            "ez-gfx-compile.exe",
            &["slang.dll", "dxcompiler.dll"],
            Archive::Zip,
        ))
    } else if target.contains("linux") {
        Ok((
            &["libez_gfx_ffi.so", "libez_gfx_ffi.a"],
            "ez-gfx-compile",
            &["libslang.so", "libdxcompiler.so"],
            Archive::TarGz,
        ))
    } else if target.ends_with("apple-darwin") {
        Ok((
            &["libez_gfx_ffi.dylib", "libez_gfx_ffi.a"],
            "ez-gfx-compile",
            &["libslang.dylib", "libdxcompiler.dylib"],
            Archive::TarGz,
        ))
    } else {
        Err(format!("unsupported target packaging matrix: {target}"))
    }
}

// Runtime dependency inspection treats command failure and any Slang/compiler match as fatal.
fn verify_runtime_tree(target: &str) -> Result<(), String> {
    let output = Command::new("cargo")
        .args([
            "tree",
            "-p",
            "ez-gfx-runtime",
            "--target",
            target,
            "--no-default-features",
        ])
        .output()
        .map_err(|error| format!("run cargo tree: {error}"))?;
    if !output.status.success() {
        return Err("cargo tree failed".to_owned());
    }
    let tree = String::from_utf8_lossy(&output.stdout).to_ascii_lowercase();
    if ["shader-slang", "ez-gfx-compiler", "slang"]
        .iter()
        .any(|name| tree.contains(name))
    {
        return Err("runtime dependency tree contains compiler or Slang".to_owned());
    }
    Ok(())
}

// Search is restricted to configured SDK roots and rejects absent or non-file matches.
fn find_native_library(name: &str) -> Result<PathBuf, String> {
    for root in [env::var_os("SLANG_DIR"), env::var_os("VULKAN_SDK")]
        .into_iter()
        .flatten()
    {
        let root = PathBuf::from(root);
        if !root.is_dir() {
            continue;
        }
        if let Some(path) = WalkDir::new(root)
            .into_iter()
            .filter_map(Result::ok)
            .find(|entry| entry.file_type().is_file() && entry.file_name() == OsStr::new(name))
            .map(|entry| entry.into_path())
        {
            return Ok(path);
        }
    }
    Err(format!("required native compiler library missing: {name}"))
}

// Existing package directories are removed, while absent paths are accepted.
fn remove_dir_if_present(path: &Path) -> Result<(), String> {
    match fs::remove_dir_all(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("remove {}: {error}", path.display())),
    }
}

// Required inputs must be regular files; directories and missing paths fail closed.
fn copy_required(source: impl AsRef<Path>, destination: impl AsRef<Path>) -> Result<(), String> {
    let source = source.as_ref();
    if !source.is_file() {
        return Err(format!("required artifact missing: {}", source.display()));
    }
    fs::copy(source, destination.as_ref())
        .map(|_| ())
        .map_err(|error| format!("copy {}: {error}", source.display()))
}

// Manifest paths are normalized to forward slashes and sorted for reproducible content.
fn write_manifest(directory: &Path, target: &str, version: &str) -> Result<(), String> {
    let manifest_name = format!("manifest-{target}-{version}.sha256");
    let mut lines = Vec::new();
    for entry in WalkDir::new(directory).into_iter().filter_map(Result::ok) {
        if !entry.file_type().is_file() || entry.file_name() == OsStr::new(&manifest_name) {
            continue;
        }
        let relative = entry
            .path()
            .strip_prefix(directory)
            .map_err(|error| error.to_string())?;
        let bytes = fs::read(entry.path()).map_err(io_error("read package file"))?;
        let digest = Sha256::digest(bytes);
        lines.push(format!(
            "{digest:x}  {}",
            relative.to_string_lossy().replace('\\', "/")
        ));
    }
    lines.sort_by(|left, right| {
        left.split_once("  ")
            .unwrap()
            .1
            .cmp(right.split_once("  ").unwrap().1)
    });
    fs::write(
        directory.join(manifest_name),
        format!("{}\n", lines.join("\n")),
    )
    .map_err(io_error("write package manifest"))
}

// ZIP entries use normalized relative names; empty directories are intentionally omitted.
fn write_zip(directory: &Path, output: &Path) -> Result<(), String> {
    let file = File::create(output).map_err(io_error("create ZIP archive"))?;
    let mut zip = ZipWriter::new(file);
    let options = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    for entry in WalkDir::new(directory)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
    {
        let relative = entry
            .path()
            .strip_prefix(directory)
            .map_err(|error| error.to_string())?;
        zip.start_file(relative.to_string_lossy().replace('\\', "/"), options)
            .map_err(|error| error.to_string())?;
        let mut source = File::open(entry.path()).map_err(io_error("open ZIP input"))?;
        io::copy(&mut source, &mut zip).map_err(io_error("write ZIP entry"))?;
    }
    zip.finish().map_err(|error| error.to_string())?;
    Ok(())
}

// TAR archives contain one top-level package directory and use deterministic gzip compression settings.
fn write_tar_gz(directory: &Path, output: &Path) -> Result<(), String> {
    let file = File::create(output).map_err(io_error("create tar archive"))?;
    let encoder = GzEncoder::new(file, Compression::default());
    let mut archive = tar::Builder::new(encoder);
    let name = directory
        .file_name()
        .ok_or_else(|| "package directory has no name".to_owned())?;
    archive
        .append_dir_all(name, directory)
        .map_err(io_error("write tar archive"))?;
    archive.finish().map_err(io_error("finish tar archive"))
}

// Child commands inherit stdio so compiler diagnostics remain actionable.
fn run(program: &str, args: &[&str]) -> Result<(), String> {
    let status = Command::new(program)
        .args(args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|error| format!("run {program}: {error}"))?;
    status
        .success()
        .then_some(())
        .ok_or_else(|| format!("{program} exited with {status}"))
}

// Static operation labels keep I/O errors concise while preserving their causes.
fn io_error(operation: &'static str) -> impl FnOnce(io::Error) -> String {
    move |error| format!("{operation}: {error}")
}
