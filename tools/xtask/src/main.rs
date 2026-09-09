//! Packaging and repository quality checks.
mod runtime_audit;

use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Parser, Subcommand};
use flate2::{Compression, write::GzEncoder};
use sha2::{Digest, Sha256};
use std::{
    env,
    ffi::OsStr,
    fs::{self, File},
    io,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};
use walkdir::WalkDir;
use zip::{ZipWriter, write::SimpleFileOptions};

#[derive(Debug, Parser)]
#[command(
    name = "xtask",
    about = "Repository packaging and quality tasks",
    arg_required_else_help = true
)]
struct Cli {
    #[command(subcommand)]
    task: Task,
}

#[derive(Debug, Subcommand)]
enum Task {
    /// Build compiler and runtime distribution archives.
    Package(PackageCommand),
    /// Check tracked Rust files against the source-line limit.
    SourceLines,
}

#[derive(Args, Debug)]
struct PackageCommand {
    /// Rust target triple. Defaults to the host target.
    target: Option<String>,
    /// Package version.
    version: Option<String>,
    /// Archive output directory.
    output: Option<PathBuf>,
}

fn main() {
    let cli = Cli::parse();
    if let Err(error) = dispatch(cli) {
        eprintln!("xtask failed: {error:#}");
        std::process::exit(1);
    }
}

fn dispatch(cli: Cli) -> Result<()> {
    match cli.task {
        Task::Package(command) => package(&command.resolve()?),
        Task::SourceLines => source_lines(),
    }
}

struct PackageArgs {
    target: String,
    version: String,
    output: PathBuf,
}

impl PackageCommand {
    fn resolve(self) -> Result<PackageArgs> {
        Ok(PackageArgs {
            target: self.target.map_or_else(host_target, Ok)?,
            version: self.version.unwrap_or_else(|| "0.1.0".to_owned()),
            output: self.output.unwrap_or_else(|| PathBuf::from("dist")),
        })
    }
}

// A malformed rustc version response is rejected rather than producing an incorrectly named package.
fn host_target() -> Result<String> {
    let output = Command::new("rustc")
        .arg("-vV")
        .output()
        .context("run rustc -vV")?;
    if !output.status.success() {
        bail!("rustc -vV failed");
    }
    String::from_utf8(output.stdout)
        .context("rustc output is not UTF-8")?
        .lines()
        .find_map(|line| line.strip_prefix("host: ").map(str::to_owned))
        .ok_or_else(|| anyhow!("rustc host target is missing"))
}

// Packaging is fail-fast: missing products, compiler libraries, or dependency isolation abort before archives are emitted.
fn package(args: &PackageArgs) -> Result<()> {
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
    fs::create_dir_all(&runtime)
        .with_context(|| format!("create runtime package {}", runtime.display()))?;
    fs::create_dir_all(&compiler)
        .with_context(|| format!("create compiler package {}", compiler.display()))?;

    copy_required(
        "bindings/c/include/ez_gfx_api.h",
        runtime.join("ez_gfx_api.h"),
    )?;
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

    runtime_audit::audit_runtime_package_contents(&runtime)?;
    let imports = runtime_audit::audit_runtime_library_file(&runtime.join(runtime_files[0]))?;
    println!(
        "runtime-import-audit: imports [{}], forbidden 0",
        imports.into_iter().collect::<Vec<_>>().join(", ")
    );

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
fn platform_products(target: &str) -> Result<PlatformProducts> {
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
        bail!("unsupported target packaging matrix: {target}")
    }
}

// Runtime dependency inspection treats command failure and any Slang/compiler match as fatal.
fn verify_runtime_tree(target: &str) -> Result<()> {
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
        .context("run cargo tree")?;
    if !output.status.success() {
        bail!("cargo tree failed");
    }
    let tree = String::from_utf8_lossy(&output.stdout).to_ascii_lowercase();
    if ["shader-slang", "ez-gfx-compiler", "slang"]
        .iter()
        .any(|name| tree.contains(name))
    {
        bail!("runtime dependency tree contains compiler or Slang");
    }
    Ok(())
}

// Search is restricted to configured SDK roots and rejects absent or non-file matches.
fn find_native_library(name: &str) -> Result<PathBuf> {
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
            .filter_map(std::result::Result::ok)
            .find(|entry| entry.file_type().is_file() && entry.file_name() == OsStr::new(name))
            .map(walkdir::DirEntry::into_path)
        {
            return Ok(path);
        }
    }
    bail!("required native compiler library missing: {name}")
}

// Existing package directories are removed, while absent paths are accepted.
fn remove_dir_if_present(path: &Path) -> Result<()> {
    match fs::remove_dir_all(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("remove {}", path.display())),
    }
}

// Required inputs must be regular files; directories and missing paths fail closed.
fn copy_required(source: impl AsRef<Path>, destination: impl AsRef<Path>) -> Result<()> {
    let source = source.as_ref();
    if !source.is_file() {
        bail!("required artifact missing: {}", source.display());
    }
    fs::copy(source, destination.as_ref())
        .with_context(|| format!("copy {}", source.display()))
        .map(|_| ())
}

// Manifest paths are normalized to forward slashes and sorted for reproducible content.
fn write_manifest(directory: &Path, target: &str, version: &str) -> Result<()> {
    let manifest_name = format!("manifest-{target}-{version}.sha256");
    let mut lines = Vec::new();
    for entry in WalkDir::new(directory)
        .into_iter()
        .filter_map(std::result::Result::ok)
    {
        if !entry.file_type().is_file() || entry.file_name() == OsStr::new(&manifest_name) {
            continue;
        }
        let relative = entry
            .path()
            .strip_prefix(directory)
            .context("resolve package manifest path")?;
        let bytes = fs::read(entry.path()).context("read package file")?;
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
    .context("write package manifest")
}

// ZIP entries use normalized relative names; empty directories are intentionally omitted.
fn write_zip(directory: &Path, output: &Path) -> Result<()> {
    let file = File::create(output).context("create ZIP archive")?;
    let mut zip = ZipWriter::new(file);
    let options = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    for entry in WalkDir::new(directory)
        .into_iter()
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.file_type().is_file())
    {
        let relative = entry
            .path()
            .strip_prefix(directory)
            .context("resolve ZIP entry path")?;
        zip.start_file(relative.to_string_lossy().replace('\\', "/"), options)
            .context("start ZIP entry")?;
        let mut source = File::open(entry.path()).context("open ZIP input")?;
        io::copy(&mut source, &mut zip).context("write ZIP entry")?;
    }
    zip.finish().context("finish ZIP archive")?;
    Ok(())
}

// TAR archives contain one top-level package directory and use deterministic gzip compression settings.
fn write_tar_gz(directory: &Path, output: &Path) -> Result<()> {
    let file = File::create(output).context("create tar archive")?;
    let encoder = GzEncoder::new(file, Compression::default());
    let mut archive = tar::Builder::new(encoder);
    let name = directory
        .file_name()
        .ok_or_else(|| anyhow!("package directory has no name"))?;
    archive
        .append_dir_all(name, directory)
        .context("write tar archive")?;
    archive.finish().context("finish tar archive")
}

// Child commands inherit stdio so compiler diagnostics remain actionable.
fn run(program: &str, args: &[&str]) -> Result<()> {
    let status = Command::new(program)
        .args(args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .with_context(|| format!("run {program}"))?;
    if !status.success() {
        bail!("{program} exited with {status}");
    }
    Ok(())
}

const SOURCE_LINE_LIMIT: usize = 1_200;

fn source_lines() -> Result<()> {
    let root = env::current_dir().context("find repository root")?;
    let root_string = root
        .to_str()
        .ok_or_else(|| anyhow!("repository root is not UTF-8"))?;
    let output = Command::new("git")
        .args([
            "-C",
            root_string,
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "*.rs",
        ])
        .output()
        .context("run git ls-files")?;

    if !output.status.success() {
        bail!("git ls-files exited with {}", output.status);
    }

    let listed = String::from_utf8(output.stdout).context("git ls-files output is not UTF-8")?;
    let files = existing_rust_paths(&root, &parse_rust_paths(&listed)?)?;
    let diagnostics = source_line_diagnostics(&root, &files, SOURCE_LINE_LIMIT)?;
    if diagnostics.is_empty() {
        println!(
            "source-lines: all tracked and unignored Rust files are at most {SOURCE_LINE_LIMIT} lines"
        );
        return Ok(());
    }
    for diagnostic in &diagnostics {
        eprintln!("{diagnostic}");
    }
    bail!(
        "source-lines: {} file(s) exceed the {SOURCE_LINE_LIMIT}-line limit",
        diagnostics.len()
    )
}

fn parse_rust_paths(output: &str) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for line in output.lines() {
        if line.is_empty() {
            continue;
        }
        let path = PathBuf::from(line);
        if path.is_absolute()
            || path
                .components()
                .any(|component| component == std::path::Component::ParentDir)
        {
            bail!("git ls-files returned invalid relative path: {line}");
        }
        if path.extension() == Some(OsStr::new("rs")) {
            paths.push(path);
        }
    }
    paths.sort();
    paths.dedup();
    Ok(paths)
}

fn existing_rust_paths(root: &Path, paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    if !root.is_dir() {
        bail!("repository root is not a directory: {}", root.display());
    }
    Ok(paths
        .iter()
        .filter(|path| root.join(path).is_file())
        .cloned()
        .collect())
}

fn source_line_diagnostics(root: &Path, files: &[PathBuf], limit: usize) -> Result<Vec<String>> {
    if !root.is_dir() {
        bail!("repository root is not a directory: {}", root.display());
    }
    if limit == 0 {
        bail!("source line limit must be greater than zero");
    }

    let mut diagnostics = Vec::new();
    for path in files {
        if path.is_absolute()
            || path
                .components()
                .any(|component| component == std::path::Component::ParentDir)
        {
            bail!("invalid relative Rust path: {}", path.display());
        }
        let full_path = root.join(path);
        let bytes =
            fs::read(&full_path).with_context(|| format!("read {}", full_path.display()))?;
        let lines = physical_line_count(&bytes);
        if lines > limit {
            diagnostics.push(format!(
                "source-lines: {} has {lines} physical lines (maximum {limit})",
                path.display()
            ));
        }
    }
    Ok(diagnostics)
}

fn physical_line_count(bytes: &[u8]) -> usize {
    // Empty input has no lines; a final newline terminates the preceding line
    // without creating an additional empty line.
    let newline_count = bytes
        .iter()
        .fold(0, |count, &byte| count + usize::from(byte == b'\n'));
    newline_count + usize::from(bytes.last().is_some_and(|&byte| byte != b'\n'))
}

#[cfg(test)]
mod source_line_tests {
    use super::{
        Cli, PackageCommand, Task, existing_rust_paths, parse_rust_paths, physical_line_count,
        source_line_diagnostics,
    };
    use clap::{Parser, error::ErrorKind};
    use std::{
        fs,
        path::{Path, PathBuf},
    };

    #[test]
    fn cli_preserves_subcommands_and_package_positionals() {
        assert!(matches!(
            Cli::try_parse_from(["xtask", "source-lines"]).unwrap().task,
            Task::SourceLines
        ));

        let cli = Cli::try_parse_from([
            "xtask",
            "package",
            "x86_64-pc-windows-msvc",
            "1.2.3",
            "release",
        ])
        .unwrap();
        assert!(matches!(
            cli.task,
            Task::Package(PackageCommand {
                target: Some(target),
                version: Some(version),
                output: Some(output),
            }) if target == "x86_64-pc-windows-msvc"
                && version == "1.2.3"
                && output == PathBuf::from("release")
        ));
    }

    #[test]
    fn cli_help_and_invalid_arguments_use_clap_errors() {
        assert_eq!(
            Cli::try_parse_from(["xtask"]).unwrap_err().kind(),
            ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
        );
        assert_eq!(
            Cli::try_parse_from(["xtask", "--help"]).unwrap_err().kind(),
            ErrorKind::DisplayHelp
        );
        assert_eq!(
            Cli::try_parse_from(["xtask", "unknown"])
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidSubcommand
        );
        assert_eq!(
            Cli::try_parse_from(["xtask", "package", "a", "b", "c", "extra"])
                .unwrap_err()
                .kind(),
            ErrorKind::UnknownArgument
        );
    }

    #[test]
    fn physical_line_count_handles_empty_and_boundary_newlines() {
        assert_eq!(physical_line_count(b""), 0);
        assert_eq!(physical_line_count(b"\n"), 1);
        assert_eq!(physical_line_count(b"\n\n"), 2);
        assert_eq!(physical_line_count(b"one"), 1);
        assert_eq!(physical_line_count(b"one\n"), 1);
        assert_eq!(physical_line_count(b"one\ntwo"), 2);
        assert_eq!(physical_line_count(b"one\n\ntwo\n"), 3);
    }

    #[test]
    fn parse_paths_filters_sorts_deduplicates_and_rejects_escape() {
        assert_eq!(
            parse_rust_paths("z.rs\na.txt\nz.rs\na.rs\n").unwrap(),
            vec![PathBuf::from("a.rs"), PathBuf::from("z.rs")]
        );
        assert!(parse_rust_paths("../escape.rs").is_err());
    }

    #[test]
    fn diagnostics_are_deterministic_and_validate_inputs() {
        let root = std::env::temp_dir().join(format!("xtask-source-lines-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("short.rs"), b"a\n").unwrap();
        fs::write(root.join("long.rs"), b"a\nb\nc").unwrap();
        let files = vec!["long.rs".into(), "short.rs".into()];
        assert_eq!(
            source_line_diagnostics(Path::new(&root), &files, 2).unwrap(),
            vec!["source-lines: long.rs has 3 physical lines (maximum 2)"]
        );
        assert!(source_line_diagnostics(Path::new(&root), &files, 0).is_err());
        let _ = fs::remove_dir_all(root);
    }
    #[test]
    fn source_paths_skip_deleted_files() {
        let root = std::env::temp_dir().join(format!("xtask-source-paths-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("tracked.rs"), b"fn tracked() {}\n").unwrap();
        fs::write(root.join("untracked.rs"), b"fn untracked() {}\n").unwrap();
        // `git ls-files --exclude-standard` omits ignored paths before this helper runs.
        let listed = parse_rust_paths("tracked.rs\ndeleted.rs\nuntracked.rs\n").unwrap();
        assert_eq!(
            existing_rust_paths(Path::new(&root), &listed).unwrap(),
            vec![PathBuf::from("tracked.rs"), PathBuf::from("untracked.rs")]
        );
        let _ = fs::remove_dir_all(root);
    }
}
