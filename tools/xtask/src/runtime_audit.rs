use anyhow::{Context, Result, anyhow, bail};
use object::Object;
use std::{collections::BTreeSet, fs, path::Path};
use walkdir::WalkDir;

const FORBIDDEN_COMPILER_NATIVE_STEMS: [&str; 5] = ["slang", "dxcompiler", "dxil", "dxc", "dxcapi"];
const FORBIDDEN_COMPILER_TOOL_STEMS: [&str; 3] = ["ez-gfx-compile", "slangc", "dxc"];

fn parse_dynamic_imports(input: &[u8]) -> Result<BTreeSet<String>> {
    let file = object::File::parse(input).context("parse runtime object imports")?;
    let mut imports = BTreeSet::new();
    for library in file
        .import_libraries()
        .context("read runtime import libraries")?
    {
        let library = library.context("read runtime import library")?;
        let name =
            std::str::from_utf8(library.name()).context("runtime import library is not UTF-8")?;
        imports.insert(normalize_import_basename(name));
    }
    Ok(imports)
}

fn normalize_import_basename(name: &str) -> String {
    name.rsplit(['/', '\\'])
        .next()
        .unwrap_or(name)
        .to_ascii_lowercase()
}

fn library_stem(name: &str) -> &str {
    let stem = name.find(".so").map_or(name, |suffix| &name[..suffix]);
    let stem = [".dll", ".dylib", ".exe"]
        .iter()
        .find_map(|suffix| stem.strip_suffix(suffix))
        .unwrap_or(stem);
    stem.strip_prefix("lib").unwrap_or(stem)
}

fn is_forbidden_runtime_name(name: &str) -> bool {
    let normalized = normalize_import_basename(name);
    let stem = library_stem(&normalized);
    FORBIDDEN_COMPILER_NATIVE_STEMS.contains(&stem) || FORBIDDEN_COMPILER_TOOL_STEMS.contains(&stem)
}

fn forbidden_runtime_names(names: impl IntoIterator<Item = impl AsRef<str>>) -> BTreeSet<String> {
    names
        .into_iter()
        .map(|name| normalize_import_basename(name.as_ref()))
        .filter(|name| is_forbidden_runtime_name(name))
        .collect()
}

fn audit_runtime_library(input: &[u8]) -> Result<BTreeSet<String>> {
    let imports = parse_dynamic_imports(input)?;
    let forbidden = forbidden_runtime_names(&imports);
    if !forbidden.is_empty() {
        bail!(
            "runtime library imports forbidden compiler dependencies: [{}]",
            forbidden.into_iter().collect::<Vec<_>>().join(", ")
        );
    }
    Ok(imports)
}

pub(super) fn audit_runtime_package_contents(root: &Path) -> Result<()> {
    let mut forbidden = BTreeSet::new();
    for entry in WalkDir::new(root) {
        let entry = entry.with_context(|| format!("walk runtime package {}", root.display()))?;
        if !entry.file_type().is_file() {
            continue;
        }
        let name = entry.file_name().to_str().ok_or_else(|| {
            anyhow!(
                "runtime package filename is not UTF-8: {}",
                entry.path().display()
            )
        })?;
        if is_forbidden_runtime_name(name) {
            forbidden.insert(
                entry
                    .path()
                    .strip_prefix(root)
                    .context("resolve runtime package entry")?
                    .to_string_lossy()
                    .replace('\\', "/"),
            );
        }
    }
    if !forbidden.is_empty() {
        bail!(
            "runtime package contains forbidden compiler files: [{}]",
            forbidden.into_iter().collect::<Vec<_>>().join(", ")
        );
    }
    Ok(())
}

pub(super) fn audit_runtime_library_file(path: &Path) -> Result<BTreeSet<String>> {
    let bytes =
        fs::read(path).with_context(|| format!("read runtime library {}", path.display()))?;
    audit_runtime_library(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forbidden_runtime_names_are_normalized_across_platforms() {
        assert_eq!(
            forbidden_runtime_names([
                "C:\\sdk\\slang.dll",
                "/usr/lib/libdxcompiler.so.1",
                "libsafe.dylib",
                "tools/ez-gfx-compile.exe",
            ]),
            BTreeSet::from([
                "ez-gfx-compile.exe".to_owned(),
                "libdxcompiler.so.1".to_owned(),
                "slang.dll".to_owned(),
            ])
        );
    }
}
