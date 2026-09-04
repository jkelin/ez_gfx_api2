use crate::declaration_parity::{PUBLIC_PREFIX, is_identifier_byte, validated_contract};
use anyhow::{Context, Result, anyhow, bail};
use object::{BinaryFormat, Object};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};
use walkdir::WalkDir;

const RUNTIME_LIBRARY_NAMES: [&str; 3] =
    ["ez_gfx_ffi.dll", "libez_gfx_ffi.so", "libez_gfx_ffi.dylib"];
const FORBIDDEN_COMPILER_NATIVE_STEMS: [&str; 5] = ["slang", "dxcompiler", "dxil", "dxc", "dxcapi"];
const FORBIDDEN_COMPILER_TOOL_STEMS: [&str; 3] = ["ez-gfx-compile", "slangc", "dxc"];

// The header is the reference set; bindings and the selected binary must match exactly.
pub(super) fn check(library_input: &Path) -> Result<()> {
    let root = std::env::current_dir().context("find repository root")?;
    let header_path = root.join("include/ez_gfx_api.h");
    let bindings_path = root.join("bindings/bindings.xml");
    let library_path = resolve_library(library_input)?;

    let header = fs::read_to_string(&header_path)
        .with_context(|| format!("read public header {}", header_path.display()))?;
    let bindings = fs::read_to_string(&bindings_path)
        .with_context(|| format!("read bindings {}", bindings_path.display()))?;
    let library = fs::read(&library_path)
        .with_context(|| format!("read runtime library {}", library_path.display()))?;
    let imports = audit_runtime_library(&library)?;
    if library_input.is_dir() {
        audit_runtime_package_contents(library_input)?;
    }

    let contract = validated_contract(&header, &bindings)?;
    let binary_functions = parse_binary_functions(&library)?;
    let failures = parity_failures(
        &contract.functions,
        &[("runtime library", &binary_functions)],
    );
    if !failures.is_empty() {
        bail!("export parity failed:\n{}", failures.join("\n"));
    }

    println!(
        "export-parity: {} functions, {} parameters, {} pointers, {} counted pointers, {} handles, {} structs/{} fields, {} enums/{} discriminants, and {} managed overrides match declarations; exports match {}; runtime imports [{}], forbidden 0",
        contract.counts.functions,
        contract.counts.parameters,
        contract.counts.pointers,
        contract.counts.counted_pointers,
        contract.counts.handles,
        contract.counts.structs,
        contract.counts.fields,
        contract.counts.enums,
        contract.counts.discriminants,
        contract.counts.managed_overrides,
        library_path.display(),
        imports.iter().cloned().collect::<Vec<_>>().join(", ")
    );
    Ok(())
}

// A file is used directly; a package root must contain exactly one supported runtime library.
fn resolve_library(input: &Path) -> Result<PathBuf> {
    if input.is_file() {
        return Ok(input.to_path_buf());
    }
    if !input.is_dir() {
        bail!(
            "runtime library or package root does not exist: {}",
            input.display()
        );
    }

    let matches = RUNTIME_LIBRARY_NAMES
        .iter()
        .map(|name| input.join(name))
        .filter(|path| path.is_file())
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [library] => Ok(library.clone()),
        [] => bail!(
            "package root {} contains none of: {}",
            input.display(),
            RUNTIME_LIBRARY_NAMES.join(", ")
        ),
        _ => bail!(
            "package root {} contains multiple runtime libraries: {}",
            input.display(),
            display_paths(&matches)
        ),
    }
}

// Package roots are shallow, so diagnostics join direct candidates without recursion.
fn display_paths(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

// Dynamic import libraries are parsed from the same object representation as exports.
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

// Both slash forms are accepted because PE paths can be inspected on Unix and vice versa.
fn normalize_import_basename(name: &str) -> String {
    name.rsplit(['/', '\\'])
        .next()
        .unwrap_or(name)
        .to_ascii_lowercase()
}

// Versioned ELF names normalize to the same compiler-native stem as PE and Mach-O names.
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

// Ordinal-only and unrelated exports are ignored; malformed public exports fail closed.
fn parse_binary_functions(input: &[u8]) -> Result<BTreeSet<String>> {
    let file = object::File::parse(input).context("parse runtime object")?;
    let macho = file.format() == BinaryFormat::MachO;
    let mut functions = BTreeSet::new();
    for export in file.exports().context("read runtime exports")? {
        let export = export.context("read runtime export")?;
        let exported_name = export.name();
        let Some(name) = exported_name.name() else {
            continue;
        };
        insert_public_export(&mut functions, name, macho)?;
    }
    if functions.is_empty() {
        bail!("runtime library contains no public {PUBLIC_PREFIX} exports");
    }
    Ok(functions)
}

// Tests exercise export normalization without constructing platform object fixtures.
#[cfg(test)]
fn parse_public_export_names<'a>(
    names: impl IntoIterator<Item = &'a [u8]>,
    macho: bool,
) -> Result<BTreeSet<String>> {
    let mut functions = BTreeSet::new();
    for name in names {
        insert_public_export(&mut functions, name, macho)?;
    }
    Ok(functions)
}

// Mach-O stores C exports with one leading underscore; PE and ELF use the source name.
fn insert_public_export(
    functions: &mut BTreeSet<String>,
    raw_name: &[u8],
    macho: bool,
) -> Result<()> {
    let name = if macho {
        raw_name.strip_prefix(b"_").unwrap_or(raw_name)
    } else {
        raw_name
    };
    if !name.starts_with(PUBLIC_PREFIX.as_bytes()) {
        return Ok(());
    }
    let name = std::str::from_utf8(name).context("public runtime export is not UTF-8")?;
    if name.len() == PUBLIC_PREFIX.len()
        || !name[PUBLIC_PREFIX.len()..].bytes().all(is_identifier_byte)
    {
        bail!("invalid public runtime export name: {name}");
    }
    if !functions.insert(name.to_owned()) {
        bail!("duplicate public runtime export: {name}");
    }
    Ok(())
}

// Every candidate reports both set directions so one run explains the complete repair.
fn parity_failures(
    expected: &BTreeSet<String>,
    candidates: &[(&str, &BTreeSet<String>)],
) -> Vec<String> {
    let mut failures = Vec::new();
    for &(label, actual) in candidates {
        let missing = expected.difference(actual).cloned().collect::<Vec<_>>();
        let extra = actual.difference(expected).cloned().collect::<Vec<_>>();
        if !missing.is_empty() {
            failures.push(format!("{label} missing: [{}]", missing.join(", ")));
        }
        if !extra.is_empty() {
            failures.push(format!("{label} extra: [{}]", extra.join(", ")));
        }
    }
    failures
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::declaration_parity::{DeclarationCounts, compare_declarations};

    fn names(values: &[&str]) -> BTreeSet<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    const VALID_HEADER: &str = r"
        #define EZ_GFX_ABI_VERSION 19u
        /* ez_gfx_documented_only() */
        uint32_t ez_gfx_one(void);
        uint32_t ez_gfx_two(void);
    ";

    fn bindings(functions: &str) -> String {
        format!(
            r#"<ez-gfx-bindings abi-version="19">
                <handles></handles><enums></enums><structs></structs>
                <functions>{functions}</functions>
            </ez-gfx-bindings>"#
        )
    }

    fn valid_bindings() -> String {
        bindings(
            r#"<function name="ez_gfx_one" return="uint32_t"/>
               <function name="ez_gfx_two" return="uint32_t"/>"#,
        )
    }

    #[test]
    fn validated_contract_ignores_comments_and_collects_public_declarations() {
        assert_eq!(
            validated_contract(VALID_HEADER, &valid_bindings())
                .unwrap()
                .functions,
            names(&["ez_gfx_one", "ez_gfx_two"])
        );
    }

    #[test]
    fn validated_contract_rejects_duplicate_and_malformed_header_declarations() {
        let duplicate =
            VALID_HEADER.replace("uint32_t ez_gfx_two(void);", "uint32_t ez_gfx_one(void);");
        assert!(
            validated_contract(&duplicate, &valid_bindings())
                .unwrap_err()
                .to_string()
                .contains("duplicate")
        );

        for declaration in [
            "void ez_gfx_missing_parentheses;",
            "void ez_gfx_bad-name(void);",
            "/* no declarations */",
        ] {
            let header = format!("#define EZ_GFX_ABI_VERSION 19u\n{declaration}");
            assert!(validated_contract(&header, &valid_bindings()).is_err());
        }
    }

    #[test]
    fn validated_contract_rejects_malformed_missing_and_duplicate_binding_names() {
        assert!(validated_contract(VALID_HEADER, "<functions>").is_err());
        assert!(
            validated_contract(VALID_HEADER, &bindings(r#"<function return="uint32_t"/>"#))
                .is_err()
        );
        assert!(
            validated_contract(
                VALID_HEADER,
                &bindings(
                    r#"<function name="ez_gfx_one" return="uint32_t"/>
                       <function name="ez_gfx_one" return="uint32_t"/>"#
                )
            )
            .unwrap_err()
            .to_string()
            .contains("duplicate")
        );
    }

    #[test]
    fn export_name_parser_normalizes_macho_and_rejects_bad_public_names() {
        assert_eq!(
            parse_public_export_names([b"_ez_gfx_one".as_slice(), b"_system".as_slice()], true)
                .unwrap(),
            names(&["ez_gfx_one"])
        );
        assert_eq!(
            parse_public_export_names([b"ez_gfx_one".as_slice(), b"system".as_slice()], false)
                .unwrap(),
            names(&["ez_gfx_one"])
        );
        assert!(
            parse_public_export_names(
                [b"ez_gfx_same".as_slice(), b"ez_gfx_same".as_slice()],
                false
            )
            .unwrap_err()
            .to_string()
            .contains("duplicate")
        );
        assert!(parse_public_export_names([b"ez_gfx_\xff".as_slice()], false).is_err());
    }
    #[test]
    fn binary_parser_rejects_malformed_objects() {
        assert!(parse_binary_functions(b"not an object").is_err());
    }

    #[test]
    fn import_basename_normalizes_case_and_path_variants() {
        for (input, expected) in [
            (r"C:\SDK\SLANG.DLL", "slang.dll"),
            ("/opt/lib/libDxCompiler.so", "libdxcompiler.so"),
            ("@rpath/LibSlang.DyLiB", "libslang.dylib"),
            ("libSlang.so.1", "libslang.so.1"),
        ] {
            assert_eq!(normalize_import_basename(input), expected);
        }
    }

    #[test]
    fn forbidden_import_matching_covers_compiler_libraries_and_tooling() {
        assert_eq!(
            forbidden_runtime_names([
                r"C:\bin\DXCOMPILER.DLL",
                "/usr/lib/libslang.so.1",
                "@rpath/libdxil.dylib",
                "tools/EZ-GFX-COMPILE.EXE",
                "dxc",
                "slangc.exe",
            ]),
            names(&[
                "dxc",
                "dxcompiler.dll",
                "ez-gfx-compile.exe",
                "libdxil.dylib",
                "libslang.so.1",
                "slangc.exe",
            ])
        );
    }

    #[test]
    fn graphics_and_operating_system_imports_are_allowed() {
        assert!(
            forbidden_runtime_names([
                "KERNEL32.dll",
                "vulkan-1.dll",
                "d3d12.dll",
                "dxgi.dll",
                "/System/Library/Frameworks/Metal.framework/Metal",
                "libvulkan.so.1",
                "libc.so.6",
            ])
            .is_empty()
        );
    }

    #[test]
    fn dynamic_import_parser_rejects_malformed_objects() {
        assert!(parse_dynamic_imports(b"not an object").is_err());
    }

    #[test]
    fn parity_failures_report_sorted_missing_and_extra_sets() {
        let expected = names(&["ez_gfx_a", "ez_gfx_b"]);
        let bindings = names(&["ez_gfx_b", "ez_gfx_c"]);
        let binary = names(&["ez_gfx_a", "ez_gfx_b"]);

        assert_eq!(
            parity_failures(
                &expected,
                &[("bindings.xml", &bindings), ("runtime library", &binary)]
            ),
            vec![
                "bindings.xml missing: [ez_gfx_a]".to_owned(),
                "bindings.xml extra: [ez_gfx_c]".to_owned(),
            ]
        );
    }

    const CONTRACT_HEADER: &str = r"
        #define EZ_GFX_ABI_VERSION 19u
        #define EZ_GFX_ACCESS(...)
        typedef uint64_t EzGfxContext;
        typedef uint8_t EzGfxResult;
        enum { EzGfxResult_Ok = 0, EzGfxResult_InvalidArgument = 1 };
        typedef uint8_t EzGfxTextureAddressMode;
        enum {
            EzGfxTextureAddressMode_Repeat = 0,
            EzGfxTextureAddressMode_ClampToEdge = 1
        };
        typedef struct EzGfxHandleParts {
            uint32_t context_slot;
            uint8_t is_context;
            uint8_t _padding[3];
        } EzGfxHandleParts;
        EzGfxResult ez_gfx_context_create(const void *desc, EzGfxContext *out_context)
            EZ_GFX_ACCESS(write_only, 2);
        EzGfxResult ez_gfx_vertex_upload_indices(
            const uint32_t *data, uint32_t count, uint32_t *out_start_index, EzGfxContext context)
            EZ_GFX_ACCESS(read_only, 1, 2) EZ_GFX_ACCESS(write_only, 3);
        EzGfxResult ez_gfx_semantic_id(
            const uint8_t *name, size_t length, uint8_t *out_id)
            EZ_GFX_ACCESS(read_only, 1, 2) EZ_GFX_ACCESS(write_only, 3);
    ";

    const CONTRACT_BINDINGS: &str = r#"
        <ez-gfx-bindings abi-version="19">
          <handles><handle name="EzGfxContext"/></handles>
          <enums>
            <enum name="EzGfxResult" underlying="uint8_t">
              <value name="EzGfxResult_Ok" value="0"/>
              <value name="EzGfxResult_InvalidArgument" value="1"/>
            </enum>
            <enum name="EzGfxTextureAddressMode" underlying="uint8_t">
              <value name="EzGfxTextureAddressMode_Repeat" value="0"/>
              <value name="EzGfxTextureAddressMode_ClampToEdge" value="1"/>
            </enum>
          </enums>
          <structs>
            <struct name="EzGfxHandleParts">
              <field name="context_slot" type="uint32_t"/>
              <field name="is_context" type="uint8_t"/>
              <field name="_padding" type="uint8_t" array-length="3"/>
            </struct>
          </structs>
          <functions>
            <function name="ez_gfx_context_create" return="EzGfxResult" managed="context-create">
              <param name="desc" type="const void *" direction="in"/>
              <param name="out_context" type="EzGfxContext *" direction="out"/>
            </function>
            <function name="ez_gfx_vertex_upload_indices" return="EzGfxResult">
              <param name="data" type="const uint32_t *" direction="in"
                     array-length="count" array-byte-size="4"/>
              <param name="count" type="uint32_t"/>
              <param name="out_start_index" type="uint32_t *" direction="out"/>
              <param name="context" type="EzGfxContext"/>
            </function>
            <function name="ez_gfx_semantic_id" return="EzGfxResult">
              <param name="name" type="const uint8_t *" direction="in" array-length="length"/>
              <param name="length" type="size_t"/>
              <param name="out_id" type="uint8_t *" direction="out" array-length="16"/>
            </function>
          </functions>
        </ez-gfx-bindings>
    "#;

    fn contract_error(header: &str, bindings: &str) -> String {
        compare_declarations(header, bindings)
            .unwrap_err()
            .to_string()
    }

    #[test]
    fn declaration_parity_accepts_complete_controlled_contract() {
        let counts = compare_declarations(CONTRACT_HEADER, CONTRACT_BINDINGS).unwrap();
        assert_eq!(
            counts,
            DeclarationCounts {
                functions: 3,
                parameters: 9,
                pointers: 6,
                counted_pointers: 3,
                handles: 1,
                structs: 1,
                fields: 3,
                enums: 2,
                discriminants: 4,
                managed_overrides: 1,
            }
        );
    }

    #[test]
    fn declaration_parity_reports_version_signature_handle_and_struct_drift() {
        for (header, bindings, expected) in [
            (
                CONTRACT_HEADER.replace("19u", "20u"),
                CONTRACT_BINDINGS.to_owned(),
                "ABI version",
            ),
            (
                CONTRACT_HEADER.replace("const void *desc", "void *desc"),
                CONTRACT_BINDINGS.to_owned(),
                "ez_gfx_context_create",
            ),
            (
                CONTRACT_HEADER.replace(
                    "typedef uint64_t EzGfxContext",
                    "typedef uint32_t EzGfxContext",
                ),
                CONTRACT_BINDINGS.to_owned(),
                "EzGfxContext",
            ),
            (
                CONTRACT_HEADER.replace("uint8_t is_context;", "uint32_t is_context;"),
                CONTRACT_BINDINGS.to_owned(),
                "EzGfxHandleParts",
            ),
        ] {
            assert!(contract_error(&header, &bindings).contains(expected));
        }
    }

    #[test]
    fn declaration_parity_reports_missing_struct_and_enum_discriminant_drift() {
        let missing_struct = CONTRACT_BINDINGS.replace(
            "<struct name=\"EzGfxHandleParts\">",
            "<struct name=\"Other\">",
        );
        assert!(contract_error(CONTRACT_HEADER, &missing_struct).contains("EzGfxHandleParts"));

        for bindings in [
            CONTRACT_BINDINGS.replace(
                "<value name=\"EzGfxTextureAddressMode_Repeat\" value=\"0\"/>",
                "<value name=\"EzGfxTextureAddressMode_Repeat\" value=\"1\"/>",
            ),
            CONTRACT_BINDINGS.replace(
                "</enum>\n          </enums>",
                "<value name=\"EzGfxTextureAddressMode_FrontAndBack\" value=\"3\"/></enum>\n          </enums>",
            ),
        ] {
            assert!(contract_error(CONTRACT_HEADER, &bindings).contains("EzGfxTextureAddressMode"));
        }
    }

    #[test]
    fn bindings_reject_missing_fixed_extents_and_unknown_count_references() {
        for (bindings, expected) in [
            (
                CONTRACT_BINDINGS.replace(" array-byte-size=\"4\"", ""),
                "array-byte-size",
            ),
            (
                CONTRACT_BINDINGS.replace(
                    " direction=\"out\" array-length=\"16\"",
                    " direction=\"out\"",
                ),
                "array-length=\"16\"",
            ),
            (
                CONTRACT_BINDINGS.replace("array-length=\"count\"", "array-length=\"missing\""),
                "unknown array-length",
            ),
            (
                CONTRACT_BINDINGS.replace("array-byte-size=\"4\"", "array-byte-size=\"missing\""),
                "unknown array-byte-size",
            ),
        ] {
            assert!(contract_error(CONTRACT_HEADER, &bindings).contains(expected));
        }
    }

    #[test]
    fn bindings_reject_malformed_duplicate_and_invalid_managed_metadata() {
        for (bindings, expected) in [
            (
                CONTRACT_BINDINGS.replace(
                    "</functions>",
                    "<function name=\"ez_gfx_semantic_id\" return=\"EzGfxResult\"/></functions>",
                ),
                "duplicate",
            ),
            (
                CONTRACT_BINDINGS.replace("managed=\"context-create\"", "managed=\"unknown\""),
                "managed override",
            ),
            (
                CONTRACT_BINDINGS.replace("direction=\"out\"", "direction=\"sideways\""),
                "direction",
            ),
            (
                CONTRACT_BINDINGS.replace("array-length=\"16\"", "array-length=\"0\""),
                "positive",
            ),
        ] {
            assert!(contract_error(CONTRACT_HEADER, &bindings).contains(expected));
        }
        assert!(compare_declarations(CONTRACT_HEADER, "<ez-gfx-bindings>").is_err());
    }
}
