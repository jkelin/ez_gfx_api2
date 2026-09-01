use anyhow::{Context, Result, anyhow, bail};
use quick_xml::{
    Reader, XmlVersion,
    events::{BytesStart, Event},
};
type Discriminants = Vec<(String, i64)>;
type EnumDeclaration = (String, Discriminants);
type CurrentEnum = (String, String, Discriminants);
type AccessAnnotations = BTreeMap<usize, (String, Option<usize>)>;
use std::collections::{BTreeMap, BTreeSet};

pub(super) const PUBLIC_PREFIX: &str = "ez_gfx_";

#[derive(Debug, PartialEq, Eq)]
pub(super) struct DeclarationCounts {
    pub(super) functions: usize,
    pub(super) parameters: usize,
    pub(super) pointers: usize,
    pub(super) counted_pointers: usize,
    pub(super) handles: usize,
    pub(super) structs: usize,
    pub(super) fields: usize,
    pub(super) enums: usize,
    pub(super) discriminants: usize,
    pub(super) managed_overrides: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Field {
    name: String,
    ty: String,
    extent: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Parameter {
    name: String,
    ty: String,
    direction: Option<String>,
    array_length: Option<String>,
    array_byte_size: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Function {
    result: String,
    parameters: Vec<Parameter>,
    managed: Option<String>,
}

#[derive(Debug, Default, Clone)]
struct Contract {
    abi_version: u32,
    handles: BTreeMap<String, String>,
    enums: BTreeMap<String, EnumDeclaration>,
    structs: BTreeMap<String, Vec<Field>>,
    functions: BTreeMap<String, Function>,
}

#[derive(Debug)]
pub(super) struct ValidatedAbiContract {
    pub(super) counts: DeclarationCounts,
    pub(super) functions: BTreeSet<String>,
}

pub(super) fn validated_contract(header: &str, bindings: &str) -> Result<ValidatedAbiContract> {
    let header = parse_header(header)?;
    let mut bindings = parse_bindings(bindings)?;
    let parameters = bindings
        .functions
        .values()
        .map(|function| function.parameters.len())
        .sum();
    let pointers = bindings
        .functions
        .values()
        .flat_map(|function| &function.parameters)
        .filter(|parameter| parameter.ty.contains('*'))
        .count();
    let counted_pointers = bindings
        .functions
        .values()
        .flat_map(|function| &function.parameters)
        .filter(|parameter| parameter.array_length.is_some())
        .count();
    let managed_overrides = bindings
        .functions
        .values()
        .filter(|function| function.managed.is_some())
        .count();
    normalize_binding_overrides(&mut bindings);
    let mut failures = Vec::new();

    if header.abi_version != bindings.abi_version {
        failures.push(format!(
            "ABI version differs: header {}, bindings {}",
            header.abi_version, bindings.abi_version
        ));
    }
    compare_maps("handle", &header.handles, &bindings.handles, &mut failures);
    compare_maps("enum", &header.enums, &bindings.enums, &mut failures);
    compare_maps("struct", &header.structs, &bindings.structs, &mut failures);
    compare_maps(
        "function",
        &header.functions,
        &bindings.functions,
        &mut failures,
    );
    if !failures.is_empty() {
        bail!("declaration parity failed:\n{}", failures.join("\n"));
    }

    Ok(ValidatedAbiContract {
        functions: header.functions.into_keys().collect(),
        counts: DeclarationCounts {
            functions: bindings.functions.len(),
            parameters,
            pointers,
            counted_pointers,
            handles: bindings.handles.len(),
            structs: bindings.structs.len(),
            fields: bindings.structs.values().map(Vec::len).sum(),
            enums: bindings.enums.len(),
            discriminants: bindings
                .enums
                .values()
                .map(|(_, values)| values.len())
                .sum(),
            managed_overrides,
        },
    })
}

#[cfg(test)]
pub(super) fn compare_declarations(header: &str, bindings: &str) -> Result<DeclarationCounts> {
    validated_contract(header, bindings).map(|contract| contract.counts)
}

fn compare_maps<T: std::fmt::Debug + PartialEq>(
    kind: &str,
    expected: &BTreeMap<String, T>,
    actual: &BTreeMap<String, T>,
    failures: &mut Vec<String>,
) {
    for (name, expected_value) in expected {
        match actual.get(name) {
            None => failures.push(format!("bindings missing {kind} {name}")),
            Some(actual_value) if actual_value != expected_value => failures.push(format!(
                "{kind} {name} differs: header {expected_value:?}, bindings {actual_value:?}"
            )),
            Some(_) => {}
        }
    }
    for name in actual.keys() {
        if !expected.contains_key(name) {
            failures.push(format!("bindings has extra {kind} {name}"));
        }
    }
}

fn parse_header(input: &str) -> Result<Contract> {
    let source = strip_c_comments(input)?;
    let abi_version = source
        .lines()
        .find_map(|line| line.trim().strip_prefix("#define EZ_GFX_ABI_VERSION "))
        .and_then(parse_unsigned)
        .ok_or_else(|| anyhow!("header has no valid ABI version"))?;
    let mut aliases = BTreeMap::new();
    for statement in source.split(';') {
        let words = statement.split_whitespace().collect::<Vec<_>>();
        if words.len() >= 3
            && words[words.len() - 3] == "typedef"
            && words[words.len() - 2] != "struct"
        {
            aliases.insert(
                words[words.len() - 1].to_owned(),
                normalize_type(words[words.len() - 2]),
            );
        }
    }

    let structs = parse_header_structs(&source)?;
    let enum_values = parse_header_enum_values(&source)?;
    let functions = parse_header_functions(&source)?;
    let mut contract = Contract {
        abi_version,
        structs,
        functions,
        ..Contract::default()
    };
    for (name, underlying) in aliases {
        if let Some(values) = enum_values.get(&name) {
            contract.enums.insert(name, (underlying, values.clone()));
        } else if underlying == "uint64_t" && name.starts_with("EzGfx") {
            contract.handles.insert(name, underlying);
        }
    }
    if contract.functions.is_empty() {
        bail!("header contains no public function declarations");
    }
    Ok(contract)
}

fn parse_header_structs(source: &str) -> Result<BTreeMap<String, Vec<Field>>> {
    let mut structs = BTreeMap::new();
    let mut remaining = source;
    while let Some(relative) = remaining.find("typedef struct ") {
        remaining = &remaining[relative + "typedef struct ".len()..];
        let open = remaining
            .find('{')
            .ok_or_else(|| anyhow!("malformed header struct"))?;
        let name = remaining[..open].trim();
        let close = remaining[open + 1..]
            .find('}')
            .ok_or_else(|| anyhow!("malformed header struct {name}"))?
            + open
            + 1;
        let mut fields = Vec::new();
        for declaration in remaining[open + 1..close]
            .split(';')
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            let declaration = declaration.trim();
            let (base, extent) = if let Some(open) = declaration.rfind('[') {
                let close = declaration[open..]
                    .find(']')
                    .ok_or_else(|| anyhow!("malformed array field in {name}"))?
                    + open;
                (
                    &declaration[..open],
                    Some(declaration[open + 1..close].trim().to_owned()),
                )
            } else {
                (declaration, None)
            };
            let (ty, field_name) =
                split_c_name(base).with_context(|| format!("parse field in {name}"))?;
            fields.push(Field {
                name: field_name,
                ty,
                extent,
            });
        }
        if structs.insert(name.to_owned(), fields).is_some() {
            bail!("duplicate header struct: {name}");
        }
        remaining = &remaining[close + 1..];
    }
    Ok(structs)
}

fn parse_header_enum_values(source: &str) -> Result<BTreeMap<String, Discriminants>> {
    let mut all_values = BTreeMap::new();
    let mut remaining = source;
    while let Some(relative) = remaining.find("enum {") {
        remaining = &remaining[relative + "enum {".len()..];
        let close = remaining
            .find('}')
            .ok_or_else(|| anyhow!("malformed header enum"))?;
        let mut values = Vec::new();
        let mut enum_name = None;
        for item in remaining[..close]
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            let (name, value) = item
                .split_once('=')
                .ok_or_else(|| anyhow!("malformed header enum value: {item}"))?;
            let name = name.trim();
            let prefix = name
                .rsplit_once('_')
                .map(|(prefix, _)| prefix)
                .ok_or_else(|| anyhow!("malformed header enum name: {name}"))?;
            if enum_name
                .as_deref()
                .is_some_and(|existing| existing != prefix)
            {
                bail!("mixed header enum prefixes");
            }
            enum_name = Some(prefix.to_owned());
            values.push((name.to_owned(), parse_signed(value.trim())?));
        }
        if let Some(name) = enum_name {
            if all_values.insert(name.clone(), values).is_some() {
                bail!("duplicate header enum: {name}");
            }
        }
        remaining = &remaining[close + 1..];
    }
    // Numeric public defines are enum discriminants too (used for platform-gated values).
    for line in source.lines().map(str::trim) {
        let Some(rest) = line.strip_prefix("#define EzGfx") else {
            continue;
        };
        let Some((suffix, raw_value)) = rest.split_once(char::is_whitespace) else {
            continue;
        };
        let name = format!("EzGfx{suffix}");
        let Some((prefix, _)) = name.rsplit_once('_') else {
            continue;
        };
        if let Some(values) = all_values.get_mut(prefix) {
            let value = parse_signed(raw_value.trim())?;
            if values.iter().any(|(existing, _)| existing == &name) {
                bail!("duplicate header enum discriminant: {name}");
            }
            values.push((name, value));
        }
    }
    Ok(all_values)
}

fn parse_header_functions(source: &str) -> Result<BTreeMap<String, Function>> {
    let mut functions = BTreeMap::new();
    for statement in source.split(';') {
        let Some(name_offset) = statement.find(PUBLIC_PREFIX) else {
            continue;
        };
        let declaration_start = statement[..name_offset]
            .rfind(['\n', '}'])
            .map_or(0, |offset| offset + 1);
        let result = normalize_type(statement[declaration_start..name_offset].trim());
        if result.is_empty() || result.starts_with('#') {
            continue;
        }
        let name_end = statement[name_offset..]
            .find('(')
            .ok_or_else(|| anyhow!("malformed header function"))?
            + name_offset;
        let name = statement[name_offset..name_end].trim().to_owned();
        // Empty or punctuated public suffixes are malformed ABI names, not export candidates.
        if name.len() == PUBLIC_PREFIX.len() || !name.bytes().all(is_identifier_byte) {
            bail!("invalid header function name: {name}");
        }
        let params_end = find_matching_paren(statement, name_end)?;
        let raw_params = &statement[name_end + 1..params_end];
        let access = parse_access_annotations(&statement[params_end + 1..])?;
        let mut parameters = Vec::new();
        if raw_params.trim() != "void" && !raw_params.trim().is_empty() {
            for (index, raw) in raw_params.split(',').enumerate() {
                let (ty, param_name) = split_c_name(raw)?;
                let access = access.get(&(index + 1));
                let direction = if ty.contains('*') {
                    access
                        .map(|(direction, _)| direction.clone())
                        .or_else(|| ty.starts_with("const ").then(|| "in".to_owned()))
                } else {
                    None
                };
                let array_length =
                    access.and_then(|(_, count)| count.map(|count| count.to_string()));
                parameters.push(Parameter {
                    name: param_name,
                    ty,
                    direction,
                    array_length,
                    array_byte_size: None,
                });
            }
        }
        // Header access counts are one-based parameter positions; bind them to names.
        let parameter_names = parameters
            .iter()
            .map(|parameter| parameter.name.clone())
            .collect::<Vec<_>>();
        for parameter in &mut parameters {
            if let Some(raw) = parameter.array_length.take() {
                let index = raw.parse::<usize>().context("parse header count index")?;
                parameter.array_length = parameter_names.get(index - 1).cloned();
            }
        }
        if functions
            .insert(
                name.clone(),
                Function {
                    result,
                    parameters,
                    managed: None,
                },
            )
            .is_some()
        {
            bail!("duplicate header function: {name}");
        }
    }
    Ok(functions)
}

fn parse_access_annotations(input: &str) -> Result<AccessAnnotations> {
    let mut access = BTreeMap::new();
    let mut remaining = input;
    while let Some(relative) = remaining.find("EZ_GFX_ACCESS(") {
        remaining = &remaining[relative + "EZ_GFX_ACCESS(".len()..];
        let close = remaining
            .find(')')
            .ok_or_else(|| anyhow!("malformed EZ_GFX_ACCESS"))?;
        let parts = remaining[..close]
            .split(',')
            .map(str::trim)
            .collect::<Vec<_>>();
        if !(2..=3).contains(&parts.len()) {
            bail!("malformed EZ_GFX_ACCESS arguments");
        }
        let direction = match parts[0] {
            "read_only" => "in",
            "write_only" => "out",
            other => bail!("unsupported access direction: {other}"),
        };
        let position = parts[1].parse::<usize>().context("parse access position")?;
        let count = parts
            .get(2)
            .map(|value| {
                value
                    .parse::<usize>()
                    .context("parse access count position")
            })
            .transpose()?;
        if access
            .insert(position, (direction.to_owned(), count))
            .is_some()
        {
            bail!("duplicate access annotation for parameter {position}");
        }
        remaining = &remaining[close + 1..];
    }
    Ok(access)
}

fn parse_bindings(input: &str) -> Result<Contract> {
    let mut reader = Reader::from_str(input);
    reader.config_mut().trim_text(true);
    let mut contract = Contract::default();
    let mut sections = BTreeSet::new();
    let mut section = None;
    let mut current_enum: Option<CurrentEnum> = None;
    let mut current_struct: Option<(String, Vec<Field>)> = None;
    let mut current_function: Option<(String, Function)> = None;
    loop {
        match reader.read_event().context("parse bindings XML")? {
            Event::Start(element) => match element.name().as_ref() {
                b"ez-gfx-bindings" => {
                    contract.abi_version = required_attr(&reader, &element, "abi-version")?
                        .parse()
                        .context("parse bindings ABI version")?;
                }
                b"handles" | b"enums" | b"structs" | b"functions" => {
                    let name = String::from_utf8_lossy(element.name().as_ref()).into_owned();
                    if !sections.insert(name.clone()) {
                        bail!("bindings XML contains duplicate {name} sections");
                    }
                    section = Some(name);
                }
                b"enum" if section.as_deref() == Some("enums") => {
                    current_enum = Some((
                        required_attr(&reader, &element, "name")?,
                        normalize_type(&required_attr(&reader, &element, "underlying")?),
                        Vec::new(),
                    ));
                }
                b"struct" if section.as_deref() == Some("structs") => {
                    current_struct = Some((required_attr(&reader, &element, "name")?, Vec::new()));
                }
                b"function" if section.as_deref() == Some("functions") => {
                    current_function = Some((
                        required_attr(&reader, &element, "name")?,
                        Function {
                            result: normalize_type(&required_attr(&reader, &element, "return")?),
                            parameters: Vec::new(),
                            managed: optional_attr(&reader, &element, "managed")?,
                        },
                    ));
                }
                _ => parse_binding_empty(
                    &reader,
                    &element,
                    section.as_deref(),
                    &mut contract,
                    &mut current_enum,
                    &mut current_struct,
                    &mut current_function,
                )?,
            },
            Event::Empty(element) => parse_binding_empty(
                &reader,
                &element,
                section.as_deref(),
                &mut contract,
                &mut current_enum,
                &mut current_struct,
                &mut current_function,
            )?,
            Event::End(element) => match element.name().as_ref() {
                b"enum" => {
                    let (name, underlying, values) = current_enum
                        .take()
                        .ok_or_else(|| anyhow!("unexpected enum end"))?;
                    insert_unique(&mut contract.enums, name, (underlying, values), "enum")?;
                }
                b"struct" => {
                    let (name, fields) = current_struct
                        .take()
                        .ok_or_else(|| anyhow!("unexpected struct end"))?;
                    insert_unique(&mut contract.structs, name, fields, "struct")?;
                }
                b"function" => {
                    let (name, mut function) = current_function
                        .take()
                        .ok_or_else(|| anyhow!("unexpected function end"))?;
                    validate_function_metadata(&name, &mut function)?;
                    insert_unique(&mut contract.functions, name, function, "function")?;
                }
                b"handles" | b"enums" | b"structs" | b"functions" => section = None,
                _ => {}
            },
            Event::Eof => break,
            _ => {}
        }
    }
    if contract.abi_version == 0
        || sections.len() != 4
        || current_enum.is_some()
        || current_struct.is_some()
        || current_function.is_some()
    {
        bail!("bindings XML is incomplete");
    }
    Ok(contract)
}

#[allow(
    clippy::too_many_arguments,
    reason = "the XML event parser updates each declaration context in one pass"
)]
fn parse_binding_empty(
    reader: &Reader<&[u8]>,
    element: &BytesStart<'_>,
    section: Option<&str>,
    contract: &mut Contract,
    current_enum: &mut Option<CurrentEnum>,
    current_struct: &mut Option<(String, Vec<Field>)>,
    current_function: &mut Option<(String, Function)>,
) -> Result<()> {
    match element.name().as_ref() {
        b"function" if section == Some("functions") => {
            let name = required_attr(reader, element, "name")?;
            if current_function.is_some() || contract.functions.contains_key(&name) {
                bail!("duplicate bindings function: {name}");
            }
            let mut function = Function {
                result: normalize_type(&required_attr(reader, element, "return")?),
                parameters: Vec::new(),
                managed: optional_attr(reader, element, "managed")?,
            };
            validate_function_metadata(&name, &mut function)?;
            insert_unique(&mut contract.functions, name, function, "function")?;
        }
        b"handle" if section == Some("handles") => {
            let name = required_attr(reader, element, "name")?;
            insert_unique(&mut contract.handles, name, "uint64_t".to_owned(), "handle")?;
        }
        b"value" if current_enum.is_some() => {
            let name = required_attr(reader, element, "name")?;
            let value = parse_signed(&required_attr(reader, element, "value")?)?;
            let values = &mut current_enum.as_mut().unwrap().2;
            if values.iter().any(|(existing, _)| existing == &name) {
                bail!("duplicate enum discriminant: {name}");
            }
            values.push((name, value));
        }
        b"field" if current_struct.is_some() => {
            let struct_name = &current_struct.as_ref().unwrap().0;
            let field_name = required_attr(reader, element, "name")?;
            // `counted_by` is valid only on C99 flexible array members, never pointer fields.
            if optional_attr(reader, element, "counted-by")?.is_some() {
                bail!("counted-by is invalid on struct field {struct_name}.{field_name}");
            }
            let field = Field {
                name: field_name,
                ty: normalize_type(&required_attr(reader, element, "type")?),
                extent: optional_attr(reader, element, "array-length")?,
            };
            let fields = &mut current_struct.as_mut().unwrap().1;
            if fields.iter().any(|existing| existing.name == field.name) {
                bail!("duplicate struct field: {}", field.name);
            }
            fields.push(field);
        }
        b"param" if current_function.is_some() => {
            let parameter_type = normalize_type(&required_attr(reader, element, "type")?);
            let direction = optional_attr(reader, element, "direction")?.or_else(|| {
                parameter_type
                    .starts_with("const ")
                    .then(|| "in".to_owned())
            });
            let parameter = Parameter {
                name: required_attr(reader, element, "name")?,
                ty: parameter_type,
                direction,
                array_length: optional_attr(reader, element, "array-length")?,
                array_byte_size: optional_attr(reader, element, "array-byte-size")?,
            };
            current_function
                .as_mut()
                .unwrap()
                .1
                .parameters
                .push(parameter);
        }
        _ => {}
    }
    Ok(())
}

fn validate_function_metadata(name: &str, function: &mut Function) -> Result<()> {
    const MANAGED: [&str; 4] = [
        "context-create",
        "surface-create",
        "utf8-string",
        "structured-write",
    ];
    if function
        .managed
        .as_deref()
        .is_some_and(|value| !MANAGED.contains(&value))
    {
        bail!("invalid managed override on {name}");
    }
    let names = function
        .parameters
        .iter()
        .map(|parameter| parameter.name.as_str())
        .collect::<BTreeSet<_>>();
    if names.len() != function.parameters.len() {
        bail!("duplicate parameter in {name}");
    }
    for parameter in &function.parameters {
        if parameter
            .direction
            .as_deref()
            .is_some_and(|value| value != "in" && value != "out")
        {
            bail!("invalid direction on {name}.{}", parameter.name);
        }
        if !parameter.ty.contains('*')
            && (parameter.direction.is_some()
                || parameter.array_length.is_some()
                || parameter.array_byte_size.is_some())
        {
            bail!("pointer metadata on non-pointer {name}.{}", parameter.name);
        }
        for (kind, reference) in [
            ("array-length", parameter.array_length.as_deref()),
            ("array-byte-size", parameter.array_byte_size.as_deref()),
        ] {
            if let Some(reference) = reference {
                if let Ok(value) = reference.parse::<u64>() {
                    if value == 0 {
                        bail!("{kind} must be positive on {name}.{}", parameter.name);
                    }
                } else if !names.contains(reference) {
                    bail!(
                        "unknown {kind} reference {reference} on {name}.{}",
                        parameter.name
                    );
                }
            }
        }
    }
    match name {
        "ez_gfx_vertex_upload_indices"
            if function
                .parameters
                .first()
                .and_then(|parameter| parameter.array_byte_size.as_deref())
                != Some("4") =>
        {
            bail!("ez_gfx_vertex_upload_indices data requires array-byte-size=\"4\"")
        }
        "ez_gfx_semantic_id"
            if function
                .parameters
                .get(2)
                .and_then(|parameter| parameter.array_length.as_deref())
                != Some("16") =>
        {
            bail!("ez_gfx_semantic_id out_id requires array-length=\"16\"")
        }
        _ => {}
    }
    Ok(())
}

fn normalize_binding_overrides(contract: &mut Contract) {
    // Fixed extents and byte-size overrides are managed-binding invariants, not C syntax.
    for function in contract.functions.values_mut() {
        for parameter in &mut function.parameters {
            if parameter
                .array_length
                .as_deref()
                .is_some_and(|value| value.parse::<u64>().is_ok())
            {
                parameter.array_length = None;
            }
            parameter.array_byte_size = None;
        }
        function.managed = None;
    }
}

fn required_attr(reader: &Reader<&[u8]>, element: &BytesStart<'_>, name: &str) -> Result<String> {
    optional_attr(reader, element, name)?.ok_or_else(|| {
        anyhow!(
            "{} is missing {name}",
            String::from_utf8_lossy(element.name().as_ref())
        )
    })
}

fn optional_attr(
    reader: &Reader<&[u8]>,
    element: &BytesStart<'_>,
    name: &str,
) -> Result<Option<String>> {
    let mut value = None;
    for attribute in element.attributes() {
        let attribute = attribute.context("parse bindings attribute")?;
        if attribute.key.as_ref() == name.as_bytes() {
            if value.is_some() {
                bail!("duplicate {name} attribute");
            }
            value = Some(
                attribute
                    .decoded_and_normalized_value(XmlVersion::Implicit1_0, reader.decoder())
                    .context("decode bindings attribute")?
                    .into_owned(),
            );
        }
    }
    Ok(value)
}

fn insert_unique<T>(
    map: &mut BTreeMap<String, T>,
    name: String,
    value: T,
    kind: &str,
) -> Result<()> {
    match map.entry(name) {
        std::collections::btree_map::Entry::Vacant(entry) => {
            entry.insert(value);
            Ok(())
        }
        std::collections::btree_map::Entry::Occupied(entry) => {
            bail!("duplicate bindings {kind}: {}", entry.key())
        }
    }
}

fn split_c_name(input: &str) -> Result<(String, String)> {
    let input = input.trim();
    let end = input.trim_end().len();
    let start = input[..end]
        .rfind(|character: char| !(character.is_ascii_alphanumeric() || character == '_'))
        .map_or(0, |offset| offset + 1);
    let name = input[start..end].to_owned();
    if name.is_empty() || !name.bytes().all(is_identifier_byte) {
        bail!("malformed C declaration: {input}");
    }
    Ok((normalize_type(input[..start].trim()), name))
}

// ABI identifiers permit only ASCII alphanumeric bytes and underscores.
pub(super) fn is_identifier_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

// Unterminated block comments are malformed input; line comments end at EOF or a newline.
fn strip_c_comments(input: &str) -> Result<String> {
    let mut output = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(current) = chars.next() {
        if current != '/' {
            output.push(current);
            continue;
        }
        match chars.peek() {
            Some('*') => {
                chars.next();
                let mut closed = false;
                while let Some(comment) = chars.next() {
                    if comment == '*' && chars.peek() == Some(&'/') {
                        chars.next();
                        closed = true;
                        break;
                    }
                }
                if !closed {
                    bail!("unterminated C block comment");
                }
                output.push(' ');
            }
            Some('/') => {
                chars.next();
                for comment in chars.by_ref() {
                    if comment == '\n' {
                        output.push('\n');
                        break;
                    }
                }
            }
            _ => output.push(current),
        }
    }
    Ok(output)
}

fn normalize_type(input: &str) -> String {
    input
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace(" *", "*")
        .replace('*', " *")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn find_matching_paren(input: &str, open: usize) -> Result<usize> {
    let mut depth = 0;
    for (offset, byte) in input.as_bytes()[open..].iter().enumerate() {
        match byte {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Ok(open + offset);
                }
            }
            _ => {}
        }
    }
    bail!("unterminated function parameter list")
}

fn parse_unsigned(input: &str) -> Option<u32> {
    input.trim().trim_end_matches(['u', 'U']).parse().ok()
}

fn parse_signed(input: &str) -> Result<i64> {
    input
        .trim()
        .trim_end_matches(['u', 'U', 'l', 'L'])
        .parse()
        .with_context(|| format!("parse integer {input}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bindings_reject_counted_by_on_struct_pointer_fields() {
        let bindings = r#"
            <ez-gfx-bindings abi-version="18">
              <handles></handles>
              <enums></enums>
              <structs>
                <struct name="EzGfxByteBuffer">
                  <field name="length" type="size_t"/>
                  <field name="data" type="const uint8_t *" counted-by="length"/>
                </struct>
              </structs>
              <functions></functions>
            </ez-gfx-bindings>
        "#;

        assert!(
            parse_bindings(bindings)
                .unwrap_err()
                .to_string()
                .contains("counted-by is invalid on struct field EzGfxByteBuffer.data")
        );
    }

    #[test]
    fn generated_header_matches_portable_bindings_contract() {
        validated_contract(
            include_str!("../../../include/ez_gfx_api.h"),
            include_str!("../../../bindings/bindings.xml"),
        )
        .unwrap();
    }
}
