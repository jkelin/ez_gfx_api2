use super::{Child, Contract, Decl, HEADER_NOTICE};
use anyhow::{Context, Result, anyhow, bail};
use quick_xml::{
    Reader,
    events::{BytesStart, Event},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
};

#[allow(
    clippy::too_many_lines,
    reason = "single-pass state tracking keeps fail-closed XML structure validation atomic"
)]
pub(super) fn parse_xml(input: &str) -> Result<Contract> {
    let mut reader = Reader::from_str(input);
    reader.config_mut().trim_text(true);
    let mut contract = Contract::default();
    let mut root_seen = false;
    let mut root_closed = false;
    let mut description_seen = false;
    let mut in_description = false;
    let mut section = String::new();
    let mut sections = BTreeSet::new();
    let mut current: Option<(String, Decl)> = None;
    let mut declarations = BTreeSet::new();

    loop {
        match reader.read_event().context("parse bindings XML")? {
            Event::Decl(_) | Event::Comment(_) => {}
            Event::Start(element) => {
                let tag = String::from_utf8_lossy(element.name().as_ref()).into_owned();
                match tag.as_str() {
                    "ez-gfx-bindings" if !root_seen && section.is_empty() => {
                        let root_attrs = attrs(&reader, &element)?;
                        if root_attrs.len() != 3
                            || root_attrs.get("version").map(String::as_str) != Some("1")
                            || root_attrs.get("library").map(String::as_str) != Some("ez_gfx_ffi")
                        {
                            bail!("bindings XML root must have version 1 and library ez_gfx_ffi");
                        }
                        contract.abi = root_attrs
                            .get("abi-version")
                            .cloned()
                            .ok_or_else(|| anyhow!("bindings XML root missing abi-version"))?;
                        root_seen = true;
                    }
                    "description"
                        if root_seen && !root_closed && section.is_empty() && !description_seen =>
                    {
                        if !attrs(&reader, &element)?.is_empty() {
                            bail!("bindings description cannot have attributes");
                        }
                        description_seen = true;
                        in_description = true;
                    }
                    "handles" | "enums" | "structs" | "callbacks" | "functions"
                        if root_seen && !root_closed && section.is_empty() && current.is_none() =>
                    {
                        if !attrs(&reader, &element)?.is_empty() {
                            bail!("bindings section {tag} cannot have attributes");
                        }
                        if !sections.insert(tag.clone()) {
                            bail!("duplicate bindings section {tag}");
                        }
                        section = tag;
                    }
                    "enum" | "struct" | "callback" | "function"
                        if expected_item(&section) == Some(tag.as_str()) && current.is_none() =>
                    {
                        current = Some((tag, read_decl(&reader, &element)?));
                    }
                    _ => bail!("unexpected bindings XML start tag {tag}"),
                }
            }
            Event::Empty(element) => {
                let tag = String::from_utf8_lossy(element.name().as_ref()).into_owned();
                match tag.as_str() {
                    "handle" if section == "handles" && current.is_none() => {
                        push_decl(
                            "handle",
                            read_decl(&reader, &element)?,
                            &mut contract,
                            &mut declarations,
                        )?;
                    }
                    "callback" | "function"
                        if expected_item(&section) == Some(tag.as_str()) && current.is_none() =>
                    {
                        push_decl(
                            &tag,
                            read_decl(&reader, &element)?,
                            &mut contract,
                            &mut declarations,
                        )?;
                    }
                    "value" | "field" | "param" => {
                        let (kind, parent) = current
                            .as_mut()
                            .ok_or_else(|| anyhow!("orphan bindings XML {tag}"))?;
                        if expected_child(kind) != Some(tag.as_str()) {
                            bail!("unexpected {tag} in {kind}");
                        }
                        let child = read_child(&reader, &element)?;
                        if parent.children.iter().any(|value| value.name == child.name) {
                            bail!("duplicate {tag} {}", child.name);
                        }
                        parent.children.push(child);
                    }
                    "returns" => {
                        let (kind, parent) = current
                            .as_mut()
                            .ok_or_else(|| anyhow!("orphan bindings XML returns"))?;
                        if kind != "function" || parent.returns_seen {
                            bail!("unexpected or duplicate function returns");
                        }
                        let return_attrs = attrs(&reader, &element)?;
                        if return_attrs.keys().any(|key| key != "doc") {
                            bail!("unknown returns attribute");
                        }
                        parent.returns_seen = true;
                        parent.returns_doc = return_attrs.get("doc").cloned().unwrap_or_default();
                    }
                    _ => bail!("unexpected bindings XML empty tag {tag}"),
                }
            }
            Event::Text(text) => {
                if !in_description && !text.decode()?.trim().is_empty() {
                    bail!("unexpected bindings XML text");
                }
            }
            Event::End(element) => {
                let tag = String::from_utf8_lossy(element.name().as_ref()).into_owned();
                if tag == "description" && in_description {
                    in_description = false;
                } else if tag == section && current.is_none() {
                    section.clear();
                } else if current.as_ref().is_some_and(|(kind, _)| kind == &tag) {
                    let (_, declaration) = current
                        .take()
                        .ok_or_else(|| anyhow!("unexpected {tag} end"))?;
                    push_decl(&tag, declaration, &mut contract, &mut declarations)?;
                } else if tag == "ez-gfx-bindings"
                    && root_seen
                    && section.is_empty()
                    && current.is_none()
                {
                    root_closed = true;
                } else {
                    bail!("unexpected bindings XML end tag {tag}");
                }
            }
            Event::Eof => break,
            _ => bail!("unsupported bindings XML event"),
        }
    }

    let required_sections = ["handles", "enums", "structs", "callbacks", "functions"]
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    if !root_seen
        || !root_closed
        || !description_seen
        || in_description
        || !section.is_empty()
        || current.is_some()
        || sections != required_sections
        || contract.abi.parse::<u32>().is_err()
        || contract.abi == "0"
    {
        bail!("bindings XML is incomplete");
    }
    validate_contract(&contract)?;
    Ok(contract)
}

fn expected_item(section: &str) -> Option<&'static str> {
    match section {
        "enums" => Some("enum"),
        "structs" => Some("struct"),
        "callbacks" => Some("callback"),
        "functions" => Some("function"),
        _ => None,
    }
}

fn expected_child(kind: &str) -> Option<&'static str> {
    match kind {
        "enum" => Some("value"),
        "struct" => Some("field"),
        "callback" | "function" => Some("param"),
        _ => None,
    }
}

fn push_decl(
    kind: &str,
    declaration: Decl,
    contract: &mut Contract,
    names: &mut BTreeSet<String>,
) -> Result<()> {
    if !names.insert(format!("{kind}:{}", declaration.name)) {
        bail!("duplicate {kind} declaration {}", declaration.name);
    }
    match kind {
        "handle" => contract.handles.push(declaration),
        "enum" => contract.enums.push(declaration),
        "struct" => contract.structs.push(declaration),
        "callback" => contract.callbacks.push(declaration),
        "function" => contract.functions.push(declaration),
        _ => bail!("unsupported declaration kind {kind}"),
    }
    Ok(())
}

fn validate_contract(contract: &Contract) -> Result<()> {
    let known_types = contract
        .handles
        .iter()
        .chain(&contract.enums)
        .chain(&contract.structs)
        .chain(&contract.callbacks)
        .map(|declaration| declaration.name.as_str())
        .chain([
            "void", "char", "uint8_t", "uint16_t", "uint32_t", "uint64_t", "int32_t", "float",
            "size_t",
        ])
        .collect::<BTreeSet<_>>();

    for declaration in &contract.handles {
        validate_attrs("handle", &declaration.attrs, &[])?;
    }
    for declaration in &contract.enums {
        validate_attrs("enum", &declaration.attrs, &["underlying"])?;
        require_attr(declaration, "underlying")?;
        if declaration.children.is_empty() {
            bail!("enum {} has no values", declaration.name);
        }
        for child in &declaration.children {
            validate_attrs("enum value", &child.attrs, &[])?;
            child
                .value
                .as_deref()
                .ok_or_else(|| anyhow!("enum value {} missing value", child.name))?
                .parse::<i64>()
                .with_context(|| format!("enum value {} is not an integer", child.name))?;
        }
    }
    for declaration in &contract.structs {
        validate_attrs("struct", &declaration.attrs, &[])?;
        if declaration.children.is_empty() {
            bail!("struct {} has no fields", declaration.name);
        }
        for child in &declaration.children {
            validate_child("field", child, &known_types, &declaration.children)?;
            if let Some(length) = child.attrs.get("array-length") {
                validate_positive_integer(length, "struct array-length")?;
            }
        }
    }
    for declaration in &contract.callbacks {
        validate_callable("callback", declaration, &["return"], &known_types)?;
    }
    for declaration in &contract.functions {
        validate_callable(
            "function",
            declaration,
            &["return", "context", "managed"],
            &known_types,
        )?;
    }
    if contract.functions.is_empty() {
        bail!("bindings XML contains no functions");
    }
    Ok(())
}

fn validate_callable(
    kind: &str,
    declaration: &Decl,
    allowed_attrs: &[&str],
    known_types: &BTreeSet<&str>,
) -> Result<()> {
    validate_attrs(kind, &declaration.attrs, allowed_attrs)?;
    let result = require_attr(declaration, "return")?;
    validate_type(result, known_types)
        .with_context(|| format!("{kind} {} return type", declaration.name))?;
    for child in &declaration.children {
        validate_child("param", child, known_types, &declaration.children)?;
    }
    Ok(())
}

fn validate_child(
    kind: &str,
    child: &Child,
    known_types: &BTreeSet<&str>,
    siblings: &[Child],
) -> Result<()> {
    let allowed = if kind == "field" {
        &["type", "array-length", "nullable", "validation"][..]
    } else {
        &[
            "type",
            "array-length",
            "array-byte-size",
            "direction",
            "nullable",
            "validation",
        ][..]
    };
    validate_attrs(kind, &child.attrs, allowed)?;
    let ty = child
        .attrs
        .get("type")
        .ok_or_else(|| anyhow!("{kind} {} missing type", child.name))?;
    validate_type(ty, known_types)?;
    if let Some(nullable) = child.attrs.get("nullable")
        && !matches!(nullable.as_str(), "true" | "false")
    {
        bail!("{kind} {} has invalid nullable value", child.name);
    }
    if let Some(direction) = child.attrs.get("direction") {
        if !matches!(direction.as_str(), "in" | "out") || !ty.contains('*') {
            bail!("{kind} {} has invalid direction", child.name);
        }
    }
    for attribute in ["array-length", "array-byte-size"] {
        if let Some(length) = child.attrs.get(attribute) {
            if kind != "field" && !ty.contains('*') {
                bail!("{kind} {} has {attribute} on a non-pointer", child.name);
            }
            if let Ok(value) = length.parse::<usize>() {
                if value == 0 {
                    bail!("{kind} {} has zero {attribute}", child.name);
                }
            } else if !siblings
                .iter()
                .any(|candidate| candidate.name == *length && candidate.name != child.name)
            {
                bail!(
                    "{kind} {} has invalid {attribute} reference {length}",
                    child.name
                );
            }
        }
    }
    Ok(())
}

fn validate_attrs(kind: &str, attrs: &BTreeMap<String, String>, allowed: &[&str]) -> Result<()> {
    if let Some(key) = attrs.keys().find(|key| !allowed.contains(&key.as_str())) {
        bail!("{kind} has unknown attribute {key}");
    }
    Ok(())
}

fn require_attr<'a>(declaration: &'a Decl, key: &str) -> Result<&'a str> {
    declaration
        .attrs
        .get(key)
        .map(String::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow!("{} missing {key}", declaration.name))
}

fn validate_positive_integer(value: &str, label: &str) -> Result<()> {
    if value.parse::<usize>().is_err() || value == "0" {
        bail!("{label} must be a positive integer");
    }
    Ok(())
}

fn validate_type(ty: &str, known_types: &BTreeSet<&str>) -> Result<()> {
    let base = ty.replace("const", "").replace('*', "").trim().to_owned();
    if base.is_empty() || !known_types.contains(base.as_str()) {
        bail!("unknown binding type {ty}");
    }
    Ok(())
}
fn read_decl(reader: &Reader<&[u8]>, e: &BytesStart<'_>) -> Result<Decl> {
    let attrs = attrs(reader, e)?;
    let name = attrs
        .get("name")
        .cloned()
        .ok_or_else(|| anyhow!("declaration missing name"))?;
    let doc = attrs.get("doc").cloned().unwrap_or_default();
    Ok(Decl {
        name,
        doc,
        returns_doc: String::new(),
        returns_seen: false,
        attrs: attrs
            .into_iter()
            .filter(|(k, _)| k != "name" && k != "doc")
            .collect(),
        child_attrs: BTreeMap::new(),
        child_order: Vec::new(),
        children: Vec::new(),
    })
}
fn read_child(reader: &Reader<&[u8]>, e: &BytesStart<'_>) -> Result<Child> {
    let attrs = attrs(reader, e)?;
    let name = attrs
        .get("name")
        .cloned()
        .ok_or_else(|| anyhow!("child missing name"))?;
    Ok(Child {
        name,
        doc: attrs.get("doc").cloned().unwrap_or_default(),
        value: attrs.get("value").cloned(),
        attrs: attrs
            .into_iter()
            .filter(|(k, _)| !matches!(k.as_str(), "name" | "doc" | "value"))
            .collect(),
    })
}
#[allow(deprecated, reason = "quick-xml 0.41 compatibility API")]
fn attrs(reader: &Reader<&[u8]>, e: &BytesStart<'_>) -> Result<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    for attr in e.attributes() {
        let attr = attr?;
        let key = String::from_utf8(attr.key.as_ref().to_vec())?;
        let value = attr
            .decode_and_unescape_value(reader.decoder())?
            .into_owned();
        if out.insert(key.clone(), value).is_some() {
            bail!("duplicate attribute {key}")
        }
    }
    Ok(out)
}

pub(super) fn render_header(c: &Contract) -> Result<String> {
    let mut out = String::new();
    writeln!(
        out,
        "{HEADER_NOTICE}\n#ifndef EZ_GFX_API_H\n#define EZ_GFX_API_H\n\n#include <stddef.h>\n#include <stdint.h>\n\n#define EZ_GFX_ABI_VERSION {}u\n\n#if defined(__clang__)\n#  if __has_attribute(access)\n#    define EZ_GFX_ACCESS(...) __attribute__((access(__VA_ARGS__)))\n#  else\n#    define EZ_GFX_ACCESS(...)\n#  endif\n#elif defined(__GNUC__) && (__GNUC__ >= 10)\n#  define EZ_GFX_ACCESS(...) __attribute__((access(__VA_ARGS__)))\n#else\n#  define EZ_GFX_ACCESS(...)\n#endif\n\n/* ABI string contract: each const char* has an explicit byte length, denotes exactly that many UTF-8 bytes without scanning for a terminator, and rejects embedded NUL. */\n\n#ifdef __cplusplus\nextern \"C\" {{\n#endif\n",
        c.abi
    )?;
    for h in &c.handles {
        c_doc(&mut out, &h.doc);
        writeln!(out, "typedef uint64_t {};\n", h.name)?;
    }
    for e in &c.enums {
        emit_enum(&mut out, e)?;
    }
    for s in ordered_structs(c)? {
        emit_struct(&mut out, s)?;
    }
    for cb in &c.callbacks {
        emit_callback(&mut out, cb)?;
    }
    for f in &c.functions {
        emit_function(&mut out, f)?;
    }
    out.push_str("#ifdef __cplusplus\n}\n#endif\n\n#endif\n");
    Ok(out)
}

// Shared Doxygen decision for every documented declaration: full block when
// child or returns docs exist, single line for a lone short body, and nothing
// when undocumented.
fn emit_doc(out: &mut String, name: &str, children: &[Child], body: &str, returns: &str) {
    let items: Vec<(&str, &str)> = children
        .iter()
        .map(|child| (child.name.as_str(), child.doc.as_str()))
        .collect();
    if body.is_empty() && !has_docs(&items) && returns.is_empty() {
        return;
    }
    if has_docs(&items) || !returns.is_empty() || body.contains('\n') {
        c_block(out, name, &items, body, returns);
    } else {
        c_doc(out, body);
    }
}

fn emit_enum(out: &mut String, declaration: &Decl) -> Result<()> {
    emit_doc(
        out,
        &declaration.name,
        &declaration.children,
        &declaration.doc,
        "",
    );
    writeln!(
        out,
        "typedef {} {};\nenum {{",
        declaration
            .attrs
            .get("underlying")
            .ok_or_else(|| anyhow!("enum {} missing underlying", declaration.name))?,
        declaration.name
    )?;
    for value in &declaration.children {
        writeln!(
            out,
            "    {} = {},",
            value.name,
            value
                .value
                .as_deref()
                .ok_or_else(|| anyhow!("enum value missing"))?
        )?;
    }
    writeln!(out, "}};\n")?;
    Ok(())
}

fn emit_struct(out: &mut String, declaration: &Decl) -> Result<()> {
    emit_doc(
        out,
        &declaration.name,
        &declaration.children,
        &declaration.doc,
        "",
    );
    writeln!(out, "typedef struct {} {{", declaration.name)?;
    for field in &declaration.children {
        let ty = field
            .attrs
            .get("type")
            .ok_or_else(|| anyhow!("field missing type"))?;
        if let Some(length) = field.attrs.get("array-length") {
            writeln!(out, "    {ty} {}[{length}];", field.name)?;
        } else {
            writeln!(out, "    {ty} {};", field.name)?;
        }
    }
    writeln!(out, "}} {};\n", declaration.name)?;
    Ok(())
}

fn emit_callback(out: &mut String, declaration: &Decl) -> Result<()> {
    emit_doc(
        out,
        &declaration.name,
        &declaration.children,
        &declaration.doc,
        "",
    );
    write!(
        out,
        "typedef {} (*{})(",
        declaration
            .attrs
            .get("return")
            .map_or("void", String::as_str),
        declaration.name
    )?;
    emit_param_list(
        out,
        &declaration.children,
        "callback parameter missing type",
    )?;
    out.push_str(");\n\n");
    Ok(())
}

fn emit_function(out: &mut String, declaration: &Decl) -> Result<()> {
    emit_doc(
        out,
        &declaration.name,
        &declaration.children,
        &declaration.doc,
        &declaration.returns_doc,
    );
    let result = declaration
        .attrs
        .get("return")
        .ok_or_else(|| anyhow!("function missing return"))?;
    write!(out, "{result} {}(", declaration.name)?;
    emit_param_list(out, &declaration.children, "parameter missing type")?;
    out.push(')');
    emit_access(out, declaration)?;
    out.push_str(";\n\n");
    Ok(())
}

fn emit_param_list(out: &mut String, params: &[Child], missing: &str) -> Result<()> {
    if params.is_empty() {
        out.push_str("void");
        return Ok(());
    }
    for (index, param) in params.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        write!(
            out,
            "{} {}",
            param
                .attrs
                .get("type")
                .ok_or_else(|| anyhow!("{missing}"))?,
            param.name
        )?;
    }
    Ok(())
}

fn emit_access(out: &mut String, declaration: &Decl) -> Result<()> {
    for (index, param) in declaration.children.iter().enumerate() {
        let Some(direction) = param.attrs.get("direction") else {
            continue;
        };
        if !param.attrs.get("type").is_some_and(|ty| ty.contains('*')) {
            continue;
        }
        write!(
            out,
            " EZ_GFX_ACCESS({}, {}",
            if direction == "out" {
                "write_only"
            } else {
                "read_only"
            },
            index + 1
        )?;
        if let Some(length) = param.attrs.get("array-length") {
            if let Some(position) = declaration
                .children
                .iter()
                .position(|candidate| candidate.name == *length)
            {
                write!(out, ", {}", position + 1)?;
            }
        }
        out.push(')');
    }
    Ok(())
}
fn ordered_structs(contract: &Contract) -> Result<Vec<&Decl>> {
    let mut pending = contract.structs.iter().collect::<Vec<_>>();
    let mut emitted = BTreeSet::new();
    let mut ordered = Vec::with_capacity(pending.len());

    while !pending.is_empty() {
        let before = pending.len();
        let mut index = 0;
        while index < pending.len() {
            let declaration = pending[index];
            let ready = declaration.children.iter().all(|field| {
                let ty = field.attrs.get("type").map_or("", String::as_str);
                contract.structs.iter().all(|dependency| {
                    dependency.name == declaration.name
                        || !ty.contains(&dependency.name)
                        || emitted.contains(&dependency.name)
                })
            });
            if ready {
                let declaration = pending.remove(index);
                emitted.insert(declaration.name.clone());
                ordered.push(declaration);
            } else {
                index += 1;
            }
        }
        if pending.len() == before {
            bail!("binding structs contain an unresolved dependency cycle");
        }
    }
    Ok(ordered)
}

fn sanitize_doc(doc: &str) -> String {
    doc.replace("*/", "* /")
}

fn has_docs(items: &[(&str, &str)]) -> bool {
    items.iter().any(|(_, doc)| !doc.is_empty())
}

fn c_doc(out: &mut String, doc: &str) {
    if doc.is_empty() {
        return;
    }
    let clean = sanitize_doc(doc);
    if clean.contains('\n') {
        c_block(out, "", &[], &clean, "");
    } else {
        out.push_str("/** ");
        out.push_str(&clean);
        out.push_str(" */\n");
    }
}

// Deterministic Doxygen block in the historical header style: a `Name:` title,
// `@child: doc` entries, a blank-separated body, and a `Returns:` trailer.
// Empty sections are omitted; callers keep the single-line form when no child
// or returns docs exist.
fn c_block(out: &mut String, title: &str, items: &[(&str, &str)], body: &str, returns: &str) {
    out.push_str("/**\n");
    if !title.is_empty() {
        out.push_str(" * ");
        out.push_str(&sanitize_doc(title));
        out.push_str(":\n");
    }
    let mut documented = false;
    for (name, doc) in items {
        if doc.is_empty() {
            continue;
        }
        out.push_str(" * @");
        out.push_str(name);
        out.push_str(": ");
        out.push_str(&sanitize_doc(doc));
        out.push('\n');
        documented = true;
    }
    let body = sanitize_doc(body);
    if documented && !body.is_empty() {
        out.push_str(" *\n");
    }
    for line in body.lines() {
        if line.trim().is_empty() {
            out.push_str(" *\n");
        } else {
            out.push_str(" * ");
            out.push_str(line);
            out.push('\n');
        }
    }
    let returns = sanitize_doc(returns);
    if !returns.is_empty() {
        if documented || !body.is_empty() {
            out.push_str(" *\n");
        }
        out.push_str(" * Returns: ");
        out.push_str(&returns);
        out.push('\n');
    }
    out.push_str(" */\n");
}

// Minimal contract exercising every documented header position: enum
// values, struct fields, callback params, function params, and returns.
#[cfg(test)]
pub(super) const DOC_XML: &str = concat!(
    "<?xml version=\"1.0\" encoding=\"utf-8\"?>",
    "<ez-gfx-bindings version=\"1\" abi-version=\"33\" library=\"ez_gfx_ffi\">",
    "<description>Test.</description>",
    "<handles><handle name=\"EzGfxTestHandle\" doc=\"Opaque test handle.\"/></handles>",
    "<enums><enum name=\"EzGfxTest\" underlying=\"uint8_t\" doc=\"Test enum.\">",
    "<value name=\"EzGfxTest_Ok\" value=\"0\" doc=\"Success.\"/>",
    "</enum></enums>",
    "<structs><struct name=\"EzGfxTestDesc\" doc=\"Test descriptor.\">",
    "<field name=\"count\" type=\"uint32_t\" doc=\"Entry count.\"/>",
    "</struct></structs>",
    "<callbacks><callback name=\"EzGfxTestCallback\" return=\"void\" doc=\"Test callback.\">",
    "<param name=\"count\" type=\"uint32_t\" doc=\"Entry count.\"/>",
    "</callback></callbacks>",
    "<functions><function name=\"ez_gfx_test\" return=\"uint32_t\" doc=\"Runs the test.\">",
    "<param name=\"desc\" type=\"const EzGfxTestDesc *\" direction=\"in\" doc=\"Borrowed descriptor.\"/>",
    "<param name=\"out_count\" type=\"uint32_t *\" direction=\"out\" doc=\"Receives the count.\"/>",
    "<returns doc=\"Returns the test status.\"/>",
    "</function></functions>",
    "</ez-gfx-bindings>",
);

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_header(input: &str) -> Result<String> {
        render_header(&parse_xml(input)?)
    }

    #[test]
    fn rejects_malformed_binding_contracts() {
        for invalid in [
            DOC_XML.replace("version=\"1\"", "version=\"2\""),
            DOC_XML.replace("<handles>", "<widgets/><handles>"),
            DOC_XML.replace("</handles>", "<handle name=\"EzGfxTestHandle\"/></handles>"),
            DOC_XML.replace("<callbacks>", "<missing-callbacks>"),
            DOC_XML.replace("</callbacks>", "</missing-callbacks>"),
            DOC_XML.replace(" return=\"uint32_t\"", ""),
            DOC_XML.replace(" value=\"0\"", ""),
            DOC_XML.replace(
                "</struct>",
                "<field name=\"count\" type=\"uint32_t\"/></struct>",
            ),
            DOC_XML.replace("direction=\"in\"", "direction=\"sideways\""),
            DOC_XML.replace(
                "direction=\"in\"",
                "direction=\"in\" array-length=\"missing\"",
            ),
        ] {
            assert!(generate_header(&invalid).is_err(), "accepted {invalid}");
        }
    }

    #[test]
    fn rejects_incomplete_xml() {
        assert!(generate_header("<ez-gfx-bindings abi-version=\"1\"/>").is_err());
    }

    #[test]
    fn header_retains_child_and_returns_docs() {
        let header = generate_header(DOC_XML).unwrap();
        for expected in [
            " * EzGfxTest:\n",
            " * @EzGfxTest_Ok: Success.\n",
            " * EzGfxTestDesc:\n",
            " * @count: Entry count.\n",
            " * EzGfxTestCallback:\n",
            " * @count: Entry count.\n",
            " * ez_gfx_test:\n",
            " * @desc: Borrowed descriptor.\n",
            " * @out_count: Receives the count.\n",
            " * Returns: Returns the test status.\n",
        ] {
            assert!(header.contains(expected), "missing {expected:?}");
        }
    }

    #[test]
    fn header_keeps_single_line_without_child_docs() {
        let xml = DOC_XML
            .replace(" doc=\"Success.\"", "")
            .replace(" doc=\"Entry count.\"", "")
            .replace(" doc=\"Borrowed descriptor.\"", "")
            .replace(" doc=\"Receives the count.\"", "")
            .replace("<returns doc=\"Returns the test status.\"/>", "<returns/>");
        let header = generate_header(&xml).unwrap();
        assert!(header.contains("/** Test enum. */\n"));
        assert!(header.contains("/** Runs the test. */\n"));
        assert!(!header.contains(" * @"));
        assert!(!header.contains("Returns:"));
    }
}
