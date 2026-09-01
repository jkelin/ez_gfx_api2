//! Shader target parsing tests.

use ez_gfx_compiler::Target;
use std::str::FromStr;

#[test]
fn target_names_are_stable() {
    for (name, target) in [
        ("spirv", Target::Spirv),
        ("dxil", Target::Dxil),
        ("metal", Target::Metal),
    ] {
        assert_eq!(Target::from_str(name).unwrap(), target);
        assert_eq!(target.to_string(), name);
    }
}

#[test]
fn implementation_target_names_are_rejected() {
    assert!(Target::from_str("msl").is_err());
    assert!(Target::from_str("metallib").is_err());
    assert!(Target::from_str("").is_err());
}
