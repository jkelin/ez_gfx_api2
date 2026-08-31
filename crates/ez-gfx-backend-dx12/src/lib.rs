//! Direct3D 12 backend for ez-gfx native resource creation and frame execution.
//!
//! The native implementation is available only for Windows targets.

use ez_gfx_core::{Backend, capability::MAX_BINDLESS_SAMPLED_TEXTURES};

/// Identifies this adapter as the Direct3D 12 backend.
pub const BACKEND: Backend = Backend::Dx12;
/// Reports whether the compilation target is Windows and supports Direct3D 12 object creation.
pub const SUPPORTED_ON_TARGET: bool = cfg!(windows);
/// Maximum number of textures admitted by the shader-visible descriptor heap.
pub const TEXTURE_DESCRIPTOR_CAPACITY: u32 = MAX_BINDLESS_SAMPLED_TEXTURES;

#[cfg(windows)]
/// Windows-only Direct3D 12 implementation.
pub mod native;
