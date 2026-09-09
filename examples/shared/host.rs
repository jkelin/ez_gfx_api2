use clap::ValueEnum;
use ez_gfx::Backend;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendConfig {
    pub backend: Backend,
    pub name: &'static str,
}

/// Builds the backend configuration selected at the process boundary.
pub fn backend_config(backend: Backend) -> BackendConfig {
    BackendConfig {
        backend,
        name: backend_name_for(backend),
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BackendArgument {
    Vulkan,
    Dx12,
    Metal,
}

pub(crate) fn parse_backend(value: Option<&str>) -> crate::shared::Result<Backend> {
    let requested = value
        .map(|value| {
            BackendArgument::from_str(value, false).map_err(|_| {
                crate::shared::Error::message(format!("unsupported EZ_GFX_BACKEND `{value}`"))
            })
        })
        .transpose()?;
    match requested {
        #[cfg(target_vendor = "apple")]
        None | Some(BackendArgument::Metal) => Ok(Backend::Metal),
        #[cfg(not(target_vendor = "apple"))]
        None | Some(BackendArgument::Vulkan) => Ok(Backend::Vulkan),
        #[cfg(windows)]
        Some(BackendArgument::Dx12) => Ok(Backend::Dx12),
        Some(_) => Err(crate::shared::Error::message(format!(
            "unsupported EZ_GFX_BACKEND `{}`",
            value.expect("a rejected backend was explicitly provided")
        ))),
    }
}

pub const fn backend_name_for(backend: Backend) -> &'static str {
    match backend {
        Backend::Vulkan => "Vulkan",
        Backend::Dx12 => "DX12",
        Backend::Metal => "Metal",
    }
}

/// Maps a backend to its clip-space convention.
pub const fn clip_y(backend: Backend) -> crate::shared::math::ClipY {
    match backend {
        Backend::Vulkan => crate::shared::math::ClipY::Vulkan,
        Backend::Dx12 => crate::shared::math::ClipY::Dx12,
        Backend::Metal => crate::shared::math::ClipY::Metal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_defaults_to_host_backend() {
        #[cfg(target_vendor = "apple")]
        assert_eq!(parse_backend(None).unwrap(), Backend::Metal);
        #[cfg(not(target_vendor = "apple"))]
        assert_eq!(parse_backend(None).unwrap(), Backend::Vulkan);
    }

    #[test]
    fn explicit_supported_backends_report_stable_names() {
        #[cfg(not(target_vendor = "apple"))]
        assert_eq!(parse_backend(Some("vulkan")).unwrap(), Backend::Vulkan);
        #[cfg(windows)]
        assert_eq!(parse_backend(Some("dx12")).unwrap(), Backend::Dx12);
        #[cfg(target_vendor = "apple")]
        assert_eq!(parse_backend(Some("metal")).unwrap(), Backend::Metal);
    }

    #[test]
    fn unsupported_backend_is_rejected() {
        assert!(parse_backend(Some("unsupported")).is_err());
        #[cfg(not(target_vendor = "apple"))]
        assert!(parse_backend(Some("metal")).is_err());
        #[cfg(not(windows))]
        assert!(parse_backend(Some("dx12")).is_err());
    }
}
