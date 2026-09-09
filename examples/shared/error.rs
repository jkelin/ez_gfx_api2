use std::num::TryFromIntError;

/// Errors produced by the shared example host and data helpers.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    Cli(#[from] clap::Error),
    #[error(transparent)]
    Environment(#[from] std::env::VarError),
    #[error(transparent)]
    IntegerConversion(#[from] TryFromIntError),
    #[error(transparent)]
    IntegerParsing(#[from] std::num::ParseIntError),
    #[error(transparent)]
    Gltf(#[from] gltf::Error),
    #[error(transparent)]
    Image(#[from] image::ImageError),
    #[error("update snapshot `{path}`: {source}")]
    SnapshotUpdate {
        path: String,
        #[source]
        source: image::ImageError,
    },
    #[error("open snapshot `{path}`: {source}")]
    SnapshotOpen {
        path: String,
        #[source]
        source: image::ImageError,
    },
    #[error(
        "snapshot dimensions differ: expected_path={path} expected={expected_width}x{expected_height} actual_path=<captured frame> actual={actual_width}x{actual_height}"
    )]
    SnapshotDimensions {
        path: String,
        expected_width: u32,
        expected_height: u32,
        actual_width: u32,
        actual_height: u32,
    },
    #[error(
        "snapshot pixels differ: expected_path={path} expected_dimensions={expected_width}x{expected_height} expected_bytes={expected_len} expected_blake3={expected_hash} actual_path=<captured frame> actual_dimensions={actual_width}x{actual_height} actual_bytes={actual_len} actual_blake3={actual_hash} first_difference_index={first_difference_index} expected_byte={expected_byte:?} actual_byte={actual_byte:?}"
    )]
    SnapshotPixels {
        path: String,
        expected_width: u32,
        expected_height: u32,
        expected_len: usize,
        expected_hash: blake3::Hash,
        actual_width: u32,
        actual_height: u32,
        actual_len: usize,
        actual_hash: blake3::Hash,
        first_difference_index: usize,
        expected_byte: Option<u8>,
        actual_byte: Option<u8>,
    },
    #[error("write example report: {0}")]
    ReportOutput(#[source] std::io::Error),
    #[error(transparent)]
    WindowHandle(#[from] raw_window_handle::HandleError),
    #[error(transparent)]
    EventLoop(#[from] winit::error::EventLoopError),
    #[error(transparent)]
    Window(#[from] winit::error::OsError),
    #[error(transparent)]
    Graphics(#[from] ez_gfx::Error),
    #[error("example callback failed: {0}")]
    Callback(String),
}

impl Error {
    pub fn message(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }

    pub fn callback(error: impl std::fmt::Display) -> Self {
        Self::Callback(error.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
