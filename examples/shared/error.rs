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
