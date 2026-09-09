//! ez-gfx binding generator.

use anyhow::Result;
use clap::{Parser, Subcommand};
use ez_gfx_bindgen::{repository_root, write_generated};

#[derive(Parser)]
#[command(
    name = "bindgen",
    about = "Generate ez-gfx foreign-language bindings",
    arg_required_else_help = true
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate `bindings/bindings.xml` from ez-gfx-ffi Rust declarations.
    RustToXml,
    /// Generate `bindings/c/include/ez_gfx_api.h` from `bindings.xml`.
    XmlToC,
    /// Generate both outputs in dependency order.
    All,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("bindgen failed: {error:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let root = repository_root();
    match Cli::parse().command {
        Command::RustToXml => write_generated(&root, true, false),
        Command::XmlToC => write_generated(&root, false, true),
        Command::All => write_generated(&root, true, true),
    }
}
