# Publish all distributable crates to crates.io in dependency order.
publish:
    cargo publish --package ez-gfx-core
    cargo publish --package ez-gfx-artifact
    cargo publish --package ez-gfx-assets
    cargo publish --package ez-gfx-hal
    cargo publish --package ez-gfx-compiler
    cargo publish --package ez-gfx-geometry-manager
    cargo publish --package ez-gfx-texture-manager
    cargo publish --package ez-gfx-runtime
    cargo publish --package ez-gfx-backend-vulkan
    cargo publish --package ez-gfx-backend-dx12
    cargo publish --package ez-gfx-backend-metal
    cargo publish --package ez-gfx
    cargo publish --package ez-gfx-ffi
