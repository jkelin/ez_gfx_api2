use super::*;

pub(super) struct Decoder(pub(super) u8);

impl Decoder {
    pub(super) fn register(id: u8, format: TextureFormat, width: u32, height: u32) -> Self {
        // These fixtures contain exactly two block columns and one (possibly clipped) block row.
        assert!((5..=8).contains(&width) && (1..=4).contains(&height));
        Context::register_texture_decoder(
            id,
            Arc::new(move |bytes, _| {
                // Reject malformed custom payloads instead of padding missing compressed blocks.
                let expected = match format {
                    TextureFormat::Bc1Unorm | TextureFormat::Bc1Srgb => 16,
                    TextureFormat::Rgba8Unorm => width as usize * height as usize * 4,
                    _ => 32,
                };
                if bytes.len() != expected {
                    return Err(TextureError::InvalidData);
                }
                Ok(DecodedTexture {
                    width,
                    height,
                    mip_count: 1,
                    format,
                    mips: vec![DecodedMip {
                        width,
                        height,
                        bytes: bytes.to_vec(),
                    }],
                })
            }),
        )
        .unwrap();
        Self(id)
    }

    pub(super) fn register_chain(id: u8, format: TextureFormat) -> Self {
        // Three real compressed mips use ceil-divided block counts, including clipped mip2 edges.
        let block_bytes = solid_block(format, 0).len();
        Context::register_texture_decoder(
            id,
            Arc::new(move |bytes, _| {
                if bytes.len() != (21 + 8 + 2) * block_bytes {
                    return Err(TextureError::InvalidData);
                }
                let mut offset = 0;
                let mips = [(28, 12, 21), (14, 6, 8), (7, 3, 2)]
                    .into_iter()
                    .map(|(width, height, blocks)| {
                        let end = offset + blocks * block_bytes;
                        let mip = DecodedMip {
                            width,
                            height,
                            bytes: bytes[offset..end].to_vec(),
                        };
                        offset = end;
                        mip
                    })
                    .collect();
                Ok(DecodedTexture {
                    width: 28,
                    height: 12,
                    mip_count: 3,
                    format,
                    mips,
                })
            }),
        )
        .unwrap();
        Self(id)
    }

    pub(super) fn source(&self) -> TextureSource {
        TextureSource::Custom(self.0)
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // All loads copy/retain their callback at admission; unregister cannot invalidate work.
        Context::unregister_texture_decoder(self.0).unwrap();
    }
}
