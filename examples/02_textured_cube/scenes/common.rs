use std::{ffi::CString, mem::size_of_val};

use ez_gfx_ffi::{
    EzGfxBinding, EzGfxDrawIndexedCommand, EzGfxDynamicState, EzGfxResult, EzGfxShaderEntry,
    EzGfxTextureDesc, ez_gfx_acquire_indirect, ez_gfx_index_heap_create, ez_gfx_index_heap_destroy,
    ez_gfx_indirect_release, ez_gfx_indirect_set_draw_count, ez_gfx_indirect_write_draw,
    ez_gfx_render_add_vertex_pipeline, ez_gfx_shader_destroy, ez_gfx_shader_load_artifact,
    ez_gfx_structured_acquire, ez_gfx_structured_release, ez_gfx_structured_write,
    ez_gfx_texture_get_binding, ez_gfx_texture_load, ez_gfx_texture_unload,
    ez_gfx_vertex_upload_indices,
};

pub fn status(result: EzGfxResult, operation: &str) -> Result<(), String> {
    match result {
        EzGfxResult::Ok => Ok(()),
        error => Err(format!("{operation}: {error:?}")),
    }
}

pub struct Shader(pub u64);

impl Shader {
    pub fn load(context: u64, artifact: &[u8], compute: bool) -> Result<Self, String> {
        let vertex = CString::new("vertexmain").unwrap();
        let fragment = CString::new("fragmentmain").unwrap();
        let compute_entry = CString::new("computemain").unwrap();
        let mut entries = vec![
            EzGfxShaderEntry {
                entry: vertex.as_ptr(),
                stage: 1,
                _padding: [0; 7],
            },
            EzGfxShaderEntry {
                entry: fragment.as_ptr(),
                stage: 2,
                _padding: [0; 7],
            },
        ];
        if compute {
            entries.push(EzGfxShaderEntry {
                entry: compute_entry.as_ptr(),
                stage: 3,
                _padding: [0; 7],
            });
        }
        let mut handle = 0;
        status(
            {
                // SAFETY: Artifact bytes, aligned entries, NUL-terminated entry names, and output storage remain live for this call.
                unsafe {
                    ez_gfx_shader_load_artifact(
                        artifact.as_ptr(),
                        artifact.len(),
                        entries.as_ptr(),
                        entries.len(),
                        &raw mut handle,
                        context,
                    )
                }
            },
            "load scene artifact",
        )?;
        Ok(Self(handle))
    }

    pub fn destroy(self, context: u64) {
        ez_gfx_shader_destroy(self.0, context);
    }
}

pub struct Structured(pub u64);

impl Structured {
    pub fn upload<T>(context: u64, name: &str, values: &[T]) -> Result<Self, String> {
        if values.is_empty() {
            return Err(format!("{name} cannot be empty"));
        }
        let count = u32::try_from(values.len()).map_err(|_| format!("{name} count exceeds ABI"))?;
        let element_size = u32::try_from(std::mem::size_of::<T>())
            .map_err(|_| format!("{name} stride exceeds ABI"))?;
        let name = CString::new(name).map_err(|_| "buffer name contains NUL".to_owned())?;
        let mut handle = 0;
        status(
            {
                // SAFETY: The NUL-terminated name and aligned output storage remain live for this call.
                unsafe {
                    ez_gfx_structured_acquire(
                        element_size,
                        count,
                        name.as_ptr(),
                        &raw mut handle,
                        context,
                    )
                }
            },
            "acquire structured buffer",
        )?;
        if let Err(error) = status(
            {
                // SAFETY: The data pointer references a live byte range of the declared size for this call.
                unsafe {
                    ez_gfx_structured_write(
                        handle,
                        values.as_ptr().cast(),
                        size_of_val(values) as u64,
                        context,
                    )
                }
            },
            "upload structured buffer",
        ) {
            ez_gfx_structured_release(handle, context);
            return Err(error);
        }
        Ok(Self(handle))
    }

    pub fn destroy(self, context: u64) {
        ez_gfx_structured_release(self.0, context);
    }
}

pub struct Indirect {
    pub handle: u64,
}

impl Indirect {
    pub fn create(context: u64, capacity: u32, name: &str) -> Result<Self, String> {
        if capacity == 0 {
            return Err("indirect capacity cannot be zero".to_owned());
        }
        let name = CString::new(name).map_err(|_| "indirect name contains NUL".to_owned())?;
        let mut handle = 0;
        status(
            {
                // SAFETY: The name is a live NUL-terminated string and the output is live aligned storage for this call.
                unsafe {
                    ez_gfx_acquire_indirect(capacity, name.as_ptr(), &raw mut handle, context)
                }
            },
            "acquire indirect buffer",
        )?;
        Ok(Self { handle })
    }

    pub fn write(
        &self,
        context: u64,
        index: u32,
        command: EzGfxDrawIndexedCommand,
    ) -> Result<(), String> {
        status(
            {
                // SAFETY: The command pointer references one live, aligned draw command for this call.
                unsafe {
                    ez_gfx_indirect_write_draw(self.handle, index, &raw const command, context)
                }
            },
            "write indirect draw",
        )
    }

    pub fn set_count(&self, context: u64, count: u32) -> Result<(), String> {
        status(
            ez_gfx_indirect_set_draw_count(self.handle, count, context),
            "set indirect draw count",
        )
    }

    pub fn destroy(self, context: u64) {
        ez_gfx_indirect_release(self.handle, context);
    }
}

pub struct IndexHeap;

impl IndexHeap {
    pub fn upload(context: u64, indices: &[u32], name: &str) -> Result<(Self, u32), String> {
        let bytes = u64::try_from(size_of_val(indices))
            .map_err(|_| "index heap size exceeds ABI".to_owned())?;
        let name = CString::new(name).map_err(|_| "index heap name contains NUL".to_owned())?;
        status(
            {
                // SAFETY: The debug name is a live NUL-terminated string for this call.
                unsafe { ez_gfx_index_heap_create(bytes, name.as_ptr(), context) }
            },
            "create index heap",
        )?;
        let mut start = 0;
        if let Err(error) = status(
            {
                // SAFETY: The index byte range and aligned output remain live with the declared sizes for this call.
                unsafe {
                    ez_gfx_vertex_upload_indices(
                        indices.as_ptr().cast(),
                        indices.len() as u32,
                        &raw mut start,
                        context,
                    )
                }
            },
            "upload indices",
        ) {
            ez_gfx_index_heap_destroy(context);
            return Err(error);
        }
        Ok((Self, start))
    }

    pub fn destroy(self, context: u64) {
        ez_gfx_index_heap_destroy(context);
    }
}

pub struct Texture {
    pub handle: u64,
    pub binding: u32,
}

impl Texture {
    #[allow(
        clippy::too_many_arguments,
        reason = "Example host callbacks mirror the rendering API contract."
    )]
    pub fn load(
        context: u64,
        bytes: &[u8],
        source_format: u8,
        width: u32,
        height: u32,
        mips: bool,
        anisotropy: f32,
        repeat: bool,
        label: &str,
    ) -> Result<Self, String> {
        let label = CString::new(label).map_err(|_| "texture label contains NUL".to_owned())?;
        let desc = EzGfxTextureDesc {
            source_format,
            destination_format: 0,
            width,
            height,
            mip_count: 0,
            generate_mips: u8::from(mips),
            min_filter: 1,
            mag_filter: 1,
            max_anisotropy: anisotropy,
            address_mode_u: u8::from(repeat),
            address_mode_v: u8::from(repeat),
            address_mode_w: u8::from(repeat),
            debug_label: label.as_ptr(),
        };
        let mut handle = 0;
        status(
            {
                // SAFETY: Texture bytes, the aligned descriptor and optional NUL-terminated label, and output storage remain live for this call; nulls only exercise checked rejection.
                unsafe {
                    ez_gfx_texture_load(
                        bytes.as_ptr(),
                        bytes.len(),
                        &raw const desc,
                        &raw mut handle,
                        context,
                    )
                }
            },
            "load texture",
        )?;
        let mut binding = 0;
        if let Err(error) = status(
            {
                // SAFETY: The non-null output points to one live, aligned u32 for this call.
                unsafe { ez_gfx_texture_get_binding(handle, &raw mut binding, context) }
            },
            "resolve texture binding",
        ) {
            ez_gfx_texture_unload(handle, context);
            return Err(error);
        }
        Ok(Self { handle, binding })
    }

    pub fn destroy(self, context: u64) {
        ez_gfx_texture_unload(self.handle, context);
    }
}

pub fn binding(name: &CString, structured: u64, indirect: u64) -> EzGfxBinding {
    EzGfxBinding {
        name: name.as_ptr(),
        structured,
        indirect,
        render_target: 0,
    }
}

pub fn record_graphics(
    context: u64,
    shader: u64,
    indirect: u64,
    bindings: &[EzGfxBinding],
    dynamic: Option<EzGfxDynamicState>,
    push: &[u8],
) -> Result<(), String> {
    status(
        {
            // SAFETY: Binding entries and names, optional dynamic state, and the declared push-byte range remain live and properly aligned for this call.
            unsafe {
                ez_gfx_render_add_vertex_pipeline(
                    shader,
                    indirect,
                    bindings.as_ptr(),
                    bindings.len() as u32,
                    dynamic.as_ref().map_or(core::ptr::null(), |value| value),
                    push.as_ptr().cast(),
                    push.len() as u32,
                    context,
                )
            }
        },
        "record graphics pipeline",
    )
}

pub fn bytes_of<T>(value: &T) -> &[u8] {
    // SAFETY: The host contract guarantees valid graphics resource pointers.
    unsafe {
        core::slice::from_raw_parts(
            std::ptr::from_ref::<T>(value).cast(),
            std::mem::size_of::<T>(),
        )
    }
}
