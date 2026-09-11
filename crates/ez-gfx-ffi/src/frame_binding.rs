#[derive(Clone, Copy)]
enum ValidatedBindingResource {
    Buffer { handle: u64, kind: buffer::Kind },
    RenderTarget(RenderTargetHandle),
}

type FrameBindingDraft = HashMap<String, ValidatedBindingResource>;
static FRAME_BINDINGS: LazyLock<Mutex<HashMap<EzGfxFrame, FrameBindingDraft>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Validates one foreign binding completely before it can enter a frame draft.
fn validate_binding(
    frame: EzGfxFrame,
    binding: EzGfxBinding,
) -> Result<(String, ValidatedBindingResource), EzGfxResult> {
    if binding.name_length > 255 {
        return Err(EzGfxResult::InvalidArgument);
    }
    let name = read_bounded_string(binding.name, binding.name_length)?;
    let resource = match (
        binding.buffer != 0,
        binding.counter_buffer != 0,
        binding.render_target != 0,
    ) {
        (true, false, false) => ValidatedBindingResource::Buffer {
            handle: binding.buffer,
            kind: buffer::Kind::Buffer,
        },
        (false, true, false) => ValidatedBindingResource::Buffer {
            handle: binding.counter_buffer,
            kind: buffer::Kind::Counter,
        },
        (false, false, true) => {
            let target = RenderTargetHandle::from_raw(binding.render_target)
                .map_err(|_| EzGfxResult::InvalidContext)?;
            ValidatedBindingResource::RenderTarget(target)
        }
        _ => return Err(EzGfxResult::InvalidArgument),
    };
    let frame_entry = frame::get(frame)?;
    match resource {
        ValidatedBindingResource::Buffer { handle, kind } => {
            buffer::validate(frame, handle, kind)?;
        }
        ValidatedBindingResource::RenderTarget(target) => {
            raw::render_target_extent(frame_entry.owner, target).map_err(EzGfxResult::from)?;
        }
    }
    Ok((name, resource))
}

fn materialized_bindings(frame: EzGfxFrame) -> Result<Vec<PublicBinding>, EzGfxResult> {
    let drafts = FRAME_BINDINGS
        .lock()
        .map_err(|_| EzGfxResult::NativeFailure)?;
    let Some(draft) = drafts.get(&frame) else {
        return Ok(Vec::new());
    };
    let mut bindings = Vec::with_capacity(draft.len());
    for (name, binding) in draft {
        let resource = match *binding {
            ValidatedBindingResource::Buffer { handle, kind } => {
                buffer::materialize(frame, handle, kind)?
            }
            ValidatedBindingResource::RenderTarget(target) => {
                ResourceIdentity::RenderTarget(target)
            }
        };
        bindings.push(PublicBinding {
            name: name.clone(),
            resource,
        });
    }
    Ok(bindings)
}

fn clear_binding_draft(frame: EzGfxFrame) {
    if let Ok(mut drafts) = FRAME_BINDINGS.lock() {
        drafts.remove(&frame);
    }
}

fn clear_owner_binding_drafts(owner: ContextHandle) {
    if let Ok(mut drafts) = FRAME_BINDINGS.lock() {
        drafts.retain(|frame, _| frame::get(*frame).is_ok_and(|entry| entry.owner != owner));
    }
}
