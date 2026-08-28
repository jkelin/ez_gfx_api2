use ez_gfx_core::handle::{LocalHandle, PackedHandle};
use ez_gfx_runtime::{ContextHealth, ContextIdentity, LifecycleError, ResourceKind};

#[test]
fn identity_table_checks_owner_kind_generation_and_destroy() {
    let mut first = ContextIdentity::new(LocalHandle::new(3, 7).unwrap()).unwrap();
    let surface = first.insert(ResourceKind::Surface).unwrap();
    let shader = first.insert(ResourceKind::Shader).unwrap();

    assert_eq!(first.resolve(surface, ResourceKind::Surface), Ok(()));
    assert_eq!(
        first.resolve(surface, ResourceKind::Shader),
        Err(LifecycleError::WrongResourceKind)
    );
    assert_eq!(
        first.resolve(shader, ResourceKind::Surface),
        Err(LifecycleError::WrongResourceKind)
    );

    let second = ContextIdentity::new(LocalHandle::new(8, 2).unwrap()).unwrap();
    assert_eq!(
        second.resolve(surface, ResourceKind::Surface),
        Err(LifecycleError::WrongOwner)
    );

    first.remove(surface, ResourceKind::Surface).unwrap();
    assert_eq!(
        first.resolve(surface, ResourceKind::Surface),
        Err(LifecycleError::StaleHandle)
    );
    let reused = first.insert(ResourceKind::Surface).unwrap();
    assert_ne!(surface, reused);
}

#[test]
fn context_handle_and_child_layout_match_existing_packing() {
    let mut identity = ContextIdentity::new(LocalHandle::new(1, 2).unwrap()).unwrap();
    let child = identity.insert(ResourceKind::Texture).unwrap();

    assert_eq!(identity.context_handle().get(), 2 | (2 << 20));
    assert!(matches!(
        PackedHandle::from_raw(child.get())
            .unwrap()
            .parts()
            .unwrap(),
        ez_gfx_core::handle::HandleParts::Child { .. }
    ));
}

#[test]
fn loss_is_terminal_exactly_once_and_rejects_new_work() {
    let identity = ContextIdentity::new(LocalHandle::new(0, 1).unwrap()).unwrap();
    assert_eq!(identity.health(), ContextHealth::Healthy);
    assert_eq!(identity.mark_lost(), Ok(()));
    assert_eq!(identity.health(), ContextHealth::Lost);
    assert_eq!(identity.mark_lost(), Err(LifecycleError::AlreadyLost));
    assert_eq!(
        identity.check_thread_and_health(),
        Err(LifecycleError::DeviceLost)
    );
}

#[test]
fn context_is_affine_to_its_creation_thread() {
    let identity = ContextIdentity::new(LocalHandle::new(0, 1).unwrap()).unwrap();
    std::thread::scope(|scope| {
        scope.spawn(|| {
            assert_eq!(
                identity.check_thread_and_health(),
                Err(LifecycleError::WrongThread)
            )
        });
    });
    assert_eq!(identity.check_thread_and_health(), Ok(()));
}
