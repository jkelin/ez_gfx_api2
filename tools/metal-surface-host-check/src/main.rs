//! Apple-only executable proof for the Metal surface host-lifetime ownership assumptions.

#[cfg(target_vendor = "apple")]
fn main() {
    use core::{ffi::c_void, ptr::NonNull};

    use objc2::{MainThreadMarker, rc::autoreleasepool};
    use objc2_app_kit::NSView;
    use objc2_core_foundation::CGSize;
    use objc2_quartz_core::CAMetalLayer;
    use raw_window_metal::Layer;

    autoreleasepool(|_| {
        let main_thread =
            MainThreadMarker::new().expect("ownership check must run on AppKit's main thread");
        let extent = CGSize {
            width: 17.0,
            height: 23.0,
        };

        // Pre-existing branch: the windowless NSView owns a CAMetalLayer root.
        let existing_view = NSView::new(main_thread);
        let existing_root = CAMetalLayer::new();
        existing_view.setWantsLayer(true);
        existing_view.setLayer(Some(&existing_root));
        let root_pointer = NonNull::from(&*existing_root);
        // SAFETY: existing_view is a live NSView with the retained CAMetalLayer root above.
        let existing_layer =
            unsafe { Layer::from_ns_view(NonNull::from(&*existing_view).cast::<c_void>()) };
        assert!(existing_layer.pre_existing());
        assert_eq!(
            existing_layer.as_ptr().cast::<CAMetalLayer>().as_ptr(),
            root_pointer.as_ptr(),
            "from_ns_view must retain the NSView's CAMetalLayer root"
        );
        drop(existing_root);
        drop(existing_view);
        // SAFETY: Layer independently retains this non-null CAMetalLayer after host release.
        let retained_existing =
            unsafe { &*existing_layer.as_ptr().cast::<CAMetalLayer>().as_ptr() };
        retained_existing.setDrawableSize(extent);
        assert_eq!(retained_existing.drawableSize(), extent);
        drop(existing_layer);

        // Observer branch: from_ns_view layer-backs a plain windowless NSView and adds a sublayer.
        let observer_view = NSView::new(main_thread);
        // SAFETY: observer_view is a live NSView retained throughout Layer construction.
        let observer_layer =
            unsafe { Layer::from_ns_view(NonNull::from(&*observer_view).cast::<c_void>()) };
        assert!(!observer_layer.pre_existing());
        // SAFETY: Layer independently retains this non-null observer-backed CAMetalLayer.
        let retained_observer =
            unsafe { &*observer_layer.as_ptr().cast::<CAMetalLayer>().as_ptr() };
        assert!(retained_observer.superlayer().is_some());
        retained_observer.removeFromSuperlayer();
        retained_observer.removeFromSuperlayer();
        assert!(
            retained_observer.superlayer().is_none(),
            "detach must synchronously and idempotently sever the weak host relationship"
        );
        drop(observer_view);
        retained_observer.setDrawableSize(extent);
        assert_eq!(retained_observer.drawableSize(), extent);
        drop(observer_layer);
    });
}

#[cfg(not(target_vendor = "apple"))]
fn main() {
    // This verification target is intentionally inert off Apple platforms.
}
