use super::*;
use raw_window_handle::{
    AndroidDisplayHandle, AndroidNdkWindowHandle, AppKitDisplayHandle, AppKitWindowHandle,
    RawDisplayHandle, RawWindowHandle, WebDisplayHandle, WebWindowHandle, Win32WindowHandle,
    WindowsDisplayHandle,
};
use std::{num::NonZeroIsize, ptr::NonNull};

fn extension(name: &CStr) -> vk::ExtensionProperties {
    let mut property = vk::ExtensionProperties::default();
    // Test names are shorter than Vulkan's fixed extension-name storage and include the NUL.
    for (destination, source) in property
        .extension_name
        .iter_mut()
        .zip(name.to_bytes_with_nul())
    {
        *destination = i8::try_from(*source).unwrap();
    }
    property
}

fn enabled_has(enabled: &[*const core::ffi::c_char], name: &CStr) -> bool {
    enabled.iter().any(|pointer| {
        // SAFETY: extension policy returns pointers to static NUL-terminated Vulkan names.
        (unsafe { CStr::from_ptr(*pointer) }) == name
    })
}

#[test]
fn native_extent_distinguishes_known_minimized_and_host_managed() {
    assert_eq!(
        classify_surface_extent(vk::Extent2D {
            width: 640,
            height: 480,
        }),
        NativeWindowExtent::Known(640, 480)
    );
    assert_eq!(
        classify_surface_extent(vk::Extent2D {
            width: 0,
            height: 0,
        }),
        NativeWindowExtent::Minimized
    );
    assert_eq!(
        classify_surface_extent(vk::Extent2D {
            width: u32::MAX,
            height: u32::MAX,
        }),
        NativeWindowExtent::HostManaged
    );
}

#[test]
fn present_mode_prefers_nonblocking_mailbox_then_immediate() {
    assert_eq!(
        preferred_present_mode(&[
            vk::PresentModeKHR::FIFO,
            vk::PresentModeKHR::IMMEDIATE,
            vk::PresentModeKHR::MAILBOX,
        ]),
        vk::PresentModeKHR::MAILBOX
    );
    assert_eq!(
        preferred_present_mode(&[vk::PresentModeKHR::FIFO, vk::PresentModeKHR::IMMEDIATE]),
        vk::PresentModeKHR::IMMEDIATE
    );
    assert_eq!(
        preferred_present_mode(&[vk::PresentModeKHR::FIFO]),
        vk::PresentModeKHR::FIFO
    );
    assert_eq!(preferred_present_mode(&[]), vk::PresentModeKHR::FIFO);
}

#[test]
fn instance_policy_enables_only_available_native_wsi_extensions() {
    let available = [
        extension(khr::surface::NAME),
        extension(khr::win32_surface::NAME),
        extension(khr::android_surface::NAME),
        extension(ash::ext::metal_surface::NAME),
    ];
    let (enabled, flags, headless) = instance_extensions(&available, true);

    assert_eq!(enabled.len(), 2);
    assert!(enabled_has(&enabled, khr::surface::NAME));
    assert!(enabled_has(&enabled, khr::win32_surface::NAME));
    assert_eq!(flags, vk::InstanceCreateFlags::empty());
    assert!(!headless);
}

#[test]
fn portability_and_headless_policy_follow_advertised_extensions() {
    let available = [
        extension(khr::surface::NAME),
        extension(ash::ext::headless_surface::NAME),
        extension(khr::portability_enumeration::NAME),
    ];

    let (enabled, flags, headless) = instance_extensions(&available, false);

    assert!(enabled_has(&enabled, ash::ext::headless_surface::NAME));
    assert!(enabled_has(&enabled, khr::portability_enumeration::NAME));
    assert_eq!(flags, vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);
    assert!(headless);
}

#[test]
fn surface_pair_requires_its_enabled_wsi_extension() {
    let window = RawWindowHandle::Win32(Win32WindowHandle::new(NonZeroIsize::new(1).unwrap()));
    let display = RawDisplayHandle::Windows(WindowsDisplayHandle::new());

    assert!(!supported_surface_pair(
        WsiCapabilities {
            surface: true,
            ..WsiCapabilities::default()
        },
        display,
        window,
    ));
    assert!(supported_surface_pair(
        WsiCapabilities {
            surface: true,
            win32: true,
            ..WsiCapabilities::default()
        },
        display,
        window,
    ));
}

#[test]
fn surface_pair_policy_rejects_unverified_and_mismatched_systems() {
    let win32 = RawWindowHandle::Win32(Win32WindowHandle::new(NonZeroIsize::new(1).unwrap()));
    assert!(supported_surface_pair(
        WsiCapabilities {
            surface: true,
            win32: true,
            ..WsiCapabilities::default()
        },
        RawDisplayHandle::Windows(WindowsDisplayHandle::new()),
        win32,
    ));
    assert!(!supported_surface_pair(
        WsiCapabilities {
            surface: true,
            win32: true,
            ..WsiCapabilities::default()
        },
        RawDisplayHandle::AppKit(AppKitDisplayHandle::new()),
        win32,
    ));

    let pointer = NonNull::dangling();
    assert!(!supported_surface_pair(
        WsiCapabilities::default(),
        RawDisplayHandle::AppKit(AppKitDisplayHandle::new()),
        RawWindowHandle::AppKit(AppKitWindowHandle::new(pointer)),
    ));
    assert!(!supported_surface_pair(
        WsiCapabilities::default(),
        RawDisplayHandle::Web(WebDisplayHandle::new()),
        RawWindowHandle::Web(WebWindowHandle::new(1)),
    ));
    assert!(!supported_surface_pair(
        WsiCapabilities::default(),
        RawDisplayHandle::Android(AndroidDisplayHandle::new()),
        RawWindowHandle::AndroidNdk(AndroidNdkWindowHandle::new(pointer)),
    ));
}
