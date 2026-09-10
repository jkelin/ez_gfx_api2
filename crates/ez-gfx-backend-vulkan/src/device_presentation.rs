use super::{CStr, khr, vk};

pub(super) const FIFO_LATEST_READY_EXT: &CStr = c"VK_EXT_present_mode_fifo_latest_ready";
pub(super) const FIFO_LATEST_READY_KHR: &CStr = c"VK_KHR_present_mode_fifo_latest_ready";
const PHYSICAL_DEVICE_PRESENT_MODE_FIFO_LATEST_READY_FEATURES: vk::StructureType =
    vk::StructureType::from_raw(1_000_361_000);

#[repr(C)]
pub(super) struct PhysicalDevicePresentModeFifoLatestReadyFeatures {
    pub(super) s_type: vk::StructureType,
    pub(super) p_next: *mut core::ffi::c_void,
    pub(super) present_mode_fifo_latest_ready: vk::Bool32,
}

impl Default for PhysicalDevicePresentModeFifoLatestReadyFeatures {
    fn default() -> Self {
        Self {
            s_type: PHYSICAL_DEVICE_PRESENT_MODE_FIFO_LATEST_READY_FEATURES,
            p_next: core::ptr::null_mut(),
            present_mode_fifo_latest_ready: vk::FALSE,
        }
    }
}

fn fifo_latest_ready_extension(
    available: &[vk::ExtensionProperties],
) -> Option<*const core::ffi::c_char> {
    available_extension(available, FIFO_LATEST_READY_KHR)
        .or_else(|| available_extension(available, FIFO_LATEST_READY_EXT))
}

pub(super) fn supports_fifo_latest_ready(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    extension_available: bool,
) -> bool {
    if !extension_available {
        return false;
    }
    let mut feature = PhysicalDevicePresentModeFifoLatestReadyFeatures::default();
    let mut features = vk::PhysicalDeviceFeatures2 {
        p_next: (&raw mut feature).cast(),
        ..Default::default()
    };
    // SAFETY: the physical device belongs to `instance`; both output structures live through the query.
    unsafe { instance.get_physical_device_features2(physical, &mut features) };
    feature.present_mode_fifo_latest_ready != vk::FALSE
}

pub(crate) fn available_extension(
    available: &[vk::ExtensionProperties],
    name: &'static CStr,
) -> Option<*const core::ffi::c_char> {
    available
        .iter()
        .any(|extension| {
            // SAFETY: Vulkan guarantees a NUL-terminated fixed-size extension name.
            (unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) }) == name
        })
        .then_some(name.as_ptr())
}

pub(crate) fn device_extensions(
    available: &[vk::ExtensionProperties],
) -> (Vec<*const core::ffi::c_char>, bool, bool) {
    let swapchain = available_extension(available, khr::swapchain::NAME).is_some();
    let fifo_latest_ready = fifo_latest_ready_extension(available);
    let mut enabled = Vec::with_capacity(3);
    enabled.extend(swapchain.then_some(khr::swapchain::NAME.as_ptr()));
    enabled.extend(
        available_extension(available, khr::portability_subset::NAME)
            .map(|_| khr::portability_subset::NAME.as_ptr()),
    );
    enabled.extend(fifo_latest_ready);
    (enabled, swapchain, fifo_latest_ready.is_some())
}
