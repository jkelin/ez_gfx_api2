use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use super::{
    AllocationCreateDesc, AllocationError, AllocationScheme, CStr, DepthTarget, HalError,
    MemoryLocation, NativeContext, NativeSurface, device::available_extension, khr,
    map_allocation_hal, map_allocator, map_vk, vk,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "WSI extensions are independent Vulkan capabilities, not mutually exclusive states"
)]
pub(crate) struct WsiCapabilities {
    pub(crate) surface: bool,
    pub(crate) win32: bool,
    pub(crate) wayland: bool,
    pub(crate) xcb: bool,
    pub(crate) xlib: bool,
}

impl WsiCapabilities {
    pub(super) fn from_enabled(enabled: &[*const core::ffi::c_char]) -> Self {
        let has = |name: &'static CStr| {
            enabled.iter().any(|pointer| {
                // SAFETY: instance extension policy returns only static Vulkan extension names.
                unsafe { CStr::from_ptr(*pointer) == name }
            })
        };
        Self {
            surface: has(khr::surface::NAME),
            win32: has(khr::win32_surface::NAME),
            wayland: has(khr::wayland_surface::NAME),
            xcb: has(khr::xcb_surface::NAME),
            xlib: has(khr::xlib_surface::NAME),
        }
    }
}

/// Drawable extent disposition reported by a Vulkan window system.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeWindowExtent {
    /// The window system reports an authoritative drawable extent.
    Known(u32, u32),
    /// The drawable is minimized and cannot present.
    Minimized,
    /// The window system requires the host to supply its configured extent.
    HostManaged,
}

pub(crate) const fn classify_surface_extent(extent: vk::Extent2D) -> NativeWindowExtent {
    if extent.width == u32::MAX || extent.height == u32::MAX {
        NativeWindowExtent::HostManaged
    } else if extent.width == 0 || extent.height == 0 {
        NativeWindowExtent::Minimized
    } else {
        NativeWindowExtent::Known(extent.width, extent.height)
    }
}

pub(crate) fn preferred_present_mode(modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    // Mailbox preserves tear-free display pacing without blocking each present. Immediate is the
    // low-latency fallback; FIFO is required by Vulkan when neither optional mode is available.
    [vk::PresentModeKHR::MAILBOX, vk::PresentModeKHR::IMMEDIATE]
        .into_iter()
        .find(|candidate| modes.contains(candidate))
        .unwrap_or(vk::PresentModeKHR::FIFO)
}

pub(crate) fn instance_extensions(
    available: &[vk::ExtensionProperties],
    enable_debug: bool,
) -> (Vec<*const core::ffi::c_char>, vk::InstanceCreateFlags, bool) {
    let native_wsi = [
        khr::surface::NAME,
        khr::win32_surface::NAME,
        khr::wayland_surface::NAME,
        khr::xcb_surface::NAME,
        khr::xlib_surface::NAME,
        ash::ext::headless_surface::NAME,
        khr::portability_enumeration::NAME,
    ];
    let mut enabled = native_wsi
        .into_iter()
        .filter_map(|name| available_extension(available, name))
        .collect::<Vec<_>>();
    if enable_debug {
        enabled.extend(available_extension(available, ash::ext::debug_utils::NAME));
    }
    let portability = available_extension(available, khr::portability_enumeration::NAME).is_some();
    let headless = available_extension(available, ash::ext::headless_surface::NAME).is_some();
    (
        enabled,
        if portability {
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::empty()
        },
        headless,
    )
}

pub(crate) const fn matched_surface_pair(
    display: RawDisplayHandle,
    window: RawWindowHandle,
) -> bool {
    matches!(
        (display, window),
        (RawDisplayHandle::Windows(_), RawWindowHandle::Win32(_))
            | (RawDisplayHandle::Xlib(_), RawWindowHandle::Xlib(_))
            | (RawDisplayHandle::Xcb(_), RawWindowHandle::Xcb(_))
            | (RawDisplayHandle::Wayland(_), RawWindowHandle::Wayland(_))
    )
}
pub(crate) const fn supported_surface_pair(
    capabilities: WsiCapabilities,
    display: RawDisplayHandle,
    window: RawWindowHandle,
) -> bool {
    // Both VK_KHR_surface and the exact platform extension must be enabled before dispatch.
    capabilities.surface
        && match (display, window) {
            (RawDisplayHandle::Windows(_), RawWindowHandle::Win32(_)) => capabilities.win32,
            (RawDisplayHandle::Xlib(_), RawWindowHandle::Xlib(_)) => capabilities.xlib,
            (RawDisplayHandle::Xcb(_), RawWindowHandle::Xcb(_)) => capabilities.xcb,
            (RawDisplayHandle::Wayland(_), RawWindowHandle::Wayland(_)) => capabilities.wayland,
            _ => false,
        }
}
impl NativeContext {
    /// Creates an owned Vulkan surface from a matched borrowed raw handle pair.
    ///
    /// # Safety
    ///
    /// `display` and `window` must be a matched pair belonging to the same live host. This function
    /// must run on that host's creator thread. The host must remain alive until the returned surface
    /// is destroyed by [`NativeContext::destroy_surface`] or native teardown is abandoned, after
    /// which it must remain alive for the process lifetime.
    ///
    /// # Errors
    ///
    /// Returns an error when the pair is unsupported, mismatched, or native creation fails.
    pub unsafe fn create_surface(
        &self,
        display: RawDisplayHandle,
        window: RawWindowHandle,
    ) -> Result<NativeSurface, HalError> {
        if !matched_surface_pair(display, window) {
            return Err(HalError::InvalidArgument);
        }
        if !supported_surface_pair(self.wsi_capabilities, display, window) {
            return Err(HalError::Unsupported);
        }
        // SAFETY: the pair was matched above and the safe owner retains the host value until
        // after this surface is destroyed.
        let handle = unsafe {
            ash_window::create_surface(&self.entry_loader, &self.instance, display, window, None)
        }
        .map_err(map_vk)?;
        if handle == vk::SurfaceKHR::null() {
            return Err(HalError::NativeFailure);
        }
        Ok(NativeSurface {
            handle,
            presented_rgba8: Vec::new(),
        })
    }
    /// Creates a headless target without native host handles.
    ///
    /// Uses `VK_EXT_headless_surface` when the active ICD exposes it. Otherwise
    /// the returned logical target supports device initialization and target-only
    /// work but not presentation.
    ///
    /// # Errors
    ///
    /// Returns an error if an advertised headless-surface creation call fails.
    pub fn create_headless_surface(&self) -> Result<NativeSurface, HalError> {
        let handle = if self.headless_surface_enabled {
            let loader =
                ash::ext::headless_surface::Instance::new(&self.entry_loader, &self.instance);
            let create = vk::HeadlessSurfaceCreateInfoEXT::default();
            // SAFETY: the instance enabled the advertised headless extension and `create` spans the call.
            unsafe { loader.create_headless_surface(&create, None) }.map_err(map_vk)?
        } else {
            vk::SurfaceKHR::null()
        };
        Ok(NativeSurface {
            handle,
            presented_rgba8: Vec::new(),
        })
    }

    /// Reads the drawable extent disposition reported by the window system.
    ///
    /// `HostManaged` means Vulkan cannot provide the size (notably on Wayland); the toolkit-neutral
    /// caller must forward its configure-event extent through the resize API.
    ///
    /// # Errors
    ///
    /// Returns an error when no device is initialized or Vulkan rejects the query.
    pub fn window_extent(&self, surface: &NativeSurface) -> Result<NativeWindowExtent, HalError> {
        if surface.is_headless() {
            return Err(HalError::InvalidArgument);
        }
        let physical = self.physical_device.ok_or(HalError::NotReady)?;
        // SAFETY: `physical` and `surface` belong to this live instance.
        let capabilities = unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(physical, surface.handle)
        }
        .map_err(map_vk)?;
        Ok(classify_surface_extent(capabilities.current_extent))
    }
    /// Acquires and presents one surface image; zero extents remain minimized.
    ///
    /// # Errors
    ///
    /// Returns `HalError::NotReady` for zero extents or missing swapchain state, and propagates swapchain recreation and Vulkan image-acquisition or presentation errors.
    pub fn acquire_present(
        &mut self,
        surface: &NativeSurface,
        width: u32,
        height: u32,
    ) -> Result<(), HalError> {
        if width == 0 || height == 0 {
            return Err(HalError::NotReady);
        }
        if self.swapchain.is_none()
            || self.swapchain_extent.width != width
            || self.swapchain_extent.height != height
        {
            self.recreate_swapchain(surface, width, height)?;
        }
        let loader = self.swapchain_loader.as_ref().ok_or(HalError::NotReady)?;
        let semaphore = self.image_available.ok_or(HalError::NotReady)?;
        let swapchain = self.swapchain.ok_or(HalError::NotReady)?;
        // SAFETY: `acquire_next_image` uses the swapchain and acquisition semaphore stored for this loader's device, both retained through the call; the null fence has no lifetime requirement.
        let (image, suboptimal) = match unsafe {
            loader.acquire_next_image(swapchain, u64::MAX, semaphore, vk::Fence::null())
        } {
            Ok(value) => value,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swapchain(surface, width, height)?;
                let loader = self.swapchain_loader.as_ref().ok_or(HalError::NotReady)?;
                // SAFETY: the failed acquire left `semaphore` unsignaled, and `recreate_swapchain` waited for device idle and installed the swapchain used by this loader; the fence is null.
                unsafe {
                    loader.acquire_next_image(
                        self.swapchain.ok_or(HalError::NotReady)?,
                        u64::MAX,
                        semaphore,
                        vk::Fence::null(),
                    )
                }
                .map_err(map_vk)?
            }
            Err(error) => return Err(map_vk(error)),
        };
        let swapchain = self.swapchain.ok_or(HalError::NotReady)?;
        let present = vk::PresentInfoKHR::default()
            .wait_semaphores(core::slice::from_ref(&semaphore))
            .swapchains(core::slice::from_ref(&swapchain))
            .image_indices(core::slice::from_ref(&image));
        // SAFETY: `image` came from the preceding acquire for `swapchain`, `semaphore` is that
        // acquire's signal, and the three slices backing `present` outlive `queue_present`.
        let presented = {
            let _queue_guard = self.graphics_queue_lock.lock();
            // SAFETY: the queue lock serializes this present with all graphics submissions.
            unsafe {
                self.swapchain_loader
                    .as_ref()
                    .ok_or(HalError::NotReady)?
                    .queue_present(self.graphics_queue.ok_or(HalError::NotReady)?, &present)
            }
        };
        match presented {
            Ok(present_suboptimal) => {
                if suboptimal || present_suboptimal {
                    self.recreate_swapchain(surface, width, height)?;
                }
                Ok(())
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swapchain(surface, width, height)
            }
            Err(error) => Err(map_vk(error)),
        }
    }

    ///
    /// # Errors
    ///
    /// Returns an error if required context state is absent, the surface has no usable format, extent, usage, or composite alpha, depth-target cleanup fails, or a Vulkan wait, query, or swapchain operation fails.
    pub(super) fn recreate_swapchain(
        &mut self,
        surface: &NativeSurface,
        requested_width: u32,
        requested_height: u32,
    ) -> Result<(), HalError> {
        self.wait_idle()?;
        self.destroy_depth_target().map_err(map_allocation_hal)?;
        let physical = self.physical_device.ok_or(HalError::NotReady)?;
        let device = self.device.as_ref().ok_or(HalError::NotReady)?;
        // SAFETY: `physical` was enumerated from `surface_loader`'s instance and `surface.handle` was created on that instance; ash owns the capabilities output storage through the query.
        let capabilities = unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(physical, surface.handle)
        }
        .map_err(map_vk)?;
        // SAFETY: `physical` and `surface.handle` belong to `surface_loader`'s instance, and ash keeps its surface-format enumeration storage allocated until the query returns.
        let formats = unsafe {
            self.surface_loader
                .get_physical_device_surface_formats(physical, surface.handle)
        }
        .map_err(map_vk)?;
        // SAFETY: `physical` and `surface.handle` belong to `surface_loader`'s instance, and ash
        // owns the returned mode storage.
        let present_modes = unsafe {
            self.surface_loader
                .get_physical_device_surface_present_modes(physical, surface.handle)
        }
        .map_err(map_vk)?;
        let present_mode = preferred_present_mode(&present_modes);
        let chosen = formats
            .iter()
            .copied()
            .find(|format| format.format == vk::Format::B8G8R8A8_SRGB)
            .or_else(|| formats.first().copied())
            .ok_or(HalError::Unsupported)?;
        let extent = match classify_surface_extent(capabilities.current_extent) {
            NativeWindowExtent::Known(width, height) => vk::Extent2D { width, height },
            NativeWindowExtent::Minimized => return Err(HalError::NotReady),
            NativeWindowExtent::HostManaged => {
                if requested_width < capabilities.min_image_extent.width
                    || requested_width > capabilities.max_image_extent.width
                    || requested_height < capabilities.min_image_extent.height
                    || requested_height > capabilities.max_image_extent.height
                {
                    return Err(HalError::InvalidArgument);
                }
                vk::Extent2D {
                    width: requested_width,
                    height: requested_height,
                }
            }
        };
        if extent.width == 0 || extent.height == 0 {
            return Err(HalError::NotReady);
        }
        let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC;
        if !capabilities.supported_usage_flags.contains(usage) {
            return Err(HalError::Unsupported);
        }
        let alpha_bits = capabilities.supported_composite_alpha.as_raw();
        if alpha_bits == 0 {
            return Err(HalError::Unsupported);
        }
        let composite_alpha =
            vk::CompositeAlphaFlagsKHR::from_raw(alpha_bits & alpha_bits.wrapping_neg());
        for view in self.swapchain_views.drain(..) {
            // SAFETY: `wait_idle` completed all uses of each drained `view`; every view was created by `device` without custom allocation callbacks.
            unsafe { device.destroy_image_view(view, None) };
        }
        for semaphore in self.swapchain_finished.drain(..) {
            // SAFETY: device idle completed all presentation waits on the old swapchain.
            unsafe { device.destroy_semaphore(semaphore, None) };
        }
        let old = self.swapchain.unwrap_or(vk::SwapchainKHR::null());
        let count = capabilities.min_image_count.saturating_add(1).min(
            if capabilities.max_image_count == 0 {
                u32::MAX
            } else {
                capabilities.max_image_count
            },
        );
        let create = vk::SwapchainCreateInfoKHR::default()
            .surface(surface.handle)
            .min_image_count(count)
            .image_format(chosen.format)
            .image_color_space(chosen.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(usage)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(composite_alpha)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old);
        let loader = self.swapchain_loader.as_ref().ok_or(HalError::NotReady)?;
        // SAFETY: `create` uses capabilities and a format queried for `surface.handle`, `old` is null or this loader's idle swapchain, and the stack create-info outlives `create_swapchain`.
        let swapchain = unsafe { loader.create_swapchain(&create, None) }.map_err(map_vk)?;
        // SAFETY: `swapchain` is the undestroyed result of `loader.create_swapchain` above, and ash keeps its image-enumeration buffer allocated until `get_swapchain_images` returns.
        let images = unsafe { loader.get_swapchain_images(swapchain) }.map_err(map_vk)?;
        let mut views = Vec::with_capacity(images.len());
        for image in images {
            // SAFETY: `image` was enumerated from the new swapchain on `device`, its format and one-level color range match that swapchain, and the temporary create-info outlives `create_image_view`.
            match unsafe {
                device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(chosen.format)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        }),
                    None,
                )
            } {
                Ok(view) => views.push(view),
                Err(error) => {
                    for view in views {
                        // SAFETY: `view` was returned by `device` in this loop, was never exposed or submitted for use, and was created without custom allocation callbacks.
                        unsafe { device.destroy_image_view(view, None) };
                    }
                    // SAFETY: `swapchain` was created by `loader` here and never acquired or queued, and every successfully created image view was destroyed before `destroy_swapchain`.
                    unsafe { loader.destroy_swapchain(swapchain, None) };
                    return Err(map_vk(error));
                }
            }
        }
        let mut finished = Vec::with_capacity(views.len());
        for _ in 0..views.len() {
            // SAFETY: the semaphore belongs to this device and starts unsignaled.
            match unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) } {
                Ok(semaphore) => finished.push(semaphore),
                Err(error) => {
                    // SAFETY: this new swapchain and its views/semaphores were never submitted.
                    unsafe {
                        for semaphore in finished {
                            device.destroy_semaphore(semaphore, None);
                        }
                        for view in views {
                            device.destroy_image_view(view, None);
                        }
                        loader.destroy_swapchain(swapchain, None);
                    }
                    return Err(map_vk(error));
                }
            }
        }
        if old != vk::SwapchainKHR::null() {
            // SAFETY: `wait_idle` completed all use of `old`, its image views were destroyed, and successful creation with `old_swapchain(old)` retired it before this destroy.
            unsafe { loader.destroy_swapchain(old, None) };
        }
        self.swapchain = Some(swapchain);
        self.swapchain_views = views;
        self.swapchain_finished = finished;
        self.swapchain_initialized = vec![false; self.swapchain_views.len()];
        self.swapchain_format = chosen.format;
        self.swapchain_extent = extent;
        Ok(())
    }

    ///
    /// # Errors
    ///
    /// Returns an error if required device or allocator state is absent, depth-target cleanup or allocation fails, or a Vulkan wait, image-creation, memory-binding, or image-view operation fails.
    pub(super) fn ensure_depth_target(&mut self, extent: vk::Extent2D) -> Result<(), HalError> {
        if self
            .depth_target
            .as_ref()
            .is_some_and(|target| target.extent == extent)
        {
            return Ok(());
        }
        self.wait_idle()?;
        self.destroy_depth_target().map_err(map_allocation_hal)?;
        let device = self.device.as_ref().ok_or(HalError::NotReady)?.clone();
        let create = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: `create` has null extension and queue-family pointers, its stack storage outlives `create_image`, and `None` passes no allocation-callback pointer.
        let image = unsafe { device.create_image(&create, None) }.map_err(map_vk)?;
        // SAFETY: `image` was returned by `device.create_image` immediately above and has not been destroyed; ash supplies writable requirements storage for the call.
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let allocation = match self.allocator.as_mut().ok_or(HalError::NotReady)?.allocate(
            &AllocationCreateDesc {
                name: "ez-gfx-depth",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            },
        ) {
            Ok(allocation) => allocation,
            Err(error) => {
                // SAFETY: `image` was created by `device`, remains unbound, and was never submitted; it was created without custom allocation callbacks.
                unsafe { device.destroy_image(image, None) };
                return Err(map_allocation_hal(map_allocator(&error)));
            }
        };
        if let Err(error) =
            // SAFETY: `allocation` was made from this image's requirements, so its device memory, offset, size, alignment, and memory type satisfy `bind_image_memory`; `image` is unbound.
            unsafe {
                device.bind_image_memory(image, allocation.memory(), allocation.offset())
            }
        {
            let _ = self
                .allocator
                .as_mut()
                .expect("allocator remains initialized")
                .free(allocation);
            // SAFETY: failed `bind_image_memory` left `image` unbound; it was never submitted and was created by `device` without custom allocation callbacks.
            unsafe { device.destroy_image(image, None) };
            return Err(map_vk(error));
        }
        // SAFETY: `image` is bound to the retained `allocation`, its D32 format and one-level depth range match creation, and the temporary view-info outlives `create_image_view`.
        let view = match unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::D32_SFLOAT)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::DEPTH,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )
        } {
            Ok(view) => view,
            Err(error) => {
                // SAFETY: `image` was created by `device`, never submitted, and remains bound to `allocation` until after `destroy_image`; no custom allocation callbacks were used.
                unsafe { device.destroy_image(image, None) };
                let _ = self
                    .allocator
                    .as_mut()
                    .expect("allocator remains initialized")
                    .free(allocation);
                return Err(map_vk(error));
            }
        };
        self.depth_target = Some(DepthTarget {
            image,
            view,
            allocation,
            extent,
        });
        Ok(())
    }

    ///
    /// # Errors
    ///
    /// Returns an error if the device or allocator is absent or freeing the depth-target allocation fails.
    pub(super) fn destroy_depth_target(&mut self) -> Result<(), AllocationError> {
        let Some(target) = self.depth_target.take() else {
            return Ok(());
        };
        let device = self.device.as_ref().ok_or(AllocationError::NativeFailure)?;
        // SAFETY: each depth-target replacement path waits for device idle first; `target.view` references `target.image`, and `target.allocation` remains allocated until both destroys finish.
        unsafe {
            device.destroy_image_view(target.view, None);
            device.destroy_image(target.image, None);
        }
        self.allocator
            .as_mut()
            .ok_or(AllocationError::NativeFailure)?
            .free(target.allocation)
            .map_err(|error| map_allocator(&error))
    }
}

#[cfg(test)]
#[path = "surface_tests.rs"]
mod tests;
