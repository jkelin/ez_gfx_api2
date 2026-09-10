use super::{
    AllocationCreateDesc, CreateDXGIFactory1, D3D12_CLEAR_VALUE, D3D12_CLEAR_VALUE_0,
    D3D12_DEPTH_STENCIL_VALUE, D3D12_DESCRIPTOR_HEAP_DESC, D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
    D3D12_DESCRIPTOR_HEAP_TYPE_DSV, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, D3D12_RENDER_TARGET_VIEW_DESC,
    D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_TEXTURE2D,
    D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, D3D12_RESOURCE_STATE_DEPTH_WRITE,
    D3D12_RTV_DIMENSION_TEXTURE2D, D3D12_TEXTURE_LAYOUT_UNKNOWN, DXGI_ALPHA_MODE_IGNORE,
    DXGI_FEATURE_PRESENT_ALLOW_TEARING, DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM,
    DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_PRESENT, DXGI_PRESENT_ALLOW_TEARING, DXGI_SAMPLE_DESC,
    DXGI_SCALING_STRETCH, DXGI_SWAP_CHAIN_DESC1, DXGI_SWAP_CHAIN_FLAG,
    DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING, DXGI_SWAP_EFFECT_FLIP_DISCARD,
    DXGI_USAGE_RENDER_TARGET_OUTPUT, HWND, HalError, ID3D12DescriptorHeap, ID3D12Resource,
    IDXGIFactory4, IDXGIFactory5, Interface, MemoryLocation, NativeContext, NativeSurface,
    PRESENT_SYNC_INTERVAL, SurfaceDepth, map_allocator_hal, map_windows,
};

fn factory_allows_tearing(factory: &IDXGIFactory4) -> bool {
    let Ok(factory) = factory.cast::<IDXGIFactory5>() else {
        return false;
    };
    let mut supported = 0_i32;
    // SAFETY: `supported` is writable BOOL-compatible storage of the exact size declared to DXGI.
    unsafe {
        factory.CheckFeatureSupport(
            DXGI_FEATURE_PRESENT_ALLOW_TEARING,
            (&raw mut supported).cast(),
            u32::try_from(core::mem::size_of_val(&supported)).unwrap_or(u32::MAX),
        )
    }
    .is_ok()
        && supported != 0
}

const fn swapchain_flags(allow_tearing: bool) -> DXGI_SWAP_CHAIN_FLAG {
    if allow_tearing {
        DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING
    } else {
        DXGI_SWAP_CHAIN_FLAG(0)
    }
}

pub(super) const fn dxgi_present_flags(allow_tearing: bool) -> DXGI_PRESENT {
    if allow_tearing {
        DXGI_PRESENT_ALLOW_TEARING
    } else {
        DXGI_PRESENT(0)
    }
}

impl NativeContext {
    /// Acquires and presents one surface image; zero extents remain minimized.
    ///
    /// # Errors
    ///
    /// Returns an error if swap-chain setup fails, no swap chain is available, or presentation fails.
    pub fn acquire_present(
        &mut self,
        surface: &mut NativeSurface,
        width: u32,
        height: u32,
    ) -> Result<(), HalError> {
        self.ensure_swapchain(surface, width, height)?;
        // SAFETY: `surface.swapchain.as_ref()` retains the `IDXGISwapChain3` COM reference for `Present`, which receives only value arguments.
        unsafe {
            surface
                .swapchain
                .as_ref()
                .ok_or(HalError::NotReady)?
                .Present(
                    PRESENT_SYNC_INTERVAL,
                    dxgi_present_flags(surface.allow_tearing),
                )
        }
        .ok()
        .map_err(map_windows)?;
        Ok(())
    }

    ///
    /// # Errors
    ///
    /// Returns an error for zero dimensions or if factory creation, swap-chain creation or resizing, synchronization, depth destruction, or surface-target creation fails.
    pub(super) fn ensure_swapchain(
        &mut self,
        surface: &mut NativeSurface,
        width: u32,
        height: u32,
    ) -> Result<(), HalError> {
        if width == 0 || height == 0 {
            return Err(HalError::NotReady);
        }
        if surface.swapchain.is_none() {
            // SAFETY: factory creation takes no caller-provided pointers.
            let factory: IDXGIFactory4 = unsafe { CreateDXGIFactory1() }.map_err(map_windows)?;
            let allow_tearing = factory_allows_tearing(&factory);
            let flags = swapchain_flags(allow_tearing);
            let desc_flags = u32::try_from(flags.0).map_err(|_| HalError::NativeFailure)?;
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: width,
                Height: height,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                Stereo: false.into(),
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: 3,
                Scaling: DXGI_SCALING_STRETCH,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                AlphaMode: DXGI_ALPHA_MODE_IGNORE,
                Flags: desc_flags,
            };
            // SAFETY: `desc` is fully initialized and lives through `CreateSwapChainForHwnd`; `self.queue` is retained for the call, and both optional descriptor pointers are null.
            let created = unsafe {
                factory.CreateSwapChainForHwnd(
                    &self.queue,
                    HWND(surface.window as *mut _),
                    &raw const desc,
                    None,
                    None,
                )
            }
            .map_err(map_windows)?;
            surface.allow_tearing = allow_tearing;
            surface.swapchain = Some(created.cast().map_err(map_windows)?);
            surface.width = width;
            surface.height = height;
        } else if surface.width != width || surface.height != height {
            self.wait_idle()?;
            self.destroy_surface_depth(surface)?;
            surface.buffers.clear();
            surface.rtv_heap = None;
            // SAFETY: `wait_idle` completed and clearing `surface.buffers` released this backend's back-buffer references before `ResizeBuffers`, which takes only value arguments.
            unsafe {
                surface
                    .swapchain
                    .as_ref()
                    .expect("initialized")
                    .ResizeBuffers(
                        3,
                        width,
                        height,
                        DXGI_FORMAT_R8G8B8A8_UNORM,
                        swapchain_flags(surface.allow_tearing),
                    )
            }
            .map_err(map_windows)?;
            surface.width = width;
            surface.height = height;
        }
        self.ensure_surface_targets(surface)
    }

    ///
    /// # Errors
    ///
    /// Returns an error if no swap chain is available or if RTV heap creation or swap-chain buffer retrieval fails.
    pub(super) fn ensure_surface_targets(
        &self,
        surface: &mut NativeSurface,
    ) -> Result<(), HalError> {
        if !surface.buffers.is_empty() {
            return Ok(());
        }
        let swapchain = surface.swapchain.as_ref().ok_or(HalError::NotReady)?;
        // SAFETY: `self.device.CreateDescriptorHeap` reads the fully initialized three-entry RTV heap descriptor temporary only for the duration of the call.
        let heap: ID3D12DescriptorHeap = unsafe {
            self.device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: 3,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                    NodeMask: 0,
                })
        }
        .map_err(map_windows)?;
        // SAFETY: `self.device` retains the `ID3D12Device` COM reference through `GetDescriptorHandleIncrementSize`, which takes only the RTV heap-type value.
        let stride = unsafe {
            self.device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV)
        } as usize;
        // SAFETY: `heap` retains the `ID3D12DescriptorHeap` COM reference through `GetCPUDescriptorHandleForHeapStart`, which takes no pointer arguments.
        let mut handle = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
        for index in 0..3 {
            let buffer: ID3D12Resource =
                // SAFETY: `index` is in `0..3`, matching the swap chain's buffer count, and `swapchain` is retained through `GetBuffer`.
                unsafe { swapchain.GetBuffer(index) }.map_err(map_windows)?;
            let desc = D3D12_RENDER_TARGET_VIEW_DESC {
                Format: DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
                ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
                ..Default::default()
            };
            // SAFETY: `handle` addresses the current slot of the three-entry RTV `heap`; `desc` selects an sRGB-compatible view of the retained UNORM swap-chain buffer.
            unsafe {
                self.device
                    .CreateRenderTargetView(&buffer, Some(&raw const desc), handle);
            }
            surface.buffers.push(buffer);
            handle.ptr += stride;
        }
        surface.rtv_heap = Some(heap);
        Ok(())
    }

    ///
    /// # Errors
    ///
    /// Returns an error if the allocator is unavailable, allocation fails, placed-resource creation fails or returns no resource, or DSV heap creation fails.
    pub(super) fn ensure_surface_depth(
        &mut self,
        surface: &mut NativeSurface,
    ) -> Result<(), HalError> {
        if surface.depth.is_some() {
            return Ok(());
        }
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: u64::from(surface.width),
            Height: surface.height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_D32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        };
        let allocation_desc = AllocationCreateDesc::from_d3d12_resource_desc(
            self.allocator
                .as_ref()
                .ok_or(HalError::NativeFailure)?
                .device(),
            &desc,
            "ez-gfx-depth",
            MemoryLocation::GpuOnly,
        );
        let allocation = self
            .allocator
            .as_mut()
            .ok_or(HalError::NativeFailure)?
            .allocate(&allocation_desc)
            .map_err(|error| map_allocator_hal(&error))?;
        let clear = D3D12_CLEAR_VALUE {
            Format: DXGI_FORMAT_D32_FLOAT,
            Anonymous: D3D12_CLEAR_VALUE_0 {
                DepthStencil: D3D12_DEPTH_STENCIL_VALUE {
                    Depth: 1.0,
                    Stencil: 0,
                },
            },
        };
        let mut resource = None;
        // SAFETY: `allocation` was obtained for `desc` from this device, so its heap and offset satisfy `CreatePlacedResource`; initialized `desc`, `clear`, and writable `resource` storage live through the call.
        if let Err(error) = unsafe {
            self.device.CreatePlacedResource(
                allocation.heap(),
                allocation.offset(),
                &raw const desc,
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                Some(&raw const clear),
                &raw mut resource,
            )
        } {
            let _ = self
                .allocator
                .as_mut()
                .expect("allocator remains initialized")
                .free(allocation);
            return Err(map_windows(error));
        }
        let Some(resource) = resource else {
            let _ = self
                .allocator
                .as_mut()
                .expect("allocator remains initialized")
                .free(allocation);
            return Err(HalError::NativeFailure);
        };
        // SAFETY: `self.device.CreateDescriptorHeap` reads the fully initialized one-entry DSV heap descriptor temporary only for the duration of the call.
        let heap: ID3D12DescriptorHeap = match unsafe {
            self.device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                    NumDescriptors: 1,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                    NodeMask: 0,
                })
        } {
            Ok(heap) => heap,
            Err(error) => {
                drop(resource);
                let _ = self
                    .allocator
                    .as_mut()
                    .expect("allocator remains initialized")
                    .free(allocation);
                return Err(map_windows(error));
            }
        };
        // SAFETY: the start handle names the sole slot of the retained DSV `heap`, and the D32_FLOAT depth `resource` is retained through `CreateDepthStencilView`.
        unsafe {
            self.device.CreateDepthStencilView(
                &resource,
                None,
                heap.GetCPUDescriptorHandleForHeapStart(),
            );
        };
        surface.depth = Some(SurfaceDepth {
            resource,
            allocation,
            heap,
        });
        Ok(())
    }

    ///
    /// # Errors
    ///
    /// Returns an error if the allocator is unavailable or freeing the depth allocation fails.
    pub(super) fn destroy_surface_depth(
        &mut self,
        surface: &mut NativeSurface,
    ) -> Result<(), HalError> {
        let Some(depth) = surface.depth.take() else {
            return Ok(());
        };
        drop(depth.resource);
        drop(depth.heap);
        self.allocator
            .as_mut()
            .ok_or(HalError::NativeFailure)?
            .free(depth.allocation)
            .map_err(|error| map_allocator_hal(&error))
    }

    /// Destroys backend-owned state associated with a borrowed host surface.
    ///
    /// Returns `false` only when native work could not drain and the surface was abandoned.
    pub fn destroy_surface(&mut self, mut surface: NativeSurface) -> bool {
        let _ = self.wait_idle();
        if !self.is_drained() {
            // Preserve swapchain buffers, depth storage, and the host retained by the safe layer.
            core::mem::forget(surface);
            return false;
        }
        let _ = self.destroy_surface_depth(&mut surface);
        true
    }
}
#[cfg(test)]
mod tests {
    use super::{PRESENT_SYNC_INTERVAL, dxgi_present_flags, swapchain_flags};

    #[test]
    fn presentation_tearing_flags_follow_capability() {
        assert_eq!(PRESENT_SYNC_INTERVAL, 0);
        assert_eq!(swapchain_flags(false), super::DXGI_SWAP_CHAIN_FLAG(0));
        assert_eq!(dxgi_present_flags(false), super::DXGI_PRESENT(0));
        assert_eq!(
            swapchain_flags(true),
            super::DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING
        );
        assert_eq!(dxgi_present_flags(true), super::DXGI_PRESENT_ALLOW_TEARING);
    }
}
