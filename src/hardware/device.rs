use std::os::fd::RawFd;

#[cfg(feature = "cuda")]
use crate::availability::AvailabilityReport;
#[cfg(feature = "cuda")]
use crate::hardware::cache::DEFAULT_MAX_REGISTERED_MEMORY_BYTES;

use crate::cufile::{self, CUfileHandle};
use crate::error::CudaError;
use crate::hardware::cache::RegistrationCache;

/// Runtime wrapper around a successfully initialized `GPUDirect` Storage session.
///
/// # Registration Cache
///
/// This type caches `cuFileHandleRegister` results across reads to the same file
/// descriptor, avoiding the ~50µs driver round-trip on every call. See the
/// [module-level documentation](crate#registration-cache-lifecycle) for cache
/// lifecycle details.
///
/// # Fast-Path Optimization
///
/// For file descriptors < 1024, a fixed-size array provides O(1) lookup without
/// hashing overhead. This optimizes the common case of sequential fd scanning.
/// FDs >= 1024 use a `HashMap` fallback.
///
/// # Thread Safety
///
/// `CuFileHardware` is `!Send` and `!Sync` because NVIDIA's cuFile driver
/// maintains global state and is not documented as safe for concurrent use.
pub struct CuFileHardware {
    pub(crate) library: cufile::CuFileLibrary,
    pub(crate) cache: RegistrationCache,
}

impl CuFileHardware {
    /// Initializes a `GPUDirect` Storage session when the runtime stack is ready.
    ///
    /// # Errors
    ///
    /// Returns an error when the crate was built without CUDA support, when
    /// `libcufile` cannot be loaded, or when `cuFileDriverOpen` fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudagrep::CuFileHardware;
    ///
    /// match CuFileHardware::try_init() {
    ///     Ok(mut gds) => {
    ///         // GDS session is ready
    ///         let _ = gds.evict_fd(3);
    ///     }
    ///     Err(e) => eprintln!("GDS unavailable: {}", e),
    /// }
    /// ```
    pub fn try_init() -> Result<Self, CudaError> {
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::FeatureDisabled)
        }

        #[cfg(feature = "cuda")]
        {
            let library = cufile::CuFileLibrary::load()
                .map_err(|error| map_load_error_to_unavailable(&error))?;
            library
                .driver_open()
                .map_err(|code| CudaError::DriverInitRejected { code })?;

            Ok(Self {
                library,
                cache: RegistrationCache::new(DEFAULT_MAX_REGISTERED_MEMORY_BYTES),
            })
        }
    }

    /// Retrieve a cached registration or register a new one.
    ///
    /// Uses a fast-path array for fds < 1024, falling back to `HashMap` for larger fds.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::DescriptorRegistrationFailed`] when the driver
    /// rejects the file descriptor registration.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudagrep::CuFileHardware;
    ///
    /// # fn example() -> Result<(), cudagrep::CudaError> {
    /// let mut gds = CuFileHardware::try_init()?;
    /// let handle = gds.get_or_register(3)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_or_register(&mut self, fd: RawFd) -> Result<CUfileHandle, CudaError> {
        self.cache.get_or_register_for_bytes(fd, 0, &self.library)
    }

    /// Evict a file descriptor from the registration cache.
    ///
    /// Call this when an fd is closed to prevent stale handle reuse.
    /// Returns `Ok(true)` if removed, `Ok(false)` if not cached.
    ///
    /// # Errors
    ///
    /// Returns driver errors from the underlying deregistration call.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudagrep::CuFileHardware;
    ///
    /// # fn example() -> Result<(), cudagrep::CudaError> {
    /// let mut gds = CuFileHardware::try_init()?;
    /// let was_cached = gds.evict_fd(3)?;
    /// if was_cached {
    ///     println!("fd 3 deregistered from GDS cache");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn evict_fd(&mut self, fd: RawFd) -> Result<bool, CudaError> {
        self.cache.evict_fd(fd, &self.library)
    }
}

impl Drop for CuFileHardware {
    fn drop(&mut self) {
        // Deregister all cached handles before closing the driver.
        // Partial failures are logged but do not prevent cleanup of remaining handles.
        for (fd, entry) in self.cache.drain_cached_handles() {
            if let Err(error) = self.library.deregister_raw(fd, entry.handle) {
                tracing::warn!(
                    ?error,
                    fd,
                    "cuFileHandleDeregister failed during CuFileHardware drop"
                );
            }
        }

        if let Err(error) = self.library.driver_close() {
            tracing::warn!(
                ?error,
                "cuFileDriverClose failed during CuFileHardware drop"
            );
        }
    }
}

#[cfg(feature = "cuda")]
fn map_load_error_to_unavailable(error: &libloading::Error) -> CudaError {
    CudaError::Unavailable {
        report: AvailabilityReport::new(
            false,
            crate::availability::map_load_reason(error),
            vec![error.to_string()],
        ),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::alignment::GDS_ALIGNMENT;
    use crate::cufile::ffi::{CUfileDescr, CUfileError};
    use crate::cufile::library::{Api, CuFileLibrary};
    use std::ffi::c_void;

    thread_local! {
        pub(crate) static REGISTER_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        pub(crate) static DEREGISTER_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        // Handle table for safe pointer generation - uses NonNull::dangling with offset
        pub(crate) static HANDLE_TABLE: std::cell::RefCell<Vec<CUfileHandle>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    unsafe extern "C" fn driver_open_stub() -> CUfileError {
        CUfileError { err: 0, cu_err: 0 }
    }

    unsafe extern "C" fn driver_close_stub() -> CUfileError {
        CUfileError { err: 0, cu_err: 0 }
    }

    unsafe extern "C" fn register_stub(
        handle: *mut CUfileHandle,
        _descr: *mut CUfileDescr,
    ) -> CUfileError {
        REGISTER_CALLS.with(|c| c.set(c.get() + 1));
        // Use a dangling pointer with alignment-based address - sound because we never dereference
        let stub_handle = CUfileHandle(
            std::ptr::NonNull::<u8>::dangling()
                .as_ptr()
                .cast::<c_void>(),
        );
        HANDLE_TABLE.with(|table| {
            let mut t = table.borrow_mut();
            t.push(stub_handle);
        });
        unsafe {
            *handle = stub_handle;
        }
        CUfileError { err: 0, cu_err: 0 }
    }

    unsafe extern "C" fn deregister_stub(_handle: CUfileHandle) -> CUfileError {
        DEREGISTER_CALLS.with(|c| c.set(c.get() + 1));
        CUfileError { err: 0, cu_err: 0 }
    }

    unsafe extern "C" fn noop_read(
        _handle: CUfileHandle,
        _device_ptr: *mut c_void,
        _size: usize,
        _file_offset: i64,
        _device_offset: i64,
    ) -> i64 {
        0
    }

    unsafe extern "C" fn noop_buf_register(
        _device_ptr: *const c_void,
        _size: usize,
        _flags: i32,
    ) -> CUfileError {
        CUfileError { err: 0, cu_err: 0 }
    }

    unsafe extern "C" fn noop_buf_deregister(_device_ptr: *const c_void) -> CUfileError {
        CUfileError { err: 0, cu_err: 0 }
    }

    unsafe extern "C" fn noop_batch_io_setup(
        _batch_params: *mut c_void,
        _count: i32,
    ) -> CUfileError {
        CUfileError { err: 0, cu_err: 0 }
    }

    unsafe extern "C" fn noop_batch_io_submit(
        _batch_params: *mut c_void,
        _count: i32,
        _handle: *mut c_void,
        _flags: i32,
    ) -> CUfileError {
        CUfileError { err: 0, cu_err: 0 }
    }

    unsafe extern "C" fn noop_set_size(_size: usize) -> CUfileError {
        CUfileError { err: 0, cu_err: 0 }
    }

    unsafe extern "C" fn noop_batch_cancel(_batch_id: u64) -> CUfileError {
        CUfileError { err: 0, cu_err: 0 }
    }

    pub(crate) fn test_hardware(max_registered_memory: usize) -> CuFileHardware {
        REGISTER_CALLS.with(|c| c.set(0));
        DEREGISTER_CALLS.with(|c| c.set(0));

        let api = Api {
            driver_open: driver_open_stub,
            driver_close: driver_close_stub,
            get_version: None,
            handle_register: register_stub,
            handle_deregister: deregister_stub,
            read: noop_read,
            buf_register: noop_buf_register,
            buf_deregister: noop_buf_deregister,
            batch_io_setup: noop_batch_io_setup,
            batch_io_submit: noop_batch_io_submit,
            driver_set_max_pinned_mem_size: noop_set_size,
            driver_set_max_cache_size: noop_set_size,
            batch_io_cancel: noop_batch_cancel,
        };

        CuFileHardware {
            library: CuFileLibrary::from_api_for_tests(api),
            cache: RegistrationCache::new(max_registered_memory),
        }
    }

    #[test]
    fn registration_cache_evicts_least_recently_used_fd() {
        let mut hardware = test_hardware(GDS_ALIGNMENT * 2);

        hardware
            .cache
            .get_or_register_for_bytes(3, GDS_ALIGNMENT, &hardware.library)
            .unwrap_or_else(|e| panic!("fd 3 should register: {e}"));
        hardware
            .cache
            .get_or_register_for_bytes(4, GDS_ALIGNMENT, &hardware.library)
            .unwrap_or_else(|e| panic!("fd 4 should register: {e}"));
        hardware
            .cache
            .get_or_register_for_bytes(3, GDS_ALIGNMENT, &hardware.library)
            .unwrap_or_else(|e| panic!("fd 3 should be refreshed: {e}"));
        hardware
            .cache
            .get_or_register_for_bytes(5, GDS_ALIGNMENT, &hardware.library)
            .unwrap_or_else(|e| panic!("fd 5 should register after evicting fd 4: {e}"));

        assert!(hardware.cache.cached_handle(3).is_some());
        assert!(hardware.cache.cached_handle(4).is_none());
        assert!(hardware.cache.cached_handle(5).is_some());
        assert_eq!(hardware.cache.total_registered_memory, GDS_ALIGNMENT * 2);
        assert_eq!(REGISTER_CALLS.with(std::cell::Cell::get), 3);
        assert_eq!(DEREGISTER_CALLS.with(std::cell::Cell::get), 1);
    }

    #[test]
    fn cache_growth_evicts_other_lru_entries_before_expanding() {
        let mut hardware = test_hardware(GDS_ALIGNMENT * 2);

        hardware
            .cache
            .get_or_register_for_bytes(7, GDS_ALIGNMENT, &hardware.library)
            .unwrap_or_else(|e| panic!("fd 7 should register: {e}"));
        hardware
            .cache
            .get_or_register_for_bytes(8, GDS_ALIGNMENT, &hardware.library)
            .unwrap_or_else(|e| panic!("fd 8 should register: {e}"));
        hardware
            .cache
            .get_or_register_for_bytes(7, GDS_ALIGNMENT * 2, &hardware.library)
            .unwrap_or_else(|e| panic!("fd 7 should expand after evicting fd 8: {e}"));

        let fd7 = hardware
            .cache
            .cached_handle(7)
            .unwrap_or_else(|| panic!("fd 7 must remain cached"));
        assert_eq!(fd7.registered_bytes, GDS_ALIGNMENT * 2);
        assert!(hardware.cache.cached_handle(8).is_none());
        assert_eq!(hardware.cache.total_registered_memory, GDS_ALIGNMENT * 2);
        assert_eq!(DEREGISTER_CALLS.with(std::cell::Cell::get), 1);
    }

    #[test]
    fn evict_fd_releases_accounted_memory() {
        let mut hardware = test_hardware(GDS_ALIGNMENT * 4);

        hardware
            .cache
            .get_or_register_for_bytes(11, GDS_ALIGNMENT * 2, &hardware.library)
            .unwrap_or_else(|e| panic!("fd 11 should register: {e}"));
        assert_eq!(hardware.cache.total_registered_memory, GDS_ALIGNMENT * 2);

        let evicted = hardware
            .evict_fd(11)
            .unwrap_or_else(|e| panic!("eviction should succeed: {e}"));

        assert!(evicted);
        assert_eq!(hardware.cache.total_registered_memory, 0);
        assert!(hardware.cache.cached_handle(11).is_none());
        assert_eq!(DEREGISTER_CALLS.with(std::cell::Cell::get), 1);
    }
}
