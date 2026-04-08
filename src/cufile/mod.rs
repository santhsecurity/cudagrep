//! Safe wrappers around the `libcufile` FFI surface.

pub mod error;
pub mod ffi;
pub mod library;
pub mod ops;

pub use error::{CuFileDriverError, ReadError};
pub use ffi::{
    CUfileBatchIOFlags, CUfileError, CUfileHandle, CUfileHandleType, CUfileIOOpType,
    CUfileIOParams, CUfileIOStatus,
};
pub(crate) use library::CuFileLibrary;
pub use ops::DevicePointer;

#[cfg(test)]
mod tests {
    #![allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::cufile::ffi::CUfileDescr;
    use crate::cufile::library::Api;
    use std::ffi::c_void;

    thread_local! {
        static DEREGISTER_CALLS: std::cell::Cell<i32> = const { std::cell::Cell::new(0) };
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
        unsafe {
            *handle = CUfileHandle(std::ptr::dangling_mut::<c_void>());
        }

        CUfileError { err: 0, cu_err: 0 }
    }

    unsafe extern "C" fn deregister_stub(_handle: CUfileHandle) -> CUfileError {
        DEREGISTER_CALLS.with(|c| c.set(c.get() + 1));
        CUfileError { err: 0, cu_err: 0 }
    }

    unsafe extern "C" fn read_stub(
        _handle: CUfileHandle,
        _device_ptr: *mut c_void,
        _size: usize,
        _file_offset: i64,
        _device_offset: i64,
    ) -> i64 {
        -5
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

    unsafe extern "C" fn noop_set_max_pinned(_size: usize) -> CUfileError {
        CUfileError { err: 0, cu_err: 0 }
    }

    unsafe extern "C" fn noop_set_max_cache(_size: usize) -> CUfileError {
        CUfileError { err: 0, cu_err: 0 }
    }

    unsafe extern "C" fn noop_batch_io_cancel(_batch_id: u64) -> CUfileError {
        CUfileError { err: 0, cu_err: 0 }
    }

    #[test]
    fn cufile_error_success_mapping_uses_os_error_field() {
        assert!(CUfileError {
            err: 0,
            cu_err: 700
        }
        .is_success());
        assert!(!CUfileError { err: -1, cu_err: 0 }.is_success());
    }

    #[test]
    fn registered_handle_drop_cleans_up_after_read_error() {
        DEREGISTER_CALLS.with(|c| c.set(0));
        let library = CuFileLibrary::from_api_for_tests(Api {
            driver_open: driver_open_stub,
            driver_close: driver_close_stub,
            get_version: None,
            handle_register: register_stub,
            handle_deregister: deregister_stub,
            read: read_stub,
            buf_register: noop_buf_register,
            buf_deregister: noop_buf_deregister,
            batch_io_setup: noop_batch_io_setup,
            batch_io_submit: noop_batch_io_submit,
            driver_set_max_pinned_mem_size: noop_set_max_pinned,
            driver_set_max_cache_size: noop_set_max_cache,
            batch_io_cancel: noop_batch_io_cancel,
        });

        let mut byte = 0_u8;
        let device_pointer =
            unsafe { DevicePointer::new(std::ptr::addr_of_mut!(byte).cast::<c_void>()) }
                .unwrap_or_else(|| panic!("stack pointer must be non-null"));

        let result = library.read_from_fd(3, device_pointer, 16, 0, 0);
        assert!(result.is_err(), "read stub should fail");
        let error = result.unwrap_err();

        assert_eq!(
            error,
            ReadError::Read {
                fd: 3,
                code: CuFileDriverError::InvalidDevicePointer
            }
        );
        assert_eq!(DEREGISTER_CALLS.with(std::cell::Cell::get), 1);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn loading_a_missing_library_returns_a_clear_error() {
        let result = CuFileLibrary::load_from_path("/definitely/missing/libcufile.so");
        let Err(error) = result else {
            panic!("test path should not exist");
        };
        assert!(error.to_string().contains("missing"));
    }
}
