use std::marker::PhantomData;

use libloading::Library;

use crate::cufile::error::CuFileDriverError;
use crate::cufile::library::{Api, CuFileLibrary, SUPPORTED_CUFILE_API_MAJOR_VERSION};

impl CuFileLibrary {
    /// Loads `libcufile.so.0` from the standard dynamic linker search path.
    ///
    /// # Errors
    ///
    /// Returns an error when the shared object cannot be loaded or a required
    /// symbol is missing.
    pub fn load() -> Result<Self, libloading::Error> {
        Self::load_from_path("libcufile.so.0")
    }

    /// Loads `libcufile` from a caller-provided path.
    ///
    /// # Errors
    ///
    /// Returns an error when the shared object cannot be loaded or a required
    /// symbol is missing.
    pub fn load_from_path(path: &str) -> Result<Self, libloading::Error> {
        // SAFETY: Loading a shared object is inherently unsafe because the process
        // trusts the library's ABI. This wrapper immediately resolves the exact
        // symbols it needs and retains the `Library` for the lifetime of the
        // copied function pointers.
        unsafe {
            let library = Library::new(path)?;
            let api = Api {
                driver_open: load_symbol(&library, b"cuFileDriverOpen")?,
                driver_close: load_symbol(&library, b"cuFileDriverClose")?,
                get_version: load_optional_symbol(&library, b"cuFileGetVersion"),
                handle_register: load_symbol(&library, b"cuFileHandleRegister")?,
                handle_deregister: load_symbol(&library, b"cuFileHandleDeregister")?,
                read: load_symbol(&library, b"cuFileRead")?,
                buf_register: load_symbol(&library, b"cuFileBufRegister")?,
                buf_deregister: load_symbol(&library, b"cuFileBufDeregister")?,
                batch_io_setup: load_symbol(&library, b"cuFileBatchIOSetUp")?,
                batch_io_submit: load_symbol(&library, b"cuFileBatchIOSubmit")?,
                driver_set_max_pinned_mem_size: load_symbol(
                    &library,
                    b"cuFileDriverSetMaxPinnedMemSize",
                )?,
                driver_set_max_cache_size: load_symbol(&library, b"cuFileDriverSetMaxCacheSize")?,
                batch_io_cancel: load_symbol(&library, b"cuFileBatchIOCancel")?,
            };
            log_version_check_result(check_cufile_version(&api));

            Ok(Self {
                _library: Some(library),
                api,
                _not_send_sync: PhantomData,
            })
        }
    }

    /// Opens the global `libcufile` driver session.
    ///
    /// # Errors
    ///
    /// Returns a typed `CuFileDriverError` when `cuFileDriverOpen` fails.
    pub fn driver_open(&self) -> Result<(), CuFileDriverError> {
        let result = unsafe { (self.api.driver_open)() };
        if result.is_success() {
            Ok(())
        } else {
            Err(CuFileDriverError::from_raw(result.err))
        }
    }

    /// Closes the global `libcufile` driver session.
    ///
    /// # Errors
    ///
    /// Returns a typed `CuFileDriverError` when `cuFileDriverClose` fails.
    pub fn driver_close(&self) -> Result<(), CuFileDriverError> {
        let result = unsafe { (self.api.driver_close)() };
        if result.is_success() {
            Ok(())
        } else {
            Err(CuFileDriverError::from_raw(result.err))
        }
    }

    /// Set maximum pinned memory allocated by the cuFile driver.
    ///
    /// # Errors
    ///
    /// Returns a typed `CuFileDriverError` when the driver rejects the limit.
    pub fn set_max_pinned_mem_size(&self, size: usize) -> Result<(), CuFileDriverError> {
        let result = unsafe { (self.api.driver_set_max_pinned_mem_size)(size) };
        if result.is_success() {
            Ok(())
        } else {
            Err(CuFileDriverError::from_raw(result.err))
        }
    }

    /// Set maximum buffer cache size for the cuFile driver.
    ///
    /// # Errors
    ///
    /// Returns a typed `CuFileDriverError` when the driver rejects the limit.
    pub fn set_max_cache_size(&self, size: usize) -> Result<(), CuFileDriverError> {
        let result = unsafe { (self.api.driver_set_max_cache_size)(size) };
        if result.is_success() {
            Ok(())
        } else {
            Err(CuFileDriverError::from_raw(result.err))
        }
    }
}

unsafe fn load_symbol<T>(library: &Library, symbol: &[u8]) -> Result<T, libloading::Error>
where
    T: Copy,
{
    let symbol = unsafe { library.get::<T>(symbol)? };
    Ok(*symbol)
}

unsafe fn load_optional_symbol<T>(library: &Library, symbol: &[u8]) -> Option<T>
where
    T: Copy,
{
    let symbol = unsafe { library.get::<T>(symbol).ok()? };
    Some(*symbol)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CuFileVersion {
    pub(crate) major: i32,
    pub(crate) minor: i32,
}

impl std::fmt::Display for CuFileVersion {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}.{}", self.major, self.minor)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CuFileVersionCheck {
    SymbolUnavailable,
    Compatible {
        loaded: CuFileVersion,
    },
    Mismatch {
        loaded: CuFileVersion,
        expected_major: i32,
    },
    InvalidEncoding {
        raw: i32,
    },
    QueryFailed {
        code: CuFileDriverError,
    },
}

pub(crate) fn check_cufile_version(api: &Api) -> CuFileVersionCheck {
    let Some(get_version) = api.get_version else {
        return CuFileVersionCheck::SymbolUnavailable;
    };

    let mut raw_version = 0_i32;
    let status = unsafe { get_version(std::ptr::addr_of_mut!(raw_version)) };
    if !status.is_success() {
        return CuFileVersionCheck::QueryFailed {
            code: CuFileDriverError::from_raw(status.err),
        };
    }

    let Some(loaded) = decode_cufile_version(raw_version) else {
        return CuFileVersionCheck::InvalidEncoding { raw: raw_version };
    };

    if loaded.major == SUPPORTED_CUFILE_API_MAJOR_VERSION {
        CuFileVersionCheck::Compatible { loaded }
    } else {
        CuFileVersionCheck::Mismatch {
            loaded,
            expected_major: SUPPORTED_CUFILE_API_MAJOR_VERSION,
        }
    }
}

pub(crate) fn decode_cufile_version(raw: i32) -> Option<CuFileVersion> {
    if raw < 0 {
        return None;
    }

    let major = raw / 1000;
    let minor = (raw % 1000) / 10;
    Some(CuFileVersion { major, minor })
}

pub(crate) fn log_version_check_result(check: CuFileVersionCheck) {
    match check {
        CuFileVersionCheck::SymbolUnavailable | CuFileVersionCheck::Compatible { .. } => {}
        CuFileVersionCheck::Mismatch {
            loaded,
            expected_major,
        } => {
            tracing::warn!(
                loaded_version = %loaded,
                expected_major,
                "loaded libcufile major version differs from the supported API major version"
            );
        }
        CuFileVersionCheck::InvalidEncoding { raw } => {
            tracing::warn!(
                raw_version = raw,
                "cuFileGetVersion returned an invalid version encoding; skipping compatibility check"
            );
        }
        CuFileVersionCheck::QueryFailed { code } => {
            tracing::warn!(
                ?code,
                "cuFileGetVersion failed; skipping libcufile compatibility check"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::cufile::ffi::{CUfileDescr, CUfileError, CUfileHandle};
    use std::ffi::c_void;

    unsafe extern "C" fn version_match_stub(version: *mut i32) -> CUfileError {
        unsafe {
            *version = 1160;
        }
        CUfileError { err: 0, cu_err: 0 }
    }

    unsafe extern "C" fn version_mismatch_stub(version: *mut i32) -> CUfileError {
        unsafe {
            *version = 2010;
        }
        CUfileError { err: 0, cu_err: 0 }
    }

    unsafe extern "C" fn version_failure_stub(_version: *mut i32) -> CUfileError {
        CUfileError { err: -3, cu_err: 0 }
    }

    fn minimal_api(get_version: Option<crate::cufile::ffi::CuFileGetVersionFn>) -> Api {
        unsafe extern "C" fn noop_driver() -> CUfileError {
            CUfileError { err: 0, cu_err: 0 }
        }

        unsafe extern "C" fn noop_register(
            _handle: *mut CUfileHandle,
            _descr: *mut CUfileDescr,
        ) -> CUfileError {
            CUfileError { err: 0, cu_err: 0 }
        }

        unsafe extern "C" fn noop_deregister(_handle: CUfileHandle) -> CUfileError {
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

        Api {
            driver_open: noop_driver,
            driver_close: noop_driver,
            get_version,
            handle_register: noop_register,
            handle_deregister: noop_deregister,
            read: noop_read,
            buf_register: noop_buf_register,
            buf_deregister: noop_buf_deregister,
            batch_io_setup: noop_batch_io_setup,
            batch_io_submit: noop_batch_io_submit,
            driver_set_max_pinned_mem_size: noop_set_size,
            driver_set_max_cache_size: noop_set_size,
            batch_io_cancel: noop_batch_cancel,
        }
    }

    #[test]
    fn decode_cufile_version_uses_documented_encoding() {
        assert_eq!(
            decode_cufile_version(1160),
            Some(CuFileVersion {
                major: 1,
                minor: 16
            })
        );
        assert_eq!(
            decode_cufile_version(1070),
            Some(CuFileVersion { major: 1, minor: 7 })
        );
        assert_eq!(decode_cufile_version(-1), None);
    }

    #[test]
    fn version_check_accepts_supported_major_version() {
        let check = check_cufile_version(&minimal_api(Some(version_match_stub)));
        assert_eq!(
            check,
            CuFileVersionCheck::Compatible {
                loaded: CuFileVersion {
                    major: 1,
                    minor: 16
                },
            }
        );
    }

    #[test]
    fn version_check_reports_major_version_mismatch() {
        let check = check_cufile_version(&minimal_api(Some(version_mismatch_stub)));
        assert_eq!(
            check,
            CuFileVersionCheck::Mismatch {
                loaded: CuFileVersion { major: 2, minor: 1 },
                expected_major: SUPPORTED_CUFILE_API_MAJOR_VERSION,
            }
        );
    }

    #[test]
    fn version_check_propagates_driver_failures() {
        let check = check_cufile_version(&minimal_api(Some(version_failure_stub)));
        assert_eq!(
            check,
            CuFileVersionCheck::QueryFailed {
                code: CuFileDriverError::NotSupported,
            }
        );
    }
}
