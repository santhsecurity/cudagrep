//! Driver-level error types.

use std::os::fd::RawFd;

/// Errors that can occur while performing a registered read.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReadError {
    /// `cuFileHandleRegister` failed.
    Register {
        /// The file descriptor passed to the driver.
        fd: RawFd,
        /// The driver-provided error code.
        code: CuFileDriverError,
    },
    /// `cuFileRead` failed.
    Read {
        /// The file descriptor used for the transfer.
        fd: RawFd,
        /// The driver-provided error code.
        code: CuFileDriverError,
    },
    /// `cuFileHandleDeregister` failed during cleanup.
    Cleanup {
        /// The file descriptor being cleaned up.
        fd: RawFd,
        /// The driver-provided error code.
        code: CuFileDriverError,
    },
}

/// Known cuFile driver error codes mapped to Rust representation.
///
/// These errors are returned by various `libcufile` operations when they fail.
/// Use `CuFileDriverError::from_raw` to convert from raw i32 codes.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum CuFileDriverError {
    /// The CUDA driver returned a device-side failure code.
    #[error("CUDA driver error (status: {0})")]
    CudaDriverError(i32),
    /// The operating system reported a host-side failure code.
    #[error("OS error (status: {0})")]
    OsError(i32),
    /// The driver could not locate usable device memory.
    #[error("No Device memory found")]
    NoDeviceMemory,
    /// The supplied device pointer was rejected by the driver.
    #[error("Invalid Device Pointer")]
    InvalidDevicePointer,
    /// The requested operation is not supported by this driver stack.
    #[error("Operation not supported")]
    NotSupported,
    /// The driver returned an unmapped status code.
    #[error("Unknown error code {0}")]
    Unknown(i32),
}

impl CuFileDriverError {
    /// Maps a raw `libcufile` i32 error code to the typed `CuFileDriverError`.
    ///
    /// # Mapping
    ///
    /// | Code | Variant |
    /// |------|---------|
    /// | -1 | `OsError` |
    /// | -2 | `CudaDriverError` |
    /// | -3 | `NotSupported` |
    /// | -4 | `NoDeviceMemory` |
    /// | -5 | `InvalidDevicePointer` |
    /// | other | `Unknown` |
    ///
    /// # Examples
    ///
    /// ```
    /// use cudagrep::cufile::CuFileDriverError;
    ///
    /// assert!(matches!(CuFileDriverError::from_raw(-1), CuFileDriverError::OsError(_)));
    /// assert_eq!(CuFileDriverError::from_raw(-3), CuFileDriverError::NotSupported);
    /// assert_eq!(CuFileDriverError::from_raw(999), CuFileDriverError::Unknown(999));
    /// ```
    #[must_use]
    pub fn from_raw(code: i32) -> Self {
        match code {
            -1 => Self::OsError(code), // usually -1 means check errno, or CUDA driver error in context
            -2 => Self::CudaDriverError(code),
            -3 => Self::NotSupported,
            -4 => Self::NoDeviceMemory,
            -5 => Self::InvalidDevicePointer,
            _ => Self::Unknown(code),
        }
    }
}
