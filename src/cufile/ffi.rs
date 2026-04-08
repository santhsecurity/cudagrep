//! Raw FFI types and bindings for `libcufile`.

use std::ffi::c_void;
use std::os::fd::RawFd;

/// Opaque cuFile handle returned by `cuFileHandleRegister`.
///
/// This is an opaque pointer type wrapping the underlying driver handle.
/// Cloning copies the pointer value (the handle is reference-counted by the driver).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CUfileHandle(pub *mut c_void);

/// Handle type used by `CUfileDescr`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub enum CUfileHandleType {
    /// A Linux file descriptor.
    OpaqueFd = 1,
}

/// File descriptor descriptor passed to `cuFileHandleRegister`.
///
/// # Safety
///
/// The `fd` field must be a valid, open file descriptor for the duration of
/// the registration. Closing the fd without deregistering first is undefined.
#[repr(C)]
#[derive(Debug)]
pub struct CUfileDescr {
    /// The type of handle stored in `fd`.
    pub handle_type: CUfileHandleType,
    /// The Linux file descriptor to register.
    pub fd: RawFd,
}

/// Error struct returned by several `libcufile` entry points.
///
/// The `err` field contains the primary error code. When `err == 0`, the
/// operation succeeded regardless of the `cu_err` field.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CUfileError {
    /// Operating-system level error code.
    pub err: i32,
    /// CUDA-level error code.
    pub cu_err: i32,
}

impl CUfileError {
    /// Returns whether the operating-system portion of the status reports success.
    ///
    /// Success is defined as `err == 0`. The `cu_err` field is ignored.
    ///
    /// # Examples
    ///
    /// ```
    /// use cudagrep::cufile::CUfileError;
    ///
    /// assert!(CUfileError { err: 0, cu_err: 700 }.is_success());
    /// assert!(!CUfileError { err: -1, cu_err: 0 }.is_success());
    /// ```
    #[must_use]
    pub fn is_success(self) -> bool {
        self.err == 0
    }
}

/// IO operation type for batch operations.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub enum CUfileIOOpType {
    /// Read operation.
    Read = 0,
    /// Write operation.
    Write = 1,
}

/// IO parameters for a single entry in a batch operation.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CUfileIOParams {
    /// Registered cuFile handle for this operation.
    pub handle: CUfileHandle,
    /// Operation type (read or write).
    pub op_type: CUfileIOOpType,
    /// Source/destination device pointer.
    pub device_ptr: *mut c_void,
    /// Byte offset in file.
    pub file_offset: i64,
    /// Byte offset in device memory.
    pub device_offset: i64,
    /// Number of bytes to transfer.
    pub size: usize,
}

/// Status for a completed batch IO operation.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CUfileIOStatus {
    /// Number of bytes transferred, or negative error code.
    pub bytes_transferred: i64,
    /// Detailed error code.
    pub status: CUfileError,
}

/// Batch IO submission flags.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CUfileBatchIOFlags {
    /// Raw bitmask representing the submission flags.
    pub bits: i32,
}

impl CUfileBatchIOFlags {
    /// Create empty flags (no special behavior).
    ///
    /// # Examples
    ///
    /// ```
    /// use cudagrep::cufile::CUfileBatchIOFlags;
    ///
    /// let flags = CUfileBatchIOFlags::empty();
    /// assert_eq!(flags.bits, 0);
    /// ```
    #[must_use]
    pub const fn empty() -> Self {
        Self { bits: 0 }
    }
}

pub(crate) type CuFileDriverOpenFn = unsafe extern "C" fn() -> CUfileError;
pub(crate) type CuFileDriverCloseFn = unsafe extern "C" fn() -> CUfileError;
pub(crate) type CuFileHandleRegisterFn =
    unsafe extern "C" fn(handle: *mut CUfileHandle, descr: *mut CUfileDescr) -> CUfileError;
pub(crate) type CuFileHandleDeregisterFn =
    unsafe extern "C" fn(handle: CUfileHandle) -> CUfileError;
pub(crate) type CuFileReadFn = unsafe extern "C" fn(
    handle: CUfileHandle,
    device_ptr: *mut c_void,
    size: usize,
    file_offset: i64,
    device_offset: i64,
) -> i64;
pub(crate) type CuFileBufRegisterFn =
    unsafe extern "C" fn(device_ptr: *const c_void, size: usize, flags: i32) -> CUfileError;
pub(crate) type CuFileBufDeregisterFn =
    unsafe extern "C" fn(device_ptr: *const c_void) -> CUfileError;
pub(crate) type CuFileBatchIOSetUpFn =
    unsafe extern "C" fn(batch_params: *mut c_void, count: i32) -> CUfileError;
pub(crate) type CuFileBatchIOSubmitFn = unsafe extern "C" fn(
    batch_params: *mut c_void,
    count: i32,
    handle: *mut c_void,
    flags: i32,
) -> CUfileError;
pub(crate) type CuFileDriverSetMaxPinnedMemSizeFn =
    unsafe extern "C" fn(size: usize) -> CUfileError;
pub(crate) type CuFileDriverSetMaxCacheSizeFn = unsafe extern "C" fn(size: usize) -> CUfileError;
pub(crate) type CuFileBatchIOCancelFn = unsafe extern "C" fn(batch_id: u64) -> CUfileError;
pub(crate) type CuFileGetVersionFn = unsafe extern "C" fn(version: *mut i32) -> CUfileError;
