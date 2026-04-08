//! Batch IO operations for `GPUDirect` Storage.

use std::os::fd::RawFd;

use crate::cufile::DevicePointer;

/// A single read operation for use with [`CuFileHardware::read_batch`](crate::CuFileHardware::read_batch).
///
/// # Examples
///
/// ```no_run
/// use cudagrep::{ReadOp, GDS_ALIGNMENT};
/// use cudagrep::cufile::DevicePointer;
/// use std::ffi::c_void;
///
/// # fn example(ptr: *mut c_void) -> Option<()> {
/// let device_pointer = unsafe { DevicePointer::new(ptr)? };
/// let op = ReadOp {
///     fd: 3,
///     device_pointer,
///     size: GDS_ALIGNMENT,
///     file_offset: 0,
///     device_offset: 0,
/// };
/// # Some(())
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ReadOp {
    /// Linux file descriptor backing the transfer.
    pub fd: RawFd,
    /// Target GPU device pointer.
    pub device_pointer: DevicePointer,
    /// Number of bytes to transfer.
    pub size: usize,
    /// Byte offset within the file.
    pub file_offset: i64,
    /// Byte offset within the device allocation.
    pub device_offset: i64,
}
