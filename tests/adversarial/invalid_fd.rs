//! Tests for invalid file descriptor handling.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_wrap
)]

use cudagrep::{CuFileHardware, CudaError, DevicePointer, GDS_ALIGNMENT};
use std::ffi::c_void;
use std::os::fd::AsRawFd;

#[test]
fn read_to_device_with_negative_fd_returns_error() {
    let mut gds = match CuFileHardware::try_init() {
        Ok(gds) => gds,
        Err(_) => return,
    };

    let mut byte = 0u8;
    let ptr = std::ptr::addr_of_mut!(byte).cast::<c_void>();
    let device_ptr = unsafe { DevicePointer::new(ptr) }.unwrap();

    let result = gds.read_to_device(-1, device_ptr, GDS_ALIGNMENT, 0, 0);
    assert!(result.is_err(), "Reading from negative fd must return an error");
}

#[test]
fn read_to_device_with_huge_fd_returns_error() {
    let mut gds = match CuFileHardware::try_init() {
        Ok(gds) => gds,
        Err(_) => return,
    };

    let mut byte = 0u8;
    let ptr = std::ptr::addr_of_mut!(byte).cast::<c_void>();
    let device_ptr = unsafe { DevicePointer::new(ptr) }.unwrap();

    let result = gds.read_to_device(i32::MAX, device_ptr, GDS_ALIGNMENT, 0, 0);
    assert!(result.is_err(), "Reading from huge fd must return an error");
}

#[test]
fn read_to_device_with_closed_fd_returns_error() {
    let mut gds = match CuFileHardware::try_init() {
        Ok(gds) => gds,
        Err(_) => return,
    };

    let mut byte = 0u8;
    let ptr = std::ptr::addr_of_mut!(byte).cast::<c_void>();
    let device_ptr = unsafe { DevicePointer::new(ptr) }.unwrap();

    // Create a temporary file and immediately close it
    let fd = {
        let file = tempfile::tempfile().unwrap();
        file.as_raw_fd()
    }; // file is closed here because it goes out of scope

    let result = gds.read_to_device(fd, device_ptr, GDS_ALIGNMENT, 0, 0);
    assert!(result.is_err(), "Reading from closed fd must return an error");
}

#[test]
fn evict_fd_with_invalid_fds_returns_ok_false() {
    let mut gds = match CuFileHardware::try_init() {
        Ok(gds) => gds,
        Err(_) => return,
    };

    assert_eq!(gds.evict_fd(-1).unwrap(), false);
    assert_eq!(gds.evict_fd(i32::MAX).unwrap(), false);
}
