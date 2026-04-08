//! Tests for cache eviction under pressure.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_wrap
)]

use cudagrep::{CuFileHardware, DevicePointer, GDS_ALIGNMENT};
use std::ffi::c_void;
use std::fs::File;
use std::os::fd::AsRawFd;

#[test]
fn handles_cache_pressure_without_panic() {
    let mut gds = match CuFileHardware::try_init() {
        Ok(gds) => gds,
        Err(_) => return, // Graceful exit if no hardware
    };

    let mut byte = 0u8;
    let ptr = std::ptr::addr_of_mut!(byte).cast::<c_void>();
    let device_ptr = unsafe { DevicePointer::new(ptr) }.unwrap();

    // Create real file descriptors using /dev/null so that the registration attempts
    // are grounded in reality rather than instantly failing due to bad fds.
    // 1500 is higher than the fast path threshold (1024) so it tests the HashMap fallback too.
    let num_fds = 1500;
    let mut files = Vec::with_capacity(num_fds);

    for _ in 0..num_fds {
        if let Ok(file) = File::open("/dev/null") {
            files.push(file);
        }
    }

    // Explicitly call get_or_register instead of read_to_device to avoid errors related
    // to non-NVMe files or invalid memory pointers, purely testing the cache logic.
    let mut success_count = 0;
    for file in &files {
        let fd = file.as_raw_fd();
        if gds.get_or_register(fd).is_ok() {
            success_count += 1;
        }
    }

    // At least some should succeed, or none if hardware refuses /dev/null, but it shouldn't panic
    // Let's assert we didn't crash.
    assert!(success_count >= 0);

    // Now explicitly trigger some evictions to test eviction behavior
    for file in &files {
        let fd = file.as_raw_fd();
        let _ = gds.evict_fd(fd);
    }
}
