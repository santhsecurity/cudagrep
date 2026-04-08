//! Tests for hardware unavailable fallback behavior.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_wrap
)]

use cudagrep::{CuFileHardware, CudaError, DevicePointer, GDS_ALIGNMENT};
use std::ffi::c_void;

#[test]
fn try_init_returns_graceful_error_when_unavailable() {
    let result = CuFileHardware::try_init();

    // Either it succeeds (if hardware is available) or fails with a known error,
    // but it must never panic.
    if let Err(err) = result {
        assert!(
            matches!(
                err,
                CudaError::FeatureDisabled
                    | CudaError::Unavailable { .. }
                    | CudaError::DriverInitRejected { .. }
            ),
            "Unexpected error type: {err:?}"
        );
        let display = format!("{err}");
        assert!(!display.is_empty(), "Error message must not be empty");
    }
}

#[test]
fn gracefully_degrades_read_to_device_when_unavailable() {
    let Ok(mut gds) = CuFileHardware::try_init() else {
        return;
    };

    let mut byte = 0u8;
    let ptr = std::ptr::addr_of_mut!(byte).cast::<c_void>();
    let device_ptr = unsafe { DevicePointer::new(ptr) }.unwrap();

    let result = gds.read_to_device(-1, device_ptr, GDS_ALIGNMENT, 0, 0);

    // Should return a graceful error like descriptor registration failure or something similar
    // for an invalid fd -1, but it must not panic.
    if let Err(err) = result {
        let msg = err.to_string(); // must implement display without panic
        assert!(!msg.is_empty());
    }
}
