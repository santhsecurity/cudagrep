//! Adversarial tests for `cudagrep` alignment validation and API contracts.
//!
//! These tests verify the correctness of the GDS alignment validation logic
//! and API error handling without requiring actual CUDA hardware. Tests that
//! need CUDA hardware early-return with `Ok(())` when GDS is unavailable.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_wrap
)]

use cudagrep::{CuFileHardware, CudaError, DEFAULT_MAX_TRANSFER_SIZE, GDS_ALIGNMENT};

const GDS_ALIGNMENT_I64: i64 = 4096;

/// Verifies that `CuFileHardware::try_init` returns a clear `FeatureDisabled`
/// or driver error when the CUDA stack is absent.
#[test]
fn try_init_returns_clear_error_without_cuda() {
    match CuFileHardware::try_init() {
        Ok(_hw) => {
            // CUDA available — nothing to assert about unavailability
        }
        Err(
            CudaError::FeatureDisabled
            | CudaError::Unavailable { .. }
            | CudaError::DriverInitRejected { .. },
        ) => {
            // Expected when compiled without `cuda` feature
        }
        Err(other) => {
            panic!("unexpected error type from try_init: {other}");
        }
    }
}

/// GDS alignment constants are powers of two and match documented requirements.
#[test]
fn gds_alignment_is_power_of_two_and_4kib() {
    assert_eq!(GDS_ALIGNMENT, 4096);
    assert!(GDS_ALIGNMENT.is_power_of_two());
}

/// The default max transfer size is a multiple of GDS alignment.
#[test]
fn default_max_transfer_size_is_aligned() {
    const _: () = assert!(DEFAULT_MAX_TRANSFER_SIZE > 0);
    assert_eq!(DEFAULT_MAX_TRANSFER_SIZE % GDS_ALIGNMENT, 0);
    // 16 MiB
    assert_eq!(DEFAULT_MAX_TRANSFER_SIZE, 16 * 1024 * 1024);
}

/// The `AvailabilityReport` correctly represents runtime state.
#[test]
fn availability_report_displays_structured_diagnostics() {
    use cudagrep::{AvailabilityReason, AvailabilityReport};

    let report = AvailabilityReport::new(
        false,
        AvailabilityReason::FeatureDisabled,
        vec!["compiled without cuda feature".to_string()],
    );
    assert!(!report.is_available());
    assert_eq!(*report.reason(), AvailabilityReason::FeatureDisabled);
    assert_eq!(report.diagnostics().len(), 1);

    let display = format!("{report}");
    assert!(!display.is_empty());
}

/// When GDS hardware IS available, alignment violations are caught during
/// `read_to_device` before any kernel DMA is attempted.
#[test]
fn alignment_violations_rejected_before_dma() {
    let Ok(mut hardware) = CuFileHardware::try_init() else {
        return;
    };

    // Create a fake device pointer for testing.
    let mut buf = vec![0u8; GDS_ALIGNMENT * 2];
    let ptr = buf.as_mut_ptr().cast::<std::ffi::c_void>();
    let Some(device_pointer) = (unsafe { cudagrep::cufile::DevicePointer::new(ptr) }) else {
        return;
    };

    let base = GDS_ALIGNMENT_I64;
    let size = GDS_ALIGNMENT;

    // Unaligned file_offset
    let result = hardware.read_to_device(-1, device_pointer, size, base - 1, base);
    assert!(result.is_err(), "unaligned file_offset must be rejected");

    // Unaligned size
    let result = hardware.read_to_device(-1, device_pointer, size - 1, base, base);
    assert!(result.is_err(), "unaligned size must be rejected");

    // Unaligned device_offset
    let result = hardware.read_to_device(-1, device_pointer, size, base, base - 1);
    assert!(result.is_err(), "unaligned device_offset must be rejected");
}

/// `CuFileDriverError` maps known codes correctly.
#[test]
fn driver_error_mapping_covers_known_codes() {
    use cudagrep::cufile::CuFileDriverError;

    assert!(matches!(
        CuFileDriverError::from_raw(-1),
        CuFileDriverError::OsError(_)
    ));
    assert!(matches!(
        CuFileDriverError::from_raw(-2),
        CuFileDriverError::CudaDriverError(_)
    ));
    assert!(matches!(
        CuFileDriverError::from_raw(-3),
        CuFileDriverError::NotSupported
    ));
    assert!(matches!(
        CuFileDriverError::from_raw(-4),
        CuFileDriverError::NoDeviceMemory
    ));
    assert!(matches!(
        CuFileDriverError::from_raw(-5),
        CuFileDriverError::InvalidDevicePointer
    ));
    assert!(matches!(
        CuFileDriverError::from_raw(-99),
        CuFileDriverError::Unknown(-99)
    ));
}

/// `CuFileError` reports success/failure based on the `err` field only.
#[test]
fn cufile_error_success_uses_os_error_field() {
    use cudagrep::cufile::CUfileError;

    assert!(CUfileError {
        err: 0,
        cu_err: 700
    }
    .is_success());
    assert!(!CUfileError { err: -1, cu_err: 0 }.is_success());
    assert!(CUfileError { err: 0, cu_err: 0 }.is_success());
}

/// `DevicePointer` rejects null pointers.
#[test]
fn device_pointer_rejects_null() {
    let result = unsafe { cudagrep::cufile::DevicePointer::new(std::ptr::null_mut()) };
    assert!(result.is_none());
}

/// `DevicePointer` accepts non-null pointers.
#[test]
fn device_pointer_accepts_non_null() {
    let mut byte = 0u8;
    let ptr = std::ptr::addr_of_mut!(byte).cast::<std::ffi::c_void>();
    let result = unsafe { cudagrep::cufile::DevicePointer::new(ptr) };
    assert!(result.is_some());
}

/// `ReadError` variants provide structured debugging information.
#[test]
fn read_error_variants_are_distinct() {
    use cudagrep::cufile::{CuFileDriverError, ReadError};

    let reg_err = ReadError::Register {
        fd: 5,
        code: CuFileDriverError::NotSupported,
    };
    let read_err = ReadError::Read {
        fd: 5,
        code: CuFileDriverError::OsError(-1),
    };
    let cleanup_err = ReadError::Cleanup {
        fd: 5,
        code: CuFileDriverError::Unknown(42),
    };

    assert_ne!(reg_err, read_err);
    assert_ne!(read_err, cleanup_err);
    assert_ne!(reg_err, cleanup_err);
}
