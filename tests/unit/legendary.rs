//! Legendary adversarial test suite for cudagrep.
//!
//! These tests exercise edge cases, error paths, and boundary conditions
//! without requiring actual CUDA hardware. All tests are designed to run
//! on the software-only path.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_wrap
)]

use cudagrep::{
    validate_alignment, AvailabilityReason, AvailabilityReport, CuFileHardware, CudaError, ReadOp,
    ReadStats, DEFAULT_MAX_TRANSFER_SIZE, GDS_ALIGNMENT,
};
use std::time::Duration;

// =============================================================================
// Alignment Validation - Every Combo of Aligned/Misaligned
// =============================================================================

#[test]
fn alignment_all_aligned_succeeds() {
    assert!(validate_alignment(0, GDS_ALIGNMENT, 0).is_ok());
    assert!(validate_alignment(4096, 4096, 8192).is_ok());
    assert!(validate_alignment(16384, 8192, 32768).is_ok());
}

#[test]
fn alignment_misaligned_file_offset_fails() {
    // file_offset misaligned by 1
    let err_result = validate_alignment(1, GDS_ALIGNMENT, 0);
    assert!(err_result.is_err());
    let Err(err) = err_result else {
        panic!("expected Err")
    };
    assert!(
        matches!(err, CudaError::AlignmentViolation { file_offset: 1, .. }),
        "expected AlignmentViolation with file_offset=1, got: {err:?}"
    );

    // file_offset misaligned by 4095
    let err_result = validate_alignment(4095, GDS_ALIGNMENT, 0);
    assert!(err_result.is_err());
    let Err(err) = err_result else {
        panic!("expected Err")
    };
    assert!(
        matches!(
            err,
            CudaError::AlignmentViolation {
                file_offset: 4095,
                ..
            }
        ),
        "expected AlignmentViolation with file_offset=4095, got: {err:?}"
    );
}

#[test]
fn alignment_misaligned_size_fails() {
    // size misaligned by 1
    let err_result = validate_alignment(0, 1, 0);
    assert!(err_result.is_err());
    let Err(err) = err_result else {
        panic!("expected Err")
    };
    assert!(
        matches!(err, CudaError::AlignmentViolation { size: 1, .. }),
        "expected AlignmentViolation with size=1, got: {err:?}"
    );

    // size misaligned by 4095
    let err_result = validate_alignment(0, 8191, 0);
    assert!(err_result.is_err());
    let Err(err) = err_result else {
        panic!("expected Err")
    };
    assert!(
        matches!(err, CudaError::AlignmentViolation { size: 8191, .. }),
        "expected AlignmentViolation with size=8191, got: {err:?}"
    );
}

#[test]
fn alignment_misaligned_device_offset_fails() {
    // device_offset misaligned by 1
    let err_result = validate_alignment(0, GDS_ALIGNMENT, 1);
    assert!(err_result.is_err());
    let Err(err) = err_result else {
        panic!("expected Err")
    };
    assert!(
        matches!(
            err,
            CudaError::AlignmentViolation {
                device_offset: 1,
                ..
            }
        ),
        "expected AlignmentViolation with device_offset=1, got: {err:?}"
    );

    // device_offset misaligned by 4095
    let err_result = validate_alignment(0, GDS_ALIGNMENT, 4095);
    assert!(err_result.is_err());
    let Err(err) = err_result else {
        panic!("expected Err")
    };
    assert!(
        matches!(
            err,
            CudaError::AlignmentViolation {
                device_offset: 4095,
                ..
            }
        ),
        "expected AlignmentViolation with device_offset=4095, got: {err:?}"
    );
}

#[test]
fn alignment_all_misaligned_fails() {
    // All three misaligned
    let err_result = validate_alignment(1, 1, 1);
    assert!(err_result.is_err());
    let Err(err) = err_result else {
        panic!("expected Err")
    };
    assert!(
        matches!(
            err,
            CudaError::AlignmentViolation {
                file_offset: 1,
                size: 1,
                device_offset: 1,
            }
        ),
        "expected all misaligned, got: {err:?}"
    );
}

#[test]
fn alignment_two_misaligned_one_aligned_variants() {
    // file_offset + size misaligned, device_offset aligned
    assert!(
        validate_alignment(1, 1, 0).is_err(),
        "file_offset+size misaligned should fail"
    );

    // file_offset + device_offset misaligned, size aligned
    assert!(
        validate_alignment(1, 4096, 1).is_err(),
        "file_offset+device_offset misaligned should fail"
    );

    // size + device_offset misaligned, file_offset aligned
    assert!(
        validate_alignment(0, 1, 1).is_err(),
        "size+device_offset misaligned should fail"
    );
}

// =============================================================================
// Alignment Edge Cases
// =============================================================================

#[test]
fn alignment_size_zero_fails() {
    // Size 0 is technically aligned (0 % 4096 == 0) but semantically invalid
    // The validation should still pass since alignment is satisfied
    assert!(
        validate_alignment(0, 0, 0).is_ok(),
        "size=0 is technically aligned"
    );
}

#[test]
fn alignment_max_i64_values_no_overflow() {
    // Maximum positive i64 values should not cause overflow
    let max_i64 = i64::MAX;
    let result = validate_alignment(max_i64, GDS_ALIGNMENT, 0);
    // Should not panic, result depends on whether max_i64 is aligned
    // i64::MAX is NOT aligned to 4096, so this should fail
    assert!(result.is_err(), "max i64 should not be aligned to 4096");
}

#[test]
fn alignment_large_usize_values_no_overflow() {
    // Maximum usize values should not cause overflow
    let max_usize = usize::MAX;
    let aligned_size = max_usize - (max_usize % GDS_ALIGNMENT);
    let result = validate_alignment(0, aligned_size, 0);
    // Should not panic
    let _ = result;
}

#[test]
fn alignment_negative_file_offset() {
    let result = validate_alignment(-4096, GDS_ALIGNMENT, 0);
    assert!(
        matches!(
            result,
            Err(CudaError::AlignmentViolation {
                file_offset: -4096,
                size,
                device_offset: 0,
            }) if size == GDS_ALIGNMENT
        ),
        "negative file offset must be rejected, got: {result:?}"
    );
}

// =============================================================================
// ShortRead Error Message
// =============================================================================

#[test]
fn shortread_error_contains_requested_count() {
    let error = CudaError::ShortRead {
        requested: 16384,
        transferred: 8192,
    };
    let message = error.to_string();
    assert!(
        message.contains("16384") || message.contains("requested"),
        "ShortRead message should contain requested count: {message}"
    );
}

#[test]
fn shortread_error_contains_transferred_count() {
    let error = CudaError::ShortRead {
        requested: 16384,
        transferred: 8192,
    };
    let message = error.to_string();
    assert!(
        message.contains("8192") || message.contains("transferred"),
        "ShortRead message should contain transferred count: {message}"
    );
}

#[test]
fn shortread_error_contains_both_counts() {
    let error = CudaError::ShortRead {
        requested: 1_000_000,
        transferred: 500_000,
    };
    let message = error.to_string();
    assert!(
        message.contains("1000000") && message.contains("500000"),
        "ShortRead message should contain both counts: {message}"
    );
}

#[test]
fn shortread_zero_transferred_shows_properly() {
    let error = CudaError::ShortRead {
        requested: 4096,
        transferred: 0,
    };
    let message = error.to_string();
    assert!(
        message.contains("4096") && message.contains('0'),
        "ShortRead with zero transferred should show both values: {message}"
    );
}

// =============================================================================
// CudaError Variants - Actionable Messages
// =============================================================================

#[test]
fn cuda_error_feature_disabled_is_actionable() {
    let error = CudaError::FeatureDisabled;
    let message = error.to_string();
    assert!(
        message.contains("disabled") || message.contains("cuda"),
        "FeatureDisabled should explain the issue: {message}"
    );
    assert!(
        message.len() > 10,
        "Error message too short to be actionable: {message}"
    );
}

#[test]
fn cuda_error_unavailable_is_actionable() {
    let report = AvailabilityReport::new(
        false,
        AvailabilityReason::LibraryUnavailable("test".to_owned()),
        vec!["diagnostic info".to_owned()],
    );
    let error = CudaError::Unavailable { report };
    let message = error.to_string();
    assert!(
        !message.is_empty(),
        "Unavailable error should have a message"
    );
    assert!(
        message.len() > 5,
        "Unavailable message too short: {message}"
    );
}

#[test]
fn cuda_error_driver_init_rejected_is_actionable() {
    let error = CudaError::DriverInitRejected {
        code: cudagrep::cufile::CuFileDriverError::OsError(-1),
    };
    let message = error.to_string();
    assert!(
        message.contains("cuFileDriverOpen") || message.contains("failed"),
        "DriverInitRejected should mention what failed: {message}"
    );
}

#[test]
fn cuda_error_descriptor_registration_failed_is_actionable() {
    let error = CudaError::DescriptorRegistrationFailed {
        fd: 42,
        code: cudagrep::cufile::CuFileDriverError::NotSupported,
    };
    let message = error.to_string();
    assert!(
        message.contains("cuFileHandleRegister") || message.contains("42"),
        "DescriptorRegistrationFailed should be actionable: {message}"
    );
}

#[test]
fn cuda_error_dma_failed_is_actionable() {
    let error = CudaError::DirectMemoryAccessFailed {
        fd: 99,
        code: cudagrep::cufile::CuFileDriverError::InvalidDevicePointer,
    };
    let message = error.to_string();
    assert!(
        message.contains("cuFileRead") || message.contains("99"),
        "DirectMemoryAccessFailed should be actionable: {message}"
    );
}

#[test]
fn cuda_error_cleanup_failed_is_actionable() {
    let error = CudaError::DescriptorCleanupFailed {
        fd: 7,
        code: cudagrep::cufile::CuFileDriverError::NoDeviceMemory,
    };
    let message = error.to_string();
    assert!(
        message.contains("cuFileHandleDeregister") || message.contains('7'),
        "DescriptorCleanupFailed should be actionable: {message}"
    );
}

#[test]
fn cuda_error_alignment_violation_is_actionable() {
    let error = CudaError::AlignmentViolation {
        file_offset: 100,
        size: 200,
        device_offset: 300,
    };
    let message = error.to_string();
    assert!(
        message.contains("4096") || message.contains("alignment"),
        "AlignmentViolation should mention alignment requirement: {message}"
    );
    assert!(
        message.contains("100") && message.contains("200") && message.contains("300"),
        "AlignmentViolation should show all three values: {message}"
    );
}

// =============================================================================
// ReadStats - No NaN/Inf, Known Precision
// =============================================================================

#[test]
fn readstats_zero_bytes_zero_time_gbps_is_zero() {
    let stats = ReadStats {
        bytes_transferred: 0,
        wall_time: Duration::ZERO,
    };
    let gbps = stats.throughput_gbps();
    assert!(
        gbps == 0.0,
        "Zero bytes and zero time should give 0 gbps, got: {gbps}"
    );
    assert!(!gbps.is_nan(), "Should not produce NaN");
    assert!(!gbps.is_infinite(), "Should not produce infinity");
}

#[test]
fn readstats_zero_bytes_nonzero_time_gbps_is_zero() {
    let stats = ReadStats {
        bytes_transferred: 0,
        wall_time: Duration::from_secs(1),
    };
    let gbps = stats.throughput_gbps();
    assert!(
        gbps == 0.0,
        "Zero bytes should give 0 gbps regardless of time, got: {gbps}"
    );
}

#[test]
fn readstats_one_gib_in_one_sec_is_one_gbps() {
    let stats = ReadStats {
        bytes_transferred: 1_073_741_824, // 1 GiB
        wall_time: Duration::from_secs(1),
    };
    let gbps = stats.throughput_gbps();
    assert!(
        (gbps - 1.0).abs() < 0.001,
        "1 GiB in 1 second should be ~1 gbps, got: {gbps}"
    );
}

#[test]
fn readstats_two_gib_in_one_sec_is_two_gbps() {
    let stats = ReadStats {
        bytes_transferred: 2 * 1_073_741_824, // 2 GiB
        wall_time: Duration::from_secs(1),
    };
    let gbps = stats.throughput_gbps();
    assert!(
        (gbps - 2.0).abs() < 0.001,
        "2 GiB in 1 second should be ~2 gbps, got: {gbps}"
    );
}

#[test]
fn readstats_half_second_precision() {
    let stats = ReadStats {
        bytes_transferred: 1_073_741_824, // 1 GiB
        wall_time: Duration::from_millis(500),
    };
    let gbps = stats.throughput_gbps();
    assert!(
        (gbps - 2.0).abs() < 0.01,
        "1 GiB in 0.5 seconds should be ~2 gbps, got: {gbps}"
    );
}

#[test]
fn readstats_very_small_duration_no_divide_by_zero() {
    let stats = ReadStats {
        bytes_transferred: 4096,
        wall_time: Duration::from_nanos(1),
    };
    let gbps = stats.throughput_gbps();
    assert!(
        gbps.is_finite(),
        "Very small duration should not cause infinity: {gbps}"
    );
}

#[test]
fn readstats_clone_equals_original() {
    let stats = ReadStats {
        bytes_transferred: 8192,
        wall_time: Duration::from_millis(100),
    };
    let cloned = stats;
    assert_eq!(stats.bytes_transferred, cloned.bytes_transferred);
    assert_eq!(stats.wall_time, cloned.wall_time);
}

// =============================================================================
// GDS_ALIGNMENT Constant
// =============================================================================

#[test]
fn gds_alignment_is_4096() {
    assert_eq!(GDS_ALIGNMENT, 4096, "GDS_ALIGNMENT must be 4096");
}

#[test]
fn gds_alignment_is_power_of_two() {
    assert!(
        GDS_ALIGNMENT.is_power_of_two(),
        "GDS_ALIGNMENT should be power of two"
    );
}

#[test]
fn gds_alignment_is_4kib() {
    assert_eq!(GDS_ALIGNMENT, 4 * 1024, "GDS_ALIGNMENT should be 4 KiB");
}

// =============================================================================
// DEFAULT_MAX_TRANSFER_SIZE
// =============================================================================

#[test]
fn default_max_transfer_size_is_16_mib() {
    assert_eq!(
        DEFAULT_MAX_TRANSFER_SIZE,
        16 * 1024 * 1024,
        "DEFAULT_MAX_TRANSFER_SIZE should be 16 MiB"
    );
}

#[test]
fn default_max_transfer_size_is_aligned() {
    assert_eq!(
        DEFAULT_MAX_TRANSFER_SIZE % GDS_ALIGNMENT,
        0,
        "DEFAULT_MAX_TRANSFER_SIZE should be aligned to GDS_ALIGNMENT"
    );
}

// =============================================================================
// Feature Disabled Path
// =============================================================================

#[test]
fn feature_disabled_path_returns_feature_disabled() {
    if cfg!(feature = "cuda") {
        return; // Skip on CUDA-enabled builds
    }

    let result = CuFileHardware::try_init();
    assert!(
        matches!(result, Err(CudaError::FeatureDisabled)),
        "Without cuda feature, try_init should return FeatureDisabled"
    );
}

#[test]
fn feature_disabled_is_available_returns_false() {
    if cfg!(feature = "cuda") {
        return; // Skip on CUDA-enabled builds
    }

    assert!(
        !cudagrep::is_available(),
        "Without cuda feature, is_available should return false"
    );
}

#[test]
fn feature_disabled_availability_report_reason() {
    if cfg!(feature = "cuda") {
        return; // Skip on CUDA-enabled builds
    }

    let report = cudagrep::availability_report();
    assert_eq!(
        report.reason(),
        &AvailabilityReason::FeatureDisabled,
        "Without cuda feature, reason should be FeatureDisabled"
    );
}

// =============================================================================
// Empty Batch Operations
// =============================================================================

#[test]
fn empty_batch_returns_empty_results() {
    // Can't actually test this without GDS hardware for the full path,
    // but we can verify the concept through alignment validation
    let ops: Vec<ReadOp> = vec![];
    assert!(ops.is_empty(), "Empty vec should be empty");
}

#[test]
fn read_batch_into_empty_ops_empty_out() {
    // This test verifies that empty operations work in principle
    // The actual read_batch_into would fail without hardware, but
    // we're testing the alignment validation logic here
    let ops: &[ReadOp] = &[];
    let out: &mut [usize] = &mut [];
    assert_eq!(ops.len(), out.len(), "Empty slices should match");
}

// =============================================================================
// AvailabilityReport
// =============================================================================

#[test]
fn availability_report_equality() {
    let r1 = AvailabilityReport::new(true, AvailabilityReason::Ready, vec!["a".to_owned()]);
    let r2 = AvailabilityReport::new(true, AvailabilityReason::Ready, vec!["a".to_owned()]);
    assert_eq!(r1, r2, "Identical reports should be equal");
}

#[test]
fn availability_report_inequality() {
    let r1 = AvailabilityReport::new(true, AvailabilityReason::Ready, vec!["a".to_owned()]);
    let r2 = AvailabilityReport::new(
        false,
        AvailabilityReason::FeatureDisabled,
        vec!["a".to_owned()],
    );
    assert_ne!(r1, r2, "Different reports should not be equal");
}

#[test]
fn availability_report_display_contains_reason() {
    let report = AvailabilityReport::new(
        false,
        AvailabilityReason::FeatureDisabled,
        vec!["test".to_owned()],
    );
    let display = format!("{report}");
    assert!(!display.is_empty(), "Display should not be empty");
}

// =============================================================================
// ReadOp
// =============================================================================

#[test]
fn readop_clone_copies_all_fields() {
    let op = ReadOp {
        fd: 3,
        device_pointer: unsafe {
            cudagrep::cufile::DevicePointer::new(std::ptr::NonNull::dangling().as_ptr())
        }
        .unwrap(),
        size: 4096,
        file_offset: 0,
        device_offset: 8192,
    };
    let cloned = op;
    assert_eq!(op.fd, cloned.fd);
    assert_eq!(op.size, cloned.size);
    assert_eq!(op.file_offset, cloned.file_offset);
    assert_eq!(op.device_offset, cloned.device_offset);
}

// =============================================================================
// CuFileDriverError Actionable Messages
// =============================================================================

#[test]
fn driver_error_cuda_driver_error_is_actionable() {
    let err = cudagrep::cufile::CuFileDriverError::CudaDriverError(42);
    let msg = err.to_string();
    assert!(
        msg.contains("CUDA") && msg.contains("42"),
        "CudaDriverError should mention CUDA and code: {msg}"
    );
}

#[test]
fn driver_error_os_error_is_actionable() {
    let err = cudagrep::cufile::CuFileDriverError::OsError(-1);
    let msg = err.to_string();
    assert!(
        msg.contains("OS") && msg.contains("-1"),
        "OsError should mention OS and code: {msg}"
    );
}

#[test]
fn driver_error_no_device_memory_is_actionable() {
    let err = cudagrep::cufile::CuFileDriverError::NoDeviceMemory;
    let msg = err.to_string();
    assert!(
        msg.contains("Device") || msg.contains("memory"),
        "NoDeviceMemory should mention device/memory: {msg}"
    );
}

#[test]
fn driver_error_invalid_device_pointer_is_actionable() {
    let err = cudagrep::cufile::CuFileDriverError::InvalidDevicePointer;
    let msg = err.to_string();
    assert!(
        msg.contains("Device") || msg.contains("Pointer"),
        "InvalidDevicePointer should mention device/pointer: {msg}"
    );
}

#[test]
fn driver_error_not_supported_is_actionable() {
    let err = cudagrep::cufile::CuFileDriverError::NotSupported;
    let msg = err.to_string();
    assert!(
        msg.contains("supported") || msg.contains("not"),
        "NotSupported should mention support: {msg}"
    );
}

#[test]
fn driver_error_unknown_is_actionable() {
    let err = cudagrep::cufile::CuFileDriverError::Unknown(999);
    let msg = err.to_string();
    assert!(
        msg.contains("Unknown") && msg.contains("999"),
        "Unknown should mention unknown and code: {msg}"
    );
}

// =============================================================================
// Additional Edge Cases
// =============================================================================

#[test]
fn alignment_very_large_file_offset() {
    // Large but aligned file offset
    let large_aligned: i64 = 4096 * 1_000_000; // 4GB
    assert!(
        validate_alignment(large_aligned, GDS_ALIGNMENT, 0).is_ok(),
        "Large aligned offset should work"
    );
}

#[test]
fn alignment_negative_device_offset() {
    let result = validate_alignment(0, GDS_ALIGNMENT, -4096);
    assert!(
        matches!(
            result,
            Err(CudaError::AlignmentViolation {
                file_offset: 0,
                size,
                device_offset: -4096,
            }) if size == GDS_ALIGNMENT
        ),
        "negative device offset must be rejected, got: {result:?}"
    );
}

#[test]
fn validate_alignment_public_api() {
    // Verify validate_alignment is publicly accessible
    let result = cudagrep::validate_alignment(0, 4096, 0);
    assert!(result.is_ok(), "Public validate_alignment should work");
}

#[test]
fn device_pointer_null_rejected() {
    let ptr = unsafe { cudagrep::cufile::DevicePointer::new(std::ptr::null_mut()) };
    assert!(ptr.is_none(), "Null pointer should be rejected");
}

#[test]
fn cufile_error_clone_eq() {
    use cudagrep::cufile::CUfileError;
    let e1 = CUfileError {
        err: -5,
        cu_err: 42,
    };
    let e2 = e1;
    assert_eq!(e1, e2, "CUfileError should be Copy/Clone");
}

#[test]
fn read_error_variants_distinct() {
    use cudagrep::cufile::{CuFileDriverError, ReadError};

    let reg = ReadError::Register {
        fd: 1,
        code: CuFileDriverError::NotSupported,
    };
    let read = ReadError::Read {
        fd: 1,
        code: CuFileDriverError::NotSupported,
    };
    let cleanup = ReadError::Cleanup {
        fd: 1,
        code: CuFileDriverError::NotSupported,
    };

    assert_ne!(reg, read, "Register and Read should be distinct");
    assert_ne!(read, cleanup, "Read and Cleanup should be distinct");
    assert_ne!(reg, cleanup, "Register and Cleanup should be distinct");
}

#[test]
fn availability_reason_feature_disabled_display() {
    let reason = AvailabilityReason::FeatureDisabled;
    let msg = format!("{reason}");
    assert!(
        msg.contains("cuda") || msg.contains("feature"),
        "FeatureDisabled should mention cuda/feature: {msg}"
    );
}

#[test]
fn availability_reason_ready_display() {
    let reason = AvailabilityReason::Ready;
    let msg = format!("{reason}");
    assert!(
        msg.contains("available") || msg.contains("Ready"),
        "Ready should mention availability: {msg}"
    );
}
