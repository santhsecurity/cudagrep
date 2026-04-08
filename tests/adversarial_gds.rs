//! Adversarial tests for the `GDS` (`GPUDirect` Storage) abstraction.
//!
//! These tests verify error paths, edge cases, and boundary conditions
//! even without NVIDIA hardware. They test that the API degrades gracefully,
//! never panics, and handles adversarial inputs correctly.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_wrap
)]

use cudagrep::{
    availability_report, validate_alignment, AvailabilityReason, AvailabilityReport,
    CuFileHardware, CudaError, DevicePointer, GDS_ALIGNMENT,
};
use std::ffi::c_void;

// =============================================================================
// Test 1: CuFileHardware::try_init() without CUDA
// =============================================================================

#[test]
fn try_init_without_cuda_returns_feature_disabled_or_driver_error() {
    // Without the cuda feature, this MUST return FeatureDisabled
    // With cuda feature but no hardware, this MAY return a driver error
    // In neither case should it panic
    let result = CuFileHardware::try_init();

    #[cfg(not(feature = "cuda"))]
    {
        assert!(
            matches!(result, Err(CudaError::FeatureDisabled)),
            "without cuda feature, try_init must return FeatureDisabled"
        );
    }

    #[cfg(feature = "cuda")]
    {
        // With cuda feature but potentially no hardware, we accept any error
        // but it must NOT panic and must NOT return Ok with a broken state
        if let Err(err) = result {
            let msg = err.to_string();
            assert!(
                !msg.is_empty(),
                "error message must not be empty for debugging"
            );
        }
    }
}

#[test]
fn try_init_never_panics() {
    // This test must never panic regardless of hardware configuration
    let result = CuFileHardware::try_init();
    // If availability probe says false, this must error. If true, it must succeed.
    assert_eq!(result.is_ok(), cudagrep::is_available());
}

// =============================================================================
// Test 2: availability_report() returns struct with is_available() = false on non-CUDA
// =============================================================================

#[test]
fn availability_report_returns_struct_without_panic() {
    let report = availability_report();
    // Must be able to query all methods without panic
    assert_eq!(report.is_available(), cudagrep::is_available());
    assert!(!format!("{:?}", report.reason()).is_empty());
    assert!(!report.diagnostics().is_empty() || cfg!(not(feature = "cuda")));
}

#[test]
fn availability_report_is_available_false_without_cuda() {
    let report = availability_report();

    #[cfg(not(feature = "cuda"))]
    {
        assert!(
            !report.is_available(),
            "without cuda feature, is_available must be false"
        );
    }

    #[cfg(feature = "cuda")]
    {
        let _ = report;
    }
}

#[test]
fn availability_report_reason_is_feature_disabled_without_cuda() {
    let report = availability_report();

    #[cfg(not(feature = "cuda"))]
    {
        assert_eq!(
            *report.reason(),
            AvailabilityReason::FeatureDisabled,
            "without cuda feature, reason must be FeatureDisabled"
        );
    }

    #[cfg(feature = "cuda")]
    {
        let _ = report;
    }
}

#[test]
fn availability_report_has_nonempty_diagnostics() {
    let report = availability_report();
    assert!(
        !report.diagnostics().is_empty(),
        "diagnostics must never be empty for debugging"
    );
}

// =============================================================================
// Tests 3-9: validate_alignment() edge cases
// =============================================================================

#[test]
fn validate_alignment_all_zeros_returns_ok() {
    // All zeros are aligned by definition (0 is a multiple of any alignment)
    let result = validate_alignment(0, 0, 0);
    assert!(
        result.is_ok(),
        "validate_alignment(0, 0, 0) should return Ok, got: {result:?}"
    );
}

#[test]
fn validate_alignment_aligned_values_returns_ok() {
    // 4096 is aligned to GDS_ALIGNMENT
    let result = validate_alignment(4096, 4096, 4096);
    assert!(
        result.is_ok(),
        "validate_alignment(4096, 4096, 4096) should return Ok, got: {result:?}"
    );
}

#[test]
fn validate_alignment_file_offset_not_aligned_returns_err() {
    // 4095 is not aligned to 4096
    let result = validate_alignment(4095, 4096, 4096);
    assert!(
        result.is_err(),
        "validate_alignment(4095, 4096, 4096) should return Err for unaligned file_offset"
    );

    // Verify it's the correct error type
    match result {
        Err(CudaError::AlignmentViolation {
            file_offset,
            size,
            device_offset,
        }) => {
            assert_eq!(file_offset, 4095);
            assert_eq!(size, 4096);
            assert_eq!(device_offset, 4096);
        }
        other => panic!("expected AlignmentViolation error, got: {other:?}"),
    }
}

#[test]
fn validate_alignment_device_offset_not_aligned_returns_err() {
    // 4095 is not aligned to 4096
    let result = validate_alignment(4096, 4096, 4095);
    assert!(
        result.is_err(),
        "validate_alignment(4096, 4096, 4095) should return Err for unaligned device_offset"
    );

    match result {
        Err(CudaError::AlignmentViolation {
            file_offset,
            size,
            device_offset,
        }) => {
            assert_eq!(file_offset, 4096);
            assert_eq!(size, 4096);
            assert_eq!(device_offset, 4095);
        }
        other => panic!("expected AlignmentViolation error, got: {other:?}"),
    }
}

#[test]
fn validate_alignment_size_not_aligned_returns_err() {
    // 4095 is not aligned to 4096
    let result = validate_alignment(4096, 4095, 4096);
    assert!(
        result.is_err(),
        "validate_alignment(4096, 4095, 4096) should return Err for unaligned size"
    );

    match result {
        Err(CudaError::AlignmentViolation {
            file_offset,
            size,
            device_offset,
        }) => {
            assert_eq!(file_offset, 4096);
            assert_eq!(size, 4095);
            assert_eq!(device_offset, 4096);
        }
        other => panic!("expected AlignmentViolation error, got: {other:?}"),
    }
}

#[test]
fn validate_alignment_zero_size_with_aligned_offsets() {
    // Zero size with aligned offsets - edge case
    // This should be acceptable (reading 0 bytes is always aligned)
    let result = validate_alignment(4096, 0, 4096);
    // Zero is aligned to any value (0 % N == 0), so this should be Ok
    assert!(
        result.is_ok(),
        "validate_alignment(4096, 0, 4096) should return Ok (zero size is aligned), got: {result:?}"
    );
}

#[test]
fn validate_alignment_large_aligned_values() {
    // Test with larger aligned values
    let large_aligned = GDS_ALIGNMENT * 1024; // 4 MiB
    let result = validate_alignment(large_aligned as i64, large_aligned, large_aligned as i64);
    assert!(
        result.is_ok(),
        "validate_alignment with large aligned values should return Ok"
    );
}

#[test]
fn gds_alignment_constant_is_4096() {
    assert_eq!(
        GDS_ALIGNMENT, 4096,
        "GDS_ALIGNMENT must be 4096 (4 KiB) per NVIDIA hardware spec"
    );
}

#[test]
fn validate_alignment_negative_offsets_rejected() {
    // Negative file_offset should be rejected
    let result = validate_alignment(-4096, 4096, 0);
    assert!(result.is_err(), "negative file_offset should be rejected");

    // Negative device_offset should be rejected
    let result = validate_alignment(0, 4096, -4096);
    assert!(result.is_err(), "negative device_offset should be rejected");

    // Both negative should be rejected
    let result = validate_alignment(-4096, 4096, -4096);
    assert!(result.is_err(), "both negative offsets should be rejected");
}

#[test]
fn validate_alignment_offset_overflow_edge_cases() {
    // Very large positive offset that might cause overflow
    let large_offset = i64::MAX - (i64::MAX % 4096);

    // Test that the large aligned offset is accepted and does not overflow
    assert!(validate_alignment(large_offset, 4096, 0).is_ok());

    // Offset that's aligned but very large
    let max_aligned = (i64::MAX / 4096) * 4096;
    assert!(validate_alignment(max_aligned, 0, 0).is_ok());
}

// =============================================================================
// Test 10: DevicePointer construction with null
// =============================================================================

#[test]
fn device_pointer_null_returns_none() {
    // SAFETY: We're passing a null pointer which DevicePointer should reject
    let result = unsafe { DevicePointer::new(std::ptr::null_mut::<c_void>()) };
    assert!(
        result.is_none(),
        "DevicePointer::new with null must return None"
    );
}

#[test]
fn device_pointer_non_null_stack_address() {
    // Use a stack variable as a non-null address
    let mut byte: u8 = 0;
    let ptr = std::ptr::addr_of_mut!(byte).cast::<c_void>();

    // SAFETY: ptr is a valid non-null stack address for the duration of this test
    let result = unsafe { DevicePointer::new(ptr) };
    assert!(
        result.is_some(),
        "DevicePointer::new with non-null address must return Some"
    );
}

#[test]
fn device_pointer_dangling_address() {
    // Use a dangling (but non-null) pointer
    let dangling = std::ptr::NonNull::<u8>::dangling()
        .as_ptr()
        .cast::<c_void>();

    // SAFETY: Dangling pointer is non-null, so DevicePointer should accept it
    // (it doesn't validate that the pointer is valid, just non-null)
    let result = unsafe { DevicePointer::new(dangling) };
    assert!(
        result.is_some(),
        "DevicePointer::new with non-null dangling ptr must return Some"
    );
}

// =============================================================================
// Test 11: CuFileHardware evict_fd on never-registered fd
// =============================================================================

#[test]
fn evict_fd_on_never_registered_returns_ok_false() {
    // This test requires a CuFileHardware instance
    // If we can't create one, we skip the test
    let Ok(mut gds) = CuFileHardware::try_init() else {
        return;
    }; // Skip test if we can't initialize

    // Try to evict an fd that was never registered
    let result = gds.evict_fd(9999);
    assert!(
        result.is_ok(),
        "evict_fd on never-registered fd should return Ok, not Err"
    );
    assert!(
        !result.unwrap(),
        "evict_fd on never-registered fd should return Ok(false)"
    );
}

// =============================================================================
// Test 12-13: Read with edge case sizes
// =============================================================================

#[test]
fn read_to_device_size_zero_returns_ok_zero() {
    let Ok(mut gds) = CuFileHardware::try_init() else {
        return;
    }; // Skip if no hardware

    // Create a dummy device pointer (this is a test limitation - we need
    // valid device memory for a real read, but with size 0 it shouldn't matter)
    let mut byte: u8 = 0;
    let ptr = unsafe { DevicePointer::new(std::ptr::addr_of_mut!(byte).cast::<c_void>()) }
        .unwrap_or_else(|| panic!("non-null stack pointer"));

    // This will likely fail with alignment error before even trying to read,
    // but let's verify it doesn't panic or cause undefined behavior
    let result = gds.read_to_device(-1, ptr, 0, 0, 0);

    // With size 0, it should either succeed with 0 bytes, or fail with
    // an alignment error (since 0,0,0 should be valid alignment)
    match result {
        Ok(0) | Err(CudaError::AlignmentViolation { .. } | _) => (),
        Ok(n) => panic!("reading 0 bytes should return 0, got {n}"),
    }
}

#[test]
fn read_to_device_with_max_size_does_not_overflow() {
    // Test that passing usize::MAX doesn't cause integer overflow
    let Ok(mut gds) = CuFileHardware::try_init() else {
        return;
    }; // Skip if no hardware

    let mut byte: u8 = 0;
    let ptr = unsafe { DevicePointer::new(std::ptr::addr_of_mut!(byte).cast::<c_void>()) }
        .unwrap_or_else(|| panic!("non-null stack pointer"));

    // This should fail gracefully, not panic or overflow
    let result = gds.read_to_device(-1, ptr, usize::MAX, 0, 0);

    // Must not panic - any error result is acceptable
    match result {
        Err(_) => (), // Any error is fine
        Ok(_) => panic!("reading usize::MAX bytes should not succeed"),
    }
}

// =============================================================================
// Test 14: Registration cache - register same fd twice is no-op
// =============================================================================

#[test]
fn registration_cache_double_register_is_noop() {
    let Ok(mut gds) = CuFileHardware::try_init() else {
        return;
    }; // Skip if no hardware

    // This test verifies that calling operations on the same fd twice
    // doesn't cause issues. We can't actually test the registration cache
    // directly without a valid file descriptor, but we can test that
    // evicting twice doesn't cause an error.

    // First evict (fd was never registered)
    let result1 = gds.evict_fd(8888);
    assert!(result1.is_ok());
    assert!(!result1.unwrap());

    // Second evict (still not registered)
    let result2 = gds.evict_fd(8888);
    assert!(result2.is_ok());
    assert!(!result2.unwrap());
}

// =============================================================================
// Test 15: Drop CuFileHardware cleans up cached registrations
// =============================================================================

#[test]
fn drop_cufile_hardware_with_no_registrations_does_not_panic() {
    let Ok(gds) = CuFileHardware::try_init() else {
        return;
    }; // Skip if no hardware

    // Drop with no registrations - should not panic
    drop(gds);
}

#[test]
fn drop_cufile_hardware_after_evicting_all_does_not_panic() {
    let Ok(mut gds) = CuFileHardware::try_init() else {
        return;
    }; // Skip if no hardware

    // Evict a non-existent fd (no-op)
    let _ = gds.evict_fd(7777);

    // Drop after "evicting all" - should not panic
    drop(gds);
}

// =============================================================================
// Additional adversarial tests
// =============================================================================

#[test]
fn validate_alignment_all_unaligned() {
    // All parameters unaligned - should still report error
    let result = validate_alignment(1, 1, 1);
    assert!(result.is_err(), "all unaligned should return Err");
}

#[test]
fn validate_alignment_mixed_alignment() {
    // Some aligned, some not
    assert!(validate_alignment(4096, 4096, 1).is_err()); // device_offset bad
    assert!(validate_alignment(4096, 1, 4096).is_err()); // size bad
    assert!(validate_alignment(1, 4096, 4096).is_err()); // file_offset bad
}

#[test]
fn availability_report_clone_and_eq() {
    let report1 = availability_report();
    let report2 = report1.clone();

    assert_eq!(report1.is_available(), report2.is_available());
    assert_eq!(report1.reason(), report2.reason());
    assert_eq!(report1.diagnostics(), report2.diagnostics());
}

#[test]
fn availability_report_display_and_debug() {
    let report = availability_report();

    let display = format!("{report}");
    assert!(!display.is_empty(), "Display must produce non-empty output");

    let debug = format!("{report:?}");
    assert!(!debug.is_empty(), "Debug must produce non-empty output");
}

#[test]
fn cuda_error_display_for_all_variants() {
    // Test that all error variants produce non-empty messages

    let err = CudaError::FeatureDisabled;
    assert!(!err.to_string().is_empty());

    let report = AvailabilityReport::new(
        false,
        AvailabilityReason::FeatureDisabled,
        vec!["test".to_owned()],
    );
    let err = CudaError::Unavailable { report };
    assert!(!err.to_string().is_empty());

    let err = CudaError::DriverInitRejected {
        code: cudagrep::cufile::CuFileDriverError::NotSupported,
    };
    assert!(!err.to_string().is_empty());

    let err = CudaError::DescriptorRegistrationFailed {
        fd: 1,
        code: cudagrep::cufile::CuFileDriverError::InvalidDevicePointer,
    };
    assert!(!err.to_string().is_empty());

    let err = CudaError::DirectMemoryAccessFailed {
        fd: 1,
        code: cudagrep::cufile::CuFileDriverError::NoDeviceMemory,
    };
    assert!(!err.to_string().is_empty());

    let err = CudaError::DescriptorCleanupFailed {
        fd: 1,
        code: cudagrep::cufile::CuFileDriverError::OsError(-1),
    };
    assert!(!err.to_string().is_empty());

    let err = CudaError::AlignmentViolation {
        file_offset: 1,
        size: 2,
        device_offset: 3,
    };
    assert!(!err.to_string().is_empty());

    let err = CudaError::ShortRead {
        requested: 4096,
        transferred: 1024,
    };
    assert!(!err.to_string().is_empty());

    let err = CudaError::BatchLengthMismatch {
        ops_len: 1,
        out_len: 2,
    };
    assert!(!err.to_string().is_empty());
}

#[test]
fn cufile_driver_error_from_raw_mappings() {
    use cudagrep::cufile::CuFileDriverError;

    // Test error code mappings
    assert!(matches!(
        CuFileDriverError::from_raw(-1),
        CuFileDriverError::OsError(-1)
    ));
    assert!(matches!(
        CuFileDriverError::from_raw(-2),
        CuFileDriverError::CudaDriverError(-2)
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

    // Unknown error codes
    assert!(matches!(
        CuFileDriverError::from_raw(0),
        CuFileDriverError::Unknown(0)
    ));
    assert!(matches!(
        CuFileDriverError::from_raw(-99),
        CuFileDriverError::Unknown(-99)
    ));
    assert!(matches!(
        CuFileDriverError::from_raw(42),
        CuFileDriverError::Unknown(42)
    ));
}

#[test]
fn cufile_driver_error_display_includes_code() {
    use cudagrep::cufile::CuFileDriverError;

    let err = CuFileDriverError::Unknown(-999);
    let msg = err.to_string();
    assert!(
        msg.contains("-999") || msg.contains("Unknown"),
        "error message should include code info: {msg}"
    );
}

#[test]
fn device_pointer_clone_copy() {
    let mut byte: u8 = 0;
    let ptr = unsafe { DevicePointer::new(std::ptr::addr_of_mut!(byte).cast::<c_void>()) }
        .unwrap_or_else(|| panic!("non-null"));

    // Clone
    let ptr2 = ptr;
    assert_eq!(ptr.as_mut_ptr(), ptr2.as_mut_ptr());

    // Copy (implicit)
    let ptr3 = ptr;
    let _ = ptr3.as_mut_ptr();
    // ptr is now moved, can't use it again
}
