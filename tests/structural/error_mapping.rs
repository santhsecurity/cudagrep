//! Adversarial tests for `CuFileDriverError` and `CUfileError` mappings.
//!
//! Ensures that raw error codes from the CUDA driver are correctly mapped
//! to typed Rust error variants, and that no mapping silently swallows
//! information.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_wrap
)]

use cudagrep::cufile::{CUfileError, CuFileDriverError};

// =============================================================================
// CuFileDriverError::from_raw Exhaustive Mapping
// =============================================================================

#[test]
fn from_raw_minus_1_is_os_error() {
    let error = CuFileDriverError::from_raw(-1);
    assert!(
        matches!(error, CuFileDriverError::OsError(_)),
        "code -1 should map to OsError, got: {error:?}"
    );
}

#[test]
fn from_raw_minus_2_is_cuda_driver_error() {
    let error = CuFileDriverError::from_raw(-2);
    assert!(
        matches!(error, CuFileDriverError::CudaDriverError(_)),
        "code -2 should map to CudaDriverError, got: {error:?}"
    );
}

#[test]
fn from_raw_minus_3_is_not_supported() {
    let error = CuFileDriverError::from_raw(-3);
    assert!(
        matches!(error, CuFileDriverError::NotSupported),
        "code -3 should map to NotSupported, got: {error:?}"
    );
}

#[test]
fn from_raw_minus_4_is_no_device_memory() {
    let error = CuFileDriverError::from_raw(-4);
    assert!(
        matches!(error, CuFileDriverError::NoDeviceMemory),
        "code -4 should map to NoDeviceMemory, got: {error:?}"
    );
}

#[test]
fn from_raw_minus_5_is_invalid_device_pointer() {
    let error = CuFileDriverError::from_raw(-5);
    assert!(
        matches!(error, CuFileDriverError::InvalidDevicePointer),
        "code -5 should map to InvalidDevicePointer, got: {error:?}"
    );
}

#[test]
fn from_raw_unknown_positive_code() {
    let error = CuFileDriverError::from_raw(42);
    assert!(
        matches!(error, CuFileDriverError::Unknown(42)),
        "positive code should map to Unknown, got: {error:?}"
    );
}

#[test]
fn from_raw_unknown_negative_code() {
    let error = CuFileDriverError::from_raw(-99);
    assert!(
        matches!(error, CuFileDriverError::Unknown(-99)),
        "unmapped negative code should map to Unknown, got: {error:?}"
    );
}

#[test]
fn from_raw_zero_is_unknown() {
    let error = CuFileDriverError::from_raw(0);
    assert!(
        matches!(error, CuFileDriverError::Unknown(0)),
        "code 0 should map to Unknown, got: {error:?}"
    );
}

#[test]
fn from_raw_i32_min_does_not_panic() {
    let error = CuFileDriverError::from_raw(i32::MIN);
    assert!(
        matches!(error, CuFileDriverError::Unknown(i32::MIN)),
        "i32::MIN should map to Unknown, got: {error:?}"
    );
}

#[test]
fn from_raw_i32_max_does_not_panic() {
    let error = CuFileDriverError::from_raw(i32::MAX);
    assert!(
        matches!(error, CuFileDriverError::Unknown(i32::MAX)),
        "i32::MAX should map to Unknown, got: {error:?}"
    );
}

// =============================================================================
// CuFileDriverError Display
// =============================================================================

#[test]
fn every_variant_has_non_empty_display() {
    let variants: Vec<CuFileDriverError> = vec![
        CuFileDriverError::CudaDriverError(-2),
        CuFileDriverError::OsError(-1),
        CuFileDriverError::NoDeviceMemory,
        CuFileDriverError::InvalidDevicePointer,
        CuFileDriverError::NotSupported,
        CuFileDriverError::Unknown(0),
    ];
    for variant in &variants {
        let display = format!("{variant}");
        assert!(
            !display.is_empty(),
            "Display for {variant:?} must be non-empty"
        );
    }
}

#[test]
fn driver_error_clone_eq() {
    let error = CuFileDriverError::from_raw(-3);
    let cloned = error;
    assert_eq!(error, cloned);
}

// =============================================================================
// CUfileError
// =============================================================================

#[test]
fn cufile_error_zero_err_is_success() {
    let status = CUfileError { err: 0, cu_err: 0 };
    assert!(status.is_success());
}

#[test]
fn cufile_error_nonzero_err_is_failure() {
    let status = CUfileError { err: -1, cu_err: 0 };
    assert!(!status.is_success());
}

#[test]
fn cufile_error_zero_err_nonzero_cu_err_is_still_success() {
    // The success check uses only the `err` field.
    let status = CUfileError {
        err: 0,
        cu_err: 700,
    };
    assert!(status.is_success());
}

#[test]
fn cufile_error_positive_err_is_failure() {
    let status = CUfileError { err: 1, cu_err: 0 };
    assert!(!status.is_success());
}

#[test]
fn cufile_error_debug_includes_fields() {
    let status = CUfileError {
        err: -5,
        cu_err: 42,
    };
    let debug = format!("{status:?}");
    assert!(debug.contains("-5"), "debug should include err field");
    assert!(debug.contains("42"), "debug should include cu_err field");
}

#[test]
fn cufile_error_clone_eq() {
    let a = CUfileError { err: -1, cu_err: 3 };
    let b = a;
    assert_eq!(a, b);
}

// =============================================================================
// All Known Error Codes Covered
// =============================================================================

#[test]
fn all_documented_error_codes_have_distinct_variants() {
    let codes = [-5, -4, -3, -2, -1];
    let variants: Vec<_> = codes
        .iter()
        .map(|&c| CuFileDriverError::from_raw(c))
        .collect();
    // Each code maps to a different variant.
    for i in 0..variants.len() {
        for j in (i + 1)..variants.len() {
            assert_ne!(
                variants[i], variants[j],
                "codes {} and {} should map to different variants",
                codes[i], codes[j]
            );
        }
    }
}
