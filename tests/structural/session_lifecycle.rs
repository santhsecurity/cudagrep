//! Adversarial tests for `CuFileHardware` session lifecycle.
//!
//! Without actual CUDA hardware, these tests verify that session creation
//! fails gracefully with clear errors, and that the error types carry
//! actionable information.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_wrap
)]

use cudagrep::{CuFileHardware, CudaError};

// =============================================================================
// Session Creation Without CUDA
// =============================================================================

#[test]
fn try_init_without_cuda_feature_returns_feature_disabled() {
    if cfg!(feature = "cuda") {
        return; // skip on CUDA-enabled builds
    }
    let result = CuFileHardware::try_init();
    assert!(result.is_err());
    let Err(error) = result else {
        panic!("expected FeatureDisabled error, got Ok")
    };
    assert!(
        matches!(error, CudaError::FeatureDisabled),
        "expected FeatureDisabled, got: {error}"
    );
}

#[test]
fn try_init_error_display_is_actionable() {
    let result = CuFileHardware::try_init();
    if let Err(error) = result {
        let message = error.to_string();
        assert!(!message.is_empty(), "error display must be non-empty");
        // The message should explain what's wrong.
        assert!(
            message.len() > 10,
            "error message too short to be useful: {message}"
        );
    }
}

#[test]
fn try_init_repeated_calls_same_result() {
    let first = CuFileHardware::try_init().err().map(|e| format!("{e}"));
    let second = CuFileHardware::try_init().err().map(|e| format!("{e}"));
    assert_eq!(first, second, "repeated try_init must produce same error");
}

// =============================================================================
// CudaError Variants
// =============================================================================

#[test]
fn cuda_error_feature_disabled_display() {
    let error = CudaError::FeatureDisabled;
    let message = error.to_string();
    assert!(
        message.contains("disabled"),
        "FeatureDisabled message should mention 'disabled': {message}"
    );
}

#[test]
fn cuda_error_driver_rejected_display() {
    let error = CudaError::DriverInitRejected {
        code: cudagrep::cufile::CuFileDriverError::Unknown(-42),
    };
    let message = error.to_string();
    assert!(
        message.contains("-42") || message.contains("Unknown"),
        "DriverInitRejected must describe the error: {message}"
    );
}

#[test]
fn cuda_error_descriptor_registration_failed_display() {
    let error = CudaError::DescriptorRegistrationFailed {
        fd: 7,
        code: cudagrep::cufile::CuFileDriverError::OsError(-1),
    };
    let message = error.to_string();
    assert!(
        message.contains('7'),
        "DescriptorRegistrationFailed must include the fd: {message}"
    );
}

#[test]
fn cuda_error_dma_failed_display() {
    let error = CudaError::DirectMemoryAccessFailed {
        fd: 12,
        code: cudagrep::cufile::CuFileDriverError::InvalidDevicePointer,
    };
    let message = error.to_string();
    assert!(
        message.contains("12"),
        "DirectMemoryAccessFailed must include the fd: {message}"
    );
}

#[test]
fn cuda_error_cleanup_failed_display() {
    let error = CudaError::DescriptorCleanupFailed {
        fd: 99,
        code: cudagrep::cufile::CuFileDriverError::NotSupported,
    };
    let message = error.to_string();
    assert!(
        message.contains("99"),
        "DescriptorCleanupFailed must include the fd: {message}"
    );
}

#[test]
fn cuda_error_unavailable_with_report() {
    let report = cudagrep::AvailabilityReport::new(
        false,
        cudagrep::AvailabilityReason::FeatureDisabled,
        vec!["test diagnostic".to_owned()],
    );
    let error = CudaError::Unavailable { report };
    let message = error.to_string();
    assert!(
        !message.is_empty(),
        "Unavailable error must produce a non-empty message"
    );
}
