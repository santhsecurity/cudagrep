//! Regression tests for graceful degradation when CUDA/GDS is unavailable.
//!
//! These tests verify that every public API entry point degrades cleanly
//! into an error or a false-availability result, never panics, and always
//! provides enough context to diagnose the issue.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_wrap
)]

use cudagrep::{AvailabilityReason, CuFileHardware, CudaError};

// =============================================================================
// No Panic on Any Public API
// =============================================================================

#[test]
fn is_available_no_panic() {
    let _ = cudagrep::is_available();
}

#[test]
fn availability_report_no_panic() {
    let _ = cudagrep::availability_report();
}

#[test]
fn try_init_no_panic() {
    let _ = CuFileHardware::try_init();
}

// =============================================================================
// Error Chain Completeness
// =============================================================================

#[test]
fn feature_disabled_error_is_source_chain_terminal() {
    let error = CudaError::FeatureDisabled;
    // thiserror-generated source() should be None for this variant.
    assert!(
        std::error::Error::source(&error).is_none(),
        "FeatureDisabled should not have a source error"
    );
}

#[test]
fn driver_rejected_error_includes_code() {
    let error = CudaError::DriverInitRejected {
        code: cudagrep::cufile::CuFileDriverError::from_raw(-1),
    };
    let message = error.to_string();
    assert!(
        message.contains("-1") || message.contains("OS error"),
        "error message should describe the rejection: {message}"
    );
}

// =============================================================================
// Availability Reason Display
// =============================================================================

#[test]
fn reason_feature_disabled_display() {
    let reason = AvailabilityReason::FeatureDisabled;
    assert!(format!("{reason}").contains("cuda"));
}

#[test]
fn reason_library_unavailable_display() {
    let reason = AvailabilityReason::LibraryUnavailable("not found".to_owned());
    let display = format!("{reason}");
    assert!(display.contains("libcufile"));
    assert!(display.contains("not found"));
}

#[test]
fn reason_symbol_unavailable_display() {
    let reason = AvailabilityReason::SymbolUnavailable("cuFileRead".to_owned());
    let display = format!("{reason}");
    assert!(display.contains("cuFileRead"));
}

#[test]
fn reason_driver_rejected_display() {
    let reason =
        AvailabilityReason::DriverRejected(cudagrep::cufile::CuFileDriverError::Unknown(-7));
    let display = format!("{reason}");
    assert!(
        display.contains("-7") || display.contains("Unknown"),
        "display should describe the rejection: {display}"
    );
}

#[test]
fn reason_ready_display() {
    let reason = AvailabilityReason::Ready;
    let display = format!("{reason}");
    assert!(display.contains("available"));
}

// =============================================================================
// AvailabilityReason Equality
// =============================================================================

#[test]
fn reason_eq_same_variant() {
    assert_eq!(AvailabilityReason::Ready, AvailabilityReason::Ready);
    assert_eq!(
        AvailabilityReason::FeatureDisabled,
        AvailabilityReason::FeatureDisabled
    );
}

#[test]
fn reason_ne_different_variant() {
    assert_ne!(
        AvailabilityReason::Ready,
        AvailabilityReason::FeatureDisabled
    );
}

#[test]
fn reason_ne_same_variant_different_payload() {
    assert_ne!(
        AvailabilityReason::DriverRejected(cudagrep::cufile::CuFileDriverError::OsError(-1)),
        AvailabilityReason::DriverRejected(cudagrep::cufile::CuFileDriverError::CudaDriverError(
            -2
        ))
    );
}
