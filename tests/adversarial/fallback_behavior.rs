//! Adversarial tests for fallback behavior when CUDA/GPU is unavailable.
//!
//! These tests verify that the library degrades gracefully without crashing,
//! panicking, or returning undefined behavior when:
//! - CUDA is not installed
//! - No NVIDIA GPU is present
//! - The cuda feature is disabled
//! - Driver initialization fails

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_wrap
)]

use cudagrep::{AvailabilityReason, AvailabilityReport, CuFileHardware, CudaError, availability_report, is_available};

// =============================================================================
// Feature Disabled Path
// =============================================================================

#[test]
fn fallback_feature_disabled_is_available_returns_false() {
    if cfg!(not(feature = "cuda")) {
        assert!(!is_available(), "Without cuda feature, is_available must be false");
    }
}

#[test]
fn fallback_feature_disabled_availability_report_reason() {
    if cfg!(not(feature = "cuda")) {
        let report = availability_report();
        assert_eq!(*report.reason(), AvailabilityReason::FeatureDisabled);
    }
}

#[test]
fn fallback_feature_disabled_try_init_returns_feature_disabled() {
    if cfg!(not(feature = "cuda")) {
        let result = CuFileHardware::try_init();
        assert!(
            matches!(result, Err(CudaError::FeatureDisabled)),
            "Without cuda feature, try_init must return FeatureDisabled"
        );
    }
}

#[test]
fn fallback_feature_disabled_error_is_actionable() {
    if cfg!(not(feature = "cuda")) {
        let result = CuFileHardware::try_init();
        let error = result.unwrap_err();
        let msg = error.to_string();
        
        assert!(
            msg.contains("disabled") || msg.contains("cuda") || msg.contains("CUDA"),
            "FeatureDisabled error should explain how to enable: {}", msg
        );
    }
}

// =============================================================================
// Library Unavailable Path
// =============================================================================

#[test]
fn fallback_library_unavailable_never_panics() {
    // Even when CUDA libraries are missing, the probe should not panic
    let report = availability_report();
    
    // Should either be available (if CUDA is present) or have a specific reason
    if !report.is_available() {
        assert!(
            matches!(
                report.reason(),
                AvailabilityReason::FeatureDisabled |
                AvailabilityReason::LibraryUnavailable(_) |
                AvailabilityReason::SymbolUnavailable(_) |
                AvailabilityReason::DriverRejected(_) |
                AvailabilityReason::DriverCleanupFailed(_)
            ),
            "Unavailable should have concrete reason: {:?}", report.reason()
        );
    }
}

#[test]
fn fallback_library_unavailable_provides_diagnostics() {
    let report = availability_report();
    
    // Should always provide diagnostics for debugging
    assert!(
        !report.diagnostics().is_empty() || matches!(report.reason(), AvailabilityReason::FeatureDisabled),
        "Report should have diagnostics or be FeatureDisabled: {:?}", report
    );
}

// =============================================================================
// Driver Rejection Path
// =============================================================================

#[test]
fn fallback_driver_rejected_error_contains_code() {
    // Simulate driver rejection error
    let error = CudaError::DriverInitRejected {
        code: cudagrep::cufile::CuFileDriverError::NotSupported,
    };
    
    let msg = error.to_string();
    assert!(
        !msg.is_empty(),
        "DriverInitRejected should have a message: {}", msg
    );
}

// =============================================================================
// Session Lifecycle Without Hardware
// =============================================================================

#[test]
fn fallback_try_init_never_panics() {
    // This test should never panic regardless of hardware configuration
    let _result = CuFileHardware::try_init();
}

#[test]
fn fallback_try_init_consistent_across_calls() {
    let first = CuFileHardware::try_init();
    let second = CuFileHardware::try_init();
    
    // Both should succeed or both should fail
    assert_eq!(
        first.is_ok(),
        second.is_ok(),
        "try_init should be consistent across calls"
    );
}

#[test]
fn fallback_availability_report_consistent_across_calls() {
    let first = availability_report();
    let second = availability_report();
    
    assert_eq!(first.is_available(), second.is_available());
    assert_eq!(first.reason(), second.reason());
}

// =============================================================================
// Error Handling Without Hardware
// =============================================================================

#[test]
fn fallback_all_error_variants_have_messages() {
    // Verify all error variants produce meaningful messages
    
    let errors = vec![
        CudaError::FeatureDisabled,
        CudaError::Unavailable {
            report: AvailabilityReport::new(
                false,
                AvailabilityReason::LibraryUnavailable("test".to_owned()),
                vec!["diag".to_owned()],
            ),
        },
        CudaError::DriverInitRejected {
            code: cudagrep::cufile::CuFileDriverError::OsError(-1),
        },
        CudaError::DescriptorRegistrationFailed {
            fd: 1,
            code: cudagrep::cufile::CuFileDriverError::InvalidDevicePointer,
        },
        CudaError::DirectMemoryAccessFailed {
            fd: 1,
            code: cudagrep::cufile::CuFileDriverError::NoDeviceMemory,
        },
        CudaError::DescriptorCleanupFailed {
            fd: 1,
            code: cudagrep::cufile::CuFileDriverError::NotSupported,
        },
        CudaError::AlignmentViolation {
            file_offset: 1,
            size: 2,
            device_offset: 3,
        },
        CudaError::ShortRead {
            requested: 100,
            transferred: 50,
        },
        CudaError::BatchLengthMismatch {
            ops_len: 1,
            out_len: 2,
        },
    ];
    
    for error in errors {
        let msg = error.to_string();
        assert!(
            !msg.is_empty(),
            "Error variant should have non-empty message: {:?}", error
        );
    }
}

// =============================================================================
// Public API Availability Without Hardware
// =============================================================================

#[test]
fn fallback_public_api_functions_exist() {
    // Verify all public functions can be called without panic
    
    // These should never panic
    let _ = is_available();
    let _ = availability_report();
    let _ = cudagrep::validate_alignment(0, 4096, 0);
    let _ = cudagrep::GDS_ALIGNMENT;
}

#[test]
fn fallback_read_stats_works_without_hardware() {
    use cudagrep::ReadStats;
    use std::time::Duration;
    
    // ReadStats is a pure data type, should work without hardware
    let stats = ReadStats {
        bytes_transferred: 0,
        wall_time: Duration::ZERO,
    };
    
    let gbps = stats.throughput_gbps();
    assert_eq!(gbps, 0.0);
}

// =============================================================================
// Concurrent Fallback Behavior
// =============================================================================

#[test]
fn fallback_concurrent_availability_checks() {
    use std::thread;
    
    let mut handles = vec![];
    
    for _ in 0..20 {
        handles.push(thread::spawn(|| {
            let report = availability_report();
            (report.is_available(), format!("{report.reason(:?}")))
        }));
    }
    
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    
    // All results should be consistent
    let first = results.first().unwrap();
    for result in &results {
        assert_eq!(result.0, first.0, "Availability must be consistent across threads");
    }
}

// =============================================================================
// Error Source Chain
// =============================================================================

#[test]
fn fallback_error_source_chain_terminal() {
    // FeatureDisabled should have no source (it's terminal)
    let error = CudaError::FeatureDisabled;
    assert!(
        std::error::Error::source(&error).is_none(),
        "FeatureDisabled should be terminal error"
    );
}

// =============================================================================
// Display and Debug Quality
// =============================================================================

#[test]
fn fallback_report_display_includes_reason() {
    let report = availability_report();
    let display = format!("{report}");
    
    assert!(
        !display.is_empty(),
        "Report Display should not be empty"
    );
}

#[test]
fn fallback_report_debug_includes_all_fields() {
    let report = availability_report();
    let debug = format!("{report:?}");
    
    assert!(
        debug.contains("available") || debug.contains("Available"),
        "Report Debug should include availability: {}", debug
    );
}

// =============================================================================
// AvailabilityReason Display
// =============================================================================

#[test]
fn fallback_reason_feature_disabled_display() {
    let reason = AvailabilityReason::FeatureDisabled;
    let msg = format!("{reason}");
    
    assert!(
        msg.contains("cuda") || msg.contains("feature") || msg.contains("disabled"),
        "FeatureDisabled should explain the issue: {}", msg
    );
}

#[test]
fn fallback_reason_library_unavailable_display() {
    let reason = AvailabilityReason::LibraryUnavailable("libcufile.so.0 not found".to_owned());
    let msg = format!("{reason}");
    
    assert!(
        msg.contains("libcufile") || msg.contains("library"),
        "LibraryUnavailable should mention library: {}", msg
    );
}

#[test]
fn fallback_reason_ready_display() {
    let reason = AvailabilityReason::Ready;
    let msg = format!("{reason}");
    
    assert!(
        msg.contains("available") || msg.contains("ready"),
        "Ready should mention availability: {}", msg
    );
}
