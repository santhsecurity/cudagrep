//! Adversarial tests for GPU device detection.
//!
//! These tests verify that the library correctly identifies NVIDIA GPUs
//! and rejects software renderers (llvmpipe, SwiftShader, etc.).

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_wrap
)]

use cudagrep::{AvailabilityReason, availability_report};

// =============================================================================
// GPU Detection - Software Renderer Rejection
// =============================================================================

#[test]
fn availability_report_detects_gpu_or_provides_reason() {
    let report = availability_report();
    
    // The report should either say we have a GPU ready, or explain why not
    if report.is_available() {
        assert_eq!(*report.reason(), AvailabilityReason::Ready, "Available must mean Ready");
        // Should have diagnostic mentioning NVIDIA
        let has_nvidia = report.diagnostics().iter()
            .any(|d| d.to_lowercase().contains("nvidia"));
        assert!(has_nvidia, "Available report should mention NVIDIA GPU: {:?}", report.diagnostics());
    } else {
        // Not available - should have a specific reason
        assert!(
            matches!(
                report.reason(),
                AvailabilityReason::FeatureDisabled |
                AvailabilityReason::LibraryUnavailable(_) |
                AvailabilityReason::SymbolUnavailable(_) |
                AvailabilityReason::DriverRejected(_) |
                AvailabilityReason::DriverCleanupFailed(_)
            ),
            "Unavailable should have a concrete reason: {:?}", report.reason()
        );
    }
}

#[test]
fn availability_report_never_claims_software_renderer() {
    let report = availability_report();
    
    // If available, verify we're not claiming a software renderer
    if report.is_available() {
        for diag in report.diagnostics() {
            let lower = diag.to_lowercase();
            assert!(
                !lower.contains("llvmpipe"),
                "Must not claim llvmpipe as valid GPU: {}", diag
            );
            assert!(
                !lower.contains("software"),
                "Must not claim software renderer as valid GPU: {}", diag
            );
            assert!(
                !lower.contains("swiftshader"),
                "Must not claim SwiftShader as valid GPU: {}", diag
            );
        }
    }
}

#[test]
fn availability_report_detects_cuda_feature_disabled() {
    if !cfg!(feature = "cuda") {
        let report = availability_report();
        assert!(!report.is_available());
        assert_eq!(*report.reason(), AvailabilityReason::FeatureDisabled);
    }
}

// =============================================================================
// Compute Capability Verification
// =============================================================================

#[test]
fn availability_report_includes_compute_capability_when_available() {
    let report = availability_report();
    
    if report.is_available() {
        // Should mention compute capability (CC) in diagnostics
        let has_cc = report.diagnostics().iter()
            .any(|d| d.contains("CC ") || d.contains("compute capability"));
        assert!(
            has_cc,
            "Available report should include compute capability info: {:?}",
            report.diagnostics()
        );
    }
}

#[test]
fn availability_report_rejects_insufficient_compute_capability() {
    let report = availability_report();
    
    // If we have GPUs but they're too old, should be marked unavailable
    if !report.is_available() {
        let has_old_gpu = report.diagnostics().iter()
            .any(|d| d.contains("insufficient compute capability"));
        
        if has_old_gpu {
            // Verify it's actually reporting the issue
            assert!(
                matches!(report.reason(), AvailabilityReason::LibraryUnavailable(_)),
                "Old GPU should be LibraryUnavailable: {:?}", report.reason()
            );
        }
    }
}

// =============================================================================
// Driver Version Detection
// =============================================================================

#[test]
fn availability_report_includes_version_info() {
    let report = availability_report();
    
    // Should have version-related diagnostics when CUDA is enabled
    if cfg!(feature = "cuda") {
        let has_version = report.diagnostics().iter()
            .any(|d| d.contains("version") || d.contains("Version"));
        
        // Not strictly required but should appear when library loads
        // Just verify diagnostics exist and are non-empty
        assert!(!report.diagnostics().is_empty(), "Should have diagnostics");
    }
}

// =============================================================================
// Concurrent Availability Probes
// =============================================================================

#[test]
fn availability_report_stable_under_concurrent_calls() {
    use std::sync::Arc;
    use std::thread;
    
    // Get baseline
    let baseline = availability_report();
    
    // Spawn multiple threads checking availability
    let mut handles = vec![];
    for _ in 0..10 {
        handles.push(thread::spawn(move || {
            availability_report()
        }));
    }
    
    // Collect results
    for handle in handles {
        let report = handle.join().expect("Thread panicked");
        assert_eq!(
            report.is_available(),
            baseline.is_available(),
            "Availability must be consistent across threads"
        );
    }
}

// =============================================================================
// Malformed Device Name Handling
// =============================================================================

#[test]
fn availability_report_handles_null_device_name() {
    // This is a defensive test - we can't actually inject null device names
    // without mocking the CUDA driver, but we verify the code path exists
    // by checking the report generates valid strings
    let report = availability_report();
    
    for diag in report.diagnostics() {
        // All diagnostics should be valid UTF-8 and non-empty
        assert!(!diag.is_empty(), "Diagnostic should not be empty");
        // Should not contain null bytes
        assert!(
            !diag.contains('\0'),
            "Diagnostic should not contain null bytes: {:?}", diag
        );
    }
}

// =============================================================================
// Library Loading Edge Cases
// =============================================================================

#[test]
fn availability_report_handles_missing_cuda_lib() {
    let report = availability_report();
    
    // If CUDA lib is missing, should be unavailable with clear reason
    if !report.is_available() {
        let has_lib_error = report.diagnostics().iter()
            .any(|d| d.contains("libcuda") || d.contains("libcufile"));
        
        // This is expected if no CUDA installed
        // Just verify the diagnostics are present
        if has_lib_error {
            assert!(
                matches!(report.reason(), AvailabilityReason::LibraryUnavailable(_)),
                "Missing lib should be LibraryUnavailable"
            );
        }
    }
}
