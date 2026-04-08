//! Adversarial tests for GDS availability probing.
//!
//! These tests verify that the availability probe never panics, returns stable
//! results, completes quickly, and provides actionable diagnostics regardless
//! of whether CUDA hardware is present.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_wrap
)]

use cudagrep::{AvailabilityReason, AvailabilityReport};

// =============================================================================
// Probe Safety
// =============================================================================

#[test]
fn probe_returns_bool_not_panic() {
    // Must never panic, even without CUDA hardware.
    let result = cudagrep::is_available();

    // Test that the availability report matches the probe result
    assert_eq!(result, cudagrep::availability_report().is_available());
}

#[test]
fn probe_result_stable_across_100_calls() {
    let first = cudagrep::is_available();
    for _ in 0..100 {
        assert_eq!(
            cudagrep::is_available(),
            first,
            "availability probe must be deterministic within a process"
        );
    }
}

#[test]
fn availability_report_returns_without_panic() {
    let report = cudagrep::availability_report();
    // Must be a valid report regardless of hardware.
    assert_eq!(report.is_available(), cudagrep::is_available());
    assert!(!format!("{:?}", report.reason()).is_empty());
    assert!(!report.diagnostics().is_empty() || cfg!(not(feature = "cuda")));
}

#[test]
fn availability_report_stable_across_calls() {
    let first = cudagrep::availability_report();
    let second = cudagrep::availability_report();
    assert_eq!(
        first.is_available(),
        second.is_available(),
        "availability report must be deterministic"
    );
    assert_eq!(first.reason(), second.reason());
}

// =============================================================================
// Diagnostics Quality
// =============================================================================

#[test]
fn availability_report_has_at_least_one_diagnostic() {
    let report = cudagrep::availability_report();
    assert!(
        !report.diagnostics().is_empty(),
        "report must provide at least one diagnostic message"
    );
}

#[test]
fn availability_report_diagnostics_are_actionable() {
    let report = cudagrep::availability_report();
    // Every diagnostic must contain a non-empty string.
    for diagnostic in report.diagnostics() {
        assert!(
            !diagnostic.trim().is_empty(),
            "diagnostic messages must not be blank"
        );
    }
}

#[test]
fn availability_report_display_is_nonempty() {
    let report = cudagrep::availability_report();
    let display = format!("{report}");
    assert!(
        !display.is_empty(),
        "Display impl must produce a non-empty string"
    );
}

// =============================================================================
// Reason Consistency
// =============================================================================

#[test]
fn available_true_implies_reason_ready() {
    let report = cudagrep::availability_report();
    if report.is_available() {
        assert_eq!(
            *report.reason(),
            AvailabilityReason::Ready,
            "available=true must have Ready reason"
        );
    }
}

#[test]
fn available_false_implies_reason_not_ready() {
    let report = cudagrep::availability_report();
    if !report.is_available() {
        assert_ne!(
            *report.reason(),
            AvailabilityReason::Ready,
            "available=false must not have Ready reason"
        );
    }
}

#[test]
fn without_cuda_feature_reason_is_feature_disabled() {
    // Without the `cuda` feature, we know the exact reason.
    if !cfg!(feature = "cuda") {
        let report = cudagrep::availability_report();
        assert_eq!(*report.reason(), AvailabilityReason::FeatureDisabled);
    }
}

// =============================================================================
// AvailabilityReport Construction
// =============================================================================

#[test]
fn report_new_available_true() {
    let report = AvailabilityReport::new(true, AvailabilityReason::Ready, vec!["test".to_owned()]);
    assert!(report.is_available());
    assert_eq!(*report.reason(), AvailabilityReason::Ready);
    assert_eq!(report.diagnostics().len(), 1);
}

#[test]
fn report_new_available_false() {
    let report = AvailabilityReport::new(
        false,
        AvailabilityReason::FeatureDisabled,
        vec!["reason1".to_owned(), "reason2".to_owned()],
    );
    assert!(!report.is_available());
    assert_eq!(*report.reason(), AvailabilityReason::FeatureDisabled);
    assert_eq!(report.diagnostics().len(), 2);
}

#[test]
fn report_new_empty_diagnostics() {
    let report = AvailabilityReport::new(false, AvailabilityReason::FeatureDisabled, Vec::new());
    assert!(report.diagnostics().is_empty());
}

#[test]
fn report_clone_equals_original() {
    let report = AvailabilityReport::new(true, AvailabilityReason::Ready, vec!["diag".to_owned()]);
    let cloned = report.clone();
    assert_eq!(report, cloned);
}

#[test]
fn report_debug_is_nonempty() {
    let report = cudagrep::availability_report();
    let debug = format!("{report:?}");
    assert!(!debug.is_empty());
}
