//! Unit tests for cudagrep.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_wrap
)]

use cudagrep::{
    validate_alignment, AvailabilityReason, AvailabilityReport, CudaError, ReadStats, GDS_ALIGNMENT,
};
use std::time::Duration;

const GDS_ALIGNMENT_I64: i64 = 4096;

#[test]
fn test_read_stats_throughput_gbps() {
    // Zero time case
    let stats = ReadStats {
        bytes_transferred: 1024,
        wall_time: Duration::from_secs(0),
    };
    assert!(stats.throughput_gbps() == 0.0);

    // Normal case (1 GiB in 1 sec = 1.0 GiB/s)
    let stats = ReadStats {
        bytes_transferred: 1_073_741_824,
        wall_time: Duration::from_secs(1),
    };
    assert!((stats.throughput_gbps() - 1.0).abs() < 1e-9);

    // Half GiB in 0.5 sec = 1.0 GiB/s
    let stats = ReadStats {
        bytes_transferred: 1_073_741_824 / 2,
        wall_time: Duration::from_millis(500),
    };
    assert!((stats.throughput_gbps() - 1.0).abs() < 1e-9);
}

#[test]
fn test_availability_report_creation_and_getters() {
    let report = AvailabilityReport::new(
        true,
        AvailabilityReason::Ready,
        vec!["Everything is awesome".to_string()],
    );

    assert!(report.is_available());
    assert_eq!(report.reason(), &AvailabilityReason::Ready);
    assert_eq!(report.diagnostics(), &["Everything is awesome".to_string()]);
}

#[test]
fn test_availability_report_display() {
    let report = AvailabilityReport::new(
        false,
        AvailabilityReason::FeatureDisabled,
        vec!["Reason 1".to_string(), "Reason 2".to_string()],
    );

    let display = format!("{report}");
    assert!(display.contains("compiled without the `cuda` feature"));
    assert!(display.contains("[Reason 1; Reason 2]"));
}

#[test]
fn test_validate_alignment() {
    // Valid alignments
    assert!(validate_alignment(0, 0, 0).is_ok());
    assert!(validate_alignment(GDS_ALIGNMENT_I64, GDS_ALIGNMENT, GDS_ALIGNMENT_I64).is_ok());

    // Invalid file offset
    let err_result = validate_alignment(1, GDS_ALIGNMENT, 0);
    assert!(err_result.is_err());
    let Err(err) = err_result else {
        panic!("expected Err")
    };
    match err {
        CudaError::AlignmentViolation {
            file_offset,
            size,
            device_offset,
        } => {
            assert_eq!(file_offset, 1);
            assert_eq!(size, GDS_ALIGNMENT);
            assert_eq!(device_offset, 0);
        }
        _ => panic!("Expected AlignmentViolation"),
    }

    // Invalid size
    assert!(validate_alignment(0, 1, 0).is_err());

    // Invalid device offset
    assert!(validate_alignment(0, GDS_ALIGNMENT, 1).is_err());
}
