//! Tests for alignment boundary errors.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_wrap
)]

use cudagrep::{validate_alignment, CudaError, GDS_ALIGNMENT};

const GDS_ALIGNMENT_I64: i64 = 4096;

#[test]
fn validate_alignment_rejects_off_by_one_offsets() {
    // Off by one positive
    assert!(validate_alignment(GDS_ALIGNMENT_I64 + 1, GDS_ALIGNMENT, 0).is_err());
    assert!(validate_alignment(0, GDS_ALIGNMENT, GDS_ALIGNMENT_I64 + 1).is_err());
    assert!(validate_alignment(0, GDS_ALIGNMENT + 1, 0).is_err());

    // Off by one negative (from aligned boundary)
    assert!(validate_alignment(GDS_ALIGNMENT_I64 - 1, GDS_ALIGNMENT, 0).is_err());
    assert!(validate_alignment(0, GDS_ALIGNMENT, GDS_ALIGNMENT_I64 - 1).is_err());
    assert!(validate_alignment(0, GDS_ALIGNMENT - 1, 0).is_err());
}

#[test]
fn validate_alignment_rejects_negative_offsets() {
    assert!(validate_alignment(-1, GDS_ALIGNMENT, 0).is_err());
    assert!(validate_alignment(0, GDS_ALIGNMENT, -1).is_err());
    assert!(validate_alignment(-GDS_ALIGNMENT_I64, GDS_ALIGNMENT, 0).is_err());
    assert!(validate_alignment(0, GDS_ALIGNMENT, -GDS_ALIGNMENT_I64).is_err());
}

#[test]
fn validate_alignment_handles_large_offsets_without_overflow() {
    // Large but aligned
    let large_aligned = (i64::MAX / GDS_ALIGNMENT_I64) * GDS_ALIGNMENT_I64;
    assert!(validate_alignment(large_aligned, GDS_ALIGNMENT, 0).is_ok());
    
    // Large unaligned
    assert!(validate_alignment(large_aligned - 1, GDS_ALIGNMENT, 0).is_err());

    // Boundary check around i64::MAX
    let result = validate_alignment(i64::MAX, GDS_ALIGNMENT, 0);
    assert!(result.is_err());
}

#[test]
fn validate_alignment_error_type() {
    let result = validate_alignment(1, GDS_ALIGNMENT, 0);
    match result {
        Err(CudaError::AlignmentViolation { file_offset, size, device_offset }) => {
            assert_eq!(file_offset, 1);
            assert_eq!(size, GDS_ALIGNMENT);
            assert_eq!(device_offset, 0);
        }
        _ => panic!("Expected AlignmentViolation"),
    }
}
