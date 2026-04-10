//! Adversarial tests for cudagrep.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_wrap
)]

use cudagrep::{validate_alignment, CudaError, GDS_ALIGNMENT};

const GDS_ALIGNMENT_I64: i64 = 4096;

#[test]
fn test_adversarial_alignment() {
    // Negative values
    let err_result = validate_alignment(-4096, GDS_ALIGNMENT, 0);
    assert!(err_result.is_err());
    let Err(err) = err_result else {
        panic!("expected Err")
    };
    assert!(matches!(err, CudaError::AlignmentViolation { .. }));

    let err_result = validate_alignment(0, GDS_ALIGNMENT, -4096);
    assert!(err_result.is_err());
    let Err(err) = err_result else {
        panic!("expected Err")
    };
    assert!(matches!(err, CudaError::AlignmentViolation { .. }));

    // Very large sizes
    let large_size = usize::MAX;
    let err_result = validate_alignment(0, large_size, 0);
    assert!(err_result.is_err());
    let Err(err) = err_result else {
        panic!("expected Err")
    };
    assert!(matches!(err, CudaError::AlignmentViolation { .. }));

    // Edge sizes exactly 1 byte off
    let err_result = validate_alignment(0, GDS_ALIGNMENT - 1, 0);
    assert!(err_result.is_err());
    let Err(err) = err_result else {
        panic!("expected Err")
    };
    assert!(matches!(err, CudaError::AlignmentViolation { .. }));
    let err_result = validate_alignment(0, GDS_ALIGNMENT + 1, 0);
    assert!(err_result.is_err());
    let Err(err) = err_result else {
        panic!("expected Err")
    };
    assert!(matches!(err, CudaError::AlignmentViolation { .. }));

    let err_result = validate_alignment(GDS_ALIGNMENT_I64 - 1, GDS_ALIGNMENT, 0);
    assert!(err_result.is_err());
    let Err(err) = err_result else {
        panic!("expected Err")
    };
    assert!(matches!(err, CudaError::AlignmentViolation { .. }));

    let err_result = validate_alignment(0, GDS_ALIGNMENT, GDS_ALIGNMENT_I64 - 1);
    assert!(err_result.is_err());
    let Err(err) = err_result else {
        panic!("expected Err")
    };
    assert!(matches!(err, CudaError::AlignmentViolation { .. }));

    // Offsets past i64::MAX (if passed as e.g., max values)
    assert!(validate_alignment(i64::MAX, GDS_ALIGNMENT, 0).is_err());
    assert!(validate_alignment(0, GDS_ALIGNMENT, i64::MAX).is_err());

    // Test that MAX values that might be multiples don't overflow wrongly.
    let max_aligned = (i64::MAX / GDS_ALIGNMENT_I64) * GDS_ALIGNMENT_I64;
    assert!(validate_alignment(max_aligned, GDS_ALIGNMENT, 0).is_ok());
    assert!(validate_alignment(0, GDS_ALIGNMENT, max_aligned).is_ok());
}
