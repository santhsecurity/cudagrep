//! Adversarial tests for buffer alignment requirements.
//!
//! GPUDirect Storage requires 4 KiB alignment for all parameters.
//! These tests verify alignment validation handles edge cases correctly.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_wrap
)]

use cudagrep::{validate_alignment, CudaError, GDS_ALIGNMENT};

// =============================================================================
// Basic Alignment Tests
// =============================================================================

#[test]
fn alignment_rejects_all_misaligned_combinations() {
    // Test every combination of aligned/misaligned parameters
    let aligned = GDS_ALIGNMENT_I64;
    let misaligned = aligned + 1;
    
    // All aligned - should succeed
    assert!(validate_alignment(aligned, GDS_ALIGNMENT, aligned).is_ok());
    
    // One misaligned - should fail
    assert!(validate_alignment(misaligned, GDS_ALIGNMENT, aligned).is_err());
    assert!(validate_alignment(aligned, GDS_ALIGNMENT + 1, aligned).is_err());
    assert!(validate_alignment(aligned, GDS_ALIGNMENT, misaligned).is_err());
    
    // Two misaligned - should fail
    assert!(validate_alignment(misaligned, GDS_ALIGNMENT + 1, aligned).is_err());
    assert!(validate_alignment(misaligned, GDS_ALIGNMENT, misaligned).is_err());
    assert!(validate_alignment(aligned, GDS_ALIGNMENT + 1, misaligned).is_err());
    
    // All misaligned - should fail
    assert!(validate_alignment(misaligned, GDS_ALIGNMENT + 1, misaligned).is_err());
}

// =============================================================================
// Boundary Value Tests
// =============================================================================

#[test]
fn alignment_boundary_just_below_aligned() {
    // One byte below alignment boundary
    let offset = GDS_ALIGNMENT_I64 - 1;
    let result = validate_alignment(offset, GDS_ALIGNMENT, 0);
    assert!(result.is_err());
    
    match result {
        Err(CudaError::AlignmentViolation { file_offset, .. }) => {
            assert_eq!(file_offset, offset);
        }
        _ => panic!("Expected AlignmentViolation"),
    }
}

#[test]
fn alignment_boundary_just_above_aligned() {
    // One byte above alignment boundary
    let offset = GDS_ALIGNMENT_I64 + 1;
    let result = validate_alignment(offset, GDS_ALIGNMENT, 0);
    assert!(result.is_err());
    
    match result {
        Err(CudaError::AlignmentViolation { file_offset, .. }) => {
            assert_eq!(file_offset, offset);
        }
        _ => panic!("Expected AlignmentViolation"),
    }
}

#[test]
fn alignment_exact_boundary_accepted() {
    // Exactly at alignment boundary - should succeed
    assert!(validate_alignment(0, GDS_ALIGNMENT, 0).is_ok());
    assert!(validate_alignment(GDS_ALIGNMENT_I64, GDS_ALIGNMENT, 0).is_ok());
    assert!(validate_alignment((GDS_ALIGNMENT * 2) as i64, GDS_ALIGNMENT, 0).is_ok());
}

// =============================================================================
// Integer Overflow Protection
// =============================================================================

#[test]
fn alignment_handles_i64_max_without_panic() {
    // i64::MAX is definitely not aligned (it's odd)
    let result = validate_alignment(i64::MAX, GDS_ALIGNMENT, 0);
    assert!(result.is_err());
}

#[test]
fn alignment_handles_i64_max_minus_alignment() {
    // Largest aligned value less than i64::MAX
    let max_aligned = (i64::MAX / GDS_ALIGNMENT_I64) * GDS_ALIGNMENT_I64;
    let result = validate_alignment(max_aligned, GDS_ALIGNMENT, 0);
    assert!(result.is_ok());
}

#[test]
fn alignment_handles_usize_max_size() {
    // Very large size that might overflow
    let aligned_size = usize::MAX - (usize::MAX % GDS_ALIGNMENT);
    let result = validate_alignment(0, aligned_size, 0);
    // Should not panic - result depends on implementation
    let _ = result;
}

#[test]
fn alignment_zero_values_accepted() {
    // Zero is aligned to any value (0 % N == 0)
    assert!(validate_alignment(0, 0, 0).is_ok());
    assert!(validate_alignment(0, GDS_ALIGNMENT, 0).is_ok());
    assert!(validate_alignment(GDS_ALIGNMENT_I64, 0, GDS_ALIGNMENT_I64).is_ok());
}

// =============================================================================
// Negative Offset Handling
// =============================================================================

#[test]
fn alignment_rejects_negative_file_offset() {
    assert!(validate_alignment(-1, GDS_ALIGNMENT, 0).is_err());
    assert!(validate_alignment(-4096, GDS_ALIGNMENT, 0).is_err());
    assert!(validate_alignment(i64::MIN, GDS_ALIGNMENT, 0).is_err());
}

#[test]
fn alignment_rejects_negative_device_offset() {
    assert!(validate_alignment(0, GDS_ALIGNMENT, -1).is_err());
    assert!(validate_alignment(0, GDS_ALIGNMENT, -4096).is_err());
    assert!(validate_alignment(0, GDS_ALIGNMENT, i64::MIN).is_err());
}

#[test]
fn alignment_negative_offset_error_contains_value() {
    let result = validate_alignment(-4096, GDS_ALIGNMENT, 0);
    match result {
        Err(CudaError::AlignmentViolation { file_offset, .. }) => {
            assert_eq!(file_offset, -4096);
        }
        _ => panic!("Expected AlignmentViolation with negative offset"),
    }
}

// =============================================================================
// Error Message Quality
// =============================================================================

#[test]
fn alignment_error_includes_all_parameters() {
    let result = validate_alignment(100, 200, 300);
    match result {
        Err(CudaError::AlignmentViolation { file_offset, size, device_offset }) => {
            assert_eq!(file_offset, 100);
            assert_eq!(size, 200);
            assert_eq!(device_offset, 300);
        }
        _ => panic!("Expected AlignmentViolation"),
    }
}

#[test]
fn alignment_error_display_includes_requirement() {
    let err = CudaError::AlignmentViolation {
        file_offset: 1,
        size: 2,
        device_offset: 3,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("4096") || msg.contains("alignment"),
        "Error should mention alignment requirement: {}", msg
    );
}

// =============================================================================
// Large Alignment Values
// =============================================================================

#[test]
fn alignment_handles_very_large_aligned_offset() {
    // 1 TB aligned offset
    let tb: i64 = 1024 * 1024 * 1024 * 1024;
    let aligned_tb = (tb / GDS_ALIGNMENT_I64) * GDS_ALIGNMENT_I64;
    assert!(validate_alignment(aligned_tb, GDS_ALIGNMENT, 0).is_ok());
}

#[test]
fn alignment_handles_very_large_unaligned_offset() {
    // 1 TB + 1 byte
    let tb_plus_1: i64 = 1024 * 1024 * 1024 * 1024 + 1;
    assert!(validate_alignment(tb_plus_1, GDS_ALIGNMENT, 0).is_err());
}

// =============================================================================
// Size Alignment Edge Cases
// =============================================================================

#[test]
fn alignment_size_one_byte_over() {
    // Size that's one byte over alignment
    let result = validate_alignment(0, GDS_ALIGNMENT + 1, 0);
    assert!(result.is_err());
}

#[test]
fn alignment_size_one_byte_under() {
    // Size that's one byte under alignment
    let result = validate_alignment(0, GDS_ALIGNMENT - 1, 0);
    assert!(result.is_err());
}

#[test]
fn alignment_size_exact_multiple() {
    // Sizes that are exact multiples of alignment
    for n in 1..=10 {
        let size = GDS_ALIGNMENT * n;
        assert!(
            validate_alignment(0, size, 0).is_ok(),
            "Size {} * {} should be aligned", GDS_ALIGNMENT, n
        );
    }
}

// =============================================================================
// Concurrent Alignment Validation
// =============================================================================

#[test]
fn alignment_validation_thread_safe() {
    use std::thread;

const GDS_ALIGNMENT_I64: i64 = 4096;
    
    let mut handles = vec![];
    
    // Spawn threads validating different alignments
    for i in 0..100 {
        handles.push(thread::spawn(move || {
            let offset = (i * GDS_ALIGNMENT) as i64;
            validate_alignment(offset, GDS_ALIGNMENT, 0)
        }));
    }
    
    // All should succeed
    for handle in handles {
        let result = handle.join().expect("Thread panicked");
        assert!(result.is_ok());
    }
}
