//! Gap tests for cudagrep.
//! These tests might fail if the implementation does not fully honor the API contract or edge cases in a correct way.
//! They represent findings and should ideally pass.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_wrap
)]

use cudagrep::{validate_alignment, ReadStats, GDS_ALIGNMENT};
use std::time::Duration;

const GDS_ALIGNMENT_I64: i64 = 4096;

#[test]
fn test_gap_alignment_max_i64_multiple() {
    // Determine the highest multiple of GDS_ALIGNMENT that fits in i64.
    let max_aligned = (i64::MAX / GDS_ALIGNMENT_I64) * GDS_ALIGNMENT_I64;

    // According to contract, any multiple of GDS_ALIGNMENT >= 0 is valid.
    // If there is an integer overflow or cast issue in `validate_alignment`, this might fail.
    assert!(
        validate_alignment(max_aligned, GDS_ALIGNMENT, 0).is_ok(),
        "Failed for max valid file_offset"
    );
    assert!(
        validate_alignment(0, GDS_ALIGNMENT, max_aligned).is_ok(),
        "Failed for max valid device_offset"
    );
}

#[test]
fn test_gap_alignment_max_usize_multiple() {
    // Determine the highest multiple of GDS_ALIGNMENT that fits in usize.
    let max_size = (usize::MAX / GDS_ALIGNMENT) * GDS_ALIGNMENT;

    // According to contract, any multiple of GDS_ALIGNMENT is a valid size.
    assert!(
        validate_alignment(0, max_size, 0).is_ok(),
        "Failed for max valid size"
    );
}

#[test]
fn test_gap_readstats_throughput_nan() {
    // If bytes_transferred is 0 and wall_time is 0, throughput_gbps should gracefully handle it.
    // Based on the code:
    // `if secs <= 0.0 { return 0.0; }`
    // So 0 time returns 0.0 throughput.

    let stats = ReadStats {
        bytes_transferred: 0,
        wall_time: Duration::from_secs(0),
    };

    let tp = stats.throughput_gbps();
    assert!(
        !tp.is_nan(),
        "Throughput should not be NaN for zero time/bytes"
    );
    assert!(tp == 0.0);
}

#[test]
fn test_gap_readstats_throughput_infinity() {
    // Extremely small non-zero duration shouldn't cause weird issues.
    let stats = ReadStats {
        bytes_transferred: usize::MAX,
        wall_time: Duration::from_nanos(1), // smallest positive non-zero
    };

    let tp = stats.throughput_gbps();
    assert!(!tp.is_infinite(), "Throughput should not be Infinity");
    assert!(!tp.is_nan(), "Throughput should not be NaN");
    assert!(tp > 0.0);
}
