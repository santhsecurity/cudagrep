//! Property tests for cudagrep.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_wrap
)]

use cudagrep::{validate_alignment, ReadStats, GDS_ALIGNMENT};
use proptest::prelude::*;
use std::time::Duration;

const GDS_ALIGNMENT_I64: i64 = 4096;

proptest! {
    #[test]
    fn prop_validate_alignment_valid(
        file_factor in 0..100_000i64,
        size_factor in 0..100_000usize,
        device_factor in 0..100_000i64
    ) {
        let file_offset = file_factor * GDS_ALIGNMENT_I64;
        let size = size_factor * GDS_ALIGNMENT;
        let device_offset = device_factor * GDS_ALIGNMENT_I64;

        prop_assert!(validate_alignment(file_offset, size, device_offset).is_ok());
    }

    #[test]
    fn prop_validate_alignment_invalid_file_offset(
        file_factor in 0..100_000i64,
        size_factor in 0..100_000usize,
        device_factor in 0..100_000i64,
        offset in 1..GDS_ALIGNMENT_I64
    ) {
        let file_offset = file_factor * GDS_ALIGNMENT_I64 + offset;
        let size = size_factor * GDS_ALIGNMENT;
        let device_offset = device_factor * GDS_ALIGNMENT_I64;

        prop_assert!(validate_alignment(file_offset, size, device_offset).is_err());
    }

    #[test]
    fn prop_validate_alignment_invalid_size(
        file_factor in 0..100_000i64,
        size_factor in 0..100_000usize,
        device_factor in 0..100_000i64,
        offset in 1..GDS_ALIGNMENT
    ) {
        let file_offset = file_factor * GDS_ALIGNMENT_I64;
        let size = size_factor * GDS_ALIGNMENT + offset;
        let device_offset = device_factor * GDS_ALIGNMENT_I64;

        prop_assert!(validate_alignment(file_offset, size, device_offset).is_err());
    }

    #[test]
    fn prop_validate_alignment_invalid_device_offset(
        file_factor in 0..100_000i64,
        size_factor in 0..100_000usize,
        device_factor in 0..100_000i64,
        offset in 1..GDS_ALIGNMENT_I64
    ) {
        let file_offset = file_factor * GDS_ALIGNMENT_I64;
        let size = size_factor * GDS_ALIGNMENT;
        let device_offset = device_factor * GDS_ALIGNMENT_I64 + offset;

        prop_assert!(validate_alignment(file_offset, size, device_offset).is_err());
    }

    #[test]
    fn prop_read_stats_throughput_non_negative(
        bytes in 0..usize::MAX,
        secs in 0..u64::MAX,
        nanos in 0..1_000_000_000u32
    ) {
        let duration = Duration::new(secs, nanos);
        let stats = ReadStats {
            bytes_transferred: bytes,
            wall_time: duration,
        };
        prop_assert!(stats.throughput_gbps() >= 0.0);
    }
}
