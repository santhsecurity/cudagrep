#![allow(missing_docs)]

//! Legendary benchmark suite for cudagrep.
//!
//! These benchmarks measure software-path performance:
//! - Alignment validation speed
//! - `ReadStats` calculation
//! - Error formatting
//!
//! Since GPU hardware is unavailable in CI, we focus on the overhead of
//! the Rust-side wrappers and error handling code.

#![allow(
    clippy::unwrap_used,
    clippy::cast_possible_wrap,
    clippy::used_underscore_binding
)]

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use cudagrep::{
    validate_alignment, AvailabilityReason, AvailabilityReport, CudaError, ReadStats, GDS_ALIGNMENT,
};
use std::ffi::c_void;
use std::time::Duration;

/// Benchmark alignment validation for various scenarios.
fn bench_alignment_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("alignment_validation");

    // Aligned case (fast path)
    group.bench_function("aligned", |b| {
        b.iter(|| {
            black_box(validate_alignment(
                black_box(0),
                black_box(GDS_ALIGNMENT),
                black_box(0),
            ))
        });
    });

    // Misaligned file_offset
    group.bench_function("misaligned_file_offset", |b| {
        b.iter(|| {
            black_box(validate_alignment(
                black_box(1),
                black_box(GDS_ALIGNMENT),
                black_box(0),
            ))
        });
    });

    // Misaligned size
    group.bench_function("misaligned_size", |b| {
        b.iter(|| black_box(validate_alignment(black_box(0), black_box(1), black_box(0))));
    });

    // Misaligned device_offset
    group.bench_function("misaligned_device_offset", |b| {
        b.iter(|| {
            black_box(validate_alignment(
                black_box(0),
                black_box(GDS_ALIGNMENT),
                black_box(1),
            ))
        });
    });

    // All misaligned (worst case)
    group.bench_function("all_misaligned", |b| {
        b.iter(|| black_box(validate_alignment(black_box(1), black_box(1), black_box(1))));
    });

    // Large aligned values
    group.bench_function("large_aligned", |b| {
        b.iter(|| {
            black_box(validate_alignment(
                black_box(4096 * 1_000_000),
                black_box(4096 * 1000),
                black_box(4096 * 5000),
            ))
        });
    });

    group.finish();
}

/// Benchmark `ReadStats` throughput calculation.
fn bench_readstats_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("readstats");

    // Zero bytes (edge case)
    group.bench_function("throughput_zero_bytes", |b| {
        let stats = ReadStats {
            bytes_transferred: 0,
            wall_time: Duration::from_secs(1),
        };
        b.iter(|| black_box(stats.throughput_gbps()));
    });

    // Zero time (edge case)
    group.bench_function("throughput_zero_time", |b| {
        let stats = ReadStats {
            bytes_transferred: 1_073_741_824,
            wall_time: Duration::ZERO,
        };
        b.iter(|| black_box(stats.throughput_gbps()));
    });

    // 1 GiB/s
    group.bench_function("throughput_1gibps", |b| {
        let stats = ReadStats {
            bytes_transferred: 1_073_741_824,
            wall_time: Duration::from_secs(1),
        };
        b.iter(|| black_box(stats.throughput_gbps()));
    });

    // 10 GiB/s (high bandwidth)
    group.bench_function("throughput_10gibps", |b| {
        let stats = ReadStats {
            bytes_transferred: 10 * 1_073_741_824,
            wall_time: Duration::from_secs(1),
        };
        b.iter(|| black_box(stats.throughput_gbps()));
    });

    // Very short duration (nanoseconds)
    group.bench_function("throughput_nanosecond", |b| {
        let stats = ReadStats {
            bytes_transferred: 4096,
            wall_time: Duration::from_nanos(1),
        };
        b.iter(|| black_box(stats.throughput_gbps()));
    });

    group.finish();
}

/// Benchmark error message formatting.
fn bench_error_formatting(c: &mut Criterion) {
    let mut group = c.benchmark_group("error_formatting");

    // FeatureDisabled (simple variant)
    group.bench_function("feature_disabled", |b| {
        b.iter(|| {
            let err = CudaError::FeatureDisabled;
            black_box(format!("{err}"));
        });
    });

    // AlignmentViolation (complex message with multiple values)
    group.bench_function("alignment_violation", |b| {
        b.iter(|| {
            let err = CudaError::AlignmentViolation {
                file_offset: 12345,
                size: 67890,
                device_offset: 11111,
            };
            black_box(format!("{err}"));
        });
    });

    // ShortRead (with counts)
    group.bench_function("short_read", |b| {
        b.iter(|| {
            let err = CudaError::ShortRead {
                requested: 16_777_216,
                transferred: 8_388_608,
            };
            black_box(format!("{err}"));
        });
    });

    // DescriptorRegistrationFailed (nested error)
    group.bench_function("registration_failed", |b| {
        b.iter(|| {
            let err = CudaError::DescriptorRegistrationFailed {
                fd: 42,
                code: cudagrep::cufile::CuFileDriverError::NotSupported,
            };
            black_box(format!("{err}"));
        });
    });

    // Unavailable (with report)
    group.bench_function("unavailable", |b| {
        b.iter_batched(
            || {
                AvailabilityReport::new(
                    false,
                    AvailabilityReason::LibraryUnavailable("libcufile.so.0 not found".to_owned()),
                    vec![
                        "Check LD_LIBRARY_PATH".to_owned(),
                        "Install NVIDIA cuFile".to_owned(),
                    ],
                )
            },
            |report| {
                let err = CudaError::Unavailable { report };
                black_box(format!("{err}"));
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

/// Benchmark `AvailabilityReport` operations.
fn bench_availability_report(c: &mut Criterion) {
    let mut group = c.benchmark_group("availability_report");

    group.bench_function("creation", |b| {
        b.iter(|| {
            black_box(AvailabilityReport::new(
                black_box(true),
                black_box(AvailabilityReason::Ready),
                black_box(vec!["test".to_owned()]),
            ));
        });
    });

    group.bench_function("display", |b| {
        let report = AvailabilityReport::new(
            false,
            AvailabilityReason::FeatureDisabled,
            vec!["compiled without cuda feature".to_owned()],
        );
        b.iter(|| {
            black_box(format!("{report}"));
        });
    });

    group.bench_function("display_with_diagnostics", |b| {
        let report = AvailabilityReport::new(
            false,
            AvailabilityReason::DriverRejected(cudagrep::cufile::CuFileDriverError::OsError(-1)),
            vec![
                "Loaded libcufile.so.0".to_owned(),
                "cuFileDriverOpen returned error".to_owned(),
                "Check driver version".to_owned(),
            ],
        );
        b.iter(|| {
            black_box(format!("{report}"));
        });
    });

    group.finish();
}

/// Benchmark `CuFileDriverError` `from_raw` mapping.
fn bench_driver_error_mapping(c: &mut Criterion) {
    use cudagrep::cufile::CuFileDriverError;

    let mut group = c.benchmark_group("driver_error_mapping");

    // Known error codes
    group.bench_function("known_os_error", |b| {
        b.iter(|| {
            black_box(CuFileDriverError::from_raw(black_box(-1)));
        });
    });

    group.bench_function("known_cuda_error", |b| {
        b.iter(|| {
            black_box(CuFileDriverError::from_raw(black_box(-2)));
        });
    });

    group.bench_function("known_not_supported", |b| {
        b.iter(|| {
            black_box(CuFileDriverError::from_raw(black_box(-3)));
        });
    });

    // Unknown error codes
    group.bench_function("unknown_positive", |b| {
        b.iter(|| {
            black_box(CuFileDriverError::from_raw(black_box(42)));
        });
    });

    group.bench_function("unknown_negative", |b| {
        b.iter(|| {
            black_box(CuFileDriverError::from_raw(black_box(-99)));
        });
    });

    // Range of codes (simulates real-world error handling)
    group.bench_function("range_of_codes", |b| {
        b.iter(|| {
            for code in -100..=100_i32 {
                black_box(CuFileDriverError::from_raw(black_box(code)));
            }
        });
    });

    group.finish();
}

/// Benchmark batch validation scenarios.
fn bench_batch_validation(c: &mut Criterion) {
    use cudagrep::cufile::DevicePointer;
    use cudagrep::ReadOp;

    let mut group = c.benchmark_group("batch_validation");

    // Single operation validation (via construction)
    group.bench_function("single_op_valid", |b| {
        b.iter(|| {
            let op = ReadOp {
                fd: 3,
                device_pointer: unsafe {
                    DevicePointer::new(
                        std::ptr::NonNull::<u8>::dangling()
                            .as_ptr()
                            .cast::<c_void>(),
                    )
                }
                .unwrap(),
                size: GDS_ALIGNMENT,
                file_offset: 0,
                device_offset: 0,
            };
            black_box(op);
        });
    });

    // Multiple operation creation
    group.bench_function("ten_ops_creation", |b| {
        b.iter(|| {
            let ptr = unsafe {
                DevicePointer::new(
                    std::ptr::NonNull::<u8>::dangling()
                        .as_ptr()
                        .cast::<c_void>(),
                )
            }
            .unwrap();
            let ops: Vec<ReadOp> = (0..10)
                .map(|i| ReadOp {
                    fd: 3,
                    device_pointer: ptr,
                    size: GDS_ALIGNMENT,
                    file_offset: (i * GDS_ALIGNMENT) as i64,
                    device_offset: (i * GDS_ALIGNMENT) as i64,
                })
                .collect();
            black_box(ops);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_alignment_validation,
    bench_readstats_calculation,
    bench_error_formatting,
    bench_availability_report,
    bench_driver_error_mapping,
    bench_batch_validation,
);
criterion_main!(benches);
