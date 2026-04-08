#![allow(missing_docs)]

//! Benchmarks for cudagrep mock driver path.
//!
//! These measure overhead of the Rust-side code (error construction,
//! alignment validation, availability checking) since actual GPU
//! operations require CUDA hardware.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use cudagrep::cufile::CuFileDriverError;

fn bench_error_from_raw(c: &mut Criterion) {
    c.bench_function("CuFileDriverError::from_raw", |b| {
        b.iter(|| {
            for code in -20..20_i32 {
                black_box(CuFileDriverError::from_raw(black_box(code)));
            }
        });
    });
}

fn bench_availability_check(c: &mut Criterion) {
    c.bench_function("is_available", |b| {
        b.iter(|| {
            black_box(cudagrep::is_available());
        });
    });
}

fn bench_availability_report(c: &mut Criterion) {
    c.bench_function("availability_report", |b| {
        b.iter(|| {
            black_box(cudagrep::availability_report());
        });
    });
}

criterion_group!(
    benches,
    bench_error_from_raw,
    bench_availability_check,
    bench_availability_report,
);
criterion_main!(benches);
