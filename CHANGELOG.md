# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Comprehensive module-level documentation explaining GDS alignment requirements
- Documented registration cache lifecycle (cache on first use, evict with `evict_fd`, cleanup on Drop)
- Documented ShortRead error semantics and EOF/hardware fault detection
- Added `#[must_use]` attribute to `ReadStats`
- Enhanced `Drop` implementation with documented partial failure handling
- Legendary adversarial test suite (`tests/legendary.rs`) with 20+ tests:
  - Alignment validation for all combinations of aligned/misaligned offsets
  - Alignment edge cases (size=0, max values, overflow prevention)
  - ShortRead error message validation (contains both counts)
  - Actionable error messages for all `CudaError` variants
  - `ReadStats` zero value handling (no NaN/Inf)
  - `GDS_ALIGNMENT` constant validation (== 4096)
  - Feature disabled path testing
  - Empty batch operation handling
- Criterion benchmarks (`benches/legendary_bench.rs`):
  - Alignment validation speed
  - `ReadStats` calculation performance
  - Error formatting overhead
  - Availability report creation
  - Driver error code mapping
- Enhanced CI workflow:
  - Test without `cuda` feature
  - Clippy with pedantic warnings
  - Documentation build checks
  - Benchmark compilation verification
- Comprehensive README.md with:
  - Hardware requirements
  - Setup instructions
  - API examples for common use cases
  - Windows DirectStorage extension guide

### Changed

- Made `validate_alignment` public for pre-validation use cases
- Enhanced doc comments on all public items with examples

## [0.1.0] - 2026-03-29

### Added

- Initial release
- Extracted library logic into `src/lib.rs`, thin `src/main.rs` wrapper
- Added `is_available()` and `availability_report()` for graceful runtime diagnostics
- Confined all `unsafe` code to `src/cufile.rs` with documented safety invariants
- Safe wrappers and RAII cleanup for registered `cuFile` handles
- Non-CUDA and feature-gated CUDA tests
- Repository metadata, CI, contribution guidance, and licensing files
