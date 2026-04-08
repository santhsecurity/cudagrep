#![allow(clippy::too_many_lines)]
//! `cudagrep` provides safe Rust bindings for NVIDIA `GPUDirect` Storage (GDS)
//! so applications can DMA data from `NVMe` storage directly into GPU memory.
//!
//! The crate wraps NVIDIA's `libcufile` runtime and exposes a safe API for:
//!
//! - probing whether the current process can use GDS,
//! - validating the 4 KiB alignment rules required by the hardware, and
//! - issuing direct NVMe-to-GPU reads once the driver is initialized.
//!
//! # Design Philosophy
//!
//! - **Graceful Degradation**: The crate degrades cleanly when CUDA/GDS is unavailable.
//!   Without the `cuda` feature or on machines without NVIDIA hardware, all APIs
//!   return clear error messages instead of panicking.
//! - **Safety**: All `unsafe` code is confined to [`cufile`].
//!   The public API is entirely safe Rust.
//! - **Performance**: Registration caching avoids ~50µs driver round-trips per
//!   file descriptor reuse.
//!
//! # Registration Cache Lifecycle
//!
//! The [`CuFileHardware`] type maintains an internal cache of `cuFileHandle`
//! registrations:
//!
//! 1. **Cache on first use**: When `read_to_device` or `read_batch` is called
//!    with a file descriptor for the first time, the fd is registered with the
//!    driver and cached for subsequent operations.
//!
//! 2. **Eviction**: Call [`CuFileHardware::evict_fd`] when a file descriptor
//!    is closed to prevent stale handle reuse. Returns `Ok(true)` if the fd
//!    was cached and successfully deregistered.
//!
//! 3. **Cleanup on Drop**: When [`CuFileHardware`] is dropped, all cached
//!    registrations are deregistered. Individual deregistration failures are
//!    logged but do not prevent cleanup of remaining handles.
//!
//! # GDS Alignment Requirements
//!
//! `GPUDirect` Storage requires **4 KiB (4096-byte) alignment** for all transfer
//! parameters. This is documented by NVIDIA as a hardware requirement:
//!
//! - File offsets must be 4 KiB aligned
//! - Device offsets must be 4 KiB aligned  
//! - Transfer sizes must be multiples of 4 KiB
//!
//! Use [`validate_alignment`] to pre-check parameters before initiating transfers.
//!
//! # `ShortRead` Error
//!
//! The [`CudaError::ShortRead`] error indicates that the driver returned 0 bytes
//! for a non-zero request. This typically means:
//!
//! - **EOF**: The file offset is at or past end-of-file
//! - **Hardware fault**: An unrecoverable `NVMe` or GPU error occurred
//!
//! # Quick Start
//!
//! ```no_run
//! use cudagrep::{availability_report, CuFileHardware, GDS_ALIGNMENT};
//! # use std::os::fd::RawFd;
//! # fn read_one_page(
//! #     fd: RawFd,
//! #     device_ptr: cudagrep::DevicePointer,
//! # ) -> Result<(), cudagrep::CudaError> {
//!     let report = availability_report();
//!     if report.is_available() {
//!         let mut gds = CuFileHardware::try_init()?;
//!         let bytes = gds.read_to_device(fd, device_ptr, GDS_ALIGNMENT, 0, 0)?;
//!         assert_eq!(bytes, GDS_ALIGNMENT);
//!     }
//! #   Ok(())
//! # }
//! ```
//!
//! # Features
//!
//! - `cuda`: Enables runtime probing and use of `libcufile.so.0`. Without this
//!   feature, the crate compiles but all operations return [`CudaError::FeatureDisabled`].

#![warn(missing_docs, clippy::pedantic)]
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

pub mod alignment;
pub mod availability;
/// Low-level `libcufile` FFI types and driver wrappers.
pub mod cufile;
pub mod error;
pub mod hardware;
pub mod ops;
pub mod stats;

pub use alignment::{validate_alignment, GDS_ALIGNMENT};
pub use availability::{availability_report, is_available, AvailabilityReason, AvailabilityReport};
pub use cufile::{
    CUfileBatchIOFlags, CUfileIOOpType, CUfileIOParams, CUfileIOStatus, DevicePointer,
};
pub use error::CudaError;
pub use hardware::{CuFileHardware, DEFAULT_MAX_TRANSFER_SIZE};
pub use ops::ReadOp;
pub use stats::ReadStats;

#[cfg(test)]
mod tests {
    #![allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn is_available_is_false_without_cuda_feature() {
        #[cfg(not(feature = "cuda"))]
        assert!(!is_available());

        #[cfg(feature = "cuda")]
        {
            let report = availability_report();
            assert_eq!(is_available(), report.is_available());
        }
    }

    #[test]
    fn availability_report_explains_feature_disabled_state() {
        let report = availability_report();

        #[cfg(not(feature = "cuda"))]
        {
            assert!(!report.is_available());
            assert_eq!(report.reason(), &AvailabilityReason::FeatureDisabled);
            assert!(report
                .diagnostics()
                .iter()
                .any(|item| item.contains("Rebuild with `--features cuda`")));
        }

        #[cfg(feature = "cuda")]
        {
            assert_eq!(
                report.is_available(),
                matches!(report.reason(), AvailabilityReason::Ready)
            );
            assert!(
                !report.diagnostics().is_empty(),
                "runtime probe should produce diagnostics"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn availability_report_handles_missing_library_gracefully() {
        let load_result = cufile::CuFileLibrary::load_from_path("/definitely/missing/libcufile.so");
        let Err(error) = load_result else {
            panic!("test path should not exist");
        };
        let report = AvailabilityReport::new(
            false,
            availability::map_load_reason(&error),
            vec!["forced missing library".to_owned()],
        );

        assert!(!report.is_available());
        assert!(matches!(
            report.reason(),
            AvailabilityReason::LibraryUnavailable(_)
        ));
    }
}
