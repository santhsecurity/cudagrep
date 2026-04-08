//! Public error types for `cudagrep`.

use std::os::fd::RawFd;

use crate::alignment::GDS_ALIGNMENT;
use crate::availability::AvailabilityReport;
use crate::cufile;

/// Errors returned by the safe GDS wrapper.
#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    /// The crate was compiled without the `cuda` feature.
    #[error("GPUDirect Storage support is disabled at compile time. Fix: rebuild with `--features cuda`.")]
    FeatureDisabled,
    /// GDS is unavailable on this machine.
    #[error("{report}. Fix: install NVIDIA GDS drivers and ensure libcufile.so is loadable.")]
    Unavailable {
        /// Detailed diagnostics describing the failure.
        report: AvailabilityReport,
    },
    /// The NVIDIA driver rejected initialization.
    #[error("cuFileDriverOpen failed: {code}. Fix: verify NVIDIA GDS driver installation and GPU compatibility.")]
    DriverInitRejected {
        /// The driver-provided error.
        code: cufile::CuFileDriverError,
    },
    /// Registering a file descriptor with GDS failed.
    #[error("cuFileHandleRegister failed for fd {fd}: {code}. Fix: ensure the file is on an NVMe device with GDS support.")]
    DescriptorRegistrationFailed {
        /// The file descriptor that was passed to `cuFileHandleRegister`.
        fd: RawFd,
        /// The driver-provided error code.
        code: cufile::CuFileDriverError,
    },
    /// Reading directly into GPU memory failed.
    #[error("cuFileRead failed for fd {fd}: {code}. Fix: verify GPU VRAM availability and alignment requirements.")]
    DirectMemoryAccessFailed {
        /// The file descriptor that was used for the transfer.
        fd: RawFd,
        /// The driver-provided error code.
        code: cufile::CuFileDriverError,
    },
    /// Dropping a registered handle failed during cleanup.
    #[error("cuFileHandleDeregister failed for fd {fd}: {code}. Fix: non-critical cleanup error — the handle may leak until process exit.")]
    DescriptorCleanupFailed {
        /// The file descriptor whose registration failed to clean up.
        fd: RawFd,
        /// The driver-provided error code.
        code: cufile::CuFileDriverError,
    },
    /// A file offset or transfer size was not aligned to the GDS requirement.
    #[error("GDS requires {GDS_ALIGNMENT}-byte alignment: file_offset={file_offset}, size={size}, device_offset={device_offset}. Fix: align all parameters to {GDS_ALIGNMENT} bytes (4 KiB).")]
    AlignmentViolation {
        /// The file offset that violated alignment.
        file_offset: i64,
        /// The transfer size that violated alignment.
        size: usize,
        /// The device offset that violated alignment.
        device_offset: i64,
    },
    /// The driver returned zero bytes for a non-zero request, indicating EOF
    /// or an unrecoverable hardware condition.
    #[error("short read: requested {requested} bytes, only {transferred} transferred. Fix: verify file length and check for EOF or NVMe errors.")]
    ShortRead {
        /// Total bytes originally requested.
        requested: usize,
        /// Bytes successfully transferred before the stall.
        transferred: usize,
    },
    /// The `ops` and `out` slices passed to `read_batch_into` have different lengths.
    #[error("batch length mismatch: ops.len()={ops_len} != out.len()={out_len}. Fix: ensure ops and out slices have equal length.")]
    BatchLengthMismatch {
        /// Length of the operations slice.
        ops_len: usize,
        /// Length of the output slice.
        out_len: usize,
    },
}

/// Maps low-level `ReadError` variants to public `CudaError` variants.
pub(crate) fn map_read_error(error: cufile::ReadError) -> CudaError {
    match error {
        cufile::ReadError::Register { fd, code } => {
            CudaError::DescriptorRegistrationFailed { fd, code }
        }
        cufile::ReadError::Read { fd, code } => CudaError::DirectMemoryAccessFailed { fd, code },
        cufile::ReadError::Cleanup { fd, code } => CudaError::DescriptorCleanupFailed { fd, code },
    }
}
