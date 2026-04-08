//! Alignment validation for `GPUDirect` Storage.

use crate::error::CudaError;

/// GDS alignment requirement in bytes (4 KiB).
///
/// NVIDIA GDS hardware requires 4 KiB alignment because `NVMe` DMA and GPU BAR
/// mappings operate at the memory page granularity. This is a hardware
/// requirement that cannot be relaxed.
pub const GDS_ALIGNMENT: usize = 4096;

/// Validates that all parameters meet GDS 4 KiB alignment requirements.
///
/// # Errors
///
/// Returns [`CudaError::AlignmentViolation`] if any parameter is not aligned
/// or if either offset is negative.
///
/// # Examples
///
/// ```
/// use cudagrep::validate_alignment;
/// use cudagrep::GDS_ALIGNMENT;
///
/// // Valid alignment
/// assert!(validate_alignment(0, GDS_ALIGNMENT, 0).is_ok());
///
/// // Invalid: file_offset not aligned
/// assert!(validate_alignment(1, GDS_ALIGNMENT, 0).is_err());
/// ```
pub fn validate_alignment(
    file_offset: i64,
    size: usize,
    device_offset: i64,
) -> Result<(), CudaError> {
    let aligned = file_offset >= 0
        && device_offset >= 0
        && usize::try_from(file_offset).is_ok_and(|offset| offset % GDS_ALIGNMENT == 0)
        && size % GDS_ALIGNMENT == 0
        && usize::try_from(device_offset).is_ok_and(|offset| offset % GDS_ALIGNMENT == 0);

    if aligned {
        Ok(())
    } else {
        Err(CudaError::AlignmentViolation {
            file_offset,
            size,
            device_offset,
        })
    }
}
