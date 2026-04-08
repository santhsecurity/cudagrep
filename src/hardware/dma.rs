use std::os::fd::RawFd;

use crate::alignment::validate_alignment;
use crate::cufile::{CUfileBatchIOFlags, CUfileHandle, CUfileIOParams, DevicePointer};
use crate::error::{map_read_error, CudaError};
use crate::hardware::device::CuFileHardware;
use crate::hardware::DEFAULT_MAX_TRANSFER_SIZE;
use crate::ops::ReadOp;
use crate::stats::ReadStats;

impl CuFileHardware {
    /// Reads data from an NVMe-backed file descriptor directly into GPU memory.
    ///
    /// The `file_offset`, `size`, and `device_offset` must each be aligned to
    /// [`GDS_ALIGNMENT`](crate::alignment::GDS_ALIGNMENT) (4 KiB). Reads larger than [`DEFAULT_MAX_TRANSFER_SIZE`]
    /// are automatically chunked.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::AlignmentViolation`] when any parameter breaks the
    /// 4 KiB alignment contract. Returns driver errors for registration,
    /// transfer, and cleanup failures.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudagrep::{CuFileHardware, GDS_ALIGNMENT};
    /// use cudagrep::cufile::DevicePointer;
    /// use std::ffi::c_void;
    ///
    /// # fn example(fd: i32, gpu_ptr: *mut c_void) -> Result<(), cudagrep::CudaError> {
    /// let mut gds = CuFileHardware::try_init()?;
    /// let device_pointer = unsafe { DevicePointer::new(gpu_ptr).unwrap() };
    /// let bytes = gds.read_to_device(fd, device_pointer, GDS_ALIGNMENT, 0, 0)?;
    /// assert_eq!(bytes, GDS_ALIGNMENT);
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_to_device(
        &mut self,
        fd: RawFd,
        device_pointer: DevicePointer,
        size: usize,
        file_offset: i64,
        device_offset: i64,
    ) -> Result<usize, CudaError> {
        validate_alignment(file_offset, size, device_offset)?;
        let handle = self
            .cache
            .get_or_register_for_bytes(fd, size, &self.library)?;
        self.read_with_handle(handle, fd, device_pointer, size, file_offset, device_offset)
    }

    /// Reads data from an NVMe-backed file descriptor directly into GPU memory.
    ///
    /// This is the unchecked variant of [`read_to_device`](Self::read_to_device) that
    /// skips alignment validation. Use this when alignment has been pre-verified.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `file_offset`, `size`, and `device_offset` are
    /// all aligned to [`GDS_ALIGNMENT`](crate::alignment::GDS_ALIGNMENT) (4 KiB). Violating this precondition may
    /// result in undefined behavior or hardware faults.
    ///
    /// # Errors
    ///
    /// Returns driver errors for registration, transfer, and cleanup failures.
    pub unsafe fn read_to_device_unchecked(
        &mut self,
        fd: RawFd,
        device_pointer: DevicePointer,
        size: usize,
        file_offset: i64,
        device_offset: i64,
    ) -> Result<usize, CudaError> {
        let handle = self
            .cache
            .get_or_register_for_bytes(fd, size, &self.library)?;
        self.read_with_handle(handle, fd, device_pointer, size, file_offset, device_offset)
    }

    #[allow(clippy::cast_possible_wrap)]
    fn read_with_handle(
        &self,
        handle: CUfileHandle,
        fd: RawFd,
        device_pointer: DevicePointer,
        size: usize,
        file_offset: i64,
        device_offset: i64,
    ) -> Result<usize, CudaError> {
        let mut total = 0_usize;
        let mut remaining = size;
        let mut f_off = file_offset;
        let mut d_off = device_offset;

        while remaining > 0 {
            let chunk = remaining.min(DEFAULT_MAX_TRANSFER_SIZE);
            let bytes = self
                .library
                .read_registered(handle, fd, device_pointer, chunk, f_off, d_off)
                .map_err(map_read_error)?;

            // A zero-byte transfer on a non-zero request means the device
            // hit EOF or an unrecoverable hardware condition. Looping would
            // spin forever. Return what we have, matching POSIX read() semantics.
            if bytes == 0 {
                return Err(CudaError::ShortRead {
                    requested: size,
                    transferred: total,
                });
            }

            if bytes < chunk {
                return Err(CudaError::ShortRead {
                    requested: size,
                    transferred: total + bytes,
                });
            }

            total = total.checked_add(bytes).ok_or(CudaError::ShortRead {
                requested: size,
                transferred: total,
            })?;
            remaining -= bytes;
            let bytes_i64 = i64::try_from(bytes).map_err(|_| CudaError::ShortRead {
                requested: size,
                transferred: total,
            })?;
            f_off = f_off.checked_add(bytes_i64).ok_or(CudaError::ShortRead {
                requested: size,
                transferred: total,
            })?;
            d_off = d_off.checked_add(bytes_i64).ok_or(CudaError::ShortRead {
                requested: size,
                transferred: total,
            })?;
        }

        Ok(total)
    }

    /// Executes a batch of read operations and returns per-operation byte counts.
    ///
    /// # Errors
    ///
    /// Returns the first error encountered during the batch.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudagrep::{CuFileHardware, ReadOp, GDS_ALIGNMENT};
    /// use cudagrep::cufile::DevicePointer;
    ///
    /// # fn example(fd: i32, ptr: DevicePointer) -> Result<(), cudagrep::CudaError> {
    /// let mut gds = CuFileHardware::try_init()?;
    /// let ops = vec![
    ///     ReadOp {
    ///         fd,
    ///         device_pointer: ptr,
    ///         size: GDS_ALIGNMENT,
    ///         file_offset: 0,
    ///         device_offset: 0,
    ///     },
    /// ];
    /// let (bytes, stats) = gds.read_batch(&ops)?;
    /// println!("Transferred {:?} bytes at {:.2} GiB/s", bytes, stats.throughput_gbps());
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_batch(&mut self, ops: &[ReadOp]) -> Result<(Vec<usize>, ReadStats), CudaError> {
        let start = std::time::Instant::now();
        let mut results = Vec::with_capacity(ops.len());
        let mut total_bytes = 0_usize;

        for op in ops {
            validate_alignment(op.file_offset, op.size, op.device_offset)?;
            let handle = self
                .cache
                .get_or_register_for_bytes(op.fd, op.size, &self.library)?;
            let bytes = self.read_with_handle(
                handle,
                op.fd,
                op.device_pointer,
                op.size,
                op.file_offset,
                op.device_offset,
            )?;
            total_bytes += bytes;
            results.push(bytes);
        }

        let stats = ReadStats {
            bytes_transferred: total_bytes,
            wall_time: start.elapsed(),
        };
        Ok((results, stats))
    }

    /// Execute a batch of read operations, writing per-operation byte counts
    /// into a caller-provided buffer.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::BatchLengthMismatch`] if `ops.len() != out.len()`.
    /// Returns the first transfer error encountered during the batch.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudagrep::{CuFileHardware, ReadOp, GDS_ALIGNMENT};
    /// use cudagrep::cufile::DevicePointer;
    ///
    /// # fn example(fd: i32, ptr: DevicePointer) -> Result<(), cudagrep::CudaError> {
    /// let mut gds = CuFileHardware::try_init()?;
    /// let ops = vec![
    ///     ReadOp {
    ///         fd,
    ///         device_pointer: ptr,
    ///         size: GDS_ALIGNMENT,
    ///         file_offset: 0,
    ///         device_offset: 0,
    ///     },
    /// ];
    /// let mut results = vec![0; ops.len()];
    /// let stats = gds.read_batch_into(&ops, &mut results)?;
    /// println!("Results: {:?} @ {:.2} GiB/s", results, stats.throughput_gbps());
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_batch_into(
        &mut self,
        ops: &[ReadOp],
        out: &mut [usize],
    ) -> Result<ReadStats, CudaError> {
        if ops.len() != out.len() {
            return Err(CudaError::BatchLengthMismatch {
                ops_len: ops.len(),
                out_len: out.len(),
            });
        }

        let start = std::time::Instant::now();
        let mut total_bytes = 0_usize;

        for (i, op) in ops.iter().enumerate() {
            validate_alignment(op.file_offset, op.size, op.device_offset)?;
            let handle = self
                .cache
                .get_or_register_for_bytes(op.fd, op.size, &self.library)?;
            let bytes = self.read_with_handle(
                handle,
                op.fd,
                op.device_pointer,
                op.size,
                op.file_offset,
                op.device_offset,
            )?;
            out[i] = bytes;
            total_bytes += bytes;
        }

        Ok(ReadStats {
            bytes_transferred: total_bytes,
            wall_time: start.elapsed(),
        })
    }

    /// Submits a batch of IO operations using the cuFile batch API.
    ///
    /// # Errors
    ///
    /// Returns a `CudaError::DirectMemoryAccessFailed` if submission fails.
    ///
    /// # Safety
    ///
    /// - The `params` slice must not be moved or reallocated until the batch completes.
    /// - All handles in `params` must be valid, registered cuFile handles.
    /// - All device pointers must be valid CUDA device pointers with sufficient size.
    /// - Alignment requirements (4 KiB) must be satisfied by the caller.
    pub unsafe fn submit_batch_io(
        &self,
        params: &[CUfileIOParams],
        flags: Option<CUfileBatchIOFlags>,
    ) -> Result<(), CudaError> {
        self.library
            .submit_batch_io(
                params,
                None,
                flags.unwrap_or_else(CUfileBatchIOFlags::empty),
            )
            .map_err(|code| CudaError::DirectMemoryAccessFailed { fd: -1, code })
    }
}
