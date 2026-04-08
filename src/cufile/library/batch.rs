use std::ffi::c_void;

use crate::cufile::error::CuFileDriverError;
use crate::cufile::ffi::{CUfileBatchIOFlags, CUfileIOParams};
use crate::cufile::library::CuFileLibrary;

impl CuFileLibrary {
    /// Sets up batch IO parameters.
    ///
    /// # Errors
    ///
    /// Returns a typed `CuFileDriverError` when setup fails.
    ///
    /// # Safety
    ///
    /// The `batch_params` pointer must point to valid memory that the driver
    /// can write to.
    pub unsafe fn batch_io_setup(
        &self,
        batch_params: *mut c_void,
        count: i32,
    ) -> Result<(), CuFileDriverError> {
        let result = unsafe { (self.api.batch_io_setup)(batch_params, count) };
        if result.is_success() {
            Ok(())
        } else {
            Err(CuFileDriverError::from_raw(result.err))
        }
    }

    /// Submits a batch of IO operations.
    ///
    /// # Errors
    ///
    /// Returns a typed `CuFileDriverError` when submission fails.
    ///
    /// # Safety
    ///
    /// The pointers passed must be valid for the duration of the operation.
    pub unsafe fn batch_io_submit(
        &self,
        batch_params: *mut c_void,
        count: i32,
        handle: *mut c_void,
        flags: i32,
    ) -> Result<(), CuFileDriverError> {
        let result = unsafe { (self.api.batch_io_submit)(batch_params, count, handle, flags) };
        if result.is_success() {
            Ok(())
        } else {
            Err(CuFileDriverError::from_raw(result.err))
        }
    }

    /// Submits a batch of IO operations using safe slices.
    ///
    /// # Errors
    ///
    /// Returns a typed `CuFileDriverError` when submission fails.
    ///
    /// # Safety
    ///
    /// - `params` must remain valid until the batch completes.
    /// - All handles in `params` must be valid registered cuFile handles.
    /// - All device pointers must be valid for the specified sizes.
    pub unsafe fn submit_batch_io(
        &self,
        params: &[CUfileIOParams],
        handle: Option<*mut c_void>,
        flags: CUfileBatchIOFlags,
    ) -> Result<(), CuFileDriverError> {
        let count = i32::try_from(params.len()).map_err(|_| {
            CuFileDriverError::OsError(-22) // EINVAL
        })?;

        let handle_ptr = handle.unwrap_or(std::ptr::null_mut());

        let result = unsafe {
            (self.api.batch_io_submit)(
                params.as_ptr().cast_mut().cast(),
                count,
                handle_ptr,
                flags.bits,
            )
        };

        if result.is_success() {
            Ok(())
        } else {
            Err(CuFileDriverError::from_raw(result.err))
        }
    }

    /// Cancels a pending batch IO operation.
    ///
    /// # Errors
    ///
    /// Returns a typed `CuFileDriverError` when cancellation fails.
    pub fn batch_io_cancel(&self, batch_id: u64) -> Result<(), CuFileDriverError> {
        let result = unsafe { (self.api.batch_io_cancel)(batch_id) };
        if result.is_success() {
            Ok(())
        } else {
            Err(CuFileDriverError::from_raw(result.err))
        }
    }
}
