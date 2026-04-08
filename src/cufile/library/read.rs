use std::marker::PhantomData;
use std::os::fd::RawFd;

use crate::cufile::error::{CuFileDriverError, ReadError};
use crate::cufile::ffi::{CUfileDescr, CUfileHandle, CUfileHandleType};
use crate::cufile::library::CuFileLibrary;
use crate::cufile::ops::DevicePointer;

impl CuFileLibrary {
    /// Reads from a file descriptor directly into GPU memory.
    ///
    /// # Errors
    ///
    /// Returns an error when handle registration fails, `cuFileRead` returns a
    /// negative status, or registered-handle cleanup fails.
    pub fn read_from_fd(
        &self,
        fd: RawFd,
        device_pointer: DevicePointer,
        size: usize,
        file_offset: i64,
        device_offset: i64,
    ) -> Result<usize, ReadError> {
        let registered = self.register(fd)?;
        let bytes_read = self.read_registered(
            registered.handle,
            fd,
            device_pointer,
            size,
            file_offset,
            device_offset,
        )?;

        drop(registered);
        Ok(bytes_read)
    }

    /// Registers a device buffer for use with cuFile operations.
    ///
    /// # Errors
    ///
    /// Returns a typed `CuFileDriverError` when registration fails.
    pub fn buf_register(
        &self,
        device_ptr: DevicePointer,
        size: usize,
    ) -> Result<(), CuFileDriverError> {
        let result =
            unsafe { (self.api.buf_register)(device_ptr.as_mut_ptr().cast_const(), size, 0) };
        if result.is_success() {
            Ok(())
        } else {
            Err(CuFileDriverError::from_raw(result.err))
        }
    }

    /// Deregisters a previously registered device buffer.
    ///
    /// # Errors
    ///
    /// Returns a typed `CuFileDriverError` when deregistration fails.
    pub fn buf_deregister(&self, device_ptr: DevicePointer) -> Result<(), CuFileDriverError> {
        let result = unsafe { (self.api.buf_deregister)(device_ptr.as_mut_ptr().cast_const()) };
        if result.is_success() {
            Ok(())
        } else {
            Err(CuFileDriverError::from_raw(result.err))
        }
    }

    pub(crate) fn register(&self, fd: RawFd) -> Result<RegisteredHandle<'_>, ReadError> {
        let handle = self.register_raw(fd)?;
        Ok(RegisteredHandle::new(self, fd, handle))
    }

    /// Register a file descriptor and return the raw handle without RAII.
    pub(crate) fn register_raw(&self, fd: RawFd) -> Result<CUfileHandle, ReadError> {
        let mut handle = CUfileHandle(std::ptr::null_mut());
        let mut descriptor = CUfileDescr {
            handle_type: CUfileHandleType::OpaqueFd,
            fd,
        };

        let result = unsafe { (self.api.handle_register)(&mut handle, &mut descriptor) };
        if result.is_success() {
            Ok(handle)
        } else {
            Err(ReadError::Register {
                fd,
                code: CuFileDriverError::from_raw(result.err),
            })
        }
    }

    /// Deregister a raw handle without RAII.
    pub(crate) fn deregister_raw(&self, fd: RawFd, handle: CUfileHandle) -> Result<(), ReadError> {
        self.deregister(fd, handle)
    }

    pub(crate) fn deregister(&self, fd: RawFd, handle: CUfileHandle) -> Result<(), ReadError> {
        let result = unsafe { (self.api.handle_deregister)(handle) };
        if result.is_success() {
            Ok(())
        } else {
            Err(ReadError::Cleanup {
                fd,
                code: CuFileDriverError::from_raw(result.err),
            })
        }
    }

    /// Reads from a registered handle into GPU memory.
    ///
    /// # Errors
    ///
    /// Returns `ReadError::Read` if the driver reports a failure.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid, registered cuFile handle
    /// - `device_pointer` must be a valid CUDA device pointer.
    pub(crate) fn read_registered(
        &self,
        handle: CUfileHandle,
        fd: RawFd,
        device_pointer: DevicePointer,
        size: usize,
        file_offset: i64,
        device_offset: i64,
    ) -> Result<usize, ReadError> {
        let result = unsafe {
            (self.api.read)(
                handle,
                device_pointer.as_mut_ptr(),
                size,
                file_offset,
                device_offset,
            )
        };

        if result < 0 {
            // Handle error code conversion without truncation
            // If the value is outside i32 range, clamp to i32::MIN which maps to Unknown
            let code = i32::try_from(result).unwrap_or_else(|_| {
                tracing::warn!(
                    raw_error = result,
                    "cuFileRead returned error code outside i32 range"
                );
                i32::MIN
            });
            Err(ReadError::Read {
                fd,
                code: CuFileDriverError::from_raw(code),
            })
        } else {
            usize::try_from(result).map_err(|_| ReadError::Read {
                fd,
                code: CuFileDriverError::Unknown(i32::MAX),
            })
        }
    }
}

/// RAII wrapper for a registered file handle.
pub(crate) struct RegisteredHandle<'library> {
    pub(crate) library: &'library CuFileLibrary,
    pub(crate) fd: RawFd,
    pub(crate) handle: CUfileHandle,
    _not_send_or_sync: PhantomData<*mut ()>,
}

impl<'library> RegisteredHandle<'library> {
    pub(crate) fn new(library: &'library CuFileLibrary, fd: RawFd, handle: CUfileHandle) -> Self {
        Self {
            library,
            fd,
            handle,
            _not_send_or_sync: PhantomData,
        }
    }
}

impl Drop for RegisteredHandle<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.library.deregister(self.fd, self.handle) {
            tracing::warn!(
                fd = self.fd,
                ?error,
                "cuFileHandleDeregister failed during cleanup"
            );
        }
    }
}
