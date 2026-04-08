//! Device-side pointers and transfer operations.

use std::ffi::c_void;
use std::ptr::NonNull;

/// Safe wrapper around a non-null CUDA device pointer.
///
/// This type guarantees the wrapped pointer is non-null. It does NOT guarantee
/// the pointer is valid — that remains the caller's responsibility.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DevicePointer(NonNull<c_void>);

impl DevicePointer {
    /// Creates a device pointer from a raw CUDA allocation.
    ///
    /// Returns `None` if `ptr` is null.
    ///
    /// # Safety
    ///
    /// `ptr` must be a valid, writable CUDA device pointer for the duration of
    /// any transfer that uses it. The safety requirements are identical to
    /// `std::ptr::NonNull::new_unchecked`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cudagrep::cufile::DevicePointer;
    /// use std::ffi::c_void;
    ///
    /// // Null pointer is rejected
    /// assert!(unsafe { DevicePointer::new(std::ptr::null_mut::<c_void>()) }.is_none());
    ///
    /// // Non-null pointer is accepted (but not validated)
    /// let mut byte = 0u8;
    /// let ptr = std::ptr::addr_of_mut!(byte).cast::<c_void>();
    /// assert!(unsafe { DevicePointer::new(ptr) }.is_some());
    /// ```
    pub unsafe fn new(ptr: *mut c_void) -> Option<Self> {
        NonNull::new(ptr).map(Self)
    }

    /// Returns the underlying raw pointer.
    #[must_use]
    pub fn as_mut_ptr(self) -> *mut c_void {
        self.0.as_ptr()
    }
}
