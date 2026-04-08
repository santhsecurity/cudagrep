//! Adversarial tests for DMA (Direct Memory Access) safety.
//!
//! These tests verify that NVMe-to-GPU transfers handle edge cases safely,
//! including invalid file descriptors, null pointers, and concurrent access.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_wrap
)]

use cudagrep::{CuFileHardware, CudaError, DevicePointer, GDS_ALIGNMENT, ReadOp};
use std::ffi::c_void;
use std::os::fd::AsRawFd;

// =============================================================================
// Invalid File Descriptor Handling
// =============================================================================

#[test]
fn dma_rejects_negative_fd() {
    let mut gds = match CuFileHardware::try_init() {
        Ok(gds) => gds,
        Err(_) => return, // Skip if no hardware
    };
    
    let mut byte = 0u8;
    let ptr = std::ptr::addr_of_mut!(byte).cast::<c_void>();
    let device_ptr = unsafe { DevicePointer::new(ptr) }.unwrap();
    
    let result = gds.read_to_device(-1, device_ptr, GDS_ALIGNMENT, 0, 0);
    assert!(result.is_err(), "Negative fd must be rejected");
}

#[test]
fn dma_rejects_very_large_fd() {
    let mut gds = match CuFileHardware::try_init() {
        Ok(gds) => gds,
        Err(_) => return, // Skip if no hardware
    };
    
    let mut byte = 0u8;
    let ptr = std::ptr::addr_of_mut!(byte).cast::<c_void>();
    let device_ptr = unsafe { DevicePointer::new(ptr) }.unwrap();
    
    let result = gds.read_to_device(i32::MAX, device_ptr, GDS_ALIGNMENT, 0, 0);
    assert!(result.is_err(), "Very large fd must be rejected");
}

#[test]
fn dma_rejects_closed_fd() {
    let mut gds = match CuFileHardware::try_init() {
        Ok(gds) => gds,
        Err(_) => return, // Skip if no hardware
    };
    
    let mut byte = 0u8;
    let ptr = std::ptr::addr_of_mut!(byte).cast::<c_void>();
    let device_ptr = unsafe { DevicePointer::new(ptr) }.unwrap();
    
    // Create a file and get its fd, then close it
    let fd = {
        let file = tempfile::tempfile().unwrap();
        file.as_raw_fd()
    }; // File is closed here
    
    let result = gds.read_to_device(fd, device_ptr, GDS_ALIGNMENT, 0, 0);
    assert!(result.is_err(), "Closed fd must be rejected");
}

// =============================================================================
// Device Pointer Validation
// =============================================================================

#[test]
fn device_pointer_null_rejected() {
    let ptr = unsafe { DevicePointer::new(std::ptr::null_mut::<c_void>()) };
    assert!(ptr.is_none(), "Null pointer must be rejected");
}

#[test]
fn device_pointer_non_null_accepted() {
    let mut byte = 0u8;
    let raw = std::ptr::addr_of_mut!(byte).cast::<c_void>();
    let ptr = unsafe { DevicePointer::new(raw) };
    assert!(ptr.is_some(), "Non-null pointer must be accepted");
}

#[test]
fn device_pointer_dangling_accepted() {
    // DevicePointer only checks for null, not validity
    let dangling = std::ptr::NonNull::<u8>::dangling().as_ptr().cast::<c_void>();
    let ptr = unsafe { DevicePointer::new(dangling) };
    assert!(ptr.is_some(), "Non-null dangling pointer is accepted (validity not checked)");
}

// =============================================================================
// Zero-Size Transfer Handling
// =============================================================================

#[test]
fn dma_zero_size_alignment_check_only() {
    let mut gds = match CuFileHardware::try_init() {
        Ok(gds) => gds,
        Err(_) => return, // Skip if no hardware
    };
    
    let mut byte = 0u8;
    let ptr = std::ptr::addr_of_mut!(byte).cast::<c_void>();
    let device_ptr = unsafe { DevicePointer::new(ptr) }.unwrap();
    
    // Zero size with aligned offsets - alignment check only
    let result = gds.read_to_device(-1, device_ptr, 0, 0, 0);
    
    // Should either succeed with 0 bytes or fail due to invalid fd
    // but must not panic or cause undefined behavior
    match result {
        Ok(0) => (), // Expected
        Err(_) => (), // Also acceptable
    }
}

// =============================================================================
// Batch Operation Safety
// =============================================================================

#[test]
fn batch_empty_operations_succeeds() {
    let mut gds = match CuFileHardware::try_init() {
        Ok(gds) => gds,
        Err(_) => return, // Skip if no hardware
    };
    
    let ops: Vec<ReadOp> = vec![];
    let result = gds.read_batch(&ops);
    
    // Empty batch should succeed with empty results
    assert!(result.is_ok());
    let (results, _) = result.unwrap();
    assert!(results.is_empty());
}

#[test]
fn batch_single_invalid_fd_fails_gracefully() {
    let mut gds = match CuFileHardware::try_init() {
        Ok(gds) => gds,
        Err(_) => return, // Skip if no hardware
    };
    
    let mut byte = 0u8;
    let ptr = std::ptr::addr_of_mut!(byte).cast::<c_void>();
    let device_ptr = unsafe { DevicePointer::new(ptr) }.unwrap();
    
    let ops = vec![ReadOp {
        fd: -1,
        device_pointer: device_ptr,
        size: GDS_ALIGNMENT,
        file_offset: 0,
        device_offset: 0,
    }];
    
    let result = gds.read_batch(&ops);
    assert!(result.is_err(), "Batch with invalid fd must fail");
}

#[test]
fn batch_mixed_validity_all_fail_on_first_error() {
    let mut gds = match CuFileHardware::try_init() {
        Ok(gds) => gds,
        Err(_) => return, // Skip if no hardware
    };
    
    let mut byte = 0u8;
    let ptr = std::ptr::addr_of_mut!(byte).cast::<c_void>();
    let device_ptr = unsafe { DevicePointer::new(ptr) }.unwrap();
    
    // Multiple operations with invalid fd
    let ops = vec![
        ReadOp {
            fd: -1,
            device_pointer: device_ptr,
            size: GDS_ALIGNMENT,
            file_offset: 0,
            device_offset: 0,
        },
        ReadOp {
            fd: -2,
            device_pointer: device_ptr,
            size: GDS_ALIGNMENT,
            file_offset: GDS_ALIGNMENT as i64,
            device_offset: GDS_ALIGNMENT as i64,
        },
    ];
    
    let result = gds.read_batch(&ops);
    assert!(result.is_err(), "Batch with invalid fds must fail");
}

// =============================================================================
// Large Transfer Handling
// =============================================================================

#[test]
fn dma_very_large_size_fails_gracefully() {
    let mut gds = match CuFileHardware::try_init() {
        Ok(gds) => gds,
        Err(_) => return, // Skip if no hardware
    };
    
    let mut byte = 0u8;
    let ptr = std::ptr::addr_of_mut!(byte).cast::<c_void>();
    let device_ptr = unsafe { DevicePointer::new(ptr) }.unwrap();
    
    // Request larger than address space
    let result = gds.read_to_device(-1, device_ptr, usize::MAX, 0, 0);
    
    // Must not panic - error is expected
    assert!(result.is_err());
}

#[test]
fn dma_max_aligned_size() {
    let mut gds = match CuFileHardware::try_init() {
        Ok(gds) => gds,
        Err(_) => return, // Skip if no hardware
    };
    
    let mut byte = 0u8;
    let ptr = std::ptr::addr_of_mut!(byte).cast::<c_void>();
    let device_ptr = unsafe { DevicePointer::new(ptr) }.unwrap();
    
    // Large but aligned size
    let large_size = 16 * 1024 * 1024; // 16 MiB (default chunk size)
    assert_eq!(large_size % GDS_ALIGNMENT, 0);
    
    let result = gds.read_to_device(-1, device_ptr, large_size, 0, 0);
    // Will fail due to invalid fd but shouldn't panic on size
    assert!(result.is_err());
}

// =============================================================================
// Offset Overflow Protection
// =============================================================================

#[test]
fn dma_large_file_offset_no_overflow() {
    let mut gds = match CuFileHardware::try_init() {
        Ok(gds) => gds,
        Err(_) => return, // Skip if no hardware
    };
    
    let mut byte = 0u8;
    let ptr = std::ptr::addr_of_mut!(byte).cast::<c_void>();
    let device_ptr = unsafe { DevicePointer::new(ptr) }.unwrap();
    
    // Very large file offset (near i64 max but aligned)
    let large_offset = (i64::MAX / GDS_ALIGNMENT as i64) * GDS_ALIGNMENT as i64;
    
    let result = gds.read_to_device(-1, device_ptr, GDS_ALIGNMENT, large_offset, 0);
    // Should not panic - will fail due to invalid fd
    assert!(result.is_err());
}

#[test]
fn dma_large_device_offset_no_overflow() {
    let mut gds = match CuFileHardware::try_init() {
        Ok(gds) => gds,
        Err(_) => return, // Skip if no hardware
    };
    
    let mut byte = 0u8;
    let ptr = std::ptr::addr_of_mut!(byte).cast::<c_void>();
    let device_ptr = unsafe { DevicePointer::new(ptr) }.unwrap();
    
    // Very large device offset (near i64 max but aligned)
    let large_offset = (i64::MAX / GDS_ALIGNMENT as i64) * GDS_ALIGNMENT as i64;
    
    let result = gds.read_to_device(-1, device_ptr, GDS_ALIGNMENT, 0, large_offset);
    // Should not panic - will fail due to invalid fd
    assert!(result.is_err());
}

// =============================================================================
// Cache Eviction Safety
// =============================================================================

#[test]
fn evict_never_registered_fd_returns_false() {
    let mut gds = match CuFileHardware::try_init() {
        Ok(gds) => gds,
        Err(_) => return, // Skip if no hardware
    };
    
    // Evict an fd that was never registered
    let result = gds.evict_fd(9999);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), false, "Evicting unregistered fd returns false");
}

#[test]
fn evict_negative_fd_returns_false() {
    let mut gds = match CuFileHardware::try_init() {
        Ok(gds) => gds,
        Err(_) => return, // Skip if no hardware
    };
    
    let result = gds.evict_fd(-1);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), false);
}

#[test]
fn evict_large_fd_returns_false() {
    let mut gds = match CuFileHardware::try_init() {
        Ok(gds) => gds,
        Err(_) => return, // Skip if no hardware
    };
    
    let result = gds.evict_fd(i32::MAX);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), false);
}

// =============================================================================
// Drop Safety
// =============================================================================

#[test]
fn drop_with_no_registrations_does_not_panic() {
    let gds = match CuFileHardware::try_init() {
        Ok(gds) => gds,
        Err(_) => return, // Skip if no hardware
    };
    
    // Drop with no cached registrations
    drop(gds);
}

#[test]
fn drop_after_evicting_all_does_not_panic() {
    let mut gds = match CuFileHardware::try_init() {
        Ok(gds) => gds,
        Err(_) => return, // Skip if no hardware
    };
    
    // Evict some fds (that don't exist)
    let _ = gds.evict_fd(100);
    let _ = gds.evict_fd(200);
    
    // Drop after evicting
    drop(gds);
}

// =============================================================================
// Unchecked Read Safety
// =============================================================================

#[test]
fn unchecked_read_with_invalid_fd_fails() {
    let mut gds = match CuFileHardware::try_init() {
        Ok(gds) => gds,
        Err(_) => return, // Skip if no hardware
    };
    
    let mut byte = 0u8;
    let ptr = std::ptr::addr_of_mut!(byte).cast::<c_void>();
    let device_ptr = unsafe { DevicePointer::new(ptr) }.unwrap();
    
    // Unchecked read still validates fd
    let result = unsafe {
        gds.read_to_device_unchecked(-1, device_ptr, GDS_ALIGNMENT, 0, 0)
    };
    
    assert!(result.is_err());
}

// =============================================================================
// Batch Length Mismatch
// =============================================================================

#[test]
fn read_batch_into_mismatched_lengths_fails() {
    let mut gds = match CuFileHardware::try_init() {
        Ok(gds) => gds,
        Err(_) => return, // Skip if no hardware
    };
    
    let mut byte = 0u8;
    let ptr = std::ptr::addr_of_mut!(byte).cast::<c_void>();
    let device_ptr = unsafe { DevicePointer::new(ptr) }.unwrap();
    
    let ops = vec![
        ReadOp {
            fd: -1,
            device_pointer: device_ptr,
            size: GDS_ALIGNMENT,
            file_offset: 0,
            device_offset: 0,
        },
    ];
    
    // Output buffer too small
    let mut out = vec![];
    
    let result = gds.read_batch_into(&ops, &mut out);
    
    match result {
        Err(CudaError::BatchLengthMismatch { ops_len, out_len }) => {
            assert_eq!(ops_len, 1);
            assert_eq!(out_len, 0);
        }
        _ => panic!("Expected BatchLengthMismatch"),
    }
}
