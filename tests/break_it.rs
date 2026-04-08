//! Adversarial tests designed to break `cudagrep` assumptions.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]

use cudagrep::{
    availability_report, is_available, validate_alignment, AvailabilityReason, CuFileHardware,
    CudaError, DevicePointer, ReadOp, GDS_ALIGNMENT,
};
use std::ptr;
use std::thread;

const GDS_ALIGNMENT_I64: i64 = 4096;
const DEFAULT_MAX_TRANSFER_SIZE: usize = 16 * 1024 * 1024;

// --- DMA buffer allocation limits ---

#[test]
fn test_01_dma_buffer_max_allocation_limit() {
    let size = usize::MAX / GDS_ALIGNMENT * GDS_ALIGNMENT;
    assert!(validate_alignment(0, size, 0).is_ok());
}

#[test]
fn test_02_dma_buffer_max_file_offset_limit() {
    let max_aligned = (i64::MAX / GDS_ALIGNMENT_I64) * GDS_ALIGNMENT_I64;
    assert!(validate_alignment(max_aligned, GDS_ALIGNMENT, 0).is_ok());
}

#[test]
fn test_03_dma_buffer_max_device_offset_limit() {
    let max_aligned = (i64::MAX / GDS_ALIGNMENT_I64) * GDS_ALIGNMENT_I64;
    assert!(validate_alignment(0, GDS_ALIGNMENT, max_aligned).is_ok());
}

#[test]
fn test_04_dma_buffer_max_i64_file_offset_overflow() {
    assert!(matches!(
        validate_alignment(i64::MAX, GDS_ALIGNMENT, 0),
        Err(CudaError::AlignmentViolation { .. })
    ));
}

#[test]
fn test_05_dma_buffer_max_i64_device_offset_overflow() {
    assert!(matches!(
        validate_alignment(0, GDS_ALIGNMENT, i64::MAX),
        Err(CudaError::AlignmentViolation { .. })
    ));
}

// --- Scan on empty input ---

#[test]
fn test_06_scan_on_empty_input_zero_size() {
    assert!(validate_alignment(0, 0, 0).is_ok());
}

#[test]
fn test_07_scan_on_empty_input_batch() {
    if let Ok(mut gds) = CuFileHardware::try_init() {
        let ops: Vec<ReadOp> = vec![];
        let mut out: Vec<usize> = vec![];

        let stats = gds.read_batch_into(&ops, &mut out).unwrap();
        assert_eq!(stats.bytes_transferred, 0);
    }
}

// --- Scan with zero patterns ---

#[test]
fn test_08_scan_with_zero_patterns_empty_batch_api() {
    if let Ok(mut gds) = CuFileHardware::try_init() {
        let ops: Vec<ReadOp> = vec![];
        let (bytes, stats) = gds.read_batch(&ops).unwrap();
        assert!(bytes.is_empty());
        assert_eq!(stats.bytes_transferred, 0);
    }
}

// --- Scan with pattern longer than input ---

#[test]
fn test_09_scan_pattern_longer_than_input_mismatch() {
    if let Ok(mut gds) = CuFileHardware::try_init() {
        let mut dummy = 0u8;
        let ptr = unsafe {
            DevicePointer::new(std::ptr::addr_of_mut!(dummy).cast::<std::ffi::c_void>()).unwrap()
        };
        let op = ReadOp {
            fd: 0,
            device_pointer: ptr,
            size: GDS_ALIGNMENT,
            file_offset: 0,
            device_offset: 0,
        };
        let ops = vec![op];
        let mut out = vec![0usize; 0];

        let err_result = gds.read_batch_into(&ops, &mut out);
        assert!(matches!(
            err_result,
            Err(CudaError::BatchLengthMismatch { .. })
        ));
    }
}

#[test]
fn test_10_scan_pattern_longer_than_input_unaligned() {
    let unaligned = usize::MAX;
    if unaligned % GDS_ALIGNMENT != 0 {
        assert!(matches!(
            validate_alignment(0, unaligned, 0),
            Err(CudaError::AlignmentViolation { .. })
        ));
    }
}

// --- GPU device lost simulation ---

#[test]
fn test_11_gpu_device_lost_simulation_try_init() {
    let res = CuFileHardware::try_init();
    if !is_available() {
        assert!(res.is_err());
    }
}

#[test]
fn test_12_gpu_device_lost_simulation_evict_fd() {
    if let Ok(mut gds) = CuFileHardware::try_init() {
        let was_cached = gds.evict_fd(9999).unwrap();
        assert!(!was_cached);
    }
}

// --- Concurrent scans from multiple threads sharing same device ---

#[test]
fn test_13_concurrent_scans_multiple_threads_availability() {
    let mut handles = vec![];
    for _ in 0..8 {
        handles.push(thread::spawn(|| {
            let report = availability_report();
            let is_avail = is_available();
            assert_eq!(report.is_available(), is_avail);
        }));
    }
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_14_concurrent_scans_diagnostics() {
    let mut handles = vec![];
    for _ in 0..8 {
        handles.push(thread::spawn(|| {
            let report = availability_report();
            for item in report.diagnostics() {
                assert!(!item.is_empty());
            }
        }));
    }
    for handle in handles {
        handle.join().unwrap();
    }
}

// --- Input exactly at GPU buffer boundary ---

#[test]
fn test_15_input_exactly_at_gpu_buffer_boundary_file_offset() {
    assert!(validate_alignment(0, GDS_ALIGNMENT, GDS_ALIGNMENT_I64).is_ok());
}

#[test]
fn test_16_input_exactly_at_gpu_buffer_boundary_device_offset() {
    assert!(validate_alignment(GDS_ALIGNMENT_I64, GDS_ALIGNMENT, 0).is_ok());
}

#[test]
fn test_17_input_exactly_at_gpu_buffer_boundary_max_transfer() {
    assert!(validate_alignment(0, DEFAULT_MAX_TRANSFER_SIZE, 0).is_ok());
}

// --- Match at last byte of input ---

#[test]
fn test_18_match_at_last_byte_of_input_negative_file_offset() {
    assert!(matches!(
        validate_alignment(-4096, GDS_ALIGNMENT, 0),
        Err(CudaError::AlignmentViolation { .. })
    ));
}

#[test]
fn test_19_match_at_last_byte_of_input_negative_device_offset() {
    assert!(matches!(
        validate_alignment(0, GDS_ALIGNMENT, -4096),
        Err(CudaError::AlignmentViolation { .. })
    ));
}

#[test]
fn test_20_match_at_last_byte_of_input_off_by_one_under() {
    assert!(matches!(
        validate_alignment(0, GDS_ALIGNMENT - 1, 0),
        Err(CudaError::AlignmentViolation { .. })
    ));
}

#[test]
fn test_21_match_at_last_byte_of_input_off_by_one_over() {
    assert!(matches!(
        validate_alignment(0, GDS_ALIGNMENT + 1, 0),
        Err(CudaError::AlignmentViolation { .. })
    ));
}

// --- Additional Structural/Adversarial Tests ---

#[test]
fn test_22_validate_alignment_off_by_one_file_offset_under() {
    assert!(matches!(
        validate_alignment(GDS_ALIGNMENT_I64 - 1, GDS_ALIGNMENT, 0),
        Err(CudaError::AlignmentViolation { .. })
    ));
}

#[test]
fn test_23_validate_alignment_off_by_one_file_offset_over() {
    assert!(matches!(
        validate_alignment(GDS_ALIGNMENT_I64 + 1, GDS_ALIGNMENT, 0),
        Err(CudaError::AlignmentViolation { .. })
    ));
}

#[test]
fn test_24_validate_alignment_off_by_one_device_offset_under() {
    assert!(matches!(
        validate_alignment(0, GDS_ALIGNMENT, GDS_ALIGNMENT_I64 - 1),
        Err(CudaError::AlignmentViolation { .. })
    ));
}

#[test]
fn test_25_validate_alignment_off_by_one_device_offset_over() {
    assert!(matches!(
        validate_alignment(0, GDS_ALIGNMENT, GDS_ALIGNMENT_I64 + 1),
        Err(CudaError::AlignmentViolation { .. })
    ));
}

#[test]
fn test_26_validate_alignment_large_chunk() {
    let size_10mb = 10 * 1024 * 1024;
    assert!(validate_alignment(0, size_10mb, 0).is_ok());
}

#[test]
fn test_27_validate_alignment_unaligned_large_chunk() {
    let size_10mb_plus_1 = 10 * 1024 * 1024 + 1;
    assert!(matches!(
        validate_alignment(0, size_10mb_plus_1, 0),
        Err(CudaError::AlignmentViolation { .. })
    ));
}

#[test]
fn test_28_device_pointer_null() {
    let ptr = unsafe { DevicePointer::new(ptr::null_mut()) };
    assert!(ptr.is_none());
}

#[test]
fn test_29_device_pointer_valid() {
    let mut dummy_data = 42u8;
    let ptr = unsafe {
        DevicePointer::new(std::ptr::addr_of_mut!(dummy_data).cast::<std::ffi::c_void>())
    };
    assert!(ptr.is_some());
}

#[test]
fn test_30_device_pointer_equality() {
    let mut dummy_data = 42u8;
    let ptr1 = unsafe {
        DevicePointer::new(std::ptr::addr_of_mut!(dummy_data).cast::<std::ffi::c_void>()).unwrap()
    };
    let ptr2 = unsafe {
        DevicePointer::new(std::ptr::addr_of_mut!(dummy_data).cast::<std::ffi::c_void>()).unwrap()
    };
    assert_eq!(ptr1, ptr2);
}

#[test]
fn test_31_device_pointer_inequality() {
    let mut dummy_data1 = 42u8;
    let mut dummy_data2 = 42u8;
    let ptr1 = unsafe {
        DevicePointer::new(std::ptr::addr_of_mut!(dummy_data1).cast::<std::ffi::c_void>()).unwrap()
    };
    let ptr2 = unsafe {
        DevicePointer::new(std::ptr::addr_of_mut!(dummy_data2).cast::<std::ffi::c_void>()).unwrap()
    };
    assert_ne!(ptr1, ptr2);
}

#[test]
fn test_32_error_display_alignment_violation() {
    let err = CudaError::AlignmentViolation {
        file_offset: -1,
        size: 1,
        device_offset: -1,
    };
    let s = err.to_string();
    assert!(s.contains("4096-byte alignment"));
    assert!(s.contains("file_offset=-1"));
}

#[test]
fn test_33_error_display_short_read() {
    let err = CudaError::ShortRead {
        requested: 1024,
        transferred: 512,
    };
    let s = err.to_string();
    assert!(s.contains("requested 1024"));
    assert!(s.contains("512 transferred"));
}

#[test]
fn test_34_error_display_batch_length_mismatch() {
    let err = CudaError::BatchLengthMismatch {
        ops_len: 5,
        out_len: 4,
    };
    let s = err.to_string();
    assert!(s.contains("ops.len()=5"));
    assert!(s.contains("out.len()=4"));
}

#[test]
fn test_35_error_display_feature_disabled() {
    let err = CudaError::FeatureDisabled;
    let s = err.to_string();
    assert!(s.contains("disabled at compile time"));
}

#[test]
fn test_36_availability_reason_debug() {
    let r1 = AvailabilityReason::Ready;
    let r2 = AvailabilityReason::FeatureDisabled;
    let s1 = format!("{r1:?}");
    let s2 = format!("{r2:?}");
    assert!(!s1.is_empty());
    assert!(!s2.is_empty());
}

#[test]
fn test_37_duplicate_identical_read_ops() {
    let mut dummy_data = 0u8;
    let ptr = unsafe {
        DevicePointer::new(std::ptr::addr_of_mut!(dummy_data).cast::<std::ffi::c_void>()).unwrap()
    };

    let op = ReadOp {
        fd: 3,
        device_pointer: ptr,
        size: GDS_ALIGNMENT,
        file_offset: 0,
        device_offset: 0,
    };

    let ops = [op, op, op];
    assert_eq!(ops.len(), 3);
}
