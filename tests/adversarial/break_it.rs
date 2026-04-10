//! Adversarial tests designed to break `cudagrep` assumptions.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_wrap
)]
use cudagrep::{
    availability_report, is_available, validate_alignment, AvailabilityReason, CuFileHardware,
    CudaError, DevicePointer, ReadOp, GDS_ALIGNMENT,
};
use std::ptr;
use std::thread;

const GDS_ALIGNMENT_I64: i64 = 4096;

// --- 1. Empty input / zero-length slices ---

#[test]
fn test_01_validate_alignment_zero_size() {
    // 0 is a multiple of 4096. This is valid by definition.
    assert!(
        validate_alignment(0, 0, 0).is_ok(),
        "Zero size should be valid GDS alignment"
    );
}

#[test]
fn test_02_validate_alignment_zero_file_offset() {
    assert!(validate_alignment(0, GDS_ALIGNMENT, GDS_ALIGNMENT_I64).is_ok());
}

#[test]
fn test_03_validate_alignment_zero_device_offset() {
    assert!(validate_alignment(GDS_ALIGNMENT_I64, GDS_ALIGNMENT, 0).is_ok());
}

// --- 2. Null bytes in input ---
// Not applicable to numerical public API for alignment, but we can test
// availability report string formatting logic to ensure it doesn't crash on null strings
// in mocked dependencies, though the public API doesn't accept strings directly.
#[test]
fn test_04_availability_report_diagnostics_formatting() {
    let report = availability_report();
    let diag = report.diagnostics();
    // length is valid as long as we can map over it and check empty
    for item in diag {
        assert!(!item.is_empty(), "Diagnostic string should not be empty");
    }
}

// --- 3. Maximum u32/u64 values for any numeric parameter ---

#[test]
fn test_05_validate_alignment_max_i64_file_offset() {
    // i64::MAX is 9223372036854775807, which is NOT a multiple of 4096
    assert!(matches!(
        validate_alignment(i64::MAX, GDS_ALIGNMENT, 0),
        Err(CudaError::AlignmentViolation { .. })
    ));
}

#[test]
fn test_06_validate_alignment_max_i64_device_offset() {
    assert!(matches!(
        validate_alignment(0, GDS_ALIGNMENT, i64::MAX),
        Err(CudaError::AlignmentViolation { .. })
    ));
}

#[test]
fn test_07_validate_alignment_max_usize_size() {
    // usize::MAX is not a multiple of 4096 usually.
    let max_aligned = usize::MAX / GDS_ALIGNMENT * GDS_ALIGNMENT;
    assert!(validate_alignment(0, max_aligned, 0).is_ok());

    // usize::MAX itself should fail
    let unaligned = usize::MAX;
    if unaligned % GDS_ALIGNMENT != 0 {
        assert!(matches!(
            validate_alignment(0, unaligned, 0),
            Err(CudaError::AlignmentViolation { .. })
        ));
    }
}

#[test]
fn test_08_validate_alignment_negative_file_offset() {
    // -4096 is a multiple of 4096 in absolute terms but GDS offset must be >= 0
    assert!(matches!(
        validate_alignment(-4096, GDS_ALIGNMENT, 0),
        Err(CudaError::AlignmentViolation { .. })
    ));
}

#[test]
fn test_09_validate_alignment_negative_device_offset() {
    assert!(matches!(
        validate_alignment(0, GDS_ALIGNMENT, -4096),
        Err(CudaError::AlignmentViolation { .. })
    ));
}

// --- 4. 1MB+ input ---

#[test]
fn test_10_validate_alignment_large_chunk() {
    let size_10mb = 10 * 1024 * 1024;
    assert!(
        validate_alignment(0, size_10mb, 0).is_ok(),
        "10MB aligned size should pass validation"
    );
}

#[test]
fn test_11_validate_alignment_unaligned_large_chunk() {
    let size_10mb_plus_1 = 10 * 1024 * 1024 + 1;
    assert!(matches!(
        validate_alignment(0, size_10mb_plus_1, 0),
        Err(CudaError::AlignmentViolation { .. })
    ));
}

// --- 5. Concurrent access from 8 threads ---

#[test]
fn test_12_concurrent_availability_checks() {
    let mut handles = vec![];
    for _ in 0..8 {
        handles.push(thread::spawn(|| {
            let report = availability_report();
            let is_avail = is_available();
            assert_eq!(
                report.is_available(),
                is_avail,
                "Availability state mismatch in thread"
            );
        }));
    }
    for handle in handles {
        handle.join().unwrap_or_else(|_| panic!("Thread panicked"));
    }
}

// --- 6. Malformed/truncated input ---

#[test]
fn test_13_device_pointer_null() {
    // DevicePointer::new(null) should return None
    let ptr = unsafe { DevicePointer::new(ptr::null_mut()) };
    assert!(ptr.is_none(), "DevicePointer should reject null pointers");
}

#[test]
fn test_14_device_pointer_valid() {
    let mut dummy_data = 42u8;
    let ptr = unsafe {
        DevicePointer::new(std::ptr::addr_of_mut!(dummy_data).cast::<std::ffi::c_void>())
    };
    assert!(
        ptr.is_some(),
        "DevicePointer should accept valid non-null pointers"
    );
}

// --- 7. Unicode edge cases (BOM, overlong sequences, surrogates) ---
// Note: error messages could theoretically be broken by weird values if they included paths.
// Since error struct fields are numerical, we test error string rendering.

#[test]
fn test_15_error_display_alignment_violation() {
    let err = CudaError::AlignmentViolation {
        file_offset: -1,
        size: 1,
        device_offset: -1,
    };
    let s = err.to_string();
    assert!(
        s.contains("4096-byte alignment"),
        "Error string should mention 4096-byte alignment"
    );
    assert!(
        s.contains("file_offset=-1"),
        "Error string should contain file_offset"
    );
}

#[test]
fn test_16_error_display_short_read() {
    let err = CudaError::ShortRead {
        requested: 1024,
        transferred: 512,
    };
    let s = err.to_string();
    assert!(
        s.contains("requested 1024"),
        "Error string should contain requested bytes"
    );
    assert!(
        s.contains("512 transferred"),
        "Error string should contain transferred bytes"
    );
}

#[test]
fn test_17_error_display_batch_length_mismatch() {
    let err = CudaError::BatchLengthMismatch {
        ops_len: 5,
        out_len: 4,
    };
    let s = err.to_string();
    assert!(
        s.contains("ops.len()=5"),
        "Error string should contain ops_len"
    );
    assert!(
        s.contains("out.len()=4"),
        "Error string should contain out_len"
    );
}

#[test]
fn test_18_error_display_feature_disabled() {
    let err = CudaError::FeatureDisabled;
    let s = err.to_string();
    assert!(
        s.contains("disabled at compile time"),
        "Error string should explain disabled state"
    );
}

// --- 8. Duplicate entries ---

#[test]
fn test_19_duplicate_identical_read_ops() {
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

    // Testing constructability and duplication. Batch submission logic is internal and hardware bound,
    // so we just test the struct definition here.
    let ops = [op, op, op];
    assert_eq!(ops.len(), 3);
}

// --- 9. Off-by-one: first byte, last byte, boundary between chunks ---

#[test]
fn test_20_validate_alignment_off_by_one_size_under() {
    assert!(matches!(
        validate_alignment(0, GDS_ALIGNMENT - 1, 0),
        Err(CudaError::AlignmentViolation { .. })
    ));
}

#[test]
fn test_21_validate_alignment_off_by_one_size_over() {
    assert!(matches!(
        validate_alignment(0, GDS_ALIGNMENT + 1, 0),
        Err(CudaError::AlignmentViolation { .. })
    ));
}

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

// --- 10. Resource exhaustion ---

#[test]
fn test_26_try_init_hardware() {
    // try_init should either return Ok (if hardware is mocked/available)
    // or return a clear error (e.g. FeatureDisabled or Unavailable), not panic.
    let res = CuFileHardware::try_init();
    if !is_available() {
        assert!(
            res.is_err(),
            "Hardware init should fail if is_available() is false"
        );
    }
}

#[test]
fn test_27_availability_reason_debug() {
    // Ensure all AvailabilityReason variants can be formatted
    let r1 = AvailabilityReason::Ready;
    let r2 = AvailabilityReason::FeatureDisabled;

    let s1 = format!("{r1:?}");
    let s2 = format!("{r2:?}");

    assert!(!s1.is_empty());
    assert!(!s2.is_empty());
}

#[test]
fn test_28_read_batch_into_mismatch() {
    // If hardware initializes, test batch mismatch without reading
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
        let mut out = vec![0usize; 0]; // 0 != 1

        let err_result = gds.read_batch_into(&ops, &mut out);
        assert!(err_result.is_err());
        let Err(err) = err_result else {
            panic!("expected Err")
        };
        assert!(matches!(err, CudaError::BatchLengthMismatch { .. }));
    }
}

#[test]
fn test_29_validate_alignment_near_overflow() {
    // Test i64 offset that when added to size could overflow if internally cast incorrectly,
    // though validate_alignment only checks modulo.
    let max_aligned = (i64::MAX / GDS_ALIGNMENT_I64) * GDS_ALIGNMENT_I64;
    assert!(validate_alignment(max_aligned, GDS_ALIGNMENT, 0).is_ok());
}

#[test]
fn test_30_validate_alignment_large_device_offset() {
    let max_aligned = (i64::MAX / GDS_ALIGNMENT_I64) * GDS_ALIGNMENT_I64;
    assert!(validate_alignment(0, GDS_ALIGNMENT, max_aligned).is_ok());
}

#[test]
fn test_31_device_pointer_equality() {
    let mut dummy_data = 42u8;
    let ptr1 = unsafe {
        DevicePointer::new(std::ptr::addr_of_mut!(dummy_data).cast::<std::ffi::c_void>()).unwrap()
    };
    let ptr2 = unsafe {
        DevicePointer::new(std::ptr::addr_of_mut!(dummy_data).cast::<std::ffi::c_void>()).unwrap()
    };
    assert_eq!(ptr1, ptr2, "Pointers to the same location should be equal");
}

#[test]
fn test_32_device_pointer_inequality() {
    let mut dummy_data1 = 42u8;
    let mut dummy_data2 = 42u8;
    let ptr1 = unsafe {
        DevicePointer::new(std::ptr::addr_of_mut!(dummy_data1).cast::<std::ffi::c_void>()).unwrap()
    };
    let ptr2 = unsafe {
        DevicePointer::new(std::ptr::addr_of_mut!(dummy_data2).cast::<std::ffi::c_void>()).unwrap()
    };
    assert_ne!(
        ptr1, ptr2,
        "Pointers to different locations should not be equal"
    );
}

#[test]
fn test_33_read_batch_into_empty() {
    // If hardware initializes, test batch with empty slices
    if let Ok(mut gds) = CuFileHardware::try_init() {
        let ops: Vec<ReadOp> = vec![];
        let mut out: Vec<usize> = vec![];

        let stats = gds.read_batch_into(&ops, &mut out).unwrap();
        assert_eq!(stats.bytes_transferred, 0);
    }
}
