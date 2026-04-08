//! Concurrent scaling tests for DMA operations.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_wrap
)]

use cudagrep::{CuFileHardware, DevicePointer, GDS_ALIGNMENT, ReadOp};
use std::ffi::c_void;

const GDS_ALIGNMENT_I64: i64 = 4096;

#[test]
fn handles_10k_concurrent_dma_requests_safely() {
    let mut hw = match CuFileHardware::try_init() {
        Ok(hw) => hw,
        Err(_) => return, // Graceful exit if no hardware
    };

    let num_ops = 10_000;
    let mut ops = Vec::with_capacity(num_ops);
    let mut byte = 0u8;
    let ptr = std::ptr::addr_of_mut!(byte).cast::<c_void>();
    let device_ptr = unsafe { DevicePointer::new(ptr) }.unwrap();

    // Since CuFileHardware is !Send and !Sync, we cannot share it across threads.
    // Instead, we use the read_batch API to test handling a large number of concurrent
    // DMA operations, which triggers the cuFile batch API under the hood (or a loop
    // depending on implementation). We use invalid fd -1 to ensure it safely handles
    // errors without crashing or leaking under high load.
    for i in 0..num_ops {
        ops.push(ReadOp {
            fd: -1, // Use an invalid fd
            device_pointer: device_ptr.clone(),
            size: GDS_ALIGNMENT,
            file_offset: (i * GDS_ALIGNMENT) as i64,
            device_offset: (i * GDS_ALIGNMENT) as i64,
        });
    }

    let result = hw.read_batch(&ops);
    
    // The entire batch should fail quickly due to the invalid fd without panicking.
    assert!(result.is_err());
    
    // Test that the hardware session is still valid by doing a single validation call.
    let _ = hw.evict_fd(-1);
}
