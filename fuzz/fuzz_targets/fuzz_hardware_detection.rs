#![no_main]

use cudagrep::{availability_report, validate_alignment};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut ints = [0_i64; 2];
    for (slot, chunk) in ints.iter_mut().zip(data.chunks(8)) {
        let mut bytes = [0_u8; 8];
        bytes[..chunk.len()].copy_from_slice(chunk);
        *slot = i64::from_le_bytes(bytes);
    }

    let mut size_bytes = [0_u8; 8];
    if data.len() > 16 {
        let end = (data.len() - 16).min(8);
        size_bytes[..end].copy_from_slice(&data[16..16 + end]);
    }
    let size = usize::from_le_bytes(size_bytes);

    let report = availability_report();
    let _ = report.is_available();
    let _ = report.reason();
    let _ = report.diagnostics();
    let _ = report.to_string();
    let _ = validate_alignment(ints[0], size, ints[1]);
});
