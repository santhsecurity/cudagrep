# cudagrep

[![crates.io](https://img.shields.io/crates/v/cudagrep.svg)](https://crates.io/crates/cudagrep)
[![docs.rs](https://docs.rs/cudagrep/badge.svg)](https://docs.rs/cudagrep)
[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Safe Rust bindings for NVIDIA **GPUDirect Storage** (`libcufile`) — zero-copy NVMe-to-GPU transfers without kernel bouncing.

## Why `cudagrep`?

- **Graceful degradation**: Compiles and runs on machines without CUDA. No panics, no stubbed APIs.
- **Airtight safety**: All `unsafe` code is confined to [`cufile`](src/cufile). The public API is 100% safe Rust.
- **Zero-overhead caching**: File descriptor registrations are cached, avoiding ~50 µs driver round-trips on every read.
- **Crates.io ready**: `no_std`-friendly error types, exhaustive docs, and adversarial tests.

## Hardware Requirements

To use GPUDirect Storage with actual hardware:

- **NVIDIA GPU**: Compute Capability 6.0+ (Pascal or newer)
- **NVMe SSD**: Direct-attached NVMe storage (not SATA, not RAID controller)
- **Linux Kernel**: 5.3+ with `CONFIG_NVME_FABRICS` support
- **NVIDIA Driver**: R470+ with GPUDirect Storage support
- **cuFile Library**: Part of NVIDIA CUDA Toolkit 11.4+ or standalone cuFile SDK

### Filesystem Requirements

- Files must be on a filesystem that supports `O_DIRECT` (ext4, XFS with `nobarrier`, etc.)
- Files must not be compressed or encrypted at the filesystem level

## Quick Start

### Add to `Cargo.toml`

```toml
[dependencies]
cudagrep = "0.1"

# Or with runtime CUDA probing:
cudagrep = { version = "0.1", features = ["cuda"] }
```

### Check Availability

```rust
use cudagrep::{is_available, availability_report};

if is_available() {
    println!("GDS is available!");
} else {
    let report = availability_report();
    println!("GDS unavailable: {}", report);
    for diag in report.diagnostics() {
        println!("  - {}", diag);
    }
}
```

### Read to GPU Memory

```rust
use cudagrep::{CuFileHardware, GDS_ALIGNMENT};
use cudagrep::cufile::DevicePointer;
use std::os::fd::RawFd;

fn read_file_to_gpu(
    fd: RawFd,
    gpu_ptr: *mut std::ffi::c_void,
) -> Result<(), cudagrep::CudaError> {
    let mut gds = CuFileHardware::try_init()?;
    let device_pointer = unsafe { DevicePointer::new(gpu_ptr).unwrap() };

    let bytes = gds.read_to_device(fd, device_pointer, GDS_ALIGNMENT, 0, 0)?;
    println!("Transferred {} bytes", bytes);

    // All registrations and the driver session are cleaned up on drop
    Ok(())
}
```

### Batch Operations

```rust
use cudagrep::{CuFileHardware, ReadOp, GDS_ALIGNMENT};
use cudagrep::cufile::DevicePointer;

fn read_batch(
    gds: &mut CuFileHardware,
    fd: std::os::fd::RawFd,
    ptr: DevicePointer,
) -> Result<(), cudagrep::CudaError> {
    let ops = vec![
        ReadOp {
            fd,
            device_pointer: ptr,
            size: GDS_ALIGNMENT,
            file_offset: 0,
            device_offset: 0,
        },
        ReadOp {
            fd,
            device_pointer: ptr,
            size: GDS_ALIGNMENT,
            file_offset: 4096,
            device_offset: 4096,
        },
    ];

    let (bytes_per_op, stats) = gds.read_batch(&ops)?;
    println!("Per-op bytes: {:?}", bytes_per_op);
    println!("Throughput: {:.2} GiB/s", stats.throughput_gbps());

    Ok(())
}
```

### Pre-allocate Result Buffer

```rust
use cudagrep::{CuFileHardware, ReadOp};

fn read_batch_no_alloc(
    gds: &mut CuFileHardware,
    ops: &[ReadOp],
) -> Result<(), cudagrep::CudaError> {
    let mut results = vec![0; ops.len()];
    let stats = gds.read_batch_into(ops, &mut results)?;
    println!("Results: {:?}", results);
    println!("Throughput: {:.2} GiB/s", stats.throughput_gbps());
    Ok(())
}
```

### Registration Cache Management

```rust
use cudagrep::CuFileHardware;

fn manage_fd(gds: &mut CuFileHardware) -> Result<(), cudagrep::CudaError> {
    // gds.read_to_device(3, ...) would register fd=3 automatically

    // Before closing the file, evict from cache to prevent stale handle reuse
    let was_cached = gds.evict_fd(3)?;
    if was_cached {
        println!("Fd 3 deregistered from GDS cache");
    }
    Ok(())
}
```

### Alignment Validation

GPUDirect Storage requires **4 KiB alignment** for all transfer parameters. Use `validate_alignment` to pre-check:

```rust
use cudagrep::{validate_alignment, GDS_ALIGNMENT};

fn validate_before_transfer(
    file_offset: i64,
    size: usize,
    device_offset: i64,
) -> Result<(), cudagrep::CudaError> {
    validate_alignment(file_offset, size, device_offset)?;
    println!("All parameters are properly aligned");
    Ok(())
}
```

## Safety Model

All `unsafe` code is confined to [`src/cufile/`](src/cufile/).

- **FFI wrapping**: Dynamic loading and raw `libcufile` calls are hidden behind safe Rust methods.
- **RAII cleanup**: Registered `cuFile` handles are deregistered automatically on drop. Errors during cleanup are logged, not swallowed.
- **No leaks on error paths**: If `cuFileRead` fails, the temporary registration is still deregistered before the error is returned.
- **Documented invariants**: Every `unsafe` block has a `// SAFETY:` comment explaining why the preconditions hold.

## Key Concepts

### GDS Alignment (4 KiB)

GPUDirect Storage requires 4 KiB (4096-byte) alignment for all transfer parameters because NVMe DMA and GPU BAR memory mappings operate at page granularity:

- File offsets must be 4 KiB aligned
- Device offsets must be 4 KiB aligned
- Transfer sizes must be multiples of 4 KiB

Use [`validate_alignment`](https://docs.rs/cudagrep/latest/cudagrep/fn.validate_alignment.html) to pre-check parameters.

### Registration Cache Lifecycle

[`CuFileHardware`](https://docs.rs/cudagrep/latest/cudagrep/struct.CuFileHardware.html) maintains a cache of registered file descriptors:

1. **Cache on first use**: File descriptors are registered with the driver on first read and cached for subsequent operations.
2. **Eviction**: Call [`evict_fd`](https://docs.rs/cudagrep/latest/cudagrep/struct.CuFileHardware.html#method.evict_fd) before closing a file descriptor to prevent stale handle reuse.
3. **Cleanup on Drop**: When `CuFileHardware` is dropped, all cached registrations are deregistered. Individual failures are logged but do not prevent cleanup of remaining handles.

### ShortRead Error

The [`CudaError::ShortRead`](https://docs.rs/cudagrep/latest/cudagrep/enum.CudaError.html#variant.ShortRead) error indicates that the driver returned 0 bytes for a non-zero request, typically meaning EOF or a hardware fault. Without this check, the transfer loop would spin forever. The error includes both `requested` and `transferred` counts.

## Features

- `cuda` — Enables runtime probing and use of `libcufile.so.0`.

## Development

### Testing

Default CI validates the non-CUDA build:

```bash
cargo fmt --all --check
cargo clippy --all-targets --no-default-features -- -D warnings
cargo test --no-default-features
```

Run the legendary adversarial test suite:

```bash
cargo test --test legendary --no-default-features
```

Validate the feature-gated CUDA path without a GPU:

```bash
cargo test --no-run --features cuda
cargo clippy --all-targets --features cuda -- -D warnings
```

### Benchmarking

```bash
cargo bench --no-default-features
```

### Documentation

```bash
cargo doc --no-deps --no-default-features
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
