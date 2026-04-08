# Contributing

## Development standards

- keep non-test code free of `unwrap()` and `todo!()`
- keep all `unsafe` code in `src/cufile.rs`
- add a `// SAFETY:` comment to every `unsafe` block
- document every public item with `///`
- document every Rust source file with `//!`

## Validation

Run these commands before submitting changes:

```bash
cargo fmt --all --check
cargo clippy --all-targets --no-default-features -- -D warnings
cargo test --no-default-features
```

If you touch CUDA-specific code, also run:

```bash
cargo test --no-run --features cuda
cargo clippy --all-targets --features cuda -- -D warnings
```

## Pull requests

- keep changes focused
- include tests for behavior changes
- update `CHANGELOG.md` when the public crate behavior changes
