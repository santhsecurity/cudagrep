#![warn(missing_docs, clippy::pedantic)]
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::todo,
        clippy::unimplemented,
        clippy::panic
    )
)]
//! Binary entry point for `cudagrep`.
//!
//! The binary is intentionally thin and only reports runtime availability.

use std::process::ExitCode;

/// Runs the `cudagrep` availability probe.
fn main() -> ExitCode {
    tracing_subscriber::fmt::init();

    let report = cudagrep::availability_report();
    if report.is_available() {
        tracing::info!("GPUDirect Storage available: {report}");
        ExitCode::SUCCESS
    } else {
        tracing::warn!("GPUDirect Storage unavailable: {report}");
        ExitCode::FAILURE
    }
}
