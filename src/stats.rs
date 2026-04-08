//! Performance measurement for GDS transfers.

/// Performance counters for a completed read or batch.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct ReadStats {
    /// Total bytes successfully transferred to device memory.
    pub bytes_transferred: usize,
    /// Wall-clock duration of the transfer.
    pub wall_time: std::time::Duration,
}

impl ReadStats {
    /// Throughput in gigabytes per second (GiB/s).
    ///
    /// Returns 0.0 if the transfer took zero time (to avoid division by zero).
    /// Uses binary gigabytes (1 GiB = 2^30 bytes = 1,073,741,824 bytes).
    ///
    /// # Examples
    ///
    /// ```
    /// use cudagrep::ReadStats;
    /// use std::time::Duration;
    ///
    /// let stats = ReadStats {
    ///     bytes_transferred: 1_073_741_824, // 1 GiB
    ///     wall_time: Duration::from_secs(1),
    /// };
    /// assert!((stats.throughput_gbps() - 1.0).abs() < 0.001);
    /// ```
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn throughput_gbps(&self) -> f64 {
        let secs = self.wall_time.as_secs_f64();
        if secs <= 0.0 {
            return 0.0;
        }
        (self.bytes_transferred as f64) / secs / 1_073_741_824.0
    }
}
