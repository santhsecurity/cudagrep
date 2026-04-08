//! High-level hardware session management for GDS.

/// Registration cache logic.
pub mod cache;
/// High-level device API.
pub mod device;
/// DMA transfer logic.
pub mod dma;

pub use device::CuFileHardware;
use std::os::fd::RawFd;

/// Default maximum single transfer size in bytes (16 MiB).
///
/// `cuFileRead` implementations have a driver-specific ceiling on a single
/// transfer. Larger requests are automatically chunked.
pub const DEFAULT_MAX_TRANSFER_SIZE: usize = 16 * 1024 * 1024;

#[cfg(not(test))]
#[allow(clippy::unnecessary_cast)]
pub(crate) fn get_file_identity(fd: RawFd) -> Result<(u64, u64), std::io::Error> {
    let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
    if unsafe { libc::fstat(fd, stat.as_mut_ptr()) } < 0 {
        return Err(std::io::Error::last_os_error());
    }
    let stat = unsafe { stat.assume_init() };
    Ok((stat.st_ino as u64, stat.st_dev as u64))
}

#[cfg(test)]
#[allow(clippy::unnecessary_wraps)]
pub(crate) fn get_file_identity(fd: RawFd) -> Result<(u64, u64), std::io::Error> {
    #[allow(clippy::cast_sign_loss)]
    let inode = fd as u64;
    Ok((inode, 1))
}
