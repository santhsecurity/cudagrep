use std::collections::HashMap;
use std::os::fd::RawFd;

use crate::alignment::GDS_ALIGNMENT;
use crate::cufile::{CUfileHandle, CuFileLibrary};
use crate::error::{map_read_error, CudaError};

pub(crate) const FD_FAST_PATH_SIZE: usize = 1024;

#[cfg(feature = "cuda")]
pub(crate) const DEFAULT_MAX_REGISTERED_MEMORY_BYTES: usize = 128 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub(crate) struct CachedHandle {
    pub(crate) handle: CUfileHandle,
    pub(crate) registered_bytes: usize,
    pub(crate) last_used: u64,
    pub(crate) inode: u64,
    pub(crate) device: u64,
}

pub(crate) struct RegistrationCache {
    pub(crate) fd_array: [Option<CachedHandle>; FD_FAST_PATH_SIZE],
    pub(crate) fd_map: HashMap<RawFd, CachedHandle>,
    pub(crate) total_registered_memory: usize,
    pub(crate) access_clock: u64,
    pub(crate) max_registered_memory: usize,
}

impl RegistrationCache {
    #[allow(dead_code, clippy::large_stack_arrays)]
    pub(crate) fn new(max_registered_memory: usize) -> Self {
        Self {
            fd_array: [None; FD_FAST_PATH_SIZE],
            fd_map: HashMap::new(),
            total_registered_memory: 0,
            access_clock: 0,
            max_registered_memory,
        }
    }

    pub(crate) fn evict_fd(
        &mut self,
        fd: RawFd,
        library: &CuFileLibrary,
    ) -> Result<bool, CudaError> {
        let Some(entry) = self.remove_cached_handle(fd) else {
            return Ok(false);
        };

        self.total_registered_memory = self
            .total_registered_memory
            .saturating_sub(entry.registered_bytes);
        library
            .deregister_raw(fd, entry.handle)
            .map_err(map_read_error)?;
        Ok(true)
    }

    pub(crate) fn get_or_register_for_bytes(
        &mut self,
        fd: RawFd,
        requested_bytes: usize,
        library: &CuFileLibrary,
    ) -> Result<CUfileHandle, CudaError> {
        let (inode, device) = crate::hardware::get_file_identity(fd).map_err(|e| {
            let os_code = e.raw_os_error().unwrap_or_else(|| {
                tracing::error!(error = %e, fd, "fstat failed with no OS error code");
                -1
            });
            CudaError::DescriptorRegistrationFailed {
                fd,
                code: crate::cufile::CuFileDriverError::OsError(os_code),
            }
        })?;

        if let Some(entry) = self.cached_handle(fd) {
            if entry.inode == inode && entry.device == device {
                self.touch_cached_handle(fd, requested_bytes, library)?;
                return Ok(entry.handle);
            }
            self.evict_fd(fd, library)?;
        }

        let registered_bytes = registered_bytes_for_request(requested_bytes);
        self.evict_lru_until_within_budget(registered_bytes, Some(fd), library)?;

        let handle = library.register_raw(fd).map_err(map_read_error)?;
        let entry = CachedHandle {
            handle,
            registered_bytes,
            last_used: self.next_access_stamp(),
            inode,
            device,
        };
        self.insert_cached_handle(fd, entry);
        self.total_registered_memory = self
            .total_registered_memory
            .saturating_add(registered_bytes);
        Ok(handle)
    }

    fn touch_cached_handle(
        &mut self,
        fd: RawFd,
        requested_bytes: usize,
        library: &CuFileLibrary,
    ) -> Result<(), CudaError> {
        let Some(current) = self.cached_handle(fd) else {
            return Ok(());
        };
        let required_bytes = registered_bytes_for_request(requested_bytes);
        let additional_bytes = required_bytes.saturating_sub(current.registered_bytes);

        if additional_bytes > 0 {
            self.evict_lru_until_within_budget(additional_bytes, Some(fd), library)?;
        }

        let stamp = self.next_access_stamp();
        if let Some(entry) = self.cached_handle_mut(fd) {
            entry.last_used = stamp;

            if required_bytes > entry.registered_bytes {
                let delta = required_bytes - entry.registered_bytes;
                entry.registered_bytes = required_bytes;
                self.total_registered_memory = self.total_registered_memory.saturating_add(delta);
            }
        }

        Ok(())
    }

    fn evict_lru_until_within_budget(
        &mut self,
        additional_bytes: usize,
        exclude_fd: Option<RawFd>,
        library: &CuFileLibrary,
    ) -> Result<(), CudaError> {
        while self
            .total_registered_memory
            .saturating_add(additional_bytes)
            > self.max_registered_memory
        {
            let Some(lru_fd) = self.find_lru_fd(exclude_fd) else {
                break;
            };

            let removed = self.evict_fd(lru_fd, library)?;
            if !removed {
                break;
            }
        }

        Ok(())
    }

    fn find_lru_fd(&self, exclude_fd: Option<RawFd>) -> Option<RawFd> {
        let mut lru: Option<(RawFd, u64)> = None;

        for (index, entry) in self.fd_array.iter().enumerate() {
            let Some(entry) = entry else {
                continue;
            };

            let Ok(fd) = i32::try_from(index) else {
                continue;
            };
            if Some(fd) == exclude_fd {
                continue;
            }

            match lru {
                Some((_, last_used)) if entry.last_used >= last_used => {}
                _ => lru = Some((fd, entry.last_used)),
            }
        }

        for (&fd, entry) in &self.fd_map {
            if Some(fd) == exclude_fd {
                continue;
            }

            match lru {
                Some((_, last_used)) if entry.last_used >= last_used => {}
                _ => lru = Some((fd, entry.last_used)),
            }
        }

        lru.map(|(fd, _)| fd)
    }

    fn next_access_stamp(&mut self) -> u64 {
        self.access_clock = self.access_clock.saturating_add(1);
        self.access_clock
    }

    pub(crate) fn cached_handle(&self, fd: RawFd) -> Option<CachedHandle> {
        if let Some(index) = fd_array_index(fd) {
            return self.fd_array[index];
        }
        self.fd_map.get(&fd).copied()
    }

    fn cached_handle_mut(&mut self, fd: RawFd) -> Option<&mut CachedHandle> {
        if let Some(index) = fd_array_index(fd) {
            return self.fd_array[index].as_mut();
        }
        self.fd_map.get_mut(&fd)
    }

    fn insert_cached_handle(&mut self, fd: RawFd, entry: CachedHandle) {
        if let Some(index) = fd_array_index(fd) {
            self.fd_array[index] = Some(entry);
        } else {
            self.fd_map.insert(fd, entry);
        }
    }

    fn remove_cached_handle(&mut self, fd: RawFd) -> Option<CachedHandle> {
        if let Some(index) = fd_array_index(fd) {
            return self.fd_array[index].take();
        }
        self.fd_map.remove(&fd)
    }

    pub(crate) fn drain_cached_handles(&mut self) -> Vec<(RawFd, CachedHandle)> {
        let mut drained = Vec::with_capacity(FD_FAST_PATH_SIZE + self.fd_map.len());

        for (index, entry) in self.fd_array.iter_mut().enumerate() {
            let Some(entry) = entry.take() else {
                continue;
            };
            if let Ok(fd) = i32::try_from(index) {
                drained.push((fd, entry));
            }
        }

        drained.extend(self.fd_map.drain());
        self.total_registered_memory = 0;
        drained
    }
}

pub(crate) fn registered_bytes_for_request(requested_bytes: usize) -> usize {
    if requested_bytes == 0 {
        return 0;
    }

    let alignment = GDS_ALIGNMENT;
    requested_bytes.saturating_add(alignment - 1) / alignment * alignment
}

pub(crate) fn fd_array_index(fd: RawFd) -> Option<usize> {
    usize::try_from(fd)
        .ok()
        .filter(|index| *index < FD_FAST_PATH_SIZE)
}
