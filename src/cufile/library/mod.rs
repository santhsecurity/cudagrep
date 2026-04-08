//! Shared library loading and `libcufile` session management.

/// Batch operations.
pub mod batch;
/// Initialization operations.
pub mod init;
/// Read operations.
pub mod read;

use std::marker::PhantomData;

use libloading::Library;

use crate::cufile::ffi::{
    CuFileBatchIOCancelFn, CuFileBatchIOSetUpFn, CuFileBatchIOSubmitFn, CuFileBufDeregisterFn,
    CuFileBufRegisterFn, CuFileDriverCloseFn, CuFileDriverOpenFn, CuFileDriverSetMaxCacheSizeFn,
    CuFileDriverSetMaxPinnedMemSizeFn, CuFileGetVersionFn, CuFileHandleDeregisterFn,
    CuFileHandleRegisterFn, CuFileReadFn,
};

pub(crate) const SUPPORTED_CUFILE_API_MAJOR_VERSION: i32 = 1;

/// Loaded `libcufile` entry points.
///
/// This type deliberately is `!Send` and `!Sync` because NVIDIA's
/// cuFile driver maintains global state and is not documented as safe
/// for concurrent use from multiple threads.
pub struct CuFileLibrary {
    pub(crate) _library: Option<Library>,
    pub(crate) api: Api,
    pub(crate) _not_send_sync: PhantomData<*mut ()>,
}

#[derive(Clone, Copy)]
pub(crate) struct Api {
    pub(crate) driver_open: CuFileDriverOpenFn,
    pub(crate) driver_close: CuFileDriverCloseFn,
    pub(crate) get_version: Option<CuFileGetVersionFn>,
    pub(crate) handle_register: CuFileHandleRegisterFn,
    pub(crate) handle_deregister: CuFileHandleDeregisterFn,
    pub(crate) read: CuFileReadFn,
    pub(crate) buf_register: CuFileBufRegisterFn,
    pub(crate) buf_deregister: CuFileBufDeregisterFn,
    pub(crate) batch_io_setup: CuFileBatchIOSetUpFn,
    pub(crate) batch_io_submit: CuFileBatchIOSubmitFn,
    pub(crate) driver_set_max_pinned_mem_size: CuFileDriverSetMaxPinnedMemSizeFn,
    pub(crate) driver_set_max_cache_size: CuFileDriverSetMaxCacheSizeFn,
    pub(crate) batch_io_cancel: CuFileBatchIOCancelFn,
}

impl CuFileLibrary {
    #[cfg(test)]
    pub(crate) fn from_api_for_tests(api: Api) -> Self {
        Self {
            _library: None,
            api,
            _not_send_sync: PhantomData,
        }
    }
}
