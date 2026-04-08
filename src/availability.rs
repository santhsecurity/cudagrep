//! Runtime availability probing for `GPUDirect` Storage.

use std::fmt;

use crate::cufile;

/// Detailed runtime availability diagnostics for `GPUDirect` Storage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AvailabilityReport {
    available: bool,
    reason: AvailabilityReason,
    diagnostics: Vec<String>,
}

impl AvailabilityReport {
    /// Creates a new availability report.
    ///
    /// # Examples
    ///
    /// ```
    /// use cudagrep::{AvailabilityReport, AvailabilityReason};
    ///
    /// let report = AvailabilityReport::new(
    ///     false,
    ///     AvailabilityReason::FeatureDisabled,
    ///     vec!["compiled without cuda feature".to_owned()],
    /// );
    ///
    /// assert!(!report.is_available());
    /// ```
    #[must_use]
    pub fn new(available: bool, reason: AvailabilityReason, diagnostics: Vec<String>) -> Self {
        Self {
            available,
            reason,
            diagnostics,
        }
    }

    /// Returns whether GDS can be used by this crate in the current process.
    ///
    /// # Examples
    ///
    /// ```
    /// use cudagrep::AvailabilityReport;
    ///
    /// let report = AvailabilityReport::new(
    ///     true,
    ///     cudagrep::AvailabilityReason::Ready,
    ///     vec!["ready".to_owned()],
    /// );
    ///
    /// assert!(report.is_available());
    /// ```
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.available
    }

    /// Returns the primary reason explaining the current availability state.
    ///
    /// # Examples
    ///
    /// ```
    /// use cudagrep::{AvailabilityReport, AvailabilityReason};
    ///
    /// let report = AvailabilityReport::new(
    ///     false,
    ///     AvailabilityReason::FeatureDisabled,
    ///     vec!["test".to_owned()],
    /// );
    ///
    /// assert_eq!(report.reason(), &AvailabilityReason::FeatureDisabled);
    /// ```
    #[must_use]
    pub fn reason(&self) -> &AvailabilityReason {
        &self.reason
    }

    /// Returns human-readable diagnostics collected during probing.
    ///
    /// # Examples
    ///
    /// ```
    /// use cudagrep::{AvailabilityReport, AvailabilityReason};
    ///
    /// let report = AvailabilityReport::new(
    ///     false,
    ///     AvailabilityReason::FeatureDisabled,
    ///     vec!["Rebuild with --features cuda".to_owned()],
    /// );
    ///
    /// assert_eq!(report.diagnostics(), &["Rebuild with --features cuda"]);
    /// ```
    #[must_use]
    pub fn diagnostics(&self) -> &[String] {
        &self.diagnostics
    }
}

impl fmt::Display for AvailabilityReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.reason)?;
        if !self.diagnostics.is_empty() {
            write!(f, " [{}]", self.diagnostics.join("; "))?;
        }

        Ok(())
    }
}

/// High-level explanation for why `GPUDirect` Storage is or is not available.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AvailabilityReason {
    /// The crate was compiled without GPU support.
    FeatureDisabled,
    /// `libcufile.so.0` could not be loaded.
    LibraryUnavailable(String),
    /// A required symbol was missing from `libcufile`.
    SymbolUnavailable(String),
    /// The NVIDIA driver rejected `cuFileDriverOpen`.
    DriverRejected(cufile::CuFileDriverError),
    /// The NVIDIA driver failed `cuFileDriverClose` after opening successfully.
    DriverCleanupFailed(cufile::CuFileDriverError),
    /// All required checks succeeded.
    Ready,
}

impl fmt::Display for AvailabilityReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FeatureDisabled => f.write_str("compiled without the `cuda` feature"),
            Self::LibraryUnavailable(message) => {
                write!(f, "failed to load libcufile.so.0: {message}")
            }
            Self::SymbolUnavailable(symbol) => {
                write!(f, "missing required libcufile symbol `{symbol}`")
            }
            Self::DriverRejected(code) => {
                write!(
                    f,
                    "cuFileDriverOpen rejected initialization with error code {code}"
                )
            }
            Self::DriverCleanupFailed(code) => {
                write!(
                    f,
                    "cuFileDriverClose failed during cleanup with error code {code}"
                )
            }
            Self::Ready => f.write_str("GPUDirect Storage is available"),
        }
    }
}

/// Returns whether `GPUDirect` Storage is available for this build and machine.
///
/// # Examples
///
/// ```
/// let available = cudagrep::is_available();
/// // Result depends on whether the crate was compiled with the cuda feature
/// // and whether libcufile.so.0 is present on the host.
/// ```
#[must_use]
pub fn is_available() -> bool {
    availability_report().is_available()
}

/// Probes `GPUDirect` Storage availability and explains the result.
///
/// # Examples
///
/// ```
/// let report = cudagrep::availability_report();
///
/// println!("GDS available: {}", report.is_available());
/// for diag in report.diagnostics() {
///     println!("  - {}", diag);
/// }
/// ```
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn availability_report() -> AvailabilityReport {
    #[cfg(not(feature = "cuda"))]
    {
        AvailabilityReport::new(
            false,
            AvailabilityReason::FeatureDisabled,
            vec!["Rebuild with `--features cuda` to enable libcufile probing.".to_owned()],
        )
    }

    #[cfg(feature = "cuda")]
    {
        let library = match cufile::CuFileLibrary::load() {
            Ok(library) => library,
            Err(error) => {
                return AvailabilityReport::new(
                    false,
                    map_load_reason(&error),
                    vec![error.to_string()],
                );
            }
        };

        match library.driver_open() {
            Ok(()) => {
                let mut diagnostics = vec![
                    "Loaded libcufile.so.0.".to_owned(),
                    "cuFileDriverOpen succeeded.".to_owned(),
                ];

                let mut gpu_accessible = false;
                if let Ok(cuda_lib) = unsafe { libloading::Library::new("libcuda.so.1") } {
                    unsafe {
                        type CuInitFn = unsafe extern "C" fn(u32) -> i32;
                        type CuDeviceGetCountFn = unsafe extern "C" fn(*mut i32) -> i32;

                        const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
                        const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;

                        let cu_init: Result<libloading::Symbol<CuInitFn>, _> =
                            cuda_lib.get(b"cuInit\0");
                        let cu_device_get_count: Result<libloading::Symbol<CuDeviceGetCountFn>, _> =
                            cuda_lib.get(b"cuDeviceGetCount\0");

                        if let (Ok(init), Ok(get_count)) = (cu_init, cu_device_get_count) {
                            if init(0) == 0 {
                                let mut count = 0;
                                if get_count(&mut count) == 0 && count > 0 {
                                    // Verify at least one device is a real NVIDIA GPU with sufficient compute capability
                                    type CuDevice = i32; // CUdevice is an opaque integer handle
                                    let cu_device_get: Result<
                                        libloading::Symbol<
                                            unsafe extern "C" fn(*mut CuDevice, i32) -> i32,
                                        >,
                                        _,
                                    > = cuda_lib.get(b"cuDeviceGet\0");
                                    let cu_device_get_name: Result<
                                        libloading::Symbol<
                                            unsafe extern "C" fn(*mut u8, i32, CuDevice) -> i32,
                                        >,
                                        _,
                                    > = cuda_lib.get(b"cuDeviceGetName\0");
                                    let cu_device_get_attribute: Result<
                                        libloading::Symbol<
                                            unsafe extern "C" fn(*mut i32, i32, CuDevice) -> i32,
                                        >,
                                        _,
                                    > = cuda_lib.get(b"cuDeviceGetAttribute\0");

                                    if let (Ok(get), Ok(get_name), Ok(get_attr)) =
                                        (cu_device_get, cu_device_get_name, cu_device_get_attribute)
                                    {
                                        for i in 0..count {
                                            let mut device: CuDevice = 0;
                                            if get(&mut device, i) != 0 {
                                                continue;
                                            }

                                            // Get device name to verify it's NVIDIA (not llvmpipe, etc.)
                                            let mut name_buf = [0u8; 256];
                                            if get_name(name_buf.as_mut_ptr(), 256, device) == 0 {
                                                let name_len = name_buf
                                                    .iter()
                                                    .position(|&b| b == 0)
                                                    .unwrap_or(256);
                                                let name =
                                                    String::from_utf8_lossy(&name_buf[..name_len]);
                                                let name_lower = name.to_lowercase();

                                                // Reject known software renderers
                                                if name_lower.contains("llvmpipe")
                                                    || name_lower.contains("software")
                                                    || name_lower.contains("swiftshader")
                                                    || name_lower.contains("microsoft")
                                                {
                                                    diagnostics.push(format!("Device {i} is a software renderer ({name}), skipping."));
                                                    continue;
                                                }

                                                // Check compute capability (GPUDirect Storage requires Pascal+ = 6.0+)
                                                let mut major = 0;
                                                let mut minor = 0;
                                                if get_attr(&mut major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) == 0 &&
                                                   get_attr(&mut minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device) == 0 {
                                                    let compute_capability = major * 10 + minor;
                                                    if compute_capability >= 60 { // Pascal or newer
                                                        gpu_accessible = true;
                                                        diagnostics.push(format!("Found NVIDIA GPU {i}: {name} (CC {major}.{minor})"));
                                                    } else {
                                                        diagnostics.push(format!("Device {i}: {name} has insufficient compute capability ({major}.{minor}, need 6.0+)."));
                                                    }
                                                } else {
                                                    diagnostics.push(format!("Device {i}: {name} (could not determine compute capability)."));
                                                }
                                            }
                                        }

                                        if !gpu_accessible {
                                            diagnostics.push("No suitable NVIDIA GPUs found (need Pascal+ with real hardware).".to_owned());
                                        }
                                    } else {
                                        diagnostics.push("Could not load device query symbols, assuming GPU available.".to_owned());
                                        gpu_accessible = true; // Fallback for older CUDA versions
                                    }
                                } else {
                                    diagnostics
                                        .push("CUDA initialized but 0 devices found.".to_owned());
                                }
                            } else {
                                diagnostics.push("cuInit failed.".to_owned());
                            }
                        } else {
                            diagnostics
                                .push("Could not load cuInit/cuDeviceGetCount symbols.".to_owned());
                        }
                    }
                } else {
                    diagnostics.push(
                        "libcuda.so.1 could not be loaded to verify GPU presence.".to_owned(),
                    );
                }

                if let Err(code) = library.driver_close() {
                    diagnostics.push(format!("cuFileDriverClose failed with error code {code}."));
                    return AvailabilityReport::new(
                        false,
                        AvailabilityReason::DriverCleanupFailed(code),
                        diagnostics,
                    );
                }

                if !gpu_accessible {
                    return AvailabilityReport::new(
                        false,
                        AvailabilityReason::LibraryUnavailable(
                            "No accessible CUDA devices found".to_owned(),
                        ),
                        diagnostics,
                    );
                }

                AvailabilityReport::new(true, AvailabilityReason::Ready, diagnostics)
            }
            Err(code) => AvailabilityReport::new(
                false,
                AvailabilityReason::DriverRejected(code),
                vec![
                    "Loaded libcufile.so.0.".to_owned(),
                    format!("cuFileDriverOpen returned error code {code}."),
                ],
            ),
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn map_load_reason(error: &libloading::Error) -> AvailabilityReason {
    match error {
        libloading::Error::DlOpen { .. } => {
            AvailabilityReason::LibraryUnavailable(error.to_string())
        }
        libloading::Error::DlSym { .. } => AvailabilityReason::SymbolUnavailable(error.to_string()),
        _ => AvailabilityReason::LibraryUnavailable(error.to_string()),
    }
}
