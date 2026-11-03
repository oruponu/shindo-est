//! The C ABI of shindo-est, declared in `include/shindo_est.h`.
//!
//! - Ownership: a model handle is opened by the library and freed with its
//!   close function. Strings the library returns are static, or owned by the
//!   handle and valid until it is closed. Prediction results go into memory
//!   the caller allocates.
//! - Threads: a model handle may be shared by any number of threads for
//!   prediction, as long as no thread closes it meanwhile.
//! - Errors: functions that can fail return an `int32_t` code, 0 for success.
//!   The library never lets a panic cross the ABI: one becomes
//!   `SHINDO_EST_ERROR_PANIC`.
//! - Options: `shindo_est_options` starts with its own size, and every other
//!   field is 0 by default. The library reads only the fields that fit in
//!   that size and takes defaults for the rest, and it rejects a nonzero field
//!   it does not know, so callers built against older or newer headers keep
//!   working.

mod codes;
mod guard;
mod info;
mod model;
mod pointer;
mod predict;
mod types;

pub use codes::*;
pub use info::*;
pub use model::*;
pub use predict::*;
pub use types::*;
