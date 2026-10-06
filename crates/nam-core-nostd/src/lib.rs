//! `no_std` Neural Amp Modeler inference core.
//!
//! Scope: WaveNet A1 (two-array) models, `Tanh`/`ReLU` activations, `Original`
//! and `Interleaved4WaveNet` `.namb` layouts. Fixed-capacity, allocation-free,
//! panic-free hot path.
//!
//! ```ignore
//! let mut engine = Engine::from_slice(flash_bytes)?;
//! engine.prewarm(2048);
//! engine.process(&input_block, &mut output_block);
//! ```

#![no_std]
#![forbid(unsafe_op_in_unsafe_fn)]

mod activation;
mod error;
mod model;

pub use error::EngineError;
pub use model::{Engine, MAX_CH, MAX_FRAMES, MAX_HEAD, MAX_K, MAX_LAYERS, STATE_CAP, WEIGHT_CAP};
