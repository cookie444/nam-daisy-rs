//! Activation functions. `no_std`-safe.
//!
//! Tanh uses `libm` so results are identical between the host parity harness
//! and the Cortex-M7 firmware (the M7 FPU has no transcendental unit).

use nam_namb::wavenet::Activation;

/// Apply an activation to a single value.
#[inline(always)]
pub fn apply(act: Activation, x: f32) -> f32 {
    match act {
        Activation::Relu => {
            if x > 0.0 {
                x
            } else {
                0.0
            }
        }
        // Keep the (large) libm tanh out of the hot kernel; only Tanh models
        // pay for it.
        Activation::Tanh => tanh_slow(x),
    }
}

#[inline(never)]
#[cold]
fn tanh_slow(x: f32) -> f32 {
    libm::tanhf(x)
}
