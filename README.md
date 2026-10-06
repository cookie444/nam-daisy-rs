# nam-daisy-rs

A pure-Rust, `no_std`, allocation-free **Neural Amp Modeler** inference engine
running real-time on the Electrosmith **Daisy Seed** (STM32H750, Cortex-M7 @
480 MHz).

This is an adaptation of [`fabiohl/NeuralAmpModeler-rs`](https://github.com/fabiohl/NeuralAmpModeler-rs)
(Apache-2.0): the NAM numerics and `.namb` format are reused, re-implemented
without `std`, without heap allocation, and without the x86-64-v3 requirement.

## Status

| | |
|---|---|
| G2 desktop parity vs NAMCore | **PASS** — ESR 7.5e-14 (2 048 smp), 2.2e-13 (240 000 smp) |
| ReLU nano vs independent f64 oracle | **PASS** — ESR 7.0e-14 |
| `.namb` Original loader | **PASS** (GateMajor/Interleaved4 not yet consumed) |
| `thumbv7em-none-eabihf` build (engine + firmware) | **PASS** — zero allocator symbols |
| On-hardware CPU load / xruns / HIL | **PENDING** (no board attached) |

Full gate trail: [`docs/PARITY.md`](docs/PARITY.md). Frozen interfaces:
[`docs/CONTRACTS.md`](docs/CONTRACTS.md).

## Layout

```
crates/nam-namb        no_std .namb parser + CRC32 + metadata topology + encoder
crates/nam-core-nostd  no_std WaveNet A1 engine (fixed-capacity, panic-free)
crates/nam-hal         no_std ModelSource trait (QSPI / SD / RAM)
crates/nam-parity      std  desktop golden harness (NAMCore + f64 oracle)
tools/nam-to-namb      .nam -> .namb converter
tools/reference_wavenet.py   independent f64 oracle
tools/stage_model.py   length-prefix a .namb for QSPI staging
firmware/nam-pedal     daisy-embassy application (standalone workspace)
docs/                  CONTRACTS.md, PARITY.md
```

## Quick start

```bash
# Desktop parity (G0/G2/parser)
cargo test --workspace --release

# Cross-build the engine crates for Cortex-M7 (G3)
cargo build --release --target thumbv7em-none-eabihf \
    -p nam-namb -p nam-core-nostd -p nam-hal

# Build the firmware image (G1/G5 scaffolding)
cd firmware/nam-pedal && cargo build --release
```

### Convert and stage a model

```bash
cargo run -p nam-to-namb -- model.nam model.namb
python tools/stage_model.py model.namb staged.bin
# flash staged.bin to QSPI offset 0, then flash the firmware with probe-rs
```

The firmware stages the model in QSPI as `[u32 LE length][.namb bytes]` at
offset 0 and falls back to clean passthrough when no model is present.

## Engine

```rust
use nam_core_nostd::Engine;

let mut engine = Engine::from_slice(namb_bytes)?; // load-time only
engine.prewarm(0);                                // stationary-state fill
engine.process(&input[..32], &mut output[..32]);  // RT-safe, zero alloc
```

- Scope: WaveNet A1 (two-array) models, Tanh/ReLU, `Original` layout.
- Capacities: `CH ≤ 8`, `HEAD ≤ 8`, `K ≤ 8`, ≤16 layers/array, 2048 f32 weights.
- `process` performs no allocation, no locking, and cannot panic.
- Addresses are validated at build time; block lengths are chunked internally.

## Design notes

- **Prewarm** mirrors NAMCore's `Prewarm()`: each layer's delay line is filled
  with its zero-input stationary value, reaching the exact fixpoint in O(layers).
- **Weight parsing** is allocation-free; metadata JSON is read with a purpose-
  built tokenizer.
- **Firmware** sets `FPSCR.FZ`, adds a `1e-11` denormal dither, and measures
  per-callback CPU load with the DWT cycle counter against a 320 000-cycle
  budget (32 frames @ 48 kHz @ 480 MHz).

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
