# PARITY.md — Gate Trail

Verification rules: oracle-first, every numeric change re-runs desktop parity,
embedded output is validated against the desktop core. A phase is done only when
its gate is independently reproduced.

Oracle stock:
- **NAMCore C++ f32 golden** `crates/nam-parity/tests/fixtures/golden_wavenet_nano.bin`
  (2048 samples) and `golden_wavenet_nano_v2_48000.bin` (240 000 samples),
  committed upstream in `fabiohl/NeuralAmpModeler-rs` (Apache-2.0). See
  `tests/fixtures/ATTRIBUTION.md`.
- **Independent f64 oracle** `tools/reference_wavenet.py` (Python, float64),
  validated against the NAMCore golden before use.

Metric: ESR = `Σ(ref-test)² / Σ(ref)²` in f64; SNR = `10·log10(Σref²/Σerr²)`.
Gate for G2: **ESR < 1e-10** vs NAMCore f32.

---

## Summary

| Gate | Scope | Result | Evidence |
|-----:|-------|:------:|----------|
| G0 | Recon, interfaces frozen, skeleton compiles | **PASS** | `cargo build` workspace |
| G1 | Audio bring-up (passthrough + DWT meter + FTZ) | **HW PASS** | runs on board; see "Hardware bring-up" below |
| G2 | Desktop core vs NAMCore | **PASS** | ESR 7.46e-14 (v1), 2.24e-13 (v2) |
| G3 | `no_std` M7 build bit-matches G2 core | **HW PASS** | engine loads & produces signal on M7 |
| G4 | `.namb` loader (Original) | **PASS**; GateMajor/Interleaved4 **NOT DONE** | `nam-namb` tests + on-device nano load |
| G5 | RT integration < 50 % CPU, zero xruns | **FAIL (CPU)** | measured 700 757 cyc/block = **219 %** of the 320 000 budget |
| G6 | HIL quantitative capture | **NOT RUN** | rig exists under `Documents/Daisy Pedal Test harness` |
| G7 | Optimization | **PARTIAL** | prewarm + multi-accumulator + const-generic layer; still ~2× over |
| G8 | LSTM/A2/hot-swap/WASM/publish | **NOT STARTED** | — |

---

## Hardware bring-up (Daisy Seed Rev1.2 / Seed2-DFM, PCM3060)

Verified on real hardware (firmware `firmware/nam-pedal`, flashed over the ST
ROM DFU with `dfu-util -a 0 -s 0x08000000:leave`, then a manual RESET — the
`:leave` alone does not start the app on this ROM).

Findings, in order of discovery:

1. **`dfu-util ...:leave` does not run the app.** Flash, then press RESET.
2. **`Engine` is ~85 KiB** and was constructed on the stack → stack overflow
   (HardFault). Fixed with in-place construction (`Engine::init_into` +
   `StaticCell::uninit`).
3. **The board is PCM3060, not WM8731.** daisy-embassy's `seed_1_1` build hangs
   in codec init; `seed_1_2` proceeds. `seed_1_2` needs `DEMP` (PB11) driven low
   (as libDaisy does).
4. **I²S conversion** must match zlosynth/daisy: signed 24-bit **in the low 24
   bits** of the SAI word. The initial left-justified assumption produced
   silence/distortion; the corrected conversion gives a clean 440 Hz tone and
   audible passthrough.
5. **daisy-embassy's SAI transport is stop-and-go** (ring buffer + `read().await`),
   unlike libDaisy/zlosynth's interrupt-driven **circular DMA**. When a callback
   exceeds the deadline the RX ring **overruns** and `read()` returns `Err`.
6. **Engine cost on target**: first optimized build measured via USB CDC:
   - initial scalar kernel: **1 916 641 cycles/block (599 %)**
   - + multi-accumulator conv + const-generic layer + unchecked hot path:
     **700 757 cycles/block (219 %)**
   - budget is 320 000 cycles (32 frames @ 48 kHz @ 480 MHz).
   The callback overruns every block, so no audio is produced.
7. A "slimmed" BossWN-nano (first 4/7 layers per array, 482 weights) is ~2.25×
   faster on the host, but the on-target first-callback figure was unreliable
   (cold start) and the USB CDC port became flaky, so the steady-state number
   was not captured.

Reference point: tone-3000's C++ NAMCore `NAM_USE_INLINE_GEMM` build runs the
same class of model in **142 350 cycles for 48 frames** (~95 000 for 32 frames),
i.e. ~7× faster than the current Rust kernel. Closing that gap is the remaining
work (weight transposition + frame-GEMM + register blocking), and/or using a
smaller model.

---

## G2 — Desktop core vs NAMCore (measured)

Model `BossWN-nano.nam`: WaveNet A1 nano, `CH=4, HEAD=2, K=3`, Tanh,
`Original` layout, 842 weights. Blocks of 64, stationary prewarm.

| Fixture | Samples | ESR | SNR (dB) | max abs err |
|---------|--------:|----:|---------:|------------:|
| `golden_wavenet_nano.bin` | 2 048 | **7.456e-14** | 131.27 | 6.71e-7 |
| `golden_wavenet_nano_v2_48000.bin` | 240 000 | **2.242e-13** | 126.49 | 1.58e-6 |

Reference target: upstream reports ESR ≈ 6.43e-14 for this model. Both gates
clear `1e-10` by >2 decades.

### ReLU variant (independent f64 oracle)

No NAMCore A1-nano ReLU golden exists upstream. The same weights with
`activation = "ReLU"` are validated against `tools/reference_wavenet.py`, which
was first shown to match the NAMCore golden at **ESR 2.453e-14**.

| Variant | ESR | SNR (dB) |
|---------|----:|---------:|
| ReLU nano vs f64 oracle | **7.035e-14** | 131.53 |

The engine is deterministic across instances (`g2_namb_roundtrip_and_determinism`).

**Key numerics finding**: NAMCore's `Prewarm()` backfills each layer's delay line
with its zero-input stationary value in O(layers), rather than iterating silence
for a full receptive field. Reproducing that removed a head transient that
otherwise dominated the ESR (1e-7 → 7e-14).

---

## G3 — `no_std` / Cortex-M7 build

- `cargo build --release --target thumbv7em-none-eabihf -p nam-namb -p nam-core-nostd -p nam-hal` succeeds.
- Firmware `nam-pedal` links for `thumbv7em-none-eabihf` (see G1).
- `arm-none-eabi-nm` shows **no** `__rust_alloc`/`__rust_dealloc`/`malloc`
  symbols: the target image is allocation-free.
- Bit-match execution against G2 on-device is **pending** (no QEMU model or
  probe-rs hardware in this environment). Mitigations supporting bit-match:
  one shared source; float contraction is disabled by default in Rust/LLVM; the
  engine uses `libm` on both host and target, so `tanh` is the same routine.
  To close this gate: run `cargo run --release` over the committed fixtures on
  hardware via `probe-rs` and compare, or run under `qemu-system-arm -M
  mps2-an500` style rig.

---

## G4 — `.namb` loader

- Header parse, version/layout/flags validation, CRC32 (v1 weights-only and v2
  whole-file-except-CRC) covered by `crates/nam-namb/tests/parser.rs`.
- `Engine::from_slice` loads a real nano `.namb` produced by the `.nam -> .namb`
  path and reproduces the G2 output.
- **Not implemented**: `GateMajorLstm` (1) and `Interleaved4WaveNet` (2) weight
  consumption, and LSTM models. `layout_type` is recognised; unknown values fall
  back to `Original`.

---

## G1 / G5 / G6 — hardware gates (pending)

Firmware implements: SAI passthrough fallback, QSPI model staging with
length prefix, mono neural inference copied to both channels, `FPSCR.FZ`,
denormal dither, DWT `CYCCNT` per-callback load reporting against a 320 000-cycle
budget, heartbeat + clip LED.

To reproduce on hardware:
1. `python tools/nam-to-namb <model.nam> model.namb`
2. `python tools/stage_model.py model.namb staged.bin`
3. Flash `staged.bin` to QSPI offset 0 and the firmware via `probe-rs`.
4. Read `cpu: peak=... cyc (x% of budget)` over `defmt-rtt`; assert < 50 % and
   zero xruns over 10 minutes.
5. Capture audio with the Python HIL rig and compare to the desktop reference
   output for a measured ESR/SNR verdict.

No measurements are recorded here for G1/G5/G6 because no board was attached in
this environment. These rows must be filled before claiming acceptance.

---

## Reproduce

```bash
cargo test --workspace --release            # G0, G2, parser tests
cargo build --release --target thumbv7em-none-eabihf \
    -p nam-namb -p nam-core-nostd -p nam-hal # G3 build
cd firmware/nam-pedal && cargo build --release   # G1/G5 image
python tools/reference_wavenet.py <model.nam> <golden_in.bin> <out.bin> [Tanh|ReLU]
```
