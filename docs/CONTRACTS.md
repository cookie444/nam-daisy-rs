# nam-daisy-rs — Frozen Contracts

Status: **frozen for Phase 0–5** (2026-09-29). Any change to these interfaces
requires re-running `cargo test --workspace` and the desktop parity harness.

This document is the single source of truth for cross-crate boundaries.

---

## 1. Crate graph

```
nam-namb        no_std  .namb header/CRC/JSON-topology parser + encoder
nam-core-nostd  no_std  WaveNet A1 inference Engine (consumes .namb bytes)
nam-hal         no_std  ModelSource abstraction (QSPI / SD / RAM)
nam-parity      std     desktop golden harness (dev-only)
nam-to-namb     std     .nam -> .namb converter (dev tooling)
firmware/nam-pedal  no_std  daisy-embassy application (standalone workspace)
```

Dependency direction: `nam-core-nostd -> nam-namb`. `nam-hal` is independent.
`nam-parity`/`nam-to-namb` are desktop-only and are never linked on target.

---

## 2. `.namb` container (implemented subset)

80-byte packed header, little-endian:

| Off | Size | Field | Notes |
|----:|-----:|-------|-------|
| 0x00 | 4 | `magic` | `0x4E414D42` ("NAMB") |
| 0x04 | 2 | `version` | `1` legacy, `2` pre-transposed |
| 0x06 | 1 | `layout_type` | `0` Original, `1` GateMajorLstm, `2` Interleaved4WaveNet (v2 only) |
| 0x07 | 1 | `flags` | bit0 `FLAG_HAS_CRC32` (v1: reserved, forced 0) |
| 0x08 | 4 | `reserved_v2` | 0 |
| 0x0C | 4 | `weights_offset` | `>= 80`, `<= file_len` |
| 0x10 | 8 | `reserved1` | 0 |
| 0x18 | 4 | `crc32` | IEEE 802.3, poly `0xEDB88320` |
| 0x1C | 4 | `reserved2` | 0 |
| 0x20 | 32 | `version_str` | informational |
| 0x40 | 4 | `sample_rate` f32 | |
| 0x44 | 4 | `input_level_dbu` f32 | |
| 0x48 | 4 | `output_level_dbu` f32 | |
| 0x4C | 4 | `reserved3` | 0 |

**CRC coverage**
- v1: weights block `bytes[weights_offset..]` only; `crc32 == 0` is rejected.
- v2: whole file except the CRC field, i.e. `bytes[..24] ++ bytes[28..]`.
  `FLAG_HAS_CRC32` is mandatory.

**Sections**: `[header 80][optional null-terminated JSON metadata][contiguous LE f32 weights]`.
Trailing bytes that do not complete an f32 are rejected.

**Implemented layouts**: `Original` (0). `GateMajorLstm` (1) and
`Interleaved4WaveNet` (2) are parsed and named but the engine does **not** yet
consume them (see docs/PARITY.md G4). Unknown `layout_type` falls back to
`Original`.

---

## 3. WaveNet A1 `Original` weight order (exact)

This is the order of the `.nam` JSON `weights` array and of the `.namb` v1/v2
`Original` block. `IN` = layer input size, `COND` = condition size,
`CH` = layer channels, `K` = kernel, `HEAD` = head size.

```
For array a in [array0, array1]:
  rechannel.W          [OUT=CH][IN]
  for each dilation d, in order:
    conv1d.W           [OUT=CH][IN=CH][K]      # out-major, then in, then tap
    conv1d.bias        [CH]
    input_mixin.W      [OUT=CH][IN=COND]
    one_by_one.W       [OUT=CH][IN=CH]
    one_by_one.bias    [CH]
  head_rechannel.W     [OUT=HEAD][IN=CH]
  head_rechannel.bias  [HEAD]                   # only if layer.head_bias
head_scale             1
```

Array 0: `IN = layer0.input_size`, `CH = layer0.channels`, `HEAD = layer0.head_size`,
`head_bias = false`, `COND = layer0.condition_size`.
Array 1: `IN = layer0.channels`, `CH = layer1.channels`, `HEAD = layer1.head_size`
(== 1), `head_bias = true`.

**A1 nano** (BossWN-nano): `CH=4, HEAD(0)=2, K=3`, dilations0 `[1,2,4,8,16,32,64]`,
dilations1 `[128,256,512,1,2,4,8,16,32,64,128,256,512]`, 842 weights,
`Original` layout.

Runtime math (scalar f32, no contraction):

```
# per frame t, per layer, per output channel oc
conv_pre[oc]   = bias[oc] + Σ_j mixin[oc,j]·cond[t,j]
               + Σ_{k=0..K-1} Σ_ic Wc[oc,ic,k]·layer_in[t - d·(K-1-k)][ic]
act[oc]        = ReLU(conv_pre) or tanh(conv_pre)
head[oc]       += act[oc]                         # array0 layer0 overwrites; array1 layer0 seeds with array0 head
layer_out[oc]  = layer_in[t][oc]                  # residual
               + b1x1[oc] + Σ_ic W1[oc,ic]·act[ic]
head_out[h]    = (bias[h] if any) + Σ_c Whead[h,c]·head[c]
output[t]      = head_scale · array1.head_out[t]
```

**Prewarm** replicates NAMCore `Prewarm()`: each layer delay line is filled with
its zero-input stationary input vector (computed once), not by iterating silence
for a full receptive field.

---

## 4. `Engine` API (`nam-core-nostd`)

```rust
pub struct Engine { /* fixed-capacity; ~85 KiB */ }

impl Engine {
    pub fn from_slice(bytes: &[u8]) -> Result<Engine, EngineError>;
    pub fn build(topo: &WavenetTopo, weights: &[f32]) -> Result<Engine, EngineError>;
    pub fn reset(&mut self);
    pub fn prewarm(&mut self, num_samples: usize);   // ignored arg; one-shot
    pub fn stabilize(&mut self);
    pub fn process(&mut self, input: &[f32], output: &mut [f32]);
    pub fn input_mult(&self) -> f32;
    pub fn output_mult(&self) -> f32;
    pub fn sample_rate(&self) -> f32;
    pub fn receptive_field(&self) -> usize;
}
```

**Guarantees** on `process`:
- zero heap allocation, zero locks, no panic (geometry validated at build);
- accepts any block length; internally chunks at `MAX_FRAMES = 64`;
- reads/writes at most `min(input.len(), output.len())` samples.

**Compiled capacities** (`CapacityExceeded` otherwise): per-array `CH ≤ 8`,
`HEAD ≤ 8`, `K ≤ 8`, layers `≤ 16`, 2 arrays, `WEIGHT_CAP = 2048` f32,
`STATE_CAP = 16384` f32.

**Not on `process`**: input/output dBu gain. `input_mult`/`output_mult` are
computed from the header (input reference 12 dBu; loudness reference -18 dBFS)
and applied by the host, mirroring the reference core.

---

## 5. `ModelSource` trait (`nam-hal`)

```rust
pub trait ModelSource {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn read(&mut self, offset: usize, dst: &mut [u8]) -> Result<(), SourceError>;
    fn as_slice(&self) -> Option<&[u8]> { None }              // mmap fast path
    fn load_into<'a>(&mut self, staging: &'a mut [u8]) -> Result<&'a [u8], SourceError>;
}
pub enum SourceError { TooLarge { size, capacity }, Io, NotFound }
```

`SliceSource` implements the mmap fast path over a `&[u8]` (RAM or memory-mapped
QSPI). A QSPI/SD source stages bytes with `read` and returns `None` from
`as_slice`.

---

## 6. Memory map (Daisy Seed, Rev1.1 / STM32H750)

| Region | Address | Size | Use |
|--------|---------|------|-----|
| ITCM | 0x0000_0000 | 64 KiB | (reserved) |
| DTCM | 0x2000_0000 | 128 KiB | `Engine` state + scratch + stack + `.bss` |
| AXI SRAM | 0x2400_0000 | 512 KiB | (reserved for larger models) |
| RAM_D2 | 0x3000_0000 | 288 KiB | SAI DMA buffers, QSPI model staging (`.sram1_bss`) |
| QSPI | 0x9000_0000 | 8 MiB | `.namb` storage (length-prefixed at offset 0) |

**Per-model buffer budget (nano)**: weights 842 f32 ≈ **3.4 KiB**; delay-line
state ≈ **47 KiB** (`STATE_CAP` reserved 64 KiB); scratch ≈ 12 KiB. Total
`Engine` ≈ **85 KiB**, placed in DTCM.

**Firmware image**: `text = 88.6 KiB`, `bss = 107.4 KiB` (of which 16 KiB
staging is in RAM_D2). Internal flash budget is 128 KiB — fits, but QSPI-XIP
boot (`APP_TYPE=BOOT_QSPI`) is recommended for headroom.

---

## 7. Audio / RT contract

- Format: stereo interleaved `u32`, 24-bit samples left-justified in each slot,
  32 frames per half-buffer (`HALF_DMA_BUFFER_LENGTH = 64`).
- Deadline: 32 frames @ 48 kHz = 0.667 ms; budget at 480 MHz = **320 000 cycles**.
- `FPSCR.FZ` (flush-to-zero) set at boot and on model change. Cortex-M7 has no DAZ.
- Denormal armour: `+1e-11` DC added before inference, subtracted after.
- CPU load: DWT `CYCCNT` wraps each callback; peak/budget reported once per second.

---

## 8. Error types

- `nam_namb::NambError` — `Truncated`, `InvalidMagic`, `InvalidVersion`,
  `WeightsOffsetOutOfBounds`, `InvalidWeightsOffset`, `CrcMismatch`,
  `CrcMissing`, `CrcMissingV1`, `NonFiniteWeight`, `InvalidHeaderField`,
  `MetadataJson`, `UnsupportedTopology`, `WeightCountMismatch`,
  `EncodeBufferTooSmall`.
- `nam_core_nostd::EngineError` — `Namb(NambError)`, `UnsupportedTopology`,
  `WeightCountMismatch`, `CapacityExceeded`, `BadParameter`.
- `nam_hal::SourceError` — `TooLarge`, `Io`, `NotFound`.
