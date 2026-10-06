# Golden fixture attribution

The files in this directory are copied verbatim from
[`fabiohl/NeuralAmpModeler-rs`](https://github.com/fabiohl/NeuralAmpModeler-rs)
(Licensed under Apache-2.0, Copyright (c) 2026 Fábio Henrique de Lima Silva).

- `BossWN-nano.nam` — WaveNet A1 nano (CH=4, HEAD=2, Tanh) model.
- `golden_wavenet_nano.bin` — 2048-sample input + NAMCore f32 reference output.
- `golden_wavenet_nano_v2_48000.bin` — 240000-sample stress input + reference output.

These are used as an independent NAMCore C++ oracle for the parity harness
(`tests/wavenet_nano_parity.rs`). See the upstream `tests/fixtures/` directory
and `docs/namb-spec.md`.
