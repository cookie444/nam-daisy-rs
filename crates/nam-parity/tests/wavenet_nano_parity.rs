//! G2 parity: `nam-core-nostd` vs the NAMCore C++ golden fixture.
//!
//! Oracle: `golden_wavenet_nano.bin` (and the 5 s v2 variant) rendered by the
//! reference C++ NAMCore, committed upstream. Model: `BossWN-nano.nam`
//! (WaveNet A1 nano, CH=4, HEAD=2, K=3, Tanh, Original layout).
//!
//! Gate: ESR < 1e-10 against the golden output.

use nam_parity::{encode_namb, esr, max_abs_err, parse_nam, read_golden, snr_db};

const NAM: &str = include_str!("fixtures/BossWN-nano.nam");
const GOLDEN_V1: &[u8] = include_bytes!("fixtures/golden_wavenet_nano.bin");
const GOLDEN_V2: &[u8] = include_bytes!("fixtures/golden_wavenet_nano_v2_48000.bin");
const GOLDEN_RELU: &[u8] = include_bytes!("fixtures/golden_wavenet_nano_relu.bin");

fn run(v2: bool) -> (f64, f64, f32, u32) {
    let nam = parse_nam(NAM).expect("parse .nam");
    let namb = encode_namb(&nam).expect("encode .namb");

    let golden_bytes = if v2 { GOLDEN_V2 } else { GOLDEN_V1 };
    let golden = read_golden(golden_bytes).expect("read golden");

    let mut engine = nam_core_nostd::Engine::from_slice(&namb).expect("build engine");
    // Match NAMCore's prewarm before rendering.
    engine.prewarm(2048);

    let mut out = vec![0.0f32; golden.input.len()];
    const BLOCK: usize = 64;
    let mut i = 0;
    while i < out.len() {
        let n = (out.len() - i).min(BLOCK);
        engine.process(&golden.input[i..i + n], &mut out[i..i + n]);
        i += n;
    }

    let e = esr(&golden.expected, &out);
    let s = snr_db(&golden.expected, &out);
    let m = max_abs_err(&golden.expected, &out);
    (e, s, m, golden.input.len() as u32)
}

#[test]
fn g2_nano_parity_v1_2048() {
    let (e, s, m, n) = run(false);
    eprintln!("G2 v1: n={n} ESR={e:.3e} SNR={s:.2} dB max_abs={m:.3e}");
    assert!(e < 1e-10, "ESR {e:.3e} exceeds 1e-10");
}

#[test]
fn g2_nano_parity_v2_48000() {
    let (e, s, m, n) = run(true);
    eprintln!("G2 v2: n={n} ESR={e:.3e} SNR={s:.2} dB max_abs={m:.3e}");
    assert!(e < 1e-10, "ESR {e:.3e} exceeds 1e-10");
}

/// A1 nano, ReLU activation, validated against an independent f64 Python oracle
/// (`tools/reference_wavenet.py`, itself validated against the NAMCore golden at
/// ESR 2.5e-14). No NAMCore ReLU nano golden exists upstream, so this variant
/// uses the f64 oracle.
#[test]
fn g2_nano_relu_vs_f64_oracle() {
    let nam = parse_nam(NAM).unwrap();
    let meta = String::from_utf8(nam.metadata_json.clone()).unwrap();
    let meta = meta.replace("\"Tanh\"", "\"ReLU\"");
    assert!(meta.contains("ReLU"));
    let nam = nam_parity::NamJson {
        metadata_json: meta.into_bytes(),
        ..nam
    };
    let namb = encode_namb(&nam).unwrap();

    let golden = read_golden(GOLDEN_RELU).unwrap();
    let mut engine = nam_core_nostd::Engine::from_slice(&namb).unwrap();
    engine.prewarm(2048);
    let mut out = vec![0.0f32; golden.input.len()];
    let mut i = 0;
    while i < out.len() {
        let n = (out.len() - i).min(64);
        engine.process(&golden.input[i..i + n], &mut out[i..i + n]);
        i += n;
    }
    let e = esr(&golden.expected, &out);
    let s = snr_db(&golden.expected, &out);
    eprintln!("G2 ReLU: ESR={e:.3e} SNR={s:.2} dB");
    assert!(e < 1e-10, "ReLU ESR {e:.3e} exceeds 1e-10");

    // ReLU must genuinely differ from the Tanh model.
    let tanh_engine = {
        let nam_tanh = parse_nam(NAM).unwrap();
        let namb_tanh = encode_namb(&nam_tanh).unwrap();
        let mut eng = nam_core_nostd::Engine::from_slice(&namb_tanh).unwrap();
        eng.prewarm(2048);
        eng
    };
    let mut out_tanh = vec![0.0f32; golden.input.len()];
    let mut eng = tanh_engine;
    eng.process(&golden.input, &mut out_tanh);
    assert!(out != out_tanh);
}

#[test]
fn g2_namb_roundtrip_and_determinism() {
    let nam = parse_nam(NAM).unwrap();
    let namb = encode_namb(&nam).unwrap();
    let golden = read_golden(GOLDEN_V1).unwrap();

    let mut a = nam_core_nostd::Engine::from_slice(&namb).unwrap();
    let mut b = nam_core_nostd::Engine::from_slice(&namb).unwrap();
    a.prewarm(2048);
    b.prewarm(2048);

    let mut oa = vec![0.0f32; golden.input.len()];
    let mut ob = vec![0.0f32; golden.input.len()];
    a.process(&golden.input, &mut oa);
    b.process(&golden.input, &mut ob);
    assert_eq!(oa, ob, "engine output must be deterministic");

    // A fresh engine that was not prewarmed must differ (state matters).
    let mut c = nam_core_nostd::Engine::from_slice(&namb).unwrap();
    let mut oc = vec![0.0f32; golden.input.len()];
    c.process(&golden.input, &mut oc);
    assert!(esr(&oa, &oc) > 0.0 || oa.iter().zip(oc.iter()).any(|(x, y)| x != y));
}
