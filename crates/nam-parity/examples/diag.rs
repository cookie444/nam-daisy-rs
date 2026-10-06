use nam_parity::{encode_namb, esr, parse_nam, read_golden};

const NAM: &str = include_str!("../tests/fixtures/BossWN-nano.nam");
const GOLDEN_V1: &[u8] = include_bytes!("../tests/fixtures/golden_wavenet_nano.bin");

#[allow(clippy::needless_range_loop)]
fn main() {
    let nam = parse_nam(NAM).unwrap();
    let namb = encode_namb(&nam).unwrap();
    let golden = read_golden(GOLDEN_V1).unwrap();
    let prewarm: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);
    let mut engine = nam_core_nostd::Engine::from_slice(&namb).unwrap();
    if prewarm > 0 {
        engine.prewarm(prewarm);
    }
    let mut out = vec![0.0f32; golden.input.len()];
    let mut i = 0;
    while i < out.len() {
        let n = (out.len() - i).min(64);
        engine.process(&golden.input[i..i + n], &mut out[i..i + n]);
        i += n;
    }
    let e = esr(&golden.expected, &out);
    println!(
        "prewarm={prewarm} ESR={e:.4e} head_scale={} rf={}",
        engine.head_scale(),
        engine.receptive_field()
    );

    // gain fit
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (r, t) in golden.expected.iter().zip(out.iter()) {
        num += *r as f64 * *t as f64;
        den += *t as f64 * *t as f64;
    }
    println!("ref/ours least-squares ratio = {:.9}", num / den);

    // largest errors
    let mut idx: Vec<usize> = (0..out.len()).collect();
    idx.sort_by(|&a, &b| {
        let ea = (golden.expected[a] - out[a]).abs();
        let eb = (golden.expected[b] - out[b]).abs();
        eb.partial_cmp(&ea).unwrap()
    });
    for &k in idx.iter().take(12) {
        println!(
            "i={k:5} in={:+.5} ref={:+.6} ours={:+.6} err={:+.2e}",
            golden.input[k],
            golden.expected[k],
            out[k],
            golden.expected[k] - out[k]
        );
    }
    // error by region
    for (name, a, b) in [("head", 0, 64), ("mid", 1000, 1064), ("tail", 1984, 2048)] {
        let mut sig = 0.0;
        let mut noi = 0.0;
        for k in a..b {
            sig += (golden.expected[k] as f64).powi(2);
            noi += (golden.expected[k] as f64 - out[k] as f64).powi(2);
        }
        println!("{name}: esr={:.3e}", noi / sig.max(1e-30));
    }
}
