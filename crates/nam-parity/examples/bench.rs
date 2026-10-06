use nam_parity::{encode_namb, parse_nam};
use std::time::Instant;

const NAM: &str = include_str!("../tests/fixtures/BossWN-nano.nam");

fn main() {
    let nam = parse_nam(NAM).unwrap();
    let namb = encode_namb(&nam).unwrap();
    let mut engine = nam_core_nostd::Engine::from_slice(&namb).unwrap();
    engine.stabilize();

    let mut input = [0.0f32; 32];
    for (i, s) in input.iter_mut().enumerate() {
        *s = 0.3 * (i as f32 * 0.2).sin();
    }
    let mut output = [0.0f32; 32];

    // warm up
    for _ in 0..1000 {
        engine.process(&input, &mut output);
    }

    for n in [1usize, 8, 32, 64] {
        let mut inp = [0.0f32; 64];
        let mut out = [0.0f32; 64];
        for (i, s) in inp.iter_mut().enumerate() {
            *s = 0.3 * (i as f32 * 0.2).sin();
        }
        for _ in 0..2000 {
            engine.process(&inp[..n], &mut out[..n]);
        }
        let iters: u32 = if n <= 8 { 200_000 } else { 50_000 };
        let t0 = Instant::now();
        for _ in 0..iters {
            engine.process(&inp[..n], &mut out[..n]);
        }
        let elapsed = t0.elapsed();
        let ns = elapsed.as_nanos() as f64 / iters as f64;
        println!(
            "n={:2}: {:>8.0} ns/block  ~{:>7.0} host cyc @3.7GHz  ({:>6.1} ns/frame)",
            n,
            ns,
            ns * 3.7,
            ns / n as f64
        );
    }
    println!("(budget @480MHz for 32 frames = 320000 cycles)");
}
