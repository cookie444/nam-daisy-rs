//! Time one block for a .nam file given on the command line.
use nam_parity::{encode_namb, parse_nam};
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).expect("usage: bench_file <model.nam>");
    let json = std::fs::read_to_string(&path).unwrap();
    let nam = parse_nam(&json).unwrap();
    let namb = encode_namb(&nam).unwrap();
    let mut engine = nam_core_nostd::Engine::from_slice(&namb).unwrap();
    engine.stabilize();
    println!("weights={}", nam.weights.len());

    let mut input = [0.0f32; 32];
    for (i, s) in input.iter_mut().enumerate() {
        *s = 0.3 * (i as f32 * 0.2).sin();
    }
    let mut output = [0.0f32; 32];
    for _ in 0..2000 {
        engine.process(&input, &mut output);
    }
    let iters: u32 = 100_000;
    let t0 = Instant::now();
    for _ in 0..iters {
        engine.process(&input, &mut output);
    }
    let ns = t0.elapsed().as_nanos() as f64 / iters as f64;
    println!("{:.0} ns/block  ~{:.0} host cyc @3.7GHz (budget 320000)", ns, ns * 3.7);
}
