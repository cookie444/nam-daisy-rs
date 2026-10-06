//! Micro-benchmark: clean conv kernel vs the engine, to estimate headroom.
use std::time::Instant;

#[inline(always)]
fn conv_accum<const CH: usize, const K: usize>(w: &[f32], oc: usize, st: &[f32], base: usize) -> f32 {
    let wbase = oc * CH * K;
    let mut a = [0.0f32; 4];
    let mut j = 0usize;
    for kk in 0..K {
        let wb = wbase + kk;
        for ic in 0..CH {
            let t = unsafe { *w.get_unchecked(wb + ic * K) * *st.get_unchecked(base + kk * CH + ic) };
            a[j & 3] += t;
            j += 1;
        }
    }
    (a[0] + a[1]) + (a[2] + a[3])
}

fn main() {
    const N: usize = 32;
    const CH: usize = 4;
    const K: usize = 3;
    const LAYERS: usize = 20;
    let mut w = vec![0.0f32; CH * CH * K * LAYERS];
    for (i, v) in w.iter_mut().enumerate() {
        *v = ((i * 7 % 13) as f32) * 0.01 - 0.05;
    }
    let mut st = vec![0.0f32; (N + K + 8) * CH * LAYERS];
    for (i, v) in st.iter_mut().enumerate() {
        *v = ((i * 3 % 11) as f32) * 0.01 - 0.05;
    }
    let mut conv = vec![0.0f32; N * CH];

    let iters: u32 = 200_000;
    let t0 = Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..iters {
        for l in 0..LAYERS {
            let base_l = l * (N + K + 8) * CH;
            for f in 0..N {
                for oc in 0..CH {
                    conv[f * CH + oc] = conv_accum::<CH, K>(&w, oc, &st, base_l + f * CH);
                }
            }
        }
        sink += conv[0];
    }
    let dt = t0.elapsed();
    let ns = dt.as_nanos() as f64 / iters as f64;
    let macs = (N * CH * K * CH * LAYERS) as f64;
    println!("clean conv: {:.0} ns/block  ~{:.0} host cyc @3.7GHz", ns, ns * 3.7);
    println!(
        "  MACs/block={:.0}  {:.2} ns/MAC  {:.1} cyc/MAC @3.7GHz",
        macs,
        ns / macs,
        ns * 3.7 / macs
    );
    println!("  sink={}", sink);
}
