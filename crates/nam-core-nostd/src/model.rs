//! WaveNet A1 inference engine: fixed-capacity, `no_std`, allocation-free.
//!
//! Weight order matches the NAM "Original" layout (see `docs/CONTRACTS.md`).
//! The hot path performs no heap allocation, no locking, and cannot panic:
//! all geometry is validated at construction time.

use crate::activation;
use crate::error::EngineError;
use nam_namb::wavenet::{Activation, WavenetTopo, MAX_DILATIONS};
use nam_namb::{Namb, NambError};

/// Maximum frames accepted per `process` call (internal chunking beyond this).
pub const MAX_FRAMES: usize = 64;
/// Per-array layer ceiling.
pub const MAX_LAYERS: usize = 16;
/// Channel / head / kernel ceilings for the compiled capacities.
pub const MAX_CH: usize = 8;
pub const MAX_HEAD: usize = 8;
pub const MAX_K: usize = 8;
/// Weight arena, in `f32`. Nano uses 842; 2048 = 8 KiB head-room.
pub const WEIGHT_CAP: usize = 2048;
/// State arena (history ring buffers), in `f32`. Nano needs 12148.
pub const STATE_CAP: usize = 16384;

const NO_BIAS: usize = usize::MAX;

#[derive(Debug, Clone, Copy)]
struct DenseMeta {
    w_off: usize,
    b_off: usize,
}

impl DenseMeta {
    const NONE: Self = Self {
        w_off: 0,
        b_off: NO_BIAS,
    };
}

#[derive(Debug, Clone, Copy)]
struct LayerMeta {
    conv_w: usize,
    conv_b: usize,
    mix_w: usize,
    one_w: usize,
    one_b: usize,
    dilation: usize,
    rf: usize,
    buf_off: usize,
    act: Activation,
}

impl LayerMeta {
    const NONE: Self = Self {
        conv_w: 0,
        conv_b: NO_BIAS,
        mix_w: 0,
        one_w: 0,
        one_b: NO_BIAS,
        dilation: 0,
        rf: 0,
        buf_off: 0,
        act: Activation::Tanh,
    };
}

#[derive(Debug, Clone, Copy)]
struct ArrayMeta {
    in_sz: usize,
    cond: usize,
    ch: usize,
    head: usize,
    k: usize,
    rechannel: DenseMeta,
    layers: [LayerMeta; MAX_LAYERS],
    nl: usize,
    head_re: DenseMeta,
}

impl ArrayMeta {
    const NONE: Self = Self {
        in_sz: 0,
        cond: 0,
        ch: 0,
        head: 0,
        k: 0,
        rechannel: DenseMeta::NONE,
        layers: [LayerMeta::NONE; MAX_LAYERS],
        nl: 0,
        head_re: DenseMeta::NONE,
    };
}

/// A fully-built, ready-to-run WaveNet A1 engine.
pub struct Engine {
    weights: [f32; WEIGHT_CAP],
    /// Number of valid weights in `weights`.
    pub nw: usize,
    state: [f32; STATE_CAP],
    arrays: [ArrayMeta; 2],
    positions: [[usize; MAX_LAYERS]; 2],
    head_scale: f32,
    sample_rate: f32,
    input_level_dbu: f32,
    output_level_dbu: f32,
    input_mult: f32,
    output_mult: f32,
    // scratch (owned so `process` is reentrant-free but lock-free)
    conv: [f32; MAX_FRAMES * MAX_CH],
    mix: [f32; MAX_FRAMES * MAX_CH],
    a0_out: [f32; MAX_FRAMES * MAX_CH],
    a0_head: [f32; MAX_FRAMES * MAX_HEAD],
    a1_head: [f32; MAX_FRAMES * MAX_HEAD],
    head_accum: [f32; MAX_FRAMES * MAX_CH],
    layer_out: [f32; MAX_FRAMES * MAX_CH],
}

impl Engine {
    /// Total `f32` weights the topology requires.
    pub fn expected_weight_count(topo: &WavenetTopo) -> Result<usize, EngineError> {
        validate_topology(topo)?;
        let mut n = 0usize;
        for (ai, layer) in topo.layers[..topo.num_layers].iter().enumerate() {
            let prev = if ai == 0 {
                None
            } else {
                Some(topo.layers[ai - 1])
            };
            let in_sz = layer.input_size as usize;
            let ch = layer.channels as usize;
            let cond = layer.condition_size as usize;
            let k = layer.kernel_size as usize;
            let head = layer.head_size as usize;
            // rechannel
            n += ch * in_sz;
            // per layer
            n += layer.num_dilations * (ch * ch * k + ch + ch * cond + ch * ch + ch);
            // head rechannel
            n += head * ch;
            if layer.head_bias {
                n += head;
            }
            let _ = prev;
        }
        n += 1; // head_scale
        Ok(n)
    }

    /// Build from a parsed topology plus an ordered `Original`-layout weight slice.
    ///
    /// Convenience for hosts with ample stack. On a Cortex-M use
    /// [`Engine::init_into`] instead: this returns the engine by value, which
    /// needs ~85 KiB of stack.
    pub fn build(topo: &WavenetTopo, weights: &[f32]) -> Result<Self, EngineError> {
        validate_topology(topo)?;
        let need = Self::expected_weight_count(topo)?;
        if weights.len() < need {
            return Err(EngineError::WeightCountMismatch {
                got: weights.len(),
                need,
            });
        }
        if need > WEIGHT_CAP {
            return Err(EngineError::CapacityExceeded);
        }
        // All-zero is a valid `Engine` (f32 zeros / usize zeros; no niches).
        let mut engine: Self = unsafe { core::mem::MaybeUninit::zeroed().assume_init() };
        engine.weights[..need].copy_from_slice(&weights[..need]);
        engine.finish(topo, need)?;
        Ok(engine)
    }

    /// Assign metadata offsets and reset state. `self.weights[..need]` must
    /// already hold the weights; the rest of `self` may be zeroed.
    fn finish(&mut self, topo: &WavenetTopo, need: usize) -> Result<(), EngineError> {
        self.nw = need;
        self.head_scale = topo.head_scale;

        let mut wc = 0usize;
        let mut sc = 0usize;
        for (ai, lt) in topo.layers[..topo.num_layers].iter().enumerate() {
            let lt = *lt;
            let mut a = ArrayMeta::NONE;
            a.in_sz = lt.input_size as usize;
            a.cond = lt.condition_size as usize;
            a.ch = lt.channels as usize;
            a.k = lt.kernel_size as usize;
            a.head = lt.head_size as usize;
            a.nl = lt.num_dilations;

            if a.in_sz > MAX_CH || a.cond > MAX_CH || a.ch > MAX_CH || a.head > MAX_HEAD {
                return Err(EngineError::CapacityExceeded);
            }
            if a.nl == 0 || a.nl > MAX_LAYERS || a.nl > MAX_DILATIONS {
                return Err(EngineError::CapacityExceeded);
            }
            if a.k == 0 || a.k > MAX_K {
                return Err(EngineError::CapacityExceeded);
            }

            a.rechannel = DenseMeta {
                w_off: wc,
                b_off: NO_BIAS,
            };
            wc += a.ch * a.in_sz;

            for li in 0..a.nl {
                let d = lt.dilations[li] as usize;
                if d == 0 {
                    return Err(EngineError::UnsupportedTopology);
                }
                let rf = (a.k - 1) * d;
                let mut lm = LayerMeta::NONE;
                lm.conv_w = wc;
                wc += a.ch * a.ch * a.k;
                lm.conv_b = wc;
                wc += a.ch;
                lm.mix_w = wc;
                wc += a.ch * a.cond;
                lm.one_w = wc;
                wc += a.ch * a.ch;
                lm.one_b = wc;
                wc += a.ch;
                lm.dilation = d;
                lm.rf = rf;
                lm.act = lt.activation;
                lm.buf_off = sc;
                let frames = rf + MAX_FRAMES;
                sc += frames * a.ch;
                a.layers[li] = lm;
            }

            a.head_re = DenseMeta {
                w_off: wc,
                b_off: if lt.head_bias {
                    wc + a.head * a.ch
                } else {
                    NO_BIAS
                },
            };
            wc += a.head * a.ch;
            if lt.head_bias {
                wc += a.head;
            }

            self.arrays[ai] = a;
        }
        wc += 1; // head_scale

        if wc != need || sc > STATE_CAP {
            return Err(EngineError::CapacityExceeded);
        }
        self.reset();
        Ok(())
    }

    /// Parse a `.namb` byte buffer and build the engine **in place**, writing
    /// no large temporary to the stack. This is the Cortex-M entry point.
    ///
    /// On success `dst` is initialised; on failure its contents are unspecified.
    pub fn init_into(
        bytes: &[u8],
        dst: &mut core::mem::MaybeUninit<Engine>,
    ) -> Result<(), EngineError> {
        let namb = Namb::parse(bytes).map_err(EngineError::Namb)?;
        let topo = namb.wavenet_topology().map_err(EngineError::Namb)?;
        validate_topology(&topo)?;
        let need = Self::expected_weight_count(&topo)?;
        if namb.num_weights() < need {
            return Err(EngineError::WeightCountMismatch {
                got: namb.num_weights(),
                need,
            });
        }
        if need > WEIGHT_CAP {
            return Err(EngineError::CapacityExceeded);
        }

        // Zero the destination; all-zero is a valid Engine.
        unsafe {
            core::ptr::write_bytes(dst.as_mut_ptr() as *mut u8, 0, core::mem::size_of::<Engine>());
        }
        let engine = unsafe { dst.assume_init_mut() };
        for (i, w) in namb.weights().take(need).enumerate() {
            if !w.is_finite() {
                return Err(EngineError::Namb(NambError::NonFiniteWeight { index: i }));
            }
            engine.weights[i] = w;
        }
        engine.finish(&topo, need)?;
        let h = namb.header();
        engine.sample_rate = h.sample_rate;
        engine.input_level_dbu = h.input_level_dbu;
        engine.output_level_dbu = h.output_level_dbu;
        engine.recompute_gains();
        Ok(())
    }

    /// Parse a `.namb` byte buffer and build the engine (returns by value).
    ///
    /// Host-only: needs ~85 KiB stack. Cortex-M should use [`Engine::init_into`].
    pub fn from_slice(bytes: &[u8]) -> Result<Self, EngineError> {
        let namb = Namb::parse(bytes).map_err(EngineError::Namb)?;
        let topo = namb.wavenet_topology().map_err(EngineError::Namb)?;
        let need = Self::expected_weight_count(&topo)?;
        if namb.num_weights() < need {
            return Err(EngineError::WeightCountMismatch {
                got: namb.num_weights(),
                need,
            });
        }
        let mut engine: Self = unsafe { core::mem::MaybeUninit::zeroed().assume_init() };
        for (i, w) in namb.weights().take(need).enumerate() {
            if !w.is_finite() {
                return Err(EngineError::Namb(NambError::NonFiniteWeight { index: i }));
            }
            engine.weights[i] = w;
        }
        engine.finish(&topo, need)?;
        let h = namb.header();
        engine.sample_rate = h.sample_rate;
        engine.input_level_dbu = h.input_level_dbu;
        engine.output_level_dbu = h.output_level_dbu;
        engine.recompute_gains();
        Ok(engine)
    }

    fn recompute_gains(&mut self) {
        // NAM convention: input 12 dBu reference. Output normalisation depends
        // on the model's metadata `loudness`, which this crate does not parse
        // yet; with the -18 dBFS reference this is unity gain.
        self.input_mult = db_to_linear(12.0 - self.input_level_dbu);
        self.output_mult = 1.0;
    }

    /// Clear all delay-line state. History starts as `rf` zero frames.
    pub fn reset(&mut self) {
        for s in self.state.iter_mut() {
            *s = 0.0;
        }
        for ai in 0..2 {
            for li in 0..self.arrays[ai].nl {
                self.positions[ai][li] = self.arrays[ai].layers[li].rf;
            }
        }
    }

    /// One-shot stationary-state prewarm.
    ///
    /// Replicates NAMCore's `Prewarm()`: rather than iterating silence for a
    /// full receptive field, each layer's delay line is filled with the
    /// constant input that layer sees under absolute silence. This reaches the
    /// exact zero-input fixpoint in O(layers), removing the cold-start
    /// transient. `_num_samples` is accepted for API compatibility and ignored.
    pub fn prewarm(&mut self, _num_samples: usize) {
        self.stabilize();
    }

    /// Fill every delay line with its zero-input stationary value.
    pub fn stabilize(&mut self) {
        let arrays = self.arrays;
        let w: &[f32] = &self.weights;
        let mut u_in = [0.0f32; MAX_CH];
        let mut a0_out = [0.0f32; MAX_CH];
        let mut a0_head = [0.0f32; MAX_HEAD];
        let mut a1_head = [0.0f32; MAX_HEAD];
        {
            let st: &mut [f32] = &mut self.state;
            let (p0, p1) = self.positions.split_at_mut(1);
            array_fixpoint(
                w,
                st,
                &arrays[0],
                &u_in[..arrays[0].in_sz],
                None,
                &mut p0[0],
                &mut a0_out,
                &mut a0_head,
            );
            u_in[..arrays[1].in_sz].copy_from_slice(&a0_out[..arrays[1].in_sz]);
            array_fixpoint(
                w,
                st,
                &arrays[1],
                &u_in[..arrays[1].in_sz],
                Some(&a0_head),
                &mut p1[0],
                &mut a0_out,
                &mut a1_head,
            );
        }
        // history is now saturated; discard the rest of the arena noise
        let _ = (a1_head[0], a0_head[0]);
    }

    #[inline]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }
    #[inline]
    pub fn input_mult(&self) -> f32 {
        self.input_mult
    }
    #[inline]
    pub fn output_mult(&self) -> f32 {
        self.output_mult
    }
    #[inline]
    pub fn head_scale(&self) -> f32 {
        self.head_scale
    }
    #[inline]
    pub fn receptive_field(&self) -> usize {
        let mut rf = 0;
        for a in &self.arrays {
            for li in 0..a.nl {
                rf += a.layers[li].rf;
            }
        }
        rf
    }

    /// Process one block. `input`/`output` may differ in length; the shorter wins.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        let n = input.len().min(output.len());
        let mut done = 0;
        while done < n {
            let chunk = (n - done).min(MAX_FRAMES);
            self.process_block(
                &input[done..done + chunk],
                &mut output[done..done + chunk],
                chunk,
            );
            done += chunk;
        }
    }

    fn process_block(&mut self, input: &[f32], output: &mut [f32], n: usize) {
        let arrays = self.arrays;
        let w: &[f32] = &self.weights;
        {
            let st: &mut [f32] = &mut self.state;
            let (a0, a1) = (arrays[0], arrays[1]);
            let (pos0, pos1) = self.positions.split_at_mut(1);
            let pos0 = &mut pos0[0];
            let pos1 = &mut pos1[0];

            // array 0
            array_process(
                w,
                st,
                &a0,
                input,
                input,
                None,
                n,
                pos0,
                &mut self.a0_out,
                &mut self.a0_head,
                &mut self.conv,
                &mut self.head_accum,
                &mut self.layer_out,
            );

            // array 1 — input = array0 residual output, seed = array0 head output
            let (a0_out, a0_head) = (&self.a0_out, &self.a0_head);
            array_process(
                w,
                st,
                &a1,
                a0_out,
                input,
                Some(a0_head),
                n,
                pos1,
                &mut self.mix, // unused as array output; scratch
                &mut self.a1_head,
                &mut self.conv,
                &mut self.head_accum,
                &mut self.layer_out,
            );
        }

        let head_scale = self.head_scale;
        for (o, h) in output[..n].iter_mut().zip(self.a1_head[..n].iter()) {
            *o = *h * head_scale;
        }
    }
}

fn validate_topology(topo: &WavenetTopo) -> Result<(), EngineError> {
    if topo.num_layers != 2 {
        return Err(EngineError::UnsupportedTopology);
    }
    let l0 = topo.layers[0];
    let l1 = topo.layers[1];
    if l0.gated || l1.gated {
        return Err(EngineError::UnsupportedTopology);
    }
    if l0.head_bias {
        return Err(EngineError::UnsupportedTopology);
    }
    if l0.input_size != 1 || l1.input_size as usize != l0.channels as usize {
        return Err(EngineError::UnsupportedTopology);
    }
    if l1.head_size != 1 {
        return Err(EngineError::UnsupportedTopology);
    }
    if !topo.head_scale.is_finite() {
        return Err(EngineError::UnsupportedTopology);
    }
    Ok(())
}

#[inline]
fn db_to_linear(db: f32) -> f32 {
    libm::powf(10.0, db * (1.0 / 20.0))
}

/// Compact (if needed) and return the frame index at which `n` new frames are
/// written for layer `li`.
#[inline]
fn prepare(
    st: &mut [f32],
    l: &LayerMeta,
    ch: usize,
    n: usize,
    pos: &mut [usize],
    li: usize,
) -> usize {
    let f_cap = l.rf + MAX_FRAMES;
    let p = pos[li];
    if p + n > f_cap {
        let keep = if p < l.rf { p } else { l.rf };
        if keep > 0 {
            let src = l.buf_off + (p - keep) * ch;
            let dst = l.buf_off;
            st.copy_within(src..src + keep * ch, dst);
        }
        pos[li] = keep;
    }
    pos[li]
}

/// Fill one layer array's delay lines with its zero-input stationary state and
/// return the stationary array output / head output.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn array_fixpoint(
    w: &[f32],
    st: &mut [f32],
    a: &ArrayMeta,
    u_in: &[f32],
    seed: Option<&[f32]>,
    pos: &mut [usize],
    out: &mut [f32],
    head_out: &mut [f32],
) {
    let ch = a.ch;
    let in_sz = a.in_sz;
    let k = a.k;

    // Stationary rechannel output (silence maps linearly, no bias).
    let mut u = [0.0f32; MAX_CH];
    let rw = &w[a.rechannel.w_off..];
    for oc in 0..ch {
        let mut s = 0.0f32;
        for ic in 0..in_sz {
            s += u_in[ic] * rw[oc * in_sz + ic];
        }
        u[oc] = s;
    }

    let mut head_accum = [0.0f32; MAX_CH];

    for li in 0..a.nl {
        let l = a.layers[li];

        // Saturate this layer's history with its constant input `u`.
        let boff = l.buf_off;
        for frame in 0..l.rf {
            let base = boff + frame * ch;
            st[base..base + ch].copy_from_slice(&u[..ch]);
        }
        pos[li] = l.rf;

        // conv_pre with every tap equal to `u`; mixin(cond=0) = 0.
        let cw = &w[l.conv_w..];
        let mut act_buf = [0.0f32; MAX_CH];
        for oc in 0..ch {
            let mut acc = if l.conv_b != NO_BIAS {
                w[l.conv_b + oc]
            } else {
                0.0
            };
            for kk in 0..k {
                for ic in 0..ch {
                    acc += cw[(oc * ch + ic) * k + kk] * u[ic];
                }
            }
            act_buf[oc] = activation::apply(l.act, acc);
        }

        for oc in 0..ch {
            if li == 0 {
                head_accum[oc] = match seed {
                    Some(s) => s[oc] + act_buf[oc],
                    None => act_buf[oc],
                };
            } else {
                head_accum[oc] += act_buf[oc];
            }
        }

        // y = u + bias1x1 + W1x1 * act
        let ow = &w[l.one_w..];
        let ob = if l.one_b != NO_BIAS {
            Some(l.one_b)
        } else {
            None
        };
        let mut y = [0.0f32; MAX_CH];
        for oc in 0..ch {
            let mut acc = u[oc];
            if let Some(b) = ob {
                acc += w[b + oc];
            }
            for ic in 0..ch {
                acc += act_buf[ic] * ow[oc * ch + ic];
            }
            y[oc] = acc;
        }
        u = y;
    }

    out[..ch].copy_from_slice(&u[..ch]);

    let hr = a.head_re;
    let head = a.head;
    let hw = &w[hr.w_off..];
    for h in 0..head {
        let mut acc = if hr.b_off != NO_BIAS {
            w[hr.b_off + h]
        } else {
            0.0
        };
        for c in 0..ch {
            acc += head_accum[c] * hw[h * ch + c];
        }
        head_out[h] = acc;
    }
}

/// Conv contribution for one output channel, reading the state directly (no
/// gather buffer) with 4 independent accumulators. `CH`/`K` are compile-time so
/// the loops fully unroll and vectorize.
#[inline(always)]
fn conv_accum<const CH: usize, const K: usize>(
    cw: &[f32],
    oc: usize,
    st: &[f32],
    base: usize,
    d: usize,
) -> f32 {
    let wb = oc * CH * K;
    let mut a = [0.0f32; 4];
    let mut j = 0usize;
    for kk in 0..K {
        let xf = base - d * (K - 1 - kk) * CH;
        for ic in 0..CH {
            // SAFETY: validated geometry bounds `st`; `cw` is the layer's conv.
            let t = unsafe {
                *cw.get_unchecked(wb + ic * K + kk) * *st.get_unchecked(xf + ic)
            };
            a[j & 3] += t;
            j += 1;
        }
    }
    (a[0] + a[1]) + (a[2] + a[3])
}

/// Full layer for one layer array entry: conv + activation + head + 1x1
/// residual, written to `dst`. `CH`/`K` are compile-time so the whole frame
/// loop is const-generic and unrolled.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn layer_ck<const CH: usize, const K: usize>(
    w: &[f32],
    cw: &[f32],
    bias_off: usize,
    mix: &[f32],
    cond: &[f32],
    cond_sz: usize,
    st: &[f32],
    boff: usize,
    s: usize,
    n: usize,
    d: usize,
    act: Activation,
    seed: Option<&[f32]>,
    is_first: bool,
    conv: &mut [f32],
    head_accum: &mut [f32],
    ow: &[f32],
    ob: Option<usize>,
    dst: &mut [f32],
) {
    for f in 0..n {
        let cf = f * CH;
        let cb = f * cond_sz;
        let base = boff + (s + f) * CH;
        for oc in 0..CH {
            let mut acc = if bias_off != NO_BIAS {
                w[bias_off + oc]
            } else {
                0.0
            };
            for j in 0..cond_sz {
                acc += cond[cb + j] * mix[oc * cond_sz + j];
            }
            acc += conv_accum::<CH, K>(cw, oc, st, base, d);
            let a = activation::apply(act, acc);
            conv[cf + oc] = a;
            if is_first {
                head_accum[cf + oc] = match seed {
                    Some(seed) => seed[cf + oc] + a,
                    None => a,
                };
            } else {
                head_accum[cf + oc] += a;
            }
        }
    }
    for f in 0..n {
        let cf = f * CH;
        let res = boff + (s + f) * CH;
        for oc in 0..CH {
            let mut acc = st[res + oc];
            if let Some(b) = ob {
                acc += w[b + oc];
            }
            acc += dense_accum::<CH>(ow, oc, &conv[cf..cf + CH]);
            dst[cf + oc] = acc;
        }
    }
}

/// Generic (runtime `ch`/`k`) layer kernel. Used only for shapes outside the
/// specialized `(CH,K)` set.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn layer_generic(
    w: &[f32],
    cw: &[f32],
    bias_off: usize,
    mix: &[f32],
    cond: &[f32],
    cond_sz: usize,
    st: &[f32],
    boff: usize,
    s: usize,
    n: usize,
    ch: usize,
    k: usize,
    d: usize,
    act: Activation,
    seed: Option<&[f32]>,
    is_first: bool,
    conv: &mut [f32],
    head_accum: &mut [f32],
    ow: &[f32],
    ob: Option<usize>,
    dst: &mut [f32],
) {
    for f in 0..n {
        let cf = f * ch;
        let cb = f * cond_sz;
        let base = boff + (s + f) * ch;
        for oc in 0..ch {
            let mut acc = if bias_off != NO_BIAS {
                w[bias_off + oc]
            } else {
                0.0
            };
            for j in 0..cond_sz {
                acc += cond[cb + j] * mix[oc * cond_sz + j];
            }
            let wb = oc * ch * k;
            for kk in 0..k {
                let xf = base - d * (k - 1 - kk) * ch;
                for ic in 0..ch {
                    acc += cw[wb + ic * k + kk] * st[xf + ic];
                }
            }
            let a = activation::apply(act, acc);
            conv[cf + oc] = a;
            if is_first {
                head_accum[cf + oc] = match seed {
                    Some(seed) => seed[cf + oc] + a,
                    None => a,
                };
            } else {
                head_accum[cf + oc] += a;
            }
        }
    }
    for f in 0..n {
        let cf = f * ch;
        let res = boff + (s + f) * ch;
        for oc in 0..ch {
            let mut acc = st[res + oc];
            if let Some(b) = ob {
                acc += w[b + oc];
            }
            for ic in 0..ch {
                acc += conv[cf + ic] * ow[oc * ch + ic];
            }
            dst[cf + oc] = acc;
        }
    }
}

/// 1x1 (dense) contribution for one output channel with 4 accumulators.
#[inline(always)]
fn dense_accum<const CH: usize>(ow: &[f32], oc: usize, act: &[f32]) -> f32 {
    let base = oc * CH;
    let mut a = [0.0f32; 4];
    for ic in 0..CH {
        // SAFETY: CH is the validated channel count.
        let t = unsafe { *ow.get_unchecked(base + ic) * *act.get_unchecked(ic) };
        a[ic & 3] += t;
    }
    (a[0] + a[1]) + (a[2] + a[3])
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn array_process(
    w: &[f32],
    st: &mut [f32],
    a: &ArrayMeta,
    inputs: &[f32],
    cond: &[f32],
    seed: Option<&[f32]>,
    n: usize,
    pos: &mut [usize],
    out: &mut [f32],
    head_out: &mut [f32],
    conv: &mut [f32],
    head_accum: &mut [f32],
    layer_out: &mut [f32],
) {
    let ch = a.ch;
    let cond_sz = a.cond;
    let in_sz = a.in_sz;
    let k = a.k;

    // ---- layer 0 input: rechannel(inputs) written into layer 0 buffer ----
    let start0 = prepare(st, &a.layers[0], ch, n, pos, 0);
    let buf0 = a.layers[0].buf_off;
    let rw = &w[a.rechannel.w_off..];
    for f in 0..n {
        let src = &inputs[f * in_sz..f * in_sz + in_sz];
        let dst = buf0 + (start0 + f) * ch;
        for oc in 0..ch {
            let mut s = 0.0f32;
            for ic in 0..in_sz {
                s += src[ic] * rw[oc * in_sz + ic];
            }
            st[dst + oc] = s;
        }
    }

    head_accum[..n * ch].fill(0.0);

    for li in 0..a.nl {
        let l = a.layers[li];
        let s = pos[li];
        let boff = l.buf_off;
        let cw = &w[l.conv_w..];
        let d = l.dilation;
        let bias_off = l.conv_b;
        let mix = &w[l.mix_w..];

        // Fused const-generic layer kernel (conv + activation + head + 1x1).
        // Dispatch once per layer; generic fallback otherwise.
        let ow = &w[l.one_w..];
        let ob = if l.one_b != NO_BIAS {
            Some(l.one_b)
        } else {
            None
        };
        let is_first = li == 0;

        if li + 1 < a.nl {
            let nxt = a.layers[li + 1];
            let sn = prepare(st, &nxt, ch, n, pos, li + 1);
            let dst_off = nxt.buf_off + sn * ch;
            match (ch, k) {
                (2, 3) => layer_ck::<2, 3>(
                    w, cw, bias_off, mix, cond, cond_sz, st, boff, s, n, d, l.act, seed, is_first,
                    conv, head_accum, ow, ob, &mut layer_out[..n * ch],
                ),
                (4, 3) => layer_ck::<4, 3>(
                    w, cw, bias_off, mix, cond, cond_sz, st, boff, s, n, d, l.act, seed, is_first,
                    conv, head_accum, ow, ob, &mut layer_out[..n * ch],
                ),
                _ => layer_generic(
                    w, cw, bias_off, mix, cond, cond_sz, st, boff, s, n, ch, k, d, l.act, seed,
                    is_first, conv, head_accum, ow, ob, &mut layer_out[..n * ch],
                ),
            }
            st[dst_off..dst_off + n * ch].copy_from_slice(&layer_out[..n * ch]);
        } else {
            match (ch, k) {
                (2, 3) => layer_ck::<2, 3>(
                    w, cw, bias_off, mix, cond, cond_sz, st, boff, s, n, d, l.act, seed, is_first,
                    conv, head_accum, ow, ob, out,
                ),
                (4, 3) => layer_ck::<4, 3>(
                    w, cw, bias_off, mix, cond, cond_sz, st, boff, s, n, d, l.act, seed, is_first,
                    conv, head_accum, ow, ob, out,
                ),
                _ => layer_generic(
                    w, cw, bias_off, mix, cond, cond_sz, st, boff, s, n, ch, k, d, l.act, seed,
                    is_first, conv, head_accum, ow, ob, out,
                ),
            }
        }

        // advance this layer's write position after its output was consumed
        pos[li] = s + n;
    }

    // ---- head rechannel: Dense<CH, HEAD> ----
    let hr = a.head_re;
    let head = a.head;
    let hw = &w[hr.w_off..];
    for f in 0..n {
        let cf = f * ch;
        let hf = f * head;
        for h in 0..head {
            let mut acc = if hr.b_off != NO_BIAS {
                w[hr.b_off + h]
            } else {
                0.0
            };
            for c in 0..ch {
                acc += head_accum[cf + c] * hw[h * ch + c];
            }
            head_out[hf + h] = acc;
        }
    }
}

