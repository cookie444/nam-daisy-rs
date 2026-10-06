#!/usr/bin/env python3
"""Independent f64 WaveNet-A1 oracle for the nam-parity harness.

This is a deliberately separate (Python, float64) implementation of the NAM
WaveNet A1 inference algorithm. It exists to validate ``nam-core-nostd`` on
model variants for which no C++ NAMCore golden is available (notably the A1
nano ReLU variant). It follows the same `.nam` "Original" weight order
documented in docs/CONTRACTS.md.

Usage:
  python reference_wavenet.py <model.nam> <golden_in.bin> <out_golden.bin> [activation]

`golden_in.bin` is the reference `[u32 N][f32 xN input][f32 xN expected]`
container; only its input block is reused. `activation` overrides the model
activation (`Tanh`/`ReLU`); default keeps the model's own.
"""

import json
import struct
import sys


def load_golden_input(path):
    with open(path, "rb") as f:
        blob = f.read()
    n = struct.unpack_from("<I", blob, 0)[0]
    x = struct.unpack_from(f"<{n}f", blob, 4)
    return list(x)


def activation(name, x):
    if name == "ReLU":
        return x if x > 0.0 else 0.0
    # tanh
    import math

    return math.tanh(x)


class Layer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ch = cfg["channels"]
        self.k = cfg["kernel_size"]
        self.dil = cfg["dilations"]
        self.act = cfg["activation"]
        self.rf = [ (self.k - 1) * d for d in self.dil ]
        # weights filled later
        self.conv = []   # per dilation: [out][in][k]
        self.conv_b = []
        self.mixin = []  # per dilation: [out][in=cond]
        self.one = []    # per dilation: [out][in]
        self.one_b = []


def read_weights(weights, cfg_layers, cond_sizes):
    """Consume the Original-order weight stream; return per-layer structures + head info."""
    it = iter(weights)
    take = lambda n: [next(it) for _ in range(n)]

    layers = []
    for li, cfg in enumerate(cfg_layers):
        L = Layer(cfg)
        cond = cfg["condition_size"]
        ch, k = L.ch, L.k
        for d in range(len(L.dil)):
            conv = take(ch * ch * k)  # [out][in][k]
            bias = take(ch)
            mix = take(ch * cond)     # [out][in]
            one = take(ch * ch)       # [out][in]
            oneb = take(ch)
            L.conv.append(conv)
            L.conv_b.append(bias)
            L.mixin.append(mix)
            L.one.append(one)
            L.one_b.append(oneb)
        layers.append(L)

    # For each layer array, rechannel comes *before* its layers; head_rechannel after.
    # We already consumed layers. Reconstruct by consuming in the true order instead:
    raise RuntimeError("use parse_model")


def parse_model(d):
    """Return (arrays, head_scale). arrays = list of dicts with rechannel, layers, head_re."""
    weights = d["weights"]
    pos = 0

    def take(n):
        nonlocal pos
        r = weights[pos:pos + n]
        pos += n
        return r

    cfg = d["config"]
    arrays = []
    for cfg_layer in cfg["layers"]:
        ch = cfg_layer["channels"]
        k = cfg_layer["kernel_size"]
        cond = cfg_layer["condition_size"]
        in_sz = cfg_layer["input_size"]
        head = cfg_layer["head_size"]
        dils = cfg_layer["dilations"]

        rechannel = take(ch * in_sz)  # [out=ch][in]
        layers = []
        for _ in dils:
            conv = take(ch * ch * k)
            bias = take(ch)
            mix = take(ch * cond)
            one = take(ch * ch)
            oneb = take(ch)
            layers.append(
                dict(conv=conv, bias=bias, mix=mix, one=one, oneb=oneb,
                     dilation=_)
            )
        head_re = take(head * ch)
        head_bias = take(head) if cfg_layer["head_bias"] else None
        arrays.append(dict(
            ch=ch, k=k, cond=cond, in_sz=in_sz, head=head,
            dils=dils, act=cfg_layer["activation"], layers=layers,
            rechannel=rechannel, head_re=head_re, head_bias=head_bias,
        ))

    head_scale = weights[pos]
    pos += 1
    assert pos == len(weights), (pos, len(weights))
    return arrays, head_scale


def dense(w, inp, out_sz, in_sz):
    out = [0.0] * out_sz
    for o in range(out_sz):
        s = 0.0
        for i in range(in_sz):
            s += inp[i] * w[o * in_sz + i]
        out[o] = s
    return out


def array_fixpoint(a, u_in):
    """Stationary zero-input state: fill per-layer constant input history."""
    u = dense(a["rechannel"], u_in, a["ch"], a["in_sz"])
    const_in = [None] * len(a["dils"])
    head_accum = [0.0] * a["ch"]
    for li, L in enumerate(a["layers"]):
        const_in[li] = list(u)
        ch, k = a["ch"], a["k"]
        act = [0.0] * ch
        for o in range(ch):
            acc = L["bias"][o]
            for kk in range(k):
                for ic in range(ch):
                    acc += L["conv"][(o * ch + ic) * k + kk] * u[ic]
            act[o] = activation(a["act"], acc)
        for o in range(ch):
            head_accum[o] += act[o]
        y = [0.0] * ch
        for o in range(ch):
            s = u[o] + L["oneb"][o]
            for ic in range(ch):
                s += act[ic] * L["one"][o * ch + ic]
            y[o] = s
        u = y
    head_out = [0.0] * a["head"]
    for h in range(a["head"]):
        s = a["head_bias"][h] if a["head_bias"] else 0.0
        for c in range(a["ch"]):
            s += head_accum[c] * a["head_re"][h * a["ch"] + c]
        head_out[h] = s
    return const_in, u, head_out


def process(d, x, act_override=None):
    if act_override:
        for L in d["config"]["layers"]:
            L["activation"] = act_override
    arrays, head_scale = parse_model(d)

    # Prewarm: constants per layer + array0 outputs/head.
    a0 = arrays[0]
    c0, a0out, a0head = array_fixpoint(a0, [0.0] * a0["in_sz"])
    a1 = arrays[1]
    c1, _, _ = array_fixpoint(a1, a0out)

    # Histories per array/layer (list of input frames). Init with rf constants.
    def init_hist(a, const_in):
        h = []
        for li, L in enumerate(a["layers"]):
            rf = (a["k"] - 1) * a["dils"][li]
            h.append([list(const_in[li]) for _ in range(rf)])
        return h

    hist0 = init_hist(a0, c0)
    hist1 = init_hist(a1, c1)

    out = []
    for t in range(len(x)):
        # ---- array 0 ----
        u = dense(a0["rechannel"], [x[t]], a0["ch"], a0["in_sz"])
        a0_head_acc = [0.0] * a0["ch"]
        for li in range(len(a0["dils"])):
            L = a0["layers"][li]
            hist0[li].append(list(u))
            ch, k, d = a0["ch"], a0["k"], a0["dils"][li]
            H = hist0[li]
            act = [0.0] * ch
            for o in range(ch):
                acc = L["bias"][o]
                # mixin cond = raw input
                for j in range(a0["cond"]):
                    acc += x[t] * L["mix"][o * a0["cond"] + j]
                for kk in range(k):
                    off = d * (k - 1 - kk)
                    fr = H[len(H) - 1 - off]
                    for ic in range(ch):
                        acc += L["conv"][(o * ch + ic) * k + kk] * fr[ic]
                act[o] = activation(a0["act"], acc)
            for o in range(ch):
                a0_head_acc[o] += act[o]
            y = [0.0] * ch
            for o in range(ch):
                s = u[o] + L["oneb"][o]
                for ic in range(ch):
                    s += act[ic] * L["one"][o * ch + ic]
                y[o] = s
            u = y
        a0_out = u
        # array0 head
        a0_head = [0.0] * a0["head"]
        for h in range(a0["head"]):
            s = 0.0
            for c in range(a0["ch"]):
                s += a0_head_acc[c] * a0["head_re"][h * a0["ch"] + c]
            a0_head[h] = s

        # ---- array 1 ----
        u = dense(a1["rechannel"], a0_out, a1["ch"], a1["in_sz"])
        a1_head_acc = [0.0] * a1["ch"]
        for li in range(len(a1["dils"])):
            L = a1["layers"][li]
            hist1[li].append(list(u))
            ch, k, d = a1["ch"], a1["k"], a1["dils"][li]
            H = hist1[li]
            act = [0.0] * ch
            for o in range(ch):
                acc = L["bias"][o]
                for j in range(a1["cond"]):
                    acc += x[t] * L["mix"][o * a1["cond"] + j]
                for kk in range(k):
                    off = d * (k - 1 - kk)
                    fr = H[len(H) - 1 - off]
                    for ic in range(ch):
                        acc += L["conv"][(o * ch + ic) * k + kk] * fr[ic]
                act[o] = activation(a1["act"], acc)
            for o in range(ch):
                if li == 0:
                    a1_head_acc[o] += a0_head[o] + act[o]
                else:
                    a1_head_acc[o] += act[o]
            y = [0.0] * ch
            for o in range(ch):
                s = u[o] + L["oneb"][o]
                for ic in range(ch):
                    s += act[ic] * L["one"][o * ch + ic]
                y[o] = s
            u = y
        head = [0.0] * a1["head"]
        for h in range(a1["head"]):
            s = a1["head_bias"][h] if a1["head_bias"] else 0.0
            for c in range(a1["ch"]):
                s += a1_head_acc[c] * a1["head_re"][h * a1["ch"] + c]
            head[h] = s
        out.append(head[0] * head_scale)
    return out


def main():
    model_path, golden_in, out_path = sys.argv[1:4]
    act = sys.argv[4] if len(sys.argv) > 4 else None
    d = json.load(open(model_path))
    x = load_golden_input(golden_in)
    y = process(d, x, act)
    n = len(x)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", n))
        f.write(struct.pack(f"<{n}f", *x))
        f.write(struct.pack(f"<{n}f", *y))
    print(f"wrote {out_path}: n={n} act={act or 'model'}")


if __name__ == "__main__":
    main()
