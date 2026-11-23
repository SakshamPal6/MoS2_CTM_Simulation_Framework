from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "applications"))
import digit_network as e8
import trained_networks

SEED, LR, BATCH, EPOCHS, NLEV, RESET_AT = 0, 0.05, 32, 3, 50, 40
STOCHASTIC = "--stochastic" in sys.argv
ROUND_RNG = np.random.default_rng(SEED + 1)
E_PULSE, E_RESET = 273.0, 2 * 426.3


@trained_networks.reuse
def hidden_layer(xtr, ytr):
    sel = ytr < 5
    torch.manual_seed(SEED)
    net = torch.nn.Sequential(torch.nn.Linear(784, 128), torch.nn.ReLU(), torch.nn.Linear(128, 5))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    X, Y = torch.from_numpy(xtr[sel]), torch.from_numpy(ytr[sel])
    g = torch.Generator().manual_seed(SEED)
    for _ in range(10):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(X[idx]), Y[idx]).backward()
            opt.step()
    return net[0].weight.detach().numpy().T.copy(), net[0].bias.detach().numpy().copy()


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def learn(H, y, weight_fn, update_fn, state, rng):
    b = np.zeros(10)
    for _ in range(EPOCHS):
        perm = rng.permutation(len(H))
        for i in range(0, len(H), BATCH):
            idx = perm[i:i + BATCH]
            p = softmax(H[idx] @ weight_fn(state) + b)
            p[np.arange(len(idx)), y[idx]] -= 1.0
            dw = -LR * H[idx].T @ p / len(idx)
            b -= LR * p.mean(0)
            update_fn(state, dw)
    return state, b


def main(path):
    t0 = time.time()
    xtr, ytr, xte, yte = e8.load_mnist(path)
    w1, b1 = hidden_layer(xtr, ytr)
    Htr, Hte = np.maximum(xtr @ w1 + b1, 0), np.maximum(xte @ w1 + b1, 0)
    syn = json.loads((ROOT / "results" / "synaptic_response.json").read_text())["pairs"]["-8.0"]
    L = np.r_[syn["G_start_S"], syn["cycles"][0]["potentiation_S"]][:NLEV]
    span = L[-1] - L[0]
    lev = (L - L[0]) / span
    step = 1.0 / (NLEV - 1)
    out = {"label": "MODEL PREDICTION: on-chip learning of the output layer with model synapses", "levels": NLEV, "step_weight_units": step}

    st = {"W": np.zeros((128, 10))}

    def upd_f(s, dw):
        s["W"] += dw
    st, bf = learn(Htr, ytr, lambda s: s["W"], upd_f, st, np.random.default_rng(SEED))
    acc = lambda W, b, H, y: float(np.mean(np.argmax(H @ W + b, 1) == y))
    out["float"] = {"all_digits": acc(st["W"], bf, Hte, yte), "digits_5_9": acc(st["W"], bf, Hte[yte >= 5], yte[yte >= 5]),
                    "digits_0_4": acc(st["W"], bf, Hte[yte < 5], yte[yte < 5])}

    cs = {"np": np.zeros((128, 10), int), "nm": np.zeros((128, 10), int), "pulses": 0, "resets": 0}

    def wfun(s):
        return lev[s["np"]] - lev[s["nm"]]

    def upd_c(s, dw):
        q = np.abs(dw) / step
        n = (np.floor(q) + (ROUND_RNG.random(q.shape) < q - np.floor(q))).astype(int) if STOCHASTIC else np.rint(q).astype(int)
        up, dn = (dw > 0) & (n > 0), (dw < 0) & (n > 0)
        before = s["np"].sum() + s["nm"].sum()
        s["np"] = np.where(up, np.minimum(s["np"] + n, NLEV - 1), s["np"])
        s["nm"] = np.where(dn, np.minimum(s["nm"] + n, NLEV - 1), s["nm"])
        s["pulses"] += int(s["np"].sum() + s["nm"].sum() - before)
        full = (s["np"] >= RESET_AT) & (s["nm"] >= RESET_AT)
        if full.any():
            d = s["np"][full] - s["nm"][full]
            s["np"][full], s["nm"][full] = np.maximum(d, 0), np.maximum(-d, 0)
            s["resets"] += int(full.sum())
            s["pulses"] += int(np.abs(d).sum())
    cs, bc = learn(Htr, ytr, wfun, upd_c, cs, np.random.default_rng(SEED))
    Wc = wfun(cs)
    out["on_chip"] = {"all_digits": acc(Wc, bc, Hte, yte), "digits_5_9": acc(Wc, bc, Hte[yte >= 5], yte[yte >= 5]),
                      "digits_0_4": acc(Wc, bc, Hte[yte < 5], yte[yte < 5]),
                      "pulses": cs["pulses"], "resets": cs["resets"], "cells": 2 * 128 * 10,
                      "energy_nJ": (cs["pulses"] * E_PULSE + cs["resets"] * E_RESET) * 1e-6,
                      "pulse_time_ms_serial": cs["pulses"] * 1e-6 * 1e3,
                      "updates": EPOCHS * int(np.ceil(len(Htr) / BATCH))}
    out["seconds"] = round(time.time() - t0, 1)
    out["rounding"] = "stochastic" if STOCHASTIC else "nearest"
    (ROOT / "results" / ("on_chip_learning.json" if STOCHASTIC else "on_chip_learning_rounded.json")).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.startswith('--')][0])
