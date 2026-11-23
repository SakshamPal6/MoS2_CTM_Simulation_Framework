from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT
sys.path.insert(0, str(PKG))
from model import final_model as fm
import trained_networks

SIGMAS, DRAWS, SEED = [0.05, 0.1, 0.2], 10, 0
TIMES = ["1", "11", "101", "601"]


def load_mnist(path):
    d = np.load(path, allow_pickle=False)
    f = lambda x: (x.reshape(len(x), -1).astype(np.float32) / 255.0)
    return f(d["x_train"]), d["y_train"].astype(np.int64), f(d["x_test"]), d["y_test"].astype(np.int64)


@trained_networks.reuse
def train(xtr, ytr):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    net = torch.nn.Sequential(torch.nn.Linear(784, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    X, Y = torch.from_numpy(xtr), torch.from_numpy(ytr)
    g = torch.Generator().manual_seed(SEED)
    for ep in range(10):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(net(X[idx]), Y[idx])
            loss.backward()
            opt.step()
    w1, b1 = net[0].weight.detach().numpy().T.copy(), net[0].bias.detach().numpy().copy()
    w2, b2 = net[2].weight.detach().numpy().T.copy(), net[2].bias.detach().numpy().copy()
    return w1, b1, w2, b2


def accuracy(x, y, w1, b1, w2, b2):
    h = np.maximum(x @ w1 + b1, 0.0)
    return float(np.mean(np.argmax(h @ w2 + b2, 1) == y))


def map_pairs(w, g):
    s = np.abs(w).max() / (g.max() - g.min())
    a_i, b_i = np.meshgrid(np.arange(len(g)), np.arange(len(g)), indexing="ij")
    diff = (g[a_i] - g[b_i]).ravel()
    order = np.argsort(diff)
    ds = diff[order]
    t = (w / s).ravel()
    k = np.clip(np.searchsorted(ds, t), 1, len(ds) - 1)
    k = np.where(np.abs(ds[k - 1] - t) <= np.abs(ds[k] - t), k - 1, k)
    pick = order[k]
    return a_i.ravel()[pick].reshape(w.shape), b_i.ravel()[pick].reshape(w.shape), s


def main(mnist_path):
    t0 = time.time()
    lev = json.loads((ROOT / "results" / "cell_levels.json").read_text())
    vd = lev["read"]["V_D"]
    G = {t: np.array([L["I_A"][t] for L in lev["levels"]]) / vd for t in TIMES}
    dv1 = np.array([L["dV_V"] for L in lev["levels"]])
    vprog = [L["v_prog_V"] for L in lev["levels"]]
    r = fm.readout()
    grid = np.linspace(dv1.min() - 1.0, dv1.max() + 1.0, 801)
    lg = np.log10([r.current(0.0, vd, x) for x in grid])
    i_of = lambda dv: 10 ** np.interp(dv, grid, lg) / vd

    xtr, ytr, xte, yte = load_mnist(mnist_path)
    w1, b1, w2, b2 = train(xtr, ytr)
    out = {"label": "MODEL PREDICTION: MNIST on an ideal crossbar of calibrated model cells",
           "network": "784-128-10 ReLU, float32, Adam 10 epochs, batch 128, seed 0", "n_test": int(len(yte))}
    out["A_float"] = accuracy(xte, yte, w1, b1, w2, b2)
    p1, p2 = map_pairs(w1, G["1"]), map_pairs(w2, G["1"])
    weff = lambda p, g: (g[p[0]] - g[p[1]]) * p[2]
    out["B_levels_1s"] = accuracy(xte, yte, weff(p1, G["1"]), b1, weff(p2, G["1"]), b2)
    out["C_retention"] = {"%s_s_after_programming" % t: accuracy(xte, yte, weff(p1, G[t]), b1, weff(p2, G[t]), b2)
                          for t in TIMES[1:]}
    rng = np.random.default_rng(SEED)
    D = {}
    for sg in SIGMAS:
        acc = []
        for _ in range(DRAWS):
            ws = []
            for (a, b, s) in (p1, p2):
                ga = i_of(dv1[a] + rng.normal(0.0, sg, a.shape))
                gb = i_of(dv1[b] + rng.normal(0.0, sg, b.shape))
                ws.append((ga - gb) * s)
            acc.append(accuracy(xte, yte, ws[0], b1, ws[1], b2))
        D["sigma_%.2f_V" % sg] = {"mean": float(np.mean(acc)), "min": float(np.min(acc)), "max": float(np.max(acc))}
    out["D_threshold_spread"] = D

    we = json.loads((PKG / "results" / "write_energy.json").read_text())
    e10 = we["channel_half_CV2_fJ"] + we["gate_line_half_CV2_fJ"]
    cells = np.concatenate([np.ravel(p1[0]), np.ravel(p1[1]), np.ravel(p2[0]), np.ravel(p2[1])])
    v_of = np.array([0.0 if v is None else v for v in vprog])
    programmed = cells[cells > 0]
    e_prog = float(np.sum(e10 * (v_of[programmed] / 10.0) ** 2))
    e_erase = float(len(cells) * e10)
    rows = w1.shape[0] + w2.shape[0]
    levels_used = len(set(programmed.tolist()))
    out["cost"] = {"cells": int(len(cells)), "programmed_cells": int(len(programmed)),
                   "levels_used": levels_used, "energy_program_nJ": e_prog * 1e-6, "energy_erase_nJ": e_erase * 1e-6,
                   "energy_per_cell_fJ": (e_prog + e_erase) / len(cells),
                   "time_serial_ms": len(programmed) * 1e-6 * 1e3,
                   "time_row_parallel_ms": rows * levels_used * 1e-6 * 1e3,
                   "erase_once_ms": fm.ERASE_S * 1e3,
                   "assumptions": "energy = (channel + gate line) 1/2 C V^2 per pulse, scaled by (V/10 V)^2; row-parallel: all "
                                  "cells of a row that share a level programmed by one pulse; erase is one block pulse"}
    out["levels_G_S"] = {t: G[t].tolist() for t in TIMES}
    out["seconds"] = round(time.time() - t0, 1)
    (ROOT / "results" / "digit_network.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "levels_G_S"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
