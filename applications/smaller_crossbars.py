from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "applications"))
import digit_network as e8
import digit_network_device_aware as e9
import trained_networks

SEED = 0


def resize14(x):
    imgs = (x.reshape(-1, 28, 28) * 255).astype(np.uint8)
    return np.stack([np.asarray(Image.fromarray(im).resize((14, 14), Image.BICUBIC), np.float32) / 255.0
                     for im in imgs]).reshape(len(x), -1)


@trained_networks.reuse
def train(x, y, hidden=None):
    torch.manual_seed(SEED)
    net = torch.nn.Linear(x.shape[1], 10) if hidden is None else torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 10))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    X, Y = torch.from_numpy(x), torch.from_numpy(y)
    gen = torch.Generator().manual_seed(SEED)
    for _ in range(10):
        perm = torch.randperm(len(X), generator=gen)
        for i in range(0, len(X), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(X[idx]), Y[idx]).backward()
            opt.step()
    lins = [net] if hidden is None else [net[0], net[2]]
    return [(l.weight.detach().numpy().T.copy(), l.bias.detach().numpy().copy()) for l in lins]


def energy(x, layers, G, scales):
    pairs = [e9.map_with_scale(W, G, s) for (W, _), s in zip(layers, scales)]
    gsum = [G[a] + G[b] for a, b, _ in pairs]
    h, e = x, np.zeros(len(x))
    for j, ((W, b), (a, bb, s)) in enumerate(zip(layers, pairs)):
        e += np.sum(h @ gsum[j], axis=1) * 1e-6
        h = h @ ((G[a] - G[bb]) * s) + b
        if j < len(layers) - 1:
            h = np.maximum(h, 0.0)
            h = h / max(h.max(), 1e-30)
    return float(np.mean(e)), pairs


def main(path):
    t0 = time.time()
    lev = json.loads((ROOT / "results" / "cell_levels.json").read_text())
    vd = lev["read"]["V_D"]
    G = {t: np.array([L["I_A"][t] for L in lev["levels"]]) / vd for t in e9.TIMES}
    dv1 = np.array([L["dV_V"] for L in lev["levels"]])
    r = e8.fm.readout()
    grid = np.linspace(dv1.min() - 1.0, dv1.max() + 1.0, 801)
    lg = np.log10([r.current(0.0, vd, z) for z in grid])
    i_of = lambda dv: 10 ** np.interp(dv, grid, lg) / vd
    vals = e9.pair_values(G["1"])
    xtr, ytr, xte, yte = e8.load_mnist(path)
    x14tr, x14te = resize14(xtr), resize14(xte)
    out = {"label": "MODEL PREDICTION: MNIST on multi-level differential-pair crossbars of model cells "
                    ""}
    figdata = {}
    for name, (tr, te, hidden) in {"A_14x14_single_196x20": (x14tr, x14te, None),
                                   "B_28x28_single_784x20": (xtr, xte, None),
                                   "C_14x14_two_layer_196x64_32x20": (x14tr, x14te, 32)}.items():
        layers = train(tr, ytr, hidden)
        scales = [np.abs(W).max() / (G["1"].max() - G["1"].min()) for W, _ in layers]
        res = {"float": e9.forward_acc(te, yte, [W for W, _ in layers], [b for _, b in layers]),
               "cells": int(sum(2 * W.size for W, _ in layers)),
               "crossbars": [[W.shape[0], 2 * W.shape[1]] for W, _ in layers]}
        res["before_device_aware"] = e9.evaluate(te, yte, layers, G, dv1, i_of, scales)
        ft = e9.fine_tune(layers, tr, ytr, vals, scales)
        res["after_device_aware"] = e9.evaluate(te, yte, ft, G, dv1, i_of, scales)
        e_j, pairs = energy(te, ft, G["1"], scales)
        res["cell_energy_per_digit_pJ_1V_1us"] = e_j * 1e12
        h = te
        for j, ((W, b), (a, bb, s)) in enumerate(zip(ft, pairs)):
            h = h @ ((G["1"][a] - G["1"][bb]) * s) + b
            if j < len(ft) - 1:
                h = np.maximum(h, 0.0)
        pred = np.argmax(h, 1)
        figdata[name + "_confusion"] = np.array([[int(np.sum((yte == t) & (pred == q))) for q in range(10)] for t in range(10)])
        out[name] = res
        ad = res["after_device_aware"]
        print("%s: float %.4f | cells %.4f (10 min %.4f) | spread 0.05/0.10/0.20 V %.4f/%.4f/%.4f | %d cells | %.2f pJ/digit"
              % (name, res["float"], ad["model_cells_1_s"], ad["model_cells_601_s"], ad["spread_0.05_V"]["mean"],
                 ad["spread_0.10_V"]["mean"], ad["spread_0.20_V"]["mean"], res["cells"], res["cell_energy_per_digit_pJ_1V_1us"]),
              flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    (ROOT / "results" / "smaller_crossbars.json").write_text(json.dumps(out, indent=1))
    np.savez(ROOT / "results" / "smaller_crossbars_figure_data.npz", **figdata)


if __name__ == "__main__":
    main(sys.argv[1])
