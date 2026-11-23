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
import trained_networks

SEED, TIMES = 0, ["1", "601"]


def pair_values(g):
    return np.unique((g[:, None] - g[None, :]).ravel())


def snap_np(t, vals):
    k = np.clip(np.searchsorted(vals, t), 1, len(vals) - 1)
    return np.where(np.abs(vals[k - 1] - t) <= np.abs(vals[k] - t), vals[k - 1], vals[k])


class Snap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, vals):
        k = torch.clamp(torch.searchsorted(vals, w.contiguous()), 1, len(vals) - 1)
        lo, hi = vals[k - 1], vals[k]
        return torch.where((w - lo).abs() <= (hi - w).abs(), lo, hi)

    @staticmethod
    def backward(ctx, g):
        return g, None


def map_with_scale(w, g, s):
    a_i, b_i = np.meshgrid(np.arange(len(g)), np.arange(len(g)), indexing="ij")
    diff = (g[a_i] - g[b_i]).ravel()
    order = np.argsort(diff)
    ds, t = diff[order], (w / s).ravel()
    k = np.clip(np.searchsorted(ds, t), 1, len(ds) - 1)
    k = np.where(np.abs(ds[k - 1] - t) <= np.abs(ds[k] - t), k - 1, k)
    pick = order[k]
    return a_i.ravel()[pick].reshape(w.shape), b_i.ravel()[pick].reshape(w.shape), s


@trained_networks.reuse
def fine_tune(layers, x, y, vals, scales, epochs=5):
    torch.manual_seed(SEED)
    P = [(torch.tensor(W, requires_grad=True), torch.tensor(b, requires_grad=True)) for W, b in layers]
    V = torch.tensor(vals, dtype=torch.float32)
    opt = torch.optim.Adam([p for wb in P for p in wb], lr=1e-3)
    X, Y = torch.from_numpy(x), torch.from_numpy(y)
    gen = torch.Generator().manual_seed(SEED)
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=gen)
        for i in range(0, len(X), 128):
            idx = perm[i:i + 128]
            h = X[idx]
            for j, ((W, b), s) in enumerate(zip(P, scales)):
                h = h @ (Snap.apply(W / s, V) * s) + b
                if j < len(P) - 1:
                    h = torch.relu(h)
            opt.zero_grad()
            torch.nn.functional.cross_entropy(h, Y[idx]).backward()
            opt.step()
    return [(W.detach().numpy().copy(), b.detach().numpy().copy()) for W, b in P]


def forward_acc(x, y, Ws, bs):
    h = x
    for j, (W, b) in enumerate(zip(Ws, bs)):
        h = h @ W + b
        if j < len(Ws) - 1:
            h = np.maximum(h, 0.0)
    return float(np.mean(np.argmax(h, 1) == y))


def evaluate(x, y, layers, G, dv1, i_of, scales):
    pairs = [map_with_scale(W, G["1"], s) for (W, _), s in zip(layers, scales)]
    bs = [b for _, b in layers]
    out = {}
    for t in TIMES:
        out["model_cells_%s_s" % t] = forward_acc(x, y, [(G[t][a] - G[t][b]) * s for a, b, s in pairs], bs)
    rng = np.random.default_rng(SEED)
    for sg in e8.SIGMAS:
        acc = []
        for _ in range(e8.DRAWS):
            Ws = [(i_of(dv1[a] + rng.normal(0, sg, a.shape)) - i_of(dv1[b] + rng.normal(0, sg, b.shape))) * s
                  for a, b, s in pairs]
            acc.append(forward_acc(x, y, Ws, bs))
        out["spread_%.2f_V" % sg] = {"mean": float(np.mean(acc)), "min": float(np.min(acc)), "max": float(np.max(acc))}
    return out


def small_binary(x):
    imgs = (x.reshape(-1, 28, 28) * 255).astype(np.uint8)
    out = np.stack([np.asarray(Image.fromarray(im).resize((13, 13), Image.BICUBIC), np.float32) / 255.0 for im in imgs])
    return (out.reshape(len(x), -1) >= 0.5).astype(np.float32)


@trained_networks.reuse
def train_single(x, y):
    torch.manual_seed(SEED)
    lin = torch.nn.Linear(x.shape[1], 10)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-3)
    X, Y = torch.from_numpy(x), torch.from_numpy(y)
    gen = torch.Generator().manual_seed(SEED)
    for _ in range(10):
        perm = torch.randperm(len(X), generator=gen)
        for i in range(0, len(X), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(lin(X[idx]), Y[idx]).backward()
            opt.step()
    return lin.weight.detach().numpy().T.copy(), lin.bias.detach().numpy().copy()


def main(path):
    t0 = time.time()
    lev = json.loads((ROOT / "results" / "cell_levels.json").read_text())
    vd = lev["read"]["V_D"]
    G = {t: np.array([L["I_A"][t] for L in lev["levels"]]) / vd for t in TIMES}
    dv1 = np.array([L["dV_V"] for L in lev["levels"]])
    r = e8.fm.readout()
    grid = np.linspace(dv1.min() - 1.0, dv1.max() + 1.0, 801)
    lg = np.log10([r.current(0.0, vd, z) for z in grid])
    i_of = lambda dv: 10 ** np.interp(dv, grid, lg) / vd
    vals = pair_values(G["1"])
    xtr, ytr, xte, yte = e8.load_mnist(path)
    out = {"label": "MODEL PREDICTION: device-aware training on the calibrated cell levels"}

    w1, b1, w2, b2 = e8.train(xtr, ytr)
    scales = [np.abs(w).max() / (G["1"].max() - G["1"].min()) for w in (w1, w2)]
    p1 = {"float": forward_acc(xte, yte, [w1, w2], [b1, b2])}
    p1["before_device_aware"] = evaluate(xte, yte, [(w1, b1), (w2, b2)], G, dv1, i_of, scales)
    ft = fine_tune([(w1, b1), (w2, b2)], xtr, ytr, vals, scales)
    p1["after_device_aware"] = evaluate(xte, yte, ft, G, dv1, i_of, scales)
    out["part1_784_128_10"] = p1
    print(json.dumps(p1, indent=1), flush=True)

    s_tr, s_te = small_binary(xtr), small_binary(xte)
    w, b = train_single(s_tr, ytr)
    sc = [np.abs(w).max() / (G["1"].max() - G["1"].min())]
    p2 = {"float": forward_acc(s_te, yte, [w], [b])}
    p2["before_device_aware"] = evaluate(s_te, yte, [(w, b)], G, dv1, i_of, sc)
    ft2 = fine_tune([(w, b)], s_tr, ytr, vals, sc, epochs=5)
    p2["after_device_aware"] = evaluate(s_te, yte, ft2, G, dv1, i_of, sc)
    p2["reported_for_measured_crossbar"] = {"simulated": 0.881, "hardware": 0.856, "test_images": 1000,
                                            "weights": "binary", "source": "base paper Fig. 3"}
    out["part2_169_10_binarized_13x13"] = p2
    print(json.dumps(p2, indent=1), flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    (ROOT / "results" / "digit_network_device_aware.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
