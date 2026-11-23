from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "applications"))
sys.argv = sys.argv[:2] + ["--stochastic"]
import digit_network as e8
import digit_network_device_aware as e9
import newton_solver_bratu as e3
import on_chip_learning as e4


def main(path):
    out, arr = {}, {}
    lev = json.loads((ROOT / "results" / "cell_levels.json").read_text())
    G = np.array([L["I_A"]["1"] for L in lev["levels"]]) / lev["read"]["V_D"]
    xtr, ytr, xte, yte = e8.load_mnist(path)

    w1, b1, w2, b2 = e8.train(xtr, ytr)
    scales = [np.abs(w).max() / (G.max() - G.min()) for w in (w1, w2)]
    ft = e9.fine_tune([(w1, b1), (w2, b2)], xtr, ytr, e9.pair_values(G), scales)
    pairs = [e9.map_with_scale(W, G, s) for (W, _), s in zip(ft, scales)]
    Ws = [(G[a] - G[b]) * s for a, b, s in pairs]
    h = np.maximum(xte @ Ws[0] + ft[0][1], 0)
    logits = h @ Ws[1] + ft[1][1]
    pred = np.argmax(logits, 1)
    acc = float(np.mean(pred == yte))
    same_run = json.loads((ROOT / "results" / "digit_network_device_aware.json").read_text())["part1_784_128_10"]
    assert abs(acc - same_run["after_device_aware"]["model_cells_1_s"]) < 1e-4, acc
    cm = np.zeros((10, 10), int)
    for t, p in zip(yte, pred):
        cm[t, p] += 1
    arr.update(G_levels=G, w1_pos=pairs[0][0], w1_neg=pairs[0][1], s1=pairs[0][2], pred=pred, logits=logits,
               x_examples=xte[:48], y_examples=yte[:48], confusion=cm,
               out_currents=(h @ (G[pairs[1][0]] - G[pairs[1][1]]))[:48])
    out["mnist_accuracy"] = acc

    W1h, B1h = e4.hidden_layer(xtr, ytr)
    Htr, Hte = np.maximum(xtr @ W1h + B1h, 0), np.maximum(xte @ W1h + B1h, 0)
    syn = json.loads((ROOT / "results" / "synaptic_response.json").read_text())["pairs"]["-8.0"]
    L = np.r_[syn["G_start_S"], syn["cycles"][0]["potentiation_S"]][:e4.NLEV]
    lev_w = (L - L[0]) / (L[-1] - L[0])
    step = 1.0 / (e4.NLEV - 1)
    s = {"np": np.zeros((128, 10), int), "nm": np.zeros((128, 10), int)}
    b = np.zeros(10)
    rng, curve, k = np.random.default_rng(e4.SEED), [], 0
    new = yte >= 5
    for _ in range(e4.EPOCHS):
        perm = rng.permutation(len(Htr))
        for i in range(0, len(Htr), e4.BATCH):
            idx = perm[i:i + e4.BATCH]
            p = e4.softmax(Htr[idx] @ (lev_w[s["np"]] - lev_w[s["nm"]]) + b)
            p[np.arange(len(idx)), ytr[idx]] -= 1.0
            dw = -e4.LR * Htr[idx].T @ p / len(idx)
            b -= e4.LR * p.mean(0)
            q = np.abs(dw) / step
            n = (np.floor(q) + (e4.ROUND_RNG.random(q.shape) < q - np.floor(q))).astype(int)
            up, dn = (dw > 0) & (n > 0), (dw < 0) & (n > 0)
            s["np"] = np.where(up, np.minimum(s["np"] + n, e4.NLEV - 1), s["np"])
            s["nm"] = np.where(dn, np.minimum(s["nm"] + n, e4.NLEV - 1), s["nm"])
            full = (s["np"] >= e4.RESET_AT) & (s["nm"] >= e4.RESET_AT)
            if full.any():
                d = s["np"][full] - s["nm"][full]
                s["np"][full], s["nm"][full] = np.maximum(d, 0), np.maximum(-d, 0)
            k += 1
            if k % 125 == 0 or k == 1:
                W = lev_w[s["np"]] - lev_w[s["nm"]]
                pr = np.argmax(Hte @ W + b, 1)
                curve.append([k, float(np.mean(pr == yte)), float(np.mean(pr[new] == yte[new]))])
    W = lev_w[s["np"]] - lev_w[s["nm"]]
    final = float(np.mean(np.argmax(Hte @ W + b, 1) == yte))
    assert abs(final - 0.9169) < 1e-4, final
    arr["learning_curve"] = np.array(curve)
    out["learning_final"] = final

    e3.SLICES, e3.VERIFY_TOL = 3, 0.01
    cells = e3.Cells(G, None, None)
    rng = np.random.default_rng(0)
    u = np.zeros(e3.N * e3.N)
    for _ in range(e3.MAX_NEWTON):
        f = e3.F(u, 5.0)
        if np.max(np.abs(f)) < e3.TOL:
            break
        Jx = e3.J(u, 5.0)
        Jh, _ = cells.write(Jx, 0.0, rng)
        dlt = np.zeros_like(u)
        for _ in range(e3.MAX_REF):
            r = -f - Jx @ dlt
            if np.linalg.norm(r) < e3.REFINE_TOL * np.linalg.norm(f):
                break
            dlt += e3.cg(Jh, r, e3.INNER)[0]
        u = u + dlt
    ud = np.zeros(e3.N * e3.N)
    for _ in range(e3.MAX_NEWTON):
        f = e3.F(ud, 5.0)
        if np.max(np.abs(f)) < e3.TOL:
            break
        ud = ud + np.linalg.solve(e3.J(ud, 5.0), -f)
    arr.update(u_cells=u.reshape(e3.N, e3.N), u_digital=ud.reshape(e3.N, e3.N))
    out["solver_max_difference"] = float(np.max(np.abs(u - ud)))
    np.savez(ROOT / "results" / "network_figure_data.npz", **arr)
    (ROOT / "results" / "network_figure_data.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    main(sys.argv[1])
