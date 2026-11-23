from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "applications"))
import digit_network as e8
import digit_network_device_aware as e9

V_READ, T_READ = (1.0, 0.5, 0.25), (1e-6, 1e-7)


def main(path):
    t0 = time.time()
    lev = json.loads((ROOT / "results" / "cell_levels.json").read_text())
    G = np.array([L["I_A"]["1"] for L in lev["levels"]]) / lev["read"]["V_D"]
    xtr, ytr, xte, yte = e8.load_mnist(path)
    w1, b1, w2, b2 = e8.train(xtr, ytr)
    scales = [np.abs(w).max() / (G.max() - G.min()) for w in (w1, w2)]
    ft = e9.fine_tune([(w1, b1), (w2, b2)], xtr, ytr, e9.pair_values(G), scales)
    pairs = [e9.map_with_scale(W, G, s) for (W, _), s in zip(ft, scales)]
    gsum = [G[a] + G[b] for a, b, _ in pairs]
    W1 = (G[pairs[0][0]] - G[pairs[0][1]]) * pairs[0][2]
    h = np.maximum(xte @ W1 + ft[0][1], 0.0)
    hn = h / h.max()
    acc = e9.forward_acc(xte, yte, [(G[a] - G[b]) * s for a, b, s in pairs], [b for _, b in ft])
    macs = 784 * 128 + 128 * 10
    c188, c194 = json.loads((ROOT / "results" / "digit_network.json").read_text())["cost"], \
        json.loads((ROOT / "results" / "on_chip_learning.json").read_text())["on_chip"]
    write_nj = c188["energy_program_nJ"] + c188["energy_erase_nJ"]
    update_fj = c194["energy_nJ"] * 1e6 / c194["updates"]
    out = {"label": "MODEL PREDICTION: cell energy of one MNIST inference in place, read-time encoding; peripheral circuits excluded", "accuracy_check": acc,
           "macs_per_inference": macs, "cases": {}, "accuracy_at_V_D": {}, "current_ratio_to_1V": {}}
    r = e8.fm.readout()
    dvs = [L["dV_V"] for L in lev["levels"]]
    i1 = np.array([r.current(0.0, 1.0, d) for d in dvs])
    assert np.allclose(i1, G * lev["read"]["V_D"], rtol=1e-9), "levels and readout disagree"
    for v in V_READ:
        iv = np.array([r.current(0.0, v, d) for d in dvs])
        isum = [iv[a] + iv[b] for a, b, _ in pairs]
        gv = iv / i1 * G
        out["current_ratio_to_1V"]["V%.2f" % v] = (iv / i1).tolist()
        out["accuracy_at_V_D"]["V%.2f" % v] = e9.forward_acc(xte, yte, [(gv[a] - gv[b]) * s for a, b, s in pairs],
                                                             [b for _, b in ft])
        for t in T_READ:
            e1 = np.sum(xte @ isum[0], axis=1) * v * t
            e2 = np.sum(hn @ isum[1], axis=1) * v * t
            e = e1 + e2
            out["cases"]["V%.2f_t%.0e" % (v, t)] = {
                "energy_per_inference_fJ": float(np.mean(e) * 1e15),
                "energy_per_MAC_aJ": float(np.mean(e) / macs * 1e18),
                "layer1_share": float(np.mean(e1) / np.mean(e)),
                "inferences_per_network_write": float(write_nj * 1e-9 / np.mean(e)),
                "inferences_per_learning_update": float(update_fj * 1e-15 / np.mean(e))}
    out["network_write_nJ"] = write_nj
    out["learning_update_fJ"] = update_fj
    out["seconds"] = round(time.time() - t0, 1)
    (ROOT / "results" / "read_energy.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
