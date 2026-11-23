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

fm = e8.fm
TEMPS_C = [25, 50, 85, 125]
V_PROG = [round(v, 1) for v in np.arange(-4.0, -10.01, -0.5)]
WIDTH, REST, HOLD, VD = 1e-6, 1.0, 600.0, 1.0


def readout_at(T):
    ro = fm._json("readout_fit.json")
    from model.readout import Readout
    cox = fm.EPS0 / sum(t / k for t, k in fm.DEV["base"]["stack"])
    return Readout(ro["vth"], ro["ss"] * T / 300.0, ro["mu"], cox, 1.0, ro["r_s"], 1e-12, 1.0)


def states(m):
    er = m.hold(m.hold(m.fresh(), 10.0, fm.ERASE_S), 0.0, REST)
    return [er] + [m.hold(m.hold(er, v, WIDTH), 0.0, REST) for v in V_PROG]


def main(path):
    t0 = time.time()
    z = fm.load_parameters()
    lev = json.loads((ROOT / "results" / "cell_levels.json").read_text())
    G1 = np.array([L["I_A"]["1"] for L in lev["levels"]]) / VD
    xtr, ytr, xte, yte = e8.load_mnist(path)
    w1, b1, w2, b2 = e8.train(xtr, ytr)
    scales = [np.abs(w).max() / (G1.max() - G1.min()) for w in (w1, w2)]
    vals = e9.pair_values(G1)
    ft = e9.fine_tune([(w1, b1), (w2, b2)], xtr, ytr, vals, scales)
    pairs = [e9.map_with_scale(W, G1, s) for (W, _), s in zip(ft, scales)]
    bs = [b for _, b in ft]
    acc = lambda G: e9.forward_acc(xte, yte, [(G[a] - G[b]) * s for a, b, s in pairs], bs)
    out = {"label": "MODEL PREDICTION: accuracy of the device-aware network at high temperature",
           "check_25C_mapping_equals_device_aware": acc(G1)}
    m300 = fm.cell(z, T=300.0)
    st300 = states(m300)
    for tc in TEMPS_C:
        T = tc + 273.15
        mT, rT = fm.cell(z, T=T), readout_at(T)
        GA = np.array([rT.current(0.0, VD, mT.dv(mT.hold(s, 0.0, HOLD))) for s in st300]) / VD
        GB = np.array([rT.current(0.0, VD, mT.dv(s)) for s in states(mT)]) / VD
        out["T_%dC" % tc] = {"A_programmed_25C_held_10min_read_hot": acc(GA), "B_programmed_and_read_hot": acc(GB),
                             "levels_A_S": GA.tolist(), "levels_B_S": GB.tolist()}
        print(tc, "C: A %.4f  B %.4f" % (out["T_%dC" % tc]["A_programmed_25C_held_10min_read_hot"],
                                         out["T_%dC" % tc]["B_programmed_and_read_hot"]), flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    (ROOT / "results" / "network_temperature.json").write_text(json.dumps(out, indent=1))
    print("check 25 C mapping:", out["check_25C_mapping_equals_device_aware"])


if __name__ == "__main__":
    main(sys.argv[1])
