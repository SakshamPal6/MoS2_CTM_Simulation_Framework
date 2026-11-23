from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT
sys.path.insert(0, str(PKG))
from model import final_model as fm

V_LIST, N, CYCLES, WIDTH, REST, VD = [-7.5, -8.0, -8.5], 50, 3, 1e-6, 1e-3, 1.0


def train(m, r, st, v):
    g = []
    for _ in range(N):
        st = m.hold(m.hold(st, v, WIDTH), 0.0, REST)
        g.append(r.current(0.0, VD, m.dv(st)) / VD)
    return st, np.array(g)


def states(g):
    ch = np.abs(g - g[0])
    tot = abs(g[-1] - g[0])
    return int(np.argmax(ch >= 0.95 * tot)) + 1 if tot > 0 else 0


def nonlin(g):
    if g[-1] == g[0]:
        return float("nan")
    g = (np.asarray(g) - g[0]) / (g[-1] - g[0])
    n = np.arange(1, len(g) + 1)
    g0, g1 = g[0], g[-1]
    f = lambda n, a: g0 + (g1 - g0) * (1 - np.exp(-(n - 1) / a)) / (1 - np.exp(-(len(g) - 1) / a))
    try:
        a, _ = curve_fit(f, n, g, p0=[10.0], bounds=(0.05, 1e4))
        return float(a[0])
    except Exception:
        return float("nan")


def main():
    t0 = time.time()
    z = fm.load_parameters()
    r = fm.readout()
    we = json.loads((PKG / "results" / "write_energy.json").read_text())
    e10 = we["channel_half_CV2_fJ"] + we["gate_line_half_CV2_fJ"]
    out = {"label": "MODEL PREDICTION: one calibrated cell as a synapse",
           "protocol": {"width_s": WIDTH, "rest_s": REST, "pulses_per_train": N, "cycles": CYCLES, "read": [0.0, VD]},
           "pairs": {}}
    for vp in V_LIST:
        m = fm.cell(z)
        st = m.hold(m.hold(m.fresh(), 10.0, fm.ERASE_S), 0.0, 1.0)
        g_start = r.current(0.0, VD, m.dv(st)) / VD
        cyc = []
        for c in range(CYCLES):
            st, gp = train(m, r, st, vp)
            st, gd = train(m, r, st, -vp)
            cyc.append({"potentiation_S": gp.tolist(), "depression_S": gd.tolist()})
        p, d = np.array(cyc[0]["potentiation_S"]), np.array(cyc[0]["depression_S"])
        gall = np.concatenate([[g_start], p, d])
        row = {"G_start_S": g_start, "cycles": cyc,
               "G_max_S": float(p.max()), "G_min_S": float(min(g_start, d.min())),
               "dynamic_range": float(p.max() / max(min(g_start, d.min()), 1e-15)),
               "states_potentiation": states(np.concatenate([[g_start], p])),
               "states_depression": states(np.concatenate([[p[-1]], d])),
               "nonlinearity_A_potentiation": nonlin(np.concatenate([[g_start], p])),
               "nonlinearity_A_depression": nonlin(np.concatenate([[p[-1]], d])),
               "Gmax_per_cycle_S": [float(np.max(c["potentiation_S"])) for c in cyc],
               "Gmin_end_of_depression_per_cycle_S": [float(c["depression_S"][-1]) for c in cyc],
               "energy_per_update_fJ": e10 * (abs(vp) / 10.0) ** 2, "time_per_update_s": WIDTH}
        row["asymmetry_states_P_over_D"] = row["states_potentiation"] / max(row["states_depression"], 1)
        out["pairs"]["%.1f" % vp] = row
        print("V_P %.1f V: range %.3g, states P %d D %d, A_P %.2f A_D %.2f, Gmax/cycle %s, E %.0f fJ" % (
            vp, row["dynamic_range"], row["states_potentiation"], row["states_depression"],
            row["nonlinearity_A_potentiation"], row["nonlinearity_A_depression"],
            ["%.3g" % x for x in row["Gmax_per_cycle_S"]], row["energy_per_update_fJ"]), flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    (ROOT / "results" / "synaptic_response.json").write_text(json.dumps(out, indent=1))
    print("done in %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
