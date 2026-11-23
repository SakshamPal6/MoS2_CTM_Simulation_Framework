from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import brentq, curve_fit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "applications"))
import digit_network as e8
import digit_network_device_aware as e9
import network_temperature as e5

fm = e8.fm
YEAR = 3.156e7
TIMES = [1.0, 10.0, 100.0, 600.0, 3600.0, 86400.0, 7 * 86400.0, 30 * 86400.0, YEAR]
V_SAFE, LOG_OFF = (-3.0, 2.5), math.log10(2e-12)


def slow(t, a, tau):
    return -a * np.log(1.0 + t / tau) ** 2


def main(path):
    t0 = time.time()
    p172 = json.loads((ROOT / "results" / "retention_projection.json").read_text())
    lt_m, m_y = np.log10(p172["t_s"]), np.array(p172["T300"]["dlogI_from_5s"])
    slope = 1.0 / json.loads((ROOT / "data" / "measured" /
                              "readout_fit.json").read_text())["ss"]
    model = lambda t: np.interp(np.log10(t), lt_m, m_y)
    runs = yaml.safe_load((ROOT / "data" / "measured" / "base_si12_traces.yaml").read_text(encoding="utf-8"))["runs"]
    data = {}
    for k, r in runs.items():
        t, y = np.array(r["t_s"], float), np.array(r["log10_I_A"], float)
        data[k] = (t, y, (y - y[0]) - (model(t) - model(t[0])))

    def fit(keys):
        T = np.concatenate([data[k][0] for k in keys])
        T0 = np.concatenate([np.full(len(data[k][0]), data[k][0][0]) for k in keys])
        Rs = np.concatenate([data[k][2] for k in keys])
        f = lambda X, a, tau: slow(X[0], a, tau) - slow(X[1], a, tau)
        p, c = curve_fit(f, np.vstack([T, T0]), Rs, p0=[0.01, 10.0], bounds=([0, 1e-2], [10, 1e5]))
        return p, c, float(np.sqrt(np.mean((f(np.vstack([T, T0]), *p) - Rs) ** 2)))

    y_start, t_start = data["si12_1h"][1][0], data["si12_1h"][0][0]
    tgrid = np.logspace(np.log10(t_start), np.log10(10 * YEAR), 400)

    def proj(a, tau):
        return y_start + (model(tgrid) - model(t_start)) + slow(tgrid, a, tau) - slow(t_start, a, tau)

    def hold(a, tau):
        y = proj(a, tau)
        below = np.nonzero(y <= LOG_OFF)[0]
        return float(tgrid[below[0]]) if len(below) else float("inf")

    out = {"label": "MODEL PREDICTION with a calibrated slow-loss term", "fits": {}}
    for name, keys in (("both_runs", ["si12_10min", "si12_1h"]), ("10min_run", ["si12_10min"]), ("1h_run", ["si12_1h"])):
        p, c, rms = fit(keys)
        out["fits"][name] = {"a_dec": float(p[0]), "tau_s": float(p[1]), "rms_dec": rms, "hold_time_s": hold(*p)}
    p, c, _ = fit(["si12_10min", "si12_1h"])
    rng = np.random.default_rng(0)
    draws = rng.multivariate_normal(p, c, size=400)
    hd = [hold(max(a, 0), max(tau, 1e-2)) for a, tau in draws]
    out["hold_time_95pc_s"] = [float(np.percentile(hd, 2.5)), float(np.percentile(hd, 97.5))]
    out["projection"] = {"t_s": tgrid.tolist(), "log10_I_model_only": (y_start + model(tgrid) - model(t_start)).tolist(),
                         "log10_I_model_plus_slow": proj(*p).tolist(), "off_level_log10_A": LOG_OFF}
    print("fits", json.dumps(out["fits"], indent=1), "95 %", out["hold_time_95pc_s"], flush=True)

    a_f, tau_f = p
    dvslow = lambda t: -(slow(t, a_f, tau_f) - slow(1.0, a_f, tau_f)) / slope
    z = fm.load_parameters()
    m, r = fm.cell(z), fm.readout()
    st = e5.states(m)
    lev = json.loads((ROOT / "results" / "cell_levels.json").read_text())
    G1 = np.array([L["I_A"]["1"] for L in lev["levels"]]) / 1.0
    xtr, ytr, xte, yte = e8.load_mnist(path)
    w1, b1, w2, b2 = e8.train(xtr, ytr)
    scales = [np.abs(w).max() / (G1.max() - G1.min()) for w in (w1, w2)]
    ft = e9.fine_tune([(w1, b1), (w2, b2)], xtr, ytr, e9.pair_values(G1), scales)
    pairs = [e9.map_with_scale(W, G1, s) for (W, _), s in zip(ft, scales)]
    acc = lambda G: e9.forward_acc(xte, yte, [(G[a] - G[b]) * s for a, b, s in pairs], [b for _, b in ft])
    ref = len(st) - 1
    states_t, prev = list(st), 1.0
    rows = {}
    i_ref0 = r.current(0.0, 1.0, m.dv(st[ref]))
    for t in TIMES:
        states_t = [m.hold(s, 0.0, t - prev) for s in states_t] if t > prev else states_t
        prev = max(prev, t)
        dv = np.array([m.dv(s) for s in states_t])
        dv[1:] = dv[1:] + dvslow(t)
        G0 = np.array([r.current(0.0, 1.0, x) for x in dv])
        f = lambda vg: math.log10(r.current(vg, 1.0, dv[ref])) - math.log10(i_ref0)
        lo, hi = V_SAFE
        vg = lo if f(lo) > 0 else hi if f(hi) < 0 else brentq(f, lo, hi)
        Gc = np.array([r.current(vg, 1.0, x) for x in dv])
        rows["%g" % t] = {"accuracy_no_compensation": acc(G0), "accuracy_compensated": acc(Gc), "read_gate_V": float(vg),
                          "slow_shift_V": float(dvslow(t))}
        print("t %10.0f s: %.4f -> %.4f at V_RG %+.2f V" % (t, rows["%g" % t]["accuracy_no_compensation"],
                                                         rows["%g" % t]["accuracy_compensated"], vg), flush=True)
    out["network"] = rows
    for key in ("accuracy_no_compensation", "accuracy_compensated"):
        ok = [float(t) for t in rows if rows[t][key] >= 0.95]
        out["refresh_interval_s_" + key.split("_", 1)[1]] = max(ok) if ok else None
    out["seconds"] = round(time.time() - t0, 1)
    (ROOT / "results" / "network_retention.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
