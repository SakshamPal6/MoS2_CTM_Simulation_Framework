from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import curve_fit
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import final_model as fm

YEAR = 3.156e7
T_END = 10 * YEAR
LOG_2PA = math.log10(2e-12)
TOL = 0.15
SI12 = yaml.safe_load((ROOT / "data" / "measured" / "base_si12_retention.yaml").read_text(encoding="utf-8"))["runs"]
MARKS = np.unique(np.round(np.concatenate([np.logspace(np.log10(5.0), np.log10(T_END), 141), [3600.0]] +
                                          [[r["t_first_s"], r["t_end_s"], r["t_end_s"] / 10] for r in SI12.values()]), 6))


def trace(m, first, second):
    st = m.hold(m.hold(m.fresh(), first, 0.1), second, 0.1)
    out, tp = [], 0.0
    for t in MARKS:
        st = m.hold(st, 0.0, t - tp)
        out.append(m.dv(st))
        tp = t
    return np.array(out)


def at(y, t):
    return float(np.interp(np.log10(t), np.log10(MARKS), y))


def crossing(y, level):
    i = np.where(y <= level)[0]
    if not len(i):
        return float("inf")
    k = i[0]
    if k == 0:
        return float(MARKS[0])
    u = np.log10(MARKS[k - 1:k + 1])
    return float(10 ** np.interp(level, [y[k], y[k - 1]], [u[1], u[0]]))


def outputs(z, slope, T):
    m = fm.cell(z, "base", T)
    on, off = trace(m, 10.0, -10.0), trace(m, -10.0, 10.0)
    y = -(on - at(on, 5.0)) * slope
    win = off - on
    res = {"dlogI_from_5s": y.tolist(), "window_V": win.tolist()}
    for name, r in SI12.items():
        a = lambda tt: -(at(on, tt)) * slope
        res[name] = {"model_first_to_end": a(r["t_end_s"]) - a(r["t_first_s"]), "model_last_decade": a(r["t_end_s"]) - a(r["t_end_s"] / 10),
                     "t_2pA_s": crossing(r["logI_first"] + (-(on - at(on, r["t_first_s"])) * slope), LOG_2PA)}
    w0 = at(win, 5.0)
    res["t_window_half_s"] = crossing(np.abs(win) / abs(w0), 0.5)
    res["window_5s_V"] = w0
    return res


def laws(y):
    u, u5 = np.log10(MARKS), math.log10(5.0)
    forms = {"power_law": (lambda u, a, b: a + b * (u - u5), [0.0, -0.3]),
             "quadratic_in_log_t": (lambda u, a, b, c: a + b * (u - u5) + c * (u - u5) ** 2, [0.0, -0.3, 0.0]),
             "gaussian_band_of_emission_times": (lambda u, A, u0, s: -A * (norm.cdf((u - u0) / s) - norm.cdf((u5 - u0) / s)), [3.0, 4.0, 2.0])}
    out, short = {}, MARKS <= 3600.0 * 1.0001
    for k, (f, p0) in forms.items():
        row = {}
        try:
            p, _ = curve_fit(f, u, y, p0=p0, maxfev=20000)
            row["rms_full_span_dec"] = float(np.sqrt(np.mean((f(u, *p) - y) ** 2)))
            row["params_full_span"] = [float(q) for q in p]
        except RuntimeError as ex:
            row["rms_full_span_dec"], row["fit_error_full"] = None, str(ex)[:100]
        try:
            q, _ = curve_fit(f, u[short], y[short], p0=p0, maxfev=20000)
            row["error_at_10y_when_fitted_to_1h_dec"] = float(f(math.log10(T_END), *q) - y[-1])
        except RuntimeError as ex:
            row["error_at_10y_when_fitted_to_1h_dec"], row["fit_error_1h"] = None, str(ex)[:100]
        out[k] = row
    ok = {k: v["rms_full_span_dec"] for k, v in out.items() if v.get("rms_full_span_dec") is not None}
    return out, min(ok, key=ok.get)


def covariance(z):
    lo, hi, s = fm.LO, fm.HI, fm.SCALE
    path = ROOT / "results" / "fits" / "full_covariance.json"
    if path.exists():
        d = json.loads(path.read_text())
        return np.array(d["cov"]), d["free"], lo, hi
    inp, r = fm.inputs(), fm.readout()
    r0 = fm.residuals(z, inp, r)
    J = np.zeros((r0.size, z.size))
    h = np.maximum(1e-3 * np.abs(z), 1e-3) * np.where(z + 1e-3 * np.abs(z) + 1e-3 > hi, -1.0, 1.0)
    for k in range(z.size):
        q = z.copy()
        q[k] += h[k]
        J[:, k] = (fm.residuals(q, inp, r) - r0) / h[k]
    free = [k for k in range(z.size) if z[k] - lo[k] > 1e-3 * (hi[k] - lo[k]) and hi[k] - z[k] > 1e-3 * (hi[k] - lo[k])]
    u_, sv, vt = np.linalg.svd(J[:, free] * s[free], full_matrices=False)
    keep = sv > 1e-6 * sv[0]
    red = round(float(r0 @ r0) / (fm.N_VALUES - z.size), 3)
    cov = ((vt[keep].T * (1.0 / sv[keep] ** 2)) @ vt[keep]) * np.outer(s[free], s[free]) * red
    path.write_text(json.dumps({"label": "MODEL: local covariance of the fitted parameters at the optimum, scaled by the reduced chi2 "
                                         "(free parameters only)", "free": free, "cov": cov.tolist()}, indent=1))
    return cov, free, lo, hi


def main(n_draws):
    z = fm.load_parameters()
    inp = fm.inputs()
    slope = float(inp["slopes"][0])
    tg = inp["ret"]
    t0 = time.time()
    out = {"label": "MODEL PREDICTION: calibrated model, no fit", "t_s": MARKS.tolist(), "subthreshold_slope_dec_per_V": slope}
    r11, r15 = fm.retention_base(fm.cell(z), slope)
    out["sanity_fitted_points"] = {"SI11_model": r11, "SI11_meas": tg["SI11_dlogI_from_5s"], "SI15_model": r15, "SI15_meas": tg["SI15_dlogI_from_60s"],
                                   "chi2_block": float(sum(((a - b) / fm.SIG["ret"]) ** 2 for a, b in zip(r11 + r15, tg["SI11_dlogI_from_5s"] + tg["SI15_dlogI_from_60s"])))}
    for T in (300.0, 358.0):
        o_ = outputs(z, slope, T)
        law, best = laws(np.array(o_["dlogI_from_5s"]))
        y = np.array(o_["dlogI_from_5s"])
        u = np.log10(MARKS)
        n_loc = -np.gradient(y, u)
        o_["local_exponent"] = {"%g" % tt: float(np.interp(math.log10(tt), u, n_loc)) for tt in (10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, T_END)}
        o_["laws"], o_["governing_law"] = law, best
        out["T%d" % T] = o_
    chk = {}
    for name, r in SI12.items():
        mm = out["T300"][name]
        chk[name] = {"meas_first_to_end": r["dlogI_first_to_end"], "model_first_to_end": mm["model_first_to_end"],
                     "meas_last_decade": r["dlogI_last_decade"], "model_last_decade": mm["model_last_decade"],
                     "pass": bool(abs(mm["model_first_to_end"] - r["dlogI_first_to_end"]) <= TOL and abs(mm["model_last_decade"] - r["dlogI_last_decade"]) <= TOL)}
    out["check_SI12"] = chk
    out["check_pass"] = all(v["pass"] for v in chk.values())
    print("check", chk, flush=True)
    cov, free, lo, hi = covariance(z)
    rng = np.random.default_rng(172)
    draws = rng.multivariate_normal(np.zeros(len(free)), cov, size=n_draws, check_valid="ignore")
    keys = [("T300", "si12_10min", "t_2pA_s"), ("T300", "si12_1h", "t_2pA_s"), ("T300", "t_window_half_s"), ("T358", "si12_10min", "t_2pA_s"),
            ("T358", "si12_1h", "t_2pA_s"), ("T358", "t_window_half_s"), ("T300", "dlogI_10y"), ("T358", "dlogI_10y"),
            ("T300", "si12_10min", "model_first_to_end"), ("T300", "si12_1h", "model_first_to_end")]
    rows, failed = [], 0
    for i, dq in enumerate(draws):
        q = z.copy()
        q[free] = np.clip(z[free] + dq, lo[free], hi[free])
        try:
            res = {"T300": outputs(q, slope, 300.0), "T358": outputs(q, slope, 358.0)}
        except (FloatingPointError, ValueError):
            failed += 1
            continue
        for T in ("T300", "T358"):
            res[T]["dlogI_10y"] = res[T]["dlogI_from_5s"][-1]
        rows.append([res[k[0]][k[1]] if len(k) == 2 else res[k[0]][k[1]][k[2]] for k in keys])
        if i % 20 == 0:
            print("draw", i, "%.0f s" % (time.time() - t0), flush=True)
    a = np.array(rows, float)

    def rng95(col):
        return [float("%.4g" % v) for v in np.percentile(col, [2.5, 50, 97.5])] if np.all(np.isfinite(col)) else None

    out["draws"] = {"n_ok": len(rows), "n_failed": failed,
                    "ranges_95": {"/".join(k): rng95(a[:, j]) for j, k in enumerate(keys)},
                    "fraction_beyond_10y": {"/".join(k): float(np.mean(np.isinf(a[:, j]))) for j, k in enumerate(keys) if k[-1].startswith("t_")},
                    "largest_model_loss_dec": {"/".join(k): float(np.min(a[:, j])) for j, k in enumerate(keys) if k[-1] == "model_first_to_end"}}
    out["years"] = {"/".join(k): ([v / YEAR for v in out["draws"]["ranges_95"]["/".join(k)]] if out["draws"]["ranges_95"]["/".join(k)] else "beyond 10 years in every draw")
                    for k in keys if k[-1].startswith("t_")}
    (ROOT / "results" / "retention_projection.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("check_pass", "draws", "years")}, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
