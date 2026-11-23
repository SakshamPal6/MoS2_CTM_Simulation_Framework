from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import final_model as fm

OUT = ROOT / "results"
FITS = ROOT / "results" / "fits"
WIDTHS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1)


def best_run(mode):
    run = json.loads((FITS / ("%s.json" % mode)).read_text())
    z = fm.load_parameters()
    z[fm.free_indices(mode)] = run["best_z"]
    return z, run


def report():
    z = fm.load_parameters()
    inp, r = fm.inputs(), fm.readout()
    rr, det = fm.residuals(z, inp, r, "full", detail=True)
    chi2 = float(rr @ rr)
    near = [n for n, val, l_, h_ in zip(fm.NAMES, z, fm.LO, fm.HI) if min(val - l_, h_ - val) < 0.02 * (h_ - l_)]
    out = {"label": "MODEL: calibrated model of record", "chi2": round(chi2, 3), "n_values": fm.N_VALUES, "k_free": len(z),
           "reduced_chi2": round(chi2 / (fm.N_VALUES - len(z)), 3), "chi2_blocks": fm.blocks(rr), "near_bound": near,
           "parameters": dict(zip(fm.NAMES, np.round(z, 4).tolist())), "detail": det}
    (OUT / "model_report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("chi2", "reduced_chi2", "chi2_blocks", "near_bound")}, indent=1))


def views():
    z = fm.load_parameters()
    m = fm.cell(z)
    inp = fm.inputs()
    p = fm.parts(z, inp)
    w1us = float(fm.si6_window(m))
    t4 = list(fm.NBTI_TIMES)
    dense = [float(t) for t in np.unique(np.concatenate([np.logspace(-1, 3, 33), t4]))]
    curve = fm.bias_stress(z, times=dense)
    pts = inp["pts"]
    fig1d = {"amps_V": [q[0] for q in pts], "measured_hole_V": [q[1] for q in pts], "measured_hole_sd": [q[2] for q in pts],
             "measured_electron_V": [q[3] for q in pts], "measured_electron_sd": [q[4] for q in pts],
             "model_hole_V": [round(a, 3) for a, _ in p["branches"]], "model_electron_V": [round(b, 3) for _, b in p["branches"]]}
    nb = {}
    for tl in fm.NBTI_T:
        nb[tl] = {"t_s": t4, "measured_dV_V": [float(v) for v in inp["nbti"][tl]],
                  "model_dV_V": [round(float(v), 4) for v in p["nbti"][tl]],
                  "model_curve_t_s": list(dense), "model_curve_dV_V": [round(float(v), 4) for v in curve[tl]]}
        at4 = [curve[tl][dense.index(t)] for t in t4]
        assert np.allclose(at4, p["nbti"][tl], atol=5e-3), (tl, at4, p["nbti"][tl])
    pulse = {"sequence_V": [q[0] for q in fm.PULSE], "measured_mean_V": [q[1] for q in fm.PULSE],
             "measured_sd_V": [q[2] for q in fm.PULSE], "model_V": [round(d, 3) for d in fm.pulse_sequence(z)]}
    out = {"label": "MODEL against the calibration data (not predictions)", "fig1d": fig1d, "nbti_2023": nb,
           "pulse_100ms": pulse, "si6_window_V": {"model": round(w1us, 3), "measured": 3.45}}
    (OUT / "model_views.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("fig1d", "pulse_100ms", "si6_window_V")}, indent=0)[:1400])


def blocks():
    inp, r = fm.inputs(), fm.readout()
    out = {"label": "chi2 by data group of the full model and of the refits"}
    for mode in ("full", "nohole", "direct"):
        z, _ = best_run(mode)
        out[mode] = fm.blocks(fm.residuals(z, inp, r, mode))
    (OUT / "chi2_blocks.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def refits():
    bic = lambda chi2, k, n: chi2 + k * math.log(n)
    out = {"label": "MODEL: full model and refits with one path removed or one value left out", "variants": {}}
    for mode, n in (("full", 94), ("nohole", 94), ("direct", 94), ("loo", 93)):
        z, b = best_run(mode)
        k = len(fm.free_indices(mode))
        m = fm.cell(z, mode=mode)
        row = {"chi2": round(b["chi2"], 2), "k": k, "n": n, "BIC": round(bic(b["chi2"], k, n), 2),
               "window_1us_V": round(fm.window(m, 1e-6), 3), "window_100ms_V": round(fm.window(m, 0.1), 3),
               "z": {fm.NAMES[i]: round(float(z[i]), 4) for i in fm.free_indices(mode)}}
        row["release_1us"] = fm.release(m, 1e-6)
        if mode != "nohole":
            hv = m.hv.copy()
            m.hv = np.zeros_like(hv)
            row["frozen_no_hole_window_1us_V"] = round(fm.window(m, 1e-6), 3)
            m.hv = hv
        out["variants"][mode] = row
    (OUT / "refits.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {q: v[q] for q in ("chi2", "BIC", "window_1us_V", "window_100ms_V")} for k, v in out["variants"].items()}, indent=1))


def _rates(m, st, v_g):
    dva, dvb = m.dv_a(st[0]), m.dv_b(st[1])
    u = v_g - m.v_ref - dva - dvb
    ps = min(max(u / m.m, m.p.psi_lo), m.p.psi_hi)
    fld = (u - ps) / m.l_eq
    onset = m.p.hole_onset if np.isfinite(m.p.hole_onset) else m.p.psi_lo
    sh = 1.0 / (1.0 + np.exp(np.clip((u / m.m - onset) / m.p.hole_width, -700.0, 700.0)))
    n_tot = m.x.shape[0]
    e = m.et0 + fld * m.x
    ke = m.nux * np.exp(-np.clip((m.eb + 0.5 * e + m.inv4lam * e * e) / m.kt, 0.0, 700))
    ev = e - m.p.hole_e_gap
    kh = m.hv * sh * m.nux * np.exp(-np.clip((m.eb + 0.5 * ev + m.inv4lam * ev * ev) / m.kt, 0.0, 700))
    return fld, sh, ke[:n_tot], kh[:n_tot], u


def _shares(m, st0, v_g, width, n=240):
    n_tot = m.x.shape[0]
    n_cv, _, _, _, kxe = m._conv_args()
    grid = np.concatenate([[0.0], np.logspace(np.log10(width) - 6, np.log10(width), n)])
    st, qe, qh, trace = st0, 0.0, 0.0, []
    for t0, t1 in zip(grid[:-1], grid[1:]):
        mid = m.hold(st, v_g, 0.5 * (t1 - t0))
        fld, sh, ke, kh, u = _rates(m, mid, v_g)
        ca, fa = m.ca[:n_tot], mid[0][:n_tot]
        qe += float(np.sum(ca * fa * ke)) * (t1 - t0)
        qh += float(np.sum(ca * fa * kh)) * (t1 - t0)
        if n_cv:
            cd, fd = m.ca[n_tot:], mid[0][n_tot:]
            qe += float(np.sum(cd * fd * ke[:n_cv]) * kxe) * (t1 - t0)
            qh += float(np.sum(cd * fd * kh[:n_cv]) * kxe) * (t1 - t0)
        st = m.hold(st, v_g, t1 - t0)
        trace.append([float(t1), float(fld), float(sh)])
    i = [0, n // 3, 2 * n // 3, n - 1]
    return {"released_by_electron_emission_V": qe, "released_by_hole_assisted_path_V": qh,
            "hole_assisted_share": qh / (qe + qh) if qe + qh else None,
            "field_MV_cm_and_hole_availability": [[float("%.3g" % trace[k][0]), round(trace[k][1] / 1e8, 3), round(trace[k][2], 4)] for k in i]}


def paths():
    z = fm.load_parameters()
    out = {"label": "MODEL CALCULATION: calibrated model, base cell, 300 K; a path removed at frozen parameters (not a refit)",
           "protocol": {"erase_s": fm.ERASE_S, "rest_s": fm.REST_S}}
    m = fm.cell(z)
    full = {w: fm.window(m, w) for w in WIDTHS}
    out["window_V_full_model"] = {"%g" % k: round(v, 3) for k, v in full.items()}
    er = m.hold(m.hold(m.fresh(), 10.0, fm.ERASE_S), 0.0, fm.REST_S)
    out["shares_erased_cell"] = {"%g" % w: _shares(m, er, -10.0, w) for w in (1e-6, 0.1)}
    hv = m.hv.copy()
    m.hv = np.zeros_like(hv)
    out["window_V_no_hole_path"] = {"%g" % w: round(fm.window(m, w), 3) for w in full}
    m.hv = hv
    nux = m.nux.copy()
    m.nux = m.p.nu0 * np.exp(-m.p.two_kappa_nm * m.x * 1e9)
    out["direct_tunnelling_rate_s"] = {"nu0": m.p.nu0, "two_kappa_per_nm": m.p.two_kappa_nm,
                                       "at_x_lo": float(m.nux.max()), "chain_rate_at_x_lo": float(nux.max())}
    out["window_V_no_hopping_chain"] = {"%g" % w: round(fm.window(m, w), 3) for w in full}
    m.nux = nux
    m = fm.cell(z)
    out["release_1us"] = fm.release(m, 1e-6)
    out["release_100ms"] = fm.release(m, 0.1)
    for mode, key in (("nohole", "window_V_refit_no_hole"), ("direct", "window_V_refit_direct")):
        zm, _ = best_run(mode)
        mm = fm.cell(zm, mode=mode)
        out[key] = {"%g" % w: round(fm.window(mm, w), 3) for w in WIDTHS}
    (OUT / "charge_paths.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in out if k.startswith(("window", "release"))}, indent=1))


def jac():
    z = fm.load_parameters()
    inp, r = fm.inputs(), fm.readout()
    lo, hi, s = fm.LO, fm.HI, fm.SCALE
    r0 = fm.residuals(z, inp, r)
    chi2 = float(r0 @ r0)
    J = np.zeros((r0.size, z.size))
    h = np.maximum(1e-3 * np.abs(z), 1e-3) * np.where(z + 1e-3 * np.abs(z) + 1e-3 > hi, -1.0, 1.0)
    for k in range(z.size):
        q = z.copy()
        q[k] += h[k]
        J[:, k] = (fm.residuals(q, inp, r) - r0) / h[k]
    free = [k for k in range(z.size) if z[k] - lo[k] > 1e-3 * (hi[k] - lo[k]) and hi[k] - z[k] > 1e-3 * (hi[k] - lo[k])]
    Js = J[:, free] * s[free]
    u, sv, vt = np.linalg.svd(Js, full_matrices=False)
    keep = sv > 1e-6 * sv[0]
    red = round(chi2 / (fm.N_VALUES - z.size), 3)
    cov_s = (vt[keep].T * (1.0 / sv[keep] ** 2)) @ vt[keep]
    cov = cov_s * np.outer(s[free], s[free]) * red
    sd = np.sqrt(np.diag(cov))
    rng = np.random.default_rng(161)
    draws = rng.multivariate_normal(np.zeros(len(free)), cov, size=200, check_valid="ignore")
    rows = []
    for dq in draws:
        q = z.copy()
        q[free] = np.clip(z[free] + dq, lo[free], hi[free])
        try:
            m = fm.cell(q)
            w1, w2 = fm.window(m, 1e-6), fm.window(m, 0.1)
            n = fm.cell(q, mode="nohole")
            rows.append((w1, w2, fm.window(n, 1e-6)))
        except FloatingPointError:
            pass
    a = np.array(rows)
    out = {"label": "MODEL: local covariance at the optimum scaled by the reduced chi2, 200 draws", "n_draws": len(a),
           "params_at_bound": [fm.NAMES[k] for k in range(z.size) if k not in free], "svd_rank": int(keep.sum()),
           "n_free_used": len(free), "condition_number": float(sv[0] / sv[keep][-1]),
           "sd": {fm.NAMES[k]: float("%.3g" % sd[i]) for i, k in enumerate(free)},
           "z": {fm.NAMES[k]: float("%.4g" % z[k]) for k in free},
           "window_1us_full_V": [float("%.3f" % q) for q in np.percentile(a[:, 0], [2.5, 50, 97.5])],
           "window_100ms_full_V": [float("%.3f" % q) for q in np.percentile(a[:, 1], [2.5, 50, 97.5])],
           "window_1us_no_hole_V": [float("%.3f" % q) for q in np.percentile(a[:, 2], [2.5, 50, 97.5])]}
    (OUT / "parameter_uncertainty.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"report": report, "views": views, "blocks": blocks, "refits": refits, "paths": paths, "jac": jac}[sys.argv[1]]()
