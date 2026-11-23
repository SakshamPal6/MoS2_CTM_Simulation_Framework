from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import brentq
from scipy.stats import norm

from model.physics import EPS0, Q, STACK, Cell, CellParams
from model.readout import Readout

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
NM = 1e-9

NAMES = ["log10_N_t", "x_lo_nm", "x_width_nm", "E_phys", "sigma_b", "E_t0", "lambda_eV", "log10_rate", "log10_dt_pt",
         "log10_tau_B", "sigma_tau", "log10_asym", "E_B_eV", "log10_s_C25", "log10_s_G24", "dvref_C25", "dvref_G24",
         "log10_h", "psi_h", "log10_conv", "conv_delta", "conv_dx", "log10_N_S", "dE_S", "sigma_S"]
LO = np.array([12.5, 0.5, 0.2, 0.6, 0.03, -0.5, 1.0, -1.0, -2.7, 0.0, 0.05, -2.0, 0.0, -1.0, -1.0, -3.0, -3.0,
               -4.0, -2.5, -2.0, 0.0, 0.0, 13.0, 0.0, 0.03])
HI = np.array([14.5, 15.0, 20.0, 2.2, 0.5, 0.5, 30.0, 2.0, -1.3, 5.0, 2.0, 3.0, 1.2, 1.0, 1.0, 3.0, 3.0,
               4.0, -1.0, 9.0, 0.6, 0.8, 15.0, 0.8, 0.5])
SCALE = np.array([0.3, 1.0, 1.0, 0.1, 0.05, 0.1, 2.0, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.2, 0.2, 0.3, 0.3,
                  0.5, 0.1, 0.5, 0.05, 0.05, 0.3, 0.05, 0.05])
HOLE = [NAMES.index("log10_h"), NAMES.index("psi_h")]
I_WINDOW_1US = 12
N_VALUES = 94
PARAM_FILE = ROOT / "results" / "model_parameters.json"


def load_parameters(path=PARAM_FILE):
    d = json.loads(Path(path).read_text())
    return np.array([d["parameters"][n] for n in NAMES], float)


HOP = (9.28e8, 1.49)
RES = (3, 33, 9)
MAX_STEPS = 2_000_000
NU0, TWO_KAPPA = 1e13, 8.9
DEV = {"base": {"stack": STACK, "ss": 0.398},
       "s2023": {"stack": ((40e-9, 10.0), (3e-9, 25.0), (7e-9, 10.0)), "ss": 0.231},
       "C25": {"stack": ((15 * NM, 10.0), (3 * NM, 25.0), (7 * NM, 10.0)), "ss": 0.210, "vth0": 1.15,
               "read": (-2.0, 5.0, 0.07, True)},
       "G24": {"stack": ((15 * NM, 10.0), (7 * NM, 25.0), (3 * NM, 10.0)), "ss": 0.150, "vth0": 0.20,
               "read": (-3.0, 3.0, 0.05, False)}}
ERASE_S, REST_S = 0.1, 1.0
DWELL = 48.0
NBTI_T = {"25C": 298.15, "125C": 398.15}
NBTI_TIMES = [1.0, 10.0, 100.0, 1000.0]
T11 = [10.0, 30.0, 100.0, 300.0, 600.0]
T15 = [120.0, 300.0, 600.0, 1200.0, 2400.0, 3600.0]
W100_S2023 = (4.49, 0.30, 12.0)
SS_C25, SS_2023 = 0.210, 0.231
PULSE = [(-10.0, -1.52, 0.76), (10.0, 0.32, 1.0), (-10.0, -0.39, 0.35), (-10.0, -0.71, 0.35)]
TW, SD_OFF = 0.1, 0.23
ON_3G, ON_19, OFF_19 = -7.26, -7.275, 0.0089
S7A = (-0.230, 0.15, 1.167, 0.3)
SIG = {"branch": 0.25, "w1us": 0.15, "drift": 0.03, "ret": 0.05, "s23": 0.10, "series": 0.20, "c25_ret": 0.05}
HOOKS = []


def d_b_log10(stack, ss):
    c_ox = EPS0 / sum(t / k for t, k in stack)
    return math.log10(c_ox * (ss / 60e-3 - 1.0) / Q / 1e4)


def cell(z, dev="base", T=300.0, mode="full"):
    P = dict(zip(NAMES, z))
    d = DEV[dev]
    st, ss = d["stack"], d["ss"]
    total_eq = 10.0 * sum(t / k for t, k in st) / NM
    hi = min(P["x_lo_nm"] + P["x_width_nm"], total_eq - 0.5)
    lo = min(P["x_lo_nm"], hi - 0.1)
    scale = 10 ** P["log10_s_" + dev] if dev in ("C25", "G24") else 1.0
    v_ref = P["dvref_" + dev] if dev in ("C25", "G24") else 0.0
    k, r = HOP
    p = CellParams(log10_nt_cm2=P["log10_N_t"], x_min_nm=lo, x_max_nm=hi, n_x=RES[0], e_b=P["E_phys"], sigma_b=P["sigma_b"],
                   e_t0=P["E_t0"], lam=P["lambda_eV"], k_hop=k, r_hop_nm=r, n_scale=scale,
                   log10_db=d_b_log10(st, ss), log10_tau_b=P["log10_tau_B"], sigma_tau=P["sigma_tau"],
                   log10_asym=P["log10_asym"], e_act_b=P["E_B_eV"], n_e=RES[1], n_b=RES[2],
                   hole_log10_scale=P["log10_h"], hole_onset=P["psi_h"],
                   conv_log10_rate=P["log10_conv"], conv_delta=P["conv_delta"], conv_dx=P["conv_dx"],
                   s_log10_nt=P["log10_N_S"] + np.log10(scale), s_e_b=P["E_phys"] + P["dE_S"], s_sigma_b=P["sigma_S"],
                   s_e_t0=P["E_t0"])
    m = Cell(p, stack=st, ss=ss, T=T, v_ref=v_ref, max_steps=MAX_STEPS)
    if mode == "nohole":
        m.hv = np.zeros_like(m.hv)
    elif mode == "direct":
        m.nux = NU0 * np.exp(-TWO_KAPPA * m.x * 1e9)
    for f in HOOKS:
        m = f(m, dev)
    return m


def window(m, width):
    er = m.hold(m.hold(m.fresh(), 10.0, ERASE_S), 0.0, REST_S)
    pg = m.hold(m.hold(er, -10.0, width), 0.0, REST_S)
    return float(m.dv(er) - m.dv(pg))


def _yaml(name):
    return yaml.safe_load((DATA / "measured" / name).read_text(encoding="utf-8"))


def _json(name):
    return json.loads((DATA / "measured" / name).read_text())


def readout():
    ro = _json("readout_fit.json")
    return Readout(ro["vth"], ro["ss"], ro["mu"], EPS0 / sum(t / k for t, k in DEV["base"]["stack"]), 1.0, ro["r_s"], 1e-12, 1.0)


def inputs():
    d = _yaml("base_fig1d_window_vs_sweep.yaml")
    pts = d["branch_shifts"]["points"]
    w6 = _yaml("base_si_pulse_shifts.yaml")["si6_1us"]["V_th_at_1e-10"]["window"]
    seg = _yaml("base_si7_vmod_segments.yaml")["segments"]
    bounds = [0.0] + [0.5 * (seg[k][2] + seg[k + 1][1]) for k in range(len(seg) - 1)] + [seg[-1][2]]
    lv = {s[0]: np.log10(s[5]) for s in seg[:7]}
    slopes = {3: lv[3] - lv[2], 2: 0.5 * ((lv[3] - lv[2]) + (lv[2] - lv[1])), 1: 0.5 * ((lv[2] - lv[1]) + (lv[1] - lv[0])),
              0: 0.5 * ((lv[1] - lv[0]) + (lv[0] - lv[-1])), -1: 0.5 * ((lv[0] - lv[-1]) + (lv[-1] - lv[-2])),
              -2: 0.5 * ((lv[-1] - lv[-2]) + (lv[-2] - lv[-3])), -3: lv[-2] - lv[-3]}
    ref = _json("base_fig1d_ref_levels.json")["reference_4V_centre_V_by_level"]
    nb = _yaml("s2023_nbti.yaml")["dVth_V"]
    c25, g24 = _json("bt_c25_extracted.json"), _json("bt_g24_extracted.json")
    sw = c25["program"]["sweeps"][:8]
    return {"pts": pts, "w6": w6, "seg": seg, "bounds": bounds, "slopes": slopes,
            "levels": np.array([ref[k] for k in sorted(ref, key=float)]),
            "ret": _json("retention_targets.json"),
            "nbti": {"25C": [nb["pre_25C"]["%g" % t] for t in NBTI_TIMES], "125C": [nb["pre_125C"]["%g" % t] for t in NBTI_TIMES]},
            "s23": _yaml("s2023_retention.yaml"),
            "c25_prog_fwd": [s["vth_up_V"] - sw[0]["vth_up_V"] for s in sw][1:],
            "c25_prog_ret": [s["vth_down_V"] - sw[0]["vth_up_V"] for s in sw],
            "c25_eras": [v - c25["erase"]["vth_plotted_V"][0] for v in c25["erase"]["vth_plotted_V"]][1:],
            "c25_states": c25["retention"]["states"],
            "c25_ret": [s["end_log10I"] - s["first_log10I"] for s in c25["retention"]["states"]],
            "g24_prog": [v - g24["program"]["vth_V"][0] for v in g24["program"]["vth_V"]][1:],
            "g24_eras": [v - g24["erase"]["vth_V"][0] for v in g24["erase"]["vth_V"]][1:]}


def si6_window(m):
    st = m.hold(m.hold(m.fresh(), 10.0, ERASE_S), 0.0, REST_S)
    pg = m.hold(m.hold(st, -10.0, 1e-6), 0.0, REST_S)
    return m.dv(st) - m.dv(pg)


def simulate_si7(m, seg, bounds, slopes):
    st = m.hold(m.hold(m.fresh(), 10.0, 0.1), -10.0, 0.1)
    drifts, mids = [], {}
    for k, s in enumerate(seg):
        v = float(s[0])
        st = m.hold(st, v, max(s[1] - bounds[k], 0.0))
        a = m.dv(st)
        mids[k] = m.dv(m.hold(st, v, 0.5 * (s[2] - s[1])))
        st = m.hold(st, v, s[2] - s[1])
        b = m.dv(st)
        drifts.append(-(b - a) * slopes[int(v)])
        st = m.hold(st, v, max(bounds[k + 1] - s[2], 0.0))
    return drifts, -(mids[7] - mids[3]) * slopes[0]


def sweep_and_reads(m, rate, inp):
    u4, d4 = m.sweep_branches(4.0, rate)
    c4 = 0.5 * (u4 + d4)
    v0 = inp["levels"] - c4
    u4l, d4l = m.sweep_branches_at(4.0, rate, v0)
    c4l = 0.5 * (u4l + d4l)
    br = []
    for a, *_ in inp["pts"]:
        u, dd = m.sweep_branches_at(float(a), rate, v0)
        br.append((float(np.mean(u - c4l)), float(np.mean(dd - c4l))))
    win = si6_window(m)
    drifts, off = simulate_si7(m, inp["seg"], inp["bounds"], inp["slopes"])
    return br, win, drifts, off


def retention_base(m, slope0):
    st = m.hold(m.hold(m.fresh(), 10.0, 0.1), -10.0, 0.1)
    marks = sorted({5.0, 60.0, *T11, *T15})
    dv, tp = {}, 0.0
    for t in marks:
        st = m.hold(st, 0.0, t - tp)
        dv[t] = m.dv(st)
        tp = t
    return [-(dv[t] - dv[5.0]) * slope0 for t in T11], [-(dv[t] - dv[60.0]) * slope0 for t in T15]


def bias_stress(z, mode="full", times=NBTI_TIMES):
    out = {}
    for tl, T in NBTI_T.items():
        m = cell(z, "s2023", T, mode)
        st, tp, vals = m.fresh(), 0.0, []
        for t in times:
            st = m.hold(st, -3.0, t - tp)
            vals.append(m.dv(st))
            tp = t
        out[tl] = vals
    return out


def window_100ms(m, erase):
    s_prog = m.hold(m.hold(m.fresh(), erase, 0.1), -10.0, 0.1)
    return m.dv(m.hold(s_prog, erase, 0.1)) - m.dv(s_prog)


def retention_s2023(m, s23):
    st = m.hold(m.hold(m.fresh(), 12.0, 0.1), -10.0, 0.1)
    r = s23["retention"]
    marks, dv, tp = [r["t_first_s"], 0.1 * r["t_end_s"], r["t_end_s"]], {}, 0.0
    for t in marks:
        st = m.hold(st, 0.0, t - tp)
        dv[t] = m.dv(st)
        tp = t
    k = s23["readout_slope_dec_per_V_at_0V"]
    t1, t9, te = marks
    return -(dv[te] - dv[t1]) * k, -(dv[te] - dv[t9]) * k


def _read(m, st, dev, dt_pt):
    v0, v1, step, double = DEV[dev]["read"]
    st, vth = m.read_sweep(st, v0, v1, step, dt_pt, DEV[dev]["vth0"], double=double)
    n_pts = (round((v1 - v0) / step) + 1) * (2 if double else 1)
    return st, vth, n_pts * dt_pt


def series_loop(m, dev, amps, width, dt_pt, loops=2):
    st = m.fresh()
    res = {}
    for _ in range(loops):
        for sign in (1.0, -1.0):
            rows = []
            for a in amps:
                st = m.hold(m.hold(st, sign * a, width), 0.0, 1.0)
                st, vth, t_read = _read(m, st, dev, dt_pt)
                st = m.hold(st, 0.0, max(DWELL - 1.0 - width - t_read, 0.0))
                rows.append(vth)
            res[sign] = rows
    return res


def _readout_c25(vth):
    return -7.0 - vth / SS_C25 if vth >= 0 else -7.0 - 0.5 * vth


def _inv_readout_c25(li):
    return (-7.0 - li) * SS_C25 if li <= -7.0 else -(li + 7.0) / 0.5


def retention_c25(m, states):
    out = []
    for s in states:
        dv1 = _inv_readout_c25(s["first_log10I"]) - DEV["C25"]["vth0"]
        lo, hi = 0.0, 16.0
        st1 = None
        for _ in range(22):
            a = 0.5 * (lo + hi)
            st1 = m.hold(m.hold(m.fresh(), -a, 0.01), 0.0, 1.0)
            if m.dv(st1) > dv1:
                lo = a
            else:
                hi = a
        st300 = m.hold(st1, 0.0, 299.0)
        vth1 = dv1 + DEV["C25"]["vth0"]
        out.append(_readout_c25(vth1 + m.dv(st300) - m.dv(st1)) - s["first_log10I"])
    return out


def pulse_sequence(z, mode="full"):
    m = cell(z, "base", 300.0, mode)
    st, out = m.fresh(), []
    for v, *_ in PULSE:
        st = m.hold(m.hold(st, v, 0.1), 0.0, 1.0)
        out.append(m.dv(st))
    return out


def log_g(r, dv):
    return math.log10(r.current(0.0, 1.0, dv) / 1.0)


def array_write(z, r, mode="full"):
    m = cell(z, "base", 300.0, mode)
    blk = m.hold(m.fresh(), 10.0, 0.1)
    lg = lambda s: log_g(r, m.dv(m.hold(s, 0.0, 1.0)))
    pr = m.hold(blk, -10.0, TW)
    s = pr
    for _ in range(12):
        s = m.hold(m.hold(s, 5.0, 28 * TW), -5.0, TW)
    on3g = lg(s)
    s = pr
    for _ in range(13):
        s = m.hold(m.hold(s, 5.0, 9 * TW), -5.0, TW)
    on19 = lg(s)
    s = blk
    for _ in range(26):
        s = m.hold(m.hold(s, 5.0, 9 * TW), -5.0, TW)
    dstar = brentq(lambda d: log_g(r, d) + 10.5, -8.0, 10.0)
    f19 = float(norm.cdf((dstar - m.dv(m.hold(s, 0.0, 1.0))) / SD_OFF))
    return {"on_3g": on3g, "on_si19": on19, "partially_on_si19": f19}


def cycling_s2023(z, s23, mode="full"):
    m = cell(z, "s2023", 300.0, mode)
    st, rows = m.fresh(), []
    for _ in range(50):
        st = m.hold(m.hold(st, -10.0, 0.1), 0.0, 2.5)
        p = m.dv(st)
        st = m.hold(m.hold(m.hold(st, 0.0, 2.55), 12.0, 0.1), 0.0, 2.5)
        e = m.dv(st)
        st = m.hold(st, 0.0, 2.55)
        rows.append((p, e))
    k = s23["readout_slope_dec_per_V_at_0V"]
    return -k * (rows[-1][0] - rows[0][0]), -(rows[-1][1] - rows[0][1]) / SS_2023


def parts(z, inp, mode="full", times=NBTI_TIMES):
    dt_pt = 10 ** z[NAMES.index("log10_dt_pt")]
    mb = cell(z, "base", 300.0, mode)
    br, win, drifts, off = sweep_and_reads(mb, 10 ** z[NAMES.index("log10_rate")], inp)
    r11, r15 = retention_base(cell(z, "base", 300.0, mode), float(inp["slopes"][0]))
    out = {"branches": br, "w1us": win, "drifts": drifts, "offset": off, "r11": r11, "r15": r15,
           "nbti": bias_stress(z, mode, times),
           "w100_s2023": window_100ms(cell(z, "s2023", 300.0, mode), W100_S2023[2]),
           "s23_ret": retention_s2023(cell(z, "s2023", 300.0, mode), inp["s23"])}
    m = cell(z, "C25", 300.0, mode)
    r = series_loop(m, "C25", list(range(4, 13)), 0.01, dt_pt)
    p, ee = r[-1.0][1:], r[1.0][1:]
    out["c25_prog_fwd"] = [q[0] - p[0][0] for q in p][1:]
    out["c25_prog_ret"] = [q[1] - p[0][0] for q in p]
    out["c25_eras"] = [q[1] - ee[0][1] for q in ee][1:]
    out["c25_ret"] = retention_c25(cell(z, "C25", 300.0, mode), inp["c25_states"])
    m = cell(z, "G24", 300.0, mode)
    r = series_loop(m, "G24", list(range(4, 13)), 1e-4, dt_pt)
    out["g24_prog"] = [q[0] - r[-1.0][0][0] for q in r[-1.0]][1:]
    out["g24_eras"] = [q[0] - r[1.0][0][0] for q in r[1.0]][1:]
    return out


def _core_residuals(p, inp):
    pts, w6, seg, tg, s23 = inp["pts"], inp["w6"], inp["seg"], inp["ret"], inp["s23"]
    r = []
    for (a, hs, _, es, _), (mh, me) in zip(pts, p["branches"]):
        r += [(mh - hs) / SIG["branch"], (me - es) / SIG["branch"]]
    r.append((p["w1us"] - w6) / SIG["w1us"])
    meas = [s[4] - s[3] for s in seg]
    r += [(p["drifts"][k] - meas[k]) / SIG["drift"] for k in range(6)]
    r.append((p["offset"] - float(np.log10(seg[7][5]) - np.log10(seg[3][5]))) / SIG["drift"])
    r += [(a - b) / SIG["ret"] for a, b in zip(p["r11"], tg["SI11_dlogI_from_5s"])]
    r += [(a - b) / SIG["ret"] for a, b in zip(p["r15"], tg["SI15_dlogI_from_60s"])]
    for tl in NBTI_T:
        r += [(q - y) / (0.05 + 0.15 * abs(y)) for q, y in zip(p["nbti"][tl], inp["nbti"][tl])]
    r.append((p["w100_s2023"] - W100_S2023[0]) / W100_S2023[1])
    r.append((p["s23_ret"][0] - s23["retention"]["dlogI_first_to_end"]) / SIG["s23"])
    r.append((p["s23_ret"][1] - s23["retention"]["dlogI_last_decade"]) / SIG["s23"])
    for k in ("c25_prog_fwd", "c25_prog_ret", "c25_eras"):
        r += [(a - b) / SIG["series"] for a, b in zip(p[k], inp[k])]
    r += [(a - b) / SIG["c25_ret"] for a, b in zip(p["c25_ret"], inp["c25_ret"])]
    for k in ("g24_prog", "g24_eras"):
        r += [(a - b) / SIG["series"] for a, b in zip(p[k], inp[k])]
    return np.nan_to_num(np.array(r, float), nan=50.0, posinf=50.0, neginf=-50.0)


N_CORE = 85
ERRORS = (FloatingPointError, ValueError, ZeroDivisionError, OverflowError)
BLOCKS = {"sweep_branches": (0, 12), "window_1us": (12, 13), "read_gate": (13, 20), "retention_base": (20, 31),
          "bias_stress_40nm": (31, 39), "window_100ms_40nm": (39, 40), "retention_40nm": (40, 42), "c25_series": (42, 64),
          "c25_retention": (64, 69), "g24": (69, 85), "pulse_100ms": (85, 89), "arrays": (89, 92), "s7a": (92, 94)}


def residuals(z, inp, r, mode="full", detail=False):
    try:
        try:
            rc = _core_residuals(parts(z, inp, mode), inp)
        except ERRORS:
            rc = np.full(N_CORE, 50.0)
        sh = pulse_sequence(z, mode)
        r1 = [(d - meas) / sig for d, (_, meas, sig) in zip(sh, PULSE)]
        ar = array_write(z, r, mode)
        r2 = [(ar["on_3g"] - ON_3G) / 0.2, (ar["on_si19"] - ON_19) / 0.2, (ar["partially_on_si19"] - OFF_19) / 0.02]
        on_dec, off_dec = cycling_s2023(z, inp["s23"], mode)
        r3 = [(on_dec - S7A[0]) / S7A[1], (off_dec - S7A[2]) / S7A[3]]
        out = np.concatenate([rc, r1, r2, r3])
        det = {"pulse": {"shifts_V": [round(d, 3) for d in sh]}, "array": ar,
               "s7a": {"program_dec": on_dec, "erase_dec": off_dec}}
    except ERRORS as e:
        out, det = np.full(N_VALUES, 50.0), {"error": type(e).__name__}
    if mode == "loo":
        out = np.array(out)
        out[I_WINDOW_1US] = 0.0
    return (out, det) if detail else out


def blocks(rr):
    row = {k: round(float(rr[a:b] @ rr[a:b]), 2) for k, (a, b) in BLOCKS.items()}
    row["total"] = round(float(rr @ rr), 2)
    return row


def free_indices(mode):
    return [k for k in range(len(NAMES)) if not (mode == "nohole" and k in HOLE)]


def release(m, width, n=240):
    n_tot = m.x.shape[0]
    n_cv, _, _, _, kxe = m._conv_args()
    st = m.hold(m.hold(m.fresh(), 10.0, ERASE_S), 0.0, REST_S)
    grid = np.concatenate([[0.0], np.logspace(np.log10(width) - 6, np.log10(width), n)])
    q = {"e_pos": 0.0, "e_neg": 0.0, "h_pos": 0.0, "h_neg": 0.0}
    amin, rmax, fl = 1e9, 0.0, []
    onset = m.p.hole_onset if np.isfinite(m.p.hole_onset) else m.p.psi_lo
    for t0, t1 in zip(grid[:-1], grid[1:]):
        mid = m.hold(st, -10.0, 0.5 * (t1 - t0))
        u = -10.0 - m.v_ref - m.dv_a(mid[0]) - m.dv_b(mid[1])
        ps = min(max(u / m.m, m.p.psi_lo), m.p.psi_hi)
        fld = (u - ps) / m.l_eq
        sh = 1.0 / (1.0 + np.exp(np.clip((u / m.m - onset) / m.p.hole_width, -700.0, 700.0)))
        en = m.et0 + fld * m.x
        ev = en - m.p.hole_e_gap
        ae = (m.eb + 0.5 * en + m.inv4lam * en * en) / m.kt
        ah = (m.eb + 0.5 * ev + m.inv4lam * ev * ev) / m.kt
        ke = m.nux * np.exp(-np.maximum(ae, 0.0))
        kh = m.hv * sh * m.nux * np.exp(-np.maximum(ah, 0.0))
        w = m.ca[:n_tot] * mid[0][:n_tot]
        if n_cv:
            w = w.copy()
            w[:n_cv] = w[:n_cv] + m.ca[n_tot:] * mid[0][n_tot:] * kxe
        dt = t1 - t0
        q["e_pos"] += float(np.sum((w * ke)[ae >= 0])) * dt
        q["e_neg"] += float(np.sum((w * ke)[ae < 0])) * dt
        q["h_pos"] += float(np.sum((w * kh)[(ah >= 0) & (m.hv > 0)])) * dt
        q["h_neg"] += float(np.sum((w * kh)[(ah < 0) & (m.hv > 0)])) * dt
        i = m.nux > 0
        amin = min(amin, float(np.min(ae[i] * m.kt)), float(np.min(ah[i & (m.hv > 0)] * m.kt)) if np.any(i & (m.hv > 0)) else 1e9)
        rmax = max(rmax, float(np.max(np.maximum(ke, kh)[i] / m.nux[i])))
        fl.append(fld / 1e8)
        st = m.hold(st, -10.0, dt)
    tot = sum(q.values())
    return {"released_V": round(tot, 4), "share_above_attempt_rate": round((q["e_neg"] + q["h_neg"]) / tot, 4) if tot else None,
            "share_by_path": {k: round(v / tot, 4) for k, v in q.items()} if tot else None,
            "lowest_activation_eV": round(amin, 3), "largest_rate_over_prefactor": float("%.3g" % rmax),
            "largest_rate_prefactor_s": float("%.3g" % float(m.nux.max())),
            "field_MV_cm_start_end": [round(fl[0], 3), round(fl[-1], 3)]}
