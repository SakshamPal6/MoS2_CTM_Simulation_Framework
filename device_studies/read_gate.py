from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import final_model as fm


def split(m, st):
    return float(m.dv_a(st[0])), float(m.dv_b(st[1]))


def read_gate(m, inp):
    seg, bounds, slopes = inp["seg"], inp["bounds"], inp["slopes"]
    drifts, off = fm.simulate_si7(m, seg, bounds, slopes)
    st = m.hold(m.hold(m.fresh(), 10.0, 0.1), -10.0, 0.1)
    mids = {}
    for k, s in enumerate(seg):
        v = float(s[0])
        st = m.hold(st, v, max(s[1] - bounds[k], 0.0))
        mids[k] = split(m, m.hold(st, v, 0.5 * (s[2] - s[1])))
        st = m.hold(st, v, s[2] - s[1])
        st = m.hold(st, v, max(bounds[k + 1] - s[2], 0.0))
    sl = float(slopes[0])
    da, db = mids[7][0] - mids[3][0], mids[7][1] - mids[3][1]
    assert abs(-(da + db) * sl - off) < 5e-4, (da, db, sl, off)
    rec, t_prev = {}, 0.0
    for t in (1e2, 1e3, 1e4, 1e5):
        st = m.hold(st, 0.0, t - t_prev)
        t_prev = t
        p = split(m, st)
        rec["%g" % t] = {"A_minus_first_read_V": p[0] - mids[3][0], "B_minus_first_read_V": p[1] - mids[3][1],
                         "offset_dec": -(p[0] - mids[3][0] + p[1] - mids[3][1]) * sl}
    share = abs(db) / (abs(da) + abs(db))
    st0 = m.hold(m.hold(m.fresh(), 10.0, 0.1), -10.0, 0.1)
    mid3, mid7 = 0.5 * (seg[3][1] + seg[3][2]), 0.5 * (seg[7][1] + seg[7][2])
    a3 = split(m, m.hold(st0, 0.0, mid3))
    a7 = split(m, m.hold(st0, 0.0, mid7))
    ctl = -(a7[0] - a3[0] + a7[1] - a3[1]) * sl
    return {"model_offset_dec": off, "model_final_0V_drift_dec": float(drifts[7]) if len(drifts) > 7 else None,
            "n_segments": len(seg), "measured_final_0V_drift_dec": -0.102, "slope_dec_per_V": sl, "dA_V": da, "dB_V": db,
            "share_surface": share, "offset_same_time_at_0V_no_read_gate_dec": ctl, "recovery_after_sequence": rec,
            "measured_offset_dec": 0.202, "first_read_s": mid3, "second_read_s": mid7}


def hole_onset(m, log10_rate):
    onset = m.p.hole_onset if np.isfinite(m.p.hole_onset) else m.p.psi_lo
    u_half = onset * m.m
    out = {"psi_h": float(onset), "m": float(m.m), "u_half_V": float(u_half), "v_ref_V": float(m.v_ref)}
    st = m.hold(m.fresh(), 10.0, fm.ERASE_S)
    st = m.hold(st, 0.0, fm.REST_S)
    dv0 = float(m.dv_a(st[0]) + m.dv_b(st[1]))
    out["erased_dV_V"] = dv0
    out["V_G_half_from_erased_V"] = float(u_half + m.v_ref + dv0)
    rate = 10 ** log10_rate
    out["sweep_rate_V_s"] = rate
    dv = 0.1
    st = m.fresh()
    for v in np.arange(0.0, 10.0 + dv / 2, dv):
        st = m.hold(st, float(v), dv / rate)
    cross, umin = None, 1e9
    for v in np.arange(10.0, -10.0 - dv / 2, -dv):
        st = m.hold(st, float(v), dv / rate)
        u = float(v) - m.v_ref - m.dv_a(st[0]) - m.dv_b(st[1])
        umin = min(umin, u)
        if cross is None and u <= u_half:
            cross = float(v)
    w = float(m.p.hole_width)
    out["V_G_half_down_sweep_V"] = cross
    out["lowest_u_down_sweep_V"] = umin
    out["largest_s_h_over_h_down_sweep"] = 1.0 / (1.0 + float(np.exp(min((umin / m.m - onset) / w, 700.0))))
    out["u_at_minus_10V_erased_V"] = -10.0 - m.v_ref - dv0
    return out


def main():
    z = fm.load_parameters()
    m = fm.cell(z)
    out = {"label": "MODEL CALCULATION: calibrated model, no fit", "A_read_gate": read_gate(m, fm.inputs()),
           "B_hole_onset": hole_onset(m, float(np.round(z, 4)[fm.NAMES.index("log10_rate")]))}
    (ROOT / "results" / "read_gate.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
