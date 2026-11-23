from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT
sys.path.insert(0, str(PKG))
from model import final_model as fm

VP = [round(v, 1) for v in np.arange(-6.0, -11.01, -0.5)]
WIDTHS = np.logspace(-7, -2, 11)
RATE_S = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
BLOCK_NM = [15, 20, 25, 30, 40]
VREAD = [round(v, 1) for v in np.arange(-3.0, 3.01, 0.5)]
E10 = 426.3


def window(m, v, t):
    er = m.hold(m.hold(m.fresh(), 10.0, fm.ERASE_S), 0.0, 1.0)
    pg = m.hold(m.hold(er, v, t), 0.0, 1.0)
    return float(m.dv(er) - m.dv(pg))


def main():
    t0 = time.time()
    z = fm.load_parameters()
    out = {"label": "MODEL PREDICTION: design maps of the calibrated cell"}
    m = fm.cell(z)
    A = np.array([[window(m, v, t) for t in WIDTHS] for v in VP])
    out["a"] = {"V_P": VP, "width_s": WIDTHS.tolist(), "window_V": A.tolist(),
                "energy_fJ": [E10 * (v / 10) ** 2 for v in VP], "extrapolated_V": [v for v in VP if abs(v) > 10]}
    i1 = int(np.argmin(np.abs(np.log10(WIDTHS) + 6)))
    for target in (2.0, 3.0):
        ok = [k for k, v in enumerate(VP) if A[k, i1] >= target]
        out["a"]["design_%gV_window_1us" % target] = ({"V_P": VP[min(ok, key=lambda k: abs(VP[k]))],
                                                      "energy_fJ": E10 * (VP[min(ok, key=lambda k: abs(VP[k]))] / 10) ** 2}
                                                     if ok else None)
    print("a done", out["a"]["design_2V_window_1us"], out["a"]["design_3V_window_1us"], flush=True)
    B = np.zeros((len(RATE_S), len(BLOCK_NM)))
    for j, nm in enumerate(BLOCK_NM):
        key = "blk%d" % nm
        fm.DEV[key] = {"stack": ((nm * 1e-9, 10.0), (3e-9, 25.0), (7e-9, 10.0)), "ss": fm.DEV["base"]["ss"]}
        for i, s in enumerate(RATE_S):
            def hook(c, dev, s=s):
                c.nux = c.nux * s
                return c
            fm.HOOKS[:] = [hook]
            B[i, j] = window(fm.cell(z, dev=key), -10.0, 1e-6)
            fm.HOOKS[:] = []
    out["b"] = {"rate_scale": RATE_S, "density_ratio": [s ** (1 / 3.1) for s in RATE_S], "blocking_nm": BLOCK_NM,
                "window_1us_V": B.tolist(), "calibrated_blocking_nm": [15, 40]}
    ok = [(i, j) for i in range(len(RATE_S)) for j in range(len(BLOCK_NM)) if B[i, j] >= 3.0]
    out["b"]["design_3V"] = ([{"density_ratio": RATE_S[i] ** (1 / 3.1), "blocking_nm": BLOCK_NM[j]} for i, j in
                              sorted(ok, key=lambda q: (BLOCK_NM[q[1]], RATE_S[q[0]]))][:3] if ok else None)
    print("b done", out["b"]["design_3V"], flush=True)
    r = fm.readout()
    m = fm.cell(z)
    er = m.hold(m.hold(m.fresh(), 10.0, fm.ERASE_S), 0.0, 1.0)
    C = {}
    for vp in (-10.0, -8.5):
        st0 = m.hold(m.hold(er, vp, 1e-6), 0.0, 1.0)
        ref = m.hold(st0, 0.0, 2.0)
        lref = np.log10(r.current(0.0, 1.0, m.dv(ref)))
        row = []
        for vr in VREAD:
            st = st0
            for _ in range(1000):
                st = m.hold(m.hold(st, vr, 1e-3), 0.0, 1e-3)
            row.append(float(np.log10(r.current(0.0, 1.0, m.dv(st))) - lref))
        C["%.1f" % vp] = row
    safe = [v for k, v in enumerate(VREAD) if all(abs(C[s][k]) < 0.05 for s in C)]
    out["c"] = {"V_read": VREAD, "dlog10I_after_1000_reads": C, "safe_window_V": [min(safe), max(safe)] if safe else None}
    print("c done", out["c"]["safe_window_V"], flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    (ROOT / "results" / "design_maps.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
