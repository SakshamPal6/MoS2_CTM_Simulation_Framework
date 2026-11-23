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

V_PROG = [round(v, 1) for v in np.arange(-4.0, -10.01, -0.5)]
WIDTH, REST, HOLDS, VD = 1e-6, 1.0, [10.0, 100.0, 600.0], 1.0


def main():
    t0 = time.time()
    z = fm.load_parameters()
    m = fm.cell(z)
    r = fm.readout()
    er = m.hold(m.hold(m.fresh(), 10.0, fm.ERASE_S), 0.0, REST)
    cur = lambda st: float(r.current(0.0, VD, m.dv(st)))
    rows = [{"v_prog_V": None, "label": "erased", "dV_V": m.dv(er), "I_A": {"1": cur(er)}}]
    st_er = er
    prev = 0.0
    for h in HOLDS:
        st_er = m.hold(st_er, 0.0, h - prev)
        prev = h
        rows[0]["I_A"]["%g" % (h + REST)] = cur(st_er)
    for v in V_PROG:
        st = m.hold(m.hold(er, v, WIDTH), 0.0, REST)
        row = {"v_prog_V": v, "dV_V": m.dv(st), "I_A": {"1": cur(st)}}
        prev = 0.0
        for h in HOLDS:
            st = m.hold(st, 0.0, h - prev)
            prev = h
            row["I_A"]["%g" % (h + REST)] = cur(st)
        rows.append(row)
        print("%5.1f V  dV %+.3f V  I(1 s) %.3e A  I(601 s) %.3e A" % (v, row["dV_V"], row["I_A"]["1"], row["I_A"]["601"]),
              flush=True)
    out = {"label": "MODEL PREDICTION: weight levels of the calibrated cell (erase +10 V 100 ms, rest 1 s, one 1 us pulse)",
           "registration": "", "width_s": WIDTH, "read": {"V_G": 0.0, "V_D": VD},
           "levels": rows, "seconds": round(time.time() - t0, 1)}
    (ROOT / "results" / "cell_levels.json").write_text(json.dumps(out, indent=1))
    print("done in %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
