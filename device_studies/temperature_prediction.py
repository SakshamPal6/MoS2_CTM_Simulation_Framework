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
KB = 8.617333262e-5
T_REF = 300.0
EA_SCAN = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
T_C = (25.0, 50.0, 75.0, 100.0, 125.0)
WIDTHS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1)
EA = {"v": 0.0}


def arrhenius(m, dev):
    if EA["v"]:
        m.nux = m.nux * math.exp(-EA["v"] / KB * (1.0 / m.T - 1.0 / T_REF))
    return m


fm.HOOKS.append(arrhenius)


def bound():
    z = fm.load_parameters()
    inp, r = fm.inputs(), fm.readout()
    rows = {}
    for ea in EA_SCAN:
        EA["v"] = ea
        rows["%g" % ea] = fm.blocks(fm.residuals(z, inp, r))
        print(ea, rows["%g" % ea], flush=True)
    EA["v"] = 0.0
    base = rows["0"]["total"]
    allowed = [ea for ea in EA_SCAN if rows["%g" % ea]["total"] - base < 4.0]
    out = {"label": "MODEL CALCULATION: chi2 by block against the activation energy of the hopping rate, all fitted parameters frozen",
           "E_a_assumed_scan_eV": list(EA_SCAN), "rows": rows, "E_a_bound_eV": max(allowed),
           "rule": "largest scanned E_a with total chi2 rise < 4 over E_a = 0"}
    (OUT / "temperature_bound.json").write_text(json.dumps(out, indent=1))
    print("bound", out["E_a_bound_eV"])


def predict():
    z = fm.load_parameters()
    bnd = json.loads((OUT / "temperature_bound.json").read_text())["E_a_bound_eV"]
    eas = sorted({0.0, bnd})
    out = {"label": "MODEL PREDICTION: window of the crossbar cell (erase +10 V, program -10 V, read 1 s after each pulse) against "
                    "pulse width and temperature, inside the calibrated range 25-125 C",
           "widths_s": list(WIDTHS), "T_C": list(T_C), "E_a_eV": eas, "window_V": {}}
    m300 = fm.cell(z)
    EA["v"] = 0.0
    out["window_V_300K_set_cell_1us"] = round(fm.window(m300, 1e-6), 4)
    for ea in eas:
        EA["v"] = ea
        for tc in T_C:
            m = fm.cell(z, "base", 273.15 + tc)
            out["window_V"]["%g|%g" % (ea, tc)] = [round(fm.window(m, w), 4) for w in WIDTHS]
            print(ea, tc, out["window_V"]["%g|%g" % (ea, tc)], flush=True)
    EA["v"] = 0.0
    w = out["window_V"]
    out["change_25_to_125C_V"] = {"%g|%g" % (ea, wd): round(w["%g|125" % ea][i] - w["%g|25" % ea][i], 4)
                                  for ea in eas for i, wd in enumerate(WIDTHS)}
    (OUT / "temperature_prediction.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["change_25_to_125C_V"], indent=1))


def diagnose():
    z = fm.load_parameters()
    out = {"label": "MODEL CALCULATION: levels of the crossbar cell (dV, V) against temperature, E_a = 0", "rows": {}}
    EA["v"] = 0.0
    for tc in T_C:
        m = fm.cell(z, "base", 273.15 + tc)
        er0 = m.hold(m.fresh(), 10.0, fm.ERASE_S)
        er = m.hold(er0, 0.0, fm.REST_S)
        row = {"fresh": m.dv(m.fresh()), "erase_end": m.dv(er0), "erase_after_rest": m.dv(er)}
        for w in (1e-6, 0.1):
            pg0 = m.hold(er, -10.0, w)
            pg = m.hold(pg0, 0.0, fm.REST_S)
            row["program_end_%g" % w], row["program_after_rest_%g" % w] = m.dv(pg0), m.dv(pg)
            row["window_%g" % w] = m.dv(er) - m.dv(pg)
        out["rows"]["%g" % tc] = {k: round(float(v), 4) for k, v in row.items()}
        print(tc, out["rows"]["%g" % tc], flush=True)
    out["erase_s"], out["rest_s"] = fm.ERASE_S, fm.REST_S
    (OUT / "temperature_levels.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"bound": bound, "predict": predict, "diagnose": diagnose}[sys.argv[1]]()
