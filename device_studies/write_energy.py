from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from device_studies._io import write_json

EPS0 = 8.8541878128e-12


def permittivity_range():
    rows = {}
    for ka, kh in ((10.0, 25.0), (8.0, 25.0), (10.0, 16.0), (8.0, 16.0)):
        c = 1e-12 / sum(t / (EPS0 * k) for t, k in ((15e-9, ka), (3e-9, kh), (7e-9, ka)))
        rows["Al2O3 %g, HfO2 %g" % (ka, kh)] = round(float(0.5 * c * 100 * 1e15), 1)
    v = list(rows.values())
    return {"half_CV2_fJ": rows, "range_fJ": [min(v), max(v)]}


def main():
    v = 10.0
    c_ox_reported = 3.8e-15
    stack = [(15e-9, 10.0), (3e-9, 25.0), (7e-9, 10.0)]
    c_area = 1.0 / sum(t / (EPS0 * k) for t, k in stack)
    c_ox_calc = c_area * 1e-12
    pitch = 676e-12 ** 0.5
    c_line = EPS0 * 3.9 / 285e-9 * 1.5e-6 * pitch
    e_low = 0.5 * c_ox_reported * v ** 2
    e_line = 0.5 * c_line * v ** 2
    e_leak = 0.2e-15
    out = {"label": "CALCULATION; inputs reported for the crossbar cell",
           "inputs": {"C_ox_stated_fF": 3.8, "C_ox_from_stack_fF": round(c_ox_calc * 1e15, 3), "V_write_V": v,
                      "line_width_um": 1.5, "SiO2_nm": 285, "cell_area_um2": 676, "cell_pitch_um_assumed": round(pitch * 1e6, 2)},
           "read_gate_energy_check_fJ": round(0.5 * c_ox_reported * 2.0 ** 2 * 1e15, 2),
           "channel_half_CV2_fJ": round(e_low * 1e15, 1), "channel_CV2_from_supply_fJ": round(2 * e_low * 1e15, 1),
           "gate_line_to_substrate_fF_per_cell": round(c_line * 1e15, 2), "gate_line_half_CV2_fJ": round(e_line * 1e15, 1),
           "range_fJ": [round(e_low * 1e15, 0), round((e_low + e_line) * 1e15, 0)],
           "ratio_to_stated_0.2fJ": [round(e_low / e_leak), round((e_low + e_line) / e_leak)],
           "channel_term_permittivity_range": permittivity_range()}
    write_json("write_energy.json", out)
    print(json.dumps({k: out[k] for k in ("channel_half_CV2_fJ", "gate_line_half_CV2_fJ", "range_fJ", "ratio_to_stated_0.2fJ")}))


if __name__ == "__main__":
    main()
