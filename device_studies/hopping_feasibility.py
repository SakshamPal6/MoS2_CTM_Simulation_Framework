from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from device_studies._io import write_json

Q, HBAR, M0, KT = 1.602176634e-19, 1.054571817e-34, 9.1093837015e-31, 0.025852
N_REQ = 1.2e13
T_PULSE = 1e-6
D_TUN = 7e-9


def hopping(n_cm3, phi_ev, m_rel, field, nu=1e13):
    r = (n_cm3 * 1e6) ** (-1.0 / 3.0)
    kappa = math.sqrt(2 * m_rel * M0 * phi_ev * Q) / HBAR
    k = nu * math.exp(-2 * kappa * r)
    hops = max(D_TUN / r, 1.0)
    occ_per_layer_cm2 = n_cm3 * r * 1e2
    flux = occ_per_layer_cm2 * k
    return {"spacing_nm": round(r * 1e9, 2), "hop_rate_s": float("%.3g" % k), "transit_s": float("%.3g" % (hops / k)),
            "charge_in_1us_cm2": float("%.3g" % (flux * T_PULSE)), "feasible": bool(flux * T_PULSE >= N_REQ and hops / k <= T_PULSE)}


def ion_drift(ea_ev, z, field, a_nm=0.3, nu=1e13):
    a = a_nm * 1e-9
    v = 2 * a * nu * math.exp(-ea_ev / KT) * math.sinh(z * a * field / (2 * KT))
    return {"velocity_m_s": float("%.3g" % v), "distance_in_1us_nm": float("%.3g" % (v * T_PULSE * 1e9))}


def ion_needed_distance(n_cm2, k=25.0):
    eps0 = 8.8541878128e-12
    return 3.45 * eps0 * k / (Q * n_cm2 * 1e4) * 1e9


def run():
    out = {"label": "CALCULATION (feasibility bounds), no fit", "hopping": {}, "ions": {}}
    for n in (2e19, 5e19, 1e20, 3e20):
        for phi in (1.6, 1.9, 2.3, 2.7):
            for m in (0.2, 0.23):
                out["hopping"]["N%.0e_phi%.1f_m%.2f" % (n, phi, m)] = hopping(n, phi, m, 5.8e8)
    for ea in (0.3, 0.5, 0.7, 1.0, 1.5):
        out["ions"]["Ea%.1f_z2_HfO2" % ea] = ion_drift(ea, 2, 1.7e8)
        out["ions"]["Ea%.1f_z1_Al2O3" % ea] = ion_drift(ea, 1, 5.8e8)
    out["ions"]["needed_distance_nm"] = {"N1e13_HfO2": round(ion_needed_distance(1e13), 2),
                                         "N1e14_HfO2": round(ion_needed_distance(1e14), 2),
                                         "N1e14_Al2O3": round(ion_needed_distance(1e14, 10.0), 2)}
    out["hopping_feasible_cases"] = [k for k, v in out["hopping"].items() if v["feasible"]]
    write_json("hopping_feasibility.json", out)
    print(json.dumps(out["hopping_feasible_cases"]))


if __name__ == "__main__":
    run()
