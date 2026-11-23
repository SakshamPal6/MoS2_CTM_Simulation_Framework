from __future__ import annotations

import json
import math
from pathlib import Path

from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
NU0 = 1e13
K_HOP, R_NM = 9.28e8, 1.49
TWO_KAPPA = math.log(NU0 / K_HOP) / R_NM


def density_ratio(s: float) -> float:
    rp = brentq(lambda r: -TWO_KAPPA * (r - R_NM) + math.log(r / R_NM) - math.log(s), 0.2, 10.0)
    return (R_NM / rp) ** 3


def main():
    b = json.loads((ROOT / "results" / "design_maps.json").read_text())["b"]
    exact = [density_ratio(s) for s in b["rate_scale"]]
    for s, e in zip(b["rate_scale"], exact):
        rp = R_NM * e ** (-1 / 3)
        assert abs(math.exp(-TWO_KAPPA * (rp - R_NM)) * rp / R_NM - s) < 1e-9 * s
    W = b["window_1us_V"]
    rows3 = lambda j: [i for i in range(len(W)) if W[i][j] >= 3.0]
    out = {"label": "CALCULATION: exact defect-density axis of design_maps map (b)",
           "two_kappa_r": TWO_KAPPA * R_NM, "rate_scale": b["rate_scale"],
           "density_ratio_exact": exact, "density_ratio_power_law_design_maps": b["density_ratio"],
           "local_exponent_d_ln_nu_d_ln_N": (TWO_KAPPA * R_NM - 1) / 3,
           "min_density_ratio_3V_15nm": min(exact[i] for i in rows3(0)),
           "min_density_ratio_3V_20nm": min(exact[i] for i in rows3(1)) if rows3(1) else None}
    (ROOT / "results" / "design_map_density_axis.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
