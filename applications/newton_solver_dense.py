from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "applications"))
import newton_solver_bratu as e3
import newton_solver_energy as e2

e3.SLICES = 3
SIZES, BS, DRAWS, SEED = (16, 32, 48), (2.0, 5.0), 10, 0
E_OP, T_READ, V_IN, TOL = 10e-12, 1e-6, 1.0, 1e-9


def kernel(m):
    h = 1.0 / (m + 1)
    g = (np.arange(m) + 1) * h
    X, Y = np.meshgrid(g, g)
    P = np.c_[X.ravel(), Y.ravel()]
    r2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    return h * h / np.sqrt(r2 + h * h)


def digital(K, b):
    n = len(b)
    u, newton, cg_steps, ops_cg = np.zeros(n), 0, 0, 0
    while True:
        f = K @ u + np.exp(u) - b
        ops_cg += 2 * n * n + 5 * n
        if np.max(np.abs(f)) < TOL:
            break
        J = K + np.diag(np.exp(u))
        x, r = np.zeros(n), -f.copy()
        p, rr = r.copy(), r @ r
        while np.sqrt(rr) > 1e-10 * np.linalg.norm(f):
            Mp = J @ p
            a = rr / (p @ Mp)
            x += a * p
            r -= a * Mp
            rn = r @ r
            p = r + (rn / rr) * p
            rr = rn
            cg_steps += 1
            ops_cg += 2 * n * n + 10 * n
        u += x
        newton += 1
    ops_chol = newton * (n ** 3 / 3 + 2 * n * n) + (newton + 1) * (2 * n * n + 5 * n)
    best = min(ops_cg, ops_chol)
    return {"newton": newton, "cg_steps": cg_steps, "ops_cg": ops_cg, "ops_cholesky": ops_chol,
            "best": "CG" if ops_cg <= ops_chol else "Cholesky", "energy_J": best * E_OP}


def crossbar(K, b, cells, rng, e_pulse, k_once=False):
    n = len(b)
    u = np.zeros(n)
    st = {"newton": 0, "matvec": 0, "refinements": 0, "write_J": 0.0, "write_s": 0.0, "read_J": 0.0, "ops": 0}
    Kh = gK = None
    for _ in range(e3.MAX_NEWTON):
        f = K @ u + np.exp(u) - b
        st["ops"] += 2 * n * n + 5 * n
        if np.max(np.abs(f)) < TOL:
            break
        Jx = K + np.diag(np.exp(u))
        if k_once:
            if Kh is None:
                Kh, gK, lv = e2.write_rows(cells, K, rng)
                st["write_J"] += np.sum(lv > 0) * 4 * e_pulse
                st["write_s"] += 2 * e3.SLICES * n * (len(set(lv[lv > 0].tolist())) + 3) * 1e-6
            Jh, gin, dg = Kh, gK, np.exp(u)
        else:
            Jh, gin, lv = e2.write_rows(cells, Jx, rng)
            st["write_J"] += np.sum(lv > 0) * 4 * e_pulse
            st["write_s"] += 2 * e3.SLICES * n * (len(set(lv[lv > 0].tolist())) + 3) * 1e-6
            dg = None
        d = np.zeros(n)
        for _ in range(e3.MAX_REF):
            r = -f - Jx @ d
            st["ops"] += 2 * n * n + 2 * n
            if np.linalg.norm(r) < e3.REFINE_TOL * np.linalg.norm(f):
                break
            x, rv = np.zeros(n), r.copy()
            p, rr = rv.copy(), rv @ rv
            for _ in range(e3.INNER):
                Mp = Jh @ p if dg is None else Jh @ p + dg * p
                st["matvec"] += 1
                st["read_J"] += T_READ * float(np.sum(np.abs(p) / np.abs(p).max() * V_IN ** 2 * gin))
                st["ops"] += 10 * n + (n if dg is not None else 0)
                den = p @ Mp
                if den <= 0 or not np.isfinite(den):
                    break
                a = rr / den
                x += a * p
                rv -= a * Mp
                rn = rv @ rv
                if rn < 1e-30:
                    break
                p = rv + (rn / rr) * p
                rr = rn
            d += x
            st["refinements"] += 1
        u = u + d
        st["newton"] += 1
        if not np.all(np.isfinite(u)):
            break
    st["converged"] = bool(np.all(np.isfinite(u)) and np.max(np.abs(K @ u + np.exp(u) - b)) < TOL)
    st["digital_J"] = st["ops"] * E_OP
    st["total_J"] = st["write_J"] + st["read_J"] + st["digital_J"]
    st["time_s"] = st["write_s"] + st["matvec"] * T_READ
    return st


def main():
    t0 = time.time()
    sys.path.insert(0, str(ROOT))
    from model import final_model as fm
    lev = json.loads((ROOT / "results" / "cell_levels.json").read_text())
    vd = lev["read"]["V_D"]
    g = np.array([L["I_A"]["1"] for L in lev["levels"]]) / vd
    dv = np.array([L["dV_V"] for L in lev["levels"]])
    vprog = np.array([0.0 if L["v_prog_V"] is None else L["v_prog_V"] for L in lev["levels"]])
    rd = fm.readout()
    grid = np.linspace(dv.min() - 1, dv.max() + 1, 801)
    lg = np.log10([rd.current(0.0, vd, x) for x in grid])
    cells = e3.Cells(g, dv, lambda x: 10 ** np.interp(x, grid, lg) / vd)
    we = json.loads((ROOT / "results" / "write_energy.json").read_text())
    e_pulse = float(np.mean((we["channel_half_CV2_fJ"] + we["gate_line_half_CV2_fJ"]) * (vprog[vprog < 0] / 10) ** 2)) * 1e-15
    out = {"label": "MODEL PREDICTION (crossbar) + CALCULATION (digital, 10 pJ/op from yao2020): Newton-Raphson on dense "
                    "systems", "digital_energy_per_op_J": E_OP}
    for m in SIZES:
        K = kernel(m)
        n = m * m
        for bv in BS:
            b = np.full(n, bv)
            dig = digital(K, b)
            res = {"n": n, "digital": dig}
            for name, once in (("rewrite_J", False), ("K_written_once", True)):
                rng = np.random.default_rng(SEED)
                runs = [crossbar(K, b, cells, rng, e_pulse, once) for _ in range(DRAWS)]
                ok = [q for q in runs if q["converged"]]
                mean = lambda k: float(np.mean([q[k] for q in ok])) if ok else None
                row = {"converged_fraction": len(ok) / DRAWS}
                for k in ("newton", "matvec", "refinements", "write_J", "read_J", "digital_J", "total_J", "time_s", "write_s"):
                    row[k] = mean(k)
                if ok:
                    row["ratio_digital_over_crossbar"] = dig["energy_J"] / row["total_J"]
                    row["shares"] = {k: row[k + "_J"] / row["total_J"] for k in ("write", "read", "digital")}
                res[name] = row
            out["n%d_b%g" % (n, bv)] = res
            rj = res["rewrite_J"]
            print("n %d b %g: digital %s %.3g J (newton %d, CG %d) | crossbar conv %.1f, newton %s, ref %s, total %s J, ratio %s, "
                  "time %s s | K once: total %s J, ratio %s, time %s s" % (
                      n, bv, dig["best"], dig["energy_J"], dig["newton"], dig["cg_steps"], rj["converged_fraction"],
                      rj["newton"], rj["refinements"], rj["total_J"], rj.get("ratio_digital_over_crossbar"), rj["time_s"],
                      res["K_written_once"]["total_J"], res["K_written_once"].get("ratio_digital_over_crossbar"),
                      res["K_written_once"]["time_s"]), flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    (ROOT / "results" / "newton_solver_dense.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
