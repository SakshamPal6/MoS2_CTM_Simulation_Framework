from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "applications"))
import newton_solver_bratu as e3

e3.SLICES = 3
TOL_V, DRAWS, SEED = 0.01, 10, 0
E_OP_J, T_READ, VINS, ADC_RATIO = 10e-12, 1e-6, (1.0,), 14.4
N2 = e3.N * e3.N
AS = sp.csr_matrix(e3.A)
NNZ = AS.nnz


def write_rows(cells, M, rng):
    span = cells.g.max() - cells.g.min()
    s = np.abs(M).max() / span
    held = np.zeros_like(M)
    gin = np.zeros(M.shape[1])
    lv = []
    rest = M / s
    for k in range(e3.SLICES):
        scale = 8.0 ** (-k)
        a, b = cells._pairs(rest / scale)
        ga, gb = cells.g[a], cells.g[b]
        ga = ga * (1 + np.where(a > 0, rng.uniform(-TOL_V, TOL_V, a.shape), 0.0))
        gb = gb * (1 + np.where(b > 0, rng.uniform(-TOL_V, TOL_V, b.shape), 0.0))
        held += (ga - gb) * scale * s
        gin += (ga + gb).sum(0)
        rest = rest - (cells.g[a] - cells.g[b]) * scale
        lv += [a.ravel(), b.ravel()]
    return held, gin, np.concatenate(lv)


def crossbar_newton(lam, cells, rng):
    u = np.zeros(N2)
    st = {"newton": 0, "matvec": 0, "refinements": 0, "cg_steps": 0, "read_J": {v: 0.0 for v in VINS}, "programmed": [],
          "levels": [], "ops_digital": 0}
    for _ in range(e3.MAX_NEWTON):
        f = e3.F(u, lam)
        st["ops_digital"] += 2 * NNZ + 5 * N2
        if np.max(np.abs(f)) < e3.TOL:
            break
        Jx = e3.J(u, lam)
        Jh, gin, lv = write_rows(cells, Jx, rng)
        st["programmed"].append(int(np.sum(lv > 0)))
        st["levels"].append(int(len(set(lv[lv > 0].tolist()))))
        d = np.zeros_like(u)
        for _ in range(e3.MAX_REF):
            r = -f - Jx @ d
            st["ops_digital"] += 2 * NNZ + 2 * N2
            if np.linalg.norm(r) < e3.REFINE_TOL * np.linalg.norm(f):
                break
            x = np.zeros_like(r)
            rr_, p = r.copy(), r.copy()
            rr = rr_ @ rr_
            for _ in range(e3.INNER):
                Mp = Jh @ p
                st["matvec"] += 1
                w = p / np.abs(p).max()
                for v in VINS:
                    st["read_J"][v] += T_READ * float(np.sum(np.abs(w) * v ** 2 * gin))
                st["cg_steps"] += 1
                st["ops_digital"] += 10 * N2
                den = p @ Mp
                if den <= 0 or not np.isfinite(den):
                    break
                a = rr / den
                x += a * p
                rr_ -= a * Mp
                rn = rr_ @ rr_
                if rn < 1e-30:
                    break
                p = rr_ + (rn / rr) * p
                rr = rn
            d += x
            st["refinements"] += 1
        u = u + d
        st["newton"] += 1
    st["converged"] = bool(np.max(np.abs(e3.F(u, lam))) < e3.TOL)
    return st


def digital_newton(lam):
    u, ops, it_n, it_cg = np.zeros(N2), 0, 0, 0
    for _ in range(e3.MAX_NEWTON):
        f = e3.F(u, lam)
        ops += 2 * NNZ + 5 * N2
        if np.max(np.abs(f)) < e3.TOL:
            break
        Jx = sp.csr_matrix(AS - lam * sp.diags(np.exp(u)))
        ops += N2
        b = -f
        x = np.zeros(N2)
        r = b.copy()
        p = r.copy()
        rr = r @ r
        while np.sqrt(rr) > 1e-10 * np.linalg.norm(b):
            Mp = Jx @ p
            a = rr / (p @ Mp)
            x += a * p
            r -= a * Mp
            rn = r @ r
            p = r + (rn / rr) * p
            rr = rn
            ops += 2 * Jx.nnz + 10 * N2
            it_cg += 1
        u = u + x
        it_n += 1
    assert np.max(np.abs(e3.F(u, lam))) < e3.TOL
    return {"newton": it_n, "cg_steps": it_cg, "ops": ops, "energy_J": ops * E_OP_J}


def main():
    t0 = time.time()
    sys.path.insert(0, str(ROOT))
    from model import final_model as fm
    lev = json.loads((ROOT / "results" / "cell_levels.json").read_text())
    vd = lev["read"]["V_D"]
    g = np.array([L["I_A"]["1"] for L in lev["levels"]]) / vd
    dv = np.array([L["dV_V"] for L in lev["levels"]])
    vprog = np.array([0.0 if L["v_prog_V"] is None else L["v_prog_V"] for L in lev["levels"]])
    r = fm.readout()
    grid = np.linspace(dv.min() - 1, dv.max() + 1, 801)
    lg = np.log10([r.current(0.0, vd, x) for x in grid])
    cells = e3.Cells(g, dv, lambda x: 10 ** np.interp(x, grid, lg) / vd)
    we = json.loads((ROOT / "results" / "write_energy.json").read_text())
    e_pulse_J = float(np.mean((we["channel_half_CV2_fJ"] + we["gate_line_half_CV2_fJ"]) * (vprog[vprog < 0] / 10) ** 2)) * 1e-15
    ref = json.loads((ROOT / "results" / "newton_solver_bratu_verified.json").read_text())
    out = {"label": "MODEL PREDICTION (crossbar) + CALCULATION (digital at 10 pJ/op, yao2020): energy of one Newton-Raphson "
                    "solution", "digital_energy_per_op_J": E_OP_J, "read_time_s": T_READ}
    for lam in e3.LAMBDAS:
        rng = np.random.default_rng(SEED)
        runs = [crossbar_newton(lam, cells, rng) for _ in range(DRAWS)]
        ok = [q for q in runs if q["converged"]]
        assert len(ok) == DRAWS
        m = lambda k: float(np.mean([q[k] for q in ok]))
        assert abs(m("matvec") - ref["lambda_%g" % lam]["tol_0.01"]["matvec_mean"]) < 1e-9, (m("matvec"), ref)
        e_write = float(np.mean([sum(c * 4 * e_pulse_J for c in q["programmed"]) for q in ok]))
        t_write = float(np.mean([sum(2 * e3.SLICES * N2 * (k + 3) * 1e-6 for k in q["levels"]) for q in ok]))
        e_dig_part = m("ops_digital") * E_OP_J
        dig = digital_newton(lam)
        dense = m("matvec") * 2 * N2 * N2 * E_OP_J + e_dig_part
        res = {"newton": m("newton"), "matvec": m("matvec"), "cg_steps": m("cg_steps"),
               "crossbar_write_J": e_write, "crossbar_digital_part_J": e_dig_part,
               "crossbar_time_s": t_write + m("matvec") * T_READ, "crossbar_write_time_s": t_write,
               "digital_sparse": dig, "digital_dense_same_algorithm_J": dense}
        for v in VINS:
            rd = float(np.mean([q["read_J"][v] for q in ok]))
            tot = e_write + rd + e_dig_part
            tot_p = e_write + rd * (1 + ADC_RATIO) + e_dig_part
            res["Vin_%g" % v] = {"crossbar_read_J": rd, "crossbar_total_J": tot, "crossbar_total_with_periphery_J": tot_p,
                                 "write_share": e_write / tot, "ratio_digital_sparse_over_crossbar": dig["energy_J"] / tot,
                                 "ratio_digital_sparse_over_crossbar_with_periphery": dig["energy_J"] / tot_p,
                                 "ratio_digital_dense_over_crossbar": dense / tot}
        out["lambda_%g" % lam] = res
        print("lambda %g: newton %.1f, matvec %.0f | crossbar write %.3g J, read (1 V, read-time encoding) %.3g J, digital part "
              "%.3g J, time %.3g s | digital sparse %.3g J (%d CG steps), dense same-algorithm %.3g J | ratio sparse/crossbar %.2f" % (
                  lam, res["newton"], res["matvec"], e_write, res["Vin_1"]["crossbar_read_J"], e_dig_part,
                  res["crossbar_time_s"], dig["energy_J"], dig["cg_steps"], dense,
                  res["Vin_1"]["ratio_digital_sparse_over_crossbar"]),
              flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    (ROOT / "results" / "newton_solver_energy.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
