from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N, LAMBDAS, TOL, INNER, REFINE_TOL, MAX_REF, MAX_NEWTON = 16, (1.0, 5.0), 1e-9, 25, 1e-3, 50, 30
SIGMAS, DRAWS, SEED = (0.05, 0.10), 10, 0
SLICES = int(sys.argv[sys.argv.index("--slices") + 1]) if "--slices" in sys.argv else 1
VERIFY_TOL = 0.0


def laplacian(n):
    h = 1.0 / (n + 1)
    T = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    return (np.kron(np.eye(n), T) + np.kron(T, np.eye(n))) / h ** 2


A = laplacian(N)


def F(u, lam):
    return A @ u - lam * np.exp(u)


def J(u, lam):
    return A - lam * np.diag(np.exp(u))


class Cells:

    def __init__(self, levels_S, dv_V, i_of):
        self.g, self.dv, self.i_of = levels_S, dv_V, i_of
        a_i, b_i = np.meshgrid(np.arange(len(levels_S)), np.arange(len(levels_S)), indexing="ij")
        d = (levels_S[a_i] - levels_S[b_i]).ravel()
        o = np.argsort(d)
        self.ds, self.ai, self.bi = d[o], a_i.ravel()[o], b_i.ravel()[o]

    def _pairs(self, t):
        k = np.clip(np.searchsorted(self.ds, t), 1, len(self.ds) - 1)
        k = np.where(np.abs(self.ds[k - 1] - t) <= np.abs(self.ds[k] - t), k - 1, k)
        return self.ai[k], self.bi[k]

    def write(self, M, sigma=0.0, rng=None):
        span = self.g.max() - self.g.min()
        s = np.abs(M).max() / span
        held = np.zeros_like(M)
        cells = []
        rest = M / s
        for k in range(SLICES):
            scale = 8.0 ** (-k)
            a, b = self._pairs(rest / scale)
            ga, gb = self.g[a], self.g[b]
            if VERIFY_TOL > 0:
                ga = ga * (1 + np.where(a > 0, rng.uniform(-VERIFY_TOL, VERIFY_TOL, a.shape), 0.0))
                gb = gb * (1 + np.where(b > 0, rng.uniform(-VERIFY_TOL, VERIFY_TOL, b.shape), 0.0))
            elif sigma > 0:
                ga = self.i_of(self.dv[a] + rng.normal(0, sigma, a.shape))
                gb = self.i_of(self.dv[b] + rng.normal(0, sigma, b.shape))
            part = (ga - gb) * scale
            held += part * s
            rest = rest - (self.g[a] - self.g[b]) * scale
            cells += [a.ravel(), b.ravel()]
        return held, np.concatenate(cells)


def cg(M, b, steps):
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rr = r @ r
    mv = 0
    for _ in range(steps):
        Mp = M @ p
        mv += 1
        den = p @ Mp
        if den <= 0 or not np.isfinite(den):
            break
        a = rr / den
        x += a * p
        r -= a * Mp
        rn = r @ r
        if rn < 1e-30:
            break
        p = r + (rn / rr) * p
        rr = rn
    return x, mv


def newton(lam, cells=None, sigma=0.0, rng=None):
    u = np.zeros(N * N)
    stats = {"newton": 0, "refinements": 0, "matvec": 0, "writes": 0, "programmed_cells": [], "levels_used": []}
    for it in range(MAX_NEWTON):
        f = F(u, lam)
        if np.max(np.abs(f)) < TOL:
            break
        Jx = J(u, lam)
        if cells is None:
            d = np.linalg.solve(Jx, -f)
        else:
            Jh, lv = cells.write(Jx, sigma, rng)
            stats["writes"] += 1
            stats["programmed_cells"].append(int(np.sum(lv > 0)))
            stats["levels_used"].append(int(len(set(lv[lv > 0].tolist()))))
            d = np.zeros_like(u)
            for _ in range(MAX_REF):
                r = -f - Jx @ d
                if np.linalg.norm(r) < REFINE_TOL * np.linalg.norm(f):
                    break
                c, mv = cg(Jh, r, INNER)
                d += c
                stats["refinements"] += 1
                stats["matvec"] += mv
        u = u + d
        stats["newton"] += 1
        if not np.all(np.isfinite(u)):
            break
    stats["final_residual"] = float(np.max(np.abs(F(u, lam)))) if np.all(np.isfinite(u)) else float("inf")
    stats["converged"] = bool(stats["final_residual"] < TOL)
    stats["u_max"] = float(np.max(u)) if np.all(np.isfinite(u)) else float("nan")
    return stats


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
    i_of = lambda x: 10 ** np.interp(x, grid, lg) / vd
    cells = Cells(g, dv, i_of)
    we = json.loads((ROOT / "results" / "write_energy.json").read_text())
    e10 = we["channel_half_CV2_fJ"] + we["gate_line_half_CV2_fJ"]
    if "--verify" in sys.argv:
        return verify_runs(cells, e10, vprog)
    out = {"label": "MODEL PREDICTION: Newton-Raphson with the Jacobian in model cells",
           "problem": "Bratu, %d x %d" % (N, N), "slices": SLICES}
    for lam in LAMBDAS:
        res = {"digital": newton(lam)}
        ideal = newton(lam, cells)
        rows = 2 * SLICES * N * N
        ideal["write_time_ms_per_write"] = float(np.mean([rows * k * 1e-3 for k in ideal["levels_used"]])) if ideal["levels_used"] else 0
        ideal["write_energy_nJ_per_write"] = float(np.mean([c for c in ideal["programmed_cells"]]) * np.mean(e10 * (vprog[vprog < 0] / 10) ** 2) * 1e-6) if ideal["programmed_cells"] else 0
        res["ideal_cells"] = ideal
        rng = np.random.default_rng(SEED)
        for sg in SIGMAS:
            runs = [newton(lam, cells, sg, rng) for _ in range(DRAWS)]
            res["spread_%.2f_V" % sg] = {"converged_fraction": float(np.mean([q["converged"] for q in runs])),
                                         "newton_mean": float(np.mean([q["newton"] for q in runs])),
                                         "refinements_mean": float(np.mean([q["refinements"] for q in runs])),
                                         "final_residual_max": float(np.max([q["final_residual"] for q in runs]))}
        for k in ("digital", "ideal_cells"):
            res[k].pop("programmed_cells", None) if k == "digital" else None
        out["lambda_%g" % lam] = res
        print("lambda", lam, json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("programmed_cells", "levels_used")}
                                         for k, v in res.items()}, indent=1), flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    name = "newton_solver_bratu.json" if SLICES == 1 else "newton_solver_bratu_slices%d.json" % SLICES
    (ROOT / "results" / name).write_text(json.dumps(out, indent=1))


def verify_runs(cells, e10, vprog):
    global VERIFY_TOL
    out = {"label": "MODEL PREDICTION: Newton-Raphson, Jacobian in model cells with write-and-verify", "slices": SLICES, "verify_pulses_per_cell": 3}
    t0 = time.time()
    for lam in LAMBDAS:
        res = {"digital_newton": newton(lam)["newton"]}
        for tol in (0.01, 0.02, 0.05):
            VERIFY_TOL = tol
            rng = np.random.default_rng(SEED)
            runs = [newton(lam, cells, 0.0, rng) for _ in range(DRAWS)]
            ok = [q for q in runs if q["converged"]]
            cells_per_write = float(np.mean([c for q in runs for c in q["programmed_cells"]]))
            e_write = cells_per_write * (1 + 3) * float(np.mean(e10 * (vprog[vprog < 0] / 10) ** 2)) * 1e-6
            res["tol_%g" % tol] = {"converged_fraction": len(ok) / len(runs),
                                   "newton_mean": float(np.mean([q["newton"] for q in ok])) if ok else None,
                                   "refinements_mean": float(np.mean([q["refinements"] for q in ok])) if ok else None,
                                   "matvec_mean": float(np.mean([q["matvec"] for q in ok])) if ok else None,
                                   "final_residual_max": float(np.max([q["final_residual"] for q in runs])),
                                   "write_energy_nJ_per_write_with_verify": e_write,
                                   "write_time_ms_per_write_with_verify": float(np.mean(
                                       [2 * SLICES * N * N * (k + 3) * 1e-3 for q in runs for k in q["levels_used"]]))}
        VERIFY_TOL = 0.0
        out["lambda_%g" % lam] = res
        print("lambda", lam, json.dumps(res, indent=1), flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    (ROOT / "results" / "newton_solver_bratu_verified.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
