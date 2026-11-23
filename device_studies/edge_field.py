from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from device_studies._io import write_json

NM = 1e-9
VG = -10.0
T_MOS2, EPS_MOS2 = 0.65 * NM, 6.0
T_CONTACT = 40 * NM
X_LEFT, X_RIGHT = -300 * NM, 500 * NM
Z_TOP = 2000 * NM


def axis_geometric(a, b, h0, growth, hmax):
    pts, h = [a], h0
    while pts[-1] < b:
        pts.append(min(pts[-1] + h, b))
        h = min(h * growth, hmax)
    return np.array(pts)


def grid(stack):
    t_stack = sum(t for t, _ in stack)
    z0 = np.arange(0.0, t_stack + 1e-15, 0.25 * NM)
    z1 = np.linspace(t_stack, t_stack + T_MOS2, 4)[1:]
    z2 = axis_geometric(t_stack + T_MOS2, Z_TOP, 0.25 * NM, 1.08, 50 * NM)[1:]
    z = np.unique(np.round(np.concatenate([z0, z1, z2]) / 1e-13) * 1e-13)
    xr = axis_geometric(0.0, X_RIGHT, 0.2 * NM, 1.06, 10 * NM)
    xl = -axis_geometric(0.0, -X_LEFT, 0.2 * NM, 1.06, 10 * NM)[1:][::-1]
    return np.concatenate([xl, xr]), z, t_stack


def eps_at(xv, zv, stack, t_stack):
    if zv < t_stack:
        acc = 0.0
        for t, k in stack:
            if zv < acc + t:
                return k
            acc += t
        return stack[-1][1]
    if zv <= t_stack + T_MOS2 + 1e-15:
        return EPS_MOS2
    return 1.0


def solve(stack, channel_conductor):
    x, z, t_stack = grid(stack)
    nx, nz = len(x), len(z)
    idx = np.arange(nx * nz).reshape(nz, nx)
    eps = np.array([[eps_at(xv, zv, stack, t_stack) for xv in x] for zv in z])
    fixed = np.zeros((nz, nx), bool)
    val = np.zeros((nz, nx))
    fixed[0, :] = True
    val[0, :] = VG
    zc0 = t_stack
    contact = (x[None, :] <= 0.0) & (z[:, None] >= zc0 - 1e-15) & (z[:, None] <= zc0 + T_MOS2 + T_CONTACT + 1e-15)
    fixed |= contact
    if channel_conductor:
        chan = (x[None, :] > 0.0) & (np.abs(z[:, None] - (zc0 + 0.5 * T_MOS2)) <= 0.5 * T_MOS2 + 1e-15)
        fixed |= chan
    rows, cols, data = [], [], []
    b = np.zeros(nx * nz)
    for j in range(nz):
        for i in range(nx):
            p = idx[j, i]
            if fixed[j, i]:
                rows.append(p); cols.append(p); data.append(1.0); b[p] = val[j, i]
                continue
            hxm = x[i] - x[i - 1] if i > 0 else None
            hxp = x[i + 1] - x[i] if i < nx - 1 else None
            hzm = z[j] - z[j - 1] if j > 0 else None
            hzp = z[j + 1] - z[j] if j < nz - 1 else None
            wx = 0.5 * ((hxm or 0) + (hxp or 0))
            wz = 0.5 * ((hzm or 0) + (hzp or 0))
            diag = 0.0
            for (jj, ii, h, w) in ((j, i - 1, hxm, wz), (j, i + 1, hxp, wz), (j - 1, i, hzm, wx), (j + 1, i, hzp, wx)):
                if h is None:
                    continue
                e_face = 2.0 * eps[j, i] * eps[jj, ii] / (eps[j, i] + eps[jj, ii])
                c = e_face * w / h
                rows.append(p); cols.append(idx[jj, ii]); data.append(-c)
                diag += c
            rows.append(p); cols.append(p); data.append(diag)
    A = sp.csr_matrix((data, (rows, cols)), shape=(nx * nz, nx * nz))
    phi = spla.spsolve(A, b).reshape(nz, nx)
    t_b, t_h, t_t = stack[0][0], stack[1][0], stack[2][0]
    j0 = int(np.argmin(np.abs(z - (t_b + t_h))))
    j1 = int(np.argmin(np.abs(z - (t_b + t_h + t_t))))
    f_t = (phi[j1, :] - phi[j0, :]) / (z[j1] - z[j0])
    return x, z, phi, f_t, t_stack


def analytic_1d(stack):
    k_t = stack[-1][1]
    return abs(VG) / sum(t * k_t / k for t, k in stack)


def run():
    stacks = {"base_15_3_7": ((15 * NM, 10.0), (3 * NM, 25.0), (7 * NM, 10.0)),
              "s2023_40_3_7": ((40 * NM, 10.0), (3 * NM, 25.0), (7 * NM, 10.0))}
    res, prof = {}, {}
    for case, cond in (("depleted", False), ("conductor_1D_ref", True)):
        for name, st in stacks.items():
            x, z, phi, f_t, t_stack = solve(st, cond)
            ch = x > 0
            xm = x[ch]
            fm = np.abs(f_t[ch])
            w = np.gradient(xm)
            res[(case, name)] = {"max_abs_F_MV_cm": round(float(fm.max()) / 1e8, 3),
                                 "x_at_max_nm": None if fm.min() >= 0.999 * fm.max() else round(float(xm[np.argmax(fm)]) / NM, 1),
                                 "mean_abs_F_MV_cm": round(float(np.sum(fm * w) / np.sum(w)) / 1e8, 3),
                                 "F_centre_MV_cm": round(float(fm[-1]) / 1e8, 3),
                                 "F_under_contact_MV_cm": round(float(np.abs(f_t[x < -50 * NM]).mean()) / 1e8, 3),
                                 "width_F_ge_half_max_nm": round(float(np.sum(w[fm >= 0.5 * fm.max()])) / NM, 1),
                                 "analytic_1D_MV_cm": round(analytic_1d(st) / 1e8, 3)}
            prof[(case, name)] = (xm / NM, fm / 1e8)
            print(case, name, res[(case, name)], flush=True)
    out = {"label": "CALCULATION: two-dimensional electrostatics, no fit", "VG": VG, "cases": {}}
    for case in ("depleted", "conductor_1D_ref"):
        b, s = res[(case, "base_15_3_7")], res[(case, "s2023_40_3_7")]
        out["cases"][case] = {"base": b, "s2023": s, "R_edge": round(s["max_abs_F_MV_cm"] / b["max_abs_F_MV_cm"], 3),
                              "R_mean": round(s["mean_abs_F_MV_cm"] / b["mean_abs_F_MV_cm"], 3)}
    ref = out["cases"]["conductor_1D_ref"]
    out["solver_check_centre_vs_analytic"] = {k: round(ref[k]["F_centre_MV_cm"] / ref[k]["analytic_1D_MV_cm"], 4) for k in ("base", "s2023")}
    sel = lambda xm: np.unique(np.concatenate([np.searchsorted(xm, np.geomspace(0.2, 500, 120)), [len(xm) - 1]]).clip(0, len(xm) - 1))
    out["profiles"] = {"%s|%s" % k: {"x_nm": np.round(v[0][sel(v[0])], 3).tolist(), "F_MV_cm": np.round(v[1][sel(v[0])], 4).tolist()}
                       for k, v in prof.items()}
    write_json("edge_field.json", out)


if __name__ == "__main__":
    run()
