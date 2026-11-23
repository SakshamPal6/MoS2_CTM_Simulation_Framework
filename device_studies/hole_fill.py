from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from device_studies import edge_field as e99

OUT = ROOT / "results" / "hole_fill.json"
Q, KB, H, HBAR, M0, EPS0 = 1.602176634e-19, 1.380649e-23, 6.62607015e-34, 1.054571817e-34, 9.1093837015e-31, 8.8541878128e-12
T = 300.0
KT = KB * T / Q
NM = e99.NM
STACKS = {"base": ((15 * NM, 10.0), (3 * NM, 25.0), (7 * NM, 10.0)), "s2023": ((40 * NM, 10.0), (3 * NM, 25.0), (7 * NM, 10.0))}


def electrostatics(stack):
    x, z, ts = e99.grid(stack)
    nx, nz = len(x), len(z)
    idx = np.arange(nx * nz).reshape(nz, nx)
    eps = np.array([[e99.eps_at(xv, zv, stack, ts) for xv in x] for zv in z])
    fixed = np.zeros((nz, nx), bool)
    fixed[0, :] = True
    contact = (x[None, :] <= 0.0) & (z[:, None] >= ts - 1e-15) & (z[:, None] <= ts + e99.T_MOS2 + e99.T_CONTACT + 1e-15)
    fixed |= contact
    rows, cols, data = [], [], []
    b = np.zeros(nx * nz)
    for j in range(nz):
        for i in range(nx):
            p = idx[j, i]
            if fixed[j, i]:
                rows.append(p); cols.append(p); data.append(1.0)
                b[p] = -1.0 if j == 0 else 0.0
                continue
            hxm = x[i] - x[i - 1] if i > 0 else None
            hxp = x[i + 1] - x[i] if i < nx - 1 else None
            hzm = z[j] - z[j - 1] if j > 0 else None
            hzp = z[j + 1] - z[j] if j < nz - 1 else None
            wx, wz = 0.5 * ((hxm or 0) + (hxp or 0)), 0.5 * ((hzm or 0) + (hzp or 0))
            diag = 0.0
            for (jj, ii, h, w) in ((j, i - 1, hxm, wz), (j, i + 1, hxp, wz), (j - 1, i, hzm, wx), (j + 1, i, hzp, wx)):
                if h is None:
                    continue
                c = 2.0 * eps[j, i] * eps[jj, ii] / (eps[j, i] + eps[jj, ii]) * w / h
                rows.append(p); cols.append(idx[jj, ii]); data.append(-c)
                diag += c
            rows.append(p); cols.append(p); data.append(diag)
    lu = spla.splu(sp.csc_matrix((data, (rows, cols)), shape=(nx * nz, nx * nz)))
    jc = int(np.argmin(np.abs(z - (ts + 0.5 * e99.T_MOS2))))
    ch = np.where(x > 0)[0]
    width = 0.5 * (x[np.minimum(ch + 1, nx - 1)] - x[ch - 1])
    phi0 = lu.solve(b).reshape(nz, nx)[jc, ch]
    B = np.zeros((nx * nz, len(ch)))
    B[idx[jc, ch], np.arange(len(ch))] = width / EPS0
    G = lu.solve(B)[idx[jc, ch], :]
    return x[ch], width, phi0, G


class Stop(Exception):
    pass


class Channel:
    deadline = float("inf")

    def __init__(self, es, vg, phi_b, mu=5e-4, m_h=0.6, m_b=0.6):
        self.x, self.w, phi1, self.G = es
        self.phi0, self.vg, self.phi_b, self.mu = phi1 * abs(vg), vg, phi_b, mu
        self.dos = m_h * M0 / (np.pi * HBAR ** 2) * Q
        self.kap = np.sqrt(2 * m_b * M0 * Q) / HBAR
        self.kk = np.sqrt(2 * m_h * M0 * Q) / HBAR
        self.xp = np.concatenate([[0.0], self.x])
        self.dx = np.diff(self.x)
        self.keep = self.x < 60 * NM
        self.nE = 90

    def v(self, p):
        return self.phi0 + self.G @ (Q * p * 1.0)

    def eta(self, p):
        a = np.maximum(p, 1.0) / (self.dos * KT)
        return np.where(a < 30.0, np.log(np.expm1(np.minimum(a, 30.0))), a)

    def inject(self, p, V):
        n = int(self.keep.sum())
        ev = self.phi_b + np.concatenate([[0.0], V[:n]])
        xx = self.xp[:n + 1]
        emin = ev.min()
        E = emin + (self.phi_b + 12 * KT - emin) * (np.linspace(0.0, 1.0, self.nE) ** 2) + 1e-5
        k = self.kap * np.sqrt(np.maximum(ev[None, :] - E[:, None], 0.0))
        T_ = np.exp(-2.0 * np.sum(0.5 * (k[:, 1:] + k[:, :-1]) * np.diff(xx), axis=1))
        U = V + KT * self.eta(p)
        mu_c = self.phi_b + U[n - 1]
        f = 1.0 / (1.0 + np.exp(np.clip(E / KT, -500, 500))) - 1.0 / (1.0 + np.exp(np.clip((E - mu_c) / KT, -500, 500)))
        modes = self.kk * np.sqrt(E - emin) / np.pi
        return 2.0 / (H / Q) * np.trapezoid(modes * T_ * f, E)

    def rhs(self, t, y):
        if time.time() > self.deadline:
            raise Stop
        p = np.maximum(y * 1e16, 1.0)
        V = self.v(p)
        U = V + KT * self.eta(p)
        p1, p2 = p[:-1], p[1:]
        r = p2 / p1
        with np.errstate(divide="ignore", invalid="ignore"):
            pf = np.where(np.abs(r - 1.0) < 1e-6, 0.5 * (p1 + p2), (p2 - p1) / np.log(r))
        J = -self.mu * pf * np.diff(U) / self.dx
        d = np.zeros_like(p)
        d[:-1] -= J
        d[1:] += J
        d[0] += self.inject(p, V)
        return d / self.w / 1e16


def run(es, vg=-10.0, phi_b=1.6, mu=5e-4, m_b=0.6, t_end=1.0):
    c = Channel(es, vg, phi_b, mu=mu, m_b=m_b)
    c.deadline = time.time() + 600.0
    n = len(c.x)
    te = np.logspace(-12, np.log10(t_end), 46)
    t0 = time.time()
    y, tl, yl, nfev, stalled = np.full(n, 1e-14), [], [], 0, None
    last_t = 0.0
    for tk in te:
        try:
            r = solve_ivp(c.rhs, (last_t, tk), y, method="BDF", rtol=1e-3, atol=1e-5,
                          first_step=min(1e-14, 0.1 * (tk - last_t)) if last_t == 0.0 else 1e-2 * (tk - last_t))
        except (ValueError, Stop):
            stalled = float(last_t)
            break
        nfev += r.nfev
        if not r.success:
            stalled = float(last_t)
            break
        y, last_t = r.y[:, -1], tk
        tl.append(tk)
        yl.append(y.copy())
    ts, Y = np.array(tl), np.array(yl).T
    P = np.maximum(Y * 1e16, 1.0)
    Qt = P.T @ c.w / c.w.sum()
    Vt = np.array([c.v(P[:, i]) for i in range(P.shape[1])])
    frac = ((Vt - vg) * c.w).sum(1) / c.w.sum() / (-vg)
    last = Qt[-1]
    Pe, Ve = P[:, -1], Vt[-1]
    current = {}
    for dv in (0.5, 1.0):
        Vd = Ve * (1.0 + dv / abs(Ve[-1]))
        current["%g" % dv] = float(c.inject(Pe, Vd) * Q * 1e-6)
    t90 = float(np.interp(np.log10(0.9 * last), np.log10(np.maximum(Qt, 1.0)), np.log10(ts)))
    at = lambda tt: float(np.interp(np.log10(tt), np.log10(ts), frac))
    pat = lambda tt: float(np.interp(np.log10(tt), np.log10(ts), Qt))
    return {"vg": vg, "phi_Bp": phi_b, "mu_m2_Vs": mu, "m_b": m_b, "t": ts.tolist(), "stack_fraction": frac.tolist(), "p_mean_m2": Qt.tolist(),
            "t_fill_90pc_of_end_s": 10 ** t90, "stack_fraction_at_1us": at(1e-6), "stack_fraction_at_100ms": at(0.1),
            "stack_fraction_at_end": float(frac[-1]), "hole_current_A_um_forward_V": current, "p_mean_cm2_at_1us": pat(1e-6) / 1e4,
            "p_mean_cm2_at_end": last / 1e4, "seconds": round(time.time() - t0, 1), "n_rhs": int(nfev), "stalled_at_s": stalled}


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "one":
        r = run(electrostatics(STACKS["base"]), phi_b=float(sys.argv[2]), t_end=1e-3)
        print({k: v for k, v in r.items() if k not in ("t", "stack_fraction", "p_mean_m2")})
        return
    out = json.loads(OUT.read_text()) if OUT.exists() else {"label": "CALCULATION: hole filling of the depleted channel", "cases": {}}
    es = {k: electrostatics(v) for k, v in STACKS.items()}
    for st in STACKS:
        for phi in (1.2, 1.6, 2.0, 2.2):
            key = "%s_phi%.1f" % (st, phi)
            if key in out["cases"]:
                continue
            out["cases"][key] = run(es[st], phi_b=phi)
            OUT.write_text(json.dumps(out, indent=1))
    for key, kw in (("sens_mu1", dict(mu=1e-4)), ("sens_mu20", dict(mu=20e-4)), ("sens_mb0.3", dict(m_b=0.3)), ("sens_VG-12.9", dict(vg=-12.9))):
        if key in out["cases"]:
            continue
        out["cases"][key] = run(es["base"], phi_b=2.0, **kw)
        OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
