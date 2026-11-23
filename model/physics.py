from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

Q, EPS0, KB = 1.602176634e-19, 8.8541878128e-12, 8.617333262e-5
STACK = ((15e-9, 10.0), (3e-9, 25.0), (7e-9, 10.0))


@njit(cache=True)
def _kernel(fa, fb, v_g, duration, tol, dt0, x, eb, nux, ca, f0a, et0, l_eq, m, psi_lo, psi_hi, inv4lam, kt,
            cb, tau_b, asym, max_steps, steps, v_ref, b0, hv, e_gap, h_width, h_onset, n_cv, rf, rb, kxc, kxe):
    hmax = 0.0
    for k in range(hv.shape[0]):
        if hv[k] > hmax:
            hmax = hv[k]
    fa = fa.copy()
    fb = fb.copy()
    na, nb = fa.shape[0], fb.shape[0]
    dva = 0.0
    for k in range(na):
        dva += ca[k] * (fa[k] - f0a[k])
    dvb = 0.0
    for j in range(nb):
        dvb += cb[j] * (fb[j] - b0)
    n_tot = na - n_cv
    fan = np.empty(na)
    fbn = np.empty(nb)
    t, dt, last = 0.0, dt0, dt0
    while t < duration:
        steps += 1
        if steps > max_steps:
            return fa, fb, last, steps, True
        if dt > duration - t:
            dt = duration - t
        u = v_g - v_ref - dva - dvb
        ps = min(max(u / m, psi_lo), psi_hi)
        fld = (u - ps) / l_eq
        sh = 0.0
        if hmax > 0.0:
            sh = 1.0 / (1.0 + np.exp(min(max((u / m - h_onset) / h_width, -700.0), 700.0)))
        dva_n = 0.0
        for k in range(n_tot):
            e = et0[k] + fld * x[k]
            q2 = inv4lam * e * e
            ac = min(max((eb[k] - 0.5 * e + q2) / kt, 0.0), 700.0)
            ae = min(max((eb[k] + 0.5 * e + q2) / kt, 0.0), 700.0)
            kc = nux[k] * np.exp(-ac)
            ke = nux[k] * np.exp(-ae)
            tot = kc + ke
            kh = 0.0
            if sh > 0.0 and hv[k] > 0.0:
                ev = e - e_gap
                ah = min(max((eb[k] + 0.5 * ev + inv4lam * ev * ev) / kt, 0.0), 700.0)
                kh = hv[k] * sh * nux[k] * np.exp(-ah)
                tot += kh
            if k < n_cv:
                j = n_tot + k
                kc2 = kc * kxc
                ke2 = (ke + kh) * kxe
                a11 = -(kc + ke + kh + rf)
                a12 = rb - kc
                a21 = rf - kc2
                a22 = -(kc2 + ke2 + rb)
                tr = a11 + a22
                det = a11 * a22 - a12 * a21
                sq = np.sqrt(max(tr * tr - 4.0 * det, 0.0))
                l1 = 0.5 * (tr + sq)
                l2 = 0.5 * (tr - sq)
                e1 = np.exp(max(l1 * dt, -700.0))
                e2 = np.exp(max(l2 * dt, -700.0))
                p1 = dt * (1.0 + 0.5 * l1 * dt) if abs(l1 * dt) < 1e-8 else (e1 - 1.0) / l1
                p2 = dt * (1.0 + 0.5 * l2 * dt) if abs(l2 * dt) < 1e-8 else (e2 - 1.0) / l2
                if sq > 1e-9 * abs(tr) and sq > 0.0:
                    c1 = (e1 - e2) / (l1 - l2)
                    c0 = (l1 * e2 - l2 * e1) / (l1 - l2)
                    g1 = (p1 - p2) / (l1 - l2)
                    g0 = (l1 * p2 - l2 * p1) / (l1 - l2)
                else:
                    mu = 0.5 * tr
                    em = np.exp(max(mu * dt, -700.0))
                    ph = dt * (1.0 + 0.5 * mu * dt) if abs(mu * dt) < 1e-8 else (em - 1.0) / mu
                    ch = 0.5 * dt * dt if abs(mu * dt) < 1e-6 else (dt * em - ph) / mu
                    c1 = dt * em
                    c0 = em * (1.0 - mu * dt)
                    g1 = ch
                    g0 = ph - mu * ch
                s0 = fa[k]
                d0 = fa[j]
                fan[k] = c0 * s0 + c1 * (a11 * s0 + a12 * d0) + g0 * kc + g1 * (a11 * kc + a12 * kc2)
                fan[j] = c0 * d0 + c1 * (a21 * s0 + a22 * d0) + g0 * kc2 + g1 * (a21 * kc + a22 * kc2)
                dva_n += ca[k] * (fan[k] - f0a[k]) + ca[j] * (fan[j] - f0a[j])
                continue
            feq = kc / tot
            fan[k] = feq + (fa[k] - feq) * np.exp(-tot * dt)
            dva_n += ca[k] * (fan[k] - f0a[k])
        dvb_n = 0.0
        for j in range(nb):
            tj = tau_b[j] * asym if ps < fb[j] else tau_b[j]
            fbn[j] = ps + (fb[j] - ps) * np.exp(-dt / tj)
            dvb_n += cb[j] * (fbn[j] - b0)
        if abs(dva_n + dvb_n - dva - dvb) > tol and dt > 1e-15:
            dt *= 0.5
            continue
        fa[:] = fan
        fb[:] = fbn
        dva, dvb = dva_n, dvb_n
        t += dt
        last = dt
        dt *= 1.6
    return fa, fb, last, steps, False


@dataclass
class CellParams:
    log10_nt_cm2: float = 13.5
    x_min_nm: float = 0.3
    x_max_nm: float = 3.0
    e_b: float = 1.3
    sigma_b: float = 0.25
    e_t0: float = 0.0
    lam: float = 5.0
    k_hop: float = 9.28e8
    r_hop_nm: float = 1.49
    nu0: float = 1e13
    two_kappa_nm: float = 8.9
    n_scale: float = 1.0
    s_log10_nt: float = -np.inf
    s_e_b: float = 0.6
    s_sigma_b: float = 0.05
    s_e_t0: float = 0.0
    s_n_e: int = 7
    hole_log10_scale: float = -np.inf
    hole_e_gap: float = 2.0
    hole_width: float = 0.0259
    hole_onset: float = float("nan")
    conv_log10_rate: float = -np.inf
    conv_delta: float = 0.3
    conv_dx: float = 0.3
    log10_db: float = -np.inf
    log10_tau_b: float = 3.0
    sigma_tau: float = 0.5
    log10_asym: float = 0.0
    e_act_b: float = 0.0
    psi_lo: float = -1.5
    psi_hi: float = 0.15
    n_x: int = 3
    n_e: int = 33
    n_b: int = 9


class Cell:

    def __init__(self, p: CellParams, stack=None, ss=0.398, T=300.0, T0=300.0, tol=0.01, max_steps=50_000_000,
                 v_ref=0.0):
        self.p, self.T, self.tol, self.max_steps = p, float(T), float(tol), int(max_steps)
        self.v_ref = float(v_ref)
        self.stack = tuple(stack) if stack is not None else STACK
        self.ss = float(ss)
        self.m = self.ss / 60e-3
        self.kt = KB * self.T
        t_t, k_t = self.stack[-1]
        s = sum(t / k for t, k in self.stack)
        self.l_eq = k_t * s
        self.lever_b = s / EPS0
        self.c_ox = 1.0 / self.lever_b
        dx = (p.x_max_nm - p.x_min_nm) / p.n_x
        xs = (p.x_min_nm + dx * (np.arange(p.n_x) + 0.5)) * 1e-9
        z = np.linspace(-3.0, 3.0, p.n_e)
        g = np.exp(-0.5 * z ** 2)
        g /= g.sum()
        self.x = np.repeat(xs, p.n_e)
        self.eb = np.tile(p.e_b + p.sigma_b * z, p.n_x)
        w = np.tile(g, p.n_x)
        self.nux = p.k_hop * p.r_hop_nm / np.maximum(self.x * 1e9, p.r_hop_nm)
        lever = (s - self.x / k_t) / EPS0
        n_area = p.n_scale * 10 ** p.log10_nt_cm2 * 1e4 / p.n_x
        self.ca = Q * n_area * w * lever
        self.inv4lam = 1.0 / (4.0 * p.lam)
        self.n_a = self.x.shape[0]
        self.b0 = min(max(-self.v_ref / self.m, p.psi_lo), p.psi_hi)
        fld0 = (-self.v_ref - self.b0) / self.l_eq
        self.f0a = 1.0 / (1.0 + np.exp(-np.clip((p.e_t0 + fld0 * self.x) / self.kt, -700, 700)))
        self.et0 = np.full(self.n_a, p.e_t0)
        self.n_s = 0
        if np.isfinite(p.s_log10_nt):
            zs = np.linspace(-3.0, 3.0, p.s_n_e)
            gs = np.exp(-0.5 * zs ** 2)
            gs /= gs.sum()
            xg = np.full(p.s_n_e, 0.5 * (p.x_min_nm + p.x_max_nm) * 1e-9)
            nuxg = p.k_hop * p.r_hop_nm / np.maximum(xg * 1e9, p.r_hop_nm)
            self.x = np.concatenate([self.x, xg])
            self.eb = np.concatenate([self.eb, p.s_e_b + p.s_sigma_b * zs])
            self.nux = np.concatenate([self.nux, nuxg])
            self.ca = np.concatenate([self.ca, Q * 10 ** p.s_log10_nt * 1e4 * gs * (s - xg / k_t) / EPS0])
            self.et0 = np.concatenate([self.et0, np.full(p.s_n_e, p.s_e_t0)])
            self.f0a = np.concatenate([self.f0a, 1.0 / (1.0 + np.exp(-np.clip((p.s_e_t0 + fld0 * xg) / self.kt, -700, 700)))])
            self.n_s = p.s_n_e
        self.n_cv = 0
        if np.isfinite(p.conv_log10_rate):
            eps0 = (p.e_t0 + fld0 * self.x[:self.n_a]) / self.kt
            lx = np.clip(eps0, -700, 700)
            ly = p.conv_delta / self.kt
            m_ = np.maximum(0.0, np.maximum(lx, lx + ly))
            e0 = np.exp(-m_) / (np.exp(-m_) + np.exp(lx - m_) + np.exp(lx + ly - m_))
            s0 = np.exp(lx - m_) / (np.exp(-m_) + np.exp(lx - m_) + np.exp(lx + ly - m_))
            d0 = 1.0 - e0 - s0
            self.f0a = np.concatenate([self.f0a[:self.n_a] * 0.0 + s0, self.f0a[self.n_a:], d0])
            self.ca = np.concatenate([self.ca, self.ca[:self.n_a]])
            self.n_cv = self.n_a
        zb = np.linspace(-3.0, 3.0, p.n_b)
        gb = np.exp(-0.5 * zb ** 2)
        gb /= gb.sum()
        d_b = 10 ** p.log10_db * 1e4 if np.isfinite(p.log10_db) else 0.0
        self.cb = Q * d_b * self.lever_b * gb
        self.tau_b = 10 ** (p.log10_tau_b + p.sigma_tau * zb)
        if self.T != T0:
            self.tau_b = self.tau_b * math.exp(p.e_act_b / KB * (1.0 / self.T - 1.0 / T0))
        self.asym = 10 ** p.log10_asym
        self.hv = np.zeros(self.x.shape[0])
        if np.isfinite(p.hole_log10_scale):
            self.hv[:] = 10 ** p.hole_log10_scale
        self._h = 1e-13

    def fresh(self):
        return self.f0a.copy(), np.full(self.p.n_b, self.b0)

    def dv_a(self, fa):
        return float(self.ca @ (fa - self.f0a))

    def dv_b(self, fb):
        return float(self.cb @ (fb - self.b0))

    def dv(self, st):
        return self.dv_a(st[0]) + self.dv_b(st[1])

    def _conv_args(self):
        if not self.n_cv:
            return 0, 0.0, 0.0, 1.0, 1.0
        p = self.p
        rf = 10 ** p.conv_log10_rate
        return (int(self.n_cv), float(rf), float(rf * math.exp(-p.conv_delta / self.kt)),
                float(math.exp(-p.conv_dx / self.kt)), float(math.exp(-(p.conv_dx + p.conv_delta) / self.kt)))

    def hold(self, st, v_g, duration, dt0=None):
        if duration <= 0:
            return st
        h0 = max(self._h * 1e-3, 1e-13) if dt0 is None else dt0
        fa, fb, last, steps, hit = _kernel(st[0], st[1], float(v_g), float(duration), self.tol, float(h0),
                                           self.x, self.eb, self.nux, self.ca, self.f0a, self.et0,
                                           float(self.l_eq), float(self.m), float(self.p.psi_lo),
                                           float(self.p.psi_hi), float(self.inv4lam), float(self.kt), self.cb,
                                           self.tau_b, float(self.asym), self.max_steps, 0,
                                           self.v_ref, float(self.b0), self.hv, float(self.p.hole_e_gap),
                                           float(self.p.hole_width),
                                           float(self.p.hole_onset if np.isfinite(self.p.hole_onset) else self.p.psi_lo),
                                           *self._conv_args())
        if hit:
            raise FloatingPointError("integration step limit")
        self._h = last
        return fa, fb

    def sweep_branches(self, amp, rate, cycles=3, dv_step=0.1):
        st = self.fresh()
        dt = dv_step / rate
        up = np.arange(-amp, amp + 1e-9, dv_step)
        st = self.hold(st, -amp, dt)
        u = d = None
        for _ in range(cycles):
            for seg, tag in ((up, "up"), (up[::-1], "down")):
                for v in seg[1:]:
                    st = self.hold(st, float(v), dt)
                    if abs(v) < 1e-9:
                        if tag == "up":
                            u = self.dv(st)
                        else:
                            d = self.dv(st)
        return u, d

    def sweep_branches_at(self, amp, rate, v_levels, cycles=3, dv_step=0.1):
        st = self.fresh()
        dt = dv_step / rate
        up = np.arange(-amp, amp + 1e-9, dv_step)
        st = self.hold(st, -amp, dt)
        lv = np.asarray(v_levels, float)
        out = {}
        for c in range(cycles):
            for seg, tag in ((up, "up"), (up[::-1], "down")):
                prev = None
                got = np.full(lv.shape, np.nan)
                for v in seg[1:]:
                    st = self.hold(st, float(v), dt)
                    if c < cycles - 1:
                        continue
                    d = self.dv(st)
                    g = float(v) - lv - d
                    if prev is not None:
                        hit = np.isnan(got) & (prev[0] * g <= 0.0) & (prev[0] != g)
                        got[hit] = prev[1] + (d - prev[1]) * (0.0 - prev[0][hit]) / (g[hit] - prev[0][hit])
                    prev = (g, d)
                out[tag] = got
        if np.isnan(out["up"]).any() or np.isnan(out["down"]).any():
            raise ValueError("sweep branch does not cross a read level")
        return out["up"], out["down"]

    def read_sweep(self, st, v_start, v_stop, step, dt_pt, vth0, double=True):
        vs = np.arange(v_start, v_stop + 0.5 * step, step)
        branches = [vs] + ([vs[::-1]] if double else [])
        out = []
        for br in branches:
            prev, vth, near = None, float("nan"), (float("inf"), float("nan"))
            for v in br:
                st = self.hold(st, float(v), dt_pt)
                d = self.dv(st)
                g = float(v) - vth0 - d
                if abs(g) < near[0]:
                    near = (abs(g), vth0 + d)
                if prev is not None and math.isnan(vth) and (prev[1] <= 0.0 < g or prev[1] >= 0.0 > g):
                    vth = prev[0] + (0.0 - prev[1]) * (float(v) - prev[0]) / (g - prev[1])
                prev = (float(v), g)
            out.append(vth if not math.isnan(vth) else near[1])
        return st, out
