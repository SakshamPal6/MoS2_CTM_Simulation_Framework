from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

PHI_T = 0.025852
LN10 = math.log(10.0)


def _f(x):
    h = 0.5 * np.asarray(x, float)
    return np.where(h > 40, h, np.log1p(np.exp(np.minimum(h, 40)))) ** 2


class Readout:
    def __init__(self, vth0=1.0, ss=0.398, mu_cm2=5.0, c_ox=3.816e-3, w_over_l=1.0, r_s=0.0, i_floor=1e-12, w_um=1.0):
        self.vth0, self.m = vth0, ss / (PHI_T * LN10)
        self.i_spec = 2 * self.m * mu_cm2 * 1e-4 * c_ox * w_over_l * PHI_T ** 2
        self.r_s, self.i_floor, self.w_um = r_s, i_floor, w_um

    def _ich(self, vg, vd, i, dvth):
        vp = (vg - self.vth0 - dvth) / self.m
        vs_, vd_ = i * self.r_s, vd - i * self.r_s
        return self.i_spec * (_f((vp - vs_) / PHI_T) - _f((vp - vd_) / PHI_T))

    def current(self, vg, vd=1.0, dvth=0.0):
        i0 = float(self._ich(vg, vd, 0.0, dvth))
        if self.r_s <= 0 or i0 <= 0:
            return max(i0, 0.0) + self.i_floor
        g = lambda i: i - float(self._ich(vg, vd, i, dvth))
        hi = min(i0, vd / (2 * self.r_s))
        return brentq(g, 0.0, hi, xtol=1e-18, rtol=1e-10) + self.i_floor if g(hi) > 0 else i0 + self.i_floor
