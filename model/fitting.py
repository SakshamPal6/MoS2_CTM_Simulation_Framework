from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from model import final_model as fm

STOP_WINDOW, STOP_DCHI2, MAX_EVAL = 100, 0.1, 1500


class Stop(Exception):
    pass


def fit(mode, z_start, out_path, z_ref=None):
    inp, r = fm.inputs(), fm.readout()
    keep = fm.free_indices(mode)
    z_ref = fm.load_parameters() if z_ref is None else np.asarray(z_ref, float)
    lo, hi, scale = fm.LO[keep], fm.HI[keep], fm.SCALE[keep]
    z0 = np.clip(np.asarray(z_start, float), lo + 1e-6, hi - 1e-6)
    out_path = Path(out_path)
    st = {"best": None, "chi2": math.inf, "hist": [], "t0": time.time()}

    def full(zs):
        z = z_ref.copy()
        z[keep] = zs
        return z

    def fun(zs):
        rr = fm.residuals(full(zs), inp, r, mode)
        c = float(rr @ rr)
        st["hist"].append(min(c, st["chi2"]))
        if c < st["chi2"]:
            st["chi2"], st["best"] = c, [float(q) for q in zs]
        out_path.write_text(json.dumps({"variant": mode, "names": [fm.NAMES[k] for k in keep], "best_z": st["best"],
                                        "chi2": st["chi2"], "n_eval": len(st["hist"]),
                                        "minutes": round((time.time() - st["t0"]) / 60, 1), "done": False}))
        h = st["hist"]
        if len(h) > STOP_WINDOW and h[-STOP_WINDOW - 1] - h[-1] < STOP_DCHI2:
            raise Stop("best chi2 fell by < %.1f over %d evaluations" % (STOP_DCHI2, STOP_WINDOW))
        if len(h) >= MAX_EVAL:
            raise Stop("%d evaluations" % MAX_EVAL)
        return rr

    reason = "least_squares converged"
    try:
        least_squares(fun, z0, bounds=(lo, hi), diff_step=1e-3, x_scale=scale, max_nfev=400)
    except Stop as e:
        reason = str(e)
    d = json.loads(out_path.read_text())
    d.update(done=True, stop_reason=reason)
    out_path.write_text(json.dumps(d, indent=1))
    return d
