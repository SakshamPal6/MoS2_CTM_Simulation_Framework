from __future__ import annotations

import fnmatch
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MNIST = "data/mnist/mnist.npz"

LONG = ["device_studies/fit_model.py full", "device_studies/fit_model.py loo", "device_studies/fit_model.py nohole",
        "device_studies/fit_model.py direct", "device_studies/hole_fill.py"]
CHECK = ["check_model.py"]
DEVICE = ["device_studies/model_outputs.py report", "device_studies/model_outputs.py views", "device_studies/model_outputs.py paths",
          "device_studies/model_outputs.py jac", "device_studies/model_outputs.py refits", "device_studies/hopping_feasibility.py",
          "device_studies/edge_field.py", "device_studies/write_energy.py", "device_studies/temperature_prediction.py bound",
          "device_studies/temperature_prediction.py predict", "device_studies/temperature_prediction.py diagnose",
          "device_studies/read_gate.py", "device_studies/retention_projection.py"]
APPLICATIONS = ["applications/cell_levels.py", "applications/synaptic_response.py", "applications/design_maps.py",
                "applications/design_map_density_axis.py", "applications/digit_network.py " + MNIST,
                "applications/digit_network_device_aware.py " + MNIST, "applications/on_chip_learning.py %s --stochastic" % MNIST,
                "applications/read_energy.py " + MNIST, "applications/network_temperature.py " + MNIST,
                "applications/network_retention.py " + MNIST, "applications/smaller_crossbars.py " + MNIST,
                "applications/newton_solver_bratu.py --verify --slices 3", "applications/newton_solver_energy.py",
                "applications/newton_solver_dense.py", "applications/network_figure_data.py " + MNIST]
FIGURES = ["plotting/make_figures.py"]
TOL = 1e-3
SKIP_KEYS = {"seconds", "label", "script"}
ROUNDING = {"final_residual_max", "solver_max_difference"}
ROUNDING_LIMIT = 1e-9
SENSITIVE = {
    "newton_solver_bratu_verified.json": [("*/write_time_ms_per_write_with_verify", 0.01,
                                           "number of verify pulses, set by values that fall near the write tolerance")],
    "newton_solver_energy.json": [("*/digital_sparse/*", 0.1, "iteration count of the digital conjugate-gradient baseline"),
                                  ("*/ratio_digital_sparse_over_crossbar*", 0.1,
                                   "iteration count of the digital conjugate-gradient baseline")],
}
NOT_COMPARED = {
    "retention_projection.json": [("*/laws/*/params_full_span*", "parameters of trial laws that do not describe the trace"),
                                  ("draws/ranges_95/*/dlogI_10y", "outer edges of the 10-year range of 200 draws")],
}


def run(steps):
    bad = 0
    for s in steps:
        t0 = time.time()
        r = subprocess.run([sys.executable] + s.split(), cwd=ROOT, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                           env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        print("%-75s %s  %5.0f s" % (s, "ok" if r.returncode == 0 else "FAILED", time.time() - t0), flush=True)
        if r.returncode:
            bad += 1
            print(r.stderr[-2000:])
    return bad


def _match(path, pattern):
    return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path, "*/" + pattern)


def leaves(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            yield path, float("inf")
            return
        for k in a:
            if k not in SKIP_KEYS:
                yield from leaves(a[k], b[k], path + "/" + k if path else k)
        return
    if isinstance(a, (list, tuple, np.ndarray)) and isinstance(b, (list, tuple, np.ndarray)):
        x, y = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
        if x.shape != y.shape:
            yield path, float("inf")
            return
        fa, fb = np.asarray(a), np.asarray(b)
        if fa.dtype.kind in "fiub" and fb.dtype.kind in "fiub":
            fa, fb = fa.astype(float), fb.astype(float)
            d = np.abs(fa - fb) / np.maximum(np.maximum(np.abs(fa), np.abs(fb)), 1e-12)
            d = np.where((np.isnan(fa) & np.isnan(fb)) | (fa == fb), 0.0, d)
            if path.split("/")[-1] in ROUNDING:
                d = np.where((np.abs(fa) <= ROUNDING_LIMIT) & (np.abs(fb) <= ROUNDING_LIMIT), 0.0, d)
            yield path, float(d.max()) if d.size else 0.0
            return
        for i, (p_, q_) in enumerate(zip(list(a), list(b))):
            yield from leaves(p_, q_, path + "[%d]" % i)
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if a == b or (a != a and b != b):
            yield path, 0.0
        elif path.split("/")[-1] in ROUNDING and abs(a) <= ROUNDING_LIMIT and abs(b) <= ROUNDING_LIMIT:
            yield path, 0.0
        else:
            yield path, abs(a - b) / max(abs(a), abs(b), 1e-12)
        return
    yield path, 0.0 if a == b else float("inf")


def judge(name, a, b):
    worst, sens, skipped = 0.0, {}, set()
    for path, d in leaves(a, b):
        p0 = path.split("[")[0]
        nc = [r for pat, r in NOT_COMPARED.get(name, []) if _match(p0, pat)]
        if nc:
            skipped.add(nc[0])
            continue
        st = [(tol, r) for pat, tol, r in SENSITIVE.get(name, []) if _match(p0, pat)]
        if st and d > TOL:
            tol, r = st[0]
            sens[r] = max(sens.get(r, 0.0), d)
            if d > tol:
                worst = max(worst, d / tol * TOL)
            continue
        worst = max(worst, d)
    return worst, sens, skipped


def compare():
    bad = 0
    for ref in sorted((ROOT / "reference_results").glob("*")):
        new = ROOT / "results" / ref.name
        sens, skipped = {}, set()
        if not new.exists():
            d = float("inf")
        elif ref.suffix == ".npz":
            a, b = np.load(new), np.load(ref)
            d = max([judge(ref.name, a[k], b[k])[0] for k in b.files], default=0.0) if set(a.files) == set(b.files) else float("inf")
        else:
            d, sens, skipped = judge(ref.name, json.loads(new.read_text()), json.loads(ref.read_text()))
        ok = d <= TOL
        print("%-45s %s" % (ref.name, ("agrees" + (" (largest relative difference %.1e)" % d if d else "")) if ok
                                       else "DIFFERS (largest relative difference %.1e)" % d))
        for r, v in sens.items():
            print("%-45s   within its tolerance for processor-dependent rounding (%.1e): %s" % ("", v, r))
        for r in sorted(skipped):
            print("%-45s   not compared (processor-dependent, not used in the paper): %s" % ("", r))
        bad += not ok
    return bad


def main():
    args = sys.argv[1:]
    if "--compare" in args:
        bad = compare()
    elif "--figures" in args:
        bad = run(FIGURES)
    else:
        bad = (run(LONG) if "--long" in args else 0) + run(CHECK + DEVICE + APPLICATIONS + FIGURES) + compare()
    print("ALL RESULTS AGREE" if bad == 0 else "%d STEP(S) FAILED OR DIFFER" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
