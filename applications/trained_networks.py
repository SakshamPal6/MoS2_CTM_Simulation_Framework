from __future__ import annotations

import functools
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ROOT / "data" / "trained_networks"
RETRAINED = ROOT / "results" / "trained_networks"


def _arrays(obj, out):
    if isinstance(obj, np.ndarray):
        out.append(obj)
        return len(out) - 1
    if isinstance(obj, (list, tuple)):
        return {"seq": "tuple" if isinstance(obj, tuple) else "list", "items": [_arrays(o, out) for o in obj]}
    raise TypeError(type(obj))


def _build(spec, arrays):
    if isinstance(spec, int):
        return arrays[spec]
    items = [_build(s, arrays) for s in spec["items"]]
    return tuple(items) if spec["seq"] == "tuple" else items


def _describe(obj, shapes, values):
    if isinstance(obj, np.ndarray) or (isinstance(obj, (list, tuple)) and obj and all(isinstance(v, float) for v in obj)):
        a = np.asarray(obj)
        shapes.append([str(a.dtype)] + list(a.shape))
        f = a.astype(np.float64).ravel()
        values += [float(f.sum()), float((f * f).sum()), float(f[0]), float(f[-1])] if f.size else []
    elif isinstance(obj, (list, tuple)):
        shapes.append(len(obj))
        for o in obj:
            _describe(o, shapes, values)
    elif isinstance(obj, float):
        values.append(obj)
    else:
        shapes.append(repr(obj))


def _file_name(fn, shapes):
    sig = json.dumps([Path(fn.__code__.co_filename).stem, fn.__name__, shapes])
    h = 0
    for ch in sig:
        h = (h * 131 + ord(ch)) % (1 << 61)
    return "%s_%s_%016x.npz" % (Path(fn.__code__.co_filename).stem, fn.__name__, h)


def reuse(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        shapes, values = [], []
        _describe(list(args) + [kwargs[k] for k in sorted(kwargs)], shapes, values)
        shapes.append(sorted(kwargs))
        name = _file_name(fn, shapes)
        path = SHIPPED / name
        if os.environ.get("RETRAIN_NETWORKS") != "1" and path.exists():
            d = np.load(path, allow_pickle=False)
            stored = np.asarray(json.loads(str(d["fingerprint"])))
            now = np.asarray(values)
            if stored.shape == now.shape and np.allclose(stored, now, rtol=1e-9, atol=1e-12):
                n = int(d["n_arrays"])
                return _build(json.loads(str(d["structure"])), [d["a%d" % i] for i in range(n)])
            print("trained_networks: the inputs of %s changed, training again" % name, file=sys.stderr, flush=True)
        result = fn(*args, **kwargs)
        arrays = []
        spec = _arrays(result, arrays)
        target = SHIPPED if os.environ.get("STORE_TRAINED_NETWORKS") == "1" else RETRAINED
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target / name, structure=json.dumps(spec), fingerprint=json.dumps(values), n_arrays=len(arrays),
                            **{"a%d" % i: a for i, a in enumerate(arrays)})
        return result

    return wrapper
