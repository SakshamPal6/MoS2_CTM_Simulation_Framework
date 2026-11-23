from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import final_model as fm
from model.fitting import fit

FITS = ROOT / "results" / "fits"


def main(mode):
    start = json.loads((FITS / ("%s_start.json" % mode)).read_text())
    d = fit(mode, start["start_z"], FITS / ("%s.json" % mode))
    print({k: d[k] for k in ("chi2", "n_eval", "stop_reason")})
    if mode == "full":
        out = {"label": "Fitted parameters of the model of record", "chi2": round(d["chi2"], 3), "n_values": fm.N_VALUES,
               "parameters": dict(zip(fm.NAMES, d["best_z"]))}
        (ROOT / "results" / "model_parameters.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
