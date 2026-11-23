from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from model import final_model as fm


def main():
    z = fm.load_parameters()
    assert len(z) == len(fm.NAMES) == 25 and np.all(z >= fm.LO) and np.all(z <= fm.HI), "parameters outside their bounds"
    r = fm.residuals(z, fm.inputs(), fm.readout())
    chi2 = float(r @ r)
    w = fm.window(fm.cell(z), 1e-6)
    print("fitted parameters:          %d" % len(z))
    print("calibration values:         %d" % r.size)
    print("chi2 of the fit:            %.2f   (paper: 199.6)" % chi2)
    print("reduced chi2:               %.2f   (paper: 2.89)" % (chi2 / (r.size - len(z))))
    print("1 us window at -10 V:       %.2f V (paper: 3.47 V model, 3.45 V measured)" % w)
    widths = (1e-6, 1e-4, 1e-2)
    ws = [fm.window(fm.cell(z), t) for t in widths]
    print("window at 1 us, 100 us, 10 ms: %s V" % ", ".join("%.2f" % x for x in ws))
    ok = r.size == 94 and abs(chi2 - 199.57) < 0.05 and abs(w - 3.47) < 0.01 and ws[0] < ws[1] < ws[2]
    print("MODEL CHECK PASSED" if ok else "MODEL CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
