from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES, MEAS = ROOT / "results", ROOT / "data" / "measured"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9.5, "axes.labelweight": "bold", "legend.fontsize": 7,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.8, "xtick.major.width": 0.8,
                     "ytick.major.width": 0.8, "lines.linewidth": 1.3, "savefig.dpi": 300, "font.family": "DejaVu Sans",
                     "legend.frameon": False, "legend.handlelength": 1.6, "axes.titlesize": 8.5})
C_AL, C_HF, C_MOS, C_PT, C_NI = "#cfe0ef", "#f6d7ad", "#a9d8a4", "#bdbdbd", "#dcc98a"
RED, BLUE, GREY, GREEN, ORANGE = "#c0392b", "#2c6fbb", "0.45", "#2e8b57", "#e08a1e"


def J(name):
    return json.loads((RES / name).read_text())


LEGEND_SIZE = 9.0


LEGEND_MIN = 8.0


def _legend_size(lg, size):
    for t in lg.get_texts() + [lg.get_title()]:
        t.set_fontweight("bold")
        t.set_fontsize(size)
    lg.prop.set_size(size)


def bold_legends(fig):
    legends = list(fig.legends) + [a.get_legend() for a in fig.axes if a.get_legend() is not None]
    for lg in legends:
        _legend_size(lg, LEGEND_SIZE)
    for _ in range(4):
        fig.canvas.draw()
        width, done = fig.bbox.width, True
        for lg in legends:
            box, size = lg.get_window_extent(), lg.get_texts()[0].get_fontsize()
            if (box.x0 < 2 or box.x1 > width - 2) and size > LEGEND_MIN:
                _legend_size(lg, max(LEGEND_MIN, size * (width - 10) / box.width))
                done = False
        if done:
            break
    for lg in legends:
        box = lg.get_window_extent()
        print("  legend %.1f pt%s" % (lg.get_texts()[0].get_fontsize(), "  STILL WIDER THAN THE PANEL" if box.x0 < 0 or box.x1 > fig.bbox.width else ""))


def save(fig, name, tight=True):
    bold_legends(fig)
    fig.canvas.draw()
    for ax_ in fig.axes:
        for lab in ax_.get_xticklabels() + ax_.get_yticklabels():
            lab.set_fontweight("bold")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / (name + "." + ext), **({"bbox_inches": "tight", "pad_inches": 0.03} if tight else {}))
    plt.close(fig)
    print("wrote", name)
    return OUT / (name + ".png")


PW, PH = 3.4, 2.75


def panel(w=PW, h=PH):
    return plt.subplots(figsize=(w, h), layout="constrained")


LAYERS = [("Al$_2$O$_3$", 15.0, 10.0, C_AL), ("HfO$_2$", 3.0, 25.0, C_HF), ("Al$_2$O$_3$", 7.0, 10.0, C_AL)]
VBO = 3.19
EG_MOS2, EG_HF, EG_AL = 1.9, 5.8, 6.9
CBO = EG_AL - EG_MOS2 - VBO
CBO_ALHF = 1.3
TRAP_HF = (1.22, 2.02)
TRAP_AL = (1.6, 1.8)


def bands(vg):
    d_over_eps = sum(t / k for _, t, k, _ in LAYERS)
    e_al = -vg / (d_over_eps * 10.0)
    segs, x0, phi = [], 0.0, -vg
    for name, t, k, c in LAYERS:
        drop = e_al * 10.0 / k * t
        cb_off = CBO if k == 10.0 else CBO - CBO_ALHF
        gap = EG_AL if k == 10.0 else EG_HF
        segs.append({"name": name, "xa": x0, "xb": x0 + t, "cb": (phi + cb_off, phi - drop + cb_off),
                     "vb": (phi + cb_off - gap, phi - drop + cb_off - gap), "c": c, "k": k})
        phi -= drop
        x0 += t
    return segs, x0


def draw_bands(ax, vg):
    segs, xend = bands(vg)
    for s in segs:
        ax.axvspan(s["xa"], s["xb"], color=s["c"], lw=0, zorder=0)
        ax.plot([s["xa"], s["xb"]], s["cb"], "k", lw=0.9)
        ax.plot([s["xa"], s["xb"]], s["vb"], "k", lw=0.9)
        lo, hi = TRAP_AL if s["k"] == 10.0 else TRAP_HF
        for f in np.linspace(0.08, 0.92, 7 if s["k"] == 10.0 else 3):
            xx = s["xa"] + f * (s["xb"] - s["xa"])
            cbx = s["cb"][0] + f * (s["cb"][1] - s["cb"][0])
            ax.plot([xx, xx], [cbx - hi, cbx - lo], color=RED if s["k"] == 25.0 else BLUE, lw=1.4, solid_capstyle="butt")
    ax.plot([xend, xend + 3], [0, 0], color=GREEN, lw=1.2)
    ax.plot([xend, xend + 3], [-EG_MOS2] * 2, color=GREEN, lw=1.2)
    ax.plot([-3, 0], [-vg - 0.4] * 2, color="0.25", lw=1.4)
    ax.set_xlim(-3.5, xend + 3.5)
    ax.set_xlabel("distance from gate (nm)")
    return segs, xend
