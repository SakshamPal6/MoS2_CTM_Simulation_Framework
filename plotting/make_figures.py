from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "applications"))
import style as mf

plt, J, panel = mf.plt, mf.J, mf.panel
RED, BLUE, GREY, GREEN, ORANGE = mf.RED, mf.BLUE, mf.GREY, mf.GREEN, mf.ORANGE
MEAS = mf.MEAS
C_AL, C_HF, C_MOS, C_PT, C_NI = "#9fd3c7", "#f7b267", "#5aa469", "#8d99ae", "#e9c46a"
mf.LAYERS[:] = [("Al$_2$O$_3$", 15.0, 10.0, C_AL), ("HfO$_2$", 3.0, 25.0, C_HF), ("Al$_2$O$_3$", 7.0, 10.0, C_AL)]
PURPLE, MAGENTA = "#6a4c93", "#c2187f"
NPZ = lambda name: np.load(ROOT / "results" / name)
BAND_W, BAND_H, LEG_H, LEG_ROW, LEG_CH = 3.4, 2.75, 0.95, 0.165, 0.68


def load(n):
    return yaml.safe_load((MEAS / n).read_text(encoding="utf-8"))


def label_and_save(fig, name, letter=None):
    if letter:
        fig.suptitle("(%s)" % letter, x=0.01, ha="left", fontsize=11, fontweight="bold")
    mf.save(fig, name, tight=False)


def legend_below(fig, handles, ncol=1, fontsize=7.5):
    fig.legend(handles=handles, loc="outside lower center", ncol=ncol, fontsize=fontsize, frameon=False, handlelength=1.8,
               columnspacing=1.2)


def _legend_below(fig, handles, ncol=2):
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=ncol, fontsize=8, frameon=False,
               handlelength=1.8, columnspacing=1.2, labelspacing=0.45)


def _legend_outside_below(fig, ax, ncol=2, fontsize=7.5, handlelength=1.8):
    h, lb = ax.get_legend_handles_labels()
    fig.legend(h, lb, loc="outside lower center", ncol=ncol, fontsize=fontsize, frameon=False, handlelength=handlelength,
               columnspacing=1.2)


def leg_h(rows):
    return 0.10 + LEG_ROW * rows


def _band_axes(legend_h=LEG_H, lift=0.0):
    fig = plt.figure(figsize=(BAND_W, BAND_H + legend_h))
    hh = BAND_H + legend_h
    return fig, fig.add_axes([0.75 / BAND_W, (legend_h + 0.5 + lift) / hh, (BAND_W - 0.85) / BAND_W, (BAND_H - 0.75 - lift) / hh])


def _fitted_traps():
    z = J("model_report.json")["parameters"]
    return z["x_lo_nm"], z["x_lo_nm"] + z["x_width_nm"], z["E_t0"]


def _band_panel(vg, ax):
    segs, xend = mf.draw_bands(ax, vg)
    tun = segs[-1]
    x_lo, x_hi, e_t0 = _fitted_traps()
    xa, xb = xend - x_hi, xend - x_lo
    cb_at = lambda xx: tun["cb"][0] + (xx - tun["xa"]) / (tun["xb"] - tun["xa"]) * (tun["cb"][1] - tun["cb"][0])
    lev = [e_t0 + cb_at(xx) - tun["cb"][1] for xx in (xa, xb)]
    ft = (0.5 * (xa + xb), 0.5 * sum(lev))
    ax.plot([ft[0]], [ft[1]], "s", color=MAGENTA, ms=5.5, zorder=6)
    ax.axhline(0.0, color=GREEN, ls=(0, (4, 3)), lw=0.8, zorder=1)
    ax.set_ylim(-8.0, 13.5)
    ax.set_yticks([-5, 0, 5, 10])
    ax.set_ylabel("electron energy (eV)")
    return segs, xend, tun, ft


def _band_legend(fig, extra, base=True):
    hd = [Patch(fc=C_AL, label="Al$_2$O$_3$"), Patch(fc=C_HF, label="HfO$_2$"),
          Line2D([], [], color="k", lw=0.9, label="oxide $E_C$ and $E_V$"),
          Line2D([], [], color=GREEN, lw=1.2, label="MoS$_2$ $E_C$ and $E_V$"),
          Line2D([], [], color=GREEN, lw=0.8, ls=(0, (4, 3)), label="channel $E_C$ level"),
          Line2D([], [], color="0.25", lw=1.4, label="gate $E_F$"),
          Line2D([], [], color=RED, lw=0, marker="|", ms=7, mew=1.4, label="HfO$_2$ traps"),
          Line2D([], [], color=BLUE, lw=0, marker="s", ms=2.5, label="shallow Al$_2$O$_3$ defects"),
          Line2D([], [], color=MAGENTA, lw=0, marker="s", ms=5, label="fitted border traps")] + list(extra)
    fig.legend(handles=hd if base else list(extra), loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2 if base else 1, fontsize=7,
               frameon=False, handlelength=1.6, columnspacing=1.0, labelspacing=0.4)


def _band_steps(ax, segs, xend):
    for a, b in zip(segs[:-1], segs[1:]):
        for key in ("cb", "vb"):
            ax.plot([a["xb"], a["xb"]], [a[key][1], b[key][0]], "k", lw=0.9)
    ax.plot([0, 0], [segs[0]["vb"][0], segs[0]["cb"][0]], "k", lw=0.9)
    ax.plot([xend, xend], [segs[-1]["cb"][1], 0.0], color=GREEN, lw=1.0)
    ax.plot([xend, xend], [segs[-1]["vb"][1], -mf.EG_MOS2], color=GREEN, lw=1.0)


def model_cell():
    from model import final_model as fm
    z = fm.load_parameters()
    return fm, fm.cell(z), fm.readout()


def figure1():
    x_lo, x_hi, _ = _fitted_traps()
    fa, ax = panel(w=mf.PW, h=BAND_H + LEG_H)
    y, tops = 0.0, {}
    for key, name, t, c in [("gate", "Ti/Pt gate", 4.0, C_PT), ("blk", "Al$_2$O$_3$, 15 nm", 15.0, C_AL),
                            ("hf", "HfO$_2$, 3 nm", 3.0, C_HF), ("tun", "Al$_2$O$_3$, 7 nm", 7.0, C_AL)]:
        ax.add_patch(Rectangle((0, y), 10, t, fc=c, ec="k", lw=0.5))
        ax.text(0.3, y + t - (1.6 if key == "tun" else 0.9), name, ha="left", va="center", fontsize=7.5)
        tops[key] = (y, y + t)
        y += t
    ych = y
    ax.add_patch(Rectangle((2.0, ych), 6.0, 0.9, fc=C_MOS, ec="k", lw=0.5))
    for xa_, lab in ((0.0, "S"), (8.0, "D")):
        ax.add_patch(Rectangle((xa_, ych), 2.0, 3.0, fc=C_NI, ec="k", lw=0.5))
        ax.text(xa_ + 1.0, ych + 1.9, lab, ha="center", va="center", fontsize=9, fontweight="bold", gid="keep")
        ax.text(xa_ + 1.0, ych + 0.8, "Ni/Au", ha="center", va="center", fontsize=7, fontweight="bold", gid="keep")
    ax.text(5.0, ych + 3.7, "monolayer MoS$_2$", ha="center", fontsize=7.5, fontweight="bold", gid="keep")
    ax.text(5.0, ych + 1.65, "L = W = 1 µm", ha="center", fontsize=7.5, fontweight="bold", gid="keep")
    ax.text(5.0, -1.4, "$V_G$", ha="center", fontsize=8)
    ax.text(10.0, ych + 3.7, "$V_D$", ha="right", fontsize=8, fontweight="bold", gid="keep")
    ax.text(0.0, ych + 3.7, "0 V", ha="left", fontsize=8, fontweight="bold", gid="keep")
    rng = np.random.default_rng(1)
    for key, n in (("blk", 26), ("tun", 16)):
        y0, y1 = tops[key]
        xs, ys = rng.uniform(0.4, 9.6, n), rng.uniform(y0 + 0.4, y1 - 0.4, n)
        keep = ~((key == "tun") & (np.abs(ys - (ych - 0.5 * (x_lo + x_hi))) < 0.55))
        keep &= ~((xs < 4.8) & (ys > y1 - 1.9))
        ax.plot(xs[keep], ys[keep], ".", color=BLUE, ms=3.2)
    hy0, hy1 = tops["hf"]
    for xx in np.linspace(5.0, 9.4, 6):
        ax.plot([xx, xx], [hy0 + 0.6, hy1 - 0.6], color=RED, lw=1.4)
    yb = ych - 0.5 * (x_lo + x_hi)
    ax.plot(np.linspace(2.3, 7.7, 10), [yb] * 10, "s", color=MAGENTA, ms=3.0)
    for xx in np.linspace(2.2, 7.8, 13):
        ax.plot(xx, ych - 0.35, "|", color=PURPLE, ms=5)
    y_sh = ych - 0.3 - 0.42 * (ych - 0.3 - yb)
    for yy, t, col in [(ych - 0.3, "surface states", PURPLE), (y_sh, "shallow Al$_2$O$_3$\ndefects (hopping)", BLUE),
                       (yb, "fitted border traps\n%.1f-%.1f nm from channel" % (x_lo, x_hi), MAGENTA),
                       (0.5 * (hy0 + hy1), "HfO$_2$ traps", RED)]:
        ax.annotate(t, xy=(10.0, yy), xytext=(11.0, yy), fontsize=6.5, color=col, va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=col))
    ax.set_xlim(-0.3, 17.2); ax.set_ylim(-2.4, ych + 4.4); ax.axis("off")

    fb, ax = _band_axes(lift=0.15)
    segs, xend, tun, ft = _band_panel(-10.0, ax)
    _band_steps(ax, segs, xend)
    cb_at = lambda xx: tun["cb"][0] + (xx - tun["xa"]) / (tun["xb"] - tun["xa"]) * (tun["cb"][1] - tun["cb"][0])
    sites = [ft] + [(xx, cb_at(xx) - 1.7) for xx in np.arange(ft[0] + 1.5, xend, 1.5)] + [(xend + 0.6, 0.0)]
    assert all(b[1] < a[1] for a, b in zip(sites[:-1], sites[1:])), sites
    ax.plot([s[0] for s in sites[1:-1]], [s[1] for s in sites[1:-1]], "o", ms=4.5, mfc="white", mec=BLUE, mew=1.3, zorder=7)
    axi = ax.inset_axes([0.60, 0.63, 0.38, 0.35])
    x0, x1, y0, y1 = ft[0] - 0.8, xend + 1.4, -0.5, ft[1] + 1.0
    axi.axvspan(x0, xend, color=C_AL, lw=0, zorder=0)
    xx = np.linspace(x0, xend, 20)
    axi.plot(xx, [cb_at(q) for q in xx], "k", lw=0.9)
    axi.plot(xx, [cb_at(q) - 1.7 for q in xx], color=BLUE, lw=0.6, ls=(0, (1, 2)))
    axi.plot([xend, x1], [0, 0], color=GREEN, lw=1.2)
    axi.axhline(0.0, color=GREEN, ls=(0, (4, 3)), lw=0.8)
    axi.plot([ft[0]], [ft[1]], "s", color=MAGENTA, ms=6, zorder=6)
    axi.plot([s[0] for s in sites[1:-1]], [s[1] for s in sites[1:-1]], "o", ms=5.5, mfc="white", mec=BLUE, mew=1.4, zorder=7)
    for a, b in zip(sites[:-1], sites[1:]):
        axi.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=BLUE, lw=1.4, shrinkA=4, shrinkB=4,
                                      connectionstyle="arc3,rad=-0.25", zorder=8))
    axi.set_xlim(x0, x1); axi.set_ylim(y0, y1); axi.set_xticks([]); axi.set_yticks([])
    ax.indicate_inset_zoom(axi, edgecolor="0.35")
    _band_legend(fb, [Line2D([], [], color=BLUE, lw=0, marker="o", ms=4.5, mfc="white", mew=1.3,
                             label="hop sites of the\nreleased electron"),
                      Line2D([], [], color=BLUE, lw=1.1, marker=">", ms=4, label="hops during the write")])
    return [fa, fb]


def figure2():
    fm, m, r = model_cell()
    er = m.hold(m.hold(m.fresh(), 10.0, fm.ERASE_S), 0.0, 1.0)
    fa, ax = panel()
    vg = np.linspace(-4, 4, 161)
    cols = {None: "0.4", -8.0: PURPLE, -8.5: ORANGE, -9.0: GREEN, -9.5: BLUE, -10.0: RED}
    for vp, col in cols.items():
        st = er if vp is None else m.hold(m.hold(er, vp, 1e-6), 0.0, 1.0)
        ax.semilogy(vg, [r.current(x, 1.0, m.dv(st)) for x in vg], "-", color=col, lw=1.3)
    ax.set_xlabel("gate voltage (V)"); ax.set_ylabel("drain current (A)")
    ax.set_ylim(5e-13, 2e-6)
    legend_below(fa, [Line2D([], [], color=c, lw=1.3, label="erased" if v is None else "%.1f V, 1 µs" % v)
                      for v, c in cols.items()], ncol=2, fontsize=7)
    A = J("design_maps.json")["a"]
    fb, ax = panel()
    W = np.array(A["window_V"])
    for vp, col in ((-8.0, PURPLE), (-9.0, GREEN), (-10.0, RED)):
        ax.semilogx(A["width_s"], W[A["V_P"].index(vp)], "o-", color=col, ms=3.5, label="%.0f V" % vp)
    ax.axvline(1e-6, color="0.6", ls=":", lw=1)
    ax.set_xlabel("program pulse width (s)"); ax.set_ylabel("window (V)")
    _legend_outside_below(fb, ax, ncol=3)
    B = J("design_maps.json")["b"]
    fc, ax = panel()
    WB = np.array(B["window_1us_V"])
    im = ax.imshow(WB, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=4.5)
    for i in range(WB.shape[0]):
        for j in range(WB.shape[1]):
            ax.text(j, i, "%.1f" % WB[i, j], ha="center", va="center", fontsize=7, color="white" if WB[i, j] < 2.8 else "black")
    ax.set_xticks(range(len(B["blocking_nm"]))); ax.set_xticklabels(["%d" % n for n in B["blocking_nm"]])
    DR = J("design_map_density_axis.json")
    assert DR["rate_scale"] == B["rate_scale"]
    ax.set_yticks(range(len(DR["density_ratio_exact"]))); ax.set_yticklabels(["%.2f" % x for x in DR["density_ratio_exact"]])
    ax.set_xlabel("blocking Al$_2$O$_3$ (nm)"); ax.set_ylabel("relative defect density")
    cb = fc.colorbar(im, ax=ax, pad=0.02); cb.set_label("1 µs window (V)", fontsize=8.5)
    legend_below(fc, [Patch(fc="none", ec="none", label="write at -10 V for 1 µs")])
    return [fa, fb, fc]


def figure3():
    A, e197, w = J("design_maps.json")["a"], J("read_energy.json"), J("write_energy.json")
    e10 = w["channel_half_CV2_fJ"] + w["gate_line_half_CV2_fJ"]
    fa, ax = panel()
    i1 = A["width_s"].index(min(A["width_s"], key=lambda t: abs(math.log10(t) + 6)))
    vv = [v for v in A["V_P"] if abs(v) <= 10]
    E = [e10 * (v / 10) ** 2 for v in vv]
    Wd = [A["window_V"][A["V_P"].index(v)][i1] for v in vv]
    ax.plot(E, Wd, "o-", color=BLUE, ms=4)
    for v, e, y in zip(vv, E, Wd):
        if v in (-7.0, -8.0, -9.0, -10.0):
            ax.annotate("%.0f V" % v, (e, y), xytext=(-7, 1) if v == -10.0 else (4, -9), textcoords="offset points",
                        fontsize=7, color="0.3", ha="right" if v == -10.0 else "left")
    ax.set_xlabel("energy per write (fJ)"); ax.set_ylabel("window of a 1 µs write (V)")
    legend_below(fa, [Line2D([], [], color=BLUE, marker="o", ms=4, label="program voltage from -6 to -10 V")])
    fb, ax = panel()
    tr = np.logspace(-7, -5, 30)
    for vd, col in ((1.0, BLUE), (0.5, GREEN), (0.25, ORANGE)):
        c1, c2 = e197["cases"]["V%.2f_t1e-06" % vd], e197["cases"]["V%.2f_t1e-07" % vd]
        assert abs(c1["energy_per_inference_fJ"] / c2["energy_per_inference_fJ"] - 10.0) < 1e-6
        ax.loglog(tr * 1e6, c1["energy_per_inference_fJ"] * tr / 1e-6 / 1e3, "-", color=col, lw=1.4,
                  label="V$_D$ = %g V" % vd)
    ax.set_xlabel("read time of a full-scale input (µs)"); ax.set_ylabel("energy per digit (pJ)")
    _legend_outside_below(fb, ax, ncol=3)
    c188, c194 = J("digit_network.json")["cost"], J("on_chip_learning.json")["on_chip"]
    budget = [("MAC", e197["cases"]["V1.00_t1e-06"]["energy_per_MAC_aJ"] * 1e-3, GREEN), ("cell\nwrite", e10, BLUE),
              ("digit", e197["cases"]["V1.00_t1e-06"]["energy_per_inference_fJ"], GREEN),
              ("learning\nstep", c194["energy_nJ"] * 1e6 / c194["updates"], ORANGE),
              ("network\nwrite", (c188["energy_program_nJ"] + c188["energy_erase_nJ"]) * 1e6, BLUE)]
    fc, ax = panel()
    ax.bar(range(len(budget)), [b[1] for b in budget], color=[b[2] for b in budget], width=0.6)
    ax.set_yscale("log"); ax.set_ylim(1e-2, 1e9)
    ax.set_xticks(range(len(budget))); ax.set_xticklabels([b[0] for b in budget], fontsize=6.8)
    ax.set_ylabel("energy (fJ)")
    legend_below(fc, [Patch(fc=GREEN, label="reading in place (V$_D$ = 1 V, 1 µs)"), Patch(fc=BLUE, label="writing (-10 V, 1 µs)"),
                      Patch(fc=ORANGE, label="learning on the chip")], ncol=1, fontsize=7)
    return [fa, fb, fc]


def figure4():
    d = J("network_retention.json")
    P = d["projection"]
    t = np.array(P["t_s"])
    base = np.array(P["log10_I_model_only"])
    slow = lambda tt, a, tau: -a * np.log(1 + tt / tau) ** 2
    fa, ax = panel()
    for key, ls, lw, lab in (("both_runs", "-", 1.4, "model with the slow loss"), ("10min_run", ":", 1.0, "slow loss from the 10 min trend"),
                             ("1h_run", "--", 1.0, "slow loss from the 1 h trend")):
        f = d["fits"][key]
        y = base + slow(t, f["a_dec"], f["tau_s"]) - slow(t[0], f["a_dec"], f["tau_s"])
        ax.semilogx(t / 86400, 10 ** y, ls, color=RED, lw=lw, label=lab)
    ax.semilogx(t / 86400, 10 ** base, "-", color=BLUE, lw=1.2, label="model, early loss only")
    ax.set_yscale("log")
    ax.axhline(2e-12, color="0.4", ls="-.", lw=0.9, label="OFF level")
    hd = d["fits"]["both_runs"]["hold_time_s"] / 86400
    lo95, hi95 = d["hold_time_95pc_s"][0] / 86400, d["hold_time_95pc_s"][1] / 86400
    ax.errorbar([hd], [2e-12], xerr=[[hd - lo95], [hi95 - hd]], fmt="D", color=RED, ms=5, capsize=3, lw=1.2, zorder=6,
                label="hold time with 95 % range")
    ax.set_xlabel("time after writing"); ax.set_ylabel("ON current (A)")
    ax.set_xlim(1e-4, 3650); ax.set_ylim(1e-12, 1e-8)
    ax.set_xticks([60 / 86400, 1 / 24, 1, 30, 365, 3650])
    ax.set_xticklabels(["1 min", "1 h", "1 day", "1 mo", "1 yr", "10 yr"], fontsize=7.5)
    ax.minorticks_off(); ax.set_yticks([1e-12, 1e-11, 1e-10, 1e-9, 1e-8])
    _legend_outside_below(fa, ax, ncol=1, fontsize=7)
    N = d["network"]
    ts = np.array([float(k) for k in N])
    fb, ax = panel()
    ax.semilogx(ts, [100 * N[k]["accuracy_no_compensation"] for k in N], "o-", color=BLUE, ms=4, label="normal read (gate at 0 V)")
    ax.semilogx(ts, [100 * N[k]["accuracy_compensated"] for k in N], "s--", color=GREEN, ms=4,
                label="read gate raised, set by a reference cell")
    ax.axhline(95, color="0.45", ls=":", lw=1.2, label="95 % target")
    ax.axhline(10, color="0.6", ls="-.", lw=1.0, label="chance level (random guess, 10 %)")
    ax.axvline(3600, color=ORANGE, ls="--", lw=1.0, label="hourly refresh")
    ax.set_xticks([1, 60, 3600, 86400, 2.592e6, 3.156e7])
    ax.set_xticklabels(["1 s", "1 min", "1 h", "1 day", "1 mo", "1 yr"], fontsize=7.5)
    ax.minorticks_off()
    ax.set_xlabel("time after writing"); ax.set_ylabel("accuracy (%)"); ax.set_ylim(0, 100)
    _legend_outside_below(fb, ax, ncol=1, fontsize=7)
    C = J("design_maps.json")["c"]
    fc, ax = panel()
    lo, hi = C["safe_window_V"]
    for v in (lo, hi):
        ax.axvline(v, color=GREEN, ls="--", lw=1.1)
    for v in (-0.05, 0.05):
        ax.axhline(v, color="0.5", ls=":", lw=1.1)
    for k, col in (("-10.0", BLUE), ("-8.5", RED)):
        ax.plot(C["V_read"], C["dlog10I_after_1000_reads"][k], "o-", color=col, ms=3.5)
    vgs = [N[k]["read_gate_V"] for k in N]
    ax.plot(vgs, [-0.085] * len(vgs), "v", color=GREEN, ms=4)
    ax.set_xlabel("read-gate voltage (V)"); ax.set_ylabel("change (decade)")
    legend_below(fc, [Line2D([], [], color=BLUE, marker="o", ms=3.5, label="cell written at -10 V"),
                      Line2D([], [], color=RED, marker="o", ms=3.5, label="cell written at -8.5 V"),
                      Line2D([], [], color=GREEN, ls="--", lw=1.1, label="limits of the safe read window (%.1f and +%.1f V)" % (lo, hi)),
                      Line2D([], [], color="0.5", ls=":", lw=1.1, label="allowed change (±0.05 decade)"),
                      Line2D([], [], color=GREEN, lw=0, marker="v", ms=4, label="read-gate voltages used in (b)")], ncol=1, fontsize=7)
    return [fa, fb, fc]


def _network_diagram():
    f = plt.figure(figsize=(mf.PW, mf.PH))
    ax = f.add_axes([0.02, 0.0, 0.96, 0.9]); ax.axis("off")
    W, H = mf.PW * 0.96, mf.PH * 0.9
    ax.set_xlim(0, 10); ax.set_ylim(0, 10 * H / W)
    ys_in, ys_h, ys_o = np.linspace(5.2, 0.9, 6), np.linspace(4.9, 1.2, 5), np.linspace(5.3, 0.6, 10)
    xi, xh, xo = 1.6, 4.8, 8.0
    for yi in ys_in:
        for yh in ys_h:
            ax.plot([xi, xh], [yi, yh], color="0.65", lw=0.6, zorder=1)
    for yh in ys_h:
        for yo in ys_o:
            ax.plot([xh, xo], [yh, yo], color="0.65", lw=0.6, zorder=1)
    for yi in ys_in:
        ax.add_patch(Circle((xi, yi), 0.36, fc="#F3D9A4", ec="0.3", lw=0.6, zorder=2))
    for yh in ys_h:
        ax.add_patch(Circle((xh, yh), 0.36, fc="#BBD4EE", ec="0.3", lw=0.6, zorder=2))
    for k, yo in enumerate(ys_o):
        ax.add_patch(Circle((xo, yo), 0.23, fc="#D5E8D4", ec="0.3", lw=0.6, zorder=2))
        ax.text(xo + 0.4, yo, str(k), ha="left", va="center", fontsize=9)
    ax.text(xi, 5.75, "input layer\n784 pixels", ha="center", va="bottom", fontsize=9.5)
    ax.text(xh, 5.75, "hidden layer\n128, ReLU", ha="center", va="bottom", fontsize=9.5)
    ax.text(xo, 5.75, "output layer\n10 digits", ha="center", va="bottom", fontsize=9.5)
    return f


def figure5():
    D = NPZ("network_figure_data.npz")
    X, Y, P, Iout = D["x_examples"], D["y_examples"], D["pred"], D["out_currents"]
    fb = plt.figure(figsize=(mf.PW, mf.PH))
    gs = fb.add_gridspec(2, 4, height_ratios=[1, 0.9], hspace=0.45, wspace=0.45, left=0.1, right=0.98, top=0.86, bottom=0.1)
    for j in range(4):
        a = fb.add_subplot(gs[0, j])
        a.imshow(X[j].reshape(28, 28), cmap="gray_r"); a.set_xticks([]); a.set_yticks([])
        a.set_title("input digit", fontsize=6.5)
        b = fb.add_subplot(gs[1, j])
        cur = Iout[j] / np.abs(Iout[j]).max()
        b.bar(range(10), cur, color=[GREEN if i == np.argmax(cur) else "0.7" for i in range(10)], width=0.8)
        b.set_xticks(range(0, 10, 3)); b.set_yticks([]); b.tick_params(labelsize=5.5)
        b.axhline(0, color="0.5", lw=0.5)
        if j == 0:
            b.set_ylabel("output", fontsize=7)
        b.set_title("read as %d" % int(np.argmax(cur)), fontsize=6.5, color=GREEN)
    fc = plt.figure(figsize=(mf.PW, mf.PH))
    G, a1, b1 = D["G_levels"], D["w1_pos"], D["w1_neg"]
    gs = fc.add_gridspec(2, 4, wspace=0.08, hspace=0.12, left=0.03, right=0.84, top=0.86, bottom=0.06)
    vmax = (G.max() - G.min()) * 1e9
    for j in range(8):
        a = fc.add_subplot(gs[j // 4, j % 4])
        im = a.imshow(((G[a1[:, j]] - G[b1[:, j]]) * 1e9).reshape(28, 28), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a.set_xticks([]); a.set_yticks([])
    cax = fc.add_axes([0.87, 0.12, 0.03, 0.68])
    cb = fc.colorbar(im, cax=cax); cb.set_label("G$^+$ - G$^-$ (nS)", fontsize=7); cb.ax.tick_params(labelsize=6)
    fd, ax = panel()
    cm = D["confusion"]
    cmn = cm / cm.sum(1, keepdims=True) * 100
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=100)
    for i in range(10):
        ax.text(i, i, "%.0f" % cmn[i, i], ha="center", va="center", fontsize=6, color="white")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("digit read by the crossbar"); ax.set_ylabel("true digit")
    cbar = fd.colorbar(im, ax=ax, pad=0.02); cbar.set_label("share of test digits (%)", fontsize=8)
    fe = plt.figure(figsize=(mf.PW, mf.PH))
    gs = fe.add_gridspec(4, 8, wspace=0.08, hspace=0.55, left=0.02, right=0.98, top=0.86, bottom=0.03)
    for k in range(32):
        a = fe.add_subplot(gs[k // 8, k % 8])
        a.imshow(X[k].reshape(28, 28), cmap="gray_r"); a.set_xticks([]); a.set_yticks([])
        a.set_title(str(P[k]), fontsize=7, color=GREEN if P[k] == Y[k] else RED, pad=1)
    return [_network_diagram(), fb, fc, fd, fe]


def solver_history(lam, tol, seed=0):
    import newton_solver_bratu as e3
    e3.SLICES, e3.VERIFY_TOL = 3, (tol or 0.0)
    lev = J("cell_levels.json")
    g = np.array([L["I_A"]["1"] for L in lev["levels"]]) / lev["read"]["V_D"]
    cells = e3.Cells(g, None, None)
    rng = np.random.default_rng(seed)
    u = np.zeros(e3.N * e3.N)
    hist = [float(np.max(np.abs(e3.F(u, lam))))]
    for _ in range(e3.MAX_NEWTON):
        f = e3.F(u, lam)
        if np.max(np.abs(f)) < e3.TOL:
            break
        Jx = e3.J(u, lam)
        if tol is None:
            dlt = np.linalg.solve(Jx, -f)
        else:
            Jh, _ = cells.write(Jx, 0.0, rng)
            dlt = np.zeros_like(u)
            for _ in range(e3.MAX_REF):
                r = -f - Jx @ dlt
                if np.linalg.norm(r) < e3.REFINE_TOL * np.linalg.norm(f):
                    break
                dlt += e3.cg(Jh, r, e3.INNER)[0]
        u = u + dlt
        hist.append(float(np.max(np.abs(e3.F(u, lam)))))
    e3.VERIFY_TOL = 0.0
    return hist


def figure6():
    D = NPZ("network_figure_data.npz")
    fa, ax = panel()
    im = ax.imshow(D["u_cells"], origin="lower", cmap="magma", extent=(0, 1, 0, 1))
    ax.set_xlabel("x"); ax.set_ylabel("y")
    cb = fa.colorbar(im, ax=ax, pad=0.02); cb.set_label("u, solved with the crossbar", fontsize=8)
    legend_below(fa, [Patch(fc="none", ec="none", label="-∇²u = 5 e$^u$, Jacobian stored in model cells")], fontsize=7)
    fb, ax = panel()
    for tol, col, mk, lab in ((None, "0.3", "o", "digital solver"), (0.01, BLUE, "s", "cells written to within 1 % of target"),
                              (0.02, ORANGE, "^", "cells written to within 2 % of target")):
        h = solver_history(5.0, tol)
        ax.semilogy(range(len(h)), np.maximum(h, 1e-16), "-", marker=mk, color=col, ms=4, label=lab)
    ax.axhline(1e-9, color="0.6", ls=":", lw=1, label="stop criterion")
    ax.set_xlabel("Newton iteration"); ax.set_ylabel("max |residual|")
    _legend_outside_below(fb, ax, ncol=1, fontsize=7)
    E = J("newton_solver_dense.json")
    fc, ax = panel()
    ns = sorted({v["n"] for k, v in E.items() if k.startswith("n")})
    for bv, ls, mfc in ((2, "-", None), (5, "--", "white")):
        row = lambda f: [f(E["n%d_b%d" % (n, bv)]) for n in ns]
        for f, col, mk in ((lambda c: c["digital"]["energy_J"], "0.3", "o"), (lambda c: c["rewrite_J"]["total_J"], BLUE, "s")):
            ax.loglog(ns, row(f), ls, marker=mk, color=col, ms=4.5, mfc=mfc or col, lw=1.3)
    ax.set_xlabel("number of unknowns"); ax.set_ylabel("energy per"+"\n"+"solution (J)")
    ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns]); ax.minorticks_off()
    legend_below(fc, [Line2D([], [], color="0.3", marker="o", lw=1.3, label="Tesla V100 graphics processor"),
                      Line2D([], [], color=BLUE, marker="s", lw=1.3, label="crossbar of model cells"),
                      Line2D([], [], color="0.5", ls="-", lw=1.3, label="source strength 2"),
                      Line2D([], [], color="0.5", ls="--", lw=1.3, label="source strength 5")], ncol=1, fontsize=7)
    return [fa, fb, fc]


def figure7():
    C = NPZ("network_figure_data.npz")["learning_curve"][1:]
    o = J("on_chip_learning.json")
    fa, ax = panel()
    ax.plot(C[:, 0], 100 * C[:, 1], "-", color=BLUE, lw=1.8)
    ax.plot(C[:, 0], 100 * C[:, 2], "-", color=ORANGE, lw=1.8)
    ax.axhline(100 * o["float"]["all_digits"], color=BLUE, ls="--", lw=1.2)
    ax.axhline(100 * o["float"]["digits_5_9"], color=ORANGE, ls="--", lw=1.2)
    ax.set_xlabel("learning steps (32 images each)"); ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(76, 96); ax.set_xlim(0, C[-1, 0] + 100)
    legend_below(fa, [Line2D([], [], color=BLUE, lw=1.8, label="all digits, on chip"),
                      Line2D([], [], color=BLUE, ls="--", lw=1.2, label="all digits, software"),
                      Line2D([], [], color=ORANGE, lw=1.8, label="new digits 5-9, on chip"),
                      Line2D([], [], color=ORANGE, ls="--", lw=1.2, label="new digits 5-9, software")], ncol=1, fontsize=7.5)
    t = J("network_temperature.json")
    fb, ax = panel()
    T = [25, 50, 85, 125]
    ax.plot(T, [100 * t["T_%dC" % c]["B_programmed_and_read_hot"] for c in T], "o-", color=GREEN, ms=4,
            label="written and read at the temperature")
    ax.plot(T, [100 * t["T_%dC" % c]["A_programmed_25C_held_10min_read_hot"] for c in T], "s--", color=RED, ms=4,
            label="written at 25 °C, held 10 min\nat the temperature")
    ax.set_xlabel("temperature (°C)"); ax.set_ylabel("MNIST accuracy (%)"); ax.set_ylim(40, 100)
    _legend_outside_below(fb, ax, ncol=1, fontsize=7)
    return [fa, fb]


def figureS1():
    v = J("model_views.json")
    f = v["fig1d"]
    fa, ax = _band_axes(leg_h(2))
    ax.errorbar(f["amps_V"], f["measured_hole_V"], yerr=f["measured_hole_sd"], fmt="o", color=BLUE, ms=3.5, capsize=2)
    ax.errorbar(f["amps_V"], f["measured_electron_V"], yerr=f["measured_electron_sd"], fmt="s", color=RED, ms=3.5, capsize=2)
    ax.plot(f["amps_V"], f["model_hole_V"], "-", color=BLUE)
    ax.plot(f["amps_V"], f["model_electron_V"], "-", color=RED)
    ax.axhline(0, color="0.85", lw=0.6)
    ax.set_xlabel("sweep amplitude (V)"); ax.set_ylabel("branch shift (V)")
    _legend_below(fa, [Line2D([], [], color=BLUE, lw=0, marker="o", ms=4, label="measured, hole"),
                       Line2D([], [], color=BLUE, lw=1.4, label="model, hole"),
                       Line2D([], [], color=RED, lw=0, marker="s", ms=4, label="measured, electron"),
                       Line2D([], [], color=RED, lw=1.4, label="model, electron")], ncol=2)
    fb, ax = _band_axes(leg_h(2))
    for tl, col in (("25C", BLUE), ("125C", RED)):
        n = v["nbti_2023"][tl]
        ax.semilogx(n["t_s"], n["measured_dV_V"], "o", color=col, ms=3.5)
        ax.semilogx(n["model_curve_t_s"], n["model_curve_dV_V"], "-", color=col)
    ax.set_xlim(0.7, 1.4e3)
    ax.set_xlabel("stress time (s)"); ax.set_ylabel("ΔV$_{th}$ (V)")
    _legend_below(fb, [Line2D([], [], color=BLUE, lw=0, marker="o", ms=4, label="measured, 25 °C"),
                       Line2D([], [], color=BLUE, lw=1.4, label="model, 25 °C"),
                       Line2D([], [], color=RED, lw=0, marker="o", ms=4, label="measured, 125 °C"),
                       Line2D([], [], color=RED, lw=1.4, label="model, 125 °C")])
    return [fa, fb]


def figureS2():
    out = []
    for vg in (0.0, 10.0):
        f, ax = _band_axes()
        segs, xend, tun, ft = _band_panel(vg, ax)
        _band_steps(ax, segs, xend)
        extra = []
        if vg > 0:
            ax.set_ylim(-15.0, 6.5); ax.set_yticks([-15, -10, -5, 0, 5])
            cb_at = lambda xx: tun["cb"][0] + (xx - tun["xa"]) / (tun["xb"] - tun["xa"]) * (tun["cb"][1] - tun["cb"][0])
            sites = [(xend + 0.6, 0.0)] + [(xx, cb_at(xx) - 1.7) for xx in np.arange(ft[0] + 1.5, xend, 1.5)[::-1]] + [ft]
            assert all(b[1] < a[1] for a, b in zip(sites[:-1], sites[1:])), sites
            axi = ax.inset_axes([0.05, 0.63, 0.40, 0.35])
            x0, x1, y0, y1 = ft[0] - 0.8, xend + 1.4, ft[1] - 0.6, 0.8
            axi.axvspan(x0, xend, color=C_AL, lw=0, zorder=0)
            xx = np.linspace(x0, xend, 20)
            axi.plot(xx, [cb_at(q) for q in xx], "k", lw=0.9)
            axi.plot(xx, [cb_at(q) - 1.7 for q in xx], color=BLUE, lw=0.6, ls=(0, (1, 2)))
            axi.plot([xend, x1], [0, 0], color=GREEN, lw=1.2)
            axi.plot([ft[0]], [ft[1]], "s", color=MAGENTA, ms=6, zorder=6)
            for ax_, ms in ((ax, 4.5), (axi, 5.5)):
                ax_.plot([s[0] for s in sites[1:-1]], [s[1] for s in sites[1:-1]], "o", ms=ms, mfc="white", mec=BLUE, mew=1.3, zorder=7)
            for a, b in zip(sites[:-1], sites[1:]):
                axi.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=BLUE, lw=1.4, shrinkA=4,
                                              shrinkB=4, connectionstyle="arc3,rad=0.25", zorder=8))
            axi.set_xlim(x0, x1); axi.set_ylim(y0, y1); axi.set_xticks([]); axi.set_yticks([])
            ax.indicate_inset_zoom(axi, edgecolor="0.35")
            extra = [Line2D([], [], color=BLUE, lw=0, marker="o", ms=4.5, mfc="white", mew=1.3, label="hop sites of the captured electron"),
                     Line2D([], [], color=BLUE, lw=1.1, marker=">", ms=4, label="hops during the erase")]
        _band_legend(f, extra, base=(vg == 0.0))
        out.append(f)
    m81, v = J("hopping_feasibility.json"), J("model_views.json")
    eps0, q = 8.8541878128e-12, 1.602176634e-19
    need = float("%.3g" % (eps0 * v["si6_window_V"]["measured"] / (q * (15e-9 / 10.0 + 3e-9 / 25.0)) * 1e-4))
    assert abs(need - 1.18e13) < 0.01e13, need
    fc, ax = panel()
    Ns, phis = [2e19, 5e19, 1e20, 3e20], [1.6, 1.9, 2.3, 2.7]
    Z = np.array([[math.log10(max(m81["hopping"]["N%.0e_phi%.1f_m0.20" % (n, p)]["charge_in_1us_cm2"], 1.0) / need)
                   for n in Ns] for p in phis])
    im = ax.imshow(np.clip(Z, -6, 6), origin="lower", cmap="RdBu", vmin=-6, vmax=6, aspect="auto")
    for i in range(len(phis)):
        for j in range(len(Ns)):
            ax.text(j, i, "0" if abs(Z[i, j]) < 0.5 else "%+.0f" % Z[i, j], ha="center", va="center", fontsize=8)
    xb = math.log(1.6e19 / Ns[0]) / math.log(Ns[1] / Ns[0])
    yb0, yb1 = ((e - phis[0]) / (phis[1] - phis[0]) for e in mf.TRAP_AL)
    ax.plot([xb, xb], [yb0, yb1], color="k", lw=4, solid_capstyle="butt", label="trap band reported for Al$_2$O$_3$")
    ax.set_xlim(-0.5, len(Ns) - 0.5)
    ax.set_xticks(range(len(Ns))); ax.set_xticklabels(["2e19", "5e19", "1e20", "3e20"])
    ax.set_yticks(range(len(phis))); ax.set_yticklabels([str(p) for p in phis])
    ax.set_xlabel("defect density (cm$^{-3}$)"); ax.set_ylabel("trap depth φ (eV)")
    cb = fc.colorbar(im, ax=ax, pad=0.02); cb.set_label("log$_{10}$(moved/required)", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    _legend_outside_below(fc, ax, ncol=1)
    return out + [fc]


def _fixed_axes(leg_h_):
    fig = plt.figure(figsize=(BAND_W, BAND_H + leg_h_))
    return fig, fig.add_axes([0.75 / BAND_W, (leg_h_ + 0.5) / (BAND_H + leg_h_), (BAND_W - 0.85) / BAND_W,
                              (BAND_H - 0.75) / (BAND_H + leg_h_)])


def figureS3():
    e2 = J("edge_field.json")
    fa, ax = _fixed_axes(LEG_CH)
    for key, ls, lab in (("depleted|base_15_3_7", "-", "depleted channel"), ("conductor_1D_ref|base_15_3_7", ":", "conducting channel")):
        p = e2["profiles"][key]
        ax.loglog(np.array(p["x_nm"]), np.maximum(np.array(p["F_MV_cm"]), 1e-3), ls, color=BLUE, lw=1.5, label=lab)
    ax.set_xlim(0.2, 500); ax.set_ylim(5e-3, 12)
    ax.set_xlabel("distance from contact edge (nm)"); ax.set_ylabel("tunnel field (MV/cm)")
    h, lb = ax.get_legend_handles_labels()
    fa.legend(h, lb, loc="upper center", bbox_to_anchor=(0.5, LEG_CH / (BAND_H + LEG_CH) + 0.005), ncol=1, fontsize=7.5,
              frameon=False, labelspacing=0.4, handlelength=2.2)
    fl = J("hole_fill.json")["cases"]
    fb, ax = _fixed_axes(LEG_CH)
    ax.axvspan(1e-10, 1e-6, color="0.88", lw=0, zorder=0, label="within the shortest measured write (1 µs)")
    for key, col, ls, lab in (("base_phi2.0", BLUE, "-", "no stored charge, 2.0 eV contact barrier"),
                              ("base_phi2.2", RED, "--", "no stored charge, 2.2 eV contact barrier"),
                              ("sens_VG-12.9", ORANGE, "-.", "erased cell, 2.0 eV contact barrier")):
        t, p = np.array(fl[key]["t"]), np.array(fl[key]["p_mean_m2"]) * 1e-4 / 1e13
        ax.semilogx(t[t > 0], p[t > 0], ls=ls, color=col, lw=1.5, label=lab, zorder=3)
    ax.set_xlim(1e-10, 1e-1); ax.set_ylim(0, 2.8)
    ax.set_xticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2])
    ax.set_xlabel("time into the -10 V pulse (s)"); ax.set_ylabel("hole density (10$^{13}$ cm$^{-2}$)")
    h, lb = ax.get_legend_handles_labels()
    fb.legend(h[1:] + h[:1], lb[1:] + lb[:1], loc="upper center", bbox_to_anchor=(0.5, LEG_CH / (BAND_H + LEG_CH) + 0.005), ncol=1,
              fontsize=7.5, frameon=False, labelspacing=0.4, handlelength=2.2)
    return [fa, fb]


def figureS4():
    t178 = J("temperature_prediction.json")
    widths = [float(x) for x in t178["widths_s"]]
    Ts = t178["T_C"]
    f, ax = panel()
    for wd, col, wl in ((1e-6, BLUE, "1 µs"), (0.1, ORANGE, "100 ms")):
        j = widths.index(wd)
        for ea, ls, el in ((0.0, "-", ""), (0.05, "--", ", hop rate activated by 0.05 eV")):
            ax.plot(Ts, [t178["window_V"]["%g|%g" % (ea, T)][j] for T in Ts], ls, marker="o", ms=3.2, color=col, lw=1.4,
                    label="%s pulse%s" % (wl, el))
    ax.set_xlabel("temperature (°C)"); ax.set_ylabel("window (V)")
    ax.set_xlim(15, 135); ax.set_ylim(0, 6)
    _legend_outside_below(f, ax, ncol=1, fontsize=7, handlelength=3.6)
    return [f]


def figureS5():
    P = J("synaptic_response.json")["pairs"]
    cols = {"-7.5": PURPLE, "-8.0": GREEN, "-8.5": RED}
    h = [Line2D([], [], color=c, marker="o", ms=3, lw=1.0, label="%s V" % k) for k, c in cols.items()]
    fa, ax = panel()
    for k, c in cols.items():
        g = np.r_[P[k]["G_start_S"], P[k]["cycles"][0]["potentiation_S"]]
        ax.semilogy(range(len(g)), g * 1e9, "o-", color=c, ms=2.5, lw=1.0)
    ax.set_xlabel("number of 1 µs program pulses"); ax.set_ylabel("conductance (nS)")
    fa.legend(handles=h, loc="outside lower center", ncol=3, frameon=False, fontsize=7.5)
    fb, ax = panel()
    for k, c in cols.items():
        g = np.r_[P[k]["G_start_S"], P[k]["cycles"][0]["potentiation_S"]]
        ax.plot(range(len(g)), (g - g[0]) / (g[-1] - g[0]), "o-", color=c, ms=2.5, lw=1.0)
    ax.set_xlabel("number of 1 µs program pulses"); ax.set_ylabel("relative change"); ax.set_ylim(-0.03, 1.05)
    fb.legend(handles=h, loc="outside lower center", ncol=3, frameon=False, fontsize=7.5)
    return [fa, fb]


def figureS6():
    tr, m172 = load("base_si12_traces.yaml")["runs"], J("retention_projection.json")
    lt, la = np.log10(m172["t_s"]), np.array(m172["T300"]["dlogI_from_5s"])
    f, ax = panel()
    for k, col, lab in (("si12_10min", BLUE, "10 min run"), ("si12_1h", ORANGE, "1 h run")):
        t, l = np.array(tr[k]["t_s"]), np.array(tr[k]["log10_I_A"])
        ax.plot(t, l - l[0], "-", color=col, lw=1.6, label=lab + ", measured")
        tm = np.logspace(np.log10(t[0]), np.log10(t[-1]), 80)
        ax.plot(tm, np.interp(np.log10(tm), lt, la) - np.interp(np.log10(t[0]), lt, la), "--", color=col, lw=1.4,
                label=lab + ", model")
    ax.set_xscale("log"); ax.set_xlim(4, 5e3); ax.set_ylim(-1.4, 0.05)
    ax.set_xlabel("time after the write (s)"); ax.set_ylabel("Δlog$_{10}$ I$_{ON}$ (decade)")
    _legend_outside_below(f, ax, ncol=2, fontsize=7)
    return [f]


def figureS7():
    s7 = load("base_si7_redigitized.yaml")
    f, ax = panel()
    cmap = plt.get_cmap("cool")
    for vm, t0, t1, l0, l1, *_ in s7["segments"]:
        ax.plot([t0, t1], [l0, l1], color=cmap((3 - vm) / 6.0), lw=2.2)
    for z in [x for x in s7["segments"] if x[0] == 0]:
        ax.plot([z[1], z[2]], [z[3], z[4]], color="k", lw=0.7)
    ax.set_xlabel("time (s)"); ax.set_ylabel("log$_{10}$ I$_{DS}$ (A/µm)")
    ax.set_ylim(-11.6, -5.5)
    volts = sorted({x[0] for x in s7["segments"]}, reverse=True)
    hd = [Line2D([], [], color=cmap((3 - vm) / 6.0), lw=2.2, label="%+d V" % vm if vm else "0 V") for vm in volts]
    f.legend(handles=hd, loc="outside lower center", ncol=4, fontsize=7, frameon=False, handlelength=1.6, columnspacing=1.0,
             title="read-gate voltage", title_fontsize=7)
    return [f]


def figureS8():
    g = plt.figure(figsize=(6.2, 2.9))
    ax = g.add_axes([0.0, 0.0, 1.0, 0.95]); ax.set_xlim(0.5, 16.2); ax.set_ylim(0, 8.3); ax.axis("off")
    x0, x1, yt, yb = 2.4, 6.8, 7.0, 2.3
    rows = np.linspace(yt, yb, 5)
    cols = np.linspace(x0 + 0.35, x1 - 0.35, 6)
    for k, y in enumerate(rows):
        ax.plot([x0 - 0.3, x1], [y, y], color=GREEN, lw=1.0)
        lab = "x$_{%s}$" % ({len(rows) - 1: "784", len(rows) - 2: "i"}.get(k, str(k + 1)))
        ax.text(x0 - 0.4, y, lab, ha="right", va="center", fontsize=6.5, color=GREEN)
    for j, x in enumerate(cols):
        ax.plot([x, x], [yt + 0.3, yb - 0.55], color=RED if j % 2 == 0 else BLUE, lw=1.0)
        for y in rows:
            ax.add_patch(Rectangle((x - 0.16, y - 0.16), 0.32, 0.32, fc="white", ec="0.25", lw=0.6, zorder=3))
        ax.text(x, yt + 0.45, "G$^+$" if j % 2 == 0 else "G$^-$", ha="center", va="bottom", fontsize=6.5,
                color=RED if j % 2 == 0 else BLUE)
    for j in range(0, len(cols), 2):
        xm = 0.5 * (cols[j] + cols[j + 1])
        ax.add_patch(FancyBboxPatch((xm - 0.42, yb - 1.25), 0.84, 0.5, boxstyle="round,pad=0.02", fc="0.93", ec="0.3", lw=0.6))
        ax.text(xm, yb - 1.0, "I$^+$ - I$^-$", ha="center", va="center", fontsize=5.5)
    ax.text(0.5 * (x0 + x1), 0.45, "crossbar 1: 784 rows × 128 pairs", ha="center", va="center", fontsize=7.5, fontweight="bold")
    arr = dict(arrowstyle="-|>", color="0.25", lw=1.0, mutation_scale=8)
    ax.annotate("", xy=(7.75, 4.6), xytext=(6.95, 4.6), arrowprops=arr)
    ax.add_patch(FancyBboxPatch((7.8, 4.15), 1.1, 0.9, boxstyle="round,pad=0.04", fc="0.93", ec="0.3", lw=0.7))
    ax.text(8.35, 4.6, "ReLU", ha="center", va="center", fontsize=7)
    ax.text(8.35, 3.85, "128 hidden\nread pulses", ha="center", va="top", fontsize=6)
    ax.annotate("", xy=(9.8, 4.6), xytext=(9.0, 4.6), arrowprops=arr)
    bx0, bx1, byt, byb = 9.9, 11.4, 6.0, 3.2
    for y in np.linspace(byt, byb, 4):
        ax.plot([bx0, bx1], [y, y], color=GREEN, lw=0.9)
    for j, x in enumerate(np.linspace(bx0 + 0.25, bx1 - 0.25, 4)):
        ax.plot([x, x], [byt + 0.2, byb - 0.3], color=RED if j % 2 == 0 else BLUE, lw=0.9)
        for y in np.linspace(byt, byb, 4):
            ax.add_patch(Rectangle((x - 0.12, y - 0.12), 0.24, 0.24, fc="white", ec="0.25", lw=0.5, zorder=3))
    ax.text(0.5 * (bx0 + bx1), 2.5, "crossbar 2:\n128 rows × 10 pairs", ha="center", va="top", fontsize=6.5)
    ax.annotate("", xy=(12.4, 4.6), xytext=(11.55, 4.6), arrowprops=arr)
    ax.text(13.2, 4.6, "largest\noutput\n= digit", ha="center", va="center", fontsize=6.5)
    ax.text(13.7, 7.75, "inputs as 1 V read pulses of\n0-1 µs on the drain lines\ngates at V$_G$ = 0 V\nweight = G$^+$ - G$^-$, written by\n1 µs gate pulses",
            ha="center", va="top", fontsize=8, fontweight="bold", color="0.25", gid="keep")
    return [g]


def figureS9():
    cm = NPZ("smaller_crossbars_figure_data.npz")["B_28x28_single_784x20_confusion"]
    f, ax = panel()
    cmn = cm / cm.sum(1, keepdims=True) * 100
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=100)
    for i in range(10):
        ax.text(i, i, "%.0f" % cmn[i, i], ha="center", va="center", fontsize=6, color="white" if cmn[i, i] > 50 else "black")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("digit read by the crossbar"); ax.set_ylabel("true digit")
    cb = f.colorbar(im, ax=ax, pad=0.02); cb.set_label("share of test digits (%)", fontsize=8)
    return [f]


MAIN = [("figure1", figure1), ("figure2", figure2), ("figure3", figure3), ("figure4", figure4), ("figure5", figure5),
        ("figure6", figure6), ("figure7", figure7)]
SI = [("figureS1", figureS1), ("figureS2", figureS2), ("figureS3", figureS3), ("figureS4", figureS4), ("figureS5", figureS5),
      ("figureS6", figureS6), ("figureS7", figureS7), ("figureS8", figureS8), ("figureS9", figureS9)]


RESERVE_TOP = {"figureS4", "figureS6", "figureS7"}


def main(which):
    for name, fn in (MAIN if which != "si" else []) + (SI if which != "main" else []):
        figs = fn()
        if len(figs) == 1:
            if name in RESERVE_TOP:
                figs[0].suptitle("(a)", x=0.01, ha="left", fontsize=11, fontweight="bold", color="none")
            label_and_save(figs[0], name)
        else:
            for letter, fig in zip("abcdefgh", figs):
                label_and_save(fig, "%s_%s" % (name, letter), letter)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
