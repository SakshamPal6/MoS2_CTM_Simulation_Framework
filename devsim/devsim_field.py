from __future__ import annotations

import json
import sys
import traceback

import numpy as np

try:
    import devsim as ds
except ImportError:
    ds = None

NM = 1e-7
EPS0 = 8.8541878128e-14
VG = -10.0
T_MOS2, EPS_MOS2, T_CONTACT = 0.65 * NM, 6.0, 40 * NM
XL, XR, YTOP = -300 * NM, 500 * NM, 2000 * NM
STACKS = {"base": ((15 * NM, 10.0), (3 * NM, 25.0), (7 * NM, 10.0)),
          "s2023": ((40 * NM, 10.0), (3 * NM, 25.0), (7 * NM, 10.0))}


def geometric(a, b, h0, growth, hmax):
    pts, h = [a], h0
    while pts[-1] < b:
        pts.append(min(pts[-1] + h, b))
        h = min(h * growth, hmax)
    return np.array(pts)


def clean(v):
    return np.unique(np.round(np.asarray(v) / 1e-12) * 1e-12)


def mesh_lines(stack, conductor, refine=1.0):
    ts = sum(t for t, _ in stack)
    y = [np.arange(0.0, ts + 1e-15, 0.25 * NM / refine)]
    if not conductor:
        y.append(np.linspace(ts, ts + T_MOS2, 4))
        y.append(geometric(ts + T_MOS2, YTOP, 0.25 * NM / refine, 1.08, 50 * NM))
        y.append([ts + T_MOS2 + T_CONTACT])
    xr = geometric(0.0, XR, 0.2 * NM / refine, 1.06, 10 * NM)
    xl = -geometric(0.0, -XL, 0.2 * NM / refine, 1.06, 10 * NM)
    return clean(np.concatenate([xl, xr])), clean(np.concatenate(y)), ts


def layout(stack, conductor):
    tb, th, tt = (t for t, _ in stack)
    ts = tb + th + tt
    yc = ts + T_MOS2 + T_CONTACT
    regions = [("blk", "oxide", XL, XR, 0.0, tb, stack[0][1]), ("hfo", "oxide", XL, XR, tb, tb + th, stack[1][1]),
               ("tun", "oxide", XL, XR, tb + th, ts, stack[2][1])]
    contacts = [("gate", "blk", VG)]
    interfaces = [("i_blk_hfo", "blk", "hfo"), ("i_hfo_tun", "hfo", "tun")]
    if conductor:
        contacts.append(("top", "tun", 0.0))
    else:
        regions += [("mos2", "mos2", 0.0, XR, ts, ts + T_MOS2, EPS_MOS2),
                    ("gas_r", "gas", 0.0, XR, ts + T_MOS2, YTOP, 1.0), ("gas_l", "gas", XL, 0.0, yc, YTOP, 1.0)]
        contacts += [("s_tun", "tun", 0.0), ("s_mos2", "mos2", 0.0), ("s_gasr", "gas_r", 0.0), ("s_gasl", "gas_l", 0.0)]
        interfaces += [("i_tun_mos2", "tun", "mos2"), ("i_mos2_gas", "mos2", "gas_r"), ("i_gas", "gas_r", "gas_l")]
    return regions, contacts, interfaces


def tri_mesh(xs, ys, stack, conductor):
    regions, contacts, interfaces = layout(stack, conductor)
    ts = sum(t for t, _ in stack)
    yc = ts + T_MOS2 + T_CONTACT
    nx, ny = len(xs), len(ys)
    reg = -np.ones((nx - 1, ny - 1), int)
    xm, ym = 0.5 * (xs[1:] + xs[:-1]), 0.5 * (ys[1:] + ys[:-1])
    for k, (_, _, x0, x1, y0, y1, _) in enumerate(regions):
        reg[np.ix_((xm > x0) & (xm < x1), (ym > y0) & (ym < y1))] = k
    names = [r[0] for r in regions]

    def cell(i, j):
        return names[reg[i, j]] if 0 <= i < nx - 1 and 0 <= j < ny - 1 and reg[i, j] >= 0 else None

    tol = 1e-12

    def on_y(a, b, v):
        return abs(a[1] - v) < tol and abs(b[1] - v) < tol

    def on_x(a, b, v):
        return abs(a[0] - v) < tol and abs(b[0] - v) < tol

    zone = {"gate": lambda a, b: on_y(a, b, 0.0), "top": lambda a, b: on_y(a, b, ts),
            "s_tun": lambda a, b: on_y(a, b, ts), "s_mos2": lambda a, b: on_x(a, b, 0.0),
            "s_gasr": lambda a, b: on_x(a, b, 0.0), "s_gasl": lambda a, b: on_y(a, b, yc)}
    groups = {c[0]: [] for c in contacts}
    igroups = {i[0]: [] for i in interfaces}
    edges = []
    for i in range(nx - 1):
        for j in range(ny):
            edges.append(((i, j), (i + 1, j), cell(i, j - 1), cell(i, j)))
    for i in range(nx):
        for j in range(ny - 1):
            edges.append(((i, j), (i, j + 1), cell(i - 1, j), cell(i, j)))
    for a, b, r1, r2 in edges:
        pa, pb = (xs[a[0]], ys[a[1]]), (xs[b[0]], ys[b[1]])
        for c, r, _ in contacts:
            if ((r1 == r and r2 is None) or (r2 == r and r1 is None)) and zone[c](pa, pb):
                groups[c].append((a, b))
    cnodes = {n for e in groups.values() for ab in e for n in ab}
    for a, b, r1, r2 in edges:
        for name, r0, rr in interfaces:
            if {r1, r2} == {r0, rr} and a not in cnodes and b not in cnodes:
                igroups[name].append((a, b))
    used = sorted({(i + di, j + dj) for i in range(nx - 1) for j in range(ny - 1) if reg[i, j] >= 0
                   for di in (0, 1) for dj in (0, 1)})
    nid = {n: k + 1 for k, n in enumerate(used)}
    tris = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            if reg[i, j] >= 0:
                n00, n10, n11, n01 = nid[(i, j)], nid[(i + 1, j)], nid[(i + 1, j + 1)], nid[(i, j + 1)]
                tris += [(names[reg[i, j]], (n00, n10, n11)), (names[reg[i, j]], (n00, n11, n01))]
    nodes = [(xs[i], ys[j]) for i, j in used]
    lines = {g: [(nid[a], nid[b]) for a, b in e] for g, e in list(groups.items()) + list(igroups.items())}
    return nodes, tris, lines, regions, contacts, interfaces


def write_msh(path, nodes, tris, lines):
    rnames = list(dict.fromkeys(r for r, _ in tris))
    tag = {g: k + 1 for k, g in enumerate(list(lines) + rnames)}
    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(tag))]
    out += ['1 %d "%s"' % (tag[g], g) for g in lines] + ['2 %d "%s"' % (tag[r], r) for r in rnames]
    out += ["$EndPhysicalNames", "$Nodes", str(len(nodes))]
    out += ["%d %.17g %.17g 0" % (k + 1, x, y) for k, (x, y) in enumerate(nodes)]
    out += ["$EndNodes", "$Elements", str(sum(len(v) for v in lines.values()) + len(tris))]
    k = 0
    for g, segs in lines.items():
        for a, b in segs:
            k += 1
            out.append("%d 1 2 %d %d %d %d" % (k, tag[g], tag[g], a, b))
    for r, (a, b, c) in tris:
        k += 1
        out.append("%d 2 2 %d %d %d %d %d" % (k, tag[r], tag[r], a, b, c))
    out.append("$EndElements")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def build(name, stack, conductor, refine=1.0):
    xs, ys, ts = mesh_lines(stack, conductor, refine)
    nodes, tris, lines, regions, contacts, interfaces = tri_mesh(xs, ys, stack, conductor)
    empty = [g for g, v in lines.items() if not v]
    assert not empty, "empty contact / interface groups: %s" % empty
    m = name
    write_msh(m + ".msh", nodes, tris, lines)
    ds.create_gmsh_mesh(mesh=m, file=m + ".msh")
    for r, mat, *_ in regions:
        ds.add_gmsh_region(mesh=m, gmsh_name=r, region=r, material=mat)
    for c, r, _ in contacts:
        ds.add_gmsh_contact(mesh=m, gmsh_name=c, name=c, region=r, material="metal")
    for i, r0, r1 in interfaces:
        ds.add_gmsh_interface(mesh=m, gmsh_name=i, name=i, region0=r0, region1=r1)
    ds.finalize_mesh(mesh=m)
    ds.create_device(mesh=m, device=m)
    print(m, "mesh:", len(nodes), "nodes,", len(tris), "triangles;",
          ", ".join("%s %d" % (g, len(v)) for g, v in lines.items()), flush=True)
    for r, _, _, _, _, _, eps in regions:
        poisson(m, r, eps)
    for c, _, v in contacts:
        ds.set_parameter(device=m, name=f"{c}_bias", value=v)
        ds.contact_node_model(device=m, contact=c, name=f"{c}_bc", equation=f"Potential - {c}_bias")
        ds.contact_node_model(device=m, contact=c, name=f"{c}_bc:Potential", equation="1")
        ds.contact_equation(device=m, contact=c, name="PotentialEquation", node_model=f"{c}_bc",
                            edge_charge_model="DField")
    for i, _, _ in interfaces:
        ds.interface_model(device=m, interface=i, name="contPot", equation="Potential@r0 - Potential@r1")
        ds.interface_model(device=m, interface=i, name="contPot:Potential@r0", equation="1")
        ds.interface_model(device=m, interface=i, name="contPot:Potential@r1", equation="-1")
        ds.interface_equation(device=m, interface=i, name="PotentialEquation", interface_model="contPot",
                              type="continuous")
    return m, xs, ts, stack


def poisson(dev, region, eps):
    ds.node_solution(device=dev, region=region, name="Potential")
    ds.edge_from_node_model(device=dev, region=region, node_model="Potential")
    ds.set_parameter(device=dev, region=region, name="Permittivity", value=eps * EPS0)
    ds.edge_model(device=dev, region=region, name="ElectricField",
                  equation="(Potential@n0 - Potential@n1) * EdgeInverseLength")
    ds.edge_model(device=dev, region=region, name="ElectricField:Potential@n0", equation="EdgeInverseLength")
    ds.edge_model(device=dev, region=region, name="ElectricField:Potential@n1", equation="-EdgeInverseLength")
    ds.edge_model(device=dev, region=region, name="DField", equation="Permittivity * ElectricField")
    ds.edge_model(device=dev, region=region, name="DField:Potential@n0", equation="Permittivity * EdgeInverseLength")
    ds.edge_model(device=dev, region=region, name="DField:Potential@n1", equation="-Permittivity * EdgeInverseLength")
    ds.equation(device=dev, region=region, name="PotentialEquation", variable_name="Potential",
                edge_model="DField", variable_update="default")


def tunnel_field(dev, stack, ts):
    x = np.array(ds.get_node_model_values(device=dev, region="tun", name="x"))
    y = np.array(ds.get_node_model_values(device=dev, region="tun", name="y"))
    p = np.array(ds.get_node_model_values(device=dev, region="tun", name="Potential"))
    tb, th, tt = (t for t, _ in stack)
    bot, top = np.abs(y - (tb + th)) < 1e-11, np.abs(y - ts) < 1e-11
    pb = dict(zip(np.round(x[bot], 13), p[bot]))
    pt = dict(zip(np.round(x[top], 13), p[top]))
    xs = np.array(sorted(set(pb) & set(pt)))
    f = np.array([abs(pt[k] - pb[k]) / tt / 1e6 for k in xs])
    return xs, f


def metrics(xs, f):
    ch = xs > 0
    xm, fm = xs[ch], f[ch]
    w = np.gradient(xm)
    return {"max_abs_F_MV_cm": round(float(fm.max()), 3), "x_at_max_nm": round(float(xm[np.argmax(fm)]) / NM, 2),
            "mean_abs_F_MV_cm": round(float(np.sum(fm * w) / np.sum(w)), 3),
            "F_centre_MV_cm": round(float(fm[-1]), 4),
            "F_under_contact_MV_cm": round(float(f[xs < -50 * NM].mean()), 3),
            "width_F_ge_half_max_nm": round(float(np.sum(w[fm >= 0.5 * fm.max()])) / NM, 2)}


def dc_solve():
    kw = dict(type="dc", absolute_error=1e10, relative_error=1e-7, maximum_iterations=30)
    try:
        info = ds.solve(info=True, **kw)
    except Exception as e:
        if "info" not in str(e):
            raise
        ds.solve(**kw)
        return {"converged": True, "info": "no info flag"}
    if isinstance(info, dict):
        its = info.get("iterations", [])
        last = its[-1] if its and isinstance(its[-1], dict) else {}
        return {"converged": bool(info.get("converged")), "iterations": len(its),
                "last": {k: last[k] for k in list(last)[:6] if isinstance(last.get(k), (int, float, str, bool))}}
    return {"converged": None, "info": str(info)[:300]}


def solve_case(tag, stack, conductor, refine):
    print(tag, "stage: mesh + device", flush=True)
    dev, xs, ts, st = build(tag, stack, conductor, refine)
    print(tag, "stage: solve", flush=True)
    conv = dc_solve()
    print(tag, "convergence:", conv, flush=True)
    out = metrics(*tunnel_field(dev, st, ts))
    out["convergence"] = conv
    print(tag, out, flush=True)
    return out


def run(refine=1.0):
    print("DEVSIM", getattr(ds, "__version__", "?"), "direct solver: default (MKL PARDISO when MKL is loaded)", flush=True)
    res = {}
    for case, cond in (("conductor", True), ("depleted", False)):
        for s, st in STACKS.items():
            tag = f"{case}_{s}_r{refine:g}"
            try:
                res[(case, s)] = solve_case(tag, st, cond, refine)
            except Exception as e:
                print(tag, "FAILED:", repr(e), flush=True)
                traceback.print_exc()
                res[(case, s)] = {"error": repr(e)[:500]}
    return res


if __name__ == "__main__":
    refine = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    r = run(refine)
    out = {"tool": f"DEVSIM {getattr(ds, '__version__', '?')}", "refine": refine, "VG": VG,
           "results": {f"{c}_{s}": m for (c, s), m in r.items()},
           "label": "CALCULATION with an independent code: field in the tunnel oxide"}
    json.dump(out, open(f"devsim_field_r{refine:g}.json", "w"), indent=1)
