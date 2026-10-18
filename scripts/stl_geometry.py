"""Haptic motor holes and band outline from the headband CAD (HH_CAD_V4_loops.stl), for the dashboard's calibration
tab: writes webapp/static/headband_geometry.json and docs/headband_geometry.png.

What is in the model (binary STL, mm, one closed mesh): a head-sized egg-shaped band wall, 20 mm tall (z 0..20), with
8 mm strap loops (horizontal tubes) along its bottom edge reaching z -4, and eight round through-holes of ~10 mm in the
wall at mid-height, one per coin motor (10 mm coin vibration motors), most with a small slit or diamond next to them
for the motor wires (the front and back holes also have a notch down to the bottom edge).

Detection: faces whose normal is perpendicular to the band's radial direction (|n . r| < 0.45) between z 4.4 and 19.6
(above the loops) are the bores of holes and slits through the wall; they are grouped by single-linkage clustering
(9 mm). For each group, its points are projected onto the wall's local frame (s along the wall, z) and a circle of
radius 4-7 mm is fitted by RANSAC (so the wire slit next to the hole is rejected), then refined by least squares on
the inliers. The hole centre is put at the middle of the wall thickness there.

Head frame: x right, y forward, z up (STL z: the bottom edge of the wall is z 0), origin at the centre of the wall's
outline. Front = STL -y by default (see FRONT_NOTE); --front +y uses the other end.

Usage:  python scripts/stl_geometry.py [--stl HH_CAD_V4_loops.stl] [--front -y|+y]
                                       [--out-json webapp/static/headband_geometry.json] [--out-png docs/headband_geometry.png]
"""
import argparse, json, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FRONT_NOTE = {
    "-y": ("Front = the STL's -y end (the narrower end of the band), up = STL +z (the strap loops hang below the wall). "
           "Evidence: the band outline is egg-shaped and wider behind than in front, like a skull seen from above "
           "(widest at STL y ~ +20 mm; at |y| = 100 mm the -y end is ~34 mm wide against ~41 mm at +y), and the STL "
           "was exported by Autodesk (ATF), whose FRONT view faces -Y in a Z-up model. Moderate confidence: check on "
           "the real band that H1 sits at the forehead; scripts/stl_geometry.py --front +y regenerates the other way."),
    "+y": ("Front = the STL's +y end (the wider end of the band), up = STL +z, chosen with --front +y. The default "
           "(-y) follows the skull-shape and Autodesk-front evidence; check on the real band that H1 is at the forehead."),
}


def load_stl(path):
    raw = open(path, "rb").read()
    n = int(np.frombuffer(raw, "<u4", 1, 80)[0])
    if 84 + 50 * n != len(raw):
        raise ValueError(f"{path}: not a binary STL (or truncated)")
    dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    return np.frombuffer(raw, dt, count=n, offset=84)["v"].astype(np.float64)        # (n, 3, 3)


def circle_lsq(s, z):
    """Algebraic least-squares circle through points (s, z): (s0, z0, R)."""
    M = np.stack([s, z, np.ones_like(s)], 1)
    a, *_ = np.linalg.lstsq(M, s ** 2 + z ** 2, rcond=None)
    s0, z0 = a[0] / 2, a[1] / 2
    return s0, z0, float(np.sqrt(max(a[2] + s0 ** 2 + z0 ** 2, 0.0)))


def circle_ransac(s, z, rmin=4.0, rmax=7.0, tol=0.3, iters=4000, seed=0, sectors=24):
    """Circle of radius rmin..rmax whose points within tol cover the most of its circumference (sectors of
    360/sectors deg with at least one point; ties: more points), refined on its inliers. Scoring by coverage, not by
    point count, keeps a densely meshed wire slit next to the hole from winning over the hole's own bore.
    (s0, z0, R, inlier mask)."""
    rng = np.random.default_rng(seed)
    best, best_key = None, (-1, -1)
    for _ in range(iters):
        i = rng.choice(len(s), 3, replace=False)
        try:
            s0, z0, R = circle_lsq(s[i], z[i])
        except np.linalg.LinAlgError:
            continue
        if not rmin <= R <= rmax:
            continue
        inl = np.abs(np.hypot(s - s0, z - z0) - R) < tol
        ang = np.arctan2(z[inl] - z0, s[inl] - s0)
        cover = len(np.unique(np.floor((ang + np.pi) / (2 * np.pi) * sectors).astype(int)))
        key = (cover, int(inl.sum()))
        if key > best_key:
            best, best_key = inl, key
    if best is None:
        raise RuntimeError("no circle found")
    s0, z0, R = circle_lsq(s[best], z[best])
    inl = np.abs(np.hypot(s - s0, z - z0) - R) < tol
    return s0, z0, R, inl


def analyse(V):
    C = V.mean(1)
    Nv = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0])
    area = np.linalg.norm(Nv, axis=1) / 2
    Nv /= np.maximum(2 * area, 1e-12)[:, None]
    P = V.reshape(-1, 3)
    # the wall alone (above the loops, below the top face): its outline centre is the head-frame origin
    wall_pts = P[(P[:, 2] > 5) & (P[:, 2] < 19)]
    cx = 0.5 * (wall_pts[:, 0].min() + wall_pts[:, 0].max())
    cy = 0.5 * (wall_pts[:, 1].min() + wall_pts[:, 1].max())
    xy = C[:, :2] - [cx, cy]
    r = np.hypot(xy[:, 0], xy[:, 1])
    ur = xy / np.maximum(r, 1e-9)[:, None]
    nr = np.abs(Nv[:, 0] * ur[:, 0] + Nv[:, 1] * ur[:, 1])
    bore = (nr < 0.45) & (C[:, 2] > 4.4) & (C[:, 2] < 19.6)
    wall = (nr > 0.8) & (C[:, 2] > 0.5) & (C[:, 2] < 19.5)
    from scipy.cluster.hierarchy import fcluster, linkage
    lab = fcluster(linkage(C[bore], "single"), t=9.0, criterion="distance")
    Bv = V[bore]; Ba = area[bore]; Bc = C[bore]
    holes = []
    for c in np.unique(lab):
        m = lab == c
        if Ba[m].sum() < 20:                                   # specks: not a hole
            continue
        ctr = Bc[m].mean(0)[:2] - [cx, cy]
        u = ctr / np.hypot(*ctr); t = np.array([u[1], -u[0]])
        pts = np.unique(np.round(Bv[m].reshape(-1, 3), 4), axis=0)
        d = pts[:, :2] - [cx, cy] - ctr
        s, z = d @ t, pts[:, 2]
        s0, z0, R, inl = circle_ransac(s, z)
        # wall mid-depth around the hole (inner and outer faces of the wall)
        dw = C[wall][:, :2] - [cx, cy] - ctr
        sw, zw, depth = dw @ t, C[wall][:, 2], dw @ u
        near = (np.abs(sw - s0) < R + 6) & (np.abs(zw - z0) < R + 4) & (np.hypot(sw - s0, zw - z0) > R + 0.5) \
            & (np.abs(depth) < 8)
        mid = 0.5 * (np.percentile(depth[near], 5) + np.percentile(depth[near], 95)) if near.sum() > 10 else 0.0
        p = ctr + s0 * t + mid * u
        holes.append(dict(x=float(p[0]), y=float(p[1]), z=float(z0), d=float(2 * R),
                          inlier_frac=float(inl.mean()), n_bore_faces=int(m.sum())))
    # outer outline of the wall for drawing: 90th-percentile radius per 2 degree bin, then a circular 5-bin median, so
    # hole rims and wire slits do not show up as spikes
    wq = wall_pts[:, :2] - [cx, cy]
    th = np.degrees(np.arctan2(wq[:, 0], wq[:, 1])); rq = np.hypot(wq[:, 0], wq[:, 1])
    bins = np.arange(-180, 180, 2.0)
    rad = np.array([np.percentile(rq[(th >= b) & (th < b + 2)], 90) for b in bins])
    rad = np.median(np.stack([np.roll(rad, k) for k in range(-2, 3)]), axis=0)
    mid = np.radians(bins + 1.0)
    outline = np.stack([rad * np.sin(mid), rad * np.cos(mid)], 1)
    return dict(centre=(float(cx), float(cy)), holes=holes, outline=outline)


def to_head(x, y, front):
    """STL xy (relative to the band centre) -> head frame. front -y: a 180 degree turn about z."""
    return (-x, -y) if front == "-y" else (x, y)


def theta_deg(x, y):
    t = float(np.degrees(np.arctan2(x, y)))
    return 180.0 if t <= -180.0 + 1e-9 else t


def build(stl, front):
    res = analyse(load_stl(stl))
    holes = []
    for h in res["holes"]:
        x, y = to_head(h["x"], h["y"], front)
        holes.append(dict(theta_deg=round(theta_deg(x, y), 1), x_mm=round(x, 1), y_mm=round(y, 1), z_mm=round(h["z"], 1),
                          diameter_mm=round(h["d"], 1)))
    holes.sort(key=lambda h: h["theta_deg"] if h["theta_deg"] >= 0 else h["theta_deg"] + 360)   # clockwise from front
    holes = [dict(id=f"H{i + 1}", **h) for i, h in enumerate(holes)]
    outline = [[round(float(a), 1), round(float(b), 1)] for a, b in (to_head(p[0], p[1], front) for p in res["outline"])]
    outline.sort(key=lambda p: theta_deg(*p) if theta_deg(*p) >= 0 else theta_deg(*p) + 360)
    geo = {"source": os.path.basename(stl), "units": "mm",
           "frame": "x right, y forward, z up, origin at the band centre (z 0 = bottom edge of the wall); "
                    "theta_deg = atan2(x, y): 0 front, +90 right",
           "front_axis_note": FRONT_NOTE[front],
           "outline": outline, "holes": holes, "mic_sites": []}
    return geo, res


def plot(geo, res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    ink, muted, blue, orange = "#0b0b0b", "#898781", "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw=dict(width_ratios=[1, 1.5]))
    o = np.array(geo["outline"] + geo["outline"][:1])
    a = ax[0]
    a.plot(o[:, 0], o[:, 1], color=muted, lw=1.5)
    for h in geo["holes"]:
        a.add_patch(plt.Circle((h["x_mm"], h["y_mm"]), h["diameter_mm"] / 2, color=orange, alpha=0.85))
        rr = np.hypot(h["x_mm"], h["y_mm"]); k = (rr + 16) / rr
        a.text(h["x_mm"] * k, h["y_mm"] * k, f"{h['id']}\n{h['theta_deg']:+.0f}°", ha="center", va="center",
               fontsize=9, color=ink)
    a.annotate("", xy=(0, 145), xytext=(0, 120), arrowprops=dict(arrowstyle="-|>", color=blue, lw=2))
    a.text(0, 150, "FRONT (nose)", ha="center", va="bottom", color=blue, fontsize=10, weight="bold")
    a.text(140, 0, "right", ha="center", va="center", color=muted); a.text(-140, 0, "left", ha="center", color=muted)
    a.set_xlim(-160, 160); a.set_ylim(-150, 170); a.set_aspect("equal")
    a.set_title("Top view, head frame (x right, y forward)", loc="left", fontsize=11)
    a.set_xlabel("x (mm)"); a.set_ylabel("y (mm)")
    b = ax[1]
    for h in geo["holes"]:
        rr = np.hypot(h["x_mm"], h["y_mm"])
        w = np.degrees(h["diameter_mm"] / rr)
        b.add_patch(Ellipse((h["theta_deg"], h["z_mm"]), w, h["diameter_mm"], color=orange, alpha=0.85))
        b.text(h["theta_deg"], h["z_mm"] + h["diameter_mm"] / 2 + 1.5, f"{h['id']} {h['theta_deg']:+.0f}°\n"
               f"z {h['z_mm']:.0f}, ⌀{h['diameter_mm']:.1f}", ha="center", va="bottom", fontsize=8, color=ink)
    b.axhspan(0, 20, color="#e1e0d9", alpha=0.6, lw=0); b.axhspan(-4, 0, color="#f0efec", alpha=0.8, lw=0)
    b.text(-178, 19, "band wall (z 0..20)", va="top", fontsize=8, color=muted)
    b.text(-178, -3.5, "strap loops (z -4..4)", va="bottom", fontsize=8, color=muted)
    b.set_xlim(-185, 185); b.set_ylim(-6, 26); b.set_xticks(range(-180, 181, 45))
    b.set_xlabel("theta (deg): 0 front, +90 right, ±180 back"); b.set_ylabel("z (mm)")
    b.set_title("Unwrapped band: motor holes", loc="left", fontsize=11)
    fig.suptitle(f"{geo['source']}: {len(geo['holes'])} motor holes", fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stl", default=os.path.join(ROOT, "HH_CAD_V4_loops.stl"))
    ap.add_argument("--front", choices=["-y", "+y"], default="-y", help="which STL end is the front (default -y)")
    ap.add_argument("--out-json", default=os.path.join(ROOT, "webapp", "static", "headband_geometry.json"))
    ap.add_argument("--out-png", default=os.path.join(ROOT, "docs", "headband_geometry.png"))
    args = ap.parse_args()
    geo, res = build(args.stl, args.front)
    with open(args.out_json, "w") as f:
        json.dump(geo, f, indent=1)
    plot(geo, res, args.out_png)
    print(f"band centre in STL coordinates: {res['centre'][0]:.2f}, {res['centre'][1]:.2f} mm; front = STL {args.front}")
    for h in geo["holes"]:
        print(f"  {h['id']}: theta {h['theta_deg']:+7.1f}  x {h['x_mm']:7.1f} y {h['y_mm']:7.1f} z {h['z_mm']:5.1f}  "
              f"diameter {h['diameter_mm']:.1f} mm")
    print("RANSAC inlier fraction of each bore (rest = wire slit):",
          ", ".join(f"{q['inlier_frac']:.2f}" for q in res["holes"]))
    print("wrote", args.out_json, "and", args.out_png)


if __name__ == "__main__":
    main()
