"""Final comparison of detector configurations on every group, from the cached records (offline replay, exact).

  board today            the shipped rules alone (USE_TFLM 0 on the ESP32: no classifier)
  shipped fused          the shipped rules + the classifier veto (what the laptop tools run)
  new rules              tonality test on local flatness (smooth shape divided out) instead of global flatness
  new rules + classifier
  new rules + DSP gate   the classifier's veto replaced by a linear gate on six tilt-invariant rolling features
                         (chosen on dev by interp.part_c --invariant; weights read from its JSON)
  shipped config         droneears/config.py as it stands, from the detector's own recorded flat_local and gate score
                         (the DSP gate threshold is the dev choice + 0.5; see config.py)

Usage: python -m interp.final
"""
import collections, json, os
import numpy as np
from interp import harness as H
from interp.part_c import load_all, key_of, REPORT
from droneears import config as C


def gate_from(path, creg):
    d = json.load(open(os.path.join(H.OUT, path)))
    g = next(g for g in d["gates"] if g["kind"] == "linear" and abs(g["C"] - creg) < 1e-9)
    return dict(weights=g["weights"], threshold=g["threshold"], flat_key=d["flat_key"], flat_min=d["flat_min"],
                flat_max=d["flat_max"])


def candidates():
    g6 = gate_from("part_c_flat_local0.3-1_inv.json", 0.03)
    g7 = gate_from("part_c_flat_local0.25-1_inv.json", 0.03)
    old = dict(flat_key="flat", flat_min=C.FLAT_MIN, flat_max=C.FLAT_THRESH)
    new = dict(flat_key="f:flat_local", flat_min=0.3, flat_max=1.0)
    return {
        "board today (rules only)": dict(old, gate=None),
        "shipped fused (rules + classifier)": dict(old, gate="ml"),
        "new rules (local flatness >= 0.3)": dict(new, gate=None),
        "new rules + classifier": dict(new, gate="ml"),
        "new rules + DSP gate (6 features)": dict(new, gate=g6),
        "new rules 0.25 + DSP gate (7 features)": dict(flat_key="f:flat_local", flat_min=0.25, flat_max=1.0, gate=g7),
        "shipped config (real detector)": dict(flat_key="flat_local", flat_min=C.FLAT_LOCAL_MIN, flat_max=C.FLAT_LOCAL_MAX,
                                               gate="detector"),
    }


def gate_mask(rec, gate):
    if gate is None:
        return None
    if gate == "ml":
        return rec["ml_ok"] > 0
    if gate == "detector":                       # the detector's own gate score (recorded), at config.GATE_THRESH
        return rec["gate"] >= C.GATE_THRESH
    return np.stack([rec[f] for f in gate["weights"]], 1) @ np.array(list(gate["weights"].values())) >= gate["threshold"]


def evaluate(clips, cfg):
    by = collections.defaultdict(lambda: dict(n=0, det=0, frames=0, det_frames=0, first=[]))
    for c in clips:
        rec = c["rec"]
        ok = H.rules_frame_ok(rec, flat_min=cfg["flat_min"], flat_max=cfg["flat_max"], flat_key=cfg["flat_key"])
        det = H.replay(ok, gate_mask(rec, cfg["gate"]))
        b = by[key_of(c["job"])]
        b["n"] += 1; b["det"] += int(det.any()); b["frames"] += len(det); b["det_frames"] += int(det.sum())
        if det.any():
            b["first"].append(int(np.argmax(det)) * C.HOP / C.FS)
    return {k: dict(clips=v["n"], detected=v["det"], frame_rate=v["det_frames"] / max(1, v["frames"]),
                    median_first_s=float(np.median(v["first"])) if v["first"] else None) for k, v in by.items()}


def main():
    clips = load_all()
    cands = candidates()
    res = {name: evaluate(clips, cfg) for name, cfg in cands.items()}
    keys = [k for k, _, _ in REPORT]
    print("| configuration | " + " | ".join(f"{s} {d}" for _, s, d in REPORT) + " |")
    print("|" + "---|" * (len(keys) + 1))
    for name, r in res.items():
        print(f"| {name} | " + " | ".join(f"{r[k]['detected']}/{r[k]['clips']}" if k in r else "-" for k in keys) + " |")
    print("\nmedian time to first detection (s) / share of frames DETECTED, drone groups")
    for name, r in res.items():
        cells = [f"{k.split(':')[0]} {r[k]['median_first_s'] or float('nan'):.1f} s / {r[k]['frame_rate']:.0%}"
                 for k in ("phone:dev", "phonemix:test", "dads:test")]
        print(f"  {name:40s} " + "   ".join(cells))
    out = dict(candidates={n: dict(c, gate=c["gate"] if isinstance(c["gate"], (dict, type(None))) else c["gate"])
                           for n, c in cands.items()}, results=res)
    json.dump(out, open(os.path.join(H.OUT, "final.json"), "w"), indent=1)
    print("wrote", os.path.join(H.OUT, "final.json"))


if __name__ == "__main__":
    main()
