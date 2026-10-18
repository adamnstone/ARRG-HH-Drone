"""Figures for the interp write-up, from interp/out/part_{a,b,c}.json. Static PNGs (matplotlib), light theme.

Usage: python -m interp.figures [--parts a,b,c]
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from interp import harness as H

# reference palette (dataviz skill): categorical slot 1 = classifier, slot 2 = rules; ink and chrome tokens
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CLS, RULES = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 2,
})

NUMERIC = {"f0": "fundamental f0 (Hz)", "n_harm": "number of harmonics", "inharm": "mistuning of partials (%)",
           "snr": "drone level vs ambient (dB SNR)", "aero": "broadband airflow floor (rel. RMS)",
           "wobble_depth": "pitch wobble depth (%)", "wobble_rate": "pitch wobble rate (Hz)",
           "am_depth": "volume wobble depth", "fill": "fraction of the window the sound fills",
           "n_motors": "number of motors", "rpm_spread": "motor speed spread (%)"}
CATEGORICAL = {"band_only": "harmonics in one octave band only (f0 200 Hz)", "drop_one": "one harmonic removed (k)",
               "structure": "which partials are present"}
LOGX = {"f0", "wobble_rate"}


def part_a():
    d = json.load(open(os.path.join(H.OUT, "part_a.json")))
    sw = d["sweeps"]
    names = [k for k in NUMERIC if k in sw]
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 2.7 * rows), squeeze=False)
    for ax, name in zip(axes.flat, names):
        lst = sw[name]
        x = np.array([float(r["value"]) for r in lst])
        ax.plot(x, [r["p_mean"] for r in lst], color=CLS, marker="o", ms=4, label="classifier: mean p(drone)")
        if lst[0]["rules_detect"] is not None:
            ax.plot(x, [r["rules_detect"] for r in lst], color=RULES, marker="s", ms=4,
                    label="shipped rules: share of 4 s clips DETECTED")
        ax.axhline(0.7, color=MUTED, lw=1, ls=(0, (3, 3)))
        ax.set_ylim(-0.03, 1.03); ax.set_xlabel(NUMERIC[name])
        if name in LOGX:
            ax.set_xscale("log")
            ticks = [50, 100, 200, 400, 800] if name == "f0" else [0.1, 0.3, 1, 3, 10]
            ax.set_xticks(ticks, [f"{t:g}" for t in ticks]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        if name == "f0":
            ax.axvspan(120, 400, color=GRID, alpha=0.5, lw=0)
            ax.text(125, 0.04, "rules' f0 search", color=MUTED, fontsize=8)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("What the classifier calls a drone: synthetic probes (dashed: 0.7 veto threshold)", y=1.04, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(H.OUT, "part_a_numeric.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    cats = [k for k in CATEGORICAL if k in sw]
    fig, axes = plt.subplots(1, len(cats), figsize=(5 * len(cats), 3.6), squeeze=False)
    for ax, name in zip(axes.flat, cats):
        lst = sw[name]
        labels = [str(r["value"]) for r in lst]
        y = np.arange(len(lst))
        ax.barh(y + 0.2, [r["p_mean"] for r in lst], height=0.38, color=CLS, label="classifier: mean p(drone)")
        if lst[0]["rules_detect"] is not None:
            ax.barh(y - 0.2, [r["rules_detect"] for r in lst], height=0.38, color=RULES, label="shipped rules: share DETECTED")
        ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlim(0, 1.0)
        ax.axvline(0.7, color=MUTED, lw=1, ls=(0, (3, 3)))
        ax.set_title(CATEGORICAL[name], loc="left"); ax.grid(axis="y", visible=False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    fig.savefig(os.path.join(H.OUT, "part_a_structure.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def part_b():
    d = json.load(open(os.path.join(H.OUT, "part_b.json")))
    drones = {k: v for k, v in d.items() if k.startswith("drone:")}
    edits = [e for e in next(iter(drones.values())) if e != "orig"]
    order = ["no_peaks", "no_harm_peaks", "no_other_peaks", "flat_shape", "shuffle", "time_avg"] + \
            [e for e in edits if e.startswith("no_peaks_")] + [e for e in edits if e.startswith("only_peaks_")]
    order = [e for e in order if e in edits]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(order) + 1.6))
    y = np.arange(len(order) + 1)

    def nice(e):
        fixed = dict(no_peaks="remove all peaks", no_harm_peaks="remove peaks at k*f0", no_other_peaks="keep only peaks at k*f0",
                     flat_shape="flatten the smooth shape", shuffle="shuffle frames in time", time_avg="average the window")
        if e in fixed:
            return fixed[e]
        kind, band = e.rsplit("_", 1)
        return ("remove peaks " if kind == "no_peaks" else "keep only peaks ") + band.replace("-", "–") + " Hz"
    labels = ["original"] + [nice(e) for e in order]
    for i, e in enumerate(["orig"] + order):
        ps = np.array([v[e]["cls"]["median_p"] for v in drones.values()])
        ax.scatter(ps, np.full(len(ps), i) + np.linspace(-0.15, 0.15, len(ps)), s=14, color=CLS, alpha=0.75, lw=0)
        ax.plot([np.median(ps)] * 2, [i - 0.3, i + 0.3], color=INK, lw=2)
        r = [v[e]["rules"]["frame_ok_frac"] for v in drones.values() if v[e]["rules"]]
        if r:
            ax.scatter([np.median(r)], [i], marker="D", s=26, color=RULES, zorder=3)
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlim(-0.02, 1.02)
    ax.axvline(0.7, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.set_xlabel("median p(drone) per phone recording (dots), median over recordings (bar)")
    ax.scatter([], [], s=14, color=CLS, label="classifier p(drone), one dot per recording")
    ax.scatter([], [], marker="D", s=26, color=RULES, label="rules: median share of frames passing")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Edits to the 9 real drone recordings", loc="left"); ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(H.OUT, "part_b_edits.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def part_c(name="part_c_flat_local0.3-1_inv.json"):
    d = json.load(open(os.path.join(H.OUT, name)))
    rk = d["ranking"][:15]
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(rk) + 1.2))
    y = np.arange(len(rk))
    ax.barh(y, [r["auc"] for r in rk], color=CLS, height=0.6)
    ax.set_yticks(y, [f"{r['feature'][2:]} ({r['direction']})" for r in rk]); ax.invert_yaxis()
    ax.set_xlim(0.5, 1.0); ax.set_xlabel("AUC: drone clips vs. rules' false-alarm clips, at the moment the rules fire (dev)")
    for i, r in enumerate(rk):
        ax.text(r["auc"] + 0.005, i, f"{r['auc']:.2f}", va="center", fontsize=8, color=INK2)
    ax.set_title("Tilt-invariant measurements that separate drones from the rules' false alarms", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(H.OUT, "part_c_ranking.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


FINAL_SHOW = [("board today (rules only)", "board today: rules, no classifier", "#2a78d6"),
              ("shipped fused (rules + classifier)", "laptop today: rules + classifier", "#eb6834"),
              ("shipped config (real detector)", "new: local flatness + DSP gate", "#1baf7a")]


def final():
    d = json.load(open(os.path.join(H.OUT, "final.json")))["results"]
    from interp.part_c import REPORT
    short = {"phone:dev": "phone\ndrones", "phonemix:test": "phone drones\nin noise", "dads:test": "DADS drones\n(held out)",
             "tilt+:robust": "drones,\nmic tilt", "esc50:test": "ESC-50\nfold 5", "confmix:test": "confusers\nin noise",
             "dadsbg:test": "DADS\nno-drone", "ambient:test": "phone\nambient", "tilt-:robust": "rule-firing\nESC-50, tilt"}
    pos = [(k, short[k]) for k, s, _ in REPORT if s == "+"]
    neg = [(k, short[k]) for k, s, _ in REPORT if s == "-"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[len(pos), len(neg)]))
    for ax, rows, title in ((axes[0], pos, "Drone clips detected (higher is better)"),
                            (axes[1], neg, "Clips with a false alarm (lower is better)")):
        x = np.arange(len(rows)); w = 0.26
        for i, (name, label, color) in enumerate(FINAL_SHOW):
            r = d[name]
            v = [r[k]["detected"] / r[k]["clips"] for k, _ in rows]
            bars = ax.bar(x + (i - 1) * w, v, width=w - 0.02, color=color, label=label)
            for b, k in zip(bars, [k for k, _ in rows]):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012, f"{r[k]['detected']}",
                        ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_xticks(x, [f"{desc}\n(n={d[FINAL_SHOW[0][0]][k]['clips']})" for k, desc in rows], fontsize=8)
        ax.set_ylim(0, 1.1); ax.set_title(title, loc="left"); ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.suptitle("Detector configurations on held-out and stress-test audio (bar labels: clip counts)", y=1.08,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(H.OUT, "final_comparison.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="a,b,c,final")
    args = ap.parse_args()
    files = dict(a="part_a.json", b="part_b.json", c="part_c_flat_local0.3-1_inv.json", final="final.json")
    for p in args.parts.split(","):
        if os.path.exists(os.path.join(H.OUT, files[p])):
            {"a": part_a, "b": part_b, "c": part_c, "final": final}[p]()
            print("figures for", p)


if __name__ == "__main__":
    main()
