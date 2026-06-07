"""Paper-style mechanism figure (Fig. 2) from exp10 result.json.

Reads <exp_dir>/result.json -- curves of decode(w_t) Chamfer-to-GT vs reverse step
for OURS (per-step latent) vs the one-shot One-shot+DPS latent -- and renders a clean
vector figure. Torch-free (matplotlib only); re-style without re-running exp10.

One noise  -> single compact single-column panel (recommended for the paper; pick the
              heaviest noise where the gap is clearest).
N noises   -> N side-by-side panels sharing the y-axis (double-column / supplementary).

Examples (from project/diffusion-point-cloud):
    # single-column paper figure (sigma_b=0.3 only)
    python experiments/plot_mechanism.py --exp_dir output/mechanism \
        --noises 0.3 --out output/mechanism/figures/fig_mechanism.pdf
    # both noises side by side
    python experiments/plot_mechanism.py --exp_dir output/mechanism
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANALYZE_DIR = os.path.dirname(os.path.abspath(__file__))
if ANALYZE_DIR not in sys.path:
    sys.path.insert(0, ANALYZE_DIR)
from paper_style import apply as apply_paper_style, PALETTE

# (result.json policy key, legend label, color, linestyle, marker). A series with no
# matching curve in result.json is silently skipped.
SERIES = [
    ("ours",            "Ours (per-step latent)",           PALETTE["ours"],     "-",  "o"),
    ("oneshot_kw",      "One-shot ($K_w$, $=$ours step 1)", "#9E78B5",           ":",  "^"),
    ("oneshot_matched", "One-shot",                         PALETTE["baseline"], "--", "s"),
]


def parse_csv_floats(v):
    if v is None or v == "":
        return None
    return [float(s) for s in str(v).split(",") if s.strip()]


def load_result(exp_dir):
    with open(os.path.join(exp_dir, "result.json")) as f:
        r = json.load(f)
    return r.get("config", {}), r.get("curves", {})


def get_series(curves, noise, policy):
    return curves.get("noise%g_%s" % (noise, policy))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--noises", type=parse_csv_floats, default=None,
                    help="noise panels to draw (default: all in result.json). One -> single panel.")
    ap.add_argument("--out", default=None, help="output path; .pdf for paper (default png under figures/).")
    ap.add_argument("--width", type=float, default=3.4, help="single-panel width (in).")
    ap.add_argument("--panel_width", type=float, default=2.8, help="per-panel width when >1 noise.")
    ap.add_argument("--height", type=float, default=2.5)
    ap.add_argument("--no_band", action="store_true", help="hide the +/- std shaded band.")
    ap.add_argument("--y0", action="store_true", help="force y-axis to start at 0.")
    ap.add_argument("--sigma_title", action="store_true",
                    help="show the sigma_b title even for a single panel (default off -> put "
                         "sigma_b in the LaTeX caption). Multi-panel always titles each panel.")
    ap.add_argument("--legend_loc", default="above",
                    help="'above' (horizontal strip above the axes -> never overlaps), 'none', "
                         "or any matplotlib loc (e.g. 'upper right', 'center right').")
    ap.add_argument("--title", default=None, help="optional figure suptitle (off by default).")
    ap.add_argument("--series", default=None,
                    help="comma list of curve keys to draw (default all): ours, oneshot_kw, "
                         "oneshot_matched. e.g. --series ours,oneshot_matched")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    apply_paper_style()
    want = set(s.strip() for s in args.series.split(",")) if args.series else None
    series = [t for t in SERIES if want is None or t[0] in want]
    config, curves = load_result(args.exp_dir)
    noises = args.noises or config.get("noises") or []
    noises = [n for n in noises if get_series(curves, n, "ours")]
    if not noises:
        raise SystemExit("No matching curves in %s/result.json" % args.exp_dir)

    npanel = len(noises)
    figw = args.width if npanel == 1 else args.panel_width * npanel
    figh = args.height + (0.35 if args.legend_loc == "above" else 0.0)   # room for the top strip
    fig, axes = plt.subplots(1, npanel, figsize=(figw, figh),
                             squeeze=False, sharey=True, constrained_layout=True)
    axes = axes[0]

    for pi, noise in enumerate(noises):
        ax = axes[pi]
        for policy, label, color, ls, marker in series:
            s = get_series(curves, noise, policy)
            if not s:
                continue
            t, mean = s["t"], s["cd_mean"]
            std = s.get("cd_std")
            ax.plot(t, mean, ls, color=color, marker=marker, markersize=3.2,
                    markeredgewidth=0, label=label)
            if std and not args.no_band:
                lo = [m - e for m, e in zip(mean, std)]
                hi = [m + e for m, e in zip(mean, std)]
                ax.fill_between(t, lo, hi, color=color, alpha=0.12, linewidth=0)
        ax.invert_xaxis()                       # t_start (left) -> 1 (right) = denoising progress
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_xlabel(r"reverse step $t$  (denoising $\rightarrow$)")
        if npanel > 1 or args.sigma_title:      # single panel -> sigma_b goes in the caption
            ax.set_title(r"$\sigma_b=%g$" % noise)
        if args.y0:
            ax.set_ylim(bottom=0)
        if pi == 0:
            ax.set_ylabel(r"CD-to-GT of $\mathrm{decode}(w_t)$  ($\downarrow$)")

    if args.legend_loc == "above":              # horizontal strip above the axes -> no overlap
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=len(handles),
                   frameon=False, columnspacing=1.6, handlelength=2.2)
    elif args.legend_loc != "none":
        axes[0].legend(loc=args.legend_loc)
    if args.title:
        fig.suptitle(args.title)

    n_shapes = config.get("num_shapes", "?")
    out = args.out or os.path.join(args.exp_dir, "figures", "exp10_mechanism_paper.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote %s  (noises=%s, n=%s shapes)" % (out, noises, n_shapes))


if __name__ == "__main__":
    main()
