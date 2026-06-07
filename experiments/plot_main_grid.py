"""Plot the exp8 qualitative grid: rows = noise levels, cols = policies.

Reads the per-shape .npz arrays written by run_main_results.py and renders, for ONE
shape, a grid whose rows are the noise levels and whose columns are the SELECTED
policies (optionally preceded by GT / observation reference columns). Only the
policies passed via --policies (default: all that were run) are drawn.

Torch-free (numpy + matplotlib only) -> runnable on the host, no docker needed.

Paper-ready by default: no title, no per-cell numbers, single blue point color,
column headers = method names, row labels = sigma_b. Numbers live in table.csv.

Examples (run from project/diffusion-point-cloud):
    # 7-column paper figure: GT | Input | Encoder | Encoder+DPS | One-shot | One-shot+DPS | Ours
    python experiments/plot_main_grid.py --exp_dir output/main_results \
        --policies enc_no_dps,enc_dps,oneshot,oneshot_dps,ours --shape 0 --out figures/exp8_grid.pdf
    python experiments/plot_main_grid.py --exp_dir output/main_results --all_shapes
    # chair: view along x (y-z side view, upright); add --captions / --title for diagnostics
    python experiments/plot_main_grid.py --exp_dir output/main_results_chair \
        --policies enc_no_dps,enc_dps,oneshot,oneshot_dps,ours --view x --all_shapes
"""
import argparse
import glob
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

ANALYZE_DIR = os.path.dirname(os.path.abspath(__file__))
if ANALYZE_DIR not in sys.path:
    sys.path.insert(0, ANALYZE_DIR)

from policies import POLICY_REGISTRY, parse_policy_list, policy_label, noise_tag
from paper_style import apply as apply_paper_style, PALETTE


def parse_csv_floats(value):
    if value is None or value == "":
        return None
    return [float(s.strip()) for s in str(value).split(",") if s.strip()]


def finite_points(arr):
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.empty((0, 3))
    return arr[np.isfinite(arr).all(axis=1)]


VIEW_AXES = {  # viewing axis -> (horizontal idx, vertical idx); the up axis is y (1)
    "x": (2, 1),  # look ALONG x -> y-z plane (side view; vertical = up = y). Good for chairs.
    "y": (0, 2),  # look along y -> x-z plane (top-down; default, matches the airplane figs)
    "z": (0, 1),  # look along z -> x-y plane (front view; vertical = up = y)
}


def proj_limits(arrays, h, v):
    """Shared square limits in the (h, v) projection across every array shown."""
    pts = []
    for arr in arrays:
        f = finite_points(arr)
        if len(f) > 0:
            pts.append(f[:, [h, v]])
    if not pts:
        return (-1.0, 1.0), (-1.0, 1.0)
    stacked = np.concatenate(pts, axis=0)
    mins, maxs = stacked.min(axis=0), stacked.max(axis=0)
    center = (mins + maxs) / 2
    span = float((maxs - mins).max())
    if not np.isfinite(span) or span <= 0:
        span = 2.0
    half = span / 2 + max(span * 0.08, 1e-6)
    return (center[0] - half, center[0] + half), (center[1] - half, center[1] + half)


def scatter_proj(ax, arr, xlim, ylim, h, v, size, alpha, color):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    f = finite_points(arr)
    if len(f) == 0:
        ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="crimson")
        return
    ax.scatter(f[:, h], f[:, v], s=size, c=color, alpha=alpha, linewidths=0, rasterized=True)


def ref_npz(exp_dir, noise, idx):
    return os.path.join(exp_dir, "reference", "arrays", noise_tag(noise), "shape%d.npz" % idx)


def policy_npz(exp_dir, policy, noise, idx):
    return os.path.join(exp_dir, "policies", policy, "arrays", noise_tag(noise), "shape%d.npz" % idx)


def load_npz(path):
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def discover_shapes(exp_dir, noise):
    """Shape indices available for a given noise (from the reference arrays)."""
    pat = os.path.join(exp_dir, "reference", "arrays", noise_tag(noise), "shape*.npz")
    idxs = []
    for p in glob.glob(pat):
        m = re.search(r"shape(\d+)\.npz$", p)
        if m:
            idxs.append(int(m.group(1)))
    return sorted(idxs)


def build_columns(show_gt, show_obs, policies):
    cols = []
    if show_gt:
        cols.append(("gt", "GT"))
    if show_obs:
        cols.append(("obs", "Observation"))
    for p in policies:
        cols.append(("policy", p))
    return cols


def cell_array_and_caption(exp_dir, kind, key, noise, idx):
    """Return (array, caption) for one grid cell."""
    if kind in ("gt", "obs"):
        d = load_npz(ref_npz(exp_dir, noise, idx))
        if d is None:
            return None, ""
        return (d["gt"] if kind == "gt" else d["observation"]), ""
    d = load_npz(policy_npz(exp_dir, key, noise, idx))
    if d is None:
        return None, ""
    cap = "CD %.4f\nsNLL %.3f" % (float(d["cd"]), float(d["snll"]))
    return d["points"], cap


def column_title(kind, key):
    if kind == "gt":
        return "GT"
    if kind == "obs":
        return "Input"
    return policy_label(key)


def render_grid(exp_dir, idx, noises, columns, args):
    nrow, ncol = len(noises), len(columns)
    h, v = VIEW_AXES[args.view]
    # gather everything once for shared limits + cache arrays
    cache = {}
    all_arrays = []
    for noise in noises:
        for ci, (kind, key) in enumerate(columns):
            arr, cap = cell_array_and_caption(exp_dir, kind, key, noise, idx)
            cache[(noise, ci)] = (arr, cap)
            if arr is not None:
                all_arrays.append(arr)
    xlim, ylim = proj_limits(all_arrays, h, v)

    fig, axes = plt.subplots(nrow, ncol, figsize=(args.cell * ncol, args.cell * nrow),
                             squeeze=False, constrained_layout=True)
    for ri, noise in enumerate(noises):
        for ci, (kind, key) in enumerate(columns):
            ax = axes[ri][ci]
            arr, cap = cache[(noise, ci)]
            scatter_proj(ax, arr, xlim, ylim, h, v, args.point_size, args.alpha, args.color)
            if args.captions and cap:
                ax.text(0.5, -0.02, cap, transform=ax.transAxes, ha="center", va="top", fontsize=7)
            if ri == 0:
                ax.set_title(column_title(kind, key), fontsize=args.header_size)
            if ci == 0:
                ax.text(-0.06, 0.5, r"$\sigma_b=%g$" % noise, transform=ax.transAxes,
                        ha="right", va="center", rotation=90, fontsize=args.header_size)
    if args.title:
        fig.suptitle(args.title, fontsize=args.header_size + 1)
    ext = os.path.splitext(args.out)[1] if args.out else ".png"
    out = args.out or os.path.join(exp_dir, "figures", "exp8_grid_shape%d%s" % (idx, ext))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", required=True, help="run_main_results.py output dir.")
    parser.add_argument("--policies", default=None,
                        help="comma list of policies to show (default: all run). order respected.")
    parser.add_argument("--noises", type=parse_csv_floats, default=None,
                        help="noise rows (default: from result.json).")
    parser.add_argument("--shape", type=int, default=None, help="shape index (default: first available).")
    parser.add_argument("--all_shapes", action="store_true", help="emit one grid per available shape.")
    parser.add_argument("--no_gt", dest="show_gt", action="store_false", default=True)
    parser.add_argument("--no_obs", dest="show_obs", action="store_false", default=True)
    parser.add_argument("--out", default=None,
                        help="output path; extension picks the format (.pdf for paper). "
                             "Ignored with --all_shapes.")
    parser.add_argument("--captions", action="store_true",
                        help="overlay per-cell CD/sNLL (off by default -> clean paper figure).")
    parser.add_argument("--title", default=None, help="optional figure title (off by default).")
    parser.add_argument("--view", default="y", choices=["x", "y", "z"],
                        help="viewing axis: x -> y-z side view (chairs), y -> x-z top-down "
                             "(default, airplanes), z -> x-y front view.")
    parser.add_argument("--color", default=PALETTE["points"], help="single point color for ALL cells.")
    parser.add_argument("--cell", type=float, default=1.9, help="per-cell size in inches.")
    parser.add_argument("--header_size", type=float, default=12.0)
    parser.add_argument("--point_size", type=float, default=1.4)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    apply_paper_style()   # STIX serif headers/labels, refined defaults
    cfg_path = os.path.join(args.exp_dir, "result.json")
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f).get("config", {})

    # policies: requested subset (validated against the registry) intersected with on-disk
    run_policies = cfg.get("policies") or list(POLICY_REGISTRY.keys())
    if args.policies:
        want = parse_policy_list(args.policies)
    else:
        want = run_policies
    policies = [p for p in want if os.path.isdir(os.path.join(args.exp_dir, "policies", p))]
    missing = [p for p in want if p not in policies]
    if missing:
        print("WARNING: not on disk (skipped): %s" % missing)
    if not policies:
        raise SystemExit("No requested policies found under %s/policies" % args.exp_dir)

    noises = args.noises or cfg.get("noises") or [0.1, 0.2, 0.3]
    columns = build_columns(args.show_gt, args.show_obs, policies)

    # which shapes?
    avail = discover_shapes(args.exp_dir, noises[0])
    if not avail:
        raise SystemExit("No reference arrays under %s (run with --save_arrays)." % args.exp_dir)
    if args.all_shapes:
        shapes = avail
        if args.out:
            print("WARNING: --out ignored with --all_shapes (per-shape default paths used).")
            args.out = None
    elif args.shape is not None:
        shapes = [args.shape]
    else:
        shapes = [avail[0]]

    print("policies: %s" % ", ".join(policies))
    print("noises:   %s" % noises)
    print("shapes:   %s" % shapes)
    for idx in shapes:
        out = render_grid(args.exp_dir, idx, noises, columns, args)
        print("grid -> %s" % out)


if __name__ == "__main__":
    main()
