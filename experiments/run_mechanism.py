"""Experiment 10: mechanism plot -- why per-step latent refinement helps.

Tracks the reconstruction quality of the CURRENT latent along the reverse chain:
at selected reverse steps t we decode the latent and measure CD-to-GT of decode(w).

All three start from the SAME encoder latent w_obs=E_phi(b) (the paper's method):
    ours            : re-solve J_t every reverse step (Kw warm-started steps), w tracks X_t.
    oneshot_kw      : ONE-SHOT with Kw steps @ t' -- exactly ours' first-step solve, frozen.
    oneshot_matched : ONE-SHOT with Kw*#reverse_steps (=ours' TOTAL budget) @ t', frozen.

oneshot_kw is the identical latent to ours at step 1, so the two COINCIDE at the first step
and then diverge -> the gain is purely per-step refinement AFTER the first solve.
oneshot_matched spends ours' FULL budget at the fixed t' -> shows per-step beats one-shot
even at matched optimization budget (it is the per-step tracking of X_t, not more compute).
decode via the full generative sampler decode_latent(F(w)).

Outputs: output/mechanism/{result.json, figures/}
    figures/exp10_latent_vs_step_noise{n}.png  (decode(w) CD-GT vs reverse step t)

Run from project/diffusion-point-cloud (inside docker).
"""
import argparse
import json
import os
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch

ANALYZE_DIR = os.path.dirname(os.path.abspath(__file__))
if ANALYZE_DIR not in sys.path:
    sys.path.insert(0, ANALYZE_DIR)

from core import (
    add_latent_init_args, load_model, make_observation, make_x_start,
    encode_observation_z, z_to_w, flow_to_z, decode_latent,
    dps_step, predict_eps_and_x0, soft_chamfer_nll, get_sigma_b,
    np_chamfer_to_gt, parse_csv_floats,
)
# Canonical FINAL-policy objective (single source of truth, shared with exp8):
from run_main_results import solve_latent_map, ANCHOR_WEIGHT
from utils.dataset import ShapeNetCore
from utils.misc import seed_all


def set_runtime_defaults(args):
    args.dps_schedule = "ratio"
    args.eta_min = None
    args.eta_max = None
    args.normalize_dps_grad = False
    args.dps_grad_eps = 1e-8
    args.dps_step_size = 1e-6
    args.emd_epsilon = 0.03
    args.emd_iters = 50
    args.emd_max_points = 0
    args.trace_every = 100
    args.x_obs_loss = args.dps_loss


def run_ours_capture(model, b, x_start, start_t, w_init, args):
    """Run the FINAL OURS policy (per-step anchor solve of J_t) and capture the latent
    AFTER each step's solve -- the latent that drives the reverse transition at step t.
    w_at[start_t] = the Kw-step solve from w_obs at t', identical to the one-shot@Kw baseline,
    so the two COINCIDE at the first step; later steps show the per-step refinement as X_t
    denoises. Returns {t: w_t}."""
    x_t = x_start.detach()
    w = w_init.detach().clone()
    w_at = OrderedDict()
    for t in range(start_t, 0, -1):
        w = solve_latent_map(model, x_t, b, t, w, args.ours_inner_steps, args, use_anchor=True)
        w_at[t] = w.detach().clone()                      # latent AFTER the step-t solve
        z = flow_to_z(model, w).detach()
        x_t = dps_step(model, x_t, b, z, t, args)[0].detach()
    return w_at


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="pretrained/GEN_airplane.pt")
    parser.add_argument("--dataset_path", default="data/shapenet.hdf5")
    parser.add_argument("--category", default="airplane")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--scale_mode", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--flexibility", type=float, default=None)

    parser.add_argument("--noises", type=parse_csv_floats, default=[0.2, 0.3])
    parser.add_argument("--ratios", type=parse_csv_floats, default=[0.3, 0.3])
    parser.add_argument("--dps_loss", default="soft_nll", choices=["mse_nll", "cd", "emd", "soft_nll"])
    parser.add_argument("--num_shapes", type=int, default=10)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--noise_std", type=float, default=0.2)
    parser.add_argument("--sigma_b", type=float, default=None)
    parser.add_argument("--encoder_sample", action="store_true")
    parser.add_argument("--x_init", default="observation", choices=["observation", "random"])
    parser.add_argument("--t_start", type=int, default=30)
    parser.add_argument("--reverse_noise_scale", type=float, default=0.0)
    parser.add_argument("--ours_inner_steps", type=int, default=15,
                        help="Kw: per-step anchor-solve steps for OURS (matches exp8).")
    parser.add_argument("--inv450_extra", type=int, default=None,
                        help="steps for the budget-matched one-shot (all at t' from w_obs); "
                             "default ours_inner_steps * t_start = ours' TOTAL per-step budget.")
    parser.add_argument("--log_stride", type=int, default=3, help="decode every this many reverse steps.")
    add_latent_init_args(parser)

    parser.add_argument("--output_dir", default="output/mechanism")
    parser.add_argument("--plot_dpi", type=int, default=180)
    args = parser.parse_args()

    if len(args.ratios) == 1:
        args.ratios = args.ratios * len(args.noises)
    set_runtime_defaults(args)
    seed_all(args.seed)
    model, ckpt = load_model(args)
    scale_mode = args.scale_mode or ckpt["args"].scale_mode
    dataset = ShapeNetCore(path=args.dataset_path, cates=[args.category], split=args.split, scale_mode=scale_mode)
    indices = list(range(args.start, min(args.start + args.num_shapes, len(dataset))))
    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    start_t = args.t_start
    log_ts = sorted({t for t in range(start_t, 0, -max(1, args.log_stride))} | {start_t, 1}, reverse=True)
    ratio_of = dict(zip(args.noises, args.ratios))

    # store[(noise, policy)] = { t: [cd per shape] }
    store = OrderedDict()

    def add(noise, policy, t, cd):
        store.setdefault((noise, policy), OrderedDict()).setdefault(t, []).append(cd)

    for noise in args.noises:
        args.target_update_ratio = ratio_of[noise]
        for shape_pos, idx in enumerate(indices):
            seed_all(args.seed + 1000 * shape_pos + int(round(noise * 100)))
            x0 = dataset[idx]["pointcloud"].unsqueeze(0).to(args.device)
            gt = x0.squeeze(0).detach().cpu().numpy()
            args.noise_std = noise
            b, _ = make_observation(x0, args)
            num_points = b.size(1)
            x_start, st, _ = make_x_start(model, b, args)

            z_obs = encode_observation_z(model, b, sample_encoder=args.encoder_sample)
            w_obs = z_to_w(model, z_obs).detach()
            # Shared init: single-t' anchor one-shot (identical to exp8). Both OURS and
            # the one-shot One-shot+DPS baseline start here -> coincide at the first step.
            # OURS: from the raw encoder latent, re-solve J at every reverse step.
            w_at = run_ours_capture(model, b, x_start, st, w_obs, args)
            # One-shot @ Kw: exactly OURS' first-step solve (Kw steps @ t' from w_obs), frozen
            # -> identical latent to ours at step 1, so the two COINCIDE at the first step.
            w_oneshot_kw = solve_latent_map(model, x_start, b, st, w_obs, args.ours_inner_steps, args, use_anchor=True)
            # One-shot @ matched budget: OURS' TOTAL per-step budget (Kw * #reverse_steps)
            # spent all at the fixed (x_start, t') from w_obs, frozen. Same start + same total
            # budget as ours -> the ONLY variable is one-shot-at-t' vs per-step (tracking X_t).
            extra = args.inv450_extra if args.inv450_extra else args.ours_inner_steps * st
            w_oneshot_matched = solve_latent_map(model, x_start, b, st, w_obs, extra, args, use_anchor=True)

            for t in log_ts:
                z_ours = flow_to_z(model, w_at[t]).detach()
                xn = decode_latent(model, z_ours, num_points, args).squeeze(0).detach().cpu().numpy()
                add(noise, "ours", t, np_chamfer_to_gt(xn, gt))
                z_kw = flow_to_z(model, w_oneshot_kw).detach()
                xnk = decode_latent(model, z_kw, num_points, args).squeeze(0).detach().cpu().numpy()
                add(noise, "oneshot_kw", t, np_chamfer_to_gt(xnk, gt))
                z_m = flow_to_z(model, w_oneshot_matched).detach()
                xnm = decode_latent(model, z_m, num_points, args).squeeze(0).detach().cpu().numpy()
                add(noise, "oneshot_matched", t, np_chamfer_to_gt(xnm, gt))
            print("noise=%g shape=%d done" % (noise, idx))

    def stat(noise, policy):
        d = store[(noise, policy)]
        ts = sorted(d.keys(), reverse=True)
        mean = [float(np.mean(d[t])) for t in ts]
        std = [float(np.std(d[t])) for t in ts]
        return ts, mean, std

    n_extra = args.inv450_extra if args.inv450_extra else args.ours_inner_steps * start_t
    result = OrderedDict(config=OrderedDict(
        noises=args.noises, ratios=args.ratios, dps_loss=args.dps_loss, num_shapes=len(indices),
        t_start=start_t, log_stride=args.log_stride, ours_inner_steps=args.ours_inner_steps,
        anchor_weight=ANCHOR_WEIGHT, invert_lr=args.invert_lr,
        oneshot_kw_steps=args.ours_inner_steps, oneshot_matched_steps=n_extra,
        objective="softNLL + 0.5*anchor_weight*||w-mu||^2"),
        curves=OrderedDict())

    for noise in args.noises:
        fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
        for policy, color, ls, label in [("ours", "#2ca02c", "-", "Ours (per-step latent)"),
                                         ("oneshot_kw", "#d62728", "--", "One-shot (Kw=%d, =ours step1)" % args.ours_inner_steps),
                                         ("oneshot_matched", "#9E78B5", ":", "One-shot (matched, %d it)" % n_extra)]:
            ts, mean, std = stat(noise, policy)
            result["curves"]["noise%g_%s" % (noise, policy)] = OrderedDict(t=ts, cd_mean=mean, cd_std=std)
            mean = np.array(mean); std = np.array(std)
            ax.plot(ts, mean, ls, color=color, linewidth=2.0, marker="o", markersize=3.5, label=label)
            ax.fill_between(ts, mean - std, mean + std, color=color, alpha=0.13)
        ax.invert_xaxis()  # chain progresses t_start -> 1, left to right
        ax.set_xlabel("reverse step $t$  (chain progresses $\\rightarrow$)")
        ax.set_ylabel("decode($w_t$) CD-to-GT (↓)")
        ax.set_title("Why per-step refinement helps ($\\sigma_b=%g$, %d shapes)" % (noise, len(indices)), fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)
        out = os.path.join(fig_dir, "exp10_latent_vs_step_noise%s.png" % ("%g" % noise).replace(".", "p"))
        fig.savefig(out, dpi=args.plot_dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("wrote %s" % out)

    with open(os.path.join(args.output_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("\nWrote -> %s" % args.output_dir)


if __name__ == "__main__":
    main()
