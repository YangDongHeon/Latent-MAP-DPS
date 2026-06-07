"""Experiment 8: OURS (Latent-MAP Guided SDEdit) vs baselines -- per-policy runner.

Observation-anchored reverse sampling (SDEdit init at t_start, x_init=observation),
multi-shape, noise {0.1,0.2,0.3}. Six policies (select a subset with --policies;
default 'all'); paper-figure column names in [brackets]:

    enc_no_dps  : w=E_phi(b) frozen,     reverse chain, no DPS          [Encoder]
    enc_dps     : w=E_phi(b) frozen,     reverse chain + DPS-on-x       [Encoder+DPS]
    oneshot       : w_obs -> solve J_t' ONCE at t' (invert_iters), freeze [One-shot]
    oneshot_dps    : oneshot + DPS (z FIXED for the whole chain)             [One-shot+DPS]
    ours_legacy : w_obs -> re-solve every step (ORIGIN prior), + DPS    [Ours (prior)]
    ours        : w_obs -> re-solve J_t every step (anchor), + DPS      [Ours]

ALL policies start from the SAME encoder latent w_obs=E_phi(b) and use the IDENTICAL
latent-MAP objective (no lambda/beta knobs)

    J_t(w) = softNLL(b; x0hat(X_t,t,F(w)))  +  0.5 * ||w - mu_t||^2
    mu_t   = sg[ F^{-1}( E_phi( x0hat(X_t,t,F(w_warm)) ) ) ]            (encoder anchor)

They differ ONLY in HOW J is used (the paper's Encoder / One-shot / Ours split):
    One-shot: solve J_t' ONCE at the SDEdit init (X_t', t') for invert_iters steps, freeze.
    OURS     : re-solve J_t at EVERY reverse step (Kw warm-started steps), w tracks X_t.
OURS does NOT get the one-shot for free: it starts from the raw encoder latent and only
spends Kw*T per-step steps total. The comparison isolates PER-STEP REFINEMENT (tracking
the evolving X_t) vs a single up-front solve.

The quadratic term is the Gaussian/Laplace surrogate for the dropped factor
    log p_t(X_t|w) + log p(w) = log p_t(w|X_t) + C(X_t),  p_t(w|X_t) ~ N(mu_t, I),
which REPLACES the bare ||w||^2 prior so w is anchored to the current denoised
geometry's latent instead of collapsing to the origin. beta = 1 (knob removed):
the noise-adaptive data/prior balance already lives INSIDE softNLL via sigma_b.
ours_legacy swaps mu_t -> 0 (same weight), an ablation of the anchor CENTER.

START: every policy is seeded from w_obs. Encoder freezes it; One-shot solves once at
t'; Ours re-solves per step. (One-shot is only computed when oneshot/oneshot_dps is selected.)

Hierarchical output (so plot_main_grid.py can render a noise x policy grid for ANY
subset of the policies that were run):

    EXP_DIR/
      result.json                                  overall config + summary table
      table.csv                                    flat metrics
      reference/arrays/<noise_tag>/shape<idx>.npz  {gt, observation}
      policies/<key>/metrics.json                  per-policy aggregates + per-shape
      policies/<key>/arrays/<noise_tag>/shape<idx>.npz  {points, cd, snll}
      figures/exp8_summary.png                     CD/sNLL vs noise (selected policies)
      figures/exp8_shape<idx>.png                  per-shape diagnostic (optional)

Metric: decode CD-GT (sigma-free, cross-noise comparable) PRIMARY; sNLL-GT
(sigma=noise, within-noise only) secondary. Mean +- std over shapes.

Run from project/diffusion-point-cloud (inside docker).
"""
import argparse
import csv
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
    encode_observation_z, z_to_w, flow_to_z, run_enc_chain,
    dps_step, predict_eps_and_x0, soft_chamfer_nll, get_sigma_b,
    np_chamfer_to_gt, np_soft_nll_to_gt, parse_csv_floats,
    scatter_xz, xz_limits,
)
from policies import (
    POLICY_REGISTRY, ALL_POLICIES, parse_policy_list, policy_label,
    policy_color, needs_oneshot, noise_tag,
)
from utils.dataset import ShapeNetCore
from utils.misc import seed_all

INVERSION_LOSS = "soft_nll"


def set_runtime_defaults(args):
    args.dps_schedule = "ratio"
    args.eta_min = None
    args.eta_max = None
    args.normalize_dps_grad = False
    args.dps_grad_eps = 1e-8
    args.dps_step_size = 1e-6           # unused by ratio schedule
    args.emd_epsilon = 0.03
    args.emd_iters = 50
    args.emd_max_points = 0
    args.trace_every = 100
    args.plot_point_size = 1.2
    args.plot_alpha = 0.9
    args.plot_color = "#1f77b4"


def np_(t):
    return t.squeeze(0).detach().cpu().numpy()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# Latent-MAP objective J_t(w) -- shared by One-shot (solve once at t') and OURS
# (re-solve every reverse step). The objective is IDENTICAL; only the solve
# frequency differs. No lambda/beta knobs:
#   J_t(w) = softNLL(b; x0hat(X_t,t,F(w))) + 0.5 * ANCHOR_WEIGHT * ||w - mu_t||^2
#   mu_t   = sg[ F^{-1}( E_phi( x0hat(X_t,t,F(w_init)) ) ) ]
# The noise-adaptive data/prior balance lives INSIDE softNLL via sigma_b, so the
# anchor precision beta is fixed to 1 (and there is no separate softNLL weight).
# --------------------------------------------------------------------------- #
ANCHOR_WEIGHT = 1.0   # beta_t in the appendix; fixed to 1 (knob removed).


def solve_latent_map(model, x_t, b, t, w_init, n_steps, args, use_anchor=True):
    """argmin_w J_t(w) by warm-started Adam from w_init; mu_t fixed (stop-grad).

    use_anchor=True : reg = 0.5 * ANCHOR_WEIGHT * ||w - mu_t||^2  (encoder anchor)
    use_anchor=False: reg = 0.5 * ANCHOR_WEIGHT * ||w||^2         (origin-prior ablation)
    """
    sigma_b = get_sigma_b(args)
    mu = None
    if use_anchor:
        with torch.no_grad():
            z0 = flow_to_z(model, w_init)
            _, x0_warm = predict_eps_and_x0(model, x_t, t, z0)
            z_anchor = encode_observation_z(model, x0_warm, sample_encoder=args.encoder_sample)
            mu = z_to_w(model, z_anchor).detach()
    w = w_init.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=args.invert_lr)
    for _ in range(max(1, n_steps)):
        z = flow_to_z(model, w)
        _, x0_hat = predict_eps_and_x0(model, x_t, t, z)
        data = soft_chamfer_nll(x0_hat, b.detach(), sigma_b)
        center = (w - mu) if use_anchor else w
        reg = 0.5 * ANCHOR_WEIGHT * center.pow(2).sum(dim=1).mean()
        opt.zero_grad()
        (data + reg).backward()
        opt.step()
    return w.detach()


def invert_latent_anchor(model, b, x_start, start_t, w_enc, args):
    """One-shot baseline: solve the SAME J_t' ONCE at the SDEdit init (X_t', t')."""
    return solve_latent_map(model, x_start, b, start_t, w_enc, args.invert_iters, args, use_anchor=True)


def run_ours_chain(model, b, x_start, start_t, w_init, args, refine):
    """Algorithm 1: re-solve J_t every reverse step (warm-started), then a DPS step.

    refine='anchor': J_t uses the encoder-consistency anchor 0.5*||w - mu_t||^2.
    refine='legacy': J_t uses the origin prior 0.5*||w||^2 (ablates the anchor center).
    """
    use_anchor = (refine == "anchor")
    x_t = x_start.detach()
    w = w_init.detach().clone()
    for t in range(start_t, 0, -1):
        w = solve_latent_map(model, x_t, b, t, w, args.ours_inner_steps, args, use_anchor=use_anchor)
        z = flow_to_z(model, w).detach()
        x_t = dps_step(model, x_t, b, z, t, args)[0].detach()   # (x_next, ...)
    return x_t.detach()


def run_policy_result(key, model, b, x_start, start_t, ctx, args):
    """Dispatch one policy -> result point cloud tensor [1, N, 3]."""
    if key == "enc_no_dps":
        return run_enc_chain(model, b, x_start, start_t, ctx["z_obs"], False, args)[0]
    if key == "enc_dps":
        return run_enc_chain(model, b, x_start, start_t, ctx["z_obs"], True, args)[0]
    if key == "oneshot":
        return run_enc_chain(model, b, x_start, start_t, ctx["z_inv"], False, args)[0]
    if key == "oneshot_dps":
        return run_enc_chain(model, b, x_start, start_t, ctx["z_inv"], True, args)[0]
    if key == "ours_legacy":
        return run_ours_chain(model, b, x_start, start_t, ctx["w_obs"], args, "legacy")
    if key == "ours":
        return run_ours_chain(model, b, x_start, start_t, ctx["w_obs"], args, "anchor")
    raise ValueError("Unknown policy %r" % key)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_shape_figure(idx, per_noise, noises, policies, ratio_of, fig_dir, args):
    """Per-shape diagnostic: rows = gt/observation/selected policies, cols = noises."""
    rows = ["gt", "observation"] + list(policies)
    all_arrays = [per_noise[n]["arrays"][r] for n in noises for r in rows]
    xlim, zlim = xz_limits(all_arrays)
    ncol, nrow = len(noises), len(rows)
    fig, axes = plt.subplots(nrow, ncol, figsize=(max(2.7 * ncol, 5.4), max(1.9 * nrow, 7)),
                             squeeze=False, constrained_layout=True)
    for col, noise in enumerate(noises):
        arrs = per_noise[noise]["arrays"]
        labels = per_noise[noise]["labels"]
        for ri, row in enumerate(rows):
            ax = axes[ri][col]
            scatter_xz(ax, arrs[row], xlim, zlim, args)
            ax.text(0.5, -0.04, labels.get(row, ""), transform=ax.transAxes,
                    ha="center", va="top", fontsize=6.5)
            if ri == 0:
                ax.set_title("noise %g (r%g)" % (noise, ratio_of[noise]), fontsize=10)
            if col == 0:
                disp = policy_label(row) if row in POLICY_REGISTRY else row
                ax.text(-0.08, 0.5, disp, transform=ax.transAxes, ha="right", va="center",
                        rotation=90, fontsize=8, fontweight="bold")
    fig.suptitle("exp8 shape %d  (dps=%s, Kw=%d, J=softNLL+0.5||w-mu||^2, invert_iters=%d)"
                 % (idx, args.dps_loss, args.ours_inner_steps, args.invert_iters), fontsize=11)
    out = os.path.join(fig_dir, "exp8_shape%d.png" % idx)
    fig.savefig(out, dpi=args.plot_dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def make_summary_figure(agg, noises, policies, fig_dir, args, n_shapes):
    """CD-GT / sNLL-GT vs noise, one line per selected policy."""
    def mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    def std(xs):
        return float(np.std(xs)) if xs else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for ax, (akey, title, ylabel, use_err) in zip(
            axes, [("cd", "decode CD-GT (down)", "CD-GT", True),
                   ("sn", "decode sNLL-GT (down, within-noise)", "sNLL-GT", False)]):
        for policy in policies:
            xs, ys, es = [], [], []
            for noise in noises:
                a = agg[(policy, noise)]
                xs.append(noise); ys.append(mean(a[akey])); es.append(std(a[akey]))
            color = policy_color(policy)
            if use_err:
                ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, linewidth=1.7,
                            label=policy_label(policy), color=color)
            else:
                ax.plot(xs, ys, marker="o", linewidth=1.7, label=policy_label(policy), color=color)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("observation noise std"); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25); ax.legend(fontsize=8)
    fig.suptitle("exp8: ours vs baselines (obs-anchored, dps=%s, %d shapes)" % (args.dps_loss, n_shapes),
                 fontsize=12)
    out = os.path.join(fig_dir, "exp8_summary.png")
    fig.savefig(out, dpi=args.plot_dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
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

    parser.add_argument("--policies", default="all",
                        help="comma list of policies to run (or 'all'). Choices: %s" % ALL_POLICIES)
    parser.add_argument("--noises", type=parse_csv_floats, default=[0.1, 0.2, 0.3])
    parser.add_argument("--ratios", type=parse_csv_floats, default=[0.5, 0.3, 0.3],
                        help="one DPS X-update ratio per noise (or a single shared value).")
    parser.add_argument("--dps_loss", default="soft_nll", choices=["mse_nll", "cd", "emd", "soft_nll"],
                        help="DPS guidance loss (same for enc_dps / winit_dps / ours).")
    parser.add_argument("--num_shapes", type=int, default=10)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--noise_std", type=float, default=0.2)   # overwritten per noise
    parser.add_argument("--sigma_b", type=float, default=None)
    parser.add_argument("--encoder_sample", action="store_true")
    parser.add_argument("--x_init", default="observation", choices=["observation", "random"])
    parser.add_argument("--t_start", type=int, default=30)
    parser.add_argument("--reverse_noise_scale", type=float, default=0.0)
    parser.add_argument("--ours_inner_steps", type=int, default=15,
                        help="Kw: per-step warm-started Adam steps for OURS (solve J_t). "
                             "One-shot uses --invert_iters for its single-t' solve.")
    add_latent_init_args(parser)

    parser.add_argument("--output_dir", default="output/main_results")
    parser.add_argument("--no_save_arrays", dest="save_arrays", action="store_false", default=True,
                        help="skip writing per-shape .npz arrays (then plot_main_grid.py can't run).")
    parser.add_argument("--no_shape_figs", dest="save_shape_figs", action="store_false", default=True)
    parser.add_argument("--no_summary_fig", dest="save_summary_fig", action="store_false", default=True)
    parser.add_argument("--plot_dpi", type=int, default=180)
    args = parser.parse_args()

    policies = parse_policy_list(args.policies)
    if len(args.ratios) == 1:
        args.ratios = args.ratios * len(args.noises)
    if len(args.ratios) != len(args.noises):
        raise ValueError("--ratios must have one value per --noises (or a single value).")

    set_runtime_defaults(args)
    # One-shot/OURS always use the softNLL data term + unit-precision anchor (no knobs).
    args.x_obs_loss = args.dps_loss
    seed_all(args.seed)
    model, ckpt = load_model(args)
    scale_mode = args.scale_mode or ckpt["args"].scale_mode
    dataset = ShapeNetCore(path=args.dataset_path, cates=[args.category], split=args.split, scale_mode=scale_mode)
    indices = list(range(args.start, min(args.start + args.num_shapes, len(dataset))))

    exp_dir = ensure_dir(args.output_dir)
    fig_dir = ensure_dir(os.path.join(exp_dir, "figures"))
    pol_dir = ensure_dir(os.path.join(exp_dir, "policies"))
    ref_dir = ensure_dir(os.path.join(exp_dir, "reference", "arrays"))
    oneshot_needed = needs_oneshot(policies)

    print("policies: %s" % ", ".join(policies))
    print("one-shot solve needed: %s" % oneshot_needed)

    ratio_of = dict(zip(args.noises, args.ratios))
    agg = OrderedDict()           # (policy, noise) -> dict(cd=[], sn=[], shapes=[])
    shape_store = OrderedDict()   # idx -> { noise -> {"arrays":{row:arr}, "labels":{row:str}} }

    def add(policy, noise, idx, cd, sn):
        a = agg.setdefault((policy, noise), OrderedDict(cd=[], sn=[], shapes=[]))
        a["cd"].append(cd); a["sn"].append(sn); a["shapes"].append(int(idx))

    def save_npz(path_dir, fname, **arrays):
        ensure_dir(path_dir)
        np.savez_compressed(os.path.join(path_dir, fname), **arrays)

    for noise in args.noises:
        args.target_update_ratio = ratio_of[noise]
        ntag = noise_tag(noise)
        for shape_pos, idx in enumerate(indices):
            seed_all(args.seed + 1000 * shape_pos + int(round(noise * 100)))
            data = dataset[idx]
            x0 = data["pointcloud"].unsqueeze(0).to(args.device)
            gt = np_(x0)
            args.noise_std = noise
            b, _ = make_observation(x0, args)
            obs = np_(b)
            num_points = b.size(1)
            x_start, start_t, _ = make_x_start(model, b, args)   # shared SDEdit init

            ctx = OrderedDict()
            ctx["z_obs"] = encode_observation_z(model, b, sample_encoder=args.encoder_sample)
            ctx["w_obs"] = z_to_w(model, ctx["z_obs"]).detach()
            if oneshot_needed:
                # One-shot = solve the SAME J_t' ONCE at the SDEdit init (X_t', t'),
                # warm-started from the encoder latent. Same objective as OURS.
                ctx["oneshot"] = invert_latent_anchor(model, b, x_start, start_t, ctx["w_obs"], args)
                ctx["z_inv"] = flow_to_z(model, ctx["oneshot"]).detach()

            if args.save_arrays:
                save_npz(os.path.join(ref_dir, ntag), "shape%d.npz" % idx,
                         gt=gt.astype(np.float32), observation=obs.astype(np.float32),
                         noise=np.float32(noise), shape_idx=np.int64(idx))

            arrays = OrderedDict(gt=gt, observation=obs)
            labels = OrderedDict(gt="", observation="")
            for policy in policies:
                xn = np_(run_policy_result(policy, model, b, x_start, start_t, ctx, args))
                cd = np_chamfer_to_gt(xn, gt)
                sn = np_soft_nll_to_gt(xn, gt, noise)
                add(policy, noise, idx, cd, sn)
                arrays[policy] = xn
                labels[policy] = "CD %.4f\nsNLL %.3f" % (cd, sn)
                if args.save_arrays:
                    save_npz(os.path.join(pol_dir, policy, "arrays", ntag), "shape%d.npz" % idx,
                             points=xn.astype(np.float32), cd=np.float32(cd), snll=np.float32(sn),
                             noise=np.float32(noise), shape_idx=np.int64(idx))
            if args.save_shape_figs:   # only the per-shape figure needs the arrays in RAM
                shape_store.setdefault(idx, OrderedDict())[noise] = OrderedDict(arrays=arrays, labels=labels)
            print("noise=%g shape=%d done" % (noise, idx))

    # ---- aggregates ----
    def mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    def std(xs):
        return float(np.std(xs)) if xs else float("nan")

    rows = []
    print("\n%-12s %-5s | %9s %8s | %9s   (n=%d, dps=%s)" %
          ("policy", "noise", "CD-GT", "cd_std", "sNLL-GT", len(indices), args.dps_loss))
    for policy in policies:
        for noise in args.noises:
            a = agg[(policy, noise)]
            row = OrderedDict(policy=policy, label=policy_label(policy), noise=noise,
                              ratio=ratio_of[noise], cd_to_gt_mean=mean(a["cd"]),
                              cd_to_gt_std=std(a["cd"]), snll_to_gt_mean=mean(a["sn"]),
                              snll_to_gt_std=std(a["sn"]))
            rows.append(row)
            print("%-12s %-5.3g | %9.5f %8.5f | %9.4f" %
                  (policy, noise, row["cd_to_gt_mean"], row["cd_to_gt_std"], row["snll_to_gt_mean"]))

    # per-policy metrics.json
    for policy in policies:
        pmetrics = OrderedDict(policy=policy, label=policy_label(policy), noises=OrderedDict())
        for noise in args.noises:
            a = agg[(policy, noise)]
            pmetrics["noises"]["%g" % noise] = OrderedDict(
                cd_mean=mean(a["cd"]), cd_std=std(a["cd"]),
                snll_mean=mean(a["sn"]), snll_std=std(a["sn"]),
                per_shape=[OrderedDict(idx=s, cd=c, snll=n)
                           for s, c, n in zip(a["shapes"], a["cd"], a["sn"])])
        with open(os.path.join(ensure_dir(os.path.join(pol_dir, policy)), "metrics.json"), "w") as f:
            json.dump(pmetrics, f, indent=2)

    config = OrderedDict(
        policies=policies, policy_labels={p: policy_label(p) for p in policies},
        noises=args.noises, ratios=args.ratios, dps_loss=args.dps_loss,
        num_shapes=len(indices), indices=indices, t_start=args.t_start,
        x_init=args.x_init, objective="softNLL + 0.5*anchor_weight*||w-mu||^2",
        anchor_weight=ANCHOR_WEIGHT, data_term=INVERSION_LOSS, invert_lr=args.invert_lr,
        invert_iters=args.invert_iters, ours_inner_steps=args.ours_inner_steps)

    with open(os.path.join(exp_dir, "table.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    with open(os.path.join(exp_dir, "result.json"), "w") as f:
        json.dump(OrderedDict(config=config, summary=rows), f, indent=2)

    # ---- figures ----
    if args.save_summary_fig:
        out = make_summary_figure(agg, args.noises, policies, fig_dir, args, len(indices))
        print("summary fig -> %s" % out)
    if args.save_shape_figs:
        for idx in indices:
            out = make_shape_figure(idx, shape_store[idx], args.noises, policies, ratio_of, fig_dir, args)
            print("shape fig -> %s" % out)

    print("\nWrote -> %s" % exp_dir)
    print("Grid figure: python experiments/plot_main_grid.py --exp_dir %s --policies %s"
          % (exp_dir, ",".join(policies)))


if __name__ == "__main__":
    main()
