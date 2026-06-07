import argparse
import csv
import json
import math
import os
import sys
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.vae_flow import FlowVAE
from utils.dataset import ShapeNetCore
from utils.misc import seed_all


"""
Clean inverse experiment script.

This version intentionally isolates the first question:

    If z is obtained from the pretrained encoder, does DPS correctly update X_t?

Runnable policies:

    enc_no_dps: z = E_phi(b), pure pretrained reverse chain
    enc_dps:    z = E_phi(b), reverse chain plus posterior X_t guidance
    wenc_no_dps:
        w_init = F^{-1}(E_phi(b)), then w is optimized by the posterior loss
        while X follows the pretrained reverse chain.
    wenc_dps:
        same w update as wenc_no_dps, plus posterior X_t guidance.

The equations below are the source of truth for this file.

Observation model for denoising:

    b = A(X_0) + n,    n ~ N(0, sigma_b^2 I),    A = I

Clean estimate from the pretrained epsilon network:

    eps_theta = eps_theta(X_t, t, z)
    X0_hat(X_t, z) = (X_t - sqrt(1 - alpha_bar_t) eps_theta) / sqrt(alpha_bar_t)

Gaussian negative log likelihood used for guidance:

    L_obs(X_t, z)
      = 1 / (2 sigma_b^2) || b - A(X0_hat(X_t, z)) ||_F^2

DPS X update:

    X'_{t-1} = Reverse_theta(X_t, t, z)
    X_{t-1} = X'_{t-1} - eta_t * grad_{X_t} L_obs(X_t, z)

Future w update, with grad_w log p_t(X_t | w) intentionally omitted:

    z = F(w)
    L_w(w) = 1 / (2 sigma_b^2) || b - A(X0_hat(X_t, F(w))) ||_F^2
             + 0.5 * lambda_w ||w||_2^2
    w <- w - rho_t * grad_w L_w(w)

Parameter meanings:

    --sigma_b:
        Observation noise std in the likelihood. If omitted, it defaults to
        --noise_std. Smaller sigma_b means b is trusted more strongly because the
        likelihood weight 1/(2 sigma_b^2) gets larger.

    --dps_step_size:
        eta_t base scale in the X update. Since L_obs is now the exact summed
        Gaussian NLL, reasonable values are much smaller than with mean-MSE loss.

    --dps_schedule:
        Defines eta_t. For constant/one_minus_alpha_bar/sigma2/x_std,
        eta_t = dps_step_size * schedule(t). For ratio, eta_t is chosen so
        ||eta_t * grad L_obs|| / ||X'_{t-1} - X_t|| equals target_update_ratio.

    --target_update_ratio:
        Used only by --dps_schedule ratio. This is the desired relative size of
        the DPS correction compared with the pretrained reverse prior step.

    --eta_min, --eta_max:
        Optional safety clamps for dynamic eta. They are disabled by default.

    --reverse_noise_scale:
        Multiplies the pretrained DDPM reverse sigma_t. This is model sampling
        noise, not observation noise sigma_b. Set 0 for deterministic reverse
        transitions when testing whether stochasticity is causing drift.

    --rho_w, --lambda_w:
        rho_w is the latent gradient descent step size. lambda_w is the
        standard-normal prior strength in 0.5 * lambda_w * ||w||_2^2.
"""


POLICIES = ("enc_no_dps", "enc_dps", "wenc_no_dps", "wenc_dps")


def parse_csv_ints(value):
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_policies(value):
    if value is None or value == "":
        return list(POLICIES)
    items = [item.strip() for item in value.split(",") if item.strip()]
    for item in items:
        if item not in POLICIES:
            raise ValueError("Unknown policy %r, choose from %s" % (item, POLICIES))
    return items


def parse_csv_floats(value):
    if value is None or value == "":
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def write_pcd_ascii(path, points):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("PCD expects shape (N, 3), got %s" % (points.shape,))
    header = "\n".join([
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS x y z",
        "SIZE 4 4 4",
        "TYPE F F F",
        "COUNT 1 1 1",
        "WIDTH %d" % points.shape[0],
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        "POINTS %d" % points.shape[0],
        "DATA ascii",
    ])
    with open(path, "w") as f:
        f.write(header + "\n")
        np.savetxt(f, points, fmt="%.8f %.8f %.8f")


def pairwise_sq_dist(a, b):
    """Exact pairwise squared Euclidean distance [B,N,M].

    Computed as (a-b)^2 directly rather than via torch.cdist's matmul expansion
    (||a||^2+||b||^2-2a.b), which suffers catastrophic cancellation at small
    distances -- exactly the nearest-neighbor regime CD/EMD rely on. This makes
    the guidance gradient (and the loss) match the exact numpy reference.
    """
    return (a.unsqueeze(2) - b.unsqueeze(1)).pow(2).sum(dim=3)


def chamfer_distance(a, b):
    dist = pairwise_sq_dist(a, b)
    return dist.min(dim=2)[0].mean(dim=1) + dist.min(dim=1)[0].mean(dim=1)


def soft_chamfer_nll(x_model, b_obs, sigma):
    """Exact NLL of the unordered Gaussian-mixture observation model:

        -log p(b|X) = -sum_i log( (1/N) sum_j N(b_i; x_j, sigma^2 I) )
                    = -sum_i [ logsumexp_j( -||b_i - x_j||^2 / (2 sigma^2) ) - log N ]

    One-directional (each observed point b_i explained by model points x_j). The
    1/(2 sigma^2) is built in; the soft assignment width is 2 sigma^2, so it
    tolerates b being off by ~sigma (noise-aware). As sigma->0 this ->
    (1/(2 sigma^2)) * one-directional Chamfer (hard nearest-neighbor).

    Averaged over observed points (not summed): the summed NLL over ~2048 points
    has huge magnitude/gradients that swamp the standard-normal w-prior under Adam
    (w then blows up). The mean keeps the data term on a scale where 0.5*lambda*||w||^2
    actually regularizes. (Mean vs sum only rescales; the minimizer is unchanged up
    to an effective prior strength.)
    """
    d = pairwise_sq_dist(b_obs, x_model)            # [B, M(obs), N(model)]
    inv = 1.0 / (2.0 * sigma * sigma)
    log_n = math.log(d.size(2))
    lse = torch.logsumexp(-d * inv, dim=2) - log_n  # [B, M]
    return (-lse).mean(dim=1).mean()




def maybe_subsample_for_emd(a, b, max_points):
    if max_points is None or max_points <= 0:
        return a, b
    if a.size(1) <= max_points and b.size(1) <= max_points:
        return a, b
    idx_a = torch.linspace(0, a.size(1) - 1, steps=min(max_points, a.size(1)), device=a.device).long()
    idx_b = torch.linspace(0, b.size(1) - 1, steps=min(max_points, b.size(1)), device=b.device).long()
    return a[:, idx_a, :], b[:, idx_b, :]


def sinkhorn_emd_loss(a, b, args):
    """Differentiable entropic OT / Sinkhorn approximation to EMD.

    The loss is permutation-invariant like EMD, but differentiable and practical in
    PyTorch. It minimizes a soft transport cost between two uniform point sets:

        min_P sum_ij P_ij ||a_i-b_j||^2 + epsilon * entropy(P)

    with row/column marginals fixed to uniform masses. Smaller epsilon is closer to
    hard EMD but can be less stable; more iterations improves marginal matching.
    """
    a, b = maybe_subsample_for_emd(a, b, args.emd_max_points)
    cost = pairwise_sq_dist(a, b)
    batch_size, n, m = cost.shape
    eps = max(float(args.emd_epsilon), 1e-8)
    log_k = -cost / eps
    log_mu = -torch.log(torch.tensor(float(n), device=a.device, dtype=a.dtype)).expand(batch_size, n)
    log_nu = -torch.log(torch.tensor(float(m), device=a.device, dtype=a.dtype)).expand(batch_size, m)
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(max(1, args.emd_iters)):
        u = log_mu - torch.logsumexp(log_k + v[:, None, :], dim=2)
        v = log_nu - torch.logsumexp(log_k + u[:, :, None], dim=1)
    log_plan = log_k + u[:, :, None] + v[:, None, :]
    plan = torch.exp(log_plan)
    return (plan * cost).sum(dim=(1, 2)).mean()

def get_sigma_b(args):
    sigma_b = args.sigma_b if args.sigma_b is not None else args.noise_std
    if sigma_b <= 0:
        raise ValueError("sigma_b must be positive, got %s" % sigma_b)
    return sigma_b


def gaussian_obs_nll(x0_hat, b, args):
    """Exact Gaussian NLL term: 1/(2 sigma_b^2) ||b - A(x0_hat)||_F^2.

    For the current denoising setup A = I and point order is assumed to be matched.
    This function deliberately uses a summed squared error, not mean MSE, so it
    matches the mathematical likelihood. Use dps_step_size to control update size.
    """
    if x0_hat.shape != b.shape:
        raise ValueError("Gaussian denoising NLL needs matched shapes, got %s and %s" % (x0_hat.shape, b.shape))
    sigma_b = get_sigma_b(args)
    squared_sum = (x0_hat - b).pow(2).flatten(1).sum(dim=1).mean()
    return 0.5 * squared_sum / (sigma_b * sigma_b)




def x_observation_loss(x0_hat, b, args):
    """Observation loss used for X DPS guidance.

    mse_nll is the Gaussian likelihood term and assumes point order is meaningful.
    cd is an unordered set-level surrogate, useful when diffusion particles do not
    preserve point-index correspondence. emd is a differentiable Sinkhorn OT loss
    with a more global soft matching than CD.
    """
    if args.x_obs_loss == "mse_nll":
        return gaussian_obs_nll(x0_hat, b, args)
    if args.x_obs_loss == "cd":
        return chamfer_distance(x0_hat, b).mean()
    if args.x_obs_loss == "emd":
        return sinkhorn_emd_loss(x0_hat, b, args)
    if args.x_obs_loss == "soft_nll":
        return soft_chamfer_nll(x0_hat, b, get_sigma_b(args))
    raise ValueError("Unsupported x_obs_loss: %s" % args.x_obs_loss)

def mse_mean(x, y):
    if x.shape != y.shape:
        return float("nan")
    return F.mse_loss(x, y).detach()


def predict_eps_and_x0(model, x_t, t, z):
    batch_size = x_t.size(0)
    sched = model.diffusion.var_sched
    beta = sched.betas[[t] * batch_size].to(x_t.device)
    alpha_bar = sched.alpha_bars[t].to(x_t.device)
    eps = model.diffusion.net(x_t, beta=beta, context=z)
    x0_hat = (x_t - torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
    return eps, x0_hat


def reverse_prior_step(model, x_t, t, z, args):
    """Pretrained DDPM reverse transition X'_{t-1} = Reverse_theta(X_t,t,z)."""
    batch_size = x_t.size(0)
    sched = model.diffusion.var_sched
    alpha = sched.alphas[t].to(x_t.device)
    alpha_bar = sched.alpha_bars[t].to(x_t.device)
    sigma_t = sched.get_sigmas(t, args.flexibility).to(x_t.device)
    eps, x0_hat = predict_eps_and_x0(model, x_t, t, z)

    noise = torch.randn_like(x_t) if t > 1 else torch.zeros_like(x_t)
    noise = args.reverse_noise_scale * noise

    c0 = 1.0 / torch.sqrt(alpha)
    c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
    x_prev = c0 * (x_t - c1 * eps) + sigma_t * noise
    return x_prev, x0_hat


def dps_schedule_scale(model, t, x_prior, args):
    """Return schedule(t), where eta_t = dps_step_size * schedule(t)."""
    sched = model.diffusion.var_sched
    if args.dps_schedule == "constant":
        return 1.0
    if args.dps_schedule == "one_minus_alpha_bar":
        return float((1.0 - sched.alpha_bars[t]).detach().cpu().item())
    if args.dps_schedule == "sigma2":
        sigma_t = sched.get_sigmas(t, args.flexibility)
        return float((sigma_t * sigma_t).detach().cpu().item())
    if args.dps_schedule == "x_std":
        return float(x_prior.detach().flatten(1).std(dim=1).mean().cpu().item())
    if args.dps_schedule == "ratio":
        raise ValueError("ratio schedule computes eta_t from grad/prior norms inside dps_step().")
    raise ValueError("Unsupported dps_schedule: %s" % args.dps_schedule)


def clamp_eta(eta_t, args):
    if args.eta_min is not None:
        eta_t = torch.clamp_min(eta_t, args.eta_min)
    if args.eta_max is not None:
        eta_t = torch.clamp_max(eta_t, args.eta_max)
    return eta_t


def encode_observation_z(model, b, sample_encoder=False):
    with torch.no_grad():
        z_mu, z_logvar = model.encoder(b)
        if sample_encoder:
            std = torch.exp(0.5 * z_logvar)
            z = z_mu + std * torch.randn_like(std)
        else:
            z = z_mu
    return z.detach()


def flow_to_z(model, w):
    return model.flow(w, reverse=True).view(w.size(0), -1)


def z_to_w(model, z):
    zeros = torch.zeros([z.size(0), 1], device=z.device)
    return model.flow(z, zeros, reverse=False)[0].view(z.size(0), -1)


def decode_latent(model, z, num_points, args):
    """Generate a point cloud from a latent z via the pretrained reverse chain.

    Uses diffusion.sample, which detaches internally (non-differentiable) -- fine
    for the reencode bootstrap which never needs gradients through decode.
    """
    with torch.no_grad():
        return model.diffusion.sample(num_points, context=z, flexibility=args.flexibility)


def reencode_latent(model, b, args):
    """Manifold-projection bootstrap to fix the encoder's OOD behaviour on noisy b.

        z <- E_phi(b)
        repeat reencode_steps:
            x_hat <- decode(z)          # project onto the data manifold
            z     <- E_phi(x_hat)       # x_hat is in-distribution -> cleaner z

    reencode_samples averages z over several decoded samples to cut sampling noise.
    """
    num_points = b.size(1)
    z = encode_observation_z(model, b, sample_encoder=args.encoder_sample)
    for _ in range(max(1, args.reencode_steps)):
        zs = []
        for _ in range(max(1, args.reencode_samples)):
            x_hat = decode_latent(model, z, num_points, args)
            zs.append(encode_observation_z(model, x_hat, sample_encoder=False))
        z = torch.stack(zs, dim=0).mean(dim=0).detach()
    return z


def invert_latent(model, b, args, w_init, return_trace=False, w_ref=None, rich_trace=False):
    """CD-based latent inversion, decoupled from the reverse chain (Adam on w).

        w* = argmin_w  E_t[ CD( X0_hat(q(b,t), t, F(w)), b ) ] + 0.5*lambda*||w||^2

    where q(b,t) noises the observation to level t. CD is used (not EMD) for speed
    and order-invariance. Single-step X0_hat per iteration -> cheap and
    differentiable (unlike full sampling, which detaches).

    If return_trace, also returns the data term being optimized (soft_nll or cd)
    measured at a FIXED (t_eval, eps_eval) snapshot every invert_log_every iters --
    a clean convergence curve for the actual objective.
    """
    num_steps = model.diffusion.var_sched.num_steps
    t_min = max(1, args.invert_t_min)
    t_max = args.invert_t_max if args.invert_t_max and args.invert_t_max > 0 else num_steps
    t_max = min(t_max, num_steps)
    w = w_init.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=args.invert_lr)
    b_fixed = b.detach()
    iters = max(1, args.invert_iters)

    # Optional norm projection: pin ||w|| to a target radius each step (projected
    # gradient on the sphere) so only the DIRECTION is optimized -- prevents the
    # collapse/blowup the ||w||^2 prior caused. Target R is GT-free (a population
    # constant, e.g. mean clean-latent norm), NOT the per-instance GT norm.
    norm_target = float(getattr(args, "invert_norm_project", 0.0) or 0.0)

    def project_norm():
        if norm_target > 0:
            with torch.no_grad():
                cur = w.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-8)
                w.mul_(norm_target / cur)

    project_norm()  # start on the sphere
    # MAP objective 1/(2 sigma^2) CD + 0.5 lambda ||w||^2, rescaled (Adam is invariant
    # to overall loss scale) to CD + 0.5 * lambda * sigma^2 * ||w||^2. So the EFFECTIVE
    # w-prior weight grows with sigma_b^2 -> noisier obs => trust b less, prior more.
    if getattr(args, "invert_sigma_weight", True):
        sigma_b = get_sigma_b(args)
        prior_weight = args.invert_lambda * sigma_b * sigma_b
    else:
        prior_weight = args.invert_lambda

    # Fixed evaluation snapshot for an interpretable CD-to-b curve.
    trace = []
    log_every = max(1, getattr(args, "invert_log_every", 10))
    t_eval = max(t_min, min(t_max, (t_min + t_max) // 2))
    abar_eval = model.diffusion.var_sched.alpha_bars[t_eval].to(b_fixed.device)
    eps_eval = torch.randn_like(b_fixed)
    x_t_eval = torch.sqrt(abar_eval) * b_fixed + torch.sqrt(1 - abar_eval) * eps_eval

    def data_at(x_t_snap, t_snap):
        z = flow_to_z(model, w)
        _, x0_hat = predict_eps_and_x0(model, x_t_snap, t_snap, z)
        if args.invert_loss == "soft_nll":
            return float(soft_chamfer_nll(x0_hat, b_fixed, get_sigma_b(args)).detach().cpu().item())
        return float(chamfer_distance(x0_hat, b_fixed).mean().detach().cpu().item())

    def eval_data_loss():
        # data term at the single fixed (t_eval, eps_eval) snapshot.
        with torch.no_grad():
            return data_at(x_t_eval, t_eval)

    # rich_trace: average data over a fixed t-grid + cossim-to-w_ref + ||w||.
    if rich_trace:
        n_grid = 8
        t_grid = sorted(set(int(round(v)) for v in np.linspace(t_min, t_max, n_grid)))
        xt_grid = []
        for tg in t_grid:
            ab = model.diffusion.var_sched.alpha_bars[tg].to(b_fixed.device)
            xt_grid.append((torch.sqrt(ab) * b_fixed + torch.sqrt(1 - ab) * torch.randn_like(b_fixed), tg))

    def eval_rich(i):
        with torch.no_grad():
            data = float(np.mean([data_at(xt, tg) for xt, tg in xt_grid]))
            cos = w_cosine(w, w_ref) if w_ref is not None else float("nan")
            wn = float(w.flatten(1).norm(dim=1).mean().detach().cpu().item())
            return OrderedDict(it=int(i), data=data, cos=cos, w_norm=wn)

    for i in range(iters):
        if rich_trace and (i % log_every == 0):
            trace.append(eval_rich(i))
        elif return_trace and (i % log_every == 0):
            trace.append((i, eval_data_loss()))
        t = int(torch.randint(t_min, t_max + 1, (1,)).item())
        alpha_bar = model.diffusion.var_sched.alpha_bars[t].to(b_fixed.device)
        eps = torch.randn_like(b_fixed)
        x_t = torch.sqrt(alpha_bar) * b_fixed + torch.sqrt(1 - alpha_bar) * eps
        z = flow_to_z(model, w)
        _, x0_hat = predict_eps_and_x0(model, x_t, t, z)
        if args.invert_loss == "soft_nll":
            data = soft_chamfer_nll(x0_hat, b_fixed, get_sigma_b(args))
        else:  # "cd": scale-free symmetric Chamfer
            data = chamfer_distance(x0_hat, b_fixed).mean()
        # sigma^2-weighted w-prior (when --invert_sigma_weight): the data-vs-prior
        # balance auto-adapts to noise -- higher sigma_b => stronger pull toward the
        # N(0,I) prior (posterior -> prior when the likelihood is weak / obs noisy).
        loss = data + 0.5 * prior_weight * w.pow(2).sum(dim=1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        project_norm()   # re-project onto ||w||=R (no-op if norm_target<=0)
    if rich_trace:
        trace.append(eval_rich(iters))
        return w.detach(), trace
    if return_trace:
        trace.append((iters, eval_data_loss()))
        return w.detach(), trace
    return w.detach()


def add_latent_init_args(parser):
    """Latent initialization strategy + its hyperparameters (shared by scripts)."""
    parser.add_argument("--latent_init", default="observation",
                        choices=["observation", "gt", "reencode", "invert"],
                        help="observation: E_phi(b). gt: E_phi(x0) (oracle). "
                             "reencode: decode->reencode bootstrap. invert: CD latent inversion.")
    # reencode (manifold-projection bootstrap)
    parser.add_argument("--reencode_steps", type=int, default=1, help="Bootstrap iterations (K).")
    parser.add_argument("--reencode_samples", type=int, default=1, help="Decodes averaged per bootstrap step.")
    # invert (latent inversion via Adam on w)
    parser.add_argument("--invert_loss", default="cd", choices=["cd", "soft_nll"],
                        help="cd: fast hard Chamfer. soft_nll: exact Gaussian-mixture NLL "
                             "(soft assignment, width 2*sigma_b^2, noise-aware).")
    parser.add_argument("--invert_iters", type=int, default=200)
    parser.add_argument("--invert_lr", type=float, default=1e-2)
    parser.add_argument("--invert_lambda", type=float, default=1e-3, help="Standard-normal prior on w.")
    parser.add_argument("--invert_t_min", type=int, default=1)
    parser.add_argument("--invert_t_max", type=int, default=0, help="<=0 means num_steps.")
    parser.add_argument("--invert_log_every", type=int, default=10, help="CD-to-b logging stride for invert.")
    parser.add_argument("--invert_norm_project", type=float, default=0.0,
                        help="If >0, project ||w|| onto this radius each step (direction-only inversion). GT-free target.")
    parser.add_argument("--invert_sigma_weight", dest="invert_sigma_weight", action="store_true", default=True,
                        help="Scale the w-prior by sigma_b^2 (noise-adaptive MAP balance).")
    parser.add_argument("--no_invert_sigma_weight", dest="invert_sigma_weight", action="store_false")


def select_latent(model, b, args, z_enc, w_enc_init, z_gt, w_gt):
    """Resolve the (z, w) used to seed the reverse chain for the chosen strategy.

        observation: E_phi(b)              (inverse-problem default)
        gt:          E_phi(x0)             (oracle upper bound)
        reencode:    decode->reencode bootstrap of E_phi(b)
        invert:      CD latent inversion, warm-started from E_phi(b)
    """
    if args.latent_init == "gt":
        return z_gt, w_gt
    if args.latent_init == "reencode":
        z = reencode_latent(model, b, args)
        return z, z_to_w(model, z).detach()
    if args.latent_init == "invert":
        w = invert_latent(model, b, args, w_init=w_enc_init)
        return flow_to_z(model, w).detach(), w
    return z_enc, w_enc_init


def w_update_step(model, x_t, b, w, t, args, w_target=None):
    """Latent posterior step from the practical joint posterior approximation.

    Implements:

        grad_w log p_t(X_t,w|b)
          approx grad_w log p(w) + grad_w log p_t(b|X_t,w)

    as minimization of

        L_w = L_w_obs(b, f_theta(X_t,t,F(w)))
              + 0.5 * lambda_w * ||w||_2^2
              + 0.5 * beta_w_target * ||w - w_target||_2^2

    where --w_obs_loss selects L_w_obs:
        none:    no observation term for w; useful for oracle diagnostics
        mse_nll: 1/(2 sigma_b^2)||b - X0_hat||_F^2
        cd:      ChamferDistance(b, X0_hat)
        emd:     differentiable Sinkhorn EMD / OT loss

    Parameter effects:
        rho_w:     rho_t, how far w moves per gradient descent step.
        lambda_w:  strength of the standard-normal prior pull toward 0.
        sigma_b:   observation confidence through 1/(2 sigma_b^2).
        beta_w_target: oracle/diagnostic pull toward a target w, usually GT w.
    """
    x_fixed = x_t.detach()
    w_current = w.detach()
    info = OrderedDict(
        w_loss=0.0,
        w_obs_nll=0.0,
        w_prior_nll=0.0,
        w_obs_cd=0.0,
        w_obs_emd=0.0,
        w_target_nll=0.0,
        w_grad_norm=0.0,
        w_step_norm=0.0,
    )
    for _ in range(max(1, args.w_update_steps)):
        w_req = w_current.detach().requires_grad_(True)
        z = flow_to_z(model, w_req)
        _, x0_hat = predict_eps_and_x0(model, x_fixed, t, z)
        obs_nll = gaussian_obs_nll(x0_hat, b.detach(), args)
        obs_cd = chamfer_distance(x0_hat, b.detach()).mean()
        obs_emd = sinkhorn_emd_loss(x0_hat, b.detach(), args) if args.w_obs_loss == "emd" else torch.zeros((), device=w_req.device, dtype=w_req.dtype)
        if args.w_obs_loss == "none":
            obs_loss = torch.zeros((), device=w_req.device, dtype=w_req.dtype)
        elif args.w_obs_loss == "mse_nll":
            obs_loss = obs_nll
        elif args.w_obs_loss == "cd":
            obs_loss = obs_cd
        elif args.w_obs_loss == "emd":
            obs_loss = obs_emd
        else:
            raise ValueError("Unsupported w_obs_loss: %s" % args.w_obs_loss)
        prior_nll = 0.5 * args.lambda_w * w_req.pow(2).sum(dim=1).mean()
        if args.beta_w_target > 0 and w_target is not None:
            target_nll = 0.5 * args.beta_w_target * (w_req - w_target.detach()).pow(2).sum(dim=1).mean()
        else:
            target_nll = torch.zeros((), device=w_req.device, dtype=w_req.dtype)
        loss = obs_loss + prior_nll + target_nll
        grad = torch.autograd.grad(loss, w_req)[0]
        w_next = (w_req - args.rho_w * grad).detach()
        info = OrderedDict(
            w_loss=float(loss.detach().cpu().item()),
            w_obs_loss=float(obs_loss.detach().cpu().item()),
            w_obs_nll=float(obs_nll.detach().cpu().item()),
            w_obs_cd=float(obs_cd.detach().cpu().item()),
            w_obs_emd=float(obs_emd.detach().cpu().item()),
            w_prior_nll=float(prior_nll.detach().cpu().item()),
            w_target_nll=float(target_nll.detach().cpu().item()),
            w_grad_norm=float(grad.detach().flatten(1).norm(dim=1).mean().cpu().item()),
            w_step_norm=float((w_next - w_req.detach()).flatten(1).norm(dim=1).mean().cpu().item()),
        )
        w_current = w_next
    return w_current, info

def make_observation(x0, args):
    noise = args.noise_std * torch.randn_like(x0)
    b = x0 + noise
    return b, OrderedDict(problem="denoising", noise_std=args.noise_std, sigma_b=get_sigma_b(args))


def make_x_start(model, b, args):
    total_steps = model.diffusion.var_sched.num_steps
    start_t = args.t_start if args.t_start is not None else total_steps
    if start_t < 1 or start_t > total_steps:
        raise ValueError("t_start must be in [1, %d], got %d" % (total_steps, start_t))

    if args.x_init == "random":
        x_t = torch.randn_like(b)
        return x_t, start_t, OrderedDict(x_init="random", t_start=start_t)

    if args.x_init == "observation":
        alpha_bar = model.diffusion.var_sched.alpha_bars[start_t].to(b.device)
        eps = torch.randn_like(b)
        x_t = torch.sqrt(alpha_bar) * b + torch.sqrt(1 - alpha_bar) * eps
        return x_t, start_t, OrderedDict(x_init="observation", t_start=start_t)

    raise ValueError("Unsupported x_init: %s" % args.x_init)


def should_trace(t, start_t, args):
    every = max(1, args.trace_every)
    return t == start_t or t == 1 or t % every == 0


def dps_step(model, x_t, b, z, t, args):
    """Apply X_{t-1} = Reverse_theta(X_t,t,z) - eta_t grad_{X_t} L_obs."""
    x_req = x_t.detach().requires_grad_(True)
    x_prior, x0_hat = reverse_prior_step(model, x_req, t, z.detach(), args)
    obs_nll = gaussian_obs_nll(x0_hat, b, args)
    obs_loss = x_observation_loss(x0_hat, b, args)
    grad = torch.autograd.grad(obs_loss, x_req)[0]

    if args.normalize_dps_grad:
        # Optional diagnostic stabilizer. Off by default because it changes the
        # exact update direction scale from the Gaussian posterior equation.
        grad_norm = grad.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(args.dps_grad_eps)
        grad = grad / grad_norm

    grad_norm = grad.detach().flatten(1).norm(dim=1).mean().clamp_min(args.dps_grad_eps)
    prior_delta = x_prior.detach() - x_t.detach()
    prior_delta_norm = prior_delta.flatten(1).norm(dim=1).mean()

    if args.dps_schedule == "ratio":
        # Dynamic eta: choose eta_t so the DPS correction has a target size
        # relative to the pretrained reverse prior step. No clamp is applied
        # unless --eta_min or --eta_max is explicitly provided.
        eta_t = args.target_update_ratio * prior_delta_norm / grad_norm
        eta_t_unclamped = eta_t.detach().clone()
        eta_t = clamp_eta(eta_t, args)
    else:
        eta_scalar = args.dps_step_size * dps_schedule_scale(model, t, x_prior, args)
        eta_t = torch.as_tensor(eta_scalar, device=x_t.device, dtype=x_t.dtype)
        eta_t_unclamped = eta_t.detach().clone()

    dps_delta = eta_t * grad.detach()
    dps_delta_norm = dps_delta.flatten(1).norm(dim=1).mean()
    update_ratio = dps_delta_norm / prior_delta_norm.clamp_min(1e-12)
    x_next = x_prior.detach() - dps_delta
    eta_value = float(eta_t.detach().cpu().item())
    diagnostics = OrderedDict(
        eta_unclamped=float(eta_t_unclamped.detach().cpu().item()),
        dps_delta_norm=float(dps_delta_norm.detach().cpu().item()),
        prior_delta_norm=float(prior_delta_norm.detach().cpu().item()),
        update_ratio=float(update_ratio.detach().cpu().item()),
    )
    diagnostics["x_obs_loss"] = float(obs_loss.detach().cpu().item())
    diagnostics["x_obs_nll"] = float(obs_nll.detach().cpu().item())
    diagnostics["x_obs_cd"] = float(chamfer_distance(x0_hat, b).mean().detach().cpu().item())
    diagnostics["x_obs_emd"] = float(obs_loss.detach().cpu().item()) if args.x_obs_loss == "emd" else 0.0
    return x_next, x0_hat.detach(), obs_loss.detach(), eta_value, grad.detach(), diagnostics


def run_enc_chain(model, b, x_start, start_t, z, use_dps, args):
    x_t = x_start.detach()
    trace = []
    x0hat_trace = OrderedDict()

    for t in tqdm(range(start_t, 0, -1), desc="enc_dps" if use_dps else "enc_no_dps", leave=False):
        if use_dps:
            x_t, x0_hat, obs_nll, eta_t, grad, diagnostics = dps_step(model, x_t, b, z, t, args)
            grad_norm = float(grad.flatten(1).norm(dim=1).mean().detach().cpu().item())
        else:
            with torch.no_grad():
                x_t, x0_hat = reverse_prior_step(model, x_t, t, z, args)
                obs_nll = gaussian_obs_nll(x0_hat, b, args)
            eta_t = 0.0
            grad_norm = 0.0
            diagnostics = OrderedDict(dps_delta_norm=0.0, prior_delta_norm=0.0, update_ratio=0.0)
            x_t = x_t.detach()
            x0_hat = x0_hat.detach()

        if should_trace(t, start_t, args):
            x0hat_trace[t] = x0_hat.squeeze(0).detach().cpu().numpy()
            trace.append(OrderedDict(
                t=int(t),
                obs_nll=float(obs_nll.detach().cpu().item()),
                mse_to_obs=float(mse_mean(x0_hat, b).cpu().item()),
                x0hat_cd_to_obs=float(chamfer_distance(x0_hat, b).mean().detach().cpu().item()),
                eta_t=float(eta_t),
                grad_norm=grad_norm,
                eta_unclamped=diagnostics.get("eta_unclamped", float(eta_t)),
                dps_delta_norm=diagnostics["dps_delta_norm"],
                prior_delta_norm=diagnostics["prior_delta_norm"],
                update_ratio=diagnostics["update_ratio"],
                x_obs_loss=diagnostics.get("x_obs_loss", float(obs_nll.detach().cpu().item())),
                x_obs_nll=diagnostics.get("x_obs_nll", float(obs_nll.detach().cpu().item())),
                x_obs_cd=diagnostics.get("x_obs_cd", float(chamfer_distance(x0_hat, b).mean().detach().cpu().item())),
                x_obs_emd=diagnostics.get("x_obs_emd", 0.0),
            ))

    return x_t.detach(), trace, x0hat_trace



def run_w_chain(model, b, x_start, start_t, w_init, use_dps, args, w_refs=None):
    """Reverse chain with encoder-initialized w posterior updates.

    Initial condition:
        z_b = E_phi(b)
        w_init = F^{-1}(z_b)

    At every selected timestep, update only w using
        w <- w - rho_w * grad_w [L_obs(X_t,F(w)) + 0.5*lambda_w*||w||^2]

    Then run the same X transition as the enc policies, using z = F(w).
    """
    x_t = x_start.detach()
    w_t = w_init.detach()
    w0 = w_init.detach()
    trace = []
    x0hat_trace = OrderedDict()
    w_trace = []

    desc = "wenc_dps" if use_dps else "wenc_no_dps"
    update_every = max(1, args.w_update_every)
    for t in tqdm(range(start_t, 0, -1), desc=desc, leave=False):
        if t % update_every == 0:
            w_t, w_info = w_update_step(model, x_t, b, w_t, t, args, w_target=(w_refs or {}).get("gt"))
        else:
            w_info = OrderedDict(
                w_loss=0.0,
                w_obs_nll=0.0,
                w_prior_nll=0.0,
                w_obs_cd=0.0,
                w_obs_emd=0.0,
                w_target_nll=0.0,
                w_grad_norm=0.0,
                w_step_norm=0.0,
            )

        with torch.no_grad():
            z_t = flow_to_z(model, w_t).detach()

        if use_dps:
            x_t, x0_hat, obs_nll, eta_t, grad, diagnostics = dps_step(model, x_t, b, z_t, t, args)
            grad_norm = float(grad.flatten(1).norm(dim=1).mean().detach().cpu().item())
        else:
            with torch.no_grad():
                x_t, x0_hat = reverse_prior_step(model, x_t, t, z_t, args)
                obs_nll = gaussian_obs_nll(x0_hat, b, args)
            eta_t = 0.0
            grad_norm = 0.0
            diagnostics = OrderedDict(dps_delta_norm=0.0, prior_delta_norm=0.0, update_ratio=0.0)
            x_t = x_t.detach()
            x0_hat = x0_hat.detach()

        if should_trace(t, start_t, args):
            x0hat_trace[t] = x0_hat.squeeze(0).detach().cpu().numpy()
            if w_refs is not None:
                w_row = OrderedDict(t=int(t))
                if "gt" in w_refs:
                    w_row["l2_to_gt_w"] = w_l2(w_t, w_refs["gt"])
                    w_row["cos_to_gt_w"] = w_cosine(w_t, w_refs["gt"])
                if "obs" in w_refs:
                    w_row["l2_to_obs_w"] = w_l2(w_t, w_refs["obs"])
                    w_row["cos_to_obs_w"] = w_cosine(w_t, w_refs["obs"])
                w_row["l2_from_init"] = w_l2(w_t, w0)
                w_row["w_norm"] = float(w_t.flatten(1).norm(dim=1).mean().detach().cpu().item())
                w_trace.append(w_row)
            trace.append(OrderedDict(
                t=int(t),
                obs_nll=float(obs_nll.detach().cpu().item()),
                mse_to_obs=float(mse_mean(x0_hat, b).cpu().item()),
                x0hat_cd_to_obs=float(chamfer_distance(x0_hat, b).mean().detach().cpu().item()),
                eta_t=float(eta_t),
                grad_norm=grad_norm,
                eta_unclamped=diagnostics.get("eta_unclamped", float(eta_t)),
                dps_delta_norm=diagnostics["dps_delta_norm"],
                prior_delta_norm=diagnostics["prior_delta_norm"],
                update_ratio=diagnostics["update_ratio"],
                x_obs_loss=diagnostics.get("x_obs_loss", float(obs_nll.detach().cpu().item())),
                x_obs_nll=diagnostics.get("x_obs_nll", float(obs_nll.detach().cpu().item())),
                x_obs_cd=diagnostics.get("x_obs_cd", float(chamfer_distance(x0_hat, b).mean().detach().cpu().item())),
                x_obs_emd=diagnostics.get("x_obs_emd", 0.0),
                w_norm=float(w_t.flatten(1).norm(dim=1).mean().detach().cpu().item()),
                w_l2_from_init=float((w_t - w0).flatten(1).norm(dim=1).mean().detach().cpu().item()),
                w_loss=w_info["w_loss"],
                w_obs_loss=w_info.get("w_obs_loss", 0.0),
                w_obs_nll=w_info["w_obs_nll"],
                w_obs_cd=w_info.get("w_obs_cd", 0.0),
                w_obs_emd=w_info.get("w_obs_emd", 0.0),
                w_prior_nll=w_info["w_prior_nll"],
                w_target_nll=w_info.get("w_target_nll", 0.0),
                w_grad_norm=w_info["w_grad_norm"],
                w_step_norm=w_info["w_step_norm"],
            ))

    return x_t.detach(), trace, x0hat_trace, w_t.detach(), w_trace

def tensor_metrics(points, x0, b, args):
    """Final reported GT/obs metrics, computed in numpy so they are identical to
    the plot labels (single source of truth). The differentiable torch CD/EMD
    are used only for DPS guidance, not for reporting."""
    p = points.squeeze(0).detach().cpu().numpy()
    g = x0.squeeze(0).detach().cpu().numpy()
    o = b.squeeze(0).detach().cpu().numpy()
    sigma_b = get_sigma_b(args)
    if p.shape == o.shape and np.isfinite(p).all() and np.isfinite(o).all():
        obs_nll = float(0.5 * np.sum((p - o) ** 2) / (sigma_b * sigma_b))
    else:
        obs_nll = float("nan")
    return OrderedDict(
        cd_to_gt=np_chamfer_to_gt(p, g),
        emd_to_gt=np_emd_to_gt(p, g, args),
        obs_cd=np_chamfer_to_gt(p, o),
        mse_to_gt=np_mse_to_gt(p, g),
        mse_to_obs=np_mse_to_gt(p, o),
        obs_nll=obs_nll,
    )


def load_model(args):
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if getattr(ckpt["args"], "model", None) != "flow":
        raise ValueError("This script expects a flow GEN checkpoint.")
    model = FlowVAE(ckpt["args"]).to(args.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    args.flexibility = ckpt["args"].flexibility if args.flexibility is None else args.flexibility
    args.latent_dim = ckpt["args"].latent_dim
    return model, ckpt


def save_arrays(case_dir, arrays, save_pcd):
    for name, arr in arrays.items():
        np.save(os.path.join(case_dir, "%s.npy" % name), arr)
        if save_pcd:
            write_pcd_ascii(os.path.join(case_dir, "%s.pcd" % name), arr)


def save_x0hat_trace(case_dir, traces, save_pcd):
    root = os.path.join(case_dir, "x0_hat_trace")
    os.makedirs(root, exist_ok=True)
    index = OrderedDict()
    for policy, snapshots in traces.items():
        policy_dir = os.path.join(root, policy)
        os.makedirs(policy_dir, exist_ok=True)
        index[policy] = []
        for t, arr in snapshots.items():
            stem = "t%03d" % int(t)
            path = os.path.join(policy_dir, "%s.npy" % stem)
            np.save(path, arr)
            if save_pcd:
                write_pcd_ascii(os.path.join(policy_dir, "%s.pcd" % stem), arr)
            index[policy].append(OrderedDict(t=int(t), file=os.path.relpath(path, case_dir)))
    with open(os.path.join(root, "index.json"), "w") as f:
        json.dump(index, f, indent=2)


def finite_points(points):
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3:
        return np.zeros((0, 3), dtype=np.float32)
    mask = np.isfinite(points).all(axis=1)
    return points[mask]


def xz_limits(arrays):
    finite_arrays = []
    for arr in arrays:
        pts = finite_points(arr)
        if len(pts) > 0:
            finite_arrays.append(pts[:, [0, 2]])
    if not finite_arrays:
        return (-1.0, 1.0), (-1.0, 1.0)
    stacked = np.concatenate(finite_arrays, axis=0)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    center = (mins + maxs) / 2
    span = float((maxs - mins).max())
    if not np.isfinite(span) or span <= 0:
        span = 2.0
    pad = max(span * 0.08, 1e-6)
    half = span / 2 + pad
    return (center[0] - half, center[0] + half), (center[1] - half, center[1] + half)


def np_chamfer_to_gt(points, gt):
    if not np.isfinite(points).all() or not np.isfinite(gt).all():
        return float("nan")
    diff = points[:, None, :] - gt[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    return float(dist2.min(axis=1).mean() + dist2.min(axis=0).mean())


def np_mse_to_gt(points, gt):
    if points.shape != gt.shape or not np.isfinite(points).all() or not np.isfinite(gt).all():
        return float("nan")
    return float(np.mean((points - gt) ** 2))


def _np_logsumexp(a, axis):
    amax = np.max(a, axis=axis, keepdims=True)
    out = amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def np_emd_to_gt(points, gt, args):
    """Numpy Sinkhorn EMD-GT (single source of truth for both records and plots).

    Uses full point sets by default (more accurate); only subsamples if the user
    explicitly sets --emd_max_points > 0, matching sinkhorn_emd_loss.
    """
    points = finite_points(points)
    gt = finite_points(gt)
    if len(points) == 0 or len(gt) == 0:
        return float("nan")
    cap = args.emd_max_points if (args.emd_max_points and args.emd_max_points > 0) else None
    if cap is not None and len(points) > cap:
        points = points[np.linspace(0, len(points) - 1, cap).astype(int)]
    if cap is not None and len(gt) > cap:
        gt = gt[np.linspace(0, len(gt) - 1, cap).astype(int)]
    diff = points[:, None, :] - gt[None, :, :]
    cost = np.sum(diff * diff, axis=2)
    n, m = cost.shape
    eps = max(float(args.emd_epsilon), 1e-8)
    log_k = -cost / eps
    log_mu = -np.log(n) * np.ones(n)
    log_nu = -np.log(m) * np.ones(m)
    u = np.zeros(n)
    v = np.zeros(m)
    for _ in range(max(1, args.emd_iters)):
        u = log_mu - _np_logsumexp(log_k + v[None, :], axis=1)
        v = log_nu - _np_logsumexp(log_k + u[:, None], axis=0)
    plan = np.exp(log_k + u[:, None] + v[None, :])
    return float(np.sum(plan * cost))


def np_soft_nll_to_gt(points, gt, sigma):
    """Numpy soft-NLL of a result to the clean shape, matching the torch
    soft_chamfer_nll(points, gt, sigma): x_model=points, b_obs=gt.

        -mean_i log( (1/N) sum_j exp(-||gt_i - points_j||^2 / (2 sigma^2)) )
    """
    points = finite_points(points)
    gt = finite_points(gt)
    if len(points) == 0 or len(gt) == 0:
        return float("nan")
    diff = gt[:, None, :] - points[None, :, :]
    d = np.sum(diff * diff, axis=2)                       # [M_gt, N_pts]
    inv = 1.0 / (2.0 * max(float(sigma), 1e-8) ** 2)
    lse = _np_logsumexp(-d * inv, axis=1) - np.log(points.shape[0])
    return float(np.mean(-lse))


def gt_metric_label(points, gt, args):
    return "CD-GT %.4g\nEMD-GT %.4g" % (np_chamfer_to_gt(points, gt), np_emd_to_gt(points, gt, args))


def scatter_xz(ax, points, xlim, zlim, args):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.axis("off")
    finite = finite_points(points)
    if len(finite) == 0:
        ax.text(0.5, 0.5, "invalid", transform=ax.transAxes, ha="center", va="center", fontsize=9, color="crimson")
        return
    ax.scatter(finite[:, 0], finite[:, 2], s=args.plot_point_size, c=args.plot_color, alpha=args.plot_alpha, linewidths=0)
    bad_count = int(np.asarray(points).shape[0] - finite.shape[0])
    if bad_count > 0:
        ax.text(0.5, 0.5, "nonfinite %d" % bad_count, transform=ax.transAxes, ha="center", va="center", fontsize=8, color="crimson")



def w_l2(a, b):
    return float((a - b).flatten(1).norm(dim=1).mean().detach().cpu().item())


def w_cosine(a, b):
    return float(F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1).mean().detach().cpu().item())


def make_w_trace_plot(case_dir, w_diagnostics, args):
    if not args.make_plot or not w_diagnostics:
        return
    traces = w_diagnostics.get("traces", {})
    if not traces:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    colors = {"wenc_no_dps": "#1f77b4", "wenc_dps": "#d62728"}
    for policy, rows in traces.items():
        if not rows:
            continue
        ts = [row["t"] for row in rows]
        l2_gt = [row["l2_to_gt_w"] for row in rows]
        l2_obs = [row["l2_to_obs_w"] for row in rows]
        cos_gt = [row["cos_to_gt_w"] for row in rows]
        color = colors.get(policy, None)
        axes[0].plot(ts, l2_gt, marker="o", markersize=2.5, linewidth=1.4, label="%s to GT w" % policy, color=color)
        axes[0].plot(ts, l2_obs, marker="x", markersize=3.0, linewidth=1.0, linestyle="--", label="%s to obs enc w" % policy, color=color, alpha=0.75)
        axes[1].plot(ts, cos_gt, marker="o", markersize=2.5, linewidth=1.4, label="%s to GT w" % policy, color=color)

    init = w_diagnostics.get("init", {})
    title_extra = "GT/obs enc cos %.3f, L2 %.3g" % (init.get("obs_cos_to_gt", float("nan")), init.get("obs_l2_to_gt", float("nan")))
    axes[0].set_title("w L2 distance\n" + title_extra, fontsize=10)
    axes[0].set_xlabel("timestep")
    axes[0].set_ylabel("L2 distance")
    axes[0].invert_xaxis()
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7)

    axes[1].set_title("w cosine similarity to GT encoder w", fontsize=10)
    axes[1].set_xlabel("timestep")
    axes[1].set_ylabel("cosine")
    axes[1].invert_xaxis()
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=7)

    fig.suptitle("w trajectory diagnostics", fontsize=13)
    fig.savefig(os.path.join(case_dir, "w_trace.png"), dpi=args.plot_dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def make_case_plots(case_dir, arrays, x0hat_traces, records, args):
    if not args.make_plot:
        return

    rows = ["gt", "observation", "x_start", "enc_no_dps", "enc_dps", "wenc_no_dps", "wenc_dps"]
    available_rows = [row for row in rows if row in arrays]
    xlim, zlim = xz_limits([arrays[row] for row in available_rows])

    fig, axes = plt.subplots(len(available_rows), 1, figsize=(3.2, max(2.1 * len(available_rows), 6)), squeeze=False, constrained_layout=True)
    for row_idx, name in enumerate(available_rows):
        ax = axes[row_idx][0]
        scatter_xz(ax, arrays[name], xlim, zlim, args)
        label = name + "\n" + gt_metric_label(arrays[name], arrays["gt"], args)
        ax.text(-0.04, 0.5, label, transform=ax.transAxes, ha="right", va="center", rotation=90, fontsize=8, fontweight="bold")
    fig.suptitle("Final outputs, y-axis view", fontsize=13)
    fig.savefig(os.path.join(case_dir, "final_grid.png"), dpi=args.plot_dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    timesteps = sorted({t for snapshots in x0hat_traces.values() for t in snapshots.keys()}, reverse=True)
    if not timesteps:
        return
    trace_rows = [("gt", arrays["gt"]), ("observation", arrays["observation"]), ("x_start", arrays["x_start"])]
    trace_rows.extend((policy, None) for policy in POLICIES if policy in x0hat_traces)
    trace_arrays = [arrays["gt"], arrays["observation"], arrays["x_start"]]
    for snapshots in x0hat_traces.values():
        trace_arrays.extend(snapshots.values())
    xlim, zlim = xz_limits(trace_arrays)

    fig_w = max(2.2 * len(timesteps), 7)
    fig_h = max(2.0 * len(trace_rows), 6)
    fig, axes = plt.subplots(len(trace_rows), len(timesteps), figsize=(fig_w, fig_h), squeeze=False, constrained_layout=True)
    for col, t in enumerate(timesteps):
        for row_idx, (name, fixed) in enumerate(trace_rows):
            ax = axes[row_idx][col]
            points = fixed if fixed is not None else x0hat_traces[name].get(t)
            if points is None:
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlim(*xlim)
                ax.set_ylim(*zlim)
                ax.axis("off")
                ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center")
            else:
                scatter_xz(ax, points, xlim, zlim, args)
                ax.text(0.5, -0.03, gt_metric_label(points, arrays["gt"], args), transform=ax.transAxes, ha="center", va="top", fontsize=6.5)
            if row_idx == 0:
                ax.set_title("t=%d" % int(t), fontsize=9)
            if col == 0:
                ax.text(-0.04, 0.5, name, transform=ax.transAxes, ha="right", va="center", rotation=90, fontsize=9, fontweight="bold")
    fig.suptitle("x0_hat trace, y-axis view", fontsize=13)
    fig.savefig(os.path.join(case_dir, "x0hat_trace_grid.png"), dpi=args.plot_dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)



def unpack_run_output(item):
    if len(item) == 2:
        return item[0], item[1], None
    return item[0], item[1], item[2]


def make_w_sweep_plot(run_outputs, args):
    if not args.make_sweep_plot or len(run_outputs) <= 1:
        return
    items = [unpack_run_output(item) for item in run_outputs]
    if not any(metadata and metadata.get("w_trace") for _, _, metadata in items):
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), squeeze=False, constrained_layout=True)
    policy_styles = {
        "wenc_no_dps": dict(linestyle="--", marker="x", alpha=0.75),
        "wenc_dps": dict(linestyle="-", marker="o", alpha=0.95),
    }
    cmap = plt.get_cmap("viridis")
    denom = max(1, len(items) - 1)

    for idx, (sweep_value, arrays, metadata) in enumerate(items):
        if metadata is None:
            continue
        traces = metadata.get("w_trace", {})
        color = cmap(idx / denom)
        label_prefix = column_title(args, sweep_value)
        for policy, rows in traces.items():
            if not rows:
                continue
            style = policy_styles.get(policy, dict(linestyle="-", marker="o", alpha=0.85))
            ts = [row["t"] for row in rows]
            label = "%s %s" % (label_prefix, policy.replace("wenc_", ""))
            axes[0][0].plot(ts, [row.get("l2_to_gt_w", float("nan")) for row in rows], color=color, linewidth=1.25, markersize=2.5, label=label, **style)
            axes[0][1].plot(ts, [row.get("cos_to_gt_w", float("nan")) for row in rows], color=color, linewidth=1.25, markersize=2.5, label=label, **style)
            axes[1][0].plot(ts, [row.get("l2_from_init", float("nan")) for row in rows], color=color, linewidth=1.25, markersize=2.5, label=label, **style)
            axes[1][1].plot(ts, [row.get("w_norm", float("nan")) for row in rows], color=color, linewidth=1.25, markersize=2.5, label=label, **style)

    titles = [
        "L2 distance to GT encoder w",
        "Cosine similarity to GT encoder w",
        "L2 movement from observation encoder w",
        "w norm",
    ]
    ylabels = ["L2", "cosine", "L2", "norm"]
    for ax, title, ylabel in zip(axes.flatten(), titles, ylabels):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("timestep")
        ax.set_ylabel(ylabel)
        ax.invert_xaxis()
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)
    os.makedirs(args.plot_output_dir, exist_ok=True)
    base = args.plot_output_name or "%s_sweep_grid.png" % args.base_run_name
    stem = os.path.splitext(base)[0]
    out_name = "%s_w_trace.png" % stem
    fig.suptitle((args.plot_title or args.base_run_name) + " - w trajectory sweep", fontsize=13)
    fig.savefig(os.path.join(args.plot_output_dir, out_name), dpi=args.plot_dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def make_sweep_plot(run_outputs, args):
    if not args.make_sweep_plot or len(run_outputs) <= 1:
        return
    rows = ["gt", "observation", "x_start", "enc_no_dps", "enc_dps", "wenc_no_dps", "wenc_dps"]
    all_arrays = []
    for item in run_outputs:
        _, arrays, _ = unpack_run_output(item)
        for row in rows:
            if row in arrays:
                all_arrays.append(arrays[row])
    if not all_arrays:
        return
    xlim, zlim = xz_limits(all_arrays)
    fig_w = max(2.4 * len(run_outputs), 7)
    fig_h = max(2.0 * len(rows), 6)
    fig, axes = plt.subplots(len(rows), len(run_outputs), figsize=(fig_w, fig_h), squeeze=False, constrained_layout=True)
    for col, item in enumerate(run_outputs):
        dps_value, arrays, _ = unpack_run_output(item)
        for row_idx, row in enumerate(rows):
            ax = axes[row_idx][col]
            points = arrays.get(row)
            if points is None:
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlim(*xlim)
                ax.set_ylim(*zlim)
                ax.axis("off")
                ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center")
            else:
                scatter_xz(ax, points, xlim, zlim, args)
                ax.text(0.5, -0.03, gt_metric_label(points, arrays["gt"], args), transform=ax.transAxes, ha="center", va="top", fontsize=6.5)
            if row_idx == 0:
                ax.set_title(column_title(args, dps_value), fontsize=10)
            if col == 0:
                ax.text(-0.04, 0.5, row, transform=ax.transAxes, ha="right", va="center", rotation=90, fontsize=9, fontweight="bold")
    os.makedirs(args.plot_output_dir, exist_ok=True)
    out_name = args.plot_output_name or "%s_sweep_grid.png" % args.base_run_name
    fig.suptitle(args.plot_title or args.base_run_name, fontsize=13)
    fig.savefig(os.path.join(args.plot_output_dir, out_name), dpi=args.plot_dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_case(model, dataset, dataset_idx, args, output_root):
    data = dataset[dataset_idx]
    x0 = data["pointcloud"].unsqueeze(0).to(args.device)
    b, obs_meta = make_observation(x0, args)
    x_start, start_t, init_meta = make_x_start(model, b, args)
    z_gt = encode_observation_z(model, x0, sample_encoder=False)
    w_gt = z_to_w(model, z_gt).detach()
    z_enc = encode_observation_z(model, b, sample_encoder=args.encoder_sample)
    w_enc_init = z_to_w(model, z_enc).detach()
    w_refs = OrderedDict(gt=w_gt, obs=w_enc_init)

    # Latent source for the reverse chain (observation / gt / reencode / invert).
    z_for_enc, w_for_wenc = select_latent(model, b, args, z_enc, w_enc_init, z_gt, w_gt)

    selected = args.policies

    enc_results = OrderedDict()   # policy -> (final, trace, x0hat)
    if "enc_no_dps" in selected:
        enc_results["enc_no_dps"] = run_enc_chain(model, b, x_start, start_t, z_for_enc, False, args)
    if "enc_dps" in selected:
        enc_results["enc_dps"] = run_enc_chain(model, b, x_start, start_t, z_for_enc, True, args)

    w_results = OrderedDict()     # policy -> (final, trace, x0hat, w_final, wtrace)
    if "wenc_no_dps" in selected:
        w_results["wenc_no_dps"] = run_w_chain(model, b, x_start, start_t, w_for_wenc, False, args, w_refs=w_refs)
    if "wenc_dps" in selected:
        w_results["wenc_dps"] = run_w_chain(model, b, x_start, start_t, w_for_wenc, True, args, w_refs=w_refs)

    case_name = "single"
    case_dir = output_root
    os.makedirs(case_dir, exist_ok=True)

    arrays = OrderedDict(
        gt=x0.squeeze(0).detach().cpu().numpy(),
        observation=b.squeeze(0).detach().cpu().numpy(),
        x_start=x_start.squeeze(0).detach().cpu().numpy(),
    )
    final_tensors = OrderedDict()
    x0hat_traces = OrderedDict()
    for policy in POLICIES:
        if policy in enc_results:
            final, _, x0hat = enc_results[policy]
        elif policy in w_results:
            final, _, x0hat = w_results[policy][0], w_results[policy][1], w_results[policy][2]
        else:
            continue
        arrays[policy] = final.squeeze(0).detach().cpu().numpy()
        final_tensors[policy] = final
        x0hat_traces[policy] = x0hat

    if args.save_arrays:
        save_arrays(case_dir, arrays, args.save_pcd)
        save_x0hat_trace(case_dir, x0hat_traces, args.save_pcd)

    records = OrderedDict()
    for policy, tensor in final_tensors.items():
        records[policy] = tensor_metrics(tensor, x0, b, args)
        records[policy].update(OrderedDict(
            case=case_name,
            dataset_idx=dataset_idx,
            category=data["cate"],
            mode=policy,
            dps_step_size=args.dps_step_size,
            target_update_ratio=args.target_update_ratio,
            rho_w=args.rho_w,
            lambda_w=args.lambda_w,
            sigma_b=get_sigma_b(args),
        ))

    traces_meta = OrderedDict()
    for policy in POLICIES:
        if policy in enc_results:
            traces_meta[policy] = enc_results[policy][1]
        elif policy in w_results:
            traces_meta[policy] = w_results[policy][1]

    w_stats = OrderedDict(
        gt_norm=float(w_gt.flatten(1).norm(dim=1).mean().detach().cpu().item()),
        obs_enc_norm=float(w_enc_init.flatten(1).norm(dim=1).mean().detach().cpu().item()),
        obs_l2_to_gt=w_l2(w_enc_init, w_gt),
        obs_cos_to_gt=w_cosine(w_enc_init, w_gt),
    )
    if "wenc_no_dps" in w_results:
        w_final_no_dps = w_results["wenc_no_dps"][3]
        w_stats["wenc_no_dps_final_l2_to_gt"] = w_l2(w_final_no_dps, w_gt)
        w_stats["wenc_no_dps_final_cos_to_gt"] = w_cosine(w_final_no_dps, w_gt)
        w_stats["wenc_no_dps_final_l2_from_init"] = w_l2(w_final_no_dps, w_enc_init)
    if "wenc_dps" in w_results:
        w_final_dps = w_results["wenc_dps"][3]
        w_stats["wenc_dps_final_l2_to_gt"] = w_l2(w_final_dps, w_gt)
        w_stats["wenc_dps_final_cos_to_gt"] = w_cosine(w_final_dps, w_gt)
        w_stats["wenc_dps_final_l2_from_init"] = w_l2(w_final_dps, w_enc_init)

    w_trace_meta = OrderedDict()
    for policy in ("wenc_no_dps", "wenc_dps"):
        if policy in w_results:
            w_trace_meta[policy] = w_results[policy][4]

    metadata = OrderedDict(
        dataset_idx=dataset_idx,
        category=data["cate"],
        pointcloud_id=int(data["id"]),
        observation=obs_meta,
        initialization=init_meta,
        reverse_noise_scale=args.reverse_noise_scale,
        dps_schedule=args.dps_schedule,
        x_obs_loss=args.x_obs_loss,
        latent_init=args.latent_init,
        policies=list(selected),
        rho_w=args.rho_w,
        lambda_w=args.lambda_w,
        w_obs_loss=args.w_obs_loss,
        beta_w_target=args.beta_w_target,
        w_update_every=args.w_update_every,
        w_update_steps=args.w_update_steps,
        w_stats=w_stats,
        w_trace=w_trace_meta,
        traces=traces_meta,
        formula=OrderedDict(
            obs_nll="1/(2*sigma_b^2) * ||b - X0_hat||_F^2",
            dps="X_{t-1} = Reverse_theta(X_t,t,z) - eta_t * grad_{X_t} L_X",
            x_obs_loss="mse_nll uses Gaussian NLL; cd uses ChamferDistance; emd uses Sinkhorn OT",
            eta_t="constant family: dps_step_size * dps_schedule(t); ratio: target_update_ratio * ||X_prior-X_t|| / (||grad||+eps)",
            w="w <- w - rho_w * grad_w [L_w_obs(X_t,F(w)) + 0.5*lambda_w*||w||^2]",
            w_obs_loss="none disables w observation loss; mse_nll uses Gaussian NLL; cd uses ChamferDistance; emd uses Sinkhorn OT",
            w_target="optional oracle term: 0.5*beta_w_target*||w-w_gt||^2",
        ),
    )
    if args.save_reports:
        with open(os.path.join(case_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    make_case_plots(case_dir, arrays, x0hat_traces, records, args)
    make_w_trace_plot(case_dir, OrderedDict(
        init=OrderedDict(obs_l2_to_gt=w_l2(w_enc_init, w_gt), obs_cos_to_gt=w_cosine(w_enc_init, w_gt)),
        traces=w_trace_meta,
    ), args)
    return list(records.values()), arrays, metadata


def write_results(output_root, records, metadata, args):
    """Always write numeric experiment results; arrays/PCDs remain optional."""
    os.makedirs(output_root, exist_ok=True)
    result_json = os.path.join(output_root, "result.json")
    result_txt = os.path.join(output_root, "result.txt")

    config = OrderedDict((k, v) for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, list, type(None))))
    payload = OrderedDict(config=config, records=records, metadata=metadata)
    with open(result_json, "w") as f:
        json.dump(payload, f, indent=2)

    with open(result_txt, "w") as f:
        f.write("run_name: %s\n" % args.base_run_name)
        if args.dps_schedule == "ratio":
            f.write("target_update_ratio: %.10g\n" % args.target_update_ratio)
        f.write("eta/dps_step_size: %.10g\n" % args.dps_step_size)
        f.write("eta_min: %s\n" % str(args.eta_min))
        f.write("eta_max: %s\n" % str(args.eta_max))
        f.write("sigma_b: %.10g\n" % get_sigma_b(args))
        f.write("noise_std: %.10g\n" % args.noise_std)
        f.write("t_start: %d\n" % args.t_start)
        f.write("x_init: %s\n" % args.x_init)
        f.write("latent_init: %s\n" % args.latent_init)
        f.write("reverse_noise_scale: %.10g\n" % args.reverse_noise_scale)
        f.write("dps_schedule: %s\n" % args.dps_schedule)
        f.write("x_obs_loss: %s\n" % args.x_obs_loss)
        f.write("emd_epsilon: %.10g\n" % args.emd_epsilon)
        f.write("emd_iters: %d\n" % args.emd_iters)
        f.write("emd_max_points: %s\n" % str(args.emd_max_points))
        f.write("rho_w: %.10g\n" % args.rho_w)
        f.write("lambda_w: %.10g\n" % args.lambda_w)
        f.write("w_obs_loss: %s\n" % args.w_obs_loss)
        f.write("beta_w_target: %.10g\n" % args.beta_w_target)
        f.write("w_update_every: %d\n" % args.w_update_every)
        f.write("w_update_steps: %d\n" % args.w_update_steps)
        enc_dps_trace = metadata.get("traces", {}).get("enc_dps", [])
        ratios = [item.get("update_ratio", 0.0) for item in enc_dps_trace if "update_ratio" in item]
        if ratios:
            f.write("update_ratio_mean: %.10g\n" % (sum(ratios) / len(ratios)))
            f.write("update_ratio_max: %.10g\n" % max(ratios))
        f.write("\n")
        for record in records:
            f.write("[%s]\n" % record["mode"])
            f.write("CD-GT: %.10g\n" % record["cd_to_gt"])
            f.write("EMD-GT: %.10g\n" % record["emd_to_gt"])
            f.write("CD-OBS: %.10g\n" % record["obs_cd"])
            f.write("MSE-OBS: %.10g\n" % record["mse_to_obs"])
            f.write("obs_nll: %.10g\n" % record["obs_nll"])
            f.write("\n")

    # Backward-friendly CSV is useful when quickly scanning multiple folders.
    csv_path = os.path.join(output_root, "result.csv")
    fieldnames = [
        "case", "dataset_idx", "category", "mode", "dps_step_size", "target_update_ratio",
        "rho_w", "lambda_w", "sigma_b", "cd_to_gt", "emd_to_gt", "obs_cd", "mse_to_gt",
        "mse_to_obs", "obs_nll",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fieldnames})


def select_indices(dataset, args):
    if args.indices is not None:
        return args.indices
    end = min(args.start + args.num_shapes, len(dataset))
    return list(range(args.start, end))


def dps_name(value):
    return ("%g" % value).replace("-", "m").replace(".", "p")


def sweep_label(args):
    if getattr(args, "sweep_kind", None) == "t_start":
        return "t"
    if getattr(args, "sweep_kind", None) == "rho_w":
        return "rho"
    return "ratio" if args.dps_schedule == "ratio" else "dps"


def column_title(args, value):
    if getattr(args, "sweep_kind", None) == "t_start":
        return "t_start %s" % str(value)
    if getattr(args, "sweep_kind", None) == "rho_w":
        return "rho_w %.4g" % value
    if args.dps_schedule == "ratio":
        return "ratio %.4g" % value
    return "eta %.4g" % value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="pretrained/GEN_airplane.pt")
    parser.add_argument("--dataset_path", default="data/shapenet.hdf5")
    parser.add_argument("--category", default="airplane")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--scale_mode", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2020)

    parser.add_argument("--noise_std", type=float, default=0.1)
    parser.add_argument("--sigma_b", type=float, default=None)
    parser.add_argument("--x_init", default="observation", choices=["observation", "random"])
    parser.add_argument("--t_start", type=int, default=30)
    parser.add_argument("--t_starts", type=parse_csv_ints, default=None)
    parser.add_argument("--encoder_sample", action="store_true")
    add_latent_init_args(parser)

    parser.add_argument("--dps_step_size", type=float, default=1e-6)  # eta base scale; constant schedule means eta_t equals this value.
    parser.add_argument("--dps_step_sizes", type=parse_csv_floats, default=None)
    parser.add_argument("--dps_schedule", default="constant", choices=["constant", "one_minus_alpha_bar", "sigma2", "x_std", "ratio"])
    parser.add_argument("--x_obs_loss", default="mse_nll", choices=["mse_nll", "cd", "emd", "soft_nll"])
    parser.add_argument("--emd_epsilon", type=float, default=0.03)
    parser.add_argument("--emd_iters", type=int, default=50)
    parser.add_argument("--emd_max_points", type=int, default=0)
    parser.add_argument("--target_update_ratio", type=float, default=1.5)
    parser.add_argument("--target_update_ratios", type=parse_csv_floats, default=None)
    parser.add_argument("--eta_min", type=float, default=None)
    parser.add_argument("--eta_max", type=float, default=None)
    parser.add_argument("--normalize_dps_grad", action="store_true")
    parser.add_argument("--dps_grad_eps", type=float, default=1e-8)
    parser.add_argument("--reverse_noise_scale", type=float, default=1.0)
    parser.add_argument("--flexibility", type=float, default=None)

    # Latent posterior parameters for wenc_* policies.
    # rho_w is the gradient descent step size in w <- w - rho_w * grad_w L_w.
    # lambda_w multiplies the standard-normal prior term 0.5 * lambda_w * ||w||^2.
    parser.add_argument("--rho_w", type=float, default=1e-7)
    parser.add_argument("--rho_ws", type=parse_csv_floats, default=None)
    parser.add_argument("--lambda_w", type=float, default=1.0)
    parser.add_argument("--w_obs_loss", default="mse_nll", choices=["none", "mse_nll", "cd", "emd"])
    parser.add_argument("--beta_w_target", type=float, default=0.0)
    parser.add_argument("--w_update_every", type=int, default=1)
    parser.add_argument("--w_update_steps", type=int, default=1)

    parser.add_argument("--policies", type=parse_policies, default=list(POLICIES),
                        help="Comma-separated subset of %s. e.g. enc_no_dps,enc_dps to skip wenc." % (POLICIES,))
    parser.add_argument("--indices", type=parse_csv_ints, default=[0])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num_shapes", type=int, default=1)
    parser.add_argument("--trace_every", type=int, default=5)

    parser.add_argument("--output_dir", default="output/inverse_experiments")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--save_arrays", action="store_true")
    parser.add_argument("--save_pcd", action="store_true")
    parser.add_argument("--save_reports", action="store_true")  # Deprecated: result.txt/json/csv are always saved.
    parser.add_argument("--make_plot", action="store_true", default=True,
                        help="On by default: per-case plots (final_grid, x0hat_trace, w_trace).")
    parser.add_argument("--make_sweep_plot", action="store_true", default=False,
                        help="Opt-in: cross-run sweep grids. Off by default (each panel re-computes CD/EMD labels).")
    parser.add_argument("--plot_output_dir", default="output/figures")
    parser.add_argument("--plot_output_name", default=None)
    parser.add_argument("--plot_title", default=None)
    parser.add_argument("--plot_dpi", type=int, default=180)
    parser.add_argument("--plot_point_size", type=float, default=1.2)
    parser.add_argument("--plot_alpha", type=float, default=0.9)
    parser.add_argument("--plot_color", default="#1f77b4")
    args = parser.parse_args()

    seed_all(args.seed)
    model, ckpt = load_model(args)
    scale_mode = args.scale_mode or ckpt["args"].scale_mode
    dataset = ShapeNetCore(path=args.dataset_path, cates=[args.category], split=args.split, scale_mode=scale_mode)
    indices = select_indices(dataset, args)[:1]

    base_run_name = args.run_name or "enc_dps_%s_%d" % (args.category, int(time.time()))
    args.base_run_name = base_run_name
    if args.t_starts is not None:
        args.sweep_kind = "t_start"
        sweep_values = args.t_starts
    elif args.rho_ws is not None:
        args.sweep_kind = "rho_w"
        sweep_values = args.rho_ws
    elif args.dps_schedule == "ratio":
        args.sweep_kind = "ratio"
        sweep_values = args.target_update_ratios if args.target_update_ratios is not None else [args.target_update_ratio]
    else:
        args.sweep_kind = "eta"
        sweep_values = args.dps_step_sizes if args.dps_step_sizes is not None else [args.dps_step_size]

    run_dirs = []
    run_outputs = []
    for sweep_value in sweep_values:
        seed_all(args.seed)
        if args.sweep_kind == "t_start":
            args.t_start = sweep_value
        elif args.sweep_kind == "rho_w":
            args.rho_w = sweep_value
        elif args.dps_schedule == "ratio":
            args.target_update_ratio = sweep_value
        else:
            args.dps_step_size = sweep_value
        if len(sweep_values) == 1:
            run_name = base_run_name
        else:
            run_name = "%s_%s_%s" % (base_run_name, sweep_label(args), dps_name(sweep_value))
        output_root = os.path.join(args.output_dir, run_name)
        os.makedirs(output_root, exist_ok=True)
        run_dirs.append((sweep_value, output_root))

        all_records = []
        dataset_idx = indices[0]
        args.case_order = 0
        case_records, arrays, metadata = run_case(model, dataset, dataset_idx, args, output_root)
        all_records.extend(case_records)
        run_outputs.append((sweep_value, arrays, metadata))
        write_results(output_root, all_records, metadata, args)

        print("Wrote %s" % output_root)
        print("Result: %s" % os.path.join(output_root, "result.txt"))

    make_sweep_plot(run_outputs, args)
    make_w_sweep_plot(run_outputs, args)


if __name__ == "__main__":
    main()
