# Latent-MAP DPS

**Latent-MAP Posterior Sampling for Training-Free Point-Cloud Inverse Problems**

Training-free posterior sampling for point-cloud inverse problems under a
**frozen, latent-conditioned** point-cloud diffusion prior. Such priors expose
only a *conditional* score for a fixed global shape latent $w$, not the
marginal score that diffusion posterior sampling (DPS) assumes. Latent-MAP DPS
derives the latent-marginal posterior score, approximates its time-dependent
latent posterior by a **per-step MAP estimate** (anchored to the prior's own
encoder, no added learned parameters), and combines it with DPS guidance under
an SDEdit-style initialization. The prior is never retrained.

At every reverse step the shape latent $w$ is re-solved on

```math
\begin{aligned}
J_t(w)
&= \mathrm{softNLL}(b; \hat{x}_0(X_t,t,F_\alpha(w)))
   + \frac{1}{2}\|w - \mu_t\|_2^2 \\
\mu_t
&= \mathrm{sg}[F_\alpha^{-1}(E_\phi(\hat{x}_0))]
   \quad \text{(encoder-consistency anchor)}
\end{aligned}
```

and the points are updated with the resulting measurement-guided posterior
score. Evaluated here on ShapeNet airplane Gaussian denoising.

> Built on the latent point-cloud diffusion prior of Luo & Hu, *Diffusion
> Probabilistic Models for 3D Point Cloud Generation* (CVPR 2021),
> <https://github.com/luost26/diffusion-point-cloud>. The model code under
> `models/` and the data loader under `utils/` derive from that project; see
> `LICENSE`.

## Repository layout

```
src/
├── models/                 # frozen latent point-cloud diffusion prior (FlowVAE)
│   ├── vae_flow.py         #   encoder, flow, latent-cond. diffusion
│   ├── diffusion.py  flow.py  common.py
│   └── encoders/           #   PointNet encoder
├── utils/
│   ├── dataset.py          # ShapeNetCore loader (HDF5)
│   └── misc.py             # seed_all, helpers
├── experiments/
│   ├── core.py             # shared sampler library (model loading, DPS step,
│   │                       #   Tweedie estimate, soft-NLL, encode/decode helpers, metrics)
│   ├── policies.py         # baseline/ablation registry (Encoder, One-shot, Ours, ...)
│   ├── paper_style.py      # matplotlib style for the paper figures
│   ├── run_main_results.py # Experiment 1: Table 1 (all policies, all noise levels)
│   ├── run_main_results.sh
│   ├── run_mechanism.py    # Experiment 2: Fig. 2 (latent-quality vs reverse step)
│   ├── run_mechanism.sh
│   ├── plot_main_grid.py   # renders the qualitative grid (Fig. 1)
│   └── plot_mechanism.py   # renders the mechanism plot (Fig. 2)
├── pretrained/             # place GEN_airplane.pt here (not tracked)
├── data/                   # place shapenet.hdf5 here (not tracked)
├── output/                 # experiment outputs (created at run time, not tracked)
├── env.yml                 # conda environment
└── LICENSE
```

## Setup

```bash
conda env create -f env.yml      # creates env "dpm-pc-gen" (PyTorch + h5py + matplotlib)
conda activate dpm-pc-gen
```

Download the pretrained generator and the ShapeNet HDF5 from the base project's
drive (<https://drive.google.com/drive/folders/1Su0hCuGFo1AGrNb_VMNnlF7qeQwKjfhZ>)
and place them as:

```
src/pretrained/GEN_airplane.pt
src/data/shapenet.hdf5
```

All commands below are run **from `src/`**.

## Reproduce the paper

### Experiment 1 — main results (Table 1 + Fig. 1)

```bash
bash experiments/run_main_results.sh
# -> output/main_results/{table.csv, result.json, policies/<key>/..., figures/}
```

Runs the ablation ladder along two axes (how the latent is obtained × DPS
on/off) plus our method:

| key           | label          | latent                                             | DPS |
|---------------|----------------|----------------------------------------------------|-----|
| `enc_no_dps`  | Encoder        | $w = F_\alpha^{-1}(E_\phi(b))$, frozen             | no  |
| `enc_dps`     | Encoder+DPS    | encoder latent, frozen                             | yes |
| `oneshot`       | One-shot       | solve $J_t$ once at $t'$ ($K_w t'$ steps), frozen  | no  |
| `oneshot_dps`    | One-shot+DPS   | one-shot latent, frozen                            | yes |
| `ours`        | **Ours**       | **re-solve $J_t$ at every reverse step** ($K_w$ each) | yes |

All policies seed from the encoder latent $w_{\mathrm{obs}}$. **One-shot is
budget-matched to Ours**: it spends the same total number of optimization steps
$K_w t'$, but all at once at $t'$ and then freezes — so the One-shot+DPS vs
Ours gap isolates *when* the latent is optimized.

Then render the qualitative grid (Fig. 1):

```bash
python experiments/plot_main_grid.py --exp_dir output/main_results --policies ours,oneshot_dps,enc_dps
```

Defaults reproduce the paper: airplane, $K_w=25$, $t'=30$, noise $\{0.1,0.2,0.3\}$,
**10-shape subset**. For the full airplane test set:

```bash
NUM_SHAPES=0 bash experiments/run_main_results.sh
```

### Experiment 2 — mechanism (Fig. 2)

```bash
bash experiments/run_mechanism.sh
python experiments/plot_mechanism.py --exp_dir output/mechanism
# -> output/mechanism/{result.json, figures/exp10_latent_vs_step_noise*.png}
```

At every few reverse steps the current latent is **decoded independently to
completion** (a full latent-conditioned generation, not the running reverse
state) and scored by CD-to-GT — i.e. how representative the latent is. Ours'
latent starts worse but improves along the chain and crosses the frozen
one-shot latents.

## Key knobs

| flag                  | meaning                                            | default |
|-----------------------|----------------------------------------------------|---------|
| `--ours_inner_steps`  | $K_w$, latent gradient steps per reverse step      | 25      |
| `--t_start`           | $t'$, SDEdit start step (of $T=100$)               | 30      |
| `--invert_iters`      | one-shot budget (set to $K_w t'$ to budget-match)  | 750     |
| `--noises`            | observation noise levels $\sigma_b$                | 0.1,0.2,0.3 |
| `--num_shapes`        | test shapes (`0` = full set)                       | 10      |

The latent objective is parameter-free: the soft-NLL already carries $\sigma_b$
and the latent posterior is unit-variance, so the two terms combine with weight
1 (no $\lambda/\beta$). The reverse-step DPS guidance keeps a step size $\zeta_t$
(the `--ratios` schedule), the standard DPS hyperparameter.

## Citation

If you use the underlying prior, please cite Luo & Hu (CVPR 2021). This
repository implements Latent-MAP DPS on top of it.
