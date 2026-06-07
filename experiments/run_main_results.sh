#!/usr/bin/env bash
# Experiment 1 (main results, paper Table 1 + Fig. 2): Latent-MAP DPS vs baselines
# for airplane Gaussian denoising, multi-shape, noise {0.1,0.2,0.3}.
#
# One-shot and Ours share the IDENTICAL latent objective (no lambda/beta knobs):
#   J_t(w) = softNLL(b; x0hat(X_t, t, F(w)))  +  0.5 * || w - mu_t ||^2
#   mu_t   = sg[ F^{-1}(E_phi(x0hat)) ]      (encoder-consistency anchor; beta = 1)
# They differ only in WHEN the optimization steps are spent.
#
# Policies (--policies, default 'all'):
#   enc_no_dps  [Encoder]        z = E_phi(b), reverse chain, no DPS
#   enc_dps     [Encoder+DPS]    z = E_phi(b), reverse chain + DPS
#   w_inv       [One-shot]       solve J once at t' (Kw*t_start steps), z = F(w) frozen, no DPS
#   winv_dps    [One-shot+DPS]   one-shot latent frozen + DPS
#   ours_legacy [Ours (prior)]   re-solve every step with origin prior (mu -> 0), then DPS
#   ours        [Ours]           re-solve J_t every step (encoder anchor), then DPS
#
# All policies seed from the encoder latent w_obs. The One-shot baseline solves J once at
# t' for Kw*t_start steps (= Ours' TOTAL per-step budget, so it is budget-matched); Ours
# re-solves J at every reverse step (Kw steps each).
#
# Output: output/main_results/{result.json, table.csv, policies/<key>/..., figures/}
# Then render the qualitative grid (paper Fig. 2):
#   python experiments/plot_main_grid.py --exp_dir output/main_results --policies ours,winv_dps,enc_dps
#
# Run from src/ (the parent of this script's directory).
set -euo pipefail
cd "$(dirname "$0")/.."

# Defaults reproduce the paper (airplane, Kw=25, 10-shape subset). Override via env vars:
#   NUM_SHAPES=0 ...                              # 0 = full airplane test set
#   CATEGORY=chair CKPT=pretrained/GEN_chair.pt EXP_DIR=output/main_results_chair \
#     bash experiments/run_main_results.sh
CATEGORY="${CATEGORY:-airplane}"
CKPT="${CKPT:-pretrained/GEN_airplane.pt}"
NOISES="${NOISES:-0.1,0.2,0.3}"
RATIOS="${RATIOS:-0.5,0.3,0.3}"
NUM_SHAPES="${NUM_SHAPES:-10}"
KW="${KW:-25}"
T_START="${T_START:-30}"
INVERT_ITERS="${INVERT_ITERS:-$((KW * T_START))}"
EXP_DIR="${EXP_DIR:-output/main_results}"
rm -rf "${EXP_DIR}"
mkdir -p "${EXP_DIR}"

python experiments/run_main_results.py \
  --ckpt "${CKPT}" \
  --dataset_path data/shapenet.hdf5 \
  --category "${CATEGORY}" \
  --policies all \
  --noises "${NOISES}" \
  --ratios "${RATIOS}" \
  --dps_loss soft_nll \
  --num_shapes "${NUM_SHAPES}" \
  --t_start "${T_START}" \
  --reverse_noise_scale 0 \
  --invert_iters "${INVERT_ITERS}" \
  --invert_lr 0.02 \
  --ours_inner_steps "${KW}" \
  --output_dir "${EXP_DIR}"

echo "DONE main results (${CATEGORY}, Kw=${KW}) -> ${EXP_DIR}"
echo "Grid (Fig. 2): python experiments/plot_main_grid.py --exp_dir ${EXP_DIR} --policies ours,winv_dps,enc_dps"
