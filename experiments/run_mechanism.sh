#!/usr/bin/env bash
# Experiment 2 (mechanism, paper Fig. 3): "is the latent representative?"
# At every --log_stride reverse steps we decode the current latent INDEPENDENTLY to
# completion (a full latent-conditioned generation, not the running reverse state) and
# measure its CD-to-GT. All curves seed from the encoder latent w_obs:
#   ours             per-step re-solve (Kw steps each step)
#   oneshot_kw       Kw steps once at t'  (= ours' first step, coincides at step 1)
#   oneshot_matched  Kw*t_start steps once at t'  (budget-matched, = the One-shot baseline)
# No lambda/beta knobs. Ours' latent improves along the chain; the frozen one-shots stay flat.
#
# Output: output/mechanism/{result.json, figures/exp10_latent_vs_step_noise*.png}
# Paper figure: python experiments/plot_mechanism.py --exp_dir output/mechanism
#
# NOTE: heavy (many full decodes). Lower --num_shapes or raise --log_stride for a quick look.
# Run from src/.
set -euo pipefail
cd "$(dirname "$0")/.."

# Defaults match the main-results run (airplane, Kw=25). Override via env vars.
CATEGORY="${CATEGORY:-airplane}"
CKPT="${CKPT:-pretrained/GEN_airplane.pt}"
NOISES="${NOISES:-0.2,0.3}"
RATIOS="${RATIOS:-0.3,0.3}"
NUM_SHAPES="${NUM_SHAPES:-10}"
KW="${KW:-25}"
EXP_DIR="${EXP_DIR:-output/mechanism}"
rm -rf "${EXP_DIR}"
mkdir -p "${EXP_DIR}"

python experiments/run_mechanism.py \
  --ckpt "${CKPT}" \
  --dataset_path data/shapenet.hdf5 \
  --category "${CATEGORY}" \
  --noises "${NOISES}" \
  --ratios "${RATIOS}" \
  --dps_loss soft_nll \
  --num_shapes "${NUM_SHAPES}" \
  --t_start 30 \
  --reverse_noise_scale 0 \
  --invert_lr 0.02 \
  --ours_inner_steps "${KW}" \
  --log_stride 3 \
  --output_dir "${EXP_DIR}"

echo "DONE mechanism (${CATEGORY}, Kw=${KW}) -> ${EXP_DIR}"
echo "Figure (Fig. 3): python experiments/plot_mechanism.py --exp_dir ${EXP_DIR}"
