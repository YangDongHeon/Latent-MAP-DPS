"""Shared policy registry for exp8 (OURS vs baselines).

Deliberately has NO heavy dependencies (no torch) so BOTH the runner
(run_main_results.py, needs torch/docker) and the plot script (plot_main_grid.py,
numpy+matplotlib only, runnable on the host) can import it.

All latent policies start from the SAME encoder latent w_obs=E_phi(b) and use the
IDENTICAL latent-MAP objective (no lambda/beta knobs)
    J_t(w) = softNLL(b; x0hat(X_t,t,F(w))) + 0.5*||w - mu_t||^2,
    mu_t = sg[F^{-1}(E_phi(x0hat(X_t,t,F(w_warm))))]   (encoder-consistency anchor).
They differ ONLY in HOW J is used (this is the paper's Encoder / One-shot / Ours split):
    Encoder   : no latent optimization (w = w_obs, frozen).
    One-shot : solve J ONCE at t' (invert_iters steps), then freeze.
    Ours      : re-solve J at EVERY reverse step (Kw warm-started steps), w tracks X_t.
So Ours and One-shot both start at w_obs and optimize the same J; the ONLY difference
is one-shot-at-t' vs per-step. Ours does NOT get the one-shot for free.

Each policy entry:
    label   : paper-figure column title
    start   : initial latent -- always 'encoder' (w_obs); 'one-shot' marks the
              one-shot t' solve baselines (oneshot = invert_iters-step solve from w_obs)
    refine  : per-step latent update -- None (freeze) | 'anchor' (re-solve J_t with
              the encoder anchor mu_t) | 'legacy' (re-solve with the ORIGIN prior
              mu_t->0, same weight -- an ablation of the anchor CENTER)
    dps     : whether DPS guidance is applied to X along the chain
    color   : line color for the summary plot
"""
from collections import OrderedDict

# Colors mirror paper_style.POLICY_COLORS (refined palette: muted baselines -> blue hero).
POLICY_REGISTRY = OrderedDict([
    ("enc_no_dps",  dict(label="Encoder",        start="encoder",   refine=None,     dps=False, color="#B8C0CC")),
    ("enc_dps",     dict(label="Encoder+DPS",    start="encoder",   refine=None,     dps=True,  color="#6BAED6")),
    ("oneshot",       dict(label="One-shot",       start="oneshot", refine=None,     dps=False, color="#F4A259")),
    ("oneshot_dps",    dict(label="One-shot+DPS",   start="oneshot", refine=None,     dps=True,  color="#E07B39")),
    ("ours_legacy", dict(label="Ours (prior)",   start="encoder",   refine="legacy", dps=True,  color="#9E78B5")),
    ("ours",        dict(label="Ours",           start="encoder",   refine="anchor", dps=True,  color="#2166AC")),
])

ALL_POLICIES = list(POLICY_REGISTRY.keys())


def parse_policy_list(value):
    """'all'/empty -> every policy; CSV -> that subset (returned in registry order)."""
    if value is None or str(value).strip() == "" or str(value).strip().lower() == "all":
        return list(ALL_POLICIES)
    items = [s.strip() for s in str(value).split(",") if s.strip()]
    bad = [s for s in items if s not in POLICY_REGISTRY]
    if bad:
        raise ValueError("Unknown policy %s; choose from %s" % (bad, ALL_POLICIES))
    return [k for k in ALL_POLICIES if k in items]   # keep canonical order


def policy_label(key):
    return POLICY_REGISTRY[key]["label"]


def policy_color(key):
    return POLICY_REGISTRY[key]["color"]


def needs_oneshot(keys):
    return any(POLICY_REGISTRY[k]["start"] == "oneshot" for k in keys)


def noise_tag(noise):
    """Filesystem-safe noise folder tag, e.g. 0.1 -> 'noise0p1'."""
    return "noise" + ("%g" % noise).replace(".", "p").replace("-", "m")
