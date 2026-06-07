"""Shared paper-figure styling: refined palette + Matplotlib rcParams (torch-free).

Import `apply()` and the palette in the plot scripts so every figure shares ONE look.
Restyle all figures at once by editing the palette / rcParams here.
"""
import matplotlib

# ColorBrewer-inspired, print-friendly. The cool deep-blue 'Ours' hero stands out
# against muted/warm baselines; readable under common color-vision deficiencies.
PALETTE = {
    "ours":     "#2166AC",   # deep blue  -- Ours hero (exp10 line; exp8 point clouds)
    "baseline": "#98A2B3",   # cool gray  -- receding one-shot baseline (exp10)
    "ink":      "#1A1A1A",
    "grid":     "#CBD2D9",
    "points":   "#2166AC",   # exp8 point-cloud color
}

# exp8 per-policy line colors (muted baselines -> blue hero). Keep in sync with
# policies.POLICY_REGISTRY.
POLICY_COLORS = {
    "enc_no_dps":  "#B8C0CC",
    "enc_dps":     "#6BAED6",
    "oneshot":       "#F4A259",
    "oneshot_dps":    "#E07B39",
    "ours_legacy": "#9E78B5",
    "ours":        "#2166AC",
}

PAPER_RC = {
    "font.family": "STIXGeneral",       # bundled with matplotlib; Times-like serif (IEEE)
    "mathtext.fontset": "stix",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.linewidth": 0.7, "axes.edgecolor": "#3A3A3A",
    "axes.labelcolor": "#1A1A1A", "text.color": "#1A1A1A",
    "xtick.color": "#3A3A3A", "ytick.color": "#3A3A3A",
    "xtick.direction": "out", "ytick.direction": "out",
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
    "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#CBD2D9", "grid.linewidth": 0.6, "grid.alpha": 0.7,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
}


def apply():
    matplotlib.rcParams.update(PAPER_RC)


def policy_color(key, default="#666666"):
    return POLICY_COLORS.get(key, default)
