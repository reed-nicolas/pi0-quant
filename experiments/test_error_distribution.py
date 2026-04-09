"""
Quick test: what does the FP8 quantization error distribution actually look like?

Part 1 — Element-wise quant error by input distribution
Part 2 — After matmul propagation (does CLT kick in?)
Part 3 — Chained matmuls (error accumulation through layers)
Part 4 — RMSE growth vs. depth
Part 5 — Goodness-of-fit summary: Gaussian vs Laplace (KS statistic + log-likelihood)

Standalone — only needs torch + numpy + matplotlib + scipy. No project imports.
Reproduces the quant_fp8_po2 logic from pi0_inout/quant_types.py inline.
"""

import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats as sp_stats


# -- Inline FP8 E4M3 po2 quant (copied from pi0_inout/quant_types.py) --------

FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max      # 448.0
FP8_E4M3_MAX_PO2 = 2 ** math.floor(math.log2(FP8_E4M3_MAX))  # 256.0


def quant_fp8_po2(x: torch.Tensor) -> torch.Tensor:
    x_f32 = x.float()
    amax = x_f32.abs().max()
    if amax == 0:
        return x_f32
    raw_scale = amax / FP8_E4M3_MAX_PO2
    scale = 2.0 ** math.floor(math.log2(raw_scale.item()))
    x_scaled = (x_f32 / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    x_q = x_scaled.to(torch.float8_e4m3fn).to(torch.float32)
    return (x_q * scale).to(x.dtype)


def quant_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor = None) -> torch.Tensor:
    """Quantized linear: quant inputs, matmul in float, quant output."""
    x_q = quant_fp8_po2(x)
    w_q = quant_fp8_po2(w)
    y = x_q.float() @ w_q.float().T
    if b is not None:
        y = y + b.float()
    return quant_fp8_po2(y.to(torch.bfloat16))


def ideal_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor = None) -> torch.Tensor:
    """Ideal linear: full-precision matmul."""
    y = x.float() @ w.float().T
    if b is not None:
        y = y + b.float()
    return y.to(torch.bfloat16)


# -- Goodness-of-fit helper ---------------------------------------------------

def gof_metrics(data: np.ndarray) -> dict:
    """
    Compute goodness-of-fit metrics for Gaussian and Laplace fits.
    Returns dict with KS stats, p-values, and mean log-likelihoods.
    """
    mu, sigma = data.mean(), data.std()
    loc_lap, scale_lap = sp_stats.laplace.fit(data)

    # KS test: lower statistic = better fit
    ks_gauss, p_gauss = sp_stats.kstest(data, 'norm', args=(mu, sigma))
    ks_lap, p_lap = sp_stats.kstest(data, 'laplace', args=(loc_lap, scale_lap))

    # Mean log-likelihood: higher = better fit
    ll_gauss = sp_stats.norm.logpdf(data, mu, sigma).mean()
    ll_lap = sp_stats.laplace.logpdf(data, loc_lap, scale_lap).mean()

    return {
        "gauss_mu": mu, "gauss_sigma": sigma,
        "lap_loc": loc_lap, "lap_scale": scale_lap,
        "ks_gauss": ks_gauss, "p_gauss": p_gauss,
        "ks_lap": ks_lap, "p_lap": p_lap,
        "ll_gauss": ll_gauss, "ll_lap": ll_lap,
        "skew": sp_stats.skew(data),
        "kurt": sp_stats.kurtosis(data),
        "winner": "Laplace" if ks_lap < ks_gauss else "Gaussian",
    }


def plot_with_gof(ax, rel_errors, title, color="steelblue"):
    """Plot histogram with Gaussian/Laplace overlays and GOF annotation."""
    m = gof_metrics(rel_errors)

    ax.hist(rel_errors, bins=200, density=True, alpha=0.7, color=color,
            edgecolor="none", label="Empirical")

    x_fit = np.linspace(
        m["gauss_mu"] - 4 * m["gauss_sigma"],
        m["gauss_mu"] + 4 * m["gauss_sigma"],
        300,
    )
    ax.plot(x_fit, sp_stats.norm.pdf(x_fit, m["gauss_mu"], m["gauss_sigma"]),
            'r-', lw=1.5,
            label=f"Gaussian  KS={m['ks_gauss']:.3f}  LL={m['ll_gauss']:.2f}")
    ax.plot(x_fit, sp_stats.laplace.pdf(x_fit, m["lap_loc"], m["lap_scale"]),
            'g--', lw=1.5,
            label=f"Laplace   KS={m['ks_lap']:.3f}  LL={m['ll_lap']:.2f}")

    winner_color = "green" if m["winner"] == "Laplace" else "red"
    ax.set_title(f"{title}\nskew={m['skew']:.3f}  kurt={m['kurt']:.3f}  "
                 f"better fit: {m['winner']}",
                 fontsize=9)
    # Color the "better fit" portion
    ax.title.set_color("black")
    ax.legend(fontsize=6.5)
    ax.set_xlabel("Relative error")
    return m


# ============================================================================
# PART 1 — Element-wise quantization error by input distribution
# ============================================================================

def collect_element_errors(shape, input_dist, n_trials=200, eps=1e-12):
    all_rel_errors = []
    for _ in range(n_trials):
        x = input_dist(shape)
        x_q = quant_fp8_po2(x)
        rel_err = ((x_q - x).float() / (x.float().abs() + eps)).flatten()
        all_rel_errors.append(rel_err)
    return torch.cat(all_rel_errors).numpy()


input_dists = {
    "Normal(0,1)":         lambda s: torch.randn(*s, dtype=torch.bfloat16),
    "Uniform(-1,1)":       lambda s: (2 * torch.rand(*s) - 1).to(torch.bfloat16),
    "Normal(0,0.01)":      lambda s: (0.01 * torch.randn(*s)).to(torch.bfloat16),
    "Heavy-tail (t, df=3)":lambda s: torch.tensor(
                                np.random.default_rng(42).standard_t(3, size=s),
                                dtype=torch.bfloat16),
    "Realistic activations\n(ReLU of Normal)": lambda s: torch.randn(*s, dtype=torch.bfloat16).clamp(min=0),
}

shape = (256, 256)
n_trials = 500

# Collect all GOF results for summary table
all_gof: list[tuple[str, str, dict]] = []  # (part, label, metrics)

print("=" * 60)
print("PART 1: Element-wise quantization error")
print("=" * 60)

fig1, axes1 = plt.subplots(2, 3, figsize=(15, 9))
axes1 = axes1.flatten()

for idx, (name, dist_fn) in enumerate(input_dists.items()):
    print(f"  Collecting: {name}...")
    rel_errors = collect_element_errors(shape, dist_fn, n_trials=n_trials)
    rel_errors = rel_errors[np.abs(rel_errors) < 1.0]
    m = plot_with_gof(axes1[idx], rel_errors, name, color="steelblue")
    all_gof.append(("P1:element", name.replace("\n", " "), m))

ax = axes1[5]
ax.axis("off")
ax.text(0.05, 0.5,
    "Goodness-of-fit metrics:\n\n"
    "  KS = Kolmogorov-Smirnov statistic\n"
    "        (lower = better fit)\n\n"
    "  LL = mean log-likelihood\n"
    "        (higher = better fit)\n\n"
    "  kurt = excess kurtosis\n"
    "    Gaussian:  0\n"
    "    Laplace:   3\n"
    "    Uniform:  -1.2",
    fontsize=11, verticalalignment="center", family="monospace",
    transform=ax.transAxes)

fig1.suptitle("Part 1: FP8 E4M3 (po2) \u2014 Element-wise Relative Error by Input Type",
              fontsize=13, fontweight="bold")
fig1.tight_layout()
fig1.savefig("experiments/results/part1_element_error.png", dpi=150, bbox_inches="tight")
print("  Saved part1_element_error.png")


# ============================================================================
# PART 2 — After matmul: per-output-element error (does CLT help?)
# ============================================================================

print()
print("=" * 60)
print("PART 2: Error after single matmul (CLT test)")
print("=" * 60)

dims = [32, 128, 256, 512, 1024]
n_trials_mm = 500

fig2, axes2 = plt.subplots(2, 3, figsize=(15, 9))
axes2 = axes2.flatten()

for idx, d in enumerate(dims):
    print(f"  dim={d}...")
    all_rel_errors = []
    for _ in range(n_trials_mm):
        x = torch.randn(1, d, dtype=torch.bfloat16)
        w = torch.randn(d, d, dtype=torch.bfloat16) / math.sqrt(d)
        y_ideal = ideal_linear(x, w)
        y_quant = quant_linear(x, w)
        rel_err = ((y_quant - y_ideal).float() / (y_ideal.float().abs() + 1e-12)).flatten()
        all_rel_errors.append(rel_err)
    rel_errors = torch.cat(all_rel_errors).numpy()
    rel_errors = rel_errors[np.abs(rel_errors) < 2.0]
    m = plot_with_gof(axes2[idx], rel_errors, f"matmul dim={d}", color="coral")
    all_gof.append(("P2:matmul", f"dim={d}", m))

ax = axes2[5]
ax.axis("off")
ax.text(0.05, 0.5,
    "CLT prediction:\n\n"
    "  Each output = sum of d products.\n"
    "  As d grows, expect Gaussian (CLT).\n\n"
    "  Watch: does KS(Gaussian) improve\n"
    "  with dimension?\n\n"
    "  If KS(Laplace) < KS(Gaussian)\n"
    "  at all dims: CLT is NOT rescuing\n"
    "  the Gaussian assumption.\n"
    "  (shared po2 scaling breaks\n"
    "  the independence CLT needs)",
    fontsize=11, verticalalignment="center", family="monospace",
    transform=ax.transAxes)

fig2.suptitle("Part 2: FP8 E4M3 (po2) \u2014 Matmul Output Error vs. Dimension (CLT test)",
              fontsize=13, fontweight="bold")
fig2.tight_layout()
fig2.savefig("experiments/results/part2_matmul_clt.png", dpi=150, bbox_inches="tight")
print("  Saved part2_matmul_clt.png")


# ============================================================================
# PART 3 — Chained matmuls (error accumulation through layers)
# ============================================================================

print()
print("=" * 60)
print("PART 3: Error accumulation through chained matmuls")
print("=" * 60)

d = 256
n_layers_list = [1, 2, 4, 8, 16, 32]
n_trials_chain = 300

fig3, axes3 = plt.subplots(2, 3, figsize=(15, 9))
axes3 = axes3.flatten()

torch.manual_seed(42)
max_layers = max(n_layers_list)
weights = [torch.randn(d, d, dtype=torch.bfloat16) / math.sqrt(d) for _ in range(max_layers)]

for idx, n_layers in enumerate(n_layers_list):
    print(f"  layers={n_layers}...")
    all_rel_errors = []
    all_rmse = []
    for _ in range(n_trials_chain):
        x = torch.randn(1, d, dtype=torch.bfloat16)
        y_ideal = x
        y_quant = x
        for L in range(n_layers):
            y_ideal = ideal_linear(y_ideal, weights[L])
            y_quant = quant_linear(y_quant, weights[L])

        diff = (y_quant - y_ideal).float()
        ref = y_ideal.float()
        rel_err = (diff / (ref.abs() + 1e-12)).flatten()
        all_rel_errors.append(rel_err)
        rmse = diff.pow(2).mean().sqrt().item()
        ref_rms = ref.pow(2).mean().sqrt().item()
        all_rmse.append(rmse / (ref_rms + 1e-12))

    rel_errors = torch.cat(all_rel_errors).numpy()
    rel_errors = rel_errors[np.abs(rel_errors) < 5.0]
    mean_rel_rmse = np.mean(all_rmse)

    m = plot_with_gof(axes3[idx], rel_errors,
                      f"{n_layers} layers (dim={d})  rel_rmse={mean_rel_rmse:.4f}",
                      color="mediumpurple")
    all_gof.append(("P3:chain", f"{n_layers} layers", m))

fig3.suptitle("Part 3: FP8 E4M3 (po2) \u2014 Error Accumulation Through Chained Matmuls",
              fontsize=13, fontweight="bold")
fig3.tight_layout()
fig3.savefig("experiments/results/part3_chain_accumulation.png", dpi=150, bbox_inches="tight")
print("  Saved part3_chain_accumulation.png")


# ============================================================================
# PART 4 — Summary: rel RMSE growth vs. depth
# ============================================================================

print()
print("=" * 60)
print("PART 4: RMSE growth curve vs. depth")
print("=" * 60)

depths = list(range(1, 33))
n_trials_curve = 200
mean_rel_rmses = []
p95_rel_rmses = []
p99_rel_rmses = []

torch.manual_seed(42)
weights_curve = [torch.randn(d, d, dtype=torch.bfloat16) / math.sqrt(d) for _ in range(max(depths))]

for depth in depths:
    rel_rmses = []
    for _ in range(n_trials_curve):
        x = torch.randn(1, d, dtype=torch.bfloat16)
        y_ideal = x
        y_quant = x
        for L in range(depth):
            y_ideal = ideal_linear(y_ideal, weights_curve[L])
            y_quant = quant_linear(y_quant, weights_curve[L])
        diff = (y_quant - y_ideal).float()
        ref = y_ideal.float()
        rmse = diff.pow(2).mean().sqrt().item()
        ref_rms = ref.pow(2).mean().sqrt().item()
        rel_rmses.append(rmse / (ref_rms + 1e-12))
    mean_rel_rmses.append(np.mean(rel_rmses))
    p95_rel_rmses.append(np.percentile(rel_rmses, 95))
    p99_rel_rmses.append(np.percentile(rel_rmses, 99))
    if depth % 8 == 0:
        print(f"  depth={depth:2d}  mean_rel_rmse={mean_rel_rmses[-1]:.6f}  "
              f"p95={p95_rel_rmses[-1]:.6f}  p99={p99_rel_rmses[-1]:.6f}")

fig4, ax4 = plt.subplots(1, 1, figsize=(10, 6))
ax4.plot(depths, mean_rel_rmses, 'b-o', markersize=4, label="Mean rel RMSE")
ax4.fill_between(depths, mean_rel_rmses, p95_rel_rmses, alpha=0.2, color="orange", label="P95 band")
ax4.fill_between(depths, p95_rel_rmses, p99_rel_rmses, alpha=0.2, color="red", label="P99 band")
ax4.plot(depths, p95_rel_rmses, 'orange', lw=1, alpha=0.7)
ax4.plot(depths, p99_rel_rmses, 'r--', lw=1, alpha=0.7)
ref_sqrt = mean_rel_rmses[0] * np.sqrt(depths)
ax4.plot(depths, ref_sqrt, 'k:', lw=1.5, alpha=0.5, label=r"$\propto \sqrt{depth}$ reference")
ax4.set_xlabel("Number of chained matmul layers", fontsize=12)
ax4.set_ylabel("Relative RMSE (output vs. ideal)", fontsize=12)
ax4.set_title(f"FP8 E4M3 (po2) \u2014 RMSE Growth vs. Depth (dim={d}, {n_trials_curve} trials)",
              fontsize=13, fontweight="bold")
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
fig4.tight_layout()
fig4.savefig("experiments/results/part4_rmse_vs_depth.png", dpi=150, bbox_inches="tight")
print("  Saved part4_rmse_vs_depth.png")


# ============================================================================
# PART 5 — Goodness-of-fit summary table
# ============================================================================

print()
print("=" * 60)
print("PART 5: Gaussian vs Laplace — Goodness-of-fit summary")
print("=" * 60)

header = (f"  {'Part':<12s}  {'Label':<28s}  {'Kurt':>6s}  "
          f"{'KS(G)':>7s}  {'KS(L)':>7s}  {'LL(G)':>8s}  {'LL(L)':>8s}  {'Winner':<8s}")
print(header)
print("  " + "-" * (len(header) - 2))

laplace_wins = 0
total = 0
for part, label, m in all_gof:
    winner = m["winner"]
    print(f"  {part:<12s}  {label:<28s}  {m['kurt']:>6.1f}  "
          f"{m['ks_gauss']:>7.4f}  {m['ks_lap']:>7.4f}  "
          f"{m['ll_gauss']:>8.2f}  {m['ll_lap']:>8.2f}  {winner:<8s}")
    if winner == "Laplace":
        laplace_wins += 1
    total += 1

print()
print(f"  Laplace wins: {laplace_wins}/{total} ({100*laplace_wins/total:.0f}%)")
print()
print("  KS = Kolmogorov-Smirnov statistic (lower = better fit)")
print("  LL = mean log-likelihood (higher = better fit)")
print()

# Summary figure
fig5, ax5 = plt.subplots(1, 1, figsize=(14, max(4, 0.4 * len(all_gof) + 2)))
ax5.axis("off")

col_labels = ["Part", "Label", "Kurt", "KS(Gauss)", "KS(Laplace)", "LL(Gauss)", "LL(Laplace)", "Winner"]
table_data = []
cell_colors = []
for part, label, m in all_gof:
    winner = m["winner"]
    row = [
        part, label,
        f"{m['kurt']:.1f}",
        f"{m['ks_gauss']:.4f}",
        f"{m['ks_lap']:.4f}",
        f"{m['ll_gauss']:.2f}",
        f"{m['ll_lap']:.2f}",
        winner,
    ]
    table_data.append(row)
    # Color the winner column
    if winner == "Laplace":
        row_colors = ["white"] * 7 + ["#c8e6c9"]  # light green
    else:
        row_colors = ["white"] * 7 + ["#ffcdd2"]  # light red
    cell_colors.append(row_colors)

table = ax5.table(
    cellText=table_data,
    colLabels=col_labels,
    cellColours=cell_colors,
    colColours=["#e0e0e0"] * len(col_labels),
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.4)

fig5.suptitle(
    f"Part 5: Gaussian vs Laplace Fit \u2014 Laplace wins {laplace_wins}/{total} "
    f"({100*laplace_wins/total:.0f}%)\n"
    f"KS = Kolmogorov-Smirnov (lower=better)   LL = mean log-likelihood (higher=better)",
    fontsize=12, fontweight="bold",
)
fig5.tight_layout()
fig5.savefig("experiments/results/part5_gof_summary.png", dpi=150, bbox_inches="tight")
print("  Saved part5_gof_summary.png")

print()
print("Done. All plots saved to experiments/results/")
