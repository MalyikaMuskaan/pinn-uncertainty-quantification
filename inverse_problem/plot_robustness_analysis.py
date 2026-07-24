"""
plot_robustness_analysis.py
---------------------------
Generates two publication-quality figures from outputs/robustness/summary.json:

  Fig 1 — robustness_error_vs_sensors.png
      Error % vs sensor count, one line per noise level.

  Fig 2 — robustness_nu_estimates.png
      Recovered nu (mean ± 90% CI) for each condition,
      with a horizontal reference at the true nu value.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ------------------------------------------------------------------ #
#  Config                                                              #
# ------------------------------------------------------------------ #
SUMMARY_JSON = os.path.join(os.path.dirname(__file__),
                            "outputs", "robustness", "summary.json")
OUT_DIR       = os.path.join(os.path.dirname(__file__),
                             "outputs", "robustness")
NU_TRUE       = 0.01 / np.pi   # ≈ 0.003183

NOISE_COLOURS = {
    0.005: "#3b82d4",   # blue
    0.010: "#e05c2a",   # orange-red
    0.020: "#7c5cd8",   # purple
}
NOISE_LABELS = {
    0.005: "noise 0.5%",
    0.010: "noise 1.0%",
    0.020: "noise 2.0%",
}

# ------------------------------------------------------------------ #
#  Load data                                                           #
# ------------------------------------------------------------------ #
with open(SUMMARY_JSON) as fh:
    results = json.load(fh)

results_sorted = sorted(results, key=lambda r: (r["noise_frac"], r["n_sensors"]))
noise_levels   = sorted({r["noise_frac"] for r in results})
sensor_counts  = sorted({r["n_sensors"]  for r in results})

# ------------------------------------------------------------------ #
#  Fig 1 — Error % vs sensor count, grouped by noise level            #
# ------------------------------------------------------------------ #
fig1, ax1 = plt.subplots(figsize=(7, 4.5))

for noise in noise_levels:
    group = sorted(
        [r for r in results if abs(r["noise_frac"] - noise) < 1e-9],
        key=lambda r: r["n_sensors"],
    )
    xs  = [r["n_sensors"]      for r in group]
    ys  = [r["mean_err_pct"]   for r in group]
    col = NOISE_COLOURS[noise]
    ax1.plot(xs, ys, marker="o", markersize=7, linewidth=2.0,
             color=col, label=NOISE_LABELS[noise])
    for x, y in zip(xs, ys):
        ax1.annotate(f"{y:.0f}%", xy=(x, y),
                     xytext=(5, 4), textcoords="offset points",
                     fontsize=8, color=col)

ax1.set_xlabel("Number of sensors", fontsize=11)
ax1.set_ylabel("Mean absolute error in ν  (%)", fontsize=11)
ax1.set_title("Inverse PINN — recovery error vs. sensor density\n"
              "(10 ensemble members per condition)", fontsize=11)
ax1.set_xticks(sensor_counts)
ax1.xaxis.set_major_formatter(mticker.FixedFormatter([str(s) for s in sensor_counts]))
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9, framealpha=0.85)
plt.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, "robustness_error_vs_sensors.png"),
             dpi=150, bbox_inches="tight")
plt.close(fig1)
print("[plot] saved robustness_error_vs_sensors.png")


# ------------------------------------------------------------------ #
#  Fig 2 — Recovered ν mean ± 90% CI, all conditions                 #
# ------------------------------------------------------------------ #
# Build ordered label list: group by sensor count so each noise set
# is a cluster; within each cluster, sensors increase left→right.
ordered = sorted(results_sorted, key=lambda r: (r["n_sensors"], r["noise_frac"]))

labels     = [f"N={r['n_sensors']}\n{r['noise_frac']:.1%}" for r in ordered]
means      = np.array([r["nu_mean"]   for r in ordered])
ci_lo      = np.array([r["ci_90_lo"]  for r in ordered])
ci_hi      = np.array([r["ci_90_hi"]  for r in ordered])
colours    = [NOISE_COLOURS[r["noise_frac"]] for r in ordered]
true_in_ci = [r["true_in_ci"] for r in ordered]

xs = np.arange(len(ordered))

fig2, ax2 = plt.subplots(figsize=(10, 5))

for i, (x, mean, lo, hi, col, tic) in enumerate(
        zip(xs, means, ci_lo, ci_hi, colours, true_in_ci)):
    # CI error bars
    ax2.errorbar(x, mean,
                 yerr=[[mean - lo], [hi - mean]],
                 fmt="o", markersize=8,
                 color=col, ecolor=col,
                 elinewidth=2, capsize=5, capthick=2,
                 zorder=3)
    # Stars above points that capture true value
    if tic:
        ax2.annotate("★", xy=(x, hi + 0.0005), fontsize=11,
                     color="#2a9d2a", ha="center")

# True nu reference line
ax2.axhline(NU_TRUE, color="#e05c2a", linewidth=1.8, linestyle="--",
            label=f"True ν = {NU_TRUE:.6f}")

# Legend proxies for noise colours
for noise in noise_levels:
    ax2.plot([], [], "o", color=NOISE_COLOURS[noise],
             label=NOISE_LABELS[noise])
ax2.plot([], [], "o", color="gray", label="mean ± 90% CI")

ax2.set_xticks(xs)
ax2.set_xticklabels(labels, fontsize=8)
ax2.set_ylabel("Recovered ν", fontsize=11)
ax2.set_title("Inverse PINN — recovered ν with 90% CI by condition\n"
              "(★ = true value captured inside CI)", fontsize=11)
ax2.grid(True, axis="y", alpha=0.3)
ax2.legend(fontsize=8.5, framealpha=0.9, loc="upper left")
plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, "robustness_nu_estimates.png"),
             dpi=150, bbox_inches="tight")
plt.close(fig2)
print("[plot] saved robustness_nu_estimates.png")
