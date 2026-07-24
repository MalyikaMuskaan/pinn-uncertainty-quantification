"""
plot.py
-------
All visualisations for the inverse Burgers PINN.

Plot A — nu_convergence.png
    ν estimate vs. epoch for every ensemble member (thin grey lines) +
    ensemble mean (thick colour line) + true ν as a horizontal reference.

Plot B — solution_comparison.png
    PINN mean solution u(x,t) (using learned ν) vs FD reference at three
    time slices, with ensemble uncertainty band.

Plot C — robustness_summary.png
    2-D grid figure: rows = noise levels, cols = sensor counts.
    Each cell shows a box-plot of the M recovered ν values with the true
    value marked.

Plot D — robustness_table.png  (optional — text table rendered to figure)
    Summary table of mean/std/error/CI-coverage for all conditions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch

from data import NU_TRUE, fd_reference, make_evaluation_grid, X_MIN, X_MAX


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _eval_model(model, device, n_x=256, n_t=100):
    """Evaluate a single InverseBurgersPINN on a dense grid."""
    x_flat, t_flat, x_grid, t_grid = make_evaluation_grid(n_x, n_t, device)
    model.eval()
    with torch.no_grad():
        u_flat = model(x_flat, t_flat)
    u_pred = u_flat.cpu().numpy().reshape(n_t, n_x)
    return x_grid, t_grid, u_pred


def _ensemble_predict(members, device, n_x=256, n_t=100):
    """Run all ensemble members and return mean + std."""
    x_flat, t_flat, x_grid, t_grid = make_evaluation_grid(n_x, n_t, device)
    preds = []
    for m in members:
        m.eval()
        with torch.no_grad():
            u = m(x_flat, t_flat).cpu().numpy().reshape(n_t, n_x)
        preds.append(u)
    u_all  = np.stack(preds)        # (M, n_t, n_x)
    return x_grid, t_grid, u_all.mean(0), u_all.std(0)


# ------------------------------------------------------------------ #
#  Plot A — ν convergence                                              #
# ------------------------------------------------------------------ #

def plot_nu_convergence(
    nu_histories: list[list[tuple[int, float]]],
    save_path: str = "outputs/nu_convergence.png",
    title_suffix: str = "",
) -> None:
    """
    ν estimate vs training epoch for every ensemble member.

    Parameters
    ----------
    nu_histories : list of length M; each element is a list of (epoch, nu) tuples
    save_path    : output file path
    title_suffix : extra info appended to the figure title (e.g. noise / sensor info)
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    # ---- Individual members (thin grey) ----
    for hist in nu_histories:
        epochs = [ep for ep, _ in hist]
        nus    = [nu for _, nu in hist]
        ax.semilogy(epochs, nus, color="#aaaaaa", linewidth=0.9, alpha=0.6)

    # ---- Ensemble mean trajectory ----
    all_epochs = [ep for ep, _ in nu_histories[0]]
    mean_nu    = np.mean(
        [[nu for _, nu in h] for h in nu_histories], axis=0
    )
    ax.semilogy(all_epochs, mean_nu, color="#3b82d4", linewidth=2.2,
                label="Ensemble mean nu")

    # ---- True value reference ----
    ax.axhline(NU_TRUE, color="#e05c2a", linewidth=1.8, linestyle="--",
               label=f"True nu = {NU_TRUE:.6f}")

    # ---- Initial guess marker ----
    nu_init = nu_histories[0][0][1]
    ax.axhline(nu_init, color="#7c5cd8", linewidth=1.0, linestyle=":",
               alpha=0.7, label=f"Initial guess nu0 = {nu_init:.4f}")

    ax.set_xlabel("Training epoch", fontsize=11)
    ax.set_ylabel("Estimated nu  (log scale)", fontsize=11)
    ttl = "Convergence of recovered viscosity nu over training"
    if title_suffix:
        ttl += f"\n{title_suffix}"
    ax.set_title(ttl, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot] nu convergence plot saved to '{save_path}'")


# ------------------------------------------------------------------ #
#  Plot B — solution comparison                                        #
# ------------------------------------------------------------------ #

def plot_solution_comparison(
    members:      list,
    nu_learned:   float,
    device:       torch.device,
    time_slices:  list[float] = [0.25, 0.50, 0.75],
    save_path:    str = "outputs/solution_comparison.png",
) -> None:
    """
    PINN mean ± 1σ vs FD reference (computed with the LEARNED ν) at fixed times.

    Also shows the FD reference computed with the TRUE ν as a third curve so the
    reader can see how closely the learned ν reproduces the true dynamics.
    """
    x_grid, t_grid, u_mean, u_std = _ensemble_predict(members, device)
    x_vals = x_grid[0, :]
    t_vals = t_grid[:, 0]

    # FD references
    u_ref_true    = fd_reference(x_vals, t_vals, NU_TRUE)
    u_ref_learned = fd_reference(x_vals, t_vals, nu_learned)

    n = len(time_slices)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, t_star in zip(axes, time_slices):
        idx = int(np.argmin(np.abs(t_vals - t_star)))
        t_act = t_vals[idx]

        ax.plot(x_vals, u_ref_true[idx],
                color="#333333", lw=2.0, label=f"FD true nu")
        ax.plot(x_vals, u_ref_learned[idx],
                color="#7c5cd8", lw=1.6, ls="-.",
                label=f"FD learned nu={nu_learned:.5f}")
        ax.plot(x_vals, u_mean[idx],
                color="#3b82d4", lw=2.0, ls="--",
                label="PINN mean")
        ax.fill_between(x_vals,
                        u_mean[idx] - u_std[idx],
                        u_mean[idx] + u_std[idx],
                        alpha=0.18, color="#3b82d4", label="±1σ ensemble")

        ax.set_title(f"t = {t_act:.2f}", fontsize=11)
        ax.set_xlabel("x", fontsize=11)
        ax.set_xlim(X_MIN, X_MAX)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5, loc="lower center")

    axes[0].set_ylabel("u(x, t)", fontsize=11)
    fig.suptitle("PINN solution with learned nu vs. FD reference", fontsize=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Solution comparison saved to '{save_path}'")


# ------------------------------------------------------------------ #
#  Plot C — robustness summary (box-plots grid)                        #
# ------------------------------------------------------------------ #

def plot_robustness_summary(
    results:    list[dict],
    save_path:  str = "outputs/robustness_summary.png",
) -> None:
    """
    Grid of box-plots for each (noise, n_sensors) condition.

    Rows = noise levels (ascending), Cols = sensor counts (ascending).
    Each box-plot shows the M recovered ν values; a horizontal red dashed
    line marks the true ν.
    """
    noise_levels  = sorted(set(r["noise_frac"]  for r in results))
    sensor_counts = sorted(set(r["n_sensors"]   for r in results))

    nrows = len(noise_levels)
    ncols = len(sensor_counts)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 3.5 * nrows),
                             sharey=True, squeeze=False)

    for ri, noise in enumerate(noise_levels):
        for ci, n_s in enumerate(sensor_counts):
            ax = axes[ri][ci]

            # Find the matching result
            cond = next((r for r in results
                         if abs(r["noise_frac"] - noise) < 1e-9
                         and r["n_sensors"] == n_s), None)
            if cond is None:
                ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center")
                continue

            vals = np.array(cond["nu_estimates"])
            ax.boxplot(vals, vert=True, patch_artist=True,
                       boxprops    = dict(facecolor="#dce8fa", color="#3b82d4"),
                       medianprops = dict(color="#3b82d4", linewidth=2),
                       whiskerprops= dict(color="#3b82d4"),
                       capprops    = dict(color="#3b82d4"),
                       flierprops  = dict(marker="o", color="#3b82d4",
                                          markersize=4, alpha=0.6))
            ax.axhline(NU_TRUE, color="#e05c2a", linewidth=1.4, linestyle="--",
                       label=f"True nu")

            err_str = f"err={cond['mean_err_pct']:.1f}%"
            in_ci   = "YES" if cond["true_in_ci"] else "NO"
            ax.set_title(f"noise={noise:.1%}, N={n_s}\n{err_str}  CI:{in_ci}",
                         fontsize=9)
            ax.set_xticks([])
            ax.grid(True, axis="y", alpha=0.3)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    axes[0][0].legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Robustness sweep: recovered nu distribution\n"
        f"(true nu = {NU_TRUE:.6f}, 10 ensemble members per condition)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Robustness summary saved to '{save_path}'")


# ------------------------------------------------------------------ #
#  Plot D — summary table (rendered as figure)                         #
# ------------------------------------------------------------------ #

def plot_robustness_table(
    results:   list[dict],
    save_path: str = "outputs/robustness_table.png",
) -> None:
    """
    Render a colour-coded table of ν recovery metrics across all conditions.

    Columns: noise | N_sensors | ν_mean | ν_std | err% | 90%CI | true_in_CI
    Rows: sorted by (noise, n_sensors)
    """
    rows_sorted = sorted(results, key=lambda r: (r["noise_frac"], r["n_sensors"]))

    col_labels = ["noise", "N sensors", "nu mean", "nu std",
                  "err %", "90% CI", "true in CI"]
    table_data = []
    for r in rows_sorted:
        table_data.append([
            f"{r['noise_frac']:.1%}",
            str(r["n_sensors"]),
            f"{r['nu_mean']:.6f}",
            f"{r['nu_std']:.6f}",
            f"{r['mean_err_pct']:.2f}",
            f"[{r['ci_90_lo']:.5f}, {r['ci_90_hi']:.5f}]",
            "YES" if r["true_in_ci"] else "NO",
        ])

    nrows_t = len(table_data)
    fig, ax = plt.subplots(figsize=(13, 0.5 + 0.4 * nrows_t))
    ax.axis("off")

    t = ax.table(
        cellText   = table_data,
        colLabels  = col_labels,
        loc        = "center",
        cellLoc    = "center",
    )
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1.0, 1.5)

    # Highlight header
    for j in range(len(col_labels)):
        t[0, j].set_facecolor("#3b82d4")
        t[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight "true_in_CI" column
    for i, r in enumerate(rows_sorted, start=1):
        cell = t[i, 6]
        cell.set_facecolor("#d4f0d4" if r["true_in_ci"] else "#f0d4d4")

    fig.suptitle(
        f"Inverse PINN: nu recovery across noise x sparsity conditions\n"
        f"(true nu = {NU_TRUE:.6f},  10 ensemble members per cell)",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Robustness table saved to '{save_path}'")


# ------------------------------------------------------------------ #
#  Plot E — per-condition ν histograms (1 histogram per condition)     #
# ------------------------------------------------------------------ #

def plot_nu_distributions(
    results:   list[dict],
    save_path: str = "outputs/nu_distributions.png",
) -> None:
    """
    Histogram of recovered ν values for every condition, arranged in the
    same noise × sensor-count grid as plot_robustness_summary.
    """
    noise_levels  = sorted(set(r["noise_frac"]  for r in results))
    sensor_counts = sorted(set(r["n_sensors"]   for r in results))
    nrows, ncols  = len(noise_levels), len(sensor_counts)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 3 * nrows),
                             squeeze=False)

    for ri, noise in enumerate(noise_levels):
        for ci, n_s in enumerate(sensor_counts):
            ax = axes[ri][ci]
            cond = next((r for r in results
                         if abs(r["noise_frac"] - noise) < 1e-9
                         and r["n_sensors"] == n_s), None)
            if cond is None:
                continue
            vals = np.array(cond["nu_estimates"])
            ax.hist(vals, bins=min(8, len(vals)), color="#3b82d4",
                    edgecolor="white", alpha=0.85)
            ax.axvline(NU_TRUE, color="#e05c2a", lw=1.8, ls="--", label="true nu")
            ax.axvline(cond["nu_mean"], color="#7c5cd8", lw=1.4, ls="-",
                       label=f"mean={cond['nu_mean']:.5f}")
            ax.set_title(f"noise={noise:.1%}, N={n_s}", fontsize=9)
            ax.set_xlabel("nu", fontsize=9)
            ax.grid(True, alpha=0.25)
            if ri == 0 and ci == 0:
                ax.legend(fontsize=7.5)

    fig.suptitle("Distribution of recovered nu values per condition", fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] nu distributions saved to '{save_path}'")
