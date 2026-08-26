"""
Plot budget-aware UQ experiment results.

This script reads the summary CSV produced by run_budgeted_uq_experiments.py and
creates a collection of plots:
  - coverage versus budget,
  - average upper bound versus budget,
  - average point estimate versus budget,
  - empirical variance / relative efficiency versus budget,
  - probability of realized improvement,
  - allocation diagnostics,
  - correlation / alpha diagnostics,
  - horizontal-interval plots with the true gap shown as a black dashed line.

Example Usage
-----
python plot_budgeted_uq_results.py path/to/budgeted_uq_summary.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sparow.conf_intervals.plotting_helpers import (
    wrap_title,
    wrap_ylabel,
    add_bottom_legend,
    apply_grid,
    finalize_standard_plot,
    compute_axis_limits_from_intervals,
    draw_horizontal_interval,
    finalize_interval_plot,
    make_standard_figure,
    make_interval_figure,

    COLOR_ACV,
    COLOR_HF_BUDGET,
    COLOR_HF_PAIRED,
    COLOR_TRUE
)

csv_path = Path(sys.argv[1]).resolve()
df = pd.read_csv(csv_path)

true_gap = df["true_gap"].iloc[0] if "true_gap" in df.columns else np.nan
R_global = int(df["R"].iloc[0]) if "R" in df.columns else None

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
    }
)

# ----------------------------------------------------------------------
# Write plots next to the CSV, inside a dedicated plots subdirectory
# ----------------------------------------------------------------------
output_dir = csv_path.parent / "plots"
output_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Core line plots versus budget
# =============================================================================

for n, sub in df.groupby("n"):
    sub = sub.sort_values("budget").copy()
    R = int(sub["R"].iloc[0])

    # -------------------------------------------------------------------------
    # Coverage versus budget
    # -------------------------------------------------------------------------
    fig, ax = make_standard_figure()
    ax.plot(
        sub["budget"],
        sub["empirical_coverage_acv"],
        marker="o",
        color=COLOR_ACV,
        label=r"Multifidelity empirical coverage",
    )
    ax.plot(
        sub["budget"],
        sub["empirical_coverage_hf_budget"],
        marker="s",
        color=COLOR_HF_BUDGET,
        label=r"HF-only empirical coverage (same total budget)",
    )
    ax.plot(
        sub["budget"],
        sub["empirical_coverage_hf_paired_only"],
        marker="^",
        color=COLOR_HF_PAIRED,
        label=r"HF-only empirical coverage (paired-count baseline, same $m$)",
    )
    ax.axhline(
        1.0 - sub["alpha"].iloc[0],
        color=COLOR_TRUE,
        linestyle="--",
        label=rf"Nominal coverage $1-\alpha={1.0 - sub['alpha'].iloc[0]:.2f}$",
    )
    apply_grid(ax)
    finalize_standard_plot(
        fig,
        ax,
        xlabel=r"Total wall-clock budget",
        ylabel="Empirical coverage probability",
        title=rf"Empirical coverage versus budget for fixed replication batch size $n={n}$ ($R={R}$ macro-replications)",
    )
    fig.savefig(output_dir / f"coverage_vs_budget_n_{n}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Average upper bound versus budget
    # -------------------------------------------------------------------------
    fig, ax = make_standard_figure()
    ax.plot(
        sub["budget"],
        sub["avg_acv_ci_upper"],
        marker="o",
        color=COLOR_ACV,
        label=r"Multifidelity average upper confidence bound",
    )
    ax.plot(
        sub["budget"],
        sub["avg_hf_budget_ci_upper"],
        marker="s",
        color=COLOR_HF_BUDGET,
        label=r"HF-only average upper confidence bound (same total budget)",
    )
    ax.plot(
        sub["budget"],
        sub["avg_hf_paired_only_ci_upper"],
        marker="^",
        color=COLOR_HF_PAIRED,
        label=r"HF-only average upper confidence bound (paired-count baseline, same $m$)",
    )
    ax.axhline(
        true_gap,
        color=COLOR_TRUE,
        linestyle="--",
        label=rf"True optimality gap $\Delta_f(\hat{{x}})$",
    )
    apply_grid(ax)
    finalize_standard_plot(
        fig,
        ax,
        xlabel=r"Total wall-clock budget",
        ylabel="Average one-sided upper confidence bound",
        title=rf"Average upper confidence bound versus budget for fixed $n={n}$ ($R={R}$ macro-replications)",
    )
    fig.savefig(output_dir / f"avg_ci_upper_vs_budget_n_{n}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Average point estimate versus budget
    # -------------------------------------------------------------------------
    fig, ax = make_standard_figure()
    ax.plot(
        sub["budget"],
        sub["avg_acv_point_estimate"],
        marker="o",
        color=COLOR_ACV,
        label=r"Multifidelity average point estimate",
    )
    ax.plot(
        sub["budget"],
        sub["avg_hf_budget_point_estimate"],
        marker="s",
        color=COLOR_HF_BUDGET,
        label=r"HF-only average point estimate (same total budget)",
    )
    ax.plot(
        sub["budget"],
        sub["avg_hf_paired_only_point_estimate"],
        marker="^",
        color=COLOR_HF_PAIRED,
        label=r"HF-only average point estimate (paired-count baseline, same $m$)",
    )
    ax.axhline(
        true_gap,
        color=COLOR_TRUE,
        linestyle="--",
        label=rf"True optimality gap $\Delta_f(\hat{{x}})$",
    )
    apply_grid(ax)
    finalize_standard_plot(
        fig,
        ax,
        xlabel=r"Total wall-clock budget",
        ylabel="Average point estimate",
        title=rf"Average point estimate versus budget for fixed $n={n}$ ($R={R}$ macro-replications)",
    )
    fig.savefig(output_dir / f"avg_point_estimate_vs_budget_n_{n}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Average margin of error versus budget
    # -------------------------------------------------------------------------
    fig, ax = make_standard_figure()
    ax.plot(
        sub["budget"],
        sub["avg_acv_half_width"],
        marker="o",
        color=COLOR_ACV,
        label=r"Multifidelity average margin of error",
    )
    ax.plot(
        sub["budget"],
        sub["avg_hf_budget_half_width"],
        marker="s",
        color=COLOR_HF_BUDGET,
        label=r"HF-only average margin of error (same total budget)",
    )
    ax.plot(
        sub["budget"],
        sub["avg_hf_paired_only_half_width"],
        marker="^",
        color=COLOR_HF_PAIRED,
        label=r"HF-only average margin of error (paired-count baseline, same $m$)",
    )
    apply_grid(ax)
    finalize_standard_plot(
        fig,
        ax,
        xlabel=r"Total wall-clock budget",
        ylabel="Average one-sided margin of error",
        title=rf"Average margin of error versus budget for fixed $n={n}$ ($R={R}$ macro-replications)",
    )
    fig.savefig(output_dir / f"avg_margin_of_error_vs_budget_n_{n}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Improvement probability versus budget
    # -------------------------------------------------------------------------
    fig, ax = make_standard_figure()
    ax.plot(
        sub["budget"],
        sub["prob_acv_improves_over_hf_budget"],
        marker="o",
        color=COLOR_HF_BUDGET,
        label=r"$\mathbb{P}(U_f^{\mathrm{ACV}} < U_f^{\mathrm{HF,same\ budget}})$",
    )
    ax.plot(
        sub["budget"],
        sub["prob_acv_improves_over_hf_paired_only"],
        marker="s",
        color=COLOR_HF_PAIRED,
        label=r"$\mathbb{P}(U_f^{\mathrm{ACV}} < U_f^{\mathrm{HF,paired-only}})$",
    )
    apply_grid(ax)
    finalize_standard_plot(
        fig,
        ax,
        xlabel=r"Total wall-clock budget",
        ylabel="Probability of realized improvement",
        title=rf"Probability Multifidelity yields a smaller realized upper bound, fixed $n={n}$ ($R={R}$ macro-replications)",
    )
    fig.savefig(output_dir / f"prob_improvement_vs_budget_n_{n}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Relative efficiency versus budget
    # -------------------------------------------------------------------------
    fig, ax = make_standard_figure()
    ax.plot(
        sub["budget"],
        sub["relative_efficiency_hf_budget_over_acv"],
        marker="o",
        color=COLOR_HF_BUDGET,
        label=r"Empirical relative efficiency: HF-only same budget / Multifidelity (values > 1 indicate Multifidelity is more efficient)",
    )
    ax.plot(
        sub["budget"],
        sub["relative_efficiency_hf_paired_only_over_acv"],
        marker="s",
        color=COLOR_HF_PAIRED,
        label=r"Empirical relative efficiency: HF paired-only / Multifidelity (values > 1 indicate Multifidelity is more efficient)",
    )
    ax.axhline(1.0, color=COLOR_TRUE, linestyle="--", label="Parity")
    apply_grid(ax)
    finalize_standard_plot(
        fig,
        ax,
        xlabel=r"Total wall-clock budget",
        ylabel="Empirical variance ratio",
        title=rf"Empirical relative efficiency versus budget for fixed $n={n}$ ($R={R}$ macro-replications)",
    )
    fig.savefig(output_dir / f"relative_efficiency_vs_budget_n_{n}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Empirical variance of point estimates versus budget
    # -------------------------------------------------------------------------
    fig, ax = make_standard_figure()
    ax.plot(
        sub["budget"],
        sub["empirical_variance_acv_point"],
        marker="o",
        color=COLOR_ACV,
        label=r"Multifidelity empirical variance of point estimates",
    )
    ax.plot(
        sub["budget"],
        sub["empirical_variance_hf_budget_point"],
        marker="s",
        color=COLOR_HF_BUDGET,
        label=r"HF-only empirical variance (same total budget)",
    )
    ax.plot(
        sub["budget"],
        sub["empirical_variance_hf_paired_only_point"],
        marker="^",
        color=COLOR_HF_PAIRED,
        label=r"HF-only empirical variance (paired-count baseline, same $m$)",
    )
    apply_grid(ax)
    finalize_standard_plot(
        fig,
        ax,
        xlabel=r"Total wall-clock budget",
        ylabel="Empirical variance across macro-replications",
        title=rf"Empirical variance of point estimates versus budget for fixed $n={n}$ ($R={R}$ macro-replications)",
    )
    fig.savefig(output_dir / f"empirical_variance_point_vs_budget_n_{n}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Allocation diagnostics versus budget
    # -------------------------------------------------------------------------
    fig, ax = make_standard_figure()
    ax.plot(
        sub["budget"],
        sub["m_paired_mean"],
        marker="o",
        color=COLOR_ACV,
        label=r"Average paired replication count $m$ recommended by PyApprox",
    )
    ax.plot(
        sub["budget"],
        sub["M_additional_lf_mean"],
        marker="s",
        color=COLOR_ACV,
        linestyle="--",
        label=r"Average additional LF replication count $M$ recommended by PyApprox",
    )
    ax.plot(
        sub["budget"],
        sub["hf_same_budget_m_mean"],
        marker="^",
        color=COLOR_HF_BUDGET,
        label=r"Average HF-only replication count under same total budget",
    )
    apply_grid(ax)
    finalize_standard_plot(
        fig,
        ax,
        xlabel=r"Total wall-clock budget",
        ylabel="Average replication count",
        title=rf"Budget-aware allocation counts versus budget for fixed $n={n}$ ($R={R}$ macro-replications)",
    )
    fig.savefig(output_dir / f"allocation_counts_vs_budget_n_{n}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Pilot and ACV correlation diagnostics versus budget
    # -------------------------------------------------------------------------
    fig, ax = make_standard_figure()
    ax.plot(
        sub["budget"],
        sub["pilot_rho_hat_mean"],
        marker="o",
        color=COLOR_HF_BUDGET,
        label=r"Average pilot correlation estimate $\hat{\rho}_{fg}^{\mathrm{pilot}}$",
    )
    ax.plot(
        sub["budget"],
        sub["avg_acv_rho_hat"],
        marker="s",
        color=COLOR_ACV,
        label=r"Average multifidelity paired-replication correlation estimate $\hat{\rho}_{fg}$",
    )
    apply_grid(ax)
    finalize_standard_plot(
        fig,
        ax,
        xlabel=r"Total wall-clock budget",
        ylabel="Estimated correlation",
        title=rf"Estimated HF/LF correlation versus budget for fixed $n={n}$ ($R={R}$ macro-replications)",
    )
    fig.savefig(output_dir / f"correlation_vs_budget_n_{n}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Alpha-hat (estimated control-variate coefficient) versus budget
    # -------------------------------------------------------------------------
    fig, ax = make_standard_figure()
    ax.plot(
        sub["budget"],
        sub["avg_acv_alpha_hat"],
        marker="o",
        color=COLOR_ACV,
        label=r"Average estimated control-variate coefficient $\hat{\alpha}$",
    )
    apply_grid(ax)
    finalize_standard_plot(
        fig,
        ax,
        xlabel=r"Total wall-clock budget",
        ylabel="Average estimated control-variate coefficient",
        title=rf"Average estimated control-variate coefficient versus budget for fixed $n={n}$ ($R={R}$ macro-replications)",
    )
    fig.savefig(output_dir / f"alpha_hat_vs_budget_n_{n}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Predicted PyApprox std versus budget
    # -------------------------------------------------------------------------
    fig, ax = make_standard_figure()
    ax.plot(
        sub["budget"],
        sub["predicted_pyapprox_std_mean"],
        marker="o",
        color=COLOR_ACV,
        label=r"Average PyApprox-predicted standard deviation",
    )
    apply_grid(ax)
    finalize_standard_plot(
        fig,
        ax,
        xlabel=r"Total wall-clock budget",
        ylabel="Predicted standard deviation",
        title=rf"PyApprox-predicted estimator standard deviation versus budget for fixed $n={n}$ ($R={R}$ macro-replications)",
    )
    fig.savefig(output_dir / f"predicted_pyapprox_std_vs_budget_n_{n}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Elapsed time profiling versus budget
    # -------------------------------------------------------------------------
    fig, ax = make_standard_figure()
    ax.plot(
        sub["budget"],
        sub["avg_elapsed_acv_seconds"],
        marker="o",
        color=COLOR_ACV,
        label="Average multifidelity elapsed time",
    )
    ax.plot(
        sub["budget"],
        sub["avg_elapsed_hf_budget_seconds"],
        marker="s",
        color=COLOR_HF_BUDGET,
        label="Average HF-only same-budget elapsed time",
    )
    ax.plot(
        sub["budget"],
        sub["avg_elapsed_hf_paired_only_seconds"],
        marker="^",
        color=COLOR_HF_PAIRED,
        label="Average HF-only paired-only elapsed time",
    )
    apply_grid(ax)
    finalize_standard_plot(
        fig,
        ax,
        xlabel=r"Total wall-clock budget",
        ylabel="Average elapsed runtime (seconds)",
        title=rf"Observed runtime profiling versus budget for fixed $n={n}$ ($R={R}$ macro-replications)",
    )
    fig.savefig(output_dir / f"elapsed_runtime_vs_budget_n_{n}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Plots versus n for fixed budget
# =============================================================================

for budget, sub in df.groupby("budget"):
    sub = sub.sort_values("n").copy()
    R = int(sub["R"].iloc[0])

    # Point estimate versus n
    fig, ax = make_standard_figure()
    ax.plot(
        sub["n"],
        sub["avg_acv_point_estimate"],
        marker="o",
        color=COLOR_ACV,
        label=r"Multifidelity average point estimate",
    )
    ax.plot(
        sub["n"],
        sub["avg_hf_budget_point_estimate"],
        marker="s",
        color=COLOR_HF_BUDGET,
        label=r"HF-only average point estimate (same total budget)",
    )
    ax.plot(
        sub["n"],
        sub["avg_hf_paired_only_point_estimate"],
        marker="^",
        color=COLOR_HF_PAIRED,
        label=r"HF-only average point estimate (paired-count baseline, same $m$)",
    )
    ax.axhline(
        true_gap,
        color=COLOR_TRUE,
        linestyle="--",
        label=rf"True optimality gap $\Delta_f(\hat{{x}})$",
    )
    apply_grid(ax)
    finalize_standard_plot(
        fig,
        ax,
        xlabel=r"Replication batch size $n$",
        ylabel="Average point estimate",
        title=rf"Average point estimate versus replication batch size $n$ for fixed budget={budget} ($R={R}$ macro-replications)",
    )
    fig.savefig(output_dir / f"avg_point_estimate_vs_n_budget_{budget}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Upper bound versus n
    fig, ax = make_standard_figure()
    ax.plot(
        sub["n"],
        sub["avg_acv_ci_upper"],
        marker="o",
        color=COLOR_ACV,
        label=r"Multifidelity average upper confidence bound",
    )
    ax.plot(
        sub["n"],
        sub["avg_hf_budget_ci_upper"],
        marker="s",
        color=COLOR_HF_BUDGET,
        label=r"HF-only average upper confidence bound (same total budget)",
    )
    ax.plot(
        sub["n"],
        sub["avg_hf_paired_only_ci_upper"],
        marker="^",
        color=COLOR_HF_PAIRED,
        label=r"HF-only average upper confidence bound (paired-count baseline, same $m$)",
    )
    ax.axhline(
        true_gap,
        color=COLOR_TRUE,
        linestyle="--",
        label=rf"True optimality gap $\Delta_f(\hat{{x}})$",
    )
    apply_grid(ax)
    finalize_standard_plot(
        fig,
        ax,
        xlabel=r"Replication batch size $n$",
        ylabel="Average upper confidence bound",
        title=rf"Average upper confidence bound versus replication batch size $n$ for fixed budget={budget} ($R={R}$ macro-replications)",
    )
    fig.savefig(output_dir / f"avg_ci_upper_vs_n_budget_{budget}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Coverage versus n
    fig, ax = make_standard_figure()
    ax.plot(
        sub["n"],
        sub["empirical_coverage_acv"],
        marker="o",
        color=COLOR_ACV,
        label=r"Multifidelity empirical coverage",
    )
    ax.plot(
        sub["n"],
        sub["empirical_coverage_hf_budget"],
        marker="s",
        color=COLOR_HF_BUDGET,
        label=r"HF-only empirical coverage (same total budget)",
    )
    ax.plot(
        sub["n"],
        sub["empirical_coverage_hf_paired_only"],
        marker="^",
        color=COLOR_HF_PAIRED,
        label=r"HF-only empirical coverage (paired-count baseline, same $m$)",
    )
    ax.axhline(
        1.0 - sub["alpha"].iloc[0],
        color=COLOR_TRUE,
        linestyle="--",
        label=rf"Nominal coverage $1-\alpha={1.0 - sub['alpha'].iloc[0]:.2f}$",
    )
    apply_grid(ax)
    finalize_standard_plot(
        fig,
        ax,
        xlabel=r"Replication batch size $n$",
        ylabel="Empirical coverage probability",
        title=rf"Empirical coverage versus replication batch size $n$ for fixed budget={budget} ($R={R}$ macro-replications)",
    )
    fig.savefig(output_dir / f"coverage_vs_n_budget_{budget}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Horizontal interval plots:
# compare average point estimate and average upper bound as intervals
# =============================================================================

# For fixed n, vary budget on y-axis
for n, sub in df.groupby("n"):
    sub = sub.sort_values("budget").copy()
    R = int(sub["R"].iloc[0])

    y_positions = np.arange(len(sub))
    offset = 0.12
    cap_half_height = 0.08

    # -------------------------------------------------------------------------
    # Same-total-budget comparison: ACV vs HF-only same budget
    # -------------------------------------------------------------------------
    x_left, x_right = compute_axis_limits_from_intervals(
        sub,
        acv_upper_col="avg_acv_ci_upper",
        hf_upper_col="avg_hf_budget_ci_upper",
        hf_point_col="avg_hf_budget_point_estimate",
        acv_point_col="avg_acv_point_estimate",
        true_gap=true_gap,
    )

    fig, ax = make_interval_figure(len(sub))
    for y, (_, row) in zip(y_positions, sub.iterrows()):
        draw_horizontal_interval(
            ax,
            y=y + offset,
            ci_lower=0.0,
            ci_upper=row["avg_hf_budget_ci_upper"],
            point_estimate=row["avg_hf_budget_point_estimate"],
            x_left=x_left,
            line_color=COLOR_HF_BUDGET,
            marker="o",
            marker_color=COLOR_HF_BUDGET,
            marker_size=8,
            alpha=0.9,
            cap_half_height=cap_half_height,
        )
        draw_horizontal_interval(
            ax,
            y=y - offset,
            ci_lower=0.0,
            ci_upper=row["avg_acv_ci_upper"],
            point_estimate=row["avg_acv_point_estimate"],
            x_left=x_left,
            line_color=COLOR_ACV,
            marker="o",
            marker_color=COLOR_ACV,
            marker_size=8,
            alpha=0.9,
            cap_half_height=cap_half_height,
        )

    legend_handles = [
        Line2D([0], [0], color=COLOR_HF_BUDGET, lw=2, label="HF-only same-total-budget average confidence interval"),
        Line2D([0], [0], marker="o", color=COLOR_HF_BUDGET, lw=0, markersize=6,
               label=r"HF-only same-total-budget average point estimate"),
        Line2D([0], [0], color=COLOR_ACV, lw=2, label="Multifidelity average confidence interval"),
        Line2D([0], [0], marker="o", color=COLOR_ACV, lw=0, markersize=6,
               label=r"Multifidelity average point estimate"),
        Line2D([0], [0], color=COLOR_TRUE, lw=1.5, linestyle="--",
               label=rf"True optimality gap $\Delta_f(\hat{{x}})$"),
    ]

    finalize_interval_plot(
        fig,
        ax,
        true_gap=true_gap,
        x_left=x_left,
        x_right=x_right,
        y_positions=y_positions,
        ytick_labels=[rf"budget={b:g}" for b in sub["budget"]],
        xlabel=r"Average point estimate / average upper confidence bound",
        ylabel=r"Total budget",
        title=rf"Same-total-budget comparison of average one-sided confidence intervals for fixed $n={n}$ ($R={R}$ macro-replications)",
        legend_handles=legend_handles,
    )
    fig.savefig(output_dir / f"avg_intervals_same_budget_compare_fixed_n_{n}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Paired-HF-only comparison: ACV vs HF-only same paired count m
    # -------------------------------------------------------------------------
    x_left, x_right = compute_axis_limits_from_intervals(
        sub,
        acv_upper_col="avg_acv_ci_upper",
        hf_upper_col="avg_hf_paired_only_ci_upper",
        hf_point_col="avg_hf_paired_only_point_estimate",
        acv_point_col="avg_acv_point_estimate",
        true_gap=true_gap,
    )

    fig, ax = make_interval_figure(len(sub))
    for y, (_, row) in zip(y_positions, sub.iterrows()):
        draw_horizontal_interval(
            ax,
            y=y + offset,
            ci_lower=0.0,
            ci_upper=row["avg_hf_paired_only_ci_upper"],
            point_estimate=row["avg_hf_paired_only_point_estimate"],
            x_left=x_left,
            line_color=COLOR_HF_PAIRED,
            marker="o",
            marker_color=COLOR_HF_PAIRED,
            marker_size=8,
            alpha=0.9,
            cap_half_height=cap_half_height,
        )
        draw_horizontal_interval(
            ax,
            y=y - offset,
            ci_lower=0.0,
            ci_upper=row["avg_acv_ci_upper"],
            point_estimate=row["avg_acv_point_estimate"],
            x_left=x_left,
            line_color=COLOR_ACV,
            marker="o",
            marker_color=COLOR_ACV,
            marker_size=8,
            alpha=0.9,
            cap_half_height=cap_half_height,
        )

    legend_handles = [
        Line2D([0], [0], color=COLOR_HF_PAIRED, lw=2, label=r"HF-only average confidence interval using the same paired count $m$"),
        Line2D([0], [0], marker="o", color=COLOR_HF_PAIRED, lw=0, markersize=6,
               label=r"HF-only average point estimate using the same paired count $m$"),
        Line2D([0], [0], color=COLOR_ACV, lw=2, label="Multifidelity average confidence interval"),
        Line2D([0], [0], marker="o", color=COLOR_ACV, lw=0, markersize=6,
               label=r"Multifidelity average point estimate"),
        Line2D([0], [0], color=COLOR_TRUE, lw=1.5, linestyle="--",
               label=rf"True optimality gap $\Delta_f(\hat{{x}})$"),
    ]

    finalize_interval_plot(
        fig,
        ax,
        true_gap=true_gap,
        x_left=x_left,
        x_right=x_right,
        y_positions=y_positions,
        ytick_labels=[rf"budget={b:g}" for b in sub["budget"]],
        xlabel=r"Average point estimate / average upper confidence bound",
        ylabel=r"Total budget",
        title=rf"Paired-count comparison of average one-sided confidence intervals for fixed $n={n}$ ($R={R}$ macro-replications)",
        legend_handles=legend_handles,
    )
    fig.savefig(output_dir / f"avg_intervals_paired_compare_fixed_n_{n}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# For fixed budget, vary n on y-axis
for budget, sub in df.groupby("budget"):
    sub = sub.sort_values("n").copy()
    R = int(sub["R"].iloc[0])

    y_positions = np.arange(len(sub))
    offset = 0.12
    cap_half_height = 0.08

    x_left, x_right = compute_axis_limits_from_intervals(
        sub,
        acv_upper_col="avg_acv_ci_upper",
        hf_upper_col="avg_hf_budget_ci_upper",
        hf_point_col="avg_hf_budget_point_estimate",
        acv_point_col="avg_acv_point_estimate",
        true_gap=true_gap,
    )

    fig, ax = make_interval_figure(len(sub))
    for y, (_, row) in zip(y_positions, sub.iterrows()):
        draw_horizontal_interval(
            ax,
            y=y + offset,
            ci_lower=0.0,
            ci_upper=row["avg_hf_budget_ci_upper"],
            point_estimate=row["avg_hf_budget_point_estimate"],
            x_left=x_left,
            line_color=COLOR_HF_BUDGET,
            marker="o",
            marker_color=COLOR_HF_BUDGET,
            marker_size=8,
            alpha=0.9,
            cap_half_height=cap_half_height,
        )
        draw_horizontal_interval(
            ax,
            y=y - offset,
            ci_lower=0.0,
            ci_upper=row["avg_acv_ci_upper"],
            point_estimate=row["avg_acv_point_estimate"],
            x_left=x_left,
            line_color=COLOR_ACV,
            marker="o",
            marker_color=COLOR_ACV,
            marker_size=8,
            alpha=0.9,
            cap_half_height=cap_half_height,
        )

    legend_handles = [
        Line2D([0], [0], color=COLOR_HF_BUDGET, lw=2, label="HF-only same-total-budget average confidence interval"),
        Line2D([0], [0], marker="o", color=COLOR_HF_BUDGET, lw=0, markersize=6,
               label=r"HF-only same-total-budget average point estimate"),
        Line2D([0], [0], color=COLOR_ACV, lw=2, label="Multifidelity average confidence interval"),
        Line2D([0], [0], marker="o", color=COLOR_ACV, lw=0, markersize=6,
               label=r"Multifidelity average point estimate"),
        Line2D([0], [0], color=COLOR_TRUE, lw=1.5, linestyle="--",
               label=rf"True optimality gap $\Delta_f(\hat{{x}})$"),
    ]

    finalize_interval_plot(
        fig,
        ax,
        true_gap=true_gap,
        x_left=x_left,
        x_right=x_right,
        y_positions=y_positions,
        ytick_labels=[rf"$n={int(n)}$" for n in sub["n"]],
        xlabel=r"Average point estimate / average upper confidence bound",
        ylabel=r"Replication batch size",
        title=rf"Same-total-budget comparison of average one-sided confidence intervals for fixed budget={budget} ($R={R}$ macro-replications)",
        legend_handles=legend_handles,
    )
    fig.savefig(output_dir / f"avg_intervals_same_budget_compare_fixed_budget_{budget}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


print(f"Wrote plots to: {output_dir}")