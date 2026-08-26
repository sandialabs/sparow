import sys
import numpy as np
from pathlib import Path
from scipy import stats

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
    COLOR_HF,
    COLOR_TRUE
)

csv_path = Path(sys.argv[1]).resolve()
df = pd.read_csv(csv_path)

true_gap = df["true_gap"].iloc[0]

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

# ==========================================================
# Interval comparison helper functions
# ==========================================================

def add_hf_only_interval_columns(sub):
    """
    Reconstruct the HF-only one-sided confidence interval quantities.
    """
    sub = sub.copy()

    sub["standard_error_hf_only"] = np.sqrt(sub["sample_variance_F"] / sub["m"])
    sub["t_statistic_hf_only"] = sub["m"].apply(
        lambda mm: stats.t.ppf(1.0 - sub["alpha"].iloc[0], df=mm - 1)
    )
    sub["half_width_hf_only"] = (
        sub["t_statistic_hf_only"] * sub["standard_error_hf_only"]
    )
    sub["ci_lower_hf_only"] = 0.0
    sub["ci_upper_hf_only"] = sub["point_estimate_hf_only"] + sub["half_width_hf_only"]

    return sub


# ----------------------------------------------------------
# ACV point estimate for gap vs n, grouped by (m, M)
# ----------------------------------------------------------
for M, subM in df.groupby("M"):
    fig, ax = make_standard_figure()
    for m, sub in subM.groupby("m"):
        sub = sub.sort_values("n")
        ax.plot(
            sub["n"],
            sub["point_estimate"],
            marker="o",
            color=COLOR_ACV,
            label=rf"$\bar F^{{\mathrm{{ACV}}}}(\hat{{x}},m,M)$, $m={m},\,M={M}$",
        )

    ax.axhline(
        true_gap,
        color=COLOR_TRUE,
        linestyle="--",
        label=rf"True optimality gap $\Delta_f(\hat{{x}})$",
    )
    apply_grid(ax)
    finalize_standard_plot(
        fig, ax,
        xlabel=r"Sample size $n$",
        ylabel=r"Multifidelity point estimate",
        title=rf"Multifidelity point estimate $\bar{{F}}^\mathrm{{ACV}}$ versus $n$ for $M={M}$",
    )
    fig.savefig(output_dir / f"acv_point_estimate_vs_n_M_{M}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# ----------------------------------------------------------
# ACV confidence interval upper bound vs n
# ----------------------------------------------------------
for M, subM in df.groupby("M"):
    fig, ax = make_standard_figure()
    for m, sub in subM.groupby("m"):
        sub = sub.sort_values("n")
        ax.plot(
            sub["n"],
            sub["ci_upper"],
            marker="o",
            color=COLOR_ACV,
            label=rf"Paired reps $m={m}$, additional reps $M={M}$",
        )

    ax.axhline(
        true_gap,
        color=COLOR_TRUE,
        linestyle="--",
        label=rf"True optimality gap $\Delta_f(\hat{{x}})$",
    )
    apply_grid(ax)
    finalize_standard_plot(
        fig, ax,
        xlabel=r"Sample size $n$",
        ylabel=r"Upper confidence bound",
        title=rf"Multifidelity upper confidence bound versus $n$ for $M={M}$",
    )
    fig.savefig(output_dir / f"acv_ci_upper_vs_n_M_{M}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# ----------------------------------------------------------
# For fixed (m, M), compare ACV and HF-only point estimates versus n
# ----------------------------------------------------------
for (m, M), sub in df.groupby(["m", "M"]):
    sub = sub.sort_values("n").copy()

    fig, ax = make_standard_figure()

    ax.plot(
        sub["n"],
        sub["point_estimate_hf_only"],
        marker="o",
        linestyle="--",
        color=COLOR_HF,
        label=rf"HF-only point estimate $\bar{{F}}_n^m(\hat{{x}})$",
    )

    ax.plot(
        sub["n"],
        sub["point_estimate"],
        marker="s",
        linestyle="--",
        color=COLOR_ACV,
        label=rf"Multifidelity point estimate $\bar{{F}}^{{\mathrm{{ACV}}}}(\hat{{x}},m,M)$",
    )

    ax.axhline(
        true_gap,
        color=COLOR_TRUE,
        linestyle="--",
        label=rf"True optimality gap $\Delta_f(\hat{{x}})$",
    )

    apply_grid(ax)
    finalize_standard_plot(
        fig, ax,
        xlabel=r"Sample size $n$",
        ylabel=r"Point estimate",
        title=rf"Multifidelity versus HF-only point estimates for fixed $m={m}$ and $M={M}$",
    )
    fig.savefig(output_dir / f"compare_acv_hf_point_estimates_fixed_m_{m}_M_{M}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# ----------------------------------------------------------
# For fixed (m, M), compare ACV and HF-only upper bounds versus n
# ----------------------------------------------------------
for (m, M), sub in df.groupby(["m", "M"]):
    sub = sub.sort_values("n").copy()

    sub["standard_error_hf_only"] = np.sqrt(sub["sample_variance_F"] / sub["m"])
    sub["t_statistic_hf_only"] = stats.t.ppf(1.0 - sub["alpha"].iloc[0], df=m - 1)
    sub["half_width_hf_only"] = sub["t_statistic_hf_only"] * sub["standard_error_hf_only"]
    sub["ci_upper_hf_only"] = sub["point_estimate_hf_only"] + sub["half_width_hf_only"]

    fig, ax = make_standard_figure()

    ax.plot(
        sub["n"],
        sub["ci_upper_hf_only"],
        marker="o",
        linestyle=":",
        color=COLOR_HF,
        label=rf"HF-only upper bound $U_f^{{\mathrm{{HF}}}}$",
    )

    ax.plot(
        sub["n"],
        sub["ci_upper"],
        marker="s",
        linestyle=":",
        color=COLOR_ACV,
        label=rf"Multifidelity upper bound $U_f^{{\mathrm{{ACV}}}}$",
    )

    ax.axhline(
        true_gap,
        color=COLOR_TRUE,
        linestyle="--",
        label=rf"True optimality gap $\Delta_f(\hat{{x}})$",
    )

    apply_grid(ax)
    finalize_standard_plot(
        fig, ax,
        xlabel=r"Sample size $n$",
        ylabel=r"Upper confidence bound",
        title=rf"Multifidelity versus HF-only upper bounds for fixed $m={m}$ and $M={M}$",
    )
    fig.savefig(output_dir / f"compare_acv_hf_upper_bounds_fixed_m_{m}_M_{M}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# ----------------------------------------------------------
# For fixed (m, M), compare ACV and HF-only versus n
# ----------------------------------------------------------
for (m, M), sub in df.groupby(["m", "M"]):
    sub = sub.sort_values("n").copy()

    sub["standard_error_hf_only"] = np.sqrt(sub["sample_variance_F"] / sub["m"])
    sub["t_statistic_hf_only"] = stats.t.ppf(1.0 - sub["alpha"].iloc[0], df=m - 1)
    sub["half_width_hf_only"] = sub["t_statistic_hf_only"] * sub["standard_error_hf_only"]
    sub["ci_upper_hf_only"] = sub["point_estimate_hf_only"] + sub["half_width_hf_only"]

    fig, ax = make_standard_figure()

    ax.plot(
        sub["n"],
        sub["point_estimate_hf_only"],
        marker="o",
        linestyle="--",
        color=COLOR_HF,
        label=rf"HF-only point estimate $\bar{{F}}_n^m(\hat{{x}})$",
    )
    ax.plot(
        sub["n"],
        sub["ci_upper_hf_only"],
        marker="o",
        linestyle=":",
        color=COLOR_HF,
        label=rf"HF-only upper bound $U_f^{{\mathrm{{HF}}}}$",
    )

    ax.plot(
        sub["n"],
        sub["point_estimate"],
        marker="s",
        linestyle="--",
        color=COLOR_ACV,
        label=rf"Multifidelity point estimate $\bar{{F}}^{{\mathrm{{ACV}}}}(\hat{{x}},m,M)$",
    )
    ax.plot(
        sub["n"],
        sub["ci_upper"],
        marker="s",
        linestyle=":",
        color=COLOR_ACV,
        label=rf"Multifidelity upper bound $U_f^{{\mathrm{{ACV}}}}$",
    )

    ax.axhline(
        true_gap,
        color=COLOR_TRUE,
        linestyle="--",
        label=rf"True optimality gap $\Delta_f(\hat{{x}})$",
    )

    apply_grid(ax)
    finalize_standard_plot(
        fig, ax,
        xlabel=r"Sample size $n$",
        ylabel=r"Gap estimate / upper confidence bound",
        title=rf"Multifidelity versus HF-only for fixed $m={m}$ and $M={M}$",
    )
    fig.savefig(output_dir / f"compare_acv_hf_fixed_m_{m}_M_{M}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# ----------------------------------------------------------
# For fixed (n, m), compare ACV and HF-only versus M
# ----------------------------------------------------------
for (n, m), sub in df.groupby(["n", "m"]):
    sub = sub.sort_values("M").copy()

    sub["standard_error_hf_only"] = np.sqrt(sub["sample_variance_F"] / sub["m"])
    sub["t_statistic_hf_only"] = stats.t.ppf(1.0 - sub["alpha"].iloc[0], df=m - 1)
    sub["half_width_hf_only"] = sub["t_statistic_hf_only"] * sub["standard_error_hf_only"]
    sub["ci_upper_hf_only"] = sub["point_estimate_hf_only"] + sub["half_width_hf_only"]

    fig, ax = make_standard_figure()

    ax.plot(
        sub["M"],
        sub["point_estimate_hf_only"],
        marker="o",
        linestyle="--",
        color=COLOR_HF,
        label=rf"HF-only point estimate $\bar{{F}}_n^m(\hat{{x}})$",
    )
    ax.plot(
        sub["M"],
        sub["ci_upper_hf_only"],
        marker="o",
        linestyle=":",
        color=COLOR_HF,
        label=rf"HF-only upper bound $U_f^{{\mathrm{{HF}}}}$",
    )

    ax.plot(
        sub["M"],
        sub["point_estimate"],
        marker="s",
        linestyle="--",
        color=COLOR_ACV,
        label=rf"Multifidelity point estimate $\bar{{F}}^{{\mathrm{{ACV}}}}(\hat{{x}},m,M)$",
    )
    ax.plot(
        sub["M"],
        sub["ci_upper"],
        marker="s",
        linestyle=":",
        color=COLOR_ACV,
        label=rf"Multifidelity upper bound $U_f^{{\mathrm{{ACV}}}}$",
    )

    ax.axhline(
        true_gap,
        color=COLOR_TRUE,
        linestyle="--",
        label=rf"True optimality gap $\Delta_f(\hat{{x}})$",
    )

    apply_grid(ax)
    finalize_standard_plot(
        fig, ax,
        xlabel=r"Additional low-fidelity replication count $M$",
        ylabel=r"Gap estimate / upper confidence bound",
        title=rf"Effect of increasing $M$ for fixed $n={n}$ and $m={m}$",
    )
    fig.savefig(output_dir / f"compare_acv_hf_fixed_n_{n}_m_{m}_vs_M.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# ----------------------------------------------------------
# Actual estimator variance: ACV vs HF-only
# ----------------------------------------------------------
for M, subM in df.groupby("M"):
    fig, ax = make_standard_figure()

    for m, sub in subM.groupby("m"):
        sub = sub.sort_values("n").copy()

        sub["variance_hf_only_estimator"] = sub["sample_variance_F"] / sub["m"]
        sub["variance_acv_estimator_plot"] = sub["variance_acv_estimator"]

        ax.plot(
            sub["n"],
            sub["variance_hf_only_estimator"],
            marker="o",
            linestyle="--",
            color=COLOR_HF,
            label=rf"HF-only estimator variance, $m={m}$",
        )

        ax.plot(
            sub["n"],
            sub["variance_acv_estimator_plot"],
            marker="s",
            linestyle="-",
            color=COLOR_ACV,
            label=rf"Multifidelity estimator variance, $m={m},\,M={M}$",
        )

    apply_grid(ax)
    finalize_standard_plot(
        fig, ax,
        xlabel=r"Sample size $n$",
        ylabel="Estimator variance",
        title=rf"Estimator variance: HF-only vs multifidelity with additional $M={M}$",
    )
    fig.savefig(output_dir / f"variance_acv_vs_hf_only_M_{M}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# ----------------------------------------------------------
# ACV point estimate vs HF-only point estimate
# ----------------------------------------------------------
for M, subM in df.groupby("M"):
    fig, ax = make_standard_figure()
    for m, sub in subM.groupby("m"):
        sub = sub.sort_values("n")

        ax.plot(
            sub["n"],
            sub["point_estimate"],
            marker="o",
            linestyle="-",
            color=COLOR_ACV,
            label=rf"Multifidelity point estimate $\bar F^{{\mathrm{{ACV}}}}(\hat{{x}},m,M)$, $m={m},\,M={M}$",
        )

        ax.plot(
            sub["n"],
            sub["point_estimate_hf_only"],
            marker="s",
            linestyle="--",
            color=COLOR_HF,
            label=rf"HF-only point estimate $\bar F_n^m(\hat{{x}})$, $m={m}$",
        )

    ax.axhline(
        true_gap,
        color=COLOR_TRUE,
        linestyle=":",
        label=rf"True optimality gap $\Delta_f(\hat{{x}})$",
    )
    apply_grid(ax)
    finalize_standard_plot(
        fig, ax,
        xlabel=r"Sample size $n$",
        ylabel=r"Point estimate",
        title=rf"$Multifidelity \bar F^{{\mathrm{{ACV}}}}(\hat{{x}},m,M)$ and HF-only $\bar F_n^m(\hat{{x}})$ Point Estimates versus $n$ for $M={M}$",
    )
    fig.savefig(output_dir / f"acv_vs_hf_point_estimate_M_{M}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# ----------------------------------------------------------
# Standard error: ACV vs HF-only, normalized by true gap
# ----------------------------------------------------------
for M, subM in df.groupby("M"):
    fig, ax = make_standard_figure()

    for m, sub in subM.groupby("m"):
        sub = sub.sort_values("n").copy()

        sub["standard_error_hf_only"] = np.sqrt(sub["sample_variance_F"] / sub["m"])
        sub["standard_error_hf_only_pct_true_gap"] = (
            100.0 * sub["standard_error_hf_only"] / true_gap
        )
        sub["standard_error_acv_pct_true_gap"] = (
            100.0 * sub["standard_error_acv"] / true_gap
        )

        ax.plot(
            sub["n"],
            sub["standard_error_hf_only_pct_true_gap"],
            marker="o",
            linestyle="--",
            color=COLOR_HF,
            label=rf"HF-only, $m={m}$",
        )

        ax.plot(
            sub["n"],
            sub["standard_error_acv_pct_true_gap"],
            marker="s",
            linestyle="-",
            color=COLOR_ACV,
            label=rf"Multifidelity, $m={m},\,M={M}$",
        )

    apply_grid(ax)
    finalize_standard_plot(
        fig, ax,
        xlabel=r"Sample size $n$",
        ylabel=r"$100 \times \widehat{\operatorname{SE}} / \Delta_f(\hat{x})$",
        title=rf"Estimated standard error relative to true gap versus $n$ for $M={M}$",
    )
    fig.savefig(
        output_dir / f"standard_error_acv_vs_hf_normalized_by_true_gap_M_{M}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

# ----------------------------------------------------------
# Sample correlation versus n
# ----------------------------------------------------------
for M, subM in df.groupby("M"):
    fig, ax = make_standard_figure()
    for m, sub in subM.groupby("m"):
        sub = sub.sort_values("n")

        ax.plot(
            sub["n"],
            sub["sample_correlation"],
            marker="o",
            color=COLOR_ACV,
            label=rf"$\hat\rho_{{fg}}$, $m={m},\,M={M}$",
        )

    apply_grid(ax)
    finalize_standard_plot(
        fig, ax,
        xlabel=r"Sample size $n$",
        ylabel=r"Estimated sample correlation $\hat\rho_{fg}$",
        title=rf"Estimated sample correlation versus $n$ for $M={M}$",
    )
    fig.savefig(output_dir / f"sample_correlation_vs_n_M_{M}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# ----------------------------------------------------------
# Estimated control variate coefficient versus n
# ----------------------------------------------------------
for M, subM in df.groupby("M"):
    fig, ax = make_standard_figure()
    for m, sub in subM.groupby("m"):
        sub = sub.sort_values("n")

        ax.plot(
            sub["n"],
            sub["control_variate_coefficient"],
            marker="o",
            color=COLOR_ACV,
            label=rf"$\hat\alpha$, $m={m},\,M={M}$",
        )

    apply_grid(ax)
    finalize_standard_plot(
        fig, ax,
        xlabel=r"Sample size $n$",
        ylabel=r"Estimated control variate coefficient $\hat\alpha$",
        title=rf"Estimated control variate coefficient $\hat\alpha$ versus $n$ for $M={M}$",
    )
    fig.savefig(output_dir / f"alpha_hat_vs_n_M_{M}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# ----------------------------------------------------------
# For fixed (m, M), effect of increasing n:
#     ACV point estimate and one-sided confidence interval
# ----------------------------------------------------------
for (m, M), sub in df.groupby(["m", "M"]):
    sub = sub.sort_values("n")

    fig, ax = make_standard_figure()
    ax.plot(
        sub["n"],
        sub["point_estimate"],
        marker="o",
        color=COLOR_ACV,
        label=rf"Point estimate $\bar{{F}}^{{\mathrm{{ACV}}}}(\hat{{x}},m,M)$",
    )
    ax.plot(
        sub["n"],
        sub["ci_upper"],
        marker="s",
        color=COLOR_ACV,
        linestyle="--",
        label=rf"Upper bound $\bar{{F}}^{{\mathrm{{ACV}}}}(\hat{{x}},m,M)+\epsilon_f^{{\mathrm{{ACV}}}}$",
    )
    ax.axhline(
        true_gap,
        color=COLOR_TRUE,
        linestyle="--",
        label=rf"True optimality gap $\Delta_f(\hat{{x}})$",
    )

    ax.fill_between(
        sub["n"],
        sub["ci_lower"],
        sub["ci_upper"],
        alpha=0.2,
        color=COLOR_ACV,
        label=r"One-sided confidence interval",
    )

    y_axis_rescale = (max(sub["ci_upper"]) - true_gap) / 6
    ax.set_ylim(true_gap - y_axis_rescale, max(sub["ci_upper"]) + y_axis_rescale)

    apply_grid(ax)
    finalize_standard_plot(
        fig, ax,
        xlabel=r"Sample size $n$",
        ylabel=r"Gap estimate / confidence bound",
        title=rf"Effect of increasing $n$ for fixed number of replications $m={m}$ and $M={M}$",
    )
    fig.savefig(output_dir / f"fixed_m_{m}_M_{M}_effect_of_n.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# ----------------------------------------------------------
# Upper-bound distance above the true gap
# ----------------------------------------------------------
for M, subM in df.groupby("M"):
    fig, ax = make_standard_figure()

    for m, sub in subM.groupby("m"):
        sub = sub.sort_values("n").copy()

        sub["standard_error_hf_only"] = np.sqrt(sub["sample_variance_F"] / sub["m"])
        sub["t_statistic_hf_only"] = stats.t.ppf(1.0 - sub["alpha"].iloc[0], df=m - 1)
        sub["half_width_hf_only"] = sub["t_statistic_hf_only"] * sub["standard_error_hf_only"]
        sub["ci_upper_hf_only"] = sub["point_estimate_hf_only"] + sub["half_width_hf_only"]

        sub["hf_distance_above_true_gap"] = sub["ci_upper_hf_only"] - true_gap
        sub["acv_distance_above_true_gap"] = sub["ci_upper"] - true_gap

        ax.plot(
            sub["n"],
            sub["hf_distance_above_true_gap"],
            marker="o",
            linestyle="--",
            color=COLOR_HF,
            label=rf"$U_f^{{\mathrm{{HF}}}} - \Delta_f(\hat{{x}})$, $m={m}$",
        )

        ax.plot(
            sub["n"],
            sub["acv_distance_above_true_gap"],
            marker="s",
            linestyle="-",
            color=COLOR_ACV,
            label=rf"$U_f^{{\mathrm{{ACV}}}} - \Delta_f(\hat{{x}})$, $m={m},\,M={M}$",
        )

    ax.axhline(0.0, color=COLOR_TRUE, linestyle=":")

    apply_grid(ax)
    finalize_standard_plot(
        fig, ax,
        xlabel=r"Sample size $n$",
        ylabel=r"Distance above true gap",
        title=rf"Upper-bound conservativeness versus $n$ for $M={M}$",
    )
    fig.savefig(output_dir / f"upper_bound_distance_above_true_gap_M_{M}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# ----------------------------------------------------------
# For fixed (m, M), show HF-only and ACV one-sided confidence intervals
# as horizontal segments for each n
# ----------------------------------------------------------
for (m, M), sub in df.groupby(["m", "M"]):
    sub = sub.sort_values("n").copy()
    sub = add_hf_only_interval_columns(sub)

    y_positions = np.arange(len(sub))
    offset = 0.10
    cap_half_height = 0.08
    x_left, x_right = compute_axis_limits_from_intervals(
        sub,
        acv_upper_col="ci_upper",
        hf_upper_col="ci_upper_hf_only",
        hf_point_col="point_estimate_hf_only",
        acv_point_col="point_estimate",
        true_gap=true_gap,
    )

    fig, ax = make_interval_figure(len(sub))

    for y, (_, row) in zip(y_positions, sub.iterrows()):
        draw_horizontal_interval(
            ax,
            y=y + offset,
            ci_lower=row["ci_lower_hf_only"],
            ci_upper=row["ci_upper_hf_only"],
            point_estimate=row["point_estimate_hf_only"],
            x_left=x_left,
            line_color=COLOR_HF,
            marker="o",
            marker_color=COLOR_HF,
            marker_size=8,
            alpha=0.9,
            cap_half_height=cap_half_height,
        )

        draw_horizontal_interval(
            ax,
            y=y - offset,
            ci_lower=row["ci_lower"],
            ci_upper=row["ci_upper"],
            point_estimate=row["point_estimate"],
            x_left=x_left,
            line_color=COLOR_ACV,
            marker="o",
            marker_color=COLOR_ACV,
            marker_size=8,
            alpha=0.9,
            cap_half_height=cap_half_height,
        )

    legend_handles = [
        Line2D([0], [0], color=COLOR_HF, lw=2, label="HF-only confidence interval"),
        Line2D([0], [0], marker="o", color=COLOR_HF, lw=0, markersize=8,
               label=rf"HF-only point estimate $\bar{{F}}_n^m(\hat{{x}})$"),
        Line2D([0], [0], color=COLOR_ACV, lw=2, label="Multifidelity confidence interval"),
        Line2D([0], [0], marker="o", color=COLOR_ACV, lw=0, markersize=8,
               label=rf"Multifidelity point estimate $\bar{{F}}^{{\mathrm{{ACV}}}}(\hat{{x}},m,M)$"),
        Line2D([0], [0], color=COLOR_TRUE, lw=1.5, linestyle="--",
               label=rf"True optimality gap $\Delta_f(\hat{{x}})$"),
    ]

    finalize_interval_plot(
        fig, ax,
        true_gap=true_gap,
        x_left=x_left,
        x_right=x_right,
        y_positions=y_positions,
        ytick_labels=[rf"$n={int(n)}$" for n in sub["n"]],
        xlabel=r"Gap estimate / confidence interval",
        ylabel=r"Sample size",
        title=rf"HF-only vs Multifidelity one-sided confidence intervals for fixed $m={m}$ and $M={M}$",
        legend_handles=legend_handles,
    )

    fig.savefig(output_dir / f"confidence_intervals_compare_fixed_m_{m}_M_{M}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# ----------------------------------------------------------
# Alternate black-and-white version:
# For fixed (m, M), compare HF-only and ACV one-sided confidence intervals
# with small vertical offsets for each n
# ----------------------------------------------------------
for (m, M), sub in df.groupby(["m", "M"]):
    sub = sub.sort_values("n").copy()
    sub = add_hf_only_interval_columns(sub)

    y_positions = np.arange(len(sub))
    offset = 0.10
    cap_half_height = 0.08
    x_left, x_right = compute_axis_limits_from_intervals(
        sub,
        acv_upper_col="ci_upper",
        hf_upper_col="ci_upper_hf_only",
        hf_point_col="point_estimate_hf_only",
        acv_point_col="point_estimate",
        true_gap=true_gap,
    )

    fig, ax = make_interval_figure(len(sub))

    for y, (_, row) in zip(y_positions, sub.iterrows()):
        # In black-and-white, line style and marker style distinguish HF-only vs ACV.
        draw_horizontal_interval(
            ax,
            y=y + offset,
            ci_lower=row["ci_lower_hf_only"],
            ci_upper=row["ci_upper_hf_only"],
            point_estimate=row["point_estimate_hf_only"],
            x_left=x_left,
            line_color="black",
            line_style="-",
            marker="o",
            marker_color="black",
            marker_size=8,
            alpha=0.95,
            cap_half_height=cap_half_height,
        )

        draw_horizontal_interval(
            ax,
            y=y - offset,
            ci_lower=row["ci_lower"],
            ci_upper=row["ci_upper"],
            point_estimate=row["point_estimate"],
            x_left=x_left,
            line_color="black",
            line_style="--",
            marker="s",
            marker_color="black",
            marker_size=7,
            alpha=0.95,
            cap_half_height=cap_half_height,
        )

    legend_handles = [
        Line2D([0], [0], color="black", lw=2, linestyle="-", label="HF-only confidence interval"),
        Line2D([0], [0], marker="o", color="black", lw=0, markersize=8,
               label=rf"HF-only point estimate $\bar{{F}}_n^m(\hat{{x}})$"),
        Line2D([0], [0], color="black", lw=2, linestyle="--", label="Multifidelity confidence interval"),
        Line2D([0], [0], marker="s", color="black", lw=0, markersize=7,
               label=rf"Multifidelity point estimate $\bar{{F}}^{{\mathrm{{ACV}}}}(\hat{{x}},m,M)$"),
        Line2D([0], [0], color="black", lw=1.5, linestyle=":",
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
        xlabel=r"Gap estimate / confidence interval",
        ylabel=r"Sample size",
        title=rf"HF-only vs Multifidelity one-sided confidence intervals for fixed $m={m}$ and $M={M}$",
        legend_handles=legend_handles,
    )

    fig.savefig(
        output_dir / f"bw_offset_confidence_intervals_fixed_m_{m}_M_{M}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)