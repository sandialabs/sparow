import sys
import numpy as np
from pathlib import Path
from scipy import stats

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

# ----------------------------------------------------------
# ACV point estimator for gap vs n, grouped by (m, M)
# ----------------------------------------------------------
for M, subM in df.groupby("M"):
    plt.figure()
    for m, sub in subM.groupby("m"):
        sub = sub.sort_values("n")
        plt.plot(
            sub["n"],
            sub["point_estimate"],
            marker="o",
            label=rf"$\bar F^{{\mathrm{{ACV}}}}(\hat{{x}},m,M)$, $m={m},\,M={M}$",
        )

    plt.axhline(
        true_gap,
        color="black",
        linestyle="--",
        label=rf"True optimality gap $\Delta_f(\hat{{x}})$",
    )
    plt.grid()
    plt.xlabel(r"Sample size $n$")
    plt.ylabel(r"ACV point estimator")
    plt.title(rf"ACV point estimator $\bar{{F}}^\mathrm{{ACV}}$ versus $n$ for $M={M}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"acv_point_estimate_vs_n_M_{M}.png", dpi=200)
    plt.close()

# ----------------------------------------------------------
# ACV Confidence Interval upper bound vs n
# ----------------------------------------------------------
for M, subM in df.groupby("M"):
    plt.figure()
    for m, sub in subM.groupby("m"):
        sub = sub.sort_values("n")
        plt.plot(
            sub["n"],
            sub["ci_upper"],
            marker="o",
            label=rf"paired reps $m={m},\,$ additional reps $M={M}$",
        )

    plt.axhline(
        true_gap,
        color="black",
        linestyle="--",
        label=rf"True optimality gap $\Delta_f(\hat{{x}})$",
    )
    plt.grid()
    plt.xlabel(r"Sample size $n$")
    plt.ylabel(r"Upper CI bound")
    plt.title(rf"ACV upper confidence bound versus $n$ for $M={M}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"acv_ci_upper_vs_n_M_{M}.png", dpi=200)
    plt.close()

# ----------------------------------------------------------
# For fixed (m, M), compare ACV and HF-only point estimates versus n
# ----------------------------------------------------------
for (m, M), sub in df.groupby(["m", "M"]):
    sub = sub.sort_values("n").copy()

    plt.figure()

    plt.plot(
        sub["n"],
        sub["point_estimate_hf_only"],
        marker="o",
        linestyle="--",
        color="blue",
        label=rf"HF-only point estimator $\bar{{F}}_n^m(\hat{{x}})$",
    )

    plt.plot(
        sub["n"],
        sub["point_estimate"],
        marker="s",
        linestyle="--",
        color="green",
        label=rf"ACV point estimator $\bar{{F}}^{{\mathrm{{ACV}}}}(\hat{{x}},m,M)$",
    )

    plt.axhline(
        true_gap,
        color="black",
        linestyle="--",
        label=rf"True optimality gap $\Delta_f(\hat{{x}})$",
    )

    plt.grid()
    plt.xlabel(r"Sample size $n$")
    plt.ylabel(r"Point estimate")
    plt.title(rf"ACV-MRP versus HF-only point estimates for fixed $m={m}$ and $M={M}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"compare_acv_hf_point_estimates_fixed_m_{m}_M_{M}.png", dpi=200)
    plt.close()

# ----------------------------------------------------------
# For fixed (m, M), compare ACV and HF-only upper bounds versus n
# ----------------------------------------------------------
for (m, M), sub in df.groupby(["m", "M"]):
    sub = sub.sort_values("n").copy()

    # Reconstruct HF-only upper bound using the t-statistic
    sub["standard_error_hf_only"] = np.sqrt(sub["sample_variance_F"] / sub["m"])
    sub["t_statistic_hf_only"] = stats.t.ppf(1.0 - sub["alpha"].iloc[0], df=m - 1)
    sub["half_width_hf_only"] = sub["t_statistic_hf_only"] * sub["standard_error_hf_only"]
    sub["ci_upper_hf_only"] = sub["point_estimate_hf_only"] + sub["half_width_hf_only"]

    plt.figure()

    plt.plot(
        sub["n"],
        sub["ci_upper_hf_only"],
        marker="o",
        linestyle=":",
        color="blue",
        label=rf"HF-only upper bound $U_f^{{\mathrm{{HF}}}}$",
    )

    plt.plot(
        sub["n"],
        sub["ci_upper"],
        marker="s",
        linestyle=":",
        color="green",
        label=rf"ACV upper bound $U_f^{{\mathrm{{ACV}}}}$",
    )

    plt.axhline(
        true_gap,
        color="black",
        linestyle="--",
        label=rf"True optimality gap $\Delta_f(\hat{{x}})$",
    )

    plt.grid()
    plt.xlabel(r"Sample size $n$")
    plt.ylabel(r"Upper confidence bound")
    plt.title(rf"ACV-MRP versus HF-only upper bounds for fixed $m={m}$ and $M={M}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"compare_acv_hf_upper_bounds_fixed_m_{m}_M_{M}.png", dpi=200)
    plt.close()

# ----------------------------------------------------------
# For fixed (m, M), compare ACV and HF-only versus n
# ----------------------------------------------------------
for (m, M), sub in df.groupby(["m", "M"]):
    sub = sub.sort_values("n").copy()

    # Reconstruct HF-only upper bound using the t-statistic
    sub["standard_error_hf_only"] = np.sqrt(sub["sample_variance_F"] / sub["m"])
    sub["t_statistic_hf_only"] = stats.t.ppf(1.0 - sub["alpha"].iloc[0], df=m - 1)
    sub["half_width_hf_only"] = sub["t_statistic_hf_only"] * sub["standard_error_hf_only"]
    sub["ci_upper_hf_only"] = sub["point_estimate_hf_only"] + sub["half_width_hf_only"]

    plt.figure()

    plt.plot(
        sub["n"],
        sub["point_estimate_hf_only"],
        marker="o",
        linestyle="--",
        color="blue",
        label=rf"HF-only point estimator $\bar{{F}}_n^m(\hat{{x}})$",
    )
    plt.plot(
        sub["n"],
        sub["ci_upper_hf_only"],
        marker="o",
        linestyle=":",
        color="blue",
        label=rf"HF-only upper bound $U_f^{{\mathrm{{HF}}}}$",
    )

    plt.plot(
        sub["n"],
        sub["point_estimate"],
        marker="s",
        linestyle="--",
        color="green",
        label=rf"ACV point estimator $\bar{{F}}^{{\mathrm{{ACV}}}}(\hat{{x}},m,M)$",
    )
    plt.plot(
        sub["n"],
        sub["ci_upper"],
        marker="s",
        linestyle=":",
        color="green",
        label=rf"ACV upper bound $U_f^{{\mathrm{{ACV}}}}$",
    )

    plt.axhline(
        true_gap,
        color="black",
        linestyle="--",
        label=rf"True optimality gap $\Delta_f(\hat{{x}})$",
    )

    plt.grid()
    plt.xlabel(r"Sample size $n$")
    plt.ylabel(r"Gap estimate / upper confidence bound")
    plt.title(rf"ACV-MRP versus HF-only for fixed $m={m}$ and $M={M}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"compare_acv_hf_fixed_m_{m}_M_{M}.png", dpi=200)
    plt.close()

# ----------------------------------------------------------
# For fixed (n, m), compare ACV and HF-only versus M
# ----------------------------------------------------------
for (n, m), sub in df.groupby(["n", "m"]):
    sub = sub.sort_values("M").copy()

    # Reconstruct HF-only upper bound using the t-statistic
    sub["standard_error_hf_only"] = np.sqrt(sub["sample_variance_F"] / sub["m"])
    sub["t_statistic_hf_only"] = stats.t.ppf(1.0 - sub["alpha"].iloc[0], df=m - 1)
    sub["half_width_hf_only"] = sub["t_statistic_hf_only"] * sub["standard_error_hf_only"]
    sub["ci_upper_hf_only"] = sub["point_estimate_hf_only"] + sub["half_width_hf_only"]

    plt.figure()

    plt.plot(
        sub["M"],
        sub["point_estimate_hf_only"],
        marker="o",
        linestyle="--",
        color="blue",
        label=rf"HF-only point estimator $\bar{{F}}_n^m(\hat{{x}})$",
    )
    plt.plot(
        sub["M"],
        sub["ci_upper_hf_only"],
        marker="o",
        linestyle=":",
        color="blue",
        label=rf"HF-only upper bound $U_f^{{\mathrm{{HF}}}}$",
    )

    plt.plot(
        sub["M"],
        sub["point_estimate"],
        marker="s",
        linestyle="--",
        color="green",
        label=rf"ACV point estimator $\bar{{F}}^{{\mathrm{{ACV}}}}(\hat{{x}},m,M)$",
    )
    plt.plot(
        sub["M"],
        sub["ci_upper"],
        marker="s",
        linestyle=":",
        color="green",
        label=rf"ACV upper bound $U_f^{{\mathrm{{ACV}}}}$",
    )

    plt.axhline(
        true_gap,
        color="black",
        linestyle="--",
        label=rf"True optimality gap $\Delta_f(\hat{{x}})$",
    )

    plt.grid()
    plt.xlabel(r"Additional low-fidelity replication count $M$")
    plt.ylabel(r"Gap estimate / upper confidence bound")
    plt.title(rf"Effect of increasing $M$ for fixed $n={n}$ and $m={m}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"compare_acv_hf_fixed_n_{n}_m_{m}_vs_M.png", dpi=200)
    plt.close()

# ----------------------------------------------------------
# Actual estimator variance: ACV vs HF-only
# ----------------------------------------------------------
for M, subM in df.groupby("M"):
    plt.figure()

    for m, sub in subM.groupby("m"):
        sub = sub.sort_values("n").copy()

        # HF-only estimator variance = Var(F_bar) estimated by s_F^2 / m
        sub["variance_hf_only_estimator"] = sub["sample_variance_F"] / sub["m"]

        # ACV estimator variance already stored in CSV
        sub["variance_acv_estimator_plot"] = sub["variance_acv_estimator"]

        plt.plot(
            sub["n"],
            sub["variance_hf_only_estimator"],
            marker="o",
            linestyle="--",
            label=rf"HF-only estimator variance, $m={m}$",
        )

        plt.plot(
            sub["n"],
            sub["variance_acv_estimator_plot"],
            marker="s",
            linestyle="-",
            label=rf"Multifidelity estimator variance, $m={m},\,M={M}$",
        )

    plt.grid()
    plt.xlabel(r"Sample size $n$")
    plt.ylabel("Estimator variance")
    plt.title(rf"Estimator variance: HF-only vs Multifidelity with additional $M={M}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"variance_acv_vs_hf_only_M_{M}.png", dpi=200)
    plt.close()

# ----------------------------------------------------------
# ACV point estimator vs HF-only point estimator
# ----------------------------------------------------------
for M, subM in df.groupby("M"):
    plt.figure()
    for m, sub in subM.groupby("m"):
        sub = sub.sort_values("n")

        plt.plot(
            sub["n"],
            sub["point_estimate"],
            marker="o",
            linestyle="-",
            label=rf"$\bar F^{{\mathrm{{ACV}}}}(\hat{{x}},m,M)$, $m={m},\,M={M}$",
        )

        plt.plot(
            sub["n"],
            sub["point_estimate_hf_only"],
            marker="s",
            linestyle="--",
            label=rf"$\bar F_n^m(\hat{{x}})$, $m={m}$",
        )

    plt.axhline(
        true_gap,
        color="black",
        linestyle=":",
        label=rf"True optimality gap $\Delta_f(\hat{{x}})$",
    )
    plt.grid()
    plt.xlabel(r"Sample size $n$")
    plt.ylabel(r"Point estimator")
    plt.title(
        rf"$\bar F^{{\mathrm{{ACV}}}}(\hat{{x}},m,M)$ and $\bar F_n^m(\hat{{x}})$ versus $n$ for $M={M}$"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"acv_vs_hf_point_estimate_M_{M}.png", dpi=200)
    plt.close()

# ----------------------------------------------------------
# Standard error: ACV vs HF-only, normalized by true gap
# ----------------------------------------------------------
for M, subM in df.groupby("M"):
    plt.figure()

    for m, sub in subM.groupby("m"):
        sub = sub.sort_values("n").copy()

        sub["standard_error_hf_only"] = np.sqrt(sub["sample_variance_F"] / sub["m"])
        sub["standard_error_hf_only_pct_true_gap"] = (
            100.0 * sub["standard_error_hf_only"] / true_gap
        )
        sub["standard_error_acv_pct_true_gap"] = (
            100.0 * sub["standard_error_acv"] / true_gap
        )

        plt.plot(
            sub["n"],
            sub["standard_error_hf_only_pct_true_gap"],
            marker="o",
            linestyle="--",
            label=rf"HF-only, $m={m}$",
        )

        plt.plot(
            sub["n"],
            sub["standard_error_acv_pct_true_gap"],
            marker="s",
            linestyle="-",
            label=rf"ACV, $m={m},\,M={M}$",
        )

    plt.grid()
    plt.xlabel(r"Sample size $n$")
    plt.ylabel(r"$100 \times \widehat{\operatorname{SE}} / \Delta_f(\hat{x})$")
    plt.title(rf"Estimated standard error relative to true gap versus $n$ for $M={M}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / f"standard_error_acv_vs_hf_normalized_by_true_gap_M_{M}.png",
        dpi=200,
    )
    plt.close()

# ----------------------------------------------------------
# Sample correlation versus n
# ----------------------------------------------------------
for M, subM in df.groupby("M"):
    plt.figure()
    for m, sub in subM.groupby("m"):
        sub = sub.sort_values("n")

        plt.plot(
            sub["n"],
            sub["sample_correlation"],
            marker="o",
            label=rf"$\hat\rho_{{fg}}$, $m={m},\,M={M}$",
        )

    plt.grid()
    plt.xlabel(r"Sample size $n$")
    plt.ylabel(r"Estimated sample correlation $\hat\rho_{fg}$")
    plt.title(rf"Estimated sample correlation versus $n$ for $M={M}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"sample_correlation_vs_n_M_{M}.png", dpi=200)
    plt.close()

# ----------------------------------------------------------
# Estimated control variate coefficient versus n
# ----------------------------------------------------------
for M, subM in df.groupby("M"):
    plt.figure()
    for m, sub in subM.groupby("m"):
        sub = sub.sort_values("n")

        plt.plot(
            sub["n"],
            sub["control_variate_coefficient"],
            marker="o",
            label=rf"$\hat\alpha$, $m={m},\,M={M}$",
        )

    plt.grid()
    plt.xlabel(r"Sample size $n$")
    plt.ylabel(r"Estimated control variate coefficient $\hat\alpha$")
    plt.title(
        rf"Estimated control variate coefficient $\hat\alpha$ versus $n$ for $M={M}$"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"alpha_hat_vs_n_M_{M}.png", dpi=200)
    plt.close()

# ----------------------------------------------------------
# Effect of M for fixed n and m
# ----------------------------------------------------------
plt.figure()

for m, subm in df.groupby("m"):
    for n, submn in subm.groupby("n"):
        submn = submn.sort_values("M")

        plt.plot(
            submn["M"],
            submn["variance_reduction_factor"],
            marker="o",
            label=rf"$m={m},\,n={n}$",
        )

plt.grid()
plt.xlabel(r"Additional low-fidelity replication count $M$")
plt.ylabel(r"Variance reduction factor")
plt.title(r"Variance reduction factor versus $M$")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "variance_reduction_vs_M.png", dpi=200)
plt.close()

# ----------------------------------------------------------
# For fixed (m, M), effect of increasing n:
#     ACV point estimate and one-sided Confidence Interval
# ----------------------------------------------------------
for (m, M), sub in df.groupby(["m", "M"]):
    sub = sub.sort_values("n")

    plt.figure()
    plt.plot(
        sub["n"],
        sub["point_estimate"],
        marker="o",
        label=rf"Point estimator $\bar{{F}}^{{\mathrm{{ACV}}}}(\hat{{x}},m,M)$",
    )
    plt.plot(
        sub["n"],
        sub["ci_upper"],
        marker="s",
        label=rf"Upper bound $\bar{{F}}^{{\mathrm{{ACV}}}}(\hat{{x}},m,M)+\epsilon_f^{{\mathrm{{ACV}}}}$",
    )
    plt.axhline(
        true_gap,
        color="black",
        linestyle="--",
        label=rf"True optimality gap $\Delta_f(\hat{{x}})$",
    )

    plt.fill_between(
        sub["n"],
        sub["ci_lower"],
        sub["ci_upper"],
        alpha=0.2,
        label=r"One-sided confidence interval",
    )

    # Scale y-axis to start just below the true optimal value, end above ci_upper
    y_axis_rescale = (max(sub["ci_upper"]) - true_gap) / 6
    plt.ylim(true_gap - y_axis_rescale, max(sub["ci_upper"]) + y_axis_rescale)

    plt.grid()
    plt.xlabel(r"Sample size $n$")
    plt.ylabel(r"Gap estimate / confidence bound")
    plt.title(
        rf"Effect of increasing $n$ for fixed number of replications $m={m}$ and $M={M}$"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"fixed_m_{m}_M_{M}_effect_of_n.png", dpi=200)
    plt.close()

# ----------------------------------------------------------
# Upper-bound distance above the true gap
# ----------------------------------------------------------
for M, subM in df.groupby("M"):
    plt.figure()

    for m, sub in subM.groupby("m"):
        sub = sub.sort_values("n").copy()

        # Reconstruct HF-only upper bound using the t-statistic
        sub["standard_error_hf_only"] = np.sqrt(sub["sample_variance_F"] / sub["m"])
        sub["t_statistic_hf_only"] = stats.t.ppf(1.0 - sub["alpha"].iloc[0], df=m - 1)
        sub["half_width_hf_only"] = sub["t_statistic_hf_only"] * sub["standard_error_hf_only"]
        sub["ci_upper_hf_only"] = sub["point_estimate_hf_only"] + sub["half_width_hf_only"]

        sub["hf_distance_above_true_gap"] = sub["ci_upper_hf_only"] - true_gap
        sub["acv_distance_above_true_gap"] = sub["ci_upper"] - true_gap

        plt.plot(
            sub["n"],
            sub["hf_distance_above_true_gap"],
            marker="o",
            linestyle="--",
            label=rf"$U_f^{{\mathrm{{HF}}}} - \Delta_f(\hat{{x}})$, $m={m}$",
        )

        plt.plot(
            sub["n"],
            sub["acv_distance_above_true_gap"],
            marker="s",
            linestyle="-",
            label=rf"$U_f^{{\mathrm{{ACV}}}} - \Delta_f(\hat{{x}})$, $m={m},\,M={M}$",
        )

    plt.axhline(0.0, color="black", linestyle=":")

    plt.grid()
    plt.xlabel(r"Sample size $n$")
    plt.ylabel(r"Distance above true gap")
    plt.title(rf"Upper-bound conservativeness versus $n$ for $M={M}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"upper_bound_distance_above_true_gap_M_{M}.png", dpi=200)
    plt.close()

# ==========================================================
# Plotting the confidence intervals for direct comparison:
# Helper functions
# ==========================================================

def add_hf_only_interval_columns(sub):
    """
    Reconstruct the HF-only one-sided confidence interval quantities.
    """
    sub = sub.copy()

    sub["standard_error_hf_only"] = np.sqrt(sub["sample_variance_F"] / sub["m"])

    # Use the t-statistic for the HF-only interval 
    sub["t_statistic_hf_only"] = sub["m"].apply(
        lambda mm: stats.t.ppf(1.0 - sub["alpha"].iloc[0], df=mm - 1)
    )
    sub["half_width_hf_only"] = (sub["t_statistic_hf_only"] * sub["standard_error_hf_only"])
    sub["ci_lower_hf_only"] = 0.0
    sub["ci_upper_hf_only"] = sub["point_estimate_hf_only"] + sub["half_width_hf_only"]

    return sub


def compute_interval_axis_limits(sub, true_gap, pad_fraction=0.08, default_pad=1.0):
    """
    Compute x-axis limits that focus on the informative upper ends of the intervals.
    """
    min_point = min(
        sub["point_estimate"].min(),
        sub["point_estimate_hf_only"].min(),
        true_gap,
    )
    max_upper = max(sub["ci_upper"].max(), sub["ci_upper_hf_only"].max())

    x_span = max_upper - min_point
    x_pad = pad_fraction * x_span if x_span > 0 else default_pad

    x_left = min_point - x_pad
    x_right = max_upper + x_pad

    return x_left, x_right


def draw_horizontal_interval(
    ax,
    *,
    y,
    ci_lower,
    ci_upper,
    point_estimate,
    x_left,
    line_color,
    line_style="-",
    line_width=2,
    marker="o",
    marker_color=None,
    marker_size=6,
    alpha=0.9,
    cap_half_height=0.10,
):
    """
    Draw one horizontal one-sided confidence interval with end caps and a point marker.
    """
    visible_left = max(ci_lower, x_left)
    visible_right = ci_upper

    ax.hlines(
        y=y,
        xmin=visible_left,
        xmax=visible_right,
        color=line_color,
        linestyle=line_style,
        linewidth=line_width,
        alpha=alpha,
    )

    # Only draw the left cap if that endpoint is actually visible.
    if ci_lower >= x_left:
        ax.vlines(
            x=ci_lower,
            ymin=y - cap_half_height,
            ymax=y + cap_half_height,
            color=line_color,
            linestyle=line_style,
            linewidth=line_width,
            alpha=alpha,
        )

    # Always draw the right cap
    ax.vlines(
        x=visible_right,
        ymin=y - cap_half_height,
        ymax=y + cap_half_height,
        color=line_color,
        linestyle=line_style,
        linewidth=line_width,
        alpha=alpha,
    )

    ax.plot(
        point_estimate,
        y,
        marker=marker,
        color=marker_color if marker_color is not None else line_color,
        markersize=marker_size,
    )


def add_interval_comparison_legend(ax, *, bw=False):
    """
    Add the legend for interval-comparison plots outside the axes.
    Putting the legend outside prevents it from covering the interval endpoints.
    """
    if bw:
        legend_handles = [
            Line2D([0], [0], color="black", lw=2, linestyle="-", marker="o", markersize=6,
                   label=rf"HF-only interval and point estimator $\bar{{F}}_n^m(\hat{{x}})$"),
            Line2D([0], [0], color="black", lw=2, linestyle="--", marker="s", markersize=5,
                   label=rf"ACV interval and point estimator $\bar{{F}}^{{\mathrm{{ACV}}}}(\hat{{x}},m,M)$"),
            Line2D([0], [0], color="black", lw=1.5, linestyle=":",
                   label=rf"True optimality gap $\Delta_f(\hat{{x}})$"),
        ]
    else:
        legend_handles = [
            Line2D([0], [0], color="steelblue", lw=2, label="HF-only confidence interval"),
            Line2D([0], [0], marker="o", color="navy", lw=0, markersize=6,
                   label=rf"HF-only point estimator $\bar{{F}}_n^m(\hat{{x}})$"),
            Line2D([0], [0], color="darkgreen", lw=2, label="ACV confidence interval"),
            Line2D([0], [0], marker="o", color="green", lw=0, markersize=6,
                   label=rf"ACV point estimator $\bar{{F}}^{{\mathrm{{ACV}}}}(\hat{{x}},m,M)$"),
            Line2D([0], [0], color="black", lw=1.5, linestyle="--",
                   label=rf"True optimality gap $\Delta_f(\hat{{x}})$"),
        ]

    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )


def finalize_interval_plot(
    ax,
    *,
    true_gap,
    x_left,
    x_right,
    y_positions,
    ytick_labels,
    xlabel,
    ylabel,
    title,
    bw=False,
):
    """
    Apply the shared formatting for interval-comparison plots.
    """
    ax.axvline(
        true_gap,
        color="black",
        linestyle=":" if bw else "--",
        linewidth=1.5,
    )
    ax.set_xlim(left=x_left, right=x_right)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ytick_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)

    add_interval_comparison_legend(ax, bw=bw)

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
    x_left, x_right = compute_interval_axis_limits(sub, true_gap)

    fig, ax = plt.subplots(figsize=(10, 4 + 0.55 * len(sub)))

    for y, (_, row) in zip(y_positions, sub.iterrows()):
        # Separate vertical offsets keep HF-only and ACV visible on the same row.
        draw_horizontal_interval(
            ax,
            y=y + offset,
            ci_lower=row["ci_lower_hf_only"],
            ci_upper=row["ci_upper_hf_only"],
            point_estimate=row["point_estimate_hf_only"],
            x_left=x_left,
            line_color="steelblue",
            marker="o",
            marker_color="navy",
            marker_size=6,
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
            line_color="darkgreen",
            marker="o",
            marker_color="green",
            marker_size=6,
            alpha=0.9,
            cap_half_height=cap_half_height,
        )

    finalize_interval_plot(
        ax,
        true_gap=true_gap,
        x_left=x_left,
        x_right=x_right,
        y_positions=y_positions,
        ytick_labels=[rf"$n={int(n)}$" for n in sub["n"]],
        xlabel=r"Gap estimate / confidence interval",
        ylabel=r"Sample size",
        title=rf"HF-only vs ACV one-sided confidence intervals for fixed $m={m}$ and $M={M}$",
        bw=False,
    )

    plt.tight_layout(rect=[0.10, 0, 0.82, 1])
    plt.savefig(output_dir / f"confidence_intervals_compare_fixed_m_{m}_M_{M}.png", dpi=200)
    plt.close()

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
    x_left, x_right = compute_interval_axis_limits(sub, true_gap)

    fig, ax = plt.subplots(figsize=(10, 4 + 0.55 * len(sub)))

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
            marker_size=6,
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
            marker_size=5,
            alpha=0.95,
            cap_half_height=cap_half_height,
        )

    finalize_interval_plot(
        ax,
        true_gap=true_gap,
        x_left=x_left,
        x_right=x_right,
        y_positions=y_positions,
        ytick_labels=[rf"$n={int(n)}$" for n in sub["n"]],
        xlabel=r"Gap estimate / confidence interval",
        ylabel=r"Sample size",
        title=rf"HF-only vs ACV one-sided confidence intervals for fixed $m={m}$ and $M={M}$",
        bw=True,
    )

    plt.tight_layout(rect=[0.10, 0, 0.82, 1])
    plt.savefig(output_dir / f"bw_offset_confidence_intervals_fixed_m_{m}_M_{M}.png", dpi=200)
    plt.close()

# ----------------------------------------------------------
# For fixed (n, m), show HF-only and ACV one-sided confidence intervals
# as horizontal segments for each M
# ----------------------------------------------------------
for (n, m), sub in df.groupby(["n", "m"]):
    sub = sub.sort_values("M").copy()
    sub = add_hf_only_interval_columns(sub)

    y_positions = np.arange(len(sub))
    offset = 0.10
    cap_half_height = 0.08
    x_left, x_right = compute_interval_axis_limits(sub, true_gap)

    fig, ax = plt.subplots(figsize=(10, 4 + 0.55 * len(sub)))

    for y, (_, row) in zip(y_positions, sub.iterrows()):
        draw_horizontal_interval(
            ax,
            y=y + offset,
            ci_lower=row["ci_lower_hf_only"],
            ci_upper=row["ci_upper_hf_only"],
            point_estimate=row["point_estimate_hf_only"],
            x_left=x_left,
            line_color="steelblue",
            marker="o",
            marker_color="navy",
            marker_size=6,
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
            line_color="darkgreen",
            marker="o",
            marker_color="green",
            marker_size=6,
            alpha=0.9,
            cap_half_height=cap_half_height,
        )

    finalize_interval_plot(
        ax,
        true_gap=true_gap,
        x_left=x_left,
        x_right=x_right,
        y_positions=y_positions,
        ytick_labels=[rf"$M={int(M)}$" for M in sub["M"]],
        xlabel=r"Gap estimate / confidence interval",
        ylabel=r"Additional LF replications",
        title=rf"HF-only vs ACV one-sided confidence intervals for fixed $n={n}$ and $m={m}$",
        bw=False,
    )

    plt.tight_layout(rect=[0.10, 0, 0.82, 1])
    plt.savefig(output_dir / f"confidence_intervals_compare_fixed_n_{n}_m_{m}.png", dpi=200)
    plt.close()

# ----------------------------------------------------------
# Alternate black-and-white version:
# For fixed (n, m), compare HF-only and ACV one-sided confidence intervals
# with small vertical offsets for each M
# ----------------------------------------------------------
for (n, m), sub in df.groupby(["n", "m"]):
    sub = sub.sort_values("M").copy()
    sub = add_hf_only_interval_columns(sub)

    y_positions = np.arange(len(sub))
    offset = 0.10
    cap_half_height = 0.08
    x_left, x_right = compute_interval_axis_limits(sub, true_gap)

    fig, ax = plt.subplots(figsize=(10, 4 + 0.55 * len(sub)))

    for y, (_, row) in zip(y_positions, sub.iterrows()):
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
            marker_size=6,
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
            marker_size=5,
            alpha=0.95,
            cap_half_height=cap_half_height,
        )

    finalize_interval_plot(
        ax,
        true_gap=true_gap,
        x_left=x_left,
        x_right=x_right,
        y_positions=y_positions,
        ytick_labels=[rf"$M={int(M)}$" for M in sub["M"]],
        xlabel=r"Gap estimate / confidence interval",
        ylabel=r"Additional LF replications",
        title=rf"HF-only vs ACV one-sided confidence intervals for fixed $n={n}$ and $m={m}$",
        bw=True,
    )

    plt.tight_layout(rect=[0.10, 0, 0.82, 1])
    plt.savefig(output_dir / f"bw_offset_confidence_intervals_fixed_n_{n}_m_{m}.png", dpi=200)
    plt.close()

# ----------------------------------------------------------
# For fixed (n, M), show HF-only and ACV one-sided confidence intervals
# as horizontal segments for each m
# ----------------------------------------------------------
for (n, M), sub in df.groupby(["n", "M"]):
    sub = sub.sort_values("m").copy()
    sub = add_hf_only_interval_columns(sub)

    y_positions = np.arange(len(sub))
    offset = 0.10
    cap_half_height = 0.08
    x_left, x_right = compute_interval_axis_limits(sub, true_gap)

    fig, ax = plt.subplots(figsize=(10, 4 + 0.55 * len(sub)))

    for y, (_, row) in zip(y_positions, sub.iterrows()):
        draw_horizontal_interval(
            ax,
            y=y + offset,
            ci_lower=row["ci_lower_hf_only"],
            ci_upper=row["ci_upper_hf_only"],
            point_estimate=row["point_estimate_hf_only"],
            x_left=x_left,
            line_color="steelblue",
            marker="o",
            marker_color="navy",
            marker_size=6,
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
            line_color="darkgreen",
            marker="o",
            marker_color="green",
            marker_size=6,
            alpha=0.9,
            cap_half_height=cap_half_height,
        )

    finalize_interval_plot(
        ax,
        true_gap=true_gap,
        x_left=x_left,
        x_right=x_right,
        y_positions=y_positions,
        ytick_labels=[rf"$m={int(m)}$" for m in sub["m"]],
        xlabel=r"Gap estimate / confidence interval",
        ylabel=r"Number of paired replications",
        title=rf"HF-only vs ACV one-sided confidence intervals for fixed $n={n}$ and $M={M}$",
        bw=False,
    )

    plt.tight_layout(rect=[0.10, 0, 0.82, 1])
    plt.savefig(output_dir / f"confidence_intervals_compare_fixed_n_{n}_M_{M}.png", dpi=200)
    plt.close()

# ----------------------------------------------------------
# Alternate black-and-white version:
# For fixed (n, M), compare HF-only and ACV one-sided confidence intervals
# with small vertical offsets for each m
# ----------------------------------------------------------
for (n, M), sub in df.groupby(["n", "M"]):
    sub = sub.sort_values("m").copy()
    sub = add_hf_only_interval_columns(sub)

    y_positions = np.arange(len(sub))
    offset = 0.10
    cap_half_height = 0.08
    x_left, x_right = compute_interval_axis_limits(sub, true_gap)

    fig, ax = plt.subplots(figsize=(10, 4 + 0.55 * len(sub)))

    for y, (_, row) in zip(y_positions, sub.iterrows()):
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
            marker_size=6,
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
            marker_size=5,
            alpha=0.95,
            cap_half_height=cap_half_height,
        )

    finalize_interval_plot(
        ax,
        true_gap=true_gap,
        x_left=x_left,
        x_right=x_right,
        y_positions=y_positions,
        ytick_labels=[rf"$m={int(m)}$" for m in sub["m"]],
        xlabel=r"Gap estimate / confidence interval",
        ylabel=r"Number of paired replications",
        title=rf"HF-only vs ACV one-sided confidence intervals for fixed $n={n}$ and $M={M}$",
        bw=True,
    )

    plt.tight_layout(rect=[0.10, 0, 0.82, 1])
    plt.savefig(output_dir / f"bw_offset_confidence_intervals_fixed_n_{n}_M_{M}.png", dpi=200)
    plt.close()

