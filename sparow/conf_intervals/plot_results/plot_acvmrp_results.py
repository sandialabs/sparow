import sys
import numpy as np
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

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
# (a) ACV point estimator for gap vs n, grouped by (m, M)
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
# (b) ACV CI upper bound vs n
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
# (c) Variance reduction factor vs n
# ----------------------------------------------------------
for M, subM in df.groupby("M"):
    plt.figure()
    for m, sub in subM.groupby("m"):
        sub = sub.sort_values("n")
        plt.plot(
            sub["n"],
            sub["variance_reduction_factor"],
            marker="o",
            label=rf"paired reps $m={m},\,$ additional reps $M={M}$",
        )

    plt.grid()
    plt.xlabel(r"Sample size $n$")
    plt.ylabel("Variance reduction factor")
    plt.title(rf"Variance reduction factor versus $n$ for $M={M}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"variance_reduction_vs_n_M_{M}.png", dpi=200)
    plt.close()

# ----------------------------------------------------------
# (d) For fixed (m, M), compare ACV and HF-only versus n
# ----------------------------------------------------------
for (m, M), sub in df.groupby(["m", "M"]):
    sub = sub.sort_values("n").copy()

    sub["standard_error_hf_only"] = np.sqrt(sub["sample_variance_F"] / sub["m"])
    sub["half_width_hf_only"] = sub["z_statistic"] * sub["standard_error_hf_only"]
    sub["ci_upper_hf_only"] = sub["point_estimate_hf_only"] + sub["half_width_hf_only"]

    plt.figure()

    plt.plot(
        sub["n"],
        sub["point_estimate_hf_only"],
        marker="o",
        linestyle="--",
        label=rf"HF-only point estimator $\bar{{F}}_n^m(\hat{{x}})$",
    )
    plt.plot(
        sub["n"],
        sub["ci_upper_hf_only"],
        marker="o",
        linestyle=":",
        label=rf"HF-only upper bound $U_f^{{\mathrm{{HF}}}}$",
    )

    plt.plot(
        sub["n"],
        sub["point_estimate"],
        marker="s",
        linestyle="-",
        label=rf"ACV point estimator $\bar{{F}}^{{\mathrm{{ACV}}}}(\hat{{x}},m,M)$",
    )
    plt.plot(
        sub["n"],
        sub["ci_upper"],
        marker="s",
        linestyle="-.",
        label=rf"ACV upper bound $U_f^{{\mathrm{{ACV}}}}$",
    )

    plt.fill_between(
        sub["n"],
        sub["point_estimate_hf_only"],
        sub["ci_upper_hf_only"],
        alpha=0.15,
        label=r"HF-only one-sided interval",
    )
    plt.fill_between(
        sub["n"],
        sub["point_estimate"],
        sub["ci_upper"],
        alpha=0.15,
        label=r"ACV one-sided interval",
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
# (e) For fixed (n, m), compare ACV and HF-only versus M
# ----------------------------------------------------------
for (n, m), sub in df.groupby(["n", "m"]):
    sub = sub.sort_values("M").copy()

    sub["standard_error_hf_only"] = np.sqrt(sub["sample_variance_F"] / sub["m"])
    sub["half_width_hf_only"] = sub["z_statistic"] * sub["standard_error_hf_only"]
    sub["ci_upper_hf_only"] = sub["point_estimate_hf_only"] + sub["half_width_hf_only"]

    plt.figure()

    plt.plot(
        sub["M"],
        sub["point_estimate_hf_only"],
        marker="o",
        linestyle="--",
        label=rf"HF-only point estimator $\bar{{F}}_n^m(\hat{{x}})$",
    )
    plt.plot(
        sub["M"],
        sub["ci_upper_hf_only"],
        marker="o",
        linestyle=":",
        label=rf"HF-only upper bound $U_f^{{\mathrm{{HF}}}}$",
    )

    plt.plot(
        sub["M"],
        sub["point_estimate"],
        marker="s",
        linestyle="-",
        label=rf"ACV point estimator $\bar{{F}}^{{\mathrm{{ACV}}}}(\hat{{x}},m,M)$",
    )
    plt.plot(
        sub["M"],
        sub["ci_upper"],
        marker="s",
        linestyle="-.",
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
# (f) Actual estimator variance: ACV vs HF-only
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
            label=rf"ACV estimator variance, $m={m},\,M={M}$",
        )

    plt.grid()
    plt.xlabel(r"Sample size $n$")
    plt.ylabel("Estimator variance")
    plt.title(rf"Estimator variance: ACV vs HF-only for $M={M}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"variance_acv_vs_hf_only_M_{M}.png", dpi=200)
    plt.close()

# ----------------------------------------------------------
# (g) ACV point estimator vs HF-only point estimator
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
# (h) Standard error: ACV vs HF-only, normalized by true gap
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
# (i) Sample correlation versus n
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
# (j) Estimated control variate coefficient versus n
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
# (k) Effect of M for fixed n and m
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
# (l) For fixed (m, M), effect of increasing n:
#     ACV point estimate and one-sided CI
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
# (m) Interval-width ratio plot: ACV / HF-only
# ----------------------------------------------------------
for M, subM in df.groupby("M"):
    plt.figure()

    for m, sub in subM.groupby("m"):
        sub = sub.sort_values("n").copy()

        sub["standard_error_hf_only"] = np.sqrt(sub["sample_variance_F"] / sub["m"])
        sub["half_width_hf_only"] = sub["z_statistic"] * sub["standard_error_hf_only"]
        sub["interval_width_ratio"] = sub["half_width"] / sub["half_width_hf_only"]

        plt.plot(
            sub["n"],
            sub["interval_width_ratio"],
            marker="o",
            label=rf"$\epsilon_f^{{\mathrm{{ACV}}}} / \epsilon_f^{{\mathrm{{HF}}}}$, $m={m},\,M={M}$",
        )

    plt.axhline(1.0, color="black", linestyle="--", label=r"No improvement")

    plt.grid()
    plt.xlabel(r"Sample size $n$")
    plt.ylabel(
        r"Interval-width ratio $\epsilon_f^{\mathrm{ACV}} / \epsilon_f^{\mathrm{HF}}$"
    )
    plt.title(rf"Interval-width ratio versus $n$ for $M={M}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"interval_width_ratio_vs_n_M_{M}.png", dpi=200)
    plt.close()

# ----------------------------------------------------------
# (n) Upper-bound distance above the true gap
# ----------------------------------------------------------
for M, subM in df.groupby("M"):
    plt.figure()

    for m, sub in subM.groupby("m"):
        sub = sub.sort_values("n").copy()

        sub["standard_error_hf_only"] = np.sqrt(sub["sample_variance_F"] / sub["m"])
        sub["half_width_hf_only"] = sub["z_statistic"] * sub["standard_error_hf_only"]
        sub["ci_upper_hf_only"] = (
            sub["point_estimate_hf_only"] + sub["half_width_hf_only"]
        )

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
