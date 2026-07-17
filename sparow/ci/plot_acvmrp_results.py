import sys
import numpy as np
import importlib
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)

true_gap = df["true_gap"].iloc[0]

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
})

# ----------------------------------------------------------------------
# Infer output directory from the problem module path
# ----------------------------------------------------------------------
model_module_name = df["model_module"].iloc[0]

model_module = importlib.import_module(model_module_name)
model_file = Path(model_module.__file__).resolve()
output_dir = model_file.parent / "acvmrp_plots"
output_dir.mkdir(exist_ok=True)

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
            label=rf"paired reps $m={m},\,$ additional reps $M={M}$"
        )

    plt.axhline(
        true_gap,
        color="black",
        linestyle="--",
        label=rf"True optimality gap $\Delta_f(\hat{{x}})$"
    )
    plt.grid()
    plt.xlabel(r"Sample size $n$")
    plt.ylabel(r"ACV point estimator")
    plt.title(rf"ACV point estimator versus $n$ for $M={M}$")
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
            label=rf"paired reps $m={m},\,$ additional reps $M={M}$"
        )

    plt.axhline(
        true_gap,
        color="black",
        linestyle="--",
        label=rf"True optimality gap $\Delta_f(\hat{{x}})$"
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
            label=rf"paired reps $m={m},\,$ additional reps $M={M}$"
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
# (d) Actual estimator variance: ACV vs HF-only
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
            label=rf"HF-only estimator variance, $m={m}$"
        )

        plt.plot(
            sub["n"],
            sub["variance_acv_estimator_plot"],
            marker="s",
            linestyle="-",
            label=rf"ACV estimator variance, $m={m},\,M={M}$"
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
# (e) ACV point estimator vs HF-only point estimator
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
            label=rf"$\bar F^{{\mathrm{{ACV}}}}(\hat{{x}},m,M)$, $m={m},\,M={M}$"
        )

        plt.plot(
            sub["n"],
            sub["point_estimate_hf_only"],
            marker="s",
            linestyle="--",
            label=rf"$\bar F_n^m(\hat{{x}})$, $m={m}$"
        )

    plt.axhline(
        true_gap,
        color="black",
        linestyle=":",
        label=rf"True optimality gap $\Delta_f(\hat{{x}})$"
    )
    plt.grid()
    plt.xlabel(r"Sample size $n$")
    plt.ylabel(r"Point estimator")
    plt.title(rf"$\bar F^{{\mathrm{{ACV}}}}(\hat{{x}},m,M)$ and $\bar F_n^m(\hat{{x}})$ versus $n$ for $M={M}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"acv_vs_hf_point_estimate_M_{M}.png", dpi=200)
    plt.close()

# ----------------------------------------------------------
# (e) ACV point estimator vs HF-only point estimator
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
            label=rf"$\bar F^{{\mathrm{{ACV}}}}(\hat{{x}},m,M)$, $m={m},\,M={M}$"
        )

        plt.plot(
            sub["n"],
            sub["point_estimate_hf_only"],
            marker="s",
            linestyle="--",
            label=rf"$\bar F_n^m(\hat{{x}})$, $m={m}$"
        )

    plt.axhline(
        true_gap,
        color="black",
        linestyle=":",
        label=rf"True optimality gap $\Delta_f(\hat{{x}})$"
    )
    plt.grid()
    plt.xlabel(r"Sample size $n$")
    plt.ylabel(r"Point estimator")
    plt.title(rf"$\bar F^{{\mathrm{{ACV}}}}(\hat{{x}},m,M)$ and $\bar F_n^m(\hat{{x}})$ versus $n$ for $M={M}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"acv_vs_hf_point_estimate_M_{M}.png", dpi=200)
    plt.close()

# ----------------------------------------------------------
# (f) Standard error: ACV vs HF-only
# ----------------------------------------------------------
for M, subM in df.groupby("M"):
    plt.figure()

    for m, sub in subM.groupby("m"):
        sub = sub.sort_values("n").copy()

        sub["standard_error_hf_only"] = np.sqrt(sub["sample_variance_F"] / sub["m"])

        plt.plot(
            sub["n"],
            sub["standard_error_hf_only"],
            marker="o",
            linestyle="--",
            label=rf"HF-only $\widehat{{\operatorname{{SE}}}}(\bar F_n^m(\hat{{x}}))$, $m={m}$"
        )

        plt.plot(
            sub["n"],
            sub["standard_error_acv"],
            marker="s",
            linestyle="-",
            label=rf"ACV $\widehat{{\operatorname{{SE}}}}(\bar F^{{\mathrm{{ACV}}}}(\hat{{x}},m,M))$, $m={m},\,M={M}$"
        )

    plt.grid()
    plt.xlabel(r"Sample size $n$")
    plt.ylabel(r"Estimated standard error")
    plt.title(rf"Estimated standard error versus $n$ for $M={M}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"standard_error_acv_vs_hf_M_{M}.png", dpi=200)
    plt.close()

# ----------------------------------------------------------
# (g) Sample correlation versus n
# ----------------------------------------------------------
for M, subM in df.groupby("M"):
    plt.figure()
    for m, sub in subM.groupby("m"):
        sub = sub.sort_values("n")

        plt.plot(
            sub["n"],
            sub["sample_correlation"],
            marker="o",
            label=rf"$\hat\rho_{{fg}}$, $m={m},\,M={M}$"
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
# (h) Estimated control variate coefficient versus n
# ----------------------------------------------------------
for M, subM in df.groupby("M"):
    plt.figure()
    for m, sub in subM.groupby("m"):
        sub = sub.sort_values("n")

        plt.plot(
            sub["n"],
            sub["control_variate_coefficient"],
            marker="o",
            label=rf"$\hat\alpha$, $m={m},\,M={M}$"
        )

    plt.grid()
    plt.xlabel(r"Sample size $n$")
    plt.ylabel(r"Estimated control variate coefficient $\hat\alpha$")
    plt.title(rf"Estimated control variate coefficient $\hat\alpha$ versus $n$ for $M={M}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"alpha_hat_vs_n_M_{M}.png", dpi=200)
    plt.close()

# ----------------------------------------------------------
# (i) Effect of M for fixed n and m
# ----------------------------------------------------------
for m, subm in df.groupby("m"):
    for n, submn in subm.groupby("n"):
        plt.figure()

        submn = submn.sort_values("M")

        plt.plot(
            submn["M"],
            submn["variance_reduction_factor"],
            marker="o"
        )

        plt.grid()
        plt.xlabel(r"Additional low-fidelity replication count $M$")
        plt.ylabel(r"Variance reduction factor")
        plt.title(rf"Variance reduction factor versus $M$ for $m={m},\,n={n}$")
        plt.tight_layout()
        plt.savefig(output_dir / f"variance_reduction_vs_M_m_{m}_n_{n}.png", dpi=200)
        plt.close()