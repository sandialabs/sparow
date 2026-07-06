import sys
import os
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
output_dir = model_file.parent / "mrp_plots"
output_dir.mkdir(exist_ok=True)

# ----------------------------------------------------------
# (a) Point estimator for Gap vs n
# ----------------------------------------------------------
plt.figure()
for m, sub in df.groupby("m"):
    sub = sub.sort_values("n")
    plt.plot(
        sub["n"],
        sub["point_estimate"],
        marker="o",
        label=rf"Number of replications, $m={m}$"
    )
plt.axhline(
    true_gap,
    color="black",
    linestyle="--",
    label=rf"True optimality gap $\Delta_f(\hat{{x}})$"
)
plt.grid()
plt.xlabel(r"Sample size $n$")
plt.ylabel(r"Point estimator $\bar{F}_n^m(\hat{x})$")
plt.title(r"Point estimator $\bar{F}_n^m(\hat{x})$ versus sample size $n$")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "gap_estimate_vs_n.png", dpi=200)
plt.close()

# ----------------------------------------------------------
# (b) CI upper bound vs n
# ----------------------------------------------------------
plt.figure()
for m, sub in df.groupby("m"):
    sub = sub.sort_values("n")
    plt.plot(
        sub["n"],
        sub["ci_upper"],
        marker="o",
        label=rf"Number of replications, $m={m}$"
    )
plt.axhline(
    true_gap,
    color="black",
    linestyle="--",
    label=rf"True optimality gap $\Delta_f(\hat{{x}})$"
)
plt.grid()
plt.xlabel(r"Sample size $n$")
plt.ylabel(r"Upper CI bound $\bar{F}_n^m(\hat{x}) + \epsilon_f$")
plt.title(r"Upper confidence bound $\bar{F}_n^m(\hat{x}) + \epsilon_f$ versus $n$")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "ci_upper_vs_n.png", dpi=200)
plt.close()

# ----------------------------------------------------------
# (c) Half-width vs n
# ----------------------------------------------------------
plt.figure()
for m, sub in df.groupby("m"):
    sub = sub.sort_values("n")
    plt.plot(
        sub["n"],
        sub["half_width"],
        marker="o",
        label=rf"Number of replications, $m={m}$"
    )
plt.grid()
plt.xlabel(r"Sample size $n$")
plt.ylabel(r"Half-width $\epsilon_f$")
plt.title(r"Confidence-interval half-width $\epsilon_f$ versus sample size $n$")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "halfwidth_vs_n.png", dpi=200)
plt.close()

# --------------------------------------------------------------
# (d) Sample standard deviation (normalized by true gap) vs n
# --------------------------------------------------------------
plt.figure()
for m, sub in df.groupby("m"):
    sub = sub.sort_values("n")
    std_pct = 100.0 * sub["sample_std_dev"] / true_gap
    plt.plot(
        sub["n"],
        std_pct,
        marker="o",
        label=rf"$m={m}$"
    )
plt.grid()
plt.xlabel(r"Sample size $n$")
plt.ylabel(r"$100 \times \frac{s_F(\hat{x},m)}{\Delta_f(\hat{x})}$")
plt.title(r"Relative sample standard deviation $100 \times \frac{s_F(\hat{x},m)}{\Delta_f(\hat{x})}$ versus sample size $n$")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "sample_std_dev_vs_n.png", dpi=200)
plt.close()

# ----------------------------------------------------------
# (e) For fixed m, effect of increasing n:
#     point estimate and one-sided CI
# ----------------------------------------------------------
for m, sub in df.groupby("m"):
    sub = sub.sort_values("n")

    plt.figure()
    plt.plot(
        sub["n"],
        sub["point_estimate"],
        marker="o",
        label=rf"Point estimator $\bar{{F}}_n^m(\hat{{x}})$"
    )
    plt.plot(
        sub["n"],
        sub["ci_upper"],
        marker="s",
        label=rf"Upper bound $\bar{{F}}_n^m(\hat{{x}})+\epsilon_f$"
    )
    plt.axhline(
        true_gap,
        color="black",
        linestyle="--",
        label=rf"True optimality gap $\Delta_f(\hat{{x}})$"
    )

    plt.fill_between(
        sub["n"],
        sub["ci_lower"],
        sub["ci_upper"],
        alpha=0.2,
        label=r"One-sided confidence interval"
    )
    plt.grid()
    plt.xlabel(r"Sample size $n$")
    plt.ylabel(r"Gap estimate / confidence bound")
    plt.title(rf"Effect of increasing $n$ for fixed number of replications $m={m}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"fixed_m_{m}_effect_of_n.png", dpi=200)
    plt.close()

# ----------------------------------------------------------
# (f) Absolute error (normalized by true gap)
#     between point estimator and true optimality gap vs n
# ----------------------------------------------------------
plt.figure()
for m, sub in df.groupby("m"):
    sub = sub.sort_values("n")
    abs_error_pct = 100.0 * (sub["point_estimate"] - true_gap).abs() / true_gap
    plt.plot(
        sub["n"],
        abs_error_pct,
        marker="o",
        label=rf"Number of replications, $m={m}$"
    )
plt.grid()
plt.xlabel(r"Sample size $n$")
plt.ylabel(r"$100 \times \frac{\left|\bar{F}_n^m(\hat{x}) - \Delta_f(\hat{x})\right|}{\Delta_f(\hat{x})}$")
plt.title(r"Relative absolute error $100 \times \frac{\left|\bar{F}_n^m(\hat{x}) - \Delta_f(\hat{x})\right|}{\Delta_f(\hat{x})}$ versus sample size $n$")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "absolute_error_vs_n.png", dpi=200)
plt.close()

# ----------------------------------------------------------
# (g) Coefficient-of-variation-type quantity vs n
# ----------------------------------------------------------
plt.figure()
for m, sub in df.groupby("m"):
    sub = sub.sort_values("n")
    cv_like = sub["sample_std_dev"] / sub["point_estimate"]
    plt.plot(
        sub["n"],
        cv_like,
        marker="o",
        label=rf"$m={m}$"
    )
plt.grid()
plt.xlabel(r"Sample size $n$")
plt.ylabel(r"$\frac{s_F(\hat{x},m)}{\bar{F}_n^m(\hat{x})}$")
plt.title(r"Coefficient-of-variation-type quantity $\frac{s_F(\hat{x},m)}{\bar{F}_n^m(\hat{x})}$ versus $n$")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "cv_like_vs_n.png", dpi=200)
plt.close()