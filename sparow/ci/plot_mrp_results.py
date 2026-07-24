import sys
import os
import numpy as np
import importlib
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import textwrap

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
# (c) Normalized margin of error (half width) vs n
# ----------------------------------------------------------
fig, ax = plt.subplots()

# Different shades of blue for each m
grouped = list(df.groupby("m"))
blue_shades = plt.cm.Blues(np.linspace(0.45, 0.9, len(grouped)))

for color, (m, sub) in zip(blue_shades, grouped):
    sub = sub.sort_values("n").copy()
    sub["normalized_margin_of_error"] = sub["half_width"] / sub["point_estimate"]

    ax.plot(
        sub["n"],
        sub["normalized_margin_of_error"],
        marker="o",
        color=color,
        label=rf"Number of replications, $m={m}$"
    )

# Reference line proportional to 1/sqrt(n)
n_ref = np.sort(df["n"].unique())

# Scale reference line to roughly match the first plotted value
df_sorted = df.sort_values(["m", "n"]).copy()
df_sorted["normalized_margin_of_error"] = df_sorted["half_width"] / df_sorted["point_estimate"]
ref_scale = df_sorted["normalized_margin_of_error"].iloc[0] * np.sqrt(df_sorted["n"].iloc[0])
ref_line = ref_scale / np.sqrt(n_ref)

# ax.plot(
#     n_ref,
#     ref_line,
#     linestyle=":",
#     color="black",
#     linewidth=1.8,
#     label=r"Reference $\propto 1/\sqrt{n}$"
# )

ax.grid()
# ax.set_xlabel(r"Number of sampled scenarios per replication, $n$")
ax.set_ylabel(r"Relative margin of error")
ax.set_title("Relative margin of error for one-sided confidence interval\nvs sample size $n$")
ax.legend()

ax.set_xlabel(r"Number of sampled scenarios per replication, $n$", labelpad=12)

fig.subplots_adjust(bottom=0.42)

fig.text(
    0.5,
    0.01,
    "NOTE: $\\epsilon_f$ = uncertainty margin\n"
    "$\\bar{F}_n^m(\\hat{x})$ = point estimate of optimality gap\n"
    "$\\frac{\\epsilon_f}{\\bar{F}_n^m(\\hat{x})}$ = relative margin of error.",
    ha="center",
    va="bottom",
    fontsize=9
)

fig.tight_layout(rect=[0, 0.16, 1, 1])

fig.savefig(output_dir / "normalized_margin_of_error_vs_n.png", dpi=200)
plt.close(fig)

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

    # Scale y-axis to start just below the true optimal value, end above ci_upper
    y_axis_rescale = (max(sub["ci_upper"]) - true_gap) / 6
    plt.ylim(true_gap - y_axis_rescale, max(sub["ci_upper"]) + y_axis_rescale)

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
fig, ax = plt.subplots()

for m, sub in df.groupby("m"):
    sub = sub.sort_values("n").copy()
    cv_like_pct = 100.0 * sub["sample_std_dev"] / sub["point_estimate"]

    ax.plot(
        sub["n"],
        cv_like_pct,
        marker="o",
        label=rf"Replications, $m={m}$"
    )

# Reference trend line proportional to 1/sqrt(n)
n_ref = np.sort(df["n"].unique())
df_sorted = df.sort_values(["m", "n"]).copy()
cv_like_pct_ref = 100.0 * df_sorted["sample_std_dev"] / df_sorted["point_estimate"]
ref_scale = cv_like_pct_ref.iloc[0] * np.sqrt(df_sorted["n"].iloc[0])
ref_line = ref_scale / np.sqrt(n_ref)

ax.plot(
    n_ref,
    ref_line,
    linestyle="--",
    color="black",
    label=r"Reference trend $\propto 1/\sqrt{n}$"
)

ax.grid()
ax.set_xlabel(r"Sample size $n$")
ax.set_ylabel(r"$100 \times \frac{s_F(\hat{x},m)}{\bar{F}_n^m(\hat{x})}$")
ax.set_title(r"Relative replication variability versus $n$")
ax.legend()

caption = (
    r"Relative replication variability, measured as "
    r"$100 \times s_F(\hat{x},m)/\bar{F}_n^m(\hat{x})$, versus sample size $n$ "
    r"for several replication counts $m$. The dashed reference line shows a trend "
    r"proportional to $1/\sqrt{n}$. Large values indicate that replication-to-replication "
    r"variability is high relative to the estimated upper bound itself."
)

# Wrapping for long captions
wrapped_caption = "\n".join(textwrap.wrap(caption, width=110))

# Leave room at bottom for caption
fig.subplots_adjust(bottom=0.28)

# Add caption inside the figure canvas
fig.text(
    0.5, 0.02,
    wrapped_caption,
    ha="center",
    va="bottom",
    fontsize=9
)

fig.savefig(output_dir / "cv_like_vs_n_percent.png", dpi=200, bbox_inches="tight")
plt.close(fig)