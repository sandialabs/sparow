"""
Run budget-aware multifidelity UQ experiments for stochastic programs.

Workflow overview
-----------------
For each requested scenario-batch size n and each total budget B:

1. Load or generate one candidate first-stage solution, xhat.
2. Optionally compute the true finite-population optimality gap once as a benchmark.
3. Build a PyApprox multifidelity problem from the problem module's get_model_ensemble_for_uq(...) factory.
4. Run a pilot study to estimate:
   - HF/LF model costs (wall-clock time),
   - covariance / correlation between the HF and LF replication-level outputs.
5. Use PyApprox to recommend the total HF and LF sample counts under the requested budget.
6. Translate the PyApprox allocation into ACV-MRP counts:
       m = number of paired HF/LF replications
       M = number of additional LF-only replications
7. Run, for each macro-replication:
   - ACV-MRP using (m, M),
   - HF-only same-total-budget baseline,
   - HF-only paired-only baseline using the same paired count m.
8. Aggregate the resulting macro-replication outputs into a summary CSV.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sparow.conf_intervals.experiment_helpers import (
    parse_int_list,
    parse_float_list,
    safe_float,
    mean_or_nan,
    var_or_nan,
    make_run_directory,
    place_output_in_run_dir,
    write_csv,
    write_debug_json,
    log,
    parse_solver_options,
    load_or_generate_candidate_xhat,
    compute_true_gap_with_timer,
    load_model_ensemble_for_uq,
    elapsed_seconds,
    run_standard_mrp,
    run_acvmrp,
)

from sparow.conf_intervals.pyapprox_helpers import (
    run_pyapprox_pilot,
    allocate_pyapprox_budget,
    hf_replications_for_same_budget,
)

# =============================================================================
# CLI parsing and related helpers
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run budget-aware UQ experiments using PyApprox + ACV-MRP."
    )

    parser.add_argument("--model-module", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--lf-model-type", type=str, default=None)
    parser.add_argument("--use-integer", action="store_true")

    parser.add_argument("--solver-name", default="gurobi_direct")
    parser.add_argument("--solver-options", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.05)

    parser.add_argument("--xhat-file", required=True)
    parser.add_argument("--use-existing-xhat", action="store_true")

    parser.add_argument("--candidate-with-replacement", action="store_true")
    parser.add_argument("--candidate-without-replacement", action="store_true")

    parser.add_argument("--mrp-with-replacement", action="store_true")
    parser.add_argument("--mrp-without-replacement", action="store_true")

    parser.add_argument("--candidate-scen-count", type=int, default=None)
    parser.add_argument("--candidate-seed", type=int, default=12345)
    parser.add_argument("--main-seed", type=int, default=54321)

    parser.add_argument("--macro-replications", type=int, default=10)
    parser.add_argument("--n-values", type=str, required=True)
    parser.add_argument("--budget-values", type=str, required=True)
    parser.add_argument("--n-pilot", type=int, default=10)

    parser.add_argument("--count-pilot-cost-against-budget", action="store_true")
    parser.add_argument(
        "--reuse-pilot-across-macroreps",
        action="store_true",
        help="Reuse one pilot study for each fixed n across all macro-replications.",
    )
    parser.add_argument(
        "--redo-pilot-per-macrorep",
        action="store_true",
        help="Force a fresh pilot study within each macro-replication.",
    )

    parser.add_argument("--compute-true-gap", action="store_true")
    parser.add_argument("--save-debug-json", action="store_true")

    parser.add_argument("--hf-cost-delay-seconds", type=float, default=0.0)
    parser.add_argument("--lf-cost-delay-seconds", type=float, default=0.0)

    parser.add_argument("--output-summary-csv", type=str, required=True)
    parser.add_argument("--output-macro-csv", type=str, required=True)
    parser.add_argument("--debug-json-file", type=str, default="budgeted_uq_debug.json")

    return parser.parse_args()


def interpret_candidate_sampling_flags(args) -> bool:
    """Translate candidate sampling with vs without replacement flags into a boolean."""
    if args.candidate_with_replacement and args.candidate_without_replacement:
        raise ValueError(
            "Choose only one of --candidate-with-replacement or --candidate-without-replacement."
        )
    candidate_with_replacement = True
    if args.candidate_without_replacement:
        candidate_with_replacement = False
    return candidate_with_replacement


def interpret_mrp_sampling_flags(args) -> bool:
    """Translate main sampling with vs without replacement flags into a boolean."""
    if args.mrp_with_replacement and args.mrp_without_replacement:
        raise ValueError(
            "Choose only one of --mrp-with-replacement or --mrp-without-replacement."
        )
    mrp_with_replacement = True
    if args.mrp_without_replacement:
        mrp_with_replacement = False
    return mrp_with_replacement


def interpret_pilot_reuse_flags(args) -> bool:
    """Default behavior is to reuse pilot across macro-replications."""
    if args.reuse_pilot_across_macroreps and args.redo_pilot_per_macrorep:
        raise ValueError(
            "Choose only one of --reuse-pilot-across-macroreps or --redo-pilot-per-macrorep."
        )
    if args.redo_pilot_per_macrorep:
        return False
    return True


# ============================================================================
# Helpers for running each macro-replication
# ============================================================================


def build_macro_seed(base_seed: int, n: int, budget_idx: int, macro_rep: int) -> int:
    """
    Build a deterministic seed for one macro-replication.
    We need to use distinct streams per (n, budget, macro-rep) configuration.
    """
    return int(base_seed + 100000 * n + 1000 * budget_idx + macro_rep)


def run_one_macrorep(
    *,
    args,
    model_module_name: str,
    model_name: Optional[str],
    lf_model_type: str,
    xhat,
    n: int,
    budget: float,
    budget_idx: int,
    macro_rep: int,
    alpha: float,
    main_with_replacement: bool,
    pilot_info,
    count_pilot_cost_against_budget: bool,
    true_gap: float,
    t0: float,
):
    """
    Run one macro-replication for one (n, budget) configuration.

    STEPS:
      1. gets the PyApprox-recommended multifidelity allocation,
      2. gets the same-budget HF-only allocation,
      3. runs ACV-MRP,
      4. runs two HF-only baselines,
      5. records the resulting estimators, intervals, diagnostics, and timings.
    """
    verbose = args.verbose

    # Get specific seed for this macro-replication
    macro_seed = build_macro_seed(args.main_seed, n, budget_idx, macro_rep)

    # Rebuild the ensemble with this macro-rep's specific seed, so that the scenario
    # sampling streams used by ACV-MRP and HF-only baselines are tied to this
    # specific macro-replication.
    ensemble = load_model_ensemble_for_uq(
        model_module_name=model_module_name,
        model_name=model_name,
        use_integer=args.use_integer,
        seed=macro_seed,
        with_replacement=main_with_replacement,
        lf_model_type=lf_model_type,
    )
    hf_model = ensemble.high_fidelity_model()

    # Use the pilot covariance and cost information to allocate the total budget
    alloc = allocate_pyapprox_budget(
        pilot_info=pilot_info,
        total_budget=budget,
        n_pilot=args.n_pilot,
        count_pilot_cost_against_budget=count_pilot_cost_against_budget,
    )

    # If no positive budget remains after optionally charging the pilot cost,
    # we set "allocation_feasible" to false and end the macro-replication.
    if not alloc["allocation_feasible"]:
        return {
            "allocation_feasible": False,
            "n": n,
            "budget": budget,
            "budget_index": budget_idx,
            "macro_replication": macro_rep,
            "estimated_pilot_cost": alloc["estimated_pilot_cost"],
            "remaining_budget": alloc["remaining_budget"],
        }

    # These are the ACV-MRP counts implied by the multifidelity allocation:
    # m = paired HF/LF replications,
    # M = additional LF-only replications.
    m_paired = alloc["m_paired"]
    M_additional_lf = alloc["M_additional_lf"]

    # Construct the same-total-budget HF-only baseline using PyApprox's
    # single-fidelity MC allocator
    hf_budget_alloc = hf_replications_for_same_budget(
        pilot_info=pilot_info,
        total_budget=budget,
        n_pilot=args.n_pilot,
        count_pilot_cost_against_budget=count_pilot_cost_against_budget,
        verbose=verbose,
        t0=t0,
    )
    hf_same_budget_m = hf_budget_alloc["hf_pyapprox_count"]

    # The paired-HF-only baseline uses the same number of HF replications as the
    # paired portion of ACV, but does not spend any extra budget on LF samples.
    hf_paired_only_m = m_paired

    # Discard configurations that do not have enough samples,
    # and end the macro-replication.
    if m_paired < 2 or hf_same_budget_m < 2 or hf_paired_only_m < 2:
        return {
            "allocation_feasible": False,
            "n": n,
            "budget": budget,
            "budget_index": budget_idx,
            "macro_replication": macro_rep,
            "estimated_pilot_cost": alloc["estimated_pilot_cost"],
            "remaining_budget": alloc["remaining_budget"],
            "m_paired": m_paired,
            "M_additional_lf": M_additional_lf,
            "hf_same_budget_m": hf_same_budget_m,
            "hf_paired_only_m": hf_paired_only_m,
        }

    row = {
        "allocation_feasible": True,
        "n": n,
        "budget": float(budget),
        "budget_index": int(budget_idx),
        "macro_replication": int(macro_rep),
        "alpha": float(alpha),
        "estimated_pilot_cost": float(alloc["estimated_pilot_cost"]),
        "remaining_budget": float(alloc["remaining_budget"]),
        "estimated_hf_cost": float(pilot_info["costs_np"][0]),
        "estimated_lf_cost": float(pilot_info["costs_np"][1]),
        "pilot_rho_hat": float(pilot_info["rho_hat_pilot"]),
        "predicted_pyapprox_std": float(alloc["predicted_pyapprox_std"]),
        "predicted_pyapprox_var": float(alloc["predicted_pyapprox_var"]),
        "pyapprox_hf_total": int(alloc["pyapprox_hf_total"]),
        "pyapprox_lf_total": int(alloc["pyapprox_lf_total"]),
        "m_paired": int(m_paired),
        "M_additional_lf": int(M_additional_lf),
        "hf_same_budget_m": int(hf_same_budget_m),
        "hf_paired_only_m": int(hf_paired_only_m),
        "true_gap": float(true_gap) if true_gap is not None else float("nan"),
    }

    # ---------------------------
    # ACV-MRP
    # ---------------------------
    acv_start = time.time()
    log(
        (
            f"Running ACV-MRP for macro-rep {macro_rep + 1}/{args.macro_replications}, "
            f"n={n}, budget={budget}, m={m_paired}, M={M_additional_lf}"
        ),
        t0=t0,
        verbose=verbose,
    )

    # This is the multifidelity estimator under the PyApprox-recommended
    # budget split between HF and LF evaluations.
    acv_results = run_acvmrp(
        ensemble=ensemble,
        xhat=xhat,
        n=n,
        m=m_paired,
        M=M_additional_lf,
        alpha=alpha,
        # specific seed for this macro-replication, along with offset to use distinct stream
        seed=macro_seed + 1000000,
        with_replacement=main_with_replacement,
        solver_name=args.solver_name,
        solver_options=parse_solver_options(args.solver_options),
        verbose=False,
    )
    acv_elapsed = time.time() - acv_start

    # ---------------------------
    # HF-only same-total-budget
    # ---------------------------
    hf_budget_start = time.time()
    log(
        (
            f"Running HF-only same-budget baseline for macro-rep {macro_rep + 1}, "
            f"n={n}, budget={budget}, m={hf_same_budget_m}"
        ),
        t0=t0,
        verbose=verbose,
    )

    # This baseline spends the same total budget as ACVMRP, but allocates it all to
    # HF replications instead of splitting it between HF and LF.
    hf_budget_results = run_standard_mrp(
        model=hf_model,
        xhat=xhat,
        n=n,
        m=hf_same_budget_m,
        alpha=alpha,
        # specific seed for this macro-replication, along with offset to use distinct stream
        seed=macro_seed + 2000000,
        with_replacement=main_with_replacement,
        solver_name=args.solver_name,
        solver_options=parse_solver_options(args.solver_options),
        verbose=False,
    )
    hf_budget_elapsed = time.time() - hf_budget_start

    # ---------------------------
    # HF-only paired-only
    # ---------------------------
    hf_paired_start = time.time()
    log(
        (
            f"Running HF-only paired-count baseline for macro-rep {macro_rep + 1}, "
            f"n={n}, paired m={hf_paired_only_m}"
        ),
        t0=t0,
        verbose=verbose,
    )

    # This baseline isolates the value of the LF computation by comparing ACVMRP
    # against an HF-only run that uses only the same paired HF count m.
    hf_paired_results = run_standard_mrp(
        model=hf_model,
        xhat=xhat,
        n=n,
        m=hf_paired_only_m,
        alpha=alpha,
        # specific seed for this macro-replication,
        # so same stream of scenario batches is sampled across ACVMRP and both HF-only baselines
        seed=macro_seed,
        with_replacement=main_with_replacement,
        solver_name=args.solver_name,
        solver_options=parse_solver_options(args.solver_options),
        verbose=False,
    )
    hf_paired_elapsed = time.time() - hf_paired_start

    # Store ACVMRP outputs
    row.update(
        {
            "acv_point_estimate": float(acv_results["point_estimate"]),
            "acv_point_estimate_hf_only_paired_reference": float(
                acv_results["point_estimate_hf_only"]
            ),
            "acv_ci_lower": float(acv_results["ci_lower"]),
            "acv_ci_upper": float(acv_results["ci_upper"]),
            "acv_half_width": float(acv_results["half_width"]),
            "acv_standard_error": float(acv_results["standard_error_acv"]),
            "acv_variance_estimator": float(acv_results["variance_acv_estimator"]),
            "acv_sample_variance_F": float(acv_results["sample_variance_F"]),
            "acv_sample_variance_G_paired": float(
                acv_results["sample_variance_G_paired"]
            ),
            "acv_sample_covariance_FG": float(acv_results["sample_covariance_FG"]),
            "acv_alpha_hat": float(acv_results["control_variate_coefficient"]),
            "acv_rho_hat": float(acv_results["sample_correlation"]),
            "acv_variance_reduction_factor": float(
                acv_results["variance_reduction_factor"]
            ),
        }
    )

    # Store HF same-budget outputs
    row.update(
        {
            "hf_budget_point_estimate": float(hf_budget_results["point_estimate"]),
            "hf_budget_ci_lower": float(hf_budget_results["ci_lower"]),
            "hf_budget_ci_upper": float(hf_budget_results["ci_upper"]),
            "hf_budget_half_width": float(hf_budget_results["half_width"]),
            "hf_budget_sample_variance": float(hf_budget_results["sample_variance"]),
            "hf_budget_sample_std": float(hf_budget_results["sample_std"]),
            "hf_budget_t_statistic": float(hf_budget_results["t_statistic"]),
        }
    )

    # Store HF paired-only outputs
    row.update(
        {
            "hf_paired_only_point_estimate": float(hf_paired_results["point_estimate"]),
            "hf_paired_only_ci_lower": float(hf_paired_results["ci_lower"]),
            "hf_paired_only_ci_upper": float(hf_paired_results["ci_upper"]),
            "hf_paired_only_half_width": float(hf_paired_results["half_width"]),
            "hf_paired_only_sample_variance": float(
                hf_paired_results["sample_variance"]
            ),
            "hf_paired_only_sample_std": float(hf_paired_results["sample_std"]),
            "hf_paired_only_t_statistic": float(hf_paired_results["t_statistic"]),
        }
    )

    # Coverage indicators: recording whether the estimated confidence interval
    # contains the true population quantity of interest (obtained from benchmark)
    if true_gap is None or math.isnan(true_gap):
        row["acv_covers_true_gap"] = float("nan")
        row["hf_budget_covers_true_gap"] = float("nan")
        row["hf_paired_only_covers_true_gap"] = float("nan")
    else:
        row["acv_covers_true_gap"] = int(true_gap <= row["acv_ci_upper"])
        row["hf_budget_covers_true_gap"] = int(true_gap <= row["hf_budget_ci_upper"])
        row["hf_paired_only_covers_true_gap"] = int(
            true_gap <= row["hf_paired_only_ci_upper"]
        )

    # Realized-improvement indicators track how often ACV gives a smaller
    # realized conf interval upper bound or point estimate than the chosen HF-only baseline.
    row["acv_improves_over_hf_budget"] = int(
        row["acv_ci_upper"] < row["hf_budget_ci_upper"]
    )
    row["acv_improves_over_hf_paired_only"] = int(
        row["acv_ci_upper"] < row["hf_paired_only_ci_upper"]
    )
    row["acv_point_improves_over_hf_budget"] = int(
        row["acv_point_estimate"] < row["hf_budget_point_estimate"]
    )
    row["acv_point_improves_over_hf_paired_only"] = int(
        row["acv_point_estimate"] < row["hf_paired_only_point_estimate"]
    )

    # Keep per-method timings for profiling.
    row["elapsed_acv_seconds"] = float(acv_elapsed)
    row["elapsed_hf_budget_seconds"] = float(hf_budget_elapsed)
    row["elapsed_hf_paired_only_seconds"] = float(hf_paired_elapsed)
    row["elapsed_total_macrorep_seconds"] = float(
        acv_elapsed + hf_budget_elapsed + hf_paired_elapsed
    )

    return row


# =============================================================================
# Summary metrics
# =============================================================================


def summarize_macrorep_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate per-macro-replication rows into summary rows.

    Each summary row corresponds to one fixed (n, budget) pair. The metrics
    reported here aggregate the results to report:
      - empirical coverage,
      - average upper bound,
      - average point estimate,
      - realized improvement probability,
      - empirical variance of point estimators,
      - average correlation / alpha / allocations.
    """
    grouped = defaultdict(list)

    # Only look at experiments where enough budget was actually available to complete the experiment
    for row in rows:
        if not row.get("allocation_feasible", False):
            continue

        # Use the tuple (n, budget) as the grouping key, so that all macro-reps
        # from the same parameter configuration are collected together.
        grouped[(row["n"], row["budget"])].append(row)

    summary_rows = []

    # grouped.items() returns pairs of the form: ((n, budget), subrows)
    # where:
    #   - (n, budget) is the parameter configuration,
    #   - subrows is the list of macro-replication rows for that configuration.
    # We sort by n first, then by budget
    for (n, budget), subrows in sorted(
        grouped.items(), key=lambda x: (x[0][0], x[0][1])
    ):
        out = {
            "n": int(n),
            "budget": float(budget),
            "R": int(len(subrows)),
            "alpha": float(subrows[0]["alpha"]),
            "true_gap": float(subrows[0]["true_gap"]),
            "estimated_hf_cost_mean": mean_or_nan(
                [r["estimated_hf_cost"] for r in subrows]
            ),
            "estimated_lf_cost_mean": mean_or_nan(
                [r["estimated_lf_cost"] for r in subrows]
            ),
            "pilot_cost_mean": mean_or_nan(
                [r["estimated_pilot_cost"] for r in subrows]
            ),
            "remaining_budget_mean": mean_or_nan(
                [r["remaining_budget"] for r in subrows]
            ),
            "pilot_rho_hat_mean": mean_or_nan([r["pilot_rho_hat"] for r in subrows]),
            "predicted_pyapprox_std_mean": mean_or_nan(
                [r["predicted_pyapprox_std"] for r in subrows]
            ),
            "m_paired_mean": mean_or_nan([r["m_paired"] for r in subrows]),
            "M_additional_lf_mean": mean_or_nan(
                [r["M_additional_lf"] for r in subrows]
            ),
            "hf_same_budget_m_mean": mean_or_nan(
                [r["hf_same_budget_m"] for r in subrows]
            ),
            "hf_paired_only_m_mean": mean_or_nan(
                [r["hf_paired_only_m"] for r in subrows]
            ),
        }

        # Average point estimates and upper bounds
        out["avg_acv_point_estimate"] = mean_or_nan(
            [r["acv_point_estimate"] for r in subrows]
        )
        out["avg_acv_ci_upper"] = mean_or_nan([r["acv_ci_upper"] for r in subrows])

        out["avg_hf_budget_point_estimate"] = mean_or_nan(
            [r["hf_budget_point_estimate"] for r in subrows]
        )
        out["avg_hf_budget_ci_upper"] = mean_or_nan(
            [r["hf_budget_ci_upper"] for r in subrows]
        )

        out["avg_hf_paired_only_point_estimate"] = mean_or_nan(
            [r["hf_paired_only_point_estimate"] for r in subrows]
        )
        out["avg_hf_paired_only_ci_upper"] = mean_or_nan(
            [r["hf_paired_only_ci_upper"] for r in subrows]
        )

        # Average uncertainty margins added to each point estimator (i.e. - interval "half width")
        out["avg_acv_half_width"] = mean_or_nan([r["acv_half_width"] for r in subrows])
        out["avg_hf_budget_half_width"] = mean_or_nan(
            [r["hf_budget_half_width"] for r in subrows]
        )
        out["avg_hf_paired_only_half_width"] = mean_or_nan(
            [r["hf_paired_only_half_width"] for r in subrows]
        )

        # Empirical coverage: the fraction of macro-reps whose upper bound
        # contains the true optimality gap benchmark.
        acv_cov = [
            r["acv_covers_true_gap"]
            for r in subrows
            if not math.isnan(r["acv_covers_true_gap"])
        ]
        hf_budget_cov = [
            r["hf_budget_covers_true_gap"]
            for r in subrows
            if not math.isnan(r["hf_budget_covers_true_gap"])
        ]
        hf_paired_cov = [
            r["hf_paired_only_covers_true_gap"]
            for r in subrows
            if not math.isnan(r["hf_paired_only_covers_true_gap"])
        ]

        out["empirical_coverage_acv"] = mean_or_nan(acv_cov)
        out["empirical_coverage_hf_budget"] = mean_or_nan(hf_budget_cov)
        out["empirical_coverage_hf_paired_only"] = mean_or_nan(hf_paired_cov)

        # Realized improvement probabilities
        # How often ACV gives a smaller realized upper bound than each HF-only comparator
        out["prob_acv_improves_over_hf_budget"] = mean_or_nan(
            [r["acv_improves_over_hf_budget"] for r in subrows]
        )
        out["prob_acv_improves_over_hf_paired_only"] = mean_or_nan(
            [r["acv_improves_over_hf_paired_only"] for r in subrows]
        )

        # Average correlation / coefficient
        out["avg_acv_rho_hat"] = mean_or_nan([r["acv_rho_hat"] for r in subrows])
        out["avg_acv_alpha_hat"] = mean_or_nan([r["acv_alpha_hat"] for r in subrows])
        out["avg_acv_variance_reduction_factor"] = mean_or_nan(
            [r["acv_variance_reduction_factor"] for r in subrows]
        )

        # Empirical variances of point estimators across macro-replications
        out["empirical_variance_acv_point"] = var_or_nan(
            [r["acv_point_estimate"] for r in subrows]
        )
        out["empirical_variance_hf_budget_point"] = var_or_nan(
            [r["hf_budget_point_estimate"] for r in subrows]
        )
        out["empirical_variance_hf_paired_only_point"] = var_or_nan(
            [r["hf_paired_only_point_estimate"] for r in subrows]
        )

        # Variance-ratio relative efficiency for the same-budget HF-only comparison,
        # with values > 1 indicating ACV is more efficient.
        if (
            not math.isnan(out["empirical_variance_acv_point"])
            and out["empirical_variance_acv_point"] > 0
            and not math.isnan(out["empirical_variance_hf_budget_point"])
        ):
            out["relative_efficiency_hf_budget_over_acv"] = (
                out["empirical_variance_hf_budget_point"]
                / out["empirical_variance_acv_point"]
            )
        else:
            out["relative_efficiency_hf_budget_over_acv"] = float("nan")

        # Variance-ratio relative efficiency for the paired HF-only comparison,
        # with values > 1 indicating ACV is more efficient.
        if (
            not math.isnan(out["empirical_variance_acv_point"])
            and out["empirical_variance_acv_point"] > 0
            and not math.isnan(out["empirical_variance_hf_paired_only_point"])
        ):
            out["relative_efficiency_hf_paired_only_over_acv"] = (
                out["empirical_variance_hf_paired_only_point"]
                / out["empirical_variance_acv_point"]
            )
        else:
            out["relative_efficiency_hf_paired_only_over_acv"] = float("nan")

        # Average elapsed times for profiling.
        out["avg_elapsed_acv_seconds"] = mean_or_nan(
            [r["elapsed_acv_seconds"] for r in subrows]
        )
        out["avg_elapsed_hf_budget_seconds"] = mean_or_nan(
            [r["elapsed_hf_budget_seconds"] for r in subrows]
        )
        out["avg_elapsed_hf_paired_only_seconds"] = mean_or_nan(
            [r["elapsed_hf_paired_only_seconds"] for r in subrows]
        )
        out["avg_elapsed_total_macrorep_seconds"] = mean_or_nan(
            [r["elapsed_total_macrorep_seconds"] for r in subrows]
        )

        summary_rows.append(out)

    return summary_rows


# =============================================================================
# Main workflow
# =============================================================================


def main():
    args = parse_args()
    total_start = time.time()  # reset timer at start of run

    # Parsing CLI settings
    solver_options = parse_solver_options(args.solver_options)
    candidate_with_replacement = interpret_candidate_sampling_flags(args)
    mrp_with_replacement = interpret_mrp_sampling_flags(args)
    reuse_pilot = interpret_pilot_reuse_flags(args)

    # Grid of parameter values to sweep over
    n_values = parse_int_list(args.n_values)
    budget_values = parse_float_list(args.budget_values)

    # Basic validation
    if args.macro_replications < 1:
        raise ValueError("--macro-replications must be >= 1.")
    if args.n_pilot < 1:
        raise ValueError("--n-pilot must be >= 1.")
    if len(n_values) == 0:
        raise ValueError("--n-values must contain at least one integer.")
    if len(budget_values) == 0:
        raise ValueError("--budget-values must contain at least one float.")

    # Create one timestamped output directory so all CSVs, debug files,
    # and any later plots from this run stay grouped together.
    run_dir = make_run_directory(
        model_module_name=args.model_module,
        model_name=args.model_name,
        lf_model_type=args.lf_model_type,
        candidate_seed=args.candidate_seed,
        mrp_seed=args.main_seed,
        candidate_scen_count=args.candidate_scen_count,
        acv_mrp_flag=True,
        m_values=[args.macro_replications],
        n_values=n_values,
        M_values=None,
        base_dir="budgeted_uq_outputs",
    )

    output_summary_csv = str(place_output_in_run_dir(run_dir, args.output_summary_csv))
    output_macro_csv = str(place_output_in_run_dir(run_dir, args.output_macro_csv))
    debug_json_file = str(place_output_in_run_dir(run_dir, args.debug_json_file))

    log(f"Created run directory: {run_dir}", t0=total_start, verbose=args.verbose)

    # -------------------------------------------------------------------------
    # Build/load candidate xhat once
    # -------------------------------------------------------------------------

    # Keep one fixed candidate solution across the whole sweep
    xhat, candidate_obj, xhat_path, candidate_elapsed = load_or_generate_candidate_xhat(
        args=args,
        candidate_with_replacement=candidate_with_replacement,
        run_dir=run_dir,
        verbose=args.verbose,
        t0=total_start,
    )

    # -------------------------------------------------------------------------
    # Load ensemble once to validate the module interface
    # -------------------------------------------------------------------------
    log("Loading base multifidelity ensemble ...", t0=total_start, verbose=args.verbose)
    base_ensemble = load_model_ensemble_for_uq(
        model_module_name=args.model_module,
        model_name=args.model_name,
        use_integer=args.use_integer,
        seed=args.main_seed,
        with_replacement=mrp_with_replacement,
        lf_model_type=args.lf_model_type,
    )
    hf_model = base_ensemble.high_fidelity_model()
    full_scenarios = hf_model.scenario_population().scenarios()
    hf_model.scenario_population().validate(full_scenarios)

    # -------------------------------------------------------------------------
    # True gap benchmark once
    # -------------------------------------------------------------------------
    true_gap_results = None
    true_gap_elapsed = 0.0

    # If requested, compute the finite-population benchmark once and reuse it
    # across the whole sweep
    if args.compute_true_gap:
        true_gap_results, true_gap_elapsed = compute_true_gap_with_timer(
            model=hf_model,
            xhat=xhat,
            solver_name=args.solver_name,
            solver_options=solver_options,
            verbose=args.verbose,
            t0=total_start,
        )
        true_gap_value = true_gap_results["true_gap"]
    else:
        true_gap_value = float("nan")

    # -------------------------------------------------------------------------
    # Pilot cache by n
    # -------------------------------------------------------------------------

    # Cache pilot results by n so the same pilot covariance/cost estimates can be
    # reused across budgets and macro-reps when desired.
    pilot_cache: Dict[int, Dict[str, Any]] = {}

    macro_rows: List[Dict[str, Any]] = []

    # Keep a structured debug object so expensive runs can be audited later
    # without re-running the experiment.
    debug_obj: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "xhat_file": str(xhat_path),
        "candidate_obj": safe_float(candidate_obj),
        "candidate_elapsed_seconds": safe_float(candidate_elapsed),
        "true_gap_elapsed_seconds": safe_float(true_gap_elapsed),
        "settings": {
            "model_module": args.model_module,
            "model_name": args.model_name,
            "lf_model_type": args.lf_model_type,
            "solver_name": args.solver_name,
            "alpha": args.alpha,
            "main_seed": args.main_seed,
            "candidate_seed": args.candidate_seed,
            "n_values": n_values,
            "budget_values": budget_values,
            "macro_replications": args.macro_replications,
            "n_pilot": args.n_pilot,
            "count_pilot_cost_against_budget": args.count_pilot_cost_against_budget,
            "reuse_pilot_across_macroreps": reuse_pilot,
        },
        "pilot_by_n": {},
    }

    # -------------------------------------------------------------------------
    # Main loops: first n, then budget, then macro-replication
    # -------------------------------------------------------------------------
    # Outer loop over n:
    # pilot covariance/cost estimates are specific to that batch size.
    for n in n_values:
        log(
            f"=== Starting experiments for n={n} ===",
            t0=total_start,
            verbose=args.verbose,
        )

        # Reuse one pilot per n if requested. This avoids repeating the expensive
        # pilot phase for every budget and macro-replication.
        if reuse_pilot:
            pilot_info = run_pyapprox_pilot(
                ensemble=base_ensemble,
                xhat=xhat,
                batch_size=n,
                solver_name=args.solver_name,
                solver_options=solver_options,
                # use a different pilot seed for each n;
                # the same pilot is then reused across all budgets and macro-reps for that n
                seed=args.main_seed + 1000 * n,
                n_pilot=args.n_pilot,
                hf_cost_delay_seconds=args.hf_cost_delay_seconds,
                lf_cost_delay_seconds=args.lf_cost_delay_seconds,
                verbose=args.verbose,
                t0=total_start,
            )
            pilot_cache[n] = pilot_info

            # Store pilot diagnostics in debug object for later inspection
            debug_obj["pilot_by_n"][str(n)] = {
                "estimated_hf_cost": float(pilot_info["costs_np"][0]),
                "estimated_lf_cost": float(pilot_info["costs_np"][1]),
                "pilot_rho_hat": float(pilot_info["rho_hat_pilot"]),
                "pilot_elapsed_seconds": float(pilot_info["pilot_elapsed_seconds"]),
            }

        # Middle loop over total budget values.
        for budget_idx, budget in enumerate(budget_values):
            log(
                f"--- Budget sweep: n={n}, budget={budget} ---",
                t0=total_start,
                verbose=args.verbose,
            )

            # Inner loop over macro-replications so empirical metrics such as
            # coverage and realized improvement can be estimated by repetition.
            for macro_rep in range(args.macro_replications):
                # Optionally redo the pilot per macro-rep. This is more expensive
                # but closer to a full end-to-end use case.
                if reuse_pilot:
                    this_pilot_info = pilot_cache[n]
                else:
                    ensemble_for_pilot = load_model_ensemble_for_uq(
                        model_module_name=args.model_module,
                        model_name=args.model_name,
                        use_integer=args.use_integer,
                        seed=build_macro_seed(args.main_seed, n, budget_idx, macro_rep),
                        with_replacement=mrp_with_replacement,
                        lf_model_type=args.lf_model_type,
                    )
                    this_pilot_info = run_pyapprox_pilot(
                        ensemble=ensemble_for_pilot,
                        xhat=xhat,
                        batch_size=n,
                        solver_name=args.solver_name,
                        solver_options=solver_options,
                        seed=build_macro_seed(args.main_seed, n, budget_idx, macro_rep),
                        n_pilot=args.n_pilot,
                        hf_cost_delay_seconds=args.hf_cost_delay_seconds,
                        lf_cost_delay_seconds=args.lf_cost_delay_seconds,
                        verbose=args.verbose,
                        t0=total_start,
                    )

                # Run one full macro-replication for this (n, budget) setting:
                # allocate the budget, run ACV, run HF-only baselines, and record results.
                row = run_one_macrorep(
                    args=args,
                    model_module_name=args.model_module,
                    model_name=args.model_name,
                    lf_model_type=args.lf_model_type,
                    xhat=xhat,
                    n=n,
                    budget=budget,
                    budget_idx=budget_idx,
                    macro_rep=macro_rep,
                    alpha=args.alpha,
                    main_with_replacement=mrp_with_replacement,
                    pilot_info=this_pilot_info,
                    count_pilot_cost_against_budget=args.count_pilot_cost_against_budget,
                    true_gap=true_gap_value,
                    t0=total_start,
                )

                macro_rows.append(row)

    # -------------------------------------------------------------------------
    # Write outputs
    # -------------------------------------------------------------------------

    # Keep only feasible rows when building summaries, since infeasible rows do
    # not correspond to valid estimator runs under the requested budget.
    feasible_macro_rows = [r for r in macro_rows if r.get("allocation_feasible", False)]
    if len(feasible_macro_rows) == 0:
        raise RuntimeError(
            "No feasible experiment rows were produced. "
            "Try larger budgets or smaller n / pilot size."
        )

    # Aggregate the raw macro-rep rows into summary metrics
    summary_rows = summarize_macrorep_rows(feasible_macro_rows)

    write_csv(feasible_macro_rows, output_macro_csv)
    write_csv(summary_rows, output_summary_csv)

    log(
        f"Wrote macro-replication CSV: {output_macro_csv}",
        t0=total_start,
        verbose=args.verbose,
    )
    log(
        f"Wrote summary CSV: {output_summary_csv}", t0=total_start, verbose=args.verbose
    )

    if args.save_debug_json:
        debug_obj["true_gap_results"] = true_gap_results
        debug_obj["num_macro_rows_total"] = len(macro_rows)
        debug_obj["num_macro_rows_feasible"] = len(feasible_macro_rows)
        debug_obj["elapsed_total_seconds"] = float(elapsed_seconds(total_start))
        write_debug_json(debug_obj, debug_json_file)
        log(
            f"Wrote debug JSON: {debug_json_file}", t0=total_start, verbose=args.verbose
        )

    log("Budgeted UQ experiments complete.", t0=total_start, verbose=True)


if __name__ == "__main__":
    main()
