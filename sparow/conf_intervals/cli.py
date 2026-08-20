import os
import argparse
import csv

import numpy as np

from sparow.conf_intervals.options import UQOptions
from sparow.conf_intervals.standard_mrp import StandardMRP
from sparow.conf_intervals.acv_mrp import ACVMRP
from sparow.conf_intervals.evaluate_true_optimality_gap import (
    TrueOptimalityGapEvaluator,
)
from sparow.conf_intervals.experiment_helpers import (
    parse_int_list,
    parse_float_list,
    load_scenarios,
    load_xhat,
    save_xhat,
    load_sp_model_for_uq,
    load_model_ensemble_for_uq,
    build_candidate_solution,
    get_model_module_parent_dir,
    make_run_directory,
    place_output_in_run_dir
)

# ============================================================================
# CLI argument parsing
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run standard MRP or ACV-MRP confidence intervals."
    )

    parser.add_argument("--model-module", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--use-integer", action="store_true")

    parser.add_argument("--solver-name", default="gurobi")
    parser.add_argument("--solver-options", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.05)

    parser.add_argument("--xhat-file", required=True)

    # Candidate-solution generation sampling
    parser.add_argument("--candidate-with-replacement", action="store_true")
    parser.add_argument("--candidate-without-replacement", action="store_true")

    # MRP replication sampling
    parser.add_argument("--mrp-with-replacement", action="store_true")
    parser.add_argument("--mrp-without-replacement", action="store_true")

    # Single-run mode arguments
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument(
        "--M", type=int, default=0
    )  # Additional LF-only replications for ACV-MRP
    parser.add_argument("--compute-true-gap", action="store_true")
    parser.add_argument("--candidate-scen-count", type=int, default=None)

    # Grid-experiment mode arguments
    parser.add_argument("--grid-experiment", action="store_true")
    parser.add_argument("--candidate-seed", type=int, default=12345)
    parser.add_argument("--mrp-seed", type=int, default=54321)
    parser.add_argument("--m-values", type=str, default=None)
    parser.add_argument("--n-values", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--use-existing-xhat", action="store_true")

    # ACV-MRP specific arguments
    parser.add_argument(
        "--acv-mrp", action="store_true", help="Run ACV-MRP instead of standard MRP"
    )
    parser.add_argument(
        "--M-values",
        type=str,
        default=None,
        help="M values for ACV-MRP grid experiment",
    )
    parser.add_argument(
        "--lf-model-type",
        type=str,
        default=None,
        help="Select which concrete low-fidelity model to use when ACV-MRP requests low fidelity",
    )

    return parser.parse_args()


# ================================================================================
# Core MRP runners
# ================================================================================


def run_single_mrp_experiment(
    model,
    xhat,
    n,
    m,
    alpha,
    seed,
    with_replacement,
    solver_name,
    solver_options,
    verbose,
):
    options = UQOptions(
        n=n,
        m=m,
        alpha=alpha,
        seed=seed,
        with_replacement=with_replacement,
        solver_name=solver_name,
        solver_options=solver_options,
        verbose=verbose,
    )

    mrp = StandardMRP(
        model=model,
        options=options,
    )

    return mrp.run(xhat=xhat)


def run_single_acvmrp_experiment(
    ensemble,
    xhat,
    n,
    m,
    M,
    alpha,
    seed,
    with_replacement,
    solver_name,
    solver_options,
    verbose,
):

    options = UQOptions(
        n=n,
        m=m,
        M=M,
        alpha=alpha,
        seed=seed,
        with_replacement=with_replacement,
        solver_name=solver_name,
        solver_options=solver_options,
        verbose=verbose,
    )

    acvmrp = ACVMRP(
        hf_model=ensemble.high_fidelity_model(),
        lf_model=ensemble.low_fidelity_model(),
        options=options,
    )

    return acvmrp.run(xhat=xhat)


def run_mrp_grid_experiment(
    model_module_name,
    model_name,
    solver_name,
    solver_options,
    verbose,
    candidate_scen_count,
    candidate_seed,
    candidate_with_replacement,
    alpha,
    mrp_seed,
    mrp_with_replacement,
    m_values,
    n_values,
    xhat_file,
    use_existing_xhat,
    output_csv,
    use_integer=False,
    lf_model_type="classic",
):
    """
    Run a full grid experiment over (m, n) for one fixed candidate xhat.

    Note that we do the following to make result comparisons easier:
        - We use the same candidate xhat for all (m, n) combinations.
        - For each replication index k, first draw one superset sample of size
    n_max = max(n_values) and fix its ordering. Then for smaller n value experiments,
    use the first n scenarios from that same sampled superset.
    This means the sampled scenarios are nested:
    n_small < n_large  ==>  sample(n_small) is a subset of sample(n_large)
    """
    # Model used for MRP replications
    model = load_sp_model_for_uq(
        model_module_name=model_module_name,
        model_name=model_name,
        use_integer=use_integer,
        seed=mrp_seed,
        with_replacement=mrp_with_replacement,
    )

    full_scenarios = model.scenario_population().scenarios()
    model.scenario_population().validate(full_scenarios)

    # ------------------------------------------------------
    # Candidate xhat
    # ------------------------------------------------------
    if use_existing_xhat:
        if not os.path.exists(xhat_file):
            raise FileNotFoundError(
                f"--use-existing-xhat was requested, but xhat file does not exist: {xhat_file}"
            )
        print(f"Loading candidate solution stored at {xhat_file}")
        xhat = load_xhat(xhat_file)
        candidate_ef_objective = np.nan
        if verbose:
            print(f"xhat: {xhat}")
    else:
        print("Generating new candidate solution...")
        if verbose:
            print(f"Candidate sample size: {candidate_scen_count}")
            print(f"Candidate seed: {candidate_seed}")
            print(f"Candidate sampling with replacement: {candidate_with_replacement}")

        # Model used to draw a random scenario batch for candidate solution generation
        candidate_model = load_sp_model_for_uq(
            model_module_name=model_module_name,
            model_name=model_name,
            use_integer=use_integer,
            seed=candidate_seed,
            with_replacement=candidate_with_replacement,
        )

        xhat, candidate_ef_objective = build_candidate_solution(
            model=candidate_model,
            candidate_scen_count=candidate_scen_count,
            solver_name=solver_name,
        )
        if verbose:
            print(f"xhat: {xhat}")
        save_xhat(xhat, xhat_file)

    # ------------------------------------------------------
    # Exact true gap
    # ------------------------------------------------------
    true_gap_evaluator = TrueOptimalityGapEvaluator(
        model=model,
        solver_name=solver_name,
        solver_options=solver_options,
    )

    true_gap_results = true_gap_evaluator.compute_true_gap(xhat=xhat)

    true_optimal_value = true_gap_results["true_optimal_value"]
    candidate_true_objective = true_gap_results["xhat_true_value"]
    true_gap = true_gap_results["true_gap"]

    print("\n=== True finite-population gap ===\n")
    print(f"Total number of population scenarios: {len(full_scenarios)}")
    print(f"True optimal value: {true_optimal_value}")
    print(f"Candidate true objective: {candidate_true_objective}")
    print(f"True gap: {true_gap}")
    print("\n==================================\n")

    # ------------------------------------------------------
    # Precompute superset of sampled scenarios once
    # ------------------------------------------------------
    n_values = sorted(n_values, reverse=True)  # sorted in descending order
    n_max = max(n_values)
    m_max = max(m_values)

    # Pre-draw the superset batch for every replication up to m_max
    # Each replication k gets one sample of size n_max, and smaller n's
    # will use prefixes of that sample.

    sampled_supersets = (
        {}
    )  # key = rep_id, value = list of sampled scenarios of size n_max
    for rep_id in range(m_max):
        sampled_supersets[rep_id] = model.scenario_sampler().draw_scenarios(
            n=n_max,
            replication_id=rep_id,
        )

    rows = []

    # ------------------------------------------------------
    # Run StandardMRP for each fixed value of (m, n)
    # ------------------------------------------------------
    for m in m_values:
        for n in n_values:
            print(f"\n=== Running MRP for m={m}, n={n} ===")

            options = UQOptions(
                n=n,
                m=m,
                alpha=alpha,
                seed=mrp_seed,
                with_replacement=mrp_with_replacement,
                solver_name=solver_name,
                verbose=True,
                nested_sampling=True,
                precomputed_supersets=sampled_supersets,
            )

            mrp = StandardMRP(
                model=model,
                options=options,
            )

            mrp_results = mrp.run(xhat=xhat)

            row = {
                "model_module": model_module_name,
                "model_name": model_name,
                "lf_model_type": lf_model_type,
                "m": m,
                "n": n,
                "alpha": alpha,
                "true_optimal_value": true_optimal_value,
                "candidate_ef_objective": candidate_ef_objective,
                "candidate_true_objective": candidate_true_objective,
                "true_gap": true_gap,
                "point_estimate": mrp_results["point_estimate"],
                "sample_variance": mrp_results["sample_variance"],
                "sample_std_dev": mrp_results["sample_std"],
                "t_statistic": mrp_results["t_statistic"],
                "half_width": mrp_results["half_width"],
                "ci_lower": mrp_results["ci_lower"],
                "ci_upper": mrp_results["ci_upper"],
            }

            rows.append(row)

    fieldnames = list(rows[0].keys())

    output_csv = str(output_csv)
    output_csv_parent = os.path.dirname(output_csv)
    if output_csv_parent:
        os.makedirs(output_csv_parent, exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "xhat": xhat,
        "candidate_ef_objective": candidate_ef_objective,
        "true_optimal_value": true_optimal_value,
        "candidate_true_objective": candidate_true_objective,
        "true_gap": true_gap,
        "rows": rows,
    }


def run_acvmrp_grid_experiment(
    model_module_name,
    model_name,
    solver_name,
    solver_options,
    verbose,
    candidate_scen_count,
    candidate_seed,
    candidate_with_replacement,
    alpha,
    mrp_seed,
    mrp_with_replacement,
    m_values,
    n_values,
    M_values,
    xhat_file,
    use_existing_xhat,
    output_csv,
    use_integer=False,
    lf_model_type="classic",
):
    """
    Run a full ACV-MRP grid experiment over (m, n, M) for one fixed candidate xhat.
    """
    ensemble = load_model_ensemble_for_uq(
        model_module_name=model_module_name,
        model_name=model_name,
        use_integer=use_integer,
        seed=mrp_seed,
        with_replacement=mrp_with_replacement,
        lf_model_type=lf_model_type,
    )

    hf_model = ensemble.high_fidelity_model()
    lf_model = ensemble.low_fidelity_model()

    full_scenarios = hf_model.scenario_population().scenarios()
    hf_model.scenario_population().validate(full_scenarios)

    # ------------------------------------------------------
    # Candidate xhat
    # ------------------------------------------------------
    if use_existing_xhat:
        if not os.path.exists(xhat_file):
            raise FileNotFoundError(
                f"--use-existing-xhat was requested, but xhat file does not exist: {xhat_file}"
            )
        print(f"Loading candidate solution stored at {xhat_file}")
        xhat = load_xhat(xhat_file)
        candidate_ef_objective = np.nan
        if verbose:
            print(f"xhat: {xhat}")
    else:
        print("Generating new candidate solution...")
        if verbose:
            print(f"Candidate sample size: {candidate_scen_count}")
            print(f"Candidate seed: {candidate_seed}")
            print(f"Candidate sampling with replacement: {candidate_with_replacement}")
        candidate_model = load_sp_model_for_uq(
            model_module_name=model_module_name,
            model_name=model_name,
            use_integer=use_integer,
            seed=candidate_seed,
            with_replacement=candidate_with_replacement,
        )

        xhat, candidate_ef_objective = build_candidate_solution(
            model=candidate_model,
            candidate_scen_count=candidate_scen_count,
            solver_name=solver_name,
        )
        if verbose:
            print(f"xhat: {xhat}")
        save_xhat(xhat, xhat_file)

    # ------------------------------------------------------
    # Exact true gap
    # ------------------------------------------------------
    true_gap_evaluator = TrueOptimalityGapEvaluator(
        model=hf_model,
        solver_name=solver_name,
        solver_options=solver_options,
    )

    true_gap_results = true_gap_evaluator.compute_true_gap(xhat=xhat)

    true_optimal_value = true_gap_results["true_optimal_value"]
    candidate_true_objective = true_gap_results["xhat_true_value"]
    true_gap = true_gap_results["true_gap"]

    print("\n=== True finite-population gap ===\n")
    print(f"Total number of population scenarios: {len(full_scenarios)}")
    print(f"True optimal value: {true_optimal_value}")
    print(f"Candidate true objective: {candidate_true_objective}")
    print(f"True gap: {true_gap}")
    print("\n==================================\n")

    rows = []

    # ------------------------------------------------------
    # Precompute superset of sampled scenarios once
    # ------------------------------------------------------
    n_values = sorted(n_values, reverse=True)
    n_max = max(n_values)
    m_max = max(m_values)
    M_max = max(M_values)

    # Need enough supersets for paired replications and
    # additional LF-only replications
    total_replications_needed = m_max + M_max
    sampled_supersets = {}

    for rep_id in range(total_replications_needed):
        sampled_supersets[rep_id] = hf_model.scenario_sampler().draw_scenarios(
            n=n_max,
            replication_id=rep_id,
        )

    rows = []

    # ------------------------------------------------------
    # Run ACV-MRP for each fixed value of (m, n, M)
    # ------------------------------------------------------
    for m in m_values:
        for n in n_values:
            for M in M_values:
                print(f"\n=== Running ACV-MRP for m={m}, n={n}, M={M} ===")

                options = UQOptions(
                    n=n,
                    m=m,
                    M=M,
                    alpha=alpha,
                    seed=mrp_seed,
                    with_replacement=mrp_with_replacement,
                    solver_name=solver_name,
                    verbose=True,
                    nested_sampling=True,
                    precomputed_supersets=sampled_supersets,
                )

                acvmrp = ACVMRP(
                    hf_model=hf_model,
                    lf_model=lf_model,
                    options=options,
                )

                acv_results = acvmrp.run(xhat=xhat)

                row = {
                    "model_module": model_module_name,
                    "model_name": model_name,
                    "lf_model_type": lf_model_type,
                    "m": m,
                    "n": n,
                    "M": M,
                    "alpha": acv_results["alpha"],
                    "true_optimal_value": true_optimal_value,
                    "candidate_ef_objective": candidate_ef_objective,
                    "candidate_true_objective": candidate_true_objective,
                    "true_gap": true_gap,
                    "point_estimate": acv_results["point_estimate"],
                    "point_estimate_hf_only": acv_results["point_estimate_hf_only"],
                    "ci_lower": acv_results["ci_lower"],
                    "ci_upper": acv_results["ci_upper"],
                    "half_width": acv_results["half_width"],
                    "sample_variance_F": acv_results["sample_variance_F"],
                    "sample_variance_G_paired": acv_results["sample_variance_G_paired"],
                    "sample_covariance_FG": acv_results["sample_covariance_FG"],
                    "variance_acv_estimator": acv_results["variance_acv_estimator"],
                    "standard_error_acv": acv_results["standard_error_acv"],
                    "sample_correlation": acv_results["sample_correlation"],
                    "control_variate_coefficient": acv_results[
                        "control_variate_coefficient"
                    ],
                    "z_statistic": acv_results["z_statistic"],
                    "variance_reduction_factor": acv_results[
                        "variance_reduction_factor"
                    ],
                }

                rows.append(row)

    fieldnames = list(rows[0].keys())

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "xhat": xhat,
        "candidate_ef_objective": candidate_ef_objective,
        "true_optimal_value": true_optimal_value,
        "candidate_true_objective": candidate_true_objective,
        "true_gap": true_gap,
        "rows": rows,
    }


# ============================================================================
# Main entry point
# ============================================================================


def main():
    # ----------------------------------------------------------------------
    # Parse command-line arguments
    # ----------------------------------------------------------------------
    args = parse_args()

    if args.verbose:
        print("Parsed CLI arguments:")
        print("ARGS:", args)
        print("\n==================================\n")

    # ----------------------------------------------------------------------
    # Interpret candidate-generation sampling flags
    # ----------------------------------------------------------------------
    if args.candidate_with_replacement and args.candidate_without_replacement:
        raise ValueError(
            "Choose only one of --candidate-with-replacement or --candidate-without-replacement."
        )

    candidate_with_replacement = True
    if args.candidate_without_replacement:
        candidate_with_replacement = False

    # ----------------------------------------------------------------------
    # Interpret replication sampling flags
    # ----------------------------------------------------------------------
    if args.mrp_with_replacement and args.mrp_without_replacement:
        raise ValueError(
            "Choose only one of --mrp-with-replacement or --mrp-without-replacement."
        )

    mrp_with_replacement = True
    if args.mrp_without_replacement:
        mrp_with_replacement = False

    # ----------------------------------------------------------------------
    # Grid-experiment mode
    # ----------------------------------------------------------------------
    if args.grid_experiment:

        # --------------------------------------------------
        # Validate required grid-experiment arguments
        # --------------------------------------------------
        if args.candidate_scen_count is None:
            raise ValueError(
                "--candidate-scen-count is required in --grid-experiment mode."
            )
        if args.m_values is None or args.n_values is None:
            raise ValueError(
                "--m-values and --n-values are required in --grid-experiment mode."
            )
        if args.output_csv is None:
            raise ValueError("--output-csv is required in --grid-experiment mode.")

        # --------------------------------------------------
        # ACV-MRP grid experiment
        # --------------------------------------------------
        if args.acv_mrp:
            if args.M_values is None:
                raise ValueError("--M-values is required for ACV-MRP grid experiments.")

            parsed_m_values = parse_int_list(args.m_values)
            parsed_n_values = parse_int_list(args.n_values)
            parsed_M_values = parse_int_list(args.M_values)

            run_dir = make_run_directory(
                model_module_name=args.model_module,
                model_name=args.model_name,
                lf_model_type=args.lf_model_type,
                candidate_seed=args.candidate_seed,
                mrp_seed=args.mrp_seed,
                candidate_scen_count=args.candidate_scen_count,
                m_values=parsed_m_values,
                n_values=parsed_n_values,
                M_values=parsed_M_values,
                acv_mrp_flag=True,
            )

            if not args.use_existing_xhat:
                args.xhat_file = str(place_output_in_run_dir(run_dir, args.xhat_file))
            args.output_csv = str(place_output_in_run_dir(run_dir, args.output_csv))

            if args.verbose:
                print(f"Created run directory: {run_dir}")
                print(f"xhat file will be written to: {args.xhat_file}")
                print(f"CSV file will be written to: {args.output_csv}")

            results = run_acvmrp_grid_experiment(
                model_module_name=args.model_module,
                model_name=args.model_name,
                solver_name=args.solver_name,
                solver_options=args.solver_options,
                verbose=args.verbose,
                candidate_scen_count=args.candidate_scen_count,
                candidate_seed=args.candidate_seed,
                candidate_with_replacement=candidate_with_replacement,
                alpha=args.alpha,
                mrp_seed=args.mrp_seed,
                mrp_with_replacement=mrp_with_replacement,
                m_values=parsed_m_values,
                n_values=parsed_n_values,
                M_values=parsed_M_values,
                xhat_file=args.xhat_file,
                use_existing_xhat=args.use_existing_xhat,
                output_csv=args.output_csv,
                use_integer=args.use_integer,
                lf_model_type=args.lf_model_type,
            )

        # --------------------------------------------------
        # Standard MRP grid experiment
        # --------------------------------------------------
        else:
            parsed_m_values = parse_int_list(args.m_values)
            parsed_n_values = parse_int_list(args.n_values)

            run_dir = make_run_directory(
                model_module_name=args.model_module,
                model_name=args.model_name,
                lf_model_type=args.lf_model_type,
                candidate_seed=args.candidate_seed,
                mrp_seed=args.mrp_seed,
                candidate_scen_count=args.candidate_scen_count,
                m_values=parsed_m_values,
                n_values=parsed_n_values,
                M_values=None,
                acv_mrp_flag=False,
            )

            if not args.use_existing_xhat:
                args.xhat_file = str(place_output_in_run_dir(run_dir, args.xhat_file))
            args.output_csv = str(place_output_in_run_dir(run_dir, args.output_csv))

            if args.verbose:
                print(f"Created run directory: {run_dir}")
                print(f"xhat file will be written to: {args.xhat_file}")
                print(f"CSV file will be written to: {args.output_csv}")

            results = run_mrp_grid_experiment(
                model_module_name=args.model_module,
                model_name=args.model_name,
                solver_name=args.solver_name,
                solver_options=args.solver_options,
                verbose=args.verbose,
                candidate_scen_count=args.candidate_scen_count,
                candidate_seed=args.candidate_seed,
                candidate_with_replacement=candidate_with_replacement,
                alpha=args.alpha,
                mrp_seed=args.mrp_seed,
                mrp_with_replacement=mrp_with_replacement,
                m_values=parsed_m_values,
                n_values=parsed_n_values,
                xhat_file=args.xhat_file,
                use_existing_xhat=args.use_existing_xhat,
                output_csv=args.output_csv,
                use_integer=args.use_integer,
                lf_model_type=args.lf_model_type,
            )

        # --------------------------------------------------
        # Final grid-experiment reporting
        # --------------------------------------------------
        if args.verbose:
            print("\nGrid experiment complete.")

        if not os.path.exists(args.output_csv):
            raise RuntimeError(f"Expected CSV was not written: {args.output_csv}")

        print(f"Wrote CSV: {args.output_csv}")
        print(f"Wrote xhat: {args.xhat_file}")
        return

    # ----------------------------------------------------------------------
    # Single-run mode
    # ----------------------------------------------------------------------

    # --------------------------------------------------
    # Validate required single-run arguments
    # --------------------------------------------------
    if args.n is None or args.m is None:
        raise ValueError("--n and --m are required in single-run mode.")

    # --------------------------------------------------
    # Create output directory and xhat path
    # --------------------------------------------------
    run_dir = make_run_directory(
        model_module_name=args.model_module,
        model_name=args.model_name,
        lf_model_type=args.lf_model_type,
        candidate_seed=args.candidate_seed,
        mrp_seed=args.mrp_seed,
        candidate_scen_count=args.candidate_scen_count,
        m_values=[args.m],
        n_values=[args.n],
        M_values=[args.M] if args.acv_mrp and args.M is not None else None,
        acv_mrp_flag=args.acv_mrp,
    )

    if not args.use_existing_xhat:
        args.xhat_file = str(place_output_in_run_dir(run_dir, args.xhat_file))

    if args.verbose:
        print(f"Created run directory: {run_dir}")
        print(f"xhat file will be written to: {args.xhat_file}")

    # --------------------------------------------------
    # Load the primary single-fidelity model wrapper
    # --------------------------------------------------
    model = load_sp_model_for_uq(
        model_module_name=args.model_module,
        model_name=args.model_name,
        use_integer=args.use_integer,
        seed=args.mrp_seed,
        with_replacement=mrp_with_replacement,
    )

    scenarios = model.scenario_population().scenarios()
    model.scenario_population().validate(scenarios)

    if args.verbose:
        print(f"Loaded {len(scenarios)} scenarios from the model wrapper.")

    # --------------------------------------------------
    # Load or generate candidate solution xhat
    # --------------------------------------------------
    if os.path.exists(args.xhat_file) and args.use_existing_xhat:
        xhat = load_xhat(args.xhat_file)
        if "ROOT" in xhat:
            xhat = xhat["ROOT"]

        if args.verbose:
            print(f"Loaded candidate solution xhat from {args.xhat_file}:")
            print(f"xhat: {xhat}")
    else:
        if args.candidate_scen_count is None:
            raise ValueError(
                "--candidate-scen-count is required in single-run mode when xhat file does not already exist."
            )

        if args.verbose:
            print("Generating new candidate solution using subsampled scenarios...")
            print(f"Candidate sample size: {args.candidate_scen_count}")
            print(f"Candidate seed: {args.candidate_seed}")
            print(f"Candidate sampling with replacement: {candidate_with_replacement}")

        candidate_model = load_sp_model_for_uq(
            model_module_name=args.model_module,
            model_name=args.model_name,
            use_integer=args.use_integer,
            seed=args.candidate_seed,
            with_replacement=candidate_with_replacement,
        )

        xhat, candidate_obj = build_candidate_solution(
            model=candidate_model,
            candidate_scen_count=args.candidate_scen_count,
            solver_name=args.solver_name,
        )

        save_xhat(xhat, args.xhat_file)

        if args.verbose:
            print(
                f"Generated candidate solution xhat and wrote it to {args.xhat_file}:"
            )
            print(f"xhat: {xhat}")
            print(f"Candidate EF objective: {candidate_obj}")

    # --------------------------------------------------
    # Multifidelity ACV-MRP - single algorithm run
    # --------------------------------------------------
    if args.acv_mrp:
        ensemble = load_model_ensemble_for_uq(
            model_module_name=args.model_module,
            model_name=args.model_name,
            use_integer=args.use_integer,
            seed=args.mrp_seed,
            with_replacement=mrp_with_replacement,
            lf_model_type=args.lf_model_type,
        )

        hf_model = ensemble.high_fidelity_model()
        scenarios = hf_model.scenario_population().scenarios()
        hf_model.scenario_population().validate(scenarios)

        if args.verbose:
            print("\n=== Running ACV-MRP ===\n")

        results = run_single_acvmrp_experiment(
            ensemble=ensemble,
            xhat=xhat,
            n=args.n,
            m=args.m,
            M=args.M,
            alpha=args.alpha,
            seed=args.mrp_seed,
            with_replacement=mrp_with_replacement,
            solver_name=args.solver_name,
            solver_options=args.solver_options,
            verbose=args.verbose,
        )

        if args.verbose:
            print("\nACV-MRP Results:\n")
            print(f"ACV Point Estimator: {results['point_estimate']}")
            print(f"Sample variance (HF): {results['sample_variance_F']}")
            print(
                f"Plug-in variance estimate for ACV estimator: {results['variance_acv_estimator']}"
            )
            print(f"Variance reduction factor: {results['variance_reduction_factor']}")
            print("\n")
            print(f"Sample variance (paired LF): {results['sample_variance_G_paired']}")
            print(f"Sample covariance (F,G): {results['sample_covariance_FG']}")
            print(
                f"Estimated sample correlation (rho): {results['sample_correlation']}"
            )
            print(
                f"Estimated control variate coefficient (alpha): {results['control_variate_coefficient']}"
            )
            print(f"z_statistic: {results['z_statistic']}")
            print(
                f"Margin of Error (confidence interval upper bound - point estimate): {results['half_width']}"
            )
            print(
                f"Confidence Interval: [{results['ci_lower']}, {results['ci_upper']}]"
            )

    # -----------------------------------------------------
    # Standard single-fidelity MRP - single algorithm run
    # -----------------------------------------------------
    else:
        if args.verbose:
            print("\n=== Running Standard MRP ===\n")

        results = run_single_mrp_experiment(
            model=model,
            xhat=xhat,
            n=args.n,
            m=args.m,
            alpha=args.alpha,
            seed=args.mrp_seed,
            with_replacement=mrp_with_replacement,
            solver_name=args.solver_name,
            solver_options=args.solver_options,
            verbose=args.verbose,
        )

        if args.verbose:
            print("\nStandard MRP Results:\n")
            print(f"Point estimate: {results['point_estimate']}")
            print(f"Sample variance: {results['sample_variance']}")
            print(f"Sample std dev: {results['sample_std']}")
            print(f"t-statistic: {results['t_statistic']}")
            print(
                f"Margin of Error (confidence interval upper bound - point estimate): {results['half_width']}"
            )
            print(
                f"Confidence Interval: [{results['ci_lower']}, {results['ci_upper']}]"
            )

    # --------------------------------------------------
    # Optional true-gap computation
    # --------------------------------------------------
    if args.compute_true_gap:
        if args.acv_mrp:
            true_gap_evaluator = TrueOptimalityGapEvaluator(
                model=hf_model,
                solver_name=args.solver_name,
                solver_options=args.solver_options,
            )
        else:
            true_gap_evaluator = TrueOptimalityGapEvaluator(
                model=model,
                solver_name=args.solver_name,
                solver_options=args.solver_options,
            )

        true_gap = true_gap_evaluator.compute_true_gap(xhat=xhat)

        if args.verbose:
            print("\nTrue finite-population gap:\n")
            print(f"True optimal value: {true_gap['true_optimal_value']}")
            print(f"xhat true value: {true_gap['xhat_true_value']}")
            print(f"True gap: {true_gap['true_gap']}")


if __name__ == "__main__":
    main()
