import os
import argparse
import csv
from html import parser
import importlib
import json
from pprint import pprint
import numpy as np
from scipy import stats

from sparow.ci.mrp_options import MRPOptions
from sparow.ci.acv_mrp_options import ACVMRPOptions
from sparow.ci.standard_mrp import StandardMRP
from sparow.ci.acv_mrp import ACVMRP
from sparow.ci.evaluate_true_optimality_gap import TrueOptimalityGapEvaluator
from sparow.ci.scenario_sampler import ScenarioSampler

# ============================================================================
# CLI argument parsing
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run standard MRP or ACV-MRP confidence intervals.")

    parser.add_argument("--model-module", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--use-integer", action="store_true")

    parser.add_argument("--solver-name", default="gurobi")
    parser.add_argument("--alpha", type=float, default=0.05)

    parser.add_argument("--xhat-file", required=True)

    # Candidate-solution generation sampling
    parser.add_argument("--candidate-with-replacement", action="store_true")
    parser.add_argument("--candidate-without-replacement", action="store_true")

    # MRP replication sampling
    parser.add_argument("--mrp-with-replacement", action="store_true")
    parser.add_argument("--mrp-without-replacement", action="store_true")

    # Single-run mode arguments
    parser.add_argument("--scenario-file", default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--M", type=int, default=0)  # Additional LF-only replications for ACV-MRP
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
    parser.add_argument("--acv-mrp", action="store_true", help="Run ACV-MRP instead of standard MRP")
    parser.add_argument("--M-values", type=str, default=None, help="M values for ACV-MRP grid experiment")

    return parser.parse_args()

# ============================================================================
# Generic helpers
# ============================================================================

def parse_int_list(s):
    """This is for processing comma-separated lists of integers from the command line, e.g. "1,2,3" -> [1, 2, 3]"""
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def load_scenarios(path):
    if path.endswith(".json"):
        with open(path, "r") as f:
            obj = json.load(f)
        return obj["scenarios"]

    if path.endswith(".npy"):
        obj = np.load(path, allow_pickle=True).tolist()
        if isinstance(obj, dict) and "scenarios" in obj:
            return obj["scenarios"]
        return obj

    raise ValueError("Scenario file must be .json or .npy")


def load_xhat(path):
    return np.load(path, allow_pickle=True).item()

def save_xhat(xhat, path):
    np.save(path, xhat, allow_pickle=True)

def load_problem_adapter(model_module_name, model_name=None, use_integer=False):
    """
    This function helps keep the core CI code model-agnostic:
    each model module (farmer, newsvendor, OPF, etc.) is responsible for
    dispatching internally to itws own appropriate model-specific adapter, while
    the core CI code here only needs to call one standard factory name.

    Parameters
    ----------
    model_module_name : str
        Python module name to import, e.g.
        "sparow_examples.farmers.MRPfarmers".
    model_name : str, optional
        Name of the specific model variant within that module, e.g.
        "Basic" or "Advanced". If None, the model module's default adapter
        is requested.
    use_integer : bool, optional
        Whether the requested model instance should use integer first-stage
        variables, if supported by the model module.

    Returns
    -------
    CIProblemAdapter
        An instantiated problem adapter compatible with the generic
        Sparow confidence-interval / MRP code.
    """

    model_module = importlib.import_module(model_module_name)

    if not hasattr(model_module, "get_ci_problem_adapter"):
        raise RuntimeError(f"Model module {model_module_name} must define get_ci_problem_adapter().")

    if model_name is None:
        return model_module.get_ci_problem_adapter(use_integer=use_integer)

    return model_module.get_ci_problem_adapter(
        model_name=model_name,
        use_integer=use_integer,
    )

def build_candidate_solution(
    problem_adapter,
    full_scenarios,
    candidate_scen_count,
    candidate_seed,
    with_replacement,
    solver_name,
):
    """
    Build a candidate first-stage solution, xhat, by drawing a sampled batch 
    from the full scenario population using ScenarioSampler.
    """
    sampler = ScenarioSampler(
        scenarios=full_scenarios,
        seed=candidate_seed,
        with_replacement=with_replacement,
    )

    sampled_candidate_scenarios = sampler.draw_scenarios(
        n=candidate_scen_count,
        replication_id=0, # Only one "replication" or set of samples needed to get candidate sol
    )

    candidate_model_data = problem_adapter.build_model_data(sampled_candidate_scenarios)

    solved_candidate = problem_adapter.solve_extensive_form(
        model_data=candidate_model_data,
        solver_name=solver_name,
    )

    candidate_obj = problem_adapter.get_objective_value(solved_candidate)
    xhat = problem_adapter.get_first_stage_solution(solved_candidate)

    return xhat, candidate_obj


# ================================================================================
# Core MRP runners
# ================================================================================

def run_single_mrp_experiment(problem_adapter, scenarios, xhat, n, m,
                              alpha, seed, with_replacement, solver_name,):

    options = MRPOptions(
        n=n,
        m=m,
        alpha=alpha,
        seed=seed,
        with_replacement=with_replacement,
        solver_name=solver_name,
    )

    mrp = StandardMRP(
        problem_adapter=problem_adapter,
        scenarios=scenarios,
        options=options,
    )

    return mrp.run(xhat=xhat)


def run_single_acvmrp_experiment(problem_adapter, scenarios, xhat, n, m, M,
                                alpha, seed, with_replacement, solver_name,):

    options = ACVMRPOptions(
        n=n,
        m=m,
        M=M,
        alpha=alpha,
        seed=seed,
        with_replacement=with_replacement,
        solver_name=solver_name,
    )

    acvmrp = ACVMRP(
        problem_adapter=problem_adapter,
        scenarios=scenarios,
        options=options,
    )

    return acvmrp.run(xhat=xhat)


def run_mrp_grid_experiment(
    model_module_name,
    model_name,
    solver_name,
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
):
    """
    Run a full grid experiment over (m, n) for one fixed candidate xhat.

    This function:
      1. loads the adapter and full scenario population,
      2. optionally builds or loads xhat,
      3. computes the exact true optimality gap,
      4. runs MRP for all (m, n) parameter combinations,
      5. writes the results to a CSV file.

    Note that we do the following to make result comparisons easier:
        - We use the same candidate xhat for all (m, n) combinations.
        - For each replication index k, first draw one superset sample of size
    n_max = max(n_values) and fix its ordering. Then for smaller n value experiments, 
    use the first n scenarios from that same sampled superset.
    This means the sampled scenarios are nested:
    n_small < n_large  ==>  sample(n_small) is a subset of sample(n_large)
    """
    problem_adapter = load_problem_adapter(
        model_module_name=model_module_name,
        model_name=model_name,
        use_integer=use_integer,
    )

    full_scenarios = problem_adapter.get_scenario_population()
    problem_adapter.validate_scenario_population(full_scenarios)

    # ------------------------------------------------------
    # Candidate xhat
    # ------------------------------------------------------
    if use_existing_xhat:
        xhat = load_xhat(xhat_file)
        candidate_ef_objective = np.nan
    else:
        xhat, candidate_ef_objective = build_candidate_solution(
            problem_adapter=problem_adapter,
            full_scenarios=full_scenarios,
            candidate_scen_count=candidate_scen_count,
            candidate_seed=candidate_seed,
            with_replacement=candidate_with_replacement,
            solver_name=solver_name,
        )
        save_xhat(xhat, xhat_file)

    # ------------------------------------------------------
    # Exact true gap
    # ------------------------------------------------------
    true_gap_evaluator = TrueOptimalityGapEvaluator(
        problem_adapter=problem_adapter,
        scenarios=full_scenarios,
        solver_name=solver_name,
    )

    true_gap_results = true_gap_evaluator.compute_true_gap(xhat=xhat)

    true_optimal_value = true_gap_results["true_optimal_value"]
    candidate_true_objective = true_gap_results["xhat_true_value"]
    true_gap = true_gap_results["true_gap"]

    print("\n=== True finite-population gap ===\n")
    print(f"True optimal value: {true_optimal_value}")
    print(f"Candidate true objective: {candidate_true_objective}")
    print(f"True gap: {true_gap}")
    print("\n==================================\n")

    # ------------------------------------------------------
    # Precompute superset of sampled scenarios once
    # ------------------------------------------------------
    n_values = sorted(n_values, reverse=True)   # sorted in descending order
    n_max = max(n_values)
    m_max = max(m_values)

    sampler = ScenarioSampler(
        scenarios=full_scenarios,
        seed=mrp_seed,
        with_replacement=mrp_with_replacement,
    )

    # Pre-draw the superset batch for every replication up to m_max
    # Each replication k gets one sample of size n_max, and smaller n's
    # will use prefixes of that sample.

    sampled_supersets = {} # key = rep_id, value = list of sampled scenarios of size n_max
    for rep_id in range(m_max):
        sampled_supersets[rep_id] = sampler.draw_scenarios(
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

            options = MRPOptions(
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
                problem_adapter=problem_adapter,
                scenarios=full_scenarios,
                options=options,
            )

            mrp_results = mrp.run(xhat=xhat)

            row = {
                "model_module": model_module_name,
                "model_name": model_name,
                "m": m,
                "n": n,
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
    args = parse_args()
    print("Parsed CLI arguments:")
    print("ARGS:", args)
    print("\n==================================\n")

    # Candidate-solution generation replacement rule
    if args.candidate_with_replacement and args.candidate_without_replacement:
        raise ValueError("Choose only one of --candidate-with-replacement or --candidate-without-replacement.")

    candidate_with_replacement = True
    if args.candidate_without_replacement:
        candidate_with_replacement = False

    # MRP replication replacement rule
    if args.mrp_with_replacement and args.mrp_without_replacement:
        raise ValueError("Choose only one of --mrp-with-replacement or --mrp-without-replacement.")

    mrp_with_replacement = True
    if args.mrp_without_replacement:
        mrp_with_replacement = False

    # ----------------------------------------------------------------------
    # Grid-experiment mode
    # ----------------------------------------------------------------------
    if args.grid_experiment:
        if args.candidate_scen_count is None:
            raise ValueError("--candidate-scen-count is required in --grid-experiment mode.")
        if args.m_values is None or args.n_values is None:
            raise ValueError("--m-values and --n-values are required in --grid-experiment mode.")
        if args.output_csv is None:
            raise ValueError("--output-csv is required in --grid-experiment mode.")

        results = run_mrp_grid_experiment(
            model_module_name=args.model_module,
            model_name=args.model_name,
            solver_name=args.solver_name,
            candidate_scen_count=args.candidate_scen_count,
            candidate_seed=args.candidate_seed,
            candidate_with_replacement=candidate_with_replacement,
            alpha=args.alpha,
            mrp_seed=args.mrp_seed,
            mrp_with_replacement=mrp_with_replacement,
            m_values=parse_int_list(args.m_values),
            n_values=parse_int_list(args.n_values),
            xhat_file=args.xhat_file,
            use_existing_xhat=args.use_existing_xhat,
            output_csv=args.output_csv,
            use_integer=args.use_integer,
        )

        print("\nGrid experiment complete.")
        print(f"Wrote CSV: {args.output_csv}")
        print(f"Wrote xhat: {args.xhat_file}")
        return

    # ----------------------------------------------------------------------
    # Single-run mode
    # ----------------------------------------------------------------------
    if args.scenario_file is None:
        raise ValueError("--scenario-file is required in single-run mode.")
    if args.n is None or args.m is None:
        raise ValueError("--n and --m are required in single-run mode.")

    adapter = load_problem_adapter(
        model_module_name=args.model_module,
        model_name=args.model_name,
        use_integer=args.use_integer,
    )

    scenarios = load_scenarios(args.scenario_file)
    adapter.validate_scenario_population(scenarios)
    print(f"Loaded {len(scenarios)} scenarios from {args.scenario_file}.")

    if os.path.exists(args.xhat_file):
        xhat = load_xhat(args.xhat_file)
        if "ROOT" in xhat:
            xhat = xhat["ROOT"]
        print(f"Loaded candidate solution xhat from {args.xhat_file}:")
        print(f"xhat: {xhat}")
    else:
        if args.candidate_scen_count is None:
            raise ValueError(
                "--candidate-scen-count is required in single-run mode when xhat file does not already exist."
            )

        print(f"No existing xhat file found at {args.xhat_file}. Generating candidate solution...")
        print(f"Candidate sample size: {args.candidate_scen_count}")
        print(f"Candidate seed: {args.candidate_seed}")
        print(f"Candidate sampling with replacement: {candidate_with_replacement}")

        xhat, candidate_obj = build_candidate_solution(
            problem_adapter=adapter,
            full_scenarios=scenarios,
            candidate_scen_count=args.candidate_scen_count,
            candidate_seed=args.candidate_seed,
            with_replacement=candidate_with_replacement,
            solver_name=args.solver_name,
        )

        save_xhat(xhat, args.xhat_file)

        print(f"Generated candidate solution xhat and wrote it to {args.xhat_file}:")
        print(f"xhat: {xhat}")
        print(f"Candidate EF objective: {candidate_obj}")

    if args.acv_mrp:

        # Run ACV-MRP
        print("\n=== Running ACV-MRP ===")
        results = run_single_acvmrp_experiment(
            problem_adapter=adapter,
            scenarios=scenarios,
            xhat=xhat,
            n=args.n,
            m=args.m,
            M=args.M,
            alpha=args.alpha,
            seed=args.mrp_seed,
            with_replacement=mrp_with_replacement,
            solver_name=args.solver_name,
        )

        print("\nACV-MRPResults:\n")

        print(f"ACV Point Estimator: {results['point_estimate']}")
        print(f"Sample variance (HF): {results['sample_variance_F']}")
        print(f"Plug-in variance estimate for ACV estimator: {results['variance_acv_estimator']}")
        print(f"Variance reduction factor: {results['variance_reduction_factor']}")

        print("\n")

        print(f"Sample variance (paired LF): {results['sample_variance_G_paired']}")
        print(f"Sample covariance (F,G): {results['sample_covariance_FG']}")
        print(f"Estimated sample correlation (rho): {results['sample_correlation']}")
        print(f"Estimated control variate coefficient (alpha): {results['control_variate_coefficient']}")
        print(f"z_statistic: {results['z_statistic']}")
        print(f"Half-width: {results['half_width']}")
        print(f"CI: [{results['ci_lower']}, {results['ci_upper']}]")

    else:

        # Run standard MRP
        print("\n=== Running Standard MRP ===")
        results = run_single_mrp_experiment(
            problem_adapter=adapter,
            scenarios=scenarios,
            xhat=xhat,
            n=args.n,
            m=args.m,
            alpha=args.alpha,
            seed=args.mrp_seed,
            with_replacement=mrp_with_replacement,
            solver_name=args.solver_name,
        )

        print("\nStandard MRP Results:")
        print(f"Point estimate: {results['point_estimate']}")
        print(f"Sample variance: {results['sample_variance']}")
        print(f"Sample std dev: {results['sample_std']}")
        print(f"t-statistic: {results['t_statistic']}")
        print(f"Half-width: {results['half_width']}")
        print(f"CI: [{results['ci_lower']}, {results['ci_upper']}]")

        print("\n==================================\n")
        print("REFERENCE VALUES FOR COMPARISON AGAINST BOOT SP OUTPUTS:")
        print(f"Reference CI (two-sided normal): [{results['reference_ci_lower_two_sided_normal']}, {results['reference_ci_upper_two_sided_normal']}]")
        print("\n==================================\n")

    if args.compute_true_gap:
        true_gap_evaluator = TrueOptimalityGapEvaluator(
            problem_adapter=adapter,
            scenarios=scenarios,
            solver_name=args.solver_name,
        )
        true_gap = true_gap_evaluator.compute_true_gap(xhat=xhat)

        print("\nTrue finite-population gap:")
        print(f"True optimal value: {true_gap['true_optimal_value']}")
        print(f"xhat true value: {true_gap['xhat_true_value']}")
        print(f"True gap: {true_gap['true_gap']}")


if __name__ == "__main__":
    main()