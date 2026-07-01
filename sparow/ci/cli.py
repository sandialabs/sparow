
import argparse
import csv
from html import parser
import importlib
import json
import numpy as np

from sparow.ci.mrp_options import MRPOptions
from sparow.ci.standard_mrp import StandardMRP
from sparow.ci.evaluate_true_optimality_gap import TrueOptimalityGapEvaluator
from sparow.ci.scenario_sampler import ScenarioSampler

# ============================================================================
# CLI argument parsing
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run standard MRP confidence intervals.")

    parser.add_argument("--model-module", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--use-integer", action="store_true")

    parser.add_argument("--solver-name", default="highs")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=12345)

    parser.add_argument("--with-replacement", action="store_true")
    parser.add_argument("--without-replacement", action="store_true")

    parser.add_argument("--xhat-file", required=True)

    # Single-run mode arguments
    parser.add_argument("--scenario-file", default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--compute-true-gap", action="store_true")

    # Grid-experiment mode arguments
    parser.add_argument("--grid-experiment", action="store_true")
    parser.add_argument("--candidate-scen-count", type=int, default=None)
    parser.add_argument("--candidate-seed", type=int, default=12345)
    parser.add_argument("--mrp-seed", type=int, default=54321)
    parser.add_argument("--m-values", type=str, default=None)
    parser.add_argument("--n-values", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--use-existing-xhat", action="store_true")

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


def run_mrp_grid_experiment(
    model_module_name,
    model_name,
    solver_name,
    candidate_scen_count,
    candidate_seed,
    with_replacement,
    alpha,
    mrp_seed,
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
    """
    problem_adapter = load_problem_adapter(
        model_module_name=model_module_name,
        model_name=model_name,
        use_integer=use_integer,
    )

    full_scenarios = problem_adapter.get_scenario_population()
    problem_adapter.validate_scenario_population(full_scenarios)

    if use_existing_xhat:
        xhat = load_xhat(xhat_file)
        candidate_ef_objective = np.nan
    else:
        xhat, candidate_ef_objective = build_candidate_solution(
            problem_adapter=problem_adapter,
            full_scenarios=full_scenarios,
            candidate_scen_count=candidate_scen_count,
            candidate_seed=candidate_seed,
            with_replacement=with_replacement,
            solver_name=solver_name,
        )
        save_xhat(xhat, xhat_file)

    true_gap_evaluator = TrueOptimalityGapEvaluator(
        problem_adapter=problem_adapter,
        scenarios=full_scenarios,
        solver_name=solver_name,
    )

    true_gap_results = true_gap_evaluator.compute_true_gap(xhat=xhat)

    true_optimal_value = true_gap_results["true_optimal_value"]
    candidate_true_objective = true_gap_results["xhat_true_value"]
    true_gap = true_gap_results["true_gap"]

    rows = []

    for m in m_values:
        for n in n_values:
            print(f"\n=== Running MRP for m={m}, n={n} ===")

            mrp_results = run_single_mrp_experiment(
                problem_adapter=problem_adapter,
                scenarios=full_scenarios,
                xhat=xhat,
                n=n,
                m=m,
                alpha=alpha,
                seed=mrp_seed,
                with_replacement=with_replacement,
                solver_name=solver_name,
            )

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
    print("ARGS:", args)

    if args.with_replacement and args.without_replacement:
        raise ValueError("Choose only one of --with-replacement or --without-replacement.")

    with_replacement = True
    if args.without_replacement:
        with_replacement = False

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
            with_replacement=with_replacement,
            alpha=args.alpha,
            mrp_seed=args.mrp_seed,
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

    xhat = load_xhat(args.xhat_file)
    if "ROOT" in xhat:
        xhat = xhat["ROOT"]

    results = run_single_mrp_experiment(
        problem_adapter=adapter,
        scenarios=scenarios,
        xhat=xhat,
        n=args.n,
        m=args.m,
        alpha=args.alpha,
        seed=args.seed,
        with_replacement=with_replacement,
        solver_name=args.solver_name,
    )

    print("\nMRP results:")
    print(f"Point estimate: {results['point_estimate']}")
    print(f"Sample variance: {results['sample_variance']}")
    print(f"Sample std dev: {results['sample_std']}")
    print(f"t-statistic: {results['t_statistic']}")
    print(f"Half-width: {results['half_width']}")
    print(f"CI: [{results['ci_lower']}, {results['ci_upper']}]")

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