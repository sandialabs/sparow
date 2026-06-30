
import argparse
import importlib
import json
import numpy as np

from sparow.ci.mrp_options import MRPOptions
from sparow.ci.standard_mrp import StandardMRP
from sparow.ci.evaluate_true_optimality_gap import TrueOptimalityGapEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description="Run standard MRP confidence intervals.")

    parser.add_argument("--model-module", required=True)
    parser.add_argument("--scenario-file", required=True)
    parser.add_argument("--xhat-file", required=True)

    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=12345)

    parser.add_argument("--with-replacement", action="store_true")
    parser.add_argument("--without-replacement", action="store_true")

    parser.add_argument("--solver-name", default="highs")
    parser.add_argument("--compute-true-gap", action="store_true")

    return parser.parse_args()


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


def main():
    args = parse_args()

    if args.with_replacement and args.without_replacement:
        raise ValueError("Choose only one of --with-replacement or --without-replacement.")

    with_replacement = True
    if args.without_replacement:
        with_replacement = False

    model_module = importlib.import_module(args.model_module)

    if not hasattr(model_module, "get_ci_problem_adapter"):
        raise RuntimeError(f"Model module {args.model_module} must define get_ci_problem_adapter().")

    adapter = model_module.get_ci_problem_adapter()

    scenarios = load_scenarios(args.scenario_file)
    adapter.validate_scenario_population(scenarios)

    xhat = load_xhat(args.xhat_file)
    if "ROOT" in xhat:
        xhat = xhat["ROOT"]

    options = MRPOptions(
        n=args.n,
        m=args.m,
        alpha=args.alpha,
        seed=args.seed,
        with_replacement=with_replacement,
        solver_name=args.solver_name,
    )

    mrp = StandardMRP(
        problem_adapter=adapter,
        scenarios=scenarios,
        options=options,
    )

    results = mrp.run(xhat=xhat)

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