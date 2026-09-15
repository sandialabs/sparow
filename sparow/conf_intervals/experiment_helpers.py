import importlib
import json
import numpy as np
import os
import csv

from pathlib import Path
from datetime import datetime
import re
from typing import Any, Dict, List, Optional, Tuple
import time

from sparow.conf_intervals.protocols import (
    StochasticProgramModelProtocol,
    ModelEnsembleProtocol,
)

from sparow.conf_intervals.evaluate_true_optimality_gap import (
    TrueOptimalityGapEvaluator,
)

from sparow.conf_intervals.options import UQOptions
from sparow.conf_intervals.standard_mrp import StandardMRP
from sparow.conf_intervals.acv_mrp import ACVMRP

# ============================================================================
# Generic helpers
# ============================================================================


def parse_int_list(s):
    """This is for processing comma-separated lists of integers from the command line, e.g. "1,2,3" -> [1, 2, 3]"""
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s):
    """Parse a comma-separated list of floats."""
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def mean_or_nan(values: List[float]) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def var_or_nan(values: List[float]) -> float:
    if len(values) < 2:
        return float("nan")
    return float(np.var(values, ddof=1))


# ============================================================================
# Helpers for saving experiment results to dedicated subdirectory
# ============================================================================


def _create_safe_path_name(value):
    """Convert a value to a filesystem-friendly string."""
    s = str(value)
    s = s.replace(" ", "")
    s = s.replace("/", "-")
    s = s.replace(",", "_")
    s = re.sub(
        r"[^A-Za-z0-9_.=-]+", "-", s
    )  # Replace any other non-alphanumeric characters with hyphens
    return s


def _module_basename(model_module_name):
    """Use the last component of the module path for directory naming."""
    return model_module_name.split(".")[-1]


def get_model_module_parent_dir(model_module_name):
    """
    Return the parent directory containing the model module file.
    """
    model_module = importlib.import_module(model_module_name)
    model_file = Path(model_module.__file__).resolve()
    return model_file.parent


def make_run_directory(
    model_module_name,
    model_name,
    lf_model_type,
    candidate_seed,
    mrp_seed,
    candidate_scen_count,
    acv_mrp_flag=None,
    m_values=None,
    n_values=None,
    M_values=None,
    base_dir="experiment_outputs",
):
    """
    Create a unique output directory for one experiment run.

    The output directory is placed in a dedicated experiment_outputs folder
    located alongside the model module file.
    """
    if acv_mrp_flag is None:
        raise ValueError("acv_mrp_flag must be specified as True or False.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parts = [_create_safe_path_name(_module_basename(model_module_name))]

    if model_name is not None:
        parts.append(f"model-{_create_safe_path_name(model_name)}")

    parts.append(f"lf-{_create_safe_path_name(lf_model_type)}")
    parts.append(f"candN-{_create_safe_path_name(candidate_scen_count)}")
    parts.append(f"candSeed-{_create_safe_path_name(candidate_seed)}")
    parts.append(f"mainSeed-{_create_safe_path_name(mrp_seed)}")

    if m_values is not None:
        parts.append(f"m-{'_'.join(map(str, m_values))}")
    if n_values is not None:
        parts.append(f"n-{'_'.join(map(str, n_values))}")
    if acv_mrp_flag and M_values is not None:
        parts.append(f"M-{'_'.join(map(str, M_values))}")

    parts.append("acvmrp" if acv_mrp_flag else "mrp")
    parts.append(timestamp)

    run_name = "__".join(parts)

    # Put the run directory under the model module's parent directory
    model_parent_dir = get_model_module_parent_dir(model_module_name)
    base_dir = model_parent_dir / base_dir

    run_dir = Path(base_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def place_output_in_run_dir(run_dir, filename):
    """
    Put a filename inside run_dir unless it is already an absolute path.
    If filename includes only a basename, place it under run_dir.
    """
    p = Path(filename)
    if p.is_absolute():
        return p
    return run_dir / p.name


# =============================================================================
# CSV / JSON writing
# =============================================================================


def write_csv(rows: List[Dict[str, Any]], output_path: str):
    """Write rows to CSV, creating parent directories if needed."""
    if len(rows) == 0:
        raise RuntimeError(f"No rows available to write to CSV: {output_path}")

    output_path = str(output_path)
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_debug_json(debug_obj: Dict[str, Any], output_path: str):
    """Write optional debug metadata to JSON."""
    parent = os.path.dirname(str(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(debug_obj, f, indent=2)


# =============================================================================
# Timing / logging helpers
# =============================================================================


def elapsed_seconds(t0: float) -> float:
    return time.time() - t0


def log(msg: str, t0: Optional[float] = None, verbose: bool = True) -> None:
    """Print a timestamped progress message."""
    if not verbose:
        return
    if t0 is None:
        print(msg)
    else:
        print(f"[elapsed {(elapsed_seconds(t0)):.2f}s] {msg}")


# ============================================================================
# Helpers for stochastic programming instances in SPAROW
# ============================================================================


def parse_solver_options(solver_options_raw):
    """
    If solver_options is None, return None. If already a dict, return it.
    If it is a JSON string, parse it.
    """
    if solver_options_raw is None:
        return None
    if isinstance(solver_options_raw, dict):
        return solver_options_raw
    if isinstance(solver_options_raw, str):
        raw = solver_options_raw.strip()
        if raw == "":
            return None
        return json.loads(raw)
    return solver_options_raw


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


def build_candidate_solution(
    model,
    candidate_scen_count,
    solver_name,
):
    """
    Build a candidate first-stage solution xhat using one sampled scenario batch.

    This assumes model was instantiated with the desired seed and with_replacement flag
    for candidate solution generation. So when you load the model for candidate generation,
    you must pass the candidate seed and replacement rule into load_sp_model_for_uq(...)
    """
    scenario_batch = model.draw_batch_of_scenarios(
        n=candidate_scen_count,
        replication_id=123456789,  # arbitrary rep id, does not overlap with any existing rep ids
        nested_sampling=False,
        precomputed_supersets=None,
    )

    solved_candidate = model.solve_saa(
        sampled_scenarios=scenario_batch,
        solver_name=solver_name,
        solver_options=None,
    )

    candidate_obj = model.get_objective_value(solved_candidate)
    xhat = model.get_first_stage_solution(solved_candidate)

    return xhat, candidate_obj


def load_or_generate_candidate_xhat(
    *,
    args,
    candidate_with_replacement: bool,
    run_dir: Path,
    verbose: bool,
    t0: float,
):
    """
    Load an existing candidate xhat or generate a new one.
    The candidate is kept fixed across the entire parameter sweep.
    """
    xhat_path = (
        str(place_output_in_run_dir(run_dir, args.xhat_file))
        if not args.use_existing_xhat
        else args.xhat_file
    )

    if args.use_existing_xhat:
        if not os.path.exists(xhat_path):
            raise FileNotFoundError(
                f"--use-existing-xhat was requested, but file does not exist: {xhat_path}"
            )
        log(f"Loading existing xhat from {xhat_path}", t0=t0, verbose=verbose)
        xhat = load_xhat(xhat_path)
        if "ROOT" in xhat:
            xhat = xhat["ROOT"]
        return xhat, float("nan"), xhat_path, 0.0

    if args.candidate_scen_count is None:
        raise ValueError(
            "--candidate-scen-count is required when generating a new xhat."
        )

    log("Generating new candidate xhat ...", t0=t0, verbose=verbose)
    cand_start = time.time()

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
    save_xhat(xhat, xhat_path)

    elapsed = time.time() - cand_start
    log(
        f"Generated candidate xhat and saved to {xhat_path} in {elapsed:.2f}s",
        t0=t0,
        verbose=verbose,
    )
    return xhat, candidate_obj, xhat_path, elapsed


def build_candidate_helper_args(
    *,
    model_module_name,
    model_name,
    use_integer,
    candidate_seed,
    candidate_scen_count,
    solver_name,
    use_existing_xhat,
    xhat_file,
):
    """
    Build the minimal args-like object expected by
    load_or_generate_candidate_xhat(...).
    """

    class _Args:
        pass

    args = _Args()
    args.model_module = model_module_name
    args.model_name = model_name
    args.use_integer = use_integer
    args.candidate_seed = candidate_seed
    args.candidate_scen_count = candidate_scen_count
    args.solver_name = solver_name
    args.use_existing_xhat = use_existing_xhat
    args.xhat_file = xhat_file
    return args


def compute_true_gap_with_timer(
    model, xhat, solver_name, solver_options, verbose=False, t0=None
):
    """
    Helper for computing the exact finite-population true gap at the start of grid experiments
    or a parameter sweep, with timer.

    This benchmark is reused across all budgets and macro-replications so that
    the expensive full-population solve is not repeated unnecessarily.

    Returns result dictionary and float for time elapsed.
    """
    log("Computing true finite-population gap benchmark ...", t0=t0, verbose=verbose)
    eval_start = time.time()

    evaluator = TrueOptimalityGapEvaluator(
        model=model,
        solver_name=solver_name,
        solver_options=solver_options,
    )
    results = evaluator.compute_true_gap(xhat=xhat)

    elapsed = time.time() - eval_start
    log(
        f"Finished true-gap computation in {elapsed:.2f}s. True gap = {results['true_gap']}",
        t0=t0,
        verbose=verbose,
    )
    return results, elapsed


def load_sp_model_for_uq(
    model_module_name,
    model_name=None,
    use_integer=False,
    seed=12345,
    with_replacement=True,
):
    """
    Load a single stochastic-program wrapper from a model module.

    The model module must define a function that is named get_sp_model_for_uq(...).
    That function must return an object satisfying the StochasticProgramProtocol.
    """
    model_module = importlib.import_module(model_module_name)

    if not hasattr(model_module, "get_sp_model_for_uq"):
        raise RuntimeError(
            f"Model module {model_module_name} must define get_sp_model_for_uq()."
        )

    if model_name is None:
        model = model_module.get_sp_model_for_uq(
            use_integer=use_integer,
            seed=seed,
            with_replacement=with_replacement,
        )
    else:
        model = model_module.get_sp_model_for_uq(
            model_name=model_name,
            use_integer=use_integer,
            seed=seed,
            with_replacement=with_replacement,
        )

    # Runtime-check the returned object against the protocol.
    # This gives users a clear failure early if their problem-specific
    # factory did not return the right kind of object.
    if not isinstance(model, StochasticProgramModelProtocol):
        raise RuntimeError(
            f"Object returned by {model_module_name}.get_sp_model_for_uq(...) "
            f"does not satisfy StochasticProgramModelProtocol."
        )

    return model


def load_model_ensemble_for_uq(
    model_module_name,
    model_name=None,
    use_integer=False,
    seed=12345,
    with_replacement=True,
    lf_model_type="classic",
):
    """
    Load a multifidelity model ensemble from a model module.

    The module must define get_model_ensemble_for_uq(...).
    """
    model_module = importlib.import_module(model_module_name)

    if not hasattr(model_module, "get_model_ensemble_for_uq"):
        raise RuntimeError(
            f"Model module {model_module_name} must define get_model_ensemble_for_uq()."
        )

    if model_name is None:
        ensemble = model_module.get_model_ensemble_for_uq(
            use_integer=use_integer,
            seed=seed,
            with_replacement=with_replacement,
            lf_model_type=lf_model_type,
        )
    else:
        ensemble = model_module.get_model_ensemble_for_uq(
            model_name=model_name,
            use_integer=use_integer,
            seed=seed,
            with_replacement=with_replacement,
            lf_model_type=lf_model_type,
        )

    if not isinstance(ensemble, ModelEnsembleProtocol):
        raise RuntimeError(
            f"Object returned by {model_module_name}.get_model_ensemble_for_uq(...) "
            "does not satisfy ModelEnsembleProtocol."
        )

    return ensemble


# =============================================================================
# Experiment runners
# These are the building blocks for grid experiments & parameter sweeps
# =============================================================================


def run_standard_mrp(
    *,
    model,
    xhat,
    n: int,
    m: int,
    alpha: float,
    seed: int,
    with_replacement: bool,
    solver_name: str,
    solver_options,
    verbose: bool,
):
    """
    Run standard MRP once with specified parameters for the given model and candidate
    first-stage solution, xhat.
    """
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
    mrp = StandardMRP(model=model, options=options)
    return mrp.run(xhat=xhat)


def run_acvmrp(
    *,
    ensemble,
    xhat,
    n: int,
    m: int,
    M: int,
    alpha: float,
    seed: int,
    with_replacement: bool,
    solver_name: str,
    solver_options,
    verbose: bool,
):
    """
    Run ACV-MRP once with specified parameters for the given model ensemble and candidate
    first-stage solution, xhat.
    """
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
    acv = ACVMRP(
        hf_model=ensemble.high_fidelity_model(),
        lf_model=ensemble.low_fidelity_model(),
        options=options,
    )
    return acv.run(xhat=xhat)
