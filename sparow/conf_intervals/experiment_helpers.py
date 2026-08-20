import importlib
import json
import numpy as np

from pathlib import Path
from datetime import datetime
import re

from sparow.conf_intervals.protocols import (
    StochasticProgramModelProtocol,
    ModelEnsembleProtocol,
)

# ============================================================================
# Generic helpers
# ============================================================================


def parse_int_list(s):
    """This is for processing comma-separated lists of integers from the command line, e.g. "1,2,3" -> [1, 2, 3]"""
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def parse_float_list(s):
    """Parse a comma-separated list of floats."""
    return [float(x.strip()) for x in s.split(",") if x.strip()]

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
