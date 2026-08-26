"""
Helpers for PyApprox integration with SPAROW confidence-interval code.
These functions enable numerical experiments and parameter sweeps
that consider computational budget.
"""
import numpy as np
import math
import time

from pyapprox.statest.statistics import MultiOutputMean
from pyapprox.statest.mc_estimator import MCEstimator
from pyapprox.statest import MFMCEstimator
from pyapprox.statest.acv import default_allocator_factory
from pyapprox.statest.acv.base import FittedACVEstimator
from pyapprox.statest.allocation import MCAllocator
from pyapprox.optimization.minimize.scipy.slsqp import ScipySLSQPOptimizer

from sparow.conf_intervals.pyapprox_interface import (
    convert_pyapprox_allocation_to_acvmrp_params,
    build_pyapprox_mf_problem_from_ensemble,
)

from sparow.conf_intervals.experiment_helpers import log

# =============================================================================
# PyApprox pilot / allocation helpers
# =============================================================================


def run_pyapprox_pilot(
    *,
    ensemble,
    xhat,
    batch_size: int,
    solver_name: str,
    solver_options,
    seed: int,
    n_pilot: int,
    hf_cost_delay_seconds: float,
    lf_cost_delay_seconds: float,
    verbose: bool,
    t0: float,
):
    """
    Run one PyApprox pilot study for a fixed batch size n.

    The pilot study serves two purposes:
      1. estimate replication-level evaluation costs (wall-clock time) for each model,
      2. estimate correlation between the models' gap estimators

    In this workflow, one PyApprox sample corresponds to one full scenario batch 
    (i.e. - one replication in the MRP / ACV-MRP sense). 
    
    For each sampled batch (i.e. - each PyApprox sample), the high-fidelity 
    and low-fidelity models return one scalar replication-level optimality-gap estimate.

    Parameters
    ----------
    ensemble :
        Model ensemble satisfying the SPAROW multifidelity interface.
        Model at index 0 is HF model and model at index 1 is LF model.
    xhat : dict
        Fixed first-stage candidate solution.
    batch_size : int
        Number of iid sampled scenarios in each replication batch.
    solver_name : str
        Solver passed through to the wrapped SPAROW model evaluations.
    solver_options : dict or None
        Optional solver settings.
    seed : int
        Random seed used by the PyApprox prior when drawing pilot batches.
    n_pilot : int
        Number of pilot replications used for cost and covariance estimation.
    hf_cost_delay_seconds : float
        Optional artificial wall-clock delay added to each HF replication
        evaluation. Useful for controlled cost-ratio experiments.
    lf_cost_delay_seconds : float
        Optional artificial wall-clock delay added to each LF replication
        evaluation. Useful for controlled cost-ratio experiments.
    verbose : bool
        If True, print progress information.
    t0 : float
        Start time of the overall experiment, used only for elapsed-time logging.

    Returns
    -------
    dict
        Dictionary containing:
          - `problem` : PyApprox multifidelity problem object,
          - `bkd` : PyApprox backend,
          - `models` : PyApprox-wrapped models,
          - `variable` : PyApprox prior over full scenario batches,
          - `costs` : backend array of estimated model costs,
          - `costs_np` : NumPy array version of the estimated model costs,
          - `stat` : PyApprox statistic object with pilot quantities attached,
          - `cov_pilot` : backend pilot covariance matrix,
          - `cov_np` : NumPy version of the pilot covariance matrix,
          - `rho_hat_pilot` : pilot estimate of HF/LF correlation,
          - `pilot_elapsed_seconds` : elapsed wall-clock time for the pilot study.

    Notes
    -----
    The returned `stat` object already has the pilot covariance quantities set,
    so it can be used directly in later calls to PyApprox allocation routines.
    """
    log(
        f"Building PyApprox multifidelity problem for n={batch_size} ...",
        t0=t0,
        verbose=verbose,
    )
    pilot_start = time.time()

    # Build the PyApprox representation of the multifidelity problem.
    # One PyApprox input sample = one full replication batch of scenarios.
    problem, bkd = build_pyapprox_mf_problem_from_ensemble(
        ensemble=ensemble,
        xhat=xhat,
        batch_size=batch_size,
        solver_name=solver_name,
        solver_options=solver_options,
        seed=seed,
        hf_cost_delay_seconds=hf_cost_delay_seconds,
        lf_cost_delay_seconds=lf_cost_delay_seconds,
    )

    models = problem.models()
    variable = problem.prior()
    costs = problem.costs()

    # These costs are the inputs to PyApprox's sample allocation step.
    costs_np = bkd.to_numpy(costs)

    # Draw n_pilot independent scenario batches from the prior.
    # Each column of samples_pilot corresponds to one replication batch.
    # Each batch is one replication in the MRP / ACV-MRP sense.
    # Pilot covariance estimation uses shared pilot samples across all models.
    samples_pilot = variable.rvs(n_pilot)
    vals_pilot = [m(samples_pilot) for m in models]

    # MultiOutputMean is the statistic object representing "estimate the mean"
    # of the replication-level outputs. PyApprox uses the pilot covariance of
    # these outputs to construct sample-allocation rules.
    stat = MultiOutputMean(models[0].nqoi(), bkd)
    cov_pilot, = stat.compute_pilot_quantities(vals_pilot)
    stat.set_pilot_quantities(cov_pilot)

    # In the 2-model setting, this is the empirical analogue of \rho_{fg}.
    cov_np = bkd.to_numpy(cov_pilot)
    rho_hat_pilot = (
        cov_np[0, 1] / np.sqrt(cov_np[0, 0] * cov_np[1, 1])
        if cov_np.shape[0] >= 2 and cov_np[0, 0] > 0 and cov_np[1, 1] > 0
        else float("nan")
    )

    pilot_elapsed = time.time() - pilot_start
    log(
        (
            f"Finished pilot for n={batch_size} in {pilot_elapsed:.2f}s. "
            f"Estimated HF cost={costs_np[0]:.6f}, LF cost={costs_np[1]:.6f}, "
            f"Pilot estimated correlation rho={rho_hat_pilot:.4f}"
        ),
        t0=t0,
        verbose=verbose,
    )

    return {
        "problem": problem,
        "bkd": bkd,
        "models": models,
        "variable": variable,
        "costs": costs,
        "costs_np": costs_np,
        "stat": stat,
        "cov_pilot": cov_pilot,
        "cov_np": cov_np,
        "rho_hat_pilot": float(rho_hat_pilot),
        "pilot_elapsed_seconds": float(pilot_elapsed),
    }


def allocate_pyapprox_budget(
    *,
    pilot_info,
    total_budget: float,
    n_pilot: int,
    count_pilot_cost_against_budget: bool,
):
    """
    Allocate a given total wall-clock budget using PyApprox.

    This helper takes the output of a pilot study and determines how many
    HF and LF replication-level evaluations should be used in the main
    estimator to minimize variance under a fixed computational budget.

    If the pilot cost is counted against the budget, the allocator sees only the
    remaining budget after subtracting the pilot phase. Otherwise the pilot is
    treated as an external setup cost and the full budget is available for the
    main estimator.

    Parameters
    ----------
    pilot_info : dict
        Output of `run_pyapprox_pilot(...)`. Must contain the 
        estimated model costs and covariance information needed by PyApprox.
    total_budget : float
        Total available wall-clock budget for the experiment.
    n_pilot : int
        Number of pilot replications previously used. Used only to compute the total 
        pilot cost, if we want the pilot cost to be charged against the reported budget.
    count_pilot_cost_against_budget : bool
        If True, subtract the estimated pilot-study cost from `total_budget`
        before allocating the remaining budget.

    Returns
    -------
    dict
        Dictionary containing:
          - `allocation_feasible` : whether positive budget remained after pilot,
          - `pilot_cost` : estimated total pilot-study wall-clock cost,
          - `remaining_budget` : budget passed to the allocation routine,
          - `pyapprox_hf_total` : total HF evaluations recommended by PyApprox,
          - `pyapprox_lf_total` : total LF evaluations recommended by PyApprox,
          - `m_paired` : paired replication count for ACV-MRP,
          - `M_additional_lf` : additional LF-only replication count for ACV-MRP,
          - `predicted_pyapprox_std` : predicted standard deviation of the multifidelity estimator,
          - `predicted_pyapprox_var` : predicted variance of the multifidelity estimator.

    Notes
    -----
    If the remaining budget is nonpositive after optionally subtracting pilot
    cost, return an infeasible allocation with zero sample counts.
    """
    bkd = pilot_info["bkd"]
    costs_np = pilot_info["costs_np"]
    stat = pilot_info["stat"]
    costs = pilot_info["costs"]

    # The pilot study evaluates every model n_pilot times, so its total cost is
    # the sum of per-model costs multiplied by the number of pilot replications.
    estimated_pilot_cost = float(np.sum(costs_np) * n_pilot)

    # Either charge the pilot cost against the total
    # budget or treat it as external setup cost.
    remaining_budget = total_budget - estimated_pilot_cost if count_pilot_cost_against_budget else total_budget

    if remaining_budget <= 0:
        return {
            "allocation_feasible": False,
            "estimated_pilot_cost": estimated_pilot_cost,
            "remaining_budget": remaining_budget,
            "pyapprox_hf_total": 0,
            "pyapprox_lf_total": 0,
            "m_paired": 0,
            "M_additional_lf": 0,
            "predicted_pyapprox_std": float("nan"),
            "predicted_pyapprox_var": float("nan"),
        }

    # Build the PyApprox multifidelity estimator using the pilot
    # covariance quantities and estimated model costs.
    est = MFMCEstimator(stat, costs)

    # For internal ACV allocation optimization problem.
    optimizer = ScipySLSQPOptimizer(maxiter=200)
    allocator_factory = lambda est: default_allocator_factory(est, optimizer=optimizer)
    allocator = allocator_factory(est)

    # Ask PyApprox to allocate the remaining budget across the HF and LF
    # replication-level model evaluations.
    result = allocator.allocate(remaining_budget)
    fitted = FittedACVEstimator(est, result)

    # Convert the recommended total sample counts per model into a flat integer
    # NumPy array. In the 2-model setting:
    #   nsamples[0] = total HF evaluations,
    #   nsamples[1] = total LF evaluations.
    nsamples = np.asarray(fitted.nsamples_per_model()).astype(int).flatten()
    pyapprox_hf_total = int(nsamples[0])
    pyapprox_lf_total = int(nsamples[1])

    # For ACVMRP:
    #   m = number of paired replications, run for both HF and LF,
    #   M = additional LF model replications.
    m, M = convert_pyapprox_allocation_to_acvmrp_params(nsamples)

    pred_var = float(fitted.covariance()[0, 0])
    pred_std = math.sqrt(pred_var) if pred_var >= 0 else float("nan")

    return {
        "allocation_feasible": True,
        "estimated_pilot_cost": float(estimated_pilot_cost),
        "remaining_budget": float(remaining_budget),
        "pyapprox_hf_total": pyapprox_hf_total,
        "pyapprox_lf_total": pyapprox_lf_total,
        "m_paired": int(m),
        "M_additional_lf": int(M),
        "predicted_pyapprox_std": float(pred_std),
        "predicted_pyapprox_var": float(pred_var),
    }


def hf_replications_for_same_budget(
    *,
    pilot_info,
    total_budget: float,
    n_pilot: int,
    count_pilot_cost_against_budget: bool,
    verbose: bool = False,
    t0: float | None = None,
):
    """
    Compute the HF-only same-budget baseline using PyApprox's single-fidelity
    Monte Carlo estimator and allocator.

    This helper is the single-fidelity analogue of the multifidelity PyApprox
    allocation used for ACV-MRP.

    For transparency, the function also computes and prints the simple
    accounting heuristic: floor(usable_budget / estimated hf_cost,
    so that users can compare the PyApprox HF-only allocation against the
    direct budget-divided-by-cost rule.

    Parameters
    ----------
    pilot_info : dict
        Output of `run_pyapprox_pilot(...)`. Must contain:
          - `bkd`
          - `cov_pilot`
          - `costs`
          - `costs_np`
    total_budget : float
        Total wall-clock budget available for the experiment.
    n_pilot : int
        Number of pilot replications previously used. Used only to compute the total 
        pilot cost, if we want the pilot cost to be charged against the reported budget.
    count_pilot_cost_against_budget : bool
        If True, subtract the estimated pilot-study cost from `total_budget`
        before allocating the remaining budget.
    verbose : bool, optional
        If True, print diagnostic information to the terminal.
    t0 : float or None, optional
        Start time of the overall experiment, used for elapsed-time logging.

    Returns
    -------
    dict
        Dictionary containing:
          - `allocation_feasible` : whether positive budget remained,
          - `estimated_pilot_cost` : estimated total pilot-study cost,
          - `usable_budget` : budget passed to the HF-only allocator,
          - `hf_cost` : estimated cost per HF replication,
          - `hf_count_floor_heuristic` : floor(usable_budget / hf_cost),
          - `hf_pyapprox_count` : HF count allocated by PyApprox MCAllocator,
          - `hf_predicted_mc_std` : predicted standard deviation of the HF-only MC estimator,
          - `hf_predicted_mc_var` : predicted variance of the HF-only MC estimator,
          - `mc_fitted` : fitted PyApprox HF-only MC estimator object.

    Notes
    -----
    In the single-fidelity setting, the PyApprox MC allocation will often agree
    with the simple floor heuristic or differ by at most rounding details, but
    using `MCEstimator` + `MCAllocator` keeps the HF-only same-budget baseline
    methodologically parallel to the multifidelity PyApprox workflow.
    """
    bkd = pilot_info["bkd"]
    cov_pilot = pilot_info["cov_pilot"]
    costs = pilot_info["costs"]
    costs_np = pilot_info["costs_np"]

    # The pilot phase evaluates each model n_pilot times, so its total estimated
    # cost is the sum of per-model costs multiplied by n_pilot.
    estimated_pilot_cost = float(np.sum(costs_np) * n_pilot)

    # Either charge the pilot cost against the total
    # budget or treat it as external setup cost.
    usable_budget = total_budget - estimated_pilot_cost if count_pilot_cost_against_budget else total_budget

    hf_cost = float(costs_np[0])

    if hf_cost <= 0 or usable_budget <= 0:
        return {
            "allocation_feasible": False,
            "estimated_pilot_cost": float(estimated_pilot_cost),
            "usable_budget": float(usable_budget),
            "hf_cost": float(hf_cost),
            "hf_count_floor_heuristic": 0,
            "hf_pyapprox_count": 0,
            "hf_predicted_mc_std": float("nan"),
            "hf_predicted_mc_var": float("nan"),
            "mc_fitted": None,
        }

    # Simple accounting rule shown only for comparison against PyApprox's
    # single-fidelity MC allocation.
    hf_count_floor_heuristic = max(0, int(math.floor(usable_budget / hf_cost)))

    # Build the HF-only statistic using only the HF pilot covariance block.
    stat_mc = MultiOutputMean(1, bkd)
    stat_mc.set_pilot_quantities(cov_pilot[:1, :1])

    # Build the PyApprox HF-only MC estimator using only the HF cost entry.
    mc_est = MCEstimator(stat_mc, costs[:1])

    # Ask PyApprox to allocate the usable budget to the single-fidelity
    # high-fidelity Monte Carlo estimator.
    mc_fitted = MCAllocator(mc_est).allocate(usable_budget)

    # PyApprox returns the fitted number of HF samples as a length-1 array.
    hf_pyapprox_count = int(np.asarray(mc_fitted.nsamples_per_model()).astype(int).flatten()[0])

    hf_predicted_mc_var = float(mc_fitted.covariance()[0, 0])
    hf_predicted_mc_std = math.sqrt(hf_predicted_mc_var) if hf_predicted_mc_var >= 0 else float("nan")

    if verbose:
        log(
            (
                f"HF-only same-budget allocation: total_budget={total_budget:.6f}, "
                f"Estimated pilot_cost={estimated_pilot_cost:.6f}, usable_budget={usable_budget:.6f}"
            ),
            t0=t0,
            verbose=verbose,
        )
        log(
            f"Estimated HF replication cost: {hf_cost:.6f}",
            t0=t0,
            verbose=verbose,
        )
        log(
            f"HF-only floor heuristic floor(usable_budget / hf_cost) = {hf_count_floor_heuristic}",
            t0=t0,
            verbose=verbose,
        )
        log(
            f"PyApprox HF-only MC allocated sample count: {hf_pyapprox_count}",
            t0=t0,
            verbose=verbose,
        )
        log(
            f"PyApprox HF-only MC predicted std (same computational budget): {hf_predicted_mc_std:.6f}",
            t0=t0,
            verbose=verbose,
        )

    return {
        "allocation_feasible": True,
        "esimated_pilot_cost": float(estimated_pilot_cost),
        "usable_budget": float(usable_budget),
        "hf_cost": float(hf_cost),
        "hf_count_floor_heuristic": int(hf_count_floor_heuristic),
        "hf_pyapprox_count": int(hf_pyapprox_count),
        "hf_predicted_mc_std": float(hf_predicted_mc_std),
        "hf_predicted_mc_var": float(hf_predicted_mc_var),
        "mc_fitted": mc_fitted,
    }
