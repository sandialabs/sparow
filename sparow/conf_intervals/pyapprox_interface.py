"""
PyApprox integration for SPAROW confidence-interval code.

This module provides:
  - a PyApprox-compatible wrapper for one replication-level gap model,
  - cost estimation,
  - translation from PyApprox allocation counts to ACV-MRP counts,
  - and a builder for a 2-model multifidelity PyApprox problem.

In this interface:
  - one PyApprox sample = one full scenario batch,
  - one replication in MRP / ACV-MRP = one full scenario batch,
  - model 0 = HF replication-level gap estimator,
  - model 1 = LF replication-level gap estimator.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox_benchmarks.problems.multifidelity_forward_uq import (
    MultifidelityForwardUQProblem,
)

from sparow.conf_intervals.protocols import (
    StochasticProgramModelProtocol,
    ModelEnsembleProtocol,
)
from sparow.conf_intervals.scenario_sampler import ScenarioBatchPrior


class PyApproxModelWrapper:
    """
    PyApprox model wrapper for one replication-level gap estimate.

    One PyApprox input sample is one batch of iid scenario draws.
    One PyApprox output is one scalar replication-level gap estimate.
    """

    def __init__(
        self,
        model: StochasticProgramModelProtocol,
        xhat: Dict[str, float],
        batch_size: int,
        solver_name: str,
        solver_options: Optional[dict] = None,
        artificial_delay_seconds: float = 0.0,
    ):
        if not isinstance(model, StochasticProgramModelProtocol):
            raise TypeError("model must satisfy StochasticProgramModelProtocol.")

        self.model = model
        self.xhat = xhat
        self.batch_size = batch_size
        self.solver_name = solver_name
        self.solver_options = solver_options
        self.artificial_delay_seconds = artificial_delay_seconds

        # The scenario-batch prior flattens one batch into a single PyApprox sample.
        self._scenario_dim = self.model.scenario_population().scenario_vector_dim()

    def nvars(self) -> int:
        """
        Return the input dimension (i.e. - dimension of a flattened batch of scenarios).
        """
        return self.batch_size * self._scenario_dim

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest.
        Here, each replication returns one scalar gap estimate.
        """
        return 1

    def _unflatten_batch(self, sample_column):
        """
        Convert one flattened batch of scenarios into a list of scenario vectors.
        """
        arr = np.asarray(sample_column, dtype=float).reshape(
            self.batch_size, self._scenario_dim
        )
        return arr.tolist()

    def _batch_to_scenarios(self, batch_vectors):
        """
        Rebuild a list of scenario dictionaries from one batch of scenario vectors.

        The adapter is responsible for the scenario-specific decode logic.
        """
        prob = 1.0 / self.batch_size
        scenarios = []

        # scenario population object owns the decode logic
        scenario_population = self.model.scenario_population()
        for scen_idx_within_batch, vec in enumerate(batch_vectors):
            scen = scenario_population.decode_scenario_vector(
                vec,
                scenario_id=f"pyapprox_batch_scen_{scen_idx_within_batch}",
            )
            scen["Probability"] = prob
            scenarios.append(scen)

        return scenarios

    def __call__(self, samples):
        """
        Standard PyApprox interface for evaluating the model on one or
        more sampled scenario batches.

        Parameters
        ----------
        samples : array-like
            Shape (nvars, nsamples). Each column is one flattened batch.

        Returns
        -------
        ndarray
            Shape (1, nsamples). Each entry is one replication-level gap estimate.
        """

        # Each column of samples is a flattened vector,
        # corresponding to a batch of scenarios for a single replication
        nsamples = samples.shape[1]

        # For each input (a batch of scenarios), store a scalar ouptut (replication's gap estimate)
        outputs = np.zeros((1, nsamples), dtype=float)

        # Optional artificial delay for cost experiments.
        # This amount of delay is applied once per replication sample
        if self.artificial_delay_seconds > 0:
            time.sleep(self.artificial_delay_seconds * nsamples)

        for col_idx in range(nsamples):

            # Rebuild one scenario batch from one PyApprox input column.
            batch_vectors = self._unflatten_batch(samples[:, col_idx])
            sampled_scenarios = self._batch_to_scenarios(batch_vectors)

            # Compute one replication-level gap estimate using the wrapped model.
            rep_result = self.model.replication_gap(
                xhat=self.xhat,
                sampled_scenarios=sampled_scenarios,
                solver_name=self.solver_name,
                solver_options=self.solver_options,
            )

            outputs[0, col_idx] = rep_result["gap_estimate"]

        return outputs


def estimate_model_cost(model, prior, ntrials: int = 10) -> float:
    """
    Estimate average wall-clock evaluation cost for a given model.

    Any artificial cost inflation already attached to the model wrapper
    is automatically reflected in this estimate.
    """
    # Draw ntrials independent PyApprox samples from the prior.
    # Here, each PyApprox sample is a batch of scenarios for a replication.
    samples = prior.rvs(ntrials)

    # Time how long it takes to evaluate the model on all of those sampled batches.
    t0 = time.time()
    _ = model(samples)
    elapsed = time.time() - t0

    # Return the average time per replication-level model evaluation.
    return elapsed / ntrials


def convert_pyapprox_allocation_to_acvmrp_params(nsamples_per_model) -> Tuple[int, int]:
    """
    Helper func that converts a 2-model PyApprox allocation into ACV-MRP counts.

    If PyApprox recommends:
        N_HF evaluations of model 0 (high fidelity)
        N_LF low-fidelity of model 1 (low fidelity)

    then set:
        m = N_HF
        M = N_LF - N_HF

    because ACV-MRP uses:
      - m paired HF/LF replications,
      - M additional LF-only replications.
    """
    # Convert the allocation output into a flat integer array.
    # For a 2-model ensemble, this should contain:
    #   nsamples[0] = number of HF evaluations (paired)
    #   nsamples[1] = number of LF evaluations (paired + additional)
    nsamples = np.asarray(nsamples_per_model).astype(int).flatten()

    if len(nsamples) != 2:
        raise ValueError("Expected exactly 2 models for ACV-MRP translation.")

    m = int(nsamples[0])  # num paired evals (for both HF and LF)
    total_lf = int(nsamples[1])  # total number of LF evals (paired and additional)
    M = max(0, total_lf - m)  # num additional LF evals

    return m, M


def build_pyapprox_mf_problem_from_ensemble(
    ensemble: ModelEnsembleProtocol,
    xhat: Dict[str, float],
    batch_size: int,
    solver_name: str,
    solver_options: Optional[dict] = None,
    seed: int = 12345,
    hf_cost_delay_seconds: float = 0.0,
    lf_cost_delay_seconds: float = 0.0,
):
    """
    Build a 2-model PyApprox multifidelity problem from a SPAROW model ensemble.

    One PyApprox input sample is one batch of iid sampled scenarios.
    Model 0 is the HF replication-level gap estimator.
    Model 1 is the LF replication-level gap estimator.
    """
    if not isinstance(ensemble, ModelEnsembleProtocol):
        raise TypeError("ensemble must satisfy ModelEnsembleProtocol.")

    bkd = NumpyBkd()

    hf_model = ensemble.high_fidelity_model()
    lf_model = ensemble.low_fidelity_model()

    # Use the HF scenario population to define the prior over scenario batches.
    # In the intended ACV setting, HF and LF share the same underlying batch draws.
    prior = ScenarioBatchPrior(
        bkd=bkd,
        scenario_population=hf_model.scenario_population(),
        batch_size=batch_size,
        seed=seed,
    )

    hf_wrapper = PyApproxModelWrapper(
        model=hf_model,
        xhat=xhat,
        batch_size=batch_size,
        solver_name=solver_name,
        solver_options=solver_options,
        artificial_delay_seconds=hf_cost_delay_seconds,
    )

    lf_wrapper = PyApproxModelWrapper(
        model=lf_model,
        xhat=xhat,
        batch_size=batch_size,
        solver_name=solver_name,
        solver_options=solver_options,
        artificial_delay_seconds=lf_cost_delay_seconds,
    )

    hf_cost = estimate_model_cost(hf_wrapper, prior, ntrials=20)
    lf_cost = estimate_model_cost(lf_wrapper, prior, ntrials=20)

    costs = bkd.array([hf_cost, lf_cost])

    problem = MultifidelityForwardUQProblem(
        name="sp_optimality_gap_estimate_problem",
        models=[hf_wrapper, lf_wrapper],
        costs=costs,
        prior=prior,
        description="HF/LF replication-level optimality-gap estimators",
    )

    return problem, bkd
