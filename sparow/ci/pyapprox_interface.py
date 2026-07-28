import numpy as np
import time
from typing import Optional, Dict, Any, Tuple

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox_benchmarks.problems.multifidelity_forward_uq import (
    MultifidelityForwardUQProblem,
)

from sparow.ci.scenario_sampler import ScenarioBatchPrior


class PyApproxModelWrapper:
    """
    PyApprox model wrapper for one replication-level gap estimate.

    One PyApprox input sample is one batch of iid scenario draws.
    One PyApprox output is one scalar replication-level gap estimate.
    """

    def __init__(
        self,
        problem_adapter,
        xhat: Dict[str, float],
        batch_size: int,
        fidelity: str,
        solver_name: str,
        solver_options: Optional[dict] = None,
        artificial_delay_seconds: float = 0.0,
    ):
        self.problem_adapter = problem_adapter
        self.xhat = xhat
        self.batch_size = batch_size
        self.fidelity = fidelity
        self.solver_name = solver_name
        self.solver_options = solver_options
        self.artificial_delay_seconds = artificial_delay_seconds
        self._scenario_dim = self.problem_adapter.scenario_vector_dim()

    def nvars(self) -> int:
        """
        Return the input dimension (i.e. - dimension of a flattened batch of scenarios).

        One batch contains batch_size iid draws of scenario vectors.
        Each scenario vector has scenario_vector_dim scalar entries.
        """
        return self.batch_size * self._scenario_dim

    def nqoi(self) -> int:
        """Return the number of quantities of interest."""
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
        for scen_idx_within_batch, vec in enumerate(batch_vectors):
            scen = self.problem_adapter.decode_scenario_vector(
                vec, scenario_id=f"pyapprox_batch_scen_{scen_idx_within_batch}"
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
        values : ndarray
            Shape (1, nsamples). Each entry is one replication-level gap estimate.
        """

        # Each column of samples is a flattened vector,
        # corresponding to a batch of scenarios for a single replication
        nsamples = samples.shape[1]

        # For each input (a batch of scenarios), store a scalar ouptut (replication's gap estimate)
        outputs = np.zeros((1, nsamples), dtype=float)

        # Apply optional artificial delay once per replication sample
        if self.artificial_delay_seconds > 0:
            time.sleep(self.artificial_delay_seconds * nsamples)

        for col_idx in range(nsamples):

            # At col_idx, extract the flattened vector containing data for a full batch of scenarios,
            # and reshape it back into a list of scenario vectors
            batch_vectors = self._unflatten_batch(samples[:, col_idx])
            sampled_scenarios = self._batch_to_scenarios(batch_vectors)

            # Sanity check: ensure scenarios were reconstructed correctly
            self.problem_adapter.validate_scenario_population(sampled_scenarios)

            # Build optimization problem that is parameterized by this batch of scenarios
            model_data_k = self.problem_adapter.build_model_data(sampled_scenarios)

            # Tell the adapter which model fidelty should be used to
            # compute this replication output.
            self.problem_adapter.set_active_fidelity(self.fidelity)

            # Estimate optimality gap with SPAROW!

            solved_saa = self.problem_adapter.solve_extensive_form(
                model_data=model_data_k,
                solver_name=self.solver_name,
                solver_options=self.solver_options,
            )
            saa_optimal_value = self.problem_adapter.get_objective_value(solved_saa)

            xhat_value = self.problem_adapter.evaluate_first_stage_solution(
                xhat=self.xhat,
                model_data=model_data_k,
                solver_name=self.solver_name,
                solver_options=self.solver_options,
            )

            # The replication-level optimality-gap estimate is:
            #   candidate objective value minus SAA optimal value
            outputs[0, col_idx] = xhat_value - saa_optimal_value

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
        N_HF high-fidelity evaluations
        N_LF low-fidelity evaluations

    then set:
        m = N_HF
        M = N_LF - N_HF
    """
    # Convert the allocation output into a flat integer array.
    # For a 2-model ensemble, this should contain:
    #   nsamples[0] = number of HF evaluations (paired)
    #   nsamples[1] = number of LF evaluations (paired + additional)
    nsamples = np.asarray(nsamples_per_model).astype(int).flatten()

    # NOTE: This is currently hardcoded for two models: one HF and one LF
    if len(nsamples) != 2:
        raise ValueError("Expected exactly 2 models for ACV-MRP translation.")

    m = int(nsamples[0])  # num paired evals (for both HF and LF)
    total_lf = int(nsamples[1])  # total number of LF evals (paired and additional)
    M = max(0, total_lf - m)  # num additional LF evals

    return m, M


def build_pyapprox_mf_problem_from_adapter(
    problem_adapter,
    full_scenarios,
    xhat: Dict[str, float],
    batch_size: int,
    solver_name: str,
    solver_options: Optional[dict] = None,
    seed: int = 12345,
    hf_cost_delay_seconds: float = 0.0,
    lf_cost_delay_seconds: float = 0.0,
):
    """
    Build a PyApprox multifidelity problem from a Sparow adapter.

    One PyApprox input sample is one batch of iid sampled scenarios.
    Model 0 is the HF replication-level gap estimator.
    Model 1 is the LF replication-level gap estimator.
    """
    bkd = NumpyBkd()

    prior = ScenarioBatchPrior(
        bkd=bkd,
        problem_adapter=problem_adapter,
        scenarios=full_scenarios,
        batch_size=batch_size,
        seed=seed,
    )

    hf_model = PyApproxModelWrapper(
        problem_adapter=problem_adapter,
        xhat=xhat,
        batch_size=batch_size,
        fidelity="high",
        solver_name=solver_name,
        solver_options=solver_options,
        artificial_delay_seconds=hf_cost_delay_seconds,
    )

    lf_model = PyApproxModelWrapper(
        problem_adapter=problem_adapter,
        xhat=xhat,
        batch_size=batch_size,
        fidelity="low",
        solver_name=solver_name,
        solver_options=solver_options,
        artificial_delay_seconds=lf_cost_delay_seconds,
    )

    # These cost estimates automatically reflect the configured delay
    # stored in each model wrapper.
    hf_cost = estimate_model_cost(hf_model, prior, ntrials=20)
    lf_cost = estimate_model_cost(lf_model, prior, ntrials=20)

    costs = bkd.array([hf_cost, lf_cost])

    problem = MultifidelityForwardUQProblem(
        name="sp_optimality_gap_estimate_problem",
        models=[hf_model, lf_model],
        costs=costs,
        prior=prior,
        description="HF/LF replication-level optimality-gap estimators",
    )

    return problem, bkd
