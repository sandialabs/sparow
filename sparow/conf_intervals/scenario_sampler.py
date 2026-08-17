"""
This file contains class definitions for drawing Monte Carlo samples (batches) of scenarios
from a finite scenario population.
These batches can be used for Monte Carlo replications in the MRP or ACV-MRP algorithms,
as well as in the format required by PyApprox.
"""

import copy
import numpy as np
from typing import List, Dict, Any, Optional, Union

from sparow.conf_intervals.protocols import (
    ScenarioPopulationProtocol,
    ScenarioEncodingProtocol,
)


class ScenarioSampler:
    """
    Draw iid Monte Carlo samples (i.e. - scenario batches) from a finite scenario population.

    This class requires an object satisfying ScenarioPopulationProtocol.
    The protocol is the single accepted interface for accessing and validating
    the underlying finite scenario set.

    The actual draws are made from the list of scenario dictionaries returned by
    scenario_population.scenarios().

    The internal representation of the scenario population as a list of dictionaries
    is done for compatibility with existing SPAROW code.

    Example usage for drawing a batch of scenarios for a single replication:
    -------------

    scenarios = [
        {"ID": "s0", "Probability": 0.5, "Demand": [1.0, 2.0]},
        {"ID": "s1", "Probability": 0.5, "Demand": [3.0, 4.0]},
    ]
    pop = FiniteScenarioPopulation(
        scenarios=scenarios,
        required_scenario_keys=["Demand"],
        scenario_vector_keys=["Demand"],
    )
    sampler = ScenarioSampler(scenario_population=pop, seed=123, with_replacement=True)
    batch = sampler.draw_scenarios(n=3, replication_id=0)
    print(batch)

    Output:
    [
        {'ID': 'rep0_scen0', 'Probability': 0.333, 'Demand': [1.0, 2.0], 'Original_ID': 's0', 'Population_Index': 0},
        {'ID': 'rep0_scen1', 'Probability': 0.333, 'Demand': [3.0, 4.0], 'Original_ID': 's1', 'Population_Index': 1},
        {'ID': 'rep0_scen2', 'Probability': 0.333, 'Demand': [3.0, 4.0], 'Original_ID': 's1', 'Population_Index': 1}
    ]
    """

    def __init__(
        self,
        scenario_population: ScenarioPopulationProtocol,
        seed: int = 12345,
        with_replacement: bool = True,
    ):
        if not isinstance(scenario_population, ScenarioPopulationProtocol):
            raise TypeError(
                "scenario_population must satisfy ScenarioPopulationProtocol."
            )

        self._scenario_population = scenario_population
        self.seed = seed
        self.with_replacement = with_replacement

        # The protocol object is responsible for validation of the stored scenarios.
        self._scenario_population.validate()
        self._scenarios = self._scenario_population.scenarios()

        if len(self._scenarios) == 0:
            raise ValueError("Scenario population must be nonempty.")

        self.num_population_scenarios = len(self._scenarios)

    def scenario_population(self) -> ScenarioPopulationProtocol:
        """
        Return the scenario population object used by this sampler.
        """
        return self._scenario_population

    def scenarios(self) -> List[Dict[str, Any]]:
        """
        Return the underlying list of scenario dictionaries used for sampling.
        """
        return self._scenarios

    def _rng_for_replication(self, replication_id: int):
        """
        Create a deterministic RNG stream for one replication.

        For reproducibility of the entire algorithm run, since each
        replication needs its own independent Monte Carlo sample of n scenarios:

            - self.seed is the base seed for the entire MRP algorithm run.
            - replication_id is used to distinguish the random streams between
              different replications
        """
        seed_sequence = np.random.SeedSequence([self.seed, replication_id])
        return np.random.default_rng(seed_sequence)

    def draw_scenarios(self, n: int, replication_id: int) -> List[Dict[str, Any]]:
        """
        Draw one batch of scenarios for a given replication.

        Parameters
        ----------
        n : int
            Number of scenarios in the batch.
        replication_id : int
            Replication index used to define the RNG stream.

        Returns
        -------
        list of dict
            Sampled scenario dictionaries in native SPAROW format

        Notes
        -----
        The returned scenarios are deep copies of the original scenarios stored
        in the population-level object.
        This avoids mutating the stored scenario population when replication-
        specific metadata is added.
        """

        if (not self.with_replacement) and n > self.num_population_scenarios:
            raise ValueError(
                "Cannot sample without replacement when n exceeds the number "
                "of available population scenarios."
            )

        # This is the random number generator for this specific replication's draws.
        rng = self._rng_for_replication(replication_id)

        # Draw population indices for scenarios included in this batch.
        indices = rng.choice(
            self.num_population_scenarios,
            size=n,
            replace=self.with_replacement,
        )

        sampled = []
        for local_id, population_idx in enumerate(indices):

            # Copy the sampled scenario so replication-specific fields do not
            # overwrite the stored population scenario.
            scen = copy.deepcopy(self._scenarios[int(population_idx)])

            # Keep track of where this scenario came from in the population.
            scen["Original_ID"] = scen.get("ID", str(population_idx))
            scen["Population_Index"] = int(population_idx)

            # Required fields for the sampled scenario dictionary in native SPAROW format.
            scen["ID"] = f"rep{replication_id}_scen{local_id}"
            scen["Probability"] = 1.0 / n

            sampled.append(scen)

        return sampled


class ScenarioBatchPrior:
    """
    For PyApprox integration: Prior over full scenario batches.

    One PyApprox sample corresponds to one batch of scenarios.
    This is one replication in the MRP or ACV-MRP sense.

    Internally, the prior draws from the list of scenario dictionaries returned
    by scenario_population.scenarios().

    If the batch size is n and each scenario vector has
    dimension scenario_dim, then one replication is represented by
    a flattened vector of length n * scenario_dim.
    """

    def __init__(
        self,
        bkd,
        scenario_population: ScenarioPopulationProtocol,
        batch_size: int,
        seed: int = 12345,
    ):
        if not isinstance(scenario_population, ScenarioPopulationProtocol):
            raise TypeError(
                "scenario_population must satisfy ScenarioPopulationProtocol."
            )
        if not isinstance(scenario_population, ScenarioEncodingProtocol):
            raise TypeError(
                "scenario_population must satisfy ScenarioEncodingProtocol "
                "for PyApprox integration."
            )
        self._bkd = bkd
        self._scenario_population = scenario_population
        self._batch_size = batch_size
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Validate the stored finite scenario set once up front.
        self._scenario_population.validate()
        self._scenarios = self._scenario_population.scenarios()

        # Dimension of one encoded scenario vector.
        # this is the length of the flat numeric vector that represents one scenario.
        self._scenario_dim = self._scenario_population.scenario_vector_dim()

    def bkd(self):
        return self._bkd

    def scenario_population(self):
        """
        Return the finite scenario population object.
        """
        return self._scenario_population

    def nvars(self) -> int:
        """
        Return the dimension of one PyApprox input sample.

        One PyApprox input sample is one full batch of scenarios.

        Each individual scenario draw is a vector of length scenario_dim.

        So one full batch of scenarios is encoded as a flat vector of length
        nvars = batch_size * scenario_dim.
        """
        return self._batch_size * self._scenario_dim

    def rvs(self, nsamples: int, with_replacement_flag: bool = True) -> np.ndarray:
        """
        Draw independent scenario batches from the empirical distribution.

        The returned array has shape (nvars, nsamples):
          - each column corresponds to one independent replication's batch of scenarios,
          - each replication's batch of scenarios contains batch_size sampled scenarios,
          - the full batch data is flattened into one long column.

        So nvars = batch_size * scenario_dim is the number of scalars needed to represent
        one full replication batch,

        And nsamples is the number of independent replication batches.
        """

        # Each column of `out` stores one flattened replication batch.
        out = np.zeros((self.nvars(), nsamples), dtype=float)

        # For each replication 0, 1, ...., nsamples
        for replication_idx in range(nsamples):

            # Choose n scenarios from the population set of scenarios to form a batch
            indices_of_selected_scens = self._rng.choice(
                len(self._scenarios),
                size=self._batch_size,
                replace=with_replacement_flag,
            )

            # Encode each sampled scenario dictionary into a flat numeric vector.
            batch_vectors = [
                self._scenario_population.encode_scenario_vector(self._scenarios[ii])
                for ii in indices_of_selected_scens
            ]

            # Stack the flattened scenario vectors into a batch matrix of shape
            # (batch_size, scenario_dim), then flatten that matrix into one
            # long vector so it can serve as one PyApprox sample column.
            batch_matrix = np.asarray(batch_vectors, dtype=float)
            out[:, replication_idx] = batch_matrix.reshape(-1)

        return self._bkd.array(out)
