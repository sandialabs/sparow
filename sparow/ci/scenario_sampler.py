import copy
import numpy as np
from typing import List, Dict, Any


class ScenarioSampler:
    """
    Draw iid Monte Carlo samples from a finite scenario population.

    The input scenario population should be a list of dictionaries.
    For example, in the farmer problem, it would look like this:
        [
            {"ID": "0", "Yield": {...}, "Probability": ...},
            {"ID": "1", "Yield": {...}, "Probability": ...},
            ...
        ]
    """

    def __init__(self, scenarios, seed=12345, with_replacement=True):
        if len(scenarios) == 0:
            raise ValueError("Scenario population must be nonempty.")

        self.scenarios = scenarios
        self.seed = seed
        self.with_replacement = with_replacement

        self.num_population_scenarios = len(scenarios)

    def draw_scenarios(self, n, replication_id):
        """
        For the given replication_id, draw a batch of n iid scenarios from
        the inputted set of population scenarios.

        We keep track of the scenarios we've drawn by their population ID/ index.

        For reproducibility, since each replication needs its own independent Monte
        Carlo sample of n scenarios:

            - self.seed is the base seed for the entire MRP algorithm run.
            - replication_id is used to distinguish the random streams between
              different replications
        """

        if (not self.with_replacement) and n > self.num_population_scenarios:
            raise ValueError(
                "Cannot sample without replacement when n exceeds the number "
                "of available population scenarios."
            )

        seed_sequence = np.random.SeedSequence([self.seed, replication_id])
        rng = np.random.default_rng(seed_sequence)  # this gives one rng per replication

        indices = rng.choice(
            self.num_population_scenarios,
            size=n,
            replace=self.with_replacement,
        )

        sampled = []
        for local_id, population_idx in enumerate(indices):
            scen = copy.deepcopy(self.scenarios[int(population_idx)])
            scen["Original_ID"] = scen.get("ID", str(population_idx))
            scen["Population_Index"] = int(population_idx)
            scen["ID"] = f"rep{replication_id}_scen{local_id}"
            scen["Probability"] = 1.0 / n
            sampled.append(scen)

        return sampled


class ScenarioBatchPrior:
    """
    For PyApprox integration: Prior over full scenario batches.

    One PyApprox sample corresponds to one batch of scenarios.
    This is one replication in the MRP or ACV-MRP sense.

    If the batch size is n and each scenario vector has 
    dimension scenario_dim, then one replication is represented by
    a flattened vector of length n * scenario_dim.
    """

    def __init__(
        self,
        bkd,
        problem_adapter,
        scenarios: List[Dict[str, Any]],
        batch_size: int,
        seed: int = 12345,
    ):
        self._bkd = bkd
        self._problem_adapter = problem_adapter
        self._scenarios = scenarios
        self._batch_size = batch_size
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Dimension of one single scenario vector after converting uncertain
        # data fields into a flat numeric vector.
        self._scenario_dim = self._problem_adapter.scenario_vector_dim()

    def bkd(self):
        return self._bkd

    def nvars(self) -> int:
        """
        Return the dimension of one PyApprox input sample.

        One PyApprox input sample is one full batch of scenarios.

        Each individual scenario draw is a vector of length scenario_dim.

        So one full batch of scenarios is encoded as a flat vector of length
        nvars = batch_size * scenario_dim.
        """
        return self._batch_size * self._scenario_dim

    def rvs(self, nsamples: int):
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

        # Initialize a return structure to hold the scenario data
        out = np.zeros((self.nvars(), nsamples), dtype=float)

        # For each replication 0, 1, ...., nsamples
        for replication_idx in range(nsamples):

            # Choose n scenarios from the historical set of scenarios to form a batch
            # NOTE: this is currently hardcoded to be done by sampling with replacement
            indices_of_selected_scens = self._rng.choice(len(self._scenarios), size=self._batch_size, replace=True)

            # Encode each sampled scenario dictionary into a flat numeric vector.
            batch_vectors = [
                self._problem_adapter.encode_scenario_vector(self._scenarios[ii])
                for ii in indices_of_selected_scens
            ]

            # Stack the flattened scenario vectors into a batch matrix of shape
            # (batch_size, scenario_dim), then flatten that matrix into one
            # long vector so it can serve as one PyApprox sample column.
            batch_matrix = np.asarray(batch_vectors, dtype=float)
            out[:, replication_idx] = batch_matrix.reshape(-1)

        return self._bkd.array(out)

    
