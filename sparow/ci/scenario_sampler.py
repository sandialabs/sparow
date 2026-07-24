import copy
import numpy as np


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
