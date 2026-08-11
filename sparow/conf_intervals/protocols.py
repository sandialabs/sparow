"""
The following protocols define the interfaces for various components of the 
confidence interval estimation code:

- ScenarioPopulationProtocol
- ScenarioEncodingProtocol
- ScenarioSamplerProtocol
- StochasticProgramModelProtocol
- ModelEnsembleProtocol
"""

from __future__ import annotations

from typing import (
    Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable
)

# Return the full finite population of historical scenarios
# Validate the formatting of the scenario dictionaries
@runtime_checkable
class ScenarioPopulationProtocol(Protocol):
    """
    Protocol for storing a finite population of scenarios that we'd like to 
    sample from.

    A scenario population stores the full finite set of scenario dictionaries
    that can be sampled from. It also knows how to validate the scenario format.
    """

    def scenarios(self) -> List[Dict[str, Any]]:
        """
        Return the full finite scenario population.
        """
        ...

    def validate(self, scenarios: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Validate either the provided scenarios or the full population.

        Parameters
        ----------
        scenarios : list of dict, optional
            Scenario dictionaries to validate. If None, validate the full population.
        """
        ...


# Encode a scenario dictionary into a numeric vector
# Decode a numeric vector back into a scenario dictionary
@runtime_checkable
class ScenarioEncodingProtocol(Protocol):
    """
    Protocol for encoding and decoding scenario data as numeric vectors.

    This is useful when interfacing with PyApprox or any other method that
    expects scenario data to be represented as arrays instead of dictionaries. 
    """

    def scenario_vector_keys(self) -> List[str]:
        """
        Return the scenario dictionary keys that define the uncertain data.
        """
        ...

    def encode_scenario_vector(self, scenario: Dict[str, Any]) -> List[float]:
        """
        Convert one scenario dictionary into a flat numeric vector.
        """
        ...

    def decode_scenario_vector(
        self,
        vector: Sequence[float],
        scenario_id: str,
    ) -> Dict[str, Any]:
        """
        Convert a flat numeric vector back into a scenario dictionary.
        """
        ...

    def scenario_vector_dim(self) -> int:
        """
        Return the dimension of one encoded scenario vector.
        This is the length of the flat numeric vector that represents one scenario.
        """
        ...


# Draw i.i.d. scenarios and batches from finite scenario population
@runtime_checkable
class ScenarioSamplerProtocol(Protocol):
    """
    Protocol for sampling batches of scenarios from a finite scenario population.
    """

    def draw_scenarios(self, n: int, replication_id: int,) -> List[Dict[str, Any]]:
        """
        Draw one batch, comprised of n i.i.d.scenario draws.

        Parameters
        ----------
        n : int
            Batch size.
        replication_id : int
            Replication identifier used to make draws reproducible.
        """
        ...


# Build model data from given scenario batch
# Solve Sample Average Approximation (SAA) extensive form model
# Evaluate a fixed first-stage candidate solution in extensive form model
@runtime_checkable
class StochasticProgramModelProtocol(Protocol):
    """
    Protocol for one stochastic program model at one fidelity.

    This protocol represents a concrete stochastic program model instance
    that can:
      - build model data from scenario batches,
      - solve its SAA problem,
      - evaluate a fixed candidate solution,
      - and report replication-level gap estimates.
    """

    def name(self) -> str:
        """
        Return a human-readable model name.
        """
        ...

    def fidelity(self) -> str:
        """
        Return the fidelity label, e.g. 'high' or 'low'.
        """
        ...

    def scenario_population(self) -> ScenarioPopulationProtocol:
        """
        Return the finite scenario population used by this model.
        """
        ...

    def scenario_sampler(self) -> ScenarioSamplerProtocol:
        """
        Return the sampler used by this model.
        """
        ...

    def build_model_data(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build model data from a scenario batch.
        """
        ...

    def solve_extensive_form(
        self,
        model_data: Dict[str, Any],
        solver_name: str,
        solver_options: Optional[Dict[str, Any]] = None,
        loglevel: str = "INFO",
    ) -> Dict[str, Any]:
        """
        Solve the Sample Average Approximation (SAA) extensive form for the supplied model data.
        """
        ...

    def get_objective_value(self, solved_object: Dict[str, Any]) -> float:
        """
        Extract the objective value from solver output.
        """
        ...

    def evaluate_first_stage_solution(
        self,
        xhat: Dict[str, float],
        model_data: Dict[str, Any],
        solver_name: str,
        solver_options: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Evaluate a fixed candidate solution on the supplied model data.
        """
        ...

    def draw_batch_of_scenarios(
        self,
        n: int,
        replication_id: int,
        nested_sampling: bool = False,
        precomputed_supersets: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Draw one scenario batch for a replication.
        """
        ...

    def replication_gap(
        self,
        xhat: Dict[str, float],
        scenarios: List[Dict[str, Any]],
        solver_name: str,
        solver_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Compute one replication-level gap estimate on the supplied batch.

        Returns
        -------
        dict
            Expected keys include:
              - 'gap_estimate'
              - 'xhat_value'
              - 'saa_optimal_value'
        """
        ...


# Container for multiple stochastic program models at different fidelities
@runtime_checkable
class ModelEnsembleProtocol(Protocol):
    """
    Protocol for a collection of stochastic program models.

    This is useful for multifidelity methods such as ACV-MRP and PyApprox.
    Note that the native ACV-MRP in SPAROW expects a pair of models ('high' and 'low'), 
    but this protocol allows for more than two models when using PyApprox.
    """

    def models(self) -> List[StochasticProgramModelProtocol]:
        """
        Return all models in the ensemble.
        """
        ...

    def high_fidelity_model(self) -> StochasticProgramModelProtocol:
        """
        Return the high-fidelity model.
        """
        ...

    def low_fidelity_model(self) -> StochasticProgramModelProtocol:
        """
        Return the low-fidelity model.
        """
        ...




