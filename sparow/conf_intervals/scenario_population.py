"""
This file contains a concrete implementation of a finite scenario population object.
The scenario population object is intended to satisfy:
  - ScenarioPopulationProtocol
  - ScenarioEncodingProtocol

This object stores:
  - the full list of scenario dictionaries,
  - the required scenario keys according to native SPAROW scenario format,
  - and optional encode/decode logic for scenario vectors.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Sequence


class FiniteScenarioPopulation:
    """
    Concrete representation of a finite population of scenarios to sample from.

    This object stores:
      - the full list of scenario dictionaries,
      - the required scenario keys,
      - optional encode/decode logic for scenario vectors,
      - optional fixed metadata that is required for the given problem instance.
        These fields are not part of the uncertain scenario vector, but they
        are required when rebuilding native SPAROW scenario dictionaries.

    It is designed to be reusable across different stochastic programs.
    """

    def __init__(
        self,
        scenarios: List[Dict[str, Any]],
        required_scenario_keys: Optional[List[str]] = None,
        scenario_vector_keys: Optional[List[str]] = None,
        fixed_metadata: Optional[Dict[str, Any]] = None,
    ):
        self._scenarios = scenarios
        self._required_scenario_keys = (
            [] if required_scenario_keys is None else list(required_scenario_keys)
        )
        self._scenario_vector_keys = (
            [] if scenario_vector_keys is None else list(scenario_vector_keys)
        )

        # These are non-random fields that are not encoded into the scenario
        # vector, but must be reattached when decoding vectors back into native
        # SPAROW scenario dictionaries.
        self._fixed_metadata = {} if fixed_metadata is None else dict(fixed_metadata)

        self.validate(self._scenarios)

    def scenarios(self) -> List[Dict[str, Any]]:
        """
        Return the full finite scenario population.
        """
        return self._scenarios

    def required_scenario_keys(self) -> List[str]:
        """
        Return required nonstandard scenario keys.
        This is usually the string name identifier foruncertain problem data.
        """
        return self._required_scenario_keys

    def fixed_metadata(self) -> Dict[str, Any]:
        """
        Return the fixed metadata to be restored on decode.

        These fields are not part of the uncertain scenario vector, but they
        are required when rebuilding native SPAROW scenario dictionaries.
        """
        return dict(self._fixed_metadata)

    def validate(self, scenarios: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Validate either the provided scenarios or the stored full population.

        Every scenario must be a dictionary containing:
          - 'ID'
          - 'Probability'
          - any user-specified required keys
        """
        if scenarios is None:
            scenarios = self._scenarios

        if not isinstance(scenarios, list):
            raise RuntimeError(
                "Scenario population must be a list of scenario dictionaries."
            )

        required = ["ID", "Probability"] + self._required_scenario_keys

        for i, scen in enumerate(scenarios):
            if not isinstance(scen, dict):
                raise RuntimeError(f"Scenario at index {i} is not a dictionary.")

            missing = [key for key in required if key not in scen]
            if missing:
                raise RuntimeError(
                    f"Scenario at index {i} is missing required key(s): {missing}"
                )

    def scenario_vector_keys(self) -> List[str]:
        """
        Return the scenario fields used for vector encoding.

        These keys identify the uncertain scenario data fields.
        """
        return self._scenario_vector_keys

    def encode_scenario_vector(self, scenario: Dict[str, Any]) -> List[float]:
        """
        Convert one scenario dictionary into a flat numeric vector.

        This implementation works when the uncertain fields listed in
        scenario_vector_keys() are numeric scalars or flat lists.
        """
        vec: List[float] = []

        for key in self.scenario_vector_keys():
            value = scenario[key]

            if np.isscalar(value):
                vec.append(float(value))
            else:
                vec.extend([float(v) for v in value])

        return vec

    def decode_scenario_vector(
        self,
        vector: Sequence[float],
        scenario_id: str,
    ) -> Dict[str, Any]:
        """
        Convert a flat vector back into a scenario dictionary.

        Note that this will contain the ID field, but not the Probability field.
        Batch-construction logic assigns the probability weight later.

        So the decoded scenario contains:
          - the supplied scenario ID,
          - the uncertain fields reconstructed from the vector,
          - any fixed metadata needed by the specific problem instance/ application.
        """
        ref = self._scenarios[0]

        out = dict(self._fixed_metadata)
        out["ID"] = scenario_id

        cursor = 0
        for key in self.scenario_vector_keys():
            ref_value = ref[key]

            if np.isscalar(ref_value):
                out[key] = float(vector[cursor])
                cursor += 1
            else:
                length = len(ref_value)
                out[key] = [float(v) for v in vector[cursor : cursor + length]]
                cursor += length

        return out

    def scenario_vector_dim(self) -> int:
        """
        Return the dimension of one encoded scenario vector.
        """
        return len(self.encode_scenario_vector(self._scenarios[0]))
