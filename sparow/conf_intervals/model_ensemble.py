from __future__ import annotations

from typing import List

from sparow.conf_intervals.protocols import (
    StochasticProgramModelProtocol,
)


class ModelEnsemble:
    """
    Concrete container for a collection of stochastic-program models.

    The first model is assumed to be the high-fidelity model.
    The second model is assumed to be the low-fidelity model when a
    two-model ACV-MRP workflow is used.

    For PyApprox integration - the first model is assumed to be the
    high-fidelity model of interest, while the remaining models are of
    varying fidelities.
    """

    def __init__(self, models: List[StochasticProgramModelProtocol]):
        if len(models) == 0:
            raise ValueError("ModelEnsemble requires at least one model.")

        self._models = models

    def models(self) -> List[StochasticProgramModelProtocol]:
        """
        Return all models in the ensemble.
        """
        return self._models

    def high_fidelity_model(self) -> StochasticProgramModelProtocol:
        """
        Return the high-fidelity model.

        By convention, this is the first model in the ensemble.
        """
        return self._models[0]

    def low_fidelity_model(self) -> StochasticProgramModelProtocol:
        """
        Return the low-fidelity model.

        By convention, this is the second model in the ensemble.
        """
        if len(self._models) < 2:
            raise RuntimeError("This ensemble does not contain a low-fidelity model.")
        return self._models[1]
