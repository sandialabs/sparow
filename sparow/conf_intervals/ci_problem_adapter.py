import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from sparow.ef import ExtensiveFormSolver


class CIProblemAdapter(ABC):
    """
    Problem-specific interface required by the confidence interval code.

    Each stochastic program, e.g. farmer, newsvendor, OPF, should provide
    a subclass implementing these methods (required):
        1. get_scenario_population()
        2. build_model_data(scenarios)
        3. build_stochastic_program(model_data)
        4. first_stage_variable_order()

    The remaining methods are optional
    """

    def __init__(
        self,
        model_name,
        scenario_data,
        model_builder,
        app_data=None,
        first_stage_variables=None,
    ):
        self.model_name = model_name
        self.scenario_data = scenario_data
        self.model_builder = model_builder
        self.app_data = {} if app_data is None else dict(app_data)
        self.first_stage_variables = (
            [] if first_stage_variables is None else first_stage_variables
        )

    # ======================================================================
    # User-implemented abstract methods
    # ======================================================================

    @abstractmethod
    def get_scenario_population(self) -> List[Dict[str, Any]]:
        """
        Return the full finite / historical scenario population as a
        list of scenario dictionaries.
        """
        raise NotImplementedError

    @abstractmethod
    def build_model_data(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build the model_data dictionary expected by the downstream Sparow model.

        Parameters
        ----------
        scenarios : list of dict
            A list of scenario dictionaries, typically either:
            - the full scenario population, or
            - one sampled MRP replication batch.

        Returns
        -------
        dict
            Model data dictionary compatible with sp.initialize_model(...).
        """
        raise NotImplementedError

    @abstractmethod
    def build_stochastic_program(self, model_data: Dict[str, Any]):
        """
        Build and return a Sparow stochastic_program object from model_data.

        For single-fidelity problems, this is the only model required.
        For multifidelity problems, subclasses may additionally support a low-fidelity
        variant, e.g. by overriding build_stochastic_program(...) to depend on an
        active fidelity state.
        """
        raise NotImplementedError

    @abstractmethod
    def first_stage_variable_order(self) -> List[str]:
        """
        Return the ordered list of first-stage variable names in xhat.

        This order is used to extract xhat from solved EF results
        and convert xhat dicts to vectors for sp.evaluate(...).
        """
        raise NotImplementedError

    # ======================================================================
    # Optional user override
    # ======================================================================

    def required_scenario_keys(self) -> List[str]:
        """
        Return any additional scenario keys required by the problem.

        The base validator always requires:
            - "ID"
            - "Probability"

        Override this method in a model-specific subclass if you want to also
        require keys such as "Yield", "Demand", etc.

        Returns
        -------
        list of str
        """
        return []

    # ======================================================================
    # Core validation logic
    # ======================================================================

    def validate_scenario_population(
        self, scenarios: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Validate the format of a scenario population.

        Parameters
        ----------
        scenarios : list of dict, optional
            If None, validates self.get_scenario_population().

        Raises
        ------
        RuntimeError
            If the scenario population format is invalid.
        """
        if scenarios is None:
            scenarios = self.get_scenario_population()

        if not isinstance(scenarios, list):
            raise RuntimeError(
                "Scenario population must be a list of scenario dictionaries."
            )

        required = ["ID", "Probability"] + self.required_scenario_keys()

        for i, scen in enumerate(scenarios):
            if not isinstance(scen, dict):
                raise RuntimeError(f"Scenario at index {i} is not a dictionary.")

            missing = [key for key in required if key not in scen]
            if missing:
                raise RuntimeError(
                    f"Scenario at index {i} is missing required key(s): {missing}"
                )

    # ======================================================================
    # Generic solver logic
    # ======================================================================

    def solve_extensive_form(
        self,
        model_data: Dict[str, Any],
        solver_name: str,
        solver_options: Optional[Dict[str, Any]] = None,
        loglevel: str = "INFO",  # can replace with DEBUG, VERBOSE, etc.
    ) -> Dict[str, Any]:
        """
        Solve the extensive form for the supplied model_data using Sparow's
        ExtensiveFormSolver API.

        Returns
        -------
        dict
            Solver results converted to dictionary form.
        """
        solver = ExtensiveFormSolver()
        solver_kwargs = {"solver": solver_name, "loglevel": loglevel}

        if solver_options is not None:
            solver_kwargs.update(solver_options)

        solver.set_options(**solver_kwargs)

        sp = self.build_stochastic_program(model_data)
        results = solver.solve(sp)

        return results.to_dict()

    def get_objective_value(self, solved_object: Dict[str, Any]) -> float:
        """
        Extract the EF objective value from a Sparow results dictionary.

        Assumes the standard Sparow result format:
            results_dict["pool_config"]["best_value"]
        """
        if not isinstance(solved_object, dict):
            raise RuntimeError("Expected solved_object to be a dictionary.")

        try:
            return solved_object["pool_config"]["best_value"]
        except Exception as e:
            raise RuntimeError(
                "Could not extract objective value from solved_object['pool_config']['best_value']."
            ) from e

    def get_first_stage_solution(
        self, solved_object: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Extract the first-stage candidate solution xhat from a Sparow results dictionary.

        Returns
        -------
        dict
            Mapping from first-stage variable name to value.
        """
        if not isinstance(solved_object, dict):
            raise RuntimeError("Expected solved_object to be a dictionary.")

        try:
            variables = solved_object["solutions"][0]["variables"]
        except Exception as e:
            raise RuntimeError(
                "Could not find solved_object['solutions'][0]['variables']."
            ) from e

        order = self.first_stage_variable_order()
        xhat = {}

        for var in variables:
            name = var["name"]
            if name in order:
                xhat[name] = var["value"]

        missing = [name for name in order if name not in xhat]
        if missing:
            raise RuntimeError(
                f"Could not extract all first-stage variables from solved_object. Missing: {missing}"
            )

        return xhat

    def first_stage_solution_dict_to_vector(
        self, xhat: Dict[str, float]
    ) -> List[float]:
        """
        Convert an xhat dictionary into the ordered list expected by
        sp.evaluate(x, ...).
        """
        order = self.first_stage_variable_order()
        missing = [name for name in order if name not in xhat]

        if missing:
            raise RuntimeError(
                f"xhat is missing required first-stage variable(s): {missing}"
            )

        return [xhat[name] for name in order]

    def evaluate_first_stage_solution(
        self,
        xhat: Dict[str, float],
        model_data: Dict[str, Any],
        solver_name: str,
        solver_options: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Evaluate a fixed first-stage solution xhat on the supplied scenarios.

        Uses Sparow's public stochastic_program.evaluate(x, ...) API.

        Returns
        -------
        float
            Objective value of the fixed first-stage solution on the supplied model_data.
        """
        sp = self.build_stochastic_program(model_data)
        sp.set_solver(solver_name)

        x_vector = self.first_stage_solution_dict_to_vector(xhat)

        eval_result = sp.evaluate(
            x=x_vector,
            solver_options={} if solver_options is None else solver_options,
            cached=False,
        )

        if not eval_result.feasible:
            raise RuntimeError(
                f"Candidate first-stage solution evaluated infeasible on bundle/scenario {eval_result.bundle}."
            )

        return eval_result.objective

    # ======================================================================
    # Optional Multifidelity ACV MRP-specific methods (default to None)
    # ======================================================================

    def get_fidelity_levels(self):
        """
        Return list of supported fidelity levels. Default is ['standard']
        for standard MRP. Override in subclass if low-fidelity models are supported.

        Returns
        -------
        list of str
            Fidelity level names (e.g., ['standard'], ['LF', 'HF'])
        """
        return ["standard"]

    def supports_acv(self):
        """Whether this adapter supports ACV-MRP."""
        return False

    # Methods below are for maintaining "active fidelity state" when running
    # the ACV algorithm code.
    # Ensures functions solve_extensive_form and evaluate_first_stage_solution are
    # operating on correct model

    def set_active_fidelity(self, fidelity):
        if fidelity not in ("high", "low"):
            raise ValueError(f"Unknown fidelity level: {fidelity}")
        self._active_fidelity = fidelity

    def get_active_fidelity(self):
        return self._active_fidelity

    # ======================================================================
    # Optional for representing scenario vector data in different ways
    # This is for PyApprox integration/ compatibility
    # ======================================================================

    def scenario_vector_keys(self) -> List[str]:
        """
        Return the dictionary key names that define the uncertain problem data
        contained in a given scenario.

        Subclasses should override this if they want to support PyApprox.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must override scenario_vector_keys()"
            "to support scenario-vector encoding."
        )

    def encode_scenario_vector(self, scenario: Dict[str, Any]) -> List[float]:
        """
        Convert a scenario dictionary into a flat numeric vector (i.e. - list of floats).

        This method works when the uncertain fields listed in
        scenario_vector_keys() are either numeric scalars or flat lists.
        """
        vec = []
        for key in self.scenario_vector_keys():
            value = scenario[key]
            if np.isscalar(value):
                vec.append(float(value))
            else:
                vec.extend([float(v) for v in value])
        return vec

    def decode_scenario_vector(
        self, vector: List[float], scenario_id: str
    ) -> Dict[str, Any]:
        """
        Convert a flat numeric vector into a scenario dictionary.

        Subclasses should override this whenever the uncertain fields are not
        simple flat lists whose lengths are known from context.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must override decode_scenario_vector()"
            "to support generic scenario-vector decoding."
        )

    def scenario_vector_dim(self) -> int:
        """
        Return the dimension of a singel, encoded scenario vector.

        The default implementation computes this from the first scenario.
        """
        scenarios = self.get_scenario_population()
        return len(self.encode_scenario_vector(scenarios[0]))
