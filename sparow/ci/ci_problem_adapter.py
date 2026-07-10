from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from sparow.ef import ExtensiveFormSolver

class CIProblemAdapter(ABC):
    """
    Problem-specific interface required by the generic MRP code.

    Each stochastic program, e.g. farmer, newsvendor, OPF, should provide
    a subclass implementing these methods.

    The user is only required to implement:
        1. get_scenario_population()
        2. build_model_data(scenarios)
        3. build_stochastic_program(model_data)
        4. first_stage_variable_order()
    """

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
        NOTE: This is the standard (high-fidelity) model. 
        If your problem supports low-fidelity models, also implement 
        build_low_fidelity_stochastic_program(...) seperately from this method.
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

    def validate_scenario_population(self, scenarios: Optional[List[Dict[str, Any]]] = None) -> None:
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
            raise RuntimeError("Scenario population must be a list of scenario dictionaries.")

        required = ["ID", "Probability"] + self.required_scenario_keys()

        for i, scen in enumerate(scenarios):
            if not isinstance(scen, dict):
                raise RuntimeError(f"Scenario at index {i} is not a dictionary.")

            missing = [key for key in required if key not in scen]
            if missing:
                raise RuntimeError(f"Scenario at index {i} is missing required key(s): {missing}")

    # ======================================================================
    # Generic solver logic
    # ======================================================================

    def solve_extensive_form(
        self,
        model_data: Dict[str, Any],
        solver_name: str,
        solver_options: Optional[Dict[str, Any]] = None,
        loglevel: str = "INFO", # can replace with DEBUG, VERBOSE, etc.
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
            raise RuntimeError("Could not extract objective value from solved_object['pool_config']['best_value'].") from e

    def get_first_stage_solution(self, solved_object: Dict[str, Any]) -> Dict[str, float]:
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
            raise RuntimeError("Could not find solved_object['solutions'][0]['variables'].") from e

        order = self.first_stage_variable_order()
        xhat = {}

        for var in variables:
            name = var["name"]
            if name in order:
                xhat[name] = var["value"]

        missing = [name for name in order if name not in xhat]
        if missing:
            raise RuntimeError(f"Could not extract all first-stage variables from solved_object. Missing: {missing}")

        return xhat

    def first_stage_solution_dict_to_vector(self, xhat: Dict[str, float]) -> List[float]:
        """
        Convert an xhat dictionary into the ordered list expected by
        sp.evaluate(x, ...).
        """
        order = self.first_stage_variable_order()
        missing = [name for name in order if name not in xhat]

        if missing:
            raise RuntimeError(f"xhat is missing required first-stage variable(s): {missing}")

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
            raise RuntimeError(f"Candidate first-stage solution evaluated infeasible on bundle/scenario {eval_result.bundle}.")

        return eval_result.objective
    
    # ======================================================================
    # Optional Multifidelity ACV MRP-specific methods (default to None)
    # ======================================================================

    def build_low_fidelity_stochastic_program(self, model_data):
        """Build low-fidelity stochastic program for ACV-MRP. Returns None if not supported."""
        return None

    def get_fidelity_levels(self):
        """
        Return list of supported fidelity levels. Default is ['standard'] 
        for standard MRP. Override in subclass if low-fidelity models are supported.

        Returns
        -------
        list of str
            Fidelity level names (e.g., ['standard'], ['LF', 'HF'])
        """
        return ['standard']

    def supports_acv(self):
        """Whether this adapter supports ACV-MRP."""
        return False

    