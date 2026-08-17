"""
This class is intended to satisfy StochasticProgramModelProtocol.
Conceptually: the protocol defines the interface, while the SPModelWrapperforUQ class
is a concrete, reusable implementation of that interface.

SPModelWrapperforUQ does not need to import or subclass StochasticProgramModelProtocol in
order to satisfy it. Rather, it needs to implement the required methods with compatible
signatures. The protocol is mainly there so that type checkers and
runtime isinstance(..., ProtocolClass) checks can verify compatibility,
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sparow.ef import ExtensiveFormSolver
from sparow.sp import stochastic_program

from sparow.conf_intervals.protocols import (
    ScenarioPopulationProtocol,
    ScenarioSamplerProtocol,
)


class SPModelWrapperforUQ:
    """
    Wrapper for a single SPAROW stochastic-program model to support
    confidence interval estimation and uncertainty quantification.

    This object is not the underlying SPAROW StochasticProgram itself.
    It is a higher-level wrapper that combines:
      - one stochastic-program model builder,
      - one finite scenario population,
      - one scenario sampler,
      - one fidelity label,
      - and the replication-level solve/evaluate logic needed by
        MRP, ACV-MRP, and PyApprox.
    """

    def __init__(
        self,
        name: str,
        fidelity: str,
        scenario_population: ScenarioPopulationProtocol,
        scenario_sampler: ScenarioSamplerProtocol,
        model_builder,
        app_data: Optional[Dict[str, Any]] = None,
        first_stage_variables: Optional[List[str]] = None,
        first_stage_variable_order: Optional[List[str]] = None,
    ):

        if not isinstance(scenario_population, ScenarioPopulationProtocol):
            raise TypeError(
                "scenario_population must satisfy ScenarioPopulationProtocol."
            )
        if not isinstance(scenario_sampler, ScenarioSamplerProtocol):
            raise TypeError("scenario_sampler must satisfy ScenarioSamplerProtocol.")

        self._name = name
        self._fidelity = fidelity
        self._scenario_population = scenario_population
        self._scenario_sampler = scenario_sampler
        self._model_builder = model_builder
        self._app_data = {} if app_data is None else dict(app_data)
        self._first_stage_variables = (
            [] if first_stage_variables is None else list(first_stage_variables)
        )
        self._first_stage_variable_order = (
            []
            if first_stage_variable_order is None
            else list(first_stage_variable_order)
        )

        # Validate the stored scenario population once at construction time.
        self._scenario_population.validate()

    # ------------------------------------------------------------------
    # Basic metadata
    # ------------------------------------------------------------------

    def name(self) -> str:
        """
        Return the model name.
        """
        return self._name

    def fidelity(self) -> str:
        """
        Return the fidelity label, e.g. 'high', 'low', etc..
        """
        return self._fidelity

    def scenario_population(self) -> ScenarioPopulationProtocol:
        """
        Return the finite scenario population associated with this model.
        """
        return self._scenario_population

    def scenario_sampler(self) -> ScenarioSamplerProtocol:
        """
        Return the sampler used to draw scenario batches for this model.
        """
        return self._scenario_sampler

    def model_builder(self):
        """
        Return the underlying SPAROW model-builder function.
        """
        return self._model_builder

    def app_data(self) -> Dict[str, Any]:
        """
        Return the application data dictionary that is used to
        initialize the stochastic program.
        """
        return self._app_data

    def first_stage_variables(self) -> List[str]:
        """
        Return the high-level first-stage variables passed to SPAROW.
        """
        return self._first_stage_variables

    def first_stage_variable_order(self) -> List[str]:
        """
        Return the ordered first-stage variable names used to encode
        the first-stage candidate solution, xhat.
        """
        return self._first_stage_variable_order

    # ------------------------------------------------------------------
    # Functions for drawing scenario batches and building model data
    # ------------------------------------------------------------------

    def draw_batch_of_scenarios(
        self,
        n: int,
        replication_id: int,
        nested_sampling: bool = False,
        precomputed_supersets: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Draw one scenario batch of scenarios for a replication. If not using a
        precomputed superset, internally calls to self._scenario_sampler.draw_scenarios
        to actually draw the scenarios.

        Parameters
        ----------
        n : int
            Batch size.
        replication_id : int
            Replication index used for reproducible sampling.
        nested_sampling : bool, optional
            Whether to use a precomputed superset and truncate it to size n.
            This is useful for numerical experiments where we vary/ increment
            batch size n and want to keep the same underlying superset of scenarios for comparability.
        precomputed_supersets : dict, optional
            Mapping from replication id to a precomputed scenario list.

        Returns
        -------
        list of dict
            Native SPAROW representation for a scenario batch.
        """
        if nested_sampling:

            if precomputed_supersets is None:
                raise RuntimeError(
                    "nested_sampling=True requires precomputed_supersets."
                )
            if replication_id not in precomputed_supersets:
                raise RuntimeError(
                    f"Missing precomputed superset for replication {replication_id}."
                )

            # Nested sampling keeps the same underlying superset and takes
            # its first n scenarios.... so results are comparable across
            # different values.
            sampled_scenarios = precomputed_supersets[replication_id][:n]

        else:
            sampled_scenarios = self._scenario_sampler.draw_scenarios(
                n=n,
                replication_id=replication_id,
            )

        self._scenario_population.validate(sampled_scenarios)

        return sampled_scenarios

    def build_model_data(
        self, sampled_scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build the model_data dictionary expected by SPAROW.
        """
        return {"data": {}, "scenarios": sampled_scenarios}

    # ------------------------------------------------------------------
    # Solver / evaluation helpers
    # ------------------------------------------------------------------

    def build_stochastic_program(self, model_data: Dict[str, Any]):
        """
        Build and return a SPAROW stochastic_program object for this model.
        """
        sp = stochastic_program(first_stage_variables=self._first_stage_variables)
        sp.initialize_application(app_data=self._app_data)
        sp.initialize_model(
            name=self._name,
            model_data=model_data,
            model_builder=self._model_builder,
        )
        return sp

    def solve_extensive_form(
        self,
        model_data: Dict[str, Any],
        solver_name: str,
        solver_options: Optional[Dict[str, Any]] = None,
        loglevel: str = "INFO",
    ) -> Dict[str, Any]:
        """
        Solve the Sample Average Approximationextensive form for
        the supplied model data.

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
        Extract the objective value from a SPAROW results dictionary.
        """
        try:
            return solved_object["pool_config"]["best_value"]

        except Exception as exc:
            raise RuntimeError(
                "Could not get objective value from solved_object['pool_config']['best_value']."
            ) from exc

    def get_first_stage_solution(
        self, solved_object: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Extract the first-stage candidate solution xhat returned from solver output
        when running candidate generation code.
        """
        try:
            variables = solved_object["solutions"][0]["variables"]
        except Exception as exc:
            raise RuntimeError(
                "Could not find candidate first-stage variables in solved_object['solutions'][0]['variables']."
            ) from exc

        order = self._first_stage_variable_order
        xhat = {}

        for var in variables:
            name = var["name"]
            if name in order:
                xhat[name] = var["value"]

        missing = [name for name in order if name not in xhat]
        if missing:
            raise RuntimeError(
                f"Could not extract all first-stage vars from solved_object. Missing: {missing}"
            )

        return xhat

    def first_stage_solution_dict_to_vector(
        self, xhat: Dict[str, float]
    ) -> List[float]:
        """
        Convert an xhat dictionary into the ordered vector expected by sp.evaluate(...).
        """
        missing = [
            name for name in self._first_stage_variable_order if name not in xhat
        ]
        if missing:
            raise RuntimeError(
                f"xhat is missing required first-stage variable(s): {missing}"
            )

        return [xhat[name] for name in self._first_stage_variable_order]

    def evaluate_first_stage_solution(
        self,
        xhat: Dict[str, float],
        model_data: Dict[str, Any],
        solver_name: str,
        solver_options: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Evaluate a fixed first-stage candidate solution on the supplied model data.

        Returns
        -------
        float
            Objective value of the fixed first-stage candidate solution.
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
                f"Fixed first-stage candidate solution evaluated infeasible on bundle/scenario {eval_result.bundle}."
            )

        return eval_result.objective

    # ------------------------------------------------------------------
    # Replication-level logic
    # ------------------------------------------------------------------

    def solve_saa(
        self,
        sampled_scenarios: List[Dict[str, Any]],
        solver_name: str,
        solver_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Solve the SAA problem on one sampled scenario batch.

        This is a convenience wrapper around build_model_data and
        solve_extensive_form.
        """
        model_data = self.build_model_data(sampled_scenarios)
        return self.solve_extensive_form(
            model_data=model_data,
            solver_name=solver_name,
            solver_options=solver_options,
        )

    def evaluate_xhat(
        self,
        xhat: Dict[str, float],
        sampled_scenarios: List[Dict[str, Any]],
        solver_name: str,
        solver_options: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Evaluate a fixed first-stage candidate solution on one sampled scenario batch.
        """
        model_data = self.build_model_data(sampled_scenarios)
        return self.evaluate_first_stage_solution(
            xhat=xhat,
            model_data=model_data,
            solver_name=solver_name,
            solver_options=solver_options,
        )

    def replication_gap(
        self,
        xhat: Dict[str, float],
        sampled_scenarios: List[Dict[str, Any]],
        solver_name: str,
        solver_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Compute one replication-level gap estimate on one sampled batch of scenarios.

        Returns
        -------
        dict
            Keys:
              - gap_estimate
              - xhat_value
              - saa_optimal_value
        """
        solved_saa = self.solve_saa(
            sampled_scenarios=sampled_scenarios,
            solver_name=solver_name,
            solver_options=solver_options,
        )
        saa_optimal_value = self.get_objective_value(solved_saa)

        xhat_value = self.evaluate_xhat(
            xhat=xhat,
            sampled_scenarios=sampled_scenarios,
            solver_name=solver_name,
            solver_options=solver_options,
        )

        gap_estimate = xhat_value - saa_optimal_value

        return {
            "gap_estimate": gap_estimate,
            "xhat_value": xhat_value,
            "saa_optimal_value": saa_optimal_value,
        }
