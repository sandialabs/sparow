import sys
import json
import copy
import munch
import logging
from typing import Any

import pyomo.core.base.indexed_component
import pyomo.environ as pyo
import pyomo.repn
import pyomo.util.vars_from_expressions as vfe

import sparow.logs
from .sp import StochasticProgram, initialize_bundles
from .replace_variables_transformation import ReplaceVariablesTransformation

logger = sparow.logs.logger


def find_objective(model: Any) -> Any:
    """
    Find the active objective in a Pyomo model.

    Parameters
    ----------
    model : pyomo.ConcreteModel
        The Pyomo model to search for objectives.

    Returns
    -------
    pyomo.Objective
        The active objective found in the model.

    Raises
    ------
    AssertionError
        If multiple objectives are found in the model.
    """
    obj = None
    for comp in model.component_data_objects(pyo.Objective, active=True):
        assert obj is None, "Cannot handle multiple objectives"
        obj = comp
    return obj


def find_variables(model: Any) -> Any:
    """
    Generator function to find all active variables in a Pyomo model.

    Parameters
    ----------
    model : pyomo.ConcreteModel
        The Pyomo model to search for variables.

    Yields
    ------
    pyomo.Var
        Active variables in the model.
    """
    for comp in model.component_data_objects(pyo.Var, active=True):
        if comp.is_indexed():
            for var in comp.values():
                yield var
        else:
            yield comp


class StochasticProgram_Pyomo_Base(StochasticProgram):
    """
    Base class for Pyomo-based stochastic programs.

    Attributes
    ----------
    varcuid_to_int : dict
        Dictionary mapping variable component UIDs to integers.
    int_to_FirstStageVar : dict
        Dictionary mapping integers to first-stage variables, indexed by bundle ID.
    int_to_FirstStageVarName : dict
        Dictionary mapping integers to first-stage variable names.
    int_to_ObjectiveCoef : dict
        Dictionary mapping integers to objective coefficients.
    solver_options : dict
        Dictionary of solver options.
    pyo_solver : pyomo.SolverFactory or None
        The Pyomo solver instance.
    _model_cache : dict
        Dictionary caching models, indexed by bundle ID.
    """

    def __init__(self) -> None:
        super().__init__()
        self.varcuid_to_int: dict = {}
        self.int_to_FirstStageVar: dict = {}  # indexed by bundle id
        self.int_to_FirstStageVarName: dict = {}
        self.int_to_ObjectiveCoef: dict = {}
        self.solver_options: dict = {}
        self.pyo_solver: Any | None = None
        self._model_cache: dict = {}  # indexed by bundle id

    def _first_stage_variables(self, *, M: Any) -> Any:
        """
        Generator function to yield first-stage variables.

        Parameters
        ----------
        M : pyomo.ConcreteModel
            The Pyomo model to extract first-stage variables from.

        Yields
        ------
        tuple
            Tuples of (variable_name, variable_component).
        """
        # A generator that yields (name,component) tuples
        pass

    def _initialize_cuid_map(self, *, M: Any, b: str) -> None:
        """
        Initialize the mapping between variable component UIDs and integers.

        Parameters
        ----------
        M : pyomo.ConcreteModel
            The Pyomo model containing the variables.
        b : str
            Bundle identifier.
        """
        fsv = list(self._first_stage_variables(M=M))
        if len(self.varcuid_to_int) == 0:
            #
            # self.varcuid_to_int maps the cuids for variables to unique integers (starting with 0).
            #   The variable cuids indexed here are specified by the list self.first_stage_variables.
            #
            for varname, var in fsv:
                i = len(self.varcuid_to_int)
                self.varcuid_to_int[pyo.ComponentUID(var, context=M)] = i
                self.int_to_FirstStageVarName[i] = varname
                if var.is_binary() or var.is_integer():
                    self._binary_or_integer_fsv.add(i)
        #
        # Setup int_to_FirstStageVarName
        #
        self.int_to_FirstStageVar[b] = {
            self.varcuid_to_int[pyo.ComponentUID(var, context=M)]: var for _, var in fsv
        }

    def set_bundles(self, bundles: Any) -> None:
        """
        Set the bundles for the stochastic program and reset related mappings.

        Parameters
        ----------
        bundles : BundleObj
            The bundles to set.
        """
        self.int_to_FirstStageVar = {}
        # self.int_to_FirstStageVarName = {}
        self._model_cache = {}
        StochasticProgram.set_bundles(self, bundles)

    def continuous_fsv(self) -> bool:
        """
        Check if all first-stage variables are continuous.

        Returns
        -------
        bool
            True if all first-stage variables are continuous, False otherwise.

        Raises
        ------
        AssertionError
            If called before a model has been constructed.
        """
        assert (
            self._binary_or_integer_fsv is not None
        ), "ERROR: cannot call continuous_fsv() until a model has been constructed"
        return len(self._binary_or_integer_fsv) == 0

    def round(self, v: int, value: float) -> float | int:
        """
        Round a value if the variable is binary or integer.

        Parameters
        ----------
        v : int
            Variable identifier.
        value : float
            Value to round.

        Returns
        -------
        float or int
            Rounded value if variable is binary/integer, otherwise original value.
        """
        if v in self._binary_or_integer_fsv:
            return round(value)
        return value

    def fix_variable(self, b: str, v: int, value: float) -> None:
        """
        Fix a variable to a specific value.

        Parameters
        ----------
        b : str
            Bundle identifier.
        v : int
            Variable identifier.
        value : float
            Value to fix the variable to.
        """
        self.int_to_FirstStageVar[b][v].fix(value)

    def get_variable_value(self, b: str, v: int) -> float:
        """
        Get the value of a variable.

        Parameters
        ----------
        b : str
            Bundle identifier.
        v : int
            Variable identifier.

        Returns
        -------
        float
            The value of the variable.
        """
        return pyo.value(self.int_to_FirstStageVar[b][v])

    def get_variable_name(self, v: int) -> str:
        """
        Get the name of a variable.

        Parameters
        ----------
        v : int
            Variable identifier.

        Returns
        -------
        str
            The name of the variable.

        Raises
        ------
        AssertionError
            If the variable identifier is not found.
        """
        assert (
            v in self.int_to_FirstStageVarName
        ), f"Missing keys: {v} not in {self.int_to_FirstStageVarName}"
        return self.int_to_FirstStageVarName[v]

    def shared_variables(self) -> list:
        """
        Get the list of shared variable identifiers.

        Returns
        -------
        list
            List of variable identifiers.
        """
        return list(range(len(self.varcuid_to_int)))

    def solve(
        self,
        M: Any,
        *,
        solver_options: dict | None = None,
        tee: bool = False,
        solver: str | None = None,
    ) -> Any:
        """
        Solve a Pyomo model using the specified solver.

        Parameters
        ----------
        M : pyomo.ConcreteModel
            The Pyomo model to solve.
        solver_options : dict, optional
            Dictionary of solver options.
        tee : bool, optional
            Whether to display solver output (default is False).
        solver : str, optional
            Name of the solver to use.

        Returns
        -------
        Munch
            A Munch object containing solver results with keys:
            - obj_value (float): The objective value.
            - termination_condition: The solver termination condition.
            - status: The solver status.
        """
        options = copy.copy(self.solver_options)
        if solver_options:
            options.update(solver_options)
        tee = options.pop("tee", tee)

        if solver:
            self.solver = solver
        pyo_solver = pyo.SolverFactory(self.solver)
        if options:
            for k, v in options.items():
                pyo_solver.options[k] = v

        results = pyo_solver.solve(M, tee=tee, load_solutions=False)
        status = results.solver.status
        if not pyo.check_optimal_termination(results):
            condition = results.solver.termination_condition
            logger.debug(
                (
                    "Error solving subproblem '{}': "
                    "SolverStatus = {}, "
                    "TerminationCondition = {}"
                ).format(M.name, status.value, condition.value)
            )
            return munch.Munch(
                obj_value=None,
                termination_condition=results.solver.termination_condition,
                status=results.solver.status,
            )
        else:
            # Load the results into the model so the user can find them there
            M.solutions.load_from(results)
            if logger.isEnabledFor(logging.DEBUG):
                print("-" * 70)
                print("Solver Results")
                print("-" * 70)
                M.pprint()
                M.display()
                sys.stdout.flush()

            # Return the value of the 'first' objective

            if self.solver == "ipopt":
                return munch.Munch(
                    obj_value=pyo.value(M.obj),
                    termination_condition=results.solver.termination_condition,
                    status=results.solver.status,
                )
            else:
                return munch.Munch(
                    obj_value=pyo.value(M.obj),
                    # obj_value=list(results.Solution[0].Objective.values())[0]["Value"],
                    termination_condition=results.solver.termination_condition,
                    status=results.solver.status,
                )

    def create_EF(
        self,
        *,
        model_fidelities: dict | None = None,
        cache_bundles: bool = False,
        compact_repn: bool = True,
    ) -> Any:
        """
        Create the extensive form of the stochastic program.

        Parameters
        ----------
        model_fidelities : dict, optional
            Dictionary specifying model fidelities.
        cache_bundles : bool, optional
            Whether to cache bundles (default is False).
        compact_repn : bool, optional
            Whether to use compact representation (default is True).

        Returns
        -------
        pyomo.ConcreteModel
            The extensive form model.
        """
        if cache_bundles:
            _int_toFirstStageVar = self.int_to_FirstStageVar
            _model_cache = self._model_cache
            _bundles = self.bundles

        if model_fidelities is None:
            self.initialize_bundles(scheme="single_bundle")
        else:
            self.initialize_bundles(scheme="single_bundle", models=model_fidelities)
        assert (
            len(self.bundles) == 1
        ), f"The extensive form should only have one bundle: {len(self.bundles)}"

        b = next(iter(self.bundles))
        M = self.create_subproblem(b, compact_repn=compact_repn)

        if cache_bundles:
            self.int_to_FirstStageVar = _int_toFirstStageVar
            self._model_cache = _model_cache
            self.bundles = _bundles
        return M


class StochasticProgram_Pyomo_NamedBuilder(StochasticProgram_Pyomo_Base):
    """
    Pyomo-based stochastic program builder using named variables.

    Attributes
    ----------
    first_stage_variables : list
        List of string names of first-stage variables.
    objective : str or None
        Name of the objective function.
    model_builder : dict
        Dictionary mapping model names to their builder functions.
    """

    def __init__(self, *, first_stage_variables: list) -> None:
        super().__init__()
        #
        # A list of string names of variables, such as:
        #   [ "x", "b.y", "b[*].z[*,*]" ]
        #
        self.first_stage_variables = first_stage_variables
        # WEH - We may have different objectives for different model_builders?
        self.objective: str | None = None
        self.model_builder: dict = {}

    def initialize_model(
        self,
        *,
        name: str | None = None,
        filename: str | None = None,
        model_data: dict | None = None,
        model_builder: Any | None = None,
        default: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a model with data and builder function.

        Parameters
        ----------
        name : str, optional
            Name of the model.
        filename : str, optional
            Path to a JSON file containing model data.
        model_data : dict, optional
            Dictionary containing model data.
        model_builder : function, optional
            Function to build the model.
        default : bool, optional
            Whether this is the default model (default is True).
        **kwargs : dict
            Additional keyword arguments.
        """
        if default:
            self.default_model = name

        if filename is not None:
            with open(f"{filename}", "r") as file:
                model_data = json.load(file)

        if name in self.model_data:
            if name is not None:
                logger.warning(
                    f"Initializing model with name '{name}', which already has been initialized!  This may be a bug in the setup of this StochasticProgram instance."
                )
        if model_data is not None:
            self.model_data[name] = model_data.get("data", {})
            self.scenario_data[name] = {
                scen["ID"]: scen for scen in model_data.get("scenarios", {})
            }
        else:
            self.model_data[name] = {}
            self.scenario_data[name] = {}

        if model_builder is not None:
            self.model_builder[name] = model_builder
        if model_data is not None and default:
            self.set_bundles(
                initialize_bundles(
                    models=[name],
                    model_data=self.model_data,
                    scenario_data=self.scenario_data,
                )
            )

    def _first_stage_variables(self, *, M: Any) -> Any:
        """
        Generator function to yield first-stage variables based on their names.

        Parameters
        ----------
        M : pyomo.ConcreteModel
            The Pyomo model to extract first-stage variables from.

        Yields
        ------
        tuple
            Tuples of (variable_name, variable_component).

        Raises
        ------
        AssertionError
            If a variable name is not found in the model.
        """
        for varname in self.first_stage_variables:
            cuid = pyo.ComponentUID(varname)
            comp = cuid.find_component_on(M)
            assert comp is not None, "Pyomo error: Unknown variable '%s'" % varname
            if comp.is_indexed():
                for var in comp.values():
                    yield var.name, var
            else:
                yield varname, comp

    def _create_scenario(self, scenario_tuple: tuple) -> Any:
        """
        Create a scenario model using the model builder.

        Parameters
        ----------
        scenario_tuple : tuple
            Tuple containing model name and scenario identifier.

        Returns
        -------
        pyomo.ConcreteModel
            The constructed scenario model.

        Raises
        ------
        AssertionError
            If data keys are already specified.
        """
        model_name, scenario = scenario_tuple
        data = copy.copy(self.app_data)
        for k, v in self.model_data.get(model_name, {}).items():
            assert k not in data, f"Model data for {k} has already been specified!"
            data[k] = v
        for k, v in self.scenario_data[model_name].get(scenario, {}).items():
            assert k not in data, f"Scenario data for {k} has already been specified!"
            data[k] = v
        return self.model_builder[model_name](data, {})

    def get_objective_coef(self, v: int, cached: bool = False) -> float:
        """
        Get the objective coefficient for a variable.

        Parameters
        ----------
        v : int
            Variable identifier.
        cached : bool, optional
            Whether to use cached values (default is False).

        Returns
        -------
        float
            The objective coefficient for the variable.
        """
        if len(self.int_to_ObjectiveCoef) == 0:
            #
            # Here we build the extensive form for the 'default' model and keep its objective expression.
            # This logic mimics the logic of StochasticProgram.evaluate()
            #

            # Setup single-scenario bundles with the default model
            _int_toFirstStageVar = self.int_to_FirstStageVar
            _model_cache = self._model_cache
            _bundles = self.bundles
            # stack.append(StochasticProgram.set_bundles(self, self.bundles))

            self.set_bundles(
                initialize_bundles(
                    models=[self.default_model],
                    scheme="single_scenario",
                    model_data=self.model_data,
                    scenario_data=self.scenario_data,
                )
            )

            obj_expr = {}
            _models = {}
            for b in self.bundles:
                s = self.bundles[b].scenarios
                M = self._create_scenario(s[0])
                _models[b] = M
                self._initialize_cuid_map(M=M, b=b)
                obj_expr[b] = find_objective(M).expr
            obj = sum(self.bundles[b].probability * obj_expr[b] for b in self.bundles)

            repn = pyomo.repn.generate_standard_repn(obj, quadratic=False)

            for index in self.varcuid_to_int.values():
                self.int_to_ObjectiveCoef[index] = 0

            for i, var in enumerate(repn.linear_vars):
                cuid = pyo.ComponentUID(var)
                if cuid in self.varcuid_to_int:
                    self.int_to_ObjectiveCoef[self.varcuid_to_int[cuid]] = (
                        repn.linear_coefs[i]
                    )

            # Setup single-scenario bundles with the default model
            self.bundles = _bundles
            self._model_cache = _model_cache
            self.int_to_FirstStageVar = _int_toFirstStageVar

        return self.int_to_ObjectiveCoef[v]

    def create_bundle_EF(
        self,
        *,
        b: str,
        w: list | None = None,
        x_bar: list | None = None,
        rho: list | None = None,
        cached: bool = False,
        compact_repn: bool = True,
    ) -> Any:
        """
        Create an integer programming representation for the bundle extensive form.

        If the cached flag is on, then the model will be constructed with mutable parameter
        objects.  Repeated calls to create_bundle_EF() will avoid reconstructing the entire model.
        Instead, the mutable parameters will be set with the values of rho, w and x_bar.

        If the cached flag is False, then the model is constructed with fixed values for rho, w and x_bar.

        Parameters
        ----------
        b : str
            Bundle identifier.
        w : list, optional
            List of weights.
        x_bar : list, optional
            List of fixed variable values.
        rho : list, optional
            List of penalty parameters.
        cached : bool, optional
            Whether to cache the model (default is False).
        compact_repn : bool, optional
            Whether to use compact representation (default is True).

        Returns
        -------
        pyomo.ConcreteModel
            The extensive form model for the bundle.
        """
        if cached and b in self._model_cache:
            M = self._model_cache[b]

            if rho is None:
                for i, x in self.int_to_FirstStageVar[b].items():
                    M.sparow_params.rho[i].set_value(0.0)
            else:
                for i, x in self.int_to_FirstStageVar[b].items():
                    M.sparow_params.rho[i].set_value(rho[i])

            if w is None:
                for i in M.sparow_params.w:
                    M.sparow_params.w[i].set_value(0.0)
            else:
                assert len(w) == len(
                    M.sparow_params.w
                ), f"Inconsistent data sizes between param.w ({len(M.sparow_params.w)}) and w ({len(w)})"
                for i in M.sparow_params.w:
                    M.sparow_params.w[i].set_value(w[i])

            if x_bar is None:
                for i in M.sparow_params.x_bar:
                    M.sparow_params.x_bar[i].set_value(0.0)
            else:
                assert len(x_bar) == len(
                    M.sparow_params.x_bar
                ), f"Inconsistent data sizes between param.x_bar ({len(M.sparow_params.x_bar)}) and x_bar ({len(x_bar)})"
                for i in M.sparow_params.x_bar:
                    M.sparow_params.x_bar[i].set_value(x_bar[i])

            return M

        EF_model = self.create_bundle_EF_repn(
            b=b,
            w=w,
            x_bar=x_bar,
            rho=rho,
            cached=cached,
            compact_repn=compact_repn,
        )

        # Cache the model if the 'cached' flag has been specified
        if cached:
            self._model_cache[b] = EF_model

        return EF_model

    def create_bundle_EF_repn(
        self,
        *,
        b: str,
        w: list | None = None,
        x_bar: list | None = None,
        rho: list | None = None,
        cached: bool = False,
        compact_repn: bool = True,
    ) -> Any:
        """
        Create the extensive form representation for a bundle.

        Parameters
        ----------
        b : str
            Bundle identifier.
        w : list, optional
            List of weights.
        x_bar : list, optional
            List of fixed variable values.
        rho : list, optional
            List of penalty parameters.
        cached : bool, optional
            Whether to cache the model (default is False).
        compact_repn : bool, optional
            Whether to use compact representation (default is True).

        Returns
        -------
        pyomo.ConcreteModel
            The extensive form model for the bundle.
        """
        if not compact_repn:
            EF_model = self._create_noncompact_bundle_EF_repn(
                b=b, w=w, x_bar=x_bar, rho=rho, cached=cached
            )

        elif len(self.bundles[b].scenarios) == 1:
            EF_model = self._create_single_scenario_EF_repn(
                b=b, w=w, x_bar=x_bar, rho=rho, cached=cached
            )

        else:
            EF_model = self._create_compact_bundle_EF_repn(
                b=b, w=w, x_bar=x_bar, rho=rho, cached=cached
            )

        # Cache the model if the 'cached' flag has been specified
        if cached:
            self._model_cache[b] = EF_model

        return EF_model

    def _create_single_scenario_EF_repn(
        self,
        *,
        b: str,
        w: list | None = None,
        x_bar: list | None = None,
        rho: list | None = None,
        cached: bool = False,
    ) -> Any:
        """
        Create a pyomo model for EF with a single scenario.

        Even though there's a single scenario, we use an indexed Block to
        ensure consistency of model structure.

        Parameters
        ----------
        b : str
            Bundle identifier.
        w : list, optional
            List of weights.
        x_bar : list, optional
            List of fixed variable values.
        rho : list, optional
            List of penalty parameters.
        cached : bool, optional
            Whether to cache the model (default is False).

        Returns
        -------
        pyomo.ConcreteModel
            The extensive form model for a single scenario.
        """
        scenarios = self.bundles[b].scenarios

        # 1) create scenario dictionary
        s = scenarios[0]
        scen_model = self._create_scenario(s)
        self._initialize_cuid_map(M=scen_model, b=b)

        # 2) Loop through scenario dictionary, add block, deactivate Obj
        EF_model = pyo.ConcreteModel()
        EF_model.s = pyo.Block(scenarios)
        if self.objective is None:
            obj = {}
            EF_model.s[s].transfer_attributes_from(scen_model)
            obj[s] = find_objective(EF_model.s[s])
            assert (
                obj[s] is not None
            ), f"Cannot find objective on model for scenario '{s}'"
        else:
            objective_cuid = pyo.ComponentUID(self.objective)
            obj = {}
            EF_model.s[s].transfer_attributes_from(scen_model)
            obj[s] = objective_cuid.find_component_on(EF_model.s[s])
            assert (
                obj[s] is not None
            ), f"Cannot find objective '{self.objective}' on model for scenario '{s}'"
        obj[s].deactivate()

        # 2.5) Collect first stage variables
        EF_model.first_stage_variables = {
            i: var for i, var in self.int_to_FirstStageVar[b].items()
        }

        # 3) Store objective parameters in a common format
        if cached:
            params = pyo.Block()
            A = list(self.int_to_FirstStageVar[b].keys())
            assert len(A) > 0, f"ERROR: b {b}, {self.int_to_FirstStageVar}"
            params.rho = pyo.Param(A, mutable=True, default=0.0, domain=pyo.Reals)
            params.w = pyo.Param(A, mutable=True, default=0.0, domain=pyo.Reals)
            params.x_bar = pyo.Param(A, mutable=True, default=0.0, domain=pyo.Reals)
            EF_model.sparow_params = params
        else:
            params = munch.Munch(rho=rho, w=w, x_bar=x_bar)

        # 3) Create Obj:sum of scenario obj * probability
        obj = self.bundles[b].scenario_probability[s] * obj[s].expr
        if cached or w is not None:
            obj = (
                obj
                + sum(params.w[i] * x for i, x in self.int_to_FirstStageVar[b].items())
                + sum(
                    (params.rho[i] / 2.0) * ((x - params.x_bar[i]) ** 2)
                    for i, x in self.int_to_FirstStageVar[b].items()
                )
            )
        EF_model.obj = pyo.Objective(expr=obj)

        EF_model.scenario_varmap = {}

        return EF_model

    def _create_compact_bundle_EF_repn(
        self,
        *,
        b: str,
        w: list | None = None,
        x_bar: list | None = None,
        rho: list | None = None,
        cached: bool = False,
    ) -> Any:
        """
        Create a pyomo model for EF that does not include separate copies of first stage variables along
        with non-anticipativity constraints.  Each scenario model, after the first, is processed to
        ensure that all scenario models use the same pyomo variable objects for the first stage variables.

        This involves more up-front processing of the scenario models, but it results in a more compact EF
        representation.

        Parameters
        ----------
        b : str
            Bundle identifier.
        w : list, optional
            List of weights.
        x_bar : list, optional
            List of fixed variable values.
        rho : list, optional
            List of penalty parameters.
        cached : bool, optional
            Whether to cache the model (default is False).

        Returns
        -------
        pyomo.ConcreteModel
            The extensive form model with compact representation.
        """
        scenarios = self.bundles[b].scenarios

        # 1) create scenario dictionary
        scen_dict = {}
        for s in scenarios:
            scenario_model = self._create_scenario(s)
            self._initialize_cuid_map(M=scenario_model, b=b)
            scen_dict[s] = scenario_model

        # 2) Loop through scenario dictionary, add block, deactivate Obj
        EF_model = pyo.ConcreteModel()
        EF_model.s = pyo.Block(scenarios)
        if self.objective is None:
            obj = {}
            for s, scen_model in scen_dict.items():
                EF_model.s[s].transfer_attributes_from(scen_model)
                obj[s] = find_objective(EF_model.s[s])
                assert (
                    obj[s] is not None
                ), f"Cannot find objective on model for scenario '{s}'"
        else:
            objective_cuid = pyo.ComponentUID(self.objective)
            obj = {}
            for s, scen_model in scen_dict.items():
                EF_model.s[s].transfer_attributes_from(scen_model)
                obj[s] = objective_cuid.find_component_on(EF_model.s[s])
                assert (
                    obj[s] is not None
                ), f"Cannot find objective '{self.objective}' on model for scenario '{s}'"

        # 2.5) Find the first stage variables
        s = scenarios[0]
        EF_model.first_stage_variables = {}
        for cuid, i in self.varcuid_to_int.items():
            var = cuid.find_component_on(EF_model.s[s])
            assert (
                var is not None
            ), f"Pyomo error: Unknown variable '{cuid}' on scenario model '{s}'"
            EF_model.first_stage_variables[i] = var
        self.int_to_FirstStageVar[b] = EF_model.first_stage_variables

        # 2.6) Walk the expression trees for scenarios 1+ to use the same first stage variables as scenario 0
        xfrm = ReplaceVariablesTransformation()
        EF_model.scenario_varmap = {i: list() for i in self.shared_variables()}

        for s in scenarios[1:]:
            variable_map = {}
            for cuid, i in self.varcuid_to_int.items():
                var = cuid.find_component_on(EF_model.s[s])
                variable_map[id(var)] = EF_model.first_stage_variables[i]
                var.fix(var.lb)  # Ignore this variable
                EF_model.scenario_varmap[i].append(var)
            xfrm.apply_to(EF_model.s[s], substitution_map=variable_map)

        # 3) Store objective parameters in a common format
        if cached:
            params = pyo.Block()
            A = list(EF_model.first_stage_variables.keys())
            assert len(A) > 0, f"ERROR: b {b}, {EF_model.first_stage_variables}"
            params.rho = pyo.Param(A, mutable=True, default=0.0, domain=pyo.Reals)
            params.w = pyo.Param(A, mutable=True, default=0.0, domain=pyo.Reals)
            params.x_bar = pyo.Param(A, mutable=True, default=0.0, domain=pyo.Reals)
            EF_model.sparow_params = params
        else:
            params = munch.Munch(rho=rho, w=w, x_bar=x_bar)

        # 3) Create Obj:sum of scenario obj * probability
        for s in scenarios:
            obj[s].deactivate()

        obj = sum(
            self.bundles[b].scenario_probability[s] * obj[s].expr for s in scenarios
        )
        if cached or w is not None:
            obj = (
                obj
                + sum(
                    params.w[i] * x for i, x in EF_model.first_stage_variables.items()
                )
                + sum(
                    (params.rho[i] / 2.0) * ((x - params.x_bar[i]) ** 2)
                    for i, x in EF_model.first_stage_variables.items()
                )
            )
        EF_model.obj = pyo.Objective(expr=obj)

        return EF_model

    def _create_noncompact_bundle_EF_repn(
        self,
        *,
        b: str,
        w: list | None = None,
        x_bar: list | None = None,
        rho: list | None = None,
        cached: bool = False,
    ) -> Any:
        """
        Create a pyomo model for EF that includes separate copies of first stage variables along
        with non-anticipativity constraints that ensure that all scenario solutions are the same.

        Parameters
        ----------
        b : str
            Bundle identifier.
        w : list, optional
            List of weights.
        x_bar : list, optional
            List of fixed variable values.
        rho : list, optional
            List of penalty parameters.
        cached : bool, optional
            Whether to cache the model (default is False).

        Returns
        -------
        pyomo.ConcreteModel
            The extensive form model with non-compact representation.
        """
        scenarios = self.bundles[b].scenarios

        # 1) create scenario dictionary
        scen_dict = {}
        for s in scenarios:
            scenario_model = self._create_scenario(s)
            self._initialize_cuid_map(M=scenario_model, b=b)
            scen_dict[s] = scenario_model

        # 2) Loop through scenario dictionary, add block, deactivate Obj
        EF_model = pyo.ConcreteModel()
        EF_model.s = pyo.Block(scenarios)
        if self.objective is None:
            obj = {}
            for s, scen_model in scen_dict.items():
                EF_model.s[s].transfer_attributes_from(scen_model)
                obj[s] = find_objective(EF_model.s[s])
                assert (
                    obj[s] is not None
                ), f"Cannot find objective on model for scenario '{s}'"
                obj[s].deactivate()
        else:
            objective_cuid = pyo.ComponentUID(self.objective)
            obj = {}
            for s, scen_model in scen_dict.items():
                EF_model.s[s].transfer_attributes_from(scen_model)
                obj[s] = objective_cuid.find_component_on(EF_model.s[s])
                assert (
                    obj[s] is not None
                ), f"Cannot find objective '{self.objective}' on model for scenario '{s}'"
                obj[s].deactivate()

        # 2.5) Create first stage variables
        EF_model.first_stage_variables = pyo.Var(list(self.varcuid_to_int.values()))
        self.int_to_FirstStageVar[b] = {
            i: EF_model.first_stage_variables[i] for i in self.varcuid_to_int.values()
        }

        # 3) Store objective parameters in a common format
        if cached:
            params = pyo.Block()
            A = list(self.int_to_FirstStageVar[b].keys())
            assert len(A) > 0, f"ERROR: b {b}, {self.int_to_FirstStageVar}"
            params.rho = pyo.Param(A, mutable=True, default=0.0, domain=pyo.Reals)
            params.w = pyo.Param(A, mutable=True, default=0.0, domain=pyo.Reals)
            params.x_bar = pyo.Param(A, mutable=True, default=0.0, domain=pyo.Reals)
            EF_model.sparow_params = params
        else:
            params = munch.Munch(rho=rho, w=w, x_bar=x_bar)

        # 3) Create Obj:sum of scenario obj * probability
        obj = sum(
            self.bundles[b].scenario_probability[s] * obj[s].expr for s in scenarios
        )
        if cached or w is not None:
            obj = (
                obj
                + sum(params.w[i] * x for i, x in self.int_to_FirstStageVar[b].items())
                + sum(
                    (params.rho[i] / 2.0) * ((x - params.x_bar[i]) ** 2)
                    for i, x in self.int_to_FirstStageVar[b].items()
                )
            )
        EF_model.obj = pyo.Objective(expr=obj)

        # 4) Constrain First Stage Variable values to be equal under all scenarios
        EF_model.non_ant_cons = pyo.ConstraintList()

        for cuid, i in self.varcuid_to_int.items():
            for s in scenarios:
                var = cuid.find_component_on(EF_model.s[s])
                assert (
                    var is not None
                ), "Pyomo error: Unknown variable '%s' on scenario model '%s'" % (
                    cuid,
                    s,
                )
                EF_model.non_ant_cons.add(expr=EF_model.first_stage_variables[i] == var)

        return EF_model
