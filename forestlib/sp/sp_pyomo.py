import sys
import json
import copy
import munch
import logging

import pyomo.core.base.indexed_component
import pyomo.environ as pyo
import pyomo.repn
import pyomo.util.vars_from_expressions as vfe

import forestlib.logs
from .sp import StochasticProgram

logger = forestlib.logs.logger


def find_objective(model):
    obj = None
    for comp in model.component_data_objects(pyo.Objective, active=True):
        assert obj is None, "Cannot handle multiple objectives"
        obj = comp
    return obj


def find_variables(model):
    for comp in model.component_data_objects(pyo.Var, active=True):
        if comp.is_indexed():
            for var in comp.values():
                yield var
        else:
            yield comp


class StochasticProgram_Pyomo_Base(StochasticProgram):

    def __init__(self):
        super().__init__()
        self.varcuid_to_int = {}
        self.int_to_FirstStageVar = {}  # indexed by bundle id
        self.int_to_FirstStageVarName = {}
        self.int_to_ObjectiveCoef = {}
        self.solver_options = {}
        self.pyo_solver = None
        self._model_cache = {}  # indexed by bundle id

    def _first_stage_variables(self, *, M):
        # A generator that yields (name,component) tuples
        pass

    def _initialize_cuid_map(self, *, M, b):
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

    def set_bundles(self, bundles):
        self.int_to_FirstStageVar = {}
        # self.int_to_FirstStageVarName = {}
        self._model_cache = {}
        StochasticProgram.set_bundles(self, bundles)

    def continuous_fsv(self):
        assert (
            self._binary_or_integer_fsv is not None
        ), "ERROR: cannot call continuous_fsv() until a model has been constructed"
        return len(self._binary_or_integer_fsv) == 0

    def round(self, v, value):
        if v in self._binary_or_integer_fsv:
            return round(value)
        return value

    def fix_variable(self, b, v, value):
        self.int_to_FirstStageVar[b][v].fix(value)

    def get_variable_value(self, b, v):
        return pyo.value(self.int_to_FirstStageVar[b][v])

    def get_variable_name(self, v):
        assert (
            v in self.int_to_FirstStageVarName
        ), f"Missing keys: {v} not in {self.int_to_FirstStageVarName}"
        return self.int_to_FirstStageVarName[v]

    def shared_variables(self):
        return list(range(len(self.varcuid_to_int)))

    def solve(self, M, *, solver_options=None, tee=False, solver=None):
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
                    obj_value=list(results.Solution[0].Objective.values())[0]["Value"],
                    termination_condition=results.solver.termination_condition,
                    status=results.solver.status,
                )

    def create_EF(self, cache_bundles=False):
        if cache_bundles:
            _int_toFirstStageVar = self.int_to_FirstStageVar
            _model_cache = self._model_cache
            _bundles = self.bundles

        self.initialize_bundles(scheme="single_bundle")
        assert (
            len(self.bundles) == 1
        ), f"The extensive form should only have one bundle: {len(self.bundles)}"

        b = next(iter(self.bundles))
        M = self.create_subproblem(b)

        if cache_bundles:
            self.int_to_FirstStageVar = _int_toFirstStageVar
            self._model_cache = _model_cache
            self.bundles = _bundles
        return M


class StochasticProgram_Pyomo_NamedBuilder(StochasticProgram_Pyomo_Base):

    def __init__(self, *, first_stage_variables):
        super().__init__()
        #
        # A list of string names of variables, such as:
        #   [ "x", "b.y", "b[*].z[*,*]" ]
        #
        self.first_stage_variables = first_stage_variables
        # WEH - We may have different objectives for different model_builders?
        self.objective = None
        self.model_builder = {}

    def initialize_model(
        self,
        *,
        name=None,
        filename=None,
        model_data=None,
        model_builder=None,
        default=True,
        **kwargs,
    ):
        if default:
            self.default_model = name

        if filename is not None:
            with open(f"{filename}", "r") as file:
                model_data = json.load(filename)

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
            self.initialize_bundles(models=[name])

    def _first_stage_variables(self, *, M):
        for varname in self.first_stage_variables:
            cuid = pyo.ComponentUID(varname)
            comp = cuid.find_component_on(M)
            assert comp is not None, "Pyomo error: Unknown variable '%s'" % varname
            if comp.is_indexed():
                for var in comp.values():
                    yield var.name, var
            else:
                yield varname, comp

    def _create_scenario(self, scenario_tuple):
        model_name, scenario = scenario_tuple
        data = copy.copy(self.app_data)
        for k, v in self.model_data.get(model_name, {}).items():
            assert k not in data, f"Model data for {k} has already been specified!"
            data[k] = v
        for k, v in self.scenario_data[model_name].get(scenario, {}).items():
            assert k not in data, f"Scenario data for {k} has already been specified!"
            data[k] = v
        return self.model_builder[model_name](data, {})

    def get_objective_coef(self, v, cached=False):
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

            self.initialize_bundles(
                models=[self.default_model], scheme="single_scenario"
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

    def create_bundle_EF(self, *, b, w=None, x_bar=None, rho=None, cached=False):
        scenarios = self.bundles[b].scenarios
        if cached and b in self._model_cache:
            M = self._model_cache[b]

            if rho is None:
                for i, x in self.int_to_FirstStageVar[b].items():
                    M.forestlib_params.rho[i].set_value(0.0)
            else:
                for i, x in self.int_to_FirstStageVar[b].items():
                    M.forestlib_params.rho[i].set_value(rho[i])

            if w is None:
                for i in M.forestlib_params.w:
                    M.forestlib_params.w[i].set_value(0.0)
            else:
                assert len(w) == len(
                    M.forestlib_params.w
                ), f"Inconsistent data sizes between param.w ({len(M.forestlib_params.w)}) and w ({len(w)})"
                for i in M.forestlib_params.w:
                    M.forestlib_params.w[i].set_value(w[i])

            if x_bar is None:
                for i in M.forestlib_params.x_bar:
                    M.forestlib_params.x_bar[i].set_value(0.0)
            else:
                assert len(x_bar) == len(
                    M.forestlib_params.x_bar
                ), f"Inconsistent data sizes between param.x_bar ({len(M.forestlib_params.x_bar)}) and x_bar ({len(x_bar)})"
                for i in M.forestlib_params.x_bar:
                    M.forestlib_params.x_bar[i].set_value(x_bar[i])

            return M

        # 1) create scenario dictionary
        scen_dict = {}
        if len(scenarios) > 1:
            for s in scenarios:
                scenario_model = self._create_scenario(s)
                self._initialize_cuid_map(M=scenario_model, b=b)
                scen_dict[s] = scenario_model
        else:
            s = scenarios[0]
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
        if len(scenarios) > 1:
            EF_model.rootx = pyo.Var(list(self.varcuid_to_int.values()))
            self.int_to_FirstStageVar[b] = {
                i: EF_model.rootx[i] for i in self.varcuid_to_int.values()
            }

        # 3) Store objective parameters in a common format
        if cached:
            params = pyo.Block()
            A = list(self.int_to_FirstStageVar[b].keys())
            assert len(A) > 0, f"ERROR: b {b}, {self.int_to_FirstStageVar}"
            params.rho = pyo.Param(A, mutable=True, default=0.0, domain=pyo.Reals)
            params.w = pyo.Param(A, mutable=True, default=0.0, domain=pyo.Reals)
            params.x_bar = pyo.Param(A, mutable=True, default=0.0, domain=pyo.Reals)
            EF_model.forestlib_params = params
        else:
            params = munch.Munch(rho=rho, w=w, x_bar=x_bar)

        # 3)Create Obj:sum of scenario obj * probability
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

        # 4)Constrain First Stage Variable values to be equal under all scenarios
        if len(scenarios) > 1:
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
                    EF_model.non_ant_cons.add(expr=EF_model.rootx[i] == var)

        # Cache the model if the 'cached' flag has been specified
        if cached:
            self._model_cache[b] = EF_model

        return EF_model
