import sys
#import pprint
import json
import copy
import munch
import logging

import pyomo.core.base.indexed_component
import pyomo.environ as pyo
import pyomo.util.vars_from_expressions as vfe
#from . import scentobund

#import forestlib.util
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
        self.int_to_FirstStageVar = {}
        self.int_to_FirstStageVarName = {}
        self.solver_options = {}
        self.pyo_solver = None

    def _first_stage_variables(self, *, M):
        # A generator that yields (name,component) tuples
        pass

    def _initialize_cuid_map(self, *, M, b):
        if len(self.varcuid_to_int) == 0:
            #
            # self.varcuid_to_int maps the cuids for variables to unique integers (starting with 0).
            #   The variable cuids indexed here are specified by the list self.first_stage_variables.
            #
            self.varcuid_to_int = {}
            for varname, var in self._first_stage_variables(M=M):
                i = len(self.varcuid_to_int)
                self.varcuid_to_int[pyo.ComponentUID(var)] = i
                if var.is_binary() or var.is_integer():
                    self._binary_or_integer_fsv.add(i)
        #
        # Setup int_to_FirstStageVar and int_to_FirstStageVarName
        #
        self.int_to_FirstStageVar[b] = {}
        self.int_to_FirstStageVarName[b] = {}
        for varname, var in self._first_stage_variables(M=M):
            ndx = self.varcuid_to_int[pyo.ComponentUID(var)]
            self.int_to_FirstStageVar[b][ndx] = var
            self.int_to_FirstStageVarName[b][ndx] = varname

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

    def get_variable_name(self, b, v):
        # return self.int_to_FirstStageVar[b][v].name
        assert (
            b in self.int_to_FirstStageVarName
        ), f"Missing keys: {b} not in {self.int_to_FirstStageVarName}"
        return self.int_to_FirstStageVarName[b][v]

    def shared_variables(self):
        return list(range(len(self.varcuid_to_int)))

    def solve(self, M, *, solver_options=None, tee=False, solver=None):
        if solver_options:
            self.solver_options = solver_options
        if solver:
            self.solver = solver
        pyo_solver = pyo.SolverFactory(self.solver)
        tee = self.solver_options.get("tee", tee)
        solver_options_ = copy.copy(self.solver_options)
        if "tee" in solver_options_:
            del solver_options_["tee"]

        results = pyo_solver.solve(
            M, options=solver_options_, tee=tee, load_solutions=False
        )
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
                M.pprint()
                M.display()
                sys.stdout.flush()

            # Return the value of the 'first' objective
            return munch.Munch(
                obj_value=list(results.Solution[0].Objective.values())[0]["Value"],
                termination_condition=results.solver.termination_condition,
                status=results.solver.status,
            )


class StochasticProgram_Pyomo_NamedBuilder(StochasticProgram_Pyomo_Base):

    def __init__(
        self,
        *,
        first_stage_variables,
    ):
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
            logger.warning(
                "Initializing model with name {name}, which already has been initialized!  This may be a bug in the setup of this StochasticProgram instance."
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

    def _create_scenario(self, s):
        model_name, scenario = s
        data = copy.copy(self.app_data)
        for k, v in self.model_data.get(model_name, {}).items():
            assert k not in data, f"Model data for {k} has already been specified!"
            data[k] = v
        for k, v in self.scenario_data[model_name].get(scenario, {}).items():
            assert k not in data, f"Scenario data for {k} has already been specified!"
            data[k] = v
        return self.model_builder[model_name](data, {})

    def create_EF(self, *, b, w=None, x_bar=None, rho=None):
        scenarios = self.bundles[b].scenarios

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

        # 3)Create Obj:sum of scenario obj * probability
        obj = sum(
            self.bundles[b].scenario_probability[s] * obj[s].expr for s in scenarios
        )
        if w is not None:
            obj = (
                obj
                + sum(w[i] * x for i, x in self.int_to_FirstStageVar[b].items())
                + (rho / 2.0)
                * sum(
                    (x - x_bar[i]) ** 2 for i, x in self.int_to_FirstStageVar[b].items()
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

        return EF_model


class StochasticProgram_Pyomo_MultistageBuilder(StochasticProgram_Pyomo_Base):

    def __init__(self, *, model_builder_list):
        super().__init__()
        assert (type(model_builder_list) is list) and (
            len(model_builder_list) >= 2
        ), "Expecting a list of model_builder functions with length >= 2"
        assert (
            len(model_builder_list) == 2
        ), "WEH - This class only works for two-stages right now."
        self.model_builder_list = model_builder_list

    def initialize_model(self, *, name=None, filename=None, model_data=None, **kwargs):
        if filename is not None:
            with open(f"{filename}", "r") as file:
                model_data = json.load(filename)

        if model_data is not None:
            self.model_data[name] = model_data.get("data", {})
            self.scenario_data[name] = {
                scen["ID"]: scen for scen in model_data.get("scenarios", {})
            }

        if model_data is not None:
            self.initialize_bundles(models=[name])

    def _first_stage_variables(self, *, M):
        for var in find_variables(M):
            yield var.name, var

    def _create_scenario(self, M, s):
        model_name, scenario = s
        data = copy.copy(self.app_data)
        for k, v in self.model_data.get(model_name, {}).items():
            assert k not in data, f"Model data for {k} has already been specified!"
            data[k] = v
        for k, v in self.scenario_data[model_name].get(scenario, {}).items():
            assert k not in data, f"Scenario data for {k} has already been specified!"
            data[k] = v
        self.model_builder_list[1](M, M.s[s], data, {})

    def create_EF(self, *, w=None, x_bar=None, rho=None, b):
        scenarios = self.bundles[b].scenarios

        # 1) create EF model
        EF_model = pyo.ConcreteModel()
        self.model_builder_list[0](EF_model, self.app_data, {})
        # Find the root objective
        root_obj = find_objective(EF_model)
        # Initialize the cuid_map, and also initialize the int_to_FirstStageVar map for this bundle
        self._initialize_cuid_map(M=EF_model, b=b)

        # 2) Loop through scenario dictionary, add block, deactivate Obj
        EF_model.s = pyo.Block(scenarios)
        obj_comp = {}
        for s in scenarios:
            self._create_scenario(EF_model, s)
            obj_comp[s] = find_objective(EF_model.s[s])
            assert (
                obj_comp[s] is not None
            ), f"Cannot find objective on block for scenario '{s}'"
            obj_comp[s].deactivate()

        # 3)Create Obj: root_obj + (sum of scenario obj * probability)
        obj = 0 if root_obj is None else root_obj
        obj = obj + sum(
            self.bundles[b].scenario_probability[s] * obj_comp[s].expr
            for s in scenarios
        )
        if w is not None:
            obj = (
                obj
                + sum(w[i] * x for i, x in self.int_to_FirstStageVar[b].items())
                + (rho / 2.0)
                * sum(
                    (x - x_bar[i]) ** 2 for i, x in self.int_to_FirstStageVar[b].items()
                )
            )
        EF_model.obj = pyo.Objective(expr=obj)
        if root_obj is not None:
            root_obj.deactivate()

        return EF_model

