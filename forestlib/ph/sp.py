import pprint
import json
import copy
import munch
import pyomo.core.base.indexed_component
import pyomo.environ as pyo
import pyomo.util.vars_from_expressions as vfe
from . import scentobund

import forestlib.logs

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


class StochasticProgram(object):

    def __init__(self):
        self.solver = "gurobi"
        self.bundles = {}
        self.bundle_probability = {}
        self.scenarios_in_bundle = {}
        self.scenario_probability = {}
        self.scenario_data = {}
        self.bundle_scheme = "single_scenario"
        self.bundle_args = {}
        self.json_data = {}
        self.app_data = {}
        self._binary_or_integer_fsv = set()

    def initialize_application(self, *, filename=None, app_data=None, **kwargs):
        if filename is not None:
            with open(f"{filename}", "r") as file:
                self.app_data = json.load(file)
        elif app_data is not None:
            self.app_data = app_data

    def initialize_bundles(
        self, *, filename=None, bundle_data=None, bundle_scheme=None, **kwargs
    ):
        # returns bundles, probabilities, and list of scenarios in each bundle
        if filename is not None:
            with open(f"{filename}", "r") as file:
                self.json_data = json.load(file)
        elif bundle_data is not None:
            self.json_data = bundle_data

        self.scenario_data = {scen["ID"]: scen for scen in self.json_data["scenarios"]}

        if bundle_scheme:
            self.bundle_scheme = bundle_scheme
        self.bundle_args = kwargs
        self.bundles = scentobund.bundle_scheme(
            self.json_data, self.bundle_scheme, self.bundle_args
        )

        for key in self.bundles:
            self.bundle_probability[key] = self.bundles[key]["Probability"]
            self.scenario_probability[key] = self.bundles[key]["Scenario_Probabilities"]
            self.scenarios_in_bundle[key] = self.bundles[key]["IDs"]

        # TODO: Check here at that the scenario probabilities sum to 1.0
        # TODO: Check here at that the bundle probabilities sum to 1.0

    def get_variable_value(self, b, v):
        pass

    def get_variable_name(self, b, v):
        pass

    def fix_variable(self, b, v, value):
        pass

    def shared_variables(self):
        pass

    def set_solver(self, name):
        self.solver = name

    def solve(self, M, *, solver_options=None):
        pass

    def create_subproblem(self, b, *, w=None, x_bar=None, rho=None):
        return self.create_EF(b=b, w=w, x_bar=x_bar, rho=rho)

    def create_EF(self, *, b, w=None, x_bar=None, rho=None):
        pass

    def evaluate(self, x, solver_options=None):
        if solver_options is None:
            solver_options = {}
        obj_value = {}
        M = {}
        #
        # WEH - Should we evaluate w.r.t. scenarios?  How would we reconfigure the sp
        #           object to do that?
        #
        for b in self.bundles:
            M[b] = self.create_subproblem(b)
            for i, xval in enumerate(x):
                self.fix_variable(b, i, xval)
            obj_value[b] = self.solve(M[b], solver_options=solver_options)
            if obj_value[b] is None:
                msg = f"Error evaluating solution for scenario {b}\n\tVariables:\n\t\t"
                tmp = {
                    self.get_variable_name(b, v): self.get_variable_value(b, v)
                    for v in self.shared_variables()
                }
                msg = msg + "\n\t\t".join(
                    f"{var}:\t{tmp[var]}" for var in sorted(tmp.keys())
                )
                logger.debug(msg)
                return munch.Munch(feasible=False, bundle=b)
        obj = sum(self.bundle_probability[b] * obj_value[b] for b in self.bundles)
        # Just need to get one of the bundles to collect the variables
        for b in self.bundles:
            return munch.Munch(
                feasible=True,
                objective=obj,
                variables={
                    self.get_variable_name(b, v): self.get_variable_value(b, v)
                    for v in self.shared_variables()
                },
            )


class StochasticProgram_Pyomo_Base(StochasticProgram):

    def __init__(self):
        super().__init__()
        self.varcuid_to_int = {}
        self.int_to_FirstStageVar = {}
        self.solver_options = {}
        self.pyo_solver = None

    def _first_stage_variables(self, *, M):
        # A generator that yields (name,component) tuples
        pass

    def _initialize_cuid_map(self, *, M, b=None):
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
        # Setup int_to_FirstStageVar
        #
        if b is not None:
            self.int_to_FirstStageVar[b] = {}
            for varname, var in self._first_stage_variables(M=M):
                self.int_to_FirstStageVar[b][
                    self.varcuid_to_int[pyo.ComponentUID(var)]
                ] = var

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
        return self.int_to_FirstStageVar[b][v].name

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
            return None
        else:
            # Load the results into the model so the user can find them there
            M.solutions.load_from(results)
            # Return the value of the 'first' objective
            return list(results.Solution[0].Objective.values())[0]["Value"]


class StochasticProgram_Pyomo_SingleBuilder(StochasticProgram_Pyomo_Base):

    def __init__(self, *, objective=None, first_stage_variables, model_builder):
        super().__init__()
        #
        # A list of string names of variables, such as:
        #   [ "x", "b.y", "b[*].z[*,*]" ]
        #
        self.objective = objective
        self.first_stage_variables = first_stage_variables
        self.model_builder = model_builder
        # self.ctr = 0

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

    def _create_scenario(self, scen):
        return self.model_builder(self.app_data, scen, {})

    def create_EF(self, *, w=None, x_bar=None, rho=None, b):
        scenarios = self.scenarios_in_bundle[b]

        # 1) create scenario dictionary
        scen_dict = {}
        if len(scenarios) > 1:
            for s in scenarios:
                scenario_model = self._create_scenario(self.scenario_data[s])
                self._initialize_cuid_map(M=scenario_model)
                scen_dict[s] = scenario_model
        else:
            s = scenarios[0]
            scenario_model = self._create_scenario(self.scenario_data[s])
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
        obj = sum(self.scenario_probability[b][s] * obj[s].expr for s in scenarios)
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

        # print(f"pyomo1_{self.ctr}.lp")
        # EF_model.write(f"pyomo1_{self.ctr}.lp", io_options={"symbolic_solver_labels":True})
        # self.ctr += 1

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

    def _first_stage_variables(self, *, M):
        for var in find_variables(M):
            yield var.name, var

    def _create_scenario(self, M, S, scen):
        self.model_builder_list[1](M, S, self.app_data, scen, {})
        # for fn in self.model_builder_list[1:]:
        #    fn(M, self.app_data, scen, {})
        # return M

    def create_EF(self, *, w=None, x_bar=None, rho=None, b):
        scenarios = self.scenarios_in_bundle[b]

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
            self._create_scenario(EF_model, EF_model.s[s], self.scenario_data[s])
            obj_comp[s] = find_objective(EF_model.s[s])
            assert (
                obj_comp[s] is not None
            ), f"Cannot find objective on block for scenario '{s}'"
            obj_comp[s].deactivate()

        # 3)Create Obj: root_obj + (sum of scenario obj * probability)
        obj = 0 if root_obj is None else root_obj
        obj = obj + sum(
            self.scenario_probability[b][s] * obj_comp[s].expr for s in scenarios
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


def stochastic_program(
    *,
    model_builder_list=None,
    model_builder=None,
    first_stage_variables=None,
    aml="pyomo",
):
    """
    aml - The modeling framework used to construct the model.

    model_builder_list - A list of functions used to construct the model.

    model_builder - A single function used to construct the model.

    first_stage_variables - A list of strings that denote the first-stage variables in the model.  This is only used if model_builder is specified.
    """
    if model_builder is not None:
        assert (
            model_builder_list is None
        ), "Cannot specify both 'model_builder_list' and 'model_builder' options"
        assert (
            first_stage_variables is not None
        ), "Must specify 'first_stage_variables' with the 'model_builer' option"
    if model_builder_list is not None:
        assert (
            model_builder is None
        ), "Cannot specify both 'model_builder_list' and 'model_builder' options"

    if aml == "pyomo":
        if model_builder is not None:
            return StochasticProgram_Pyomo_SingleBuilder(
                model_builder=model_builder, first_stage_variables=first_stage_variables
            )
        else:
            return StochasticProgram_Pyomo_MultistageBuilder(
                model_builder_list=model_builder_list
            )

    else:
        raise RuntimeError(f"AML {aml} is not currently supported.")
