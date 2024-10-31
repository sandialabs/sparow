import pprint
import pyomo.core.base.indexed_component
import pyomo.environ as pyo
from . import scentobund
import json
import copy

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

    def initialize_bundles(self, *, filename=None, bundle_data=None, bundle_scheme=None, **kwargs):
        # returns bundles, probabilities, and list of scenarios in each bundle
        if filename is not None:
            with open(f'{filename}', 'r') as file:
                self.json_data = json.load(file)
        elif bundle_data is not None:
            self.json_data = bundle_data
   
        self.scenario_data = {scen['ID']:scen for scen in self.json_data['scenarios']}

        if bundle_scheme:
            self.bundle_scheme = bundle_scheme
        self.bundle_args = kwargs
        self.bundles = scentobund.bundle_scheme(self.json_data, self.bundle_scheme, self.bundle_args)
        
        for key in self.bundles:
            self.bundle_probability[key] = self.bundles[key]['Probability']
            self.scenario_probability[key] = self.bundles[key]['Scenario_Probabilities']
            self.scenarios_in_bundle[key] = self.bundles[key]['IDs']

        # TODO: Check here at that the scenario probabilities sum to 1.0
        # TODO: Check here at that the bundle probabilities sum to 1.0
            

    def get_variable_value(self, v, M):
        pass

    def shared_variables(self):
        pass

    def set_solver(self, name):
        self.solver = name

    def solve(self, M, *, solver_options=None):
        pass

    def create_subproblem(self, b, *, w=None, x_bar=None, rho=None):
        M = self.create_EF(b=b, w=w, x_bar=x_bar, rho=rho)
        #self._initialize_varmap(b=b, M=M)
        return M

    def create_EF(self, *, b, w=None, x_bar=None, rho=None):
        pass

            
class StochasticProgram_Pyomo(StochasticProgram):

    def __init__(self, *, objective, first_stage_variables, model_builder):
        StochasticProgram.__init__(self)
        #
        # A list of string names of variables, such as:
        #   [ "x", "b.y", "b[*].z[*,*]" ]
        #
        self.objective = objective
        self.first_stage_variables = first_stage_variables
        self.model_builder=model_builder
        self.varcuid_to_int = {}
        self.int_to_var = {}
        self.solver_options = {}
        self.pyo_solver=None

    def _initialize_cuid_map(self,*,M):
        if len(self.varcuid_to_int) == 0:
            #
            # self.varcuid_to_int maps the cuids for variables to unique integers (starting with 0).
            #   The variable cuids indexed here are specified by the list self.first_stage_variables.
            #
            self.varcuid_to_int = {}
            for varname in self.first_stage_variables:
                cuid = pyo.ComponentUID(varname)
                comp = cuid.find_component_on(M)
                assert comp is not None, "Pyomo error: Unknown variable '%s'" % varname
                if comp.is_indexed():
                #if comp.ctype is pyomo.core.base.indexed_component.IndexedComponent:
                    for var in comp.values():    
                        self.varcuid_to_int[ pyo.ComponentUID(var) ] = len(self.varcuid_to_int)
                elif comp.ctype is pyo.Var:
                    self.varcuid_to_int[ pyo.ComponentUID(comp) ] = len(self.varcuid_to_int)
                else:
                    raise RuntimeError("Pyomo error: Component '%s' is not a variable" % varname)

    # DEPRECATED
    def _initialize_varmap(self, *, b, M):
        
        #
        # self.int_to_var maps the tuple (b,vid) to a Pyomo Var object, where b is a bundle ID and vid
        #   is an integer variable id (0..N-1) that is associated with the variables specified in
        #   self.first_stage_variables.
        #
        self.int_to_var[b] = {}
        if len(b) > 1:
            for cuid,vid in self.varcuid_to_int.items():
                self.int_to_var[b][vid] = cuid.find_component_on(M.rootx[vid])
        else:
            for cuid,vid in self.varcuid_to_int.items():
                self.int_to_var[b][vid] = cuid.find_component_on(EF_model.s[s])

    def create_EF(self, *, w=None, x_bar=None, rho=None, b):
        scenarios = self.scenarios_in_bundle[b]
        
        #1) create scenario dictionary
        scen_dict={}
        for s in scenarios:
            scenario_model=self.create_scenario(self.scenario_data[s])
            scen_dict[s]=scenario_model

        #2) Loop through scenario dictionary, add block, deactivate Obj
        EF_model=pyo.ConcreteModel()
        EF_model.s=pyo.Block(scenarios)
        objective_cuid = pyo.ComponentUID(self.objective)
        obj = {}
        for s, scen_model in scen_dict.items():
            EF_model.s[s].transfer_attributes_from(scen_model)
            obj[s] = objective_cuid.find_component_on(EF_model.s[s])
            assert obj[s] is not None, f"Cannot find objective '{self.objective}' on model for scenario '{s}'"
            obj[s].deactivate()
            
        #2.5) Create first stage variables
        if len(b) > 1:
            EF_model.rootx = pyo.Var(list(self.varcuid_to_int.values()))
            self.int_to_var[b] = {i:EF_model.rootx[i] for i in self.varcuid_to_int.values()}

        #3)Create Obj:sum of scenario obj * probability
        obj = sum(self.scenario_probability[b][s] * obj[s].expr  for s in scenarios)
        if w is not None:
            obj = obj + sum(w[i]*x for i,x in EF_model.rootx.items()) + (rho/2.0) * sum( (x - x_bar[i])**2 for i,x in EF_model.rootx.items())
        EF_model.obj=pyo.Objective(expr=obj)

        #4)Constrain First Stage Variable values to be equal under all scenarios
        if len(b) > 1:
            EF_model.non_ant_cons=pyo.ConstraintList()
        
            for cuid,i in self.varcuid_to_int.items():
                for s in scenarios:
                    var = cuid.find_component_on(EF_model.s[s])
                    assert var is not None, "Pyomo error: Unknown variable '%s' on scenario model '%s'" % (cuid, s)
                    EF_model.non_ant_cons.add(expr=EF_model.rootx[i] == var)
    
        return EF_model
    
    def create_scenario(self, scen):
        # scen: single-element in scenario list:
        #       "low_yield" 
        model= self.model_builder(scen, {})
        self._initialize_cuid_map(M=model)
        return model
    
    def get_variable_value(self, b, v):
        return pyo.value(self.int_to_var[b][v])
            
    def shared_variables(self):
        return list(range(len(self.varcuid_to_int)))

    def solve(self, M, *, solver_options=None, tee=False,solver=None):
        if solver_options:
            self.solver_options = solver_options
        if solver:
            self.solver=solver
        pyo_solver = pyo.SolverFactory(self.solver)
        tee=solver_options.get('tee',tee)
        solver_options_=copy.copy(self.solver_options)
        if 'tee' in solver_options_:
            del solver_options_['tee']
        results = pyo_solver.solve(M, options=solver_options_, tee=tee, load_solutions=False)
        status = results.solver.status
        if not pyo.check_optimal_termination(results):
            condition = results.solver.termination_condition
            raise Exception(
                (
                    "Error solving subproblem '{}': "
                    "SolverStatus = {}, "
                    "TerminationCondition = {}"
                ).format(M.name, status.value, condition.value)
            )
        M.solutions.load_from(results)

