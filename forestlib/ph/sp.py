import pyomo.core.base.indexed_component
import pyomo.environ as pyo
from . import scentobund

class StochasticProgram(object):

    def __init__(self):
        self.solver = "gurobi"
        self.bundles = {}
        self.bundle_probability = {}
        self.scenarios_in_bundle = {}

    def initialize_bundles(self, args): # need to think about user args
        # returns bundles, probabilities, and list of scenarios in each bundle
        self.bundles = scentobund.bundle_scheme(scentobund.data, scentobund.scheme)
        
        for key in self.bundles:
            self.bundle_probability[key] = self.bundles[key]['Probability']
            self.scenarios_in_bundle[key] = self.bundles[key]['IDs']
            
        return self.bundles, self.bundle_probability, self.scenarios_in_bundle

    #def bundle_probability(self, bund_name):
    #    return self.bundles[bund_name]['Probability']


    def initialize_varmap(self, *, b, M):
        pass

    def get_variable_value(self, v, M):
        pass

    def shared_variables(self):
        pass

    def solve(self, M, *, solver_options=None):
        pass

    def create_subproblem(self, * b, w=None, x_bar=None, rho=None):
        M = self.create_EF(b=b, w=w, x_bar=x_bar, rho=rho)
        self.initialize_varmap(b=b, M=M)
        return M

    def create_EF(self, * b, w=None, x_bar=None, rho=None):
        pass

            
class StochasticProgram_Pyomo(StochasticProgram):

    def __init__(self, *, first_stage_variables,model_builder):
        StochasticProgram.__init__(self)
        #
        # A list of string names of variables, such as:
        #   [ "x", "b.y", "b[*].z[*,*]" ]
        #
        self.first_stage_variables = first_stage_variables
        self.model_builder=model_builder
        self.varcuid_to_int = []
        self.int_to_var = {}
        self.solver_options = {}
        self.pyo_solver=None
        
    def initialize_varmap(self, *, b, M):
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
                if comp.ctype is pyomo.core.base.indexed_component.IndexedComponent:
                    for var in comp.values():    
                        self.varcuid_to_int[ pyo.ComponentUID(var) ] = len(self.varcuid_to_int)
                elif comp.ctype is pyo.Var:
                    self.varcuid_to_int[ pyo.ComponentUID(comp) ] = len(self.varcuid_to_int)
                else:
                    raise RuntimeError("Pyomo error: Component '%s' is not a variable" % varname)

        #
        # self.int_to_var maps the tuple (b,vid) to a Pyomo Var object, where b is a bundle ID and vid
        #   is an integer variable id (0..N-1) that is associated with the variables specified in
        #   self.first_stage_variables.
        #
        self.int_to_var[b] = {}
        for cuid,vid in self.varcuid_to_int.items():
            self.int_to_var[b][vid] = cuid.find_component_on(M)

    def create_EF(self,scenarios,p):
        #
        # scenarios: A list of string names of scenario IDs, such as:
        #       [ "high_yield", "low_yield" ]
        # p: A dict mapping scenario IDs to their probability, such as:
        #       {'high_yield':0.5,'low_yield':0.5} 
        
        #1) create scenario dictionary
        scen_dict={}
        for scen in scenarios:
            scenario_model=self.create_scenario(scen)
            scen_dict[scen]=scenario_model
        #2) Loop through scenario dictionary, add block, deactivate Obj
        EF_model=pyo.ConcreteModel()
        EF_model.s=pyo.Block(scenarios)
        for scen in scen_dict.keys():
            EF_model.s[scen].transfer_attributes_from(scen_dict[scen])
            EF_model.s[scen].obj.deactivate()
        #3)Create Obj:sum of scenario obj * probability
        EF_model.obj=pyo.Objective(expr=sum(p[s]*EF_model.s[s].obj.expr  for s in scenarios))
        #4)Create First Stage Variables, Constrain value to be equal under all scenarios
        EF_model.non_ant_cons=pyo.ConstraintList()
        for x in self.first_stage_variables:
            EF_model.add_component(x, pyo.Var())
            for s in scenarios:
                EF_model.non_ant_cons.add(expr=EF_model.find_component(x)==EF_model.s[s].find_component(x))
    
        return EF_model
    
    def create_scenario(self,scen):
        # scen: single-element in scenario list:
        #       "low_yield" 
        model= self.model_builder(scen)
        return model
    
    def get_variable_value(self, b, v):
        return pyo.value(self.int_to_var[b][v])
            
    def shared_variables(self):
        return list(range(len(self.varcuid_to_int)))

    def solve(self, M, *, solver_options=None, tee=False):
        if solver_options:
            self.solver_options = solver_options
        if self.pyo_solver is None:
            self.pyo_solver = pyo.SolverFactory(self.solver)

        res = self.pyo_solver.solve(M, options=self.solver_options, tee=tee)
        return res
