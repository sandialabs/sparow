import pyomo.core.base.indexed_component
import pyomo.environ as pyo
from sp import StochasticProgram_Pyomo

class Stochastic_Working_EF(StochasticProgram_Pyomo):

    def __init__(self, *, first_stage_variables):
        
        #
        # A list of string names of variables, such as:
        #   [ "x", "b.y", "b[*].z[*,*]" ]
        #
        self.first_stage_variables = first_stage_variables
        StochasticProgram_Pyomo.__init__(self,first_stage_variables=self.first_stage_variables)
        self.varcuid_to_int = []
        self.int_to_var = {}
        self.solver_options = {}
        self.pyo_solver = pyo.SolverFactory('ipopt')
    
    def create_EF(self,scenarios,p):
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
        model= self.model_builder(scen)
        return model
    def model_builder(self,scen):
        M = pyo.ConcreteModel()
        M.x = pyo.Var()
        M.y = pyo.Var()
        if scen=='good':
            M.c=pyo.Constraint(expr=1*M.x**2==M.y)
        elif scen=='bad':
            M.c=pyo.Constraint(expr=(1*M.x+1)**2==M.y)
        M.obj=pyo.Objective(expr=M.y)
        return M

first_stage_vars=['x','y']
scenarios=['good','bad']
p={'good':0.5,'bad':0.5}
S_EF=Stochastic_Working_EF(first_stage_variables=first_stage_vars)
EF_model=S_EF.create_EF(scenarios,p)
res=S_EF.solve(EF_model,tee=True)
bundles={'bundle_0':{'ID':['good','bad'],'Probability':[0.5,0.5]},'bundle_1':{'ID':['good','bad'],'Probability':[0.5,0.5]}}
for b in bundles.keys():
    S_EF=Stochastic_Working_EF(first_stage_variables=first_stage_vars)
    print(b)
    EF_model=S_EF.create_EF(bundles[b]['ID'],dict(zip(bundles[b]['ID'],bundles[b]['Probability'])))
    res=S_EF.solve(EF_model,tee=True)