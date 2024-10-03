import pyomo.core.base.indexed_component
import pyomo.environ as pyo
from IPython import embed
import sys
import os

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from forestlib.ph.sp import StochasticProgram_Pyomo

def model_builder(scen):
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

S_EF=StochasticProgram_Pyomo(first_stage_variables=first_stage_vars,model_builder=model_builder)
S_EF.pyo_solver=pyo.SolverFactory('ipopt')
b='bundle_0'
EF_model=S_EF.create_EF(scenarios=scenarios,p=p,b=b)
S_EF.initialize_varmap(b=b,M=EF_model)
#embed()
res=S_EF.solve(EF_model,tee=True)
bundles={'bundle_0':{'ID':['good','bad'],'Probability':[0.5,0.5]},'bundle_1':{'ID':['good','bad'],'Probability':[0.5,0.5]}}
for b in bundles.keys():
    S_EF=StochasticProgram_Pyomo(first_stage_variables=first_stage_vars,model_builder=model_builder)
    S_EF.pyo_solver=pyo.SolverFactory('ipopt')
    print(b)
    EF_model=S_EF.create_EF(scenarios=bundles[b]['ID'],p=dict(zip(bundles[b]['ID'],bundles[b]['Probability'])),b=b)
    S_EF.initialize_varmap(b=b,M=EF_model)
    res=S_EF.solve(EF_model,tee=True)
