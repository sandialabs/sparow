import pyomo.core.base.indexed_component
import pyomo.environ as pyo
#from IPython import embed
import sys
import os

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from forestlib.ph.sp import StochasticProgram_Pyomo

def model_builder(scen,scen_args):
    M = pyo.ConcreteModel(scen["ID"])
    M.x = pyo.Var()
    M.y = pyo.Var()
    if scen["ID"]=='good':
        M.c=pyo.Constraint(expr=1*M.x**2==M.y)
    elif scen["ID"]=='bad':
        M.c=pyo.Constraint(expr=(1*M.x+1)**2==M.y)
    M.obj=pyo.Objective(expr=M.y)
    return M

first_stage_vars=['x','y']
scenarios=['good','bad']
p={'good':0.5,'bad':0.5}

S_EF=StochasticProgram_Pyomo(objective='obj',first_stage_variables=first_stage_vars,model_builder=model_builder)
#S_EF.pyo_solver=pyo.SolverFactory('ipopt')
S_EF.pyo_solver=None
b='bundle_0'
bundle_data={"scenarios": [
    { "ID": 'good',
        "Fidelity": "HF",
        "Demand": 2.1,
        "Weight": 1,
        "Probability": 0.5
    },
    { "ID": 'bad',
        "Fidelity": "HF",
        "Demand": 1.2,
        "Weight": 1,
        "Probability": 0.5
    }]}

S_EF.initialize_bundles(bundle_data=bundle_data, bundle_scheme='single_bundle', fidelity='HF')
EF_model=S_EF.create_EF(b=list(S_EF.scenarios_in_bundle.keys())[0])
res=pyo.SolverFactory('ipopt').solve(EF_model,tee=True)
#embed()
#res=S_EF.solve(EF_model,solver_options={'tee':True})
#tee=solver_options.get('tee',tee)
for b in list(S_EF.scenarios_in_bundle.keys()):
    print(b)
    EF_model=S_EF.create_EF(b=b)
    res=pyo.SolverFactory('ipopt').solve(EF_model,tee=True)

    for s in S_EF.scenarios_in_bundle[b]:
        print(pyo.value(EF_model.s[s].x))


