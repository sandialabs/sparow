import pyomo.core.base.indexed_component
import pyomo.environ as pyo
from IPython import embed
import sys
import os
from forestlib.ph import ProgressiveHedgingSolver
from forestlib.ph import stochastic_program
from mpisppy.opt.ef import ExtensiveForm
from mpisppy.opt.ph import PH
import mpisppy.utils.sputils as sputils


def model_builder(app_dta, scen, scen_args):
    M = pyo.ConcreteModel(scen["ID"])
    M.x = pyo.Var()
    M.y = pyo.Var()
    if scen["ID"] == "good":
        M.c = pyo.Constraint(expr=1 * M.x**2 == M.y)
    elif scen["ID"] == "bad":
        M.c = pyo.Constraint(expr=(1 * M.x + 1) ** 2 == M.y)
    M.obj = pyo.Objective(expr=M.y)
    return M


def model_builder_mpi(scen, scen_args):
    M = pyo.ConcreteModel(scen["ID"])
    M.x = pyo.Var()
    M.y = pyo.Var()
    if scen["ID"] == "good":
        M.c = pyo.Constraint(expr=1 * M.x**2 == M.y)
    elif scen["ID"] == "bad":
        M.c = pyo.Constraint(expr=(1 * M.x + 1) ** 2 == M.y)
    M.obj = pyo.Objective(expr=M.y)
    return M


first_stage_vars = ["x", "y"]
scenarios = ["good", "bad"]
p = {"good": 0.5, "bad": 0.5}

S_EF = stochastic_program(
    first_stage_variables=first_stage_vars, model_builder=model_builder
)

b = "bundle_0"
bundle_data = {
    "scenarios": [
        {
            "ID": "good",
            "Fidelity": "HF",
            "Demand": 2.1,
            "Weight": 1,
            "Probability": 0.5,
        },
        {"ID": "bad", "Fidelity": "HF", "Demand": 1.2, "Weight": 1, "Probability": 0.5},
    ]
}

S_EF.initialize_bundles(
    bundle_data=bundle_data, bundle_scheme="single_bundle", fidelity="HF"
)
EF_model = S_EF.create_EF(b=list(S_EF.scenarios_in_bundle.keys())[0])


for b in list(S_EF.scenarios_in_bundle.keys()):
    print(b)
    EF_model = S_EF.create_EF(b=b)
    # res=pyo.SolverFactory('ipopt').solve(EF_model,tee=True)

    for s in S_EF.scenarios_in_bundle[b]:
        break
        # print(pyo.value(EF_model.s[s].x))


FarmerSP = stochastic_program(
    first_stage_variables=first_stage_vars, model_builder=model_builder
)
FarmerSP.initialize_bundles(bundle_data=bundle_data, bundle_scheme="single_scenario")
ph = ProgressiveHedgingSolver()
ph.solve(FarmerSP, max_iterations=2, solver="ipopt", loglevel="DEBUG")


def scenario_creator(scenario_name):
    if scenario_name == "good":
        model = model_builder_mpi({"ID": "good"}, {})
    elif scenario_name == "bad":
        model = model_builder_mpi({"ID": "bad"}, {})
    else:
        raise ValueError("Unrecognized scenario name")

    sputils.attach_root_node(model, model.obj, [model.x, model.y])
    model._mpisppy_probability = 1.0 / 2
    return model


options = {
    "solver_name": "ipopt",
    "PHIterLimit": 2,
    "defaultPHrho": 1.5,
    "convthresh": 1e-3,
    "verbose": True,
    "display_progress": False,
    "display_timing": False,
    "iter0_solver_options": dict(),
    "iterk_solver_options": dict(),
}
all_scenario_names = ["good", "bad"]
ph = PH(
    options,
    all_scenario_names,
    scenario_creator,
)

results = ph.ph_main()
