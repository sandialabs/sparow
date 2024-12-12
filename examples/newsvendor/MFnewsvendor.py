import sys
import pprint
import argparse
import munch
import pyomo.environ as pyo
from forestlib.sp import stochastic_program
from forestlib.ef import ExtensiveFormSolver
from forestlib.ph import ProgressiveHedgingSolver


#
# Data for a simple newsvendor example
#
app_data = dict(c=1.0, b=1.5, h=0.1)
model_data = {
    "LF": {
        "scenarios": [
            {"ID": "1", "d": 15},
            {"ID": "2", "d": 60},
            {"ID": "3", "d": 72},
            {"ID": "4", "d": 78},
            {"ID": "5", "d": 82},
        ]
    },
    "HF": {
        "data": {"B": 0.9},
        "scenarios": [
            {"ID": "1", "d": 15, "C": 1.4},
            {"ID": "2", "d": 60, "C": 1.3},
            {"ID": "3", "d": 72, "C": 1.2},
            {"ID": "4", "d": 78, "C": 1.1},
            {"ID": "5", "d": 82, "C": 1.0},
        ],
    },
}


#
# Function that constructs a newsvendor model
# including a single second stage
#
def LF_builder(data, args):
    b = data["b"]
    c = data["c"]
    h = data["h"]
    d = data["d"]

    M = pyo.ConcreteModel(data["ID"])

    M.x = pyo.Var(within=pyo.NonNegativeReals)

    M.y = pyo.Var()
    M.greater = pyo.Constraint(expr=M.y >= (c - b) * M.x + b * d)
    M.less = pyo.Constraint(expr=M.y >= (c + h) * M.x - h * d)

    M.o = pyo.Objective(expr=M.y)

    return M


def HF_builder(data, args):
    b = data["b"]
    B = data["B"]
    c = data["c"]
    C = data["C"]
    h = data["h"]
    d = data["d"]

    M = pyo.ConcreteModel(data["ID"])

    M.x = pyo.Var(within=pyo.NonNegativeReals)

    M.y = pyo.Var()
    M.greater = pyo.Constraint(expr=M.y >= (c - b) * M.x + b * d)
    M.greaterX = pyo.Constraint(expr=M.y >= (C - B) * M.x + B * d)
    M.less = pyo.Constraint(expr=M.y >= (c + h) * M.x - h * d)

    M.o = pyo.Objective(expr=M.y)

    return M


def HF_EF():
    print("-" * 60)
    print("Running HF_EF")
    print("-" * 60)
    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="HF", model_data=model_data["HF"], model_builder=HF_builder
    )

    solver = ExtensiveFormSolver()
    solver.set_options(solver="gurobi")
    results = solver.solve(sp)
    pprint.pprint(munch.unmunchify(results), indent=4, sort_dicts=True)


def LF_EF():
    print("-" * 60)
    print("Running LF_EF")
    print("-" * 60)
    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="LF", model_data=model_data["LF"], model_builder=LF_builder
    )

    solver = ExtensiveFormSolver()
    solver.set_options(solver="gurobi")
    results = solver.solve(sp)
    pprint.pprint(munch.unmunchify(results), indent=4, sort_dicts=True)


def HF_PH():
    print("-" * 60)
    print("Running HF_PH")
    print("-" * 60)
    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="HF", model_data=model_data["HF"], model_builder=HF_builder
    )

    solver = ProgressiveHedgingSolver()
    solver.set_options(solver="gurobi", rho=0.0125, loglevel="INFO")
    results = solver.solve(sp)
    pprint.pprint(munch.unmunchify(results), indent=4, sort_dicts=True)


def LF_PH(*, cache, max_iter):
    print("-" * 60)
    print("Running LF_PH")
    print("-" * 60)
    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="LF", model_data=model_data["LF"], model_builder=LF_builder
    )

    solver = ProgressiveHedgingSolver()
    solver.set_options(solver="gurobi", rho=0.25, loglevel="INFO", cached_model_generation=cache, max_iterations=max_iter)
    results = solver.solve(sp)
    pprint.pprint(munch.unmunchify(results), indent=4, sort_dicts=True)


def MF_PH(*, cache, max_iter):
    print("-" * 60)
    print("Running MF_PH")
    print("-" * 60)
    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="HF", model_data=model_data["HF"], model_builder=HF_builder
    )
    sp.initialize_model(
        name="LF",
        model_data=model_data["LF"],
        model_builder=LF_builder,
        default=False,
    )

    bundle_num = 0
    sp.initialize_bundles(
        scheme="mf_random_nested",
        LF=2,
        seed=1234567890,
        model_weight={"HF": 2.0, "LF": 1.0},
    )
    pprint.pprint(sp.get_bundles())
    sp.save_bundles(f"MF_PH_bundle_{bundle_num}.json", indent=4, sort_keys=True)
    
    solver = ProgressiveHedgingSolver()
    solver.set_options(solver="gurobi", rho=0.25, loglevel="INFO", cached_model_generation=cache, max_iterations=max_iter)
    results = solver.solve(sp)
    pprint.pprint(munch.unmunchify(results), indent=4, sort_dicts=True)


parser = argparse.ArgumentParser()
parser.add_argument("--lf-ef", action="store_true")
parser.add_argument("--hf-ef", action="store_true")
parser.add_argument("--hf-ph", action="store_true")
parser.add_argument("--lf-ph", action="store_true")
parser.add_argument("--mf-ph", action="store_true")
parser.add_argument("--cache", action="store_true", default=False)
parser.add_argument("--max-iter", action="store", default=100, type=int)
args = parser.parse_args()  # parse sys.argv

if args.lf_ef:
    LF_EF()
elif args.hf_ef:
    HF_EF()
elif args.hf_ph:
    HF_PH(cache=args.cache, max_iter=args.max_iter)
elif args.lf_ph:
    LF_PH(cache=args.cache, max_iter=args.max_iter)
elif args.mf_ph:
    MF_PH()
