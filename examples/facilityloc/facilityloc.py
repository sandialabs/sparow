import sys
import argparse
import munch
import pyomo.environ as pyo
from forestlib.sp import stochastic_program
from forestlib.ef import ExtensiveFormSolver
from forestlib.ph import ProgressiveHedgingSolver

#
# Data for facility location problem - do I need to reference the thesis for the app data???
#
app_data = {'n': 4, 's': 11}
app_data['f'] = [44, 49, 42, 34]
app_data['c'] = [
                [34, 25, 6, 15, 44, 8, 2, 3, 44, 49, 3],
                [31, 6, 6, 15, 17, 40, 36, 44, 8, 27, 30],
                [27, 22, 14, 37, 21, 31, 16, 21, 47, 28, 21],
                [39, 44, 32, 31, 40, 4, 22, 38, 34, 36, 3]
                ]

model_data = {
    "LF": {
        "scenarios": [
            {"ID": "1", "Demand": [13, 41, 6, 46, 50, 2, 4, 48, 20, 44, 9], "Probability": 1.0},
        ]
    },
    "HF": {
        "scenarios": [
            {"ID": "1", "Demand": [13, 41, 6, 46, 50, 2, 4, 48, 20, 44, 9], "Probability": 1.0},
        ],
    },
}

def LF_builder(data, args):
    n = data['n']
    s = data['s']
    f = data['f']
    c = data['c']

    ### STOCHASTIC DATA
    d = data['Demand']

    model = pyo.ConcreteModel(data["ID"])

    ### PARAMETERS
    model.N = pyo.Set(initialize=[i for i in range(n)])
    model.S = pyo.Set(initialize=[j for j in range(s)])

    ### VARIABLES
    model.x = pyo.Var(model.N, bounds=[0,1])
    model.z = pyo.Var(model.N, model.S, bounds=[0,1])

    ### CONSTRAINTS
    def MeetDemand_rule(model, j):
        return sum(model.z[i,j] for i in range(n)) == 1
    model.MeetDemand = pyo.Constraint(model.S, rule=MeetDemand_rule)

    def VarLogic_rule(model, i):
        return sum(model.z[i,j] for j in range(s)) <= s*model.x[i]
    model.VarLogic = pyo.Constraint(model.N, rule=VarLogic_rule)

    ### OBJECTIVE
    def Obj_rule(model):
        expr = sum(sum(c[i][j]*d[j]*model.z[i,j] for j in range(s)) for i in range(n))
        expr += sum(f[i]*model.x[i] for i in range(n))
        return expr
    model.obj = pyo.Objective(rule=Obj_rule, sense=pyo.minimize)

    return model

def HF_builder(data, args):
    n = data['n']
    s = data['s']
    f = data['f']
    c = data['c']

    ### STOCHASTIC DATA
    d = data['Demand']

    model = pyo.ConcreteModel(data["ID"])

    ### PARAMETERS
    model.N = pyo.Set(initialize=[i for i in range(n)])
    model.S = pyo.Set(initialize=[j for j in range(s)])

    ### VARIABLES
    model.x = pyo.Var(model.N, bounds=[0,1])
    model.z = pyo.Var(model.N, model.S, bounds=[0,1])

    ### CONSTRAINTS
    def MeetDemand_rule(model, j):
        return sum(model.z[i,j] for i in range(n)) == 1
    model.MeetDemand = pyo.Constraint(model.S, rule=MeetDemand_rule)

    def VarLogic_rule(model, i, j):
        return model.z[i,j] <= model.x[i]
    model.VarLogic = pyo.Constraint(model.N, model.S, rule=VarLogic_rule)

    ### OBJECTIVE
    def Obj_rule(model):
        expr = sum(sum(c[i][j]*d[j]*model.z[i,j] for j in range(s)) for i in range(n))
        expr += sum(f[i]*model.x[i] for i in range(n))
        return expr
    model.obj = pyo.Objective(rule=Obj_rule, sense=pyo.minimize)

    return model

#
# options to solve, LF, HF, or MF models with PH or EF:
#

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
    results.write("results.json", indent=4)
    print("Writing results to 'results.json'")


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
    results.write("results.json", indent=4)
    print("Writing results to 'results.json'")

def HF_PH(*, cache, max_iter, loglevel, finalize_all_iters):
    print("-" * 60)
    print("Running HF_PH")
    print("-" * 60)
    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="HF", model_data=model_data["HF"], model_builder=HF_builder
    )

    solver = ProgressiveHedgingSolver(sp)
    solver.set_options(solver="gurobi", loglevel=loglevel, cached_model_generation=cache, max_iterations=max_iter, finalize_all_xbar=finalize_all_iters, rho_updates=True)
    results = solver.solve(sp)
    results.write("results.json", indent=4)
    print("Writing results to 'results.json'")


def LF_PH(*, cache, max_iter, loglevel, finalize_all_iters):
    print("-" * 60)
    print("Running LF_PH")
    print("-" * 60)
    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="LF", model_data=model_data["LF"], model_builder=LF_builder
    )

    solver = ProgressiveHedgingSolver(sp)
    solver.set_options(solver="gurobi", loglevel=loglevel, cached_model_generation=cache, max_iterations=max_iter, finalize_all_xbar=finalize_all_iters, rho_updates=True)
    results = solver.solve(sp)
    results.write("results.json", indent=4)
    print("Writing results to 'results.json'")


def dist_map(data, models):
    model0 = models[0]
    
    HFscenarios = list(data[model0].keys())
    LFscenarios = {}  # all other models are LF
    for model in models[1:]:
        LFscenarios[model] = list(data[model].keys())

    HFdemands = list(data[model0][HFkey]["d"] for HFkey in HFscenarios)
    LFdemands = list(data[model][ls]["d"] for ls in LFscenarios[model] for model in models[1:])

    # map each LF scenario to closest HF scenario using 1-norm of demand difference
    demand_diffs = {}
    for i in range(len(HFdemands)):
        for j in range(len(LFdemands)):
            demand_diffs[(i,j)] = abs(HFdemands[i] - LFdemands[j])

    return demand_diffs

def MF_PH(*, cache, max_iter, loglevel, finalize_all_iters):
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
        scheme="mf_random", #dissimilar_partitions",
        #distance_function=dist_map,
        LF=2,
        seed=1234567890,
        model_weight={"HF": 2.0, "LF": 1.0},
    )
    #pprint.pprint(sp.get_bundles())
    sp.save_bundles(f"MF_PH_bundle_{bundle_num}.json", indent=4, sort_keys=True)
    
    solver = ProgressiveHedgingSolver(sp)
    solver.set_options(solver="gurobi", loglevel=loglevel, cached_model_generation=cache, max_iterations=max_iter, finalize_all_xbar=finalize_all_iters, rho_updates=True)
    results = solver.solve(sp)
    results.write("results.json", indent=4)
    print("Writing results to 'results.json'")


parser = argparse.ArgumentParser()
parser.add_argument("--lf-ef", action="store_true")
parser.add_argument("--hf-ef", action="store_true")
parser.add_argument("--hf-ph", action="store_true")
parser.add_argument("--lf-ph", action="store_true")
parser.add_argument("--mf-ph", action="store_true")
parser.add_argument("--cache", action="store_true", default=False)
parser.add_argument("-f", "--finalize_all_iterations", action="store_true", default=False)
parser.add_argument("--max-iter", action="store", default=100, type=int)
parser.add_argument("-l", "--loglevel", action="store", default="INFO")
args = parser.parse_args()  # parse sys.argv

if args.lf_ef:
    LF_EF()
elif args.hf_ef:
    HF_EF()
elif args.hf_ph:
    HF_PH(cache=args.cache, max_iter=args.max_iter, loglevel=args.loglevel, finalize_all_iters=args.finalize_all_iterations)
elif args.lf_ph:
    LF_PH(cache=args.cache, max_iter=args.max_iter, loglevel=args.loglevel, finalize_all_iters=args.finalize_all_iterations)
elif args.mf_ph:
    MF_PH(cache=args.cache, max_iter=args.max_iter, loglevel=args.loglevel, finalize_all_iters=args.finalize_all_iterations)
else:
    parser.print_help()
