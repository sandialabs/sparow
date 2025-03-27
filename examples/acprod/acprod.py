import sys
import argparse
import munch
import pyomo.environ as pyo
from forestlib.sp import stochastic_program
from forestlib.ef import ExtensiveFormSolver
from forestlib.ph import ProgressiveHedgingSolver 


#
# Data for AC production problem - reference Birge and Louveaux???
#
app_data = {'c': [1, 3, 0.5], 'h': [1, 3, 0], 'T': 3}

model_data = {
    "LF": {
        "scenarios": [
            {"ID": "LowLow", "Demand": [1, 1, 1], "Probability": 0.25},
            {"ID": "LowHigh", "Demand": [1, 1, 3], "Probability": 0.25},
            {"ID": "HighLow", "Demand": [1, 3, 1], "Probability": 0.25},
            {"ID": "HighHigh", "Demand": [1, 3, 3], "Probability": 0.25},
        ]
    },
    "HF": {
        "scenarios": [
            {"ID": "LowLow", "Demand": [1, 1, 1], "Probability": 0.25},
            {"ID": "LowHigh", "Demand": [1, 1, 3], "Probability": 0.25},
            {"ID": "HighLow", "Demand": [1, 3, 1], "Probability": 0.25},
            {"ID": "HighHigh", "Demand": [1, 3, 3], "Probability": 0.25},
        ]
    },
}

def LF_builder(data, args):
    c = data['c']
    h = data['h']
    T = data['T']

    ### STOCHASTIC DATA
    d = data['Demand']

    model = pyo.ConcreteModel(data["ID"])

    ### PARAMETERS
    model.T = pyo.Set(initialize=[i for i in range(T)])

    ### VARIABLES
    model.x = pyo.Var(model.T, bounds=[0,2])
    model.w = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.y = pyo.Var(model.T, within=pyo.NonNegativeReals)

    ### CONSTRAINTS
    def MeetInitDemand_rule(model):
        return model.x[0] + model.w[0] - model.y[0] == 1 
    model.MeetInitDemand = pyo.Constraint(rule=MeetInitDemand_rule)

    def MeetDemand_rule(model, t):
        return model.y[t-1] + model.x[t] + model.w[t] - model.y[t] == d[t]
    model.MeetDemand = pyo.Constraint(pyo.Set(initialize=[i for i in range(1, T)]), rule=MeetDemand_rule)

    ### OBJECTIVE
    def Obj_rule(model):
        expr = c[0]*model.x[0] + c[1]*model.w[0] + c[2]*model.y[0]
        expr += sum(c[0]*model.x[t] + c[1]*model.w[t] + c[2]*model.y[t] for t in range(1,T-1))
        expr += h[0]*model.x[T-1] + h[1]*model.w[T-1] + h[2]*model.y[T-1]
        return expr
    model.obj = pyo.Objective(rule=Obj_rule, sense=pyo.minimize)

    return model

def HF_builder(data, args):
    c = data['c']
    h = data['h']
    T = data['T']

    ### STOCHASTIC DATA
    d = data['Demand']

    model = pyo.ConcreteModel(data["ID"])

    ### PARAMETERS
    model.T = pyo.Set(initialize=[i for i in range(T)])

    ### VARIABLES
    model.x = pyo.Var(model.T, within=pyo.NonNegativeIntegers, bounds=[0,2])
    model.w = pyo.Var(model.T, within=pyo.NonNegativeIntegers)
    model.y = pyo.Var(model.T, within=pyo.NonNegativeIntegers)

    ### CONSTRAINTS
    def MeetInitDemand_rule(model):
        return model.x[0] + model.w[0] - model.y[0] == 1 
    model.MeetInitDemand = pyo.Constraint(rule=MeetInitDemand_rule)

    def MeetDemand_rule(model, t):
        return model.y[t-1] + model.x[t] + model.w[t] - model.y[t] == d[t]
    model.MeetDemand = pyo.Constraint(pyo.Set(initialize=[i for i in range(1, T)]), rule=MeetDemand_rule)

    ### OBJECTIVE
    def Obj_rule(model):
        expr = c[0]*model.x[0] + c[1]*model.w[0] + c[2]*model.y[0]
        expr += sum(c[0]*model.x[t] + c[1]*model.w[t] + c[2]*model.y[t] for t in range(1,T-1))
        expr += h[0]*model.x[T-1] + h[1]*model.w[T-1] + h[2]*model.y[T-1]
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
    sp = stochastic_program(first_stage_variables=["x[0]", "w[0]", "y[0]"])
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
    sp = stochastic_program(first_stage_variables=["x[0]", "w[0]", "y[0]"])
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
    sp = stochastic_program(first_stage_variables=["x[0]", "w[0]", "y[0]"])
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
    sp = stochastic_program(first_stage_variables=["x[0]", "w[0]", "y[0]"])
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
    sp = stochastic_program(first_stage_variables=["x[0]", "w[0]", "y[0]"])
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
