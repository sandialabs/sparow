import sys
import argparse
import munch
import pyomo.environ as pyo
import itertools
import math
import random
from forestlib.sp import stochastic_program
from forestlib.ef import ExtensiveFormSolver
from forestlib.ph import ProgressiveHedgingSolver

'''
UNCAPACITATED FACILITY LOCATION
    - LF model is an approximation using continuous variables and big-M constraints
    - HF model:
        - disaggregates logical constraints
        - is continuous, but a perfect formulation (i.e., obtains integer optimal solution)
Problem data adapted from https://ampl.com/colab/notebooks/ampl-development-tutorial-26-stochastic-capacitated-facility-location-problem.html#problem-description
'''

app_data = {"n": 3, "s": 4}
app_data["f"] = [400000, 200000, 600000]
app_data["c"] = [
    [5739.725, 6539.725, 8650.40, 22372.1125],
    [6055.05, 6739.055, 8050.40, 21014.225],
    [8650.40, 7539.055, 4539.72, 15024.325],
]
app_data["k"] = [1550, 650, 1750]

# Define the customer demand data
customer_demand = {
    "San_Antonio_TX": [450, 650, 887],
    "Dallas_TX": [910, 1134, 1456],
    "Jackson_MS": [379, 416, 673],
    "Birmingham_AL": [91, 113, 207],
}
demand_levels = ["Low", "Medium", "High"]
cities = list(customer_demand.keys())
all_scenarios = list(itertools.product(demand_levels, repeat=len(cities)))

# mapping of demand levels to their corresponding values
demand_value_mapping = {"Low": 0, "Medium": 1, "High": 2}
# mapping of demand levels to their corresponding probabilities
demand_prob_mapping = {"Low": 0.25, "Medium": 0.5, "High": 0.25}

sdict = {}
for scenario in all_scenarios:
    scenario_dict = dict(zip(cities, scenario))
    sdict[scenario] = {
        "Demand": {
            city: customer_demand[city][demand_value_mapping[demand]]
            for city, demand in scenario_dict.items()
        },
        "Probability": math.prod(
            demand_prob_mapping[scenario_dict[city]] for city in scenario_dict.keys()
        ),
    }

# Each scenario is randomly assigned LF or HF
random.seed(58564564871312356)
LF_scenarios = random.choices(list(sdict.keys()), k=40)
HF_scenarios = [s for s in sdict.keys() if s not in LF_scenarios]

LFscens_list = []
for lscen in LF_scenarios:
    LFscens_list.append(
        {
            "ID": f"{scenario}",
            "Demand": list(sdict[scenario]["Demand"].values()),
            "Probability": sdict[scenario]["Probability"],
        }
    )

# normalize LF scenario probabilities
LF_norm_term = sum(LFscens_list[s_idx]["Probability"] for s_idx in range(len(LFscens_list)))
for s_idx in range(len(LFscens_list)):
    LFscens_list[s_idx]["Probability"] /= LF_norm_term

HFscens_list = []
for hscen in HF_scenarios:
    HFscens_list.append(
        {
            "ID": f"{scenario}",
            "Demand": list(sdict[scenario]["Demand"].values()),
            "Probability": sdict[scenario]["Probability"],
        }
    )

# normalize LF scenario probabilities
HF_norm_term = sum(HFscens_list[s_idx]["Probability"] for s_idx in range(len(HFscens_list)))
for s_idx in range(len(HFscens_list)):
    HFscens_list[s_idx]["Probability"] /= HF_norm_term

model_data = {"LF": {"scenarios": LFscens_list}, "HF": {"scenarios": HFscens_list}}

def LF_builder(data, args):
    n = data["n"]
    s = data["s"]
    f = data["f"]
    c = data["c"]
    k = data["k"]

    ### STOCHASTIC DATA
    d = data["Demand"]

    model = pyo.ConcreteModel(data["ID"])

    ### PARAMETERS
    model.N = pyo.Set(initialize=[i for i in range(n)])
    model.S = pyo.Set(initialize=[j for j in range(s)])

    ### VARIABLES
    model.x = pyo.Var(model.N, bounds=[0, 1])
    model.z = pyo.Var(model.N, model.S, bounds=[0, 1])

    ### CONSTRAINTS
    def MeetDemand_rule(model, j):
        return sum(model.z[i, j] for i in range(n)) == 1

    model.MeetDemand = pyo.Constraint(model.S, rule=MeetDemand_rule)

    def VarLogic_rule(model, i):
        return sum(model.z[i, j] for j in range(s)) <= s * model.x[i]

    model.VarLogic = pyo.Constraint(model.N, rule=VarLogic_rule)

    #def Capacity_rule(model, i): # remove this constraint for a lower-fidelity model
    #    return sum(model.z[i,j] for j in range(s)) <= k[i] * model.x[i]

    #model.Capacity = pyo.Constraint(model.N, rule=Capacity_rule)

    ### OBJECTIVE
    def Obj_rule(model):
        expr = sum(
            sum(c[i][j] * d[j] * model.z[i, j] for j in range(s)) for i in range(n)
        )
        expr += sum(f[i] * model.x[i] for i in range(n))
        return expr

    model.obj = pyo.Objective(rule=Obj_rule, sense=pyo.minimize)

    return model


def HF_builder(data, args):
    n = data["n"]
    s = data["s"]
    f = data["f"]
    c = data["c"]
    k = data["k"]

    ### STOCHASTIC DATA
    d = data["Demand"]

    model = pyo.ConcreteModel(data["ID"])

    ### PARAMETERS
    model.N = pyo.Set(initialize=[i for i in range(n)])
    model.S = pyo.Set(initialize=[j for j in range(s)])

    ### VARIABLES
    model.x = pyo.Var(model.N, bounds=[0, 1]) # x[i] == 1 if facility i is open
    model.z = pyo.Var(model.N, model.S, bounds=[0, 1]) # z[i, j] is proportion of customer j's demand met by facility i

    ### CONSTRAINTS
    def MeetDemand_rule(model, j):
        return sum(model.z[i, j] for i in range(n)) == 1

    model.MeetDemand = pyo.Constraint(model.S, rule=MeetDemand_rule)

    def VarLogic_rule(model, i, j):
        return model.z[i, j] <= model.x[i]

    model.VarLogic = pyo.Constraint(model.N, model.S, rule=VarLogic_rule)

    #def Capacity_rule(model, i):
    #    return sum(model.z[i,j] for j in range(s)) <= k[i] * model.x[i]

    #model.Capacity = pyo.Constraint(model.N, rule=Capacity_rule)

    ### OBJECTIVE
    def Obj_rule(model):
        expr = sum(
            sum(c[i][j] * d[j] * model.z[i, j] for j in range(s)) for i in range(n)
        )
        expr += sum(f[i] * model.x[i] for i in range(n))
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
    solver.set_options(
        solver="gurobi",
        loglevel=loglevel,
        cached_model_generation=cache,
        max_iterations=max_iter,
        finalize_all_xbar=finalize_all_iters,
        rho_updates=True,
    )
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
    solver.set_options(
        solver="gurobi",
        loglevel=loglevel,
        cached_model_generation=cache,
        max_iterations=max_iter,
        finalize_all_xbar=finalize_all_iters,
        rho_updates=True,
    )
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
    LFdemands = list(
        data[model][ls]["d"] for ls in LFscenarios[model] for model in models[1:]
    )

    # map each LF scenario to closest HF scenario using 1-norm of demand difference
    demand_diffs = {}
    for i in range(len(HFdemands)):
        for j in range(len(LFdemands)):
            demand_diffs[(i, j)] = abs(HFdemands[i] - LFdemands[j])

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
        scheme="bundle_random_partition",
        # distance_function=dist_map,
        LF=2,
        seed=1234567890,
        num_buns=2,
        # model_weight={"HF": 2.0, "LF": 1.0},
    )
    # pprint.pprint(sp.get_bundles())
    sp.save_bundles(f"MF_PH_bundle_{bundle_num}.json", indent=4, sort_keys=True)

    solver = ProgressiveHedgingSolver(sp)
    solver.set_options(
        solver="gurobi",
        loglevel=loglevel,
        cached_model_generation=cache,
        max_iterations=max_iter,
        finalize_all_xbar=finalize_all_iters,
        rho_updates=True,
    )
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
parser.add_argument(
    "-f", "--finalize_all_iterations", action="store_true", default=False
)
parser.add_argument("--max-iter", action="store", default=100, type=int)
parser.add_argument("-l", "--loglevel", action="store", default="INFO")
args = parser.parse_args()  # parse sys.argv

if args.lf_ef:
    LF_EF()
elif args.hf_ef:
    HF_EF()
elif args.hf_ph:
    HF_PH(
        cache=args.cache,
        max_iter=args.max_iter,
        loglevel=args.loglevel,
        finalize_all_iters=args.finalize_all_iterations,
    )
elif args.lf_ph:
    LF_PH(
        cache=args.cache,
        max_iter=args.max_iter,
        loglevel=args.loglevel,
        finalize_all_iters=args.finalize_all_iterations,
    )
elif args.mf_ph:
    MF_PH(
        cache=args.cache,
        max_iter=args.max_iter,
        loglevel=args.loglevel,
        finalize_all_iters=args.finalize_all_iterations,
    )
else:
    parser.print_help()
