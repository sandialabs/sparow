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

"""
CAPACITATED FACILITY LOCATION
    - LF model is an approximation using continuous variables
        - LF scenarios are restricted to Low, Medium, and High
    - HF model is a MIP (first-stage binary variables, second-stage continuous variables)
        - HF scenarios can be Low, Medium, and High for each customer (e.g., ['Low', 'High', 'Low', 'Medium'])
* Can specify number of HF scenarios by including "num_HF" key in app_data; otherwise, defaults to 8
* Problem data adapted from https://ampl.com/colab/notebooks/ampl-development-tutorial-26-stochastic-capacitated-facility-location-problem.html#problem-description
"""

app_data = {"n": 3, "t": 4}  # number of facilities & customers
app_data["f"] = [400000, 200000, 600000]  # fixed costs for opening facilities
app_data["c"] = [
    [5739.725, 6539.725, 8650.40, 22372.1125],
    [6055.05, 6739.055, 8050.40, 21014.225],
    [8650.40, 7539.055, 4539.72, 15024.325],
]  # servicing costs
app_data["k"] = [1550, 650, 1750]  # facility capacity

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

sdict = {}  # dictionary of all possible Low/Medium/High combinations for HF scenarios
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

LF_scenarios = [
    ("Low", "Low", "Low", "Low"),
    ("Medium", "Medium", "Medium", "Medium"),
    ("High", "High", "High", "High"),
]
random.seed(58564564871312356)
HF_scenarios = random.choices(
    list(sdict.keys()), k=app_data.get("num_HF", 8)
)  # randomly select HF scenarios from sdict

LFscens_list = []  # list of LF scenarios
for lscen_idx, lscen in enumerate(LF_scenarios):
    scen = demand_levels[lscen_idx]
    LFscens_list.append(
        {
            "ID": f"{scen}",
            "Demand": [
                customer_demand[key][demand_value_mapping[scen]]
                for key in customer_demand.keys()
            ],
            "Probability": demand_prob_mapping[scen],
        }
    )

HFscens_list = []  # list of HF scenarios
customer_demand_vals = list(customer_demand.values())
for hscen in HF_scenarios:
    HFscens_list.append(
        {
            "ID": f"{hscen}",
            "Demand": [
                customer_demand_vals[customer][demand_value_mapping[hscen[customer]]]
                for customer in range(len(hscen))
            ],
            "Probability": sdict[scenario]["Probability"],
        }
    )

# normalize HF scenario probabilities
HF_norm_term = sum(
    HFscens_list[s_idx]["Probability"] for s_idx in range(len(HFscens_list))
)
for s_idx in range(len(HFscens_list)):
    HFscens_list[s_idx]["Probability"] /= HF_norm_term


model_data = {"LF": {"scenarios": LFscens_list}, "HF": {"scenarios": HFscens_list}}


def LF_builder(data, args):
    n = data["n"]
    t = data["t"]
    f = data["f"]
    c = data["c"]
    k = data["k"]

    ### STOCHASTIC DATA
    d = data["Demand"]

    model = pyo.ConcreteModel(data["ID"])

    ### PARAMETERS
    model.N = pyo.Set(initialize=[i for i in range(n)])
    model.T = pyo.Set(initialize=[j for j in range(t)])

    ### VARIABLES
    model.x = pyo.Var(model.N, bounds=[0, 1])  # x[i] == 1 if facility i is open
    model.z = pyo.Var(
        model.N, model.T, within=pyo.NonNegativeReals
    )  # z[i, j] = proportion of customer j's demand met by facility i

    ### CONSTRAINTS
    def MeetDemand_rule(model, j):
        return sum(model.z[i, j] for i in range(n)) >= d[j]

    model.MeetDemand = pyo.Constraint(model.T, rule=MeetDemand_rule)

    def SufficientProduction_rule(model):
        return sum(k[i] * model.x[i] for i in range(n)) >= sum(d[j] for j in range(t))

    model.SufficientProduction = pyo.Constraint(rule=SufficientProduction_rule)

    def Capacity_rule(model, i):  # note this constraint also ensures logic between x, z
        return sum(model.z[i, j] for j in range(t)) <= k[i] * model.x[i]

    model.Capacity = pyo.Constraint(model.N, rule=Capacity_rule)

    ### OBJECTIVE
    def Obj_rule(model):
        expr = sum(sum(c[i][j] * model.z[i, j] for j in range(t)) for i in range(n))
        expr += sum(f[i] * model.x[i] for i in range(n))
        return expr

    model.obj = pyo.Objective(rule=Obj_rule, sense=pyo.minimize)

    return model


def HF_builder(data, args):
    n = data["n"]
    t = data["t"]
    f = data["f"]
    c = data["c"]
    k = data["k"]

    ### STOCHASTIC DATA
    d = data["Demand"]

    model = pyo.ConcreteModel(data["ID"])

    ### PARAMETERS
    model.N = pyo.Set(initialize=[i for i in range(n)])
    model.T = pyo.Set(initialize=[j for j in range(t)])

    ### VARIABLES
    model.x = pyo.Var(model.N, within=pyo.Binary)  # x[i] == 1 if facility i is open
    model.z = pyo.Var(
        model.N, model.T, within=pyo.NonNegativeReals
    )  # z[i, j] = proportion of customer j's demand met by facility i

    ### CONSTRAINTS
    def MeetDemand_rule(model, j):
        return sum(model.z[i, j] for i in range(n)) >= d[j]

    model.MeetDemand = pyo.Constraint(model.T, rule=MeetDemand_rule)

    def SufficientProduction_rule(model):
        return sum(k[i] * model.x[i] for i in range(n)) >= sum(d[j] for j in range(t))

    model.SufficientProduction = pyo.Constraint(rule=SufficientProduction_rule)

    def Capacity_rule(model, i):  # note this constraint also ensures logic between x, z
        return sum(model.z[i, j] for j in range(t)) <= k[i] * model.x[i]

    model.Capacity = pyo.Constraint(model.N, rule=Capacity_rule)

    ### OBJECTIVE
    def Obj_rule(model):
        expr = sum(sum(c[i][j] * model.z[i, j] for j in range(t)) for i in range(n))
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

    solver = ProgressiveHedgingSolver()
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

    solver = ProgressiveHedgingSolver()
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
        name="LF", model_data=model_data["LF"], model_builder=LF_builder, default=False
    )

    bundle_num = 0
    sp.initialize_bundles(
        scheme="mf_random",
        # distance_function=dist_map,
        LF=2,
        seed=1234567890,
        num_buns=2,
        # model_weight={"HF": 2.0, "LF": 1.0},
    )
    # pprint.pprint(sp.get_bundles())
    sp.save_bundles(f"MF_PH_bundle_{bundle_num}.json", indent=4, sort_keys=True)

    solver = ProgressiveHedgingSolver()
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
