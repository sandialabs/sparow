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
app_data = {'c': [1, 3, 0.5], 'T': 3}

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


class ScenTree(object):
    def __init__(self, T):
        self.T = T  # number of stages
        self.num_nodes = (2 ^ self.T) - 2  # number of nodes in scen tree

    def ceil_division(self, p, q):
        return -(p // -q)

    # node attributes for scenario tree w/ T stages
    def nodes(self):
        node_attrs = {
            n: {
                "stage": 0,
                "demand": None,
                "parent": None,
                "cond_prob": None,
                "nonants": None,
            }
            for n in range(self.num_nodes)
        }
        node_attrs[0]["stage"] = 1
        node_attrs[0]["demand"] = 1
        node_attrs[0]["cond_prob"] = 1.0
        for n in range(1, self.num_nodes):
            node_attrs[n]["parent"] = self.ceil_division(n, 2) - 1
            node_attrs[n]["cond_prob"] = 0.5
            if n % 2 == 0:
                node_attrs[n]["demand"] = 3
                node_attrs[n]["nonants"] = [n - 1]
            else:
                node_attrs[n]["demand"] = 1
                node_attrs[n]["nonants"] = [n + 1]
        for t in range(1, self.T):
            for n in range(2 ^ t - 1, 2 ^ (t + 1) - 1):
                node_attrs[n]["stage"] = t + 1

        return node_attrs


def LF_builder(data, args):
    c = data['c']
    T = data['T']
    g0 = 0  # initial number of AC units stored

    ### STOCHASTIC DATA
    d = data['Demand']

    model = pyo.ConcreteModel(data["ID"])

    ### PARAMETERS
    model.T = pyo.RangeSet(1, T)

    def acprod_block_rule(b, t):
        ### VARIABLES
        b.x = pyo.Var(bounds=[0, 2])
        b.w = pyo.Var(within=pyo.NonNegativeReals)
        b.y = pyo.Var(within=pyo.NonNegativeReals)
        b.g = pyo.Var(within=pyo.NonNegativeReals)

        ### CONSTRAINTS
        def MeetDemand_rule(b):
            if t == 0:
                return b.g + b.x + b.w - b.y == 1
            else:
                return b.g + b.x + b.w - b.y == d[t - 1]

        b.MeetDemand = pyo.Constraint(rule=MeetDemand_rule)

    model.ab = pyo.Block(model.T, rule=acprod_block_rule)

    def linking_rule(model, t):
        if t == 0:
            return model.ab[t].g == g0
        else:
            return model.ab[t].g == model.ab[t - 1].y

    model.linking_rule = pyo.Constraint(model.T, rule=linking_rule)

    ### OBJECTIVE
    def Obj_rule(model):
        expr = sum(
            c[0] * model.ab[t].x + c[1] * model.ab[t].w + c[2] * model.ab[t].y
            for t in range(T - 1)
        )
        expr += c[0] * model.ab[T - 1].x + c[1] * model.ab[T - 1].w
        return expr

    model.obj = pyo.Objective(rule=Obj_rule, sense=pyo.minimize)

    return model


def HF_builder(data, args):
    c = data['c']
    T = data['T']
    g0 = 0  # initial number of AC units stored

    ### STOCHASTIC DATA
    d = data['Demand']

    model = pyo.ConcreteModel(data["ID"])

    ### PARAMETERS
    model.T = pyo.RangeSet(1, T)

    def acprod_block_rule(b, t):
        ### VARIABLES
        b.x = pyo.Var(bounds=[0, 2], within=pyo.NonNegativeIntegers)
        b.w = pyo.Var(within=pyo.NonNegativeIntegers)
        b.y = pyo.Var(within=pyo.NonNegativeIntegers)
        b.g = pyo.Var(within=pyo.NonNegativeIntegers)

        ### CONSTRAINTS
        def MeetDemand_rule(b):
            if t == 0:
                return b.g + b.x + b.w - b.y == 1
            else:
                return b.g + b.x + b.w - b.y == d[t - 1]

        b.MeetDemand = pyo.Constraint(rule=MeetDemand_rule)

    model.ab = pyo.Block(model.T, rule=acprod_block_rule)

    def linking_rule(model, t):
        if t == 0:
            return model.ab[t].g == g0
        else:
            return model.ab[t].g == model.ab[t - 1].y

    model.linking_rule = pyo.Constraint(model.T, rule=linking_rule)

    ### OBJECTIVE
    def Obj_rule(model):
        expr = sum(
            c[0] * model.ab[t].x + c[1] * model.ab[t].w + c[2] * model.ab[t].y
            for t in range(T - 1)
        )
        expr += c[0] * model.ab[T - 1].x + c[1] * model.ab[T - 1].w
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
    sp = stochastic_program(
        first_stage_variables=["ab[0].x", "ab[0].w", "ab[0].y", "ab[0].g"]
    )
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
    sp = stochastic_program(
        first_stage_variables=["ab[0].x", "ab[0].w", "ab[0].y", "ab[0].g"]
    )
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
    sp = stochastic_program(
        first_stage_variables=["ab[0].x", "ab[0].w", "ab[0].y", "ab[0].g"]
    )
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
    sp = stochastic_program(
        first_stage_variables=["ab[0].x", "ab[0].w", "ab[0].y", "ab[0].g"]
    )
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
    sp = stochastic_program(
        first_stage_variables=["ab[0].x", "ab[0].w", "ab[0].y", "ab[0].g"]
    )
    sp.initialize_application(app_data=app_data)
    sp.initialize_model(
        name="HF", model_data=model_data["HF"], model_builder=HF_builder
    )
    sp.initialize_model(
        name="LF", model_data=model_data["LF"], model_builder=LF_builder, default=False
    )

    bundle_num = 0
    sp.initialize_bundles(
        scheme="mf_random",  # dissimilar_partitions",
        # distance_function=dist_map,
        LF=2,
        seed=1234567890,
        model_weight={"HF": 1.0, "LF": 1.0},
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
