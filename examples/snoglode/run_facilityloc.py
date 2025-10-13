"""
Demonstration of solving the LF farmer SP using SNoGloDe
"""
from pprint import pprint
from facilityloc import LF_builder
from forestlib.sp import stochastic_program
import math
import itertools

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

model_data = {"scenarios": LFscens_list}
sp = stochastic_program(first_stage_variables=["x"])
sp.initialize_application(app_data=app_data)
sp.initialize_model(
    name="LF", model_data=model_data, model_builder=LF_builder
)

if __name__ == "__main__":
    from forestlib.snoglode import SnoglodeSolver
    solver = SnoglodeSolver()
    solver.set_options(solver="gurobi")
    solutions = solver.solve(sp, solver="gurobi", loglevel="DEBUG")
    pprint(solutions.to_dict())