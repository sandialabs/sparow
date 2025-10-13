"""
Demonstration of solving the LF farmer SP using SNoGloDe
"""
from pprint import pprint
from MFfarmers import LFScenario_dict, LF_model_builder
from forestlib.sp import stochastic_program

class GlobalData:
    num_plots = 1
    num_scens = 3  ### should be >= 3

# list of possible per-plot scenarios for LF model:
LF_scendata = {
    "scenarios": [
        {
            "ID": "scen_0",
            "Yield": {"WHEAT": 2.0, "CORN": 2.4, "SUGAR_BEETS": 16.0},
            "Probability": 0.3,
        },
        {
            "ID": "scen_1",
            "Yield": {"WHEAT": 2.5, "CORN": 3.0, "SUGAR_BEETS": 20.0},
            "Probability": 0.3,
        },
        {
            "ID": "scen_2",
            "Yield": {"WHEAT": 3.0, "CORN": 3.6, "SUGAR_BEETS": 24.0},
            "Probability": 0.4,
        },
    ]
}

lf_scen_dict = LFScenario_dict(LF_scendata)
app_data = {"num_plots": 1, "use_integer": False}
model_data = {"LF": LF_scendata}
sp = stochastic_program(first_stage_variables=["DevotedAcreage[*,*]"])
sp.initialize_application(app_data=app_data)
sp.initialize_model(
    name="LF", model_data=model_data["LF"], model_builder=LF_model_builder
)

if __name__ == "__main__":
    from forestlib.snoglode import SnoglodeSolver
    solver = SnoglodeSolver()
    solutions = solver.solve(sp, solver="gurobi", loglevel="DEBUG")
    pprint(solutions.to_dict())