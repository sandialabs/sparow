"""
Demonstration of solving the LF farmer SP using SNoGloDe
"""

#import snoglode as sno
#from snoglode.utils.solve_stats import OneUpperBoundSolve
#import snoglode.utils.compute as compute
from pprint import pprint

#import pyomo.environ as pyo
#from pyomo.contrib.alternative_solutions.aos_utils import get_active_objective
#from pyomo.opt import TerminationCondition, SolverStatus

#try:
#    from pyomo.contrib.alternative_solutions.solnpool import (
#        PoolCounter,
#        SolutionPool_KeepBest,
#        PoolManager,
#    )
#    from pyomo.contrib.alternative_solutions.solution import Solution, PyomoSolution
#    from pyomo.contrib.alternative_solutions import Objective, Variable
#
#    alt_sol_available = True
#    print("Alternative solutions package from pyomo.contrib is available.")
#except:
#    PoolCounter, SolutionPool_KeepBest, Solution = None, None, None
#    alt_sol_available = False
#    print(
#        "Alternative solutions package from pyomo.contrib is unavailable. \
#          \nReverting to simple saving behavior."
#    )

from MFfarmers import LFScenario_dict, LF_model_builder
from forestlib.sp import stochastic_program
#from typing import Tuple

#import math


class GlobalData:
    num_plots = 1
    num_scens = 3  ### should be >= 3


#
# list of possible per-plot scenarios for LF model:
#
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
    solutions = solver.solve(sp, solver="glpk", loglevel="DEBUG")
    pprint(solutions.to_dict())
    
