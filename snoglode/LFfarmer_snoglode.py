"""
Demonstration of solving the LF farmer SP using SNoGloDe
"""

import snoglode as sno
import snoglode.utils.compute as compute

import pyomo.environ as pyo
from pyomo.contrib.alternative_solutions.aos_utils import get_active_objective

from MFfarmers import LFScenario_dict, HFScenario_dict, LF_model_builder
from forestlib.sp import stochastic_program
from typing import Tuple

import math

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
app_data = {"num_plots": 1,
            "use_integer": False}
model_data = {"LF": LF_scendata}
sp = stochastic_program(first_stage_variables=["DevotedAcreage[*,*]"])
sp.initialize_application(app_data=app_data)
sp.initialize_model(
    name="LF", 
    model_data=model_data["LF"], 
    model_builder=LF_model_builder
)

class CustomCandidateGenerator(sno.AbstractCandidateGenerator):
    """
    Here, we want to just mimic the logic of the sno.AverageLowerBoundSolution
    candidate generator, with an extra step of saving candidate solutions
    as we traverse the tree.

    Note: this saving is happening *before we actually solve for the UB,
    so we do not have a good concept of what the "best" or how close to optimal
    any solution is; this is for demonstrative purposes. 
    """
    def __init__(self, 
                 solver, 
                 subproblems: sno.Subproblems, 
                 time_ub: int) -> None:
        
        super().__init__(solver = solver, 
                         subproblems = subproblems, 
                         time_ub = time_ub)
        
        # do we ALWAYS have to solve the upper bound problem?
        # NOTE: be careful - there is very few times I expect you not to have to do this.
        self.ub_required = True
        
        # save different dicts of candidates
        self.candidates = []

    def generate_candidate(self, 
                           node: sno.Node, 
                           subproblems: sno.Subproblems) -> Tuple[bool, dict, float]:
        """
        This method should return a boolean (candidate_found) to indicate
        if the solve was successful & a dictionary, that has a key for
        each of the lifted variables and a corresponding value.

        NOTE: if we do not find a candidate, send an emtpy dict / obj back.
              They will not be checked / used if we did not find a candidate.

        Parameters
        ----------
        node : Node
            node object representing the current node we are exploring in the branch 
            and bound tree. Contains all bounding information.
        subproblems : Subproblems
            initialized subproblem manager.
            contains all subproblem names, models, probabilities, and lifted var lists/

        Returns
        ----------
        candidate_found : bool
            If this method successfully found a candidate or not
        candidate_solution : dict
            For each of the lifted variable ID's, therre should be a 
            corresponding value to fix the variable to.
        candidate_solution_obj : float
            If we do not want a global guarantee, an objective value is necessary
            to be produced.
        """
        # let's just do the average of the solutions so far
        candidate_solution = compute.average_lb_solution(node, subproblems)
        self.candidates.append(candidate_solution)

        # necessary information to return
        candidate_found = True
        candidate_solution_obj = math.nan

        return candidate_found, candidate_solution, candidate_solution_obj


def subproblem_creator(scen_name):
    """
    function is called once per scenario name passed.

    Parameters
    ------------
    scen_name : str
        name representing a unique scenario.

    Returns
    ------------
    scen_model : pyo.ConcreteModel()
        model representing this particular scenario
    lifted_var_ids : dict
        keys = first stage variable ID's (str OR tuple)
        vals = pyo.Var linked to that first stage var
    scen_prob : float
        probability associated with this scenario occuring
    """
    # grab model
    scen_model = sp.create_subproblem(scen_name)

    # get all first stage vars
    lifted_var_ids = {}
    for var in scen_model.component_data_objects(pyo.Var):
        if "DevotedAcreage" in var.name:
            if "WHEAT" in var.name:
                lifted_var_ids[f"DevotedAcreageWheat"] = var
            elif "CORN" in var.name:
                lifted_var_ids[f"DevotedAcreageCorn"] = var
            elif "SUGAR_BEETS" in var.name:
                lifted_var_ids[f"DevotedAcreageBeets"] = var

    # get probability
    id = scen_name.split("LF_")[1]
    for scen in LF_scendata["scenarios"]:
        if scen["ID"] == id:
            scen_prob = scen["Probability"]
            break

    return scen_model, lifted_var_ids, scen_prob

def run_snoglode():
    lf_scen_names = [b for b in iter(sp.bundles)]

    # create / set necessary params for snoglode
    params = sno.SolverParameters(subproblem_names = lf_scen_names,
                                  subproblem_creator = subproblem_creator,
                                  lb_solver = pyo.SolverFactory("gurobi"),
                                  cg_solver = pyo.SolverFactory("gurobi"),
                                  ub_solver = pyo.SolverFactory("gurobi"))
    params.inherit_solutions_from_parent(True)
    params.set_bounders(candidate_solution_finder = CustomCandidateGenerator)
    
    solver = sno.Solver(params)
    solver.solve(max_iter = 25)

if __name__=="__main__":
    run_snoglode()