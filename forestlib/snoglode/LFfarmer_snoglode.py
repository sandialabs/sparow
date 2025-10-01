"""
Demonstration of solving the LF farmer SP using SNoGloDe
"""

import snoglode as sno
from snoglode.utils.solve_stats import OneUpperBoundSolve
import snoglode.utils.compute as compute
from pprint import pprint

import pyomo.environ as pyo
from pyomo.contrib.alternative_solutions.aos_utils import get_active_objective
from pyomo.opt import TerminationCondition, SolverStatus

try:
    from pyomo.contrib.alternative_solutions.solnpool import (
        PoolCounter,
        SolutionPool_KeepBest,
        PoolManager,
    )
    from pyomo.contrib.alternative_solutions.solution import Solution, PyomoSolution
    from pyomo.contrib.alternative_solutions import Objective, Variable

    alt_sol_available = True
    print("Alternative solutions package from pyomo.contrib is available.")
except:
    PoolCounter, SolutionPool_KeepBest, Solution = None, None, None
    alt_sol_available = False
    print(
        "Alternative solutions package from pyomo.contrib is unavailable. \
          \nReverting to simple saving behavior."
    )

from MFfarmers import LFScenario_dict, LF_model_builder
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
app_data = {"num_plots": 1, "use_integer": False}
model_data = {"LF": LF_scendata}
sp = stochastic_program(first_stage_variables=["DevotedAcreage[*,*]"])
sp.initialize_application(app_data=app_data)
sp.initialize_model(
    name="LF", model_data=model_data["LF"], model_builder=LF_model_builder
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

    def __init__(self, solver, subproblems: sno.Subproblems, time_ub: int) -> None:

        super().__init__(solver=solver, subproblems=subproblems, time_ub=time_ub)

        # do we ALWAYS have to solve the upper bound problem?
        # NOTE: in this case no because I will hack in the UB solve to work with the candidate solutions
        self.ub_required = False
        self.opt = pyo.SolverFactory("glpk")

        # save different dicts of candidates
        if alt_sol_available:

            # init solution pool object
            # self.aos = SolutionPool_KeepBest(counter = PoolCounter(),
            #  max_pool_size = 10)
            self.pm = PoolManager()
            self.pm.add_pool("pool_1", policy="keep_best", max_pool_size=10)
            # self.pm.add_pool("pool_1", policy="keep_all")

        else:
            self.candidates = []

    def fix_to_candidate_solution(
        self,
        subproblem_lifted_variables: list,
        subproblem_var_map: pyo.ComponentMap,
        candidate_solution_state: dict,
    ) -> None:
        """
        Given the candidate solution state dictionary, fix all of the scenarios
        to the necessary values.

        Parameters
        -----------
        subproblem_lifted_variables : list
            list of all the first stage variables.
        candidate_solution : dict
            Dictionary containing all candidate solution values.
        """
        # for each of the first stage variables, retrieve value & fix
        for var in subproblem_lifted_variables:
            _, var_id, _ = subproblem_var_map[var]
            var_candidate_value = candidate_solution_state[var_id]
            var.fix(var_candidate_value)

    def solve_subproblem(
        self, subproblem_model: pyo.ConcreteModel
    ) -> Tuple[bool, float]:
        """
        Given a Pyomo model representing one of the subproblems, solve

        Parameters
        -----------
        subproblem_model : pyo.ConcreteModel()
            A pyomo model representing a single subproblem.
            Should be reflecting current node state

        Returns
        -----------
        feasible_solution : bool
            Was the model feasible / solved okay?
        scenario_objective : float
            The value of the objective; returns None if infeasible.
        """
        # solve model
        results = self.opt.solve(
            subproblem_model,
            load_solutions=False,
            symbolic_solver_labels=True,
            tee=False,
        )

        # if the solution is optimal, return objective value
        if (
            results.solver.termination_condition == TerminationCondition.optimal
            and results.solver.status == SolverStatus.ok
        ):

            # load in solutions, return [feasibility = True, obj, results]
            subproblem_model.solutions.load_from(results)

            # there should only be one objective, so return that value.
            return True, pyo.value(get_active_objective(subproblem_model))

        # if the solution is not feasible, return None
        else:
            return False, None

    def generate_candidate(
        self, node: sno.Node, subproblems: sno.Subproblems
    ) -> Tuple[bool, dict, float]:
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
        # compute the average across all LB solutions as the candidate
        candidate_solution = compute.average_lb_solution(node, subproblems)

        # init statistics object for this solve
        statistics = OneUpperBoundSolve(subproblems.names)

        # for each subproblems's model
        for subproblem_name in subproblems.names:

            # un-relax binaries, if there are any
            if subproblems.relax_binaries:
                subproblems.unrelax_all_binaries()
            if subproblems.relax_integers:
                subproblems.unrelax_all_integers()

            # fix the first stage variables to the candidate solution
            self.fix_to_candidate_solution(
                subproblem_lifted_variables=subproblems.subproblem_lifted_vars[
                    subproblem_name
                ],
                subproblem_var_map=subproblems.var_to_data,
                candidate_solution_state=candidate_solution,
            )

            # solve the current model representing this scenario
            subproblem_is_feasible, subproblem_objective = self.solve_subproblem(
                subproblem_model=subproblems.model[subproblem_name]
            )

            # if we have one infeasible scenario, the entire node is infeasible
            if not subproblem_is_feasible:

                # if we are infeasible, both UB/LB are infeasible -> add appropriate stats
                node.ub_problem.is_infeasible()
                return False, {}, math.nan

            # if we are feasible, add statistics
            statistics.update(
                subproblem_name=subproblem_name,
                subproblem_objective=subproblem_objective,
                subproblem_probabilty=subproblems.probability[subproblem_name],
            )

        # if we were successful, add statistics to node
        node.ub_problem.is_feasible(pyo.value(statistics.aggregated_objective))

        # save candidate solution
        if not alt_sol_available:
            self.candidates.append(candidate_solution)
        else:
            # store vars
            variables = []
            for var in candidate_solution:
                variables.append(Variable(name=var, value=candidate_solution[var]))

            self.pm.add(
                Solution(
                    variable=variables,
                    objectives=[
                        Objective(value=pyo.value(statistics.aggregated_objective))
                    ],
                )
            )

        return True, candidate_solution, pyo.value(statistics.aggregated_objective)


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

    lifted_var_ids = {sp.int_to_FirstStageVarName[i]:v for i,v in sp.int_to_FirstStageVar[scen_name].items()}
    scen_prob = sp.bundles[scen_name].probability

    return scen_model, lifted_var_ids, scen_prob


def run_snoglode():
    lf_scen_names = [b for b in iter(sp.bundles)]

    # create / set necessary params for snoglode
    params = sno.SolverParameters(
        subproblem_names=lf_scen_names,
        subproblem_creator=subproblem_creator,
        lb_solver=pyo.SolverFactory("glpk"),
        cg_solver=pyo.SolverFactory("glpk"),
        ub_solver=pyo.SolverFactory("glpk"),
    )
    params.inherit_solutions_from_parent(True)
    params.set_bounders(candidate_solution_finder=CustomCandidateGenerator)

    solver = sno.Solver(params)
    solver.solve(max_iter=25)

    # evaluate results
    aos = solver.upper_bounder.candidate_solution_finder.pm.to_dict()
    pprint(aos)


if __name__ == "__main__":
    run_snoglode()
