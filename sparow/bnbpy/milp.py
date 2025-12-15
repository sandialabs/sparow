###
### MILP problem class
###
# only requirement is that variables which should be binary should all be
# elements of m.y - all other variables should have names that are not y


import copy
import re
import math
from .bb import BranchAndBound
from pyomo.environ import SolverFactory, value

__author__ = "Jodie Simkoff: jsimkoff@utexas.edu"


class MILP(BranchAndBound):

    __slots__ = ["ub", "lb"]

    # attributes for each node are dicts containing ub and lb and their corresponding
    # index in y

    def __init__(self, model=None, sense=1):
        BranchAndBound.__init__(
            self, sense=sense
        )  # sense = -1 for maximization problem!
        self.context.model = model
        self.ub = {}
        self.lb = {}

    def debug(self):
        print("LOCKED IN")
        print(self.locked_in)
        print("LOCKED OUT")
        print(self.locked_out)
        print("SOLUTION")
        print(self.solution)
        print("SOLUTION VALUE")
        print(self.solution_value)
        print("BOUND")
        print(self.bound)
        print("LAST")
        print(getattr(self, "last", None))

    def compute_bound(self):
        global_model = self.context.model
        # clear all previous bounds on y
        for i in global_model.y:
            global_model.y[i].setlb(None)
            global_model.y[i].setub(None)
        # set bounds for given node
        for i in self.lb:
            global_model.y[i].setlb(self.lb[i])
        for i in self.ub:
            global_model.y[i].setub(self.ub[i])
        results = SolverFactory("glpk").solve(global_model, keepfiles=False, tee=False)
        # infeasible exit code
        if str(results.solver.termination_condition) == "infeasible":
            print("infeasible subproblem\n")
            self.bound = self.sense * float("Inf")
        # some other exit code - treating as infeasible here for now...
        elif str(results.solver.termination_condition) != "optimal":
            print(str(results.solver.termination_condition))
            self.bound = self.sense * float("Inf")
        # optimal solution
        else:
            self.bound = value(global_model.obj)
            self.solution_value = value(global_model.obj)
            self.solution = global_model.y

        return self.bound

    def make_child(self, which_child):
        global_model = self.context.model

        child = MILP(model=global_model, sense=self.sense)
        child.bound = self.bound
        child.lb = copy.copy(self.lb)
        child.ub = copy.copy(self.ub)

        # branch on most fractional value
        max_frac = None
        for i in global_model.y:
            val = global_model.y[i].value
            frac = min(val - math.floor(val), math.ceil(val) - val)

            if max_frac is None or frac > max_frac:
                max_frac = frac
                v1 = math.floor(val)
                v2 = math.ceil(val)
                index = i

        if which_child == 0:
            child.ub[index] = v1
        elif which_child == 1:
            child.lb[index] = v2
        else:
            raise RuntimeError("Unknown child %d" % which_child)

        print("making child:")
        print("bound: %f" % child.bound)
        print("ub", child.ub)
        print("lb", child.lb)
        print("\n")

        return child

    def separate(self):
        return 2

    def terminal(self):
        """
        Return True if this is a terminal (integer solution)
        """
        return self.get_solution()[0] is not None

    def get_solution(self):
        """
        Return solution and value if integer.
        """
        global_model = self.context.model

        frac = 0
        for i in global_model.y:
            val = global_model.y[i].value
            if (
                min(val - math.floor(val), math.ceil(val) - val) >= 1e-6
            ):  # tolerance on integrality violation
                frac = 1
                break
        if frac > 0:
            return (None, None)
        return (self.solution_value, copy.deepcopy(self.solution))

    def print_solution(self, solution):
        solution.display()


if __name__ == "__main__":
    from bnbpy import SerialBBSolver
    from example2 import m

    problem = MILP(model=m, sense=-1)  # 1 to minimize, -1 to maximize
    solver = SerialBBSolver()
    value, solution = solver.solve(problem=problem)
    print("opt value: %f\n" % value)
    problem.print_solution(solution)
