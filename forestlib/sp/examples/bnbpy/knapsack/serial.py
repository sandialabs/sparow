# Serial knapsack

from forestlib.bnbpy import SerialBBSolver
from knapsack import Knapsack

problem = Knapsack(filename="scor-500-1.txt")
solver = SerialBBSolver()
value, solution = solver.solve(problem=problem)
print(value)
problem.print_solution(solution)
