from sparow_examples.facilityloc.grid_facilityloc import *  # import facility location exemplar from sparow_examples
from sparow.ph.ph_mpisppy import (
    ProgressiveHedgingSolver_MPISPPY,
)  # import parallel PH solver (mpi-sppy wrapper)
import pprint  # import for solution readability

sp = random_HF_LF1_grid_facilityloc()  # SP model object imported from sparow_examples
solver = (
    ProgressiveHedgingSolver_MPISPPY()
)  # solving in parallel with the sparow wrapper around mpisppy
solver.set_options(
    solver="gurobi",  # solving with gurobi
    max_iterations=2,  # this will default to 100
    loglevel="INFO",  # can replace with DEBUG, VERBOSE, etc.
    default_rho=1.5,  # rho by default will already be 1.5
    mpisppy_options=[
        "--lagrangian",
        "--xhatshuffle",
        "--rel-gap=0.01",
    ],  # can customize these options as well
)

results_mpi = solver.solve(sp, solver="gurobi")  # returns the solution object
if getattr(solver, "mpi_rank", 0) == 0:  # all the information gets sent to rank 0
    pprint.pprint(results_mpi.to_dict())  # pretty-print results
