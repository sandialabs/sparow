import sys
import pprint
import mpisppy
#from sparow_examples.pmedian.pmedian import *
from sparow_examples.facilityloc.grid_facilityloc import *
from sparow.ph import ProgressiveHedgingSolver
#from sparow.ef import ExtensiveFormSolver
from sparow.ph.ph_mpisppy import ProgressiveHedgingSolver_MPISPPY

sp = random_HF_LF1_grid_facilityloc()
solver = ProgressiveHedgingSolver_MPISPPY() #ProgressiveHedgingSolver()
solver.set_options(
    solver="gurobi",
    max_iterations=100,
    rho_updates=True,
    loglevel="INFO",
    mpisppy_options=["--lagrangian", "--xhatshuffle", "--rel-gap=0.01"]
)

results_mpi = solver.solve(sp, solver="gurobi")
#pprint.pprint(results_mpi.to_dict())
if getattr(solver, "mpi_rank", 0) == 0:
    pprint.pprint(results_mpi.to_dict())
    #results_mpi.write("results_mpi.json", indent=4)
    #print("Writing results to 'results_mpi.json'")
