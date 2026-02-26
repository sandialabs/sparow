import sys
import pprint
import mpisppy
#from sparow_examples.pmedian.pmedian import *
from sparow_examples.facilityloc.facilityloc import *
from sparow.ph import ProgressiveHedgingSolver
#from sparow.ef import ExtensiveFormSolver
from sparow.ph.ph_mpisppy import ProgressiveHedgingSolver_MPISPPY

sp = LF_facilityloc()
solver = ProgressiveHedgingSolver() #ProgressiveHedgingSolver_MPISPPY()
solver.set_options(
    solver="gurobi",
    max_iterations=100,
    rho_updates=True,
    loglevel="INFO"
)

results_mpi = solver.solve(sp, solver="gurobi")
pprint.pprint(results_mpi.to_dict())
if getattr(solver, "mpi_rank", 0) == 0:
    pprint.pprint(results_mpi.to_dict())
    #results_mpi.write("results_mpi.json", indent=4)
    #print("Writing results to 'results_mpi.json'")
