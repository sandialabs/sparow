import pprint

# import parallel PH solver (mpi-sppy wrapper)
from sparow.ph.ph_mpisppy import ProgressiveHedgingSolver_MPISPPY

# import newsvendor example from sparow
from sparow.sp.examples import simple_newsvendor

# SP model object
example = simple_newsvendor()

# solving in parallel with the sparow wrapper around mpisppy
solver = ProgressiveHedgingSolver_MPISPPY()

solver.set_options(
    # solving with highs
    solver="highs",
    # this will default to 100
    max_iterations=2,
    # can replace with DEBUG, VERBOSE, etc.
    loglevel="INFO",
    # rho by default will already be 1.5
    default_rho=1.5,
    # mpisppy specific options
    mpisppy_options=[
        "--tee-rank0-solves",
        "--lagrangian",
        "--xhatshuffle",
        "--rel-gap=0.01",
    ],
)

# Call mpisppy and return a sparow solution object
results_mpi = solver.solve(example.sp)


if getattr(solver, "mpi_rank", 0) == 0:
    # Print results on Rank 0
    pprint.pprint(results_mpi.to_dict())
