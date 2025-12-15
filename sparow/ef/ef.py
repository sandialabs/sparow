import sys
import numpy as np
import munch
import logging
import datetime

from pyomo.common.timing import tic, toc
import sparow.logs
import sparow.solnpool

logger = sparow.logs.logger


class ExtensiveFormSolver(object):

    def __init__(self):
        self.solver_name = None
        self.solver_options = {}

    def set_options(self, *, solver=None, solver_options=None, loglevel=None):
        #
        # Misc configuration
        #
        if solver:
            self.solver_name = solver
        if solver_options:
            self.solver_options = solver_options

        if loglevel is not None:
            if loglevel == "DEBUG" or loglevel == "VERBOSE":
                sparow.logs.use_debugging_formatter()
            logger.setLevel(loglevel)

    def solve_and_return_EF(self, sp, **options):
        start_time = datetime.datetime.now()
        if len(options) > 0:
            self.set_options(**options)
        # The StochProgram object manages the sub-solver interface.  By default, we assume
        #   the user has initialized the sub-solver within the SP object.
        if self.solver_name:
            sp.set_solver(self.solver_name)

        logger.info("")
        logger.info("-" * 70)
        logger.info("ExtensiveFormSolver - START")
        if logger.isEnabledFor(logging.VERBOSE):
            print(f"  Solver: {self.solver_name}")
            print(f"  Solver Options")
            for k, v in self.solver_options.items():
                print(f"    {k}= {v}")
        tic(None)

        M = sp.create_EF(compact_repn=True)
        if logger.isEnabledFor(logging.DEBUG):
            # Print extensive form model
            M.pprint()
            sys.stdout.flush()

        toc("Created extensive form", logger=logger, level=logging.VERBOSE)
        results = sp.solve(M, solver_options=self.solver_options)

        # TODO - show value of subproblem
        toc("Optimized extensive form", logger=logger, level=logging.VERBOSE)
        end_time = datetime.datetime.now()

        solutions = sparow.solnpool.PoolManager()
        metadata = solutions.metadata
        metadata.termination_condition = str(results.termination_condition)
        metadata.status = str(results.status)
        metadata.start_time = str(start_time)
        metadata.end_time = str(end_time)
        metadata.time_elapsed = str(end_time - start_time)

        if results.obj_value is not None:
            b = next(iter(sp.bundles))
            variables = [
                sparow.solnpool.Variable(
                    value=sp.get_variable_value(b, i),
                    index=i,
                    name=sp.get_variable_name(i),
                )
                for i, _ in enumerate(sp.get_variables())
            ]
            objective = sparow.solnpool.Objective(value=results.obj_value)
            solutions.add(variables=variables, objective=objective)

        logger.info("")
        logger.info("-" * 70)
        logger.info("ExtensiveFormSolver - STOP")

        return munch.Munch(solutions=solutions, model=M)

    def solve(self, sp, **options):
        return self.solve_and_return_EF(sp, **options).solutions
