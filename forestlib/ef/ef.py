import sys
import numpy as np
import munch

import logging
import forestlib.logs

logger = forestlib.logs.logger


class ExtensiveFormSolver(object):

    def __init__(self):
        self.solver_name = None
        self.solver_options = {}

    def set_options(
        self,
        *,
        solver=None,
        solver_options=None,
        loglevel=None,
    ):
        #
        # Misc configuration
        #
        if solver:
            self.solver_name = solver
        if solver_options:
            self.solver_options = solver_options

        if loglevel is not None:
            if loglevel == "DEBUG":
                forestlib.logs.use_debugging_formatter()
            logger.setLevel(loglevel)

    def solve(self, sp, **options):
        if len(options) > 0:
            self.set_options(**options)
        # The StochProgram object manages the sub-solver interface.  By default, we assume
        #   the user has initialized the sub-solver within the SP object.
        if self.solver_name:
            sp.set_solver(self.solver_name)

        logger.info("ProgressiveHendingSolver - START")

        sp.initialize_bundles(scheme="single_bundle")
        assert (
            len(sp.bundles) == 1
        ), f"The extensive form should only have one bundle: {len(sp.bundles)}"

        logger.debug(f"Creating extensive form")
        b = next(iter(sp.bundles))
        M = sp.create_subproblem(b)
        if logger.isEnabledFor(logging.DEBUG):
            M.pprint()
            M.display()
            sys.stdout.flush()

        logger.debug(f"Optimizing extensive form")
        results = sp.solve(M, solver_options=self.solver_options)

        # TODO - show value of subproblem
        logger.debug(f"Optimization Complete")

        if results.obj_value is None:
            return munch.Munch(
                termination_condition=str(results.termination_condition),
                status=str(results.status),
                solutions=[],
            )
        else:
            return munch.Munch(
                obj_value=results.obj_value,
                termination_condition=str(results.termination_condition),
                status=str(results.status),
                solutions=[sp.get_variables()],
            )
