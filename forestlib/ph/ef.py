import logging
import numpy as np

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


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
                formatter = logging.Formatter(
                    "%(asctime)s - %(levelname)s - %(message)s"
                )
                handler.setFormatter(formatter)
            logger.setLevel(loglevel)

    def solve(self, sp, **options):
        if len(options) > 0:
            self.set_options(**options)
        # The StochProgram object manages the sub-solver interface.  By default, we assume
        #   the user has initialized the sub-solver within the SP object.
        if self.solver_name:
            sp.set_solver(self.solver_name)

        logger.info("ProgressiveHendingSolver - START")

        sp.initialize_bundles(bundle_scheme="single_bundle")

        assert len(sp.bundles) == 1, "The extensive form has a single bundle"

        logger.debug(f"Creating extensive form")
        M[b] = sp.create_subproblem(sp.bundles[0])

        logger.debug(f"Optimizing extensive form")
        sp.solve(M[b], solver_options=self.solver_options)

        # TODO - show value of subproblem
        logger.debug(f"Optimization Complete")
