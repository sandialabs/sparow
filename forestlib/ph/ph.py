import logging
import numpy as np

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class ProgressiveHedgingSolver(object):

    def __init__(self):
        self.rho = 1.5
        self.max_iterations = 100
        self.convergence_tolerance = 1e-3
        self.normalize_convergence_norm = True
        self.solver_name = None
        self.solver_options = {}

    def set_options(
        self,
        *,
        rho=None,
        max_iterations=None,
        convergence_tolerance=None,
        normalize_convergence_norm=None,
        solver=None,
        solver_options=None,
        loglevel=None,
    ):
        #
        # Misc configuration
        #
        if rho:
            self.rho = rho
        if max_iterations:
            self.max_iterations = max_iterations
        if convergence_tolerance:
            self.convergence_tolerance = convergence_tolerance
        if normalize_convergence_norm:
            self.normalize_convergence_norm = normalize_convergence_norm
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

        # Step 2
        M = {}
        for b in sp.bundles:
            logger.debug(f"Creating subproblem '{b}'")
            M[b] = sp.create_subproblem(b)
            # M[b].write(f'Iter0_PH_{b}.lp',io_options={'symbolic_solver_labels':True})
            logger.debug(f"Optimizing subproblem '{b}'")
            sp.solve(M[b], solver_options=self.solver_options)
            # TODO - show value of subproblem
            logger.debug(f"Optimization Complete")

        #
        # This is a list of shared first-stage variables amongst all bundles.
        # Note: we need to initialize this here, *after* we create our initial sub-problems.
        #
        sfs_variables = sp.shared_variables()

        # Step 3
        k = 0
        x_bar = {}
        for x in sfs_variables:
            x_bar[x] = 0.0
            for b in sp.bundles:
                x_bar[x] += sp.bundle_probability[b] * sp.get_variable_value(b, x)

        # Step 4
        w = {}
        for b in sp.bundles:
            w[b] = {}
            for x in sfs_variables:
                w[b][x] = self.rho * (sp.get_variable_value(b, x) - x_bar[x])

        while True:
            logger.info("")
            logger.info("-" * 70)
            logger.info("")
            logger.info(f"Iteration: {k}")
            logger.debug(f"x_bar:     {x_bar}")
            logger.debug(f"rho:      {self.rho}")

            # Step 5
            x_bar_prev = x_bar
            w_prev = w

            # Step 6
            M = {}
            for b in sp.bundles:
                logger.debug(f"Creating subproblem '{b}'")
                logger.debug(f"  b: {b}  w: {w[b]}")
                M[b] = sp.create_subproblem(
                    b=b, w=w_prev[b], x_bar=x_bar_prev, rho=self.rho
                )
                logger.debug(f"Optimizing subproblem '{b}'")
                sp.solve(M[b], solver_options=self.solver_options)
                logger.debug(f"Optimization Complete")

            # Step 7
            x_bar = {}
            for x in sfs_variables:
                x_bar[x] = 0.0
                for b in sp.bundles:
                    x_bar[x] += sp.bundle_probability[b] * sp.get_variable_value(b, x)
            logger.debug(f"x_bar = {x_bar}")

            # Step 8
            w = {}
            for b in sp.bundles:
                w[b] = {}
                for x in sfs_variables:
                    w[b][x] = w_prev[b][x] + self.rho * (
                        sp.get_variable_value(b, x) - x_bar[x]
                    )
                logger.debug(f"w[{b}] = {w[b]}")

            # Step 9
            g = 0.0
            for b in sp.bundles:
                g += sp.bundle_probability[b] * self.norm(
                    sp.get_variable_value(b, x) - x_bar[x] for x in sfs_variables
                )
            if self.normalize_convergence_norm:
                g /= len(sfs_variables)
            logger.info(f"g = {g}")

            # Step 10
            if g < self.convergence_tolerance:
                logger.info(
                    f"Termination: convergence tolerance ({g} < {self.convergence_tolerance})"
                )
                break

            self.update_rho()
            k += 1
            if k == self.max_iterations:
                logger.info(
                    f"Termination: max_iterations ({k} == {self.max_iterations})"
                )
                break

        logger.info("ProgressiveHendingSolver - STOP")
        self.store_results(x_bar=x_bar, w=w, g=g)

    def update_rho(self):
        # TODO HERE
        pass

    def store_results(self, *, x_bar, w, g):
        # Abstract
        pass

    def norm(self, values):
        v = np.array(list(values))
        return np.linalg.norm(v, ord=1)
