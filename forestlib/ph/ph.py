import sys
import munch
import pprint
import numpy as np

import logging
import forestlib.logs

logger = forestlib.logs.logger


def norm(values, p):
    return np.linalg.norm(np.array(values), ord=p)


class ProgressiveHedgingSolver(object):

    def __init__(self):
        self.rho = 1.5
        self.max_iterations = 100
        self.convergence_tolerance = 1e-3
        self.normalize_convergence_norm = True
        self.convergence_norm = 1
        self.solver_name = None
        self.solver_options = {}
        self.finalize_xbar_by_rounding = True

    def set_options(
        self,
        *,
        rho=None,
        max_iterations=None,
        convergence_tolerance=None,
        normalize_convergence_norm=None,
        convergence_norm=None,
        solver=None,
        solver_options=None,
        loglevel=None,
        finalize_xbar_by_rounding=None,
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
        if convergence_norm:
            self.convergence_norm = convergence_norm
        if solver:
            self.solver_name = solver
        if solver_options:
            self.solver_options = solver_options
        if finalize_xbar_by_rounding:
            self.finalize_xbar_by_rounding = finalize_xbar_by_rounding

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

        logger.info("ProgressiveHedgingSolver - START")

        # Step 2
        obj_value = {}
        for b in sp.bundles:
            logger.verbose(f"Creating subproblem '{b}'")
            M = sp.create_subproblem(b)
            # M.write(f'Iter0_PH_{b}.lp',io_options={'symbolic_solver_labels':True})
            logger.verbose(f"Optimizing subproblem '{b}'")
            results = sp.solve(M, solver_options=self.solver_options)
            assert (
                results.obj_value is not None
            ), f"ERROR solving bundle {b} in initial solve"
            obj_value[b] = results.obj_value
            logger.verbose(f"Optimization Complete")
        obj_lb = sum(sp.bundles[b].probability * obj_value[b] for b in sp.bundles)

        #
        # This is a list of shared first-stage variables amongst all bundles.
        # Note: we need to initialize this here, *after* we create our initial sub-problems.
        #
        sfs_variables = sp.shared_variables()

        # Step 3
        xbar = {}
        for x in sfs_variables:
            xbar[x] = 0.0
            for b in sp.bundles:
                xbar[x] += sp.bundles[b].probability * sp.get_variable_value(b, x)

        # Step 4
        w = {}
        for b in sp.bundles:
            w[b] = {}
            for x in sfs_variables:
                w[b][x] = self.rho * (sp.get_variable_value(b, x) - xbar[x])

        iteration = 0
        termination_condition = "Termination: unknown"
        while True:
            logger.info("")
            logger.info("-" * 70)
            logger.info(f"Iteration:    {iteration}")
            logger.info(f"obj_lb:      {obj_lb}")
            logger.verbose(f"xbar:        {xbar}")
            logger.verbose(f"rho:         {self.rho}")
            logger.info("")

            # Step 5
            xbar_prev = xbar
            w_prev = w

            # Step 6
            obj_value = {}
            for b in sp.bundles:
                logger.verbose(f"Creating subproblem '{b}'")
                logger.debug(f"  b: {b}  w: {w[b]}")
                M = sp.create_subproblem(
                    b=b, w=w_prev[b], x_bar=xbar_prev, rho=self.rho
                )
                logger.verbose(f"Optimizing subproblem '{b}'")
                results = sp.solve(M, solver_options=self.solver_options)
                assert (
                    results.obj_value is not None
                ), f"ERROR solving bundle {b} in iteration {iteration}"
                obj_value[b] = results.obj_value
                logger.verbose(f"Optimization Complete")
            obj_lb = sum(sp.bundles[b].probability * obj_value[b] for b in sp.bundles)

            # Step 7
            xbar = {}
            for x in sfs_variables:
                xbar[x] = 0.0
                for b in sp.bundles:
                    logger.debug(
                        f"Variable: {x} {b} {sp.get_variable_name(b,x)} {sp.get_variable_value(b, x)}"
                    )
                    xbar[x] += sp.bundles[b].probability * sp.get_variable_value(b, x)
            logger.debug(f"xbar = {xbar}")

            # Step 8
            w = {}
            for b in sp.bundles:
                w[b] = {}
                for x in sfs_variables:
                    w[b][x] = w_prev[b][x] + self.rho * (
                        sp.get_variable_value(b, x) - xbar[x]
                    )
                logger.debug(f"w[{b}] = {w[b]}")

            # Step 9
            g = 0.0
            for b in sp.bundles:
                g += sp.bundles[b].probability * norm(
                    [sp.get_variable_value(b, x) - xbar[x] for x in sfs_variables],
                    self.convergence_norm,
                )
            if self.normalize_convergence_norm:
                g /= len(sfs_variables)
            logger.info(f"g = {g}")

            # Step 10
            if g < self.convergence_tolerance:
                termination_condition = f"Termination: convergence tolerance ({g} < {self.convergence_tolerance})"
                logger.info(termination_condition)
                break

            iteration += 1
            if iteration == self.max_iterations:
                termination_condition = f"Termination: max_iterations ({iteration} == {self.max_iterations})"
                logger.info(termination_condition)
                break

            self.update_rho(iteration)

        logger.info("")
        logger.info("-" * 70)
        logger.info("ProgressiveHedgingSolver - FINALIZING")
        results = self.finalize_results(
            sp,
            xbar=xbar,
            w=w,
            g=g,
            iterations=iteration,
            termination_condition=termination_condition,
            obj_lb=obj_lb,
        )
        logger.info("")
        logger.info("-" * 70)
        logger.info("ProgressiveHedgingSolver - RESULTS")
        if logger.level != logging.NOTSET and logger.level <= logging.VERBOSE:
            pprint.pprint(results.toDict())
            sys.stdout.flush()

        logger.info("")
        logger.info("-" * 70)
        logger.info("ProgressiveHedgingSolver - STOP")

        return results

    def update_rho(self, iteration):
        # TODO HERE
        pass

    def finalize_results(
        self, sp, *, xbar, w, g, iterations, termination_condition, obj_lb
    ):
        #
        # We use xbar to identify a point that is feasible for all scenarios.
        #
        solutions = []
        if sp.continuous_fsv():
            logger.info("Final solution is continuous")
            #
            # Evaluate the final xbar, and keep if feasible.
            #
            sol = sp.evaluate([xbar[x] for x in sp.shared_variables()])
            if sol.feasible:
                solutions.append(sol)
        else:
            logger.info("Final solution has binary or integer variables")

            if self.finalize_xbar_by_rounding:
                #
                # Round the final xbar, and keep if feasible.
                #
                logger.info(
                    "Rounding xbar values associated with binary and integer variables"
                )
                tmpx = [sp.round(x, xbar[x]) for x in sp.shared_variables()]
                sol = sp.evaluate(tmpx)
                if sol.feasible:
                    solutions.append(sol)

        return munch.Munch(
            xbar=xbar,
            w=w,
            g=g,
            iterations=iterations,
            termination_condition=termination_condition,
            obj_lb=obj_lb,
            solutions=solutions,
        )
